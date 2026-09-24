"""Tests for CLI progress rendering (interface-quality plan T2.4).

Covers _ProgressPoller semantics: a determinate percent renders only when
the authed /generation-status payload carries progress_pct (T1.0) — never
synthesized from chunk_index (plan decision D2); stream mode never renders
a percent; the local elapsed-only ticker polls nothing; the bar owns
stderr; quiet mode is a no-op; and generate_via_server's mid-generation
model-load prompt cannot hang a non-interactive script (tty guard).
"""

import io
import sys
import unittest
from unittest.mock import MagicMock, patch

_MOD = "qwen3_tts.interface.generate_server"


class _FakeTty(io.StringIO):
    """A stdin stand-in that reports itself as interactive."""

    def isatty(self):
        return True


class _FakeNotTty(io.StringIO):
    """A stdin stand-in that reports itself as non-interactive (pipes, CI)."""

    def isatty(self):
        return False


class _FakeProgress:
    """Records rich Progress calls; shaped like the context manager."""

    def __init__(self, *args, **kwargs):
        self.add_task_calls = []
        self.update_calls = []
        self._next_id = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def add_task(self, description, total=None):
        self.add_task_calls.append((description, total))
        self._next_id += 1
        return self._next_id

    def update(self, task_id, **fields):
        self.update_calls.append((task_id, fields))


def _status_response(payload, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


def _authed_payload(**extra):
    """An authed /generation-status body (the T1.0 shape)."""
    state = {
        "active": True,
        "cancelled": False,
        "batch_index": 0,
        "chunk_index": 1,
        "elapsed_sec": 12.0,
        "batch_total": 1,
        "chunk_total": 1,
    }
    state.update(extra)
    return state


def _public_payload(**extra):
    """The unauthenticated /generation-status body — no derived fields."""
    state = {
        "active": True,
        "cancelled": False,
        "batch_index": 0,
        "chunk_index": 1,
        "elapsed_sec": 12.0,
    }
    state.update(extra)
    return state


class _PollerCase(unittest.TestCase):
    """Base: swaps stdout/stderr for StringIO, runs exactly one poller loop
    iteration against a scripted /generation-status payload."""

    def setUp(self):
        self._stdout = io.StringIO()
        self._stderr = io.StringIO()
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self.addCleanup(self._restore_streams)

    def _restore_streams(self):
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

    def _make_poller(self):
        from qwen3_tts.interface.generate_interactive import _ProgressPoller

        return _ProgressPoller

    def _run_one_iteration(self, poller, payload, runner=None):
        """Run a single poller loop pass against *payload*; return rendered text."""
        real_wait = poller._stop.wait

        def stop_after_first(timeout=None):
            poller._stop.set()
            return real_wait(0)

        poller._stop.wait = stop_after_first
        loop = runner or poller._run_fallback
        with (
            patch(
                "qwen3_tts.core.http_client.server_request",
                return_value=_status_response(payload),
            ),
            patch.object(type(poller), "HAS_RICH", False),
        ):
            loop()
        return self._stderr.getvalue()


class TestFallbackProgress(_PollerCase):
    """Print-based fallback loop: percent rules and stream separation."""

    def test_single_item_renders_server_percent(self):
        poller = self._make_poller()(batch_total=1)
        out = self._run_one_iteration(
            poller, _authed_payload(progress_pct=42.0, eta_sec=30.0)
        )
        self.assertIn("42%", out, f"authed progress_pct not rendered: {out!r}")

    def test_percent_absent_renders_elapsed_only(self):
        poller = self._make_poller()(batch_total=1)
        out = self._run_one_iteration(poller, _authed_payload())
        self.assertIn("elapsed", out)
        self.assertNotIn("%", out, f"percent invented without progress_pct: {out!r}")

    def test_public_payload_renders_elapsed_only(self):
        poller = self._make_poller()(batch_total=1)
        out = self._run_one_iteration(poller, _public_payload())
        self.assertIn("elapsed", out)
        self.assertNotIn("%", out, f"percent invented from public payload: {out!r}")

    def test_chunk_counts_never_become_percent(self):
        """D2: chunk_index is a completed-chunk COUNT; no percent is derived."""
        poller = self._make_poller()(batch_total=1)
        out = self._run_one_iteration(
            poller, _authed_payload(chunk_index=4, chunk_total=9)
        )
        self.assertIn("[chunk 5/9]", out)
        self.assertNotIn(
            "%", out, f"percent synthesized from chunk_index (banned): {out!r}"
        )

    def test_stream_mode_never_renders_percent(self):
        poller = self._make_poller()(batch_total=1, stream=True)
        out = self._run_one_iteration(
            poller, _authed_payload(progress_pct=50.0, eta_sec=10.0)
        )
        self.assertIn("Streaming", out)
        self.assertNotIn("%", out, f"stream mode rendered a percent (banned): {out!r}")

    def test_batch_multi_renders_item_counter(self):
        poller = self._make_poller()(batch_total=3)
        out = self._run_one_iteration(
            poller, _authed_payload(batch_index=1, eta_sec=18.0, progress_pct=64.0)
        )
        self.assertIn("[2/3]", out, f"batch item counter missing: {out!r}")

    def test_bar_writes_stderr_only(self):
        """Item lines own stdout; the progress bar owns stderr."""
        poller = self._make_poller()(batch_total=1)
        self._run_one_iteration(poller, _authed_payload(progress_pct=42.0))
        self.assertEqual(self._stdout.getvalue(), "", "progress bar leaked onto stdout")
        self.assertNotEqual(self._stderr.getvalue(), "", "no bar rendered on stderr")


class TestElapsedOnlyTicker(_PollerCase):
    """elapsed_only(): the local-path ticker — no server polling at all."""

    def test_classmode_shape(self):
        poller_cls = self._make_poller()
        poller = poller_cls.elapsed_only()
        self.assertIsInstance(poller, poller_cls)
        self.assertEqual(poller.batch_total, 1)
        self.assertTrue(poller._elapsed_only)
        self.assertFalse(poller.stream)

    def test_never_polls_the_server(self):
        poller = self._make_poller().elapsed_only()
        with (
            patch("qwen3_tts.core.http_client.server_request") as mock_request,
            patch.object(type(poller), "HAS_RICH", False),
        ):
            self._run_one_iteration(poller, {})
        mock_request.assert_not_called()

    def test_renders_elapsed_line(self):
        poller = self._make_poller().elapsed_only()
        with patch.object(type(poller), "HAS_RICH", False):
            out = self._run_one_iteration(poller, {})
        self.assertIn("Generating audio...", out)
        self.assertIn("elapsed", out)
        self.assertNotIn("%", out)


class TestQuietProgress(unittest.TestCase):
    """quiet_progress=True silences the poller entirely (dry-run/CI)."""

    def test_start_is_a_noop(self):
        from qwen3_tts.interface.generate_interactive import _ProgressPoller

        poller = _ProgressPoller(batch_total=1, quiet_progress=True)
        poller.start()
        self.assertIsNone(poller._thread, "quiet mode still spawned a thread")
        poller.stop()  # must not raise without a thread


class TestRichProgress(_PollerCase):
    """Rich loop: determinate only from server progress_pct; stream indeterminate."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.interface.generate_interactive import _ProgressPoller

        if not _ProgressPoller.HAS_RICH:
            raise unittest.SkipTest("rich not installed")

    def _run_rich_once(self, poller, payload):
        fake_progress = _FakeProgress()

        def make_progress(*args, **kwargs):
            return fake_progress

        real_wait = poller._stop.wait

        def stop_after_first(timeout=None):
            poller._stop.set()
            return real_wait(0)

        poller._stop.wait = stop_after_first
        with (
            patch(
                "qwen3_tts.core.http_client.server_request",
                return_value=_status_response(payload),
            ),
            patch("rich.progress.Progress", side_effect=make_progress),
            patch("rich.console.Console"),
        ):
            poller._run_rich()
        return fake_progress

    def test_single_item_becomes_determinate_on_server_percent(self):
        poller = self._make_poller()(batch_total=1)
        fake = self._run_rich_once(poller, _authed_payload(progress_pct=42.0))
        self.assertEqual(fake.add_task_calls[0][0], "Generating audio...")
        self.assertIsNone(fake.add_task_calls[0][1], "task started determinate")
        determinate = [fields for _, fields in fake.update_calls if "total" in fields]
        self.assertEqual(
            determinate,
            [{"total": 100, "completed": 42.0}],
            f"progress_pct did not drive the determinate bar: {fake.update_calls}",
        )

    def test_single_item_stays_indeterminate_without_percent(self):
        poller = self._make_poller()(batch_total=1)
        fake = self._run_rich_once(poller, _authed_payload())
        self.assertIsNone(fake.add_task_calls[0][1])
        determinate = [fields for _, fields in fake.update_calls if "total" in fields]
        self.assertEqual(determinate, [], "percent invented without progress_pct")

    def test_stream_task_is_never_determinate(self):
        poller = self._make_poller()(batch_total=1, stream=True)
        fake = self._run_rich_once(poller, _authed_payload(progress_pct=50.0))
        self.assertEqual(fake.add_task_calls[0][0], "Streaming...")
        self.assertIsNone(fake.add_task_calls[0][1])
        determinate = [fields for _, fields in fake.update_calls if "total" in fields]
        self.assertEqual(determinate, [], "stream mode went determinate")

    def test_elapsed_only_skips_polling(self):
        poller = self._make_poller().elapsed_only()
        fake = self._run_rich_once(poller, {})
        self.assertEqual(fake.add_task_calls[0][0], "Generating audio...")


class TestNonTtyModelLoadGuard(unittest.TestCase):
    """generate_via_server's model-load prompt cannot hang non-interactive runs."""

    def _patches(self, resp):
        poller = MagicMock()
        return {
            "payload": patch(f"{_MOD}._build_generation_payload", return_value={}),
            "poller": patch(
                "qwen3_tts.interface.generate_interactive._ProgressPoller",
                return_value=poller,
            ),
            "request": patch(
                "qwen3_tts.core.http_client.server_request", return_value=resp
            ),
        }

    def _resp_503(self):
        return _status_response(
            {
                "error": "model_not_loaded",
                "model_type": "clone",
                "description": "Voice cloning",
            },
            status_code=503,
        )

    def test_non_tty_stdin_raises_tts_error(self):
        from qwen3_tts.core.config import TTSError
        from qwen3_tts.interface.generate_server import generate_via_server

        patches = self._patches(self._resp_503())
        with (
            patches["payload"],
            patches["poller"],
            patches["request"],
            patch("sys.stdin", _FakeNotTty()),
            patch(
                "builtins.input",
                side_effect=AssertionError("input() must not block on non-tty stdin"),
            ),
        ):
            with self.assertRaises(TTSError) as ctx:
                generate_via_server(["Hi"], "clone", {}, {})
        self.assertIn(
            "not a tty",
            str(ctx.exception),
            f"guard message missing the non-tty reason: {ctx.exception}",
        )

    def test_tty_stdin_still_prompts(self):
        from qwen3_tts.interface.generate_server import (
            TTSGenericError,
            generate_via_server,
        )

        patches = self._patches(self._resp_503())
        with (
            patches["payload"],
            patches["poller"],
            patches["request"],
            patch("sys.stdin", _FakeTty()),
            patch("builtins.input", return_value="n"),
        ):
            with self.assertRaises(TTSGenericError):
                generate_via_server(["Hi"], "clone", {}, {})


class TestStreamingStartsPoller(unittest.TestCase):
    """--stream starts the poller (stream=True) around the streaming request."""

    def test_generate_streaming_constructs_stream_poller(self):
        from qwen3_tts.interface import generate_server

        poller = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        with (
            patch(
                "qwen3_tts.interface.generate_interactive._ProgressPoller",
                return_value=poller,
            ) as poller_cls,
            patch("qwen3_tts.core.http_client.server_request", return_value=resp),
            patch(
                "qwen3_tts.interface.generate_server.iter_stream_chunks",
                return_value=iter([]),
            ),
        ):
            generate_server.generate_streaming(
                "Hi", "clone", {}, {}, "/tmp/out.wav", prompt_file="voice.pt"
            )
        self.assertTrue(poller_cls.called, "streaming path never started a poller")
        self.assertEqual(
            poller_cls.call_args.kwargs.get("stream"),
            True,
            f"poller not constructed in stream mode: {poller_cls.call_args}",
        )
        poller.start.assert_called()
        poller.stop.assert_called()


if __name__ == "__main__":
    unittest.main()
