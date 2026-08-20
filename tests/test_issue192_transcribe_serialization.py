#!/usr/bin/env python3
"""Issue #192 follow-up — /transcribe's ASR generate must be serialized.

mlx-whisper's ``generate`` (engine/asr.py, via ``transcribe_audio``) is real
MLX inference. Until this fix it ran UNLOCKED while /generate, /generate-stream
and /ws all serialize on ``app.state.inference_lock`` — the second reachable
unsynchronized concurrent MLX inference pair, after the design warm-up closed
in PR #211. ``/create-voice-prompt`` (``create_voice_clone_prompt``,
engine/inference.py) was the last of the class
(ml-explore/mlx#3078, Blaizzy/mlx-audio#638, #733 — corruption
manifests as EOS-never-emitted runaways behind HTTP 200) and is
serialized now (tests/test_issue192_create_prompt_serialization.py) —
with it, all MLX inference reachable through the API serializes on
``app.state.inference_lock``.

Contract pinned by these tests:

  * ``transcribe_audio`` runs with ``inference_lock`` HELD (leaf acquisition —
    holds nothing else, so the global inference_lock-outermost order holds)
  * the lazy ASR model load (first /transcribe without /load-asr) runs
    OUTSIDE the lock AND before the generate — minutes of download + weight
    construction must not starve /generate (mirrors the /load-model
    load/warm-up split of PR #211; ``preload_asr_model`` never preloads on
    the MLX backend, so the lazy load is a real path)
  * the pre-load is skipped when ASR is already loaded
  * decode/tempfile/load/generate run in worker threads, never on the
    loop (pinned: the async rewrite must not lose the off-loop property
    the old route-level ``to_thread`` wrapper provided)
  * a held ``inference_lock`` DEFERS the transcription — the queueing
    behavior at the heart of the fix
  * the ``/transcribe`` route awaits the async handler directly (source-shape
    guard: wrapping it back into ``asyncio.to_thread`` returns an un-awaited
    coroutine and passes every unit test here)
  * the UI /transcribe client uses ``TRANSCRIBE_TIMEOUT_SEC`` — the old
    hardcoded 60s now fails spuriously whenever the transcription queues
    behind an in-flight generation

No GPU, models, or running server required — locks are real asyncio.Locks,
inference is patched at the engine facade (the handler imports
function-locally, so the patched attribute is what it resolves at call time).

Run: pytest tests/test_issue192_transcribe_serialization.py -v --tb=short
"""

import asyncio
import base64
import inspect
import os
import threading
import unittest
from unittest.mock import MagicMock, patch

try:
    import fastapi  # noqa: F401

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi")


def _make_state():
    """Build an app.state stand-in with a REAL asyncio lock.

    The lock must be real — the tests assert on ``lock.locked()`` observed
    from inside the work functions, so a MagicMock lock would make every
    assertion hollow.
    """
    state = MagicMock()
    state.inference_lock = asyncio.Lock()
    return state


def _audio_req(language="en"):
    req = MagicMock()
    # Valid base64 — the handler 400s before any patched engine call otherwise.
    req.audio_base64 = base64.b64encode(b"fake-audio-bytes").decode()
    req.language = language
    return req


@_skip
class TestTranscribeSerialization(unittest.TestCase):
    """/transcribe must generate under inference_lock, load without it."""

    def _run_handler(self, state, asr_preloaded=True):
        """Run the handler with engine seams patched; return what happened.

        ``events`` records the ORDER of engine calls (load before generate)
        plus lock/thread state observed from inside each call.
        """
        from qwen3_tts.server.app_models import handle_transcribe

        events = []

        def _load(*args, **kwargs):
            events.append(
                {
                    "kind": "load",
                    "lock_held": state.inference_lock.locked(),
                    "in_worker_thread": threading.current_thread()
                    is not threading.main_thread(),
                }
            )

        def _transcribe(*args, **kwargs):
            events.append(
                {
                    "kind": "generate",
                    "args": args,
                    "kwargs": kwargs,
                    "lock_held": state.inference_lock.locked(),
                    "in_worker_thread": threading.current_thread()
                    is not threading.main_thread(),
                }
            )
            return "hello transcript"

        with (
            patch(
                "qwen3_tts.core.engine.is_asr_loaded",
                return_value=asr_preloaded,
            ),
            patch("qwen3_tts.core.engine.load_asr_model", side_effect=_load),
            patch(
                "qwen3_tts.core.engine.transcribe_audio",
                side_effect=_transcribe,
            ),
        ):
            result = asyncio.run(handle_transcribe(state, _audio_req()))
        return result, events

    def test_generate_runs_with_inference_lock_held(self):
        state = _make_state()
        result, events = self._run_handler(state)

        self.assertEqual(result, {"transcript": "hello transcript"})
        kinds = [e["kind"] for e in events]
        self.assertEqual(
            kinds,
            ["generate"],
            "with ASR preloaded only the generate should run",
        )
        self.assertTrue(
            events[0]["lock_held"],
            "ASR generate must run with inference_lock held — an unlocked "
            "generate is the #192 concurrency bug",
        )

    def test_lazy_asr_load_runs_outside_lock_and_before_generate(self):
        """First-use load: unlocked (must not starve /generate), and complete
        before the generate acquires the lock."""
        state = _make_state()
        result, events = self._run_handler(state, asr_preloaded=False)

        self.assertEqual(result["transcript"], "hello transcript")
        kinds = [e["kind"] for e in events]
        self.assertEqual(
            kinds,
            ["load", "generate"],
            "the lazy ASR model load must run, and before the generate",
        )
        self.assertFalse(
            events[0]["lock_held"],
            "ASR model load (download + weight construction) must run "
            "outside inference_lock",
        )
        self.assertTrue(
            events[1]["lock_held"],
            "the generate itself must run with inference_lock held",
        )

    def test_asr_load_skipped_when_already_loaded(self):
        state = _make_state()
        _, events = self._run_handler(state, asr_preloaded=True)

        self.assertNotIn(
            "load",
            [e["kind"] for e in events],
            "a pre-loaded ASR model must not pay a load round-trip (the "
            "is_asr_loaded guard keeps the common path lock-free)",
        )

    def test_generate_runs_off_event_loop_thread(self):
        """The handler must never block the event loop: the generate executes
        in a worker thread, not the loop's thread."""
        state = _make_state()
        _, events = self._run_handler(state)

        self.assertTrue(
            events[0]["in_worker_thread"],
            "ASR generate must run via asyncio.to_thread, not on the loop",
        )

    def test_decode_and_staging_run_off_event_loop_thread(self):
        """b64decode + the tempfile write are blocking CPU/file IO — they
        must run in worker threads, not on the loop (async-offload policy,
        tests/test_server_async_offload.py; the old route-level to_thread
        wrapper used to provide this and the async rewrite must not lose
        it)."""
        import tempfile as tempfile_mod

        from qwen3_tts.server.app_models import handle_transcribe

        state = _make_state()
        events = []

        def _decode(*args, **kwargs):
            events.append(
                {
                    "kind": "decode",
                    "in_worker_thread": threading.current_thread()
                    is not threading.main_thread(),
                }
            )
            return b"decoded-bytes"

        # A real path so the handler's chmod/exists/remove behave; its
        # finally cleans it up (addCleanup covers an assertion failure).
        fd, real_path = tempfile_mod.mkstemp(suffix=".wav")
        os.close(fd)
        self.addCleanup(
            lambda: os.path.exists(real_path) and os.remove(real_path)
        )

        class _FakeTmp:
            name = real_path

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def write(self, data):
                events.append(
                    {
                        "kind": "write",
                        "in_worker_thread": threading.current_thread()
                        is not threading.main_thread(),
                    }
                )

        with (
            patch("base64.b64decode", side_effect=_decode),
            patch("tempfile.NamedTemporaryFile", return_value=_FakeTmp()),
            patch(
                "qwen3_tts.core.engine.is_asr_loaded", return_value=True
            ),
            patch("qwen3_tts.core.engine.load_asr_model"),
            patch(
                "qwen3_tts.core.engine.transcribe_audio",
                return_value="offloop transcript",
            ),
        ):
            result = asyncio.run(handle_transcribe(state, _audio_req()))

        self.assertEqual(result, {"transcript": "offloop transcript"})
        by_kind = {e["kind"]: e for e in events}
        self.assertEqual(
            set(by_kind), {"decode", "write"}, f"unexpected events: {events}"
        )
        for kind in ("decode", "write"):
            self.assertTrue(
                by_kind[kind]["in_worker_thread"],
                f"{kind} must run via asyncio.to_thread, not on the loop",
            )
        self.assertFalse(
            os.path.exists(real_path),
            "the handler's finally must still remove the tempfile",
        )

    def test_generate_deferred_while_generation_holds_lock(self):
        """The heart of the fix: a held inference_lock defers the ASR
        generate until the holder releases."""

        async def _scenario():
            from qwen3_tts.server.app_models import handle_transcribe

            state = _make_state()
            seen = []

            def _transcribe(*args, **kwargs):
                seen.append({"lock_held": state.inference_lock.locked()})
                return "deferred transcript"

            # Simulate an in-flight generation holding the GPU lock.
            async with state.inference_lock:
                with (
                    patch(
                        "qwen3_tts.core.engine.is_asr_loaded",
                        return_value=True,
                    ),
                    patch("qwen3_tts.core.engine.load_asr_model"),
                    patch(
                        "qwen3_tts.core.engine.transcribe_audio",
                        side_effect=_transcribe,
                    ),
                ):
                    task = asyncio.ensure_future(
                        handle_transcribe(state, _audio_req())
                    )
                    # Let the handler run to its lock-acquire point.
                    await asyncio.sleep(0.05)
                    assert seen == [], (
                        "ASR generate must not run while another holder "
                        "has inference_lock"
                    )
            result = await asyncio.wait_for(task, timeout=5)
            return result, seen

        result, seen = asyncio.run(_scenario())

        self.assertEqual(result, {"transcript": "deferred transcript"})
        self.assertEqual(len(seen), 1)
        self.assertTrue(
            seen[0]["lock_held"],
            "after the holder releases, the generate runs with the lock held",
        )


@_skip
class TestStageTempfileFailureCleanup(unittest.TestCase):
    """A failed stage must not leak its tempfile.

    The handler's finally never sees the path until staging returns, so
    staging owns its own cleanup on both failure modes (the pre-#192
    handler leaked the write-failure case — `tmp_path` was assigned only
    after a successful write).
    """

    def _fake_tmp(self, fail_on):
        """A NamedTemporaryFile stand-in on a REAL path (chmod/exists/
        remove behave), failing on `fail_on` in {'write', 'chmod'}."""
        import tempfile as tempfile_mod

        fd, real_path = tempfile_mod.mkstemp(suffix=".wav")
        os.close(fd)
        # Covers an assertion failure inside the test body.
        self.addCleanup(
            lambda: os.path.exists(real_path) and os.remove(real_path)
        )

        class _FakeTmp:
            name = real_path

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def write(self, data):
                if fail_on == "write":
                    raise OSError("simulated disk full")

        return _FakeTmp(), real_path

    def test_write_failure_removes_tempfile(self):
        from qwen3_tts.server.app_models import _stage_tempfile

        fake, path = self._fake_tmp("write")
        with patch("tempfile.NamedTemporaryFile", return_value=fake):
            with self.assertRaises(OSError):
                _stage_tempfile(b"data")

        self.assertFalse(
            os.path.exists(path),
            "a failed write must remove the tempfile (pre-#192 leak)",
        )

    def test_chmod_failure_removes_tempfile(self):
        from qwen3_tts.server.app_models import _stage_tempfile

        fake, path = self._fake_tmp("chmod")
        with (
            patch("tempfile.NamedTemporaryFile", return_value=fake),
            patch(
                "qwen3_tts.server.app_models.os.chmod",
                side_effect=OSError("simulated chmod failure"),
            ),
        ):
            with self.assertRaises(OSError):
                _stage_tempfile(b"data")

        self.assertFalse(
            os.path.exists(path),
            "a failed chmod must remove the tempfile",
        )


@_skip
class TestTranscribeRouteShape(unittest.TestCase):
    """/transcribe must await the now-async handler directly.

    Source-shape guard (precedent: tests/test_p3_p4_remediation.py) —
    wrapping ``handle_transcribe`` back into ``asyncio.to_thread`` would
    return an un-awaited coroutine from the endpoint and pass every unit
    test above; only the source assertion catches it.
    """

    def test_endpoint_awaits_handler_directly(self):
        from qwen3_tts.server import app as app_module

        src = inspect.getsource(app_module.transcribe)
        self.assertIn(
            "await handle_transcribe(",
            src,
            "/transcribe route must await the async handler directly",
        )
        self.assertNotIn(
            "to_thread(handle_transcribe",
            src,
            "/transcribe route must not dispatch the async handler through "
            "asyncio.to_thread",
        )


@_skip
class TestTranscribeTimeoutDrift(unittest.TestCase):
    """The UI /transcribe client must use TRANSCRIBE_TIMEOUT_SEC.

    /transcribe now queues behind in-flight generations (whole-text worst
    case ~660s documented), so the old hardcoded 60s fails spuriously —
    same defect class as the /load-model 120s clients caught for PR #211.
    """

    def test_ui_client_uses_shared_constant(self):
        from qwen3_tts.core.http_client import TRANSCRIBE_TIMEOUT_SEC

        # Must cover queue-behind-one-generation + the ASR generate itself.
        self.assertGreater(TRANSCRIBE_TIMEOUT_SEC, 660)

        from qwen3_tts.interface.ui import voice_management

        src = inspect.getsource(voice_management.auto_transcribe_audio)
        self.assertIn(
            "TRANSCRIBE_TIMEOUT_SEC",
            src,
            "auto_transcribe_audio must use the shared /transcribe timeout "
            "constant",
        )
        self.assertNotIn(
            "timeout=60",
            src,
            "auto_transcribe_audio must not hardcode the old 60s "
            "/transcribe timeout",
        )


if __name__ == "__main__":
    unittest.main()
