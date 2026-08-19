#!/usr/bin/env python3
"""Issue #192 final pair — /create-voice-prompt's clone inference must be serialized.

Engine ``create_voice_prompt`` (``model.create_voice_clone_prompt``,
engine/inference.py) is real MLX/torch inference. Until this fix it ran
UNLOCKED while /generate, /generate-stream, /ws, the design warm-up (PR
#211) and /transcribe's ASR generate (PR #212) all serialize on
``app.state.inference_lock`` — the LAST reachable unsynchronized
concurrent inference pair of the #192 class (ml-explore/mlx#3078,
Blaizzy/mlx-audio#638, #733 — corruption manifests as EOS-never-emitted
runaways behind HTTP 200).

Contract pinned by these tests:

  * ``create_voice_prompt`` runs with ``inference_lock`` HELD (leaf
    acquisition — the handler holds nothing else, so the global
    inference_lock-outermost order holds)
  * the audio staging (b64 decode, tempfile write and
    ``load_audio_for_cloning``) runs OUTSIDE the lock AND before the
    create — blocking-but-uncontended work must not starve /generate
    (mirrors the /transcribe and /load-model load/warm-up splits)
  * the .pt save (blocking disk IO) runs AFTER the lock is released
  * decode/staging/load/create/save run in worker threads, never on the
    loop (pinned: the async rewrite must not lose the off-loop property
    the old route-level ``to_thread`` wrapper provided)
  * the clone-model reference is captured ONCE before the lock — a
    concurrent /unload-model then leaves an alive local reference,
    equal-or-better than the old inline double read
  * a held ``inference_lock`` DEFERS the create — the queueing behavior
    at the heart of the fix
  * the ``/create-voice-prompt`` route awaits the async handler directly
    (source-shape guard: wrapping it back into ``asyncio.to_thread``
    returns an un-awaited coroutine and passes every unit test here)
  * the UI /create-voice-prompt client uses ``CREATE_PROMPT_TIMEOUT_SEC``
    — the old hardcoded 60s now fails spuriously whenever creation queues
    behind an in-flight generation

No GPU, models, or running server required — locks are real asyncio.Locks,
inference is patched at the engine facade (the handler imports
function-locally, so the patched attribute is what it resolves at call time).

Run: pytest tests/test_issue192_create_prompt_serialization.py -v --tb=short
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

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi")


def _make_state():
    """Build an app.state stand-in with a REAL asyncio lock.

    The lock must be real — the tests assert on ``lock.locked()`` observed
    from inside the work functions, so a MagicMock lock would make every
    assertion hollow.
    """
    state = MagicMock()
    state.models = {"clone": MagicMock()}
    state.inference_lock = asyncio.Lock()
    return state


def _prompt_req(no_transcript=False):
    req = MagicMock()
    req.name = "test_voice"
    # Valid base64 — the handler 400s before any patched engine call otherwise.
    req.audio_base64 = base64.b64encode(b"fake-audio-bytes").decode()
    req.no_transcript = no_transcript
    req.transcript = "hello world"
    return req


@_skip
class TestCreatePromptSerialization(unittest.TestCase):
    """/create-voice-prompt must create under inference_lock, stage without it."""

    def _run_handler(self, state):
        """Run the handler with engine seams patched; return what happened.

        ``events`` records the ORDER of engine calls (load before create
        before save) plus lock/thread state observed from inside each call.
        """
        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

        events = []

        def _load_audio(*args, **kwargs):
            events.append(
                {
                    "kind": "load_audio",
                    "lock_held": state.inference_lock.locked(),
                    "in_worker_thread": threading.current_thread()
                    is not threading.main_thread(),
                }
            )
            return ("fake-audio", 24000)

        def _create(*args, **kwargs):
            events.append(
                {
                    "kind": "create",
                    "args": args,
                    "kwargs": kwargs,
                    "lock_held": state.inference_lock.locked(),
                    "in_worker_thread": threading.current_thread()
                    is not threading.main_thread(),
                }
            )
            return MagicMock()

        def _save(*args, **kwargs):
            events.append(
                {
                    "kind": "save",
                    "args": args,
                    "lock_held": state.inference_lock.locked(),
                    "in_worker_thread": threading.current_thread()
                    is not threading.main_thread(),
                }
            )

        with (
            patch(
                "qwen3_tts.core.engine.load_audio_for_cloning",
                side_effect=_load_audio,
            ),
            patch("qwen3_tts.core.engine.create_voice_prompt", side_effect=_create),
            patch("qwen3_tts.server.app_prompts._save_pt", side_effect=_save),
            patch("qwen3_tts.core.engine.clear_voice_prompt_cache"),
        ):
            result = asyncio.run(handle_create_voice_prompt(state, _prompt_req()))
        return result, events

    def test_create_runs_with_inference_lock_held(self):
        state = _make_state()
        result, events = self._run_handler(state)

        self.assertEqual(result, {"status": "created", "name": "test_voice"})
        by_kind = {e["kind"]: e for e in events}
        self.assertTrue(
            by_kind["create"]["lock_held"],
            "clone inference must run with inference_lock held — an unlocked "
            "create is the #192 concurrency bug",
        )

    def test_audio_load_runs_outside_lock_and_before_create(self):
        """Audio decode is blocking-but-uncontended: unlocked (must not
        starve /generate), and complete before the create queues for the
        lock."""
        state = _make_state()
        result, events = self._run_handler(state)

        self.assertEqual(result, {"status": "created", "name": "test_voice"})
        self.assertEqual(
            [e["kind"] for e in events],
            ["load_audio", "create", "save"],
            "the audio load must run first, then the create, then the save",
        )
        self.assertFalse(
            events[0]["lock_held"],
            "load_audio_for_cloning (audio decode) must run outside "
            "inference_lock",
        )
        self.assertTrue(
            events[1]["lock_held"],
            "the create itself must run with inference_lock held",
        )

    def test_save_runs_after_lock_released(self):
        state = _make_state()
        _, events = self._run_handler(state)

        by_kind = {e["kind"]: e for e in events}
        self.assertFalse(
            by_kind["save"]["lock_held"],
            "the .pt save is disk IO and must not hold the GPU lock",
        )

    def test_create_runs_off_event_loop_thread(self):
        """The handler must never block the event loop: the create executes
        in a worker thread, not the loop's thread."""
        state = _make_state()
        _, events = self._run_handler(state)

        by_kind = {e["kind"]: e for e in events}
        self.assertTrue(
            by_kind["create"]["in_worker_thread"],
            "clone inference must run via asyncio.to_thread, not on the loop",
        )

    def test_create_uses_captured_clone_model_reference(self):
        state = _make_state()
        _, events = self._run_handler(state)

        by_kind = {e["kind"]: e for e in events}
        self.assertIs(
            by_kind["create"]["args"][0],
            state.models["clone"],
            "create_voice_prompt must receive the clone-model reference "
            "captured once before the lock — a concurrent /unload-model "
            "leaves an alive local reference",
        )

    def test_decode_and_staging_run_off_event_loop_thread(self):
        """b64decode + the tempfile write are blocking CPU/file IO — they
        must run in worker threads, not on the loop (async-offload policy,
        tests/test_server_async_offload.py; the old route-level to_thread
        wrapper used to provide this and the async rewrite must not lose
        it)."""
        import tempfile as tempfile_mod

        from qwen3_tts.server.app_prompts import handle_create_voice_prompt

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
                "qwen3_tts.core.engine.load_audio_for_cloning",
                return_value=("fake-audio", 24000),
            ),
            patch(
                "qwen3_tts.core.engine.create_voice_prompt",
                return_value=MagicMock(),
            ),
            patch("qwen3_tts.server.app_prompts._save_pt"),
            patch("qwen3_tts.core.engine.clear_voice_prompt_cache"),
        ):
            result = asyncio.run(handle_create_voice_prompt(state, _prompt_req()))

        self.assertEqual(result, {"status": "created", "name": "test_voice"})
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

    def test_create_deferred_while_generation_holds_lock(self):
        """The heart of the fix: a held inference_lock defers the clone
        inference until the holder releases."""

        async def _scenario():
            from qwen3_tts.server.app_prompts import handle_create_voice_prompt

            state = _make_state()
            staged = []
            seen = []

            def _load_audio(*args, **kwargs):
                staged.append({"lock_held": state.inference_lock.locked()})
                return ("fake-audio", 24000)

            def _create(*args, **kwargs):
                seen.append({"lock_held": state.inference_lock.locked()})
                return MagicMock()

            # The patches must OUTLIVE the lock release: the handler still
            # has post-lock work (_save_pt) after it acquires the lock, so
            # nesting them inside the async with would unpatch before the
            # task finishes (unlike the transcribe template, whose handler
            # returns straight off its generate).
            with (
                patch(
                    "qwen3_tts.core.engine.load_audio_for_cloning",
                    side_effect=_load_audio,
                ),
                patch(
                    "qwen3_tts.core.engine.create_voice_prompt",
                    side_effect=_create,
                ),
                patch("qwen3_tts.server.app_prompts._save_pt"),
                patch("qwen3_tts.core.engine.clear_voice_prompt_cache"),
            ):
                # Simulate an in-flight generation holding the GPU lock.
                async with state.inference_lock:
                    task = asyncio.ensure_future(
                        handle_create_voice_prompt(state, _prompt_req())
                    )
                    # Let the handler run to its lock-acquire point.
                    await asyncio.sleep(0.05)
                    assert seen == [], (
                        "clone inference must not run while another holder "
                        "has inference_lock"
                    )
                    assert staged and staged[0]["lock_held"], (
                        "the audio load must run while another holder has "
                        "inference_lock — staging is not gated on the GPU "
                        "lock, so it proceeds even while the lock is "
                        "unavailable"
                    )
                result = await asyncio.wait_for(task, timeout=5)
            return result, seen

        result, seen = asyncio.run(_scenario())

        self.assertEqual(result, {"status": "created", "name": "test_voice"})
        self.assertEqual(len(seen), 1)
        self.assertTrue(
            seen[0]["lock_held"],
            "after the holder releases, the create runs with the lock held",
        )


@_skip
@unittest.skipUnless(HAS_TORCH, "requires torch")
class TestSavePtHelper(unittest.TestCase):
    """``_save_pt`` must serialize via torch.save (lazy torch import)."""

    def test_save_pt_calls_torch_save(self):
        from qwen3_tts.server.app_prompts import _save_pt

        sentinel = object()
        with patch("torch.save") as mock_save:
            _save_pt(sentinel, "/tmp/x.pt")

        mock_save.assert_called_once_with(sentinel, "/tmp/x.pt")


@_skip
class TestCreatePromptRouteShape(unittest.TestCase):
    """/create-voice-prompt must await the now-async handler directly.

    Source-shape guard (precedent: tests/test_p3_p4_remediation.py) —
    wrapping ``handle_create_voice_prompt`` back into ``asyncio.to_thread``
    would return an un-awaited coroutine from the endpoint and pass every
    unit test above; only the source assertion catches it.
    """

    def test_endpoint_awaits_handler_directly(self):
        from qwen3_tts.server import app as app_module

        src = inspect.getsource(app_module.create_voice_prompt_endpoint)
        self.assertIn(
            "await handle_create_voice_prompt(",
            src,
            "/create-voice-prompt route must await the async handler directly",
        )
        self.assertNotIn(
            "to_thread(handle_create_voice_prompt",
            src,
            "/create-voice-prompt route must not dispatch the async handler "
            "through asyncio.to_thread",
        )


@_skip
class TestCreatePromptTimeoutDrift(unittest.TestCase):
    """The UI /create-voice-prompt client must use CREATE_PROMPT_TIMEOUT_SEC.

    /create-voice-prompt now queues behind in-flight generations (whole-text
    worst case ~660s documented) before the prompt creation itself runs, so
    the old hardcoded 60s fails spuriously — same defect class as the
    /load-model 120s and /transcribe 60s clients caught for PRs #211/#212.
    """

    def test_ui_client_uses_shared_constant(self):
        from qwen3_tts.core.http_client import CREATE_PROMPT_TIMEOUT_SEC

        # Must cover queue-behind-one-generation + the creation itself.
        self.assertGreater(CREATE_PROMPT_TIMEOUT_SEC, 660)

        from qwen3_tts.interface.ui import voice_management

        src = inspect.getsource(voice_management.create_voice_prompt)
        self.assertIn(
            "CREATE_PROMPT_TIMEOUT_SEC",
            src,
            "the torch-path create_voice_prompt must use the shared "
            "/create-voice-prompt timeout constant",
        )
        self.assertNotIn(
            "timeout=60",
            src,
            "the torch-path create_voice_prompt must not hardcode the old "
            "60s /create-voice-prompt timeout",
        )


if __name__ == "__main__":
    unittest.main()
