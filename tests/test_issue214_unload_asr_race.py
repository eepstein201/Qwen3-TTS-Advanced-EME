#!/usr/bin/env python3
"""Issue #214 item 2 -- the /unload-asr race.

Two halves, both defects in ``qwen3_tts/core/engine/asr.py`` /
``qwen3_tts/server/app_models.py::handle_transcribe``:

  * Half A: ``unload_asr_model()`` mutates the shared ``_asr_model_mlx`` /
    ``_asr_model_torch`` globals with NO lock, even though the same module
    defines ``_asr_lock`` and every loader (``_ensure_asr_torch_loaded``,
    ``load_asr_model``) takes it. Unload is the odd one out.
  * Half B: ``handle_transcribe`` has a check-then-use window --
    ``is_asr_loaded()`` is checked BEFORE ``inference_lock`` is acquired,
    then ``transcribe_audio`` (which lazily reloads on first call) runs
    INSIDE the lock. If ``/unload-asr`` lands in that window, the multi-
    minute model reload happens while ``inference_lock`` is held, starving
    ``/generate``. The unlocked-load split from #192
    (tests/test_issue192_transcribe_serialization.py) is intentional and
    must be preserved -- the fix is a post-lock ``is_asr_loaded()`` recheck
    that bails with a retryable error, NOT moving the load inside the lock.

Contract pinned by these tests:

  * ``unload_asr_model()`` acquires ``_asr_lock`` -- proven both by
    observing it block behind a lock held on another thread, and by
    substituting an instrumented lock and observing real acquire/release
    calls (not just "a lock object exists somewhere").
  * ``handle_transcribe`` re-checks ``is_asr_loaded()`` AFTER acquiring
    ``inference_lock``. When that recheck finds the model gone (an unload
    raced in), the handler must raise a retryable HTTPException and must
    NEVER call ``load_asr_model`` or ``transcribe_audio`` while holding
    the lock.
  * A genuine first-use load (never loaded, still absent at the pre-lock
    check) still loads OUTSIDE ``inference_lock`` and completes before the
    generate -- the recheck must not regress the #192 split.

No GPU, models, or running server required -- locks are real
``threading.Lock``/``asyncio.Lock`` instances; inference is patched at the
engine facade (the handler imports function-locally, so the patched
attribute is what it resolves at call time).

Run: pytest tests/test_issue214_unload_asr_race.py -v --tb=short
"""

import asyncio
import base64
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

    The lock must be real -- tests assert on ``lock.locked()`` observed
    from inside the patched engine calls, so a MagicMock lock would make
    every assertion hollow.
    """
    state = MagicMock()
    state.inference_lock = asyncio.Lock()
    return state


def _audio_req(language="en"):
    req = MagicMock()
    # Valid base64 -- the handler 400s before any patched engine call otherwise.
    req.audio_base64 = base64.b64encode(b"fake-audio-bytes").decode()
    req.language = language
    return req


class _RecordingLock:
    """Wraps a real ``threading.Lock``, recording every acquire/release.

    Delegates all blocking semantics to the wrapped real lock -- this
    observes calls, it does not fake concurrency safety. Substituting this
    for the module's ``_asr_lock`` lets a test assert "the code under test
    actually called acquire/release on this object" without depending on
    thread-scheduling timing.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.acquire_calls = 0
        self.release_calls = 0

    def acquire(self, *args, **kwargs):
        self.acquire_calls += 1
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self.release_calls += 1
        self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def locked(self):
        return self._lock.locked()


@_skip
class TestUnloadAsrModelTakesLock(unittest.TestCase):
    """Half A: unload_asr_model() must serialize on _asr_lock."""

    def test_unload_acquires_and_releases_the_recorded_lock(self):
        """Substitute an instrumented lock for the module's _asr_lock and
        assert unload_asr_model() actually calls acquire()/release() on it
        -- not merely that a lock attribute exists on the module."""
        from qwen3_tts.core.engine import asr as asr_module

        recorder = _RecordingLock()
        # Register restores BEFORE mutating: a failed assertion below would
        # otherwise skip cleanup and leak state into the next test in
        # full-suite order. Restore the ORIGINAL values, not a hardcoded None.
        self.addCleanup(
            setattr, asr_module, "_asr_model_mlx", asr_module._asr_model_mlx
        )
        self.addCleanup(
            setattr, asr_module, "_asr_model_torch", asr_module._asr_model_torch
        )
        asr_module._asr_model_mlx = "loaded-mlx-model"
        asr_module._asr_model_torch = None

        with patch.object(asr_module, "_asr_lock", recorder):
            asr_module.unload_asr_model()

        self.assertGreaterEqual(
            recorder.acquire_calls,
            1,
            "unload_asr_model() never acquired _asr_lock -- it mutates "
            "the shared ASR globals unlocked (issue #214 half A)",
        )
        self.assertEqual(
            recorder.acquire_calls,
            recorder.release_calls,
            "every acquire must be matched by a release (no held lock "
            "leaked out of unload_asr_model())",
        )

    def test_unload_blocks_while_lock_held_by_another_thread(self):
        """Behavioral proof: with _asr_lock held on another thread, a call
        to unload_asr_model() must block until it is released -- it must
        not complete "for free" by skipping the lock entirely."""
        from qwen3_tts.core.engine import asr as asr_module

        holder_ready = threading.Event()
        release_holder = threading.Event()

        def _hold_lock():
            with asr_module._asr_lock:
                holder_ready.set()
                # Bounded wait, not a sleep loop: this thread's only job
                # is to keep the lock held until the test releases it.
                release_holder.wait(timeout=5)

        holder = threading.Thread(target=_hold_lock, daemon=True)
        holder.start()
        self.addCleanup(lambda: (release_holder.set(), holder.join(timeout=5)))

        self.assertTrue(
            holder_ready.wait(timeout=5),
            "test setup failed: holder thread never acquired _asr_lock",
        )

        unload_done = threading.Event()

        def _unload():
            asr_module.unload_asr_model()
            unload_done.set()

        unloader = threading.Thread(target=_unload, daemon=True)
        unloader.start()
        self.addCleanup(lambda: unloader.join(timeout=5))

        completed_while_held = unload_done.wait(timeout=0.5)
        self.assertFalse(
            completed_while_held,
            "unload_asr_model() completed while another thread held "
            "_asr_lock -- it is not taking the lock at all (issue #214 "
            "half A)",
        )

        release_holder.set()
        self.assertTrue(
            unload_done.wait(timeout=5),
            "unload_asr_model() never completed after _asr_lock was "
            "released",
        )


@_skip
class TestTranscribeUnloadRace(unittest.TestCase):
    """Half B: handle_transcribe must recheck is_asr_loaded() post-lock."""

    def test_post_lock_recheck_detects_concurrent_unload(self):
        """Simulate /unload-asr landing between the pre-lock ensure and the
        locked transcribe: is_asr_loaded() reports True on the pre-lock
        check, then False once inference_lock is held. The handler must
        bail with a retryable error and must NEVER call load_asr_model or
        transcribe_audio while holding the lock -- silently reloading
        there is exactly the starvation bug from issue #214 half B."""
        from fastapi import HTTPException

        from qwen3_tts.server.app_models import handle_transcribe

        state = _make_state()
        transcribe_calls = []
        load_calls = []
        # Lock state at EACH is_asr_loaded() call. Without this the test cannot
        # tell a post-lock recheck from a pre-lock one -- mutation-proven: a
        # handler doing the second check immediately BEFORE
        # `async with inference_lock` passed every other assertion here.
        # Asserting call_count >= 2 pins that a recheck happened, not WHERE.
        recheck_lock_states = []

        def _is_asr_loaded(*args, **kwargs):
            recheck_lock_states.append(state.inference_lock.locked())
            return len(recheck_lock_states) == 1  # True pre-lock, False after

        def _transcribe(*args, **kwargs):
            transcribe_calls.append(state.inference_lock.locked())
            return "should never run"

        def _load(*args, **kwargs):
            load_calls.append(state.inference_lock.locked())

        with (
            patch(
                "qwen3_tts.core.engine.is_asr_loaded",
                side_effect=_is_asr_loaded,
            ) as mock_is_loaded,
            patch("qwen3_tts.core.engine.load_asr_model", side_effect=_load),
            patch(
                "qwen3_tts.core.engine.transcribe_audio",
                side_effect=_transcribe,
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(handle_transcribe(state, _audio_req()))

        self.assertGreaterEqual(
            mock_is_loaded.call_count,
            2,
            "handle_transcribe must recheck is_asr_loaded() AFTER "
            "acquiring inference_lock, not just once before it -- "
            "issue #214 half B",
        )
        # WHERE the recheck happens is the whole point: first check unlocked
        # (the ensure), second check with inference_lock HELD.
        self.assertEqual(
            recheck_lock_states,
            [False, True],
            "the recheck must run with inference_lock HELD -- a second check "
            "taken before the lock reintroduces the very window it closes "
            f"(observed lock states: {recheck_lock_states})",
        )
        self.assertEqual(
            transcribe_calls,
            [],
            "transcribe_audio must never run once the post-lock recheck "
            "finds ASR unloaded",
        )
        self.assertEqual(
            load_calls,
            [],
            "the handler must bail with a retryable error, not silently "
            "reload the model while inference_lock is held",
        )
        detail = ctx.exception.detail
        self.assertIsInstance(
            detail, dict, f"expected a structured error detail, got: {detail!r}"
        )
        self.assertEqual(
            detail.get("recovery"),
            "retry",
            f"expected a retryable error classification, got: {detail}",
        )

    def test_genuine_first_use_load_still_runs_outside_lock_with_recheck(self):
        """A REAL first-use load (ASR absent at the pre-lock check, present
        by the post-lock recheck because the load itself succeeded) must
        still: load outside inference_lock, complete before the generate,
        and let the generate proceed -- the half-B recheck must not
        regress the #192 unlocked-load split
        (tests/test_issue192_transcribe_serialization.py)."""
        from qwen3_tts.server.app_models import handle_transcribe

        state = _make_state()
        events = []

        def _load(*args, **kwargs):
            events.append(("load", state.inference_lock.locked()))

        def _transcribe(*args, **kwargs):
            events.append(("generate", state.inference_lock.locked()))
            return "hello transcript"

        with (
            patch(
                "qwen3_tts.core.engine.is_asr_loaded",
                side_effect=[False, True],
            ) as mock_is_loaded,
            patch("qwen3_tts.core.engine.load_asr_model", side_effect=_load),
            patch(
                "qwen3_tts.core.engine.transcribe_audio",
                side_effect=_transcribe,
            ),
        ):
            result = asyncio.run(handle_transcribe(state, _audio_req()))

        self.assertEqual(result, {"transcript": "hello transcript"})
        self.assertGreaterEqual(
            mock_is_loaded.call_count,
            2,
            "the post-lock recheck must run even on the genuine "
            "first-use load path, not only on the race path",
        )
        kinds = [kind for kind, _ in events]
        self.assertEqual(
            kinds,
            ["load", "generate"],
            "the load must still run, and still run before the generate",
        )
        self.assertFalse(
            events[0][1],
            "the first-use ASR load must still run outside inference_lock "
            "(minutes of download/weight construction must not starve "
            "/generate)",
        )
        self.assertTrue(
            events[1][1],
            "the generate must still run with inference_lock held",
        )


@_skip
class TestUnloadSerializesOnInferenceLock(unittest.TestCase):
    """The structural half: /unload-asr must take ``inference_lock``.

    ``_asr_lock`` + a post-lock recheck only NARROW the window -- an unload
    can still land between the recheck and ``transcribe_audio``'s own
    ``if _asr_model_mlx is None`` (asr.py:168), or before
    ``_transcribe_torch``'s unconditional ``_ensure_asr_torch_loaded()``
    (asr.py:190), and lazily rebuild the model INSIDE the lock. Making the
    unload itself acquire ``inference_lock`` closes it structurally rather
    than shrinking it.

    It also closes the SAME defect on the ICL echo-trim path
    (``inference.py:1155`` checks ``is_asr_loaded()`` then transcribes at
    ``:1163`` with no recheck) -- that runs inside ``run_inference`` with
    ``inference_lock`` already held, so an unload that must take the lock
    can never interleave with it.
    """

    def test_unload_asr_route_holds_inference_lock(self):
        """The route must hold inference_lock while the unload runs."""
        from qwen3_tts.server.app_models import handle_unload_asr

        state = _make_state()
        observed = {}

        def _fake_unload():
            observed["lock_held"] = state.inference_lock.locked()

        async def _drive():
            # Mirror the route body (app.py): lock, then to_thread the handler.
            async with state.inference_lock:
                return await asyncio.to_thread(handle_unload_asr, state)

        with patch("qwen3_tts.core.engine.unload_asr_model", side_effect=_fake_unload):
            asyncio.run(_drive())

        self.assertTrue(
            observed.get("lock_held"),
            "unload ran without inference_lock held; it can then interleave "
            "with an in-flight /transcribe or /generate echo-trim probe",
        )

    def test_route_source_acquires_inference_lock(self):
        """Pin the real route, not just a stand-in of it.

        The test above drives a local mirror of the route body, which would
        keep passing if app.py never took the lock.

        Parsed as an AST, NOT a substring search: the route's docstring
        explains the lock at length, so ``"inference_lock" in source`` stays
        true even with the ``async with`` deleted. Mutation-proven -- that
        weaker form let a lock-less route pass all nine tests in this module.
        Only an actual ``async with ... inference_lock`` node counts.
        """
        import ast
        import inspect
        import textwrap

        from qwen3_tts.server import app as app_module

        tree = ast.parse(textwrap.dedent(inspect.getsource(app_module.unload_asr)))

        async_with_targets = [
            ast.unparse(item.context_expr)
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncWith)
            for item in node.items
        ]
        self.assertTrue(
            any("inference_lock" in target for target in async_with_targets),
            "/unload-asr must acquire inference_lock via `async with` so an "
            "unload cannot interleave with in-flight inference (#214 item 2). "
            f"async-with contexts found: {async_with_targets or 'NONE'}",
        )


@_skip
class TestUnloadAsrClientTimeout(unittest.TestCase):
    """Blocking the unload on inference_lock makes the old 60 s client wrong."""

    def test_unload_timeout_constant_is_defined_and_generous(self):
        """Mirrors LOAD_MODEL_TIMEOUT_SEC / TRANSCRIBE_TIMEOUT_SEC (=900).

        An unload now queues behind a whole generation, so a 60 s client
        read timeout fails spuriously while the server completes the unload
        anyway -- the UI then shows an error for work that succeeded.
        """
        from qwen3_tts.core.http_client import (
            LOAD_MODEL_TIMEOUT_SEC,
            UNLOAD_ASR_TIMEOUT_SEC,
        )

        self.assertEqual(UNLOAD_ASR_TIMEOUT_SEC, 900)
        self.assertEqual(
            UNLOAD_ASR_TIMEOUT_SEC,
            LOAD_MODEL_TIMEOUT_SEC,
            "keep the queue-behind-inference timeouts in lockstep",
        )

    def test_ui_uses_the_constant_not_a_literal(self):
        """Drift guard: the UI must not reintroduce a hardcoded timeout."""
        import inspect

        from qwen3_tts.interface.ui import model_management

        src = inspect.getsource(model_management.toggle_asr)
        self.assertIn("UNLOAD_ASR_TIMEOUT_SEC", src)


@_skip
class TestTranscribeErrorSurfacing(unittest.TestCase):
    """The 503 is useless if the only client renders it as 'Unknown error'."""

    def test_ui_reads_the_nested_detail_payload(self):
        """FastAPI serializes HTTPException(detail={...}) as {"detail": {...}}.

        ``_error_response`` puts the structured body INSIDE ``detail``, so a
        client doing ``resp.json().get("error")`` reads None and shows
        "Unknown error" -- losing both the cause and the retry hint.
        """
        import inspect

        from qwen3_tts.interface.ui import voice_management

        src = inspect.getsource(voice_management.auto_transcribe_audio)
        self.assertIn(
            '"detail"',
            src,
            "the transcribe client must read the nested detail payload, not "
            'resp.json().get("error") which is always None for _error_response',
        )
        self.assertIn(
            "recovery",
            src,
            "a retryable 503 must be surfaced as retryable to the user",
        )


if __name__ == "__main__":
    unittest.main()
