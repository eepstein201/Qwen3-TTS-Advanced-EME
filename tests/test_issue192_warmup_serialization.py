#!/usr/bin/env python3
"""Issue #192 structural fix — the load-time warm-up must be serialized.

The design warm-up inference is real MLX inference. Running it while a
generation holds the GPU was the #192 trigger pair — the one this fix
closes. It was not the only unsynchronized concurrent MLX inference
reachable through the API: ``/create-voice-prompt``
(``create_voice_clone_prompt``, engine/inference.py) remains unlocked of
the same class (ml-explore/mlx#3078, Blaizzy/mlx-audio#638, #733 —
corruption manifests as EOS-never-emitted runaways behind HTTP 200);
serializing it is tracked as a #192 follow-up. ``/transcribe``
(mlx-whisper ``generate``) was the same class and is now serialized
(tests/test_issue192_transcribe_serialization.py).

Contract pinned by these tests:

  * ``handle_load_model`` runs the design warm-up with ``inference_lock``
    HELD, and the load itself with ``inference_lock`` RELEASED (minutes of
    download/weight construction must not starve ``/generate``)
  * the model becomes visible in ``state.models`` only AFTER the warm-up
  * clone/custom loads skip the warm-up path entirely (no lock queueing to
    no-op — mirrors ``_warmup_model``'s own design-only guard)
  * the engine ``load_model`` is called with ``warmup=False`` (the split is
    enforced, not assumed)
  * ``TTS_SKIP_WARMUP`` is checked BEFORE the lock (ablation runs don't
    queue behind generations for a no-op)
  * the load and the warm-up run in worker threads, never on the event loop
  * a held ``inference_lock`` DEFERS the warm-up while the load proceeds —
    the queueing behavior at the heart of the fix
  * ``_background_load`` (startup loader thread) serializes its warm-up onto
    the server's event loop the same way, and skips the warm-up — never runs
    it unsynchronized, never raises — when no loop is available or closed
  * every ``/load-model`` client (TTSClient, CLI auto-load, both UI paths)
    uses the shared ``LOAD_MODEL_TIMEOUT_SEC`` — the old hardcoded 120s now
    fails spuriously whenever a warm-up queues behind a generation
  * the ``/load-model`` route awaits the async handler directly (source-shape
    guard: wrapping it back into ``asyncio.to_thread`` returns an un-awaited
    coroutine and passes every unit test here)

No GPU, models, or running server required — locks are real asyncio.Locks,
inference is patched at the engine definition sites.

Run: pytest tests/test_issue192_warmup_serialization.py -v --tb=short
"""

import asyncio
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


def _make_state(event_loop=None):
    """Build an app.state stand-in with REAL asyncio locks.

    The locks must be real — the tests assert on ``lock.locked()`` observed
    from inside the work functions, so a MagicMock lock would make every
    assertion hollow.
    """
    state = MagicMock()
    state.models = {"clone": None, "design": None, "custom": None}
    state.models_loading = {"clone": False, "design": False, "custom": False}
    state.model_load_times = {}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
    state.inference_lock = asyncio.Lock()
    state.generation_lock = asyncio.Lock()
    state.gen_cache = {}
    if event_loop is not None:
        state.event_loop = event_loop
    return state


def _req(model_type="design"):
    req = MagicMock()
    req.model_type = model_type
    return req


def _recorder(lock, ret=None):
    """Return (mock_fn, calls) recording lock/thread state at each call."""
    calls = []

    def _record(*args, **kwargs):
        calls.append(
            {
                "args": args,
                "kwargs": kwargs,
                "lock_held": lock.locked(),
                "in_worker_thread": threading.current_thread()
                is not threading.main_thread(),
            }
        )
        return ret

    return _record, calls


@_skip
class TestLoadModelWarmupSerialization(unittest.TestCase):
    """/load-model must warm up under inference_lock, load without it."""

    def _run_handler(self, state, model_type="design"):
        # Ambient knob would flake every non-knob test for a developer who
        # exports TTS_SKIP_WARMUP. The knob test calls _run_handler_inner
        # directly so its patch.dict still governs.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_SKIP_WARMUP", None)
            return self._run_handler_inner(state, model_type)

    def _run_handler_inner(self, state, model_type="design"):
        from qwen3_tts.server.app_models import handle_load_model

        load_fn, load_calls = _recorder(state.inference_lock, ret=object())
        warm_calls = []

        def _warm_fn(*args, **kwargs):
            warm_calls.append(
                {
                    "lock_held": state.inference_lock.locked(),
                    # Sequencing claim made load-bearing: the model must not
                    # be visible to concurrent requests during the warm-up.
                    "model_visible": state.models.get(model_type) is not None,
                    "in_worker_thread": threading.current_thread()
                    is not threading.main_thread(),
                }
            )

        with (
            patch("qwen3_tts.core.engine.load_model", side_effect=load_fn),
            patch(
                "qwen3_tts.core.engine.model_loader._warmup_model",
                side_effect=_warm_fn,
            ),
            patch(
                "qwen3_tts.core.config.get_model_info",
                return_value={"name": "qwen3-tts-design"},
            ),
        ):
            result = asyncio.run(handle_load_model(state, _req(model_type)))
        return result, load_calls, warm_calls

    def test_warmup_runs_with_inference_lock_held(self):
        state = _make_state()
        result, _, warm_calls = self._run_handler(state)

        self.assertEqual(result["status"], "loaded")
        self.assertEqual(len(warm_calls), 1)
        self.assertTrue(
            warm_calls[0]["lock_held"],
            "warm-up inference must run with inference_lock held — an "
            "unlocked warm-up is the #192 concurrency bug",
        )

    def test_load_runs_outside_inference_lock(self):
        """A multi-minute model load must not starve /generate."""
        state = _make_state()
        _, load_calls, _ = self._run_handler(state)

        self.assertEqual(len(load_calls), 1)
        self.assertFalse(
            load_calls[0]["lock_held"],
            "model load (download + weight construction) must run outside "
            "inference_lock",
        )

    def test_engine_load_called_with_warmup_false(self):
        """The load/warm-up split is enforced at the call site."""
        state = _make_state()
        _, load_calls, _ = self._run_handler(state)

        self.assertIs(
            load_calls[0]["kwargs"].get("warmup", True),
            False,
            "server must call engine load_model with warmup=False and run "
            "the warm-up itself under inference_lock",
        )

    def test_model_stored_after_warmup(self):
        """Observable sequencing preserved: model visible only once usable."""
        state = _make_state()
        result, _, warm_calls = self._run_handler(state)

        self.assertEqual(result["status"], "loaded")
        self.assertFalse(
            warm_calls[0]["model_visible"],
            "model must not be visible in state.models while the warm-up "
            "is still running",
        )
        self.assertIsNotNone(state.models["design"])

    def test_clone_load_skips_warmup_entirely(self):
        """clone/custom never queue for the lock just to no-op the warm-up."""
        state = _make_state()
        result, _, warm_calls = self._run_handler(state, model_type="clone")

        self.assertEqual(result["status"], "loaded")
        self.assertEqual(warm_calls, [])
        self.assertIsNotNone(state.models["clone"])

    def test_knob_set_skips_warmup_without_locking(self):
        """TTS_SKIP_WARMUP=1 — checked BEFORE the lock, so ablation runs
        don't queue behind generations for a no-op."""
        state = _make_state()
        with patch.dict(os.environ, {"TTS_SKIP_WARMUP": "1"}):
            result, _, warm_calls = self._run_handler_inner(state)

        self.assertEqual(result["status"], "loaded")
        self.assertEqual(warm_calls, [])
        self.assertIsNotNone(state.models["design"])

    def test_load_and_warmup_run_off_event_loop_thread(self):
        """The handler must never block the event loop: both the load and
        the warm-up execute in worker threads, not the loop's thread."""
        state = _make_state()
        _, load_calls, warm_calls = self._run_handler(state)

        self.assertTrue(
            load_calls[0]["in_worker_thread"],
            "model load must run via asyncio.to_thread, not on the loop",
        )
        self.assertTrue(
            warm_calls[0]["in_worker_thread"],
            "warm-up must run via asyncio.to_thread, not on the loop",
        )

    def test_warmup_deferred_while_generation_holds_lock(self):
        """The heart of the fix: a held inference_lock defers the warm-up
        (while the load proceeds — it must not wait for the lock)."""
        from qwen3_tts.server.app_models import handle_load_model

        async def _scenario():
            state = _make_state()
            load_fn, load_calls = _recorder(state.inference_lock, ret=object())
            warm_calls = []

            def _warm_fn(*args, **kwargs):
                warm_calls.append({"lock_held": state.inference_lock.locked()})

            # Simulate an in-flight generation holding the GPU lock.
            async with state.inference_lock:
                with (
                    patch(
                        "qwen3_tts.core.engine.load_model", side_effect=load_fn
                    ),
                    patch(
                        "qwen3_tts.core.engine.model_loader._warmup_model",
                        side_effect=_warm_fn,
                    ),
                    patch(
                        "qwen3_tts.core.config.get_model_info",
                        return_value={"name": "qwen3-tts-design"},
                    ),
                ):
                    task = asyncio.ensure_future(
                        handle_load_model(state, _req("design"))
                    )
                    # Let the handler run to its lock-acquire point.
                    await asyncio.sleep(0.05)
                    assert len(load_calls) == 1, (
                        "load must complete without waiting for the lock"
                    )
                    assert warm_calls == [], (
                        "warm-up must not run while another holder has "
                        "inference_lock"
                    )
            result = await asyncio.wait_for(task, timeout=5)
            return result, warm_calls

        # Ambient TTS_SKIP_WARMUP would skip the warm-up and flake this test.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_SKIP_WARMUP", None)
            result, warm_calls = asyncio.run(_scenario())

        self.assertEqual(result["status"], "loaded")
        self.assertEqual(len(warm_calls), 1)
        self.assertTrue(
            warm_calls[0]["lock_held"],
            "after the holder releases, the warm-up runs with the lock held",
        )


@_skip
class TestBackgroundLoadWarmupSerialization(unittest.TestCase):
    """Startup loader thread must serialize its warm-up the same way."""

    def _run_background_load(self, state):
        # Ambient TTS_SKIP_WARMUP would flake the non-knob startup test.
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_SKIP_WARMUP", None)
            return self._run_background_load_inner(state)

    def _run_background_load_inner(self, state):
        from qwen3_tts.server.app_lifespan import _background_load

        load_fn, load_calls = _recorder(state.inference_lock, ret=object())
        warm_fn, warm_calls = _recorder(state.inference_lock)
        with (
            patch("qwen3_tts.core.engine.load_model", side_effect=load_fn),
            patch(
                "qwen3_tts.core.engine.model_loader._warmup_model",
                side_effect=warm_fn,
            ),
            patch("qwen3_tts.core.engine.migrate_orphan_mlx_prompts"),
            patch("qwen3_tts.server.app_lifespan.get_backend", return_value="mlx"),
        ):
            # Simulate the real server: a live event loop (captured on
            # state) with the loader running in a separate thread.
            async def _scenario():
                state.event_loop = asyncio.get_running_loop()
                await asyncio.to_thread(_background_load, state)

            asyncio.run(_scenario())
        return load_calls, warm_calls

    def test_startup_warmup_runs_with_inference_lock_held(self):
        state = _make_state()
        state.server_config = {"models": {"design": {"load_at_startup": True}}}
        load_calls, warm_calls = self._run_background_load(state)

        self.assertEqual(len(load_calls), 1)
        self.assertFalse(
            load_calls[0]["lock_held"],
            "startup model load must run outside inference_lock",
        )
        self.assertEqual(
            len(warm_calls),
            1,
            "startup warm-up must still run — serialized, not skipped",
        )
        self.assertTrue(
            warm_calls[0]["lock_held"],
            "startup warm-up must run with inference_lock held",
        )
        self.assertFalse(state.models_loading["design"])
        self.assertIsNotNone(state.models["design"])

    def _run_startup_no_lock_acquire(self, state):
        """Run _background_load synchronously (no loop anywhere) and return
        the warm-up recorder calls."""
        from qwen3_tts.server.app_lifespan import _background_load

        state.server_config = {"models": {"design": {"load_at_startup": True}}}
        warm_fn, warm_calls = _recorder(state.inference_lock)
        with (
            patch(
                "qwen3_tts.core.engine.load_model",
                side_effect=_recorder(state.inference_lock, ret=object())[0],
            ),
            patch(
                "qwen3_tts.core.engine.model_loader._warmup_model",
                side_effect=warm_fn,
            ),
            patch("qwen3_tts.core.engine.migrate_orphan_mlx_prompts"),
            patch("qwen3_tts.server.app_lifespan.get_backend", return_value="mlx"),
        ):
            _background_load(state)  # no asyncio.run — no loop anywhere
        return warm_calls

    def test_startup_warmup_skipped_when_event_loop_missing(self):
        """Explicitly no event loop (None) — skip, never run unsynchronized,
        never fail the load. Set explicitly: a bare MagicMock state would
        auto-fabricate a truthy event_loop and exercise nothing."""
        state = _make_state()
        state.event_loop = None
        warm_calls = self._run_startup_no_lock_acquire(state)

        self.assertEqual(
            warm_calls,
            [],
            "with no event loop the warm-up must be skipped, not run "
            "unsynchronized",
        )
        self.assertIsNotNone(
            state.models["design"],
            "the load itself must succeed without the warm-up",
        )
        self.assertFalse(state.models_loading["design"])

    def test_startup_warmup_skipped_when_event_loop_closed(self):
        """A REAL closed loop (not a mock) — shutdown race exercises the
        is_closed() branch of the skip guard."""
        state = _make_state()
        closed_loop = asyncio.new_event_loop()
        closed_loop.close()
        state.event_loop = closed_loop
        warm_calls = self._run_startup_no_lock_acquire(state)

        self.assertEqual(
            warm_calls,
            [],
            "with a closed event loop the warm-up must be skipped, not run "
            "unsynchronized",
        )
        self.assertIsNotNone(state.models["design"])
        self.assertFalse(state.models_loading["design"])

    def test_startup_clone_load_skips_warmup_scheduling(self):
        """clone/custom startup loads never schedule the warm-up at all."""
        from qwen3_tts.server.app_lifespan import _background_load

        state = _make_state()
        state.server_config = {"models": {"clone": {"load_at_startup": True}}}
        warm_fn, warm_calls = _recorder(state.inference_lock)
        with (
            patch(
                "qwen3_tts.core.engine.load_model",
                side_effect=_recorder(state.inference_lock, ret=object())[0],
            ),
            patch(
                "qwen3_tts.core.engine.model_loader._warmup_model",
                side_effect=warm_fn,
            ),
            patch("qwen3_tts.core.engine.migrate_orphan_mlx_prompts"),
            patch("qwen3_tts.server.app_lifespan.get_backend", return_value="mlx"),
        ):
            _background_load(state)

        self.assertEqual(warm_calls, [])
        self.assertIsNotNone(state.models["clone"])
        self.assertFalse(state.models_loading["clone"])


@_skip
class TestLoadModelRouteShape(unittest.TestCase):
    """/load-model must await the now-async handler directly.

    Source-shape guard (precedent: tests/test_p3_p4_remediation.py) —
    wrapping ``handle_load_model`` back into ``asyncio.to_thread`` would
    return an un-awaited coroutine from the endpoint and pass every unit
    test above; only the source assertion catches it.
    """

    def test_endpoint_awaits_handler_directly(self):
        import inspect

        from qwen3_tts.server import app as app_module

        src = inspect.getsource(app_module.load_model_endpoint)
        self.assertIn(
            "await handle_load_model(",
            src,
            "/load-model route must await the async handler directly",
        )
        self.assertNotIn(
            "to_thread(handle_load_model",
            src,
            "/load-model route must not dispatch the async handler through "
            "asyncio.to_thread",
        )


@_skip
class TestLoadModelTimeoutDrift(unittest.TestCase):
    """Every /load-model client must use the shared LOAD_MODEL_TIMEOUT_SEC.

    A caller reverting to a hardcoded timeout ships silently — no behavior
    test pins request timeouts, and the server now legitimately takes
    longer than the old 120s whenever a warm-up queues behind a generation
    (#192 serialization).
    """

    _CALL_SITES = (
        "qwen3_tts/server/client/models.py",
        "qwen3_tts/interface/generate_server.py",
        "qwen3_tts/interface/ui/model_management.py",
        "qwen3_tts/interface/ui/shared.py",
    )

    def test_all_load_model_call_sites_use_shared_constant(self):
        from qwen3_tts.core.http_client import LOAD_MODEL_TIMEOUT_SEC

        # Must cover load + queue-behind-one-generation + warm-up.
        self.assertGreater(LOAD_MODEL_TIMEOUT_SEC, 660)

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for rel in self._CALL_SITES:
            with open(os.path.join(root, rel)) as f:
                src = f.read()
            self.assertIn(
                "LOAD_MODEL_TIMEOUT_SEC",
                src,
                f"{rel} must use the shared /load-model timeout constant",
            )
            self.assertNotIn(
                "timeout=120",
                src,
                f"{rel} must not hardcode the old 120s /load-model timeout",
            )


if __name__ == "__main__":
    unittest.main()
