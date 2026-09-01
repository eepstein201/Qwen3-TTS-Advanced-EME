#!/usr/bin/env python3
"""Issue #214 / plan T5 -- the /unload-model queued-generation window.

Every generation path captures ``state.models[mode]`` into a LOCAL before
acquiring ``inference_lock`` and only sets ``generation_state["active"]``
AFTER the lock. An unload landing in that window passes both existing
guards (active+mode 409, load-in-flight 409), nulls the slot, bumps the
epoch, wipes ``gen_cache``, and replies 200 ``unloaded`` -- while the
queued generation runs to completion against the orphaned local (the
engine never re-reads app state). That lying 200 also lets a later
/load-model see slot ``None`` and double-allocate -- the #233 pathology
re-opened through a new route.

Contract pinned by these tests (plan Deliverable 1):

  * ``/unload-model`` acquires ``inference_lock`` as a leaf (the #214 item 2
    ``/unload-asr`` template): with the lock held elsewhere the unload must
    NOT proceed, and it must run only after release. Proven behaviorally
    AND by an AST-enclosure source guard (the #230 precedent guard only
    checked that *some* ``AsyncWith`` exists -- a lock acquired and then
    used outside its body passes it; here the ``to_thread`` call node must
    be a DESCENDANT of the lock's ``AsyncWith``).
  * The route takes the lock even on the ``already_unloaded`` fast path --
    a route-level pre-lock short-circuit would be check-then-act (the
    handler's own short-circuit inside the lock is legitimate).
  * The batch path resets ``generation_state`` INSIDE inference_lock on the
    final item. Without this the route lock alone is self-defeating: the
    batch tail (encode / cache-write / peaks, each an off-loop await)
    releases the lock long before the outer ``finally`` clears ``active``,
    so a queued unload acquires with stale ``active=True`` and 409s AFTER
    waiting out the whole generation.
  * All three generation paths re-read ``state.models.get(mode)`` after
    acquiring the lock and bail with a retryable 503 when the model was
    unloaded while queued (the capture->acquire gap; the canonical shape is
    /transcribe's post-lock ASR recheck). Behavioral for the batch path;
    structural (helper-call-inside-lock) for streaming and /ws.
  * ``/update-model-config`` -- which nulls ALL THREE slots with no
    active guard at all -- takes the same route lock.
  * Every /unload-model and /update-model-config client uses
    ``UNLOAD_MODEL_TIMEOUT_SEC`` (=900): the old hardcoded 10s client
    timeout now fails spuriously whenever anything is queued.

No GPU, models, or running server required. Single event loop per test
(the route coroutine is called directly under one ``asyncio.run`` --
holding the lock from a second loop beside a TestClient portal raises the
loop-bound RuntimeError, which masquerades as "didn't complete").

Run: pytest tests/test_issue214_unload_queued_window.py -v --tb=short
"""

import ast
import asyncio
import importlib
import inspect
import os
import textwrap
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import numpy as np
    import soundfile  # noqa: F401

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

_skip = unittest.skipUnless(_HAS_DEPS, "requires numpy + soundfile")

_ENGINE = "qwen3_tts.core.engine"
_APP_GENERATION = "qwen3_tts.server.app_generation"


def _make_state(**overrides):
    """Complete app.state stand-in (conftest-shaped).

    handle_unload_model and handle_generate read state attributes directly
    (only model_loads/model_config_epoch are getattr-tolerant), so the
    state must be fully populated -- a bare object reds for the wrong
    reason (AttributeError, not the missing lock).
    """
    state = SimpleNamespace()
    state.auth_token = "test_token"  # nosec B105
    state.models = {
        "clone": MagicMock(name="clone-model"),
        "design": MagicMock(name="design-model"),
        "custom": MagicMock(name="custom-model"),
    }
    state.model_load_times = {}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
    state.model_loads = {"clone": None, "design": None, "custom": None}
    state.model_config_epoch = 0
    state.generation_state = {
        "active": False,
        "start_time": 0.0,
        "text_length": 0,
        "mode": "",
        "batch_index": 0,
        "batch_total": 0,
        "chunk_index": 0,
        "chunk_total": 0,
        "generation_id": None,
        "cancelled": False,
    }
    state.request_queue = set()
    state.request_queue_lock = threading.Lock()
    state.generation_lock = asyncio.Lock()
    state.pending_requests = []
    state.pending_lock = asyncio.Lock()
    state.last_activity = 0
    state.models_loaded = threading.Event()
    state.models_loaded.set()
    state.gen_cache = {}
    state.gen_cache_lock = threading.Lock()
    state.inference_lock = asyncio.Lock()
    state.eta_cache = {"median_rate": None, "last_updated": 0}
    state.eta_cache_lock = threading.Lock()
    state.shutdown_timer = None
    state.server_config = {
        "security": {"max_text_length": 50000, "max_batch_size": 20},
        "auto_shutdown_minutes": 0,
        "vllm": {"enabled": False, "fallback_to_torch": True},
    }
    state.vllm_adapter = None
    state.vllm_client = None
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _make_request(state):
    """A REAL starlette Request (slowapi's decorator rejects Mocks) whose
    ``app.state`` points at *state*. ``client`` is a real Address: starlette
    <1.6 returns the raw scope tuple, and slowapi's key_func calls
    ``.host`` on it."""
    from starlette.datastructures import Address
    from starlette.requests import Request

    scope = {
        "type": "http",
        "app": SimpleNamespace(state=state),
        "headers": [],
        "path": "/unload-model",
        "method": "POST",
        "client": Address("127.0.0.1", 51000),
        "query_string": b"",
    }
    return Request(scope)


def _unload_req(model_type="design"):
    req = MagicMock()
    req.model_type = model_type
    return req


def _cleanup_gen_cache_files(state):
    """Unlink the empty cache files handle_generate's cache-write step
    creates (NamedTemporaryFile is real even with soundfile.write patched
    -- only the CONTENT is patched away). Mirrors the precedent cleanup in
    tests/test_batch_generation_state_ownership.py."""
    for entry in list(getattr(state, "gen_cache", {}).values()):
        main_file = entry.get("main_file") or entry.get("file")
        if main_file and os.path.exists(main_file):
            try:
                os.remove(main_file)
            except OSError:
                pass
    state.gen_cache.clear()


class _RecordingAsyncLock:
    """Wraps a real asyncio.Lock, recording every acquire/release.

    Delegates all scheduling semantics to the wrapped real lock -- this
    observes calls, it does not fake concurrency safety (a MagicMock lock
    is the documented anti-pattern: it makes ``locked()``-style assertions
    hollow).
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self.acquire_calls = 0
        self.release_calls = 0

    async def acquire(self):
        self.acquire_calls += 1
        return await self._lock.acquire()

    def release(self):
        self.release_calls += 1
        self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc):
        self.release()
        return False

    def locked(self):
        return self._lock.locked()


def _async_with_contexts(tree):
    """All ``async with`` context expressions in an AST, unparsed."""
    return [
        ast.unparse(item.context_expr)
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncWith)
        for item in node.items
    ]


def _calls_matching(tree, needle):
    """Unparsed source of every Call in *tree* whose IDENTIFIER TEXT (names
    and attributes, not string literals) contains *needle*.

    Matching names only: a ``logger.info("dispatching handle_unload_model")``
    inside the lock body must not satisfy the enclosure pin.
    """
    matches = []
    for sub in ast.walk(tree):
        if not isinstance(sub, ast.Call):
            continue
        identifier_text = " ".join(
            ast.unparse(n)
            for n in ast.walk(sub)
            if isinstance(n, (ast.Name, ast.Attribute))
        )
        if needle in identifier_text:
            matches.append(ast.unparse(sub))
    return matches


def _locked_with_contains_call(tree, needle):
    """True iff a call matching *needle* appears inside the body of an
    ``async with ... inference_lock`` node — the enclosure pin."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncWith):
            continue
        contexts = [ast.unparse(item.context_expr) for item in node.items]
        if not any("inference_lock" in ctx for ctx in contexts):
            continue
        if any(needle in src for src in _calls_matching(node, needle)):
            return True
    return False


# ---------------------------------------------------------------------------
# Item 1: the /unload-model route takes inference_lock
# ---------------------------------------------------------------------------


@_skip
class TestUnloadModelRouteSerializesOnInferenceLock(unittest.TestCase):
    """The route must hold inference_lock while the unload runs."""

    def test_unload_does_not_run_while_inference_lock_held_elsewhere(self):
        """Behavioral: a queued generation holds the lock -> the unload must
        wait. Pre-fix the route runs handle_unload_model immediately,
        nulling the slot and answering 200 ``unloaded`` while the
        generation is still queued (the lying 200 this PR closes)."""
        from qwen3_tts.server import app as app_module

        state = _make_state()
        design_model = state.models["design"]
        cleanup_done = threading.Event()
        # Lock state observed INSIDE the handler: proves the lock is held
        # while the unload's own work runs, not merely that the route waited
        # out the test's hold and then released early (the
        # acquire-then-release mutant).
        locked_at_cleanup = {}

        async def _scenario():
            route_task = None
            async with state.inference_lock:  # a queued generation "holds" it
                route_task = asyncio.ensure_future(
                    app_module.unload_model(_make_request(state), _unload_req(), None)
                )
                # Bounded wait, not a sleep: gives a lock-less route every
                # chance to complete so the assertion below is a real
                # observation, not a scheduling race. MUST run off-loop —
                # a blocking Event.wait on the loop thread would freeze the
                # very task this test is trying to catch in the act.
                ran_while_held = await asyncio.to_thread(
                    cleanup_done.wait, 0.5
                )
                self.assertFalse(
                    ran_while_held,
                    "unload ran to completion while inference_lock was held "
                    "by a queued generation -- the route does not take the "
                    "lock (T5: lying 200 + #233 double-allocation re-open)",
                )
                self.assertIs(
                    state.models["design"],
                    design_model,
                    "the slot must stay untouched while the lock is held "
                    "elsewhere",
                )
            result = await asyncio.wait_for(route_task, timeout=5)
            return result

        def _cleanup():
            locked_at_cleanup["held"] = state.inference_lock.locked()
            cleanup_done.set()

        with patch(
            "qwen3_tts.core.engine.unload_model_cleanup",
            side_effect=_cleanup,
        ):
            result = asyncio.run(_scenario())

        self.assertEqual(result.get("status"), "unloaded", result)
        self.assertIsNone(state.models["design"], "the slot must be nulled once the unload finally runs")
        cleanup_done_seen = cleanup_done.is_set()
        self.assertTrue(cleanup_done_seen, "cleanup never ran after release")
        self.assertIs(
            locked_at_cleanup.get("held"),
            True,
            "the handler's own work ran WITHOUT inference_lock held -- the "
            "route may have acquired and released before dispatching (T5)",
        )

    def test_route_source_encloses_to_thread_in_inference_lock(self):
        """AST-enclosure pin on the real route.

        The route's docstring explains the lock at length, so substring
        checks stay true even with the ``async with`` deleted; and the #230
        precedent guard only proves *some* AsyncWith exists -- a route that
        acquires and releases the lock, then calls ``to_thread`` OUTSIDE it,
        passes both. The ``asyncio.to_thread(handle_unload_model, ...)`` call
        node must be a descendant of the ``async with ... inference_lock``
        body. A future "409-instead-of-queue" redesign is expected to break
        this pin -- update it consciously; do not delete it as an obstacle.
        """
        from qwen3_tts.server import app as app_module

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(app_module.unload_model))
        )
        dispatches = _calls_matching(tree, "handle_unload_model")
        self.assertTrue(
            dispatches,
            "the to_thread(handle_unload_model) dispatch itself disappeared "
            "from the route -- the test target is gone",
        )
        self.assertTrue(
            _locked_with_contains_call(tree, "handle_unload_model"),
            "/unload-model must hold inference_lock AROUND the "
            "to_thread(handle_unload_model) dispatch so the unload cannot "
            "interleave with a queued generation (T5). async-with contexts "
            f"found: {_async_with_contexts(tree) or 'NONE'}",
        )

    def test_already_unloaded_still_acquires_the_lock(self):
        """No route-level pre-lock ``already_unloaded`` short-circuit: that
        would be check-then-act (the handler's own short-circuit INSIDE the
        lock at app_models.py is legitimate). An instrumented lock proves
        the route really acquires it even when there is nothing to do."""
        from qwen3_tts.server import app as app_module

        state = _make_state()
        state.models["design"] = None
        state.inference_lock = _RecordingAsyncLock()

        async def _scenario():
            return await app_module.unload_model(
                _make_request(state), _unload_req(), None
            )

        result = asyncio.run(_scenario())

        self.assertEqual(result.get("status"), "already_unloaded")
        self.assertGreaterEqual(
            state.inference_lock.acquire_calls,
            1,
            "the route skipped inference_lock on the already_unloaded path "
            "-- a pre-lock short-circuit there is check-then-act (T5)",
        )


# ---------------------------------------------------------------------------
# Item 2: the batch path resets generation_state INSIDE the lock
# ---------------------------------------------------------------------------


@_skip
class TestBatchResetsGenerationStateInsideLock(unittest.TestCase):
    """Without the in-lock reset, the route lock alone is self-defeating.

    The batch path releases inference_lock after each item but only clears
    ``active`` in the outer ``finally`` -- after the encode / cache-write /
    peaks off-loop awaits. A queued acquirer (the analogue here: a real
    second waiter on the lock) is granted the lock in that gap while
    ``active`` is still True, and the unload's 409 guard then fires AFTER
    it waited out the whole generation.
    """

    def _run_batch(self, texts):
        """Drive handle_generate; return the actives observed by a real
        second lock-waiter spawned during each item's inference."""

        async def _scenario():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            state = _make_state()
            loop = asyncio.get_running_loop()
            observed_actives = []

            async def _second_waiter():
                # FIFO first waiter: granted the moment the handler releases
                # (mid-batch or final), before the post-item tail can finish
                # -- the tail's continuation is blocked on real thread work,
                # so this observation is deterministic.
                await state.inference_lock.acquire()
                try:
                    observed_actives.append(state.generation_state["active"])
                finally:
                    state.inference_lock.release()

            def _spawn_waiter(*args, **kwargs):
                loop.call_soon_threadsafe(
                    asyncio.ensure_future, _second_waiter()
                )
                return np.zeros(4800, dtype=np.float32), 24000

            req = GenerateRequest(
                texts=list(texts), mode="design", voice_description="friendly"
            )
            try:
                with (
                    patch(
                        f"{_APP_GENERATION}._check_memory_available",
                        return_value=(True, 4096),
                    ),
                    patch(
                        "qwen3_tts.server.validation._validate_generation_request"
                    ),
                    patch(f"{_ENGINE}.run_inference", side_effect=_spawn_waiter),
                    patch("soundfile.write"),
                    patch(
                        "qwen3_tts.core.engine.audio_processing.calculate_waveform_peaks",
                        return_value=[0.1] * 500,
                    ),
                ):
                    result = await asyncio.wait_for(
                        handle_generate(
                            request=_make_request(state),
                            state=state,
                            req=req,
                            security={
                                "max_text_length": 50000,
                                "max_batch_size": 20,
                            },
                            config_provider=None,
                        ),
                        timeout=15,
                    )
            finally:
                _cleanup_gen_cache_files(state)
            return result, observed_actives, state

        return asyncio.run(_scenario())

    def test_active_is_false_when_lock_granted_after_final_item(self):
        """The queued acquirer must observe active=False at the final
        release -- otherwise an unload queued behind the batch 409s after
        waiting out the whole generation."""
        result, observed, _state = self._run_batch(["single text"])

        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 1, "batch must complete")
        self.assertEqual(
            observed,
            [False],
            f"a lock granted after the final item saw active={observed} -- "
            "the batch tail keeps 'active' stale past the lock release, so "
            "a queued /unload-model 409s AFTER the full wait (T5 item 2)",
        )

    def test_active_stays_true_between_items_of_a_multi_item_batch(self):
        """The reset must happen on the FINAL item only: between items the
        batch genuinely is active, and a mid-batch unload SHOULD 409 fast
        and honestly."""
        _result, observed, _state = self._run_batch(["first", "second"])

        self.assertEqual(
            len(observed),
            2,
            f"expected one lock observation per item, got {observed}",
        )
        self.assertIs(
            observed[0],
            True,
            "between items the batch IS active -- resetting per item would "
            "let an unload slip between the items of a running batch",
        )
        self.assertIs(
            observed[1],
            False,
            "after the FINAL item the lock must be released with active "
            f"already cleared (observed: {observed})",
        )


# ---------------------------------------------------------------------------
# Item 3: post-lock slot re-read on the generation paths
# ---------------------------------------------------------------------------


@_skip
class TestPostLockSlotReRead(unittest.TestCase):
    """The capture->acquire gap: a model captured into a local pre-lock
    must be re-validated under the lock before inference runs on it."""

    def test_batch_bails_with_retryable_503_when_slot_nulled_while_queued(self):
        """An unload lands between the model capture and the lock: the
        handler must acquire inference_lock, re-read the slot UNDER it, and
        bail with a retryable 503 -- NOT run inference against the orphaned
        local, and NOT pass because of a pre-lock re-read (a check before
        the acquire leaves the capture->acquire window open; the recording
        lock proves the acquire actually happened)."""
        from fastapi import HTTPException

        # Hoisted so the assertRaises block below can assert on them: the
        # 503 alone proves nothing about WHERE the re-read happened.
        inference_calls = []
        state = _make_state()
        state.inference_lock = _RecordingAsyncLock()

        async def _scenario():
            from qwen3_tts.server.app_generation import handle_generate
            from qwen3_tts.server.validation import GenerateRequest

            def _unload_in_window(state, prompt_file, *args, **kwargs):
                # Runs pre-lock (the prompt load sits between the model
                # capture and the acquire): a concurrent /unload-model
                # nulls the slot here.
                state.models["clone"] = None
                return MagicMock(name="voice-prompt")

            def _run_inference(*args, **kwargs):
                inference_calls.append(state.inference_lock.locked())
                return np.zeros(4800, dtype=np.float32), 24000

            req = GenerateRequest(
                text="hello clone", mode="clone", prompt_file="voice.wav"
            )
            try:
                with (
                    patch(
                        f"{_APP_GENERATION}._check_memory_available",
                        return_value=(True, 4096),
                    ),
                    patch(
                        "qwen3_tts.server.validation._validate_generation_request"
                    ),
                    patch(
                        "qwen3_tts.server.prompt_loading.load_voice_prompt_serialized",
                        side_effect=_unload_in_window,
                    ),
                    patch(f"{_ENGINE}.run_inference", side_effect=_run_inference),
                    patch("soundfile.write"),
                    patch(
                        "qwen3_tts.core.engine.audio_processing.calculate_waveform_peaks",
                        return_value=[0.1] * 500,
                    ),
                ):
                    result = await asyncio.wait_for(
                        handle_generate(
                            request=_make_request(state),
                            state=state,
                            req=req,
                            security={
                                "max_text_length": 50000,
                                "max_batch_size": 20,
                            },
                            config_provider=None,
                        ),
                        timeout=15,
                    )
            finally:
                _cleanup_gen_cache_files(state)
            return result, inference_calls

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(_scenario())

        detail = ctx.exception.detail
        self.assertIsInstance(detail, dict, f"expected structured detail: {detail!r}")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(
            detail.get("error"),
            "model_unloaded",
            "the capture-gap unload must surface a classified 503, "
            f"got: {detail}",
        )
        self.assertEqual(
            detail.get("recovery"),
            "retry",
            "the caller queued behind a generation that never ran -- the "
            "error must be retryable",
        )
        self.assertEqual(
            inference_calls,
            [],
            "inference ran against the orphaned model before the 503 -- "
            "the re-read must precede run_inference INSIDE the lock (a "
            "re-read placed after inference inside the lock would raise "
            "the same 503 having already done the harm)",
        )
        self.assertGreaterEqual(
            state.inference_lock.acquire_calls,
            1,
            "the handler never acquired inference_lock -- a pre-lock "
            "recheck raises the same 503 while leaving the window open",
        )

    def test_batch_reread_is_enclosed_in_the_lock(self):
        """Structural companion to the behavioral test above: the
        handle_generate re-read must sit INSIDE the
        ``async with ... inference_lock`` body. A pre-lock recheck would
        raise the same 503 while leaving the capture->acquire window open --
        this pin kills that mutant."""
        from qwen3_tts.server.app_generation import handle_generate

        tree = ast.parse(textwrap.dedent(inspect.getsource(handle_generate)))
        self.assertTrue(
            _calls_matching(tree, "_require_model_under_lock"),
            "the under-lock re-read helper disappeared from handle_generate "
            "-- the capture->acquire gap is unpinned",
        )
        self.assertTrue(
            _locked_with_contains_call(tree, "_require_model_under_lock"),
            "handle_generate must re-read the model slot UNDER "
            "inference_lock -- a pre-lock recheck leaves the window open",
        )

    def test_streaming_bails_with_503_when_slot_nulled_before_iteration(self):
        """Behavioral twin of the batch test, for /generate-stream: the
        model is captured into a local at handler time but the lock is only
        acquired when the response body is iterated — an unload landing in
        between (here: between the handler call and iteration) must surface
        as a retryable 503 from the under-lock re-read, with the streaming
        inference NEVER started."""
        from fastapi import HTTPException

        # Hoisted for the assertions after the run.
        streaming_calls = []
        state = _make_state()
        state.inference_lock = _RecordingAsyncLock()

        def _stream_stub(*args, **kwargs):
            streaming_calls.append(1)
            yield np.zeros(480, dtype=np.float32), 24000

        async def _scenario():
            from qwen3_tts.server.app_generation import handle_generate_stream
            from qwen3_tts.server.validation import GenerateRequest

            req = GenerateRequest(text="stream me", mode="design")
            response = await handle_generate_stream(
                request=_make_request(state),
                state=state,
                req=req,
                security={"max_text_length": 50000, "max_batch_size": 20},
                config_provider=None,
            )
            # The unload lands AFTER the handler captured the model into a
            # local but BEFORE the response body is iterated.
            state.models["design"] = None
            # Iterate manually: Starlette would normally do this.
            async for _chunk in response.body_iterator:
                pass

        with (
            patch(
                f"{_APP_GENERATION}._check_memory_available",
                return_value=(True, 4096),
            ),
            patch(
                "qwen3_tts.server.validation._validate_generation_request"
            ),
            patch(
                "qwen3_tts.core.engine.run_inference_streaming",
                side_effect=_stream_stub,
            ),
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(_scenario())

        detail = ctx.exception.detail
        self.assertIsInstance(detail, dict, f"expected structured detail: {detail!r}")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(detail.get("error"), "model_unloaded")
        self.assertEqual(detail.get("recovery"), "retry")
        self.assertEqual(
            streaming_calls,
            [],
            "streaming inference started against the orphaned model",
        )
        self.assertGreaterEqual(
            state.inference_lock.acquire_calls,
            1,
            "the streaming generator never acquired inference_lock",
        )

    def test_streaming_and_ws_re_read_inside_their_locks(self):
        """Structural: the streaming generator and the /ws stream function
        must call the same under-lock re-read helper inside their
        ``async with ... inference_lock`` bodies (behavioral coverage is the
        batch test above; these paths share the helper). A missing target is
        a LOUD failure -- a rename must update this test, not silently
        unpin it."""
        import qwen3_tts.server.app_generation as appgen
        import qwen3_tts.server.websocket as wsmod

        for module, func_name in (
            (appgen, "handle_generate_stream"),
            (wsmod, "_stream_generation"),
        ):
            func = getattr(module, func_name, None)
            if func is None:
                self.fail(
                    f"{module.__name__}.{func_name} not found -- the "
                    "streaming/WS re-read pin has lost its target; repoint "
                    "it, do not delete it"
                )
            tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
            self.assertTrue(
                _locked_with_contains_call(tree, "_require_model_under_lock"),
                f"{module.__name__}.{func_name} must re-read the model slot "
                "under inference_lock (the capture->acquire gap leaves the "
                "streaming/WS paths running inference on an orphaned model)",
            )


# ---------------------------------------------------------------------------
# Item 4: /update-model-config takes the same route lock
# ---------------------------------------------------------------------------


@_skip
class TestUpdateModelConfigRouteLock(unittest.TestCase):
    """update-model-config nulls ALL THREE slots with no active guard at
    all -- strictly weaker than pre-fix /unload-model. Same defect class,
    same lock."""

    def test_config_change_does_not_run_while_inference_lock_held(self):
        from qwen3_tts.server import app as app_module

        state = _make_state()
        save_calls = []
        # Lock state observed INSIDE the handler (at save time): kills the
        # acquire-then-release-early mutant, same as the unload-route test.
        locked_at_save = {}

        def _save(cfg):
            locked_at_save["held"] = state.inference_lock.locked()
            save_calls.append(cfg)

        async def _scenario():
            route_task = None
            async with state.inference_lock:
                req = MagicMock()
                req.model_size = "0.6B"
                req.mlx_quantization = None
                route_task = asyncio.ensure_future(
                    app_module.update_model_config(
                        _make_request(state), req, None
                    )
                )
                # Bounded wait, not a sleep-loop: a lock-less route completes
                # well within this window; a lock-taking route stays parked
                # past it. shield keeps the task alive on timeout so it can
                # still be awaited after the release below.
                completed_while_held = False
                try:
                    await asyncio.wait_for(
                        asyncio.shield(route_task), timeout=0.5
                    )
                    completed_while_held = True
                except asyncio.TimeoutError:
                    pass
                self.assertFalse(
                    completed_while_held,
                    "update-model-config completed while inference_lock was "
                    "held elsewhere -- the route does not take the lock (T5 "
                    "item 4)",
                )
            return await asyncio.wait_for(route_task, timeout=5)

        with (
            patch("qwen3_tts.server.app._get_app_config", return_value={}),
            patch(
                "qwen3_tts.server.app_models.save_config",
                side_effect=_save,
            ),
        ):
            result = asyncio.run(_scenario())

        self.assertEqual(result.get("status"), "config_updated")
        self.assertEqual(len(save_calls), 1, "config must save exactly once")
        self.assertIs(
            locked_at_save.get("held"),
            True,
            "the config save (and the slot-nulling around it) ran WITHOUT "
            "inference_lock held -- the route may acquire and release "
            "before dispatching (T5 item 4)",
        )

    def test_update_config_source_encloses_handler_in_inference_lock(self):
        """AST-enclosure pin: the handle_update_model_config dispatch must
        sit inside the route's ``async with ... inference_lock`` body, and
        must still exist."""
        from qwen3_tts.server import app as app_module

        tree = ast.parse(
            textwrap.dedent(inspect.getsource(app_module.update_model_config))
        )
        self.assertTrue(
            _calls_matching(tree, "handle_update_model_config"),
            "the handle_update_model_config dispatch disappeared from the "
            "route -- the test target is gone",
        )
        self.assertTrue(
            _locked_with_contains_call(tree, "handle_update_model_config"),
            "/update-model-config must hold inference_lock around the "
            "handler (it nulls all three slots with no active guard -- the "
            "same queued-generation window as /unload-model). async-with "
            f"contexts found: {_async_with_contexts(tree) or 'NONE'}",
        )


# ---------------------------------------------------------------------------
# Item 5: client timeouts
# ---------------------------------------------------------------------------


@_skip
class TestUnloadModelClientTimeout(unittest.TestCase):
    """Blocking on inference_lock makes short client timeouts wrong.

    The old hardcoded ``timeout=10`` on TTSClient.unload_model now fails
    spuriously whenever anything is queued or generating, while the server
    completes the unload anyway -- reporting failure for work that
    succeeded (the exact defect class /unload-asr fixed with
    UNLOAD_ASR_TIMEOUT_SEC).
    """

    def test_unload_timeout_constant_is_defined_and_lockstep(self):
        from qwen3_tts.core.http_client import (
            LOAD_MODEL_TIMEOUT_SEC,
            UNLOAD_MODEL_TIMEOUT_SEC,
        )

        self.assertEqual(UNLOAD_MODEL_TIMEOUT_SEC, 900)
        self.assertEqual(
            UNLOAD_MODEL_TIMEOUT_SEC,
            LOAD_MODEL_TIMEOUT_SEC,
            "keep the queue-behind-inference timeouts in lockstep",
        )

    def _post_timeout_is_constant(
        self, module_path, class_name, func_name, constant_name, call_needle
    ):
        """AST: EVERY call matching *call_needle* with a ``timeout=`` keyword
        in *func_name* must pass the named constant — not just the first
        (a dead-code decoy call must not satisfy the check while the real
        call keeps a literal). The module source must also reference the
        constant (covers the import)."""
        module = importlib.import_module(module_path)
        self.assertIn(
            constant_name,
            inspect.getsource(module),
            f"{module_path} must import {constant_name}",
        )
        func = getattr(getattr(module, class_name), func_name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or call_needle not in ast.unparse(
                node.func
            ):
                continue
            for kw in node.keywords:
                if kw.arg == "timeout":
                    found.append((ast.unparse(node), ast.unparse(kw.value)))
        self.assertTrue(
            found,
            f"no {call_needle}(timeout=...) call found in "
            f"{module_path}.{func_name}",
        )
        for call_src, got in found:
            self.assertEqual(
                got,
                constant_name,
                f"{module_path}.{func_name} must pass {constant_name} on "
                f"EVERY matching call ({call_src!r} passes {got!r}) -- a "
                "short client timeout fails spuriously once the unload "
                "queues behind a generation",
            )

    def test_tts_client_unload_model_uses_the_constant(self):
        self._post_timeout_is_constant(
            "qwen3_tts.server.client.models",
            "ModelManagerMixin",
            "unload_model",
            "UNLOAD_MODEL_TIMEOUT_SEC",
            call_needle="post",
        )

    def test_tts_client_update_model_config_uses_the_constant(self):
        """update-model-config takes the same route lock (item 4), so its
        client timeout must cover a whole generation too."""
        self._post_timeout_is_constant(
            "qwen3_tts.server.client.models",
            "ModelManagerMixin",
            "update_model_config",
            "UNLOAD_MODEL_TIMEOUT_SEC",
            call_needle="post",
        )

    def test_ui_toggle_model_uses_the_constant(self):
        """The UI borrows LOAD_MODEL_TIMEOUT_SEC for unload today -- the
        value happens to match, but the drift guard must cover intent, not
        coincidence. AST on the server_request call's timeout keyword, NOT
        a substring scan: a docstring or comment mentioning the constant
        would satisfy that. Accepts the house shape toggle_asr uses too:
        the kw may name a local assigned from the constant in a branch."""
        from qwen3_tts.interface.ui import model_management

        tree = ast.parse(inspect.getsource(model_management.toggle_model))
        # name -> ALL RHS sources ever assigned, for branch-assigned
        # timeouts (timeout = UNLOAD_MODEL_TIMEOUT_SEC if ... else ...).
        # EVERY assignment to the variable must derive from the constant:
        # last-assignment-wins would let a swapped-branch mutant survive.
        assignments = {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                rhs = ast.unparse(n.value)
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        assignments.setdefault(t.id, []).append(rhs)
        matched = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or "server_request" not in ast.unparse(
                node.func
            ):
                continue
            for kw in node.keywords:
                if kw.arg == "timeout":
                    got = ast.unparse(kw.value)
                    if got in assignments:
                        # House shape: branch-assigned local. The unload
                        # branch must derive from the constant, and NO
                        # branch may assign a bare literal (kills the
                        # swapped-branch mutant).
                        rhs_sources = assignments[got]
                        self.assertTrue(
                            any(
                                "UNLOAD_MODEL_TIMEOUT_SEC" in rhs
                                for rhs in rhs_sources
                            ),
                            f"timeout local {got!r} is never assigned from "
                            "UNLOAD_MODEL_TIMEOUT_SEC -- the unload path "
                            "borrows another constant",
                        )
                        for rhs in rhs_sources:
                            self.assertNotRegex(
                                rhs,
                                r"^\d+$",
                                f"timeout local {got!r} is assigned a bare "
                                f"literal ({rhs}) in some branch -- a "
                                "swapped branch would silently keep a 10s "
                                "timeout",
                            )
                    else:
                        self.assertEqual(
                            got,
                            "UNLOAD_MODEL_TIMEOUT_SEC",
                            "toggle_model must pass UNLOAD_MODEL_TIMEOUT_SEC "
                            "(directly or via a branch-assigned local, the "
                            "toggle_asr shape) to server_request -- "
                            "borrowing the load constant passes only by "
                            "coincidence",
                        )
                    matched = True
        self.assertTrue(
            matched,
            "no server_request(timeout=...) call found in toggle_model",
        )


if __name__ == "__main__":
    unittest.main()
