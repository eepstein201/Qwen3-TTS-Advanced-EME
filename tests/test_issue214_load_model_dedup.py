#!/usr/bin/env python3
"""Issue #214 item 3 -- /load-model in-flight dedup (Phase 2c).

``handle_load_model`` guards only on *already loaded*. While a load is in
flight ``state.models[t]`` is still None, so a second POST /load-model starts
a second full weight construction (~2.5 GB double allocation on MLX); both
then assign ``state.models[t]``, orphaning one model. This module pins the
replacement contract: a duplicate caller AWAITS the in-flight load through a
per-load record (claim / release / await) -- it never re-loads behind one.

Contract pinned by these tests:

  * Concurrent duplicates coalesce: exactly ONE ``load_model`` call serves N
    concurrent /load-model requests; duplicates return
    ``{"status": "loaded", "deduped": True}`` (M1).
  * ``claim_model_load`` serializes on a MODULE-SCOPE lock (fail-closed --
    the gate cannot be absent), proven by real OS-thread contention (M2).
  * A waiter's response is observed only after the owner's load finished
    (Event ordering, not sleeps) (M3).
  * A waiter classifies the outcome from the RECORD, never from shared dicts
    (M4/M7): FAILED with a message -> 500 carrying that message; FAILED with
    an empty message -> retryable 503, never a 500-with-null-body (C3).
  * The claim slot is cleared on EVERY owner exit path -- release runs in
    ``finally`` for ImportError / RuntimeError / ValueError / the catch-all,
    and asyncio.CancelledError records CANCELLED and re-raises (M6/C2).
  * A stale-epoch caller does not attach: after /update-model-config bumps
    ``model_config_epoch``, the next caller takes its own claim (C1/M9).
  * /unload-model during an in-flight load is a 409, not a silent undo, and
    a completed unload bumps the epoch (H4a/C1).
  * A warm-up failure does NOT discard weights that loaded cleanly (W1/M10):
    the model stays assigned, ``warmup_failed`` is surfaced on owner AND
    waiter responses, ``_recover_from_failed_load`` never runs, and
    ``model_load_errors`` is never written for it.
  * The startup loader WAITS on an HTTP-owned load instead of skipping it:
    ``models_loaded`` (which gates /ready) is not set while the HTTP-owned
    load is in flight, and a failed owner is recorded in ``model_load_errors``
    (H2/M5/M8).
  * The waiter's budget stays below the client's ``LOAD_MODEL_TIMEOUT_SEC``
    (drift guard); timing out is a retryable 503 that leaves the owner's
    claim untouched.
  * The two clients that read the error body flat (CLI load_model_on_server,
    UI toggle_model) unwrap the nested ``detail`` payload via
    ``_error_payload`` -- a classified 503 must not reach the user as
    "Unknown error".

No GPU, models, or running server required -- ``load_model`` is patched at
the ``qwen3_tts.core.engine`` facade (handlers import function-locally, so
the patched attribute is what they resolve at call time).

Run: pytest tests/test_issue214_load_model_dedup.py -v --tb=short
"""

import asyncio
import inspect
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

try:
    import fastapi  # noqa: F401

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides):
    """Build an app.state stand-in with the REAL primitives the gate needs.

    ``model_loads`` / ``model_config_epoch`` are plain dicts/int the claim
    mutates under the module lock; the locks are real so lock-state
    assertions are not hollow.
    """
    state = MagicMock()
    state.models = {"clone": None, "design": None, "custom": None}
    state.model_loads = {"clone": None, "design": None, "custom": None}
    state.model_config_epoch = 0
    state.model_load_times = {}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
    state.inference_lock = asyncio.Lock()
    state.generation_lock = asyncio.Lock()
    state.generation_state = {"active": False, "mode": "", "start_time": 0.0}
    state.gen_cache = {}
    state.gen_cache_lock = threading.Lock()
    state.models_loaded = threading.Event()
    state.server_config = {"models": {}}
    state.event_loop = None
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _req(model_type="clone"):
    req = MagicMock()
    req.model_type = model_type
    return req


class _HandlerBox:
    """Thread-safe result box for a handler run on its own thread/loop."""

    def __init__(self):
        self.value = None
        self.error = None
        self.done = threading.Event()

    def target(self, state, req):
        from qwen3_tts.server.app_models import handle_load_model

        try:
            self.value = asyncio.run(handle_load_model(state, req))
        except BaseException as exc:  # noqa: BLE001 — the box IS the trap
            self.error = exc
        finally:
            self.done.set()


def _run_handler_thread(state, model_type="clone"):
    box = _HandlerBox()
    thread = threading.Thread(target=box.target, args=(state, _req(model_type)), daemon=True)
    thread.start()
    return box, thread


def _poll_until(test, predicate, timeout=5.0, message="condition not met"):
    """Bounded poll for cross-thread state that has no Event of its own."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    test.fail(message)


def _blocking_load_spy(calls, release_event, sentinel, later_sentinel=None):
    """Side effect for qwen3_tts.core.engine.load_model.

    First call records itself and blocks on ``release_event`` (the owner's
    in-flight window); any later call records itself and returns immediately
    -- in a correct implementation no later call ever happens, so seeing one
    is the double-construction signature.
    """

    def _load(model_type, warmup=False):
        calls.append(model_type)
        if len(calls) == 1:
            release_event.wait(timeout=10)
            return sentinel
        return later_sentinel if later_sentinel is not None else sentinel

    return _load


@_skip
class TestDedupCoalescesConcurrentLoads(unittest.TestCase):
    """M1: a duplicate attaches to the in-flight load; weights build once."""

    def test_two_concurrent_loads_call_load_model_once(self):
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        calls = []
        release = threading.Event()
        sentinel = object()
        timer = threading.Timer(0.3, release.set)
        self.addCleanup(timer.cancel)
        timer.start()

        async def _scenario():
            return await asyncio.gather(
                handle_load_model(state, _req()),
                handle_load_model(state, _req()),
            )

        with (
            patch(
                "qwen3_tts.core.engine.load_model",
                side_effect=_blocking_load_spy(calls, release, sentinel),
            ),
            patch(
                "qwen3_tts.core.config.get_model_info",
                return_value={"name": "qwen3-tts-clone"},
            ),
        ):
            owner, waiter = asyncio.run(_scenario())

        self.assertEqual(
            calls,
            ["clone"],
            f"load_model must run exactly once for concurrent duplicates, "
            f"saw {calls!r} — a second full weight construction is the "
            f"#214 item-3 double allocation",
        )
        self.assertEqual(owner, {"status": "loaded", "model": "clone"})
        self.assertIs(state.models["clone"], sentinel)
        self.assertEqual(
            waiter,
            {"status": "loaded", "model": "clone", "deduped": True},
            "the duplicate must attach and be told so, not reload",
        )
        self.assertIsNone(
            state.model_loads["clone"], "claim slot must be released after success"
        )

    def test_three_claimants_still_build_once(self):
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        calls = []
        release = threading.Event()
        sentinel = object()
        timer = threading.Timer(0.3, release.set)
        self.addCleanup(timer.cancel)
        timer.start()

        async def _one():
            return await handle_load_model(state, _req())

        async def _all_three():
            return await asyncio.gather(_one(), _one(), _one())

        with (
            patch(
                "qwen3_tts.core.engine.load_model",
                side_effect=_blocking_load_spy(calls, release, sentinel),
            ),
            patch(
                "qwen3_tts.core.config.get_model_info",
                return_value={"name": "qwen3-tts-clone"},
            ),
        ):
            results = asyncio.run(_all_three())

        self.assertEqual(calls, ["clone"], "three claimants, one construction")
        deduped_flags = [bool(r.get("deduped")) for r in results]
        self.assertEqual(
            sorted(deduped_flags),
            [False, True, True],
            f"exactly one owner and two attachers expected, got {deduped_flags!r}",
        )
        for result in results:
            self.assertEqual(result["status"], "loaded")


@_skip
class TestClaimSerializesOnModuleLock(unittest.TestCase):
    """M2: the claim is a real CAS under a real module-scope lock."""

    def test_claim_blocks_while_lock_held_by_another_thread(self):
        from qwen3_tts.server import model_loading

        state = _make_state()
        holder_ready = threading.Event()
        release_holder = threading.Event()

        def _hold_lock():
            with model_loading.MODEL_LOAD_LOCK:
                holder_ready.set()
                release_holder.wait(timeout=5)

        holder = threading.Thread(target=_hold_lock, daemon=True)
        holder.start()
        self.addCleanup(lambda: (release_holder.set(), holder.join(timeout=5)))
        self.assertTrue(
            holder_ready.wait(timeout=5),
            "test setup failed: holder thread never acquired the module lock",
        )

        claim_done = threading.Event()
        box = {}

        def _claim():
            box["result"] = model_loading.claim_model_load(state, "clone")
            claim_done.set()

        claimer = threading.Thread(target=_claim, daemon=True)
        claimer.start()
        self.addCleanup(lambda: claimer.join(timeout=5))

        completed_while_held = claim_done.wait(timeout=0.5)
        self.assertFalse(
            completed_while_held,
            "claim_model_load completed while another thread held "
            "MODEL_LOAD_LOCK — the claim is not taking the module-scope "
            "lock (fail-closed gate violated)",
        )

        release_holder.set()
        self.assertTrue(
            claim_done.wait(timeout=5), "claim never completed after release"
        )
        result, record = box["result"]
        self.assertEqual(
            result,
            model_loading.ClaimResult.CLAIMED,
            "a free slot must yield a fresh CLAIMED record",
        )
        # Leave the gate clean for later tests in full-suite order.
        model_loading.release_model_load(
            state, "clone", record, model_loading.LoadOutcome.OK
        )


@_skip
class TestWaiterWaitsForOwner(unittest.TestCase):
    """M3: the waiter's response must not precede the owner's completion."""

    def test_waiter_returns_only_after_load_finished(self):
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        events = []
        release = threading.Event()
        sentinel = object()
        timer = threading.Timer(0.3, release.set)
        self.addCleanup(timer.cancel)
        timer.start()

        def _load(model_type, warmup=False):
            events.append("load_start")
            release.wait(timeout=10)
            events.append("load_end")
            return sentinel

        async def _waiter():
            result = await handle_load_model(state, _req())
            events.append("waiter_returned")
            return result

        async def _both():
            return await asyncio.gather(
                handle_load_model(state, _req()), _waiter()
            )

        with (
            patch("qwen3_tts.core.engine.load_model", side_effect=_load),
            patch(
                "qwen3_tts.core.config.get_model_info",
                return_value={"name": "qwen3-tts-clone"},
            ),
        ):
            asyncio.run(_both())

        self.assertIn("load_end", events, "the owner's load must have finished")
        self.assertLess(
            events.index("load_end"),
            events.index("waiter_returned"),
            f"waiter returned before the owner's load finished; ordering was "
            f"{events!r} — the waiter is not waiting on the record",
        )


@_skip
class TestWaiterClassificationFromRecord(unittest.TestCase):
    """M4/M7: the waiter classifies from the record, never shared dicts."""

    def _run_owner_then_waiter(self, exc):
        """Owner raises ``exc`` after its release; both outcomes gathered."""
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        release = threading.Event()
        timer = threading.Timer(0.3, release.set)
        self.addCleanup(timer.cancel)
        timer.start()

        def _load(model_type, warmup=False):
            release.wait(timeout=10)
            raise exc

        async def _scenario():
            return await asyncio.gather(
                handle_load_model(state, _req()),
                handle_load_model(state, _req()),
                return_exceptions=True,
            )

        with (
            patch("qwen3_tts.core.engine.load_model", side_effect=_load),
            patch(
                "qwen3_tts.core.config.get_model_info",
                return_value={"name": "qwen3-tts-clone"},
            ),
        ):
            owner, waiter = asyncio.run(_scenario())
        return state, owner, waiter

    def test_waiter_gets_owner_failure_message(self):
        from fastapi import HTTPException

        state, owner, waiter = self._run_owner_then_waiter(
            RuntimeError("boom during load")
        )

        self.assertIsInstance(owner, HTTPException, "owner must see the 500")
        self.assertIsInstance(
            waiter, HTTPException, "a waiter must not be told the load succeeded"
        )
        self.assertEqual(waiter.status_code, 500)
        self.assertIn(
            "boom during load",
            str(waiter.detail.get("detail", "")),
            "the waiter must carry the owner's message from the record",
        )
        self.assertIsNone(
            state.models["clone"], "a failed load must not leave a model behind"
        )

    def test_waiter_empty_error_maps_to_retryable_503(self):
        """C3: FAILED with an empty error is a 503, not a 500-null-body."""
        from fastapi import HTTPException

        _state, _owner, waiter = self._run_owner_then_waiter(RuntimeError())

        self.assertIsInstance(waiter, HTTPException)
        self.assertEqual(
            waiter.status_code,
            503,
            f"an empty owner error must classify as retryable 503, got "
            f"{waiter.status_code} {waiter.detail!r}",
        )
        self.assertEqual(waiter.detail.get("recovery"), "retry")
        self.assertTrue(waiter.detail.get("error"))


@_skip
class TestReleaseRunsInFinally(unittest.TestCase):
    """M6: every failure path releases the claim — a leak wedges 870s→503."""

    def test_slot_cleared_on_each_except_path(self):
        from fastapi import HTTPException

        from qwen3_tts.server.app_models import handle_load_model

        cases = [
            ("import_error", ImportError("no backend")),
            ("runtime_error", RuntimeError("boom")),
            ("value_error", ValueError("bad value")),
            ("catch_all", AttributeError("library drift")),
        ]
        for name, exc in cases:
            with self.subTest(path=name):
                state = _make_state()
                with (
                    patch("qwen3_tts.core.engine.load_model", side_effect=exc),
                    patch(
                        "qwen3_tts.core.config.get_model_info",
                        return_value={"name": "qwen3-tts-clone"},
                    ),
                ):
                    with self.assertRaises(HTTPException):
                        asyncio.run(handle_load_model(state, _req()))

                self.assertIsNone(
                    state.model_loads["clone"],
                    f"{name}: claim slot not released — release_model_load "
                    f"must run in finally, or this model type 503s forever",
                )

    def test_cancellation_records_cancelled_and_reraises(self):
        """C2: CancelledError subclasses BaseException — release must still
        run, the outcome must read CANCELLED, and the error must propagate."""
        from qwen3_tts.server import model_loading
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        release = threading.Event()
        self.addCleanup(release.set)

        def _load(model_type, warmup=False):
            release.wait(timeout=10)
            return object()

        real_release = model_loading.release_model_load
        with (
            patch("qwen3_tts.core.engine.load_model", side_effect=_load),
            patch(
                "qwen3_tts.core.config.get_model_info",
                return_value={"name": "qwen3-tts-clone"},
            ),
            patch.object(
                model_loading, "release_model_load", wraps=real_release
            ) as release_spy,
        ):

            async def _scenario():
                task = asyncio.create_task(
                    handle_load_model(state, _req())
                )
                # Await-based poll: a time.sleep here would block the very
                # loop the created task needs to reach its claim.
                for _ in range(500):
                    if state.model_loads["clone"] is not None:
                        break
                    await asyncio.sleep(0.01)
                else:
                    self.fail("owner never claimed before cancellation")
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            asyncio.run(_scenario())

            self.assertIsNone(
                state.model_loads["clone"],
                "cancelled owner must still release the claim slot",
            )
            outcome_args = [
                call.args[3] if len(call.args) > 3 else call.kwargs.get("outcome")
                for call in release_spy.call_args_list
            ]
            self.assertIn(
                model_loading.LoadOutcome.CANCELLED,
                outcome_args,
                f"release must record CANCELLED on the cancellation path, "
                f"saw {outcome_args!r}",
            )


@_skip
class TestStaleEpochDoesNotAttach(unittest.TestCase):
    """C1/M9: a waiter whose epoch differs must not attach to the record."""

    def test_update_model_config_bumps_epoch_and_caller_reclaims(self):
        from qwen3_tts.server.app_models import (
            handle_load_model,
            handle_update_model_config,
        )

        state = _make_state()
        calls = []
        release_owner = threading.Event()
        sentinel_owner = object()
        sentinel_new = object()

        def _load(model_type, warmup=False):
            calls.append(model_type)
            if len(calls) == 1:
                release_owner.wait(timeout=10)
                return sentinel_owner
            return sentinel_new

        cfg_req = MagicMock()
        cfg_req.model_size = "0.6B"
        cfg_req.mlx_quantization = None

        with (
            patch(
                "qwen3_tts.core.engine.load_model", side_effect=_load
            ),
            patch(
                "qwen3_tts.core.config.get_model_info",
                return_value={"name": "qwen3-tts-clone"},
            ),
            patch("qwen3_tts.server.app_models.save_config"),
        ):
            # Start the owner INSIDE the patch context — otherwise it calls
            # the real engine load_model from its thread.
            box, thread = _run_handler_thread(state)
            try:
                _poll_until(
                    self,
                    lambda: state.model_loads["clone"] is not None,
                    message="owner never claimed",
                )

                asyncio.run(
                    handle_update_model_config(state, cfg_req, lambda: {})
                )
                self.assertGreater(
                    state.model_config_epoch,
                    0,
                    "handle_update_model_config must bump model_config_epoch",
                )

                result = asyncio.run(handle_load_model(state, _req()))
            finally:
                release_owner.set()
                box.done.wait(timeout=5)
                thread.join(timeout=5)

        self.assertNotIn(
            "deduped",
            result,
            "a stale-epoch caller attached to the pre-update load — it "
            "would serve old weights under the new config",
        )
        self.assertEqual(result["status"], "loaded")
        self.assertEqual(
            calls,
            ["clone", "clone"],
            "the stale-epoch caller must take its own claim (second load)",
        )
        # C1 assign guard: the superseded owner finished AFTER the epoch-1
        # load — its stale-weights assignment must not have landed.
        self.assertIs(
            state.models["clone"],
            sentinel_new,
            "the superseded owner clobbered the newer-config model — "
            "the owner's assignment needs the identity check",
        )

    def test_in_flight_load_discarded_when_epoch_bumps_mid_flight(self):
        """Supersede is LAZY (fires at the next claim) — the owner must
        re-check the epoch at assign time, not rely on a later claimant.
        Without that check a load that started before /update-model-config
        installs OLD-config weights and answers 200 over the new config."""
        from fastapi import HTTPException

        from qwen3_tts.server.app_models import handle_load_model
        from qwen3_tts.server.model_loading import MODEL_LOAD_LOCK

        state = _make_state()
        calls = []
        release = threading.Event()
        sentinel = object()
        timer = threading.Timer(0.3, release.set)
        self.addCleanup(timer.cancel)
        timer.start()

        def _load(model_type, warmup=False):
            calls.append(model_type)
            release.wait(timeout=10)
            return sentinel

        async def _scenario():
            task = asyncio.create_task(handle_load_model(state, _req()))
            for _ in range(500):
                if state.model_loads["clone"] is not None:
                    break
                await asyncio.sleep(0.01)
            else:
                self.fail("owner never claimed")
            # Bump exactly as the mutators do — under the claim lock.
            with MODEL_LOAD_LOCK:
                state.model_config_epoch += 1
            release.set()
            with self.assertRaises(HTTPException) as ctx:
                await task
            return ctx

        with (
            patch("qwen3_tts.core.engine.load_model", side_effect=_load),
            patch(
                "qwen3_tts.core.config.get_model_info",
                return_value={"name": "qwen3-tts-clone"},
            ),
        ):
            ctx = asyncio.run(_scenario())

        self.assertEqual(
            ctx.exception.status_code,
            503,
            f"a load discarded for a stale epoch must not answer 200 "
            f"loaded: {ctx.exception.detail!r}",
        )
        self.assertEqual(ctx.exception.detail.get("error"), "load_in_progress")
        self.assertIsNone(
            state.models["clone"],
            "the stale-weights assignment landed after the epoch bump",
        )
        self.assertIsNone(state.model_loads["clone"], "claim must release")


@_skip
class TestWaiterDisconnect(unittest.TestCase):
    """The forwarded `request` must actually gate the waiter's poll loop."""

    def test_disconnected_client_stops_the_wait_immediately(self):
        from fastapi import HTTPException

        from qwen3_tts.server import model_loading
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        result, record = model_loading.claim_model_load(state, "clone")
        self.assertEqual(result, model_loading.ClaimResult.CLAIMED)
        self.addCleanup(
            lambda: model_loading.release_model_load(
                state, "clone", record, model_loading.LoadOutcome.CANCELLED
            )
        )

        class _Gone:
            async def is_disconnected(self):
                return True

        with patch.object(model_loading, "_POLL_INTERVAL_SEC", 0.01):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(handle_load_model(state, _req(), request=_Gone()))

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail.get("error"), "load_in_progress")

    def test_connected_client_gets_the_normal_attach_path(self):
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        calls = []
        release = threading.Event()
        self.addCleanup(release.set)
        sentinel = object()

        class _Here:
            async def is_disconnected(self):
                return False

        waiter_box = {"value": None, "error": None, "done": threading.Event()}

        def _load(model_type, warmup=False):
            calls.append(model_type)
            if len(calls) == 1:
                release.wait(timeout=10)
            return sentinel

        def _waiter():
            try:
                waiter_box["value"] = asyncio.run(
                    handle_load_model(state, _req(), request=_Here())
                )
            except BaseException as exc:  # noqa: BLE001 — the box IS the trap
                waiter_box["error"] = exc
            finally:
                waiter_box["done"].set()

        box, thread = None, None
        try:
            with (
                patch("qwen3_tts.core.engine.load_model", side_effect=_load),
                patch(
                    "qwen3_tts.core.config.get_model_info",
                    return_value={"name": "qwen3-tts-clone"},
                ),
            ):
                # Start the owner INSIDE the patch context (same trap as the
                # stale-epoch test above: outside, it calls the real engine).
                box, thread = _run_handler_thread(state)
                waiter = threading.Thread(target=_waiter, daemon=True)
                waiter.start()
                _poll_until(
                    self,
                    lambda: state.model_loads["clone"] is not None,
                    message="owner never claimed",
                )
                waiter.join(timeout=0.5)  # stays parked while the load blocks
                self.assertTrue(
                    waiter.is_alive(), "waiter finished while the load was blocked"
                )
                release.set()
                self.assertTrue(
                    waiter_box["done"].wait(timeout=5), "waiter never finished"
                )
        finally:
            if box is not None:
                box.done.wait(timeout=5)
                thread.join(timeout=5)

        self.assertIsNone(waiter_box["error"])
        self.assertEqual(calls, ["clone"])
        self.assertEqual(waiter_box["value"].get("deduped"), True)


@_skip
class TestUnloadDuringLoad(unittest.TestCase):
    """H4a: unload during an in-flight load is a 409, not a silent undo."""

    def test_unload_while_load_in_flight_is_409(self):
        from fastapi import HTTPException

        from qwen3_tts.server import model_loading
        from qwen3_tts.server.app_models import handle_unload_model

        state = _make_state()
        result, record = model_loading.claim_model_load(state, "clone")
        self.assertEqual(result, model_loading.ClaimResult.CLAIMED)
        self.addCleanup(
            lambda: model_loading.release_model_load(
                state, "clone", record, model_loading.LoadOutcome.CANCELLED
            )
        )

        with self.assertRaises(HTTPException) as ctx:
            handle_unload_model(state, _req())

        self.assertEqual(
            ctx.exception.status_code,
            409,
            f"unload during an in-flight load must 409, got "
            f"{ctx.exception.status_code}: {ctx.exception.detail!r}",
        )

    def test_completed_unload_bumps_epoch(self):
        from qwen3_tts.server.app_models import handle_unload_model

        state = _make_state()
        state.models["clone"] = object()
        epoch_before = state.model_config_epoch

        with patch("qwen3_tts.core.engine.unload_model_cleanup"):
            result = handle_unload_model(state, _req())

        self.assertEqual(result["status"], "unloaded")
        self.assertEqual(
            state.model_config_epoch,
            epoch_before + 1,
            "an unload invalidates in-flight claims — epoch must bump (C1)",
        )


@_skip
class TestWarmupFailureKeepsModel(unittest.TestCase):
    """W1/M10: a warm-up throw must not discard weights that loaded cleanly."""

    def test_warmup_throw_keeps_model_and_reports_warmup_failed(self):
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        sentinel = object()
        cleanup_calls = []

        with (
            patch(
                "qwen3_tts.core.engine.load_model", return_value=sentinel
            ),
            patch(
                "qwen3_tts.core.config.get_model_info",
                return_value={"name": "qwen3-tts-design"},
            ),
            patch(
                "qwen3_tts.core.engine.model_loader._warmup_disabled",
                return_value=False,
            ),
            patch(
                "qwen3_tts.core.engine.model_loader._warmup_model",
                side_effect=RuntimeError("warmup boom"),
            ),
            patch(
                "qwen3_tts.core.engine.unload_model_cleanup",
                side_effect=lambda: cleanup_calls.append(1),
            ),
        ):
            result = asyncio.run(handle_load_model(state, _req("design")))

        self.assertEqual(
            result.get("status"),
            "loaded",
            f"weights that loaded cleanly must be kept: {result!r}",
        )
        self.assertTrue(
            result.get("warmup_failed"),
            "the warm-up failure must be surfaced on the response",
        )
        self.assertIs(
            state.models["design"],
            sentinel,
            "a warm-up throw discarded a fully-built model (W1)",
        )
        self.assertEqual(
            cleanup_calls,
            [],
            "_recover_from_failed_load ran for a warm-up failure — the "
            "model was reclaimed even though it loaded",
        )
        self.assertIsNone(
            state.model_load_errors["design"],
            "model_load_errors must never record a warm-up failure — /health "
            "would report a usable model as broken",
        )
        self.assertIsNone(state.model_loads["design"], "claim must release")

    def test_warmup_failure_reaches_attached_waiter(self):
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        calls = []
        release = threading.Event()
        sentinel = object()
        timer = threading.Timer(0.3, release.set)
        self.addCleanup(timer.cancel)
        timer.start()

        def _load(model_type, warmup=False):
            calls.append(model_type)
            if len(calls) == 1:
                release.wait(timeout=10)
            return sentinel

        with (
            patch("qwen3_tts.core.engine.load_model", side_effect=_load),
            patch(
                "qwen3_tts.core.config.get_model_info",
                return_value={"name": "qwen3-tts-design"},
            ),
            patch(
                "qwen3_tts.core.engine.model_loader._warmup_disabled",
                return_value=False,
            ),
            patch(
                "qwen3_tts.core.engine.model_loader._warmup_model",
                side_effect=RuntimeError("warmup boom"),
            ),
        ):

            async def _both():
                return await asyncio.gather(
                    handle_load_model(state, _req("design")),
                    handle_load_model(state, _req("design")),
                    return_exceptions=True,
                )

            owner, waiter = asyncio.run(_both())

        self.assertEqual(calls, ["design"], "one construction, one warm-up")
        self.assertNotIsInstance(waiter, Exception)
        self.assertEqual(waiter.get("status"), "loaded")
        self.assertTrue(
            waiter.get("warmup_failed"),
            f"the attached waiter must also see warmup_failed: {waiter!r}",
        )
        self.assertTrue(owner.get("warmup_failed"))


@_skip
class TestBackgroundLoadWaits(unittest.TestCase):
    """H2/M5/M8: the startup loader waits on the record, never skips."""

    def _loader_thread(self, state):
        from qwen3_tts.server.app_lifespan import _background_load

        thread = threading.Thread(target=_background_load, args=(state,), daemon=True)
        thread.start()
        return thread

    def _loader_patches(self, load_side_effect):
        return (
            patch(
                "qwen3_tts.core.engine.load_model", side_effect=load_side_effect
            ),
            patch(
                "qwen3_tts.core.config.get_model_info",
                return_value={"name": "qwen3-tts-clone"},
            ),
            patch("qwen3_tts.server.app_lifespan.get_backend", return_value="mlx"),
        )

    def test_loader_waits_for_http_owned_load_m5(self):
        """Loader claims first; the HTTP duplicate attaches and waits."""
        state = _make_state(
            server_config={"models": {"clone": {"load_at_startup": True}}}
        )
        calls = []
        release = threading.Event()
        sentinel = object()
        self.addCleanup(release.set)

        patches = self._loader_patches(
            _blocking_load_spy(calls, release, sentinel)
        )
        with patches[0], patches[1], patches[2]:
            thread = self._loader_thread(state)
            self.addCleanup(lambda: thread.join(timeout=5))
            _poll_until(
                self,
                lambda: state.model_loads["clone"] is not None,
                message="loader never claimed",
            )

            box, handler_thread = _run_handler_thread(state)
            self.assertFalse(
                box.done.wait(timeout=0.5),
                "HTTP duplicate finished while the loader's load was still "
                "blocked — it re-loaded instead of waiting (M5)",
            )
            self.assertFalse(
                state.models_loaded.is_set(),
                "models_loaded set while the owned load is in flight",
            )

            release.set()
            self.assertTrue(box.done.wait(timeout=5), "waiter never finished")
            thread.join(timeout=5)

        self.assertFalse(handler_thread.is_alive())
        self.assertEqual(calls, ["clone"], "startup + HTTP must build once")
        self.assertEqual(box.value.get("deduped"), True)
        self.assertIs(state.models["clone"], sentinel)
        self.assertTrue(
            state.models_loaded.is_set(),
            "loader must still signal readiness after the shared load",
        )

    def test_ready_gate_holds_while_http_load_in_flight_m8(self):
        """HTTP owns; the loader must wait — /ready stays 503 until done."""
        state = _make_state(
            server_config={"models": {"clone": {"load_at_startup": True}}}
        )
        calls = []
        release = threading.Event()
        sentinel = object()
        self.addCleanup(release.set)

        patches = self._loader_patches(
            _blocking_load_spy(calls, release, sentinel)
        )
        with patches[0], patches[1], patches[2]:
            box, handler_thread = _run_handler_thread(state)
            self.addCleanup(lambda: handler_thread.join(timeout=5))
            _poll_until(
                self,
                lambda: state.model_loads["clone"] is not None,
                message="HTTP owner never claimed",
            )

            thread = self._loader_thread(state)
            self.addCleanup(lambda: thread.join(timeout=5))
            self.assertFalse(
                box.done.wait(timeout=0.5),
                "HTTP owner finished while its load was still blocked",
            )
            self.assertFalse(
                state.models_loaded.is_set(),
                "loader signalled readiness while the HTTP-owned load was "
                "still in flight — /ready would answer 200 early (H2)",
            )
            self.assertEqual(
                calls,
                ["clone"],
                "loader must not start its own construction while a claim "
                "is held (M8 double-build signature)",
            )

            release.set()
            self.assertTrue(box.done.wait(timeout=5))
            thread.join(timeout=5)

        self.assertEqual(calls, ["clone"])
        self.assertTrue(state.models_loaded.is_set())
        # The HTTP handler is the OWNER here (the loader attached) — no
        # deduped flag on an owner response.
        self.assertNotIn("deduped", box.value)

    def test_loader_records_error_when_http_owner_fails(self):
        """H2b: an owner failure must reach model_load_errors, not silence."""
        from fastapi import HTTPException

        state = _make_state(
            server_config={"models": {"clone": {"load_at_startup": True}}}
        )
        release = threading.Event()
        self.addCleanup(release.set)

        def _load(model_type, warmup=False):
            release.wait(timeout=10)
            raise RuntimeError("boom during load")

        patches = self._loader_patches(_load)
        with patches[0], patches[1], patches[2]:
            box, handler_thread = _run_handler_thread(state)
            self.addCleanup(lambda: handler_thread.join(timeout=5))
            _poll_until(
                self,
                lambda: state.model_loads["clone"] is not None,
                message="HTTP owner never claimed",
            )

            thread = self._loader_thread(state)
            self.addCleanup(lambda: thread.join(timeout=5))
            release.set()

            self.assertTrue(box.done.wait(timeout=5))
            thread.join(timeout=5)

        self.assertIsInstance(box.error, HTTPException)
        self.assertTrue(
            state.model_load_errors["clone"],
            "loader attached to a failed owner and recorded NOTHING — "
            "/health would go silent in exactly the raced case (H2)",
        )
        self.assertIsNone(state.models["clone"])

    def test_waiter_sees_startup_load_failure_message(self):
        """A waiter on a failed STARTUP load gets the cause, not a generic."""
        from fastapi import HTTPException

        state = _make_state(
            server_config={"models": {"clone": {"load_at_startup": True}}}
        )
        release = threading.Event()
        self.addCleanup(release.set)

        def _load(model_type, warmup=False):
            release.wait(timeout=10)
            raise RuntimeError("boom startup")

        patches = self._loader_patches(_load)
        with patches[0], patches[1], patches[2]:
            thread = self._loader_thread(state)
            self.addCleanup(lambda: thread.join(timeout=5))
            _poll_until(
                self,
                lambda: state.model_loads["clone"] is not None,
                message="loader never claimed",
            )

            box, handler_thread = _run_handler_thread(state)
            self.addCleanup(lambda: handler_thread.join(timeout=5))
            release.set()

            self.assertTrue(box.done.wait(timeout=5), "waiter never finished")
            thread.join(timeout=5)

        self.assertIsInstance(box.error, HTTPException)
        self.assertIn(
            "boom startup",
            str(box.error.detail.get("detail", "")),
            "the loader's release dropped the error message — attached "
            "waiters would surface a generic 503 instead of the cause",
        )


@_skip
class TestReleaseFirstWriterWins(unittest.TestCase):
    """A superseder's CANCELLED is never stomped by a stale owner's FAILED.

    Interleaving: the superseder marks the record CANCELLED and sets done;
    a waiter scheduled behind it may not run until AFTER the failing owner's
    release — without first-writer-wins it would read FAILED and mirror a
    500 for a load that was invalidated, instead of the retryable 503.
    """

    def test_late_failed_release_keeps_the_superseders_cancelled(self):
        from qwen3_tts.server import model_loading

        state = _make_state()
        result, record = model_loading.claim_model_load(state, "clone")
        self.assertEqual(result, model_loading.ClaimResult.CLAIMED)
        try:
            # Simulate the superseder: CANCELLED classification under the lock.
            with model_loading.MODEL_LOAD_LOCK:
                record.outcome = model_loading.LoadOutcome.CANCELLED
                record.error = "superseded by a newer model-config epoch"
                record.done.set()

            # The stale owner's failure lands late.
            model_loading.release_model_load(
                state,
                "clone",
                record,
                model_loading.LoadOutcome.FAILED,
                error="boom from the stale owner",
                code="load_failed",
                recovery="restart",
            )

            self.assertIs(
                record.outcome,
                model_loading.LoadOutcome.CANCELLED,
                "the stale owner's FAILED stomped the superseder's "
                "CANCELLED — a parked waiter would nondeterministically "
                "mirror a 500 for an invalidated load",
            )
            self.assertEqual(
                record.error, "superseded by a newer model-config epoch"
            )
        finally:
            model_loading.release_model_load(
                state, "clone", record, model_loading.LoadOutcome.CANCELLED
            )
        self.assertIsNone(state.model_loads["clone"], "slot must still clear")


@_skip
class TestWaiterTimeout(unittest.TestCase):
    """The waiter gives up retryable; the owner's claim stays untouched."""

    def test_timeout_is_retryable_503_and_keeps_owner_claim(self):
        from fastapi import HTTPException

        from qwen3_tts.server import model_loading
        from qwen3_tts.server.app_models import handle_load_model

        state = _make_state()
        result, record = model_loading.claim_model_load(state, "clone")
        self.assertEqual(result, model_loading.ClaimResult.CLAIMED)
        try:
            with (
                patch.object(
                    model_loading, "MODEL_LOAD_WAIT_TIMEOUT_SEC", 0.05
                ),
                patch.object(model_loading, "_POLL_INTERVAL_SEC", 0.01),
                patch(
                    "qwen3_tts.core.config.get_model_info",
                    return_value={"name": "qwen3-tts-clone"},
                ),
            ):
                with self.assertRaises(HTTPException) as ctx:
                    asyncio.run(handle_load_model(state, _req()))

            # Before this test's own cleanup releases the record: a timed-out
            # waiter must leave the owner's claim exactly as it found it.
            self.assertIs(
                state.model_loads["clone"],
                record,
                "the waiter must not release a claim it does not own",
            )
        finally:
            model_loading.release_model_load(
                state, "clone", record, model_loading.LoadOutcome.CANCELLED
            )

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail.get("error"), "load_in_progress")
        self.assertEqual(ctx.exception.detail.get("recovery"), "retry")

    def test_wait_budget_stays_below_client_timeout(self):
        """Drift guard: 870 < LOAD_MODEL_TIMEOUT_SEC (900)."""
        from qwen3_tts.core.http_client import LOAD_MODEL_TIMEOUT_SEC
        from qwen3_tts.server.model_loading import MODEL_LOAD_WAIT_TIMEOUT_SEC

        self.assertGreater(MODEL_LOAD_WAIT_TIMEOUT_SEC, 0)
        self.assertLess(
            MODEL_LOAD_WAIT_TIMEOUT_SEC,
            LOAD_MODEL_TIMEOUT_SEC,
            "the waiter must give up before the owner's own client does",
        )


@_skip
class TestClientErrorUnwrap(unittest.TestCase):
    """A classified 503 is useless if the client renders 'Unknown error'."""

    def test_cli_load_model_unwraps_nested_detail(self):
        from qwen3_tts.interface import generate_server

        src = inspect.getsource(generate_server.load_model_on_server)
        self.assertIn(
            "_error_payload",
            src,
            "load_model_on_server reads resp.json().get('error') flat — "
            "_error_response nests the body under detail, so the classified "
            "error reaches the CLI as 'Unknown error'",
        )

    def test_ui_toggle_model_unwraps_nested_detail(self):
        from qwen3_tts.interface.ui import model_management

        src = inspect.getsource(model_management.toggle_model)
        self.assertIn(
            "_error_payload",
            src,
            "toggle_model reads resp.json().get('error') flat — the nested "
            "detail payload must be unwrapped via _error_payload",
        )


if __name__ == "__main__":
    unittest.main()
