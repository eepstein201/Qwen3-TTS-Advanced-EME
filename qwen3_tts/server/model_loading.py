"""Per-load records for /load-model — in-flight dedup (#214 item 3, Phase 2c).

``handle_load_model`` guarded only on *already loaded*, so while a load was
in flight a second POST /load-model started a second full weight construction
(~2.5 GB double allocation on MLX); both then assigned ``state.models[t]``,
orphaning one model. This module replaces that guard with a per-load record:
a duplicate caller AWAITS the in-flight load instead of racing it.

Design notes carried from the Phase 2c review (5-agent round + Gate A):

* **Per-load record, not a shared bool.** A shared ``models_loading`` flag is
  ABA-prone: a waiter attaches to *a flag*, not *a load*, so a stale flag
  from a finished load gets misattributed to the next one. The record carries
  its own ``done`` Event, outcome, and error; waiters classify from the
  record, never from shared dicts (C3).
* **Fail-closed gate.** ``MODEL_LOAD_LOCK`` is created at module scope so it
  cannot be absent — a mutual-exclusion primitive must never silently degrade
  (r1's ``nullcontext()`` fallback would have restored the exact bug it
  guards against, with no log line). The lock is NOT consulted on state, so
  mock states work; ``state.model_loads`` / ``model_config_epoch`` should be
  real containers on hand-built states — claim/release tolerate MagicMock
  auto-attributes, but writes to them do not persist.
* **Single-process assumption** (like ``inference_lock``): the record table
  is a plain dict on ``app.state``, coherent only within one server process.
  Multi-worker deployments each get their own gate.
* **Unlocked reads / GIL.** ``record.done.is_set()`` and the final response
  reads are lock-free; under free-threaded CPython a torn read is possible
  in theory. The owner writes every field before ``done.set()``, and
  ``threading.Event.set`` is atomic, so a waiter that observes ``done`` also
  observes the completed fields on CPython-GIL builds.
* **Epoch.** ``state.model_config_epoch`` is bumped by every mutator of the
  model slots outside this module (``/update-model-config`` nulls all three
  slots; ``/unload-model`` nulls one). A caller whose epoch differs from the
  in-flight record's does not attach — it would otherwise attach to a load
  built from the OLD config while /models reports the NEW one.
* **W1 — warm-up failure keeps the model.** The design warm-up runs BEFORE
  ``state.models[t] = model``; a warm-up throw used to send a fully-built
  model into ``_recover_from_failed_load``. Startup already treats warm-up as
  non-fatal (``app_lifespan._run_warmup_under_inference_lock``), so the HTTP
  path now matches: warm-up failure logs a warning, the model stays, and
  ``warmup_failed`` is surfaced on owner and waiter responses. It is
  deliberately NEVER written to ``model_load_errors`` — its only non-display
  read sits behind ``models.get(mode) is None`` (app_generation) and
  /health + /ready list non-None entries, so recording it would report a
  usable model as broken. No warm-up retry: the first real request pays the
  cold start, the same documented residual as startup.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

from qwen3_tts.server.validation import _error_response

logger = logging.getLogger(__name__)

__all__ = [
    "MODEL_LOAD_LOCK",
    "MODEL_LOAD_WAIT_TIMEOUT_SEC",
    "ClaimResult",
    "LoadOutcome",
    "WaitResult",
    "await_in_flight_load",
    "claim_model_load",
    "load_model_deduped",
    "release_model_load",
]

# Mutual exclusion for the claim table. Module scope ON PURPOSE: the gate is
# fail-closed — a mutual-exclusion primitive must never degrade to absent
# when a hand-built state object forgets to carry one. Mock states still work
# because no state attribute is consulted for the lock itself.
MODEL_LOAD_LOCK = threading.Lock()

# Bound on a duplicate caller's wait for the in-flight load. Kept below the
# client's LOAD_MODEL_TIMEOUT_SEC (900, core/http_client.py) so the waiter
# reports a retryable 503 while the owner's client is still listening; a load
# CAN exceed 900 s (see the comment there), in which case the waiter 503s
# after the owner's client has already given up. Await converts the retry
# storm into one queued wait — it does not eliminate that residual.
MODEL_LOAD_WAIT_TIMEOUT_SEC = 870

# Poll cadence for the waiter. Sub-second patched by tests (batch 3 runs the
# suite under a 180 s wall clock — never let a test wait the real 870 s).
_POLL_INTERVAL_SEC = 0.25

# ``request.is_disconnected()`` does I/O — probe it on a slow cadence, not
# every poll tick. Starlette does not auto-cancel a plain ``async def``
# handler, so without this a dead caller polls the full 870 s budget.
_DISCONNECT_CHECK_INTERVAL_SEC = 2.0


class LoadOutcome(str, Enum):
    """Terminal state of one load, written once by the owner."""

    PENDING = "pending"
    OK = "ok"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClaimResult(str, Enum):
    """Outcome of claiming the load slot for a model type."""

    CLAIMED = "claimed"  # caller owns the load and the record
    ATTACH = "attach"  # an in-flight load exists; wait on its record


class WaitResult(str, Enum):
    """What the waiter observed when it stopped waiting."""

    OK = "ok"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    DISCONNECTED = "disconnected"


@dataclass
class _LoadRecord:
    """One in-flight model load. Written by its owner; waited on by others.

    ``ok`` mirrors ``outcome is OK`` for cheap checks; ``outcome`` is the
    authoritative classification waiters branch on. The owner fills every
    field BEFORE ``done.set()`` so a waking waiter never sees a half-written
    record.
    """

    model_type: str
    epoch: int
    done: threading.Event = field(default_factory=threading.Event)
    error: str | None = None
    outcome: LoadOutcome = LoadOutcome.PENDING
    warmup_failed: bool = False
    # The owner's error classification, mirrored to waiters so an
    # import_error (config) doesn't reach them as load_failed/retry and
    # invite a doomed retry loop.
    code: str = "load_failed"
    recovery: str = "retry"


def _is_superseded(state, record) -> bool:
    """True when the record no longer matches current model config.

    Supersede is LAZY (it fires at the next claim), so the owner must check
    the epoch itself at its mutation points — assign, error write, recovery.
    Called under ``MODEL_LOAD_LOCK``; the epoch mutators also take the lock,
    making check+act atomic against a bump.
    """
    return record.outcome is LoadOutcome.CANCELLED or record.epoch != getattr(
        state, "model_config_epoch", 0
    )


def claim_model_load(state, model_type: str):
    """Atomically claim the load slot, or attach to the in-flight record.

    Returns ``(ClaimResult, record)``. On CLAIMED the returned record is the
    caller's to release; on ATTACH it is the in-flight owner's record.
    """
    with MODEL_LOAD_LOCK:
        epoch = getattr(state, "model_config_epoch", 0)
        current = state.model_loads.get(model_type)
        if current is not None and not current.done.is_set():
            if current.epoch == epoch:
                return ClaimResult.ATTACH, current
            # Stale config: the slot holds a load built before the epoch
            # bump. Wake its waiters (they re-evaluate) and supersede it —
            # the owner's later release is identity-checked and will not
            # disturb this new record.
            current.outcome = LoadOutcome.CANCELLED
            current.error = "superseded by a newer model-config epoch"
            current.done.set()
        record = _LoadRecord(model_type=model_type, epoch=epoch)
        state.model_loads[model_type] = record
        return ClaimResult.CLAIMED, record


def release_model_load(
    state,
    model_type: str,
    record,
    outcome: LoadOutcome,
    error: str | None = None,
    warmup_failed: bool = False,
    code: str = "load_failed",
    recovery: str = "retry",
) -> None:
    """Fill the record's terminal fields, wake waiters, clear the slot.

    Safe to call by a superseded owner: the slot is only cleared when it
    still holds THIS record (identity check under the lock). First terminal
    writer wins: a superseder's CANCELLED classification is never stomped by
    a later release carrying the stale owner's FAILED — a waiter parked on
    the record keeps its retryable 503 instead of nondeterministically
    receiving a 500 mirror of a load that was invalidated.
    """
    with MODEL_LOAD_LOCK:
        keep_superseder = (
            record.outcome is LoadOutcome.CANCELLED
            and outcome is not LoadOutcome.CANCELLED
        )
        if not keep_superseder:
            record.outcome = outcome
            record.error = error
            record.warmup_failed = warmup_failed
            record.code = code
            record.recovery = recovery
        record.done.set()
        if state.model_loads.get(model_type) is record:
            state.model_loads[model_type] = None


async def await_in_flight_load(record, request=None) -> WaitResult:
    """Wait for the record's owner to finish; classify what happened.

    Polls ``record.done`` — the waiter attaches to *this load*, so a later
    claim/release cycle can never be misattributed to it. ``request`` (when
    supplied) is probed for disconnects on a slow cadence so a dead caller
    stops polling instead of burning the full budget.
    """
    deadline = time.monotonic() + MODEL_LOAD_WAIT_TIMEOUT_SEC
    next_disconnect_check = 0.0
    while not record.done.is_set():
        if time.monotonic() >= deadline:
            logger.warning(
                "Gave up waiting for the in-flight %s load after %ss "
                "(logged once — a leaked claim and a slow cold load used to "
                "be indistinguishable here)",
                record.model_type,
                MODEL_LOAD_WAIT_TIMEOUT_SEC,
            )
            return WaitResult.TIMEOUT
        now = time.monotonic()
        if request is not None and now >= next_disconnect_check:
            next_disconnect_check = now + _DISCONNECT_CHECK_INTERVAL_SEC
            try:
                if await request.is_disconnected():
                    return WaitResult.DISCONNECTED
            except Exception:  # noqa: BLE001 — probe is best-effort
                pass  # nosec B110 — a failed disconnect probe must never break the wait
        await asyncio.sleep(_POLL_INTERVAL_SEC)
    if record.outcome is LoadOutcome.OK:
        return WaitResult.OK
    if record.outcome is LoadOutcome.CANCELLED:
        return WaitResult.CANCELLED
    return WaitResult.FAILED


def _recover_from_failed_load(state, model_type: str) -> None:
    """Reclaim backend memory and reset state after a failed load (PRF-5).

    A partially-constructed model leaves allocations behind that slow down
    every later generation (upstream mlx-audio #827 reports ~2.4x on Base
    cloning, and the server has a known-red "dies under repeated
    load/unload"). Recording the error is not enough — this mirrors the
    cleanup the unload path runs and drops the state that would otherwise let
    /models describe the model as healthy.

    Never raises: the caller still has to surface the original load failure.
    Moved here from app_models with the owner body (its only callers).
    """
    try:
        state.models[model_type] = None
        state.model_load_times.pop(model_type, None)

        from qwen3_tts.core.engine import unload_model_cleanup

        unload_model_cleanup()
    except Exception as e:  # noqa: BLE001 — recovery is non-fatal by design
        logger.warning(
            "Recovery after failed %s load did not complete: %s",
            _safe(model_type),
            _safe(e),
        )


def _safe(value) -> str:
    """Best-effort sanitize for log lines the owner writes itself.

    Full ``_sanitize_error`` lives in app_lifespan and is applied to
    ``model_load_errors`` (the /health surface). These lines only go to the
    local log, but keep the path-redaction habit anyway.
    """
    import re

    text = str(value)
    # Newline/control strip: keep the log-injection defense the old
    # handler's sanitize_log habit provided.
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"(?<![\w])/(?:[^\s\"']+)", "<path>", text)
    text = re.sub(r"[A-Za-z]:\\[^\s\"']+", "<path>", text)
    return text[:200]


async def load_model_deduped(state, model_type: str, request=None) -> dict | None:
    """Load a model, or attach to the in-flight load for it.

    The load itself runs WITHOUT inference_lock — minutes of download and
    weight construction must not starve /generate. The design warm-up
    inference afterwards runs UNDER inference_lock, acquired as a leaf
    (nothing else held), matching the global lock order where
    inference_lock is outermost (/generate at app_generation.py, /ws).
    Warm-up-vs-generation was the issue #192 trigger pair. /transcribe and
    /create-voice-prompt were the same class and are both serialized now —
    with this, every MLX inference reachable through the API serializes on
    inference_lock.

    Raises HTTPException on invalid input (the caller's job) or load
    failure; returns the status dict. The ``None`` return is unreachable in
    practice (the error paths end in ``_error_response``, which raises) and
    exists purely as the no-fall-through guard the python-review-fixes suite
    pins with a mocked no-raise ``_error_response``.
    """
    # Fast path: already loaded. Not a gate — the claim below is.
    if state.models.get(model_type) is not None:
        return {"status": "already_loaded", "model": model_type}

    result, record = claim_model_load(state, model_type)

    if result is ClaimResult.ATTACH:
        wait = await await_in_flight_load(record, request)
        if wait is WaitResult.OK:
            response: dict = {"status": "loaded", "model": model_type}
            if record.warmup_failed:
                response["warmup_failed"] = True
            response["deduped"] = True
            return response
        if wait is WaitResult.FAILED and record.error:
            # Mirror the owner's classification — with the owner's message.
            # An empty error must NOT become a 500 with a null body:
            # classify it as retryable instead (C3).
            _error_response(500, record.code, record.error, record.recovery)
        if wait is WaitResult.FAILED:
            _error_response(
                503, "load_failed", "the in-flight load failed", "retry"
            )
        # CANCELLED / TIMEOUT / DISCONNECTED are all retryable.
        _error_response(
            503, "load_in_progress", "the in-flight load ended early", "retry"
        )

    # CLAIMED — this caller owns the load. The imports are the FIRST
    # statements inside the try on purpose: (a) the #192 tests patch
    # qwen3_tts.core.engine.model_loader._warmup_model with a patch window
    # that closes before their post-lock wait_for, so the binding must be
    # captured before the lock wait; (b) an ImportError here (broken native
    # install) must flow through the excepts and the release in finally —
    # outside the try it would leak the claim and wedge this model type
    # into 870 s -> 503 forever.
    outcome = LoadOutcome.OK
    error: str | None = None
    warmup_failed = False
    code, recovery = "load_failed", "retry"
    superseded = False
    try:
        from qwen3_tts.core.config import get_backend, get_model_info
        from qwen3_tts.core.engine import load_model
        from qwen3_tts.core.engine.model_loader import (
            _warmup_disabled,
            _warmup_model,
        )

        info = get_model_info(model_type)
        model_name = info.get("name", info.get("name_template", model_type))
        logger.info("Loading %s...", _safe(model_name))
        t0 = time.time()
        # Load without the warm-up — the warm-up is serialized below so it
        # never runs concurrently with a generation (#192).
        model = await asyncio.to_thread(load_model, model_type, warmup=False)
        # Only design weights warm up (see _warmup_model's own guard — keep
        # in sync), and the knob is checked BEFORE the lock so ablation
        # runs don't queue behind generations for a no-op; clone/custom
        # skip the lock round-trip entirely.
        if model_type == "design" and not _warmup_disabled():
            try:
                async with state.inference_lock:
                    await asyncio.to_thread(
                        _warmup_model, model, model_type, get_backend()
                    )
            except Exception as warmup_exc:  # noqa: BLE001 — W1: best-effort
                # The load SUCCEEDED — startup treats warm-up as non-fatal
                # (app_lifespan._run_warmup_under_inference_lock) and the
                # HTTP path now matches. Never _recover_from_failed_load
                # here: discarding fully-built weights because a warm-up
                # inference hiccuped is exactly the double construction this
                # module exists to prevent. Never model_load_errors either —
                # see the module docstring.
                warmup_failed = True
                logger.warning(
                    "%s warm-up failed non-fatally; the model is loaded and "
                    "usable, the first request pays the cold start: %s",
                    _safe(model_type),
                    _safe(warmup_exc),
                )
        # Assign only if this load still matches current config. Supersede
        # is LAZY (it fires at the next claim), so the owner checks the
        # epoch itself here, under MODEL_LOAD_LOCK — the epoch mutators take
        # the same lock, making check+assign atomic against a bump. An
        # unconditional stale-weights assignment would clobber the
        # newer-config model while /models reports the new settings (C1).
        # Checked on the record itself, not via a slot read-back, so
        # hand-built states work too.
        with MODEL_LOAD_LOCK:
            superseded = _is_superseded(state, record)
            if superseded:
                # Release must not stomp the superseder's CANCELLED to OK.
                outcome = LoadOutcome.CANCELLED
            else:
                state.models[model_type] = model
                state.model_load_times[model_type] = round(time.time() - t0, 1)
                state.model_load_errors[model_type] = None
        if not superseded:
            logger.info(
                "Loaded %s model successfully in %.1fs.",
                _safe(model_type),
                state.model_load_times[model_type],
            )
    except asyncio.CancelledError:
        outcome = LoadOutcome.CANCELLED
        raise
    except ImportError as e:
        outcome = LoadOutcome.FAILED
        code, recovery = "import_error", "config"
        error = _sanitize(str(e))
        logger.error(
            "Backend not available for model loading %s: %s",
            _safe(model_type),
            error,
            exc_info=True,
        )
        # A superseded owner must not null a NEWER load or leave a stale
        # error for /health to report against it.
        with MODEL_LOAD_LOCK:
            if not _is_superseded(state, record):
                state.model_load_errors[model_type] = error
                _recover_from_failed_load(state, model_type)
        _error_response(500, code, error, recovery)
        return None  # explicit guard — _error_response raises, but this ensures no fall-through
    except (RuntimeError, OSError, ValueError) as e:
        outcome = LoadOutcome.FAILED
        code, recovery = "load_failed", "restart"
        error = _sanitize(str(e))
        logger.error(
            "Failed to load %s model: %s",
            _safe(model_type),
            error,
            exc_info=True,
        )
        with MODEL_LOAD_LOCK:
            if not _is_superseded(state, record):
                state.model_load_errors[model_type] = error
                _recover_from_failed_load(state, model_type)
        _error_response(500, code, error, recovery)
        return None  # explicit guard
    except Exception as e:  # noqa: BLE001 — mirrors the historical catch-all
        outcome = LoadOutcome.FAILED
        code, recovery = "unknown_error", "bug"
        error = _sanitize(str(e))
        logger.error(
            "Unexpected error loading %s model: %s",
            _safe(model_type),
            error,
            exc_info=True,
        )
        with MODEL_LOAD_LOCK:
            if not _is_superseded(state, record):
                state.model_load_errors[model_type] = error
                _recover_from_failed_load(state, model_type)
        _error_response(500, code, error, recovery)
        return None  # explicit guard
    finally:
        # Literally in finally: every except above ends in `raise` or in
        # _error_response (which raises — the `return` guards exist only for
        # the mocked-no-raise case). A leaked claim wedges this model type
        # into 870 s -> 503 forever.
        release_model_load(
            state,
            model_type,
            record,
            outcome,
            error=error,
            warmup_failed=warmup_failed,
            code=code,
            recovery=recovery,
        )

    if superseded:
        logger.warning(
            "%s load superseded by a newer model-config epoch mid-flight; "
            "discarding the stale-weights assignment",
            _safe(model_type),
        )
        # A discarded load must not answer 200 "loaded" — /models would
        # describe weights that are not installed.
        _error_response(
            503,
            "load_in_progress",
            "load superseded by a newer model-config epoch; retry to build "
            "with the new settings",
            "retry",
        )
        return None  # explicit guard

    response = {"status": "loaded", "model": model_type}
    if warmup_failed:
        response["warmup_failed"] = True
    return response


def _sanitize(msg: str) -> str:
    """app_lifespan._sanitize_error, resolved late to dodge an import loop."""
    from qwen3_tts.server.app_lifespan import _sanitize_error

    return _sanitize_error(msg)
