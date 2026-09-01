"""Server lifecycle management — lifespan, cleanup, and infrastructure helpers.

Extracted from app.py to keep each module under 800 lines.
Contains: lifespan context manager, background model loading, cleanup,
auto-shutdown, ETA estimation, memory checking, error sanitization.
"""

import asyncio
import atexit
import concurrent.futures
import fcntl
import json
import logging
import os
import re as _re
import secrets
import sys
import tempfile
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

from qwen3_tts.core.config import (
    HISTORY_FILE,
    LOCK_FILE,
    TOKEN_FILE,
    cleanup_pid_file,
    get_backend,
    get_eta_cache_ttl,
    load_config,
    sanitize_log,
)
from qwen3_tts.server.model_loading import (
    MODEL_LOAD_WAIT_TIMEOUT_SEC,
    ClaimResult,
    LoadOutcome,
    claim_model_load,
    release_model_load,
)

# Optional memory monitoring for OOM safeguard
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

logger = logging.getLogger("tts")


# ---------------------------------------------------------------------------
# Error sanitization — strip filesystem paths from public error responses
# ---------------------------------------------------------------------------

MAX_ERROR_MSG_LEN = 200  # max chars for sanitized error messages


def _sanitize_error(msg: str) -> str:
    """Remove absolute filesystem paths from error messages for public endpoints."""
    # Replace Unix-style absolute paths (e.g. /Users/foo/bar.pt -> <path>)
    sanitized = _re.sub(r"/[^\s\"']+", "<path>", str(msg))
    # Replace Windows-style absolute paths
    sanitized = _re.sub(r"[A-Za-z]:\\[^\s\"']+", "<path>", sanitized)
    return sanitized[:MAX_ERROR_MSG_LEN]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _estimate_eta(app_state, text_length: int, elapsed_sec: float) -> float | None:
    """Estimate remaining seconds from history data."""
    now = time.time()

    # Refresh cache if stale — lock to prevent concurrent read-modify-write races
    with app_state.eta_cache_lock:
        if now - app_state.eta_cache["last_updated"] > get_eta_cache_ttl():
            try:
                if not os.path.exists(HISTORY_FILE):
                    app_state.eta_cache["median_rate"] = None
                else:
                    with open(HISTORY_FILE) as f:
                        lines = deque(f, maxlen=20)
                    rates = []
                    for line in lines:
                        entry = json.loads(line)
                        dur = entry.get("duration_sec")
                        tl = entry.get("text_length")
                        if dur and tl and dur > 0:
                            rates.append(tl / dur)
                    if rates:
                        rates.sort()
                        app_state.eta_cache["median_rate"] = rates[len(rates) // 2]
                    else:
                        app_state.eta_cache["median_rate"] = None
            except (json.JSONDecodeError, OSError, KeyError, ValueError):
                app_state.eta_cache["median_rate"] = None
            app_state.eta_cache["last_updated"] = now

        median_rate = app_state.eta_cache["median_rate"]
    if median_rate is None or median_rate <= 0:
        return None

    estimated_total = text_length / median_rate
    remaining = max(0, estimated_total - elapsed_sec)
    return round(remaining, 1)


# Degradation thresholds for the IN-FLIGHT generation.
#
# Measured on an M2 Pro / MLX 1.7B-8bit: a healthy generation runs roughly
# 1-4 seconds per character. A wedged server was observed completing
# "22 chars, 7314.2s" — about 332 s/char — while /health still answered
# "status": "ok" and every model was reported loaded. Every generation-bearing
# E2E test then blew its timeout and read as a code regression.
#
# 30 s/char sits ~10x above healthy and ~10x below the observed pathology, so
# it will not fire on a merely slow generation. The elapsed floor exists
# because s/char is meaningless early: a 5-character request is "1 s/char"
# after five seconds without anything being wrong.
_DEGRADED_MIN_ELAPSED_SEC = 300.0
_DEGRADED_SEC_PER_CHAR = 30.0


def detect_degraded_generation(app_state, now: float | None = None) -> dict:
    """Report whether the IN-FLIGHT generation is pathologically slow.

    Deliberately measures the *active* request rather than completed history.
    The failure this exists to catch ran for two hours before completing, so a
    completed-samples design would not have raised anything until long after
    every dependent caller had already timed out and misattributed the failure.

    No new state is tracked: ``generation_state`` already carries ``active``,
    ``start_time`` and ``text_length``, and ``/generation-status`` already
    derives ``elapsed_sec`` the same way.

    Returns a dict with ``degraded`` plus the supporting numbers. Callers on
    public endpoints must take **only** the boolean — ``text_length`` and
    ``sec_per_char`` reveal the in-flight request's size, which is exactly what
    ``/generation-status`` strips for unauthenticated callers.
    """
    gen_state = app_state.generation_state
    now = time.time() if now is None else now

    result = {
        "degraded": False,
        "elapsed_sec": None,
        "sec_per_char": None,
        "threshold_sec_per_char": _DEGRADED_SEC_PER_CHAR,
    }
    if not gen_state.get("active"):
        return result

    elapsed = now - gen_state.get("start_time", 0.0)
    result["elapsed_sec"] = round(elapsed, 1)
    if elapsed < _DEGRADED_MIN_ELAPSED_SEC:
        return result

    text_length = gen_state.get("text_length") or 0
    if text_length <= 0:
        # Unknown size — fall back to elapsed time alone rather than dividing
        # by zero or silently declaring health.
        result["degraded"] = True
        return result

    sec_per_char = elapsed / text_length
    result["sec_per_char"] = round(sec_per_char, 2)
    result["degraded"] = sec_per_char > _DEGRADED_SEC_PER_CHAR
    return result


_MEMORY_THRESHOLD_BYTES = 1024 * 1024 * 1024  # 1 GB


def _check_memory_available() -> tuple[bool, int]:
    """Check if sufficient system memory is available for generation.

    Returns:
        (ok, available_mb): ok is True if memory is above threshold,
        available_mb is the current available memory in MB.
    """
    if not _HAS_PSUTIL:
        return True, 0  # Skip check if psutil not installed
    try:
        mem = psutil.virtual_memory()
    except (RuntimeError, OSError) as e:
        # psutil can raise transiently on some platforms (e.g. macOS
        # host_statistics64 syscall races under load). Degrade gracefully
        # rather than failing the generation request.
        logger.warning("Memory check unavailable (%s); skipping guard", e)
        return True, 0
    available_mb = mem.available // (1024 * 1024)
    if mem.available < _MEMORY_THRESHOLD_BYTES:
        return False, available_mb
    if mem.available < _MEMORY_THRESHOLD_BYTES * 2:
        logger.warning("Low memory: %dMB available", available_mb)
    return True, available_mb


def _acquire_startup_lock():
    """Acquire an exclusive, non-blocking lock guarding the server-start race.

    uvicorn's Server.startup() runs the ASGI lifespan startup (this function's
    caller) BEFORE it attempts to bind the TCP port. A losing `tts server
    start`/`tts ui` invocation that races an already-running (or
    already-starting) instance therefore used to reach _write_auth_token()
    and clobber the winner's still-valid token with one that belongs to a
    process about to exit on its own bind failure — every authenticated
    endpoint (including the Gradio UI's /models poll) then 401s against the
    server that actually won and kept running. Acquiring this lock first, and
    aborting startup immediately when it's already held, means a losing
    process never reaches the token write at all.

    Raises RuntimeError if another instance already holds the lock. Returns
    the open file object — keep it referenced for the process lifetime;
    closing it (or process exit) releases the lock.
    """
    fh = open(LOCK_FILE, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        fh.close()
        raise RuntimeError(
            f"Another TTS server instance is already running or starting "
            f"(lock held on {LOCK_FILE})"
        ) from e
    return fh


def _write_auth_token(token: str) -> None:
    """Write the auth token file atomically with restricted permissions.

    Atomic: serialize to a temp file in the same directory, fsync, then
    os.replace() onto TOKEN_FILE (atomic on POSIX). A concurrent reader (the
    Gradio UI /models poll, CLI clients) therefore never observes an empty or
    partially-written token -- the previous token stays valid until the new one
    is fully in place. Mirrors the atomic writes used for config
    (core/config/io.py) and the PID file (core/config/pid.py).

    Raises RuntimeError on failure. TTSClient discovers the token ONLY by
    reading TOKEN_FILE, so a write failure must abort startup rather than leave
    every authenticated endpoint unreachable.
    """
    try:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(TOKEN_FILE.parent, 0o700)
        fd, tmp_path = tempfile.mkstemp(
            dir=TOKEN_FILE.parent, prefix=".voice_server_token.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(token)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, TOKEN_FILE)
        except BaseException:
            # Leave any existing token untouched; clean up the temp file.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as e:
        raise RuntimeError(f"Cannot write auth token to {TOKEN_FILE}: {e}") from e
    logger.info("Auth token written to %s", TOKEN_FILE)


def _get_queue_size(app_state) -> int:
    """Return request queue size (thread-safe)."""
    with app_state.request_queue_lock:
        return len(app_state.request_queue)


def reset_activity_timer(app_state):
    """Reset the auto-shutdown timer on activity."""
    app_state.last_activity = time.time()

    auto_shutdown_minutes = app_state.server_config.get("auto_shutdown_minutes", 0)
    if auto_shutdown_minutes <= 0:
        return

    # Cancel existing timer
    if app_state.shutdown_timer is not None:
        app_state.shutdown_timer.cancel()

    # Start new timer
    app_state.shutdown_timer = threading.Timer(
        auto_shutdown_minutes * 60, lambda: auto_shutdown(app_state)
    )
    app_state.shutdown_timer.daemon = True
    app_state.shutdown_timer.start()


def auto_shutdown(app_state):
    """Auto-shutdown due to inactivity."""
    logger.info(
        "Auto-shutdown: No activity for %d minutes.",
        app_state.server_config.get("auto_shutdown_minutes", 0),
    )
    cleanup_resources(app_state)
    import signal

    os.kill(os.getpid(), signal.SIGTERM)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


async def _maybe_start_vllm_adapter(state) -> None:
    """Start VLLMAdapter and AsyncVLLMClient when backend is vllm.

    Uses protocol-based dependency injection:
    - VLLMAdapter manages subprocess lifecycle
    - AsyncVLLMClient provides async HTTP interface with circuit breaker
    - Health check uses AsyncVLLMClient to avoid blocking startup
    """
    config = load_config()
    vllm_config = config.get("vllm", {})

    # Check if vLLM is enabled
    if not vllm_config.get("enabled", False):
        return

    from qwen3_tts.core.engine_vllm import VLLMAdapter
    from qwen3_tts.server.vllm_client import AsyncVLLMClient

    # Create VLLM adapter with config parameters
    adapter = VLLMAdapter(
        gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", 0.9),
        max_model_len=vllm_config.get("max_model_len", 8192),
        dtype=vllm_config.get("dtype", "bfloat16"),
        audio_sample_rate=vllm_config.get("audio_sample_rate", 24000),
        audio_chunk_size=vllm_config.get("audio_chunk_size", 2000),
        mm_processor_name=vllm_config.get(
            "mm_processor_name", "Qwen/Qwen2-Audio-7B-Instruct"
        ),
    )

    # Start VLLM subprocess (may take up to 300s)
    await adapter.start()

    # Create async HTTP client for non-blocking requests
    base_url = f"http://127.0.0.1:{adapter.port}"
    client = AsyncVLLMClient(
        base_url=base_url,
        timeout=vllm_config.get("timeout", 300.0),
        circuit_breaker_failure_threshold=3,
    )

    # Verify health via async client (non-blocking)
    healthy = await client.health_check()
    if not healthy:
        logger.warning("vLLM health check failed after startup")
        if vllm_config.get("fallback_to_torch", True):
            logger.info("vLLM fallback enabled - will use torch/MLX instead")
        else:
            raise RuntimeError("vLLM health check failed and fallback is disabled")

    state.vllm_adapter = adapter
    state.vllm_client = client
    logger.info("VLLMAdapter and AsyncVLLMClient started and attached to app state")


async def _maybe_stop_vllm_adapter(state) -> None:
    """Stop VLLMAdapter and AsyncVLLMClient on shutdown if they were started."""
    # Stop async HTTP client first
    client = getattr(state, "vllm_client", None)
    if client is not None:
        await client.close()
        state.vllm_client = None
        logger.info("AsyncVLLMClient closed")

    # Stop VLLM subprocess
    adapter = getattr(state, "vllm_adapter", None)
    if adapter is not None:
        adapter.stop()
        state.vllm_adapter = None
        logger.info("VLLMAdapter stopped")


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan — startup and shutdown."""
    # Must be first: abort before touching any shared state (including
    # TOKEN_FILE) if another instance is already running or starting.
    # See _acquire_startup_lock for why this has to precede everything else.
    lock_fh = _acquire_startup_lock()

    # Initialize app.state
    app.state.auth_token = secrets.token_hex(32)
    app.state.models = {"clone": None, "design": None, "custom": None}
    app.state.model_load_times = {}
    app.state.generation_lock = asyncio.Lock()
    app.state.generation_state = {
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
    app.state.request_queue = set()
    app.state.request_queue_lock = threading.Lock()
    app.state.pending_requests = []  # [{id, text_preview, mode, queued_at}]
    app.state.pending_lock = asyncio.Lock()
    app.state.last_activity = time.time()
    app.state.models_loaded = threading.Event()
    app.state.gen_cache = {}
    app.state.gen_cache_lock = threading.Lock()

    # CRITICAL: Add inference_lock to prevent parallel OOM
    app.state.inference_lock = asyncio.Lock()

    # Captured for the startup loader thread: _background_load schedules its
    # warm-up inference onto this loop so it can take inference_lock (#192 —
    # no MLX inference may run concurrently with a generation).
    app.state.event_loop = asyncio.get_running_loop()

    # ETA cache
    app.state.eta_cache = {"median_rate": None, "last_updated": 0}
    app.state.eta_cache_lock = threading.Lock()

    # Model load error tracking
    app.state.model_load_errors = {"clone": None, "design": None, "custom": None}

    # Per-load records (Phase 2c, #214 item 3) — the single source of truth
    # for in-flight loads. claim/release in model_loading own the mutations;
    # /models derives its `loading` display flag from this. The epoch is
    # bumped by /update-model-config and /unload-model — the mutators of the
    # model slots outside this module — so waiters never attach to a load
    # built from superseded settings.
    app.state.model_loads = {"clone": None, "design": None, "custom": None}
    app.state.model_config_epoch = 0

    # Auto-shutdown timer
    app.state.shutdown_timer = None

    # vLLM adapter and client (None unless backend="vllm")
    app.state.vllm_adapter = None
    app.state.vllm_client = None

    # Graceful shutdown event
    app.state.shutdown_event = asyncio.Event()

    # Server config (loaded from config.json)
    config = load_config()
    app.state.server_config = config.get("server", {})
    app.state.server_config["models"] = config.get("models", {})
    app.state.server_config["security"] = config.get("security", {})

    # Write token file (create directory with restricted permissions). This is
    # the only channel by which TTSClient discovers the token, so a write
    # failure must abort startup rather than silently leaving auth unreachable.
    _write_auth_token(app.state.auth_token)

    # Register atexit handler as safety net for cleanup
    atexit.register(cleanup_resources, app.state)

    logger.info("FastAPI server starting...")

    # Start background model loading
    loader = threading.Thread(target=_background_load, args=(app.state,), daemon=True)
    loader.start()

    # Start vLLM adapter if backend="vllm"
    await _maybe_start_vllm_adapter(app.state)

    yield

    # Shutdown
    logger.info("FastAPI server shutting down...")
    await _maybe_stop_vllm_adapter(app.state)
    cleanup_resources(app.state)

    # Clean up token file — only if it contains OUR token.
    # A competing instance (e.g. PM2 crash loop) that failed to bind
    # must not delete the token written by the real server.
    # IMPORTANT: Don't delete token on normal shutdown - clients need it!
    # Only delete if this is a crash cleanup scenario.
    try:
        with open(TOKEN_FILE) as f:
            on_disk = f.read().strip()
        # Only delete if token doesn't match (indicates crash scenario)
        # Never delete our own valid token on normal shutdown
        if on_disk != app.state.auth_token:
            logger.warning(
                f"Token mismatch on disk vs memory - cleaning up: {TOKEN_FILE}"
            )
            os.unlink(TOKEN_FILE)
        else:
            logger.info(f"Token file preserved for client use: {TOKEN_FILE}")
    except (FileNotFoundError, OSError):
        pass

    fcntl.flock(lock_fh, fcntl.LOCK_UN)
    lock_fh.close()


# Bounds the loader thread's wait for the locked warm-up. One runaway
# generation chunk at the 4096-token ceiling is ~328s (12.5 Hz); a
# multi-chunk /generate can hold inference_lock longer still (~660s
# documented). 600 covers the single-chunk case with slack; longer queues
# abandon the warm-up wait (best-effort — the first real request pays the
# cold start) rather than stalling /ready behind an unbounded wait.
_STARTUP_WARMUP_TIMEOUT_SEC = 600


def _run_warmup_under_inference_lock(app_state, model, model_type):
    """Run the design load-time warm-up serialized on inference_lock (#192).

    The startup loader is a plain thread and cannot await the asyncio lock,
    so the locked warm-up is scheduled onto the server's event loop
    (app_state.event_loop, captured in lifespan) and this thread blocks
    until it completes or times out. Best-effort by design — the load has
    already succeeded at this point: if the loop is gone (shutdown race)
    or scheduling fails, the warm-up is skipped — never run
    unsynchronized, never failing the load around it. On timeout the WAIT
    is abandoned and the scheduled warm-up is cancelled if it has not
    started yet; one already running finishes under the lock (still
    serialized — safety holds), it is just no longer waited for.

    This closes the warm-up-vs-generation pair; /transcribe and
    /create-voice-prompt were the same class and are both serialized
    now — all MLX inference reachable through the API serializes on
    inference_lock (#192).
    """
    if model_type != "design":
        # _warmup_model no-ops for clone/custom — don't queue behind a
        # generation just to do nothing (keep in sync with its own guard).
        return
    try:
        from qwen3_tts.core.engine.model_loader import (
            _warmup_disabled,
            _warmup_model,
        )

        if _warmup_disabled():
            # TTS_SKIP_WARMUP — check before the lock so ablation runs
            # don't queue behind generations for a no-op.
            return

        loop = getattr(app_state, "event_loop", None)
        if loop is None or loop.is_closed():
            logger.info(
                "Skipping %s warm-up (event loop unavailable)", model_type
            )
            return

        async def _locked_warmup():
            async with app_state.inference_lock:
                await asyncio.to_thread(
                    _warmup_model, model, model_type, get_backend()
                )

        future = asyncio.run_coroutine_threadsafe(_locked_warmup(), loop)
        try:
            future.result(timeout=_STARTUP_WARMUP_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            # Abandon the wait, not the safety: cancel stops the coroutine
            # if it hasn't started; if it is mid-warm-up it finishes under
            # the lock. The load has already succeeded either way.
            future.cancel()
            logger.warning(
                "%s warm-up still queued behind inference_lock after %ss — "
                "no longer waiting (cold start deferred to first request)",
                model_type,
                _STARTUP_WARMUP_TIMEOUT_SEC,
            )
    except Exception as e:  # noqa: BLE001 — best-effort; the load succeeded
        logger.warning(
            "Startup warm-up for %s could not run (non-fatal): %s",
            sanitize_log(model_type),
            sanitize_log(e),
        )


def _background_load(app_state):
    """Background thread that loads models at startup."""
    # The ENTIRE body runs under a finally that signals readiness — the try
    # opens on the first statement, before even the engine import. This thread
    # is the ONLY producer of models_loaded; if an exception escapes it, /ready
    # answers 503 forever and every waiter hangs, with no recorded reason. The
    # thread is also a daemon, so the failure is otherwise invisible.
    #
    # Two escapes lived above an earlier, later-opening try: the function-local
    # engine import (a broken native install raises ImportError) and
    # ``settings.get`` while building models_to_load (a hand-edited config.json
    # mapping a model to a bool raises AttributeError). Never move work above
    # this try. Guarded by tests/test_fastapi_app_ext2.py::TestBackgroundLoad.
    try:
        from qwen3_tts.core.engine import load_model, migrate_orphan_mlx_prompts

        config = app_state.server_config
        models_config = config.get("models", {})

        if not models_config:
            models_config = {"clone": {"load_at_startup": True}}

        models_to_load = []
        for model_type, settings in models_config.items():
            if settings.get("load_at_startup", False):
                models_to_load.append(model_type)

        if not models_to_load:
            logger.warning("No models configured to load at startup.")
        else:
            logger.info(
                "Loading %d model(s): %s",
                len(models_to_load),
                ", ".join(models_to_load),
            )

        for model_type in models_to_load:
            if app_state.models.get(model_type) is not None:
                # Already loaded — a concurrent HTTP load finished while the
                # loader worked through earlier startup models. Claiming and
                # building again here would orphan a ~2.5 GB copy: the exact
                # defect this phase exists to kill.
                logger.info(
                    "%s already loaded; skipping the startup build.", model_type
                )
                continue
            result, record = claim_model_load(app_state, model_type)
            if result is ClaimResult.ATTACH:
                # An HTTP-owned load is already building this model. WAIT,
                # never skip: skipping would drop the model and fall through
                # to models_loaded.set() below — which gates /ready — so
                # /ready would answer 200 while the HTTP-owned load is
                # minutes away (H2). Blocking is fine: this is a plain
                # thread, and the HTTP owner assigns + releases the record.
                logger.info(
                    "Waiting for the in-flight %s load owned by an HTTP "
                    "request instead of building it a second time...",
                    model_type,
                )
                record.done.wait(timeout=MODEL_LOAD_WAIT_TIMEOUT_SEC)
                if not record.done.is_set():
                    logger.error(
                        "In-flight %s load still running after %ss — giving "
                        "up the wait (the load may complete on its own; "
                        "recorded in model_load_errors so /health surfaces "
                        "it)",
                        model_type,
                        MODEL_LOAD_WAIT_TIMEOUT_SEC,
                    )
                    app_state.model_load_errors[model_type] = (
                        "startup wait for the in-flight load timed out"
                    )
                elif record.outcome is not LoadOutcome.OK:
                    logger.error(
                        "The in-flight %s load startup was waiting on failed: %s",
                        model_type,
                        record.error,
                    )
                    app_state.model_load_errors[model_type] = (
                        record.error or "the in-flight load failed"
                    )
                else:
                    logger.info(
                        "In-flight %s load completed; model is loaded.", model_type
                    )
                continue

            outcome = LoadOutcome.OK
            error_msg: str | None = None
            try:
                from qwen3_tts.core.config import get_model_info

                info = get_model_info(model_type)
                model_name = info.get("name", info.get("name_template", model_type))
                logger.info("Loading %s...", sanitize_log(model_name))
                t0 = time.time()
                # Load without the warm-up; it runs serialized below (#192).
                model = load_model(model_type, warmup=False)
                _run_warmup_under_inference_lock(app_state, model, model_type)
                app_state.models[model_type] = model
                app_state.model_load_times[model_type] = round(time.time() - t0, 1)
                logger.info(
                    "Loaded %s model successfully in %.1fs.",
                    model_type,
                    app_state.model_load_times[model_type],
                )
            except (ImportError, RuntimeError, OSError, ValueError, MemoryError) as e:
                outcome = LoadOutcome.FAILED
                error_msg = str(e)
                logger.error(
                    "Failed to load %s model: %s", model_type, error_msg, exc_info=True
                )
                # Sanitize before storing — /health is a public endpoint
                app_state.model_load_errors[model_type] = _sanitize_error(error_msg)
            except Exception as e:  # noqa: BLE001 — one model must not kill startup
                # Mirrors handle_load_model's catch-all (app_models.py). Library
                # API drift surfaces as AttributeError/TypeError/KeyError, which
                # the tuple above misses; without this the loader thread dies
                # mid-list, later models never load and no error is recorded.
                outcome = LoadOutcome.FAILED
                error_msg = str(e)
                logger.error(
                    "Unexpected error loading %s model: %s",
                    sanitize_log(model_type),
                    sanitize_log(error_msg),
                    exc_info=True,
                )
                app_state.model_load_errors[model_type] = _sanitize_error(error_msg)
            finally:
                # The startup loader deliberately does NOT run
                # _recover_from_failed_load (pre-existing asymmetry, tracked
                # as a follow-up) — but it always releases its claim, with
                # the error so attached waiters surface the cause instead of
                # a generic 503.
                release_model_load(
                    app_state,
                    model_type,
                    record,
                    outcome,
                    error=_sanitize_error(error_msg) if error_msg else None,
                )

        # MLX prompt migration for torch backend
        if get_backend() == "torch":
            try:
                migrate_orphan_mlx_prompts(clone_model=app_state.models.get("clone"))
            except Exception as e:  # noqa: BLE001 — migration is best-effort
                logger.warning(
                    "MLX prompt migration failed: %s", sanitize_log(e)
                )
    except Exception as e:  # noqa: BLE001 — startup must never die silently
        # Reached only by the setup work the per-model handlers cannot cover
        # (engine import, config parsing). The finally below still signals
        # readiness, so /ready answers 200 and /health reports the reason
        # rather than the server hanging at 503 with nothing logged.
        logger.error(
            "Background model loading failed before any model could load: %s",
            sanitize_log(e),
            exc_info=True,
        )
    finally:
        app_state.models_loaded.set()
        logger.info("Background model loading complete.")


def cleanup_resources(app_state):
    """Clean up resources on shutdown."""
    # Cancel shutdown timer
    shutdown_timer = getattr(app_state, "shutdown_timer", None)
    if shutdown_timer is not None:
        shutdown_timer.cancel()

    # Clean up models
    models = getattr(app_state, "models", None)
    if models is not None:
        for name in ("clone", "design", "custom"):
            model = models.get(name)
            if model is not None:
                try:
                    del model
                    models[name] = None
                except (TypeError, RuntimeError, OSError):
                    pass

    # Clean up generation cache temp files
    gen_cache = getattr(app_state, "gen_cache", None)
    if gen_cache:
        for entry in gen_cache.values():
            main_file = entry.get("main_file") or entry.get("file")
            if main_file and os.path.exists(main_file):
                try:
                    os.remove(main_file)
                except OSError:
                    pass
        gen_cache.clear()

    # Clean up PID file
    cleanup_pid_file()


def cleanup_pid(app_state):
    """Clean up PID file and initiate graceful shutdown."""
    shutdown_timer = getattr(app_state, "shutdown_timer", None)
    if shutdown_timer is not None:
        shutdown_timer.cancel()
    cleanup_pid_file()
    try:
        with open(TOKEN_FILE) as f:
            on_disk = f.read().strip()
        if on_disk == getattr(app_state, "auth_token", None):
            os.unlink(TOKEN_FILE)
    except (FileNotFoundError, OSError):
        pass
    # Set shutdown event for graceful termination
    shutdown_event = getattr(app_state, "shutdown_event", None)
    if shutdown_event is not None:
        shutdown_event.set()
    cleanup_resources(app_state)
    sys.exit(0)
