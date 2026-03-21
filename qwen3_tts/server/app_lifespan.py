"""Server lifecycle management — lifespan, cleanup, and infrastructure helpers.

Extracted from app.py to keep each module under 800 lines.
Contains: lifespan context manager, background model loading, cleanup,
auto-shutdown, ETA estimation, memory checking, error sanitization.
"""

import asyncio
import atexit
import json
import logging
import os
import re as _re
import secrets
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional

from qwen3_tts.core.config import (
    TOKEN_FILE,
    HISTORY_FILE,
    load_config,
    get_backend,
    get_eta_cache_ttl,
    cleanup_pid_file,
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

def _estimate_eta(app_state, text_length: int, elapsed_sec: float) -> Optional[float]:
    """Estimate remaining seconds from history data."""
    now = time.time()

    # Refresh cache if stale
    if now - app_state.eta_cache["last_updated"] > get_eta_cache_ttl():
        try:
            if not os.path.exists(HISTORY_FILE):
                app_state.eta_cache["median_rate"] = None
            else:
                with open(HISTORY_FILE, "r") as f:
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


_MEMORY_THRESHOLD_BYTES = 1024 * 1024 * 1024  # 1 GB


def _check_memory_available() -> tuple[bool, int]:
    """Check if sufficient system memory is available for generation.

    Returns:
        (ok, available_mb): ok is True if memory is above threshold,
        available_mb is the current available memory in MB.
    """
    if not _HAS_PSUTIL:
        return True, 0  # Skip check if psutil not installed
    mem = psutil.virtual_memory()
    available_mb = mem.available // (1024 * 1024)
    if mem.available < _MEMORY_THRESHOLD_BYTES:
        return False, available_mb
    if mem.available < _MEMORY_THRESHOLD_BYTES * 2:
        logger.warning("Low memory: %dMB available", available_mb)
    return True, available_mb


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
        auto_shutdown_minutes * 60,
        lambda: auto_shutdown(app_state)
    )
    app_state.shutdown_timer.daemon = True
    app_state.shutdown_timer.start()


def auto_shutdown(app_state):
    """Auto-shutdown due to inactivity."""
    logger.info("Auto-shutdown: No activity for %d minutes.",
                app_state.server_config.get("auto_shutdown_minutes", 0))
    cleanup_resources(app_state)
    sys.exit(0)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan — startup and shutdown."""
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

    # ETA cache
    app.state.eta_cache = {"median_rate": None, "last_updated": 0}

    # Model load error tracking
    app.state.model_load_errors = {"clone": None, "design": None, "custom": None}

    # Auto-shutdown timer
    app.state.shutdown_timer = None

    # Graceful shutdown event
    app.state.shutdown_event = asyncio.Event()

    # Server config (loaded from config.json)
    config = load_config()
    app.state.server_config = config.get("server", {})
    app.state.server_config["models"] = config.get("models", {})
    app.state.server_config["security"] = config.get("security", {})

    # Write token file
    with open(TOKEN_FILE, "w") as f:
        f.write(app.state.auth_token)
    os.chmod(TOKEN_FILE, 0o600)

    # Register atexit handler as safety net for cleanup
    atexit.register(cleanup_resources, app.state)

    logger.info("FastAPI server starting...")

    # Start background model loading
    loader = threading.Thread(target=_background_load, args=(app.state,), daemon=True)
    loader.start()

    yield

    # Shutdown
    logger.info("FastAPI server shutting down...")
    cleanup_resources(app.state)

    # Clean up token file
    try:
        os.unlink(TOKEN_FILE)
    except FileNotFoundError:
        pass


def _background_load(app_state):
    """Background thread that loads models at startup."""
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
        logger.info("Loading %d model(s): %s", len(models_to_load), ", ".join(models_to_load))

    for model_type in models_to_load:
        try:
            from qwen3_tts.core.config import get_model_info
            info = get_model_info(model_type)
            model_name = info.get("name", info.get("name_template", model_type))
            logger.info("Loading %s...", model_name)
            t0 = time.time()
            model = load_model(model_type)
            app_state.models[model_type] = model
            app_state.model_load_times[model_type] = round(time.time() - t0, 1)
            logger.info("Loaded %s model successfully in %.1fs.", model_type, app_state.model_load_times[model_type])
        except (ImportError, RuntimeError, OSError, ValueError, MemoryError) as e:
            error_msg = str(e)
            logger.error("Failed to load %s model: %s", model_type, error_msg, exc_info=True)
            # Sanitize before storing — /health is a public endpoint
            app_state.model_load_errors[model_type] = _sanitize_error(error_msg)

    # MLX prompt migration for torch backend
    if get_backend() == "torch":
        try:
            migrate_orphan_mlx_prompts(clone_model=app_state.models.get("clone"))
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            logger.warning("MLX prompt migration failed: %s", e)

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
                except Exception:
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
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)
    # Set shutdown event for graceful termination
    shutdown_event = getattr(app_state, "shutdown_event", None)
    if shutdown_event is not None:
        shutdown_event.set()
    cleanup_resources(app_state)
    sys.exit(0)
