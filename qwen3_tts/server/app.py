#!/usr/bin/env python3
"""FastAPI TTS server with async support and real streaming.

- Real async/await
- Proper streaming via asyncio.Queue
- app.state for worker-safe state
- inference_lock for GPU serialization
"""

import asyncio
import atexit
import hashlib
import json
import logging
import logging.handlers
import os
import re
import secrets
import signal
import struct
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger("tts")

from qwen3_tts.core.config import (
    CONFIG_PATH,
    VOICE_PROMPTS_DIR,
    PID_FILE,
    TOKEN_FILE,
    LOG_FILE,
    MODEL_INFO,
    MLX_MODEL_INFO,
    CUSTOM_VOICE_SPEAKERS,
    HISTORY_FILE,
    IN_COLAB,
    load_config,
    save_config,
    get_backend,
    get_default_clone_prompt,
    get_torch_dtype_name,
    get_mlx_quantization,
    get_model_size,
    get_model_info,
    get_mlx_model_name,
    get_generation_cache_max,
    get_eta_cache_ttl,
)

# Pre-computed valid speaker names (keys + display names)
_VALID_SPEAKER_NAMES = frozenset(CUSTOM_VOICE_SPEAKERS.keys()) | frozenset(
    v["name"] for v in CUSTOM_VOICE_SPEAKERS.values()
)


def _get_eta_cache_ttl() -> int:
    """Get ETA cache TTL from config (cached for performance)."""
    return get_eta_cache_ttl()


def _get_gen_cache_max() -> int:
    """Get generation cache max size from config (cached for performance)."""
    return get_generation_cache_max()



# ---------------------------------------------------------------------------
# Pydantic models for request validation
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    mode: str = "clone"
    prompt_file: Optional[str] = None
    voice_description: str = ""
    language: str = "English"
    speaker: Optional[str] = None
    instruct: str = ""
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048
    seed: Optional[int] = None
    max_chunk_chars: Optional[int] = None
    x_vector_only_mode: bool = False


class LoadModelRequest(BaseModel):
    model_type: str


class UnloadModelRequest(BaseModel):
    model_type: str


class UpdateModelConfigRequest(BaseModel):
    model_size: Optional[str] = None
    mlx_quantization: Optional[str] = None


class UpdateStartupConfigRequest(BaseModel):
    clone: Optional[bool] = None
    design: Optional[bool] = None
    custom: Optional[bool] = None


class DeletePromptRequest(BaseModel):
    name: str


class RenamePromptRequest(BaseModel):
    old_name: str
    new_name: str


# ---------------------------------------------------------------------------
# Pydantic models for response validation
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    recovery: str = "retry"


class GenerateResult(BaseModel):
    index: int
    audio_base64: Optional[str] = None
    sample_rate: int


class GenerateResponse(BaseModel):
    results: List[GenerateResult]


class HealthResponse(BaseModel):
    status: str
    backend: Optional[str] = None
    model_size: Optional[str] = None
    clone_model_loaded: Optional[bool] = None
    design_model_loaded: Optional[bool] = None
    custom_model_loaded: Optional[bool] = None
    model_load_times: Optional[dict] = None
    model_load_errors: Optional[dict] = None
    mlx_quantization: Optional[str] = None
    dtype: Optional[str] = None


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def verify_auth(request: Request) -> None:
    """Verify Bearer token for protected endpoints."""
    if request.url.path in ("/health", "/generation-status"):
        return
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not secrets.compare_digest(token, request.app.state.auth_token):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _validate_generation_request(req: GenerateRequest, security_config: dict) -> None:
    """Shared validation for /generate and /generate-stream.

    Raises HTTPException for:
    - Path traversal in prompt_file
    - Invalid speaker name for custom mode
    - Invalid mode
    """
    # Path traversal check
    if req.prompt_file and (".." in req.prompt_file or "/" in req.prompt_file):
        raise HTTPException(
            status_code=400,
            detail="Invalid prompt_file: path traversal not allowed",
        )

    # Speaker validation for custom mode
    if req.mode == "custom" and req.speaker:
        speaker_key = req.speaker.lower() if isinstance(req.speaker, str) else ""
        if speaker_key not in CUSTOM_VOICE_SPEAKERS and req.speaker not in _VALID_SPEAKER_NAMES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown speaker: {req.speaker}. Valid: {', '.join(CUSTOM_VOICE_SPEAKERS.keys())}",
            )

    # Mode validation
    if req.mode not in ("clone", "design", "custom"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {req.mode}. Must be clone, design, or custom",
        )


def _error_response(status_code: int, error: str, detail: str = "", recovery: str = "retry") -> None:
    """Raise HTTPException with standardized error format.

    Args:
        status_code: HTTP status code
        error: Short error identifier
        detail: Detailed error message
        recovery: Suggested recovery action

    Raises:
        HTTPException with structured detail dict
    """
    raise HTTPException(
        status_code=status_code,
        detail={"error": error, "detail": detail, "recovery": recovery},
    )


def _validate_prompt_name(name: str) -> Optional[tuple]:
    """Validate prompt name — returns error tuple or None."""
    if not name or not name.strip():
        return {"error": "Missing prompt name", "recovery": "config"}, 400
    name = name.strip()
    if len(name) > 255:
        return {"error": "Prompt name too long", "recovery": "config"}, 400
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', name):
        return {"error": "Invalid prompt name: only alphanumeric, dash, underscore, dot allowed", "recovery": "config"}, 400
    if ".." in name:
        return {"error": "Invalid prompt name", "recovery": "config"}, 400
    return None


def _strip_extension(name: str) -> str:
    """Strip .pt, .wav, or .txt extension from name."""
    base = name
    for ext in (".pt", ".wav", ".txt"):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    return base


def _gen_cache_key(text: str, mode: str, gen_params: dict, prompt_file: str = None,
                   voice_description: str = None, speaker: str = None, instruct: str = None) -> str:
    """Generate a hash key for generation cache lookup."""
    key_parts = [text, mode, str(sorted(gen_params.items()))]
    if prompt_file:
        key_parts.append(prompt_file)
    if voice_description:
        key_parts.append(voice_description)
    if speaker:
        key_parts.append(speaker)
    if instruct:
        key_parts.append(instruct)
    raw = "|".join(key_parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _estimate_eta(app_state, text_length: int, elapsed_sec: float) -> Optional[float]:
    """Estimate remaining seconds from history data."""
    now = time.time()

    # Refresh cache if stale
    if now - app_state.eta_cache["last_updated"] > _get_eta_cache_ttl():
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
    if median_rate is None:
        return None

    estimated_total = text_length / median_rate
    remaining = max(0, estimated_total - elapsed_sec)
    return round(remaining, 1)


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
    # Signal uvicorn to stop gracefully
    os.kill(os.getpid(), signal.SIGTERM)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
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
        except Exception as e:
            error_msg = str(e)
            logger.error("Failed to load %s model: %s", model_type, error_msg, exc_info=True)
            app_state.model_load_errors[model_type] = error_msg

    # MLX prompt migration for torch backend
    if get_backend() == "torch":
        try:
            migrate_orphan_mlx_prompts(clone_model=app_state.models.get("clone"))
        except Exception as e:
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
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except OSError:
        pass


def cleanup_pid(app_state):
    """Clean up PID file and initiate graceful shutdown."""
    shutdown_timer = getattr(app_state, "shutdown_timer", None)
    if shutdown_timer is not None:
        shutdown_timer.cancel()
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)
    # Set shutdown event for graceful termination
    shutdown_event = getattr(app_state, "shutdown_event", None)
    if shutdown_event is not None:
        shutdown_event.set()
    cleanup_resources(app_state)
    # Signal uvicorn to stop gracefully
    os.kill(os.getpid(), signal.SIGTERM)


# Create FastAPI app
app = FastAPI(
    title="Qwen3-TTS Server",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS: allow Gradio UI and local development
_cors_origins = ["http://localhost:7860", "http://127.0.0.1:7860"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Public endpoints (no auth)
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Health check endpoint."""
    state = request.app.state
    reset_activity_timer(state)

    if not state.models_loaded.is_set():
        from fastapi.responses import JSONResponse
        data = {"status": "loading"}
        # Include model load errors even during loading
        data["model_load_errors"] = {
            k: v for k, v in state.model_load_errors.items() if v is not None
        }
        return JSONResponse(content=data, status_code=503)

    backend = get_backend()
    data = {
        "status": "ok",
        "backend": backend,
        "model_size": get_model_size(),
        "clone_model_loaded": state.models.get("clone") is not None,
        "design_model_loaded": state.models.get("design") is not None,
        "custom_model_loaded": state.models.get("custom") is not None,
        "model_load_times": state.model_load_times,
    }
    # Include model load errors if any
    data["model_load_errors"] = {
        k: v for k, v in state.model_load_errors.items() if v is not None
    }
    if backend == "mlx":
        data["mlx_quantization"] = get_mlx_quantization()
    else:
        data["dtype"] = get_torch_dtype_name()
    return data


@app.get("/ready")
async def ready(request: Request):
    """Readiness probe endpoint for Kubernetes-style deployments.

    Returns 503 while models are loading, 200 when ready.
    Unlike /health which returns loading status, this endpoint is intended
    for container orchestration readiness probes.
    """
    state = request.app.state
    reset_activity_timer(state)

    if not state.models_loaded.is_set():
        raise HTTPException(status_code=503, detail="Models not loaded")

    return {"status": "ready"}


@app.get("/generation-status")
async def generation_status(request: Request):
    """Get current generation status."""
    state = request.app.state
    gen_state = dict(state.generation_state)
    if gen_state["active"]:
        gen_state["elapsed_sec"] = round(time.time() - gen_state["start_time"], 1)
        gen_state["eta_sec"] = _estimate_eta(state, gen_state["text_length"], gen_state["elapsed_sec"])
    return gen_state


# ---------------------------------------------------------------------------
# Protected endpoints (require auth)
# ---------------------------------------------------------------------------

@app.get("/stats")
async def stats(request: Request, _auth: None = Depends(verify_auth)):
    """Get server statistics."""
    state = request.app.state
    reset_activity_timer(state)

    idle_seconds = int(time.time() - state.last_activity)
    auto_shutdown_minutes = state.server_config.get("auto_shutdown_minutes", 0)

    from qwen3_tts.core.engine import voice_prompt_cache_info
    cache_info = voice_prompt_cache_info()

    backend = get_backend()
    stats_data = {
        "status": "ok",
        "backend": backend,
        "clone_model_loaded": state.models.get("clone") is not None,
        "design_model_loaded": state.models.get("design") is not None,
        "custom_model_loaded": state.models.get("custom") is not None,
        "voice_prompts_cached": cache_info.currsize,
        "voice_prompts_cache_hits": cache_info.hits,
        "idle_seconds": idle_seconds,
        "auto_shutdown_minutes": auto_shutdown_minutes if auto_shutdown_minutes > 0 else "disabled",
        "generation_queue_size": len(state.request_queue),
    }
    if backend == "mlx":
        stats_data["mlx_quantization"] = get_mlx_quantization()
    else:
        stats_data["dtype"] = get_torch_dtype_name()

    # GPU memory stats (lazy torch import)
    try:
        import torch
        if torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory()
                stats_data["mps_memory_allocated_mb"] = round(allocated / (1024 * 1024), 2)
            except (AttributeError, RuntimeError):
                stats_data["mps_memory_allocated_mb"] = "unavailable"

        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                stats_data["cuda_memory_allocated_mb"] = round(allocated / (1024 * 1024), 2)
                stats_data["cuda_memory_reserved_mb"] = round(reserved / (1024 * 1024), 2)
            except (AttributeError, RuntimeError):
                pass
    except ImportError:
        pass

    # MLX memory stats
    if backend == "mlx":
        try:
            import mlx.core as mx
            try:
                active_mem = mx.get_active_memory()
                peak_mem = mx.get_peak_memory()
            except AttributeError:
                active_mem = mx.metal.get_active_memory()
                peak_mem = mx.metal.get_peak_memory()
            stats_data["mlx_memory_active_mb"] = round(active_mem / (1024 * 1024), 2)
            stats_data["mlx_memory_peak_mb"] = round(peak_mem / (1024 * 1024), 2)
        except (ImportError, AttributeError, RuntimeError):
            pass

    return stats_data


@app.get("/models")
async def list_models(request: Request, _auth: None = Depends(verify_auth)):
    """List model status."""
    state = request.app.state
    reset_activity_timer(state)

    backend = get_backend()
    model_size = get_model_size()

    size_model_info = MODEL_INFO.get(model_size, MODEL_INFO["1.7B"])
    size_mlx_info = MLX_MODEL_INFO.get(model_size, MLX_MODEL_INFO["1.7B"])

    models_data = {}
    for model_type, info in size_model_info.items():
        loaded = state.models.get(model_type) is not None
        models_cfg = state.server_config.get("models", {})
        load_at_startup = models_cfg.get(model_type, {}).get("load_at_startup", False)

        entry = {
            "loaded": loaded,
            "description": info["description"],
            "memory_mb": info["memory_mb"],
            "repo_id": info["name"],
            "load_at_startup": load_at_startup,
            "load_time_sec": state.model_load_times.get(model_type),
        }
        if backend == "mlx":
            mlx_info = size_mlx_info.get(model_type)
            if mlx_info:
                entry["repo_id"] = get_mlx_model_name(model_type)
                entry["memory_mb"] = mlx_info["memory_mb"]
        models_data[model_type] = entry

    return {"models": models_data, "backend": backend, "model_size": model_size}


@app.post("/load-model")
async def load_model_endpoint(request: Request, req: LoadModelRequest, _auth: None = Depends(verify_auth)):
    """Load a model on demand."""
    state = request.app.state
    reset_activity_timer(state)

    model_type = req.model_type

    valid_types = ("clone", "design", "custom")
    if model_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type: {model_type}. Valid: {', '.join(valid_types)}",
        )

    # Check if already loaded
    if state.models.get(model_type) is not None:
        return {"status": "already_loaded", "model": model_type}

    # Load the model
    try:
        from qwen3_tts.core.engine import load_model
        from qwen3_tts.core.config import get_model_info

        info = get_model_info(model_type)
        model_name = info.get("name", info.get("name_template", model_type))
        logger.info("Loading %s...", model_name)
        t0 = time.time()
        model = load_model(model_type)
        state.models[model_type] = model
        state.model_load_times[model_type] = round(time.time() - t0, 1)
        logger.info("Loaded %s model successfully in %.1fs.", model_type, state.model_load_times[model_type])
        # Clear any previous load error for this model
        state.model_load_errors[model_type] = None
    except ImportError as e:
        logger.error("Backend not available for model loading %s: %s", model_type, e, exc_info=True)
        state.model_load_errors[model_type] = str(e)
        raise HTTPException(
            status_code=500,
            detail={"error": "import_error", "message": str(e)},
        )
    except (RuntimeError, OSError, ValueError) as e:
        logger.error("Failed to load model %s: %s", model_type, e, exc_info=True)
        state.model_load_errors[model_type] = str(e)
        raise HTTPException(
            status_code=500,
            detail={"error": "load_failed", "message": str(e)},
        )
    except Exception as e:
        logger.error("Unexpected error loading model %s: %s", model_type, e, exc_info=True)
        state.model_load_errors[model_type] = str(e)
        raise HTTPException(
            status_code=500,
            detail={"error": "unknown_error", "message": str(e)},
        )

    return {"status": "loaded", "model": model_type}


@app.post("/unload-model")
async def unload_model(request: Request, req: UnloadModelRequest, _auth: None = Depends(verify_auth)):
    """Unload a model to free memory."""
    state = request.app.state
    reset_activity_timer(state)

    model_type = req.model_type

    valid_types = ("clone", "design", "custom")
    if model_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type: {model_type}. Valid: {', '.join(valid_types)}",
        )

    # Check if generation is active for this mode
    if state.generation_state["active"] and state.generation_state["mode"] == model_type:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot unload {model_type} model while generation is active",
        )

    if state.models.get(model_type) is None:
        return {"status": "already_unloaded", "model": model_type}

    state.models[model_type] = None

    from qwen3_tts.core.engine import unload_model_cleanup
    unload_model_cleanup()

    # Invalidate generation cache
    with state.gen_cache_lock:
        for entry in state.gen_cache.values():
            try:
                main_file = entry.get("main_file") or entry.get("file")
                if main_file and os.path.exists(main_file):
                    os.remove(main_file)
            except OSError:
                pass
        state.gen_cache.clear()

    state.model_load_times.pop(model_type, None)
    logger.info("Unloaded %s model.", model_type)

    return {"status": "unloaded", "model": model_type}


@app.post("/update-model-config")
async def update_model_config(request: Request, req: UpdateModelConfigRequest, _auth: None = Depends(verify_auth)):
    """Update model size and/or quantization settings."""
    state = request.app.state
    reset_activity_timer(state)

    new_size = req.model_size
    new_quant = req.mlx_quantization

    if not new_size and not new_quant:
        raise HTTPException(
            status_code=400,
            detail="At least one of model_size or mlx_quantization required",
        )

    valid_sizes = ("1.7B", "0.6B")
    valid_quants = ("4bit", "8bit", "bf16")

    if new_size and new_size not in valid_sizes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_size: {new_size}. Valid: {', '.join(valid_sizes)}",
        )

    if new_quant and new_quant not in valid_quants:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mlx_quantization: {new_quant}. Valid: {', '.join(valid_quants)}",
        )

    config = load_config()
    if "advanced" not in config:
        config["advanced"] = {}

    changes = []
    if new_size:
        config["advanced"]["model_size"] = new_size
        changes.append(f"model_size={new_size}")
    if new_quant:
        config["advanced"]["mlx_quantization"] = new_quant
        changes.append(f"mlx_quantization={new_quant}")

    save_config(config)

    # Unload all models so new settings take effect
    async with state.generation_lock:
        for name in ("clone", "design", "custom"):
            state.models[name] = None

    # Invalidate generation cache
    with state.gen_cache_lock:
        for entry in state.gen_cache.values():
            try:
                main_file = entry.get("main_file") or entry.get("file")
                if main_file and os.path.exists(main_file):
                    os.remove(main_file)
            except OSError:
                pass
        state.gen_cache.clear()

    # Sync audio loader cache if config changed
    new_loader = config.get("advanced", {}).get("audio_loader")
    if new_loader:
        try:
            from qwen3_tts.core.engine import set_audio_loader
            set_audio_loader(new_loader)
        except (ValueError, ImportError):
            pass

    logger.info("Model config updated: %s. Models unloaded. Generation cache cleared.", ", ".join(changes))

    return {
        "status": "config_updated",
        "changes": changes,
        "models_unloaded": True,
        "note": "New model will be loaded on next generation",
    }


@app.post("/update-startup-config")
async def update_startup_config(request: Request, req: UpdateStartupConfigRequest, _auth: None = Depends(verify_auth)):
    """Update which models load at startup in config.json."""
    state = request.app.state
    reset_activity_timer(state)

    valid_types = ("clone", "design", "custom")
    changes = []
    config = load_config()
    if "models" not in config:
        config["models"] = {}

    for model_type in valid_types:
        val = getattr(req, model_type, None)
        if val is not None:
            val_bool = bool(val)
            if model_type not in config["models"]:
                config["models"][model_type] = {}
            config["models"][model_type]["load_at_startup"] = val_bool
            changes.append(f"{model_type}={'on' if val_bool else 'off'}")

    if not changes:
        raise HTTPException(status_code=400, detail="No valid model types provided")

    save_config(config)

    # Update server config cache
    state.server_config["models"] = config.get("models", {})

    logger.info("Startup config updated: %s", ", ".join(changes))
    return {"status": "updated", "changes": changes}


@app.get("/prompts")
async def list_prompts(request: Request, _auth: None = Depends(verify_auth)):
    """List voice prompts."""
    state = request.app.state
    reset_activity_timer(state)

    backend = get_backend()
    try:
        all_files = set(os.listdir(VOICE_PROMPTS_DIR))
    except OSError:
        return {"prompts": []}

    if backend == "mlx":
        # MLX uses .wav+.txt pairs
        wav_files = {f[:-4] for f in all_files if f.endswith('.wav')}
        txt_files = {f[:-4] for f in all_files if f.endswith('.txt')}
        names = sorted(wav_files & txt_files)
        prompts = [f"{n}.wav" for n in names]
    else:
        # Torch uses .pt files, but also include voices with .wav+.txt
        pt_names = {f[:-3] for f in all_files if f.endswith('.pt')}
        wav_names = {f[:-4] for f in all_files if f.endswith('.wav')}
        all_names = sorted(pt_names | wav_names)
        prompts = [f"{n}.pt" for n in all_names]

    return {"prompts": prompts}


@app.post("/delete-prompt")
async def delete_prompt(request: Request, req: DeletePromptRequest, _auth: None = Depends(verify_auth)):
    """Delete a voice prompt and all its format files."""
    state = request.app.state
    reset_activity_timer(state)

    name = req.name
    err = _validate_prompt_name(name)
    if err:
        raise HTTPException(status_code=err[1], detail=err[0]["error"])

    base = _strip_extension(name)

    # Find and delete all matching files
    files_removed = []
    for ext in (".pt", ".wav", ".txt"):
        path = os.path.join(VOICE_PROMPTS_DIR, f"{base}{ext}")
        if os.path.exists(path):
            os.remove(path)
            files_removed.append(f"{base}{ext}")

    if not files_removed:
        raise HTTPException(status_code=404, detail=f"Voice prompt '{base}' not found")

    # If deleted prompt was the default, clear it
    try:
        config = load_config()
        current_default = config.get("default_clone_prompt", "")
        default_base = _strip_extension(current_default)
        if default_base == base:
            config["default_clone_prompt"] = ""
            save_config(config)
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    # Clear voice prompt cache
    from qwen3_tts.core.engine import clear_voice_prompt_cache
    clear_voice_prompt_cache()

    logger.info("Deleted voice prompt '%s': %s", base, files_removed)
    return {"status": "deleted", "name": base, "files_removed": files_removed}


@app.post("/rename-prompt")
async def rename_prompt(request: Request, req: RenamePromptRequest, _auth: None = Depends(verify_auth)):
    """Rename a voice prompt (all format files) with rollback on partial failure."""
    state = request.app.state
    reset_activity_timer(state)

    for name_val in (req.old_name, req.new_name):
        err = _validate_prompt_name(name_val)
        if err:
            raise HTTPException(status_code=err[1], detail=err[0]["error"])

    old_base = _strip_extension(req.old_name)
    new_base = _strip_extension(req.new_name)

    if old_base == new_base:
        raise HTTPException(status_code=400, detail="Old and new names are the same")

    # Collision check
    for ext in (".pt", ".wav", ".txt"):
        if os.path.exists(os.path.join(VOICE_PROMPTS_DIR, f"{new_base}{ext}")):
            raise HTTPException(status_code=409, detail=f"Voice prompt '{new_base}' already exists")

    # Check that at least one old file exists
    old_exists = any(
        os.path.exists(os.path.join(VOICE_PROMPTS_DIR, f"{old_base}{ext}"))
        for ext in (".pt", ".wav", ".txt")
    )
    if not old_exists:
        raise HTTPException(status_code=404, detail=f"Voice prompt '{old_base}' not found")

    # Rename with rollback on partial failure
    renamed = []
    try:
        for ext in (".pt", ".wav", ".txt"):
            old_path = os.path.join(VOICE_PROMPTS_DIR, f"{old_base}{ext}")
            new_path = os.path.join(VOICE_PROMPTS_DIR, f"{new_base}{ext}")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                renamed.append((new_path, old_path))
    except OSError as e:
        # Rollback
        for current, rollback_to in renamed:
            try:
                os.rename(current, rollback_to)
            except OSError:
                pass
        logger.error("Rename failed %s -> %s: %s", req.old_name, req.new_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Rename failed. Check server logs for details.")

    # Update default if the renamed prompt was the default
    try:
        config = load_config()
        current_default = config.get("default_clone_prompt", "")
        default_base = _strip_extension(current_default)
        if default_base == old_base:
            if current_default.endswith(".pt"):
                config["default_clone_prompt"] = f"{new_base}.pt"
            else:
                config["default_clone_prompt"] = new_base
            save_config(config)
    except (OSError, json.JSONDecodeError, KeyError):
        pass

    # Clear voice prompt cache
    from qwen3_tts.core.engine import clear_voice_prompt_cache
    clear_voice_prompt_cache()

    files_renamed = [os.path.basename(new) for new, _ in renamed]
    logger.info("Renamed voice prompt '%s' -> '%s': %s", old_base, new_base, files_renamed)
    return {"status": "renamed", "old_name": old_base, "new_name": new_base, "files_renamed": files_renamed}


@app.get("/preview-prompt")
async def preview_prompt(request: Request, _auth: None = Depends(verify_auth)):
    """Return the .wav file for a voice prompt as audio/wav."""
    state = request.app.state
    reset_activity_timer(state)

    name = request.query_params.get("name", "")
    err = _validate_prompt_name(name)
    if err:
        raise HTTPException(status_code=err[1], detail=err[0]["error"])

    base = _strip_extension(name)
    wav_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.wav")

    if not os.path.exists(wav_path):
        raise HTTPException(status_code=404, detail=f"No .wav file found for prompt '{base}'")

    return FileResponse(wav_path, media_type="audio/wav")


@app.get("/prompt-details")
async def prompt_details(request: Request, _auth: None = Depends(verify_auth)):
    """Return metadata for voice prompts."""
    state = request.app.state
    reset_activity_timer(state)

    name = request.query_params.get("name")

    # Get current default
    current_default = get_default_clone_prompt() or ""
    default_base = _strip_extension(current_default)

    def _prompt_info(base):
        """Build metadata dict for a single prompt."""
        formats = []
        total_size = 0
        created = None
        for ext in (".pt", ".wav", ".txt"):
            path = os.path.join(VOICE_PROMPTS_DIR, f"{base}{ext}")
            if os.path.exists(path):
                formats.append(ext)
                total_size += os.path.getsize(path)
                mtime = os.path.getmtime(path)
                if created is None or mtime < created:
                    created = mtime
        return {
            "name": base,
            "formats": formats,
            "size_bytes": total_size,
            "created": created,
            "is_default": (base == default_base),
        }

    if name:
        err = _validate_prompt_name(name)
        if err:
            raise HTTPException(status_code=err[1], detail=err[0]["error"])
        base = _strip_extension(name)
        info = _prompt_info(base)
        if not info["formats"]:
            raise HTTPException(status_code=404, detail=f"Voice prompt '{base}' not found")
        return info

    # All prompts
    try:
        all_files = os.listdir(VOICE_PROMPTS_DIR)
    except OSError:
        return {"prompts": []}

    # Collect unique base names
    bases = set()
    for f in all_files:
        for ext in (".pt", ".wav", ".txt"):
            if f.endswith(ext):
                bases.add(f[:-len(ext)])
                break

    prompts = [_prompt_info(b) for b in sorted(bases)]
    return {"prompts": prompts}


@app.post("/cancel-generation")
async def cancel_generation(request: Request, _auth: None = Depends(verify_auth)):
    """Cancel the current streaming generation."""
    state = request.app.state
    reset_activity_timer(state)

    async with state.generation_lock:
        if not state.generation_state["active"]:
            return {"status": "no_active_generation"}
        state.generation_state["cancelled"] = True
        logger.info("Generation cancellation requested")
    return {
        "status": "cancellation_requested",
        "generation_id": state.generation_state.get("generation_id")
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: Request, req: GenerateRequest, _auth: None = Depends(verify_auth)):
    """Generate audio from text."""
    state = request.app.state
    reset_activity_timer(state)

    # Validate and normalize request
    security = state.server_config.get("security", {})
    max_text_length = security.get("max_text_length", 10000)
    max_batch_size = security.get("max_batch_size", 20)

    # Support both text and texts
    if req.text:
        texts = [req.text]
    elif req.texts:
        texts = req.texts
    else:
        raise HTTPException(status_code=400, detail="No text provided")

    if isinstance(texts, str):
        texts = [texts]

    if len(texts) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(texts)} exceeds limit of {max_batch_size}",
        )

    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            raise HTTPException(status_code=400, detail=f"Text at index {i} is empty or invalid")
        if len(t) > max_text_length:
            raise HTTPException(
                status_code=400,
                detail=f"Text at index {i} exceeds {max_text_length} character limit ({len(t)} chars)",
            )

    mode = req.mode
    if mode not in ("clone", "design", "custom"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode}. Must be clone, design, or custom",
        )

    prompt_file = req.prompt_file
    if prompt_file and (".." in prompt_file or "/" in prompt_file):
        raise HTTPException(status_code=400, detail="Invalid prompt_file: path traversal not allowed")

    # Check speaker for custom mode
    speaker = req.speaker
    if mode == "custom" and speaker:
        speaker_key = speaker.lower() if isinstance(speaker, str) else ""
        if speaker_key not in CUSTOM_VOICE_SPEAKERS and speaker not in _VALID_SPEAKER_NAMES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown speaker: {speaker}. Valid: {', '.join(CUSTOM_VOICE_SPEAKERS.keys())}",
            )

    # Check if required model is loaded
    model = state.models.get(mode)
    if model is None:
        from qwen3_tts.core.config import get_model_info
        info = get_model_info(mode)
        detail = info.get('description', '')
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_not_loaded",
                "detail": f"The '{mode}' model is not loaded. {detail}",
                "recovery": "restart",
                "model_type": mode,
            },
        )

    # Generation parameters
    gen_params = {
        "temperature": req.temperature,
        "top_k": req.top_k,
        "top_p": req.top_p,
        "repetition_penalty": req.repetition_penalty,
        "max_new_tokens": req.max_new_tokens,
    }
    if req.seed is not None:
        gen_params["seed"] = req.seed

    voice_description = req.voice_description
    language = req.language
    instruct = req.instruct
    x_vector_only_mode = req.x_vector_only_mode
    max_chunk_chars = req.max_chunk_chars

    # Track this request in queue
    request_id = id(request)
    state.request_queue.add(request_id)

    try:
        import io, base64
        import soundfile as sf
        from qwen3_tts.core.engine import load_voice_prompt, run_inference

        # Pre-lock cache check
        pre_lock_cache_keys = {}
        pre_lock_results = {}
        for i, text in enumerate(texts):
            cache_key = _gen_cache_key(
                text, mode, gen_params,
                prompt_file=prompt_file,
                voice_description=voice_description,
                speaker=speaker, instruct=instruct,
            )
            pre_lock_cache_keys[i] = cache_key
            with state.gen_cache_lock:
                entry = state.gen_cache.get(cache_key)
            if entry:
                cache_file = entry.get("main_file") or entry.get("file")
                if cache_file and os.path.exists(cache_file):
                    with open(cache_file, "rb") as f:
                        b64_audio = base64.b64encode(f.read()).decode("utf-8")
                    pre_lock_results[i] = {"index": i, "audio_base64": b64_audio, "sample_rate": entry["sample_rate"]}
                    logger.info("Generation cache hit (pre-lock) for text %d/%d", i + 1, len(texts))

        # If ALL texts hit cache, skip the lock entirely
        if len(pre_lock_results) == len(texts):
            results = [pre_lock_results[i] for i in range(len(texts))]
            state.request_queue.discard(request_id)
            return {"results": results}

        # Acquire inference_lock for GPU serialization (generation_lock used only for state updates)
        async with state.inference_lock:
            results = []

            for i, text in enumerate(texts):
                # Use pre-lock cache hit if available
                if i in pre_lock_results:
                    results.append(pre_lock_results[i])
                    continue

                # Post-lock cache check
                cache_key = pre_lock_cache_keys[i]
                with state.gen_cache_lock:
                    entry = state.gen_cache.get(cache_key)
                if entry:
                    cache_file = entry.get("main_file") or entry.get("file")
                    if cache_file and os.path.exists(cache_file):
                        with open(cache_file, "rb") as f:
                            b64_audio = base64.b64encode(f.read()).decode("utf-8")
                        results.append({"index": i, "audio_base64": b64_audio, "sample_rate": entry["sample_rate"]})
                        logger.info("Generation cache hit (post-lock) for text %d/%d", i + 1, len(texts))
                    continue

                # Brief lock to set generation state
                async with state.generation_lock:
                    state.generation_state.update({
                        "active": True,
                        "start_time": time.time(),
                        "text_length": len(text),
                        "mode": mode,
                        "batch_index": i,
                        "batch_total": len(texts),
                    })

                # Prepare mode-specific params
                voice_prompt = None
                if mode == "clone":
                    if not prompt_file:
                        raise HTTPException(status_code=400, detail="prompt_file required for clone mode")
                    voice_prompt = load_voice_prompt(prompt_file)
                    if voice_prompt is None:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Voice prompt not found: {prompt_file}",
                        )

                def _chunk_progress(chunk_idx, chunk_total):
                    state.generation_state.update({
                        "chunk_index": chunk_idx,
                        "chunk_total": chunk_total,
                    })

                # Run inference (offloaded to thread pool to avoid blocking event loop)
                wav, sr = await asyncio.to_thread(
                    run_inference,
                    model=model,
                    text=text,
                    mode=mode,
                    gen_params=gen_params,
                    language=language,
                    voice_prompt=voice_prompt,
                    voice_description=voice_description,
                    speaker=speaker,
                    instruct=instruct,
                    max_chunk_chars=max_chunk_chars,
                    progress_callback=_chunk_progress,
                    x_vector_only_mode=x_vector_only_mode,
                )

                # Encode audio to base64 WAV in memory
                buf = io.BytesIO()
                sf.write(buf, wav, sr, format="WAV")
                b64_audio = base64.b64encode(buf.getvalue()).decode("utf-8")

                # Store persistent cache file for future hits
                cache_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                cache_file.close()  # Close handle before sf.write to avoid leak
                os.chmod(cache_file.name, 0o600)
                sf.write(cache_file.name, wav, sr)

                with state.gen_cache_lock:
                    if len(state.gen_cache) >= _get_gen_cache_max():
                        oldest_key = min(state.gen_cache, key=lambda k: state.gen_cache[k]["timestamp"])
                        old_entry = state.gen_cache.pop(oldest_key)
                        old_main = old_entry.get("main_file")
                        if old_main and os.path.exists(old_main):
                            try:
                                os.remove(old_main)
                            except OSError:
                                pass
                    state.gen_cache[cache_key] = {
                        "main_file": cache_file.name,
                        "sample_rate": sr,
                        "timestamp": time.time(),
                    }

                results.append({"index": i, "audio_base64": b64_audio, "sample_rate": sr})

            # Content negotiation: return binary WAV if Accept header contains audio/wav
            accept = request.headers.get("accept", "application/json")
            if "audio/wav" in accept and len(results) == 1:
                # Single text generation with audio/wav Accept: return binary WAV directly
                result = results[0]
                if result.get("audio_base64"):
                    audio_bytes = base64.b64decode(result["audio_base64"])
                    return Response(
                        content=audio_bytes,
                        media_type="audio/wav",
                        headers={"X-Sample-Rate": str(result["sample_rate"])},
                    )

            return {"results": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Audio generation failed",
                "detail": "An internal error occurred. Check server logs for details.",
                "recovery": "retry",
            },
        )
    finally:
        # Clear generation state
        state.generation_state.update({
            "active": False,
            "start_time": 0.0,
            "text_length": 0,
            "mode": "",
            "batch_index": 0,
            "batch_total": 0,
            "chunk_index": 0,
            "chunk_total": 0,
        })
        if request_id in state.request_queue:
            state.request_queue.discard(request_id)


@app.post("/generate-stream")
async def generate_stream(request: Request, req: GenerateRequest, _auth: None = Depends(verify_auth)):
    """Stream audio generation — returns chunked audio as it's produced."""
    state = request.app.state
    reset_activity_timer(state)

    # Validate request
    security = state.server_config.get("security", {})
    max_text_length = security.get("max_text_length", 10000)

    if not req.text:
        raise HTTPException(status_code=400, detail="No text provided")

    text = req.text
    if len(text) > max_text_length:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds {max_text_length} character limit ({len(text)} chars)",
        )

    # Shared validation (path traversal, speaker, mode)
    _validate_generation_request(req, security)

    # Check if required model is loaded
    model = state.models.get(mode)
    if model is None:
        error_msg = state.model_load_errors.get(mode, "Model not loaded")
        raise HTTPException(
            status_code=503,
            detail={"error": "model_not_loaded", "message": error_msg, "model_type": mode},
        )

    # Generation parameters
    gen_params = {
        "temperature": req.temperature,
        "top_k": req.top_k,
        "top_p": req.top_p,
        "repetition_penalty": req.repetition_penalty,
        "max_new_tokens": req.max_new_tokens,
    }
    if req.seed is not None:
        gen_params["seed"] = req.seed

    # Prepare mode-specific params
    from qwen3_tts.core.engine import load_voice_prompt

    voice_prompt = None
    if mode == "clone":
        prompt_file = req.prompt_file
        if not prompt_file:
            raise HTTPException(status_code=400, detail="prompt_file required for clone mode")
        voice_prompt = load_voice_prompt(prompt_file)
        if voice_prompt is None:
            raise HTTPException(status_code=404, detail=f"Voice prompt not found: {prompt_file}")

    voice_description = req.voice_description
    language = req.language
    speaker = req.speaker
    instruct = req.instruct
    x_vector_only_mode = req.x_vector_only_mode

    # Create queue for streaming chunks
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    stop_event = threading.Event()
    inference_lock = state.inference_lock

    async def audio_stream_generator():
        """Async generator that yields audio chunks."""
        # Acquire inference_lock to serialize GPU access
        async with inference_lock:
            gen_id = str(uuid.uuid4())[:8]
            state.generation_state.update({
                "active": True,
                "start_time": time.time(),
                "text_length": len(text),
                "mode": mode,
                "generation_id": gen_id,
                "cancelled": False,
            })

            def inference_thread():
                """Run inference in a thread and push chunks to queue."""
                try:
                    from qwen3_tts.core.engine import run_inference_streaming

                    chunk_idx = 0
                    for wav_chunk, sr in run_inference_streaming(
                        model=model,
                        text=text,
                        mode=mode,
                        gen_params=gen_params,
                        language=language,
                        voice_prompt=voice_prompt,
                        voice_description=voice_description,
                        speaker=speaker,
                        instruct=instruct,
                        x_vector_only_mode=x_vector_only_mode,
                    ):
                        if stop_event.is_set():
                            logger.info("Generation cancelled after %d chunks", chunk_idx)
                            break

                        chunk_idx += 1
                        state.generation_state["chunk_index"] = chunk_idx

                        # Length-prefixed format: [sample_rate:4][length:4][audio:length]
                        audio_bytes = wav_chunk.astype("<f4").tobytes()
                        header = struct.pack("<II", sr, len(audio_bytes))

                        # Use call_soon_threadsafe to safely put from thread to async queue
                        loop.call_soon_threadsafe(queue.put_nowait, header + audio_bytes)

                except Exception as e:
                    logger.error("Streaming inference failed: %s", e, exc_info=True)
                    loop.call_soon_threadsafe(queue.put_nowait, None)
                else:
                    # Signal completion
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            # Start inference thread
            thread = threading.Thread(target=inference_thread, daemon=True)
            thread.start()

            try:
                # Yield chunks as they arrive
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk
            finally:
                stop_event.set()
                # Reset generation state if still our generation
                if state.generation_state.get("generation_id") == gen_id:
                    state.generation_state.update({
                        "active": False,
                        "start_time": 0.0,
                        "text_length": 0,
                        "mode": "",
                        "chunk_index": 0,
                        "chunk_total": 0,
                        "generation_id": None,
                        "cancelled": False,
                    })

    return StreamingResponse(
        audio_stream_generator(),
        media_type="application/octet-stream",
        headers={"X-Content-Type": "audio/raw-float32"}
    )


@app.post("/shutdown")
async def shutdown(request: Request, _auth: None = Depends(verify_auth)):
    """Graceful shutdown."""
    state = request.app.state

    # Cancel shutdown timer
    if state.shutdown_timer is not None:
        state.shutdown_timer.cancel()

    # Clean up PID and token files
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
    except OSError:
        pass
    try:
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)
    except OSError:
        pass

    # Signal uvicorn to stop gracefully
    cleanup_resources(state)
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutting down"}


# ---------------------------------------------------------------------------
# Main (for running directly)
# ---------------------------------------------------------------------------

def run_server(host="127.0.0.1", port=5123, public=False):
    """Run the FastAPI server."""
    # Configure logging
    log_fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=1)
    file_handler.setFormatter(log_fmt)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(log_fmt)
    logging.getLogger("tts").setLevel(logging.DEBUG)
    logging.getLogger("tts").addHandler(file_handler)
    logging.getLogger("tts").addHandler(stderr_handler)

    # Load config for settings
    config = load_config()
    server_config = config.get("server", {})

    if public:
        host = "0.0.0.0"
        logger.warning("Binding to 0.0.0.0 — server is accessible from the network.")

    if IN_COLAB:
        host = "0.0.0.0"
        logger.info("Colab detected — binding to 0.0.0.0 for tunnel access.")

    # Handle shutdown signals
    def _signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        # Set shutdown event
        shutdown_event = getattr(app.state, "shutdown_event", None)
        if shutdown_event is not None:
            shutdown_event.set()
        cleanup_resources(app.state)
        os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Print startup info
    print(f"\nFastAPI TTS Server starting on http://{host}:{port}")
    print("Models loading in background — /health returns 200 when ready.")
    print("Use 'tts server stop' to shut down.\n")

    uvicorn.run(
        "qwen3_tts.server.app:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FastAPI TTS Server")
    parser.add_argument("--public", action="store_true",
                        help="Bind to 0.0.0.0 (accessible from network)")
    args = parser.parse_args()

    config = load_config()
    server_config = config.get("server", {})
    host = server_config.get("host", "127.0.0.1")
    port = server_config.get("port", 5123)

    run_server(host=host, port=port, public=args.public)
