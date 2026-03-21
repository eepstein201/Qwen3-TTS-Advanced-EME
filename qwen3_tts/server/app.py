#!/usr/bin/env python3
"""FastAPI TTS server with async support and real streaming.

- Real async/await
- Proper streaming via asyncio.Queue
- app.state for worker-safe state
- inference_lock for GPU serialization

Lifecycle (lifespan, cleanup) lives in app_lifespan.py.
Generation handlers (/generate, /generate-stream) live in app_generation.py.
"""

import json
import logging
import logging.handlers
import os
import secrets
import signal
import sys
import time
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Depends, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

# Optional rate limiting (R-13)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    _HAS_SLOWAPI = True
except ImportError:
    _HAS_SLOWAPI = False

logger = logging.getLogger("tts")

from qwen3_tts.core.config import (  # noqa: E402
    VOICE_PROMPTS_DIR,
    TOKEN_FILE,
    LOG_FILE,
    MODEL_INFO,
    MLX_MODEL_INFO,
    IN_COLAB,
    ConfigLoader,
    DefaultConfigLoader,
    load_config,
    save_config,
    get_backend,
    get_default_clone_prompt,
    get_torch_dtype_name,
    get_mlx_quantization,
    get_model_size,
    get_mlx_model_name,
    cleanup_pid_file,
)

# Config loader — can be replaced in tests for config injection
_DEFAULT_CONFIG_LOADER = DefaultConfigLoader()
_app_config_provider: Optional[ConfigLoader] = None


def set_app_config_provider(provider: Optional[ConfigLoader]) -> None:
    """Set a custom config loader for testing or custom deployments."""
    global _app_config_provider
    _app_config_provider = provider


def _get_app_config() -> dict:
    """Load application config, using override provider if set."""
    return (_app_config_provider or _DEFAULT_CONFIG_LOADER).load()


from qwen3_tts.server.websocket import websocket_tts_handler  # noqa: E402

# Import validation module (models and helpers)
from qwen3_tts.server.validation import (  # noqa: E402
    # Request models
    GenerateRequest,
    LoadModelRequest,
    UnloadModelRequest,
    UpdateModelConfigRequest,
    UpdateStartupConfigRequest,
    DeletePromptRequest,
    RenamePromptRequest,
    # Response models
    ErrorResponse,  # noqa: F401 (imported by test code via app module)
    GenerateResponse,
    GenerateResult,  # noqa: F401 (imported by test code via app module)
    HealthResponse,
    # Validation helpers
    _validate_generation_request,  # noqa: F401 (re-exported for test backward compat)
    _validate_prompt_name,
    _strip_extension,
    _gen_cache_key,  # noqa: F401 (re-exported for test backward compat)
    _error_response,  # noqa: F401 (re-exported for test backward compat)
)

# Import lifecycle/infrastructure from app_lifespan
from qwen3_tts.server.app_lifespan import (  # noqa: E402
    MAX_ERROR_MSG_LEN,  # noqa: F401 (re-exported)
    _sanitize_error,  # noqa: F401 (re-exported for test backward compat)
    _estimate_eta,
    _check_memory_available,  # noqa: F401 (re-exported for test backward compat)
    _get_queue_size,
    reset_activity_timer,
    auto_shutdown,  # noqa: F401 (re-exported for test backward compat)
    lifespan,
    _background_load,  # noqa: F401 (re-exported for test backward compat)
    cleanup_resources,
    cleanup_pid,  # noqa: F401 (re-exported for test backward compat)
)

# Import generation handlers
from qwen3_tts.server.app_generation import (  # noqa: E402
    handle_generate,
    handle_generate_stream,
)


# ---------------------------------------------------------------------------
# Real client IP resolution (R-13)
# ---------------------------------------------------------------------------

def _get_real_client_ip(request: Request) -> str:
    """Extract real client IP, checking proxy headers first.

    Critical for Colab/Gradio share links where all traffic
    comes through a reverse proxy or tunnel.

    X-Forwarded-For is only trusted when the server is NOT bound exclusively
    to localhost (i.e. listening on a public interface for Colab/tunnel use).
    Trusting XFF on a loopback-only server allows any client to spoof their IP
    and bypass rate limiting.
    """
    direct_host = request.client.host if request.client else "127.0.0.1"
    # Trust XFF only when not on pure loopback — prevents rate-limit bypass
    is_loopback = direct_host in ("127.0.0.1", "::1", "localhost")
    if not is_loopback:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return direct_host


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def verify_auth(request: Request) -> None:
    """Verify Bearer token for protected endpoints."""
    if request.url.path in ("/health", "/generation-status"):
        return
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not secrets.compare_digest(token, request.app.state.auth_token):
        client_ip = _get_real_client_ip(request)
        logger.warning("Auth failure from %s on %s %s", client_ip, request.method, request.url.path)
        raise HTTPException(status_code=401, detail="Unauthorized")


# Create FastAPI app
app = FastAPI(
    title="Qwen3-TTS Server",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS: allow Gradio UI on any local port (7860, 7861, etc.)
# In Colab, also allow *.gradio.live origins (Gradio share tunnel)
_cors_regex = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
if IN_COLAB:
    _cors_regex = r"(^https?://(localhost|127\.0\.0\.1)(:\d+)?$)|(^https://[a-z0-9-]+\.gradio\.live$)"
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_cors_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting (R-13) — optional, requires slowapi
if _HAS_SLOWAPI:
    _rate_config = load_config().get("security", {}).get("rate_limits", {})
    _generate_limit = _rate_config.get("generate", "10/minute")
    _model_limit = _rate_config.get("model_ops", "5/minute")
    limiter = Limiter(key_func=_get_real_client_ip)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
else:
    limiter = None
    _generate_limit = "10/minute"
    _model_limit = "5/minute"


def _rate_limit(limit_string):
    """Return a slowapi rate limit decorator, or a no-op if slowapi is not installed."""
    if limiter is not None:
        return limiter.limit(limit_string)
    def _noop(func):
        return func
    return _noop


# ---------------------------------------------------------------------------
# Public endpoints (no auth)
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse,
         responses={503: {"model": HealthResponse, "description": "Models still loading"}})
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


@app.get("/queue-status")
async def queue_status(request: Request):
    """Get generation queue status (no auth required)."""
    state = request.app.state
    async with state.pending_lock:
        pending = list(state.pending_requests)
    gen_state = state.generation_state
    return {
        "queue_length": len(pending),
        "active": gen_state.get("active", False),
        "active_mode": gen_state.get("mode", "") if gen_state.get("active") else None,
        "positions": [
            {"id": p["id"], "mode": p["mode"]}
            for p in pending
        ],
    }


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
        "generation_queue_size": _get_queue_size(state),
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
@_rate_limit(_model_limit)
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
        _error_response(500, "import_error", str(e), "config")
    except (RuntimeError, OSError, ValueError) as e:
        logger.error("Failed to load model %s: %s", model_type, e, exc_info=True)
        state.model_load_errors[model_type] = str(e)
        _error_response(500, "load_failed", str(e), "restart")
    except Exception as e:
        logger.error("Unexpected error loading model %s: %s", model_type, e, exc_info=True)
        state.model_load_errors[model_type] = str(e)
        _error_response(500, "unknown_error", str(e), "bug")

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
@_rate_limit(_model_limit)
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

    config = _get_app_config()
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
    config = _get_app_config()
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
    """List voice prompts with optional pagination (R-24).

    Query params:
        offset: Start index (default 0).
        limit: Max results (default 0 = return all, for backward compat).
    """
    state = request.app.state
    reset_activity_timer(state)

    backend = get_backend()
    try:
        all_files = set(os.listdir(VOICE_PROMPTS_DIR))
    except OSError:
        return {"prompts": [], "total": 0}

    if backend == "mlx":
        # MLX uses .wav+.txt pairs
        wav_files = {f[:-4] for f in all_files if f.endswith('.wav')}
        txt_files = {f[:-4] for f in all_files if f.endswith('.txt')}
        names = sorted(wav_files & txt_files)
        prompts = [f"{n}.wav" for n in names]
    else:
        # Torch uses .pt files only (not .wav files)
        pt_names = {f[:-3] for f in all_files if f.endswith('.pt')}
        prompts = sorted(f"{n}.pt" for n in pt_names)

    total = len(prompts)

    # Pagination (R-24) — default limit=0 means return all (backward compat)
    try:
        offset = max(0, int(request.query_params.get("offset", 0)))
    except (ValueError, TypeError):
        offset = 0
    try:
        limit = max(0, int(request.query_params.get("limit", 0)))
    except (ValueError, TypeError):
        limit = 0

    if offset > 0 or limit > 0:
        if limit > 0:
            prompts = prompts[offset:offset + limit]
        else:
            prompts = prompts[offset:]

    return {"prompts": prompts, "total": total, "offset": offset, "limit": limit}


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
        config = _get_app_config()
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
        config = _get_app_config()
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

    # Symlink resolution — prevent path traversal via symlinks (R-20)
    real_path = os.path.realpath(wav_path)
    real_prompts_dir = os.path.realpath(VOICE_PROMPTS_DIR)
    if not real_path.startswith(real_prompts_dir + os.sep):
        raise HTTPException(status_code=403, detail="Access denied: path outside voice prompts directory")

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


# ---------------------------------------------------------------------------
# Generation endpoints — thin wrappers delegating to app_generation.py
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse,
          responses={200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}}})
@_rate_limit(_generate_limit)
async def generate(request: Request, req: GenerateRequest, _auth: None = Depends(verify_auth)):
    """Generate audio from text."""
    state = request.app.state
    reset_activity_timer(state)
    security = state.server_config.get("security", {})
    return await handle_generate(request, state, req, security, _app_config_provider)


@app.post("/generate-stream")
@_rate_limit(_generate_limit)
async def generate_stream(request: Request, req: GenerateRequest, _auth: None = Depends(verify_auth)):
    """Stream audio generation — returns chunked audio as it's produced.

    Wire format per chunk (little-endian):
        [sample_rate: 4 bytes uint32][audio_len: 4 bytes uint32][audio: audio_len bytes float32]

    Python consumer::

        import struct, httpx
        with httpx.stream("POST", url, json=payload, headers=headers) as r:
            buf = b""
            for raw in r.iter_bytes():
                buf += raw
                while len(buf) >= 8:
                    sr, n = struct.unpack_from("<II", buf, 0)
                    if len(buf) < 8 + n:
                        break
                    samples = struct.unpack_from(f'<{n//4}f', buf, 8)
                    buf = buf[8 + n:]
                    # process samples at sample rate sr

    JavaScript consumer (see wavesurfer_js.py startStreaming() for full implementation)::

        // Each chunk: Float32Array samples at the given sample rate
        // Parse header: DataView.getUint32(0, true) = sr, getUint32(4, true) = byteLen
    """
    state = request.app.state
    reset_activity_timer(state)
    security = state.server_config.get("security", {})
    return await handle_generate_stream(request, state, req, security, _app_config_provider)


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Bidirectional WebSocket endpoint for real-time TTS streaming."""
    state = websocket.app.state

    def _verify_token(token: str) -> bool:
        return secrets.compare_digest(token, state.auth_token)

    await websocket_tts_handler(websocket, state, _verify_token)


@app.post("/shutdown")
async def shutdown(request: Request, _auth: None = Depends(verify_auth)):
    """Graceful shutdown via BackgroundTask + SIGTERM for reliable termination."""
    from starlette.background import BackgroundTask

    state = request.app.state

    # Cancel shutdown timer
    if state.shutdown_timer is not None:
        state.shutdown_timer.cancel()

    def _shutdown_background():
        """Run cleanup then SIGTERM self — matches _signal_handler pattern."""
        cleanup_pid_file()
        try:
            if os.path.exists(TOKEN_FILE):
                os.remove(TOKEN_FILE)
        except OSError:
            pass
        cleanup_resources(state)
        os.kill(os.getpid(), signal.SIGTERM)

    from fastapi import Response
    return Response(
        content=json.dumps({"status": "shutting_down"}),
        media_type="application/json",
        background=BackgroundTask(_shutdown_background),
    )


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

    if public:
        host = "0.0.0.0"
        logger.warning("Binding to 0.0.0.0 — server is accessible from the network.")

    if IN_COLAB:
        host = "0.0.0.0"
        logger.info("Colab detected — binding to 0.0.0.0 for tunnel access.")

    # Handle shutdown signals
    def _signal_handler(signum, frame):
        """Handle shutdown signals gracefully."""
        # Reset handlers to default to prevent re-entry
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
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
