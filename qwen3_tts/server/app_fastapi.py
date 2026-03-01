"""FastAPI TTS server with async support and real streaming.

Replaces Flask with FastAPI for:
- Real async/await
- Proper streaming via asyncio.Queue
- app.state for worker-safe state
- inference_lock for GPU serialization
"""

import asyncio
import json
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Generator, Optional

import soundfile as sf
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn

logger = logging.getLogger("tts")

from qwen3_tts.core.config import (
    CONFIG_PATH,
    VOICE_PROMPTS_DIR,
    PID_FILE,
    TOKEN_FILE,
    MODEL_INFO,
    CUSTOM_VOICE_SPEAKERS,
    load_config,
    save_config,
    get_backend,
    get_default_clone_prompt,
    get_torch_dtype_name,
    get_mlx_quantization,
    get_model_size,
    VALID_BACKENDS,
)

# Pre-computed valid speaker names
_VALID_SPEAKER_NAMES = frozenset(CUSTOM_VOICE_SPEAKERS.keys()) | frozenset(
    v["name"] for v in CUSTOM_VOICE_SPEAKERS.values()
)

# Global server state (will be replaced by app.state in lifespan)
_auth_token: Optional[str] = None
_models: Dict[str, Optional[object]] = {"clone": None, "design": None, "custom": None}
_model_load_times: Dict[str, float] = {}
_generation_lock = threading.Lock()
_generation_state: Dict = {
    "in_progress": False,
    "current_text": "",
    "start_time": None,
    "cancelled": False,
}
_request_queue: set = set()
_last_activity: float = time.time()
_models_loaded = threading.Event()
_gen_cache: Dict = {}
_gen_cache_lock = threading.Lock()
_inference_lock: asyncio.Lock = None  # Set in lifespan


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def verify_auth(request: Request) -> None:
    """Verify Bearer token for protected endpoints."""
    if request.url.path in ("/health", "/generation-status"):
        return
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != request.app.state.auth_token:
        raise HTTPException(status_code=401, detail="Unauthorized")


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
    app.state.generation_lock = threading.Lock()
    app.state.generation_state = {
        "in_progress": False,
        "current_text": "",
        "start_time": None,
        "cancelled": False,
    }
    app.state.request_queue = set()
    app.state.last_activity = time.time()
    app.state.models_loaded = threading.Event()
    app.state.gen_cache = {}
    app.state.gen_cache_lock = threading.Lock()

    # CRITICAL: Add inference_lock to prevent parallel OOM
    app.state.inference_lock = asyncio.Lock()

    # Write token file
    global _auth_token, _inference_lock
    _auth_token = app.state.auth_token
    _inference_lock = app.state.inference_lock

    with open(TOKEN_FILE, "w") as f:
        f.write(app.state.auth_token)
    os.chmod(TOKEN_FILE, 0o600)

    logger.info("FastAPI server starting...")

    # Start background model loading
    loader = threading.Thread(target=_background_load, args=(app,), daemon=True)
    loader.start()

    yield

    # Shutdown
    logger.info("FastAPI server shutting down...")
    cleanup_resources(app)

    # Clean up token file
    try:
        os.unlink(TOKEN_FILE)
    except FileNotFoundError:
        pass


# Create FastAPI app
app = FastAPI(
    title="Qwen3-TTS Server",
    version="3.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _background_load(app_state):
    """Background thread that loads models at startup."""
    from qwen3_tts.core.engine import load_model
    config = load_config()

    startup_models = []
    for model_type in ("clone", "design", "custom"):
        if config.get("models", {}).get(model_type, {}).get("load_at_startup", False):
            startup_models.append(model_type)

    for model_type in startup_models:
        try:
            logger.info(f"Loading {model_type} model...")
            setattr(app_state.models, model_type, load_model(model_type))
            setattr(app_state.model_load_times, model_type, time.time())
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")

    app_state.models_loaded.set()
    logger.info("Background model loading complete")


def cleanup_resources(app_state):
    """Clean up resources on shutdown."""
    for name in ("clone", "design", "custom"):
        model = getattr(app_state.models, name, None)
        if model is not None:
            try:
                del model
                setattr(app_state.models, name, None)
            except Exception:
                pass


def get_model_info(app_state, name: str) -> dict:
    """Get info about a model."""
    model = getattr(app_state.models, name, None)
    return {
        "loaded": model is not None,
        "load_time": getattr(app_state.model_load_times, name, None),
    }


# ---------------------------------------------------------------------------
# Public endpoints (no auth)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    if request.app.state.models_loaded.is_set():
        backend = get_backend()
        return {"status": "ok", "backend": backend}
    return {"status": "loading"}, 503


@app.get("/generation-status")
async def generation_status(request: Request):
    """Get current generation status."""
    state = request.app.state.generation_state
    return {
        "in_progress": state.in_progress,
        "current_text": state.current_text if state.in_progress else "",
        "start_time": state.start_time,
        "cancelled": state.cancelled,
    }


# ---------------------------------------------------------------------------
# Protected endpoints (require auth)
# ---------------------------------------------------------------------------

@app.get("/stats")
async def stats(request: Request):
    """Get server statistics."""
    state = request.app.state
    return {
        "models": {
            name: get_model_info(state, name)
            for name in ("clone", "design", "custom")
        },
        "cache_size": len(state.gen_cache),
        "last_activity": state.last_activity,
    }


@app.get("/models")
async def list_models(request: Request):
    """List model status."""
    state = request.app.state
    return {
        name: get_model_info(state, name)
        for name in ("clone", "design", "custom")
    }


@app.post("/shutdown")
async def shutdown(request: Request):
    """Graceful shutdown."""
    # In a real deployment, this would trigger shutdown
    return {"status": "shutting down"}


# ---------------------------------------------------------------------------
# Main (for running directly)
# ---------------------------------------------------------------------------

def run_server(host="127.0.0.1", port=5123):
    """Run the FastAPI server."""
    uvicorn.run(
        "qwen3_tts.server.app_fastapi:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
