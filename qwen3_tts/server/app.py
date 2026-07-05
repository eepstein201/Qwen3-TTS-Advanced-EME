#!/usr/bin/env python3
"""FastAPI TTS server with async support and real streaming.

- Real async/await
- Proper streaming via asyncio.Queue
- app.state for worker-safe state
- inference_lock for GPU serialization

Lifecycle (lifespan, cleanup) lives in app_lifespan.py.
Generation handlers (/generate, /generate-stream) live in app_generation.py.
"""

import asyncio
import hashlib
import json
import logging
import logging.handlers
import os
import secrets
import signal
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Rate limiting (R-13)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

_HAS_SLOWAPI = True

logger = logging.getLogger("tts")

from qwen3_tts.core.config import (  # noqa: E402
    IN_COLAB,
    LOG_FILE,
    TOKEN_FILE,
    ConfigLoader,
    DefaultConfigLoader,
    cleanup_pid_file,
    get_backend,
    get_mlx_quantization,
    get_model_size,
    get_torch_dtype_name,
    load_config,
    sanitize_log,
)

# Config loader — can be replaced in tests for config injection
_DEFAULT_CONFIG_LOADER = DefaultConfigLoader()
_app_config_provider: ConfigLoader | None = None


def set_app_config_provider(provider: ConfigLoader | None) -> None:
    """Set a custom config loader for testing or custom deployments."""
    global _app_config_provider
    _app_config_provider = provider


def _get_app_config() -> dict:
    """Load application config, using override provider if set."""
    return (_app_config_provider or _DEFAULT_CONFIG_LOADER).load()


# Import generation handlers
from qwen3_tts.server.app_generation import (  # noqa: E402
    handle_generate,
    handle_generate_stream,
)

# Import lifecycle/infrastructure from app_lifespan
from qwen3_tts.server.app_lifespan import (  # noqa: E402
    MAX_ERROR_MSG_LEN,  # noqa: F401 (re-exported)
    _background_load,  # noqa: F401 (re-exported for test backward compat)
    _check_memory_available,  # noqa: F401 (re-exported for test backward compat)
    _estimate_eta,
    _get_queue_size,  # noqa: F401 (re-exported for test backward compat)
    _sanitize_error,  # noqa: F401 (re-exported for test backward compat)
    auto_shutdown,  # noqa: F401 (re-exported for test backward compat)
    cleanup_pid,  # noqa: F401 (re-exported for test backward compat)
    cleanup_resources,
    lifespan,
    reset_activity_timer,
)

# Import model/stats and prompt handlers
from qwen3_tts.server.app_models import (  # noqa: E402
    handle_list_models,
    handle_load_asr,
    handle_load_model,
    handle_stats,
    handle_transcribe,
    handle_unload_asr,
    handle_unload_model,
    handle_update_model_config,
    handle_update_startup_config,
)
from qwen3_tts.server.app_prompts import (  # noqa: E402
    handle_create_voice_prompt,
    handle_delete_prompt,
    handle_list_prompts,
    handle_preview_prompt,
    handle_prompt_details,
    handle_rename_prompt,
)

# Import validation module (models and helpers)
from qwen3_tts.server.validation import (  # noqa: E402
    CreateVoicePromptRequest,
    DeletePromptRequest,
    # Response models
    ErrorResponse,  # noqa: F401 (imported by test code via app module)
    # Request models
    GenerateRequest,
    GenerateResponse,
    GenerateResult,  # noqa: F401 (imported by test code via app module)
    HealthResponse,
    LoadModelRequest,
    RenamePromptRequest,
    TranscribeRequest,
    UnloadModelRequest,
    UpdateModelConfigRequest,
    UpdateStartupConfigRequest,
    _error_response,  # noqa: F401 (re-exported for test backward compat)
    _gen_cache_key,  # noqa: F401 (re-exported for test backward compat)
    # Validation helpers
    _validate_generation_request,  # noqa: F401 (re-exported for test backward compat)
)
from qwen3_tts.server.websocket import websocket_tts_handler  # noqa: E402

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


def _get_rate_limit_key(request: Request) -> str:
    """Hybrid rate limit key: combine IP and token for strictest limits.

    This ensures both per-IP AND per-token limits are enforced simultaneously.
    """
    client_ip = _get_real_client_ip(request)
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    # Hash token to avoid leaking sensitive data in rate limit keys
    token_hash = (
        hashlib.sha256(token.encode()).hexdigest()[:16] if token else "anonymous"
    )
    return f"{client_ip}:{token_hash}"


def _get_ip_key(request: Request) -> str:
    """IP-only rate limit key (current behavior).

    Rate limits based on client IP address only.
    """
    return _get_real_client_ip(request)


def _get_token_key(request: Request) -> str:
    """Token-only rate limit key.

    Rate limits based on authentication token only.
    Useful for shared IP environments (NAT, corporate proxies).
    """
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        return "anonymous"
    return hashlib.sha256(token.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def verify_auth(request: Request) -> None:
    """Verify Bearer token for protected endpoints.

    Public endpoints (/health, /ready, /generation-status, /queue-status) achieve
    no-auth access by simply not declaring Depends(verify_auth); this function only
    ever runs for protected routes.
    """
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not secrets.compare_digest(token, request.app.state.auth_token):
        client_ip = _get_real_client_ip(request)
        # Determine failure reason for audit logging (R-26)
        failure_reason = "missing_token" if not token else "invalid_token"
        logger.warning(
            "Auth failure: %s from %s on %s %s",
            failure_reason,
            sanitize_log(client_ip),
            request.method,
            request.url.path,
        )
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
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next) -> dict:
    """Add security response headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# Rate limiting (R-13)
_rate_config = load_config().get("security", {}).get("rate_limits", {})
_generate_limit = _rate_config.get("generate", "10/minute")
_model_limit = _rate_config.get("model_ops", "5/minute")
_transcribe_limit = _rate_config.get("transcribe", "10/minute")
_prompt_ops_limit = _rate_config.get("prompt_ops", "10/minute")
_config_ops_limit = _rate_config.get("config_ops", "2/minute")

# Create separate limiters for different strategies
limiter_hybrid = Limiter(key_func=_get_rate_limit_key)
limiter_ip = Limiter(key_func=_get_ip_key)
limiter_token = Limiter(key_func=_get_token_key)

app.state.limiter = limiter_hybrid  # slowapi's default handler expects this name
app.state.limiter_hybrid = limiter_hybrid
app.state.limiter_ip = limiter_ip
app.state.limiter_token = limiter_token

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _rate_limit(limit_string: str, strategy: str = "hybrid") -> callable:
    """Return a slowapi rate limit decorator with specified strategy.

    Args:
        limit_string: Rate limit string (e.g., "10/minute")
        strategy: "hybrid" (both IP+token), "ip" (IP only), "token" (token only)

    Returns:
        Decorator function.
    """
    limiter_map = {
        "hybrid": limiter_hybrid,
        "ip": limiter_ip,
        "token": limiter_token,
    }
    selected_limiter = limiter_map.get(strategy)
    return selected_limiter.limit(limit_string)


# ---------------------------------------------------------------------------
# Public endpoints (no auth)
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Models still loading"}},
)
async def health(request: Request) -> dict:
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
async def ready(request: Request) -> dict:
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
async def generation_status(request: Request) -> dict:
    """Get current generation status (public — sensitive fields stripped)."""
    state = request.app.state
    gen_state = state.generation_state
    result = {
        "active": gen_state["active"],
        "batch_index": gen_state["batch_index"],
        "batch_total": gen_state["batch_total"],
        "chunk_index": gen_state["chunk_index"],
        "chunk_total": gen_state["chunk_total"],
        "cancelled": gen_state["cancelled"],
    }
    if gen_state["active"]:
        result["elapsed_sec"] = round(time.time() - gen_state["start_time"], 1)
        result["eta_sec"] = _estimate_eta(
            state, gen_state["text_length"], result["elapsed_sec"]
        )
    return result


@app.get("/queue-status")
async def queue_status(request: Request) -> dict:
    """Get generation queue status (public — sensitive fields stripped)."""
    state = request.app.state
    async with state.pending_lock:
        pending = list(state.pending_requests)
    gen_state = state.generation_state
    return {
        "queue_length": len(pending),
        "active": gen_state.get("active", False),
    }


# ---------------------------------------------------------------------------
# Protected endpoints (require auth)
# ---------------------------------------------------------------------------


@app.get("/stats")
async def stats(request: Request, _auth: None = Depends(verify_auth)) -> dict:
    """Get server statistics."""
    state = request.app.state
    reset_activity_timer(state)
    return handle_stats(state, state.server_config)


@app.get("/models")
async def list_models(request: Request, _auth: None = Depends(verify_auth)) -> dict:
    """List model status."""
    state = request.app.state
    reset_activity_timer(state)
    return handle_list_models(state, state.server_config)


@app.post("/load-model")
@_rate_limit(_model_limit)
async def load_model_endpoint(
    request: Request, req: LoadModelRequest, _auth: None = Depends(verify_auth)
):
    """Load a model on demand."""
    state = request.app.state
    reset_activity_timer(state)
    return await asyncio.to_thread(handle_load_model, state, req)


@app.post("/unload-model")
@_rate_limit(_model_limit, strategy="hybrid")
async def unload_model(
    request: Request, req: UnloadModelRequest, _auth: None = Depends(verify_auth)
):
    """Unload a model to free memory."""
    state = request.app.state
    reset_activity_timer(state)
    return await asyncio.to_thread(handle_unload_model, state, req)


@app.post("/update-model-config")
@_rate_limit(_model_limit)
async def update_model_config(
    request: Request, req: UpdateModelConfigRequest, _auth: None = Depends(verify_auth)
):
    """Update model size and/or quantization settings."""
    state = request.app.state
    reset_activity_timer(state)
    return await handle_update_model_config(state, req, _get_app_config)


@app.post("/update-startup-config")
@_rate_limit("2/minute", strategy="hybrid")
async def update_startup_config(
    request: Request,
    req: UpdateStartupConfigRequest,
    _auth: None = Depends(verify_auth),
):
    """Update which models load at startup in config.json."""
    state = request.app.state
    reset_activity_timer(state)
    return await asyncio.to_thread(
        handle_update_startup_config, state, req, _get_app_config
    )


# ---------------------------------------------------------------------------
# ASR endpoints
# ---------------------------------------------------------------------------


@app.post("/load-asr")
@_rate_limit(_model_limit)
async def load_asr(request: Request, _auth: None = Depends(verify_auth)):
    """Load the ASR model for transcription."""
    state = request.app.state
    reset_activity_timer(state)
    return await asyncio.to_thread(handle_load_asr, state)


@app.post("/unload-asr")
@_rate_limit(_model_limit, strategy="hybrid")
async def unload_asr(request: Request, _auth: None = Depends(verify_auth)):
    """Unload the ASR model to free memory."""
    state = request.app.state
    reset_activity_timer(state)
    return await asyncio.to_thread(handle_unload_asr, state)


@app.post("/transcribe")
@_rate_limit(_generate_limit)
async def transcribe(
    request: Request, req: TranscribeRequest, _auth: None = Depends(verify_auth)
):
    """Transcribe audio to text using ASR."""
    state = request.app.state
    reset_activity_timer(state)
    return await asyncio.to_thread(handle_transcribe, state, req)


@app.get("/prompts")
async def list_prompts(request: Request, _auth: None = Depends(verify_auth)):
    """List voice prompts with optional pagination (R-24)."""
    state = request.app.state
    reset_activity_timer(state)
    return await asyncio.to_thread(
        handle_list_prompts, state, get_backend(), request.query_params
    )


@app.post("/delete-prompt")
@_rate_limit("10/minute", strategy="hybrid")
async def delete_prompt(
    request: Request, req: DeletePromptRequest, _auth: None = Depends(verify_auth)
):
    """Delete a voice prompt and all its format files."""
    state = request.app.state
    reset_activity_timer(state)
    return await asyncio.to_thread(handle_delete_prompt, state, req, _get_app_config)


@app.post("/rename-prompt")
@_rate_limit("10/minute", strategy="hybrid")
async def rename_prompt(
    request: Request, req: RenamePromptRequest, _auth: None = Depends(verify_auth)
):
    """Rename a voice prompt (all format files) with rollback on partial failure."""
    state = request.app.state
    reset_activity_timer(state)
    return await asyncio.to_thread(handle_rename_prompt, state, req, _get_app_config)


@app.get("/preview-prompt")
async def preview_prompt(request: Request, _auth: None = Depends(verify_auth)):
    """Return the .wav file for a voice prompt as audio/wav."""
    state = request.app.state
    reset_activity_timer(state)
    return await asyncio.to_thread(
        handle_preview_prompt, request.query_params.get("name", "")
    )


@app.get("/prompt-details")
async def prompt_details(request: Request, _auth: None = Depends(verify_auth)):
    """Return metadata for voice prompts."""
    state = request.app.state
    reset_activity_timer(state)
    return await asyncio.to_thread(
        handle_prompt_details, request.query_params.get("name")
    )


@app.post("/create-voice-prompt")
@_rate_limit(_model_limit)
async def create_voice_prompt_endpoint(
    request: Request, req: CreateVoicePromptRequest, _auth: None = Depends(verify_auth)
):
    """Create a voice clone prompt from uploaded audio."""
    state = request.app.state
    reset_activity_timer(state)
    return await asyncio.to_thread(handle_create_voice_prompt, state, req)


@app.post("/cancel-generation")
async def cancel_generation(
    request: Request, _auth: None = Depends(verify_auth)
) -> dict:
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
        "generation_id": state.generation_state.get("generation_id"),
    }


# ---------------------------------------------------------------------------
# Generation endpoints — thin wrappers delegating to app_generation.py
# ---------------------------------------------------------------------------


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        200: {
            "content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}
        }
    },
)
@_rate_limit(_generate_limit)
async def generate(
    request: Request, req: GenerateRequest, _auth: None = Depends(verify_auth)
):
    """Generate audio from text."""
    state = request.app.state
    reset_activity_timer(state)
    security = state.server_config.get("security", {})
    return await handle_generate(request, state, req, security, _app_config_provider)


@app.post("/generate-stream")
@_rate_limit(_generate_limit)
async def generate_stream(
    request: Request, req: GenerateRequest, _auth: None = Depends(verify_auth)
):
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
    return await handle_generate_stream(
        request, state, req, security, _app_config_provider
    )


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
async def shutdown(request: Request, _auth: None = Depends(verify_auth)) -> dict:
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
            with open(TOKEN_FILE) as f:
                on_disk = f.read().strip()
            if on_disk == state.auth_token:
                os.remove(TOKEN_FILE)
        except (FileNotFoundError, OSError):
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


def run_server(host: str = "127.0.0.1", port: int = 5123, public: bool = False) -> None:
    """Run the FastAPI server."""
    # Configure logging
    log_fmt = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=1
    )
    file_handler.setFormatter(log_fmt)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(log_fmt)
    _log_level_name = os.environ.get("TTS_LOG_LEVEL", "INFO").upper()
    _log_level = getattr(logging, _log_level_name, logging.INFO)
    logging.getLogger("tts").setLevel(_log_level)
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
    parser.add_argument(
        "--public",
        action="store_true",
        help="Bind to 0.0.0.0 (accessible from network)",
    )
    args = parser.parse_args()

    config = load_config()
    server_config = config.get("server", {})
    host = server_config.get("host", "127.0.0.1")
    port = server_config.get("port", 5123)

    run_server(host=host, port=port, public=args.public)
