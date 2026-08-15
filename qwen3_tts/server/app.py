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
import re
import secrets
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketException,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Rate limiting (R-13)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

_HAS_SLOWAPI = True

logger = logging.getLogger("tts")

from qwen3_tts.core.config import (  # noqa: E402
    IN_COLAB,
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
    _estimate_eta,  # noqa: F401 (re-exported for test backward compat)
    _get_queue_size,  # noqa: F401 (re-exported for test backward compat)
    _sanitize_error,  # noqa: F401 (re-exported for test backward compat)
    auto_shutdown,  # noqa: F401 (re-exported for test backward compat)
    cleanup_pid,  # noqa: F401 (re-exported for test backward compat)
    cleanup_resources,
    detect_degraded_generation,
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
    MAX_AUDIO_BASE64_BYTES,
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
    # Response contracts (GEN-2)
    GenerationStatusResponse,
    ModelsResponse,
    QueueStatusResponse,
    ReadyResponse,
    StatsResponse,
    TranscribeResponse,
    _error_response,  # noqa: F401 (re-exported for test backward compat)
    _gen_cache_key,  # noqa: F401 (re-exported for test backward compat)
    # Validation helpers
    _validate_generation_request,  # noqa: F401 (re-exported for test backward compat)
)
from qwen3_tts.server.websocket import websocket_tts_handler  # noqa: E402

# ---------------------------------------------------------------------------
# Real client IP resolution (R-13)
# ---------------------------------------------------------------------------


def _load_trusted_proxies() -> set[str]:
    """Trusted reverse-proxy IPs whose X-Forwarded-For header we honor.

    Defaults to loopback only (covers Colab/Gradio tunnels, which forward to
    127.0.0.1). Operators fronting the server with a real reverse proxy can add
    its IP(s) via TTS_TRUSTED_PROXIES (comma-separated).
    """
    proxies = {"127.0.0.1", "::1", "localhost"}
    env = os.environ.get("TTS_TRUSTED_PROXIES", "")
    proxies |= {ip.strip() for ip in env.split(",") if ip.strip()}
    return proxies


TRUSTED_PROXIES = _load_trusted_proxies()


def _get_real_client_ip(request: Request) -> str:
    """Extract the real client IP, honoring X-Forwarded-For only from a proxy.

    Critical for Colab/Gradio share links where traffic arrives through a
    reverse proxy or tunnel. X-Forwarded-For is trusted ONLY when the direct
    TCP peer is in TRUSTED_PROXIES; otherwise a client connecting directly on a
    public/Colab bind could spoof the header to rotate its rate-limit key and
    bypass per-IP limits. Untrusted peers always fall back to their real IP.
    """
    direct_host = request.client.host if request.client else "127.0.0.1"
    if direct_host in TRUSTED_PROXIES:
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


# Same origin allowlist as CORS, compiled for the WebSocket Origin check
# (CSWSH defense). CORS protects HTTP only; the WS handshake is a separate
# request that a browser page can attempt to steal via cookies.
_origin_re = re.compile(_cors_regex)


def validate_ws_origin(websocket: WebSocket) -> None:
    """Reject cross-origin browser WebSocket handshakes (CSWSH defense).

    Browsers always send ``Origin``; CLI/script clients (the ``tts`` client,
    tests) do not. A browser handshake is accepted only if Origin matches the
    same allowlist as CORS. An ABSENT Origin is allowed because the real auth is
    the per-message bearer token (not Origin), and a non-browser client cannot
    carry the victim's browser session that makes Cross-Site WebSocket
    Hijacking exploitable. Raised before ``accept()`` -> FastAPI rejects the
    handshake (Starlette sends HTTP 403).
    """
    origin = websocket.headers.get("origin")
    if origin is not None and not _origin_re.match(origin):
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)


# Reject request bodies larger than this before they are read/parsed (R-30).
# Sized ~2x the largest legitimate payload: MAX_AUDIO_BASE64_BYTES (50MB base64)
# plus JSON envelope overhead. Guards against memory-exhaustion via oversized
# uploads that Pydantic would only catch after buffering the whole body.
MAX_REQUEST_BODY_BYTES = 2 * MAX_AUDIO_BASE64_BYTES  # ~100MB


async def _send_413(scope, receive, send) -> None:
    """Send a minimal 413 JSON response (matches the prior middleware body)."""
    from fastapi.responses import JSONResponse

    await JSONResponse(
        status_code=413, content={"detail": "Request body too large"}
    )(scope, receive, send)


class RequestBodySizeLimitMiddleware:
    """ASGI middleware: reject oversized request bodies WITHOUT buffering.

    Replaces the prior ``@app.middleware("http")`` that did
    ``request._body = b"".join(...)`` — buffering up to MAX_REQUEST_BODY_BYTES
    of an UNAUTHENTICATED request before the 401 (a pre-auth memory-DoS: an
    attacker can send many large chunked bodies with no token). This counts
    bytes off the ``receive`` callable and aborts as soon as the cap is
    exceeded; chunks pass through unbuffered so the app's own ``request.json()``
    read works normally.

    The cap is read from the module global at request time so tests that patch
    ``MAX_REQUEST_BODY_BYTES`` take effect without re-adding the middleware.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        max_body_size = MAX_REQUEST_BODY_BYTES

        # Fast-path: Content-Length present and over cap -> reject before reading
        # any bytes. This runs BEFORE the app, so we send the 413 directly (no
        # inner ExceptionMiddleware is above us to convert a raised exception).
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > max_body_size:
                        await _send_413(scope, receive, send)
                        return
                except ValueError:
                    pass
                break

        received = 0

        async def receive_wrapper():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b"") or b"")
                if received > max_body_size:
                    # Raised inside the app's body read, so Starlette's
                    # ExceptionMiddleware (inner to us) converts it to the 413
                    # response. A plain Exception here is swallowed into a
                    # 400/500; HTTPException (a StarletteHTTPException subclass)
                    # is what it recognises.
                    raise HTTPException(
                        status_code=413, detail="Request body too large"
                    )
            return message

        await self.app(scope, receive_wrapper, send)


# Registered BEFORE security_headers so it sits INSIDE that BaseHTTPMiddleware
# in the final stack (an outer BaseHTTPMiddleware re-wraps `receive` and would
# bypass this counter), and outside CORS/ExceptionMiddleware so the
# StarletteHTTPException(413) raised on overflow is converted to the 413
# response by the inner ExceptionMiddleware. Net order (outer->inner):
# SlowAPI -> security_headers -> RequestBody -> CORS -> ExceptionMiddleware.
app.add_middleware(RequestBodySizeLimitMiddleware)


@app.middleware("http")
async def security_headers(request: Request, call_next) -> dict:
    """Add security response headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# Rate limiting (R-13)
#
# Limits are read once at import from config.json (security.rate_limits) and
# can be overridden per-limit via env vars (TTS_RATE_LIMIT_GENERATE, ...).
# Set TTS_DISABLE_RATE_LIMITING=1 to bypass rate limiting entirely — intended
# for local E2E/CI servers where the /generate cap would otherwise starve test
# suites that fire many requests. Production deployments leave it unset.
_rate_config = load_config().get("security", {}).get("rate_limits", {})

_RATE_LIMITING_DISABLED = os.environ.get("TTS_DISABLE_RATE_LIMITING", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def _rate_limit_from_env(env_key: str, config_key: str, default: str) -> str:
    """Resolve a limit string: env override > config value > literal default.

    Format is the slowapi ``<count>/<unit>`` string, e.g. ``"10/minute"``.
    """
    return os.environ.get(env_key) or _rate_config.get(config_key, default)


_generate_limit = _rate_limit_from_env("TTS_RATE_LIMIT_GENERATE", "generate", "10/minute")
_model_limit = _rate_limit_from_env("TTS_RATE_LIMIT_MODEL_OPS", "model_ops", "5/minute")
_transcribe_limit = _rate_limit_from_env("TTS_RATE_LIMIT_TRANSCRIBE", "transcribe", "10/minute")
_prompt_ops_limit = _rate_limit_from_env("TTS_RATE_LIMIT_PROMPT_OPS", "prompt_ops", "10/minute")
_config_ops_limit = _rate_limit_from_env("TTS_RATE_LIMIT_CONFIG_OPS", "config_ops", "2/minute")

# Global pre-auth FLOOD backstop, deliberately decoupled from the per-route
# generate limit. The Gradio UI polls /health + /models every 5s (~24/min), so
# a global ceiling tied to the 10/min generate limit would throttle normal
# authenticated use within ~25s and 429 /health -> is_server_running() reads
# "down" -> the UI shows "Disconnected / Server not running". This backstop
# only needs to stop floods (hundreds-thousands/min); the tight per-endpoint
# caps below (@limiter.limit, post-auth) still do the real rate limiting.
_global_limit = _rate_limit_from_env("TTS_RATE_LIMIT_GLOBAL", "global", "120/minute")

# Global pre-auth limiter: an IP-keyed default ceiling enforced by
# SlowAPIMiddleware at the ASGI layer, BEFORE routing/auth. Per-route
# @limiter.limit decorators run after Depends(verify_auth) (Starlette order:
# Middleware -> Routing -> Endpoint), so without this global ceiling an
# unauthenticated flood bypasses rate limiting entirely — every request 401s
# before the per-route limiter fires. The global limiter is a SEPARATE instance
# from the strategy limiters below; slowapi's decorator binds to its own Limiter
# at decoration time and never reads app.state.limiter, so the two coexist:
# global = pre-auth ceiling on all routes; per-route = post-auth fine-grain.
_rate_limit_enabled = not _RATE_LIMITING_DISABLED
limiter_global = Limiter(
    key_func=_get_ip_key,
    default_limits=[_global_limit],
    enabled=_rate_limit_enabled,
)

# Create separate limiters for different strategies
limiter_hybrid = Limiter(key_func=_get_rate_limit_key, enabled=_rate_limit_enabled)
limiter_ip = Limiter(key_func=_get_ip_key, enabled=_rate_limit_enabled)
limiter_token = Limiter(key_func=_get_token_key, enabled=_rate_limit_enabled)

# SlowAPIMiddleware + _rate_limit_exceeded_handler read app.state.limiter for the
# global default; the per-strategy limiters stay on their own attrs for the
# per-route decorators (and for test reset).
app.state.limiter = limiter_global
app.state.limiter_global = limiter_global
app.state.limiter_hybrid = limiter_hybrid
app.state.limiter_ip = limiter_ip
app.state.limiter_token = limiter_token

# slowapi ships the handler narrowed to RateLimitExceeded; Starlette's handler
# protocol types the parameter as Exception and does not accept the narrower
# signature (no contravariance) — under slowapi <= 0.1.9. 0.1.10 (requirements.lock)
# fixed the handler typing, so the ignore is unused there; warn_unused_ignores
# turns that into an error. Listing both codes keeps local (0.1.9) and CI (0.1.10)
# green.
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,  # type: ignore[arg-type,unused-ignore]
)
# Added last → outermost middleware → rate limit fires before CORS, body-size,
# security headers, and auth (the desired pre-auth flood ceiling).
app.add_middleware(SlowAPIMiddleware)


def _rate_limit(limit_string: str, strategy: str = "hybrid") -> Callable:
    """Return a slowapi rate limit decorator with specified strategy.

    When ``TTS_DISABLE_RATE_LIMITING`` is set, returns a no-op decorator so
    every protected endpoint runs unthrottled (local E2E/CI only).

    Args:
        limit_string: Rate limit string (e.g., "10/minute")
        strategy: "hybrid" (both IP+token), "ip" (IP only), "token" (token only)

    Returns:
        Decorator function.

    Raises:
        ValueError: If ``strategy`` is not one of the known strategies.
    """
    if _RATE_LIMITING_DISABLED:

        def _noop(func):
            return func

        return _noop

    limiter_map = {
        "hybrid": limiter_hybrid,
        "ip": limiter_ip,
        "token": limiter_token,
    }
    selected_limiter = limiter_map.get(strategy)
    if selected_limiter is None:
        raise ValueError(f"Unknown rate-limit strategy: {strategy!r}")
    return selected_limiter.limit(limit_string)


# ---------------------------------------------------------------------------
# Public endpoints (no auth)
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Models still loading"}},
)
async def health(request: Request) -> dict[str, Any] | JSONResponse:
    """Health check endpoint."""
    state = request.app.state
    reset_activity_timer(state)

    if not state.models_loaded.is_set():
        data: dict[str, Any] = {"status": "loading"}
        # Include model load errors even during loading
        data["model_load_errors"] = {
            k: v for k, v in state.model_load_errors.items() if v is not None
        }
        return JSONResponse(content=data, status_code=503)

    backend = get_backend()
    data = {
        "status": "ok",
        # Liveness alone is not usability: a wedged server answers "ok" with
        # every model loaded while taking minutes per character. `status` stays
        # "ok" because real consumers (/ready probes, tests, shell checks) match
        # on it; this is an additive flag they can start honouring.
        #
        # BOOLEAN ONLY on this public endpoint — the supporting numbers reveal
        # the in-flight request's size, which /generation-status deliberately
        # strips for unauthenticated callers. They are on /stats, behind auth.
        "degraded": detect_degraded_generation(state)["degraded"],
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


@app.get("/ready", response_model=ReadyResponse)
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


@app.get("/generation-status", response_model=GenerationStatusResponse)
async def generation_status(request: Request) -> dict:
    """Get current generation status (public — sensitive fields stripped)."""
    state = request.app.state
    gen_state = state.generation_state
    # Public/no-auth endpoint: expose only liveness, cancellation, and coarse
    # progress position. Totals (batch_total, chunk_total) and eta_sec are
    # omitted because they reveal the batch size / text length of the in-flight
    # request to unauthenticated callers.
    result = {
        "active": gen_state["active"],
        "batch_index": gen_state["batch_index"],
        "chunk_index": gen_state["chunk_index"],
        "cancelled": gen_state["cancelled"],
    }
    if gen_state["active"]:
        result["elapsed_sec"] = round(time.time() - gen_state["start_time"], 1)
    return result


@app.get("/queue-status", response_model=QueueStatusResponse)
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


@app.get("/stats", response_model=StatsResponse)
async def stats(request: Request, _auth: None = Depends(verify_auth)) -> dict:
    """Get server statistics."""
    state = request.app.state
    reset_activity_timer(state)
    return handle_stats(state, state.server_config)


@app.get("/models", response_model=ModelsResponse)
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
@_rate_limit(_config_ops_limit, strategy="hybrid")
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


@app.post("/transcribe", response_model=TranscribeResponse)
@_rate_limit(_transcribe_limit)
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
@_rate_limit(_prompt_ops_limit, strategy="hybrid")
async def delete_prompt(
    request: Request, req: DeletePromptRequest, _auth: None = Depends(verify_auth)
):
    """Delete a voice prompt and all its format files."""
    state = request.app.state
    reset_activity_timer(state)
    return await asyncio.to_thread(handle_delete_prompt, state, req, _get_app_config)


@app.post("/rename-prompt")
@_rate_limit(_prompt_ops_limit, strategy="hybrid")
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
@_rate_limit(_prompt_ops_limit)
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
async def websocket_endpoint(
    websocket: WebSocket,
    origin_ok: None = Depends(validate_ws_origin),
):
    """Bidirectional WebSocket endpoint for real-time TTS streaming."""
    state = websocket.app.state

    def _verify_token(token: str) -> bool:
        return secrets.compare_digest(token, state.auth_token)

    await websocket_tts_handler(
        websocket, state, _verify_token, _app_config_provider
    )


@app.post("/shutdown")
async def shutdown(request: Request, _auth: None = Depends(verify_auth)) -> Response:
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
    from qwen3_tts.core.config import LOG_FILE

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
        host = "0.0.0.0"  # nosec B104  # intentional network bind via --public; logged on the next line
        logger.warning("Binding to 0.0.0.0 — server is accessible from the network.")

    if IN_COLAB:
        host = "0.0.0.0"  # nosec B104  # Colab requires 0.0.0.0 for tunnel access; logged on next line
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
