"""Request/response validation helpers for TTS server.

This module contains:
- Pydantic models for request/response validation
- Validation functions for generation requests
- Helper functions for prompt name validation
- Error response helpers
"""

import hashlib
import re
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel, Field

from qwen3_tts.core.config import CUSTOM_VOICE_SPEAKERS, VOICE_PROMPTS_DIR

MAX_PROMPT_NAME_LEN = 255  # max length for voice prompt names
MAX_AUDIO_BASE64_BYTES = 50 * 1024 * 1024  # 50MB base64 ≈ 37.5MB raw audio
MAX_SEED = 2**31 - 1  # upper bound for generation seed (signed int32, safe across torch/MLX)

# Pre-computed valid speaker names (keys + display names)
_VALID_SPEAKER_NAMES = frozenset(CUSTOM_VOICE_SPEAKERS.keys()) | frozenset(
    v["name"] for v in CUSTOM_VOICE_SPEAKERS.values()
)


# ---------------------------------------------------------------------------
# Pydantic models for request validation
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request model for /generate and /generate-stream endpoints."""

    text: str | None = None
    texts: list[str] | None = None
    mode: str = "clone"
    prompt_file: str | None = None
    voice_description: str = ""
    language: str = "auto"
    speaker: str | None = None
    instruct: str = ""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=1, le=1000)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=0.5, le=2.0)
    max_new_tokens: int = Field(default=2048, ge=1, le=8192)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    max_chunk_chars: int | None = None
    x_vector_only_mode: bool = False
    seed_lock_chunks: bool = False


class LoadModelRequest(BaseModel):
    """Request model for /load-model endpoint."""

    model_type: str


class UnloadModelRequest(BaseModel):
    """Request model for /unload-model endpoint."""

    model_type: str


class UpdateModelConfigRequest(BaseModel):
    """Request model for /update-model-config endpoint."""

    model_size: str | None = None
    mlx_quantization: str | None = None


class UpdateStartupConfigRequest(BaseModel):
    """Request model for /update-startup-config endpoint."""

    clone: bool | None = None
    design: bool | None = None
    custom: bool | None = None


class DeletePromptRequest(BaseModel):
    """Request model for /delete-prompt endpoint."""

    name: str


class RenamePromptRequest(BaseModel):
    """Request model for /rename-prompt endpoint."""

    old_name: str
    new_name: str


class TranscribeRequest(BaseModel):
    """Request model for /transcribe endpoint."""

    audio_base64: str = Field(max_length=MAX_AUDIO_BASE64_BYTES)
    language: str = Field(default="en", pattern=r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")


class CreateVoicePromptRequest(BaseModel):
    """Request model for /create-voice-prompt endpoint."""

    audio_base64: str = Field(max_length=MAX_AUDIO_BASE64_BYTES)
    transcript: str = ""
    name: str
    no_transcript: bool = False


class CreateVoicePromptResponse(BaseModel):
    """Response for /create-voice-prompt (previously an untyped JSON route)."""

    status: str
    name: str


# ---------------------------------------------------------------------------
# Pydantic models for response validation
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str
    detail: str = ""
    recovery: str = "retry"


class GenerateResult(BaseModel):
    """Single generation result."""

    index: int
    audio_base64: str | None = None
    sample_rate: int
    peaks: list[float] | None = None
    chunks: int | None = None
    seed: int | None = None


class GenerateResponse(BaseModel):
    """Response model for /generate endpoint."""

    results: list[GenerateResult]
    # True when the batch stopped early because /cancel-generation fired.
    # ``results`` is then shorter than the requested texts (possibly empty), so
    # the client needs this to tell a cancellation apart from a full response.
    cancelled: bool = False


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""

    status: str
    # True when the in-flight generation is pathologically slow. `status` stays
    # "ok" in that case for compatibility, so this is the field to check for
    # usability rather than mere liveness. Boolean only — the supporting numbers
    # are on the authenticated /stats, since they reveal in-flight request size.
    degraded: bool | None = None
    backend: str | None = None
    model_size: str | None = None
    clone_model_loaded: bool | None = None
    design_model_loaded: bool | None = None
    custom_model_loaded: bool | None = None
    model_load_times: dict | None = None
    model_load_errors: dict | None = None
    mlx_quantization: str | None = None
    dtype: str | None = None


# ---------------------------------------------------------------------------
# Response contracts (GEN-2)
#
# Every model below describes an EXISTING handler's return dict. FastAPI
# response_model filters anything the model does not declare, so a wrong model
# silently drops fields — read the handler before editing these. Optional
# fields cover backend-conditional keys (mlx_quantization vs dtype) and
# hardware-absent memory stats so the same contract validates on every
# backend. The routes set response_model_exclude_unset=True so keys the
# handler did NOT emit (e.g. mlx memory stats on torch) stay absent instead
# of appearing as additive nulls — the response body is byte-identical to
# the pre-contract output. Untyped on purpose: /generate-stream and /ws
# (binary/WebSocket), /preview-prompt and /shutdown (non-JSON Response).
# ---------------------------------------------------------------------------


class ReadyResponse(BaseModel):
    """Response model for /ready (200 body; the 503 path is an HTTPException)."""

    status: str


class GenerationStatusResponse(BaseModel):
    """Response model for /generation-status (public — sensitive fields stripped)."""

    active: bool
    batch_index: int
    chunk_index: int
    cancelled: bool
    elapsed_sec: float | None = None  # only emitted while a generation is active


class QueueStatusResponse(BaseModel):
    """Response model for /queue-status."""

    queue_length: int
    active: bool


class GenerationHealth(BaseModel):
    """Nested /stats generation-health block (detect_degraded_generation)."""

    degraded: bool
    elapsed_sec: float | None = None
    sec_per_char: float | None = None
    threshold_sec_per_char: float


class StatsResponse(BaseModel):
    """Response model for /stats."""

    status: str
    backend: str
    generation_health: GenerationHealth
    model_size: str
    clone_model_loaded: bool
    design_model_loaded: bool
    custom_model_loaded: bool
    voice_prompts_cached: int
    voice_prompts_cache_hits: int
    idle_seconds: int
    auto_shutdown_minutes: int | str  # "disabled" when auto-shutdown is off
    generation_queue_size: int
    # Backend-conditional / hardware-optional fields (None on the other backend).
    mlx_quantization: str | None = None
    dtype: str | None = None
    mps_memory_allocated_mb: float | str | None = None  # "unavailable" on error
    cuda_memory_allocated_mb: float | None = None
    cuda_memory_reserved_mb: float | None = None
    mlx_memory_active_mb: float | None = None
    mlx_memory_peak_mb: float | None = None


class ModelEntry(BaseModel):
    """One model row in the /models listing."""

    loaded: bool
    loading: bool  # mutually exclusive with loaded (drives UI progress polls)
    description: str
    memory_mb: int
    repo_id: str
    load_at_startup: bool
    load_time_sec: float | None = None


class AsrInfo(BaseModel):
    """ASR block in the /models listing."""

    loaded: bool
    backend: str | None = None
    model_name: str | None = None


class ModelsResponse(BaseModel):
    """Response model for /models."""

    models: dict[str, ModelEntry]
    asr: AsrInfo
    backend: str
    model_size: str


class TranscribeResponse(BaseModel):
    """Response model for /transcribe."""

    transcript: str


class ModelOpResponse(BaseModel):
    """Response model for /load-model and /unload-model."""

    status: str  # "loaded" | "already_loaded" | "unloaded" | "already_unloaded"
    model: str
    # Phase 2c (#214 item 3): True only when this caller ATTACHED to an
    # in-flight load instead of owning it. response_model_exclude_unset
    # omits it otherwise — absence is the owner/already_loaded shape.
    deduped: bool = False
    # W1: True when the weights loaded cleanly but the design warm-up
    # failed non-fatally (the model stays; the first request pays the cold
    # start). Deliberately never mirrored into model_load_errors.
    warmup_failed: bool = False


class LoadAsrResponse(BaseModel):
    """Response model for /load-asr."""

    status: str  # "loaded" | "already_loaded"
    load_time_sec: float | None = None  # only emitted on a fresh load


class UnloadAsrResponse(BaseModel):
    """Response model for /unload-asr."""

    status: str  # "unloaded"


class UpdateModelConfigResponse(BaseModel):
    """Response model for /update-model-config."""

    status: str
    changes: list[str]
    models_unloaded: bool
    note: str


class UpdateStartupConfigResponse(BaseModel):
    """Response model for /update-startup-config."""

    status: str
    changes: list[str]


class PromptsListResponse(BaseModel):
    """Response model for /prompts (backend-aware file list)."""

    prompts: list[str]
    total: int
    offset: int = 0  # absent only on the OSError branch, which returns early
    limit: int = 0


class PromptInfo(BaseModel):
    """Metadata for one voice prompt (/prompt-details)."""

    name: str
    formats: list[str]
    size_bytes: int
    created: float | None = None  # None when no files exist for the base name
    is_default: bool


class PromptDetailsResponse(BaseModel):
    """Response model for /prompt-details with no name (all prompts)."""

    prompts: list[PromptInfo]


class DeletePromptResponse(BaseModel):
    """Response model for /delete-prompt."""

    status: str
    name: str
    files_removed: list[str]
    files_failed: list[str] | None = None  # only emitted when a removal failed


class RenamePromptResponse(BaseModel):
    """Response model for /rename-prompt."""

    status: str
    old_name: str
    new_name: str
    files_renamed: list[str]


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def _validate_generation_request(req: GenerateRequest, security_config: dict) -> None:
    """Shared validation for /generate and /generate-stream.

    Raises HTTPException for:
    - Path traversal in prompt_file
    - Invalid speaker name for custom mode
    - Invalid mode
    """
    # Path traversal check — use pathlib.resolve() to catch encoded sequences and symlinks
    if req.prompt_file:
        try:
            resolved = (Path(VOICE_PROMPTS_DIR) / req.prompt_file).resolve()
            if not resolved.is_relative_to(Path(VOICE_PROMPTS_DIR).resolve()):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid prompt_file: path traversal not allowed",
                )
        except (ValueError, OSError):
            raise HTTPException(
                status_code=400,
                detail="Invalid prompt_file: path traversal not allowed",
            )

    # Speaker validation for custom mode
    if req.mode == "custom" and req.speaker:
        speaker_key = req.speaker.lower() if isinstance(req.speaker, str) else ""
        if (
            speaker_key not in CUSTOM_VOICE_SPEAKERS
            and speaker_key not in _VALID_SPEAKER_NAMES
        ):
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


def _validate_prompt_name(name: str) -> tuple[dict, int] | None:
    """Validate prompt name — returns error tuple or None."""
    if not name or not name.strip():
        return {"error": "Missing prompt name", "recovery": "config"}, 400
    name = name.strip()
    if len(name) > MAX_PROMPT_NAME_LEN:
        return {"error": "Prompt name too long", "recovery": "config"}, 400
    if not re.match(r"^[a-zA-Z0-9_\-\.]+$", name):
        return {
            "error": "Invalid prompt name: only alphanumeric, dash, underscore, dot allowed",
            "recovery": "config",
        }, 400
    if ".." in name:
        return {"error": "Invalid prompt name", "recovery": "config"}, 400
    return None


def _strip_extension(name: str) -> str:
    """Strip .pt, .wav, or .txt extension from name."""
    base = name
    for ext in (".pt", ".wav", ".txt"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return base


def _gen_cache_key(
    text: str,
    mode: str,
    gen_params: dict,
    prompt_file: str | None = None,
    voice_description: str | None = None,
    speaker: str | None = None,
    instruct: str | None = None,
    language: str | None = None,
    x_vector_only_mode: bool = False,
    max_chunk_chars: int | None = None,
    seed_lock_chunks: bool = False,
) -> str:
    """Generate a hash key for generation cache lookup.

    The key must include EVERY input that changes the generated audio. Pre-fix
    it omitted language, x_vector_only_mode, max_chunk_chars, and
    seed_lock_chunks, so two requests differing only in one of those fields
    collided and served stale audio from the wrong configuration (e.g. an
    English request hitting a Spanish cache entry for the same text/voice).

    Note: the actual ``seed`` value is intentionally excluded (see
    app_generation.handle_generate) so blank-seed requests still cache-hit —
    only the seeding *strategy* (seed_lock_chunks) matters for correctness.
    """
    key_parts = [text, mode, str(sorted(gen_params.items()))]
    if prompt_file:
        key_parts.append(prompt_file)
    if voice_description:
        key_parts.append(voice_description)
    if speaker:
        key_parts.append(speaker)
    if instruct:
        key_parts.append(instruct)
    # Behavior toggles that alter output audio.
    key_parts.append(f"lang={language}")
    key_parts.append(f"xvo={x_vector_only_mode}")
    if max_chunk_chars is not None:
        key_parts.append(f"mcc={max_chunk_chars}")
    key_parts.append(f"slc={seed_lock_chunks}")
    raw = "|".join(key_parts)
    return hashlib.sha256(raw.encode(), usedforsecurity=False).hexdigest()[:16]


def _error_response(
    status_code: int, error: str, detail: str = "", recovery: str = "retry"
) -> None:
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
