"""Request/response validation helpers for TTS server.

This module contains:
- Pydantic models for request/response validation
- Validation functions for generation requests
- Helper functions for prompt name validation
- Error response helpers
"""
import hashlib
import re
from typing import Optional, List

from fastapi import HTTPException
from pydantic import BaseModel, Field

from qwen3_tts.core.config import CUSTOM_VOICE_SPEAKERS


# Pre-computed valid speaker names (keys + display names)
_VALID_SPEAKER_NAMES = frozenset(CUSTOM_VOICE_SPEAKERS.keys()) | frozenset(
    v["name"] for v in CUSTOM_VOICE_SPEAKERS.values()
)


# ---------------------------------------------------------------------------
# Pydantic models for request validation
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Request model for /generate and /generate-stream endpoints."""
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
    """Request model for /load-model endpoint."""
    model_type: str


class UnloadModelRequest(BaseModel):
    """Request model for /unload-model endpoint."""
    model_type: str


class UpdateModelConfigRequest(BaseModel):
    """Request model for /update-model-config endpoint."""
    model_size: Optional[str] = None
    mlx_quantization: Optional[str] = None


class UpdateStartupConfigRequest(BaseModel):
    """Request model for /update-startup-config endpoint."""
    clone: Optional[bool] = None
    design: Optional[bool] = None
    custom: Optional[bool] = None


class DeletePromptRequest(BaseModel):
    """Request model for /delete-prompt endpoint."""
    name: str


class RenamePromptRequest(BaseModel):
    """Request model for /rename-prompt endpoint."""
    old_name: str
    new_name: str


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
    audio_base64: Optional[str] = None
    sample_rate: int


class GenerateResponse(BaseModel):
    """Response model for /generate endpoint."""
    results: List[GenerateResult]


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
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
# Validation functions
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
