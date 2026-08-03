#!/usr/bin/env python3
"""Model/backend configuration getters, cache-size config, and canonical
speaker/model metadata (CUSTOM_VOICE_SPEAKERS, MODEL_INFO, MLX_MODEL_INFO).

No torch, numpy, or heavy imports.

References to ``load_config`` and ``IS_MACOS`` (defined in sibling
submodules) are resolved via a lazy per-call import from
``qwen3_tts.core.config`` (the package facade), not a static module-level
import — see qwen3_tts/core/config/__init__.py for the rationale.
"""

import json
import os
import platform
from typing import Any

# ---------------------------------------------------------------------------
# Torch dtype / backend / quantization configuration
# ---------------------------------------------------------------------------

VALID_DTYPES = ("float32", "float16", "bfloat16")
VALID_BACKENDS = ("torch", "mlx", "vllm")
VALID_MLX_QUANTIZATIONS = ("4bit", "5bit", "6bit", "8bit", "bf16")
VALID_TORCH_QUANTIZATIONS = ("none", "8bit", "4bit")
VALID_MODEL_SIZES = ("1.7B", "0.6B")

# Single source of truth for the four generation sampling-parameter defaults.
# Drives get_default_config()["generation"], the validate_config temperature
# clamp, and the Gradio UI sliders/payload fallbacks (via get_generation_defaults)
# so these values can never drift apart across those surfaces.
DEFAULT_GENERATION_PARAMS = {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
}


def get_torch_dtype_name() -> str:
    """Read the configured dtype from config.json (advanced.dtype).

    Returns:
        A string: "float32", "float16", or "bfloat16".
        Defaults to "float32" if not set or invalid.
    """
    from qwen3_tts.core.config import load_config

    try:
        config = load_config()
        dtype = config.get("advanced", {}).get("dtype", "float32")
    except (json.JSONDecodeError, ValueError, OSError):
        dtype = "float32"
    if dtype not in VALID_DTYPES:
        dtype = "float32"
    return dtype


def get_backend() -> str:
    """Read the configured backend from config.json (advanced.backend).

    The TTS_BACKEND environment variable overrides the config file,
    allowing --backend CLI flag to work without modifying config.json.

    Returns:
        A string: "torch", "mlx", or "vllm".
        Defaults to "mlx" on macOS ARM64 (Apple Silicon optimized),
        "torch" on all other platforms (Colab/Linux/Intel Mac).
    """
    from qwen3_tts.core.config import IS_MACOS, load_config

    # Platform-aware default: MLX only on Apple Silicon
    _default = "mlx" if (IS_MACOS and platform.machine() == "arm64") else "torch"

    # Environment variable override (set by --backend CLI flag)
    env_backend = os.environ.get("TTS_BACKEND")
    if env_backend and env_backend in VALID_BACKENDS:
        return env_backend
    try:
        config = load_config()
        backend = config.get("advanced", {}).get("backend", _default)
    except (json.JSONDecodeError, ValueError, OSError):
        backend = _default
    if backend not in VALID_BACKENDS:
        backend = _default
    return backend


def get_mlx_quantization() -> str:
    """Read the configured MLX quantization from config.json (advanced.mlx_quantization).

    Returns:
        A string: "4bit", "5bit", "6bit", "8bit", or "bf16".
        Defaults to "8bit" if not set or invalid.
    """
    from qwen3_tts.core.config import load_config

    try:
        config = load_config()
        quant = config.get("advanced", {}).get("mlx_quantization", "8bit")
    except (json.JSONDecodeError, ValueError, OSError):
        quant = "8bit"
    if quant not in VALID_MLX_QUANTIZATIONS:
        quant = "8bit"
    return quant


def get_model_size() -> str:
    """Read the configured model size from config.json (advanced.model_size).

    The TTS_MODEL_SIZE environment variable overrides the config file,
    allowing --model-size CLI flag to work without modifying config.json.

    Returns:
        A string: "1.7B" or "0.6B".
        Defaults to "1.7B" if not set or invalid.
    """
    from qwen3_tts.core.config import load_config

    # Environment variable override (set by --model-size CLI flag)
    env_size = os.environ.get("TTS_MODEL_SIZE")
    if env_size and env_size in VALID_MODEL_SIZES:
        return env_size
    try:
        config = load_config()
        size = config.get("advanced", {}).get("model_size", "1.7B")
    except (json.JSONDecodeError, ValueError, OSError):
        size = "1.7B"
    if size not in VALID_MODEL_SIZES:
        size = "1.7B"
    return size


def get_generation_defaults() -> dict[str, float | int]:
    """Return the configured generation sampling defaults for the UI.

    Reads the "generation" section from config.json and returns the four
    sampling parameters the Gradio UI exposes (temperature, top_k, top_p,
    repetition_penalty). Any key missing from the user's config falls back to
    DEFAULT_GENERATION_PARAMS; a corrupt or unreadable config falls back
    entirely to the canonical defaults. This keeps the UI sliders and payload
    fallbacks in lockstep with the user's config instead of hardcoded values.

    Returns:
        A dict with keys temperature (float), top_k (int), top_p (float),
        and repetition_penalty (float).
    """
    from qwen3_tts.core.config import load_config

    try:
        config = load_config()
    except (json.JSONDecodeError, ValueError, OSError):
        config = {}
    generation = config.get("generation", {}) if isinstance(config, dict) else {}
    return {
        key: generation.get(key, DEFAULT_GENERATION_PARAMS[key])
        for key in DEFAULT_GENERATION_PARAMS
    }


def get_torch_quantization() -> str:
    """Read the configured torch quantization from config.json (advanced.torch_quantization).

    Returns:
        A string: "none", "8bit", or "4bit".
        Defaults to "none" if not set or invalid.

    Note:
        4-bit quantization requires bitsandbytes on CUDA/Linux only.
        8-bit quantization requires bitsandbytes.
        "none" means no quantization (use full precision).
    """
    from qwen3_tts.core.config import load_config

    try:
        config = load_config()
        quant = config.get("advanced", {}).get("torch_quantization", "none")
    except (json.JSONDecodeError, ValueError, OSError):
        quant = "none"
    if quant not in VALID_TORCH_QUANTIZATIONS:
        quant = "none"
    return quant


# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------


def _get_config_value(
    key_path: list, default: Any, validator: Any | None = None
) -> Any:
    """Get a nested config value with fallback.

    Args:
        key_path: List of keys to traverse (e.g., ["cache", "voice_prompt_max"])
        default: Default value if key not found or invalid
        validator: Optional function to validate the value

    Returns:
        The config value or default
    """
    from qwen3_tts.core.config import load_config

    try:
        config = load_config()
        val = config
        for key in key_path:
            val = val.get(key, {})
        if val == {}:
            return default
        if validator and not validator(val):
            return default
        return val if val is not None else default
    except (json.JSONDecodeError, ValueError, OSError):
        return default


def get_voice_prompt_cache_max() -> int:
    """Read the configured voice prompt cache max size from config.json.

    Returns:
        An integer representing the maximum number of voice prompts to cache.
        Defaults to 10 if not set or invalid.
    """
    return _get_config_value(
        ["cache", "voice_prompt_max"], 10, lambda x: isinstance(x, int) and x > 0
    )


def get_generation_cache_max() -> int:
    """Read the configured generation cache max size from config.json.

    Returns:
        An integer representing the maximum number of generations to cache.
        Defaults to 5 if not set or invalid.
    """
    return _get_config_value(
        ["cache", "generation_max"], 5, lambda x: isinstance(x, int) and x > 0
    )


def get_eta_cache_ttl() -> int:
    """Read the configured ETA cache TTL from config.json.

    Returns:
        An integer representing the TTL in seconds for ETA cache entries.
        Defaults to 30 if not set or invalid.
    """
    return _get_config_value(
        ["cache", "eta_ttl_seconds"], 30, lambda x: isinstance(x, int) and x >= 0
    )


# ---------------------------------------------------------------------------
# Canonical speaker and model data
# ---------------------------------------------------------------------------

CUSTOM_VOICE_SPEAKERS = {
    "ryan": {"name": "Ryan", "lang": "English", "desc": "Dynamic male, strong rhythm"},
    "aiden": {
        "name": "Aiden",
        "lang": "English",
        "desc": "Sunny American male, clear midrange",
    },
    "vivian": {"name": "Vivian", "lang": "Chinese", "desc": "Bright young female"},
    "serena": {"name": "Serena", "lang": "Chinese", "desc": "Warm, gentle female"},
    "uncle_fu": {
        "name": "Uncle_Fu",
        "lang": "Chinese",
        "desc": "Seasoned male, mellow timbre",
    },
    "dylan": {"name": "Dylan", "lang": "Chinese", "desc": "Youthful Beijing male"},
    "eric": {"name": "Eric", "lang": "Chinese", "desc": "Lively Chengdu male"},
    "ono_anna": {"name": "Ono_Anna", "lang": "Japanese", "desc": "Playful female"},
    "sohee": {"name": "Sohee", "lang": "Korean", "desc": "Warm female, rich emotion"},
}

MODEL_INFO = {
    "1.7B": {
        "clone": {
            "name": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "description": "Voice cloning from audio samples (clone mode)",
            "memory_mb": 3500,
        },
        "design": {
            "name": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "description": "Generate voice from text description (design mode)",
            "memory_mb": 3500,
        },
        "custom": {
            "name": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "description": "9 premium pre-trained speakers (custom mode)",
            "memory_mb": 3500,
        },
    },
    "0.6B": {
        "clone": {
            "name": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "description": "Voice cloning from audio samples (clone mode, lightweight)",
            "memory_mb": 2000,
        },
        "design": {
            "name": "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
            "description": "Generate voice from text description (design mode, lightweight)",
            "memory_mb": 2000,
        },
        "custom": {
            "name": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "description": "9 premium pre-trained speakers (custom mode, lightweight)",
            "memory_mb": 2000,
        },
    },
}

MLX_MODEL_INFO = {
    "1.7B": {
        "clone": {
            "name_template": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-{quant}",
            "description": "Voice cloning from audio samples (clone mode)",
            "memory_mb": 2500,
        },
        "design": {
            "name_template": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-{quant}",
            "description": "Generate voice from text description (design mode)",
            "memory_mb": 2500,
        },
        "custom": {
            "name_template": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-{quant}",
            "description": "9 premium pre-trained speakers (custom mode)",
            "memory_mb": 2500,
        },
    },
    "0.6B": {
        "clone": {
            "name_template": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-{quant}",
            "description": "Voice cloning from audio samples (clone mode, lightweight)",
            "memory_mb": 1500,
        },
        "design": {
            "name_template": "mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-{quant}",
            "description": "Generate voice from text description (design mode, lightweight)",
            "memory_mb": 1500,
        },
        "custom": {
            "name_template": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-{quant}",
            "description": "9 premium pre-trained speakers (custom mode, lightweight)",
            "memory_mb": 1500,
        },
    },
}


def get_mlx_model_name(model_type):
    """Return the MLX HuggingFace repo ID for a given model type.

    Substitutes the configured quantization level and model size into the name template.
    """
    # Lazy facade import (not a plain call to the sibling function defined
    # above): tests patch "qwen3_tts.core.config.get_model_size" /
    # "get_mlx_quantization" at the package facade, which a same-module bare
    # call would not observe. See qwen3_tts/core/config/__init__.py.
    from qwen3_tts.core.config import get_mlx_quantization, get_model_size

    model_size = get_model_size()
    size_info = MLX_MODEL_INFO.get(model_size)
    if not size_info:
        raise ValueError(f"Unknown model size: {model_size}")
    info = size_info.get(model_type)
    if not info:
        raise ValueError(f"Unknown model type: {model_type}")
    quant = get_mlx_quantization()
    return info["name_template"].format(quant=quant)


def get_torch_model_name(model_type):
    """Return the PyTorch HuggingFace repo ID for a given model type.

    Uses the configured model size.
    """
    from qwen3_tts.core.config import get_model_size

    model_size = get_model_size()
    size_info = MODEL_INFO.get(model_size)
    if not size_info:
        raise ValueError(f"Unknown model size: {model_size}")
    info = size_info.get(model_type)
    if not info:
        raise ValueError(f"Unknown model type: {model_type}")
    return info["name"]


def get_model_info(model_type):
    """Return the model info dict for a given model type.

    Uses the configured model size and backend.
    """
    from qwen3_tts.core.config import get_backend, get_model_size

    model_size = get_model_size()
    backend = get_backend()
    if backend == "mlx":
        size_info = MLX_MODEL_INFO.get(model_size, {})
    else:
        size_info = MODEL_INFO.get(model_size, {})
    return size_info.get(model_type, {})


def get_model_revision(model_type):
    """Return the HuggingFace revision (branch/tag/SHA) to download for a model type.

    Resolution order, falling back to "main" (current behavior — no change today):
      1. config.json: models.<model_type>.revision
      2. MODEL_INFO / MLX_MODEL_INFO entry's "revision" key
      3. "main"

    Pinning a specific SHA/tag here (via config or MODEL_INFO) lets a deployment
    avoid silently tracking a repo's moving ``main`` branch.
    """
    from qwen3_tts.core.config import get_model_info as get_info
    from qwen3_tts.core.config import load_config

    config = load_config()
    pinned = config.get("models", {}).get(model_type, {}).get("revision")
    if pinned:
        return pinned
    return get_info(model_type).get("revision") or "main"


# ---------------------------------------------------------------------------
# Voice Description Builder attributes (Design mode UI)
# ---------------------------------------------------------------------------

VOICE_DESCRIPTION_ATTRIBUTES = {
    "gender": ["Male", "Female", "Androgynous"],
    "age": ["Young (18-25)", "Adult (25-45)", "Middle-aged (45-60)", "Elderly (60+)"],
    "tone": [
        "Warm",
        "Authoritative",
        "Gentle",
        "Energetic",
        "Calm",
        "Serious",
        "Playful",
    ],
    "texture": [
        "Smooth",
        "Gravelly",
        "Crisp",
        "Breathy",
        "Rich",
        "Clear",
        "Husky",
        "Raspy",
    ],
    "pace": ["Slow and deliberate", "Moderate", "Fast-paced", "Measured", "Unhurried"],
    "accent": [
        "Neutral American",
        "British RP",
        "Australian",
        "Southern American",
        "None/Default",
    ],
}
