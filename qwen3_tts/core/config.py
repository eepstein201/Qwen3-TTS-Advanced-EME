#!/usr/bin/env python3
"""Lightweight TTS configuration — no torch, numpy, or heavy imports.

This module provides:
- Path constants (CONFIG_PATH, VOICE_PROMPTS_DIR, etc.)
- Config loading/saving helpers
- Server URL and status helpers
- Canonical CUSTOM_VOICE_SPEAKERS and MODEL_INFO dicts
- Error class hierarchy
"""

import json
import logging
import os
import pathlib
import platform
import sys

logger = logging.getLogger("tts.config")


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

IN_COLAB = "google.colab" in sys.modules
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

USER_FILES_DIR = os.path.expanduser("~/Qwen3-TTS_UserFiles")
CONFIG_PATH = os.path.join(USER_FILES_DIR, "config.json")
VOICE_PROMPTS_DIR = os.path.join(USER_FILES_DIR, "voice_prompts")
HISTORY_FILE = os.path.expanduser("~/.voice_history.jsonl")
PID_FILE = pathlib.Path(os.path.join(USER_FILES_DIR, ".voice_server.pid"))
LOG_FILE = pathlib.Path(os.path.join(USER_FILES_DIR, ".voice_server.log"))
TOKEN_FILE = pathlib.Path(os.path.expanduser("~/.voice_server_token"))

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_config_cache = {"data": None, "mtime": 0}


def validate_config(config):
    """Validate config structure and values. Logs warnings for issues."""
    issues = []
    backend = config.get("advanced", {}).get("backend")
    if backend and backend not in VALID_BACKENDS:
        issues.append(f"Invalid backend: {backend}")
    size = config.get("advanced", {}).get("model_size")
    if size and size not in ("1.7B", "0.6B"):
        issues.append(f"Invalid model_size: {size}")
    temp = config.get("generation", {}).get("temperature")
    if temp is not None and not (0.0 <= temp <= 2.0):
        issues.append(f"temperature {temp} out of range 0.0-2.0")
    mtl = config.get("security", {}).get("max_text_length")
    if mtl is not None and (not isinstance(mtl, int) or mtl <= 0):
        issues.append("max_text_length must be positive integer")
    # vLLM-specific validation
    vllm_gpu = config.get("advanced", {}).get("vllm_gpu_memory_utilization")
    if vllm_gpu is not None and not (0.0 < vllm_gpu <= 1.0):
        issues.append(f"vllm_gpu_memory_utilization {vllm_gpu} out of range 0.0-1.0")
    vllm_port = config.get("advanced", {}).get("vllm_port")
    if vllm_port is not None and not (1024 <= vllm_port <= 65535):
        issues.append(f"vllm_port {vllm_port} out of range 1024-65535")
    for issue in issues:
        logger.warning("Config validation: %s", issue)
    return issues


def load_config():
    """Load configuration from config.json with mtime-based caching.

    Returns a cached copy if config.json has not been modified since the last read.
    """
    try:
        current_mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        current_mtime = 0

    if _config_cache["data"] is not None and current_mtime == _config_cache["mtime"]:
        return _config_cache["data"]

    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
    _config_cache["data"] = data
    _config_cache["mtime"] = current_mtime
    validate_config(data)
    return data


def save_config(config):
    """Save configuration to config.json and invalidate the cache."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    _config_cache["data"] = None
    _config_cache["mtime"] = 0


def get_default_clone_prompt(config=None):
    """Return the default clone prompt filename.

    Reads from config's "default_clone_prompt" key. If missing or the file
    doesn't exist, falls back to the first .pt file found in VOICE_PROMPTS_DIR.
    Returns None if no prompts are available.
    """
    if config is None:
        try:
            config = load_config()
        except (json.JSONDecodeError, OSError):
            config = {}

    configured = config.get("default_clone_prompt")
    if configured:
        # Check it exists (as .pt or as MLX .wav/.txt pair)
        base = configured[:-3] if configured.endswith(".pt") else configured
        pt_exists = os.path.exists(os.path.join(VOICE_PROMPTS_DIR, f"{base}.pt"))
        mlx_exists = (os.path.exists(os.path.join(VOICE_PROMPTS_DIR, f"{base}.wav"))
                      and os.path.exists(os.path.join(VOICE_PROMPTS_DIR, f"{base}.txt")))
        if pt_exists or mlx_exists:
            return configured

    # Fallback: first .pt file in voice_prompts/
    try:
        prompts = sorted(f for f in os.listdir(VOICE_PROMPTS_DIR) if f.endswith(".pt"))
        if prompts:
            return prompts[0]
    except OSError:
        pass

    return None


def set_default_clone_prompt(prompt_name, config=None):
    """Set the default clone prompt in config.json."""
    if config is None:
        config = load_config()
    config["default_clone_prompt"] = prompt_name
    save_config(config)


def get_device():
    """Return the PyTorch device string for this platform.

    No torch import — uses environment/platform detection only.
    Returns: 'cuda', 'mps', or 'cpu'
    """
    if IN_COLAB or IS_LINUX:
        if (os.environ.get("CUDA_VISIBLE_DEVICES") is not None
                or os.path.exists("/dev/nvidia0")):
            return "cuda"
        return "cpu"
    if IS_MACOS and platform.machine() == "arm64":
        return "mps"
    return "cpu"


def get_cuda_capability():
    """Return CUDA compute capability as (major, minor) or None if no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()
    except ImportError:
        pass
    return None


def _has_flash_attn():
    """Check if flash_attn package is importable."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def get_optimal_attn_config():
    """Return (attn_implementation, torch_dtype_name, load_in_8bit) based on GPU.

    Turing (T4, CC 7.5): sdpa, float16, True (8-bit via bitsandbytes)
    Ampere+ (L4/A100, CC >= 8.0): flash_attention_2 (if installed) or sdpa, bfloat16, False
    Non-CUDA: sdpa, float32, False
    """
    cap = get_cuda_capability()
    if cap is None:
        return "sdpa", "float32", False
    if cap[0] >= 8:
        attn = "flash_attention_2" if _has_flash_attn() else "sdpa"
        return attn, "bfloat16", False
    return "sdpa", "float16", True


def get_server_url(config):
    """Return the server base URL from a config dict."""
    server = config.get("server", {})
    host = server.get("host", "127.0.0.1")
    port = server.get("port", 5123)
    return f"http://{host}:{port}"


def is_server_running(config_or_url=None):
    """Check whether the TTS server is reachable.

    Args:
        config_or_url: A config dict, a URL string, or None (loads config).
    """
    if config_or_url is None:
        config_or_url = load_config()

    if isinstance(config_or_url, str):
        url = config_or_url
    else:
        url = get_server_url(config_or_url)

    try:
        import requests
    except ImportError:
        return False
    try:
        resp = requests.get(f"{url}/health", timeout=2)
        return resp.status_code in (200, 503)
    except (requests.RequestException, OSError):
        return False


# ---------------------------------------------------------------------------
# Torch dtype configuration
# ---------------------------------------------------------------------------

VALID_DTYPES = ("float32", "float16", "bfloat16")
VALID_BACKENDS = ("torch", "mlx", "vllm")
VALID_MLX_QUANTIZATIONS = ("4bit", "8bit", "bf16")
VALID_TORCH_QUANTIZATIONS = ("none", "8bit", "4bit")
VALID_MODEL_SIZES = ("1.7B", "0.6B")


def get_torch_dtype_name():
    """Read the configured dtype from config.json (advanced.dtype).

    Returns:
        A string: "float32", "float16", or "bfloat16".
        Defaults to "float32" if not set or invalid.
    """
    try:
        config = load_config()
        dtype = config.get("advanced", {}).get("dtype", "float32")
    except (json.JSONDecodeError, OSError):
        dtype = "float32"
    if dtype not in VALID_DTYPES:
        dtype = "float32"
    return dtype


def get_backend():
    """Read the configured backend from config.json (advanced.backend).

    The TTS_BACKEND environment variable overrides the config file,
    allowing --backend CLI flag to work without modifying config.json.

    Returns:
        A string: "torch", "mlx", or "vllm".
        Defaults to "mlx" on macOS ARM64 (Apple Silicon optimized),
        "torch" on all other platforms (Colab/Linux/Intel Mac).
    """
    # Platform-aware default: MLX only on Apple Silicon
    _default = "mlx" if (IS_MACOS and platform.machine() == "arm64") else "torch"

    # Environment variable override (set by --backend CLI flag)
    env_backend = os.environ.get("TTS_BACKEND")
    if env_backend and env_backend in VALID_BACKENDS:
        return env_backend
    try:
        config = load_config()
        backend = config.get("advanced", {}).get("backend", _default)
    except (json.JSONDecodeError, OSError):
        backend = _default
    if backend not in VALID_BACKENDS:
        backend = _default
    return backend


def get_mlx_quantization():
    """Read the configured MLX quantization from config.json (advanced.mlx_quantization).

    Returns:
        A string: "4bit", "8bit", or "bf16".
        Defaults to "8bit" if not set or invalid.
    """
    try:
        config = load_config()
        quant = config.get("advanced", {}).get("mlx_quantization", "8bit")
    except (json.JSONDecodeError, OSError):
        quant = "8bit"
    if quant not in VALID_MLX_QUANTIZATIONS:
        quant = "8bit"
    return quant


def get_model_size():
    """Read the configured model size from config.json (advanced.model_size).

    The TTS_MODEL_SIZE environment variable overrides the config file,
    allowing --model-size CLI flag to work without modifying config.json.

    Returns:
        A string: "1.7B" or "0.6B".
        Defaults to "1.7B" if not set or invalid.
    """
    # Environment variable override (set by --model-size CLI flag)
    env_size = os.environ.get("TTS_MODEL_SIZE")
    if env_size and env_size in VALID_MODEL_SIZES:
        return env_size
    try:
        config = load_config()
        size = config.get("advanced", {}).get("model_size", "1.7B")
    except (json.JSONDecodeError, OSError):
        size = "1.7B"
    if size not in VALID_MODEL_SIZES:
        size = "1.7B"
    return size


def get_torch_quantization():
    """Read the configured torch quantization from config.json (advanced.torch_quantization).

    Returns:
        A string: "none", "8bit", or "4bit".
        Defaults to "none" if not set or invalid.

    Note:
        4-bit quantization requires bitsandbytes on CUDA/Linux only.
        8-bit quantization requires bitsandbytes.
        "none" means no quantization (use full precision).
    """
    try:
        config = load_config()
        quant = config.get("advanced", {}).get("torch_quantization", "none")
    except (json.JSONDecodeError, OSError):
        quant = "none"
    if quant not in VALID_TORCH_QUANTIZATIONS:
        quant = "none"
    return quant


def get_vllm_gpu_util():
    """Read the configured vLLM GPU memory utilization from config.json.

    Returns:
        A float between 0.0 and 1.0 representing GPU memory fraction.
        Defaults to 0.7 if not set or invalid.
    """
    try:
        config = load_config()
        util = config.get("advanced", {}).get("vllm_gpu_memory_utilization", 0.7)
    except (json.JSONDecodeError, OSError):
        util = 0.7
    if not isinstance(util, (int, float)) or not (0.0 < util <= 1.0):
        util = 0.7
    return float(util)


def get_vllm_port():
    """Read the configured vLLM port from config.json.

    Returns:
        An integer port number, or None for auto-find (default).
    """
    try:
        config = load_config()
        port = config.get("advanced", {}).get("vllm_port")
    except (json.JSONDecodeError, OSError):
        port = None
    if port is not None:
        if not isinstance(port, int) or not (1024 <= port <= 65535):
            logger.warning("Invalid vllm_port %s, using auto-find", port)
            port = None
    return port


# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

def get_voice_prompt_cache_max():
    """Read the configured voice prompt cache max size from config.json.

    Returns:
        An integer representing the maximum number of voice prompts to cache.
        Defaults to 10 if not set or invalid.
    """
    try:
        config = load_config()
        max_size = config.get("cache", {}).get("voice_prompt_max", 10)
    except (json.JSONDecodeError, OSError):
        max_size = 10
    if not isinstance(max_size, int) or max_size < 1:
        max_size = 10
    return max_size


def get_generation_cache_max():
    """Read the configured generation cache max size from config.json.

    Returns:
        An integer representing the maximum number of generations to cache.
        Defaults to 5 if not set or invalid.
    """
    try:
        config = load_config()
        max_size = config.get("cache", {}).get("generation_max", 5)
    except (json.JSONDecodeError, OSError):
        max_size = 5
    if not isinstance(max_size, int) or max_size < 1:
        max_size = 5
    return max_size


def get_eta_cache_ttl():
    """Read the configured ETA cache TTL from config.json.

    Returns:
        An integer representing the TTL in seconds for ETA cache entries.
        Defaults to 30 if not set or invalid.
    """
    try:
        config = load_config()
        ttl = config.get("cache", {}).get("eta_ttl_seconds", 30)
    except (json.JSONDecodeError, OSError):
        ttl = 30
    if not isinstance(ttl, int) or ttl < 0:
        ttl = 30
    return ttl


# ---------------------------------------------------------------------------
# Canonical speaker and model data
# ---------------------------------------------------------------------------

CUSTOM_VOICE_SPEAKERS = {
    "ryan":     {"name": "Ryan",      "lang": "English",  "desc": "Dynamic male, strong rhythm"},
    "aiden":    {"name": "Aiden",     "lang": "English",  "desc": "Sunny American male, clear midrange"},
    "vivian":   {"name": "Vivian",    "lang": "Chinese",  "desc": "Bright young female"},
    "serena":   {"name": "Serena",    "lang": "Chinese",  "desc": "Warm, gentle female"},
    "uncle_fu": {"name": "Uncle_Fu",  "lang": "Chinese",  "desc": "Seasoned male, mellow timbre"},
    "dylan":    {"name": "Dylan",     "lang": "Chinese",  "desc": "Youthful Beijing male"},
    "eric":     {"name": "Eric",      "lang": "Chinese",  "desc": "Lively Chengdu male"},
    "ono_anna": {"name": "Ono_Anna",  "lang": "Japanese", "desc": "Playful female"},
    "sohee":    {"name": "Sohee",     "lang": "Korean",   "desc": "Warm female, rich emotion"},
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
    model_size = get_model_size()
    backend = get_backend()
    if backend == "mlx":
        size_info = MLX_MODEL_INFO.get(model_size, {})
    else:
        size_info = MODEL_INFO.get(model_size, {})
    return size_info.get(model_type, {})


# ---------------------------------------------------------------------------
# Prosody presets (instruct text templates for Custom & Design modes)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Voice Description Builder attributes (Design mode UI)
# ---------------------------------------------------------------------------

VOICE_DESCRIPTION_ATTRIBUTES = {
    "gender": ["Male", "Female", "Androgynous"],
    "age": ["Young (18-25)", "Adult (25-45)", "Middle-aged (45-60)", "Elderly (60+)"],
    "tone": ["Warm", "Authoritative", "Gentle", "Energetic", "Calm", "Serious", "Playful"],
    "texture": ["Smooth", "Gravelly", "Crisp", "Breathy", "Rich", "Clear", "Husky", "Raspy"],
    "pace": ["Slow and deliberate", "Moderate", "Fast-paced", "Measured", "Unhurried"],
    "accent": ["Neutral American", "British RP", "Australian", "Southern American", "None/Default"],
}


DEFAULT_PROSODY_PRESETS = {
    "excited": "Speak with excitement and high energy",
    "calm": "Speak in a calm, soothing, relaxed manner",
    "whisper": "Speak in a soft whisper",
    "authoritative": "Speak in a confident, authoritative tone",
    "slow": "Speak slowly and deliberately with clear enunciation",
    "fast": "Speak quickly with urgency",
    "dramatic": "Speak with dramatic flair and emotional intensity",
    "conversational": "Speak in a casual, natural conversational style",
}


def get_prosody_presets(config=None):
    """Return prosody presets dict (user-defined + defaults).

    User presets in config.json override defaults with the same key.
    """
    if config is None:
        try:
            config = load_config()
        except (json.JSONDecodeError, OSError):
            config = {}
    user_presets = config.get("prosody_presets", {})
    merged = dict(DEFAULT_PROSODY_PRESETS)
    merged.update(user_presets)
    return merged


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

def read_auth_token():
    """Read the server auth token from TOKEN_FILE.

    Returns:
        The token string, or None if file doesn't exist.
    """
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            return f.read().strip()
    return None


def auth_headers():
    """Return HTTP headers dict with Bearer auth token, or empty dict."""
    token = read_auth_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


class TTSError(Exception):
    """Base error for all TTS operations.

    Attributes:
        user_message:     Short message safe to show end-users.
        technical_detail: Optional developer-facing detail.
        recovery:         Hint — "restart" | "config" | "bug" | "retry".
    """

    def __init__(self, user_message, technical_detail=None, recovery="restart"):
        self.user_message = user_message
        self.technical_detail = technical_detail
        self.recovery = recovery
        super().__init__(user_message)

    def format_cli(self):
        """Format for terminal display."""
        parts = [f"Error: {self.user_message}"]
        if self.technical_detail:
            parts[0] += f" [{self.technical_detail}]"
        suggestions = {
            "restart": "Try restarting the server with 'tts server start'.",
            "config": f"Check your configuration in {CONFIG_PATH}.",
            "bug": "This is an unexpected error — please report it.",
            "retry": "Try again; the issue may be transient.",
        }
        hint = suggestions.get(self.recovery, "")
        if hint:
            parts.append(f"  Suggestion: {hint}")
        return "\n".join(parts)

    def format_gradio(self):
        """Format for Gradio UI display."""
        color = {
            "restart": "#c0392b",
            "config": "#e67e22",
            "bug": "#8e44ad",
            "retry": "#2980b9",
        }.get(self.recovery, "#333")
        html = f'<span style="color:{color};font-weight:bold;">{self.user_message}</span>'
        if self.technical_detail:
            html += f'<br><small style="color:#666;">{self.technical_detail}</small>'
        suggestions = {
            "restart": "Try restarting the server with <code>tts server start</code>.",
            "config": f"Check your configuration in <code>{CONFIG_PATH}</code>.",
            "bug": "This is an unexpected error — please report it.",
            "retry": "Try again; the issue may be transient.",
        }
        hint = suggestions.get(self.recovery, "")
        if hint:
            html += f'<br><em>{hint}</em>'
        return html


class ServerConnectionError(TTSError):
    """Server is unreachable."""

    def __init__(self, detail=None):
        super().__init__(
            "Cannot connect to TTS server.",
            technical_detail=detail,
            recovery="restart",
        )


class ModelNotLoadedError(TTSError):
    """Required model is not loaded on the server."""

    def __init__(self, model_type, detail=None):
        if not detail:
            try:
                size = get_model_size()
            except Exception:
                size = "1.7B"
            detail = MODEL_INFO.get(size, {}).get(model_type, {}).get("description", "")
        super().__init__(
            f"The '{model_type}' model is not loaded.",
            technical_detail=detail,
            recovery="restart",
        )
        self.model_type = model_type


class InvalidInputError(TTSError):
    """User-provided input failed validation."""

    def __init__(self, detail):
        super().__init__(detail, recovery="config")


class GenerationError(TTSError):
    """Generation failed at runtime."""

    def __init__(self, detail=None):
        super().__init__(
            "Audio generation failed.",
            technical_detail=detail,
            recovery="restart",
        )


class AuthenticationError(TTSError):
    """Authentication with the server failed."""

    def __init__(self, detail=None):
        super().__init__(
            "Authentication failed.",
            technical_detail=detail or "Cannot authenticate. Run 'tts server start' to generate auth token.",
            recovery="restart",
        )
