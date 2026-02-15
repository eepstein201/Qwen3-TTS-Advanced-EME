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
import os

import requests

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

USER_FILES_DIR = os.path.expanduser("~/Qwen3-TTS_UserFiles")
CONFIG_PATH = os.path.join(USER_FILES_DIR, "config.json")
VOICE_PROMPTS_DIR = os.path.join(USER_FILES_DIR, "voice_prompts")
HISTORY_FILE = os.path.expanduser("~/.voice_history.jsonl")
PID_FILE = os.path.join(USER_FILES_DIR, ".voice_server.pid")
LOG_FILE = os.path.join(USER_FILES_DIR, ".voice_server.log")
TOKEN_FILE = os.path.expanduser("~/.voice_server_token")

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config():
    """Load configuration from config.json."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(config):
    """Save configuration to config.json."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def get_default_clone_prompt(config=None):
    """Return the default clone prompt filename.

    Reads from config's "default_clone_prompt" key. If missing or the file
    doesn't exist, falls back to the first .pt file found in VOICE_PROMPTS_DIR.
    Returns None if no prompts are available.
    """
    if config is None:
        try:
            config = load_config()
        except Exception:
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
        resp = requests.get(f"{url}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Torch dtype configuration
# ---------------------------------------------------------------------------

VALID_DTYPES = ("float32", "float16", "bfloat16")
VALID_BACKENDS = ("torch", "mlx")
VALID_MLX_QUANTIZATIONS = ("4bit", "8bit", "bf16")
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
    except Exception:
        dtype = "float32"
    if dtype not in VALID_DTYPES:
        dtype = "float32"
    return dtype


def get_backend():
    """Read the configured backend from config.json (advanced.backend).

    The TTS_BACKEND environment variable overrides the config file,
    allowing --backend CLI flag to work without modifying config.json.

    Returns:
        A string: "torch" or "mlx".
        Defaults to "mlx" if not set or invalid (Apple Silicon optimized).
    """
    # Environment variable override (set by --backend CLI flag)
    env_backend = os.environ.get("TTS_BACKEND")
    if env_backend and env_backend in VALID_BACKENDS:
        return env_backend
    try:
        config = load_config()
        backend = config.get("advanced", {}).get("backend", "mlx")
    except Exception:
        backend = "mlx"
    if backend not in VALID_BACKENDS:
        backend = "mlx"
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
    except Exception:
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
    except Exception:
        size = "1.7B"
    if size not in VALID_MODEL_SIZES:
        size = "1.7B"
    return size


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
            "restart": "Try restarting the server with 'startTTSServer'.",
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
            "restart": "Try restarting the server with <code>startTTSServer</code>.",
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
        super().__init__(
            f"The '{model_type}' model is not loaded.",
            technical_detail=detail or MODEL_INFO.get(model_type, {}).get("description", ""),
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
            technical_detail=detail or "Cannot authenticate. Run 'startTTSServer' to generate auth token.",
            recovery="restart",
        )
