#!/usr/bin/env python3
"""Lightweight TTS configuration — no torch, numpy, or heavy imports.

This module provides:
- Path constants (CONFIG_PATH, VOICE_PROMPTS_DIR, etc.)
- Config loading/saving helpers
- Server URL and status helpers
- Canonical CUSTOM_VOICE_SPEAKERS and MODEL_INFO dicts
- Error class hierarchy
"""

import copy
import json
import logging
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
import threading
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

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


def _resolve_config_path():
    """Resolve config.json location across dev, CI, and packaged installs.

    Order: user files dir (normal runtime) -> repo-root config.json (CI
    checkout / source tree). The home-dir path coincides with the repo root
    on the maintainer's machine but not in CI, so the repo-root fallback
    (anchored on __file__, not on any external input) lets the committed
    config.json be found there. No environment-variable override is used:
    feeding an env var into open() is a path-injection source (CodeQL
    py/path-injection), and __file__ anchoring fixes CI without it.
    """
    home_cfg = os.path.join(USER_FILES_DIR, "config.json")
    if os.path.exists(home_cfg):
        return home_cfg
    repo_cfg = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config.json",
    )
    return repo_cfg if os.path.exists(repo_cfg) else home_cfg


CONFIG_PATH = _resolve_config_path()
VOICE_PROMPTS_DIR = pathlib.Path(USER_FILES_DIR) / "voice_prompts"
HISTORY_FILE = os.path.expanduser("~/.voice_history.jsonl")
PID_FILE = pathlib.Path(os.path.join(USER_FILES_DIR, ".voice_server.pid"))
LOG_FILE = pathlib.Path(os.path.join(USER_FILES_DIR, ".voice_server.log"))
_TOKEN_DIR = pathlib.Path(os.path.expanduser("~/.config/qwen3-tts"))
TOKEN_FILE = _TOKEN_DIR / ".voice_server_token"
_LEGACY_TOKEN_FILE = pathlib.Path(os.path.expanduser("~/.voice_server_token"))

# HuggingFace cache location (single source of truth)
HF_CACHE = pathlib.Path.home() / ".cache" / "huggingface" / "hub"

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_config_lock = threading.Lock()
_config_cache: dict[str, Any] = {"data": None, "mtime": 0}


def _validate_rate_limit_string(limit_str: str) -> bool:
    """Validate rate limit string format (e.g., '10/minute', '5/hour').

    Args:
        limit_str: String to validate

    Returns:
        True if valid format, False otherwise
    """
    if not isinstance(limit_str, str):
        return False
    parts = limit_str.split("/")
    if len(parts) != 2:
        return False
    try:
        count = int(parts[0])
        if count <= 0:
            return False
    except ValueError:
        return False
    unit = parts[1].lower()
    return unit in ("second", "minute", "hour", "day")


def _get_default_rate_limit(endpoint_type: str) -> str:
    """Get default rate limit for endpoint type.

    Args:
        endpoint_type: Type of endpoint ('generate', 'model_ops', etc.)

    Returns:
        Default rate limit string
    """
    defaults = {
        "generate": "20/minute",
        "model_ops": "3/minute",
        "transcribe": "15/minute",
        "prompt_ops": "10/minute",
        "config_ops": "1/minute",
    }
    return defaults.get(endpoint_type, "10/minute")


def validate_config(config: dict) -> tuple[dict, list[str]]:
    """Validate config structure and values, returning corrected copy.

    Invalid values are replaced with safe defaults. Returns a new config dict
    with corrections applied — the input config is never mutated.
    Missing keys are left missing (no bloat).

    Returns:
        (result, issues): result is a new dict with corrections; issues is a
        list of human-readable correction descriptions.
    """
    issues = []
    result = dict(config)
    adv = config.get("advanced")
    if isinstance(adv, dict):
        corrected_adv = dict(adv)
        backend = adv.get("backend")
        if backend and backend not in VALID_BACKENDS:
            new_backend = (
                "mlx" if (IS_MACOS and platform.machine() == "arm64") else "torch"
            )
            corrected_adv["backend"] = new_backend
            issues.append(f"corrected backend from {backend!r} to {new_backend!r}")
        size = adv.get("model_size")
        if size and size not in ("1.7B", "0.6B"):
            corrected_adv["model_size"] = "1.7B"
            issues.append(f"corrected model_size from {size!r} to '1.7B'")
        vllm_gpu = adv.get("vllm_gpu_memory_utilization")
        if vllm_gpu is not None and not (0.0 < vllm_gpu <= 1.0):
            corrected_adv["vllm_gpu_memory_utilization"] = 0.7
            issues.append(
                f"corrected vllm_gpu_memory_utilization from {vllm_gpu} to 0.7"
            )
        vllm_port = adv.get("vllm_port")
        if vllm_port is not None and not (
            isinstance(vllm_port, int) and 1024 <= vllm_port <= 65535
        ):
            corrected_adv["vllm_port"] = None
            issues.append(f"corrected vllm_port from {vllm_port} to None")
        if corrected_adv != adv:
            result["advanced"] = corrected_adv
    gen = config.get("generation")
    if isinstance(gen, dict):
        corrected_gen = dict(gen)
        temp = gen.get("temperature")
        if temp is not None and not (0.0 <= temp <= 2.0):
            corrected_gen["temperature"] = 0.7
            issues.append(f"corrected temperature from {temp} to 0.7")
        if corrected_gen != gen:
            result["generation"] = corrected_gen
    sec = config.get("security")
    if isinstance(sec, dict):
        corrected_sec = dict(sec)

        # Validate rate_limits section
        rate_limits = sec.get("rate_limits")
        if isinstance(rate_limits, dict):
            corrected_limits = {}
            for key, value in rate_limits.items():
                if isinstance(value, str) and _validate_rate_limit_string(value):
                    corrected_limits[key] = value
                else:
                    corrected_limits[key] = _get_default_rate_limit(key)
                    issues.append(
                        f"corrected rate_limits.{key} from {value!r} to {corrected_limits[key]!r}"
                    )
            corrected_sec["rate_limits"] = corrected_limits
        else:
            # Add default rate_limits if missing
            corrected_sec["rate_limits"] = {
                "generate": "20/minute",
                "model_ops": "3/minute",
                "transcribe": "15/minute",
                "prompt_ops": "10/minute",
                "config_ops": "1/minute",
            }
            issues.append("added default security.rate_limits section")

        # Existing security validations
        mtl = sec.get("max_text_length")
        if mtl is not None and (not isinstance(mtl, int) or mtl <= 0):
            corrected_sec["max_text_length"] = 10000
            issues.append(f"corrected max_text_length from {mtl} to 10000")
        if corrected_sec != sec:
            result["security"] = corrected_sec
    for issue in issues:
        logger.warning("Config validation: %s", sanitize_log(issue))
    return result, issues


def load_config() -> dict:
    """Load configuration from config.json with mtime-based caching.

    Returns a cached copy if config.json has not been modified since the last read.
    Thread-safe: uses lock to prevent concurrent reads/writes.

    Environment variable override:
        QWEN3_TTS_BACKEND: If set, overrides the backend setting in config.
                           Useful for testing in different environments.
    """
    with _config_lock:
        try:
            current_mtime = os.path.getmtime(CONFIG_PATH)
        except OSError:
            current_mtime = 0

        if (
            _config_cache["data"] is not None
            and current_mtime == _config_cache["mtime"]
        ):
            return copy.deepcopy(_config_cache["data"])

        try:
            with open(CONFIG_PATH) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"config.json is corrupt or invalid JSON: {e}\n"
                f"Run 'tts config' to reset, or fix {CONFIG_PATH} manually."
            ) from e
        _config_cache["mtime"] = current_mtime
        data, _ = validate_config(data)

        # Allow environment variable override for backend (useful for test runner)
        env_backend = os.environ.get("QWEN3_TTS_BACKEND")
        if env_backend and env_backend in VALID_BACKENDS:
            if "advanced" not in data:
                data["advanced"] = {}
            data["advanced"]["backend"] = env_backend

        _config_cache["data"] = data
        return copy.deepcopy(data)


def save_config(config: dict) -> None:
    """Save configuration to config.json and invalidate the cache.

    Writes atomically: serialize to a temp file in the same directory, fsync,
    then os.replace() onto the target (atomic on POSIX). This guarantees a crash
    mid-write leaves the previous config.json intact rather than truncated.
    Thread-safe: uses lock to prevent concurrent reads/writes.
    """
    with _config_lock:
        config_dir = os.path.dirname(CONFIG_PATH) or "."
        fd, tmp_path = tempfile.mkstemp(
            dir=config_dir, prefix=".config.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(config, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, CONFIG_PATH)
        except BaseException:
            # Leave the original config.json untouched; clean up the temp file.
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        _config_cache["data"] = None
        _config_cache["mtime"] = 0


def get_default_config(current_config: dict | None = None) -> dict:
    """Return a default config dict, preserving backend/model_size from current_config.

    Args:
        current_config: Existing config dict to preserve dynamic settings from.
            If None, platform defaults are used.

    Returns:
        A fresh default config dict.
    """
    current = current_config or {}
    return {
        "default_voice_description": "A calm, friendly male voice with clear articulation and moderate pace.",
        "default_clone_prompt": "default_clone.pt",
        "default_speaker": "ryan",
        "output_directory": "~/Downloads",
        "language": "English",
        "server": {
            "host": "127.0.0.1",
            "port": 5123,
            "auto_shutdown_minutes": 0,
        },
        "models": {
            "clone": {"load_at_startup": True},
            "design": {"load_at_startup": False},
            "custom": {"load_at_startup": False},
        },
        "security": {
            "max_text_length": 10000,
            "max_batch_size": 20,
        },
        "advanced": {
            "dtype": "bfloat16",
            "backend": current.get("advanced", {}).get("backend", "mlx"),
            "mlx_quantization": "8bit",
            "model_size": current.get("advanced", {}).get("model_size", "1.7B"),
            "torch_quantization": "none",
            "audio_loader": "torchaudio",
            "vllm_enabled": False,
            "vllm_fallback_to_torch": True,
        },
        "vllm": {
            "enabled": False,
            "fallback_to_torch": True,
            "max_model_len": 8192,
            "audio_sample_rate": 24000,
            "audio_chunk_size": 2000,
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 1,
            "mm_processor_name": "Qwen/Qwen2-Audio-7B-Instruct",
            "port": None,
            "dtype": "bfloat16",
        },
        "generation": {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "seed": None,
            "max_chunk_chars": 500,
            "max_new_tokens": 2048,
            "compile_model": True,
            "max_chunk_tokens": 200,
            "lufs_normalize": False,
            "lufs_target": -16.0,
            "silence_gap_seconds": 0.0,
        },
        "presets": {
            "consistent": {"temperature": 0.5, "top_k": 30, "seed": 42},
            "creative": {"temperature": 0.9, "top_p": 0.98},
        },
        "ui": {"port": 7860},
        "aliases": {
            "default": {"prompt": "default_clone.pt", "preset": "consistent"},
        },
        "cache": {
            "voice_prompt_max": 10,
            "generation_max": 5,
            "eta_ttl_seconds": 30,
        },
        "prosody_presets": {
            "excited": "Speak with excitement and high energy",
            "calm": "Speak in a calm, soothing, relaxed manner",
            "whisper": "Speak in a soft whisper",
            "authoritative": "Speak in a confident, authoritative tone",
            "slow": "Speak slowly and deliberately with clear enunciation",
            "fast": "Speak quickly with urgency",
            "dramatic": "Speak with dramatic flair and emotional intensity",
            "conversational": "Speak in a casual, natural conversational style",
        },
        "prompt_enhancer": {
            "enabled": False,
            "provider": "anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
            "model": "claude-haiku-4-5-20251001",
        },
    }


# ---------------------------------------------------------------------------
# ConfigLoader Protocol — enables dependency injection in tests and server
# ---------------------------------------------------------------------------


@runtime_checkable
class ConfigLoader(Protocol):
    """Protocol for config loaders — allows DI in tests and server."""

    def load(self) -> dict: ...


class DefaultConfigLoader:
    """Default implementation that reads from config.json on disk."""

    def load(self) -> dict:
        return load_config()


def get_default_clone_prompt(config: dict | None = None) -> str | None:
    """Return the default clone prompt filename.

    Reads from config's "default_clone_prompt" key. If missing or the file
    doesn't exist, falls back to the first prompt matching the current backend
    (.wav+.txt for MLX, .pt for torch/vllm). Returns None if no prompts are available.
    """
    if config is None:
        try:
            config = load_config()
        except (json.JSONDecodeError, ValueError, OSError):
            config = {}

    configured = config.get("default_clone_prompt")
    if configured:
        # Check it exists (as .pt or as MLX .wav/.txt pair)
        base = configured[:-3] if configured.endswith(".pt") else configured
        prompts_dir = str(VOICE_PROMPTS_DIR)
        pt_exists = os.path.exists(safe_path_join(prompts_dir, f"{base}.pt"))
        mlx_exists = os.path.exists(
            safe_path_join(prompts_dir, f"{base}.wav")
        ) and os.path.exists(safe_path_join(prompts_dir, f"{base}.txt"))
        if pt_exists or mlx_exists:
            return configured

    # Fallback: first prompt matching current backend
    backend = get_backend()
    try:
        all_files = os.listdir(VOICE_PROMPTS_DIR)
        if backend == "mlx":
            txt_bases = {f[:-4] for f in all_files if f.endswith(".txt")}
            for wav in sorted(f for f in all_files if f.endswith(".wav")):
                if wav[:-4] in txt_bases:
                    return wav
        else:
            prompts = sorted(f for f in all_files if f.endswith(".pt"))
            if prompts:
                return prompts[0]
    except OSError as e:
        logger.warning("Cannot scan voice_prompts dir %s: %s", VOICE_PROMPTS_DIR, e)

    return None


def set_default_clone_prompt(prompt_name: str, config: dict | None = None) -> None:
    """Set the default clone prompt in config.json."""
    if config is None:
        config = load_config()
    save_config({**config, "default_clone_prompt": prompt_name})


def get_device() -> str:
    """Return the PyTorch device string for this platform.

    No torch import — uses environment/platform detection only.
    Returns: 'cuda', 'mps', or 'cpu'
    """
    if IN_COLAB or IS_LINUX:
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None or os.path.exists(
            "/dev/nvidia0"
        ):
            return "cuda"
        return "cpu"
    if IS_MACOS and platform.machine() == "arm64":
        return "mps"
    return "cpu"


def get_cuda_capability() -> tuple[int, int] | None:
    """Return CUDA compute capability as (major, minor) or None if no CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()
    except ImportError:
        pass
    return None


def _has_flash_attn() -> bool:
    """Check if flash_attn package is importable."""
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def get_optimal_attn_config() -> tuple[str, str, bool]:
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


_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _validate_server_url(url: str) -> str:
    """Validate server URL -- allowlist approach for SSRF prevention.

    Only localhost variants are permitted as the TTS server always runs locally.
    Raises ValueError if the URL is considered invalid.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported server URL scheme: {parsed.scheme!r}")
    host = parsed.hostname or ""
    if host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"Server host {host!r} not allowed. "
            f"Valid: {', '.join(sorted(_ALLOWED_HOSTS))}"
        )
    port = parsed.port
    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"Invalid port: {port}")
    return url


def get_server_url(config: dict) -> str:
    """Return the server base URL from a config dict."""
    server = config.get("server", {})
    host = server.get("host", "127.0.0.1")
    port = server.get("port", 5123)
    # Bracket IPv6 addresses for valid URL syntax
    if ":" in host:
        url = f"http://[{host}]:{port}"
    else:
        url = f"http://{host}:{port}"
    return _validate_server_url(url)


def sanitize_log(value: Any) -> str:
    """Strip newlines and control characters for safe logging.

    Prevents log injection by replacing \\n and \\r with their escaped
    representations and removing null bytes. Non-string values are
    converted to str first (immutable — returns a new string).
    """
    s = str(value) if not isinstance(value, str) else value
    return s.replace("\n", "\\n").replace("\r", "\\r").replace("\x00", "")


def safe_path_join(base_dir: str, *parts: str) -> str:
    """Join paths safely, preventing directory traversal.

    Resolves the joined path to its real (canonical) location and verifies
    it is inside base_dir. Raises ValueError if traversal is detected.

    Returns:
        The resolved absolute path (immutable — returns a new string).
    """
    base_str = str(base_dir)  # Accept pathlib.Path
    joined = os.path.realpath(os.path.join(base_str, *parts))
    real_base = os.path.realpath(base_str)
    if not (joined == real_base or joined.startswith(real_base + os.sep)):
        raise ValueError("Path traversal detected")
    return joined


_VOICE_NAME_RE = re.compile(r"^[^/\\\x00]+$")


def validate_voice_name(name: str) -> str:
    """Validate a voice prompt name, rejecting path traversal and metacharacters.

    Raises ValueError for empty, too-long, or dangerous names.
    Returns the name unchanged if valid.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("voice name must be non-empty")
    if len(name) > 128 or ".." in name or not _VOICE_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid voice name: {name!r}")
    return name


def is_server_running(config_or_url: dict | str | None = None) -> bool:
    """Check whether the TTS server is reachable.

    Args:
        config_or_url: A config dict, a URL string, or None (loads config).
    """
    if config_or_url is None:
        config_or_url = load_config()

    if isinstance(config_or_url, str):
        url = _validate_server_url(config_or_url)
    else:
        url = get_server_url(config_or_url)

    try:
        import requests
    except ImportError:
        return False
    try:
        validated_url = _validate_server_url(url)
        resp = requests.get(f"{validated_url}/health", timeout=2)
        return resp.status_code in (200, 503)
    except (requests.RequestException, OSError):
        return False


# ---------------------------------------------------------------------------
# PID lifecycle helpers
# ---------------------------------------------------------------------------


def read_pid_file() -> int | None:
    """Read PID from .voice_server.pid. Returns None if missing/invalid."""
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None


def write_pid_file(pid: int) -> None:
    """Write PID to .voice_server.pid atomically via temp-file + os.replace()."""
    tmp = PID_FILE.with_suffix(".pid.tmp")
    tmp.write_text(str(pid))
    os.replace(tmp, PID_FILE)


def cleanup_pid_file() -> None:
    """Remove .voice_server.pid if it exists. Idempotent."""
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def find_pid_by_port(port: int) -> int | None:
    """Discover PID of process listening on a TCP port via lsof.
    Works on macOS and Linux. Returns int PID or None.
    """
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            first_line = result.stdout.strip().splitlines()[0]
            return int(first_line)
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None
    except ValueError:
        return None
    return None


def is_pid_alive(pid: int) -> bool:
    """Check if process exists via os.kill(pid, 0). Cross-platform."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Process exists but owned by another user
    except OSError:
        return False


def detect_server_state(config: dict | None = None) -> dict:
    """Unified server state combining health check + PID file + process liveness.

    Returns dict with keys:
        running (bool): True if server is definitely running
        health_ok (bool): True if /health responds
        pid (int|None): PID if known from file
        pid_alive (bool): True if PID process exists
        stale_pid (bool): True if PID file exists but process dead + health fails
    """
    health_ok = is_server_running(config)
    pid = read_pid_file()
    pid_alive = is_pid_alive(pid) if pid is not None else False

    running = health_ok  # Health check is authoritative
    stale_pid = pid is not None and not pid_alive and not health_ok

    return {
        "running": running,
        "health_ok": health_ok,
        "pid": pid,
        "pid_alive": pid_alive,
        "stale_pid": stale_pid,
    }


# ---------------------------------------------------------------------------
# Torch dtype configuration
# ---------------------------------------------------------------------------

VALID_DTYPES = ("float32", "float16", "bfloat16")
VALID_BACKENDS = ("torch", "mlx", "vllm")
VALID_MLX_QUANTIZATIONS = ("4bit", "5bit", "6bit", "8bit", "bf16")
VALID_TORCH_QUANTIZATIONS = ("none", "8bit", "4bit")
VALID_MODEL_SIZES = ("1.7B", "0.6B")


def get_torch_dtype_name() -> str:
    """Read the configured dtype from config.json (advanced.dtype).

    Returns:
        A string: "float32", "float16", or "bfloat16".
        Defaults to "float32" if not set or invalid.
    """
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
    try:
        config = load_config()
        quant = config.get("advanced", {}).get("torch_quantization", "none")
    except (json.JSONDecodeError, ValueError, OSError):
        quant = "none"
    if quant not in VALID_TORCH_QUANTIZATIONS:
        quant = "none"
    return quant


def get_vllm_gpu_util() -> float:
    """Read the configured vLLM GPU memory utilization from config.json.

    Returns:
        A float between 0.0 and 1.0 representing GPU memory fraction.
        Defaults to 0.7 if not set or invalid.
    """
    try:
        config = load_config()
        util = config.get("advanced", {}).get("vllm_gpu_memory_utilization", 0.7)
    except (json.JSONDecodeError, ValueError, OSError):
        util = 0.7
    if not isinstance(util, (int, float)) or not (0.0 < util <= 1.0):
        util = 0.7
    return float(util)


def get_vllm_port() -> int | None:
    """Read the configured vLLM port from config.json.

    Returns:
        An integer port number, or None for auto-find (default).
    """
    try:
        config = load_config()
        port = config.get("advanced", {}).get("vllm_port")
    except (json.JSONDecodeError, ValueError, OSError):
        port = None
    if port is not None:
        if not isinstance(port, int) or not (1024 <= port <= 65535):
            logger.warning("Invalid vllm_port %s, using auto-find", sanitize_log(port))
            port = None
    return port


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


def get_model_revision(model_type):
    """Return the HuggingFace revision (branch/tag/SHA) to download for a model type.

    Resolution order, falling back to "main" (current behavior — no change today):
      1. config.json: models.<model_type>.revision
      2. MODEL_INFO / MLX_MODEL_INFO entry's "revision" key
      3. "main"

    Pinning a specific SHA/tag here (via config or MODEL_INFO) lets a deployment
    avoid silently tracking a repo's moving ``main`` branch.
    """
    config = load_config()
    pinned = config.get("models", {}).get(model_type, {}).get("revision")
    if pinned:
        return pinned
    return get_model_info(model_type).get("revision") or "main"


# ---------------------------------------------------------------------------
# Prosody presets (instruct text templates for Custom & Design modes)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Generation presets (temperature/top_k/top_p/repetition_penalty)
# ---------------------------------------------------------------------------

DEFAULT_GENERATION_PRESETS = {
    "stable": {
        "temperature": 0.5,
        "top_k": 30,
        "top_p": 0.90,
        "repetition_penalty": 1.10,
    },
    "natural": {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
    },
    "expressive": {
        "temperature": 0.9,
        "top_k": 70,
        "top_p": 0.98,
        "repetition_penalty": 1.03,
    },
    "audiobook": {
        "temperature": 0.6,
        "top_k": 40,
        "top_p": 0.92,
        "repetition_penalty": 1.08,
    },
    "conversational": {
        "temperature": 0.8,
        "top_k": 60,
        "top_p": 0.97,
        "repetition_penalty": 1.04,
    },
    "broadcast": {
        "temperature": 0.55,
        "top_k": 35,
        "top_p": 0.91,
        "repetition_penalty": 1.09,
    },
    "dramatic": {
        "temperature": 1.0,
        "top_k": 80,
        "top_p": 0.99,
        "repetition_penalty": 1.02,
    },
    "whisper": {
        "temperature": 0.65,
        "top_k": 45,
        "top_p": 0.93,
        "repetition_penalty": 1.06,
    },
}


def get_generation_presets(config=None):
    """Return generation presets dict (defaults merged with user config).

    User presets in config.json override defaults with the same key.
    Additional user presets are added alongside defaults.

    Args:
        config: Optional pre-loaded config dict. If None, loads from disk.

    Returns:
        Dict mapping preset name -> parameter dict.
    """
    if config is None:
        try:
            config = load_config()
        except (json.JSONDecodeError, ValueError, OSError):
            config = {}
    user_presets = config.get("presets", {})
    return {**DEFAULT_GENERATION_PRESETS, **user_presets}


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
        except (json.JSONDecodeError, ValueError, OSError):
            config = {}
    user_presets = config.get("prosody_presets", {})
    return {**DEFAULT_PROSODY_PRESETS, **user_presets}


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


def read_auth_token():
    """Read the server auth token from TOKEN_FILE.

    Falls back to legacy path (~/.voice_server_token) with a deprecation warning.

    Returns:
        The token string, or None if file doesn't exist.
    """
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    # Backward compat: check legacy location
    if os.path.exists(_LEGACY_TOKEN_FILE):
        import logging

        logging.getLogger("tts").warning(
            "Reading auth token from legacy path %s — "
            "restart the server to migrate to %s",
            _LEGACY_TOKEN_FILE,
            TOKEN_FILE,
        )
        with open(_LEGACY_TOKEN_FILE) as f:
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
        html = (
            f'<span style="color:{color};font-weight:bold;">{self.user_message}</span>'
        )
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
            html += f"<br><em>{hint}</em>"
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
            technical_detail=detail
            or "Cannot authenticate. Run 'tts server start' to generate auth token.",
            recovery="restart",
        )


class VoicePromptError(TTSError):
    """Voice prompt operation failed."""

    def __init__(self, operation: str, detail=None):
        """Initialize VoicePromptError.

        Args:
            operation: Operation that failed (e.g., "delete", "rename", "preview")
            detail: Error details
        """
        self.operation = operation
        super().__init__(
            f"Voice prompt {operation} failed.",
            technical_detail=detail,
            recovery="retry" if operation in ("list", "preview") else "config",
        )


class ModelError(TTSError):
    """Model operation failed."""

    def __init__(self, model_type: str, operation: str, detail=None):
        """Initialize ModelError.

        Args:
            model_type: Model type (clone, design, custom)
            operation: Operation that failed (e.g., "load", "unload")
            detail: Error details
        """
        self.model_type = model_type
        self.operation = operation
        super().__init__(
            f"Model {operation} failed for {model_type}.",
            technical_detail=detail,
            recovery="restart",
        )
