#!/usr/bin/env python3
"""Config file I/O: validation, load/save, and the default config template.

No torch, numpy, or heavy imports.

References to names defined in sibling submodules (CONFIG_PATH, VALID_BACKENDS,
DEFAULT_GENERATION_PARAMS, sanitize_log, IS_MACOS) are resolved via a lazy
per-call import from ``qwen3_tts.core.config`` (the package facade), not a
static module-level import. This preserves the existing test seam where
``@patch("qwen3_tts.core.config.NAME", ...)`` patches the facade attribute —
a static ``from .paths import NAME`` would bind an independent copy in this
module's namespace that the patch would never reach.
"""

import copy
import json
import logging
import os
import platform
import tempfile
import threading
from typing import Any

logger = logging.getLogger("tts.config")

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
    if not isinstance(config, dict):
        raise ValueError(
            f"config must be a dict, got {type(config).__name__} instead"
        )

    from qwen3_tts.core.config import (
        DEFAULT_GENERATION_PARAMS,
        IS_MACOS,
        VALID_BACKENDS,
        sanitize_log,
    )

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
            corrected_gen["temperature"] = DEFAULT_GENERATION_PARAMS["temperature"]
            issues.append(
                f"corrected temperature from {temp} to "
                f"{DEFAULT_GENERATION_PARAMS['temperature']}"
            )
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
            corrected_sec["max_text_length"] = 50000
            issues.append(f"corrected max_text_length from {mtl} to 50000")
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
    from qwen3_tts.core.config import CONFIG_PATH, VALID_BACKENDS

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
        from qwen3_tts.core.config import validate_config as _validate_config

        try:
            data, _ = _validate_config(data)
        except (TypeError, ValueError) as e:
            # Mirror the corrupt-JSON path above: an unusable config is surfaced,
            # not silently swapped for defaults, so the user's real settings are
            # never quietly ignored.
            raise ValueError(
                f"config.json is not a valid config object: {e}\n"
                f"Run 'tts config' to reset, or fix {CONFIG_PATH} manually."
            ) from e

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
    from qwen3_tts.core.config import CONFIG_PATH

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
    from qwen3_tts.core.config import DEFAULT_GENERATION_PARAMS

    current = current_config or {}
    return {
        "default_voice_description": "A calm, friendly male voice with clear articulation and moderate pace.",
        # None, not a filename: no prompt ships with the package, so any seeded
        # name is a dangling reference. get_default_clone_prompt() treats a
        # falsy value and a missing file identically — both fall through to the
        # backend-aware scan for the first usable prompt on disk — so None is
        # behaviour-preserving and honest about shipping no default.
        "default_clone_prompt": None,
        "default_speaker": "ryan",
        "output_directory": "~/Downloads",
        # Web-UI generation history lives under its own parent so app-managed
        # files stay separable from the user's real downloads. Two fixed-name
        # subfolders are created beneath it: "Automated Output" (every web-UI
        # generation) and "Manual Downloads" (user-curated keepers). Distinct
        # from output_directory, which remains the CLI's save location.
        "history_output_directory": "~/Downloads/Qwen3-TTS Output",
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
            "max_text_length": 50000,
            "max_batch_size": 20,
        },
        "advanced": {
            "dtype": "bfloat16",
            "backend": current.get("advanced", {}).get("backend", "mlx"),
            "mlx_quantization": "8bit",
            "model_size": current.get("advanced", {}).get("model_size", "1.7B"),
            "torch_quantization": "none",
            "audio_loader": "torchaudio",
            "attn_implementation": "auto",
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
            # temperature/top_k/top_p/repetition_penalty come from the shared
            # DEFAULT_GENERATION_PARAMS constant (single source of truth).
            **DEFAULT_GENERATION_PARAMS,
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
        # Ship no aliases. The former seeded "default" alias pointed at
        # default_clone.pt, which does not exist in any install, and the alias
        # path (interface/generate.py: `alias_prompt or get_default_clone_prompt`)
        # short-circuits the missing-prompt fallback — so `tts --alias default`
        # raised FileNotFoundError on every fresh install. A .pt prompt is also
        # torch-only and wrong for the default MLX backend. Users create their
        # own aliases; an empty table cannot mislead.
        "aliases": {},
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
