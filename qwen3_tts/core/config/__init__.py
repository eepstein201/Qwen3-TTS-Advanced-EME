#!/usr/bin/env python3
"""Lightweight TTS configuration package — facade re-exporting all public names.

No torch, numpy, or heavy imports. This package replaces the monolithic
config.py (1473 lines, well over the project's 800-line limit) with 8
submodules. This __init__.py re-exports every public — and several
underscore-prefixed internal — names so that all existing consumers
(server, interface, tools, tests) continue to work unchanged via
``from qwen3_tts.core.config import X`` or ``qwen3_tts.core.config.X``.

Submodules:
    paths.py    — platform detection (IN_COLAB, IS_MACOS, IS_LINUX) +
                   path constants (CONFIG_PATH, VOICE_PROMPTS_DIR, PID_FILE,
                   LOG_FILE, TOKEN_FILE, HF_CACHE, ...)
    io.py       — config.json validation, load/save (atomic write), and the
                   default config template
    pid.py      — PID lifecycle helpers (read/write/cleanup, lsof lookup,
                   liveness, unified server-state detection)
    runtime.py  — ConfigLoader DI protocol, device/attention detection,
                   server URL validation, voice-prompt lookup, path/log/
                   voice-name safety helpers
    models.py   — model/backend/quantization getters, cache-size config,
                   canonical speaker + model metadata (MODEL_INFO, etc.)
    presets.py  — generation presets and prosody presets
    auth.py     — server auth token read helpers
    errors.py   — TTSError hierarchy

Why nearly every function-body import below reads ``from qwen3_tts.core.config
import NAME`` instead of a static ``from .sibling import NAME``:

This codebase's test suite patches config internals two ways:
  1. ``@patch("qwen3_tts.core.config.NAME", ...)`` / ``patch.object(cfg, "NAME", ...)``
     — mutates *this* module's (the facade's) attribute dict.
  2. ``@patch("qwen3_tts.core.config.os.listdir", ...)`` — resolves ``os`` as an
     attribute of this facade, then patches the real, shared ``os`` module
     object; this variant works regardless of which submodule performs the
     ``os.listdir`` call, and needs no special handling here beyond making
     ``os``/``subprocess``/``platform`` importable as facade attributes
     (done below).

Case 1 only works post-split if the *consuming function* re-reads the name
from this facade's namespace at call time — a static
``from .paths import PID_FILE`` at a sibling submodule's top would bind an
independent copy in that submodule's own globals, which
``patch.object(cfg, "PID_FILE", ...)`` (mutating only this module's dict)
would never reach. Every submodule function in this package that consumes a
name defined in a *different* submodule therefore does the lazy per-call
``from qwen3_tts.core.config import NAME`` import documented at the top of
each submodule — mirroring this codebase's existing external-consumer
convention (see CLAUDE.md) applied internally too.

Internal symbols (prefixed with _) are exported here as well, matching
qwen3_tts/core/engine/__init__.py's convention, because several are
themselves patched directly by tests (e.g. ``_has_flash_attn``,
``_resolve_config_path``).
"""

import copy  # noqa: F401 -- re-exported so dotted-attribute patches resolve
import json  # noqa: F401
import logging
import os  # noqa: F401
import pathlib  # noqa: F401
import platform  # noqa: F401
import re  # noqa: F401
import subprocess  # noqa: F401  # nosec B404  # re-exported for pid.py server PID management
import sys  # noqa: F401
import tempfile  # noqa: F401
import threading  # noqa: F401

# --- auth ---
from qwen3_tts.core.config.auth import (
    auth_headers,
    read_auth_token,
)

# --- errors ---
from qwen3_tts.core.config.errors import (
    AuthenticationError,
    GenerationError,
    InvalidInputError,
    ModelError,
    ModelNotLoadedError,
    ServerConnectionError,
    TTSError,
    VoicePromptError,
)

# --- io ---
from qwen3_tts.core.config.io import (
    _config_cache,  # noqa: F401
    _config_lock,  # noqa: F401
    _get_default_rate_limit,  # noqa: F401
    _validate_rate_limit_string,  # noqa: F401
    get_default_config,
    load_config,
    save_config,
    validate_config,
)

# --- models ---
from qwen3_tts.core.config.models import (
    CUSTOM_VOICE_SPEAKERS,
    DEFAULT_GENERATION_PARAMS,
    MLX_MODEL_INFO,
    MODEL_INFO,
    VALID_BACKENDS,
    VALID_DTYPES,
    VALID_MLX_QUANTIZATIONS,
    VALID_MODEL_SIZES,
    VALID_TORCH_QUANTIZATIONS,
    VOICE_DESCRIPTION_ATTRIBUTES,
    _get_config_value,  # noqa: F401
    get_backend,
    get_eta_cache_ttl,
    get_generation_cache_max,
    get_generation_defaults,
    get_mlx_model_name,
    get_mlx_quantization,
    get_model_info,
    get_model_revision,
    get_model_size,
    get_torch_dtype_name,
    get_torch_model_name,
    get_torch_quantization,
    get_voice_prompt_cache_max,
)

# --- paths ---
from qwen3_tts.core.config.paths import (
    _LEGACY_TOKEN_FILE,  # noqa: F401
    _TOKEN_DIR,  # noqa: F401
    CONFIG_PATH,
    HF_CACHE,
    HISTORY_FILE,
    IN_COLAB,
    IS_LINUX,
    IS_MACOS,
    LOCK_FILE,
    LOG_FILE,
    PID_FILE,
    TOKEN_FILE,
    USER_FILES_DIR,
    VOICE_PROMPTS_DIR,
    _resolve_config_path,  # noqa: F401
)

# --- pid ---
from qwen3_tts.core.config.pid import (
    cleanup_pid_file,
    detect_server_state,
    find_pid_by_port,
    is_pid_alive,
    read_pid_file,
    write_pid_file,
)

# --- presets ---
from qwen3_tts.core.config.presets import (
    DEFAULT_GENERATION_PRESETS,
    DEFAULT_PROSODY_PRESETS,
    get_generation_presets,
    get_prosody_presets,
)

# --- runtime ---
from qwen3_tts.core.config.runtime import (
    _ALLOWED_HOSTS,  # noqa: F401
    _VOICE_NAME_RE,  # noqa: F401
    ConfigLoader,
    DefaultConfigLoader,
    _has_flash_attn,  # noqa: F401
    _resolve_attn_implementation,
    _validate_server_url,  # noqa: F401
    get_cuda_capability,
    get_default_clone_prompt,
    get_device,
    get_optimal_attn_config,
    get_server_url,
    is_server_running,
    prompt_file_exists,
    safe_path_join,
    sanitize_log,
    set_default_clone_prompt,
    validate_voice_name,
)

from . import (
    auth,
    errors,
    io,
    models,
    paths,
    pid,
    presets,
    runtime,
)

logger = logging.getLogger("tts.config")

# Public API — all symbols without underscore prefix, plus submodules
__all__ = [
    # submodules
    "paths",
    "io",
    "pid",
    "runtime",
    "models",
    "presets",
    "auth",
    "errors",
    # paths
    "IN_COLAB",
    "IS_MACOS",
    "IS_LINUX",
    "USER_FILES_DIR",
    "CONFIG_PATH",
    "VOICE_PROMPTS_DIR",
    "HISTORY_FILE",
    "PID_FILE",
    "LOG_FILE",
    "LOCK_FILE",
    "TOKEN_FILE",
    "HF_CACHE",
    # io
    "validate_config",
    "load_config",
    "save_config",
    "get_default_config",
    # pid
    "read_pid_file",
    "write_pid_file",
    "cleanup_pid_file",
    "find_pid_by_port",
    "is_pid_alive",
    "detect_server_state",
    # runtime
    "ConfigLoader",
    "DefaultConfigLoader",
    "prompt_file_exists",
    "get_default_clone_prompt",
    "set_default_clone_prompt",
    "get_device",
    "get_cuda_capability",
    "get_optimal_attn_config",
    "_resolve_attn_implementation",
    "get_server_url",
    "sanitize_log",
    "safe_path_join",
    "validate_voice_name",
    "is_server_running",
    # models
    "VALID_DTYPES",
    "VALID_BACKENDS",
    "VALID_MLX_QUANTIZATIONS",
    "VALID_TORCH_QUANTIZATIONS",
    "VALID_MODEL_SIZES",
    "DEFAULT_GENERATION_PARAMS",
    "get_torch_dtype_name",
    "get_backend",
    "get_mlx_quantization",
    "get_model_size",
    "get_generation_defaults",
    "get_torch_quantization",
    "get_voice_prompt_cache_max",
    "get_generation_cache_max",
    "get_eta_cache_ttl",
    "CUSTOM_VOICE_SPEAKERS",
    "MODEL_INFO",
    "MLX_MODEL_INFO",
    "get_mlx_model_name",
    "get_torch_model_name",
    "get_model_info",
    "get_model_revision",
    "VOICE_DESCRIPTION_ATTRIBUTES",
    # presets
    "DEFAULT_GENERATION_PRESETS",
    "get_generation_presets",
    "DEFAULT_PROSODY_PRESETS",
    "get_prosody_presets",
    # auth
    "read_auth_token",
    "auth_headers",
    # errors
    "TTSError",
    "ServerConnectionError",
    "ModelNotLoadedError",
    "InvalidInputError",
    "GenerationError",
    "AuthenticationError",
    "VoicePromptError",
    "ModelError",
]
