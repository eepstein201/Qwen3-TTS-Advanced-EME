#!/usr/bin/env python3
"""Platform detection and path constants for the TTS configuration package.

No torch, numpy, or heavy imports. This module is the single source of
truth for filesystem locations (config.json, voice prompts, PID/log/token
files) and platform-detection flags (IN_COLAB, IS_MACOS, IS_LINUX).

Consumers elsewhere in this package read these constants via a lazy
per-call import from ``qwen3_tts.core.config`` (the package facade) rather
than a static module-level import, so that tests which patch
``qwen3_tts.core.config.CONSTANT`` (the facade attribute) are observed at
call time. See qwen3_tts/core/config/__init__.py for the rationale.
"""

import os
import pathlib
import platform
import sys

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


# Runtime config (Track T4). CONFIG_PATH is where config LIVES and is always
# the write target — like TOKEN_FILE, it sits outside the checkout so user-named
# values (presets) never land in a version-controlled file. Reads fall back
# through the legacy UserFiles location, then the tracked repo-root default
# (CI checkout / source tree; anchored on __file__, not on any external input).
# See io.resolve_config_read_path(). No environment-variable override is used:
# feeding an env var into open() is a path-injection source (CodeQL
# py/path-injection), and __file__ anchoring fixes CI without it.
_CONFIG_DIR = pathlib.Path(os.path.expanduser("~/.config/qwen3-tts"))
CONFIG_PATH = str(_CONFIG_DIR / "config.json")
_LEGACY_CONFIG_PATH = os.path.join(USER_FILES_DIR, "config.json")
_REPO_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "config.json",
)
VOICE_PROMPTS_DIR = pathlib.Path(USER_FILES_DIR) / "voice_prompts"
HISTORY_FILE = os.path.expanduser("~/.voice_history.jsonl")
PID_FILE = pathlib.Path(os.path.join(USER_FILES_DIR, ".voice_server.pid"))
LOG_FILE = pathlib.Path(os.path.join(USER_FILES_DIR, ".voice_server.log"))
LOCK_FILE = pathlib.Path(os.path.join(USER_FILES_DIR, ".voice_server.lock"))
_TOKEN_DIR = pathlib.Path(os.path.expanduser("~/.config/qwen3-tts"))
TOKEN_FILE = _TOKEN_DIR / ".voice_server_token"
_LEGACY_TOKEN_FILE = pathlib.Path(os.path.expanduser("~/.voice_server_token"))

# HuggingFace cache location (single source of truth)
HF_CACHE = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
