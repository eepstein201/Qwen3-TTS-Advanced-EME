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
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        ),
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
