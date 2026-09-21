#!/usr/bin/env python3
"""Generation presets (temperature/top_k/top_p/repetition_penalty) and
prosody presets (instruct text templates for Custom & Design modes).

No torch, numpy, or heavy imports.

``load_config`` (defined in io.py) is resolved via a lazy per-call import
from ``qwen3_tts.core.config`` (the package facade) — see
qwen3_tts/core/config/__init__.py for the rationale.
"""

import json
import re

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
    from qwen3_tts.core.config import load_config

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
    from qwen3_tts.core.config import load_config

    if config is None:
        try:
            config = load_config()
        except (json.JSONDecodeError, ValueError, OSError):
            config = {}
    user_presets = config.get("prosody_presets", {})
    return {**DEFAULT_PROSODY_PRESETS, **user_presets}


# ---------------------------------------------------------------------------
# User-defined prosody presets (Custom-tab save/delete; factory names reserved)
# ---------------------------------------------------------------------------

PROSODY_NAME_MAX_LEN = 40

# Factory names are reserved case-insensitively: a hand-edited factory-keyed
# override still applies through get_prosody_presets() (config wins) but is
# never listed or manageable as a user preset.
_FACTORY_PROSODY_NAMES = frozenset(k.lower() for k in DEFAULT_PROSODY_PRESETS)

# "(none)" and " - " are defence-in-depth rejections — the charset below
# already excludes both. ".." is deliberately allowed: a preset name is a
# config key, never a path component.
_PROSODY_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_EMPTY_INSTRUCT_MSG = (
    "Type a Style Instruction first — the preset saves exactly what's in that box."
)
_CORRUPT_SAVE_MSG = "Couldn't save the preset — config.json is corrupt or unreadable."
_CORRUPT_DELETE_MSG = (
    "Couldn't delete the preset — config.json is corrupt or unreadable."
)
_MISSING_CONFIG_MSG = "config.json is missing — run tts config to create it."


def _raw_user_prosody_entries(config):
    """The RAW prosody_presets dict, unfiltered.

    This is the WRITE base — never the filtered view: a filtered-base save
    would silently drop seeded factory entries and hand-edited
    factory-shadowed overrides from config.json.
    """
    entries = config.get("prosody_presets", {})
    return entries if isinstance(entries, dict) else {}


def _stripped(value):
    return value.strip() if isinstance(value, str) else ""


def get_user_prosody_presets(config=None):
    """Filtered view of user-defined prosody presets: factory names dropped
    (case-insensitive), str values only, non-empty after strip.

    Pure read — a corrupt or unreadable config swallows to {} so dropdown
    rendering never explodes. Returns a new dict, never an alias.
    """
    from qwen3_tts.core.config import load_config

    if config is None:
        try:
            config = load_config()
        except (json.JSONDecodeError, ValueError, OSError):
            return {}
    raw = _raw_user_prosody_entries(config)
    return {
        name: text
        for name, text in raw.items()
        if isinstance(name, str)
        and name.lower() not in _FACTORY_PROSODY_NAMES
        and isinstance(text, str)
        and text.strip()
    }


def validate_prosody_preset_name(name):
    """Return an error message for an invalid preset name, else None."""
    stripped = _stripped(name)
    if not stripped:
        return "Enter a preset name."
    if len(stripped) > PROSODY_NAME_MAX_LEN:
        return f"Preset name is too long ({PROSODY_NAME_MAX_LEN} characters max)."
    if stripped.lower() in _FACTORY_PROSODY_NAMES:
        return f"'{stripped}' is a built-in preset — pick a different name."
    if not _PROSODY_NAME_RE.match(stripped):
        return (
            "Preset names can only contain letters, numbers, dashes, "
            "underscores, and dots."
        )
    return None


def save_user_prosody_preset(name, instruct_text, config=None):
    """Save (or overwrite) a user-defined prosody preset.

    Strips name and text, stores stripped forms; the write base is the RAW
    entries dict. Swallow-on-corrupt is for pure reads only — a write-path
    load failure returns (False, msg) and never saves. OSError messages
    surface err.strerror only (never the path, CWE-209).
    """
    from qwen3_tts.core.config import load_config, save_config

    name = _stripped(name)
    text = _stripped(instruct_text)
    error = validate_prosody_preset_name(name)
    if error:
        return (False, error)
    if not text:
        return (False, _EMPTY_INSTRUCT_MSG)
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            return (False, _MISSING_CONFIG_MSG)
        except (json.JSONDecodeError, ValueError, OSError):
            return (False, _CORRUPT_SAVE_MSG)
    presets = _raw_user_prosody_entries(config)
    updated = name in presets and name.lower() not in _FACTORY_PROSODY_NAMES
    try:
        save_config({**config, "prosody_presets": {**presets, name: text}})
    except OSError as err:
        return (
            False,
            f"Couldn't save prosody preset '{name}' — the config file couldn't "
            f"be written ({err.strerror or 'unknown error'}). Check disk space "
            "and permissions, then try again.",
        )
    if updated:
        return (True, f"Updated preset '{name}'.")
    message = f"Saved preset '{name}'. It now appears in the Style Preset dropdown."
    if len(text) > 200:
        message += f" ({len(text)} characters, saved verbatim)."
    return (True, message)


def delete_user_prosody_preset(name, config=None):
    """Delete a user-defined prosody preset.

    Membership is tested against the RAW entries minus factory names —
    hand-edited junk entries stay deletable while never being listed.
    """
    from qwen3_tts.core.config import load_config, save_config

    name = _stripped(name)
    if name.lower() in _FACTORY_PROSODY_NAMES:
        return (False, f"'{name}' is a built-in preset and can't be deleted.")
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            return (False, _MISSING_CONFIG_MSG)
        except (json.JSONDecodeError, ValueError, OSError):
            return (False, _CORRUPT_DELETE_MSG)
    presets = _raw_user_prosody_entries(config)
    if name not in presets:
        return (
            False,
            f"Preset '{name}' no longer exists — it may have been removed "
            "outside the UI.",
        )
    try:
        save_config(
            {
                **config,
                "prosody_presets": {k: v for k, v in presets.items() if k != name},
            }
        )
    except OSError as err:
        return (
            False,
            f"Couldn't delete prosody preset '{name}' — the config file couldn't "
            f"be written ({err.strerror or 'unknown error'}). Check disk space "
            "and permissions, then try again.",
        )
    return (True, f"Deleted preset '{name}'.")
