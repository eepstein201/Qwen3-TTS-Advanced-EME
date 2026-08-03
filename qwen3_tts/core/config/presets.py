#!/usr/bin/env python3
"""Generation presets (temperature/top_k/top_p/repetition_penalty) and
prosody presets (instruct text templates for Custom & Design modes).

No torch, numpy, or heavy imports.

``load_config`` (defined in io.py) is resolved via a lazy per-call import
from ``qwen3_tts.core.config`` (the package facade) — see
qwen3_tts/core/config/__init__.py for the rationale.
"""

import json

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
