"""
Base infrastructure for TTSClient: shared imports, helpers, decorator, and _ClientBase.

This module NEVER imports torch or qwen3_tts.core.engine — it communicates
exclusively over HTTP to the TTS server.
"""

import functools
import json

import requests

from qwen3_tts.core.config import (
    CONFIG_PATH,
    VOICE_PROMPTS_DIR,
    get_server_url,
    is_server_running,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_BUFFER_SIZE = 100 * 1024 * 1024  # 100MB - maximum buffer size for streaming


# ---------------------------------------------------------------------------
# Helper functions for code reuse
# ---------------------------------------------------------------------------

def _normalize_speaker_name(speaker):
    """Normalize speaker name to lowercase for consistent lookup.

    Args:
        speaker: Speaker name (e.g., "Ryan", "RYAN", "ryan") or None

    Returns:
        Lowercase speaker name or None if input is None
    """
    if speaker is None:
        return None
    return speaker.lower()


def _resolve_voice_alias(alias, prompt, mode, description, speaker, instruct, preset):
    """Resolve voice alias and merge with user-provided parameters.

    User-provided parameters take precedence over alias values.

    Args:
        alias: Dict from resolve_alias() or None
        prompt: User-provided prompt or None
        mode: User-provided mode or None
        description: User-provided description or None
        speaker: User-provided speaker or None
        instruct: User-provided instruct or None
        preset: User-provided preset or None

    Returns:
        Dict with resolved values for prompt, mode, description, speaker, instruct, preset
    """
    result = {
        "prompt": prompt,
        "mode": mode,
        "description": description,
        "speaker": speaker,
        "instruct": instruct,
        "preset": preset,
    }

    if alias:
        # Only use alias value if user didn't provide one
        if "prompt" in alias and result["prompt"] is None:
            result["prompt"] = alias["prompt"]
        if "preset" in alias and result["preset"] is None:
            result["preset"] = alias["preset"]
        if "mode" in alias:
            result["mode"] = alias["mode"]  # mode always comes from alias if present
        if "description" in alias and result["description"] is None:
            result["description"] = alias["description"]
        if "speaker" in alias and result["speaker"] is None:
            result["speaker"] = alias["speaker"]
        if "instruct" in alias and result["instruct"] is None:
            result["instruct"] = alias["instruct"]

    return result


def _build_gen_params(config, temperature, top_k, top_p, repetition_penalty, max_new_tokens, seed):
    """Build generation parameters dict from config and user overrides.

    User-provided values take precedence over config defaults.

    Args:
        config: Config dict with 'generation' key
        temperature: User-provided temperature or None
        top_k: User-provided top_k or None
        top_p: User-provided top_p or None
        repetition_penalty: User-provided repetition_penalty or None
        max_new_tokens: User-provided max_new_tokens or None
        seed: User-provided seed or None

    Returns:
        Dict with generation parameters
    """
    gen_config = config.get("generation", {})
    gen_params = {
        "temperature": temperature if temperature is not None else gen_config.get("temperature", 0.7),
        "top_k": top_k if top_k is not None else gen_config.get("top_k", 50),
        "top_p": top_p if top_p is not None else gen_config.get("top_p", 0.95),
        "repetition_penalty": repetition_penalty if repetition_penalty is not None else gen_config.get("repetition_penalty", 1.05),
        "max_new_tokens": max_new_tokens if max_new_tokens is not None else gen_config.get("max_new_tokens", 2048),
    }

    if seed is not None:
        gen_params["seed"] = seed
    elif gen_config.get("seed"):
        gen_params["seed"] = gen_config["seed"]

    return gen_params


def _extract_error_message(resp, default: str = "Unknown error") -> str:
    """Extract a human-readable error message from an HTTP error response."""
    try:
        data = resp.json()
        return data.get("error") or data.get("message") or data.get("detail") or default
    except (ValueError, requests.exceptions.JSONDecodeError):
        return f"Server returned HTTP {resp.status_code}"


def _require_server(func):
    """Decorator that checks server is running before method execution."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_server_running():
            raise ConnectionError(
                "TTS server is not running. Start it with: tts server start"
            )
        return func(self, *args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# _ClientBase class
# ---------------------------------------------------------------------------

class _ClientBase:
    """Base class providing shared state, config, and HTTP session."""

    def __init__(self, config_path=None, config_provider=None):
        """Initialize the TTS client.

        Args:
            config_path: Path to config.json. Defaults to ~/Qwen3-TTS_UserFiles/config.json
            config_provider: Optional ConfigProvider instance for dependency injection.
                           If provided, config_path is ignored.
        """
        self._config_provider = config_provider
        self.config_path = config_path or CONFIG_PATH
        self.voice_prompts_dir = VOICE_PROMPTS_DIR
        self._config = None
        self._session = requests.Session()

    def close(self):
        """Close the HTTP session and release connection pool."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def config(self):
        """Load and cache configuration."""
        if self._config_provider is not None:
            # Use config provider if available
            return {
                "generation": self._config_provider.get_generation_params(),
                "server": {"url": self._config_provider.get_server_url()},
            }
        if self._config is None:
            with open(self.config_path, "r") as f:
                self._config = json.load(f)
        return self._config

    def reload_config(self):
        """Force reload of configuration."""
        self._config = None
        return self.config

    @property
    def server_url(self):
        """Get the server URL from config."""
        if self._config_provider is not None:
            return self._config_provider.get_server_url()
        return get_server_url(self.config)

    def is_server_running(self):
        """Check if the TTS server is running."""
        return is_server_running(self.config)

    @staticmethod
    def _add_mode_params(
        payload: dict,
        mode: str,
        prompt=None,
        description=None,
        speaker=None,
        instruct=None,
        x_vector_only_mode: bool = False,
    ) -> None:
        """Mutate payload in-place with mode-specific generation parameters."""
        if mode == "clone":
            payload["prompt_file"] = prompt
            if x_vector_only_mode:
                payload["x_vector_only_mode"] = True
        elif mode == "custom":
            payload["speaker"] = speaker
            payload["instruct"] = instruct or ""
        else:
            payload["voice_description"] = description
