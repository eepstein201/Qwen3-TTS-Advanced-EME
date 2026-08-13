#!/usr/bin/env python3
"""Runtime helpers: config loader DI, device/attention detection, server URL,
voice prompt lookup, and path/log/voice-name safety helpers.

No torch, numpy, or heavy imports (torch is imported lazily inside
get_cuda_capability() only, guarded by ImportError).

References to names defined in sibling submodules (VOICE_PROMPTS_DIR,
IN_COLAB, IS_LINUX, IS_MACOS, load_config, save_config, get_backend,
_has_flash_attn) are resolved via a lazy per-call import from
``qwen3_tts.core.config`` (the package facade), not a static module-level
import. This preserves the existing test seam where
``@patch("qwen3_tts.core.config.NAME", ...)`` patches the facade attribute —
a static import would bind an independent copy in this module's namespace
that the patch would never reach. See qwen3_tts/core/config/__init__.py.
"""

import json
import logging
import os
import platform
import re
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

logger = logging.getLogger("tts.config")


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
        from qwen3_tts.core.config import load_config

        return load_config()


def prompt_file_exists(name: str | None) -> bool:
    """True if *name* resolves to a usable voice prompt on disk.

    A prompt is usable as ``<base>.pt`` (torch) or as the ``<base>.wav`` +
    ``<base>.txt`` pair (MLX). Accepts all three spellings a prompt name
    travels in — a bare base, a ``.pt``, or a ``.wav`` — because callers get
    names from config, from ``--prompt``, and from this module's own scan,
    which returns the ``.wav`` filename.

    Extracted from get_default_clone_prompt() so the CLI's explicit-prompt
    check cannot drift from the default-prompt check (repo-audit-2026-07-31
    P1-2's lesson applied to P1-3's fix).
    """
    from qwen3_tts.core.config import VOICE_PROMPTS_DIR

    if not name:
        return False
    base = name
    for suffix in (".pt", ".wav"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    prompts_dir = str(VOICE_PROMPTS_DIR)
    pt_exists = os.path.exists(safe_path_join(prompts_dir, f"{base}.pt"))
    mlx_exists = os.path.exists(
        safe_path_join(prompts_dir, f"{base}.wav")
    ) and os.path.exists(safe_path_join(prompts_dir, f"{base}.txt"))
    return pt_exists or mlx_exists


def get_default_clone_prompt(config: dict | None = None) -> str | None:
    """Return the default clone prompt filename.

    Reads from config's "default_clone_prompt" key. If missing or the file
    doesn't exist, falls back to the first prompt matching the current backend
    (.wav+.txt for MLX, .pt for torch/vllm). Returns None if no prompts are available.
    """
    from qwen3_tts.core.config import VOICE_PROMPTS_DIR, get_backend, load_config
    from qwen3_tts.core.config import prompt_file_exists as _prompt_file_exists

    if config is None:
        try:
            config = load_config()
        except (json.JSONDecodeError, ValueError, OSError):
            config = {}

    configured = config.get("default_clone_prompt")
    if configured and _prompt_file_exists(configured):
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
    from qwen3_tts.core.config import load_config, save_config

    if config is None:
        config = load_config()
    save_config({**config, "default_clone_prompt": prompt_name})


def get_device() -> str:
    """Return the PyTorch device string for this platform.

    No torch import — uses environment/platform detection only.
    Returns: 'cuda', 'mps', or 'cpu'
    """
    from qwen3_tts.core.config import IN_COLAB, IS_LINUX, IS_MACOS

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


def _resolve_attn_implementation(preference, has_flash: bool) -> str:
    """Pick the torch attention implementation from a user preference.

    SDPA is the safe default — upstream #333 reports NaN logits with
    flash_attention_2 for Qwen3-TTS on L4/A100. FA2 is opt-in only and
    silently falls back to SDPA when flash_attn isn't installed.
    """
    pref = (preference or "auto").lower()
    if pref in ("flash_attention_2", "fa2", "flash"):
        return "flash_attention_2" if has_flash else "sdpa"
    if pref == "eager":
        return "eager"
    return "sdpa"  # auto / sdpa / unknown → safe default


def get_optimal_attn_config(preference: str = "auto") -> tuple[str, str, bool]:
    """Return (attn_implementation, torch_dtype_name, load_in_8bit) based on GPU.

    SDPA is the default everywhere; flash_attention_2 is opt-in via
    ``preference`` and only honoured on Ampere+ with flash_attn installed.
    Turing (T4, CC 7.5): sdpa, float16, True (8-bit via bitsandbytes)
    Ampere+ (L4/A100, CC >= 8.0): sdpa by default, bfloat16, False
    Non-CUDA: sdpa, float32, False
    """
    from qwen3_tts.core.config import _has_flash_attn as has_flash_attn
    from qwen3_tts.core.config import get_cuda_capability as get_cap

    cap = get_cap()
    if cap is None:
        return "sdpa", "float32", False
    if cap[0] >= 8:
        attn = _resolve_attn_implementation(preference, has_flash_attn())
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


def sanitize_log(value) -> str:
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
    from qwen3_tts.core.config import load_config

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
        # A 429 (rate-limited) still proves the server process is up and
        # answering; treating it as "down" makes the Gradio UI show
        # "Disconnected / Server not running" whenever the global limiter
        # trips, and misleads `tts server restart` (which gates the stop on
        # this check).
        return resp.status_code in (200, 503, 429)
    except (requests.RequestException, OSError):
        return False
