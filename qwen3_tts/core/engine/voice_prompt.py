#!/usr/bin/env python3
"""Voice prompt loading, caching, and migration.

Imports from: config, audio_processing.
Uses lazy imports for model_loader and inference to avoid circular dependencies.
"""

import logging
import os
import threading
from collections import OrderedDict

from qwen3_tts.core.config import (
    VOICE_PROMPTS_DIR,
    get_backend,
    get_voice_prompt_cache_max,
)
from qwen3_tts.core.engine.audio_processing import load_audio_for_cloning

logger = logging.getLogger("tts.engine")


def _evict_if_full(cache: OrderedDict, max_size: int) -> None:
    """Remove oldest entry from LRU cache if at or over capacity."""
    if len(cache) >= max_size:
        cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Voice prompt cache (torch backend)
# ---------------------------------------------------------------------------

_torch_prompt_cache = OrderedDict()
_torch_prompt_cache_lock = threading.Lock()
_torch_prompt_cache_hits = 0
_torch_prompt_cache_misses = 0


def _store_in_torch_cache(prompt_file: str, result) -> None:
    """Insert result into the LRU cache, evicting oldest entry if full."""
    global _torch_prompt_cache_misses
    with _torch_prompt_cache_lock:
        _evict_if_full(_torch_prompt_cache, get_voice_prompt_cache_max())
        _torch_prompt_cache[prompt_file] = result
        _torch_prompt_cache_misses += 1


def _auto_create_pt_from_wav(base_name: str, wav_path: str, txt_path: str,
                              prompt_path: str, prompt_file: str):
    """Create a .pt voice prompt from .wav + .txt files and cache the result.

    Returns the voice_prompt tensor, or None if wav_path does not exist.
    """
    import torch
    if not os.path.exists(wav_path):
        return None
    logger.info("Auto-creating .pt from .wav for %s", base_name)
    ref_audio, ref_sr = load_audio_for_cloning(wav_path)
    transcript = ""
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            transcript = f.read().strip()
    if not transcript:
        logger.warning("No transcript for %s, using empty string", base_name)
    from qwen3_tts.core.engine.model_loader import load_model
    from qwen3_tts.core.engine.inference import create_voice_prompt
    model = load_model("clone")
    voice_prompt = create_voice_prompt(model, ref_audio, ref_sr, transcript)
    torch.save(voice_prompt, prompt_path)
    logger.info("Auto-created and saved %s", prompt_path)
    _store_in_torch_cache(prompt_file, voice_prompt)
    return voice_prompt


def _load_pt_safe(prompt_path: str, prompt_file: str, device: str):
    """Load a .pt file with weights_only=True (safe deserialization only)."""
    import torch
    try:
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
        torch.serialization.add_safe_globals([VoiceClonePromptItem])
    except ImportError:
        pass
    try:
        result = torch.load(prompt_path, weights_only=True, map_location=device)
        _store_in_torch_cache(prompt_file, result)
        return result
    except (RuntimeError, ValueError, TypeError):
        # Only unpickling/weights_only failures reach here.
        # PermissionError, MemoryError, OSError propagate normally.
        real_prompt = os.path.realpath(prompt_path)
        real_prompts_dir = os.path.realpath(VOICE_PROMPTS_DIR)
        if not real_prompt.startswith(real_prompts_dir + os.sep):
            raise ValueError(f"Refusing to load {prompt_file}: outside voice_prompts/ directory")
        raise RuntimeError(
            f"Cannot load {prompt_file} safely. "
            f"Re-create with 'tts voice create'."
        )


def _load_voice_prompt_torch(prompt_file):
    """Load and cache a .pt voice prompt (torch backend).

    If the .pt file doesn't exist but .wav + .txt files do, auto-creates
    the .pt using the already-loaded clone model (avoids loading a second model).

    Results are cached (up to cache.voice_prompt_max entries, default 10) for
    repeated lookups. Cache is config-aware and respects the voice_prompt_max setting.
    """
    global _torch_prompt_cache_hits
    import torch  # noqa: F401 — needed for cache type

    with _torch_prompt_cache_lock:
        if prompt_file in _torch_prompt_cache:
            _torch_prompt_cache.move_to_end(prompt_file)
            _torch_prompt_cache_hits += 1
            return _torch_prompt_cache[prompt_file]

    prompt_path = os.path.join(VOICE_PROMPTS_DIR, prompt_file)
    if not os.path.exists(prompt_path):
        base_name = prompt_file[:-3] if prompt_file.endswith('.pt') else prompt_file
        wav_path = os.path.join(VOICE_PROMPTS_DIR, f"{base_name}.wav")
        txt_path = os.path.join(VOICE_PROMPTS_DIR, f"{base_name}.txt")
        return _auto_create_pt_from_wav(base_name, wav_path, txt_path, prompt_path, prompt_file)

    from qwen3_tts.core.config import get_device
    return _load_pt_safe(prompt_path, prompt_file, get_device())


def load_voice_prompt(prompt_file):
    """Load a voice prompt, dispatching to the correct format for the backend.

    - torch backend: loads .pt tensor file
    - mlx backend: loads .wav + .txt file pair as a dict
    """
    backend = get_backend()
    if backend == "mlx":
        return load_voice_prompt_mlx(prompt_file)
    return _load_voice_prompt_torch(prompt_file)


def clear_voice_prompt_cache():
    """Clear both torch and MLX voice prompt caches."""
    with _torch_prompt_cache_lock:
        _torch_prompt_cache.clear()
    with _mlx_prompt_cache_lock:
        _mlx_prompt_cache.clear()


def voice_prompt_cache_info():
    """Return cache statistics for the active backend.

    For torch: returns simple namespace with currsize/maxsize (manual OrderedDict cache).
    For mlx: returns a simple namespace with hits/currsize.
    """
    from types import SimpleNamespace
    backend = get_backend()
    if backend == "mlx":
        with _mlx_prompt_cache_lock:
            return SimpleNamespace(
                currsize=len(_mlx_prompt_cache),
                hits=0,  # dict cache doesn't track hits
                misses=0,
                maxsize=get_voice_prompt_cache_max(),
            )
    with _torch_prompt_cache_lock:
        return SimpleNamespace(
            currsize=len(_torch_prompt_cache),
            hits=_torch_prompt_cache_hits,
            misses=_torch_prompt_cache_misses,
            maxsize=get_voice_prompt_cache_max(),
        )


def migrate_orphan_mlx_prompts(clone_model=None):
    """Scan voice_prompts/ for .wav+.txt without .pt and auto-create .pt files.

    Supports users migrating from Mac (MLX) to Colab (PyTorch).

    Args:
        clone_model: Optional pre-loaded clone model. If None, loads on demand.
    """
    import glob

    wav_files = glob.glob(os.path.join(VOICE_PROMPTS_DIR, "*.wav"))
    migrated = 0
    model = None  # Lazy: only load if migration needed and no model passed
    for wav_path in wav_files:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        pt_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.pt")
        if not os.path.exists(pt_path):
            logger.info("Migrating orphan MLX prompt: %s", base)
            try:
                if model is None:
                    # Lazy import to avoid circular dependency
                    from qwen3_tts.core.engine.model_loader import load_model
                    model = clone_model or load_model("clone")
                ref_audio, ref_sr = load_audio_for_cloning(wav_path)
                transcript = ""
                txt_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path, "r") as f:
                        transcript = f.read().strip()
                if not transcript:
                    logger.warning("No transcript for %s, using empty string", base)
                # Lazy import to avoid circular dependency
                from qwen3_tts.core.engine.inference import create_voice_prompt
                voice_prompt = create_voice_prompt(model, ref_audio, ref_sr, transcript)
                import torch  # lazy: only needed when saving .pt
                torch.save(voice_prompt, pt_path)
                logger.info("Auto-created and saved %s", pt_path)
                migrated += 1
            except Exception as e:
                logger.warning("Failed to migrate prompt '%s': %s", base, e)
    if migrated:
        logger.info("Migrated %d orphan MLX prompt(s) to .pt format", migrated)
    return migrated


# ---------------------------------------------------------------------------
# MLX voice prompt loading
# ---------------------------------------------------------------------------

_mlx_prompt_cache = OrderedDict()
_mlx_prompt_cache_lock = threading.Lock()


def load_voice_prompt_mlx(prompt_name):
    """Load an MLX-compatible voice prompt (wav + txt file pair).

    Looks for <prompt_name>.wav and <prompt_name>.txt in VOICE_PROMPTS_DIR.
    Returns a dict with 'ref_audio' (path) and 'ref_text' (string) keys.
    Results are cached (up to 10 entries) for repeated lookups.

    Args:
        prompt_name: Base name with or without .pt extension.
                     E.g. "my_voice" or "my_voice.pt" — the .pt is stripped.
    """
    # Check cache first (move to end on hit for LRU eviction)
    with _mlx_prompt_cache_lock:
        if prompt_name in _mlx_prompt_cache:
            _mlx_prompt_cache.move_to_end(prompt_name)
            return _mlx_prompt_cache[prompt_name]

    # Strip known extensions to get the base name
    base = prompt_name
    if base.endswith(".pt"):
        base = base[:-3]
    elif base.endswith(".wav"):
        base = base[:-4]

    wav_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.wav")
    txt_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.txt")

    if not os.path.exists(wav_path) or not os.path.exists(txt_path):
        # Check if a .pt file exists (torch-only prompt)
        pt_path = os.path.join(VOICE_PROMPTS_DIR, f"{base}.pt")
        if os.path.exists(pt_path):
            raise FileNotFoundError(
                f"Voice prompt '{base}' only has a .pt file (torch format). "
                f"The MLX backend requires .wav and .txt files. "
                f"Re-create the prompt with 'tts voice create' to generate "
                f"MLX-compatible files."
            )
        raise FileNotFoundError(
            f"Voice prompt not found: looked for {wav_path} and {txt_path}"
        )

    with open(txt_path, "r") as f:
        ref_text = f.read().strip()

    result = {"ref_audio": wav_path, "ref_text": ref_text}

    # Cache the result with thread-safe lock
    with _mlx_prompt_cache_lock:
        _evict_if_full(_mlx_prompt_cache, get_voice_prompt_cache_max())
        _mlx_prompt_cache[prompt_name] = result

    return result
