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

# ---------------------------------------------------------------------------
# Voice prompt cache (torch backend)
# ---------------------------------------------------------------------------

_torch_prompt_cache = OrderedDict()
_torch_prompt_cache_lock = threading.Lock()
_torch_prompt_cache_hits = 0
_torch_prompt_cache_misses = 0


def _load_voice_prompt_torch(prompt_file):
    """Load and cache a .pt voice prompt (torch backend).

    If the .pt file doesn't exist but .wav + .txt files do, auto-creates
    the .pt using the already-loaded clone model (avoids loading a second model).

    Results are cached (up to cache.voice_prompt_max entries, default 10) for
    repeated lookups. Cache is config-aware and respects the voice_prompt_max setting.
    """
    global _torch_prompt_cache_hits, _torch_prompt_cache_misses
    import torch

    # Check cache first (move to end on hit for LRU eviction)
    with _torch_prompt_cache_lock:
        if prompt_file in _torch_prompt_cache:
            _torch_prompt_cache.move_to_end(prompt_file)
            _torch_prompt_cache_hits += 1
            return _torch_prompt_cache[prompt_file]

    prompt_path = os.path.join(VOICE_PROMPTS_DIR, prompt_file)
    if not os.path.exists(prompt_path):
        # Try to auto-create .pt from .wav + .txt
        base_name = prompt_file[:-3] if prompt_file.endswith('.pt') else prompt_file
        wav_path = os.path.join(VOICE_PROMPTS_DIR, f"{base_name}.wav")
        txt_path = os.path.join(VOICE_PROMPTS_DIR, f"{base_name}.txt")
        if os.path.exists(wav_path):
            logger.info("Auto-creating .pt from .wav for %s", base_name)
            ref_audio, ref_sr = load_audio_for_cloning(wav_path)
            transcript = ""
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    transcript = f.read().strip()
            if not transcript:
                logger.warning("No transcript for %s, using empty string", base_name)
            # Lazy import to avoid circular dependency
            from qwen3_tts.core.engine.model_loader import load_model
            from qwen3_tts.core.engine.inference import create_voice_prompt
            model = load_model("clone")
            voice_prompt = create_voice_prompt(model, ref_audio, ref_sr, transcript)
            torch.save(voice_prompt, prompt_path)
            logger.info("Auto-created and saved %s", prompt_path)
            # Cache the result
            with _torch_prompt_cache_lock:
                max_size = get_voice_prompt_cache_max()
                if len(_torch_prompt_cache) >= max_size:
                    _torch_prompt_cache.popitem(last=False)
                _torch_prompt_cache[prompt_file] = voice_prompt
                _torch_prompt_cache_misses += 1
            return voice_prompt
        return None
    from qwen3_tts.core.config import get_device
    device = get_device()
    # Register VoiceClonePromptItem as a safe global so torch.load(weights_only=True)
    # can deserialize .pt files containing this class (PyTorch 2.6+ requirement).
    try:
        from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
        torch.serialization.add_safe_globals([VoiceClonePromptItem])
    except ImportError:
        pass  # qwen_tts not installed — fall through to exception handler
    try:
        result = torch.load(prompt_path, weights_only=True, map_location=device)
        # Cache the result
        with _torch_prompt_cache_lock:
            max_size = get_voice_prompt_cache_max()
            if len(_torch_prompt_cache) >= max_size:
                _torch_prompt_cache.popitem(last=False)
            _torch_prompt_cache[prompt_file] = result
            _torch_prompt_cache_misses += 1
        return result
    except Exception:
        allow_unsafe = os.environ.get("TTS_ALLOW_UNSAFE_PICKLE") == "1"
        real_prompt = os.path.realpath(prompt_path)
        real_prompts_dir = os.path.realpath(VOICE_PROMPTS_DIR)
        if not real_prompt.startswith(real_prompts_dir + os.sep):
            raise ValueError(
                f"Refusing to load {prompt_file}: outside voice_prompts/ directory"
            )
        if not allow_unsafe:
            raise RuntimeError(
                f"Cannot load {prompt_file} with weights_only=True. "
                f"If this is a trusted file, set TTS_ALLOW_UNSAFE_PICKLE=1"
            )
        logger.warning(
            "Loading %s with weights_only=False — only load trusted .pt files",
            prompt_file,
        )
        result = torch.load(prompt_path, weights_only=False, map_location=device)  # nosec B614
        # Cache the result
        with _torch_prompt_cache_lock:
            max_size = get_voice_prompt_cache_max()
            if len(_torch_prompt_cache) >= max_size:
                _torch_prompt_cache.popitem(last=False)
            _torch_prompt_cache[prompt_file] = result
            _torch_prompt_cache_misses += 1
        return result


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
    _mlx_prompt_cache.clear()


def voice_prompt_cache_info():
    """Return cache statistics for the active backend.

    For torch: returns simple namespace with currsize/maxsize (manual OrderedDict cache).
    For mlx: returns a simple namespace with hits/currsize.
    """
    from types import SimpleNamespace
    backend = get_backend()
    if backend == "mlx":
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

    # Evict least-recently-used entry if at capacity (uses config value)
    max_cache_size = get_voice_prompt_cache_max()
    if len(_mlx_prompt_cache) >= max_cache_size:
        _mlx_prompt_cache.popitem(last=False)

    _mlx_prompt_cache[prompt_name] = result
    return result
