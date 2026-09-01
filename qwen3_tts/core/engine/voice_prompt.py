#!/usr/bin/env python3
"""Voice prompt loading, caching, and migration.

Imports from: config, audio_processing.
Uses lazy imports for model_loader and inference to avoid circular dependencies.
"""

import logging
import os
import pickle  # nosec B403  # only pickle.UnpicklingError (corrupted-.pt fallback); prompts are local trusted artifacts created by create_voice.py, torch.load does the unpickling
import threading
from collections import OrderedDict

from qwen3_tts.core.config import (
    VOICE_PROMPTS_DIR,
    get_backend,
    get_voice_prompt_cache_max,
    safe_path_join,
    sanitize_log,
)
from qwen3_tts.core.engine.audio_processing import (
    DEFAULT_SAMPLE_RATE,
    load_audio_for_cloning,
)

logger = logging.getLogger("tts.engine")


class VoicePromptCreateRequired(Exception):  # noqa: N818 -- signal, not an error (see docstring)
    """Internal control-flow signal: allow_create=False but a create is needed.

    Deliberately NOT in the TTSError hierarchy (core/config/errors.py) -- it is
    raised in a worker thread, caught one frame later in the server helper
    (server/prompt_loading.py), and never reaches a user.
    """

    def __init__(self, prompt_file: str) -> None:
        super().__init__(prompt_file)
        self.prompt_file = prompt_file


def _evict_if_full(cache: OrderedDict, max_size: int) -> None:
    """Remove oldest entry from LRU cache if at or over capacity."""
    if len(cache) >= max_size:
        cache.popitem(last=False)


# ---------------------------------------------------------------------------
# Voice prompt cache (torch backend)
# ---------------------------------------------------------------------------

_torch_prompt_cache: OrderedDict = OrderedDict()
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


def _auto_create_pt_from_wav(
    base_name: str,
    wav_path: str,
    txt_path: str,
    prompt_path: str,
    prompt_file: str,
    *,
    model=None,
):
    """Create a .pt voice prompt from .wav + .txt files and cache the result.

    Returns the voice_prompt tensor, or None if wav_path does not exist.

    Args:
        model: Optional pre-loaded clone model (#214 item 1). load_model()
            has NO memoization (model_loader.py) -- every call is a full
            multi-minute weight construction -- so a caller that has already
            built the model (e.g. server/prompt_loading.py, building it
            OUTSIDE inference_lock before re-entering locked) must forward
            it here. Only load_model("clone") internally when model is None.
    """
    import torch

    if not os.path.exists(wav_path):
        return None
    logger.info("Auto-creating .pt from .wav for %s", sanitize_log(base_name))
    ref_audio, ref_sr = load_audio_for_cloning(wav_path)
    transcript = ""
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            transcript = f.read().strip()
    if not transcript:
        logger.warning(
            "No transcript for %s, using empty string", sanitize_log(base_name)
        )
    from qwen3_tts.core.engine.inference import create_voice_prompt

    if model is None:
        from qwen3_tts.core.engine.model_loader import load_model

        model = load_model("clone")
    voice_prompt = create_voice_prompt(
        model, ref_audio, ref_sr, transcript, x_vector_only_mode=not transcript
    )
    torch.save(voice_prompt, prompt_path)
    logger.info("Auto-created and saved %s", sanitize_log(prompt_path))
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
        result = torch.load(
            prompt_path, weights_only=True, map_location=device
        )  # CodeQL: weights_only=True is safe [py/unsafe-deserialization]
        _store_in_torch_cache(prompt_file, result)
        return result
    except (RuntimeError, ValueError, TypeError, pickle.UnpicklingError):
        # Only unpickling/weights_only failures reach here.
        # PermissionError, MemoryError, OSError propagate normally.
        real_prompt = os.path.realpath(prompt_path)
        real_prompts_dir = os.path.realpath(VOICE_PROMPTS_DIR)
        if not real_prompt.startswith(real_prompts_dir + os.sep):
            raise ValueError(
                f"Refusing to load {prompt_file}: outside voice_prompts/ directory"
            )
        base = prompt_file.removesuffix(".pt")
        wav_exists = os.path.exists(
            safe_path_join(str(VOICE_PROMPTS_DIR), f"{base}.wav")
        )
        hint = (
            f"Run 'tts voice rebuild {base}' to regenerate from .wav+.txt."
            if wav_exists
            else f"No {base}.wav found — re-create with 'tts voice create'."
        )
        raise RuntimeError(
            f"Cannot load {prompt_file} safely. {hint}"
        )


def _load_voice_prompt_torch(prompt_file, *, allow_create: bool = True, clone_model=None):
    """Load and cache a .pt voice prompt (torch backend).

    If the .pt file doesn't exist but .wav + .txt files do, auto-creates
    the .pt using the already-loaded clone model (avoids loading a second model).

    Results are cached (up to cache.voice_prompt_max entries, default 10) for
    repeated lookups. Cache is config-aware and respects the voice_prompt_max setting.

    Args:
        allow_create: When False, raise VoicePromptCreateRequired instead of
            running the (real GPU inference) create inline -- used by
            server/prompt_loading.py to probe unlocked and only re-enter
            under inference_lock when a create is actually needed (#214
            item 1). When True (the default -- preserves every existing
            direct caller, including tests/test_voice_prompts.py's
            positional calls), the create runs inline exactly as before.
        clone_model: Optional pre-loaded clone model forwarded straight
            through to _auto_create_pt_from_wav (see its docstring for why
            dropping this reconstructs the model under the lock).
    """
    global _torch_prompt_cache_hits
    import torch  # noqa: F401 — needed for cache type

    with _torch_prompt_cache_lock:
        if prompt_file in _torch_prompt_cache:
            # Load-bearing for #214 item 1's concurrent-convergence guarantee:
            # two callers racing the same missing prompt each run an unlocked
            # allow_create=False probe (raises, no create), then serialize on
            # inference_lock for the allow_create=True retry. The FIRST
            # locked caller creates and populates this cache; by the time the
            # SECOND locked caller reaches this check, the entry is already
            # here, so it returns the cached result instead of creating a
            # second time. Do not remove this without preserving that
            # property some other way.
            _torch_prompt_cache.move_to_end(prompt_file)
            _torch_prompt_cache_hits += 1
            return _torch_prompt_cache[prompt_file]

    prompt_path = safe_path_join(str(VOICE_PROMPTS_DIR), prompt_file)
    if not os.path.exists(prompt_path):
        base_name = prompt_file[:-3] if prompt_file.endswith(".pt") else prompt_file
        wav_path = safe_path_join(str(VOICE_PROMPTS_DIR), f"{base_name}.wav")
        txt_path = safe_path_join(str(VOICE_PROMPTS_DIR), f"{base_name}.txt")
        # Duplicated exists() check (also done inside _auto_create_pt_from_wav
        # at its own top) -- necessary because the allow_create decision must
        # happen BEFORE calling it: we need to raise VoicePromptCreateRequired
        # to the caller instead of ever starting the create.
        if not allow_create and os.path.exists(wav_path):
            raise VoicePromptCreateRequired(prompt_file)
        return _auto_create_pt_from_wav(
            base_name, wav_path, txt_path, prompt_path, prompt_file, model=clone_model
        )

    from qwen3_tts.core.config import get_device

    try:
        return _load_pt_safe(prompt_path, prompt_file, get_device())
    except (RuntimeError, ValueError):
        base_name = prompt_file[:-3] if prompt_file.endswith(".pt") else prompt_file
        wav_path = safe_path_join(str(VOICE_PROMPTS_DIR), f"{base_name}.wav")
        txt_path = safe_path_join(str(VOICE_PROMPTS_DIR), f"{base_name}.txt")
        # Same duplicated check as above, on the corrupt-.pt fallback path.
        if not allow_create and os.path.exists(wav_path):
            raise VoicePromptCreateRequired(prompt_file) from None
        result = _auto_create_pt_from_wav(
            base_name, wav_path, txt_path, prompt_path, prompt_file, model=clone_model
        )
        if result is not None:
            logger.warning(
                "Rebuilt corrupt %s from .wav+.txt fallback",
                sanitize_log(prompt_file),
            )
            return result
        raise


def load_voice_prompt(prompt_file, *, allow_create: bool = True, clone_model=None):
    """Load a voice prompt, dispatching to the correct format for the backend.

    - torch backend: loads .pt tensor file
    - mlx backend: loads .wav + .txt file pair as a dict (never creates --
      load_voice_prompt_mlx never raises VoicePromptCreateRequired, so
      allow_create/clone_model are no-ops on this backend)

    allow_create=True (the default) never raises VoicePromptCreateRequired.
    """
    backend = get_backend()
    if backend == "mlx":
        return load_voice_prompt_mlx(prompt_file)
    return _load_voice_prompt_torch(
        prompt_file, allow_create=allow_create, clone_model=clone_model
    )


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
        pt_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.pt")
        if not os.path.exists(pt_path):
            logger.info("Migrating orphan MLX prompt: %s", base)
            try:
                if model is None:
                    # Lazy import to avoid circular dependency
                    from qwen3_tts.core.engine.model_loader import load_model

                    model = clone_model or load_model("clone")
                ref_audio, ref_sr = load_audio_for_cloning(wav_path)
                transcript = ""
                txt_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.txt")
                if os.path.exists(txt_path):
                    with open(txt_path) as f:
                        transcript = f.read().strip()
                if not transcript:
                    logger.warning("No transcript for %s, using empty string", base)
                # Lazy import to avoid circular dependency
                from qwen3_tts.core.engine.inference import create_voice_prompt

                voice_prompt = create_voice_prompt(
                    model, ref_audio, ref_sr, transcript, x_vector_only_mode=not transcript
                )
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

_mlx_prompt_cache: OrderedDict = OrderedDict()
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

    wav_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.wav")
    txt_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.txt")

    if not os.path.exists(wav_path) or not os.path.exists(txt_path):
        # Check if a .pt file exists (torch-only prompt)
        pt_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.pt")
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

    with open(txt_path) as f:
        ref_text = f.read().strip()

    # Prompts created before reference audio was resampled on write are still
    # on disk, and MLX opens this path directly — so a below-native rate here
    # is the runaway-generation bug, not a quality nit. `tts voice rebuild`
    # does NOT repair it: rebuild regenerates the .pt and leaves the .wav
    # alone. Advisory only; never block a load over it. Cached, so this fires
    # once per cache-miss rather than once per generation.
    try:
        import soundfile as sf  # lazy — optional at import time

        on_disk_sr = sf.info(wav_path).samplerate
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break loading
        logger.debug("Could not read sample rate of %s: %s", wav_path, exc)
    else:
        if on_disk_sr < DEFAULT_SAMPLE_RATE:
            logger.warning(
                "Voice prompt '%s' has a %d Hz reference (below the model's "
                "native %d Hz). This makes clone generation fail to stop and "
                "run to the token cap. Re-create it with 'tts voice create' — "
                "'tts voice rebuild' will NOT fix it (it only regenerates the "
                ".pt; MLX reads the .wav).",
                sanitize_log(base),
                on_disk_sr,
                DEFAULT_SAMPLE_RATE,
            )

    result = {"ref_audio": wav_path, "ref_text": ref_text}

    # Cache the result with thread-safe lock
    with _mlx_prompt_cache_lock:
        _evict_if_full(_mlx_prompt_cache, get_voice_prompt_cache_max())
        _mlx_prompt_cache[prompt_name] = result

    return result


class UnsupportedReferenceAudioError(RuntimeError):
    """``ensure_min_sample_rate`` could not deliver a usable reference.

    Subclasses ``RuntimeError`` so every existing ``except RuntimeError``
    caller keeps working; the /create-voice-prompt MLX branch translates it
    to a 400 (a sub-native-rate reference is a client-input problem, and
    an 8 kHz prompt made MLX clone generation run to the token cap 3/3
    times -- it must never be stored as-is).
    """


def save_voice_prompt_mlx(base: str, audio_path: str, transcript: str) -> str:
    """Validate a reference clip and store an MLX voice prompt pair.

    The server-side MLX create path (#236): MLX consumes prompts as a
    ``.wav+.txt`` pair at generation time (``prepare_zeroprompt``) -- no
    model, no ``.pt``, no inference. This writer validates the audio
    (``ensure_min_sample_rate``: always mono, never downsamples, rewritten
    to >=24 kHz or it raises) and stores the pair, mirroring the CLI/UI
    write-time policy.

    Args:
        base: Prompt base name (no extension); caller has validated it.
        audio_path: Reference audio file (any soundfile-decodable container;
            the server stages uploads with a ``.wav`` suffix, so m4a/mp3 are
            NOT convertible here -- that is a documented limitation).
        transcript: Reference transcript; stored stripped.

    Returns:
        The written ``.wav`` path (str).

    Raises:
        UnsupportedReferenceAudioError: The sample-rate guarantee is undeliverable.
        soundfile errors: The audio is undecodable (mapped to 400 by the handler).
    """
    import shutil

    import soundfile as sf

    from qwen3_tts.core.engine.audio_processing import ensure_min_sample_rate

    ref_audio, ref_sr = sf.read(audio_path, dtype="float32")
    if ref_audio.ndim > 1:
        import numpy as np

        ref_audio = np.mean(ref_audio, axis=-1).astype(np.float32)
    try:
        ref_audio, ref_sr, was_modified = ensure_min_sample_rate(ref_audio, ref_sr)
    except RuntimeError as e:
        raise UnsupportedReferenceAudioError(str(e)) from e

    wav_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.wav")
    txt_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base}.txt")

    # Same write-vs-copy policy as the UI create path: rewriting identical
    # audio is pointless; anything the rate check modified must be written.
    if was_modified:
        sf.write(wav_path, ref_audio, ref_sr)
    else:
        shutil.copy2(audio_path, wav_path)

    with open(txt_path, "w") as f:
        f.write((transcript or "").strip())

    logger.info(
        "Stored MLX voice prompt '%s' (%.1fs audio @ %d Hz)",
        sanitize_log(base),
        len(ref_audio) / ref_sr,
        ref_sr,
    )
    return wav_path
