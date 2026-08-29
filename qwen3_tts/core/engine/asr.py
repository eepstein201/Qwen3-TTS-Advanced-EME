#!/usr/bin/env python3
"""Audio transcription (ASR) — lazy loading, backend-dispatched.

Imports from: config, audio_processing.
"""

import logging
import threading

from qwen3_tts.core.config import get_backend, sanitize_log

logger = logging.getLogger("tts.engine")

# ASR model caches — NOT loaded at startup, preloaded in background by UI
_asr_model_mlx = None
_asr_model_torch = None
_asr_lock = threading.Lock()


def _ensure_asr_torch_loaded():
    """Thread-safe loading of the torch ASR pipeline. Blocks until ready."""
    global _asr_model_torch
    with _asr_lock:
        if _asr_model_torch is not None:
            return
        import time

        logger.info("Loading torch ASR model...")
        t0 = time.time()
        try:
            from transformers import pipeline as hf_pipeline

            from qwen3_tts.core.config import get_device

            device_name = get_device()
            if device_name == "cuda":
                device = 0
            elif device_name == "mps":
                device = "mps"
            else:
                device = -1
            _asr_model_torch = hf_pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                revision=_WHISPER_REVISION,
                device=device,
                chunk_length_s=30,
            )
            logger.info("Torch ASR model loaded in %.1fs", time.time() - t0)
        except ImportError as e:
            raise ImportError(
                f"ASR transcription requires transformers: {e}\n"
                "Install with: pip install transformers"
            )


def preload_asr_model():
    """Preload ASR model in a background thread. Non-blocking, non-fatal."""

    def _load():
        try:
            backend = get_backend()
            if backend != "mlx":
                _ensure_asr_torch_loaded()
        except Exception as e:
            logger.warning("ASR preload failed (will retry on first use): %s", e)

    threading.Thread(target=_load, daemon=True).start()


_MLX_WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"
_WHISPER_PROCESSOR_SOURCE = "openai/whisper-large-v3-turbo"
# HuggingFace revision for Whisper downloads. Default "main" preserves current
# behavior; pin to a tag/SHA here to avoid tracking a repo's moving main branch.
_WHISPER_REVISION = "main"
_WHISPER_PROCESSOR_FILES = (
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "generation_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
)


def _ensure_mlx_whisper_processor() -> None:
    """Copy HF processor/tokenizer files into the MLX Whisper snapshot dir.

    Why: The mlx-community/whisper-large-v3-turbo repo ships only weights +
    config.json. mlx-audio loads them via WhisperProcessor.from_pretrained,
    which then hard-fails on get_tokenizer(). We bridge by copying the
    matching files from the upstream openai/whisper-large-v3-turbo repo.
    Idempotent: skips files that already exist.
    """
    import os
    import shutil

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        return  # huggingface_hub is a hard dep of mlx_audio; if missing the next call fails clearly

    try:
        snapshot_dir = snapshot_download(
            repo_id=_MLX_WHISPER_REPO,
            revision=_WHISPER_REVISION,
            allow_patterns=["config.json"],
        )
    except Exception as e:
        logger.warning("Could not locate MLX Whisper snapshot: %s", e)
        return

    missing = [
        f
        for f in _WHISPER_PROCESSOR_FILES
        if not os.path.exists(os.path.join(snapshot_dir, f))
    ]
    if not missing:
        return

    logger.info(
        "Fetching %d missing Whisper processor files from %s",
        len(missing),
        _WHISPER_PROCESSOR_SOURCE,
    )
    for filename in missing:
        try:
            src = hf_hub_download(
                repo_id=_WHISPER_PROCESSOR_SOURCE,
                filename=filename,
                revision=_WHISPER_REVISION,
            )
            shutil.copy(src, os.path.join(snapshot_dir, filename))
        except Exception as e:
            # added_tokens.json may legitimately not exist upstream; log and continue
            logger.debug("Skipped processor file %s: %s", filename, e)


def load_asr_model():
    """Load ASR model synchronously for the current backend. Returns True on success."""
    global _asr_model_mlx
    backend = get_backend()
    if backend == "mlx":
        with _asr_lock:
            if _asr_model_mlx is not None:
                return True
            try:
                from mlx_audio.stt import load_model as load_stt_model

                _ensure_mlx_whisper_processor()
                _asr_model_mlx = load_stt_model(_MLX_WHISPER_REPO)
                logger.info("Loaded MLX ASR model")
                return True
            except ImportError as e:
                raise ImportError(f"ASR requires mlx-audio with STT support: {e}")
    else:
        _ensure_asr_torch_loaded()
        return True


def _transcribe_mlx(audio_path, language="en"):
    """Transcribe using MLX ASR (Apple Silicon)."""
    import time

    if _asr_model_mlx is None:
        load_asr_model()

    logger.info("Transcribing (MLX): %s", sanitize_log(audio_path))
    t0 = time.time()
    try:
        result = _asr_model_mlx.generate(audio_path, language=language)
        transcript = result.text.strip() if result.text else ""
        logger.info(
            "Transcription complete: %d chars in %.1fs",
            len(transcript),
            time.time() - t0,
        )
        return transcript
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")


def _transcribe_torch(audio_path, language="en"):
    """Transcribe using transformers Whisper pipeline (CUDA/CPU)."""
    import time

    _ensure_asr_torch_loaded()

    logger.info("Transcribing (torch): %s", sanitize_log(audio_path))
    t0 = time.time()
    try:
        kwargs = {}
        if language:
            kwargs["generate_kwargs"] = {"language": language}
        result = _asr_model_torch(audio_path, **kwargs)
        transcript = result["text"].strip() if result.get("text") else ""
        logger.info(
            "Transcription complete: %d chars in %.1fs",
            len(transcript),
            time.time() - t0,
        )
        return transcript
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")


def transcribe_audio(audio_path, language="en"):
    """Transcribe audio file to text.

    Dispatches to MLX ASR (Apple Silicon) or torch Whisper (CUDA/CPU)
    based on configured backend. Lazily loads the ASR model on first call.

    Args:
        audio_path: Path to audio file (.wav, .mp3, etc.).
        language: Language code for transcription (default: "en").

    Returns:
        Transcribed text string.

    Raises:
        ImportError: If required ASR library is not installed.
        RuntimeError: If transcription fails.
    """
    backend = get_backend()
    if backend == "mlx":
        return _transcribe_mlx(audio_path, language)
    else:
        return _transcribe_torch(audio_path, language)


def unload_asr_model():
    """Free ASR model from memory to reclaim VRAM/RAM for TTS generation.

    Takes ``_asr_lock`` (#214 item 2). Every loader in this module already
    serializes on it — ``_ensure_asr_torch_loaded`` and the MLX loader — but
    unload used to null the shared globals unsynchronized, so it could land
    in the middle of a loader's check-then-assign and drop a model that had
    just been constructed, or clear the slot between a caller's
    ``is_asr_loaded()`` and its use.

    Safe to take unconditionally: ``handle_unload_asr`` is the only caller and
    holds nothing, so this is a leaf acquisition. ``threading.Lock`` is NOT
    reentrant — never call this from a function that already holds the lock.

    Only the reference drop is locked. ``gc.collect()`` and the CUDA cache
    release run AFTER the lock: they are unbounded work that would otherwise
    block every concurrent loader for no benefit.

    Order matters — ``gc.collect()`` FIRST, then ``torch.cuda.empty_cache()``,
    matching ``unload_model_cleanup``. Nulling the global only drops a
    reference; the tensors return to the caching allocator when gc actually
    collects the model. Calling ``empty_cache()`` before that releases the
    allocator's free blocks while the model's are still live, so the VRAM this
    function exists to reclaim stays with the process instead of the driver.
    """
    global _asr_model_mlx, _asr_model_torch
    had_torch_model = False
    with _asr_lock:
        if _asr_model_mlx is not None:
            _asr_model_mlx = None
            logger.info("Unloaded MLX ASR model")
        if _asr_model_torch is not None:
            _asr_model_torch = None
            had_torch_model = True
            logger.info("Unloaded torch ASR model")

    import gc

    gc.collect()

    if had_torch_model:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def is_asr_available():
    """Check if ASR transcription is available.

    Returns True if the current backend has a compatible ASR library.
    Does NOT load the ASR model — just checks availability.
    """
    if get_backend() == "mlx":
        try:
            from mlx_audio.stt import load_model  # noqa: F401

            return True
        except ImportError:
            return False
    else:
        try:
            from transformers import pipeline  # noqa: F401

            return True
        except ImportError:
            return False


def is_asr_loaded():
    """Check if an ASR model is currently loaded in memory."""
    return _asr_model_mlx is not None or _asr_model_torch is not None


def get_asr_model_info():
    """Return info dict about loaded ASR model (or None if not loaded)."""
    if _asr_model_mlx is not None:
        return {
            "loaded": True,
            "backend": "mlx",
            "model_name": "whisper-large-v3-turbo",
        }
    if _asr_model_torch is not None:
        return {
            "loaded": True,
            "backend": "torch",
            "model_name": "whisper-base",
        }
    return {"loaded": False, "backend": None, "model_name": None}


def unload_model_cleanup():
    """Backend-specific memory cleanup after setting a model to None."""
    import gc

    gc.collect()
    backend = get_backend()
    if backend == "torch":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
