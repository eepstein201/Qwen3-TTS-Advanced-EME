#!/usr/bin/env python3
"""Audio transcription (ASR) — lazy loading, backend-dispatched.

Imports from: config, audio_processing.
"""

import logging
import threading

from qwen3_tts.core.config import get_backend

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
                _asr_model_mlx = load_stt_model("mlx-community/whisper-large-v3-turbo")
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

    logger.info("Transcribing (MLX): %s", audio_path)
    t0 = time.time()
    try:
        result = _asr_model_mlx.generate(audio_path, language=language)
        transcript = result.text.strip() if result.text else ""
        logger.info("Transcription complete: %d chars in %.1fs", len(transcript), time.time() - t0)
        return transcript
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")


def _transcribe_torch(audio_path, language="en"):
    """Transcribe using transformers Whisper pipeline (CUDA/CPU)."""
    import time
    _ensure_asr_torch_loaded()

    logger.info("Transcribing (torch): %s", audio_path)
    t0 = time.time()
    try:
        kwargs = {}
        if language:
            kwargs["generate_kwargs"] = {"language": language}
        result = _asr_model_torch(audio_path, **kwargs)
        transcript = result["text"].strip() if result.get("text") else ""
        logger.info("Transcription complete: %d chars in %.1fs", len(transcript), time.time() - t0)
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
    """Free ASR model from memory to reclaim VRAM/RAM for TTS generation."""
    global _asr_model_mlx, _asr_model_torch
    if _asr_model_mlx is not None:
        _asr_model_mlx = None
        logger.info("Unloaded MLX ASR model")
    if _asr_model_torch is not None:
        _asr_model_torch = None
        logger.info("Unloaded torch ASR model")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    import gc
    gc.collect()


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
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
