#!/usr/bin/env python3
"""TTS inference dispatch: stateless generation for torch and MLX backends.

Top-level orchestrator — imports from text_processing, audio_processing,
model_loader, voice_prompt, and config.
"""

import logging
import time

from qwen3_tts.core.config import (
    CONFIG_PATH,
    DefaultConfigLoader,
    get_backend,
    get_torch_dtype_name,
    load_config,
    sanitize_log,
)
from qwen3_tts.core.engine.audio_processing import process_audio, LUFS_TARGET
from qwen3_tts.core.engine.text_processing import _normalize_text, _split_text

logger = logging.getLogger("tts.engine")

_DEFAULT_CONFIG_LOADER = DefaultConfigLoader()

# ---------------------------------------------------------------------------
# Strategy registries for OCP-compliant dispatch
# ---------------------------------------------------------------------------

# Backend strategies: maps backend name -> inference function
_INFERENCE_STRATEGIES = {}

# Mode strategies for torch backend: maps mode name -> model method name
_MODE_STRATEGIES_TORCH = {
    "clone": "generate_voice_clone",
    "design": "generate_voice_design",
    "custom": "generate_custom_voice",
}


from typing import Callable, Any, Iterator

logger = logging.getLogger("tts.engine")

_DEFAULT_CONFIG_LOADER = DefaultConfigLoader()

# ---------------------------------------------------------------------------
# Strategy registries for OCP-compliant dispatch
# ---------------------------------------------------------------------------

# Backend strategies: maps backend name -> inference function
_INFERENCE_STRATEGIES = {}

# Mode strategies for torch backend: maps mode name -> model method name
_MODE_STRATEGIES_TORCH = {
    "clone": "generate_voice_clone",
    "design": "generate_voice_design",
    "custom": "generate_custom_voice",
}


def register_backend(name: str, strategy_fn: Callable) -> None:
    """Register a new inference backend strategy.

    This enables extending the system with new backends without modifying
    existing dispatch code (Open/Closed Principle).

    Args:
        name: Backend name (e.g., "mlx", "torch", "vllm")
        strategy_fn: Function with signature (model, text, mode, gen_params, **kwargs)
                     returning (wavs, sample_rate)
    """
    _INFERENCE_STRATEGIES[name] = strategy_fn
    logger.debug("Registered inference backend: %s", name)


def _get_backend_strategy(backend: str) -> Callable:
    """Get the strategy function for a backend.

    Args:
        backend: Backend name

    Returns:
        Strategy function

    Raises:
        ValueError: If backend not registered
    """
    if backend not in _INFERENCE_STRATEGIES:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available: {list(_INFERENCE_STRATEGIES.keys())}"
        )
    return _INFERENCE_STRATEGIES[backend]


# ---------------------------------------------------------------------------
# Torch backend — inference helpers (H6)
# ---------------------------------------------------------------------------

def _apply_mps_float32_guard(model: Any, mode: str) -> Any | None:
    """Override model to float32 for clone mode on MPS if needed.

    Returns the original dtype (to restore later), or None if no override was needed.
    """
    import torch
    if mode != "clone" or not torch.backends.mps.is_available():
        return None
    dtype_name = get_torch_dtype_name()
    if dtype_name == "float32":
        return None
    logger.warning(
        "Clone mode on MPS requires float32 (configured: %s). "
        "Overriding to float32 for this generation. "
        "Set advanced.dtype to 'float32' in %s to silence this warning.",
        sanitize_log(dtype_name), CONFIG_PATH,
    )
    original_dtype = next(model.parameters()).dtype
    model.float()
    return original_dtype


def _dispatch_torch_mode(model: Any, text: str, mode: str, language: str, params: dict,
                         voice_prompt: Any, voice_description: str | None, speaker: str | None,
                         instruct: str | None, x_vector_only_mode: bool) -> tuple:
    """Call the appropriate model method for the given mode. Returns (wavs, sr)."""
    import torch
    with torch.inference_mode():
        if mode == "clone":
            clone_kwargs = dict(text=text, language=language, voice_clone_prompt=voice_prompt, **params)
            if x_vector_only_mode:
                clone_kwargs["x_vector_only_mode"] = True
            return model.generate_voice_clone(**clone_kwargs)
        elif mode == "custom":
            return model.generate_custom_voice(
                text=text, speaker=speaker, instruct=instruct or "", language=language, **params,
            )
        else:  # design
            return model.generate_voice_design(
                text=text, instruct=voice_description or "", language=language, **params,
            )


def _cleanup_device_memory() -> None:
    """Release MPS or CUDA cached memory after inference and log usage."""
    import torch
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            peak = torch.mps.current_allocated_memory()
            logger.debug("MPS memory after generation: %.1f MB", peak / (1024 * 1024))
        except RuntimeError as e:
            logger.debug("MPS memory cleanup failed: %s", e)
    elif torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_allocated()
            logger.debug("CUDA memory after generation: %.1f MB", peak / (1024 * 1024))
        except RuntimeError as e:
            logger.debug("CUDA memory cleanup failed: %s", e)


# ---------------------------------------------------------------------------
# Torch backend — inference
# ---------------------------------------------------------------------------

def _run_inference_torch(model: Any, text: str, mode: str, gen_params: dict, language: str = "English",
                         voice_prompt: Any = None, voice_description: str | None = None,
                         speaker: str | None = None, instruct: str | None = None,
                         x_vector_only_mode: bool = False) -> tuple:
    """Run TTS inference using the PyTorch/MPS backend."""
    import torch

    t0 = time.time()
    original_dtype = _apply_mps_float32_guard(model, mode)
    params = _build_torch_params(gen_params)

    seed = gen_params.get("seed")
    if seed is not None:
        torch.manual_seed(seed)

    try:
        wavs, sr = _dispatch_torch_mode(
            model, text, mode, language, params,
            voice_prompt, voice_description, speaker, instruct, x_vector_only_mode,
        )
    except RuntimeError as e:
        if "inf" in str(e) or "nan" in str(e):
            dtype_name = get_torch_dtype_name()
            if dtype_name != "float32":
                logger.error(
                    "Generation produced NaN/Inf with dtype=%s. "
                    "Switch to float32 in %s under advanced.dtype for stability.",
                    sanitize_log(dtype_name), CONFIG_PATH,
                )
        raise
    finally:
        if original_dtype is not None:
            try:
                model.to(original_dtype)
            except (RuntimeError, TypeError) as restore_err:
                logger.warning("Failed to restore model dtype after MPS guard: %s", restore_err)

    _cleanup_device_memory()

    logger.info("Inference complete: %d chars, %.1fs, mode=%s [torch]", len(text), time.time() - t0, mode)
    wav = _validate_audio(wavs[0], sr, mode=mode)
    return wav, sr


def _validate_audio(wav: Any, sample_rate: int, mode: str = "unknown") -> Any:
    """Check generated audio for common quality issues.

    Logs warnings for issues found, never blocks. Returns the audio
    with issues corrected (NaN replaced, clipping normalized).

    Args:
        wav: Audio numpy array
        sample_rate: Sample rate in Hz
        mode: TTS mode (for logging)

    Returns:
        Validated/corrected audio array
    """
    import numpy as np  # lazy — heavy import
    if wav is None or len(wav) == 0:
        logger.warning("Generated audio is empty (mode=%s)", mode)
        return wav

    # NaN check
    if np.any(np.isnan(wav)):
        logger.warning("Generated audio contains NaN values (mode=%s), replacing with zeros", mode)
        wav = np.nan_to_num(wav, nan=0.0)

    # Clipping check
    peak = np.max(np.abs(wav))
    if peak > 1.0:
        logger.warning("Generated audio is clipping (peak=%.2f, mode=%s), normalizing", peak, mode)
        wav = wav / peak

    # All-silence check
    if np.max(np.abs(wav)) < 1e-6:
        logger.warning("Generated audio is silent (mode=%s)", mode)

    return wav


# ---------------------------------------------------------------------------
# MLX backend — inference
# ---------------------------------------------------------------------------

def _get_mlx_gen_params(gen_params: dict, config: dict) -> dict:
    """Merge caller gen_params with config defaults for MLX backend."""
    config_gen = config.get("generation", {})
    return {
        "temperature": gen_params.get("temperature", config_gen.get("temperature", 0.7)),
        "top_k": gen_params.get("top_k", config_gen.get("top_k", 50)),
        "top_p": gen_params.get("top_p", config_gen.get("top_p", 0.95)),
        "repetition_penalty": gen_params.get("repetition_penalty", config_gen.get("repetition_penalty", 1.05)),
        "max_new_tokens": gen_params.get("max_new_tokens", config_gen.get("max_new_tokens", 2048)),
    }


def _run_inference_mlx(model: Any, text: str, mode: str, gen_params: dict, language: str = "English",
                       voice_prompt: Any = None, voice_description: str | None = None,
                       speaker: str | None = None, instruct: str | None = None,
                       x_vector_only_mode: bool = False) -> tuple:
    """Run TTS inference using the MLX backend.

    Returns (wav_array, sample_rate) where wav_array is a float32 numpy array,
    matching the torch backend's output contract.

    Args:
        model: Loaded MLX model (from _load_model_mlx).
        text: Text to synthesize.
        mode: "clone", "design", or "custom".
        gen_params: Dict with temperature, top_k, top_p, repetition_penalty.
        language: Language string.
        voice_prompt: For clone mode — dict with "ref_audio" (path) and
                      "ref_text" (str), or a .pt prompt name (will error).
        voice_description: Voice description string (design mode).
        speaker: Speaker name string (custom mode).
        instruct: Style instruction string (custom mode).
        x_vector_only_mode: If True, use empty ref_text for speaker-embedding-only clone.
    """
    t0 = time.time()

    # Read defaults from config so MLX matches torch behavior (R-17)
    config = load_config()
    params = _get_mlx_gen_params(gen_params, config)

    if mode == "clone":
        # MLX clone mode uses ref_audio (wav path) + ref_text directly.
        # voice_prompt should be a dict {"ref_audio": path, "ref_text": str}
        # set up by the caller or by load_voice_prompt_mlx().
        if voice_prompt is None:
            raise ValueError("voice_prompt is required for clone mode")

        if isinstance(voice_prompt, dict):
            ref_audio_path = voice_prompt["ref_audio"]
            ref_text = "" if x_vector_only_mode else voice_prompt["ref_text"]
        else:
            raise TypeError(
                "MLX clone mode requires a voice prompt dict with 'ref_audio' "
                "and 'ref_text' keys. Torch .pt prompts are not compatible "
                "with the MLX backend. Re-create the prompt with 'tts voice create' "
                "to generate MLX-compatible files (.wav + .txt)."
            )

        results = list(model.generate(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            language=language,
            **params,
        ))

    elif mode == "custom":
        # MLX generate_custom_voice doesn't accept max_new_tokens
        custom_params = {k: v for k, v in params.items() if k != "max_new_tokens"}
        results = list(model.generate_custom_voice(
            text=text,
            speaker=speaker or "Ryan",
            language=language,
            instruct=instruct or "",
            **custom_params,
        ))

    else:  # design
        # MLX generate_voice_design doesn't accept max_new_tokens
        design_params = {k: v for k, v in params.items() if k != "max_new_tokens"}
        results = list(model.generate_voice_design(
            text=text,
            instruct=voice_description or "",
            language=language,
            **design_params,
        ))

    if not results:
        raise RuntimeError("MLX generation returned no results")

    # Collect audio from all segments and concatenate
    import numpy as np  # lazy — heavy import
    import mlx.core as mx

    audio_segments = [r.audio for r in results]
    if len(audio_segments) == 1:
        audio_mx = audio_segments[0]
    else:
        audio_mx = mx.concatenate(audio_segments)

    # Convert mx.array → numpy float32 (matches torch backend output contract)
    wav = np.array(audio_mx, dtype=np.float32)

    # Flatten to 1-D if needed
    if wav.ndim > 1:
        wav = wav.squeeze()

    sr = results[0].sample_rate

    elapsed = time.time() - t0
    logger.info(
        "Inference complete: %d chars, %.1fs, mode=%s [mlx]",
        len(text), elapsed, mode,
    )

    wav = _validate_audio(wav, sr, mode=mode)
    return wav, sr


def _run_inference_mlx_streaming(model: Any, text: str, mode: str, gen_params: dict, language: str = "English",
                                  voice_prompt: Any = None, voice_description: str | None = None,
                                  speaker: str | None = None, instruct: str | None = None,
                                  x_vector_only_mode: bool = False, config: dict | None = None,
                                  progress_callback: Any = None) -> Iterator[tuple]:
    """Run TTS inference using the MLX backend, yielding audio chunks as they generate.

    This is a generator that yields (audio_chunk, sample_rate) tuples as they
    become available, enabling streaming playback.

    Args:
        model: Loaded MLX model (from _load_model_mlx).
        text: Text to synthesize.
        mode: "clone", "design", or "custom".
        gen_params: Dict with temperature, top_k, top_p, repetition_penalty.
        language: Language string.
        voice_prompt: For clone mode — dict with "ref_audio" and "ref_text".
        voice_description: Voice description string (design mode).
        speaker: Speaker name string (custom mode).
        instruct: Style instruction string (custom mode).
        x_vector_only_mode: If True, use empty ref_text for speaker-embedding-only clone.
        config: Pre-loaded config dict (optional, avoids redundant disk read).
        progress_callback: Optional callable(chunk_index, chunk_total) for progress updates.

    Yields:
        (audio_chunk, sample_rate) tuples where audio_chunk is a float32 numpy array.
    """

    import numpy as np  # lazy — heavy import
    # Read defaults from config so MLX matches torch behavior (R-17)
    if config is None:
        config = load_config()
    params = _get_mlx_gen_params(gen_params, config)

    if mode == "clone":
        if voice_prompt is None:
            raise ValueError("voice_prompt is required for clone mode")
        if isinstance(voice_prompt, dict):
            ref_audio_path = voice_prompt["ref_audio"]
            ref_text = "" if x_vector_only_mode else voice_prompt["ref_text"]
        else:
            raise TypeError(
                "MLX clone mode requires a voice prompt dict with 'ref_audio' "
                "and 'ref_text' keys."
            )

        generator = model.generate(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            language=language,
            stream=True,  # Enable streaming
            **params,
        )

    elif mode == "custom":
        # MLX generate_custom_voice doesn't accept max_new_tokens
        custom_params = {k: v for k, v in params.items() if k != "max_new_tokens"}
        generator = model.generate_custom_voice(
            text=text,
            speaker=speaker or "Ryan",
            language=language,
            instruct=instruct or "",
            stream=True,  # Enable streaming
            **custom_params,
        )

    else:  # design
        # MLX generate_voice_design doesn't accept max_new_tokens
        design_params = {k: v for k, v in params.items() if k != "max_new_tokens"}
        generator = model.generate_voice_design(
            text=text,
            instruct=voice_description or "",
            language=language,
            stream=True,  # Enable streaming
            **design_params,
        )

    chunk_count = 0
    for result in generator:
        audio_mx = result.audio
        wav = np.array(audio_mx, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.squeeze()
        sr = result.sample_rate
        chunk_count += 1
        logger.debug("Streaming chunk %d: %d samples", chunk_count, len(wav))

        # Call progress callback (total unknown until completion for MLX)
        if progress_callback:
            progress_callback(chunk_count, 0)  # 0 = unknown total

        yield wav, sr

    # After completion, update with final count
    if progress_callback and chunk_count > 0:
        progress_callback(chunk_count, chunk_count)

    logger.info("Streaming complete: %d chunks, mode=%s [mlx]", chunk_count, mode)


# ---------------------------------------------------------------------------
# Chunk combination (crossfade / silence gap)
# ---------------------------------------------------------------------------

def _crossfade_chunks(chunks: list, sample_rate: int, crossfade_ms: int = 50, silence_gap_s: float | None = None) -> Any:
    """Combine audio chunks with crossfade or silence gap.

    Uses a raised-cosine (Hann) window for smooth transitions between chunks,
    eliminating audible clicks at boundaries.

    Args:
        chunks: List of numpy float32 audio arrays.
        sample_rate: Audio sample rate.
        crossfade_ms: Crossfade duration in ms (0 to disable). Default 50ms.
        silence_gap_s: If set, insert silence instead of crossfade.

    Returns:
        Combined float32 numpy array.
    """
    import numpy as np  # lazy — heavy import
    if len(chunks) == 0:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    if silence_gap_s is not None and silence_gap_s > 0:
        silence = np.zeros(int(sample_rate * silence_gap_s), dtype=np.float32)
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(chunk)
            if i < len(chunks) - 1:
                parts.append(silence)
        return np.concatenate(parts)

    if crossfade_ms <= 0:
        return np.concatenate(chunks)

    fade_samples = int(sample_rate * crossfade_ms / 1000)
    combined = chunks[0].copy()
    for chunk in chunks[1:]:
        overlap = min(fade_samples, len(combined), len(chunk))
        if overlap <= 0:
            combined = np.concatenate([combined, chunk])
            continue
        t = np.linspace(0, np.pi / 2, overlap, dtype=np.float32)
        fade_out = np.cos(t) ** 2
        fade_in = np.sin(t) ** 2
        combined[-overlap:] = combined[-overlap:] * fade_out + chunk[:overlap] * fade_in
        combined = np.concatenate([combined, chunk[overlap:]])
    return combined


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_max_chunk_chars() -> int:
    """Read max_chunk_chars from config, defaulting to 500."""
    try:
        config = load_config()
        return config.get("generation", {}).get("max_chunk_chars", 500)
    except Exception as e:
        logger.debug("Config read failed for max_chunk_chars, using default: %s", e)
        return 500


def _get_max_chunk_tokens() -> int:
    """Read max_chunk_tokens from config, defaulting to 200."""
    try:
        config = load_config()
        return config.get("generation", {}).get("max_chunk_tokens", 200)
    except Exception as e:
        logger.debug("Config read failed for max_chunk_tokens, using default: %s", e)
        return 200


def _prepare_text_chunks(text: str, language: str, model, max_chunk_chars: int) -> list[str]:
    """Normalize and chunk text for inference.

    Args:
        text: Text to process.
        language: Language string.
        model: Model with optional tokenizer attribute.
        max_chunk_chars: Max chars per chunk.

    Returns:
        List of text chunks ready for inference.
    """
    text = _normalize_text(text, language)
    tokenizer = getattr(model, "tokenizer", None)
    max_tokens = _get_max_chunk_tokens() if tokenizer is not None else None

    if tokenizer is not None and max_tokens is not None:
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count > max_tokens:
            return _split_text(
                text, max_chars=max_chunk_chars,
                language=language, tokenizer=tokenizer,
                max_tokens=max_tokens
            )
        return [text]
    elif max_chunk_chars > 0 and len(text) > max_chunk_chars:
        return _split_text(text, max_chars=max_chunk_chars, language=language)
    return [text]


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _set_seed_for_backend(seed: int | None) -> None:
    """Set the random seed for the active backend (torch or mlx).

    No-op when seed is None. Intended for seed-locking across chunks so that
    each chunk gets identical initial random state for voice consistency.
    """
    if seed is None:
        return
    backend = get_backend()
    if backend == "mlx":
        import mlx.core as mx
        mx.random.seed(seed)
    else:
        import torch
        torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def run_inference(model: Any, text: str, mode: str, gen_params: dict, language: str = "English",
                  voice_prompt: Any = None, voice_description: str | None = None,
                  speaker: str | None = None, instruct: str | None = None,
                  max_chunk_chars: int | None = None, progress_callback: Any = None,
                  x_vector_only_mode: bool = False, config_provider: Any = None,
                  seed_lock_chunks: bool = False) -> tuple:
    """Run TTS inference, dispatching to the configured backend.

    For long texts, automatically splits into chunks at sentence boundaries
    and concatenates the results with short silence gaps.

    Args:
        model: Loaded model (from load_model).
        text: Text to synthesize.
        mode: "clone", "design", or "custom".
        gen_params: Dict with temperature, top_k, top_p, repetition_penalty.
        language: Language string.
        voice_prompt: Loaded voice prompt (clone mode).
        voice_description: Voice description string (design mode).
        speaker: Speaker name string (custom mode).
        instruct: Style instruction string (custom mode).
        max_chunk_chars: Max chars per chunk (None = read from config, 0 = disable).
        progress_callback: Optional callable(chunk_index, chunk_total) for progress.

    Returns:
        (wav_array, sample_rate) tuple.
    """
    if max_chunk_chars is None:
        max_chunk_chars = _get_max_chunk_chars()

    # Normalize and split into chunks
    chunks = _prepare_text_chunks(text, language, model, max_chunk_chars)

    if len(chunks) == 1:
        # Single chunk — no overhead
        if progress_callback:
            progress_callback(0, 1)
        wav, sr = _run_inference_single(
            model, chunks[0], mode, gen_params, language,
            voice_prompt, voice_description, speaker, instruct,
            x_vector_only_mode=x_vector_only_mode,
        )
        config = (config_provider or _DEFAULT_CONFIG_LOADER).load()
        if config.get("generation", {}).get("lufs_normalize", False):
            wav, sr = process_audio(wav, sr, lufs_target=LUFS_TARGET)
        return wav, sr

    # Multi-chunk: generate each, combine with crossfade or silence gap
    logger.info("Splitting text (%d chars) into %d chunks (max %d chars each)",
                len(text), len(chunks), max_chunk_chars)

    all_audio = []
    sample_rate = None

    seed = gen_params.get("seed") if seed_lock_chunks else None

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i, len(chunks))

        # Re-seed before each chunk for voice consistency across chunks
        if seed is not None:
            _set_seed_for_backend(seed)

        preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
        logger.info("Chunk %d/%d: '%s' (%d chars)", i + 1, len(chunks), sanitize_log(preview), len(chunk))

        wav, sr = _run_inference_single(
            model, chunk, mode, gen_params, language,
            voice_prompt, voice_description, speaker, instruct,
            x_vector_only_mode=x_vector_only_mode,
        )

        if sample_rate is None:
            sample_rate = sr

        all_audio.append(wav)

    # Combine chunks: use silence_gap_seconds from config, or crossfade (default 50ms)
    config = (config_provider or _DEFAULT_CONFIG_LOADER).load()
    silence_gap = config.get("generation", {}).get("silence_gap_seconds", 0.0)
    if silence_gap > 0:
        result = _crossfade_chunks(all_audio, sample_rate, crossfade_ms=0, silence_gap_s=silence_gap)
    else:
        result = _crossfade_chunks(all_audio, sample_rate, crossfade_ms=50)

    if config.get("generation", {}).get("lufs_normalize", False):
        result, sample_rate = process_audio(result, sample_rate, lufs_target=LUFS_TARGET)

    logger.info("Combined %d chunks into %.1fs audio", len(chunks), len(result) / sample_rate)
    return result, sample_rate


def _run_inference_single(model: Any, text: str, mode: str, gen_params: dict, language: str = "English",
                          voice_prompt: Any = None, voice_description: str | None = None,
                          speaker: str | None = None, instruct: str | None = None,
                          _metal_retry_depth: int = 0, x_vector_only_mode: bool = False) -> tuple:
    """Run TTS inference for a single text chunk.

    For MLX backend, includes Metal crash recovery: on certain Metal kernel
    errors, retries with smaller sub-chunks up to depth 2.

    Dispatches to the appropriate backend strategy via the registry.
    """
    import numpy as np  # lazy — heavy import (used in Metal retry concatenation)
    backend = get_backend()

    # Get strategy from registry (OCP: extensible without modification)
    strategy = _get_backend_strategy(backend)

    # MLX backend has special Metal crash recovery logic
    if backend == "mlx":
        try:
            return strategy(
                model, text, mode, gen_params, language,
                voice_prompt, voice_description, speaker, instruct,
                x_vector_only_mode=x_vector_only_mode,
            )
        except RuntimeError as e:
            # Metal kernel crashes often contain "command buffer" or "GPU" in the message
            error_str = str(e).lower()
            is_metal_crash = any(
                keyword in error_str
                for keyword in ("command buffer", "gpu", "metal", "kernel")
            )
            if is_metal_crash and _metal_retry_depth < 2 and len(text) > 100:
                logger.warning(
                    "Metal kernel issue detected (depth %d), retrying with smaller sub-chunks: %s",
                    _metal_retry_depth, str(e)[:100],
                )
                # Split the chunk in half and process each sub-chunk
                mid = len(text) // 2
                # Find a space near the midpoint to avoid splitting mid-word
                split_idx = text.rfind(" ", mid - 50, mid + 50)
                if split_idx == -1:
                    split_idx = mid
                chunk1, chunk2 = text[:split_idx].strip(), text[split_idx:].strip()

                wav1, sr = _run_inference_single(
                    model, chunk1, mode, gen_params, language,
                    voice_prompt, voice_description, speaker, instruct,
                    _metal_retry_depth=_metal_retry_depth + 1,
                    x_vector_only_mode=x_vector_only_mode,
                )
                wav2, _ = _run_inference_single(
                    model, chunk2, mode, gen_params, language,
                    voice_prompt, voice_description, speaker, instruct,
                    _metal_retry_depth=_metal_retry_depth + 1,
                    x_vector_only_mode=x_vector_only_mode,
                )
                # Concatenate with short silence (use config gap, min 50ms)
                _retry_cfg = _DEFAULT_CONFIG_LOADER.load()
                _gap_s = _retry_cfg.get("generation", {}).get("silence_gap_seconds", 0.1)
                silence = np.zeros(int(sr * max(_gap_s, 0.05)), dtype=np.float32)
                return np.concatenate([wav1, silence, wav2]), sr
            raise

    # Default: use strategy from registry
    return strategy(
        model, text, mode, gen_params, language,
        voice_prompt, voice_description, speaker, instruct,
        x_vector_only_mode=x_vector_only_mode,
    )


def run_inference_streaming(model: Any, text: str, mode: str, gen_params: dict, language: str = "English",
                            voice_prompt: Any = None, voice_description: str | None = None,
                            speaker: str | None = None, instruct: str | None = None,
                            max_chunk_chars: int | None = None, x_vector_only_mode: bool = False,
                            config_provider: Any = None, progress_callback: Any = None) -> Iterator[tuple]:
    """Run TTS inference in streaming mode, yielding audio chunks as they generate.

    For MLX backend, uses native streaming from model.generate().
    For torch backend, falls back to text chunking — yields per-chunk audio.

    Args:
        model: Loaded model (from load_model).
        text: Text to synthesize.
        mode: "clone", "design", or "custom".
        gen_params: Dict with temperature, top_k, top_p, repetition_penalty.
        language: Language string.
        voice_prompt: Loaded voice prompt (clone mode).
        voice_description: Voice description string (design mode).
        speaker: Speaker name string (custom mode).
        instruct: Style instruction string (custom mode).
        max_chunk_chars: Max chars per chunk for torch fallback (None = config default).
        progress_callback: Optional callable(chunk_index, chunk_total) for progress updates.
                         Called as each chunk is yielded. For MLX, chunk_total=0 until completion.

    Yields:
        (audio_chunk, sample_rate) tuples where audio_chunk is float32 numpy array.
    """
    config = (config_provider or _DEFAULT_CONFIG_LOADER).load()
    backend = get_backend()

    if backend == "mlx":
        # MLX has native streaming — yield chunks as they generate
        logger.info("Starting streaming inference [mlx]")
        yield from _run_inference_mlx_streaming(
            model, text, mode, gen_params, language,
            voice_prompt, voice_description, speaker, instruct,
            x_vector_only_mode=x_vector_only_mode, config=config,
            progress_callback=progress_callback,
        )
    else:
        # Torch fallback: chunk the text and yield per-chunk audio
        logger.info("Starting chunked streaming [torch fallback]")
        if max_chunk_chars is None:
            max_chunk_chars = _get_max_chunk_chars()

        # Normalize and split into chunks
        chunks = _prepare_text_chunks(text, language, model, max_chunk_chars)
        chunk_total = len(chunks)  # Total known upfront for torch

        for i, chunk in enumerate(chunks):
            preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
            logger.info("Streaming chunk %d/%d: '%s'", i + 1, chunk_total, preview)

            # Call progress callback with known total
            if progress_callback:
                progress_callback(i + 1, chunk_total)

            wav, sr = _run_inference_single(
                model, chunk, mode, gen_params, language,
                voice_prompt, voice_description, speaker, instruct,
                x_vector_only_mode=x_vector_only_mode,
            )
            yield wav, sr


def create_voice_prompt(model: Any, ref_audio: Any, ref_sr: int, transcript: str) -> Any:
    """Create a reusable voice-clone prompt from reference audio.

    Args:
        model: Loaded clone (Base) model.
        ref_audio: numpy array of reference audio (mono).
        ref_sr: Sample rate of reference audio.
        transcript: Text transcript of the reference audio.

    Returns:
        Voice prompt tensor (suitable for torch.save / generate_voice_clone).
    """
    import numpy as np  # lazy — heavy import
    # Convert to mono if stereo
    if ref_audio.ndim > 1:
        ref_audio = np.mean(ref_audio, axis=-1).astype(np.float32)

    logger.info(
        "Creating voice prompt: %.1fs audio, %d char transcript",
        len(ref_audio) / ref_sr, len(transcript),
    )

    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=(ref_audio, ref_sr),
        ref_text=transcript,
    )
    return voice_prompt


# ---------------------------------------------------------------------------
# Register default backends (strategy pattern)
# ---------------------------------------------------------------------------

# Register torch backend
register_backend("torch", _run_inference_torch)

# Register MLX backend (will be available if MLX is installed)
# Note: The function exists regardless, but will fail at runtime if MLX not installed
register_backend("mlx", _run_inference_mlx)
