#!/usr/bin/env python3
"""TTS inference dispatch: stateless generation for torch and MLX backends.

Top-level orchestrator — imports from text_processing, audio_processing,
model_loader, voice_prompt, and config.
"""

import logging
import time

import numpy as np

from qwen3_tts.core.config import (
    CONFIG_PATH,
    get_backend,
    get_torch_dtype_name,
    load_config,
)
from qwen3_tts.core.engine.text_processing import _normalize_text, _split_text

logger = logging.getLogger("tts.engine")


# ---------------------------------------------------------------------------
# Torch backend — inference
# ---------------------------------------------------------------------------

def _run_inference_torch(model, text, mode, gen_params, language="English",
                         voice_prompt=None, voice_description=None,
                         speaker=None, instruct=None,
                         x_vector_only_mode=False):
    """Run TTS inference using the PyTorch/MPS backend."""
    import torch

    t0 = time.time()

    # Float32 guard: clone mode on MPS requires float32 to avoid NaN/Inf errors.
    # If a non-float32 dtype is configured, override for this call and warn.
    original_dtype = None
    if mode == "clone" and torch.backends.mps.is_available():
        dtype_name = get_torch_dtype_name()
        if dtype_name != "float32":
            logger.warning(
                "Clone mode on MPS requires float32 (configured: %s). "
                "Overriding to float32 for this generation. "
                "Set advanced.dtype to 'float32' in %s to silence this warning.",
                dtype_name, CONFIG_PATH,
            )
            # Save original dtype and cast model to float32 for this call
            original_dtype = next(model.parameters()).dtype
            model.float()

    params = {
        "temperature": gen_params.get("temperature", 0.7),
        "top_k": gen_params.get("top_k", 50),
        "top_p": gen_params.get("top_p", 0.95),
        "repetition_penalty": gen_params.get("repetition_penalty", 1.05),
        "max_new_tokens": gen_params.get("max_new_tokens", 2048),
    }

    seed = gen_params.get("seed")
    if seed is not None:
        torch.manual_seed(seed)

    try:
        with torch.inference_mode():
            if mode == "clone":
                clone_kwargs = dict(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_prompt,
                    **params,
                )
                if x_vector_only_mode:
                    clone_kwargs["x_vector_only_mode"] = True
                wavs, sr = model.generate_voice_clone(**clone_kwargs)
            elif mode == "custom":
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    speaker=speaker,
                    instruct=instruct or "",
                    language=language,
                    **params,
                )
            else:  # design
                wavs, sr = model.generate_voice_design(
                    text=text,
                    instruct=voice_description or "",
                    language=language,
                    **params,
                )
    except RuntimeError as e:
        if "inf" in str(e) or "nan" in str(e):
            dtype_name = get_torch_dtype_name()
            if dtype_name != "float32":
                logger.error(
                    "Generation produced NaN/Inf with dtype=%s. "
                    "Switch to float32 in %s under advanced.dtype for stability.",
                    dtype_name, CONFIG_PATH,
                )
            raise
        raise
    finally:
        # Restore original dtype if we overrode it
        if original_dtype is not None:
            model.to(original_dtype)

    # Device memory management
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            peak = torch.mps.current_allocated_memory()
            logger.debug(
                "MPS memory after generation: %.1f MB",
                peak / (1024 * 1024),
            )
        except RuntimeError as e:
            logger.debug("MPS memory cleanup failed: %s", e)
    elif torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_allocated()
            logger.debug(
                "CUDA memory after generation: %.1f MB",
                peak / (1024 * 1024),
            )
        except RuntimeError as e:
            logger.debug("CUDA memory cleanup failed: %s", e)

    elapsed = time.time() - t0
    logger.info(
        "Inference complete: %d chars, %.1fs, mode=%s [torch]",
        len(text), elapsed, mode,
    )

    wav = _validate_audio(wavs[0], sr, mode=mode)
    return wav, sr


def _validate_audio(wav, sample_rate, mode="unknown"):
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

def _run_inference_mlx(model, text, mode, gen_params, language="English",
                       voice_prompt=None, voice_description=None,
                       speaker=None, instruct=None,
                       x_vector_only_mode=False):
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
    config_gen = config.get("generation", {})
    params = {
        "temperature": gen_params.get("temperature", config_gen.get("temperature", 0.7)),
        "top_k": gen_params.get("top_k", config_gen.get("top_k", 50)),
        "top_p": gen_params.get("top_p", config_gen.get("top_p", 0.95)),
        "repetition_penalty": gen_params.get("repetition_penalty", config_gen.get("repetition_penalty", 1.05)),
        "max_new_tokens": gen_params.get("max_new_tokens", config_gen.get("max_new_tokens", 2048)),
    }

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
        results = list(model.generate_custom_voice(
            text=text,
            speaker=speaker or "Ryan",
            language=language,
            instruct=instruct or "",
            **params,
        ))

    else:  # design
        results = list(model.generate_voice_design(
            text=text,
            instruct=voice_description or "",
            language=language,
            **params,
        ))

    if not results:
        raise RuntimeError("MLX generation returned no results")

    # Collect audio from all segments and concatenate
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


def _run_inference_mlx_streaming(model, text, mode, gen_params, language="English",
                                  voice_prompt=None, voice_description=None,
                                  speaker=None, instruct=None,
                                  x_vector_only_mode=False):
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

    Yields:
        (audio_chunk, sample_rate) tuples where audio_chunk is a float32 numpy array.
    """
    import mlx.core as mx

    # Read defaults from config so MLX matches torch behavior (R-17)
    config = load_config()
    config_gen = config.get("generation", {})
    params = {
        "temperature": gen_params.get("temperature", config_gen.get("temperature", 0.7)),
        "top_k": gen_params.get("top_k", config_gen.get("top_k", 50)),
        "top_p": gen_params.get("top_p", config_gen.get("top_p", 0.95)),
        "repetition_penalty": gen_params.get("repetition_penalty", config_gen.get("repetition_penalty", 1.05)),
        "max_new_tokens": gen_params.get("max_new_tokens", config_gen.get("max_new_tokens", 2048)),
    }

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
        generator = model.generate_custom_voice(
            text=text,
            speaker=speaker or "Ryan",
            language=language,
            instruct=instruct or "",
            stream=True,  # Enable streaming
            **params,
        )

    else:  # design
        generator = model.generate_voice_design(
            text=text,
            instruct=voice_description or "",
            language=language,
            stream=True,  # Enable streaming
            **params,
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
        yield wav, sr

    logger.info("Streaming complete: %d chunks, mode=%s [mlx]", chunk_count, mode)


# ---------------------------------------------------------------------------
# Chunk combination (crossfade / silence gap)
# ---------------------------------------------------------------------------

def _crossfade_chunks(chunks, sample_rate, crossfade_ms=50, silence_gap_s=None):
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

def _get_max_chunk_chars():
    """Read max_chunk_chars from config, defaulting to 500."""
    try:
        config = load_config()
        return config.get("generation", {}).get("max_chunk_chars", 500)
    except Exception:
        return 500


def _get_max_chunk_tokens():
    """Read max_chunk_tokens from config, defaulting to 200."""
    try:
        config = load_config()
        return config.get("generation", {}).get("max_chunk_tokens", 200)
    except Exception:
        return 200


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def run_inference(model, text, mode, gen_params, language="English",
                  voice_prompt=None, voice_description=None,
                  speaker=None, instruct=None,
                  max_chunk_chars=None, progress_callback=None,
                  x_vector_only_mode=False):
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

    # Normalize text (expand numbers, dates, abbreviations) before chunking
    text = _normalize_text(text, language)

    # Resolve tokenizer for token-aware chunking (torch backend only)
    tokenizer = getattr(model, "tokenizer", None)
    max_tokens = _get_max_chunk_tokens() if tokenizer is not None else None

    # Split into chunks
    if tokenizer is not None and max_tokens is not None:
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count > max_tokens:
            chunks = _split_text(text, max_chars=max_chunk_chars,
                                 language=language, tokenizer=tokenizer,
                                 max_tokens=max_tokens)
        else:
            chunks = [text]
    elif max_chunk_chars > 0 and len(text) > max_chunk_chars:
        chunks = _split_text(text, max_chars=max_chunk_chars, language=language)
    else:
        chunks = [text]

    if len(chunks) == 1:
        # Single chunk — no overhead
        if progress_callback:
            progress_callback(0, 1)
        return _run_inference_single(
            model, chunks[0], mode, gen_params, language,
            voice_prompt, voice_description, speaker, instruct,
            x_vector_only_mode=x_vector_only_mode,
        )

    # Multi-chunk: generate each, combine with crossfade or silence gap
    logger.info("Splitting text (%d chars) into %d chunks (max %d chars each)",
                len(text), len(chunks), max_chunk_chars)

    all_audio = []
    sample_rate = None

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i, len(chunks))

        preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
        logger.info("Chunk %d/%d: '%s' (%d chars)", i + 1, len(chunks), preview, len(chunk))

        wav, sr = _run_inference_single(
            model, chunk, mode, gen_params, language,
            voice_prompt, voice_description, speaker, instruct,
            x_vector_only_mode=x_vector_only_mode,
        )

        if sample_rate is None:
            sample_rate = sr

        all_audio.append(wav)

    # Combine chunks: use silence_gap_seconds from config, or crossfade (default 50ms)
    config = load_config()
    silence_gap = config.get("generation", {}).get("silence_gap_seconds", 0.0)
    if silence_gap > 0:
        result = _crossfade_chunks(all_audio, sample_rate, crossfade_ms=0, silence_gap_s=silence_gap)
    else:
        result = _crossfade_chunks(all_audio, sample_rate, crossfade_ms=50)

    logger.info("Combined %d chunks into %.1fs audio", len(chunks), len(result) / sample_rate)
    return result, sample_rate


def _run_inference_single(model, text, mode, gen_params, language="English",
                          voice_prompt=None, voice_description=None,
                          speaker=None, instruct=None,
                          _metal_retry_depth=0, x_vector_only_mode=False):
    """Run TTS inference for a single text chunk.

    For MLX backend, includes Metal crash recovery: on certain Metal kernel
    errors, retries with smaller sub-chunks up to depth 2.
    """
    backend = get_backend()
    if backend == "mlx":
        try:
            return _run_inference_mlx(
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
                # Concatenate with short silence
                silence = np.zeros(int(sr * 0.1), dtype=np.float32)
                return np.concatenate([wav1, silence, wav2]), sr
            raise
    return _run_inference_torch(
        model, text, mode, gen_params, language,
        voice_prompt, voice_description, speaker, instruct,
        x_vector_only_mode=x_vector_only_mode,
    )


def run_inference_streaming(model, text, mode, gen_params, language="English",
                            voice_prompt=None, voice_description=None,
                            speaker=None, instruct=None,
                            max_chunk_chars=None, x_vector_only_mode=False):
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

    Yields:
        (audio_chunk, sample_rate) tuples where audio_chunk is float32 numpy array.
    """
    backend = get_backend()

    if backend == "mlx":
        # MLX has native streaming — yield chunks as they generate
        logger.info("Starting streaming inference [mlx]")
        yield from _run_inference_mlx_streaming(
            model, text, mode, gen_params, language,
            voice_prompt, voice_description, speaker, instruct,
            x_vector_only_mode=x_vector_only_mode,
        )
    else:
        # Torch fallback: chunk the text and yield per-chunk audio
        logger.info("Starting chunked streaming [torch fallback]")
        if max_chunk_chars is None:
            max_chunk_chars = _get_max_chunk_chars()

        text = _normalize_text(text, language)
        tokenizer = getattr(model, "tokenizer", None)
        max_tokens = _get_max_chunk_tokens() if tokenizer is not None else None

        if tokenizer is not None and max_tokens is not None:
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            if token_count > max_tokens:
                chunks = _split_text(text, max_chars=max_chunk_chars,
                                     language=language, tokenizer=tokenizer,
                                     max_tokens=max_tokens)
            else:
                chunks = [text]
        elif max_chunk_chars > 0 and len(text) > max_chunk_chars:
            chunks = _split_text(text, max_chars=max_chunk_chars, language=language)
        else:
            chunks = [text]

        for i, chunk in enumerate(chunks):
            preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
            logger.info("Streaming chunk %d/%d: '%s'", i + 1, len(chunks), preview)

            wav, sr = _run_inference_single(
                model, chunk, mode, gen_params, language,
                voice_prompt, voice_description, speaker, instruct,
                x_vector_only_mode=x_vector_only_mode,
            )
            yield wav, sr


def create_voice_prompt(model, ref_audio, ref_sr, transcript):
    """Create a reusable voice-clone prompt from reference audio.

    Args:
        model: Loaded clone (Base) model.
        ref_audio: numpy array of reference audio (mono).
        ref_sr: Sample rate of reference audio.
        transcript: Text transcript of the reference audio.

    Returns:
        Voice prompt tensor (suitable for torch.save / generate_voice_clone).
    """
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
