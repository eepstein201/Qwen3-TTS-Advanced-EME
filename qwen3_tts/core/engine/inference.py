#!/usr/bin/env python3
"""TTS inference dispatch: stateless generation for torch and MLX backends.

Top-level orchestrator — imports from text_processing, audio_processing,
model_loader, voice_prompt, and config.
"""

import logging
import time
from collections.abc import Callable, Iterator
from typing import Any

from qwen3_tts.core.config import (
    CONFIG_PATH,
    DefaultConfigLoader,
    get_backend,
    get_torch_dtype_name,
    load_config,
    sanitize_log,
)
from qwen3_tts.core.engine.audio_processing import (
    DEFAULT_SAMPLE_RATE,
    LUFS_TARGET,
    process_audio,
)
from qwen3_tts.core.engine.text_processing import _normalize_text, _split_text

logger = logging.getLogger("tts.engine")

_DEFAULT_CONFIG_LOADER = DefaultConfigLoader()

# ---------------------------------------------------------------------------
# Strategy registries for OCP-compliant dispatch
# ---------------------------------------------------------------------------

# Backend strategies: maps backend name -> inference function
_INFERENCE_STRATEGIES: dict[str, Any] = {}

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
        sanitize_log(dtype_name),
        sanitize_log(CONFIG_PATH),
    )
    original_dtype = next(model.parameters()).dtype
    model.float()
    return original_dtype


def _dispatch_torch_mode(
    model: Any,
    text: str,
    mode: str,
    language: str,
    params: dict,
    voice_prompt: Any,
    voice_description: str | None,
    speaker: str | None,
    instruct: str | None,
    x_vector_only_mode: bool,
) -> tuple:
    """Call the appropriate model method for the given mode. Returns (wavs, sr)."""
    import torch

    with torch.inference_mode():
        if mode == "clone":
            clone_kwargs = dict(
                text=text, language=language, voice_clone_prompt=voice_prompt, **params
            )
            if x_vector_only_mode:
                clone_kwargs["x_vector_only_mode"] = True
            return model.generate_voice_clone(**clone_kwargs)
        elif mode == "custom":
            return model.generate_custom_voice(
                text=text,
                speaker=speaker,
                instruct=instruct or "",
                language=language,
                **params,
            )
        else:  # design
            return model.generate_voice_design(
                text=text,
                instruct=voice_description or "",
                language=language,
                **params,
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


# The documented product default for `generation.max_new_tokens`
# (core/config/io.py, docs/CONFIG.md). Shared by both backends so the two
# param-merge helpers cannot drift apart.
_DEFAULT_MAX_NEW_TOKENS = 2048


def _build_torch_params(gen_params: dict) -> dict:
    """Merge caller gen_params with config defaults for torch backend."""
    config = load_config()
    config_gen = config.get("generation", {})
    return {
        "temperature": gen_params.get(
            "temperature", config_gen.get("temperature", 0.7)
        ),
        "top_k": gen_params.get("top_k", config_gen.get("top_k", 50)),
        "top_p": gen_params.get("top_p", config_gen.get("top_p", 0.95)),
        "repetition_penalty": gen_params.get(
            "repetition_penalty", config_gen.get("repetition_penalty", 1.05)
        ),
        "max_new_tokens": gen_params.get(
            "max_new_tokens",
            config_gen.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS),
        ),
    }


def _run_inference_torch(
    model: Any,
    text: str,
    mode: str,
    gen_params: dict,
    language: str = "auto",
    voice_prompt: Any = None,
    voice_description: str | None = None,
    speaker: str | None = None,
    instruct: str | None = None,
    x_vector_only_mode: bool = False,
) -> tuple:
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
            model,
            text,
            mode,
            language,
            params,
            voice_prompt,
            voice_description,
            speaker,
            instruct,
            x_vector_only_mode,
        )
    except RuntimeError as e:
        if "inf" in str(e) or "nan" in str(e):
            dtype_name = get_torch_dtype_name()
            if dtype_name != "float32":
                logger.error(
                    "Generation produced NaN/Inf with dtype=%s. "
                    "Switch to float32 in %s under advanced.dtype for stability.",
                    sanitize_log(dtype_name),
                    sanitize_log(CONFIG_PATH),
                )
        raise
    finally:
        if original_dtype is not None:
            try:
                model.to(original_dtype)
            except (RuntimeError, TypeError) as restore_err:
                logger.warning(
                    "Failed to restore model dtype after MPS guard: %s", restore_err
                )

    _cleanup_device_memory()

    if not wavs:
        raise RuntimeError("torch generation returned no audio segments")

    logger.info(
        "Inference complete: %d chars, %.1fs, mode=%s [torch]",
        len(text),
        time.time() - t0,
        mode,
    )
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
        logger.warning(
            "Generated audio contains NaN values (mode=%s), replacing with zeros", mode
        )
        wav = np.nan_to_num(wav, nan=0.0)

    # Clipping check
    peak = np.max(np.abs(wav))
    if peak > 1.0:
        logger.warning(
            "Generated audio is clipping (peak=%.2f, mode=%s), normalizing", peak, mode
        )
        wav = wav / peak

    # All-silence check
    if np.max(np.abs(wav)) < 1e-6:
        logger.warning("Generated audio is silent (mode=%s)", mode)

    return wav


# ---------------------------------------------------------------------------
# MLX backend — inference
# ---------------------------------------------------------------------------


# PRF-9 (docs/reviews/prf9-max-new-tokens-measurement-2026-08-15.md) measured
# MLX caps >= 8192 as NO-GO: non-deterministic EOS-failure runaway loops plus
# memory exhaustion (13.5 GB active at 8192; 16.5 GB over-commit at 16384 on a
# 16 GB machine). mlx-audio's own default, 4096, is the highest validated cap.
# The request schema still permits up to 8192 for other backends, so the MLX
# ceiling is enforced here, where the backend knowledge lives.
_MLX_MAX_TOKENS_CEILING = 4096


def _split_mlx_params(params: dict) -> tuple[dict, int]:
    """Split merged gen params into mlx-audio sampling kwargs + the token cap.

    Our key is ``max_new_tokens``; every mlx-audio entry point calls it
    ``max_tokens``. ``generate_custom_voice``/``generate_voice_design`` take no
    ``**kwargs``, so the key must be *renamed* rather than added — passing the
    old name raises TypeError there, and on ``generate()`` it is silently
    swallowed (the original bug).

    Returns a NEW dict; ``params`` is never mutated.
    """
    sampling = {k: v for k, v in params.items() if k != "max_new_tokens"}

    requested = params.get("max_new_tokens")
    if requested is None:
        # Absent or explicitly None: fall back to the documented product
        # default, NOT the ceiling — PRF-9 established that higher is the risk
        # direction, so "unspecified" must not mean "maximum permitted".
        return sampling, _DEFAULT_MAX_NEW_TOKENS

    # Type-normalize before comparing. `GenerateRequest` coerces this field, but
    # `validate_config()` type-checks only `generation.temperature`, so a
    # hand-edited config.json ("max_new_tokens": "4096") reaches us as a str and
    # would raise TypeError on the comparison below — breaking every MLX
    # generation. bool is excluded explicitly: it is an int subclass, and
    # `range(True)` would yield a silent 1-token generation.
    if isinstance(requested, bool) or not isinstance(requested, (int, float)):
        logger.warning(
            "max_new_tokens=%r is not a number; using the default of %d",
            requested,
            _DEFAULT_MAX_NEW_TOKENS,
        )
        return sampling, _DEFAULT_MAX_NEW_TOKENS
    requested = int(requested)

    if requested > _MLX_MAX_TOKENS_CEILING:
        logger.warning(
            "max_new_tokens=%d exceeds the MLX ceiling of %d; clamping "
            "(PRF-9 measured >=8192 as unstable on 16 GB)",
            requested,
            _MLX_MAX_TOKENS_CEILING,
        )
        return sampling, _MLX_MAX_TOKENS_CEILING

    if requested < 1:
        # mlx-audio loops `for step in range(max_tokens)`, so <=0 yields an
        # empty generation rather than an error. The request schema enforces
        # ge=1, but direct engine callers are not bound by it.
        logger.warning(
            "max_new_tokens=%d is below 1; using the default of %d",
            requested,
            _DEFAULT_MAX_NEW_TOKENS,
        )
        return sampling, _DEFAULT_MAX_NEW_TOKENS

    return sampling, requested


def _warn_if_cap_reached(results: Any, max_tokens: int, mode: str) -> None:
    """Report generations that stopped at the cap instead of on EOS.

    mlx-audio's loop is a bare ``for step in range(max_tokens)`` with no
    exhaustion signal, and ``_validate_audio`` only checks clipping/silence —
    so truncation is otherwise silent end to end: a clean 200 carrying half a
    sentence, or minutes of looped audio when EOS never fires.

    ``token_count`` is ``len(generated_codes)`` upstream. It is read
    defensively so a result object without the field degrades to silence
    rather than raising.
    """
    for result in results or ():
        count = getattr(result, "token_count", None)
        if isinstance(count, int) and not isinstance(count, bool) and count >= max_tokens:
            _warn_cap_reached(max_tokens, mode)
            return


def _warn_cap_reached(max_tokens: int, mode: str) -> None:
    """Emit the truncation warning. Shared by the batch and streaming paths."""
    logger.warning(
        "Generation hit the %d-token cap without emitting EOS "
        "(mode=%s) — the audio is truncated. For clone mode the usual "
        "cause is a voice prompt whose reference .wav is below %d Hz; "
        "re-create the prompt from higher-rate audio.",
        max_tokens,
        mode,
        DEFAULT_SAMPLE_RATE,
    )


def _mlx_lang_code(language: str | None, model: Any = None) -> str:
    """Map our language name onto mlx-audio's language code.

    mlx-audio lowercases the value and looks it up in ``codec_language_id``.
    An unrecognized value does NOT raise — it simply drops the language
    conditioning token — so an unsupported language would otherwise fail
    silently. It also does not *strip*, so a padded "  English  " would miss
    the lookup; normalizing here covers both.

    The supported set is per-checkpoint (mlx-audio builds it from the loaded
    talker config), so it is read off the model rather than hardcoded.
    """
    code = (language or "auto").strip().lower()

    if code == "auto":
        # Never looked up: both backends short-circuit "auto" before touching
        # codec_language_id, so it can never be unsupported.
        return code

    supported = getattr(model, "supported_languages", None)
    # mlx-audio deliberately omits every "*_dialect" key when it builds
    # supported_languages, but its generation-time lookup still consults the
    # full codec_language_id map — so dialect codes DO work and must not be
    # reported as unsupported (the project ships Beijing/Chengdu speakers).
    if supported and code not in supported and "dialect" not in code:
        logger.warning(
            "language %s has no mlx-audio match; generating without language "
            "conditioning (model supports: %s)",
            sanitize_log(str(language)),
            ", ".join(sorted(supported)),
        )
    return code


def _get_mlx_gen_params(gen_params: dict, config: dict) -> dict:
    """Merge caller gen_params with config defaults for MLX backend."""
    config_gen = config.get("generation", {})
    return {
        "temperature": gen_params.get(
            "temperature", config_gen.get("temperature", 0.7)
        ),
        "top_k": gen_params.get("top_k", config_gen.get("top_k", 50)),
        "top_p": gen_params.get("top_p", config_gen.get("top_p", 0.95)),
        "repetition_penalty": gen_params.get(
            "repetition_penalty", config_gen.get("repetition_penalty", 1.05)
        ),
        "max_new_tokens": gen_params.get(
            "max_new_tokens",
            config_gen.get("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS),
        ),
    }


def _run_inference_mlx(
    model: Any,
    text: str,
    mode: str,
    gen_params: dict,
    language: str = "auto",
    voice_prompt: Any = None,
    voice_description: str | None = None,
    speaker: str | None = None,
    instruct: str | None = None,
    x_vector_only_mode: bool = False,
) -> tuple:
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
    sampling, max_tokens = _split_mlx_params(params)

    seed = gen_params.get("seed")
    if seed is not None:
        _set_seed_for_backend(seed)

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

        results = list(
            model.generate(
                text=text,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                lang_code=_mlx_lang_code(language, model),
                max_tokens=max_tokens,
                **sampling,
            )
        )

    elif mode == "custom":
        results = list(
            model.generate_custom_voice(
                text=text,
                speaker=speaker or "Ryan",
                language=_mlx_lang_code(language, model),
                instruct=instruct or "",
                max_tokens=max_tokens,
                **sampling,
            )
        )

    else:  # design
        results = list(
            model.generate_voice_design(
                text=text,
                instruct=voice_description or "",
                language=_mlx_lang_code(language, model),
                max_tokens=max_tokens,
                **sampling,
            )
        )

    if not results:
        raise RuntimeError("MLX generation returned no results")

    _warn_if_cap_reached(results, max_tokens, mode)

    # Collect audio from all segments and concatenate
    import mlx.core as mx
    import numpy as np  # lazy — heavy import

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
        len(text),
        elapsed,
        mode,
    )

    wav = _validate_audio(wav, sr, mode=mode)
    return wav, sr


def _run_inference_mlx_streaming(
    model: Any,
    text: str,
    mode: str,
    gen_params: dict,
    language: str = "auto",
    voice_prompt: Any = None,
    voice_description: str | None = None,
    speaker: str | None = None,
    instruct: str | None = None,
    x_vector_only_mode: bool = False,
    config: dict | None = None,
    progress_callback: Any = None,
) -> Iterator[tuple]:
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
    sampling, max_tokens = _split_mlx_params(params)

    seed = gen_params.get("seed")
    if seed is not None:
        _set_seed_for_backend(seed)

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
            lang_code=_mlx_lang_code(language, model),
            max_tokens=max_tokens,
            stream=True,  # Enable streaming
            **sampling,
        )

    elif mode == "custom":
        generator = model.generate_custom_voice(
            text=text,
            speaker=speaker or "Ryan",
            language=_mlx_lang_code(language, model),
            instruct=instruct or "",
            max_tokens=max_tokens,
            stream=True,  # Enable streaming
            **sampling,
        )

    else:  # design
        generator = model.generate_voice_design(
            text=text,
            instruct=voice_description or "",
            language=_mlx_lang_code(language, model),
            max_tokens=max_tokens,
            stream=True,  # Enable streaming
            **sampling,
        )

    chunk_count = 0
    # Streaming yields per-chunk token DELTAS, never a cumulative count:
    # mlx-audio's streaming branch returns before reaching the
    # `token_count = len(generated_codes)` yield, so each result carries
    # `new_tokens`, bounded by `streaming_chunk_size` (25 at the default 2.0 s
    # interval). Checking a single result against `max_tokens` is therefore
    # never true, which left cap runaways silent on /generate-stream and /ws.
    #
    # The cap is applied PER SEGMENT upstream — `generate()` resets
    # `generated_codes` and `decoded_tokens` inside its segment loop — so the
    # deltas are accumulated per `segment_idx`. Summing the whole stream would
    # turn a multi-segment generation into a false positive.
    tokens_by_segment: dict[int, int] = {}
    for result in generator:
        audio_mx = result.audio
        wav = np.array(audio_mx, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.squeeze()
        sr = result.sample_rate
        chunk_count += 1
        logger.debug("Streaming chunk %d: %d samples", chunk_count, len(wav))

        _tc = getattr(result, "token_count", None)
        if isinstance(_tc, int) and not isinstance(_tc, bool):
            _seg = getattr(result, "segment_idx", 0)
            if not isinstance(_seg, int) or isinstance(_seg, bool):
                _seg = 0
            tokens_by_segment[_seg] = tokens_by_segment.get(_seg, 0) + _tc

        # Call progress callback (total unknown until completion for MLX)
        if progress_callback:
            progress_callback(chunk_count, 0)  # 0 = unknown total

        yield wav, sr

    if tokens_by_segment and max(tokens_by_segment.values()) >= max_tokens:
        _warn_cap_reached(max_tokens, mode)

    # After completion, update with final count
    if progress_callback and chunk_count > 0:
        progress_callback(chunk_count, chunk_count)

    logger.info("Streaming complete: %d chunks, mode=%s [mlx]", chunk_count, mode)


# ---------------------------------------------------------------------------
# Chunk combination (crossfade / silence gap)
# ---------------------------------------------------------------------------


# PRF-2: how far into an incoming chunk we may trim to reach a sign change.
# 2 ms is under one period of any speech fundamental, so the splice never
# skips audible content.
_ZERO_CROSS_SEARCH_MS = 2.0

# PRF-2: lag window searched when phase-aligning a crossfaded splice. 10 ms is
# a full period at 100 Hz, so it spans the phase of any normal speech
# fundamental; the trimmed audio sits at a chunk boundary.
_ALIGN_MAX_LAG_MS = 10.0

# Correlation window used to score alignment (~5 ms of the outgoing tail).
_ALIGN_CORR_SAMPLES = 120

# PRF-2: bound on the per-seam level correction. Chunks are sentences of one
# utterance, so a few dB covers generator drift; clamping keeps a genuinely
# quiet chunk from being pumped up and stops gain drifting across many seams.
_SEAM_MAX_GAIN_DB = 3.0


def _snap_to_zero_crossing(head: Any, max_search: int) -> int:
    """Return the offset of the first sign change within ``head[:max_search]``.

    Splicing at a sign change means both sides of the seam start near zero
    amplitude, which removes the step discontinuity an arbitrary splice point
    would leave. Returns 0 when there is no crossing in the window (e.g. a DC
    or near-silent head), so the caller trims nothing.
    """
    import numpy as np  # lazy — heavy import

    n = int(min(max_search, len(head)))
    if n < 2:
        return 0
    signs = np.signbit(head[:n])
    changes = np.flatnonzero(signs[:-1] != signs[1:])
    if changes.size == 0:
        return 0
    return int(changes[0]) + 1


def _align_offset(
    tail: Any, head: Any, max_lag: int, corr_samples: int = _ALIGN_CORR_SAMPLES
) -> int:
    """Return the lag into ``head`` that best phase-aligns it with ``tail``.

    Picks the offset maximising normalised cross-correlation between the end
    of ``tail`` and a same-length window of ``head``. Snapping to a *zero
    crossing* is not enough here: it ignores the crossing direction and the
    outgoing phase, so it can land anti-phase and deepen the cancellation dip
    the crossfade then bakes in (measured: a quarter-period offset gets worse,
    0.87 -> 0.71 of reference RMS). Correlation aligns to the outgoing signal
    directly. Returns 0 when either side is too short or effectively silent.
    """
    import numpy as np  # lazy — heavy import

    if max_lag <= 0:
        return 0
    ref_len = int(min(corr_samples, len(tail)))
    if ref_len < 8:
        return 0
    max_lag = int(min(max_lag, len(head) - ref_len))
    if max_lag <= 0:
        return 0

    ref = np.asarray(tail[-ref_len:], dtype=np.float64)
    ref_norm = float(np.linalg.norm(ref))
    if ref_norm <= 1e-9:
        return 0
    seg = np.asarray(head[: ref_len + max_lag], dtype=np.float64)

    dots = np.correlate(seg, ref, mode="valid")
    cumsq = np.concatenate([[0.0], np.cumsum(seg**2)])
    window_energy = cumsq[ref_len:] - cumsq[:-ref_len]
    norms = np.sqrt(np.maximum(window_energy, 1e-18))
    scores = dots / (norms * ref_norm)

    # Voiced speech is periodic, so many lags align equally well (a 200 Hz
    # sine repeats every 120 samples). Take the earliest lag that is
    # essentially as good as the best so we discard as little audio as
    # possible instead of jumping a whole extra period.
    best = float(scores.max())
    good = np.flatnonzero(scores >= best - 1e-3)
    return int(good[0]) if good.size else int(np.argmax(scores))


def _seam_gain(tail: Any, head: Any, max_db: float = _SEAM_MAX_GAIN_DB) -> float:
    """Bounded gain that RMS-matches ``head`` to ``tail`` across a seam.

    Independently generated chunks drift in level, which reads as a step at
    the splice. Returns 1.0 for empty or effectively silent input so digital
    silence can't divide by zero or explode the gain.
    """
    import numpy as np  # lazy — heavy import

    if len(tail) == 0 or len(head) == 0:
        return 1.0
    tail_rms = float(np.sqrt(np.mean(np.asarray(tail, dtype=np.float64) ** 2)))
    head_rms = float(np.sqrt(np.mean(np.asarray(head, dtype=np.float64) ** 2)))
    if tail_rms <= 1e-6 or head_rms <= 1e-6:
        return 1.0
    limit = 10.0 ** (max_db / 20.0)
    return float(min(max(tail_rms / head_rms, 1.0 / limit), limit))


def _crossfade_chunks(
    chunks: list,
    sample_rate: int,
    crossfade_ms: int = 50,
    silence_gap_s: float | None = None,
) -> Any:
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
    search_samples = int(sample_rate * _ZERO_CROSS_SEARCH_MS / 1000)
    max_lag_samples = int(sample_rate * _ALIGN_MAX_LAG_MS / 1000)
    combined = chunks[0].copy()
    for chunk in chunks[1:]:
        overlap = min(fade_samples, len(combined), len(chunk))
        if overlap <= 0:
            # Hard splice (no room to fade): snap to a sign change so both
            # sides meet near zero and the join doesn't click.
            offset = _snap_to_zero_crossing(chunk, search_samples)
            combined = np.concatenate([combined, chunk[offset:] if offset else chunk])
            continue
        # PRF-2: align phase and level *before* the raised cosine — the fade
        # shape is already right for correlated speech; the seam artifact
        # comes from splicing at an arbitrary phase and from level drift.
        lag = _align_offset(combined, chunk, max_lag_samples)
        if lag:
            chunk = chunk[lag:]
            overlap = min(fade_samples, len(combined), len(chunk))
            if overlap <= 0:
                combined = np.concatenate([combined, chunk])
                continue
        gain = _seam_gain(combined[-overlap:], chunk[:overlap])
        if gain != 1.0:
            chunk = (chunk * gain).astype(np.float32)
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


# PRF-6: clone mode ignores the model's own rate control (upstream #290 —
# output lands at 41-48 s whatever is requested), so rate is applied after
# generation. Bounds keep a fat-fingered factor from destroying the audio
# instead of speeding it up.
_CLONE_SPEED_MIN = 0.5
_CLONE_SPEED_MAX = 2.0


def _resolve_clone_speed(gen_params: dict, config: dict | None) -> float | None:
    """Return the clone rate factor to apply, or None when there is nothing to do.

    Precedence: an explicit ``gen_params["speed"]`` beats
    ``generation.clone_speed`` in config. Unusable values are ignored with a
    warning rather than failing the generation.
    """
    speed = (gen_params or {}).get("speed")
    if speed is None:
        cfg = config if config is not None else load_config()
        speed = cfg.get("generation", {}).get("clone_speed")
    if speed is None:
        return None

    try:
        speed = float(speed)
    except (TypeError, ValueError):
        logger.warning("Ignoring non-numeric clone speed: %s", sanitize_log(speed))
        return None
    if speed <= 0:
        logger.warning("Ignoring non-positive clone speed: %s", speed)
        return None
    if speed == 1.0:
        return None

    clamped = min(max(speed, _CLONE_SPEED_MIN), _CLONE_SPEED_MAX)
    if clamped != speed:
        logger.warning("Clamped clone speed %s to %s", speed, clamped)
    return clamped


def _maybe_apply_speed(audio, sample_rate, gen_params, mode, config=None):
    """Apply post-hoc rate control for clone mode (PRF-6).

    Design and custom modes keep the model's native ``instruct`` rate control,
    so they are left alone — stretching them here would apply it twice.
    Returns (audio, sample_rate); the sample rate never changes.
    """
    if mode != "clone":
        return audio, sample_rate

    speed = _resolve_clone_speed(gen_params, config)
    if speed is None:
        return audio, sample_rate

    try:
        return process_audio(audio, sample_rate, speed=speed), sample_rate
    except Exception as e:
        # A missing rubberband CLI or a librosa failure must not lose audio we
        # already spent minutes generating.
        logger.warning(
            "Clone rate control failed, returning unstretched audio: %s",
            sanitize_log(e),
        )
        return audio, sample_rate


# PRF-8: ICL cloning sometimes re-speaks the tail of the reference transcript
# before the requested text (upstream #341). Probe only the head — the echo is
# a prefix, and transcribing the whole output would cost as much as generating
# it.
_ICL_PROBE_SECONDS = 4.0
_ICL_MIN_ECHO_WORDS = 3
# Never clip more than this fraction of the output, whatever ASR claims.
_ICL_MAX_TRIM_FRACTION = 0.5


def _normalize_for_match(text: str | None) -> list[str]:
    """Lowercase, drop punctuation, split to words — for loose text comparison."""
    if not text:
        return []
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return cleaned.split()


def _detect_echo_prefix(
    head_text: str | None,
    reference_text: str | None,
    min_words: int = _ICL_MIN_ECHO_WORDS,
) -> str | None:
    """Return the echoed phrase when the output head repeats the reference tail.

    Matches the *longest* suffix of the reference that the head starts with, so
    a long echo isn't under-trimmed. Requires ``min_words`` words so an
    incidental shared word isn't mistaken for an echo.
    """
    head = _normalize_for_match(head_text)
    reference = _normalize_for_match(reference_text)
    if len(head) < min_words or len(reference) < min_words:
        return None

    limit = min(len(head), len(reference))
    for size in range(limit, min_words - 1, -1):
        if reference[-size:] == head[:size]:
            return " ".join(head[:size])
    return None


def _find_silence_boundary(audio, sample_rate, near_sample, window_s: float = 0.4):
    """Index of the quietest short frame near ``near_sample``.

    Cutting on the pause after an echo avoids clipping mid-word. Falls back to
    ``near_sample`` when the region has no clear gap.
    """
    import numpy as np  # lazy — heavy import

    n = len(audio)
    if n == 0:
        return 0
    near_sample = int(min(max(near_sample, 0), n))

    half = int(window_s * sample_rate)
    start = max(0, near_sample - half)
    end = min(n, near_sample + half)
    if end - start < 2:
        return near_sample

    frame = max(1, int(0.01 * sample_rate))  # 10 ms
    region = np.asarray(audio[start:end], dtype=np.float64)
    usable = (len(region) // frame) * frame
    if usable < frame:
        return near_sample

    frames = region[:usable].reshape(-1, frame)
    energies = np.sqrt(np.mean(frames**2, axis=1))
    return start + int(np.argmin(energies)) * frame


def _transcribe_probe(audio, sample_rate) -> str:
    """Transcribe an in-memory probe with the existing ASR.

    transcribe_audio() takes a path, so the probe goes through a temp WAV.
    """
    import os
    import tempfile
    import wave

    import numpy as np

    from qwen3_tts.core.engine.asr import transcribe_audio

    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="icl_probe_")
    os.close(fd)
    try:
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(pcm.tobytes())
        return transcribe_audio(path) or ""
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# Attribute names a voice-prompt object may use for its reference transcript.
# The prompt is built by the upstream model, so this is best-effort: callers
# that know the transcript should pass reference_text explicitly.
_PROMPT_TEXT_ATTRS = ("transcript", "text", "prompt_text", "ref_text", "reference_text")


def _reference_text_from_prompt(voice_prompt) -> str | None:
    """Best-effort read of the reference transcript off a voice prompt."""
    if voice_prompt is None:
        return None
    if isinstance(voice_prompt, dict):
        candidates = (voice_prompt.get(key) for key in _PROMPT_TEXT_ATTRS)
    else:
        candidates = (getattr(voice_prompt, attr, None) for attr in _PROMPT_TEXT_ATTRS)
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _trim_icl_echo(
    audio,
    sample_rate,
    reference_text,
    mode,
    x_vector_only_mode,
    config=None,
):
    """Clip a reference-transcript echo from the head of cloned output (PRF-8).

    Returns (audio, sample_rate). Any failure returns the audio untouched — a
    missed trim is a cosmetic problem, losing a generation is not.
    """
    if mode != "clone" or x_vector_only_mode or not reference_text:
        return audio, sample_rate

    cfg = config if config is not None else load_config()
    if not cfg.get("generation", {}).get("trim_icl_echo", True):
        return audio, sample_rate

    from qwen3_tts.core.engine import asr

    # Only opportunistic: pulling a heavy ASR model into a generation that
    # never asked for one would cost more than the artifact it removes.
    if not asr.is_asr_loaded():
        return audio, sample_rate

    try:
        probe_len = min(len(audio), int(_ICL_PROBE_SECONDS * sample_rate))
        if probe_len <= 0:
            return audio, sample_rate

        head_text = _transcribe_probe(audio[:probe_len], sample_rate)
        echo = _detect_echo_prefix(head_text, reference_text)
        if not echo:
            return audio, sample_rate

        # No word timestamps available, so estimate the echo's share of the
        # probe by character count, then snap to the pause that follows it.
        head_words = _normalize_for_match(head_text)
        echo_words = _normalize_for_match(echo)
        if not head_words:
            return audio, sample_rate
        share = min(1.0, len(echo_words) / len(head_words))
        estimate = int(probe_len * share)

        cut = _find_silence_boundary(audio, sample_rate, estimate)
        max_cut = int(len(audio) * _ICL_MAX_TRIM_FRACTION)
        cut = min(cut, max_cut)
        if cut <= 0:
            return audio, sample_rate

        logger.info(
            "Trimmed %.2fs ICL reference echo from cloned output: %s",
            cut / sample_rate,
            sanitize_log(echo),
        )
        return audio[cut:], sample_rate
    except Exception as e:
        logger.warning(
            "ICL echo trim skipped (%s) — returning untrimmed audio",
            sanitize_log(e),
        )
        return audio, sample_rate


def _maybe_apply_lufs(audio, sample_rate, config=None):
    """Apply LUFS normalization if generation.lufs_normalize is True.

    Reads target loudness from generation.lufs_target (default LUFS_TARGET = -16.0).
    Returns (audio, sample_rate) tuple regardless of whether normalization was applied.
    """
    cfg = config if config is not None else load_config()
    gen = cfg.get("generation", {})
    if not gen.get("lufs_normalize", False):
        return audio, sample_rate
    target = gen.get("lufs_target", LUFS_TARGET)
    # process_audio returns the array only — callers unpack (audio, sample_rate),
    # so returning it bare made every generation raise "too many values to
    # unpack" as soon as lufs_normalize was turned on.
    return process_audio(audio, sample_rate, lufs_target=target), sample_rate


def _prepare_text_chunks(
    text: str, language: str, model, max_chunk_chars: int
) -> list[str]:
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
                text,
                max_chars=max_chunk_chars,
                language=language,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
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


def _postprocess_chunk(
    audio,
    sample_rate,
    gen_params,
    mode,
    config,
    reference_text=None,
    x_vector_only_mode=False,
):
    """Per-chunk post-processing shared by the batch and streaming paths (WS2).

    Applies every step that is computable from a single chunk, in the order the
    batch path has always used: ICL echo trim (it removes generated content, so
    it must run first), then clone rate control, then audio validation.

    ``_maybe_apply_lufs`` is deliberately NOT here. EBU R128 integrated loudness
    applies a relative gate derived from the block statistics of the *whole*
    signal, so it cannot be computed incrementally over chunks. The batch path
    applies it once after combining; the streaming paths skip it. That
    divergence is architectural, not an oversight — see
    tests/test_engine_streaming.py::test_lufs_stays_batch_only.

    Returns (audio, sample_rate); the sample rate never changes.
    """
    audio, sample_rate = _trim_icl_echo(
        audio,
        sample_rate,
        reference_text,
        mode,
        x_vector_only_mode,
        config=config,
    )
    audio, sample_rate = _maybe_apply_speed(
        audio, sample_rate, gen_params, mode, config=config
    )
    audio = _validate_audio(audio, sample_rate, mode=mode)
    return audio, sample_rate


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------


def run_inference(
    model: Any,
    text: str,
    mode: str,
    gen_params: dict,
    language: str = "auto",
    voice_prompt: Any = None,
    voice_description: str | None = None,
    speaker: str | None = None,
    instruct: str | None = None,
    max_chunk_chars: int | None = None,
    progress_callback: Any = None,
    x_vector_only_mode: bool = False,
    config_provider: Any = None,
    seed_lock_chunks: bool = False,
    reference_text: str | None = None,
) -> tuple:
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
            model,
            chunks[0],
            mode,
            gen_params,
            language,
            voice_prompt,
            voice_description,
            speaker,
            instruct,
            x_vector_only_mode=x_vector_only_mode,
        )
        config = (config_provider or _DEFAULT_CONFIG_LOADER).load()
        # Shared per-chunk steps, then loudness — LUFS is batch-only and must
        # measure the audio that actually ships.
        wav, sr = _postprocess_chunk(
            wav,
            sr,
            gen_params,
            mode,
            config,
            reference_text=reference_text or _reference_text_from_prompt(voice_prompt),
            x_vector_only_mode=x_vector_only_mode,
        )
        wav, sr = _maybe_apply_lufs(wav, sr, config=config)
        return wav, sr

    # Multi-chunk: generate each, combine with crossfade or silence gap
    logger.info(
        "Splitting text (%d chars) into %d chunks (max %d chars each)",
        len(text),
        len(chunks),
        max_chunk_chars,
    )

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
        logger.info(
            "Chunk %d/%d: '%s' (%d chars)",
            i + 1,
            len(chunks),
            sanitize_log(preview),
            len(chunk),
        )

        wav, sr = _run_inference_single(
            model,
            chunk,
            mode,
            gen_params,
            language,
            voice_prompt,
            voice_description,
            speaker,
            instruct,
            x_vector_only_mode=x_vector_only_mode,
        )

        if sample_rate is None:
            sample_rate = sr

        all_audio.append(wav)

    # At least one chunk was processed, so the sample rate is set by now.
    if sample_rate is None:
        raise RuntimeError("no audio chunks were produced")

    # Combine chunks: use silence_gap_seconds from config, or crossfade (default 50ms)
    config = (config_provider or _DEFAULT_CONFIG_LOADER).load()
    silence_gap = config.get("generation", {}).get("silence_gap_seconds", 0.0)
    if silence_gap > 0:
        result = _crossfade_chunks(
            all_audio, sample_rate, crossfade_ms=0, silence_gap_s=silence_gap
        )
    else:
        result = _crossfade_chunks(all_audio, sample_rate, crossfade_ms=50)

    # Shared per-chunk steps run on the combined audio here (echo trim targets
    # the head of the whole generation), then loudness — LUFS is batch-only and
    # must measure the audio that actually ships.
    result, sample_rate = _postprocess_chunk(
        result,
        sample_rate,
        gen_params,
        mode,
        config,
        reference_text=reference_text or _reference_text_from_prompt(voice_prompt),
        x_vector_only_mode=x_vector_only_mode,
    )
    result, sample_rate = _maybe_apply_lufs(result, sample_rate, config=config)

    logger.info(
        "Combined %d chunks into %.1fs audio", len(chunks), len(result) / sample_rate
    )
    return result, sample_rate


def _run_inference_single(
    model: Any,
    text: str,
    mode: str,
    gen_params: dict,
    language: str = "auto",
    voice_prompt: Any = None,
    voice_description: str | None = None,
    speaker: str | None = None,
    instruct: str | None = None,
    _metal_retry_depth: int = 0,
    x_vector_only_mode: bool = False,
) -> tuple:
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
                model,
                text,
                mode,
                gen_params,
                language,
                voice_prompt,
                voice_description,
                speaker,
                instruct,
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
                    _metal_retry_depth,
                    str(e)[:100],
                )
                # Split the chunk in half and process each sub-chunk
                mid = len(text) // 2
                # Find a space near the midpoint to avoid splitting mid-word
                split_idx = text.rfind(" ", mid - 50, mid + 50)
                if split_idx == -1:
                    split_idx = mid
                chunk1, chunk2 = text[:split_idx].strip(), text[split_idx:].strip()

                wav1, sr = _run_inference_single(
                    model,
                    chunk1,
                    mode,
                    gen_params,
                    language,
                    voice_prompt,
                    voice_description,
                    speaker,
                    instruct,
                    _metal_retry_depth=_metal_retry_depth + 1,
                    x_vector_only_mode=x_vector_only_mode,
                )
                wav2, _ = _run_inference_single(
                    model,
                    chunk2,
                    mode,
                    gen_params,
                    language,
                    voice_prompt,
                    voice_description,
                    speaker,
                    instruct,
                    _metal_retry_depth=_metal_retry_depth + 1,
                    x_vector_only_mode=x_vector_only_mode,
                )
                # Concatenate with short silence (use config gap, min 50ms)
                _retry_cfg = _DEFAULT_CONFIG_LOADER.load()
                _gap_s = _retry_cfg.get("generation", {}).get(
                    "silence_gap_seconds", 0.1
                )
                silence = np.zeros(int(sr * max(_gap_s, 0.05)), dtype=np.float32)
                return np.concatenate([wav1, silence, wav2]), sr
            raise

    # Default: use strategy from registry
    return strategy(
        model,
        text,
        mode,
        gen_params,
        language,
        voice_prompt,
        voice_description,
        speaker,
        instruct,
        x_vector_only_mode=x_vector_only_mode,
    )


def run_inference_streaming(
    model: Any,
    text: str,
    mode: str,
    gen_params: dict,
    language: str = "auto",
    voice_prompt: Any = None,
    voice_description: str | None = None,
    speaker: str | None = None,
    instruct: str | None = None,
    max_chunk_chars: int | None = None,
    x_vector_only_mode: bool = False,
    config_provider: Any = None,
    progress_callback: Any = None,
) -> Iterator[tuple]:
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
        # Apply text chunking for MLX streaming, same as the batch path.
        # Without this, the full text is sent to model.generate() in one shot;
        # max_new_tokens=2048 caps audio at ~170 s (12 Hz), so long texts are
        # silently truncated. Chunking at ≤500 chars keeps each call well within
        # the model's output window.
        if max_chunk_chars is None:
            max_chunk_chars = _get_max_chunk_chars()
        chunks = _prepare_text_chunks(text, language, model, max_chunk_chars)
        chunk_total = len(chunks)
        logger.info(
            "Starting streaming inference [mlx]: %d text chunk(s)", chunk_total
        )
        ref_text = _reference_text_from_prompt(voice_prompt)
        emitted = 0
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i + 1, chunk_total)
            for wav_chunk, chunk_sr in _run_inference_mlx_streaming(
                model,
                chunk,
                mode,
                gen_params,
                language,
                voice_prompt,
                voice_description,
                speaker,
                instruct,
                x_vector_only_mode=x_vector_only_mode,
                config=config,
                progress_callback=None,  # text-chunk progress reported above
            ):
                wav_chunk, chunk_sr = _postprocess_chunk(
                    wav_chunk,
                    chunk_sr,
                    gen_params,
                    mode,
                    config,
                    # An ICL echo sits at the head of the generation, so only
                    # the first emitted chunk can contain one.
                    reference_text=ref_text if emitted == 0 else None,
                    x_vector_only_mode=x_vector_only_mode,
                )
                emitted += 1
                yield wav_chunk, chunk_sr
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
                model,
                chunk,
                mode,
                gen_params,
                language,
                voice_prompt,
                voice_description,
                speaker,
                instruct,
                x_vector_only_mode=x_vector_only_mode,
            )
            wav, sr = _postprocess_chunk(
                wav,
                sr,
                gen_params,
                mode,
                config,
                # Echo trim applies to the head of the generation only.
                reference_text=(
                    _reference_text_from_prompt(voice_prompt) if i == 0 else None
                ),
                x_vector_only_mode=x_vector_only_mode,
            )
            yield wav, sr


def create_voice_prompt(
    model: Any,
    ref_audio: Any,
    ref_sr: int,
    transcript: str,
    x_vector_only_mode: bool = False,
) -> Any:
    """Create a reusable voice-clone prompt from reference audio.

    Args:
        model: Loaded clone (Base) model.
        ref_audio: numpy array of reference audio (mono).
        ref_sr: Sample rate of reference audio.
        transcript: Text transcript of the reference audio. Ignored by the
            upstream API when x_vector_only_mode=True (speaker-embedding-only).
        x_vector_only_mode: If True, build a transcript-free prompt using only
            the speaker embedding (x-vector). ref_text may be empty; upstream
            sets ref_code=None and stores the flag inside the prompt so later
            generation runs in x-vector-only mode.

    Returns:
        Voice prompt tensor (suitable for torch.save / generate_voice_clone).
    """
    import numpy as np  # lazy — heavy import

    # Convert to mono if stereo
    if ref_audio.ndim > 1:
        ref_audio = np.mean(ref_audio, axis=-1).astype(np.float32)

    logger.info(
        "Creating voice prompt: %.1fs audio, %d char transcript (x_vector_only_mode=%s)",
        len(ref_audio) / ref_sr,
        len(transcript),
        x_vector_only_mode,
    )

    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=(ref_audio, ref_sr),
        ref_text=transcript,
        x_vector_only_mode=x_vector_only_mode,
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
