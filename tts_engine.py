#!/usr/bin/env python3
"""Heavy inference engine — imports torch, transformers, soundfile, librosa.

This module owns:
- Model loading (clone / design / custom) with configurable dtype, SDPA, MPS
- Core inference: run_inference() with NaN/Inf detection
- Voice-prompt creation from reference audio
- Audio post-processing (trim, normalize, speed, pitch)
- MPS memory management (empty_cache after generation)
- Voice prompt loading with LRU cache
"""

import logging
import os
import time
from functools import lru_cache

import librosa
import numpy as np
import soundfile as sf
import torch

from tts_config import MODEL_INFO, VOICE_PROMPTS_DIR, get_torch_dtype_name, CONFIG_PATH

logger = logging.getLogger("tts.engine")


# ---------------------------------------------------------------------------
# Voice prompt cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=10)
def load_voice_prompt(prompt_file):
    """Load and cache a .pt voice prompt."""
    prompt_path = os.path.join(VOICE_PROMPTS_DIR, prompt_file)
    if not os.path.exists(prompt_path):
        return None
    return torch.load(prompt_path, weights_only=False)


def clear_voice_prompt_cache():
    """Clear the voice prompt LRU cache."""
    load_voice_prompt.cache_clear()


def voice_prompt_cache_info():
    """Return cache statistics."""
    return load_voice_prompt.cache_info()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_type):
    """Load a TTS model by type.

    Args:
        model_type: One of "clone", "design", "custom".

    Returns:
        The loaded Qwen3TTSModel instance.
    """
    from qwen_tts import Qwen3TTSModel

    info = MODEL_INFO.get(model_type)
    if not info:
        raise ValueError(f"Unknown model type: {model_type}")

    dtype_name = get_torch_dtype_name()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype_name]

    logger.info("Loading %s (%s) with dtype=%s...", model_type, info["name"], dtype_name)
    t0 = time.time()

    model = Qwen3TTSModel.from_pretrained(
        info["name"],
        attn_implementation="sdpa",
        device_map="mps",
        dtype=torch_dtype,
    )

    elapsed = time.time() - t0
    logger.info("Loaded %s model in %.1fs", model_type, elapsed)
    return model


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(model, text, mode, gen_params, language="English",
                  voice_prompt=None, voice_description=None,
                  speaker=None, instruct=None):
    """Run TTS inference and return (wav_array, sample_rate).

    Args:
        model: Loaded Qwen3TTSModel.
        text: Text to synthesize.
        mode: "clone", "design", or "custom".
        gen_params: Dict with temperature, top_k, top_p, repetition_penalty.
        language: Language string.
        voice_prompt: Loaded voice prompt tensor (clone mode).
        voice_description: Voice description string (design mode).
        speaker: Speaker name string (custom mode).
        instruct: Style instruction string (custom mode).

    Returns:
        (wav_array, sample_rate) tuple.
    """
    t0 = time.time()

    params = {
        "temperature": gen_params.get("temperature", 0.7),
        "top_k": gen_params.get("top_k", 50),
        "top_p": gen_params.get("top_p", 0.95),
        "repetition_penalty": gen_params.get("repetition_penalty", 1.05),
    }

    seed = gen_params.get("seed")
    if seed is not None:
        torch.manual_seed(seed)

    try:
        with torch.inference_mode():
            if mode == "clone":
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=voice_prompt,
                    **params,
                )
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

    # MPS memory management
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            peak = torch.mps.current_allocated_memory()
            logger.debug(
                "MPS memory after generation: %.1f MB",
                peak / (1024 * 1024),
            )
        except Exception:
            pass

    elapsed = time.time() - t0
    logger.info(
        "Inference complete: %d chars, %.1fs, mode=%s",
        len(text), elapsed, mode,
    )

    return wavs[0], sr


# ---------------------------------------------------------------------------
# Voice prompt creation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------

def trim_silence(audio, sample_rate, threshold_db=-40, min_silence_ms=100):
    """Trim leading and trailing silence from audio."""
    threshold = 10 ** (threshold_db / 20)
    min_samples = int(sample_rate * min_silence_ms / 1000)

    abs_audio = np.abs(audio)
    non_silent = abs_audio > threshold

    if not np.any(non_silent):
        return audio

    start_idx = np.argmax(non_silent)
    start_idx = max(0, start_idx - min_samples)

    end_idx = len(audio) - np.argmax(non_silent[::-1])
    end_idx = min(len(audio), end_idx + min_samples)

    return audio[start_idx:end_idx]


def normalize_audio(audio, target_db=-3.0):
    """Normalize audio to target peak dB level."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    target_peak = 10 ** (target_db / 20)
    return audio * (target_peak / peak)


def adjust_speed(audio, sample_rate, speed_factor):
    """Adjust audio speed without changing pitch.

    Args:
        speed_factor: >1.0 = faster, <1.0 = slower.
    """
    if speed_factor == 1.0:
        return audio
    return librosa.effects.time_stretch(audio, rate=speed_factor)


def adjust_pitch(audio, sample_rate, semitones):
    """Adjust audio pitch without changing speed.

    Args:
        semitones: positive = higher pitch, negative = lower.
    """
    if semitones == 0:
        return audio
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)


def process_audio(audio, sample_rate, trim=False, normalize=False,
                  speed=None, pitch=None):
    """Apply all audio processing in canonical order.

    Args:
        audio: numpy audio array.
        sample_rate: Audio sample rate.
        trim: Trim leading/trailing silence.
        normalize: Normalize to -3dB peak.
        speed: Speed factor (None or 1.0 = unchanged).
        pitch: Pitch shift in semitones (None or 0 = unchanged).
    """
    if trim:
        audio = trim_silence(audio, sample_rate)

    if speed and speed != 1.0:
        audio = adjust_speed(audio, sample_rate, speed)

    if pitch and pitch != 0:
        audio = adjust_pitch(audio, sample_rate, pitch)

    if normalize:
        audio = normalize_audio(audio, target_db=-3.0)

    return audio
