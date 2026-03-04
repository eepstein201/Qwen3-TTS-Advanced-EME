#!/usr/bin/env python3
"""Audio processing utilities: loading, trimming, normalization, speed/pitch.

Base utility module — imports only from config.py, never from other engine submodules.
"""

import logging
import os
import threading

import numpy as np

from qwen3_tts.core.config import load_config

logger = logging.getLogger("tts.engine")

# Audio loader preference — lazy init on first access, thread-safe updates
_AUDIO_LOADER = None
_AUDIO_LOADER_LOCK = threading.Lock()


def get_audio_loader():
    """Return cached audio loader preference. Lazy-loads from config on first call."""
    global _AUDIO_LOADER
    if _AUDIO_LOADER is None:
        with _AUDIO_LOADER_LOCK:
            if _AUDIO_LOADER is None:
                _AUDIO_LOADER = load_config().get("advanced", {}).get("audio_loader", "torchaudio")
    return _AUDIO_LOADER


def set_audio_loader(loader):
    """Update audio loader preference in memory (called by config update endpoints)."""
    global _AUDIO_LOADER
    if loader not in ("torchaudio", "librosa"):
        raise ValueError(f"Invalid audio loader: {loader}")
    with _AUDIO_LOADER_LOCK:
        _AUDIO_LOADER = loader


# ---------------------------------------------------------------------------
# Smart Audio Loader (torchaudio primary, soundfile/librosa fallback)
# ---------------------------------------------------------------------------

def load_audio(file_path, target_sr=16000):
    """Load audio file, resample to target_sr. Uses cached loader preference."""
    if get_audio_loader() == "torchaudio":
        try:
            import torchaudio
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            return waveform.squeeze(0).numpy(), target_sr
        except Exception as e:
            logger.warning("torchaudio failed, falling back to soundfile: %s", e)
    import soundfile as sf
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr


def load_audio_for_cloning(file_path, max_duration=15, target_sr=16000):
    """Load audio truncated to max_duration seconds. For voice embedding only."""
    if get_audio_loader() == "torchaudio":
        try:
            import torchaudio
            info = torchaudio.info(file_path)
            max_frames = int(max_duration * info.sample_rate)
            waveform, sr = torchaudio.load(file_path, num_frames=max_frames)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            return waveform.squeeze(0).numpy(), target_sr
        except Exception as e:
            logger.warning("torchaudio failed, falling back to soundfile: %s", e)
    import soundfile as sf
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)
    max_samples = int(max_duration * sr)
    audio = audio[:max_samples]
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr


# ---------------------------------------------------------------------------
# Audio processing (backend-agnostic — uses only numpy)
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

    end_idx = len(audio) - np.argmax(np.flip(non_silent))
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

    Uses pyrubberband (Rubber Band library) for professional-quality
    time-stretching. Falls back to librosa's phase vocoder if
    pyrubberband or the rubberband CLI tool is not installed.

    Args:
        speed_factor: >1.0 = faster, <1.0 = slower.
    """
    if speed_factor == 1.0:
        return audio
    try:
        import pyrubberband as pyrb
        return pyrb.time_stretch(audio, sample_rate, speed_factor)
    except (ImportError, FileNotFoundError):
        import librosa
        return librosa.effects.time_stretch(audio, rate=speed_factor)


def adjust_pitch(audio, sample_rate, semitones):
    """Adjust audio pitch without changing speed.

    Uses pyrubberband (Rubber Band library) for professional-quality
    pitch-shifting. Falls back to librosa's phase vocoder if
    pyrubberband or the rubberband CLI tool is not installed.

    Args:
        semitones: positive = higher pitch, negative = lower.
    """
    if semitones == 0:
        return audio
    try:
        import pyrubberband as pyrb
        return pyrb.pitch_shift(audio, sample_rate, semitones)
    except (ImportError, FileNotFoundError):
        import librosa
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
