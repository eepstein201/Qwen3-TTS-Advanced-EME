#!/usr/bin/env python3
"""Audio processing utilities: loading, trimming, normalization, speed/pitch.

Base utility module — imports only from config.py, never from other engine submodules.
"""

import logging
import threading

from qwen3_tts.core.config import load_config

logger = logging.getLogger("tts.engine")


# ---------------------------------------------------------------------------
# Audio processing constants
# ---------------------------------------------------------------------------

SILENCE_THRESHOLD_DB = -40
NORMALIZATION_TARGET_DB = -3.0
LUFS_TARGET = -16.0
VOICE_EMBEDDING_MAX_DURATION = 15  # seconds
DEFAULT_SAMPLE_RATE = 24000

# Audio loader preference — lazy init on first access, thread-safe updates
_AUDIO_LOADER = None
_AUDIO_LOADER_LOCK = threading.Lock()


def get_audio_loader():
    """Return cached audio loader preference. Lazy-loads from config on first call."""
    global _AUDIO_LOADER
    if _AUDIO_LOADER is None:
        with _AUDIO_LOADER_LOCK:
            if _AUDIO_LOADER is None:
                _AUDIO_LOADER = (
                    load_config().get("advanced", {}).get("audio_loader", "torchaudio")
                )
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
    import numpy as np  # lazy — heavy import

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
        except (ImportError, RuntimeError, OSError) as e:
            logger.warning("torchaudio failed, falling back to soundfile: %s", e)
    import soundfile as sf

    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1).astype(np.float32)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr


def load_audio_for_cloning(
    file_path, max_duration=VOICE_EMBEDDING_MAX_DURATION, target_sr=DEFAULT_SAMPLE_RATE
):
    """Load audio truncated to max_duration seconds. For voice embedding only."""
    import numpy as np  # lazy — heavy import

    if get_audio_loader() == "torchaudio":
        try:
            import torchaudio

            # torchaudio.info() was removed in torchaudio >=2.11 (pyproject's
            # own floor). Load the full clip, then truncate to max_duration —
            # voice-clone sources are short, so this is cheap and avoids the
            # removed API (which raised AttributeError and aborted the build).
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            max_frames = int(max_duration * sr)
            waveform = waveform[:, :max_frames]
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
            return waveform.squeeze(0).numpy(), target_sr
        except (ImportError, RuntimeError, OSError) as e:
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


def trim_silence(
    audio, sample_rate, threshold_db=SILENCE_THRESHOLD_DB, min_silence_ms=100
):
    """Trim leading and trailing silence from audio."""
    import numpy as np  # lazy — heavy import

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


def normalize_audio(audio, target_db=NORMALIZATION_TARGET_DB):
    """Normalize audio to target peak dB level."""
    import numpy as np  # lazy — heavy import

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


def normalize_lufs(audio, sample_rate, target_lufs=LUFS_TARGET):
    """Normalize audio to target LUFS (EBU R128). Requires pyloudnorm.

    Args:
        audio: numpy float32 audio array.
        sample_rate: Audio sample rate.
        target_lufs: Target loudness in LUFS (default -16.0).

    Returns:
        Loudness-normalized audio array.

    Raises:
        ImportError: If pyloudnorm is not installed.
    """
    import pyloudnorm as pyln

    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    return pyln.normalize.loudness(audio, loudness, target_lufs)


def process_audio(
    audio,
    sample_rate,
    trim=False,
    normalize=False,
    speed=None,
    pitch=None,
    lufs_target=None,
):
    """Apply all audio processing in canonical order.

    Args:
        audio: numpy audio array.
        sample_rate: Audio sample rate.
        trim: Trim leading/trailing silence.
        normalize: Normalize to -3dB peak.
        speed: Speed factor (None or 1.0 = unchanged).
        pitch: Pitch shift in semitones (None or 0 = unchanged).
        lufs_target: Optional LUFS target (e.g. -16.0). Requires pyloudnorm.
    """
    if trim:
        audio = trim_silence(audio, sample_rate)

    if speed and speed != 1.0:
        audio = adjust_speed(audio, sample_rate, speed)

    if pitch and pitch != 0:
        audio = adjust_pitch(audio, sample_rate, pitch)

    if normalize:
        audio = normalize_audio(audio, target_db=NORMALIZATION_TARGET_DB)

    if lufs_target is not None:
        try:
            audio = normalize_lufs(audio, sample_rate, target_lufs=lufs_target)
        except ImportError:
            logger.warning("pyloudnorm not installed — skipping LUFS normalization")
        except Exception as e:
            logger.warning("LUFS normalization failed: %s", e)

    return audio


# ---------------------------------------------------------------------------
# Waveform peak calculation (for wavesurfer.js backend-side rendering)
# ---------------------------------------------------------------------------


def calculate_waveform_peaks(audio, num_peaks: int = 500) -> list[float]:
    """Pre-calculate normalized peak amplitudes for waveform visualization.

    Divides the audio into `num_peaks` bins and computes the maximum absolute
    amplitude in each bin, returning values in [-1.0, 1.0] suitable for
    wavesurfer.js `load('', [peaks], duration)`.

    Args:
        audio: Audio samples as a 1-D float array.
        num_peaks: Number of peak bins to produce.

    Returns:
        List of peak values, each in [-1.0, 1.0].
    """
    import numpy as np  # lazy — heavy import

    if audio.size == 0:
        return [0.0] * num_peaks

    # Flatten to mono if needed
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    num_peaks = min(num_peaks, audio.size)
    samples_per_bin = audio.size / num_peaks
    peaks = []

    for i in range(num_peaks):
        start = int(i * samples_per_bin)
        end = int((i + 1) * samples_per_bin)
        end = min(end, audio.size)
        if start >= end:
            peaks.append(0.0)
        else:
            peak_val = float(np.max(np.abs(audio[start:end])))
            peaks.append(max(-1.0, min(1.0, peak_val)))

    return peaks
