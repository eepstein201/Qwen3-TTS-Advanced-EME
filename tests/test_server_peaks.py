"""Tests for waveform peaks in /generate response (MED-1)."""
import numpy as np


class TestGenerateResultPeaksField:
    """MED-1: GenerateResult must carry an optional peaks field."""

    def test_generate_result_has_peaks_field(self):
        """GenerateResult accepts an optional peaks list."""
        from qwen3_tts.server.validation import GenerateResult

        result = GenerateResult(index=0, sample_rate=24000, peaks=[0.1, 0.5, 0.3])
        assert result.peaks == [0.1, 0.5, 0.3]

    def test_generate_result_peaks_default_none(self):
        """GenerateResult.peaks defaults to None when not provided."""
        from qwen3_tts.server.validation import GenerateResult

        result = GenerateResult(index=0, sample_rate=24000)
        assert result.peaks is None

    def test_generate_result_peaks_accepts_empty_list(self):
        """GenerateResult.peaks accepts an empty list."""
        from qwen3_tts.server.validation import GenerateResult

        result = GenerateResult(index=0, sample_rate=24000, peaks=[])
        assert result.peaks == []


class TestCalculateWaveformPeaksExistence:
    """Sanity-check that calculate_waveform_peaks already exists."""

    def test_calculate_waveform_peaks_returns_correct_length(self):
        """calculate_waveform_peaks returns exactly num_peaks values."""
        from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks

        audio = np.random.randn(24000).astype(np.float32)
        peaks = calculate_waveform_peaks(audio, num_peaks=500)
        assert len(peaks) == 500

    def test_calculate_waveform_peaks_values_in_range(self):
        """calculate_waveform_peaks returns values in [-1.0, 1.0]."""
        from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks

        audio = np.random.randn(24000).astype(np.float32)
        peaks = calculate_waveform_peaks(audio, num_peaks=100)
        assert all(-1.0 <= p <= 1.0 for p in peaks)
