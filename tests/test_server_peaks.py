"""Tests for waveform peaks in /generate response (MED-1).

``unittest.TestCase`` subclasses, not plain classes: this module is batched
(batch 3, ``tests/run_batches.py``) and the batch runner spawns
``python -m unittest``, which collects nothing from a plain ``class Test…:``.
As plain classes these four assertions reported "Ran 0 tests ... OK" while a
sabotaged ``calculate_waveform_peaks`` was live -- a hollow gate. The
recurrence guard is ``tests/test_batched_testcase_hygiene.py``.
"""
import unittest

import numpy as np


class TestGenerateResultPeaksField(unittest.TestCase):
    """MED-1: GenerateResult must carry an optional peaks field."""

    def test_generate_result_has_peaks_field(self):
        """GenerateResult accepts an optional peaks list."""
        from qwen3_tts.server.validation import GenerateResult

        result = GenerateResult(index=0, sample_rate=24000, peaks=[0.1, 0.5, 0.3])
        self.assertEqual(result.peaks, [0.1, 0.5, 0.3])

    def test_generate_result_peaks_default_none(self):
        """GenerateResult.peaks defaults to None when not provided."""
        from qwen3_tts.server.validation import GenerateResult

        result = GenerateResult(index=0, sample_rate=24000)
        self.assertIsNone(result.peaks)

    def test_generate_result_peaks_accepts_empty_list(self):
        """GenerateResult.peaks accepts an empty list."""
        from qwen3_tts.server.validation import GenerateResult

        result = GenerateResult(index=0, sample_rate=24000, peaks=[])
        self.assertEqual(result.peaks, [])


class TestCalculateWaveformPeaksExistence(unittest.TestCase):
    """Sanity-check that calculate_waveform_peaks already exists."""

    def test_calculate_waveform_peaks_returns_correct_length(self):
        """calculate_waveform_peaks returns exactly num_peaks values."""
        from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks

        audio = np.random.randn(24000).astype(np.float32)
        peaks = calculate_waveform_peaks(audio, num_peaks=500)
        self.assertEqual(len(peaks), 500)

    def test_calculate_waveform_peaks_values_in_range(self):
        """calculate_waveform_peaks returns values in [-1.0, 1.0]."""
        from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks

        audio = np.random.randn(24000).astype(np.float32)
        peaks = calculate_waveform_peaks(audio, num_peaks=100)
        self.assertTrue(all(-1.0 <= p <= 1.0 for p in peaks))


if __name__ == "__main__":
    unittest.main()
