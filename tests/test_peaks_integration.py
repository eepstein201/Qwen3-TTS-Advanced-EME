"""Test waveform peaks integration and performance.

Tests that peaks are calculated correctly server-side and included in responses.
"""

import unittest

import numpy as np

from qwen3_tts.core.engine.audio_processing import calculate_waveform_peaks


class TestWaveformPeaksIntegration(unittest.TestCase):
    def test_peaks_calculation(self):
        """Test peaks are calculated correctly."""
        # Create dummy audio
        audio = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz

        # Calculate peaks
        peaks = calculate_waveform_peaks(audio, num_peaks=500)

        # Verify
        self.assertEqual(len(peaks), 500)
        self.assertTrue(all(-1.0 <= p <= 1.0 for p in peaks))
        self.assertIsInstance(peaks, list)

    def test_peaks_different_sizes(self):
        """Test peaks calculation with different sizes."""
        # Test with different audio lengths
        for duration_sec in [0.5, 1.0, 2.0, 5.0]:
            samples = int(duration_sec * 24000)
            audio = np.random.randn(samples).astype(np.float32)
            peaks = calculate_waveform_peaks(audio, num_peaks=500)

            self.assertEqual(len(peaks), 500)
            self.assertTrue(all(-1.0 <= p <= 1.0 for p in peaks))

    def test_peaks_with_different resolutions(self):
        """Test peaks calculation with different resolutions."""
        audio = np.random.randn(24000).astype(np.float32)  # 1 second

        # Test different peak counts
        for num_peaks in [100, 250, 500, 1000]:
            peaks = calculate_waveform_peaks(audio, num_peaks=num_peaks)

            self.assertEqual(len(peaks), num_peaks)
            self.assertTrue(all(-1.0 <= p <= 1.0 for p in peaks))

    def test_peaks_with_silence(self):
        """Test peaks calculation with silent audio."""
        # Create silent audio
        audio = np.zeros(24000, dtype=np.float32)

        # Calculate peaks
        peaks = calculate_waveform_peaks(audio, num_peaks=500)

        # All peaks should be 0 for silence
        self.assertTrue(all(p == 0.0 for p in peaks))

    def test_peaks_with_loud_audio(self):
        """Test peaks calculation with maximum amplitude."""
        # Create maximum amplitude audio
        audio = np.ones(24000, dtype=np.float32)

        # Calculate peaks
        peaks = calculate_waveform_peaks(audio, num_peaks=500)

        # All peaks should be 1.0 for maximum amplitude
        self.assertTrue(all(p == 1.0 for p in peaks))


if __name__ == "__main__":
    unittest.main()
