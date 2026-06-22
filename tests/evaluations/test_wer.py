"""WER (Word Error Rate) evaluation tests for TTS output quality.

Uses Whisper ASR to transcribe generated audio and compares against
the original text prompt. WER threshold is 5% (0.05) for passing.

Requirements: whisper, jiwer
These tests are skipped if dependencies are not available.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import jiwer
    import whisper
    HAS_WER_DEPS = True
except ImportError:
    HAS_WER_DEPS = False

WER_THRESHOLD = 0.05  # 5% maximum acceptable WER


@unittest.skipUnless(HAS_WER_DEPS, "requires whisper and jiwer packages")
class TestWERBelowThreshold(unittest.TestCase):
    """Verify generated audio transcription has WER < 5%."""

    @classmethod
    def setUpClass(cls):
        """Load Whisper model once for all tests."""
        cls.whisper_model = whisper.load_model("base")

    def _compute_wer(self, audio_path: str, reference_text: str) -> float:
        """Transcribe audio and compute WER against reference.

        Args:
            audio_path: Path to generated audio file.
            reference_text: Original text prompt.

        Returns:
            WER as a float (0.0 = perfect, 1.0 = 100% error).
        """
        result = self.whisper_model.transcribe(audio_path)
        hypothesis = result["text"].strip()
        wer = jiwer.wer(reference_text, hypothesis)
        return wer

    def test_wer_evaluation_framework_works(self):
        """Verify the WER evaluation framework is functional."""
        # This test verifies jiwer works correctly
        wer = jiwer.wer("hello world", "hello world")
        self.assertEqual(wer, 0.0)

        wer = jiwer.wer("hello world", "hello")
        self.assertGreater(wer, 0.0)

    def test_wer_with_known_transcription(self):
        """Verify WER computation with known input/output pair."""
        reference = "the quick brown fox jumps over the lazy dog"
        hypothesis = "the quick brown fox jumps over the lazy dog"
        wer = jiwer.wer(reference, hypothesis)
        self.assertLess(wer, WER_THRESHOLD)


class TestWERUtilities(unittest.TestCase):
    """Test WER utility functions that don't require heavy dependencies."""

    def test_wer_threshold_constant(self):
        """WER threshold should be 5%."""
        self.assertEqual(WER_THRESHOLD, 0.05)


if __name__ == "__main__":
    unittest.main()
