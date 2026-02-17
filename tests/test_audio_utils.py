#!/usr/bin/env python3
"""Tests for audio processing utilities and text chunking in voice_engine.py.

Covers normalize_audio, trim_silence, process_audio, and _split_text.
No GPU, models, or running server required — uses numpy for test audio.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAudioProcessing(unittest.TestCase):
    """Tests for normalize_audio, trim_silence, and process_audio."""

    # --- normalize_audio ---

    def test_normalize_silent_audio(self):
        """All-zeros audio should be returned unchanged (peak==0 guard)."""
        from voice_engine import normalize_audio
        audio = np.zeros(16000, dtype=np.float32)
        result = normalize_audio(audio, target_db=-3.0)
        np.testing.assert_array_equal(result, audio)

    def test_normalize_loud_audio(self):
        """After normalization the peak should match the target dB level."""
        from voice_engine import normalize_audio
        audio = np.random.randn(16000).astype(np.float32)
        target_db = -3.0
        result = normalize_audio(audio, target_db=target_db)
        expected_peak = 10 ** (target_db / 20)
        actual_peak = np.max(np.abs(result))
        self.assertAlmostEqual(actual_peak, expected_peak, places=5)

    def test_normalize_clipping_prevention(self):
        """With target_db=0, normal audio values should stay within [-1, 1]."""
        from voice_engine import normalize_audio
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        result = normalize_audio(audio, target_db=0.0)
        self.assertLessEqual(np.max(np.abs(result)), 1.0 + 1e-6)

    def test_normalize_idempotent(self):
        """Normalizing twice with the same target should give the same result."""
        from voice_engine import normalize_audio
        audio = np.random.randn(16000).astype(np.float32)
        once = normalize_audio(audio, target_db=-3.0)
        twice = normalize_audio(once, target_db=-3.0)
        np.testing.assert_allclose(once, twice, atol=1e-6)

    def test_normalize_preserves_dtype(self):
        """Output dtype should remain float32."""
        from voice_engine import normalize_audio
        audio = np.random.randn(16000).astype(np.float32)
        result = normalize_audio(audio, target_db=-3.0)
        self.assertEqual(result.dtype, np.float32)

    # --- trim_silence ---

    def test_trim_all_silent(self):
        """All-zeros audio should be returned unchanged (no non-silent samples)."""
        from voice_engine import trim_silence
        audio = np.zeros(16000, dtype=np.float32)
        result = trim_silence(audio, sample_rate=16000)
        np.testing.assert_array_equal(result, audio)

    def test_trim_no_silence(self):
        """Loud audio with no silence should be returned mostly unchanged."""
        from voice_engine import trim_silence
        audio = np.ones(16000, dtype=np.float32) * 0.5
        result = trim_silence(audio, sample_rate=16000, threshold_db=-40)
        # The entire signal is non-silent, so start_idx=0 (minus padding,
        # clamped to 0) and end_idx=16000 (plus padding, clamped to 16000).
        self.assertEqual(len(result), len(audio))

    def test_trim_leading_silence(self):
        """Leading silence should be removed (minus min_silence padding)."""
        from voice_engine import trim_silence
        sr = 16000
        silence = np.zeros(8000, dtype=np.float32)
        signal = np.ones(8000, dtype=np.float32) * 0.5
        audio = np.concatenate([silence, signal])
        result = trim_silence(audio, sample_rate=sr, threshold_db=-40,
                              min_silence_ms=100)
        # Result should be shorter than original because leading silence trimmed
        self.assertLess(len(result), len(audio))

    def test_trim_trailing_silence(self):
        """Trailing silence should be removed (minus min_silence padding)."""
        from voice_engine import trim_silence
        sr = 16000
        signal = np.ones(8000, dtype=np.float32) * 0.5
        silence = np.zeros(8000, dtype=np.float32)
        audio = np.concatenate([signal, silence])
        result = trim_silence(audio, sample_rate=sr, threshold_db=-40,
                              min_silence_ms=100)
        self.assertLess(len(result), len(audio))

    def test_trim_single_sample(self):
        """Array of length 1 should not crash."""
        from voice_engine import trim_silence
        audio = np.array([0.5], dtype=np.float32)
        result = trim_silence(audio, sample_rate=16000)
        self.assertGreaterEqual(len(result), 1)

    # --- process_audio ---

    def test_process_noop(self):
        """All defaults (no trim, no normalize, no speed/pitch) returns audio unchanged."""
        from voice_engine import process_audio
        audio = np.random.randn(16000).astype(np.float32)
        result = process_audio(audio, sample_rate=16000)
        np.testing.assert_array_equal(result, audio)

    def test_process_trim_only(self):
        """With trim=True, leading/trailing silence should be trimmed."""
        from voice_engine import process_audio
        sr = 16000
        silence = np.zeros(8000, dtype=np.float32)
        signal = np.ones(4000, dtype=np.float32) * 0.5
        audio = np.concatenate([silence, signal, silence])
        result = process_audio(audio, sample_rate=sr, trim=True)
        self.assertLess(len(result), len(audio))

    def test_process_normalize_only(self):
        """With normalize=True, peak should match -3dB target."""
        from voice_engine import process_audio
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = process_audio(audio, sample_rate=16000, normalize=True)
        expected_peak = 10 ** (-3.0 / 20)
        actual_peak = np.max(np.abs(result))
        self.assertAlmostEqual(actual_peak, expected_peak, places=5)


class TestTextChunkingEdgeCases(unittest.TestCase):
    """Tests for _split_text edge cases and boundary conditions."""

    def test_split_short_text(self):
        """Text under max_chars should return a single chunk."""
        from voice_engine import _split_text
        result = _split_text("Hello world.", max_chars=500)
        self.assertEqual(result, ["Hello world."])

    def test_split_on_sentences(self):
        """Two sentences should split at the sentence boundary."""
        from voice_engine import _split_text
        text = "First sentence. Second sentence."
        result = _split_text(text, max_chars=20)
        self.assertEqual(len(result), 2)
        self.assertIn("First sentence.", result[0])
        self.assertIn("Second sentence.", result[1])

    def test_split_long_sentence_clause(self):
        """A sentence exceeding max_chars should fall back to clause splitting."""
        from voice_engine import _split_text
        # Build a long sentence with clause boundaries
        text = "This is clause one, and this is clause two, and this is clause three"
        result = _split_text(text, max_chars=30)
        self.assertGreater(len(result), 1)
        # Recombined text should contain all original words
        combined = " ".join(result)
        for word in ["clause", "one", "two", "three"]:
            self.assertIn(word, combined)

    def test_split_empty_string(self):
        """Empty string should return a list with one empty string."""
        from voice_engine import _split_text
        result = _split_text("", max_chars=500)
        # After strip, empty string has len 0 <= 500, so returns [""]
        self.assertEqual(result, [""])

    def test_split_unicode_emoji(self):
        """Text with emoji characters should not crash."""
        from voice_engine import _split_text
        text = "Hello world! \U0001f600 This is great! \U0001f389"
        result = _split_text(text, max_chars=500)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) >= 1)

    def test_split_exactly_max_chars(self):
        """Text exactly at max_chars should return a single chunk."""
        from voice_engine import _split_text
        text = "x" * 100
        result = _split_text(text, max_chars=100)
        self.assertEqual(result, [text])

    def test_split_very_long_word(self):
        """A single word exceeding max_chars should not cause infinite loop."""
        from voice_engine import _split_text
        word = "a" * 600
        text = f"Hello. {word}. World."
        result = _split_text(text, max_chars=100)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) >= 1)
        # The long word must appear somewhere in the output
        combined = " ".join(result)
        self.assertIn(word, combined)

    def test_split_no_empty_chunks(self):
        """No chunk in the result should be an empty string."""
        from voice_engine import _split_text
        text = "First sentence. Second sentence. Third sentence. Fourth one."
        result = _split_text(text, max_chars=25)
        for chunk in result:
            self.assertTrue(len(chunk) > 0, f"Empty chunk found in {result}")

    def test_split_consecutive_punctuation(self):
        """Consecutive punctuation like '!!!' should be handled gracefully."""
        from voice_engine import _split_text
        text = "Hello!!! World!!! Testing!!!"
        result = _split_text(text, max_chars=15)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) >= 1)
        combined = " ".join(result)
        self.assertIn("Hello", combined)
        self.assertIn("World", combined)


if __name__ == "__main__":
    unittest.main()
