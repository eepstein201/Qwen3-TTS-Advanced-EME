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


try:
    import num2words as _num2words  # noqa: F401
    HAS_NUM2WORDS = True
except ImportError:
    HAS_NUM2WORDS = False

try:
    import pysbd as _pysbd  # noqa: F401
    HAS_PYSBD = True
except ImportError:
    HAS_PYSBD = False

_skip_num2words = unittest.skipUnless(HAS_NUM2WORDS, "requires num2words")
_skip_pysbd = unittest.skipUnless(HAS_PYSBD, "requires pysbd")


class TestMapLanguage(unittest.TestCase):
    """Tests for _map_language() language code mapping helper."""

    def test_english_maps_to_en(self):
        from qwen3_tts.core.engine import _map_language
        self.assertEqual(_map_language("English"), "en")

    def test_spanish_maps_to_es(self):
        from qwen3_tts.core.engine import _map_language
        self.assertEqual(_map_language("Spanish"), "es")

    def test_french_maps_to_fr(self):
        from qwen3_tts.core.engine import _map_language
        self.assertEqual(_map_language("French"), "fr")

    def test_unknown_language_falls_back_to_en(self):
        from qwen3_tts.core.engine import _map_language
        self.assertEqual(_map_language("Klingon"), "en")

    def test_case_insensitive(self):
        from qwen3_tts.core.engine import _map_language
        self.assertEqual(_map_language("english"), "en")
        self.assertEqual(_map_language("ENGLISH"), "en")


@_skip_num2words
class TestNormalizeText(unittest.TestCase):
    """Tests for _normalize_text() — numbers, abbreviations, dates, currencies."""

    def test_normalize_cardinal_number(self):
        """42 → forty-two."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("I have 42 apples.", "English")
        self.assertIn("forty-two", result)
        self.assertNotIn("42", result)

    def test_normalize_dr_abbreviation(self):
        """Dr. → Doctor."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("Dr. Smith visited.", "English")
        self.assertIn("Doctor", result)

    def test_normalize_mr_abbreviation(self):
        """Mr. → Mister."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("Mr. Jones called.", "English")
        self.assertIn("Mister", result)

    def test_normalize_currency_dollars(self):
        """$42 → forty-two dollars."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("It costs $42.", "English")
        self.assertIn("dollars", result.lower())
        self.assertNotIn("$", result)

    def test_normalize_currency_euros(self):
        """€10 → ten euros."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("Pay €10 now.", "English")
        self.assertIn("euros", result.lower())
        self.assertNotIn("€", result)

    def test_normalize_ordinal(self):
        """3rd → third."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("He came in 3rd place.", "English")
        self.assertIn("third", result)
        self.assertNotIn("3rd", result)

    def test_normalize_iso_date(self):
        """2026-02-20 → February twentieth, ..."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("Today is 2026-02-20.", "English")
        self.assertIn("February", result)
        self.assertIn("twentieth", result)
        self.assertNotIn("2026-02-20", result)

    def test_normalize_etc_abbreviation(self):
        """etc. → et cetera."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("cats, dogs, etc.", "English")
        self.assertIn("et cetera", result)

    def test_normalize_eg_abbreviation(self):
        """e.g. → for example."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("e.g. apples and oranges", "English")
        self.assertIn("for example", result)

    def test_normalize_vs_abbreviation(self):
        """vs. → versus."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("cats vs. dogs", "English")
        self.assertIn("versus", result)

    def test_normalize_preserves_plain_text(self):
        """Text with no special tokens should be returned unchanged."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("Hello world.", "English")
        self.assertEqual(result, "Hello world.")

    def test_normalize_returns_string(self):
        """Always returns a string even on complex input."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("Test 123 Dr. Smith $5 etc.", "English")
        self.assertIsInstance(result, str)

    def test_normalize_email(self):
        """user@example.com → user at example dot com."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("Email user@example.com today.", "English")
        self.assertIn("at", result)
        self.assertNotIn("@", result)

    def test_normalize_url(self):
        """https://example.com → example dot com."""
        from qwen3_tts.core.engine import _normalize_text
        result = _normalize_text("Visit https://example.com for info.", "English")
        self.assertNotIn("https://", result)
        self.assertIn("example", result)


@_skip_pysbd
class TestPysbdSentenceSplitting(unittest.TestCase):
    """Tests that pySBD does not split on abbreviation dots."""

    def test_dr_smith_not_split(self):
        """'Dr. Smith' should not be treated as a sentence boundary."""
        from qwen3_tts.core.engine import _split_text
        text = "Dr. Smith visited the clinic. He was very kind."
        # With max_chars=200 both sentences should still be one chunk
        # because the total is under 200 chars
        result = _split_text(text, max_chars=200, language="English")
        self.assertEqual(len(result), 1, f"Expected 1 chunk, got: {result}")

    def test_real_sentence_boundary_still_splits(self):
        """Genuine sentence endings still trigger splitting when over limit."""
        from qwen3_tts.core.engine import _split_text
        text = "First sentence here. Second sentence here."
        result = _split_text(text, max_chars=25, language="English")
        self.assertGreater(len(result), 1)

    def test_language_param_accepted(self):
        """_split_text accepts a language keyword argument without error."""
        from qwen3_tts.core.engine import _split_text
        result = _split_text("Hello world.", max_chars=500, language="English")
        self.assertEqual(result, ["Hello world."])

    def test_pysbd_falls_back_gracefully_on_unknown_language(self):
        """Unknown language falls back to English segmenter without crashing."""
        from qwen3_tts.core.engine import _split_text
        result = _split_text("Hello world.", max_chars=500, language="Klingon")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) >= 1)


class TestGetMaxChunkTokens(unittest.TestCase):
    """Tests for _get_max_chunk_tokens() config reader."""

    def test_returns_int(self):
        from qwen3_tts.core.engine import _get_max_chunk_tokens
        result = _get_max_chunk_tokens()
        self.assertIsInstance(result, int)

    def test_returns_positive(self):
        from qwen3_tts.core.engine import _get_max_chunk_tokens
        result = _get_max_chunk_tokens()
        self.assertGreater(result, 0)


class TestTokenAwareChunking(unittest.TestCase):
    """Tests for token-aware _split_text(tokenizer=, max_tokens=)."""

    def _make_mock_tokenizer(self, tokens_per_word=1):
        """Create a mock tokenizer that assigns tokens_per_word tokens per word."""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.encode.side_effect = (
            lambda text, add_special_tokens=True:
            [1] * (len(text.split()) * tokens_per_word)
        )
        return tokenizer

    def test_token_aware_splits_when_over_max_tokens(self):
        """With tokenizer, chunks respect max_tokens not max_chars."""
        from qwen3_tts.core.engine import _split_text
        tokenizer = self._make_mock_tokenizer(tokens_per_word=1)
        # 30 words → 30 tokens; max_tokens=10 → should produce multiple chunks
        text = " ".join(["word"] * 30)
        result = _split_text(text, max_chars=10000, tokenizer=tokenizer, max_tokens=10)
        self.assertGreater(len(result), 1)

    def test_token_aware_no_split_under_limit(self):
        """Short text under max_tokens stays as one chunk."""
        from qwen3_tts.core.engine import _split_text
        tokenizer = self._make_mock_tokenizer(tokens_per_word=1)
        text = "Hello world."  # 2 tokens with mock
        result = _split_text(text, max_chars=10000, tokenizer=tokenizer, max_tokens=50)
        self.assertEqual(len(result), 1)

    def test_no_tokenizer_falls_back_to_chars(self):
        """Without tokenizer, char-based logic still applies."""
        from qwen3_tts.core.engine import _split_text
        text = "First sentence. Second sentence."
        result = _split_text(text, max_chars=20, tokenizer=None, max_tokens=None)
        self.assertGreater(len(result), 1)


if __name__ == "__main__":
    unittest.main()
