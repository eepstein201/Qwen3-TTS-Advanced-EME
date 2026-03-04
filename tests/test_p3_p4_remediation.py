#!/usr/bin/env python3
"""Tests for P3/P4 code review remediation.

Phase 2: Text processing fixes (R-21, R-22, _normalize_text, _expand_currency)
"""

import unittest
import unittest.mock


# ---------------------------------------------------------------------------
# Task 2: _normalize_text bare try/except:pass → logged warnings
# ---------------------------------------------------------------------------


class TestNormalizeTextLogging(unittest.TestCase):
    """Verify _normalize_text logs warnings instead of silently swallowing errors."""

    @unittest.mock.patch('qwen3_tts.core.engine.text_processing.logger')
    def test_normalization_step_failure_logged(self, mock_logger):
        """If a normalization step fails, a warning should be logged."""
        import qwen3_tts.core.engine.text_processing as tp
        original = tp._EMAIL_RE
        tp._EMAIL_RE = None  # Will cause AttributeError on .sub()
        try:
            result = tp._normalize_text("user@test.com hello")
            # Should still return a string (graceful degradation)
            self.assertIsInstance(result, str)
            # Should have logged a warning
            mock_logger.warning.assert_called()
            # Verify the warning mentions "email" in the format string
            call_args = mock_logger.warning.call_args_list[0]
            self.assertIn("email", call_args[0][0])
        finally:
            tp._EMAIL_RE = original

    @unittest.mock.patch('qwen3_tts.core.engine.text_processing.logger')
    def test_url_step_failure_logged(self, mock_logger):
        """URL normalization failure should be logged."""
        import qwen3_tts.core.engine.text_processing as tp
        original = tp._URL_RE
        tp._URL_RE = None
        try:
            result = tp._normalize_text("Visit https://example.com")
            self.assertIsInstance(result, str)
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args_list[0]
            self.assertIn("url", call_args[0][0])
        finally:
            tp._URL_RE = original

    @unittest.mock.patch('qwen3_tts.core.engine.text_processing.logger')
    def test_phone_step_failure_logged(self, mock_logger):
        """Phone normalization failure should be logged."""
        import qwen3_tts.core.engine.text_processing as tp
        original = tp._PHONE_RE
        tp._PHONE_RE = None
        try:
            result = tp._normalize_text("Call (800) 555-1234")
            self.assertIsInstance(result, str)
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args_list[0]
            self.assertIn("phone", call_args[0][0])
        finally:
            tp._PHONE_RE = original

    def test_normal_text_no_warnings(self):
        """Normal text should not trigger any warnings."""
        import qwen3_tts.core.engine.text_processing as tp
        with unittest.mock.patch.object(tp.logger, 'warning') as mock_warn:
            tp._normalize_text("Hello world, this is a test.")
            mock_warn.assert_not_called()


# ---------------------------------------------------------------------------
# Task 3: _expand_currency decimal handling
# ---------------------------------------------------------------------------


class TestCurrencyExpansion(unittest.TestCase):
    """Verify _expand_currency handles decimals like $5.99."""

    def test_decimal_currency(self):
        """$5.99 should expand to include cents."""
        from qwen3_tts.core.engine.text_processing import _normalize_text
        result = _normalize_text("The price is $5.99")
        self.assertIn("ninety", result.lower())
        self.assertIn("cent", result.lower())

    def test_whole_dollar(self):
        """$5 should still work as before."""
        from qwen3_tts.core.engine.text_processing import _normalize_text
        result = _normalize_text("It costs $5")
        self.assertIn("five", result.lower())
        self.assertIn("dollar", result.lower())

    def test_one_dollar(self):
        """$1 should use singular 'dollar'."""
        from qwen3_tts.core.engine.text_processing import _normalize_text
        result = _normalize_text("It costs $1")
        self.assertIn("one", result.lower())
        self.assertNotIn("dollars", result.lower())

    def test_one_cent(self):
        """$0.01 should say 'one cent'."""
        from qwen3_tts.core.engine.text_processing import _normalize_text
        result = _normalize_text("It costs $0.01")
        self.assertIn("cent", result.lower())
        self.assertNotIn("cents", result.lower())

    def test_euro_decimal(self):
        """€3.50 should handle cents."""
        from qwen3_tts.core.engine.text_processing import _normalize_text
        result = _normalize_text("It costs €3.50")
        self.assertIn("three", result.lower())
        self.assertIn("euro", result.lower())

    def test_pound_decimal(self):
        """£2.50 should use pence."""
        from qwen3_tts.core.engine.text_processing import _normalize_text
        result = _normalize_text("It costs £2.50")
        self.assertIn("two", result.lower())
        self.assertIn("pound", result.lower())
        self.assertIn("pence", result.lower())

    def test_zero_cents_no_subunit(self):
        """$5.00 should not include 'cent' text."""
        from qwen3_tts.core.engine.text_processing import _normalize_text
        result = _normalize_text("It costs $5.00")
        self.assertIn("five", result.lower())
        self.assertIn("dollar", result.lower())
        self.assertNotIn("cent", result.lower())


# ---------------------------------------------------------------------------
# Task 4: Cache pysbd.Segmenter per language (R-21)
# ---------------------------------------------------------------------------


class TestPysdbCache(unittest.TestCase):
    """Verify pysbd.Segmenter is cached per language code."""

    def test_segmenter_cached_per_language(self):
        """Second call with same language should reuse cached segmenter."""
        from qwen3_tts.core.engine.text_processing import _split_text, _SEGMENTER_CACHE
        _SEGMENTER_CACHE.clear()
        _split_text("Hello world. This is a test. Another one.", max_chars=15, language="English")
        self.assertIn("en", _SEGMENTER_CACHE)

    def test_different_languages_cached_separately(self):
        """Different languages should have separate cache entries."""
        from qwen3_tts.core.engine.text_processing import _split_text, _SEGMENTER_CACHE
        _SEGMENTER_CACHE.clear()
        _split_text("Hello world. This is a test.", max_chars=15, language="English")
        _split_text("Hola mundo. Esta es una prueba.", max_chars=15, language="Spanish")
        self.assertIn("en", _SEGMENTER_CACHE)
        self.assertIn("es", _SEGMENTER_CACHE)

    def test_cache_reuse_same_object(self):
        """Same language should return the same segmenter object."""
        from qwen3_tts.core.engine.text_processing import _split_text, _SEGMENTER_CACHE
        _SEGMENTER_CACHE.clear()
        _split_text("Hello world. This is a test.", max_chars=15, language="English")
        seg1 = _SEGMENTER_CACHE.get("en")
        _split_text("Another sentence. And one more.", max_chars=15, language="English")
        seg2 = _SEGMENTER_CACHE.get("en")
        self.assertIs(seg1, seg2)


# ---------------------------------------------------------------------------
# Task 5: Cache num2words import at module level (R-22)
# ---------------------------------------------------------------------------


class TestNum2wordsCache(unittest.TestCase):
    """Verify num2words is cached at module level after first use."""

    def test_num2words_cached_after_first_call(self):
        """num2words should be cached at module level after first use."""
        from qwen3_tts.core.engine.text_processing import _normalize_text
        _normalize_text("123")
        from qwen3_tts.core.engine import text_processing as tp
        self.assertTrue(tp._n2w_loaded)

    def test_num2words_function_cached(self):
        """_n2w_cached should hold the num2words function after first call."""
        from qwen3_tts.core.engine.text_processing import _normalize_text
        _normalize_text("456")
        from qwen3_tts.core.engine import text_processing as tp
        # num2words is installed in test env, so _n2w_cached should be set
        self.assertIsNotNone(tp._n2w_cached)


if __name__ == "__main__":
    unittest.main()
