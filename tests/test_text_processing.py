#!/usr/bin/env python3
"""SSML parsing, text chunking, and normalization tests.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_text_processing.py -v

No GPU, models, or running server required.
"""

import inspect
import unittest

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Dummy decorator for when pytest is not available
    class _DummyMarkerFunc:
        """Represents a marker function like skipif that takes condition and returns decorator."""
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition, **kwargs):
            # skipif, etc. take condition as first arg, return a decorator
            return lambda f: f
    class _DummyMarker:
        def __call__(self, func):
            return func
        def __getattr__(self, name):
            # Return special function for skipif, otherwise return a callable marker
            if name == 'skipif':
                return _DummyMarkerFunc(name)
            return _DummyMarkerFunc(name)
        @property
        def unit(self):
            return self
    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()
    class _DummyPytest:
        mark = _DummyMark()
    pytest = _DummyPytest()

# Check optional dependencies
try:
    import soundfile  # noqa: F401
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

_skip_generate = unittest.skipUnless(HAS_SOUNDFILE, "requires soundfile (voice_generate)")

# num2words drives English number/time speech; Chinese uses an inline renderer.
try:
    import num2words  # noqa: F401
    HAS_NUM2WORDS = True
except ImportError:
    HAS_NUM2WORDS = False

_skip_n2w = unittest.skipUnless(HAS_NUM2WORDS, "requires num2words (English time speech)")


# =========================================================================
# SSML Edge Cases
# =========================================================================

@pytest.mark.unit
@_skip_generate
class TestSSMLEdgeCases(unittest.TestCase):
    """Test SSML parsing edge cases."""

    def test_ssml_sub_replacement(self):
        """<sub alias='hello'>hi</sub> replaces content with alias."""
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml('Say <sub alias="hello">hi</sub> please')
        self.assertIn("hello", text)
        self.assertNotIn("<sub", text)
        self.assertNotIn("hi", text)

    def test_ssml_say_as_characters(self):
        """<say-as interpret-as='characters'>ABC</say-as> spells out as 'A B C'."""
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml('<say-as interpret-as="characters">ABC</say-as>')
        self.assertIn("A B C", text)

    def test_ssml_prosody_rate_slow(self):
        """<prosody rate='slow'> sets speed=0.8 in metadata."""
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml('<prosody rate="slow">Hello world</prosody>')
        self.assertTrue(meta["has_ssml"])
        self.assertIsNotNone(meta["prosody"])
        self.assertEqual(meta["prosody"]["speed"], 0.8)

    def test_ssml_prosody_pitch_high(self):
        """<prosody pitch='high'> sets pitch=2 in metadata."""
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml('<prosody pitch="high">Hello world</prosody>')
        self.assertTrue(meta["has_ssml"])
        self.assertIsNotNone(meta["prosody"])
        self.assertEqual(meta["prosody"]["pitch"], 2)

    def test_ssml_nested_emphasis_sub(self):
        """Nested <emphasis><sub alias='hello'>hi</sub></emphasis> produces 'hello'."""
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml('<emphasis><sub alias="hello">hi</sub></emphasis>')
        self.assertIn("hello", text)
        self.assertNotIn("<", text)


# =========================================================================
# Dry-Run and Interactive Mode Tests
# =========================================================================

@pytest.mark.unit
@_skip_generate
class TestDryRunAndInteractive(unittest.TestCase):
    """Verify dry-run flag and interactive mode exist in source."""

    def test_dry_run_flag_in_source(self):
        """qwen3_tts.interface.generate source contains '--dry-run' argument."""
        import qwen3_tts.interface.generate
        source = inspect.getsource(qwen3_tts.interface.generate)
        self.assertIn("--dry-run", source)

    def test_dry_run_marker_in_source(self):
        """qwen3_tts.interface.generate source contains 'DRY RUN' marker text."""
        import qwen3_tts.interface.generate
        source = inspect.getsource(qwen3_tts.interface.generate)
        self.assertIn("DRY RUN", source)

    def test_interactive_mode_function_exists(self):
        """voice_generate has a callable interactive_mode function."""
        import qwen3_tts.interface.generate
        self.assertTrue(hasattr(qwen3_tts.interface.generate, "interactive_mode"))
        self.assertTrue(callable(qwen3_tts.interface.generate.interactive_mode))


# =========================================================================
# _safe_transform helper
# =========================================================================

class TestSafeTransform(unittest.TestCase):
    """Tests for the _safe_transform helper in text_processing."""

    def test_applies_transform_on_success(self):
        from qwen3_tts.core.engine.text_processing import _safe_transform
        result = _safe_transform("hello world", "test", lambda t: t.upper())
        self.assertEqual(result, "HELLO WORLD")

    def test_returns_original_on_exception(self):
        from qwen3_tts.core.engine.text_processing import _safe_transform

        def bad_fn(t):
            raise ValueError("boom")

        result = _safe_transform("hello", "bad_step", bad_fn)
        self.assertEqual(result, "hello")

    def test_logs_warning_on_exception(self):
        from qwen3_tts.core.engine.text_processing import _safe_transform
        with self.assertLogs(level="WARNING") as cm:
            _safe_transform("hello", "explode", lambda t: 1 / 0)
        self.assertTrue(any("explode" in msg for msg in cm.output))

    def test_step_name_in_warning_message(self):
        from qwen3_tts.core.engine.text_processing import _safe_transform
        with self.assertLogs(level="WARNING") as cm:
            _safe_transform("text", "my_step", lambda t: (_ for _ in ()).throw(RuntimeError("err")))
        self.assertTrue(any("my_step" in msg for msg in cm.output))

    def test_returns_string_unchanged_when_no_error(self):
        from qwen3_tts.core.engine.text_processing import _safe_transform
        result = _safe_transform("unchanged", "noop", lambda t: t)
        self.assertEqual(result, "unchanged")


# =========================================================================
# Chinese number normalization (PRF-1)
# =========================================================================


@pytest.mark.unit
class TestChineseNormalization(unittest.TestCase):
    """PRF-1: Chinese cardinal/ordinal/currency/date normalization.

    num2words(lang='zh') raises NotImplementedError, which _safe_transform
    swallowed — so all Chinese number normalization silently no-op'd. These
    tests pin the local digits→汉字 converter and the normalized output.
    """

    # --- _num_to_chinese: the pure converter ---
    def test_num_to_chinese_small(self):
        from qwen3_tts.core.engine.text_processing import _num_to_chinese

        cases = {
            0: "零", 1: "一", 9: "九", 10: "十", 12: "十二", 20: "二十",
            100: "一百", 101: "一百零一", 110: "一百一十", 111: "一百一十一",
            1000: "一千", 1001: "一千零一", 1010: "一千零一十", 1100: "一千一百",
            1234: "一千二百三十四",
        }
        for n, expected in cases.items():
            self.assertEqual(_num_to_chinese(n), expected, msg=f"failed for {n}")

    def test_num_to_chinese_large(self):
        from qwen3_tts.core.engine.text_processing import _num_to_chinese

        cases = {
            10000: "一万", 10001: "一万零一", 10010: "一万零一十",
            10100: "一万零一百", 11000: "一万一千",
            12345: "一万二千三百四十五", 100000: "十万", 100001: "十万零一",
            1000000: "一百万", 10010000: "一千零一万", 100000000: "一亿",
            100001000: "一亿零一千", 110000000: "一亿一千万",
            101000000: "一亿零一百万",
            123456789: "一亿二千三百四十五万六千七百八十九",
        }
        for n, expected in cases.items():
            self.assertEqual(_num_to_chinese(n), expected, msg=f"failed for {n}")

    def test_num_to_chinese_negative(self):
        from qwen3_tts.core.engine.text_processing import _num_to_chinese

        self.assertEqual(_num_to_chinese(-5), "负五")

    # --- _normalize_text: Chinese paths ---
    def test_chinese_cardinal_delimited(self):
        from qwen3_tts.core.engine.text_processing import _normalize_text

        out = _normalize_text("我有 3 个苹果", language="chinese")
        self.assertIn("三", out)
        self.assertNotIn("3", out)

    def test_chinese_cardinal_cjk_adjacent(self):
        """Digits directly between CJK chars (no \b) must still convert."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        out = _normalize_text("我有3个苹果", language="chinese")
        self.assertIn("三", out)
        self.assertNotIn("3", out)

    def test_chinese_ordinal(self):
        from qwen3_tts.core.engine.text_processing import _normalize_text

        out = _normalize_text("the 3rd one", language="chinese")
        self.assertIn("第三", out)
        self.assertNotIn("3", out)

    def test_chinese_currency(self):
        from qwen3_tts.core.engine.text_processing import _normalize_text

        out = _normalize_text("¥5", language="chinese")
        self.assertIn("五元", out)
        self.assertNotIn("5", out)

    def test_chinese_date(self):
        from qwen3_tts.core.engine.text_processing import _normalize_text

        out = _normalize_text("2024-01-02", language="chinese")
        self.assertIn("年", out)
        self.assertIn("月", out)
        self.assertIn("日", out)
        self.assertNotRegex(out, r"\d")  # no raw digits remain

    def test_zh_input_not_silently_no_op(self):
        """Regression for the core PRF-1 bug: zh number must change the text."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        before = "我有3个苹果"
        after = _normalize_text(before, language="chinese")
        self.assertNotEqual(before, after)
        self.assertIn("三", after)

    def test_cardinal_does_not_abort_wholesale_for_zh(self):
        """Multiple zh numbers in one string all convert (no wholesale abort)."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        out = _normalize_text("count 3 apples 我有 12 个", language="chinese")
        self.assertIn("三", out)
        self.assertIn("十二", out)

    def test_english_baseline_unchanged(self):
        from qwen3_tts.core.engine.text_processing import _normalize_text

        out = _normalize_text("I have 3 apples", language="english")
        self.assertIn("three", out)


# =========================================================================
# Time (HH:MM / HH:MM:SS) normalization — PRF-3
# =========================================================================


class TestTimeNormalization(unittest.TestCase):
    """HH:MM and HH:MM:SS strings must expand to spoken time, not garbled digits.

    Without time normalization the colons flow through to the model and the
    time is read back digit-by-digit (upstream #328). These tests pin the
    spoken form for English (num2words) and Chinese (inline renderer).
    """

    # --- English (requires num2words) -------------------------------------

    @_skip_n2w
    def test_hh_mm_ss_expands_to_spoken_time(self):
        """15:16:36 → 'fifteen hours sixteen minutes and thirty-six seconds'."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        result = _normalize_text("The time is 15:16:36 now.", language="English")
        self.assertNotIn("15:16:36", result)
        self.assertIn("fifteen hours sixteen minutes and thirty-six seconds", result)

    @_skip_n2w
    def test_hh_mm_clock_reading(self):
        """9:05 → 'nine oh five'; 3:45 → 'three forty-five'."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        self.assertIn("nine oh five", _normalize_text("at 9:05 sharp", "English"))
        self.assertIn("three forty-five", _normalize_text("at 3:45 sharp", "English"))

    @_skip_n2w
    def test_hh_mm_o_clock_on_the_hour(self):
        """9:00 → 'nine o'clock' (on-the-hour clock reading)."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        self.assertIn("nine o'clock", _normalize_text("at 9:00 sharp", "English"))

    @_skip_n2w
    def test_hh_mm_tens_minutes(self):
        """12:30 → 'twelve thirty'."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        self.assertIn("twelve thirty", _normalize_text("at 12:30 sharp", "English"))

    # --- Chinese (inline renderer, no num2words) --------------------------

    def test_zh_hh_mm_ss_uses_chinese(self):
        """15:16:36 → '十五点十六分三十六秒'."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        result = _normalize_text("时间是15:16:36。", language="Chinese")
        self.assertNotIn("15:16:36", result)
        self.assertIn("十五点十六分三十六秒", result)

    def test_zh_hh_mm_minutes(self):
        """9:05 → '九点五分'; 3:45 → '三点四十五分'."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        self.assertIn("九点五分", _normalize_text("在9:05开始", "Chinese"))
        self.assertIn("三点四十五分", _normalize_text("在3:45开始", "Chinese"))

    # --- Boundaries / negatives -------------------------------------------

    @_skip_n2w
    def test_time_not_confused_with_dates_or_phones(self):
        """Colons don't bleed into slash/dash dates or dash phones."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        result = _normalize_text("Meet 01/02/2024 at 10:00 or call 555-123-4567.", "English")
        # The time expands; the US date and phone are handled by their own steps.
        self.assertIn("ten o'clock", result)

    def test_ratio_with_single_digit_minute_not_matched(self):
        """'3:1' is not a clock time (minutes must be two digits) → not expanded."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        # The standalone digits may be cardinal-expanded by a later step, but the
        # key assertion is that the colon pair is never treated as a time.
        result = _normalize_text("ratio 3:1 here", "English")
        self.assertNotIn("hours", result)  # not HH:MM:SS
        self.assertNotIn("oh", result)     # not a clock reading

    @_skip_n2w
    def test_time_mid_sentence_preserves_surrounding_text(self):
        """A time inside a sentence expands without eating surrounding words."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        result = _normalize_text("Train leaves at 8:30 AM today.", "English")
        self.assertIn("eight thirty", result)
        self.assertIn("today", result)

    # --- Unit-level helper ------------------------------------------------

    def test_zh_under_hundred(self):
        """_zh_under_hundred renders clock components 0..59 correctly."""
        from qwen3_tts.core.engine.text_processing import _zh_under_hundred

        self.assertEqual(_zh_under_hundred(0), "零")
        self.assertEqual(_zh_under_hundred(5), "五")
        self.assertEqual(_zh_under_hundred(10), "十")
        self.assertEqual(_zh_under_hundred(15), "十五")
        self.assertEqual(_zh_under_hundred(20), "二十")
        self.assertEqual(_zh_under_hundred(25), "二十五")
        self.assertEqual(_zh_under_hundred(45), "四十五")
        self.assertEqual(_zh_under_hundred(59), "五十九")


if __name__ == "__main__":
    unittest.main()
