#!/usr/bin/env python3
"""SSML parsing, text chunking, and normalization tests.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_text_processing.py -v

No GPU, models, or running server required.
"""

import inspect
import os
import sys
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

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check optional dependencies
try:
    import soundfile  # noqa: F401
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

_skip_generate = unittest.skipUnless(HAS_SOUNDFILE, "requires soundfile (voice_generate)")


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


if __name__ == "__main__":
    unittest.main()
