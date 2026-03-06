"""Tests for Gradio Audio component JavaScript reset functionality."""

import unittest
from qwen3_tts.interface.ui import _create_audio_reset_js


class TestAudioResetJS(unittest.TestCase):
    """Test the JavaScript reset function generator."""

    def test_returns_non_empty_string(self):
        """JavaScript function should return non-empty string."""
        js = _create_audio_reset_js()
        self.assertIsInstance(js, str)
        self.assertGreater(len(js), 100)

    def test_contains_audio_element_reset(self):
        """Should query and reset audio elements."""
        js = _create_audio_reset_js()
        self.assertIn("querySelectorAll('audio')", js)
        self.assertIn("pause()", js)
        self.assertIn("currentTime", js)

    def test_contains_src_clearing(self):
        """Should remove src attribute to unload buffer."""
        js = _create_audio_reset_js()
        self.assertIn("removeAttribute('src')", js)

    def test_returns_true_on_success(self):
        """Should return true to allow generation to proceed."""
        js = _create_audio_reset_js()
        self.assertIn("return true", js)

    def test_has_error_handling(self):
        """Should wrap in try/catch for non-blocking failure."""
        js = _create_audio_reset_js()
        self.assertIn("try", js)
        self.assertIn("catch", js)
        self.assertIn("console.warn", js)

    def test_no_syntax_errors(self):
        """JavaScript should be valid syntax (basic checks)."""
        js = _create_audio_reset_js()
        # Basic JS syntax validation
        self.assertIn("=>", js)  # Arrow function
        self.assertIn("{", js)   # Opening brace
        self.assertIn("}", js)   # Closing brace


if __name__ == "__main__":
    unittest.main()
