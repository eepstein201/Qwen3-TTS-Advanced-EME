"""Test TranscribeRequest language validation.

Extends existing validation tests with comprehensive language code tests.
"""

import unittest

from pydantic import ValidationError

from qwen3_tts.server.validation import TranscribeRequest


class TestTranscribeRequestValidation(unittest.TestCase):
    def test_valid_language_codes(self):
        """Test valid language code formats."""
        # 2-letter codes
        req = TranscribeRequest(audio_base64="dummy", language="en")
        self.assertEqual(req.language, "en")

        # 3-letter codes
        req = TranscribeRequest(audio_base64="dummy", language="zho")
        self.assertEqual(req.language, "zho")

        # With region
        req = TranscribeRequest(audio_base64="dummy", language="en-US")
        self.assertEqual(req.language, "en-US")

        # Default is "en"
        req = TranscribeRequest(audio_base64="dummy")
        self.assertEqual(req.language, "en")

    def test_invalid_language_codes(self):
        """Test invalid language codes are rejected."""
        # Uppercase not allowed
        with self.assertRaises(ValidationError):
            TranscribeRequest(audio_base64="dummy", language="EN")

        # Numbers not allowed
        with self.assertRaises(ValidationError):
            TranscribeRequest(audio_base64="dummy", language="e1")

        # Special chars not allowed
        with self.assertRaises(ValidationError):
            TranscribeRequest(audio_base64="dummy", language="en_US")

        # Empty string not allowed
        with self.assertRaises(ValidationError):
            TranscribeRequest(audio_base64="dummy", language="")

    def test_valid_language_with_region(self):
        """Test language codes with region variants."""
        # Valid region codes (letters only, 2-4 chars)
        req = TranscribeRequest(audio_base64="dummy", language="zh-CN")
        self.assertEqual(req.language, "zh-CN")

        req = TranscribeRequest(audio_base64="dummy", language="en-GB")
        self.assertEqual(req.language, "en-GB")

        req = TranscribeRequest(audio_base64="dummy", language="es-MX")
        self.assertEqual(req.language, "es-MX")


if __name__ == "__main__":
    unittest.main()
