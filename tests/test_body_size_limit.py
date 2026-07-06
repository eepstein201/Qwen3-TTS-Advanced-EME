"""Tests for the ASGI request-body-size cap (roadmap R-30).

Oversized bodies must be rejected by middleware via the Content-Length header
BEFORE the body is read into memory / parsed by Pydantic. A request whose
Content-Length exceeds the cap returns HTTP 413; a normal request passes
through to routing/validation.
"""

import unittest
from unittest.mock import patch

from tests.voice_test_helpers import _make_test_client, _skip_server


@_skip_server
class TestBodySizeLimit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app

        cls.app = app
        cls.client = _make_test_client(
            app, server_config={"security": {}, "auto_shutdown_minutes": 0}
        )

    def test_oversized_content_length_returns_413(self):
        """Body larger than the cap is rejected with 413 before parsing."""
        # Shrink the cap so the test does not need a real multi-MB payload.
        with patch("qwen3_tts.server.app.MAX_REQUEST_BODY_BYTES", 16):
            resp = self.client.post(
                "/generate",
                content=b"x" * 64,  # 64 bytes > 16-byte cap
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(resp.status_code, 413)

    def test_normal_request_passes_through(self):
        """A small body is not blocked by the size cap (413)."""
        resp = self.client.post(
            "/generate",
            json={"texts": ["hi"], "mode": "clone"},
        )
        # It may be rejected by auth/validation, but must NOT be 413.
        self.assertNotEqual(resp.status_code, 413)

    def test_cap_is_named_constant_above_audio_limit(self):
        """The cap is a named constant sized above the largest legit payload."""
        from qwen3_tts.server.app import MAX_REQUEST_BODY_BYTES
        from qwen3_tts.server.validation import MAX_AUDIO_BASE64_BYTES

        self.assertIsInstance(MAX_REQUEST_BODY_BYTES, int)
        self.assertGreater(MAX_REQUEST_BODY_BYTES, MAX_AUDIO_BASE64_BYTES)


if __name__ == "__main__":
    unittest.main()
