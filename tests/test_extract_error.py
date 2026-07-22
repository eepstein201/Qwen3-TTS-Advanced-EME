#!/usr/bin/env python3
"""Tests for _extract_error_message — dict detail, string detail, malformed responses.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_extract_error.py -v

No GPU, models, or running server required.
"""

import unittest
from unittest.mock import MagicMock

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from qwen3_tts.server.client._base import _extract_error_message



class TestExtractErrorMessage(unittest.TestCase):
    """Test _extract_error_message handles all FastAPI error response shapes."""

    def _make_response(self, json_data=None, status_code=500, raises=False):
        """Create a mock HTTP response.

        Args:
            json_data: dict to return from resp.json(), or None
            status_code: HTTP status code
            raises: if True, resp.json() raises ValueError
        """
        resp = MagicMock()
        resp.status_code = status_code
        if raises:
            resp.json.side_effect = ValueError("No JSON")
        else:
            resp.json.return_value = json_data
        return resp

    # --- String detail (existing behavior) ---

    def test_string_detail(self):
        """When detail is a plain string, return it."""
        resp = self._make_response({"detail": "Model not loaded"}, 503)
        result = _extract_error_message(resp)
        self.assertEqual(result, "Model not loaded")
        self.assertIsInstance(result, str)

    def test_error_field(self):
        """When 'error' key is present, prefer it."""
        resp = self._make_response({"error": "Some error"}, 500)
        result = _extract_error_message(resp)
        self.assertEqual(result, "Some error")

    def test_message_field(self):
        """When 'message' key is present (no 'error'), return it."""
        resp = self._make_response({"message": "Something went wrong"}, 500)
        result = _extract_error_message(resp)
        self.assertEqual(result, "Something went wrong")

    # --- Dict detail (Bug 5 fix) ---

    def test_dict_detail_with_detail_key(self):
        """When detail is a dict with 'detail' key, extract it as string."""
        resp = self._make_response(
            {"detail": {"error": "model_not_loaded", "detail": "The 'design' model is not loaded", "recovery": "load_model"}},
            503,
        )
        result = _extract_error_message(resp)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "The 'design' model is not loaded")

    def test_dict_detail_with_error_key(self):
        """When detail is a dict with 'error' but no 'detail' key."""
        resp = self._make_response(
            {"detail": {"error": "import_error", "recovery": "config"}},
            500,
        )
        result = _extract_error_message(resp)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "import_error")

    def test_dict_detail_fallback_to_str(self):
        """When detail is a dict with no 'detail' or 'error' key, str() it."""
        resp = self._make_response(
            {"detail": {"recovery": "restart"}},
            500,
        )
        result = _extract_error_message(resp)
        self.assertIsInstance(result, str)
        # Should be str(dict) rather than the dict itself
        self.assertIn("recovery", result)

    def test_dict_detail_never_returns_dict(self):
        """Result of _extract_error_message must always be a string, never a dict."""
        resp = self._make_response(
            {"detail": {"error": "load_failed", "detail": "CUDA OOM", "recovery": "restart"}},
            500,
        )
        result = _extract_error_message(resp)
        self.assertNotIsInstance(result, dict)

    # --- Malformed / edge cases ---

    def test_json_decode_error(self):
        """When response body is not JSON, return status code message."""
        resp = self._make_response(raises=True, status_code=502)
        result = _extract_error_message(resp)
        self.assertIsInstance(result, str)
        self.assertIn("502", result)

    def test_empty_json(self):
        """When JSON is empty dict, return default."""
        resp = self._make_response({}, 500)
        result = _extract_error_message(resp)
        self.assertEqual(result, "Unknown error")

    def test_custom_default(self):
        """Custom default message is returned when no fields match."""
        resp = self._make_response({}, 500)
        result = _extract_error_message(resp, default="Custom default")
        self.assertEqual(result, "Custom default")

    def test_none_detail(self):
        """When detail is None, fall through to default."""
        resp = self._make_response({"detail": None}, 500)
        result = _extract_error_message(resp)
        self.assertEqual(result, "Unknown error")

    def test_return_type_always_str(self):
        """Every code path must return a str."""
        cases = [
            {"detail": "simple string"},
            {"detail": {"error": "e", "detail": "d"}},
            {"detail": {"error": "e"}},
            {"detail": {}},
            {"error": "e"},
            {"message": "m"},
            {},
        ]
        for data in cases:
            resp = self._make_response(data, 500)
            result = _extract_error_message(resp)
            self.assertIsInstance(result, str, f"Failed for data={data}")


if __name__ == "__main__":
    unittest.main()
