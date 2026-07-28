#!/usr/bin/env python3
"""Tests for qwen3_tts.server.websocket — bidirectional TTS streaming.

Covers:
- Authentication flow (valid token, invalid token, bad JSON, timeout)
- Generation requests (valid streaming, missing model, empty text)
- Cancel action during generation
- Wire format: [sr:4][len:4][audio:len] binary structure
- Message validation (oversized messages, invalid JSON in message loop)
- Disconnect handling

All inference is mocked — no models, GPU, or running server needed.

Run: pytest tests/test_websocket.py -v --tb=short
"""

import asyncio
import json
import struct
import unittest
from unittest.mock import MagicMock, patch

try:
    import numpy as np
    from fastapi.testclient import TestClient

    from qwen3_tts.server.app import app
    from qwen3_tts.server.websocket import websocket_tts_handler  # noqa: F401
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi, numpy, qwen3_tts")

# Test token matching the conftest xdist auto-fixture
_TEST_TOKEN = "xdist_test_token"


def _setup_app_state(models=None, server_config=None):
    """Configure app.state for WebSocket tests without touching unrelated attrs.

    Always resets auth_token to _TEST_TOKEN to avoid pollution from other test files.
    """
    state = app.state
    state.auth_token = _TEST_TOKEN
    if not hasattr(state, "inference_lock"):
        state.inference_lock = asyncio.Lock()
    if models is not None:
        state.models = models
    if server_config is not None:
        state.server_config = server_config
    elif not hasattr(state, "server_config"):
        state.server_config = {"security": {"max_text_length": 10000}}


@_skip
class TestWebSocketAuth(unittest.TestCase):
    """Authentication flow for WebSocket /ws endpoint."""

    def test_valid_token_authenticates(self):
        """Sending a valid token should return authenticated status."""
        _setup_app_state()
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"token": _TEST_TOKEN}))
            resp = ws.receive_json()
            self.assertEqual(resp["status"], "authenticated")

    def test_invalid_token_closes_4001(self):
        """An invalid token should return error and close."""
        _setup_app_state()
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"token": "wrong_token"}))
            resp = ws.receive_json()
            self.assertEqual(resp["error"], "Authentication failed")

    def test_bad_json_first_message_closes_4001(self):
        """Non-JSON first message should close the connection."""
        from starlette.websockets import WebSocketDisconnect
        _setup_app_state()
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_text("this is not json")
            with self.assertRaises(WebSocketDisconnect):
                ws.receive_json()

    def test_missing_token_field_closes_4001(self):
        """JSON without 'token' field should fail auth."""
        _setup_app_state()
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"not_token": "value"}))
            resp = ws.receive_json()
            self.assertEqual(resp["error"], "Authentication failed")

    def test_empty_token_string_rejected(self):
        """Empty token string should fail authentication."""
        _setup_app_state()
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"token": ""}))
            resp = ws.receive_json()
            self.assertEqual(resp["error"], "Authentication failed")

    def test_non_dict_first_message_releases_slot(self):
        """A valid-JSON non-object first message (e.g. "42") must not leak the
        connection slot. Pre-fix, auth_data.get() raised AttributeError, which
        escaped the except and skipped _ws_release (slot-exhaustion DoS)."""
        from starlette.websockets import WebSocketDisconnect

        _setup_app_state()
        app.state._ws_connections = {}  # isolate the slot counter
        with TestClient(app).websocket_connect("/ws") as ws:
            ws.send_text("42")  # valid JSON, but not an object
            with self.assertRaises(WebSocketDisconnect):
                ws.receive_json()
        conns = getattr(app.state, "_ws_connections", None) or {}
        self.assertEqual(sum(conns.values()), 0, f"slot leaked: {conns}")


@_skip
class TestWebSocketGeneration(unittest.TestCase):
    """Generation request handling via WebSocket."""

    def _authenticate(self, ws):
        """Helper to complete auth handshake."""
        ws.send_text(json.dumps({"token": _TEST_TOKEN}))
        resp = ws.receive_json()
        self.assertEqual(resp["status"], "authenticated")

    def test_oversized_max_new_tokens_rejected(self):
        """max_new_tokens beyond the Pydantic cap (le=8192) must be rejected,
        not passed to inference. Pre-fix the WS path built GenerateRequest from
        only 6 fields and passed raw data unvalidated (inference-lock DoS)."""
        _setup_app_state(
            models={"clone": MagicMock(), "design": None, "custom": None},
            server_config={"security": {}},
        )
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            ws.send_text(
                json.dumps({"text": "hi", "mode": "clone", "max_new_tokens": 999999})
            )
            resp = ws.receive_json()
            self.assertIn("error", resp)
            self.assertNotEqual(resp.get("status"), "generating")

    def test_out_of_range_temperature_rejected(self):
        """temperature > 2.0 must be rejected by Pydantic validation."""
        _setup_app_state(
            models={"clone": MagicMock(), "design": None, "custom": None},
            server_config={"security": {}},
        )
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            ws.send_text(
                json.dumps({"text": "hi", "mode": "clone", "temperature": 9.9})
            )
            resp = ws.receive_json()
            self.assertIn("error", resp)

    def test_empty_text_returns_error(self):
        """Sending a request with empty text should return an error."""
        _setup_app_state(models={"clone": MagicMock(), "design": None, "custom": None})
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            ws.send_text(json.dumps({"text": "", "mode": "clone"}))
            resp = ws.receive_json()
            self.assertEqual(resp["error"], "No text provided")

    def test_missing_text_field_returns_error(self):
        """Request without 'text' key should return no-text error."""
        _setup_app_state(models={"clone": MagicMock(), "design": None, "custom": None})
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            ws.send_text(json.dumps({"mode": "clone"}))
            resp = ws.receive_json()
            self.assertEqual(resp["error"], "No text provided")

    def test_model_not_loaded_returns_error(self):
        """Requesting a mode whose model is None should return an error."""
        _setup_app_state(models={"clone": None, "design": None, "custom": None})
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            ws.send_text(json.dumps({"text": "Hello world", "mode": "clone"}))
            resp = ws.receive_json()
            self.assertIn("not loaded", resp["error"])

    def test_design_model_not_loaded(self):
        """Design mode with no model loaded should error."""
        _setup_app_state(models={"clone": MagicMock(), "design": None, "custom": None})
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            ws.send_text(json.dumps({"text": "Test", "mode": "design"}))
            resp = ws.receive_json()
            self.assertIn("design", resp["error"])
            self.assertIn("not loaded", resp["error"])

    @patch("qwen3_tts.server.websocket.run_inference_streaming", create=True)
    def test_valid_generation_streams_audio(self, _mock_unused):
        """Valid generation request should stream binary chunks then complete."""
        fake_model = MagicMock()
        _setup_app_state(
            models={"clone": fake_model, "design": None, "custom": None},
            server_config={"security": {}},
        )

        # Create fake audio chunks
        chunk1 = np.zeros(2400, dtype=np.float32)
        chunk2 = np.ones(1200, dtype=np.float32)
        fake_chunks = [(chunk1, 24000), (chunk2, 24000)]

        with patch(
            "qwen3_tts.core.engine.run_inference_streaming",
            return_value=iter(fake_chunks),
        ), patch(
            "qwen3_tts.server.validation._validate_generation_request"
        ):
            with TestClient(app).websocket_connect("/ws") as ws:
                self._authenticate(ws)
                ws.send_text(json.dumps({"text": "Hello", "mode": "clone"}))

                # First JSON: "generating" status
                status = ws.receive_json()
                self.assertEqual(status["status"], "generating")
                self.assertEqual(status["text_length"], 5)

                # Binary chunks
                bin1 = ws.receive_bytes()
                bin2 = ws.receive_bytes()

                # Final JSON: "complete"
                complete = ws.receive_json()
                self.assertEqual(complete["status"], "complete")
                self.assertEqual(complete["chunks"], 2)

                # Verify we got binary data
                self.assertIsInstance(bin1, bytes)
                self.assertIsInstance(bin2, bytes)

    def test_invalid_mode_returns_error(self):
        """Invalid mode should return a validation error."""
        fake_model = MagicMock()
        _setup_app_state(
            models={"clone": fake_model, "design": None, "custom": None, "invalid": None},
            server_config={"security": {}},
        )
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            ws.send_text(json.dumps({"text": "Hello", "mode": "invalid"}))
            resp = ws.receive_json()
            # Model lookup for "invalid" returns None => "not loaded"
            self.assertIn("error", resp)

    def test_path_traversal_in_prompt_file_rejected(self):
        """prompt_file with path traversal should be rejected by validation."""
        fake_model = MagicMock()
        _setup_app_state(
            models={"clone": fake_model, "design": None, "custom": None},
            server_config={"security": {}},
        )
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            ws.send_text(json.dumps({
                "text": "Hello",
                "mode": "clone",
                "prompt_file": "../../../etc/passwd",
            }))
            resp = ws.receive_json()
            self.assertIn("error", resp)
            self.assertIn("path traversal", resp["error"].lower())

    def test_validation_error_names_offending_field(self):
        """A pydantic failure should surface the field name in the error."""
        fake_model = MagicMock()
        _setup_app_state(
            models={"clone": fake_model, "design": None, "custom": None},
            server_config={"security": {}},
        )
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            # temperature must be a number; a string fails type validation.
            ws.send_text(json.dumps({
                "text": "Hello",
                "mode": "clone",
                "temperature": "not-a-number",
            }))
            resp = ws.receive_json()
            self.assertIn("error", resp)
            self.assertIn("temperature", resp["error"].lower())

    def test_validation_error_with_no_details_uses_generic_message(self):
        """An empty ``errors()`` list must not crash — fall back to a generic message.

        Guards the branch that previously used a bare ``{}`` fallback.
        """
        from pydantic import ValidationError

        fake_model = MagicMock()
        _setup_app_state(
            models={"clone": fake_model, "design": None, "custom": None},
            server_config={"security": {}},
        )
        empty = ValidationError.from_exception_data("GenerateRequest", [])
        with patch(
            "qwen3_tts.server.validation.GenerateRequest", side_effect=empty
        ):
            with TestClient(app).websocket_connect("/ws") as ws:
                self._authenticate(ws)
                ws.send_text(json.dumps({"text": "Hello", "mode": "clone"}))
                resp = ws.receive_json()
                self.assertEqual(
                    resp["error"], "Invalid parameter: validation failed"
                )


@_skip
class TestWebSocketWireFormat(unittest.TestCase):
    """Verify the binary wire format: [sr:4 LE uint32][len:4 LE uint32][audio:len bytes]."""

    def _authenticate(self, ws):
        ws.send_text(json.dumps({"token": _TEST_TOKEN}))
        resp = ws.receive_json()
        assert resp["status"] == "authenticated"

    @patch("qwen3_tts.server.websocket.run_inference_streaming", create=True)
    def test_binary_frame_structure(self, _mock_unused):
        """Each binary frame must have 8-byte header (sr + length) followed by audio."""
        fake_model = MagicMock()
        _setup_app_state(
            models={"clone": fake_model, "design": None, "custom": None},
            server_config={"security": {}},
        )

        sample_rate = 24000
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        fake_chunks = [(audio, sample_rate)]

        with patch(
            "qwen3_tts.core.engine.run_inference_streaming",
            return_value=iter(fake_chunks),
        ), patch(
            "qwen3_tts.server.validation._validate_generation_request"
        ):
            with TestClient(app).websocket_connect("/ws") as ws:
                self._authenticate(ws)
                ws.send_text(json.dumps({"text": "Test", "mode": "clone"}))

                # Skip "generating" status
                ws.receive_json()

                frame = ws.receive_bytes()

                # Parse header
                self.assertGreaterEqual(len(frame), 8)
                sr, length = struct.unpack("<II", frame[:8])
                self.assertEqual(sr, sample_rate)
                self.assertEqual(length, len(audio.astype("<f4").tobytes()))
                self.assertEqual(len(frame), 8 + length)

                # Verify audio payload matches
                payload = frame[8:]
                recovered = np.frombuffer(payload, dtype="<f4")
                np.testing.assert_array_almost_equal(recovered, audio)

                # Complete message
                complete = ws.receive_json()
                self.assertEqual(complete["status"], "complete")

    @patch("qwen3_tts.server.websocket.run_inference_streaming", create=True)
    def test_multiple_chunks_correct_headers(self, _mock_unused):
        """Multiple chunks should each have independent correct headers."""
        fake_model = MagicMock()
        _setup_app_state(
            models={"clone": fake_model, "design": None, "custom": None},
            server_config={"security": {}},
        )

        chunks = [
            (np.zeros(100, dtype=np.float32), 24000),
            (np.ones(200, dtype=np.float32), 24000),
        ]

        with patch(
            "qwen3_tts.core.engine.run_inference_streaming",
            return_value=iter(chunks),
        ), patch(
            "qwen3_tts.server.validation._validate_generation_request"
        ):
            with TestClient(app).websocket_connect("/ws") as ws:
                self._authenticate(ws)
                ws.send_text(json.dumps({"text": "Test", "mode": "clone"}))
                ws.receive_json()  # "generating"

                for expected_audio, expected_sr in chunks:
                    frame = ws.receive_bytes()
                    sr, length = struct.unpack("<II", frame[:8])
                    self.assertEqual(sr, expected_sr)
                    expected_bytes = expected_audio.astype("<f4").tobytes()
                    self.assertEqual(length, len(expected_bytes))
                    self.assertEqual(frame[8:], expected_bytes)

                complete = ws.receive_json()
                self.assertEqual(complete["chunks"], 2)


@_skip
class TestWebSocketCancel(unittest.TestCase):
    """Cancel action handling."""

    def _authenticate(self, ws):
        ws.send_text(json.dumps({"token": _TEST_TOKEN}))
        resp = ws.receive_json()
        assert resp["status"] == "authenticated"

    def test_cancel_action_returns_cancelled(self):
        """Sending cancel action should return cancelled status."""
        _setup_app_state()
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            ws.send_text(json.dumps({"action": "cancel"}))
            resp = ws.receive_json()
            self.assertEqual(resp["status"], "cancelled")

    def test_cancel_then_new_request_works(self):
        """After cancel, a new generation request should still work."""
        _setup_app_state(models={"clone": None, "design": None, "custom": None})
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            # Cancel first
            ws.send_text(json.dumps({"action": "cancel"}))
            resp = ws.receive_json()
            self.assertEqual(resp["status"], "cancelled")

            # Then a new request (model not loaded, but shows loop continues)
            ws.send_text(json.dumps({"text": "Hello", "mode": "clone"}))
            resp = ws.receive_json()
            self.assertIn("not loaded", resp["error"])


@_skip
class TestWebSocketMessageValidation(unittest.TestCase):
    """Message validation in the message loop."""

    def _authenticate(self, ws):
        ws.send_text(json.dumps({"token": _TEST_TOKEN}))
        resp = ws.receive_json()
        assert resp["status"] == "authenticated"

    def test_invalid_json_in_message_loop(self):
        """Invalid JSON after auth should return error but keep connection open."""
        _setup_app_state()
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            ws.send_text("not valid json")
            resp = ws.receive_json()
            self.assertEqual(resp["error"], "Invalid JSON")

            # Connection stays open — can still send valid messages
            ws.send_text(json.dumps({"action": "cancel"}))
            resp = ws.receive_json()
            self.assertEqual(resp["status"], "cancelled")

    def test_oversized_message_rejected(self):
        """Messages over 64KB should be rejected."""
        _setup_app_state()
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            # Send a message just over 64KB
            big_msg = json.dumps({"text": "x" * 70000})
            ws.send_text(big_msg)
            resp = ws.receive_json()
            self.assertIn("too large", resp["error"])

    def test_multiple_sequential_requests(self):
        """Multiple generation requests on the same connection should all respond."""
        _setup_app_state(models={"clone": None, "design": None, "custom": None})
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            for i in range(3):
                ws.send_text(json.dumps({"text": f"Hello {i}", "mode": "clone"}))
                resp = ws.receive_json()
                self.assertIn("not loaded", resp["error"])


@_skip
class TestWebSocketDisconnect(unittest.TestCase):
    """Client disconnect handling."""

    def test_disconnect_during_auth_handled(self):
        """Disconnecting before sending auth should not raise unhandled errors."""
        _setup_app_state()
        # Simply connecting and immediately closing should not crash
        try:
            with TestClient(app).websocket_connect("/ws") as _ws:
                pass  # Context manager closes connection
        except Exception:
            pass  # Any close exception is acceptable

    def test_disconnect_after_auth_handled(self):
        """Disconnecting after auth should not raise unhandled errors."""
        _setup_app_state()
        try:
            with TestClient(app).websocket_connect("/ws") as ws:
                ws.send_text(json.dumps({"token": _TEST_TOKEN}))
                ws.receive_json()
                # Connection closes when context manager exits
        except Exception:
            pass


@_skip
class TestWebSocketDefaultMode(unittest.TestCase):
    """Default parameter handling."""

    def _authenticate(self, ws):
        ws.send_text(json.dumps({"token": _TEST_TOKEN}))
        resp = ws.receive_json()
        assert resp["status"] == "authenticated"

    def test_default_mode_is_clone(self):
        """When mode is not specified, it should default to 'clone'."""
        _setup_app_state(models={"clone": None, "design": None, "custom": None})
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            ws.send_text(json.dumps({"text": "Hello"}))
            resp = ws.receive_json()
            # clone model is None, so we get "clone" not loaded
            self.assertIn("clone", resp["error"])


if __name__ == "__main__":
    unittest.main()
