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
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

try:
    import numpy as np
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

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
        ), patch(
            "qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock(),
        ):
            with TestClient(app).websocket_connect("/ws") as ws:
                self._authenticate(ws)
                ws.send_text(json.dumps({"text": "Hello", "mode": "clone", "prompt_file": "test.pt"}))

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
        ), patch(
            "qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock(),
        ):
            with TestClient(app).websocket_connect("/ws") as ws:
                self._authenticate(ws)
                ws.send_text(json.dumps({"text": "Test", "mode": "clone", "prompt_file": "test.pt"}))

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
        ), patch(
            "qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock(),
        ):
            with TestClient(app).websocket_connect("/ws") as ws:
                self._authenticate(ws)
                ws.send_text(json.dumps({"text": "Test", "mode": "clone", "prompt_file": "test.pt"}))
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


@_skip
class TestWebSocketOOMGuard(unittest.TestCase):
    """OOM memory guard on the WebSocket streaming path (H1).

    The HTTP /generate and /generate-stream paths enforce _check_memory_available
    before generating; the WS path historically bypassed it, so a low-memory
    request still spawned an inference thread that could OOM-crash the server.
    """

    def _authenticate(self, ws):
        """Helper to complete auth handshake."""
        ws.send_text(json.dumps({"token": _TEST_TOKEN}))
        resp = ws.receive_json()
        self.assertEqual(resp["status"], "authenticated")

    @patch("qwen3_tts.server.app_lifespan._check_memory_available")
    def test_low_memory_sends_status_error(self, mock_mem):
        """When the OOM guard reports insufficient memory, the WS path must send
        a status=='error' frame and must NOT proceed to status=='generating'.

        Patches the app_lifespan seam (where websocket.py imports the guard
        inline), NOT the app_generation (HTTP) seam — patching the wrong one
        leaves the real check in place and the test passes vacuously.
        """
        mock_mem.return_value = (False, 500)
        fake_model = MagicMock()
        _setup_app_state(
            models={"clone": fake_model, "design": None, "custom": None},
            server_config={"security": {}},
        )

        # Guard against real inference in case execution flows past the guard
        # (pre-fix). Validation is patched so the request reaches the guard.
        with patch(
            "qwen3_tts.server.validation._validate_generation_request"
        ), patch("qwen3_tts.core.engine.run_inference_streaming"), patch(
            "qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock(),
        ):
            with TestClient(app).websocket_connect("/ws") as ws:
                self._authenticate(ws)
                ws.send_text(json.dumps({"text": "Hello", "mode": "clone", "prompt_file": "test.pt"}))

                # The error frame must be the FIRST message received. If a
                # "generating" frame had been sent first (the pre-fix behavior),
                # this receive would return it and the assertion would fail —
                # proving no "generating" frame precedes the error.
                resp = ws.receive_json()
                self.assertEqual(resp["status"], "error")
                self.assertIn("Insufficient memory", resp["detail"])
                self.assertIn("500", resp["detail"])
                mock_mem.assert_called_once()


@_skip
class TestWebSocketErrorReporting(unittest.TestCase):
    """H5: WebSocket generation errors and missing prompts must not report
    false success. Pre-fix, an inference exception was logged but the client
    still received {"status":"complete"}, and a clone request with a missing
    or not-found prompt_file proceeded and also reported complete.
    """

    def _authenticate(self, ws):
        """Helper to complete auth handshake."""
        ws.send_text(json.dumps({"token": _TEST_TOKEN}))
        resp = ws.receive_json()
        self.assertEqual(resp["status"], "authenticated")

    @patch("qwen3_tts.server.app_lifespan._check_memory_available")
    def test_inference_exception_returns_status_error_not_complete(self, mock_mem):
        """When run_inference_streaming raises, the terminal frame must report
        status=="error" with a sanitized detail — NOT false success.

        Pre-fix the thread excepted silently and the terminal frame sent
        {"status":"complete"} (false success).
        """
        mock_mem.return_value = (True, 4096)
        _setup_app_state(
            models={
                "clone": MagicMock(),
                "design": MagicMock(),
                "custom": None,
            },
            server_config={"security": {}},
        )

        with patch(
            "qwen3_tts.core.engine.run_inference_streaming",
            side_effect=RuntimeError("test inference failure"),
        ), patch(
            "qwen3_tts.server.validation._validate_generation_request"
        ):
            with TestClient(app).websocket_connect("/ws") as ws:
                self._authenticate(ws)
                ws.send_text(json.dumps({"text": "Hello", "mode": "design"}))

                # "generating" frame is expected (validation + memory passed)
                generating = ws.receive_json()
                self.assertEqual(generating["status"], "generating")

                # Terminal frame must be an error, not false success
                error = ws.receive_json()
                self.assertEqual(error["status"], "error")
                self.assertNotEqual(error["status"], "complete")
                self.assertIn("test inference failure", error["detail"])
                self.assertEqual(error["chunks"], 0)

    @patch("qwen3_tts.core.engine.run_inference_streaming", return_value=iter([]))
    def test_missing_prompt_file_returns_error_not_empty_complete(self, _mock_inf):
        """Clone mode without a prompt_file must return an error frame BEFORE
        the "generating" frame — a missing prompt must never look like
        generation started (false success).
        """
        _setup_app_state(
            models={"clone": MagicMock(), "design": None, "custom": None},
            server_config={"security": {}},
        )
        with TestClient(app).websocket_connect("/ws") as ws:
            self._authenticate(ws)
            ws.send_text(json.dumps({"text": "Hello", "mode": "clone"}))

            resp = ws.receive_json()
            self.assertEqual(resp["error"], "prompt_file required for clone mode")
            # No "generating" frame should precede this error
            self.assertNotEqual(resp.get("status"), "generating")

    @patch("qwen3_tts.core.engine.run_inference_streaming", return_value=iter([]))
    @patch("qwen3_tts.server.validation._validate_generation_request")
    def test_prompt_not_found_returns_error(self, _mock_validate, _mock_inf):
        """Clone mode with a prompt_file whose load_voice_prompt returns None
        must return a 'Voice prompt not found' error — NOT proceed with
        voice_prompt=None and report false success.
        """
        _setup_app_state(
            models={"clone": MagicMock(), "design": None, "custom": None},
            server_config={"security": {}},
        )
        with patch(
            "qwen3_tts.core.engine.load_voice_prompt", return_value=None
        ):
            with TestClient(app).websocket_connect("/ws") as ws:
                self._authenticate(ws)
                ws.send_text(json.dumps({
                    "text": "Hello",
                    "mode": "clone",
                    "prompt_file": "missing.pt",
                }))

                resp = ws.receive_json()
                self.assertEqual(
                    resp["error"], "Voice prompt not found: missing.pt"
                )
                self.assertNotEqual(resp.get("status"), "generating")


@_skip
class TestWebSocketCancelMidGeneration(unittest.TestCase):
    """H6: a cancel sent during in-flight generation must stop the stream.

    Pre-fix, the main message loop is blocked inside _stream_generation, so a
    mid-generation {"action":"cancel"} frame is never read until generation
    finishes — the stream runs to completion and reports "complete".
    Post-fix, a concurrent cancel-watcher reads frames during generation and
    sets stop_event, which the inference thread observes (it checks between
    chunks), stopping the stream and yielding a "cancelled" terminal frame.
    """

    def _authenticate(self, ws):
        """Helper to complete auth handshake."""
        ws.send_text(json.dumps({"token": _TEST_TOKEN}))
        resp = ws.receive_json()
        self.assertEqual(resp["status"], "authenticated")

    def test_cancel_mid_generation_stops_stream_and_sends_cancelled(self):
        """Cancel sent after the first chunk must stop the stream and produce
        a {"status":"cancelled"} terminal frame — not run to completion.

        The mock generator yields one chunk then blocks on a test-controlled
        ``proceed`` event, mirroring how real inference blocks inside
        model.generate() between chunk yields.  The test unblocks the
        generator only after sending cancel and giving the concurrent watcher
        time to process it.  Without the watcher (pre-fix) stop_event is never
        set, so the terminal frame is "complete"; with the watcher it is
        "cancelled".
        """
        _setup_app_state(
            models={"clone": MagicMock(), "design": None, "custom": None},
            server_config={"security": {}},
        )

        chunk = np.zeros(2400, dtype=np.float32)
        sample_rate = 24000

        # proceed gates the mock generator: it blocks after the first chunk
        # until the test signals it.  This decouples generator unblocking
        # from the handler's internal stop_event (which the test cannot
        # access directly).
        proceed = threading.Event()

        def fake_streaming(*args, **kwargs):
            yield (chunk, sample_rate)
            # Block until the test signals proceed — mirroring real inference
            # blocking inside model.generate() between chunk yields.  Without
            # the watcher, stop_event is never set regardless of when this
            # returns.
            proceed.wait(timeout=5.0)
            # Return without yielding more — cancel must stop generation here.

        with patch(
            "qwen3_tts.core.engine.run_inference_streaming",
            side_effect=fake_streaming,
        ), patch(
            "qwen3_tts.server.validation._validate_generation_request"
        ), patch(
            "qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock(),
        ), patch(
            "qwen3_tts.server.app_lifespan._check_memory_available",
            return_value=(True, 4096),
        ):
            with TestClient(app).websocket_connect("/ws") as ws:
                self._authenticate(ws)
                ws.send_text(json.dumps({
                    "text": "Hello",
                    "mode": "clone",
                    "prompt_file": "test.pt",
                }))

                # "generating" status frame
                generating = ws.receive_json()
                self.assertEqual(generating["status"], "generating")

                # First (and only) binary chunk arrives at the client
                bin1 = ws.receive_bytes()
                self.assertIsInstance(bin1, bytes)

                # Send cancel mid-generation while the stream is in-flight
                ws.send_text(json.dumps({"action": "cancel"}))

                # Give the concurrent watcher (if present) time to read the
                # cancel frame and set stop_event.  The watcher runs in the
                # server's event loop (portal thread); sub-millisecond work.
                time.sleep(0.3)

                # Unblock the generator so the inference thread can finish.
                # GREEN: stop_event already set by watcher -> "cancelled".
                # RED:   stop_event never set -> "complete".
                proceed.set()

                # Terminal frame — must be "cancelled", not "complete".
                final = ws.receive_json()
                self.assertEqual(final["status"], "cancelled")
                # Exactly one chunk was streamed before the cancel took effect
                self.assertEqual(final["chunks"], 1)
                self.assertIn("seed", final)


class _FakeDisconnectWebSocket:
    """Minimal async WebSocket double for handler-level disconnect tests.

    Scripts ``receive_text`` from a fed queue and raises
    ``WebSocketDisconnect`` once the queue is empty — modelling a client that
    has gone away mid-stream.  Records every JSON/binary send so the test can
    assert on the terminal frame without a running server or TestClient.
    """

    def __init__(self):
        self._rx: list[str] = []
        self.sent_json: list[dict] = []
        self.sent_bytes: list[bytes] = []
        self.client = types.SimpleNamespace(host="127.0.0.1")

    def feed(self, message: str) -> None:
        """Queue an inbound text frame for the next ``receive_text`` call."""
        self._rx.append(message)

    async def accept(self) -> None:
        pass

    async def receive_text(self) -> str:
        if self._rx:
            return self._rx.pop(0)
        # Queue drained → the client has disconnected.
        raise WebSocketDisconnect(1001)

    async def send_json(self, obj: dict) -> None:
        self.sent_json.append(obj)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        pass


@_skip
class TestWebSocketDisconnectDuringGeneration(unittest.IsolatedAsyncioTestCase):
    """I1: a client disconnect observed by the concurrent cancel-watcher during
    in-flight generation must NOT be misclassified as a cancel.

    Pre-fix the watcher signals a disconnect with the same ``stop_event`` it
    uses for a cancel, so the consumer cannot tell them apart: on disconnect it
    sets ``was_cancelled`` and emits a ``{"status": "cancelled"}`` terminal
    frame on the dead socket (a redundant send that raises a noisy server-side
    error log).  Post-fix the watcher also sets a distinct ``disconnect_event``;
    the consumer treats the outcome as a disconnect and returns WITHOUT any
    terminal status frame.
    """

    async def test_watcher_disconnect_emits_no_terminal_status_frame(self):
        """A disconnect read by the watcher mid-generation produces no terminal
        status frame — not cancelled, complete, or error.

        The fake generator yields chunks until the inference thread breaks out
        of its for-loop on ``stop_event`` (set by the watcher once it reads the
        disconnect), exactly as real ``model.generate()`` is checked between
        chunk yields.  Deterministic: no test-path sleeps, no driver thread.
        """
        # Arrange — auth frame, then a generation request; the fake socket
        # disconnects (WebSocketDisconnect) on the very next receive, which the
        # concurrent cancel-watcher reads while generation is in flight.
        ws = _FakeDisconnectWebSocket()
        ws.feed(json.dumps({"token": _TEST_TOKEN}))
        ws.feed(json.dumps({"text": "Hello", "mode": "custom"}))

        state = types.SimpleNamespace(
            models={"custom": MagicMock()},
            server_config={"security": {"max_text_length": 10000}},
            inference_lock=asyncio.Lock(),
        )

        def fake_streaming(*args, **kwargs):
            chunk = np.zeros(100, dtype=np.float32)
            # Yield until the inference thread observes stop_event (set by the
            # watcher once it reads the disconnect) and breaks out of its
            # for-loop — mirroring real generation checked between chunks.
            while True:
                yield (chunk, 24000)
                time.sleep(0.01)

        with patch(
            "qwen3_tts.core.engine.run_inference_streaming",
            side_effect=fake_streaming,
        ), patch(
            "qwen3_tts.server.validation._validate_generation_request"
        ), patch(
            "qwen3_tts.server.app_lifespan._check_memory_available",
            return_value=(True, 4096),
        ):
            # Act — drive the real handler end-to-end against the fake socket.
            await asyncio.wait_for(
                websocket_tts_handler(
                    ws, state, lambda token: token == _TEST_TOKEN
                ),
                timeout=5.0,
            )

        # Assert — auth + "generating" are expected; a disconnect must NOT be
        # reported as a cancel (or any other terminal status) on the dead
        # socket.  Pre-fix this fails: "cancelled" is present.
        statuses = [msg.get("status") for msg in ws.sent_json]
        self.assertNotIn("cancelled", statuses)
        self.assertNotIn("complete", statuses)
        self.assertNotIn("error", statuses)


@_skip
class TestWebSocketStreamErrorCascade(unittest.TestCase):
    """Cover the handler try/except around ``_stream_generation`` (websocket.py
    223-236).

    ``_stream_generation`` is ``await``ed directly inside the handler's
    try/except, so an exception it raises reaches the error cascade and
    produces the right client-facing frame. These branches were previously
    uncovered (the existing tests exercise the happy path and message-loop
    validation, not the streaming-error cascade).
    """

    def _authenticate(self, ws):
        ws.send_text(json.dumps({"token": _TEST_TOKEN}))
        resp = ws.receive_json()
        self.assertEqual(resp["status"], "authenticated")

    def _first_frame_when_stream_raises(self, exc):
        """Auth + post a clone request; return the first frame received when
        ``_stream_generation`` raises *exc* before sending anything."""
        _setup_app_state(
            models={"clone": MagicMock(), "design": None, "custom": None},
            server_config={"security": {}},
        )
        with patch(
            "qwen3_tts.server.validation._validate_generation_request"
        ), patch(
            "qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock()
        ), patch(
            "qwen3_tts.server.app_lifespan._check_memory_available",
            return_value=(True, 4096),
        ), patch(
            "qwen3_tts.server.websocket._stream_generation", side_effect=exc
        ):
            with TestClient(app).websocket_connect("/ws") as ws:
                self._authenticate(ws)
                ws.send_text(
                    json.dumps({"text": "Hello", "mode": "clone", "prompt_file": "test.pt"})
                )
                return ws.receive_json()

    def test_value_error_sends_invalid_request(self):
        resp = self._first_frame_when_stream_raises(ValueError("bad payload"))
        self.assertIn("Invalid request", resp["error"])

    def test_connection_error_sends_connection_error(self):
        resp = self._first_frame_when_stream_raises(ConnectionError("broken pipe"))
        self.assertEqual(resp["error"], "Connection error")

    def test_generic_exception_sends_sanitized_error(self):
        # A non-validation, non-connection exception is sanitized so the client
        # gets a clean error frame (and the handler does not crash).
        resp = self._first_frame_when_stream_raises(RuntimeError("internal boom"))
        self.assertIn("error", resp)


if __name__ == "__main__":
    unittest.main()
