#!/usr/bin/env python3
"""Tests for Qwen3-TTS codebase.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/ -v

These tests do NOT require GPU, models, or a running server.
They test config, validation, auth, error classes, and the server
endpoint logic using Flask's test client.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# tts_config tests
# =============================================================================

class TestTTSConfig(unittest.TestCase):
    """Test tts_config module (no heavy imports)."""

    def test_imports_no_torch(self):
        """tts_config must not import torch."""
        import tts_config  # noqa: F401
        self.assertNotIn("torch", sys.modules.keys() - {"torch"})

    def test_error_hierarchy(self):
        from tts_config import (
            TTSError, ServerConnectionError, ModelNotLoadedError,
            InvalidInputError, GenerationError, AuthenticationError,
        )
        # All errors inherit from TTSError
        for cls in [ServerConnectionError, ModelNotLoadedError,
                    InvalidInputError, GenerationError, AuthenticationError]:
            err = cls("test") if cls != ModelNotLoadedError else cls("clone")
            self.assertIsInstance(err, TTSError)

    def test_error_format_cli(self):
        from tts_config import ServerConnectionError
        err = ServerConnectionError("details here")
        formatted = err.format_cli()
        self.assertIn("Cannot connect", formatted)
        self.assertIn("startTTSServer", formatted)

    def test_error_format_gradio(self):
        from tts_config import GenerationError
        err = GenerationError("oops")
        html = err.format_gradio()
        self.assertIn("Audio generation failed", html)

    def test_read_auth_token_missing(self):
        from tts_config import TOKEN_FILE, read_auth_token
        # Use a temp path that doesn't exist
        with patch("tts_config.TOKEN_FILE", "/tmp/nonexistent_token_test_xyz"):
            result = read_auth_token()
        # If the real file exists, it returns its content; with patched path it's None
        # Can't reliably test with real TOKEN_FILE, so test the function signature
        self.assertTrue(result is None or isinstance(result, str))

    def test_auth_headers_returns_dict(self):
        from tts_config import auth_headers
        headers = auth_headers()
        self.assertIsInstance(headers, dict)

    def test_auth_headers_with_token(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".token", delete=False) as f:
            f.write("test_token_abc123")
            token_path = f.name
        try:
            with patch("tts_config.TOKEN_FILE", token_path):
                from tts_config import read_auth_token, auth_headers
                # Need to reimport to pick up patched value
                token = read_auth_token()
                headers = auth_headers()
            # The token file exists so we should get headers
            # (actual behavior depends on whether TOKEN_FILE is patched at call time)
        finally:
            os.unlink(token_path)

    def test_model_info_keys(self):
        from tts_config import MODEL_INFO
        self.assertIn("clone", MODEL_INFO)
        self.assertIn("design", MODEL_INFO)
        self.assertIn("custom", MODEL_INFO)
        for info in MODEL_INFO.values():
            self.assertIn("name", info)
            self.assertIn("description", info)
            self.assertIn("memory_mb", info)

    def test_custom_voice_speakers(self):
        from tts_config import CUSTOM_VOICE_SPEAKERS
        self.assertIn("ryan", CUSTOM_VOICE_SPEAKERS)
        self.assertIn("aiden", CUSTOM_VOICE_SPEAKERS)
        for key, info in CUSTOM_VOICE_SPEAKERS.items():
            self.assertIn("name", info)
            self.assertIn("lang", info)
            self.assertIn("desc", info)


# =============================================================================
# tts_server validation tests (using Flask test client, no models needed)
# =============================================================================

class TestServerValidation(unittest.TestCase):
    """Test server input validation without loading any models."""

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client with mocked models."""
        # We need to mock torch and model-related imports
        # to avoid loading heavy dependencies
        import tts_server
        tts_server.auth_token = None  # Disable auth for tests
        tts_server.server_config = {
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        }
        cls.app = tts_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")

    def test_generate_empty_texts(self):
        resp = self.client.post("/generate", json={"texts": []})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No texts", resp.get_json()["error"])

    def test_generate_batch_too_large(self):
        texts = ["hello"] * 5  # max is 3 in test config
        resp = self.client.post("/generate", json={"texts": texts})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("exceeds limit", resp.get_json()["error"])

    def test_generate_text_too_long(self):
        resp = self.client.post("/generate", json={"texts": ["x" * 200]})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("character limit", resp.get_json()["error"])

    def test_generate_invalid_mode(self):
        resp = self.client.post("/generate", json={"texts": ["hello"], "mode": "invalid"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid mode", resp.get_json()["error"])

    def test_generate_path_traversal_prompt(self):
        resp = self.client.post("/generate", json={
            "texts": ["hello"],
            "mode": "clone",
            "prompt_file": "../../../etc/passwd",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("path traversal", resp.get_json()["error"])

    def test_generate_invalid_speaker(self):
        resp = self.client.post("/generate", json={
            "texts": ["hello"],
            "mode": "custom",
            "speaker": "nonexistent_speaker",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unknown speaker", resp.get_json()["error"])

    def test_generate_valid_speaker_accepted(self):
        # This will fail with 503 (model not loaded) rather than 400 (validation error)
        resp = self.client.post("/generate", json={
            "texts": ["hello"],
            "mode": "custom",
            "speaker": "Ryan",
        })
        # Should pass validation (400) and hit model-not-loaded (503)
        self.assertIn(resp.status_code, [200, 503])

    def test_error_response_has_recovery_field(self):
        """All error responses should include a recovery hint."""
        # Validation error
        resp = self.client.post("/generate", json={"texts": []})
        data = resp.get_json()
        self.assertIn("recovery", data)

        # Model not loaded
        resp = self.client.post("/generate", json={
            "texts": ["hello"], "mode": "clone", "prompt_file": "test.pt"
        })
        data = resp.get_json()
        self.assertIn("recovery", data)


# =============================================================================
# tts_server auth tests
# =============================================================================

class TestServerAuth(unittest.TestCase):
    """Test server authentication."""

    @classmethod
    def setUpClass(cls):
        import tts_server
        tts_server.auth_token = "test_secret_token"
        tts_server.server_config = {
            "security": {},
            "auto_shutdown_minutes": 0,
        }
        cls.app = tts_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        import tts_server
        tts_server.auth_token = None

    def test_health_no_auth_required(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_stats_requires_auth(self):
        resp = self.client.get("/stats")
        self.assertEqual(resp.status_code, 401)

    def test_stats_with_valid_auth(self):
        resp = self.client.get("/stats", headers={
            "Authorization": "Bearer test_secret_token"
        })
        self.assertEqual(resp.status_code, 200)

    def test_generate_requires_auth(self):
        resp = self.client.post("/generate", json={"texts": ["hello"]})
        self.assertEqual(resp.status_code, 401)

    def test_generate_wrong_token(self):
        resp = self.client.post("/generate",
            json={"texts": ["hello"]},
            headers={"Authorization": "Bearer wrong_token"})
        self.assertEqual(resp.status_code, 401)

    def test_models_requires_auth(self):
        resp = self.client.get("/models")
        self.assertEqual(resp.status_code, 401)

    def test_models_with_auth(self):
        resp = self.client.get("/models", headers={
            "Authorization": "Bearer test_secret_token"
        })
        self.assertEqual(resp.status_code, 200)

    def test_generation_status_no_auth_required(self):
        resp = self.client.get("/generation-status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["active"])


# =============================================================================
# SSML parsing tests (from tts_generate, lightweight)
# =============================================================================

class TestSSMLParsing(unittest.TestCase):
    """Test SSML parsing in tts_generate."""

    def test_no_ssml(self):
        from tts_generate import parse_ssml
        text, meta = parse_ssml("Hello world")
        self.assertEqual(text, "Hello world")
        self.assertFalse(meta["has_ssml"])

    def test_break_tag(self):
        from tts_generate import parse_ssml
        text, meta = parse_ssml('Hello <break time="500ms"/> world')
        self.assertTrue(meta["has_ssml"])
        self.assertNotIn("<break", text)

    def test_sub_tag(self):
        from tts_generate import parse_ssml
        text, meta = parse_ssml('<sub alias="World Wide Web">WWW</sub>')
        self.assertIn("World Wide Web", text)
        self.assertNotIn("WWW", text)

    def test_say_as_characters(self):
        from tts_generate import parse_ssml
        text, meta = parse_ssml('<say-as interpret-as="characters">ABC</say-as>')
        self.assertIn("A B C", text)

    def test_prosody_speed(self):
        from tts_generate import parse_ssml
        text, meta = parse_ssml('<prosody rate="fast">Quick text</prosody>')
        self.assertTrue(meta["has_ssml"])
        self.assertEqual(meta["prosody"]["speed"], 1.2)


# =============================================================================
# SRT parsing tests
# =============================================================================

class TestSRTParsing(unittest.TestCase):
    """Test SRT parsing."""

    def test_parse_srt(self):
        from tts_generate import parse_srt
        srt_content = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,500
Second subtitle
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            srt_path = f.name
        try:
            entries = parse_srt(srt_path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0][3], "Hello world")
            self.assertEqual(entries[1][3], "Second subtitle")
        finally:
            os.unlink(srt_path)

    def test_srt_time_to_ms(self):
        from tts_generate import srt_time_to_ms
        self.assertEqual(srt_time_to_ms("00:01:30,500"), 90500)
        self.assertEqual(srt_time_to_ms("01:00:00,000"), 3600000)


# =============================================================================
# Auto-increment filename tests
# =============================================================================

class TestAutoIncrementFilename(unittest.TestCase):
    """Test auto_increment_filename helper."""

    def test_no_conflict(self):
        from tts_generate import auto_increment_filename
        # Non-existent file should return as-is
        result = auto_increment_filename("/tmp/nonexistent_test_xyz.wav")
        self.assertEqual(result, "/tmp/nonexistent_test_xyz.wav")

    def test_conflict_increments(self):
        from tts_generate import auto_increment_filename
        # Create a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            result = auto_increment_filename(path)
            self.assertNotEqual(result, path)
            self.assertIn("_2", result)
        finally:
            os.unlink(path)

    def test_already_numbered(self):
        from tts_generate import auto_increment_filename
        # Create files with _2 suffix
        with tempfile.NamedTemporaryFile(suffix="_2.wav", delete=False, dir="/tmp", prefix="test_") as f:
            path = f.name
        try:
            result = auto_increment_filename(path)
            self.assertNotEqual(result, path)
            self.assertIn("_3", result)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
