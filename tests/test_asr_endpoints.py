#!/usr/bin/env python3
"""Tests for ASR server endpoints: /load-asr, /unload-asr, /transcribe.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_asr_endpoints.py -v

No GPU, models, or running server required.
"""

import base64
import unittest
from unittest.mock import patch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

try:
    import soundfile  # noqa: F401
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

_skip_server = unittest.skipUnless(
    HAS_FASTAPI,
    "requires fastapi + soundfile",
)



@_skip_server
class TestLoadASREndpoint(unittest.TestCase):
    """Test POST /load-asr endpoint."""

    def setUp(self):
        from qwen3_tts.server.app import app
        from tests.conftest import _init_app_state, _save_app_state

        self.app = app
        self.original_state = _save_app_state(app)
        _init_app_state(app, auth_token="test_token")
        app.state.models_loaded.set()
        app.state.server_config = {"models": {}}
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer test_token"}

    def tearDown(self):
        from tests.conftest import _restore_app_state
        _restore_app_state(self.app, self.original_state)

    @patch("qwen3_tts.core.engine.load_asr_model")
    def test_load_asr_success(self, mock_load):
        mock_load.return_value = True
        resp = self.client.post("/load-asr", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "loaded")
        mock_load.assert_called_once()

    @patch("qwen3_tts.core.engine.load_asr_model", side_effect=ImportError("mlx not installed"))
    def test_load_asr_import_error(self, mock_load):
        resp = self.client.post("/load-asr", headers=self.headers)
        self.assertEqual(resp.status_code, 500)

    @patch("qwen3_tts.core.engine.is_asr_loaded", return_value=True)
    def test_load_asr_already_loaded(self, mock_check):
        resp = self.client.post("/load-asr", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "already_loaded")

    def test_load_asr_no_auth(self):
        resp = self.client.post("/load-asr")
        self.assertEqual(resp.status_code, 401)



@_skip_server
class TestUnloadASREndpoint(unittest.TestCase):
    """Test POST /unload-asr endpoint."""

    def setUp(self):
        from qwen3_tts.server.app import app
        from tests.conftest import _init_app_state, _save_app_state

        self.app = app
        self.original_state = _save_app_state(app)
        _init_app_state(app, auth_token="test_token")
        app.state.models_loaded.set()
        app.state.server_config = {"models": {}}
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer test_token"}

    def tearDown(self):
        from tests.conftest import _restore_app_state
        _restore_app_state(self.app, self.original_state)

    @patch("qwen3_tts.core.engine.unload_asr_model")
    def test_unload_asr_success(self, mock_unload):
        resp = self.client.post("/unload-asr", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "unloaded")
        mock_unload.assert_called_once()

    def test_unload_asr_no_auth(self):
        resp = self.client.post("/unload-asr")
        self.assertEqual(resp.status_code, 401)



@_skip_server
class TestTranscribeEndpoint(unittest.TestCase):
    """Test POST /transcribe endpoint."""

    def setUp(self):
        from qwen3_tts.server.app import app
        from tests.conftest import _init_app_state, _save_app_state

        self.app = app
        self.original_state = _save_app_state(app)
        _init_app_state(app, auth_token="test_token")
        app.state.models_loaded.set()
        app.state.server_config = {"models": {}}
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer test_token"}

    def tearDown(self):
        from tests.conftest import _restore_app_state
        _restore_app_state(self.app, self.original_state)

    @patch("qwen3_tts.core.engine.transcribe_audio")
    def test_transcribe_success(self, mock_transcribe):
        mock_transcribe.return_value = "Hello world"
        audio_b64 = base64.b64encode(b"fake_audio_data").decode()
        resp = self.client.post(
            "/transcribe",
            json={"audio_base64": audio_b64, "language": "en"},
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["transcript"], "Hello world")

    @patch("qwen3_tts.core.engine.transcribe_audio")
    def test_transcribe_default_language(self, mock_transcribe):
        mock_transcribe.return_value = "Bonjour"
        audio_b64 = base64.b64encode(b"fake_audio_data").decode()
        resp = self.client.post(
            "/transcribe",
            json={"audio_base64": audio_b64},
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["transcript"], "Bonjour")

    @patch("qwen3_tts.core.engine.transcribe_audio", side_effect=RuntimeError("Transcription failed"))
    def test_transcribe_failure(self, mock_transcribe):
        audio_b64 = base64.b64encode(b"fake_audio_data").decode()
        resp = self.client.post(
            "/transcribe",
            json={"audio_base64": audio_b64},
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 500)

    def test_transcribe_no_auth(self):
        resp = self.client.post("/transcribe", json={"audio_base64": "abc"})
        self.assertEqual(resp.status_code, 401)

    def test_transcribe_missing_audio(self):
        """Missing audio_base64 field should fail validation."""
        resp = self.client.post(
            "/transcribe",
            json={},
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 422)



@_skip_server
class TestModelsEndpointASR(unittest.TestCase):
    """Test that GET /models includes ASR info."""

    def setUp(self):
        from qwen3_tts.server.app import app
        from tests.conftest import _init_app_state, _save_app_state

        self.app = app
        self.original_state = _save_app_state(app)
        _init_app_state(app, auth_token="test_token")
        app.state.models_loaded.set()
        app.state.server_config = {"models": {}}
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer test_token"}

    def tearDown(self):
        from tests.conftest import _restore_app_state
        _restore_app_state(self.app, self.original_state)

    @patch("qwen3_tts.core.engine.is_asr_loaded", return_value=False)
    @patch("qwen3_tts.core.engine.get_asr_model_info", return_value={"loaded": False, "backend": None, "model_name": None})
    def test_models_includes_asr_not_loaded(self, mock_info, mock_loaded):
        resp = self.client.get("/models", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("asr", data)
        self.assertFalse(data["asr"]["loaded"])

    @patch("qwen3_tts.core.engine.is_asr_loaded", return_value=True)
    @patch("qwen3_tts.core.engine.get_asr_model_info", return_value={"loaded": True, "backend": "mlx", "model_name": "whisper-large-v3-turbo"})
    def test_models_includes_asr_loaded(self, mock_info, mock_loaded):
        resp = self.client.get("/models", headers=self.headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("asr", data)
        self.assertTrue(data["asr"]["loaded"])
        self.assertEqual(data["asr"]["model_name"], "whisper-large-v3-turbo")


if __name__ == "__main__":
    unittest.main()
