#!/usr/bin/env python3
"""Tests for /create-voice-prompt server endpoint.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_create_voice_endpoint.py -v

No GPU, models, or running server required.
"""

import base64
import unittest
from unittest.mock import MagicMock, patch

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

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_skip_torch = unittest.skipUnless(HAS_TORCH, "requires torch")


@pytest.mark.unit
@_skip_server
class TestCreateVoicePromptEndpoint(unittest.TestCase):
    """Test POST /create-voice-prompt endpoint."""

    def setUp(self):
        from qwen3_tts.server.app import app
        from tests.conftest import _init_app_state, _save_app_state

        self.app = app
        self.original_state = _save_app_state(app)
        _init_app_state(app, auth_token="test_token")
        app.state.models_loaded.set()
        app.state.server_config = {"models": {}}
        # Provide a mock clone model
        app.state.models["clone"] = MagicMock()
        self.client = TestClient(app)
        self.headers = {"Authorization": "Bearer test_token"}

    def tearDown(self):
        from tests.conftest import _restore_app_state
        _restore_app_state(self.app, self.original_state)

    @_skip_torch
    @patch("qwen3_tts.core.engine.clear_voice_prompt_cache")
    @patch("torch.save")
    @patch("qwen3_tts.core.engine.create_voice_prompt")
    @patch("qwen3_tts.core.engine.load_audio_for_cloning")
    def test_create_prompt_success(self, mock_load_audio, mock_create, mock_save, mock_cache):
        import numpy as np
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_create.return_value = MagicMock()  # fake voice prompt tensor

        audio_b64 = base64.b64encode(b"fake_wav_bytes").decode()
        # Torch branch FORCED via the app.get_backend seam: the torch-shaped
        # patches above only intercept the torch branch, and the ambient
        # backend must not choose for us (mlx machines would take the MLX
        # branch and 400 on the undecodable b"fake_wav_bytes").
        with patch("qwen3_tts.server.app.get_backend", return_value="torch"):
            resp = self.client.post(
                "/create-voice-prompt",
                json={
                    "audio_base64": audio_b64,
                    "transcript": "Hello world",
                    "name": "test_voice",
                },
                headers=self.headers,
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "created")
        self.assertEqual(data["name"], "test_voice")

    def test_create_prompt_no_auth(self):
        resp = self.client.post(
            "/create-voice-prompt",
            json={"audio_base64": "abc", "name": "test"},
        )
        self.assertEqual(resp.status_code, 401)

    def test_create_prompt_missing_name(self):
        """Missing name field should fail validation."""
        audio_b64 = base64.b64encode(b"fake").decode()
        resp = self.client.post(
            "/create-voice-prompt",
            json={"audio_base64": audio_b64},
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 422)

    def test_create_prompt_invalid_name(self):
        """Name with special chars should be rejected."""
        audio_b64 = base64.b64encode(b"fake").decode()
        resp = self.client.post(
            "/create-voice-prompt",
            json={
                "audio_base64": audio_b64,
                "name": "../etc/passwd",
                "transcript": "test",
            },
            headers=self.headers,
        )
        self.assertEqual(resp.status_code, 400)

    def test_create_prompt_clone_not_loaded(self):
        """Should fail if clone model is not loaded (torch branch, FORCED:
        the MLX branch is inference-free and needs no clone model, so the
        gate is pinned on the explicitly-selected backend rather than the
        ambient one)."""
        self.app.state.models["clone"] = None
        audio_b64 = base64.b64encode(b"fake").decode()
        with patch("qwen3_tts.server.app.get_backend", return_value="torch"):
            resp = self.client.post(
                "/create-voice-prompt",
                json={
                    "audio_base64": audio_b64,
                    "name": "test_voice",
                    "transcript": "hello",
                },
                headers=self.headers,
            )
        self.assertEqual(resp.status_code, 503)

    @_skip_torch
    @patch("qwen3_tts.core.engine.clear_voice_prompt_cache")
    @patch("torch.save")
    @patch("qwen3_tts.core.engine.create_voice_prompt")
    @patch("qwen3_tts.core.engine.load_audio_for_cloning")
    def test_create_prompt_no_transcript_mode(self, mock_load_audio, mock_create, mock_save, mock_cache):
        """no_transcript flag should pass empty transcript."""
        import numpy as np
        mock_load_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        mock_create.return_value = MagicMock()

        audio_b64 = base64.b64encode(b"fake_wav_bytes").decode()
        # Torch branch FORCED (see test_create_prompt_success): the torch-
        # shaped patches above only intercept the torch branch.
        with patch("qwen3_tts.server.app.get_backend", return_value="torch"):
            resp = self.client.post(
                "/create-voice-prompt",
                json={
                    "audio_base64": audio_b64,
                    "name": "test_voice_nt",
                    "no_transcript": True,
                },
                headers=self.headers,
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "created")
        # no_transcript must flow to x_vector_only_mode so upstream doesn't
        # raise 'ref_text is required when x_vector_only_mode=False'.
        mock_create.assert_called_once()
        self.assertEqual(mock_create.call_args.args[3], "")  # transcript
        self.assertIs(
            mock_create.call_args.kwargs.get("x_vector_only_mode"), True
        )


@pytest.mark.unit
@_skip_server
class TestModelTableASRRow(unittest.TestCase):
    """Test that get_model_table_data includes ASR row."""

    @patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True)
    @patch("qwen3_tts.interface.ui.model_management.load_config", return_value={"models": {}})
    def test_asr_row_in_table(self, mock_config, mock_running):
        """Model table should include an ASR row."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": {
                "clone": {"loaded": True, "memory_mb": 3500},
                "design": {"loaded": False, "memory_mb": 3500},
                "custom": {"loaded": False, "memory_mb": 3500},
            },
            "asr": {"loaded": True, "model_name": "whisper-large-v3-turbo", "backend": "mlx"},
            "backend": "mlx",
            "model_size": "1.7B",
        }

        # get_model_table_data calls server_request (the single HTTP choke-point),
        # so patching it directly is sufficient. Do NOT swap sys.modules["requests"]
        # for a MagicMock: importing modules while that swap is live corrupts their
        # import state on Python 3.10 and leaks into later tests (it broke
        # test_stream_error_frame's server_request patch in batch 3).
        from qwen3_tts.interface.ui import model_management as mm

        with patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
            rows = mm.get_model_table_data()

        # Should have 4 rows: clone, design, custom, asr
        self.assertEqual(len(rows), 4)
        model_types = [row[0] for row in rows]
        self.assertIn("asr", model_types)

    @patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=False)
    @patch("qwen3_tts.interface.ui.model_management.load_config", return_value={"models": {}})
    def test_asr_row_in_fallback(self, mock_config, mock_running):
        """Fallback rows (server not running) should include ASR."""
        from qwen3_tts.interface.ui.model_management import get_model_table_data
        rows = get_model_table_data()

        self.assertEqual(len(rows), 4)
        model_types = [row[0] for row in rows]
        self.assertIn("asr", model_types)


if __name__ == "__main__":
    unittest.main()
