"""Tests for qwen3_tts/server/client/models.py.

Covers: load_model(), unload_model(), get_models(),
update_model_config(), update_startup_config() — all with mocked HTTP calls.
No running server or GPU required.

Run with:
    python -m pytest tests/test_client_models.py -v --tb=short
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(data=None):
    """Create a temp config file and return its path."""
    if data is None:
        data = {
            "server": {"host": "127.0.0.1", "port": 5123},
            "presets": {},
            "aliases": {},
            "generation": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            "output_directory": "~/Downloads",
            "default_clone_prompt": "default.pt",
            "default_voice_description": "neutral voice",
            "default_speaker": "ryan",
            "language": "English",
            "prosody_presets": {},
        }
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return f.name


def _mock_error_response(status=500, message="something went wrong"):
    """Build a mock error response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"error": message}
    return resp


def _client_with_server(config_path):
    """Return (client, session_mock) with session pre-wired."""
    from qwen3_tts.server.client import TTSClient

    client = TTSClient(config_path=config_path)
    session = MagicMock()
    client._session = session
    return client, session


# ============================================================================
# load_model()
# ============================================================================


class TestLoadModel(unittest.TestCase):
    """load_model() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_load_model_success(self):
        """load_model returns response dict on success."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "loaded"}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.load_model("clone")
        self.assertEqual(result["status"], "loaded")

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["model_type"], "clone")

    def test_load_model_error_raises(self):
        """load_model raises ModelError on failure."""
        from qwen3_tts.core.config import ModelError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(500, "OOM")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(ModelError):
                client.load_model("clone")

    def test_load_model_requires_server(self):
        """load_model raises when server is down."""
        client, _ = _client_with_server(self.cfg)

        with patch.object(client, "is_server_running", return_value=False):
            with self.assertRaises(ConnectionError):
                client.load_model("clone")


# ============================================================================
# unload_model()
# ============================================================================


class TestUnloadModel(unittest.TestCase):
    """unload_model() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_unload_model_success(self):
        """unload_model returns response dict on success."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "unloaded"}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.unload_model("design")
        self.assertEqual(result["status"], "unloaded")

    def test_unload_model_409_accepted(self):
        """unload_model accepts 409 (already unloaded) without raising."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=409,
            json=MagicMock(return_value={"status": "already_unloaded"}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.unload_model("design")
        self.assertEqual(result["status"], "already_unloaded")

    def test_unload_model_error_raises(self):
        """unload_model raises ModelError on 500."""
        from qwen3_tts.core.config import ModelError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(500, "internal error")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(ModelError):
                client.unload_model("design")


# ============================================================================
# get_models()
# ============================================================================


class TestGetModels(unittest.TestCase):
    """get_models() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_get_models_returns_data(self):
        """get_models returns model info dict."""
        client, session = _client_with_server(self.cfg)
        expected = {"models": {"clone": {"loaded": True}}, "backend": "mlx"}
        session.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=expected),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.get_models()
        self.assertEqual(result["backend"], "mlx")

    def test_get_models_requires_server(self):
        """get_models raises when server is down."""
        client, _ = _client_with_server(self.cfg)

        with patch.object(client, "is_server_running", return_value=False):
            with self.assertRaises(ConnectionError):
                client.get_models()


# ============================================================================
# update_model_config()
# ============================================================================


class TestUpdateModelConfig(unittest.TestCase):
    """update_model_config() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_update_model_config_success(self):
        """update_model_config sends correct payload and returns result."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "updated", "changes": {}}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.update_model_config(model_size="0.6B", mlx_quantization="4bit")

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["model_size"], "0.6B")
        self.assertEqual(payload["mlx_quantization"], "4bit")
        self.assertEqual(result["status"], "updated")

    def test_update_model_config_no_args_raises(self):
        """update_model_config raises ValueError with no arguments."""
        client, _ = _client_with_server(self.cfg)

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                client.update_model_config()
            self.assertIn("At least one", str(ctx.exception))

    def test_update_model_config_error_raises(self):
        """update_model_config raises ModelError on failure."""
        from qwen3_tts.core.config import ModelError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(400, "invalid size")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(ModelError):
                client.update_model_config(model_size="99B")


# ============================================================================
# update_startup_config()
# ============================================================================


class TestUpdateStartupConfig(unittest.TestCase):
    """update_startup_config() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_update_startup_config_success(self):
        """update_startup_config sends correct payload."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "updated", "changes": {}}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.update_startup_config(clone=True, design=False)

        payload = session.post.call_args[1]["json"]
        self.assertTrue(payload["clone"])
        self.assertFalse(payload["design"])
        self.assertNotIn("custom", payload)
        self.assertEqual(result["status"], "updated")

    def test_update_startup_config_no_args_raises(self):
        """update_startup_config raises ValueError with no arguments."""
        client, _ = _client_with_server(self.cfg)

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                client.update_startup_config()
            self.assertIn("At least one", str(ctx.exception))

    def test_update_startup_config_error_raises(self):
        """update_startup_config raises ModelError on failure."""
        from qwen3_tts.core.config import ModelError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(500, "server error")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(ModelError):
                client.update_startup_config(clone=True)


if __name__ == "__main__":
    unittest.main()
