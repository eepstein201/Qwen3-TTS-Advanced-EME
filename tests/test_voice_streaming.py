"""Streaming tests extracted from test_voice.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.voice_test_helpers import (
    _skip_server, _skip_client, _make_test_client,
)


class TestStreaming(unittest.TestCase):
    """Test streaming inference API."""

    def test_run_inference_streaming_exists(self):
        """run_inference_streaming function is importable."""
        from qwen3_tts.core.engine import run_inference_streaming
        self.assertTrue(callable(run_inference_streaming))

    def test_mlx_streaming_function_exists(self):
        """_run_inference_mlx_streaming function is importable."""
        from qwen3_tts.core.engine.inference import _run_inference_mlx_streaming
        self.assertTrue(callable(_run_inference_mlx_streaming))

    def test_streaming_torch_falls_back_to_chunked(self):
        """run_inference_streaming for torch uses chunked inference (not native streaming)."""
        from qwen3_tts.core.engine import run_inference_streaming
        import inspect
        source = inspect.getsource(run_inference_streaming)
        # Torch backend falls back to chunked approach
        self.assertIn("_run_inference_single", source)

    def test_streaming_mlx_function_signature(self):
        """_run_inference_mlx_streaming has correct parameters."""
        from qwen3_tts.core.engine.inference import _run_inference_mlx_streaming
        import inspect
        sig = inspect.signature(_run_inference_mlx_streaming)
        params = list(sig.parameters.keys())
        self.assertIn("model", params)
        self.assertIn("text", params)
        self.assertIn("mode", params)
        self.assertIn("gen_params", params)


@_skip_server
class TestStreamingServerEndpoint(unittest.TestCase):
    """Test /generate-stream server endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {"max_text_length": 1000, "max_batch_size": 10},
            "auto_shutdown_minutes": 0,
        })
        app.state.models_loaded.set()  # simulate models ready

    @classmethod
    def tearDownClass(cls):
        from qwen3_tts.server.app import app
        app.state.auth_token = "test_token"  # nosec B105

    def test_generate_stream_requires_auth(self):
        """POST /generate-stream requires authentication."""
        resp = self.client.post("/generate-stream", json={
            "text": "hello", "mode": "design"
        })
        self.assertEqual(resp.status_code, 401)

    def test_generate_stream_validates_text(self):
        """POST /generate-stream validates text input."""
        resp = self.client.post("/generate-stream",
            json={"mode": "design"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("text", resp.json()["detail"].lower())

    def test_generate_stream_validates_mode(self):
        """POST /generate-stream validates mode."""
        resp = self.client.post("/generate-stream",
            json={"text": "hello", "mode": "invalid"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("mode", resp.json()["detail"].lower())


@_skip_client
class TestStreamingClientMethod(unittest.TestCase):
    """Test TTSClient.generate_streaming method."""

    def test_generate_streaming_method_exists(self):
        """TTSClient has generate_streaming method."""
        from qwen3_tts.server.client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "generate_streaming"))
        self.assertTrue(callable(getattr(client, "generate_streaming")))

    def test_cancel_generation_method_exists(self):
        """TTSClient has cancel_generation method."""
        from qwen3_tts.server.client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "cancel_generation"))
        self.assertTrue(callable(getattr(client, "cancel_generation")))

    def test_generate_streaming_signature(self):
        """generate_streaming has expected parameters."""
        import inspect
        from qwen3_tts.server.client import TTSClient
        sig = inspect.signature(TTSClient.generate_streaming)
        params = list(sig.parameters.keys())
        # Should have text, mode, and various optional params
        self.assertIn("text", params)
        self.assertIn("mode", params)


if __name__ == "__main__":
    unittest.main()
