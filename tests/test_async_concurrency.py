#!/usr/bin/env python3
"""Async concurrency and base64 response tests.

Verifies:
- /generate returns base64 audio instead of file paths
- /health remains responsive during slow inference

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_async_concurrency.py -v
"""

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi.testclient import TestClient
    import soundfile  # noqa: F401
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi, soundfile, numpy")


def _setup_app_state(app):
    """Initialize app.state for tests."""
    import threading

    mock_lock = AsyncMock()
    mock_lock.__aenter__.return_value = None
    mock_lock.__aexit__.return_value = None

    app.state.auth_token = "test_token"  # nosec B105
    app.state.models = {"clone": MagicMock(), "design": None, "custom": None}
    app.state.model_load_times = {}
    app.state.generation_lock = mock_lock
    app.state.generation_state = {
        "active": False,
        "start_time": 0.0,
        "text_length": 0,
        "mode": "",
        "batch_index": 0,
        "batch_total": 0,
        "chunk_index": 0,
        "chunk_total": 0,
        "generation_id": None,
        "cancelled": False,
    }
    app.state.request_queue = set()
    app.state.last_activity = 0
    app.state.models_loaded = threading.Event()
    app.state.gen_cache = {}
    app.state.gen_cache_lock = threading.Lock()
    app.state.inference_lock = AsyncMock()
    app.state.inference_lock.__aenter__.return_value = None
    app.state.inference_lock.__aexit__.return_value = None
    app.state.eta_cache = {"median_rate": None, "last_updated": 0}
    app.state.model_load_errors = {"clone": None, "design": None, "custom": None}
    app.state.shutdown_timer = None
    app.state.server_config = {
        "security": {"max_text_length": 10000, "max_batch_size": 20},
        "auto_shutdown_minutes": 0,
    }


@_skip
class TestBase64AudioResponse(unittest.TestCase):
    """Test that /generate returns base64-encoded audio."""

    @patch("qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock())
    @patch("qwen3_tts.core.engine.run_inference")
    def test_generate_returns_base64_audio(self, mock_inference, mock_load_prompt):
        """Response should contain audio_base64 key, not file path."""
        # Return a known numpy array
        test_audio = np.sin(np.linspace(0, 2 * np.pi, 24000)).astype(np.float32)
        mock_inference.return_value = (test_audio, 24000)

        from qwen3_tts.server.app import app
        _setup_app_state(app)

        client = TestClient(app)
        resp = client.post(
            "/generate",
            json={"texts": ["Hello world"], "mode": "clone", "prompt_file": "test.pt"},
            headers={"Authorization": "Bearer test_token"},
        )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        result = data["results"][0]

        # Must have audio_base64, not file
        self.assertIn("audio_base64", result)
        self.assertNotIn("file", result)
        self.assertIn("sample_rate", result)

        # Decode and verify it's valid WAV
        import base64, io, soundfile as sf
        audio_bytes = base64.b64decode(result["audio_base64"])
        wav, sr = sf.read(io.BytesIO(audio_bytes))
        self.assertEqual(sr, 24000)
        self.assertGreater(len(wav), 0)

    @patch("qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock())
    @patch("qwen3_tts.core.engine.run_inference")
    def test_generate_no_file_key_in_response(self, mock_inference, mock_load_prompt):
        """Ensure no file paths leak into the response."""
        test_audio = np.zeros(4800, dtype=np.float32)
        mock_inference.return_value = (test_audio, 24000)

        from qwen3_tts.server.app import app
        _setup_app_state(app)

        client = TestClient(app)
        resp = client.post(
            "/generate",
            json={"texts": ["Test"], "mode": "clone", "prompt_file": "test.pt"},
            headers={"Authorization": "Bearer test_token"},
        )

        self.assertEqual(resp.status_code, 200)
        for result in resp.json()["results"]:
            self.assertNotIn("file", result)
            self.assertNotIn("cleanup", result)


@_skip
class TestHealthDuringGeneration(unittest.TestCase):
    """Test that /health responds while inference is running."""

    @patch("qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock())
    @patch("qwen3_tts.core.engine.run_inference")
    def test_health_responds_during_generation(self, mock_inference, mock_load_prompt):
        """Health endpoint should respond quickly even during slow generation."""
        import threading

        def slow_inference(*args, **kwargs):
            time.sleep(2)
            return (np.zeros(4800, dtype=np.float32), 24000)

        mock_inference.side_effect = slow_inference

        from qwen3_tts.server.app import app
        _setup_app_state(app)

        client = TestClient(app)

        # Start generation in background
        gen_result = [None]

        def do_generate():
            gen_result[0] = client.post(
                "/generate",
                json={"texts": ["Slow test"], "mode": "clone", "prompt_file": "test.pt"},
                headers={"Authorization": "Bearer test_token"},
            )

        gen_thread = threading.Thread(target=do_generate)
        gen_thread.start()

        # Give generation a moment to start
        time.sleep(0.3)

        # Health should respond quickly
        start = time.time()
        health_resp = client.get("/health")
        elapsed = time.time() - start

        self.assertIn(health_resp.status_code, (200, 503))
        self.assertLess(elapsed, 1.0, f"/health took {elapsed:.2f}s during generation")

        gen_thread.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
