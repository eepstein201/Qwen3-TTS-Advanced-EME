#!/usr/bin/env python3
"""Extended FastAPI server tests — second batch, part 2.

Split from test_fastapi_app_ext2.py (over 800 lines).
Covers: run_server, lifespan, CORS regex, ETA estimation.

Run: python -m pytest tests/test_fastapi_app_ext2_part2.py -v
"""
import pathlib
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

_APP = "qwen3_tts.server.app"
_APP_LIFESPAN = "qwen3_tts.server.app_lifespan"
_APP_GENERATION = "qwen3_tts.server.app_generation"


def _make_app_state(**overrides):
    """Create a minimal mock app_state with required attributes."""
    state = MagicMock()
    state.models = {"clone": None, "design": None, "custom": None}
    state.model_load_times = {}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
    state.last_activity = 0
    state.shutdown_timer = None
    state.server_config = {"auto_shutdown_minutes": 0, "models": {}}
    state.gen_cache = {}
    state.gen_cache_lock = threading.Lock()
    state.models_loaded = threading.Event()
    state.shutdown_event = MagicMock()
    state.generation_lock = AsyncMock()
    state.generation_lock.__aenter__ = AsyncMock(return_value=None)
    state.generation_lock.__aexit__ = AsyncMock(return_value=None)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


# ---------------------------------------------------------------------------
# run_server
# ---------------------------------------------------------------------------

class TestRunServer(unittest.TestCase):

    def test_run_server_public_binds_all(self):
        from qwen3_tts.server.app import run_server
        mock_handler = MagicMock()
        mock_handler.level = 0
        with patch(f"{_APP}.uvicorn") as mock_uv, \
             patch(f"{_APP}.IN_COLAB", False), \
             patch("qwen3_tts.core.config.LOG_FILE", "/tmp/fake_log.log"), \
             patch("logging.handlers.RotatingFileHandler", return_value=mock_handler), \
             patch("logging.StreamHandler", return_value=mock_handler), \
             patch("builtins.print"):
            run_server(host="127.0.0.1", port=5123, public=True)
        mock_uv.run.assert_called_once()
        # Verify host was changed to 0.0.0.0 for public
        call_kwargs = mock_uv.run.call_args
        self.assertEqual(call_kwargs.kwargs.get("host"), "0.0.0.0")

    def test_run_server_colab_binds_all(self):
        from qwen3_tts.server.app import run_server
        mock_handler = MagicMock()
        mock_handler.level = 0
        with patch(f"{_APP}.uvicorn") as mock_uv, \
             patch(f"{_APP}.IN_COLAB", True), \
             patch("qwen3_tts.core.config.LOG_FILE", "/tmp/fake_log.log"), \
             patch("logging.handlers.RotatingFileHandler", return_value=mock_handler), \
             patch("logging.StreamHandler", return_value=mock_handler), \
             patch("builtins.print"):
            run_server(host="127.0.0.1", port=5123, public=False)
        mock_uv.run.assert_called_once()
        self.assertEqual(mock_uv.run.call_args.kwargs.get("host"), "0.0.0.0")


# ---------------------------------------------------------------------------
# Lifespan startup/shutdown
# ---------------------------------------------------------------------------

class TestLifespan(unittest.TestCase):

    def test_lifespan_initializes_state(self):
        """Test that the lifespan context manager initializes app state correctly."""
        import asyncio

        from qwen3_tts.server.app import app, lifespan

        mock_config = {
            "server": {"auto_shutdown_minutes": 0},
            "models": {"clone": {"load_at_startup": False}},
            "security": {"max_text_length": 10000},
        }

        async def _run():
            with patch(f"{_APP_LIFESPAN}.load_config", return_value=mock_config), \
                 patch(f"{_APP_LIFESPAN}.TOKEN_FILE", pathlib.Path("/tmp/test_token_xyz")), \
                 patch(f"{_APP_LIFESPAN}._acquire_startup_lock", return_value=MagicMock()), \
                 patch(f"{_APP_LIFESPAN}._background_load"), \
                 patch(f"{_APP_LIFESPAN}.cleanup_resources"), \
                 patch(f"{_APP_LIFESPAN}.cleanup_pid_file"), \
                 patch("atexit.register"), \
                 patch("builtins.open", MagicMock()), \
                 patch("os.chmod"), \
                 patch("os.unlink"), \
                 patch("fcntl.flock"):
                async with lifespan(app):
                    # Verify state was initialized during startup
                    self.assertIsNotNone(app.state.auth_token)
                    self.assertEqual(
                        set(app.state.models.keys()),
                        {"clone", "design", "custom"},
                    )
                    self.assertIsNotNone(app.state.generation_lock)

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# IN_COLAB CORS regex
# ---------------------------------------------------------------------------

class TestColabCors(unittest.TestCase):

    def test_colab_cors_regex_matches_gradio_live(self):
        """Verify the Colab CORS regex allows *.gradio.live origins."""
        import re
        colab_regex = (
            r"(^https?://(localhost|127\.0\.0\.1)(:\d+)?$)"
            r"|(^https://[a-z0-9-]+\.gradio\.live$)"
        )
        self.assertTrue(re.match(colab_regex, "https://abc-123.gradio.live"))
        self.assertTrue(re.match(colab_regex, "http://localhost:7860"))
        self.assertFalse(re.match(colab_regex, "https://evil.com"))


# ---------------------------------------------------------------------------
# _estimate_eta
# ---------------------------------------------------------------------------

class TestEstimateEta(unittest.TestCase):

    def test_no_history(self):
        from qwen3_tts.server.app import _estimate_eta
        state = _make_app_state()
        state.eta_cache = {"median_rate": None, "last_updated": 0}
        with patch(f"{_APP_LIFESPAN}.get_eta_cache_ttl", return_value=60), \
             patch(f"{_APP_LIFESPAN}.HISTORY_FILE", "/nonexistent_history.jsonl"), \
             patch("os.path.exists", return_value=False):
            result = _estimate_eta(state, 100, 5.0)
        self.assertIsNone(result)

    def test_with_cached_rate(self):
        from qwen3_tts.server.app import _estimate_eta
        state = _make_app_state()
        # median_rate = chars/sec; fresh cache
        state.eta_cache = {"median_rate": 10.0, "last_updated": time.time()}
        with patch(f"{_APP_LIFESPAN}.get_eta_cache_ttl", return_value=60):
            result = _estimate_eta(state, 100, 5.0)
        # estimated_total = 100/10 = 10s, remaining = 10 - 5 = 5.0
        self.assertAlmostEqual(result, 5.0, places=1)


if __name__ == "__main__":
    unittest.main()
