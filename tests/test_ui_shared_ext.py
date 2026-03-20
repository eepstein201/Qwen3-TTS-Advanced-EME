#!/usr/bin/env python3
"""Extended tests for qwen3_tts.interface.ui.shared module.

Covers:
  - enhance_description_with_ai: validation, provider dispatch
  - is_enhancer_available: config checks
  - get_current_model_settings: server fetch + fallback
  - apply_model_settings: post config to server
  - get_server_status: stats parsing
  - format_status_display: HTML rendering
  - get_voice_prompts: MLX vs torch listing
  - add_to_history / get_history_data

Run: pytest tests/test_ui_shared_ext.py -v
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

_MOD = "qwen3_tts.interface.ui.shared"


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestEnhanceDescription(unittest.TestCase):
    """Tests for enhance_description_with_ai."""

    def test_empty_raises(self):
        from qwen3_tts.interface.ui.shared import enhance_description_with_ai
        with self.assertRaises(gr.Error):
            enhance_description_with_ai("")

    def test_not_enabled_raises(self):
        from qwen3_tts.interface.ui.shared import enhance_description_with_ai
        with patch(f"{_MOD}.load_config", return_value={"prompt_enhancer": {"enabled": False}}):
            with self.assertRaises(gr.Error):
                enhance_description_with_ai("warm voice")

    def test_missing_api_key_raises(self):
        from qwen3_tts.interface.ui.shared import enhance_description_with_ai
        config = {"prompt_enhancer": {"enabled": True, "api_key_env": "FAKE_KEY_12345"}}
        with patch(f"{_MOD}.load_config", return_value=config), \
             patch.dict(os.environ, {}, clear=False):
            # Make sure FAKE_KEY_12345 is not set
            os.environ.pop("FAKE_KEY_12345", None)
            with self.assertRaises(gr.Error):
                enhance_description_with_ai("warm voice")

    def test_unsupported_provider_raises(self):
        from qwen3_tts.interface.ui.shared import enhance_description_with_ai
        config = {"prompt_enhancer": {"enabled": True, "api_key_env": "TEST_KEY", "provider": "openai"}}
        with patch(f"{_MOD}.load_config", return_value=config), \
             patch.dict(os.environ, {"TEST_KEY": "sk-test"}):
            with self.assertRaises(gr.Error):
                enhance_description_with_ai("warm voice")

    def test_anthropic_success(self):
        from qwen3_tts.interface.ui.shared import enhance_description_with_ai
        config = {"prompt_enhancer": {"enabled": True, "api_key_env": "TEST_KEY", "provider": "anthropic"}}
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="  A warm, smooth male voice  ")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        mock_anthropic_mod = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        with patch(f"{_MOD}.load_config", return_value=config), \
             patch.dict(os.environ, {"TEST_KEY": "sk-test"}), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic_mod}):
            result = enhance_description_with_ai("warm voice")
        self.assertEqual(result, "A warm, smooth male voice")

    def test_api_error_raises_gr_error(self):
        from qwen3_tts.interface.ui.shared import enhance_description_with_ai
        config = {"prompt_enhancer": {"enabled": True, "api_key_env": "TEST_KEY", "provider": "anthropic"}}
        mock_anthropic_mod = MagicMock()
        mock_anthropic_mod.Anthropic.side_effect = Exception("API down")
        with patch(f"{_MOD}.load_config", return_value=config), \
             patch.dict(os.environ, {"TEST_KEY": "sk-test"}), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic_mod}):
            with self.assertRaises(gr.Error):
                enhance_description_with_ai("warm voice")


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestIsEnhancerAvailable(unittest.TestCase):

    def test_not_enabled(self):
        from qwen3_tts.interface.ui.shared import is_enhancer_available
        with patch(f"{_MOD}.load_config", return_value={"prompt_enhancer": {"enabled": False}}):
            self.assertFalse(is_enhancer_available())

    def test_enabled_with_key(self):
        from qwen3_tts.interface.ui.shared import is_enhancer_available
        config = {"prompt_enhancer": {"enabled": True, "api_key_env": "TEST_KEY"}}
        with patch(f"{_MOD}.load_config", return_value=config), \
             patch.dict(os.environ, {"TEST_KEY": "sk-test"}):
            self.assertTrue(is_enhancer_available())

    def test_enabled_without_key(self):
        from qwen3_tts.interface.ui.shared import is_enhancer_available
        config = {"prompt_enhancer": {"enabled": True, "api_key_env": "MISSING_KEY_XYZ"}}
        with patch(f"{_MOD}.load_config", return_value=config):
            os.environ.pop("MISSING_KEY_XYZ", None)
            self.assertFalse(is_enhancer_available())


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestGetCurrentModelSettings(unittest.TestCase):

    def test_server_not_running_uses_config(self):
        from qwen3_tts.interface.ui.shared import get_current_model_settings
        with patch(f"{_MOD}.get_backend", return_value="mlx"), \
             patch(f"{_MOD}.get_model_size", return_value="1.7B"), \
             patch(f"{_MOD}.get_mlx_quantization", return_value="8bit"), \
             patch(f"{_MOD}.is_server_running", return_value=False), \
             patch(f"{_MOD}.load_config", return_value={}):
            result = get_current_model_settings()
        self.assertEqual(result, ("1.7B", "8bit", "mlx"))

    def test_server_running_with_settings(self):
        from qwen3_tts.interface.ui.shared import get_current_model_settings
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"settings": {"model_size": "0.6B", "mlx_quantization": "4bit", "backend": "mlx"}}
        with patch(f"{_MOD}.get_backend", return_value="mlx"), \
             patch(f"{_MOD}.get_model_size", return_value="1.7B"), \
             patch(f"{_MOD}.get_mlx_quantization", return_value="8bit"), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch(f"{_MOD}.auth_headers", return_value={}), \
             patch("requests.get", return_value=mock_resp):
            result = get_current_model_settings()
        self.assertEqual(result, ("0.6B", "4bit", "mlx"))

    def test_server_error_falls_back(self):
        from qwen3_tts.interface.ui.shared import get_current_model_settings
        with patch(f"{_MOD}.get_backend", return_value="torch"), \
             patch(f"{_MOD}.get_model_size", return_value="1.7B"), \
             patch(f"{_MOD}.get_mlx_quantization", return_value="8bit"), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch(f"{_MOD}.auth_headers", return_value={}), \
             patch("requests.get", side_effect=Exception("timeout")):
            result = get_current_model_settings()
        self.assertEqual(result, ("1.7B", "8bit", "torch"))


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestApplyModelSettings(unittest.TestCase):

    def test_server_not_running(self):
        from qwen3_tts.interface.ui.shared import apply_model_settings
        with patch(f"{_MOD}.is_server_running", return_value=False), \
             patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.format_status_display", return_value="<html>"):
            msg, html = apply_model_settings("1.7B", "8bit")
        self.assertIn("not running", msg)

    def test_success(self):
        from qwen3_tts.interface.ui.shared import apply_model_settings
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch(f"{_MOD}.is_server_running", return_value=True), \
             patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch(f"{_MOD}.get_backend", return_value="mlx"), \
             patch(f"{_MOD}.auth_headers", return_value={}), \
             patch(f"{_MOD}.format_status_display", return_value="<html>"), \
             patch("requests.post", return_value=mock_resp):
            msg, html = apply_model_settings("0.6B", "4bit")
        self.assertIn("applied", msg)

    def test_server_error(self):
        from qwen3_tts.interface.ui.shared import apply_model_settings
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "bad config"}
        with patch(f"{_MOD}.is_server_running", return_value=True), \
             patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch(f"{_MOD}.get_backend", return_value="mlx"), \
             patch(f"{_MOD}.auth_headers", return_value={}), \
             patch(f"{_MOD}.format_status_display", return_value="<html>"), \
             patch("requests.post", return_value=mock_resp):
            msg, html = apply_model_settings("0.6B", "4bit")
        self.assertIn("Failed", msg)

    def test_exception_handled(self):
        from qwen3_tts.interface.ui.shared import apply_model_settings
        with patch(f"{_MOD}.is_server_running", return_value=True), \
             patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.get_server_url", side_effect=Exception("conn")), \
             patch(f"{_MOD}.format_status_display", return_value="<html>"):
            msg, html = apply_model_settings("0.6B", "4bit")
        self.assertIn("Error", msg)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestGetServerStatus(unittest.TestCase):

    def test_disconnected(self):
        from qwen3_tts.interface.ui.shared import get_server_status
        mock_client = MagicMock()
        mock_client.is_server_running.return_value = False
        with patch("qwen3_tts.server.client.TTSClient", return_value=mock_client):
            status, mem, models, backend = get_server_status()
        self.assertEqual(status, "Disconnected")

    def test_connected_mlx(self):
        from qwen3_tts.interface.ui.shared import get_server_status
        mock_client = MagicMock()
        mock_client.is_server_running.return_value = True
        mock_client.get_stats.return_value = {
            "mlx_memory_active_mb": 2048.5,
            "clone_model_loaded": True,
            "design_model_loaded": False,
            "custom_model_loaded": True,
            "backend": "mlx",
            "model_size": "1.7B",
            "mlx_quantization": "8bit",
        }
        with patch("qwen3_tts.server.client.TTSClient", return_value=mock_client):
            status, mem, models, backend = get_server_status()
        self.assertEqual(status, "Connected")
        self.assertIn("2048.5", mem)
        self.assertIn("Clone", models)
        self.assertIn("Custom", models)
        self.assertNotIn("Design", models)
        self.assertIn("MLX", backend)

    def test_connected_torch(self):
        from qwen3_tts.interface.ui.shared import get_server_status
        mock_client = MagicMock()
        mock_client.is_server_running.return_value = True
        mock_client.get_stats.return_value = {
            "clone_model_loaded": False,
            "design_model_loaded": True,
            "custom_model_loaded": False,
            "backend": "torch",
            "dtype": "float16",
            "model_size": "0.6B",
        }
        with patch("qwen3_tts.server.client.TTSClient", return_value=mock_client):
            status, mem, models, backend = get_server_status()
        self.assertEqual(status, "Connected")
        self.assertIn("Design", models)
        self.assertIn("PyTorch", backend)

    def test_stats_error(self):
        from qwen3_tts.interface.ui.shared import get_server_status
        mock_client = MagicMock()
        mock_client.is_server_running.return_value = True
        mock_client.get_stats.side_effect = Exception("timeout")
        with patch("qwen3_tts.server.client.TTSClient", return_value=mock_client):
            status, mem, models, backend = get_server_status()
        self.assertIn("Error", status)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestFormatStatusDisplay(unittest.TestCase):

    def test_connected_html(self):
        from qwen3_tts.interface.ui.shared import format_status_display
        with patch(f"{_MOD}.get_server_status", return_value=("Connected", "1024MB", "Clone", "MLX")):
            html = format_status_display()
        self.assertIn("green", html)
        self.assertIn("1024MB", html)

    def test_disconnected_html(self):
        from qwen3_tts.interface.ui.shared import format_status_display
        with patch(f"{_MOD}.get_server_status", return_value=("Disconnected", "N/A", "N/A", "N/A")):
            html = format_status_display()
        self.assertIn("red", html)

    def test_error_html(self):
        from qwen3_tts.interface.ui.shared import format_status_display
        with patch(f"{_MOD}.get_server_status", return_value=("Error: timeout", "N/A", "N/A", "N/A")):
            html = format_status_display()
        self.assertIn("orange", html)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestGetVoicePrompts(unittest.TestCase):

    def test_mlx_filters_wav_with_txt(self):
        from qwen3_tts.interface.ui.shared import get_voice_prompts
        files = ["voice1.wav", "voice1.txt", "voice2.wav", "orphan.wav"]
        with patch(f"{_MOD}.get_backend", return_value="mlx"), \
             patch("os.listdir", return_value=files):
            result = get_voice_prompts()
        self.assertEqual(result, ["voice1.wav"])

    def test_torch_filters_pt(self):
        from qwen3_tts.interface.ui.shared import get_voice_prompts
        files = ["voice1.pt", "voice2.pt", "voice1.wav", "voice1.txt"]
        with patch(f"{_MOD}.get_backend", return_value="torch"), \
             patch("os.listdir", return_value=files):
            result = get_voice_prompts()
        self.assertEqual(result, ["voice1.pt", "voice2.pt"])

    def test_oserror_returns_empty(self):
        from qwen3_tts.interface.ui.shared import get_voice_prompts
        with patch(f"{_MOD}.get_backend", return_value="mlx"), \
             patch("os.listdir", side_effect=OSError("missing")):
            result = get_voice_prompts()
        self.assertEqual(result, [])


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestHistoryFunctions(unittest.TestCase):

    def test_add_to_history(self):
        from qwen3_tts.interface.ui.shared import add_to_history
        result = add_to_history([], "clone", "Hello world", "/tmp/out.wav", 3)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["mode"], "Clone")
        self.assertEqual(result[0]["chunks"], 3)

    def test_add_to_history_truncates(self):
        from qwen3_tts.interface.ui.shared import add_to_history
        long_text = "A" * 50
        result = add_to_history([], "design", long_text, "/tmp/out.wav", 1)
        self.assertTrue(result[0]["text"].endswith("..."))
        self.assertEqual(len(result[0]["text"]), 43)

    def test_add_to_history_caps_at_max(self):
        from qwen3_tts.interface.ui.shared import add_to_history, MAX_HISTORY_SIZE
        history = [{"timestamp": i, "mode": "Clone", "text": "x", "path": "/tmp", "chunks": 0} for i in range(MAX_HISTORY_SIZE)]
        result = add_to_history(history, "clone", "new", "/tmp/new.wav", 1)
        self.assertEqual(len(result), MAX_HISTORY_SIZE)
        self.assertEqual(result[0]["text"], "new")

    def test_get_history_data_empty(self):
        from qwen3_tts.interface.ui.shared import get_history_data
        self.assertEqual(get_history_data([]), [])

    def test_get_history_data_formats(self):
        from qwen3_tts.interface.ui.shared import get_history_data
        history = [{"timestamp": 1710000000, "mode": "Clone", "text": "Hello", "path": "/tmp", "chunks": 2}]
        rows = get_history_data(history)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "Clone")
        self.assertEqual(rows[0][2], "Hello")
        self.assertEqual(rows[0][3], 2)


if __name__ == "__main__":
    unittest.main()
