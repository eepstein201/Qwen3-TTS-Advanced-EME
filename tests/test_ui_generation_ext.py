#!/usr/bin/env python3
"""Tests for qwen3_tts.interface.ui.generation module.

Covers:
  - cancel_streaming_generation
  - _prepare_streaming_config: validation, mode-specific payloads, Colab
  - _save_completed_audio: decode, save, error paths
  - _generate_server_side: server-side generation via TTSClient
  - _validate_inputs: basic validation

Run: pytest tests/test_ui_generation_ext.py -v
"""
import unittest
from unittest.mock import MagicMock, patch

try:
    import gradio as gr  # noqa: F401
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

# generation.py imports at MODULE scope from config, so patch at the generation module
_MOD = "qwen3_tts.interface.ui.generation"


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestCancelStreamingGeneration(unittest.TestCase):

    def test_server_not_running(self):
        from qwen3_tts.interface.ui.generation import cancel_streaming_generation
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=False), \
             patch(f"{_MOD}.format_status_display", return_value="<html>"):
            msg, html = cancel_streaming_generation()
        self.assertIn("not running", msg)

    def test_cancel_success(self):
        from qwen3_tts.interface.ui.generation import cancel_streaming_generation
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch("qwen3_tts.core.http_client.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
             patch(f"{_MOD}.format_status_display", return_value="<html>"):
            msg, html = cancel_streaming_generation()
        self.assertIn("stopped", msg)

    def test_cancel_failure(self):
        from qwen3_tts.interface.ui.generation import cancel_streaming_generation
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "internal"}
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch("qwen3_tts.core.http_client.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
             patch(f"{_MOD}.format_status_display", return_value="<html>"):
            msg, html = cancel_streaming_generation()
        self.assertIn("failed", msg)

    def test_cancel_exception(self):
        from qwen3_tts.interface.ui.generation import cancel_streaming_generation
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch("qwen3_tts.core.http_client.get_server_url", side_effect=Exception("conn")), \
             patch(f"{_MOD}.format_status_display", return_value="<html>"):
            msg, html = cancel_streaming_generation()
        self.assertIn("Error", msg)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestPrepareStreamingConfig(unittest.TestCase):

    def _call(self, **kwargs):
        from qwen3_tts.interface.ui.generation import _prepare_streaming_config
        defaults = {
            "mode": "clone", "text": "Hello", "preset": None,
            "temperature": 0.7, "top_k": 50, "top_p": 0.95,
            "repetition_penalty": 1.05, "seed": "",
            "prompt_file": "voice1.wav",
        }
        defaults.update(kwargs)
        return _prepare_streaming_config(**defaults)

    def test_empty_text(self):
        cfg, status = self._call(text="")
        self.assertIsNone(cfg)
        self.assertIn("enter text", status)

    def test_design_no_description(self):
        cfg, status = self._call(mode="design", text="hello", description="")
        self.assertIsNone(cfg)
        self.assertIn("voice description", status)

    def test_server_not_running(self):
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=False):
            cfg, status = self._call()
        self.assertIsNone(cfg)
        self.assertIn("not running", status)

    def test_clone_no_prompt(self):
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True):
            cfg, status = self._call(prompt_file=None)
        self.assertIsNone(cfg)
        self.assertIn("voice prompt", status)

    def test_custom_no_speaker(self):
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True):
            cfg, status = self._call(mode="custom", speaker=None)
        self.assertIsNone(cfg)
        self.assertIn("speaker", status)

    def test_clone_returns_server_side_config(self):
        with patch(f"{_MOD}.load_config", return_value={"language": "English"}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch(f"{_MOD}.get_prosody_presets", return_value={}):
            cfg, status = self._call()
        self.assertTrue(cfg.get("server_side"))
        self.assertEqual(status, "Generating...")

    def test_design_with_prosody(self):
        import qwen3_tts.core.config as _cfg
        orig = _cfg.IN_COLAB
        try:
            _cfg.IN_COLAB = True
            presets = {"happy": "[happy]"}
            with patch(f"{_MOD}.load_config", return_value={"language": "English"}), \
                 patch(f"{_MOD}.is_server_running", return_value=True), \
                 patch(f"{_MOD}.get_prosody_presets", return_value=presets):
                cfg, status = self._call(
                    mode="design", description="warm voice",
                    prosody_preset="happy", prompt_file=None,
                )
            self.assertEqual(cfg["payload"]["mode"], "design")
        finally:
            _cfg.IN_COLAB = orig

    def test_custom_speaker_extraction(self):
        import qwen3_tts.core.config as _cfg
        orig = _cfg.IN_COLAB
        try:
            _cfg.IN_COLAB = True
            with patch(f"{_MOD}.load_config", return_value={"language": "English"}), \
                 patch(f"{_MOD}.is_server_running", return_value=True), \
                 patch(f"{_MOD}.get_prosody_presets", return_value={}):
                cfg, status = self._call(
                    mode="custom", speaker="ryan (English) - A warm voice",
                    prompt_file=None,
                )
            self.assertEqual(cfg["payload"]["speaker"], "ryan")
        finally:
            _cfg.IN_COLAB = orig

    def test_preset_applied(self):
        import qwen3_tts.core.config as _cfg
        orig = _cfg.IN_COLAB
        try:
            _cfg.IN_COLAB = True
            config = {"language": "English", "presets": {"warm": {"temperature": 0.9}}}
            with patch(f"{_MOD}.load_config", return_value=config), \
                 patch(f"{_MOD}.is_server_running", return_value=True), \
                 patch(f"{_MOD}.get_prosody_presets", return_value={}):
                cfg, status = self._call(preset="warm")
            self.assertEqual(cfg["payload"]["temperature"], 0.9)
        finally:
            _cfg.IN_COLAB = orig

    def test_seed_parsing(self):
        import qwen3_tts.core.config as _cfg
        orig = _cfg.IN_COLAB
        try:
            _cfg.IN_COLAB = True
            with patch(f"{_MOD}.load_config", return_value={"language": "English"}), \
                 patch(f"{_MOD}.is_server_running", return_value=True), \
                 patch(f"{_MOD}.get_prosody_presets", return_value={}):
                cfg, _ = self._call(seed="42")
            self.assertEqual(cfg["payload"]["seed"], 42)
        finally:
            _cfg.IN_COLAB = orig

    def test_no_transcript_clone(self):
        import qwen3_tts.core.config as _cfg
        orig = _cfg.IN_COLAB
        try:
            _cfg.IN_COLAB = True
            with patch(f"{_MOD}.load_config", return_value={"language": "English"}), \
                 patch(f"{_MOD}.is_server_running", return_value=True), \
                 patch(f"{_MOD}.get_prosody_presets", return_value={}):
                cfg, _ = self._call(no_transcript=True)
            self.assertTrue(cfg["payload"].get("x_vector_only_mode"))
        finally:
            _cfg.IN_COLAB = orig

    def test_non_colab_returns_server_side_config(self):
        """Non-Colab path now returns server_side config (no token reading)."""
        import qwen3_tts.core.config as _cfg
        orig = _cfg.IN_COLAB
        try:
            _cfg.IN_COLAB = False
            with patch(f"{_MOD}.load_config", return_value={"language": "English"}), \
                 patch(f"{_MOD}.is_server_running", return_value=True), \
                 patch(f"{_MOD}.get_prosody_presets", return_value={}):
                cfg, status = self._call()
            self.assertTrue(cfg.get("server_side"))
            self.assertEqual(cfg["payload"]["mode"], "clone")
            self.assertEqual(status, "Generating...")
        finally:
            _cfg.IN_COLAB = orig


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestGenerateServerSide(unittest.TestCase):

    def test_none_config_preserves_error(self):
        from qwen3_tts.interface.ui.generation import _generate_server_side
        with patch(f"{_MOD}.format_status_display", return_value="<html>"):
            result = _generate_server_side("clone", "hi", [], None)
        self.assertIsNone(result[0])

    def test_non_server_side_cancelled(self):
        from qwen3_tts.interface.ui.generation import _generate_server_side
        with patch(f"{_MOD}.format_status_display", return_value="<html>"):
            result = _generate_server_side("clone", "hi", [], {"payload": {}})
        self.assertEqual(result[1], "Generation stopped")

    def test_server_side_success(self):
        from qwen3_tts.interface.ui.generation import _generate_server_side
        mock_client = MagicMock()
        mock_client.last_chunk_count = 3
        stream_config = {"server_side": True, "payload": {
            "text": "hi", "mode": "clone", "temperature": 0.7,
        }}
        with patch(f"{_MOD}.format_status_display", return_value="<html>"), \
             patch("qwen3_tts.server.client.TTSClient", return_value=mock_client), \
             patch(f"{_MOD}.add_to_history", return_value=[{"path": "/tmp/out.wav"}]), \
             patch("qwen3_tts.interface.ui.shared.get_history_data", return_value=[]), \
             patch(f"{_MOD}.shutil.copy2"), \
             patch(f"{_MOD}.save_generation_metadata"), \
             patch(f"{_MOD}.load_config", return_value={"output_directory": "/tmp"}), \
             patch("os.path.expanduser", return_value="/tmp"), \
             patch("os.makedirs"):
            result = _generate_server_side("clone", "hi", [], stream_config)
        self.assertIsNotNone(result[0])
        self.assertIn("Generated", result[1])
        mock_client.generate.assert_called_once()

    def test_server_side_error(self):
        from qwen3_tts.interface.ui.generation import _generate_server_side
        stream_config = {"server_side": True, "payload": {"text": "hi", "mode": "clone"}}
        with patch(f"{_MOD}.format_status_display", return_value="<html>"), \
             patch("qwen3_tts.server.client.TTSClient", side_effect=Exception("conn")), \
             patch("qwen3_tts.interface.ui.shared.get_history_data", return_value=[]):
            result = _generate_server_side("clone", "hi", [], stream_config)
        self.assertIsNone(result[0])
        self.assertIn("Error", result[1])

    def test_server_side_model_not_loaded_is_actionable(self):
        """When the server rejects with model_not_loaded, the UI must surface a
        clear, actionable message naming the Manage Models tab — not the bare,
        easy-to-miss "model is not loaded" string that previously read as a
        silent failure.

        Previously the client raised a generic GenerationError carrying only
        "The 'design' model is not loaded." and the UI echoed it verbatim, with
        no guidance on how to recover. This test pins the actionable message.
        """
        from qwen3_tts.core.config import ModelNotLoadedError
        from qwen3_tts.interface.ui.generation import _generate_server_side
        stream_config = {
            "server_side": True,
            "payload": {"text": "hi", "mode": "design"},
        }
        with patch(f"{_MOD}.format_status_display", return_value="<html>"), \
             patch(
                 "qwen3_tts.server.client.TTSClient",
                 side_effect=ModelNotLoadedError("design"),
             ), \
             patch("qwen3_tts.interface.ui.shared.get_history_data", return_value=[]):
            result = _generate_server_side("design", "hi", [], stream_config)
        # audio_path is None (no audio generated)
        self.assertIsNone(result[0])
        status = result[1]
        # Actionable: names the model AND points the user to Manage Models
        self.assertIn("design", status)
        self.assertIn("Manage Models", status)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestEdgeCases(unittest.TestCase):
    """Cover remaining uncovered lines in generation.py."""

    def test_invalid_seed_ignored(self):
        """Line 101-102: ValueError branch when seed is non-numeric."""
        import qwen3_tts.core.config as _cfg
        from qwen3_tts.interface.ui.generation import _prepare_streaming_config
        orig = _cfg.IN_COLAB
        try:
            _cfg.IN_COLAB = True
            with patch(f"{_MOD}.load_config", return_value={"language": "English"}), \
                 patch(f"{_MOD}.is_server_running", return_value=True), \
                 patch(f"{_MOD}.get_prosody_presets", return_value={}):
                cfg, _ = _prepare_streaming_config(
                    mode="clone", text="Hello", preset=None,
                    temperature=0.7, top_k=50, top_p=0.95,
                    repetition_penalty=1.05, seed="not_a_number",
                    prompt_file="voice1.wav",
                )
            self.assertNotIn("seed", cfg["payload"])
        finally:
            _cfg.IN_COLAB = orig

    def test_server_side_config_non_colab(self):
        """Non-Colab path returns server_side config (no token in config)."""
        import qwen3_tts.core.config as _cfg
        from qwen3_tts.interface.ui.generation import _prepare_streaming_config
        orig = _cfg.IN_COLAB
        try:
            _cfg.IN_COLAB = False
            with patch(f"{_MOD}.load_config", return_value={"language": "English"}), \
                 patch(f"{_MOD}.is_server_running", return_value=True), \
                 patch(f"{_MOD}.get_prosody_presets", return_value={}):
                cfg, status = _prepare_streaming_config(
                    mode="clone", text="Hello", preset=None,
                    temperature=0.7, top_k=50, top_p=0.95,
                    repetition_penalty=1.05, seed="",
                    prompt_file="voice1.wav",
                )
            self.assertIsNotNone(cfg)
            self.assertTrue(cfg.get("server_side"))
            self.assertNotIn("auth_token", cfg)
            self.assertEqual(status, "Generating...")
        finally:
            _cfg.IN_COLAB = orig

    def test_server_side_none_history(self):
        """history_list is None in _generate_server_side."""
        from qwen3_tts.interface.ui.generation import _generate_server_side
        with patch(f"{_MOD}.format_status_display", return_value="<html>"):
            result = _generate_server_side("clone", "hi", None, None)
        self.assertIsNone(result[0])


class TestGenerationSavesIntoAutomatedOutput(unittest.TestCase):
    """Web-UI generations must land in Automated Output, not flat in ~/Downloads."""

    def test_history_scan_reads_the_automated_output_subdir(self):
        import os
        import tempfile
        from unittest.mock import patch

        from qwen3_tts.interface.ui import shared

        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmp:
            automated = os.path.join(tmp, "Automated Output")
            os.makedirs(automated)
            wav = os.path.join(automated, "voice_ui_abc123.wav")
            with open(wav, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 40)
            with open(os.path.join(automated, "voice_ui_abc123.json"), "w") as f:
                f.write('{"timestamp": 1.0, "mode": "clone", "text": "hi", "seed": 42}')

            with patch.object(
                shared, "resolve_automated_output_dir", return_value=automated
            ):
                history = shared.load_history_from_disk_for_config({})

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["seed"], 42)
        self.assertTrue(history[0]["path"].endswith("voice_ui_abc123.wav"))


if __name__ == "__main__":
    unittest.main()
