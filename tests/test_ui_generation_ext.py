#!/usr/bin/env python3
"""Tests for qwen3_tts.interface.ui.generation module.

Covers:
  - cancel_streaming_generation
  - _prepare_streaming_config: validation, mode-specific payloads, Colab
  - _save_completed_audio: decode, save, error paths
  - _generate_colab_fallback: error/timeout/success/Colab TTSClient
  - _validate_inputs: basic validation

Run: pytest tests/test_ui_generation_ext.py -v
"""
import unittest
from unittest.mock import patch, MagicMock

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
             patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch(f"{_MOD}.auth_headers", return_value={}), \
             patch("requests.post", return_value=mock_resp), \
             patch(f"{_MOD}.format_status_display", return_value="<html>"):
            msg, html = cancel_streaming_generation()
        self.assertIn("cancelled", msg)

    def test_cancel_failure(self):
        from qwen3_tts.interface.ui.generation import cancel_streaming_generation
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "internal"}
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch(f"{_MOD}.auth_headers", return_value={}), \
             patch("requests.post", return_value=mock_resp), \
             patch(f"{_MOD}.format_status_display", return_value="<html>"):
            msg, html = cancel_streaming_generation()
        self.assertIn("failed", msg)

    def test_cancel_exception(self):
        from qwen3_tts.interface.ui.generation import cancel_streaming_generation
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch(f"{_MOD}.get_server_url", side_effect=Exception("conn")), \
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

    def test_clone_colab_mode(self):
        import qwen3_tts.core.config as _cfg
        orig = _cfg.IN_COLAB
        try:
            _cfg.IN_COLAB = True
            with patch(f"{_MOD}.load_config", return_value={"language": "English"}), \
                 patch(f"{_MOD}.is_server_running", return_value=True), \
                 patch(f"{_MOD}.get_prosody_presets", return_value={}):
                cfg, status = self._call()
            self.assertTrue(cfg.get("colab_fallback"))
            self.assertIn("Colab", status)
        finally:
            _cfg.IN_COLAB = orig

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

    def test_auth_token_not_found(self):
        import qwen3_tts.core.config as _cfg
        orig = _cfg.IN_COLAB
        try:
            _cfg.IN_COLAB = False
            with patch(f"{_MOD}.load_config", return_value={"language": "English"}), \
                 patch(f"{_MOD}.is_server_running", return_value=True), \
                 patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123"), \
                 patch(f"{_MOD}.get_prosody_presets", return_value={}), \
                 patch("builtins.open", side_effect=FileNotFoundError):
                cfg, status = self._call()
            self.assertIsNone(cfg)
            self.assertIn("Auth token not found", status)
        finally:
            _cfg.IN_COLAB = orig


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestSaveCompletedAudio(unittest.TestCase):

    def test_empty_base64(self):
        from qwen3_tts.interface.ui.generation import _save_completed_audio
        with patch(f"{_MOD}.format_status_display", return_value="<html>"):
            status, html, hist, df = _save_completed_audio("", "clone", "hi", [])
        self.assertEqual(status, "Cancelled")

    def test_timeout(self):
        from qwen3_tts.interface.ui.generation import _save_completed_audio
        with patch(f"{_MOD}.format_status_display", return_value="<html>"):
            status, html, hist, df = _save_completed_audio("TIMEOUT", "clone", "hi", [])
        self.assertIn("Timed out", status)

    def test_error_prefix(self):
        from qwen3_tts.interface.ui.generation import _save_completed_audio
        with patch(f"{_MOD}.format_status_display", return_value="<html>"):
            status, html, hist, df = _save_completed_audio("ERROR:oops", "clone", "hi", [])
        self.assertIn("oops", status)

    def test_success(self):
        import base64
        from qwen3_tts.interface.ui.generation import _save_completed_audio
        wav_b64 = base64.b64encode(b"RIFF fake wav").decode()
        with patch(f"{_MOD}.format_status_display", return_value="<html>"), \
             patch(f"{_MOD}.add_to_history", return_value=[{"path": "/tmp/out.wav"}]), \
             patch("qwen3_tts.interface.ui.shared.get_history_data", return_value=[]), \
             patch("builtins.open", MagicMock()), \
             patch("os.path.expanduser", return_value="/tmp/voice_ui_test.wav"):
            status, html, hist, df = _save_completed_audio(wav_b64, "clone", "hi", [])
        self.assertIn("Generated", status)

    def test_decode_error(self):
        from qwen3_tts.interface.ui.generation import _save_completed_audio
        with patch(f"{_MOD}.format_status_display", return_value="<html>"):
            status, html, hist, df = _save_completed_audio("!!!invalid!!!", "clone", "hi", [])
        self.assertIn("Error", status)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestGenerateColabFallback(unittest.TestCase):

    def test_error_prefix(self):
        from qwen3_tts.interface.ui.generation import _generate_colab_fallback
        with patch(f"{_MOD}.format_status_display", return_value="<html>"):
            result = _generate_colab_fallback("ERROR:fail", "clone", "hi", [], None)
        self.assertIsNone(result[0])
        self.assertIn("fail", result[1])

    def test_timeout(self):
        from qwen3_tts.interface.ui.generation import _generate_colab_fallback
        with patch(f"{_MOD}.format_status_display", return_value="<html>"):
            result = _generate_colab_fallback("TIMEOUT", "clone", "hi", [], None)
        self.assertIn("Timed out", result[1])

    def test_none_config_preserves_error(self):
        from qwen3_tts.interface.ui.generation import _generate_colab_fallback
        with patch(f"{_MOD}.format_status_display", return_value="<html>"):
            result = _generate_colab_fallback("", "clone", "hi", [], None)
        self.assertIsNone(result[0])

    def test_non_colab_cancelled(self):
        from qwen3_tts.interface.ui.generation import _generate_colab_fallback
        with patch(f"{_MOD}.format_status_display", return_value="<html>"):
            result = _generate_colab_fallback("", "clone", "hi", [], {"server_url": "x"})
        self.assertEqual(result[1], "Cancelled")

    def test_colab_fallback_success(self):
        from qwen3_tts.interface.ui.generation import _generate_colab_fallback
        mock_client = MagicMock()
        stream_config = {"colab_fallback": True, "payload": {
            "text": "hi", "mode": "clone", "temperature": 0.7,
        }}
        with patch(f"{_MOD}.format_status_display", return_value="<html>"), \
             patch("qwen3_tts.server.client.TTSClient", return_value=mock_client), \
             patch(f"{_MOD}.add_to_history", return_value=[{"path": "/tmp/out.wav"}]), \
             patch("qwen3_tts.interface.ui.shared.get_history_data", return_value=[]), \
             patch("os.path.expanduser", return_value="/tmp/voice_ui_test.wav"):
            result = _generate_colab_fallback("", "clone", "hi", [], stream_config)
        self.assertIsNotNone(result[0])
        self.assertIn("Generated", result[1])
        mock_client.generate.assert_called_once()

    def test_colab_fallback_error(self):
        from qwen3_tts.interface.ui.generation import _generate_colab_fallback
        stream_config = {"colab_fallback": True, "payload": {"text": "hi", "mode": "clone"}}
        with patch(f"{_MOD}.format_status_display", return_value="<html>"), \
             patch("qwen3_tts.server.client.TTSClient", side_effect=Exception("conn")), \
             patch("qwen3_tts.interface.ui.shared.get_history_data", return_value=[]):
            result = _generate_colab_fallback("", "clone", "hi", [], stream_config)
        self.assertIsNone(result[0])
        self.assertIn("Error", result[1])


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestValidateInputs(unittest.TestCase):

    def test_empty_text(self):
        from qwen3_tts.interface.ui.generation import _validate_inputs
        valid, msg = _validate_inputs("clone", "")
        self.assertFalse(valid)

    def test_design_no_description(self):
        from qwen3_tts.interface.ui.generation import _validate_inputs
        valid, msg = _validate_inputs("design", "hello", description="")
        self.assertFalse(valid)

    def test_valid(self):
        from qwen3_tts.interface.ui.generation import _validate_inputs
        valid, msg = _validate_inputs("clone", "hello")
        self.assertTrue(valid)
        self.assertIsNone(msg)


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

    def test_auth_token_success_non_colab(self):
        """Lines 143, 146: non-Colab path where token file exists."""
        import qwen3_tts.core.config as _cfg
        from qwen3_tts.interface.ui.generation import _prepare_streaming_config
        orig = _cfg.IN_COLAB
        try:
            _cfg.IN_COLAB = False
            mock_open = MagicMock()
            mock_open.return_value.__enter__ = MagicMock(
                return_value=MagicMock(read=MagicMock(return_value="test-token\n"))
            )
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            with patch(f"{_MOD}.load_config", return_value={"language": "English"}), \
                 patch(f"{_MOD}.is_server_running", return_value=True), \
                 patch(f"{_MOD}.get_server_url", return_value="http://127.0.0.1:5123"), \
                 patch(f"{_MOD}.get_prosody_presets", return_value={}), \
                 patch("builtins.open", mock_open):
                cfg, status = _prepare_streaming_config(
                    mode="clone", text="Hello", preset=None,
                    temperature=0.7, top_k=50, top_p=0.95,
                    repetition_penalty=1.05, seed="",
                    prompt_file="voice1.wav",
                )
            self.assertIsNotNone(cfg)
            self.assertEqual(cfg["auth_token"], "test-token")
            self.assertIn("Connecting", status)
        finally:
            _cfg.IN_COLAB = orig

    def test_save_completed_audio_none_history(self):
        """Line 158: history_list is None."""
        from qwen3_tts.interface.ui.generation import _save_completed_audio
        with patch(f"{_MOD}.format_status_display", return_value="<html>"):
            status, html, hist, df = _save_completed_audio("", "clone", "hi", None)
        self.assertEqual(status, "Cancelled")
        self.assertIsInstance(hist, list)

    def test_colab_fallback_none_history(self):
        """Line 207: history_list is None in _generate_colab_fallback."""
        from qwen3_tts.interface.ui.generation import _generate_colab_fallback
        with patch(f"{_MOD}.format_status_display", return_value="<html>"):
            result = _generate_colab_fallback("ERROR:fail", "clone", "hi", None, None)
        self.assertIsNone(result[0])
        self.assertIn("fail", result[1])


if __name__ == "__main__":
    unittest.main()
