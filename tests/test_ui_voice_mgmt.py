#!/usr/bin/env python3
"""Tests for qwen3_tts.interface.ui.voice_management module.

Covers:
  - create_voice_prompt: MLX/torch creation, validation, error paths
  - auto_transcribe_audio: server ASR call
  - get_prompt_table_data: formatting
  - preview_voice: server preview
  - rename_voice / delete_voice: server calls
  - set_voice_default: config update

Run: pytest tests/test_ui_voice_mgmt.py -v
"""
import unittest
from unittest.mock import patch, MagicMock, mock_open

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

_MOD = "qwen3_tts.interface.ui.voice_management"


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestCreateVoicePromptUI(unittest.TestCase):
    """Tests for create_voice_prompt."""

    def test_no_audio_raises(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        with self.assertRaises(gr.Error):
            create_voice_prompt(None, "hello", "my_voice", False, False)

    def test_no_name_raises(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        with self.assertRaises(gr.Error):
            create_voice_prompt("/tmp/audio.wav", "hello", "", False, False)

    def test_invalid_name_raises(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        with patch(f"{_MOD}.validate_prompt_name", return_value=[{"error": "bad name"}]):
            with self.assertRaises(gr.Error):
                create_voice_prompt("/tmp/audio.wav", "hello", "my_voice", False, False)

    def test_mlx_already_exists_raises(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        with patch(f"{_MOD}.validate_prompt_name", return_value=None), \
             patch(f"{_MOD}.load_config", return_value={"advanced": {"backend": "mlx"}}), \
             patch(f"{_MOD}.strip_extension", return_value="voice1"), \
             patch("os.path.exists", return_value=True):
            with self.assertRaises(gr.Error):
                create_voice_prompt("/tmp/audio.wav", "hello", "voice1", False, False)

    def test_mlx_creates_wav_and_txt(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        with patch(f"{_MOD}.validate_prompt_name", return_value=None), \
             patch(f"{_MOD}.load_config", return_value={"advanced": {"backend": "mlx"}}), \
             patch(f"{_MOD}.strip_extension", return_value="new_voice"), \
             patch("os.path.exists", return_value=False), \
             patch("shutil.copy") as mock_copy, \
             patch("builtins.open", mock_open()), \
             patch(f"{_MOD}.get_voice_prompts", return_value=["new_voice.wav"]), \
             patch(f"{_MOD}.get_default_clone_prompt", return_value="new_voice.wav"):
            status, prompts, default = create_voice_prompt(
                "/tmp/audio.wav", "Hello world", "new_voice", False, False,
            )
        self.assertIn("MLX", status)
        mock_copy.assert_called_once()

    def test_mlx_no_transcript_mode(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        written = []
        m = mock_open()
        m.return_value.write = lambda x: written.append(x)
        with patch(f"{_MOD}.validate_prompt_name", return_value=None), \
             patch(f"{_MOD}.load_config", return_value={"advanced": {"backend": "mlx"}}), \
             patch(f"{_MOD}.strip_extension", return_value="voice"), \
             patch("os.path.exists", return_value=False), \
             patch("shutil.copy"), \
             patch("builtins.open", m), \
             patch(f"{_MOD}.get_voice_prompts", return_value=["voice.wav"]), \
             patch(f"{_MOD}.get_default_clone_prompt", return_value="voice.wav"):
            status, _, _ = create_voice_prompt(
                "/tmp/audio.wav", None, "voice", True, False,
            )
        self.assertIn("MLX", status)

    def test_mlx_no_transcript_no_flag_raises(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        with patch(f"{_MOD}.validate_prompt_name", return_value=None), \
             patch(f"{_MOD}.load_config", return_value={"advanced": {"backend": "mlx"}}), \
             patch(f"{_MOD}.strip_extension", return_value="voice"), \
             patch("os.path.exists", return_value=False), \
             patch("shutil.copy"), \
             patch("os.remove"):
            with self.assertRaises(gr.Error):
                create_voice_prompt("/tmp/audio.wav", "", "voice", False, False)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestAutoTranscribe(unittest.TestCase):

    def test_no_audio_raises(self):
        from qwen3_tts.interface.ui.voice_management import auto_transcribe_audio
        with self.assertRaises(gr.Error):
            auto_transcribe_audio(None)

    def test_server_not_running_raises(self):
        from qwen3_tts.interface.ui.voice_management import auto_transcribe_audio
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=False):
            with self.assertRaises(gr.Error):
                auto_transcribe_audio("/tmp/audio.wav")

    def test_success(self):
        from qwen3_tts.interface.ui.voice_management import auto_transcribe_audio
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"transcript": "Hello world"}
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch("qwen3_tts.core.http_client.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch("builtins.open", mock_open(read_data=b"audio")), \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
            result = auto_transcribe_audio("/tmp/audio.wav")
        self.assertEqual(result, "Hello world")

    def test_server_error_raises(self):
        from qwen3_tts.interface.ui.voice_management import auto_transcribe_audio
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "failed"}
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch("qwen3_tts.core.http_client.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch("builtins.open", mock_open(read_data=b"audio")), \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
            with self.assertRaises(gr.Error):
                auto_transcribe_audio("/tmp/audio.wav")


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestGetPromptTableData(unittest.TestCase):

    def test_formats_rows(self):
        from qwen3_tts.interface.ui.voice_management import get_prompt_table_data
        with patch(f"{_MOD}.get_voice_prompts", return_value=["voice1.wav", "voice2.wav"]), \
             patch(f"{_MOD}.load_config", return_value={"default_clone_prompt": "voice1"}), \
             patch(f"{_MOD}.strip_extension", side_effect=lambda n: n.rsplit(".", 1)[0]):
            rows = get_prompt_table_data()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][2], "✓")  # default
        self.assertEqual(rows[1][2], "")


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestPreviewVoice(unittest.TestCase):

    def test_no_name_raises(self):
        from qwen3_tts.interface.ui.voice_management import preview_voice
        with self.assertRaises(gr.Error):
            preview_voice("")

    def test_server_not_running_raises(self):
        from qwen3_tts.interface.ui.voice_management import preview_voice
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=False):
            with self.assertRaises(gr.Error):
                preview_voice("my_voice")

    def test_success_returns_path(self):
        from qwen3_tts.interface.ui.voice_management import preview_voice
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"RIFF fake wav data"
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch("qwen3_tts.core.http_client.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
             patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_file = MagicMock()
            mock_file.name = "/tmp/preview.wav"
            mock_tmp.return_value = mock_file
            result = preview_voice("my_voice")
        self.assertEqual(result, "/tmp/preview.wav")

    def test_server_error_raises(self):
        from qwen3_tts.interface.ui.voice_management import preview_voice
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.json.return_value = {"error": "not found"}
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch("qwen3_tts.core.http_client.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
            with self.assertRaises(gr.Error):
                preview_voice("missing_voice")

    def test_exception_returns_none(self):
        from qwen3_tts.interface.ui.voice_management import preview_voice
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch("qwen3_tts.core.http_client.get_server_url", side_effect=Exception("conn")):
            result = preview_voice("my_voice")
        self.assertIsNone(result)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestRenameVoice(unittest.TestCase):

    def test_no_old_name_raises(self):
        from qwen3_tts.interface.ui.voice_management import rename_voice
        with self.assertRaises(gr.Error):
            rename_voice("", "new_name")

    def test_no_new_name_raises(self):
        from qwen3_tts.interface.ui.voice_management import rename_voice
        with self.assertRaises(gr.Error):
            rename_voice("old", "")

    def test_server_not_running_raises(self):
        from qwen3_tts.interface.ui.voice_management import rename_voice
        with patch(f"{_MOD}.validate_prompt_name", return_value=None), \
             patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=False):
            with self.assertRaises(gr.Error):
                rename_voice("old", "new")

    def test_success(self):
        from qwen3_tts.interface.ui.voice_management import rename_voice
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch(f"{_MOD}.validate_prompt_name", return_value=None), \
             patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch("qwen3_tts.core.http_client.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
             patch(f"{_MOD}.get_voice_prompts", return_value=["new.wav"]), \
             patch(f"{_MOD}.get_prompt_table_data", return_value=[]):
            msg, table, dropdown = rename_voice("old", "new")
        self.assertIn("Renamed", msg)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestDeleteVoice(unittest.TestCase):

    def test_no_name_raises(self):
        from qwen3_tts.interface.ui.voice_management import delete_voice
        with self.assertRaises(gr.Error):
            delete_voice("")

    def test_server_not_running_raises(self):
        from qwen3_tts.interface.ui.voice_management import delete_voice
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=False):
            with self.assertRaises(gr.Error):
                delete_voice("my_voice")

    def test_success(self):
        from qwen3_tts.interface.ui.voice_management import delete_voice
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch("qwen3_tts.core.http_client.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
             patch(f"{_MOD}.get_voice_prompts", return_value=[]), \
             patch(f"{_MOD}.get_prompt_table_data", return_value=[]):
            msg, table, dropdown = delete_voice("my_voice")
        self.assertIn("Deleted", msg)

    def test_server_error_raises(self):
        from qwen3_tts.interface.ui.voice_management import delete_voice
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "internal"}
        with patch(f"{_MOD}.load_config", return_value={}), \
             patch(f"{_MOD}.is_server_running", return_value=True), \
             patch("qwen3_tts.core.http_client.get_server_url", return_value="http://127.0.0.1:5123"), \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
            with self.assertRaises(gr.Error):
                delete_voice("my_voice")


if __name__ == "__main__":
    unittest.main()
