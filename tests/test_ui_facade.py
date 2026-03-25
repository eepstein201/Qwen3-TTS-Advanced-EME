#!/usr/bin/env python3
"""Tests for the Gradio UI facade module (_facade.py).

Covers:
  - stop_server: TTSClient shutdown with polling
  - _find_available_port: port scanning logic
  - on_history_select: history row click handling
  - build_ui: returns gr.Blocks with expected structure
  - main: CLI entry point with port finding and launch

Run: pytest tests/test_ui_facade.py -v
"""
import unittest
from unittest.mock import patch, MagicMock

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False


class TestStopServer(unittest.TestCase):
    """Tests for stop_server function."""

    @patch("qwen3_tts.interface.ui._facade.format_status_display", return_value="<html>stopped</html>")
    @patch("qwen3_tts.interface.ui._facade.TTSClient")
    @patch("time.sleep")
    def test_shutdown_success_immediate(self, _sleep, mock_client_cls, mock_format):
        from qwen3_tts.interface.ui._facade import stop_server
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.is_server_running.return_value = False
        result = stop_server()
        mock_client.shutdown.assert_called_once()
        self.assertEqual(result, "<html>stopped</html>")

    @patch("qwen3_tts.interface.ui._facade.format_status_display", return_value="<html>status</html>")
    @patch("qwen3_tts.interface.ui._facade.TTSClient")
    @patch("time.sleep")
    def test_shutdown_polls_until_stopped(self, _sleep, mock_client_cls, mock_format):
        from qwen3_tts.interface.ui._facade import stop_server
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        # Server running for 3 polls, then stops
        mock_client.is_server_running.side_effect = [True, True, True, False]
        stop_server()
        self.assertEqual(mock_client.is_server_running.call_count, 4)

    @patch("qwen3_tts.interface.ui._facade.format_status_display", return_value="<html>timeout</html>")
    @patch("qwen3_tts.interface.ui._facade.TTSClient")
    @patch("time.sleep")
    def test_shutdown_timeout(self, _sleep, mock_client_cls, mock_format):
        from qwen3_tts.interface.ui._facade import stop_server
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.is_server_running.return_value = True  # Never stops
        stop_server()
        # Should have polled 10 times
        self.assertEqual(mock_client.is_server_running.call_count, 10)

    @patch("qwen3_tts.interface.ui._facade.format_status_display", return_value="<html>err</html>")
    @patch("qwen3_tts.interface.ui._facade.TTSClient")
    @patch("time.sleep")
    def test_shutdown_exception_handled(self, _sleep, mock_client_cls, mock_format):
        from qwen3_tts.interface.ui._facade import stop_server
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.shutdown.side_effect = ConnectionError("refused")
        mock_client.is_server_running.return_value = False
        # Should not raise
        result = stop_server()
        self.assertIsNotNone(result)


class TestFindAvailablePort(unittest.TestCase):
    """Tests for _find_available_port."""

    @patch("qwen3_tts.interface.ui._facade.IN_COLAB", False)
    def test_preferred_port_available(self):
        from qwen3_tts.interface.ui._facade import _find_available_port
        mock_socket = MagicMock()
        with patch("socket.socket", return_value=mock_socket):
            mock_socket.__enter__ = lambda s: s
            mock_socket.__exit__ = MagicMock(return_value=False)
            port = _find_available_port(7860)
        self.assertEqual(port, 7860)

    @patch("qwen3_tts.interface.ui._facade.IN_COLAB", False)
    def test_falls_back_to_next_port(self):
        from qwen3_tts.interface.ui._facade import _find_available_port
        call_count = [0]

        class FakeSocket:
            def __init__(self, *a, **kw):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def bind(self, addr):
                call_count[0] += 1
                if call_count[0] <= 2:
                    raise OSError("in use")

        with patch("socket.socket", FakeSocket):
            port = _find_available_port(7860)
        self.assertEqual(port, 7862)

    @patch("qwen3_tts.interface.ui._facade.IN_COLAB", False)
    def test_all_ports_taken_returns_none(self):
        from qwen3_tts.interface.ui._facade import _find_available_port

        class FailSocket:
            def __init__(self, *a, **kw):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def bind(self, addr):
                raise OSError("in use")

        with patch("socket.socket", FailSocket):
            port = _find_available_port(7860, max_tries=3)
        self.assertIsNone(port)

    @patch("qwen3_tts.interface.ui._facade.IN_COLAB", True)
    def test_colab_binds_to_all_interfaces(self):
        from qwen3_tts.interface.ui._facade import _find_available_port
        bound_addr = []

        class TrackSocket:
            def __init__(self, *a, **kw):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def bind(self, addr):
                bound_addr.append(addr)

        with patch("socket.socket", TrackSocket):
            _find_available_port(7860)
        self.assertEqual(bound_addr[0][0], "0.0.0.0")


class TestOnHistorySelect(unittest.TestCase):
    """Tests for on_history_select."""

    def test_valid_selection_returns_temp_path(self):
        import tempfile, os
        from qwen3_tts.interface.ui._facade import on_history_select
        # Create a real temp file so containment and existence checks pass
        src = os.path.join(tempfile.gettempdir(), "test_on_hist.wav")
        with open(src, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 40)
        try:
            evt = MagicMock()
            evt.index = [0]
            history = [{"path": src}]
            with patch("qwen3_tts.interface.ui._facade.load_config", return_value={}):
                result = on_history_select(evt, history)
            self.assertIsNotNone(result)
            self.assertTrue(result.startswith(tempfile.gettempdir()))
        finally:
            if os.path.exists(src):
                os.remove(src)

    def test_invalid_index_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [5]
        history = [{"path": "/tmp/test.wav"}]
        result = on_history_select(evt, history)
        self.assertIsNone(result)

    def test_missing_path_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        history = [{"mode": "clone"}]
        result = on_history_select(evt, history)
        self.assertIsNone(result)

    def test_nonexistent_file_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        # Path is in a safe root (tempdir) but file doesn't exist
        import tempfile
        history = [{"path": tempfile.gettempdir() + "/nonexistent_file_xyz.wav"}]
        with patch("qwen3_tts.interface.ui._facade.load_config", return_value={}):
            result = on_history_select(evt, history)
        self.assertIsNone(result)

    def test_empty_history_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        result = on_history_select(evt, [])
        self.assertIsNone(result)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestBuildUI(unittest.TestCase):
    """Tests for build_ui — verifies the Gradio interface builds."""

    @patch("qwen3_tts.interface.ui._facade.TTSClient")
    @patch("qwen3_tts.interface.ui._facade.format_status_display", return_value="<html></html>")
    @patch("qwen3_tts.interface.ui._facade.get_model_status_html", return_value="<html></html>")
    @patch("qwen3_tts.interface.ui._facade.get_model_table_data", return_value=[])
    @patch("qwen3_tts.interface.ui._facade.get_prompt_table_data", return_value=[])
    @patch("qwen3_tts.interface.ui._facade.get_voice_prompts", return_value=["default.wav"])
    @patch("qwen3_tts.interface.ui._facade.get_presets", return_value=["(none)"])
    @patch("qwen3_tts.interface.ui._facade.get_prosody_choices", return_value=["(none)"])
    @patch("qwen3_tts.interface.ui._facade.is_enhancer_available", return_value=False)
    @patch("qwen3_tts.interface.ui._facade.get_current_model_settings", return_value=("1.7B", "8bit", "mlx"))
    @patch("qwen3_tts.interface.ui._facade.get_audio_loader_setting", return_value="torchaudio")
    @patch("qwen3_tts.interface.ui._facade.get_default_clone_prompt", return_value="default.wav")
    def test_build_ui_returns_blocks(self, *mocks):
        from qwen3_tts.interface.ui._facade import build_ui
        # Suppress ASR preload
        with patch("qwen3_tts.core.engine.is_asr_available", side_effect=ImportError):
            demo = build_ui()
        self.assertIsInstance(demo, gr.Blocks)

    @patch("qwen3_tts.interface.ui._facade.TTSClient")
    @patch("qwen3_tts.interface.ui._facade.format_status_display", return_value="<html></html>")
    @patch("qwen3_tts.interface.ui._facade.get_model_status_html", return_value="<html></html>")
    @patch("qwen3_tts.interface.ui._facade.get_model_table_data", return_value=[])
    @patch("qwen3_tts.interface.ui._facade.get_prompt_table_data", return_value=[])
    @patch("qwen3_tts.interface.ui._facade.get_voice_prompts", return_value=["default.wav"])
    @patch("qwen3_tts.interface.ui._facade.get_presets", return_value=["(none)"])
    @patch("qwen3_tts.interface.ui._facade.get_prosody_choices", return_value=["(none)"])
    @patch("qwen3_tts.interface.ui._facade.is_enhancer_available", return_value=False)
    @patch("qwen3_tts.interface.ui._facade.get_current_model_settings", return_value=("1.7B", "8bit", "mlx"))
    @patch("qwen3_tts.interface.ui._facade.get_audio_loader_setting", return_value="torchaudio")
    @patch("qwen3_tts.interface.ui._facade.get_default_clone_prompt", return_value="default.wav")
    def test_build_ui_has_title(self, *mocks):
        from qwen3_tts.interface.ui._facade import build_ui
        with patch("qwen3_tts.core.engine.is_asr_available", side_effect=ImportError):
            demo = build_ui()
        self.assertEqual(demo.title, "Qwen3-TTS Web Interface")


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestMain(unittest.TestCase):
    """Tests for main() CLI entry point."""

    @patch("qwen3_tts.interface.ui._facade.load_config", return_value={"ui": {"port": 7860}})
    @patch("qwen3_tts.interface.ui._facade._find_available_port", return_value=None)
    @patch("builtins.print")
    def test_no_available_port_exits(self, _print, _port, _config):
        from qwen3_tts.interface.ui._facade import main
        with patch("sys.argv", ["ui", "--port", "7860"]):
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertEqual(ctx.exception.code, 1)

    @patch("qwen3_tts.interface.ui._facade.load_config", return_value={"ui": {"port": 7860}})
    @patch("qwen3_tts.interface.ui._facade._find_available_port", return_value=7861)
    @patch("qwen3_tts.interface.ui._facade.build_ui")
    @patch("qwen3_tts.interface.ui._facade.TTSClient")
    @patch("qwen3_tts.interface.ui._facade.IN_COLAB", False)
    @patch("builtins.print")
    def test_port_fallback_message(self, mock_print, mock_client_cls, mock_build, _port, _config):
        from qwen3_tts.interface.ui._facade import main
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.is_server_running.return_value = True
        mock_demo = MagicMock()
        mock_build.return_value = mock_demo
        with patch("sys.argv", ["ui", "--port", "7860"]):
            main()
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("7860", output)
        self.assertIn("7861", output)
        mock_demo.launch.assert_called_once()

    @patch("qwen3_tts.interface.ui._facade.load_config", return_value={})
    @patch("qwen3_tts.interface.ui._facade._find_available_port", return_value=7860)
    @patch("qwen3_tts.interface.ui._facade.build_ui")
    @patch("qwen3_tts.interface.ui._facade.TTSClient")
    @patch("qwen3_tts.interface.ui._facade.IN_COLAB", False)
    @patch("builtins.print")
    def test_server_not_running_warning(self, mock_print, mock_client_cls, mock_build, _port, _config):
        from qwen3_tts.interface.ui._facade import main
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.is_server_running.return_value = False
        mock_demo = MagicMock()
        mock_build.return_value = mock_demo
        with patch("sys.argv", ["ui"]):
            main()
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("WARNING", output)

    @patch("qwen3_tts.interface.ui._facade.load_config", return_value={})
    @patch("qwen3_tts.interface.ui._facade._find_available_port", return_value=7860)
    @patch("qwen3_tts.interface.ui._facade.build_ui")
    @patch("qwen3_tts.interface.ui._facade.TTSClient")
    @patch("qwen3_tts.interface.ui._facade.IN_COLAB", True)
    @patch("qwen3_tts.core.config.IN_COLAB", True)
    @patch("builtins.print")
    def test_colab_mode_settings(self, _print, mock_client_cls, mock_build, _port, _config):
        from qwen3_tts.interface.ui._facade import main
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.is_server_running.return_value = True
        mock_demo = MagicMock()
        mock_build.return_value = mock_demo
        with patch("sys.argv", ["ui"]):
            main()
        launch_kwargs = mock_demo.launch.call_args[1]
        self.assertEqual(launch_kwargs["server_name"], "0.0.0.0")
        self.assertTrue(launch_kwargs["share"])
        self.assertFalse(launch_kwargs["inbrowser"])


class TestGetGradioLaunchKwargs(unittest.TestCase):
    """Tests for get_gradio_launch_kwargs helper."""

    @patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_includes_all_required_keys(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs
        kwargs = get_gradio_launch_kwargs({})
        self.assertIn("theme", kwargs)
        self.assertIn("css", kwargs)
        self.assertIn("server_name", kwargs)
        self.assertIn("allowed_paths", kwargs)

    @patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_includes_tempdir_in_allowed_paths(self):
        import tempfile
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs
        kwargs = get_gradio_launch_kwargs({"output_directory": "~/Downloads"})
        self.assertIn(tempfile.gettempdir(), kwargs["allowed_paths"])

    @patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_includes_downloads_in_allowed_paths(self):
        import os
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs
        kwargs = get_gradio_launch_kwargs({})
        downloads = os.path.realpath(os.path.expanduser("~/Downloads"))
        self.assertIn(downloads, kwargs["allowed_paths"])

    @patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_localhost_when_not_colab(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs
        kwargs = get_gradio_launch_kwargs({})
        self.assertEqual(kwargs["server_name"], "127.0.0.1")

    @patch("qwen3_tts.core.config.IN_COLAB", True)
    def test_all_interfaces_when_colab(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs
        kwargs = get_gradio_launch_kwargs({})
        self.assertEqual(kwargs["server_name"], "0.0.0.0")

    @patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_css_contains_gr_hidden(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs
        kwargs = get_gradio_launch_kwargs({})
        self.assertIn(".gr-hidden", kwargs["css"])


class TestResolveOutputDir(unittest.TestCase):
    """Tests for _resolve_output_dir helper."""

    def test_default_downloads(self):
        import os
        from qwen3_tts.interface.ui.shared import _resolve_output_dir
        result = _resolve_output_dir({})
        expected = os.path.realpath(os.path.expanduser("~/Downloads"))
        self.assertEqual(result, expected)

    def test_custom_output_dir(self):
        import os
        from qwen3_tts.interface.ui.shared import _resolve_output_dir
        result = _resolve_output_dir({"output_directory": "~/Music"})
        expected = os.path.realpath(os.path.expanduser("~/Music"))
        self.assertEqual(result, expected)

    def test_traversal_falls_back_to_downloads(self):
        import os
        from qwen3_tts.interface.ui.shared import _resolve_output_dir
        result = _resolve_output_dir({"output_directory": "~/../../etc"})
        downloads = os.path.realpath(os.path.expanduser("~/Downloads"))
        self.assertEqual(result, downloads)

    def test_returns_absolute_path(self):
        import os
        from qwen3_tts.interface.ui.shared import _resolve_output_dir
        result = _resolve_output_dir({})
        self.assertTrue(os.path.isabs(result))


class TestOnHistorySelectHardened(unittest.TestCase):
    """Tests for hardened on_history_select (containment + temp copy)."""

    def test_unsafe_path_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        history = [{"path": "/etc/passwd", "mode": "Clone", "text": "test"}]
        with patch("os.path.exists", return_value=True), \
             patch("qwen3_tts.interface.ui._facade.load_config", return_value={}):
            result = on_history_select(evt, history)
        self.assertIsNone(result)

    def test_valid_path_returns_temp_copy(self):
        import tempfile, os
        from qwen3_tts.interface.ui._facade import on_history_select
        # Create a real temp file to simulate a Downloads file
        src = os.path.join(tempfile.gettempdir(), "test_history_select.wav")
        with open(src, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 40)
        try:
            evt = MagicMock()
            evt.index = [0]
            history = [{"path": src}]
            with patch("qwen3_tts.interface.ui._facade.load_config", return_value={}):
                result = on_history_select(evt, history)
            self.assertIsNotNone(result)
            self.assertTrue(result.startswith(tempfile.gettempdir()))
        finally:
            if os.path.exists(src):
                os.remove(src)

    def test_empty_history_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        result = on_history_select(evt, [])
        self.assertIsNone(result)

    def test_none_history_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        result = on_history_select(evt, None)
        self.assertIsNone(result)


class TestSanitizeVoiceName(unittest.TestCase):
    """Tests for _sanitize_voice_name allowlist."""

    def test_valid_name(self):
        from qwen3_tts.interface.ui._facade import _sanitize_voice_name
        name, err = _sanitize_voice_name("my_voice_01")
        self.assertEqual(name, "my_voice_01")
        self.assertIsNone(err)

    def test_spaces_converted_to_underscores(self):
        from qwen3_tts.interface.ui._facade import _sanitize_voice_name
        name, err = _sanitize_voice_name("my voice")
        self.assertEqual(name, "my_voice")
        self.assertIsNone(err)

    def test_traversal_rejected(self):
        from qwen3_tts.interface.ui._facade import _sanitize_voice_name
        _, err = _sanitize_voice_name("../../etc/passwd")
        self.assertIsNotNone(err)

    def test_empty_rejected(self):
        from qwen3_tts.interface.ui._facade import _sanitize_voice_name
        _, err = _sanitize_voice_name("")
        self.assertIsNotNone(err)

    def test_too_long_rejected(self):
        from qwen3_tts.interface.ui._facade import _sanitize_voice_name
        _, err = _sanitize_voice_name("a" * 65)
        self.assertIsNotNone(err)

    def test_hyphens_allowed(self):
        from qwen3_tts.interface.ui._facade import _sanitize_voice_name
        name, err = _sanitize_voice_name("my-voice")
        self.assertEqual(name, "my-voice")
        self.assertIsNone(err)


class TestFormatStatusDisplayEscaping(unittest.TestCase):
    """Tests for HTML escaping in format_status_display."""

    @patch("qwen3_tts.interface.ui.shared.get_server_status")
    def test_xss_in_status_is_escaped(self, mock_status):
        from qwen3_tts.interface.ui.shared import format_status_display
        mock_status.return_value = ("<script>alert(1)</script>", "N/A", "N/A", "N/A")
        result = format_status_display()
        self.assertNotIn("<script>", result)
        self.assertIn("&lt;script&gt;", result)


if __name__ == "__main__":
    unittest.main()
