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
from unittest.mock import MagicMock, patch

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
    """Tests for on_history_select.

    on_history_select returns a 4-tuple: (audio_path, clone_seed, design_seed,
    custom_seed). Tests here assert on the audio_path (first element); the
    seed-broadcast behavior is covered by TestOnHistorySelectSeedBroadcast.
    """

    def test_valid_selection_returns_temp_path(self):
        import os
        import tempfile

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
                audio, *_ = on_history_select(evt, history)
            self.assertIsNotNone(audio)
            self.assertTrue(audio.startswith(tempfile.gettempdir()))
        finally:
            if os.path.exists(src):
                os.remove(src)

    def test_invalid_index_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [5]
        history = [{"path": "/tmp/test.wav"}]
        audio, *_ = on_history_select(evt, history)
        self.assertIsNone(audio)

    def test_missing_path_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        history = [{"mode": "clone"}]
        audio, *_ = on_history_select(evt, history)
        self.assertIsNone(audio)

    def test_nonexistent_file_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        # Path is in a safe root (tempdir) but file doesn't exist
        import tempfile
        history = [{"path": tempfile.gettempdir() + "/nonexistent_file_xyz.wav"}]
        with patch("qwen3_tts.interface.ui._facade.load_config", return_value={}):
            audio, *_ = on_history_select(evt, history)
        self.assertIsNone(audio)

    def test_empty_history_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        audio, *_ = on_history_select(evt, [])
        self.assertIsNone(audio)


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
    """Tests for hardened on_history_select (containment + temp copy).

    on_history_select returns a 4-tuple (audio, seed, seed, seed); tests here
    destructure the tuple and assert on the audio element.
    """

    def test_unsafe_path_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        history = [{"path": "/etc/passwd", "mode": "Clone", "text": "test"}]
        with patch("os.path.exists", return_value=True), \
             patch("qwen3_tts.interface.ui._facade.load_config", return_value={}):
            audio, *_ = on_history_select(evt, history)
        self.assertIsNone(audio)

    def test_valid_path_returns_temp_copy(self):
        import os
        import tempfile

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
                audio, *_ = on_history_select(evt, history)
            self.assertIsNotNone(audio)
            self.assertTrue(audio.startswith(tempfile.gettempdir()))
        finally:
            if os.path.exists(src):
                os.remove(src)

    def test_empty_history_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        audio, *_ = on_history_select(evt, [])
        self.assertIsNone(audio)

    def test_none_history_returns_none(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        audio, *_ = on_history_select(evt, None)
        self.assertIsNone(audio)


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


class TestExtractSeedFromHistory(unittest.TestCase):
    """Tests for extract_seed_from_history — populates seed field on row click."""

    def test_valid_seed_returns_string(self):
        from qwen3_tts.interface.ui._facade import extract_seed_from_history
        evt = MagicMock()
        evt.index = [0]
        history = [{"seed": 42, "path": "/tmp/test.wav"}]
        result = extract_seed_from_history(evt, history)
        self.assertEqual(result, "42")

    def test_none_seed_returns_empty(self):
        from qwen3_tts.interface.ui._facade import extract_seed_from_history
        evt = MagicMock()
        evt.index = [0]
        history = [{"seed": None, "path": "/tmp/test.wav"}]
        result = extract_seed_from_history(evt, history)
        self.assertEqual(result, "")

    def test_missing_seed_key_returns_empty(self):
        from qwen3_tts.interface.ui._facade import extract_seed_from_history
        evt = MagicMock()
        evt.index = [0]
        history = [{"path": "/tmp/test.wav"}]
        result = extract_seed_from_history(evt, history)
        self.assertEqual(result, "")

    def test_invalid_index_returns_empty(self):
        from qwen3_tts.interface.ui._facade import extract_seed_from_history
        evt = MagicMock()
        evt.index = [5]
        history = [{"seed": 42}]
        result = extract_seed_from_history(evt, history)
        self.assertEqual(result, "")

    def test_empty_history_returns_empty(self):
        from qwen3_tts.interface.ui._facade import extract_seed_from_history
        evt = MagicMock()
        evt.index = [0]
        result = extract_seed_from_history(evt, [])
        self.assertEqual(result, "")

    def test_none_history_returns_empty(self):
        from qwen3_tts.interface.ui._facade import extract_seed_from_history
        evt = MagicMock()
        evt.index = [0]
        result = extract_seed_from_history(evt, None)
        self.assertEqual(result, "")

    def test_no_index_attr_returns_empty(self):
        from qwen3_tts.interface.ui._facade import extract_seed_from_history
        evt = MagicMock(spec=[])  # no .index attribute
        result = extract_seed_from_history(evt, [{"seed": 42}])
        self.assertEqual(result, "")


class TestOnHistorySelectSeedBroadcast(unittest.TestCase):
    """Tests that on_history_select emits seed to all three tab outputs.

    on_history_select returns a 4-tuple: (audio_path, clone_seed, design_seed, custom_seed).
    The last three are identical — broadcast to every tab's seed textbox.
    """

    def test_broadcasts_seed_to_three_outputs(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        # Use a path outside safe_roots so audio returns None but seed still broadcasts
        history = [{"seed": 12345, "path": "/etc/passwd"}]
        result = on_history_select(evt, history)
        # 8-tuple: audio, clone_seed, design_seed, custom_seed, df, state, payload, status
        self.assertEqual(len(result), 8)
        _audio, c, d, cu, *_rest = result
        self.assertEqual((c, d, cu), ("12345", "12345", "12345"))

    def test_broadcasts_empty_when_no_seed(self):
        from qwen3_tts.interface.ui._facade import on_history_select
        evt = MagicMock()
        evt.index = [0]
        history = [{"path": "/tmp/test.wav"}]
        result = on_history_select(evt, history)
        self.assertEqual(len(result), 8)
        _audio, c, d, cu, *_rest = result
        self.assertEqual((c, d, cu), ("", "", ""))


class TestOnHistorySelectColumnRouting(unittest.TestCase):
    """Column-aware routing in on_history_select: copy / delete / replay.

    on_history_select returns an 8-tuple; the payload at index 6 carries the
    action ("copy"|"delete"|"replay") consumed by the copy .then(js=...) chain.
    """

    def _evt(self, row, col):
        evt = MagicMock()
        evt.index = [row, col]
        return evt

    def test_text_preview_column_copies_full_transcript(self):
        from qwen3_tts.interface.ui._facade import on_history_select

        full = "The quick brown fox jumps over the lazy dog."
        history = [
            {"text": full[:40], "full_text": full, "path": "/tmp/x.wav", "seed": 1}
        ]
        result = on_history_select(self._evt(0, 2), history)
        payload = result[6]
        self.assertEqual(payload["action"], "copy")
        self.assertEqual(payload["text"], full)
        # The visible "Copied" status is set optimistically by the handler (8th
        # element), not carried in the payload — the clipboard write is a JS
        # side-effect and Gradio's fn-return→output model forbids a JS return
        # from driving the status output.
        status = result[7]
        self.assertIn("Copied", status)

    def test_remove_column_deletes_row_and_clears_audio(self):
        from qwen3_tts.interface.ui._facade import HISTORY_COL_DELETE, on_history_select

        history = [
            {"text": "a", "full_text": "a", "path": "/tmp/a.wav", "seed": 1},
            {"text": "b", "full_text": "b", "path": "/tmp/b.wav", "seed": 2},
            {"text": "c", "full_text": "c", "path": "/tmp/c.wav", "seed": 3},
        ]
        result = on_history_select(self._evt(1, HISTORY_COL_DELETE), history)
        audio, _c, _d, _cu, df, state, payload, status = result
        # Player cleared on delete via None (NOT "" — "" makes gr.Audio
        # postprocess abspath "" to the CWD and crash; see on_history_select).
        self.assertIsNone(audio)
        self.assertEqual(payload["action"], "delete")
        self.assertEqual(len(state), 2)  # middle row removed
        self.assertEqual([e["text"] for e in state], ["a", "c"])
        self.assertEqual(len(df), 2)  # dataframe re-rendered from new list
        self.assertIn("Entry removed", status)

    def test_remove_column_does_not_mutate_input(self):
        from qwen3_tts.interface.ui._facade import HISTORY_COL_DELETE, on_history_select

        history = [
            {"text": "a", "path": "/tmp/a.wav"},
            {"text": "b", "path": "/tmp/b.wav"},
        ]
        on_history_select(self._evt(0, HISTORY_COL_DELETE), history)
        self.assertEqual(len(history), 2)  # input list not mutated

    def test_other_column_is_replay_and_broadcasts_seed(self):
        from qwen3_tts.interface.ui._facade import on_history_select

        history = [{"text": "hi", "full_text": "hi", "seed": 99, "path": "/tmp/x.wav"}]
        # Column 0 (Time) -> replay branch; path is outside safe roots so audio
        # is None, but the seed still broadcasts and action is "replay".
        result = on_history_select(self._evt(0, 0), history)
        self.assertEqual(result[6]["action"], "replay")
        self.assertEqual(result[1], "99")  # clone seed

    def test_legacy_row_only_event_treated_as_replay(self):
        from qwen3_tts.interface.ui._facade import on_history_select

        evt = MagicMock()
        evt.index = [0]  # no column dimension -> legacy select event
        history = [{"text": "hi", "full_text": "hi", "seed": 7, "path": "/tmp/x.wav"}]
        result = on_history_select(evt, history)
        self.assertEqual(result[6]["action"], "replay")


class TestOnClearHistoryClick(unittest.TestCase):
    """Two-step confirm for Clear All (list-only; disk files untouched)."""

    def test_first_click_arms_and_leaves_history_untouched(self):
        import time  # noqa: F401  (kept for parity with the confirm test below)

        from qwen3_tts.interface.ui._facade import on_clear_history_click

        state = {"armed": False, "ts": 0.0}
        history = [{"text": "a", "path": "/tmp/a.wav"}, {"text": "b", "path": "/tmp/b.wav"}]
        new_state, _btn, df, hist, _audio, status, payload = on_clear_history_click(state, history)
        self.assertTrue(new_state["armed"])  # armed on first click
        self.assertEqual(df, gr.update())  # table unchanged
        self.assertEqual(hist, gr.update())  # history_state unchanged
        self.assertIn("Click again", status)
        self.assertEqual(payload["action"], "replay")  # no waveform clear on arm

    def test_second_click_within_timeout_clears_history(self):
        import time

        from qwen3_tts.interface.ui._facade import on_clear_history_click

        # Armed + recent timestamp -> confirm_step confirms within 5s window.
        state = {"armed": True, "ts": time.time()}
        history = [{"text": "a", "path": "/tmp/a.wav"}, {"text": "b", "path": "/tmp/b.wav"}]
        new_state, _btn, df, hist, audio, status, payload = on_clear_history_click(state, history)
        self.assertFalse(new_state["armed"])  # disarmed after action
        self.assertEqual(hist, [])  # history_state cleared
        self.assertEqual(df, [])  # table re-rendered empty
        self.assertIsNone(audio)  # player cleared via None ("" crashes Audio postprocess)
        self.assertIn("cleared", status)
        self.assertEqual(payload["action"], "clear")  # triggers waveform clear

    def test_second_click_clears_even_when_history_none(self):
        import time

        from qwen3_tts.interface.ui._facade import on_clear_history_click

        state = {"armed": True, "ts": time.time()}
        _new_state, _btn, df, hist, _audio, _status, _payload = on_clear_history_click(state, None)
        self.assertEqual(hist, [])
        self.assertEqual(df, [])


class TestFormatStatusDisplayEscaping(unittest.TestCase):
    """Tests for HTML escaping in format_status_display."""

    @patch("qwen3_tts.interface.ui.shared.get_server_status")
    def test_xss_in_status_is_escaped(self, mock_status):
        from qwen3_tts.interface.ui.shared import format_status_display
        mock_status.return_value = ("<script>alert(1)</script>", "N/A", "N/A", "N/A")
        result = format_status_display()
        self.assertNotIn("<script>", result)
        self.assertIn("&lt;script&gt;", result)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestConfirmStep(unittest.TestCase):
    """Tests for confirm_step two-step confirmation pattern."""

    def test_first_click_arms_button(self):
        """First click returns confirmed=False and updates button label."""
        from qwen3_tts.interface.ui.components import confirm_step
        state, btn_update, confirmed = confirm_step(
            None, "Confirm Delete? (click again)", "Delete"
        )
        self.assertFalse(confirmed)
        self.assertTrue(state.get("armed", False))
        self.assertIsNotNone(btn_update)

    def test_second_click_confirms_within_timeout(self):
        """Second click within timeout returns confirmed=True."""
        from qwen3_tts.interface.ui.components import confirm_step
        # First click - arm
        state, _, confirmed = confirm_step(
            None, "Confirm Delete? (click again)", "Delete"
        )
        self.assertFalse(confirmed)
        # Second click - confirm
        state, btn_update, confirmed = confirm_step(
            state, "Confirm Delete? (click again)", "Delete"
        )
        self.assertTrue(confirmed)
        self.assertFalse(state.get("armed", True))

    def test_timeout_requires_rearming(self):
        """Click after timeout re-arms the button (requires two clicks again)."""
        import time

        from qwen3_tts.interface.ui.components import confirm_step
        # First click - arm
        state, _, _ = confirm_step(
            None, "Confirm Delete? (click again)", "Delete", timeout_s=0.1
        )
        self.assertTrue(state.get("armed", False))
        # Wait for timeout
        time.sleep(0.15)
        # Click after timeout - should re-arm, not confirm
        state, btn_update, confirmed = confirm_step(
            state, "Confirm Delete? (click again)", "Delete", timeout_s=0.1
        )
        self.assertFalse(confirmed)  # Not confirmed
        self.assertTrue(state.get("armed", False))  # Re-armed
        # Need another click to actually confirm
        state, _, confirmed = confirm_step(
            state, "Confirm Delete? (click again)", "Delete", timeout_s=0.1
        )
        self.assertTrue(confirmed)  # Now confirmed

    def test_timeout_prevents_confirmation(self):
        """Cannot confirm after timeout expires."""
        import time

        from qwen3_tts.interface.ui.components import confirm_step
        # First click - arm
        state, _, _ = confirm_step(
            None, "Confirm Delete? (click again)", "Delete", timeout_s=0.1
        )
        # Wait for timeout
        time.sleep(0.15)
        # Try to confirm - should fail
        state, _, confirmed = confirm_step(
            state, "Confirm Delete? (click again)", "Delete", timeout_s=0.1
        )
        self.assertFalse(confirmed)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestConfirmButton(unittest.TestCase):
    """Tests for ConfirmButton wrapper class."""

    def test_click_returns_four_tuple(self):
        """ConfirmButton.click returns (state, btn, status, confirmed)."""
        from qwen3_tts.interface.ui.components import ConfirmButton
        btn = ConfirmButton(
            arm_label="Confirm Delete? (click again)",
            original_label="Delete",
            timeout_s=5.0,
            status_message="Please confirm within 5 seconds"
        )
        result = btn.click(None)
        self.assertEqual(len(result), 4)
        state, btn_update, status_update, confirmed = result
        self.assertIsInstance(state, dict)
        self.assertFalse(confirmed)

    def test_first_click_shows_status_message(self):
        """First click shows status message."""
        from qwen3_tts.interface.ui.components import ConfirmButton
        btn = ConfirmButton(
            arm_label="Confirm Delete? (click again)",
            original_label="Delete",
            status_message="Please confirm within 5 seconds"
        )
        state, btn_update, status_update, confirmed = btn.click(None)
        self.assertFalse(confirmed)
        self.assertIsNotNone(status_update)

    def test_second_click_clears_status(self):
        """Second click clears status message."""
        from qwen3_tts.interface.ui.components import ConfirmButton
        btn = ConfirmButton(
            arm_label="Confirm Delete? (click again)",
            original_label="Delete",
            status_message="Please confirm within 5 seconds"
        )
        # First click
        state, _, _, confirmed = btn.click(None)
        self.assertFalse(confirmed)
        # Second click
        state, btn_update, status_update, confirmed = btn.click(state)
        self.assertTrue(confirmed)


class TestUnloadHandlerImport(unittest.TestCase):
    """Regression tests for the Manage Models 'Unload' handler (UI-1, UI-2).

    The on_unload_click closure previously imported get_model_table_data from
    qwen3_tts.interface.ui.shared, where it does not exist, so the first click
    on 'Unload' raised ImportError at runtime. It lives in model_management.
    """

    def test_get_model_table_data_importable_from_model_management(self):
        """The handler's import path must resolve (would catch the ImportError)."""
        from qwen3_tts.interface.ui.model_management import get_model_table_data
        self.assertTrue(callable(get_model_table_data))

    def test_get_model_table_data_not_in_shared(self):
        """Document that .shared does NOT provide it — the source of the old bug."""
        import qwen3_tts.interface.ui.shared as shared
        self.assertFalse(
            hasattr(shared, "get_model_table_data"),
            "get_model_table_data unexpectedly in .shared; update the handler import",
        )

    def test_facade_imports_from_correct_submodule(self):
        """Pin the fix: _facade source imports get_model_table_data from model_management."""
        import inspect

        from qwen3_tts.interface.ui import _facade
        src = inspect.getsource(_facade)
        self.assertIn(
            "from qwen3_tts.interface.ui.model_management import get_model_table_data",
            src,
        )
        self.assertNotIn(
            "from qwen3_tts.interface.ui.shared import get_model_table_data", src
        )

    def test_startup_warning_matches_table_values(self):
        """UI-2: startup warning compares against 'Yes' (what the table emits), not 'default'."""
        import inspect

        from qwen3_tts.interface.ui import _facade
        src = inspect.getsource(_facade)
        self.assertNotIn('if startup == "default":', src)
        self.assertIn('if startup == "Yes":', src)


if __name__ == "__main__":
    unittest.main()
