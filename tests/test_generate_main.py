#!/usr/bin/env python3
"""Tests for generate.py — the CLI orchestrator module.

Covers:
  - _build_parser: argument parsing
  - process_batch: server and local batch processing
  - _handle_info_commands: list-*, stats, history, prompt management
  - _handle_list_models: model listing display
  - _handle_stats: server stats display
  - _handle_dry_run: dry-run summary
  - _handle_generation: dispatch logic (voice alias, clipboard, batch, text)
  - main: entry point dispatch
  - process_dialogue / process_srt_file: delegators

Run: pytest tests/test_generate_main.py -v
"""
import argparse
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MODEL_INFO in config.py is keyed by size ("1.7B"/"0.6B") then model_type.
# _handle_list_models iterates MODEL_INFO.items() expecting model_type keys
# with 'name'/'description' — provide the shape the code consumes.
_MOCK_MODEL_INFO = {
    "clone": {"name": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "description": "Clone", "memory_mb": 3500},
    "design": {"name": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "description": "Design", "memory_mb": 3500},
    "custom": {"name": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "description": "Custom", "memory_mb": 3500},
}


class TestBuildParser(unittest.TestCase):
    """Tests for _build_parser argument parsing."""

    def _parse(self, args_list):
        from qwen3_tts.interface.generate import _build_parser
        return _build_parser().parse_args(args_list)

    def test_text_positional(self):
        args = self._parse(["Hello world"])
        self.assertEqual(args.text, ["Hello world"])

    def test_mode_flag(self):
        args = self._parse(["-m", "design", "Hello"])
        self.assertEqual(args.mode, "design")

    def test_output_flag(self):
        args = self._parse(["-o", "out.wav", "Hello"])
        self.assertEqual(args.output, "out.wav")

    def test_play_and_stream_flags(self):
        args = self._parse(["--play", "--stream", "Hello"])
        self.assertTrue(args.play)
        self.assertTrue(args.stream)

    def test_batch_flag(self):
        args = self._parse(["--batch", "texts.json"])
        self.assertEqual(args.batch, "texts.json")

    def test_dry_run_flag(self):
        args = self._parse(["--dry-run", "Hello"])
        self.assertTrue(args.dry_run)

    def test_server_mode_flag(self):
        args = self._parse(["--_server-mode", "Hello"])
        self.assertTrue(args.server_mode)

    def test_ui_flag(self):
        args = self._parse(["--ui"])
        self.assertTrue(args.ui)

    def test_repl_flag(self):
        args = self._parse(["--repl"])
        self.assertTrue(args.repl)

    def test_srt_and_dialogue_flags(self):
        args = self._parse(["--srt", "subs.srt"])
        self.assertEqual(args.srt, "subs.srt")
        args = self._parse(["--dialogue", "dialog.json"])
        self.assertEqual(args.dialogue, "dialog.json")

    def test_audio_processing_flags(self):
        args = self._parse(["--trim-silence", "--normalize", "--speed", "1.5", "--pitch", "2.0", "Hello"])
        self.assertTrue(args.trim_silence)
        self.assertTrue(args.normalize)
        self.assertEqual(args.speed, 1.5)
        self.assertEqual(args.pitch, 2.0)


class TestProcessBatch(unittest.TestCase):
    """Tests for process_batch function."""

    def _make_args(self, **kwargs):
        defaults = {
            "output": "/tmp/test_batch",
            "mode": "clone",
            "prompt": "voice.pt",
            "description": "",
            "trim_silence": False,
            "normalize": False,
            "speed": None,
            "pitch": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("qwen3_tts.interface.generate.generate_via_server")
    @patch("qwen3_tts.interface.generate._save_base64_result")
    @patch("os.makedirs")
    @patch("builtins.print")
    def test_server_mode_no_processing(self, _print, _makedirs, mock_save, mock_gen):
        from qwen3_tts.interface.generate import process_batch
        mock_gen.return_value = ["base64data1", "base64data2"]
        args = self._make_args()
        config = {"language": "English"}
        gen_params = {"temperature": 0.7}

        result = process_batch(["Hello", "World"], args, config, gen_params, use_server=True)

        self.assertEqual(len(result), 2)
        mock_gen.assert_called_once()
        self.assertEqual(mock_save.call_count, 2)

    @patch("qwen3_tts.interface.generate.generate_local")
    @patch("qwen3_tts.interface.generate.process_audio_args")
    @patch("os.makedirs")
    @patch("builtins.print")
    def test_local_mode(self, _print, _makedirs, mock_process, mock_gen):
        import numpy as np
        from qwen3_tts.interface.generate import process_batch
        wav = np.zeros(1000, dtype="float32")
        mock_gen.return_value = (wav, 24000)
        mock_process.return_value = wav
        args = self._make_args()
        config = {"language": "English"}
        gen_params = {"temperature": 0.7}

        with patch("soundfile.write"):
            result = process_batch(["Hello"], args, config, gen_params, use_server=False)

        self.assertEqual(len(result), 1)
        mock_gen.assert_called_once()

    @patch("qwen3_tts.interface.generate.generate_via_server")
    @patch("qwen3_tts.interface.generate._decode_base64_result")
    @patch("qwen3_tts.interface.generate.process_audio_args")
    @patch("os.makedirs")
    @patch("builtins.print")
    def test_server_mode_with_processing(self, _print, _makedirs, mock_process, mock_decode, mock_gen):
        import numpy as np
        from qwen3_tts.interface.generate import process_batch
        wav = np.zeros(1000, dtype="float32")
        mock_gen.return_value = ["base64data"]
        mock_decode.return_value = (wav, 24000)
        mock_process.return_value = wav
        args = self._make_args(trim_silence=True)
        config = {"language": "English"}

        with patch("soundfile.write"):
            result = process_batch(["Hello"], args, config, {"temperature": 0.7}, use_server=True)

        self.assertEqual(len(result), 1)
        mock_decode.assert_called_once()
        mock_process.assert_called_once()


class TestDelegators(unittest.TestCase):
    """Tests for process_dialogue and process_srt_file delegators."""

    @patch("qwen3_tts.interface.cli.dialogue.process_dialogue")
    def test_process_dialogue_delegates(self, mock_impl):
        from qwen3_tts.interface.generate import process_dialogue
        process_dialogue("dialog.json", {}, None, {}, True)
        mock_impl.assert_called_once_with("dialog.json", {}, None, {}, True)

    @patch("qwen3_tts.interface.cli.srt.process_srt_file")
    def test_process_srt_file_delegates(self, mock_impl):
        from qwen3_tts.interface.generate import process_srt_file
        process_srt_file("subs.srt", {}, None, {}, False)
        mock_impl.assert_called_once_with("subs.srt", {}, None, {}, False)


class TestHandleInfoCommands(unittest.TestCase):
    """Tests for _handle_info_commands dispatch."""

    def _make_args(self, **kwargs):
        defaults = {
            "list_backends": False, "list_prompts": False, "voices": False,
            "list_presets": False, "list_aliases": False, "list_prosody": False,
            "list_speakers": False, "list_models": False, "stats": False,
            "edit_config": False, "history": None, "delete_prompt": None,
            "rename_prompt": None, "preview_prompt": None, "backend": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_no_info_command_returns_none(self):
        from qwen3_tts.interface.generate import _handle_info_commands
        args = self._make_args()
        result = _handle_info_commands(args, {}, {})
        self.assertIsNone(result)

    @patch("qwen3_tts.interface.generate.get_backend", return_value="mlx")
    @patch("qwen3_tts.interface.generate.get_mlx_quantization", return_value="8bit")
    @patch("qwen3_tts.interface.generate.get_mlx_model_name", return_value="mlx-community/model")
    @patch("builtins.print")
    def test_list_backends_mlx(self, mock_print, *_mocks):
        from qwen3_tts.interface.generate import _handle_info_commands
        args = self._make_args(list_backends=True)
        result = _handle_info_commands(args, {}, {})
        self.assertFalse(result)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("mlx", output)

    @patch("qwen3_tts.interface.generate.MODEL_INFO", _MOCK_MODEL_INFO)
    @patch("qwen3_tts.interface.generate.get_backend", return_value="torch")
    @patch("qwen3_tts.interface.generate.get_torch_dtype_name", return_value="float32")
    @patch("builtins.print")
    def test_list_backends_torch(self, mock_print, *_mocks):
        from qwen3_tts.interface.generate import _handle_info_commands
        args = self._make_args(list_backends=True)
        result = _handle_info_commands(args, {}, {})
        self.assertFalse(result)

    @patch("qwen3_tts.interface.generate.list_voice_prompts", return_value=["voice1.pt", "voice2.pt"])
    @patch("builtins.print")
    def test_list_prompts(self, mock_print, _mock):
        from qwen3_tts.interface.generate import _handle_info_commands
        args = self._make_args(list_prompts=True)
        result = _handle_info_commands(args, {}, {})
        self.assertFalse(result)

    @patch("builtins.print")
    def test_list_presets(self, mock_print):
        from qwen3_tts.interface.generate import _handle_info_commands
        args = self._make_args(list_presets=True)
        config = {"presets": {"fast": {"temperature": 0.3}}}
        result = _handle_info_commands(args, config, {})
        self.assertFalse(result)

    @patch("builtins.print")
    def test_list_aliases_empty(self, mock_print):
        from qwen3_tts.interface.generate import _handle_info_commands
        args = self._make_args(list_aliases=True)
        result = _handle_info_commands(args, {}, {})
        self.assertFalse(result)

    @patch("builtins.print")
    def test_list_aliases_with_data(self, mock_print):
        from qwen3_tts.interface.generate import _handle_info_commands
        args = self._make_args(list_aliases=True)
        config = {"aliases": {"myvoice": {"prompt": "v.pt", "mode": "clone"}}}
        result = _handle_info_commands(args, config, {})
        self.assertFalse(result)

    @patch("qwen3_tts.core.config.get_prosody_presets", return_value={"calm": "Speak calmly"})
    @patch("builtins.print")
    def test_list_prosody(self, mock_print, _mock):
        from qwen3_tts.interface.generate import _handle_info_commands
        args = self._make_args(list_prosody=True)
        result = _handle_info_commands(args, {}, {})
        self.assertFalse(result)

    @patch("builtins.print")
    def test_list_speakers(self, mock_print):
        from qwen3_tts.interface.generate import _handle_info_commands
        args = self._make_args(list_speakers=True)
        result = _handle_info_commands(args, {}, {})
        self.assertFalse(result)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("Premium", output)

    @patch("qwen3_tts.interface.generate.show_history")
    def test_history(self, mock_hist):
        from qwen3_tts.interface.generate import _handle_info_commands
        args = self._make_args(history=5)
        result = _handle_info_commands(args, {}, {})
        self.assertFalse(result)
        mock_hist.assert_called_once_with(5)

    @patch("qwen3_tts.interface.generate.delete_voice_prompt")
    def test_delete_prompt(self, mock_del):
        from qwen3_tts.interface.generate import _handle_info_commands
        args = self._make_args(delete_prompt="old_voice")
        result = _handle_info_commands(args, {}, {})
        self.assertFalse(result)
        mock_del.assert_called_once_with("old_voice")

    @patch("qwen3_tts.interface.generate.rename_voice_prompt")
    def test_rename_prompt(self, mock_rename):
        from qwen3_tts.interface.generate import _handle_info_commands
        args = self._make_args(rename_prompt=["old", "new"])
        result = _handle_info_commands(args, {}, {})
        self.assertFalse(result)
        mock_rename.assert_called_once_with("old", "new")

    @patch("qwen3_tts.interface.generate.preview_voice_prompt")
    def test_preview_prompt(self, mock_preview):
        from qwen3_tts.interface.generate import _handle_info_commands
        args = self._make_args(preview_prompt="my_voice")
        result = _handle_info_commands(args, {"key": "val"}, {})
        self.assertFalse(result)
        mock_preview.assert_called_once_with("my_voice", {"key": "val"})


class TestHandleListModels(unittest.TestCase):
    """Tests for _handle_list_models."""

    @patch("qwen3_tts.interface.generate.MODEL_INFO", _MOCK_MODEL_INFO)
    @patch("qwen3_tts.interface.generate.is_server_running", return_value=False)
    @patch("builtins.print")
    def test_server_not_running(self, mock_print, _mock):
        from qwen3_tts.interface.generate import _handle_list_models
        args = argparse.Namespace(backend=None)
        result = _handle_list_models(args, {})
        self.assertFalse(result)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("server not running", output)

    @patch("qwen3_tts.interface.generate.MODEL_INFO", _MOCK_MODEL_INFO)
    @patch("qwen3_tts.interface.generate.is_server_running", return_value=True)
    @patch("qwen3_tts.interface.generate.get_server_url", return_value="http://127.0.0.1:5123")
    @patch("qwen3_tts.interface.generate.auth_headers", return_value={})
    @patch("builtins.print")
    def test_server_running_with_models(self, mock_print, _auth, _url, _running):
        from qwen3_tts.interface.generate import _handle_list_models
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": {"clone": {"loaded": True}}}
        with patch("requests.get", return_value=mock_resp):
            args = argparse.Namespace(backend=None)
            result = _handle_list_models(args, {})
        self.assertFalse(result)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("LOADED", output)


class TestHandleStats(unittest.TestCase):
    """Tests for _handle_stats."""

    @patch("qwen3_tts.interface.generate.is_server_running", return_value=False)
    @patch("builtins.print")
    def test_server_not_running(self, mock_print, _mock):
        from qwen3_tts.interface.generate import _handle_stats
        result = _handle_stats({})
        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn("not running", mock_print.call_args[0][0])

    @patch("qwen3_tts.interface.generate.is_server_running", return_value=True)
    @patch("qwen3_tts.interface.generate.get_server_url", return_value="http://127.0.0.1:5123")
    @patch("qwen3_tts.interface.generate.auth_headers", return_value={})
    @patch("builtins.print")
    def test_server_running_success(self, mock_print, _auth, _url, _running):
        from qwen3_tts.interface.generate import _handle_stats
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"generations": 5, "uptime": "1h"}
        with patch("requests.get", return_value=mock_resp):
            result = _handle_stats({})
        self.assertFalse(result)

    @patch("qwen3_tts.interface.generate.is_server_running", return_value=True)
    @patch("qwen3_tts.interface.generate.get_server_url", return_value="http://127.0.0.1:5123")
    @patch("qwen3_tts.interface.generate.auth_headers", return_value={})
    @patch("builtins.print")
    def test_server_error_response(self, mock_print, _auth, _url, _running):
        from qwen3_tts.interface.generate import _handle_stats
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("requests.get", return_value=mock_resp):
            result = _handle_stats({})
        self.assertFalse(result)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("Error", output)


class TestHandleDryRun(unittest.TestCase):
    """Tests for _handle_dry_run."""

    def _make_args(self, **kwargs):
        defaults = {
            "mode": "clone", "prompt": "voice.pt", "description": "",
            "output": None, "batch": None, "text": ["Hello"],
            "speaker": None, "instruct": None, "trim_silence": False,
            "normalize": False, "speed": None, "pitch": None,
            "ssml": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("builtins.print")
    def test_dry_run_single_text(self, mock_print):
        from qwen3_tts.interface.generate import _handle_dry_run
        args = self._make_args()
        config = {"output_directory": "/tmp"}
        result = _handle_dry_run(args, config, {"temperature": 0.7}, True, 500)
        self.assertFalse(result)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("DRY RUN", output)
        self.assertIn("clone", output)

    @patch("builtins.print")
    def test_dry_run_custom_mode(self, mock_print):
        from qwen3_tts.interface.generate import _handle_dry_run
        args = self._make_args(mode="custom", speaker="ryan", instruct="happy")
        config = {"output_directory": "/tmp"}
        result = _handle_dry_run(args, config, {}, False, None)
        self.assertFalse(result)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("custom", output)

    @patch("builtins.print")
    def test_dry_run_design_mode(self, mock_print):
        from qwen3_tts.interface.generate import _handle_dry_run
        args = self._make_args(mode="design", description="warm voice")
        config = {"output_directory": "/tmp", "default_voice_description": "warm voice"}
        result = _handle_dry_run(args, config, {}, True, 0)
        self.assertFalse(result)


class TestHandleGeneration(unittest.TestCase):
    """Tests for _handle_generation dispatch."""

    def _make_args(self, **kwargs):
        defaults = {
            "repl": False, "watch": None, "srt": None, "dialogue": None,
            "voice": None, "clipboard": False, "dry_run": False,
            "batch": None, "text": [], "text_override": None, "ssml": False,
            "output": None, "mode": None, "prompt": None, "description": None,
            "preset": None, "prosody": None, "instruct": None, "speaker": None,
            "play": False, "no_open": False, "stream": False,
            "trim_silence": False, "normalize": False, "speed": None, "pitch": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("qwen3_tts.interface.generate.run_repl")
    def test_repl_dispatch(self, mock_repl):
        from qwen3_tts.interface.generate import _handle_generation
        args = self._make_args(repl=True)
        result = _handle_generation(args, {}, {}, True, None)
        mock_repl.assert_called_once()
        self.assertTrue(result)

    @patch("qwen3_tts.interface.generate.interactive_mode", return_value=None)
    def test_interactive_mode_when_no_text(self, mock_interactive):
        from qwen3_tts.interface.generate import _handle_generation
        args = self._make_args()
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            result = _handle_generation(args, {}, {}, True, None)
        self.assertFalse(result)
        mock_interactive.assert_called_once()

    @patch("qwen3_tts.interface.generate._run_single_generation", return_value=True)
    @patch("builtins.open", mock_open())
    def test_single_text_generation(self, mock_gen):
        from qwen3_tts.interface.generate import _handle_generation
        args = self._make_args(text=["Hello world"])
        config = {"output_directory": "/tmp", "language": "English", "default_voice_description": ""}
        _handle_generation(args, config, {}, True, None)
        mock_gen.assert_called_once()

    @patch("qwen3_tts.interface.generate.get_voice_alias")
    @patch("qwen3_tts.interface.generate._run_single_generation", return_value=True)
    @patch("builtins.open", mock_open())
    @patch("builtins.print")
    def test_voice_alias_resolution(self, _print, mock_gen, mock_alias):
        from qwen3_tts.interface.generate import _handle_generation
        mock_alias.return_value = {"prompt": "narrator.pt", "mode": "clone"}
        args = self._make_args(text=["Hello"], voice="narrator")
        config = {"output_directory": "/tmp", "language": "English", "default_voice_description": ""}
        _handle_generation(args, config, {}, True, None)
        mock_alias.assert_called_once_with("narrator", config)

    @patch("qwen3_tts.interface.generate.get_voice_alias", return_value=None)
    @patch("builtins.print")
    def test_unknown_voice_alias_exits(self, _print, mock_alias):
        from qwen3_tts.interface.generate import _handle_generation
        args = self._make_args(text=["Hello"], voice="nonexistent")
        with self.assertRaises(SystemExit):
            _handle_generation(args, {"aliases": {}}, {}, True, None)


class TestMain(unittest.TestCase):
    """Tests for main() entry point."""

    @patch("qwen3_tts.interface.generate._build_parser")
    @patch("qwen3_tts.interface.generate.load_config", return_value={})
    @patch("qwen3_tts.interface.generate.get_generation_params", return_value={})
    @patch("qwen3_tts.interface.generate.launch_gradio_ui")
    def test_ui_flag_launches_gradio(self, mock_ui, _params, _config, mock_parser):
        from qwen3_tts.interface.generate import main
        mock_args = MagicMock()
        mock_args.backend = None
        mock_args.model_size = None
        mock_args.ui = True
        mock_parser.return_value.parse_args.return_value = mock_args
        result = main()
        mock_ui.assert_called_once()
        self.assertFalse(result)

    @patch("qwen3_tts.interface.generate._build_parser")
    @patch("qwen3_tts.interface.generate.load_config", return_value={})
    @patch("qwen3_tts.interface.generate.get_generation_params", return_value={})
    @patch("qwen3_tts.interface.generate._handle_info_commands", return_value=False)
    def test_info_command_handled(self, mock_info, _params, _config, mock_parser):
        from qwen3_tts.interface.generate import main
        mock_args = MagicMock()
        mock_args.backend = None
        mock_args.model_size = None
        mock_args.ui = False
        mock_parser.return_value.parse_args.return_value = mock_args
        result = main()
        self.assertFalse(result)

    @patch("qwen3_tts.interface.generate._build_parser")
    @patch("qwen3_tts.interface.generate.load_config", return_value={})
    @patch("qwen3_tts.interface.generate.get_generation_params", return_value={})
    @patch("qwen3_tts.interface.generate._handle_info_commands", return_value=None)
    @patch("qwen3_tts.interface.generate._handle_generation", return_value=True)
    def test_generation_dispatched(self, mock_gen, mock_info, _params, _config, mock_parser):
        from qwen3_tts.interface.generate import main
        mock_args = MagicMock()
        mock_args.backend = None
        mock_args.model_size = None
        mock_args.ui = False
        mock_args.server_mode = True
        mock_args.local = False
        mock_parser.return_value.parse_args.return_value = mock_args
        result = main()
        mock_gen.assert_called_once()
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
