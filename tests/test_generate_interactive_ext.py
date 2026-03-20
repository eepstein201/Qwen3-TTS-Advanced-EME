#!/usr/bin/env python3
"""Extended tests for generate_interactive module.

Covers deeper paths not in test_generate_interactive.py:
  - delete_voice_prompt: confirmed deletion, cancelled
  - rename_voice_prompt: success, already-exists, rollback on OSError
  - preview_voice_prompt: server success, server error
  - _ProgressPoller: _run dispatch, fallback loop, rich loop
  - interactive_mode: UI choice, text input, mode selection
  - run_repl: commands /quit, /voice, /preset, /prompt, /play, /speed, /pitch, /status

Run: pytest tests/test_generate_interactive_ext.py -v
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDeleteVoicePromptDeep(unittest.TestCase):
    """Deeper tests for delete_voice_prompt."""

    @patch("os.path.exists", side_effect=lambda p: p.endswith(".pt"))
    @patch("os.path.join", side_effect=lambda d, f: f"/voice_prompts/{f}")
    @patch("os.remove")
    @patch("builtins.input", return_value="y")
    @patch("builtins.print")
    def test_confirmed_deletion(self, _print, _input, mock_remove, _join, _exists):
        from qwen3_tts.interface.generate_interactive import delete_voice_prompt
        result = delete_voice_prompt("my_voice")
        self.assertTrue(result)
        mock_remove.assert_called_once()

    @patch("os.path.exists", side_effect=lambda p: p.endswith(".pt"))
    @patch("os.path.join", side_effect=lambda d, f: f"/voice_prompts/{f}")
    @patch("builtins.input", return_value="n")
    @patch("builtins.print")
    def test_cancelled_deletion(self, _print, _input, _join, _exists):
        from qwen3_tts.interface.generate_interactive import delete_voice_prompt
        result = delete_voice_prompt("my_voice")
        self.assertFalse(result)

    @patch("os.path.exists", side_effect=lambda p: p.endswith(".wav") or p.endswith(".txt"))
    @patch("os.path.join", side_effect=lambda d, f: f"/voice_prompts/{f}")
    @patch("os.remove")
    @patch("builtins.input", return_value="y")
    @patch("builtins.print")
    def test_deletes_mlx_dual_format(self, _print, _input, mock_remove, _join, _exists):
        from qwen3_tts.interface.generate_interactive import delete_voice_prompt
        result = delete_voice_prompt("my_voice.wav")
        self.assertTrue(result)
        self.assertEqual(mock_remove.call_count, 2)

    def test_backslash_rejected(self):
        from qwen3_tts.interface.generate_interactive import delete_voice_prompt
        with patch("builtins.print"):
            result = delete_voice_prompt("evil\\path")
        self.assertFalse(result)


class TestRenameVoicePromptDeep(unittest.TestCase):
    """Deeper tests for rename_voice_prompt."""

    @patch("os.path.exists")
    @patch("os.path.join", side_effect=lambda d, f: f"/voice_prompts/{f}")
    @patch("os.rename")
    @patch("builtins.print")
    def test_successful_rename(self, _print, mock_rename, _join, mock_exists):
        from qwen3_tts.interface.generate_interactive import rename_voice_prompt
        # old .pt exists, new .pt does not
        mock_exists.side_effect = lambda p: "old_voice" in p and p.endswith(".pt")
        result = rename_voice_prompt("old_voice", "new_voice")
        self.assertTrue(result)
        mock_rename.assert_called_once()

    @patch("os.path.exists")
    @patch("os.path.join", side_effect=lambda d, f: f"/voice_prompts/{f}")
    @patch("builtins.print")
    def test_target_already_exists(self, mock_print, _join, mock_exists):
        from qwen3_tts.interface.generate_interactive import rename_voice_prompt
        # Both old and new .pt files exist
        mock_exists.return_value = True
        result = rename_voice_prompt("old_voice", "existing_voice")
        self.assertFalse(result)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("already exists", output)

    @patch("os.path.exists")
    @patch("os.path.join", side_effect=lambda d, f: f"/voice_prompts/{f}")
    @patch("os.rename")
    @patch("builtins.print")
    def test_rollback_on_oserror(self, _print, mock_rename, _join, mock_exists):
        from qwen3_tts.interface.generate_interactive import rename_voice_prompt
        # old .pt and .wav exist, new files don't
        def exists_side_effect(p):
            return ("old_voice" in p) and (p.endswith(".pt") or p.endswith(".wav"))
        mock_exists.side_effect = exists_side_effect
        # First rename succeeds, second raises, third is the rollback
        mock_rename.side_effect = [None, OSError("disk full"), None]
        result = rename_voice_prompt("old_voice", "new_voice")
        self.assertFalse(result)
        # Should have attempted rollback
        self.assertGreaterEqual(mock_rename.call_count, 3)


class TestPreviewVoicePromptDeep(unittest.TestCase):
    """Deeper tests for preview_voice_prompt."""

    @patch("qwen3_tts.interface.generate_interactive.voice_prompt_exists", return_value=True)
    @patch("qwen3_tts.interface.generate_interactive.is_server_running", return_value=True)
    @patch("qwen3_tts.interface.generate_interactive.get_server_url", return_value="http://127.0.0.1:5123")
    @patch("qwen3_tts.interface.generate_interactive.auth_headers", return_value={})
    @patch("qwen3_tts.interface.generate_interactive._save_base64_result")
    @patch("qwen3_tts.interface.generate_interactive.play_audio")
    @patch("os.remove")
    @patch("builtins.print")
    def test_server_success(self, _print, _remove, mock_play, mock_save, _auth, _url, _running, _exists):
        from qwen3_tts.interface.generate_interactive import preview_voice_prompt
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": ["base64audio"]}
        with patch("requests.post", return_value=mock_resp), \
             patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_tmp.return_value.name = "/tmp/preview.wav"
            result = preview_voice_prompt("voice.pt", {})
        self.assertTrue(result)
        mock_play.assert_called_once()

    @patch("qwen3_tts.interface.generate_interactive.voice_prompt_exists", return_value=True)
    @patch("qwen3_tts.interface.generate_interactive.is_server_running", return_value=True)
    @patch("qwen3_tts.interface.generate_interactive.get_server_url", return_value="http://127.0.0.1:5123")
    @patch("qwen3_tts.interface.generate_interactive.auth_headers", return_value={})
    @patch("builtins.print")
    def test_server_error(self, mock_print, _auth, _url, _running, _exists):
        from qwen3_tts.interface.generate_interactive import preview_voice_prompt
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "Internal error"}
        with patch("requests.post", return_value=mock_resp):
            result = preview_voice_prompt("voice", {})
        self.assertFalse(result)

    def test_auto_appends_pt_extension(self):
        """preview_voice_prompt appends .pt if missing."""
        from qwen3_tts.interface.generate_interactive import preview_voice_prompt
        with patch("qwen3_tts.interface.generate_interactive.voice_prompt_exists", return_value=False) as mock_exists, \
             patch("builtins.print"):
            preview_voice_prompt("myvoice", {})
        mock_exists.assert_called_with("myvoice.pt")


class TestProgressPollerRun(unittest.TestCase):
    """Tests for _ProgressPoller _run dispatch and fallback loop."""

    def test_run_dispatches_to_fallback_without_rich(self):
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller("http://localhost:5123")
        poller._stop.set()  # Stop immediately

        with patch.object(type(poller), "HAS_RICH", False):
            with patch.object(poller, "_run_fallback") as mock_fallback:
                poller._run()
            mock_fallback.assert_called_once()

    def test_run_dispatches_to_rich_when_available(self):
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller("http://localhost:5123")
        poller._stop.set()

        with patch.object(type(poller), "HAS_RICH", True):
            with patch.object(poller, "_run_rich") as mock_rich:
                poller._run()
            mock_rich.assert_called_once()

    def test_fallback_handles_request_error(self):
        """_run_fallback gracefully handles request exceptions."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller("http://localhost:5123")

        call_count = [0]
        orig_wait = poller._stop.wait

        def stop_after_one(timeout):
            call_count[0] += 1
            if call_count[0] >= 1:
                poller._stop.set()
            return orig_wait(0)

        poller._stop.wait = stop_after_one
        with patch("requests.get", side_effect=ConnectionError("refused")):
            poller._run_fallback()
        # Should complete without exception


class TestRunRepl(unittest.TestCase):
    """Tests for run_repl REPL commands."""

    def _run_repl_with_inputs(self, inputs, use_server=True):
        """Helper to run REPL with a sequence of inputs."""
        from qwen3_tts.interface.generate_interactive import run_repl
        input_iter = iter(inputs)
        with patch("builtins.input", side_effect=input_iter), \
             patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"), \
             patch("soundfile.write"), \
             patch("qwen3_tts.interface.generate_server.generate_via_server", return_value=["base64"]), \
             patch("qwen3_tts.interface.generate_server.generate_local", return_value=(MagicMock(), 24000)), \
             patch("qwen3_tts.interface.generate_interactive._decode_base64_result", return_value=(MagicMock(), 24000)), \
             patch("qwen3_tts.interface.generate_interactive.play_audio"):
            run_repl({}, use_server)

    def test_quit_command(self):
        self._run_repl_with_inputs(["/quit"])

    def test_q_shortcut(self):
        self._run_repl_with_inputs(["/q"])

    def test_status_command(self):
        self._run_repl_with_inputs(["/status", "/quit"])

    def test_play_on_off(self):
        self._run_repl_with_inputs(["/play off", "/play on", "/quit"])

    def test_speed_command(self):
        self._run_repl_with_inputs(["/speed 1.5", "/speed", "/quit"])

    def test_pitch_command(self):
        self._run_repl_with_inputs(["/pitch 2.0", "/pitch", "/quit"])

    def test_speed_invalid(self):
        self._run_repl_with_inputs(["/speed abc", "/quit"])

    def test_pitch_invalid(self):
        self._run_repl_with_inputs(["/pitch xyz", "/quit"])

    @patch("qwen3_tts.interface.generate_interactive.get_voice_alias")
    def test_voice_command_found(self, mock_alias):
        mock_alias.return_value = {"prompt": "narrator.pt"}
        self._run_repl_with_inputs(["/voice narrator", "/quit"])

    @patch("qwen3_tts.interface.generate_interactive.get_voice_alias")
    def test_voice_command_not_found(self, mock_alias):
        mock_alias.return_value = None
        self._run_repl_with_inputs(["/voice nonexistent", "/quit"])

    def test_voice_no_arg(self):
        self._run_repl_with_inputs(["/voice", "/quit"])

    def test_preset_command(self):
        from qwen3_tts.interface.generate_interactive import run_repl
        inputs = iter(["/preset fast", "/quit"])
        config = {"presets": {"fast": {"temperature": 0.3}}, "generation": {}}
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"):
            run_repl(config, True)

    def test_preset_not_found(self):
        self._run_repl_with_inputs(["/preset nonexistent", "/quit"])

    def test_preset_no_arg(self):
        self._run_repl_with_inputs(["/preset", "/quit"])

    def test_unknown_command(self):
        self._run_repl_with_inputs(["/badcommand", "/quit"])

    def test_empty_input_skipped(self):
        self._run_repl_with_inputs(["", "/quit"])

    def test_eof_exits(self):
        from qwen3_tts.interface.generate_interactive import run_repl
        with patch("builtins.input", side_effect=EOFError), \
             patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"):
            run_repl({}, True)

    def test_prompt_command_found(self):
        from qwen3_tts.interface.generate_interactive import run_repl
        inputs = iter(["/prompt myvoice", "/quit"])
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print"), \
             patch("os.path.exists", return_value=True), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"):
            run_repl({}, True)

    def test_prompt_command_not_found(self):
        from qwen3_tts.interface.generate_interactive import run_repl
        inputs = iter(["/prompt nonexistent", "/quit"])
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print"), \
             patch("os.path.exists", return_value=False), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"):
            run_repl({}, True)

    def test_prompt_no_arg(self):
        self._run_repl_with_inputs(["/prompt", "/quit"])

    def test_text_generation_server(self):
        """Typing text generates audio via server."""
        self._run_repl_with_inputs(["Hello world", "/quit"], use_server=True)

    def test_text_generation_local(self):
        """Typing text generates audio locally."""
        self._run_repl_with_inputs(["Hello world", "/quit"], use_server=False)

    def test_generation_error_handled(self):
        """Generation errors are caught and printed."""
        from qwen3_tts.interface.generate_interactive import run_repl
        inputs = iter(["Hello", "/quit"])
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"), \
             patch("qwen3_tts.interface.generate_server.generate_via_server", side_effect=RuntimeError("gen error")):
            run_repl({}, True)


class TestInteractiveMode(unittest.TestCase):
    """Tests for interactive_mode."""

    def test_web_ui_choice(self):
        """Selecting option 2 launches the web UI."""
        from qwen3_tts.interface.generate_interactive import interactive_mode
        with patch("builtins.input", return_value="2"), \
             patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_server.launch_gradio_ui") as mock_ui:
            result = interactive_mode(True, {}, {})
        self.assertIsNone(result)
        mock_ui.assert_called_once()

    def test_cli_mode_design(self):
        """Selecting option 1 -> option 1 (design mode) generates audio."""
        from qwen3_tts.interface.generate_interactive import interactive_mode
        inputs = iter(["1", "Hello world", "1", "Y", ""])
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_server.generate_via_server", return_value=["b64data"]), \
             patch("qwen3_tts.interface.generate_interactive._save_base64_result"), \
             patch("qwen3_tts.interface.generate_interactive.open_file"):
            result = interactive_mode(True, {"default_voice_description": "warm voice"}, {})
        self.assertIsNotNone(result)

    def test_cli_mode_empty_text_exits(self):
        """Empty text input causes sys.exit."""
        from qwen3_tts.interface.generate_interactive import interactive_mode
        inputs = iter(["1", ""])
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print"), \
             self.assertRaises(SystemExit):
            interactive_mode(True, {}, {})


if __name__ == "__main__":
    unittest.main()
