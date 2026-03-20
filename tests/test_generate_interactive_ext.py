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


class TestRenameExtensionStripping(unittest.TestCase):
    """Cover lines 86, 88: extension stripping in rename_voice_prompt."""

    @patch("os.path.exists")
    @patch("os.path.join", side_effect=lambda d, f: f"/voice_prompts/{f}")
    @patch("os.rename")
    @patch("builtins.print")
    def test_rename_strips_pt_extension(self, _print, mock_rename, _join, mock_exists):
        from qwen3_tts.interface.generate_interactive import rename_voice_prompt
        # Pass names WITH .pt extension — lines 86, 88 strip them
        mock_exists.side_effect = lambda p: "old_voice" in p and p.endswith(".pt")
        result = rename_voice_prompt("old_voice.pt", "new_voice.pt")
        self.assertTrue(result)
        mock_rename.assert_called_once()

    @patch("os.path.exists")
    @patch("os.path.join", side_effect=lambda d, f: f"/voice_prompts/{f}")
    @patch("os.rename")
    @patch("builtins.print")
    def test_rename_strips_wav_extension(self, _print, mock_rename, _join, mock_exists):
        from qwen3_tts.interface.generate_interactive import rename_voice_prompt
        mock_exists.side_effect = lambda p: "old_voice" in p and p.endswith(".wav")
        result = rename_voice_prompt("old_voice.wav", "new_voice.wav")
        self.assertTrue(result)

    @patch("os.path.exists")
    @patch("os.path.join", side_effect=lambda d, f: f"/voice_prompts/{f}")
    @patch("os.rename")
    @patch("builtins.print")
    def test_rollback_also_fails(self, _print, mock_rename, _join, mock_exists):
        """Lines 115-116: rollback rename also raises OSError."""
        from qwen3_tts.interface.generate_interactive import rename_voice_prompt
        def exists_side_effect(p):
            return ("old_voice" in p) and (p.endswith(".pt") or p.endswith(".wav"))
        mock_exists.side_effect = exists_side_effect
        # First rename ok, second raises, rollback also raises
        mock_rename.side_effect = [None, OSError("disk full"), OSError("still full")]
        result = rename_voice_prompt("old_voice", "new_voice")
        self.assertFalse(result)


class TestPreviewJsonDecodeError(unittest.TestCase):
    """Cover lines 157-158: JSONDecodeError in preview_voice_prompt."""

    @patch("qwen3_tts.interface.generate_interactive.voice_prompt_exists", return_value=True)
    @patch("qwen3_tts.interface.generate_interactive.is_server_running", return_value=True)
    @patch("qwen3_tts.interface.generate_interactive.get_server_url", return_value="http://127.0.0.1:5123")
    @patch("qwen3_tts.interface.generate_interactive.auth_headers", return_value={})
    @patch("builtins.print")
    def test_json_decode_error_on_error_response(self, mock_print, _auth, _url, _running, _exists):
        from qwen3_tts.interface.generate_interactive import preview_voice_prompt
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.json.side_effect = ValueError("No JSON")
        with patch("requests.post", return_value=mock_resp):
            result = preview_voice_prompt("voice", {})
        self.assertFalse(result)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("502", output)


try:
    import watchdog  # noqa: F401
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False


class TestProgressPollerRichPath(unittest.TestCase):
    """Cover lines 204, 219-267: Rich progress bar path."""

    def test_stop_with_rich_progress(self):
        """Line 204: stop() calls _rich_progress.stop() when available."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller("http://localhost:5123")
        poller._thread = MagicMock()
        poller._rich_progress = MagicMock()
        with patch.object(type(poller), "HAS_RICH", True):
            poller.stop()
        poller._rich_progress.stop.assert_called_once()

    def _make_poller_one_iter(self, batch_total=1):
        """Helper: create a poller that stops after one iteration."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller("http://localhost:5123", batch_total=batch_total)
        call_count = [0]
        orig_wait = poller._stop.wait

        def stop_after_one(timeout):
            call_count[0] += 1
            if call_count[0] >= 1:
                poller._stop.set()
            return orig_wait(0)

        poller._stop.wait = stop_after_one
        return poller

    def test_run_rich_with_active_generation(self):
        """Lines 219-267: Rich progress loop with active generation state."""
        poller = self._make_poller_one_iter()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "active": True, "elapsed_sec": 5, "eta_sec": 10,
            "chunk_total": 2, "chunk_index": 0,
        }
        # Rich is imported lazily inside _run_rich — let it use real Rich
        with patch("requests.get", return_value=mock_resp):
            poller._run_rich()

    def test_run_rich_batch_mode_with_eta(self):
        """Lines 257-263: Rich batch mode with ETA calculation."""
        poller = self._make_poller_one_iter(batch_total=3)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "active": True, "elapsed_sec": 5, "eta_sec": 10,
            "batch_index": 1, "chunk_total": 1,
        }
        with patch("requests.get", return_value=mock_resp):
            poller._run_rich()

    def test_run_rich_request_error(self):
        """Lines 264-265: exception handling in Rich loop."""
        poller = self._make_poller_one_iter()
        with patch("requests.get", side_effect=ConnectionError("refused")):
            poller._run_rich()


class TestProgressPollerFallbackDisplay(unittest.TestCase):
    """Cover lines 276-307: fallback progress display with active generation."""

    def test_fallback_active_no_batch_with_eta(self):
        """Lines 300-302: single generation with ETA."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller("http://localhost:5123", batch_total=1)

        call_count = [0]
        orig_wait = poller._stop.wait

        def stop_after_one(timeout):
            call_count[0] += 1
            if call_count[0] >= 1:
                poller._stop.set()
            return orig_wait(0)

        poller._stop.wait = stop_after_one

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "active": True, "elapsed_sec": 5, "eta_sec": 10,
            "chunk_total": 1,
        }

        with patch("requests.get", return_value=mock_resp), \
             patch("sys.stderr") as mock_stderr:
            poller._run_fallback()
        mock_stderr.write.assert_called()

    def test_fallback_active_no_batch_no_eta(self):
        """Lines 303-304: single generation without ETA."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller("http://localhost:5123", batch_total=1)

        call_count = [0]
        orig_wait = poller._stop.wait

        def stop_after_one(timeout):
            call_count[0] += 1
            if call_count[0] >= 1:
                poller._stop.set()
            return orig_wait(0)

        poller._stop.wait = stop_after_one

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "active": True, "elapsed_sec": 5, "eta_sec": None,
            "chunk_total": 1,
        }

        with patch("requests.get", return_value=mock_resp), \
             patch("sys.stderr") as mock_stderr:
            poller._run_fallback()
        mock_stderr.write.assert_called()

    def test_fallback_batch_mode_with_eta(self):
        """Lines 290-297: batch mode with ETA and progress bar."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller("http://localhost:5123", batch_total=3)

        call_count = [0]
        orig_wait = poller._stop.wait

        def stop_after_one(timeout):
            call_count[0] += 1
            if call_count[0] >= 1:
                poller._stop.set()
            return orig_wait(0)

        poller._stop.wait = stop_after_one

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "active": True, "elapsed_sec": 5, "eta_sec": 10,
            "batch_index": 1, "chunk_total": 2, "chunk_index": 0,
        }

        with patch("requests.get", return_value=mock_resp), \
             patch("sys.stderr") as mock_stderr:
            poller._run_fallback()
        mock_stderr.write.assert_called()

    def test_fallback_batch_mode_no_eta(self):
        """Lines 298-299: batch mode without ETA."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller("http://localhost:5123", batch_total=3)

        call_count = [0]
        orig_wait = poller._stop.wait

        def stop_after_one(timeout):
            call_count[0] += 1
            if call_count[0] >= 1:
                poller._stop.set()
            return orig_wait(0)

        poller._stop.wait = stop_after_one

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "active": True, "elapsed_sec": 5, "eta_sec": None,
            "batch_index": 0, "chunk_total": 1,
        }

        with patch("requests.get", return_value=mock_resp), \
             patch("sys.stderr") as mock_stderr:
            poller._run_fallback()
        mock_stderr.write.assert_called()


class TestInteractiveModeClone(unittest.TestCase):
    """Cover lines 353-371, 378, 384, 395, 400-405: clone mode + local gen."""

    def test_clone_mode_server(self):
        """Lines 353-371, 395: clone mode via server."""
        from qwen3_tts.interface.generate_interactive import interactive_mode
        # 1=CLI, text, 2=clone, select prompt 1, output name without .wav
        inputs = iter(["1", "Hello world", "2", "1", "output"])
        config = {"default_clone_prompt": "voice1.pt"}
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_interactive.list_voice_prompts", return_value=["voice1.pt"]), \
             patch("qwen3_tts.interface.generate_helpers.get_text", return_value="Hello world"), \
             patch("qwen3_tts.interface.generate_server.generate_via_server", return_value=["b64"]), \
             patch("qwen3_tts.interface.generate_interactive._save_base64_result"), \
             patch("qwen3_tts.interface.generate_interactive.open_file"):
            result = interactive_mode(True, config, {})
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith(".wav"))

    def test_clone_mode_invalid_selection(self):
        """Lines 366-368: invalid prompt selection uses default."""
        from qwen3_tts.interface.generate_interactive import interactive_mode
        inputs = iter(["1", "Hello", "2", "invalid", "out.wav"])
        config = {"default_clone_prompt": "voice1.pt"}
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_interactive.list_voice_prompts", return_value=["voice1.pt"]), \
             patch("qwen3_tts.interface.generate_helpers.get_text", return_value="Hello"), \
             patch("qwen3_tts.interface.generate_server.generate_via_server", return_value=["b64"]), \
             patch("qwen3_tts.interface.generate_interactive._save_base64_result"), \
             patch("qwen3_tts.interface.generate_interactive.open_file"):
            result = interactive_mode(True, config, {})
        self.assertIsNotNone(result)

    def test_design_mode_custom_description(self):
        """Line 378: user enters custom voice description."""
        from qwen3_tts.interface.generate_interactive import interactive_mode
        # 1=CLI, text, 1=design, n=custom desc, enter desc, output
        inputs = iter(["1", "Hello", "1", "n", "bright voice", "out.wav"])
        config = {"default_voice_description": "warm voice"}
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_helpers.get_text", return_value="Hello"), \
             patch("qwen3_tts.interface.generate_server.generate_via_server", return_value=["b64"]), \
             patch("qwen3_tts.interface.generate_interactive._save_base64_result"), \
             patch("qwen3_tts.interface.generate_interactive.open_file"):
            result = interactive_mode(True, config, {})
        self.assertIsNotNone(result)

    def test_local_generation(self):
        """Lines 400-405: local generation path (use_server=False)."""
        from qwen3_tts.interface.generate_interactive import interactive_mode
        import numpy as np
        inputs = iter(["1", "Hello", "1", "Y", "out.wav"])
        config = {"default_voice_description": "warm", "language": "English"}
        mock_wav = np.zeros(1000, dtype=np.float32)
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_helpers.get_text", return_value="Hello"), \
             patch("qwen3_tts.interface.generate_server.generate_local", return_value=(mock_wav, 24000)), \
             patch("soundfile.write"), \
             patch("qwen3_tts.interface.generate_interactive.open_file"):
            result = interactive_mode(False, config, {})
        self.assertIsNotNone(result)

    def test_clone_no_prompts_exits(self):
        """Lines 354-356: no voice prompts available."""
        from qwen3_tts.interface.generate_interactive import interactive_mode
        inputs = iter(["1", "Hello", "2"])
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_interactive.list_voice_prompts", return_value=[]), \
             patch("qwen3_tts.interface.generate_helpers.get_text", return_value="Hello"), \
             self.assertRaises(SystemExit):
            interactive_mode(True, {}, {})


class TestReplVoiceAliasWithPreset(unittest.TestCase):
    """Cover lines 472-474, 507: voice alias with preset, /play bad arg."""

    def test_voice_alias_with_preset(self):
        """Lines 472-474: voice alias includes preset key."""
        from qwen3_tts.interface.generate_interactive import run_repl
        inputs = iter(["/voice narrator", "/quit"])
        config = {"presets": {"warm": {"temperature": 0.8}}, "generation": {}}
        alias = {"prompt": "narrator.pt", "preset": "warm"}
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print") as mock_print, \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"), \
             patch("qwen3_tts.interface.generate_interactive.get_voice_alias", return_value=alias):
            run_repl(config, True)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("narrator", output)

    def test_play_invalid_arg(self):
        """Line 507: /play with bad argument."""
        from qwen3_tts.interface.generate_interactive import run_repl
        inputs = iter(["/play maybe", "/quit"])
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print") as mock_print, \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"):
            run_repl({}, True)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("/play on|off", output)


class TestReplAudioProcessing(unittest.TestCase):
    """Cover lines 561-562, 564-565: speed/pitch adjustment in REPL gen."""

    def test_generation_with_speed_and_pitch(self):
        """Lines 561-565: speed and pitch adjustments applied during REPL gen."""
        import numpy as np
        from qwen3_tts.interface.generate_interactive import run_repl
        inputs = iter(["/speed 1.5", "/pitch 2.0", "Hello world", "/quit"])
        mock_wav = np.zeros(1000, dtype=np.float32)
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"), \
             patch("qwen3_tts.interface.generate_server.generate_local", return_value=(mock_wav, 24000)), \
             patch("soundfile.write"), \
             patch("qwen3_tts.interface.generate_interactive._decode_base64_result", return_value=(mock_wav, 24000)), \
             patch("qwen3_tts.interface.generate_interactive.play_audio"), \
             patch("qwen3_tts.core.engine.adjust_speed", return_value=mock_wav) as mock_speed, \
             patch("qwen3_tts.core.engine.adjust_pitch", return_value=mock_wav) as mock_pitch:
            run_repl({}, False)
        mock_speed.assert_called_once()
        mock_pitch.assert_called_once()


@unittest.skipUnless(HAS_WATCHDOG, "requires watchdog")
class TestRunWatchMode(unittest.TestCase):
    """Cover lines 585-670: run_watch_mode."""

    def test_watch_dir_not_found(self):
        """Lines 590-592: non-existent watch directory."""
        from qwen3_tts.interface.generate_interactive import run_watch_mode
        args = MagicMock()
        with patch("builtins.print") as mock_print:
            run_watch_mode("/nonexistent_xyz", {}, args, {}, True)
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("not found", output)

    def test_watch_mode_starts_and_stops(self):
        """Lines 652-670: observer starts and Ctrl+C stops it."""
        import tempfile
        import shutil
        from qwen3_tts.interface.generate_interactive import run_watch_mode
        tmp_dir = tempfile.mkdtemp()
        args = MagicMock()
        args.output = tmp_dir
        args.mode = "clone"
        args.prompt = "default.pt"
        args.description = "warm"
        args.play = False

        mock_observer = MagicMock()
        # Simulate KeyboardInterrupt during sleep loop
        with patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"), \
             patch("watchdog.observers.Observer", return_value=mock_observer), \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            try:
                run_watch_mode(tmp_dir, {}, args, {}, True)
            except KeyboardInterrupt:
                pass
        mock_observer.start.assert_called_once()
        mock_observer.stop.assert_called_once()
        shutil.rmtree(tmp_dir)

    def test_watch_handler_processes_txt_file(self):
        """Lines 603-650: TTSHandler.on_created processes .txt files."""
        import tempfile
        import shutil
        from qwen3_tts.interface.generate_interactive import run_watch_mode
        tmp_dir = tempfile.mkdtemp()
        args = MagicMock()
        args.output = tmp_dir
        args.mode = "clone"
        args.prompt = "default.pt"
        args.description = "warm"
        args.play = False

        captured_handler = [None]
        mock_observer = MagicMock()

        def capture_schedule(handler, *a, **kw):
            captured_handler[0] = handler

        mock_observer.schedule = capture_schedule

        with patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"), \
             patch("watchdog.observers.Observer", return_value=mock_observer), \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            try:
                run_watch_mode(tmp_dir, {}, args, {}, True)
            except KeyboardInterrupt:
                pass

        # Now test the handler
        handler = captured_handler[0]
        self.assertIsNotNone(handler)

        # Create a fake .txt file
        txt_path = os.path.join(tmp_dir, "test.txt")
        with open(txt_path, "w") as f:
            f.write("Hello world")

        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = txt_path

        import numpy as np
        mock_wav = np.zeros(1000, dtype=np.float32)
        with patch("qwen3_tts.interface.generate_server.generate_via_server", return_value=["b64"]), \
             patch("qwen3_tts.interface.generate_interactive._decode_base64_result", return_value=(mock_wav, 24000)), \
             patch("qwen3_tts.interface.generate_interactive.process_audio_args", return_value=mock_wav), \
             patch("soundfile.write"), \
             patch("time.sleep"):
            handler.on_created(mock_event)

        shutil.rmtree(tmp_dir)

    def test_watch_handler_skips_directories(self):
        """Line 606: TTSHandler skips directory events."""
        import tempfile
        import shutil
        from qwen3_tts.interface.generate_interactive import run_watch_mode
        tmp_dir = tempfile.mkdtemp()
        args = MagicMock()
        args.output = tmp_dir
        args.mode = "clone"
        args.prompt = "default.pt"
        args.description = "warm"
        args.play = False

        captured_handler = [None]
        mock_observer = MagicMock()

        def capture_schedule(handler, *a, **kw):
            captured_handler[0] = handler

        mock_observer.schedule = capture_schedule

        with patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"), \
             patch("watchdog.observers.Observer", return_value=mock_observer), \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            try:
                run_watch_mode(tmp_dir, {}, args, {}, True)
            except KeyboardInterrupt:
                pass

        handler = captured_handler[0]
        mock_event = MagicMock()
        mock_event.is_directory = True
        # Should return without processing
        handler.on_created(mock_event)
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
