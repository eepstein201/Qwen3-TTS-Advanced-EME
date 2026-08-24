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
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch


@dataclass(frozen=True)
class ReplRun:
    """Observable outcome of one REPL session.

    Frozen: the captured transcript is evidence, not scratch space.
    """

    lines: tuple[str, ...]
    write: Any
    server: Any
    local: Any
    play: Any

    @property
    def output(self) -> str:
        """Full transcript as one searchable string."""
        return "\n".join(self.lines)


class TestDeleteVoicePromptDeep(unittest.TestCase):
    """Deeper tests for delete_voice_prompt."""

    @patch("os.path.exists", side_effect=lambda p: p.endswith(".pt"))
    @patch("qwen3_tts.interface.generate_interactive.safe_path_join", side_effect=lambda d, f: f"/voice_prompts/{f}")
    @patch("os.remove")
    @patch("builtins.input", return_value="y")
    @patch("builtins.print")
    def test_confirmed_deletion(self, _print, _input, mock_remove, _join, _exists):
        from qwen3_tts.interface.generate_interactive import delete_voice_prompt
        result = delete_voice_prompt("my_voice")
        self.assertTrue(result)
        mock_remove.assert_called_once()

    @patch("os.path.exists", side_effect=lambda p: p.endswith(".pt"))
    @patch("qwen3_tts.interface.generate_interactive.safe_path_join", side_effect=lambda d, f: f"/voice_prompts/{f}")
    @patch("builtins.input", return_value="n")
    @patch("builtins.print")
    def test_cancelled_deletion(self, _print, _input, _join, _exists):
        from qwen3_tts.interface.generate_interactive import delete_voice_prompt
        result = delete_voice_prompt("my_voice")
        self.assertFalse(result)

    @patch("os.path.exists", side_effect=lambda p: p.endswith(".wav") or p.endswith(".txt"))
    @patch("qwen3_tts.interface.generate_interactive.safe_path_join", side_effect=lambda d, f: f"/voice_prompts/{f}")
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
    @patch("qwen3_tts.interface.generate_interactive.safe_path_join", side_effect=lambda d, f: f"/voice_prompts/{f}")
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
    @patch("qwen3_tts.interface.generate_interactive.safe_path_join", side_effect=lambda d, f: f"/voice_prompts/{f}")
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
    @patch("qwen3_tts.interface.generate_interactive.safe_path_join", side_effect=lambda d, f: f"/voice_prompts/{f}")
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
    @patch("qwen3_tts.interface.generate_interactive._save_base64_result")
    @patch("qwen3_tts.interface.generate_interactive.play_audio")
    @patch("os.remove")
    @patch("builtins.print")
    def test_server_success(self, _print, _remove, mock_play, mock_save, _running, _exists):
        import tempfile as real_tempfile

        from qwen3_tts.interface.generate_interactive import preview_voice_prompt

        # Create a real temp file for testing
        with real_tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as real_tmp:
            real_tmp_name = real_tmp.name

        try:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"results": ["base64audio"]}

            # Use a real file but mock the tempfile creation
            with patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
                 patch("tempfile.NamedTemporaryFile", return_value=real_tmp):
                result = preview_voice_prompt("voice.pt", {})

            self.assertTrue(result)
            mock_play.assert_called_once()
        finally:
            # Clean up the real temp file
            try:
                os.remove(real_tmp_name)
            except OSError:
                pass

    @patch("qwen3_tts.interface.generate_interactive.voice_prompt_exists", return_value=True)
    @patch("qwen3_tts.interface.generate_interactive.is_server_running", return_value=True)
    @patch("builtins.print")
    def test_server_error(self, mock_print, _running, _exists):
        from qwen3_tts.interface.generate_interactive import preview_voice_prompt
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "Internal error"}
        with patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
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
        poller = _ProgressPoller()
        poller._stop.set()  # Stop immediately

        with patch.object(type(poller), "HAS_RICH", False):
            with patch.object(poller, "_run_fallback") as mock_fallback:
                poller._run()
            mock_fallback.assert_called_once()

    def test_run_dispatches_to_rich_when_available(self):
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller()
        poller._stop.set()

        with patch.object(type(poller), "HAS_RICH", True):
            with patch.object(poller, "_run_rich") as mock_rich:
                poller._run()
            mock_rich.assert_called_once()

    def test_fallback_handles_request_error(self):
        """_run_fallback gracefully handles request exceptions."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller()

        call_count = [0]
        orig_wait = poller._stop.wait

        def stop_after_one(timeout):
            call_count[0] += 1
            if call_count[0] >= 1:
                poller._stop.set()
            return orig_wait(0)

        poller._stop.wait = stop_after_one
        with patch(
            "qwen3_tts.core.http_client.server_request",
            side_effect=ConnectionError("refused"),
        ) as mock_req:
            poller._run_fallback()

        # The point of the test: a refused connection is swallowed by the poll
        # loop rather than escaping the thread, and the loop still terminates.
        self.assertTrue(
            mock_req.called, "fallback loop never polled the server."
        )
        self.assertTrue(
            poller._stop.is_set(),
            "fallback loop did not terminate after the stop signal.",
        )
        # Should complete without exception


class TestRunRepl(unittest.TestCase):
    """Tests for run_repl REPL commands."""

    def _run_repl_with_inputs(self, inputs, use_server=True):
        """Run the REPL over *inputs* and return an observable result.

        Returns a ``ReplRun`` carrying the captured stdout lines plus the
        generation/playback mocks, so callers can assert on what the REPL
        actually did. These tests previously discarded ``print`` entirely and
        asserted nothing — they passed as long as no exception escaped, which
        made them crash-only smoke tests rather than behavioral coverage.
        """
        from qwen3_tts.interface.generate_interactive import run_repl

        printed: list[str] = []
        input_iter = iter(inputs)
        with patch("builtins.input", side_effect=input_iter), \
             patch("builtins.print", side_effect=lambda *a, **k: printed.append(" ".join(str(x) for x in a))), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"), \
             patch("soundfile.write") as mock_write, \
             patch("qwen3_tts.interface.generate_server.generate_via_server", return_value=["base64"]) as mock_server, \
             patch("qwen3_tts.interface.generate_server.generate_local", return_value=(MagicMock(), 24000)) as mock_local, \
             patch("qwen3_tts.interface.generate_interactive._decode_base64_result", return_value=(MagicMock(), 24000)), \
             patch("qwen3_tts.interface.generate_interactive.play_audio") as mock_play:
            run_repl({}, use_server)
        return ReplRun(
            lines=tuple(printed),
            write=mock_write,
            server=mock_server,
            local=mock_local,
            play=mock_play,
        )

    def assertReplExitedCleanly(self, run):
        """The REPL printed its banner, reached its exit line, and raised no
        'Unknown command' complaint. Baseline for every command test."""
        self.assertIn(
            "=== TTS REPL Mode ===",
            run.output,
            "REPL never printed its banner — it did not start.",
        )
        self.assertIn(
            "Exiting REPL.",
            run.output,
            "REPL did not reach its exit line; the input loop broke early.",
        )
        self.assertNotIn(
            "Unknown command",
            run.output,
            f"REPL rejected a valid command: {run.output!r}",
        )

    def test_repl_honors_cli_sampling_flags(self):
        """H5: run_repl threads the caller's gen_params (CLI --seed/--temperature)
        through to generation instead of rebuilding from raw config (which
        silently dropped every CLI override)."""
        from qwen3_tts.interface.generate_interactive import run_repl

        captured = {}

        def fake_generate(texts, mode, config, gen_params, **kwargs):
            captured["gen_params"] = dict(gen_params)
            return ["base64"]

        inputs = iter(["hello world", "/quit"])
        cli_params = {
            "temperature": 0.5,
            "seed": 42,
            "max_new_tokens": 2048,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
        }
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print"), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"), \
             patch("soundfile.write"), \
             patch("qwen3_tts.interface.generate_server.generate_via_server", side_effect=fake_generate), \
             patch("qwen3_tts.interface.generate_interactive._decode_base64_result", return_value=(MagicMock(), 24000)), \
             patch("qwen3_tts.interface.generate_interactive.play_audio"):
            run_repl({}, True, gen_params=cli_params)

        self.assertIn(
            "gen_params", captured, "REPL never invoked generation for text input."
        )
        self.assertEqual(
            captured["gen_params"].get("seed"),
            42,
            "CLI --seed was dropped before reaching generation.",
        )
        self.assertEqual(
            captured["gen_params"].get("temperature"),
            0.5,
            "CLI --temperature was dropped before reaching generation.",
        )

    def test_quit_command(self):
        """/quit exits the loop without consuming further input."""
        run = self._run_repl_with_inputs(["/quit"])
        self.assertReplExitedCleanly(run)
        self.assertFalse(
            run.server.called, "/quit must not trigger a generation."
        )

    def test_q_shortcut(self):
        """/q is an alias for /quit."""
        run = self._run_repl_with_inputs(["/q"])
        self.assertReplExitedCleanly(run)
        self.assertFalse(run.server.called, "/q must not trigger a generation.")

    def test_status_command(self):
        """/status reports every live setting."""
        run = self._run_repl_with_inputs(["/status", "/quit"])
        self.assertReplExitedCleanly(run)
        for field in ("Mode:", "Prompt:", "Preset:", "Auto-play:", "Speed:", "Pitch:", "Server:"):
            self.assertIn(field, run.output, f"/status omitted {field!r}")
        self.assertIn("Mode: clone", run.output)
        self.assertIn("Prompt: default.pt", run.output)

    def test_play_on_off(self):
        """/play toggles auto-play and echoes the new state each time."""
        run = self._run_repl_with_inputs(["/play off", "/play on", "/quit"])
        self.assertReplExitedCleanly(run)
        self.assertIn("Auto-play: off", run.output)
        self.assertIn("Auto-play: on", run.output)
        self.assertLess(
            run.output.index("Auto-play: off"),
            run.output.index("Auto-play: on"),
            "Auto-play states echoed out of order.",
        )

    def test_speed_command(self):
        """/speed FACTOR sets speed; bare /speed resets it."""
        run = self._run_repl_with_inputs(["/speed 1.5", "/speed", "/quit"])
        self.assertReplExitedCleanly(run)
        self.assertIn("Speed: 1.5", run.output)
        self.assertIn("Speed: reset to default", run.output)

    def test_pitch_command(self):
        """/pitch SEMI sets pitch; bare /pitch resets it."""
        run = self._run_repl_with_inputs(["/pitch 2.0", "/pitch", "/quit"])
        self.assertReplExitedCleanly(run)
        self.assertIn("Pitch: 2.0 semitones", run.output)
        self.assertIn("Pitch: reset to default", run.output)

    def test_speed_invalid(self):
        """A non-numeric /speed is reported, not crashed on."""
        run = self._run_repl_with_inputs(["/speed abc", "/quit"])
        self.assertReplExitedCleanly(run)
        self.assertIn("Invalid speed value", run.output)

    def test_pitch_invalid(self):
        """A non-numeric /pitch is reported, not crashed on."""
        run = self._run_repl_with_inputs(["/pitch xyz", "/quit"])
        self.assertReplExitedCleanly(run)
        self.assertIn("Invalid pitch value", run.output)

    @patch("qwen3_tts.interface.generate_interactive.get_voice_alias")
    def test_voice_command_found(self, mock_alias):
        """/voice NAME resolves the alias and adopts its prompt."""
        mock_alias.return_value = {"prompt": "narrator.pt"}
        run = self._run_repl_with_inputs(["/voice narrator", "/status", "/quit"])
        self.assertReplExitedCleanly(run)
        mock_alias.assert_called_with("narrator", {})
        self.assertIn(
            "Prompt: narrator.pt",
            run.output,
            f"/voice did not adopt the alias prompt: {run.output!r}",
        )

    @patch("qwen3_tts.interface.generate_interactive.get_voice_alias")
    def test_voice_command_not_found(self, mock_alias):
        """An unknown alias leaves the active prompt untouched."""
        mock_alias.return_value = None
        run = self._run_repl_with_inputs(["/voice nonexistent", "/status", "/quit"])
        self.assertReplExitedCleanly(run)
        mock_alias.assert_called_with("nonexistent", {})
        self.assertIn(
            "Prompt: default.pt",
            run.output,
            "A failed /voice lookup must not change the active prompt.",
        )

    def test_voice_no_arg(self):
        """Bare /voice is a no-op, not an error or a crash."""
        run = self._run_repl_with_inputs(["/voice", "/status", "/quit"])
        self.assertReplExitedCleanly(run)
        self.assertIn(
            "Prompt: default.pt",
            run.output,
            "Bare /voice must not clear the active prompt.",
        )

    def test_preset_command(self):
        """/preset NAME adopts a preset defined in config."""
        from qwen3_tts.interface.generate_interactive import run_repl
        printed = []
        inputs = iter(["/preset fast", "/status", "/quit"])
        config = {"presets": {"fast": {"temperature": 0.3}}, "generation": {}}
        with patch("builtins.input", side_effect=inputs), \
             patch("builtins.print", side_effect=lambda *a, **k: printed.append(" ".join(str(x) for x in a))), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"):
            run_repl(config, True)
        output = "\n".join(printed)
        self.assertIn("Exiting REPL.", output)
        self.assertNotIn(
            "Unknown preset",
            output,
            "A preset present in config was reported unknown.",
        )
        self.assertIn(
            "Preset: fast",
            output,
            f"/preset did not become the active preset: {output!r}",
        )

    def test_preset_not_found(self):
        """An undefined preset is reported by name."""
        run = self._run_repl_with_inputs(["/preset nonexistent", "/quit"])
        self.assertIn("Exiting REPL.", run.output)
        self.assertIn("Unknown preset: nonexistent", run.output)

    def test_preset_no_arg(self):
        """Bare /preset is a no-op, not an error."""
        run = self._run_repl_with_inputs(["/preset", "/status", "/quit"])
        self.assertReplExitedCleanly(run)
        self.assertIn(
            "Preset: default",
            run.output,
            "Bare /preset must leave the active preset alone.",
        )

    def test_unknown_command(self):
        """An unrecognised slash command is named back to the user."""
        run = self._run_repl_with_inputs(["/badcommand", "/quit"])
        self.assertIn("Exiting REPL.", run.output)
        self.assertIn("Unknown command: /badcommand", run.output)
        self.assertFalse(
            run.server.called,
            "An unknown command must not be sent off as text to generate.",
        )

    def test_empty_input_skipped(self):
        """Empty input is skipped without reaching generation."""
        run = self._run_repl_with_inputs(["", "/quit"])
        self.assertReplExitedCleanly(run)
        self.assertFalse(
            run.server.called, "Empty input must not trigger a generation."
        )
        self.assertFalse(
            run.write.called, "Empty input must not write an audio file."
        )

    def test_eof_exits(self):
        """EOF (Ctrl-D) exits the loop instead of propagating."""
        from qwen3_tts.interface.generate_interactive import run_repl
        printed = []
        with patch("builtins.input", side_effect=EOFError), \
             patch("builtins.print", side_effect=lambda *a, **k: printed.append(" ".join(str(x) for x in a))), \
             patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"):
            run_repl({}, True)
        self.assertIn(
            "=== TTS REPL Mode ===",
            "\n".join(printed),
            "REPL did not start before EOF was delivered.",
        )

    def _run_repl_capturing(self, inputs, extra_patches=()):
        """Run the REPL with bespoke patches, returning the printed transcript."""
        from qwen3_tts.interface.generate_interactive import run_repl
        printed = []
        started = [
            patch("builtins.input", side_effect=iter(inputs)),
            patch("builtins.print", side_effect=lambda *a, **k: printed.append(" ".join(str(x) for x in a))),
            patch("qwen3_tts.interface.generate_interactive.get_default_clone_prompt", return_value="default.pt"),
            *extra_patches,
        ]
        for p in started:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(started)])
        run_repl({}, True)
        return "\n".join(printed)

    def test_prompt_command_found(self):
        """/prompt NAME adopts an existing prompt file."""
        output = self._run_repl_capturing(
            ["/prompt myvoice", "/status", "/quit"],
            [patch("os.path.exists", return_value=True)],
        )
        self.assertIn("Exiting REPL.", output)
        self.assertNotIn("not found", output.lower())
        self.assertIn(
            "myvoice",
            output,
            f"/prompt did not adopt the named prompt: {output!r}",
        )

    def test_prompt_command_not_found(self):
        """A missing prompt file is reported and the old prompt is kept."""
        output = self._run_repl_capturing(
            ["/prompt nonexistent", "/status", "/quit"],
            [patch("os.path.exists", return_value=False)],
        )
        self.assertIn("Exiting REPL.", output)
        self.assertIn(
            "Prompt: default.pt",
            output,
            "A missing /prompt target must not replace the active prompt.",
        )

    def test_prompt_no_arg(self):
        """Bare /prompt is a no-op, not an error."""
        run = self._run_repl_with_inputs(["/prompt", "/status", "/quit"])
        self.assertReplExitedCleanly(run)
        self.assertIn(
            "Prompt: default.pt",
            run.output,
            "Bare /prompt must leave the active prompt alone.",
        )

    def test_text_generation_server(self):
        """Typing text routes to the server path and writes the audio out."""
        run = self._run_repl_with_inputs(["Hello world", "/quit"], use_server=True)
        self.assertReplExitedCleanly(run)
        run.server.assert_called_once()
        self.assertFalse(
            run.local.called,
            "Server mode must not fall back to local generation.",
        )
        self.assertIn("Hello world", str(run.server.call_args))
        run.write.assert_called_once()
        self.assertIn(".wav", run.output, "REPL did not report an output path.")

    def test_text_generation_local(self):
        """With no server, typing text routes to local generation."""
        run = self._run_repl_with_inputs(["Hello world", "/quit"], use_server=False)
        self.assertReplExitedCleanly(run)
        run.local.assert_called_once()
        self.assertFalse(
            run.server.called,
            "Local mode must not call the server generation path.",
        )
        run.write.assert_called_once()
        self.assertIn(".wav", run.output, "REPL did not report an output path.")

    def test_generation_error_handled(self):
        """A generation failure is reported and the REPL keeps running."""
        output = self._run_repl_capturing(
            ["Hello", "/quit"],
            [patch(
                "qwen3_tts.interface.generate_server.generate_via_server",
                side_effect=RuntimeError("gen error"),
            )],
        )
        self.assertIn(
            "Exiting REPL.",
            output,
            "A generation error killed the REPL loop instead of being handled.",
        )
        self.assertIn(
            "gen error",
            output,
            f"The generation failure was swallowed silently: {output!r}",
        )


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
    @patch("qwen3_tts.interface.generate_interactive.safe_path_join", side_effect=lambda d, f: f"/voice_prompts/{f}")
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
    @patch("qwen3_tts.interface.generate_interactive.safe_path_join", side_effect=lambda d, f: f"/voice_prompts/{f}")
    @patch("os.rename")
    @patch("builtins.print")
    def test_rename_strips_wav_extension(self, _print, mock_rename, _join, mock_exists):
        from qwen3_tts.interface.generate_interactive import rename_voice_prompt
        mock_exists.side_effect = lambda p: "old_voice" in p and p.endswith(".wav")
        result = rename_voice_prompt("old_voice.wav", "new_voice.wav")
        self.assertTrue(result)

    @patch("os.path.exists")
    @patch("qwen3_tts.interface.generate_interactive.safe_path_join", side_effect=lambda d, f: f"/voice_prompts/{f}")
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
    @patch("builtins.print")
    def test_json_decode_error_on_error_response(self, mock_print, _running, _exists):
        from qwen3_tts.interface.generate_interactive import preview_voice_prompt
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_resp.json.side_effect = ValueError("No JSON")
        with patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
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
        poller = _ProgressPoller()
        poller._thread = MagicMock()
        poller._rich_progress = MagicMock()
        with patch.object(type(poller), "HAS_RICH", True):
            poller.stop()
        poller._rich_progress.stop.assert_called_once()

    def _make_poller_one_iter(self, batch_total=1):
        """Helper: create a poller that stops after one iteration."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller(batch_total=batch_total)
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
        with patch(
            "qwen3_tts.core.http_client.server_request", return_value=mock_resp
        ) as mock_req:
            poller._run_rich()

        self.assertTrue(mock_req.called, "rich loop never polled /generation-status.")
        self.assertTrue(mock_resp.json.called, "rich loop never read the poll payload.")
        self.assertTrue(
            poller._stop.is_set(), "rich loop did not terminate after the stop signal."
        )

    def test_run_rich_batch_mode_with_eta(self):
        """Lines 257-263: Rich batch mode with ETA calculation."""
        poller = self._make_poller_one_iter(batch_total=3)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "active": True, "elapsed_sec": 5, "eta_sec": 10,
            "batch_index": 1, "chunk_total": 1,
        }
        with patch(
            "qwen3_tts.core.http_client.server_request", return_value=mock_resp
        ) as mock_req:
            poller._run_rich()

        self.assertTrue(mock_req.called, "batch rich loop never polled the server.")
        self.assertTrue(mock_resp.json.called, "batch rich loop never read the payload.")
        self.assertTrue(
            poller._stop.is_set(),
            "batch rich loop did not terminate after the stop signal.",
        )

    def test_run_rich_request_error(self):
        """Lines 264-265: exception handling in Rich loop."""
        poller = self._make_poller_one_iter()
        with patch(
            "qwen3_tts.core.http_client.server_request",
            side_effect=ConnectionError("refused"),
        ) as mock_req:
            poller._run_rich()

        # A refused connection must be contained inside the loop, not raised
        # out of the poller thread, and must still let the loop exit.
        self.assertTrue(mock_req.called, "rich loop never attempted a poll.")
        self.assertTrue(
            poller._stop.is_set(),
            "rich loop did not terminate after a connection error.",
        )


class TestProgressPollerFallbackDisplay(unittest.TestCase):
    """Cover lines 276-307: fallback progress display with active generation."""

    def test_fallback_active_no_batch_with_eta(self):
        """Lines 300-302: single generation with ETA."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller(batch_total=1)

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

        with patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
             patch("sys.stderr") as mock_stderr:
            poller._run_fallback()
        mock_stderr.write.assert_called()

    def test_fallback_active_no_batch_no_eta(self):
        """Lines 303-304: single generation without ETA."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller(batch_total=1)

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

        with patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
             patch("sys.stderr") as mock_stderr:
            poller._run_fallback()
        mock_stderr.write.assert_called()

    def test_fallback_batch_mode_with_eta(self):
        """Lines 290-297: batch mode with ETA and progress bar."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller(batch_total=3)

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

        with patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
             patch("sys.stderr") as mock_stderr:
            poller._run_fallback()
        mock_stderr.write.assert_called()

    def test_fallback_batch_mode_no_eta(self):
        """Lines 298-299: batch mode without ETA."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller
        poller = _ProgressPoller(batch_total=3)

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

        with patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
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
        import numpy as np

        from qwen3_tts.interface.generate_interactive import interactive_mode
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




if __name__ == "__main__":
    unittest.main()
