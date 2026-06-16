#!/usr/bin/env python3
"""Extended tests for generate_interactive module — part 2.

Split from test_generate_interactive_ext.py (over 800 lines).
Covers: REPL voice alias with preset, audio processing, watch mode.

Run: pytest tests/test_generate_interactive_ext_part2.py -v
"""
import os
import unittest
from unittest.mock import patch, MagicMock

try:
    import watchdog  # noqa: F401
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False


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
        # run_watch_mode rejects output dirs outside $HOME; keep the temp dir
        # under home so the security check passes (default macOS tmp is not).
        tmp_dir = tempfile.mkdtemp(dir=os.path.expanduser("~"))
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
        # run_watch_mode rejects output dirs outside $HOME; keep the temp dir
        # under home so the security check passes (default macOS tmp is not).
        tmp_dir = tempfile.mkdtemp(dir=os.path.expanduser("~"))
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
        # run_watch_mode rejects output dirs outside $HOME; keep the temp dir
        # under home so the security check passes (default macOS tmp is not).
        tmp_dir = tempfile.mkdtemp(dir=os.path.expanduser("~"))
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
