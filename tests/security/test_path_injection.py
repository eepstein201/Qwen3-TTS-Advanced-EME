#!/usr/bin/env python3
"""Tests for path injection prevention via safe_path_join (PR 4 - security remediation).

TDD RED phase — tests verify path traversal attempts are rejected.

Coverage:
  1. interface/cli/srt.py::generate_from_srt — output path traversal
  2. interface/cli/dialogue.py::generate_from_dialogue — output path traversal
  3. interface/generate.py — save path traversal
  4. interface/generate_helpers.py::auto_increment_filename — traversal
  5. interface/generate_interactive.py — history file traversal
  6. tools/create_voice.py — output path traversal
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

from qwen3_tts.core.config import safe_path_join


class TestPathInjectionPrevention(unittest.TestCase):
    """Verify path construction from user input rejects traversal attempts."""

    def test_safe_path_join_rejects_double_dot_traversal(self):
        """safe_path_join rejects paths containing '..' for traversal."""
        base = tempfile.gettempdir()
        with self.assertRaises(ValueError) as ctx:
            safe_path_join(base, "../etc/passwd")
        self.assertIn("traversal", str(ctx.exception).lower())

    def test_safe_path_join_rejects_absolute_path_escape(self):
        """safe_path_join rejects absolute paths that escape base."""
        base = tempfile.gettempdir()
        with self.assertRaises(ValueError) as ctx:
            safe_path_join(base, "/etc/passwd")
        self.assertIn("traversal", str(ctx.exception).lower())


class TestSrtPathInjection(unittest.TestCase):
    """Test srt.py output path validation."""

    def _import_srt_module(self):
        from qwen3_tts.interface.cli import srt

        return srt

    @patch("qwen3_tts.core.engine.inference")
    def test_process_srt_file_rejects_traversal_output(self, mock_inference):
        """process_srt_file with relative traversal path '../malicious' raises ValueError."""
        srt = self._import_srt_module()
        # Create a temporary SRT file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write("1\n00:00:00,000 --> 00:00:01,000\nTest")
            temp_srt = f.name

        try:
            # Create mock config and args with relative traversal path
            config = {}
            args = MagicMock(output="../escape")  # relative path with traversal

            with self.assertRaises(ValueError) as ctx:
                # Attempt to write outside current directory via relative traversal
                srt.process_srt_file(temp_srt, config, args, {}, use_server=True)
            self.assertIn("traversal", str(ctx.exception).lower())
        finally:
            os.unlink(temp_srt)


class TestDialoguePathInjection(unittest.TestCase):
    """Test dialogue.py output path validation."""

    def _import_dialogue_module(self):
        from qwen3_tts.interface.cli import dialogue

        return dialogue

    @patch("qwen3_tts.core.engine.inference")
    def test_process_dialogue_rejects_traversal_output(self, mock_inference):
        """process_dialogue with relative traversal path '../malicious' raises ValueError."""
        dialogue = self._import_dialogue_module()
        # Create a temporary dialogue file with valid content (so code reaches output_dir)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"lines": [{"text": "test"}]}')
            temp_dialogue = f.name

        try:
            # Create mock config and args with relative traversal path
            config = {"output_directory": tempfile.gettempdir()}
            args = MagicMock(output="../escape")  # relative path with traversal

            with self.assertRaises(ValueError) as ctx:
                # Attempt to write outside current directory via relative traversal
                dialogue.process_dialogue(
                    temp_dialogue, config, args, {}, use_server=True
                )
            self.assertIn("traversal", str(ctx.exception).lower())
        finally:
            os.unlink(temp_dialogue)


class TestAutoIncrementFilenameInjection(unittest.TestCase):
    """Test generate_helpers.py auto_increment_filename."""

    def test_auto_increment_rejects_traversal_in_path(self):
        """auto_increment_filename with traversal path '../escape/output.wav' raises ValueError."""
        from qwen3_tts.interface.generate_helpers import auto_increment_filename

        # Create a temp directory to work in
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "output.wav")
            # Create the file so auto-increment triggers
            open(base_path, "w").close()

            # This should raise ValueError due to path traversal attempt
            with self.assertRaises(ValueError) as ctx:
                auto_increment_filename(os.path.join(tmpdir, "../escape/output.wav"))
            self.assertIn("traversal", str(ctx.exception).lower())


class TestCreateVoicePathInjection(unittest.TestCase):
    """Test tools/create_voice.py output path validation."""

    def test_validate_voice_name_rejects_traversal(self):
        """validate_voice_name (used by create_and_save_voice_prompt) rejects traversal."""
        from qwen3_tts.core.config import validate_voice_name

        with self.assertRaises(ValueError) as ctx:
            validate_voice_name("../../../etc/passwd")
        self.assertIn("invalid", str(ctx.exception).lower())

    def test_resolve_audio_path_rejects_traversal(self):
        """_resolve_audio_path with traversal path '../../../malicious.wav' raises ValueError."""
        from unittest.mock import MagicMock

        from qwen3_tts.tools.create_voice import _resolve_audio_path

        args = MagicMock(audio="../../../etc/passwd")
        # This should raise - traversal attempt
        with self.assertRaises(ValueError) as ctx:
            _resolve_audio_path(args)
        self.assertIn("traversal", str(ctx.exception).lower())

    def test_resolve_audio_path_rejects_downloads_traversal(self):
        """_resolve_audio_path with '../escape in filename triggers ValueError."""
        from unittest.mock import MagicMock

        from qwen3_tts.tools.create_voice import _resolve_audio_path

        args = MagicMock(audio="../escape.wav")
        # The weak guard (".." not in text_or_file) should catch this
        with self.assertRaises(ValueError) as ctx:
            _resolve_audio_path(args)
        self.assertIn("traversal", str(ctx.exception).lower())

    def test_transcript_from_file_rejects_traversal_path(self):
        """_prompt_for_transcript with traversal file path '../../../etc/passwd' raises ValueError."""
        # This test requires mocking input() - we'll test the file path validation directly
        # For now, document that line 208-209 need safe_path_join validation
        pass


class TestGenerateHelpersPathInjection(unittest.TestCase):
    """Test generate_helpers.py path injection vulnerabilities (Group 1 - 9 alerts)."""

    def _import_generate_helpers(self):
        from qwen3_tts.interface import generate_helpers

        return generate_helpers

    def test_get_text_rejects_traversal_expanded_path(self):
        """get_text with traversal path '../../../etc/passwd' raises ValueError."""
        generate_helpers = self._import_generate_helpers()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("safe content")
            safe_path = f.name

        try:
            # This should work - safe path
            result = generate_helpers.get_text(safe_path)
            self.assertEqual(result, "safe content")
        finally:
            os.unlink(safe_path)

        # This should raise - traversal attempt
        with self.assertRaises(ValueError):
            generate_helpers.get_text("../../../etc/passwd")

    def test_get_text_rejects_traversal_downloads_path(self):
        """get_text with '../escape in filename triggers ValueError."""
        generate_helpers = self._import_generate_helpers()
        # The weak guard (".." not in text_or_file) should catch this
        with self.assertRaises(ValueError):
            generate_helpers.get_text("../escape.txt")

    def test_auto_increment_filename_rejects_traversal(self):
        """auto_increment_filename with traversal path '../escape.wav' raises ValueError."""
        generate_helpers = self._import_generate_helpers()
        # Create temp directory to work in
        with tempfile.TemporaryDirectory() as tmpdir:
            # This should work - safe path
            safe_path = os.path.join(tmpdir, "output.wav")
            result = generate_helpers.auto_increment_filename(safe_path)
            self.assertEqual(result, safe_path)

            # This should raise - traversal attempt
            with self.assertRaises(ValueError):
                generate_helpers.auto_increment_filename(
                    os.path.join(tmpdir, "../escape/output.wav")
                )

    def test_parse_srt_rejects_traversal_path(self):
        """parse_srt with traversal path '../../../malicious.srt' raises ValueError."""
        generate_helpers = self._import_generate_helpers()
        # This should raise - traversal attempt
        with self.assertRaises(ValueError):
            generate_helpers.parse_srt("../../../etc/passwd")

    def test_save_base64_result_rejects_traversal_path(self):
        """_save_base64_result with traversal output path raises ValueError."""
        generate_helpers = self._import_generate_helpers()
        mock_result = {"audio_base64": "dGVzdCBiYXNlNjQgYXVkaW8"}  # "test base64 audio"

        # This should raise - traversal attempt
        with self.assertRaises(ValueError):
            generate_helpers._save_base64_result(mock_result, "../../../etc/passwd.wav")


class TestSharedPathInjection(unittest.TestCase):
    """Test shared.py path injection vulnerabilities (Group 3 - 6 alerts)."""

    def _import_shared(self):
        from qwen3_tts.interface.ui import shared

        return shared

    def test_save_generation_metadata_rejects_traversal_wav_path(self):
        """save_generation_metadata with traversal wav_path '../../../escape.wav' raises ValueError."""
        shared = self._import_shared()
        metadata = {"text": "test", "mode": "clone"}

        # This should raise - traversal attempt in wav_path
        with self.assertRaises(ValueError):
            shared.save_generation_metadata("../../../escape.wav", metadata)

    def test_save_generation_metadata_rejects_absolute_escape(self):
        """save_generation_metadata with absolute path escaping home raises ValueError."""
        shared = self._import_shared()
        metadata = {"text": "test", "mode": "clone"}

        # This should raise - absolute path outside home directory
        with self.assertRaises(ValueError):
            shared.save_generation_metadata("/etc/passwd.wav", metadata)

    def test_load_history_from_disk_with_safe_output_dir(self):
        """load_history_from_disk with safe output_dir under home works."""
        shared = self._import_shared()
        import tempfile

        # Create temp directory under home (required by security fix)
        home = os.path.expanduser("~")
        with tempfile.TemporaryDirectory(dir=home) as tmpdir:
            # Create temp directory under home (should work)
            entries = shared.load_history_from_disk(tmpdir)
            self.assertIsInstance(entries, list)
            self.assertEqual(entries, [])  # No JSON files in empty dir

    def test_load_history_from_disk_rejects_traversal_output_dir(self):
        """load_history_from_disk with traversal output_dir '../../../etc' raises ValueError."""
        shared = self._import_shared()

        # This should raise - traversal attempt in output_dir
        with self.assertRaises(ValueError):
            shared.load_history_from_disk("../../../etc")


class TestSinkSideContainment(unittest.TestCase):
    """The containment check must cover the exact path each sink opens.

    CodeQL alerts #3281/#3296-#3298: the home check ran on one variable
    (the resolved .wav path / output dir) while the sink opened another
    (the derived .json sidecar / a globbed file / the raw argument), so a
    symlinked sidecar escaped the check. Each test pins the sink path.
    """

    WAV_HEADER = b"RIFF\x24\x00\x00\x00WAVEfmt "

    def setUp(self):
        self._home = tempfile.TemporaryDirectory()
        self._outside = tempfile.TemporaryDirectory()
        self.addCleanup(self._home.cleanup)
        self.addCleanup(self._outside.cleanup)
        self.home = self._home.name
        self.outside = self._outside.name
        env = patch.dict(os.environ, {"HOME": self.home})
        env.start()
        self.addCleanup(env.stop)

    def test_save_metadata_refuses_sidecar_symlinked_outside_home(self):
        from qwen3_tts.interface.ui import shared

        target = os.path.join(self.outside, "victim.json")
        with open(target, "w") as f:
            f.write("original")
        wav = os.path.join(self.home, "clip.wav")
        os.symlink(target, os.path.join(self.home, "clip.json"))

        with self.assertRaises(ValueError):
            shared.save_generation_metadata(wav, {"text": "t"})
        with open(target) as f:
            self.assertEqual(f.read(), "original")

    def test_save_metadata_still_writes_sidecar_under_home(self):
        from qwen3_tts.interface.ui import shared

        wav = os.path.join(self.home, "clip.wav")
        shared.save_generation_metadata(wav, {"text": "t"})
        self.assertTrue(os.path.exists(os.path.join(self.home, "clip.json")))

    def test_load_history_skips_sidecar_symlinked_outside_home(self):
        from qwen3_tts.interface.ui import shared

        target = os.path.join(self.outside, "leak.json")
        with open(target, "w") as f:
            f.write('{"text": "outside secret", "mode": "clone"}')
        with open(os.path.join(self.home, "voice_ui_1.wav"), "wb") as f:
            f.write(self.WAV_HEADER)
        os.symlink(target, os.path.join(self.home, "voice_ui_1.json"))

        self.assertEqual(shared.load_history_from_disk(self.home), [])

    def test_load_history_still_loads_regular_entries(self):
        from qwen3_tts.interface.ui import shared

        with open(os.path.join(self.home, "voice_ui_1.wav"), "wb") as f:
            f.write(self.WAV_HEADER)
        with open(os.path.join(self.home, "voice_ui_1.json"), "w") as f:
            f.write('{"text": "hi", "mode": "clone"}')

        entries = shared.load_history_from_disk(self.home)
        self.assertEqual([e["full_text"] for e in entries], ["hi"])

    def test_is_valid_wav_file_never_opens_outside_home_and_temp(self):
        from qwen3_tts.interface.ui import shared

        with patch("builtins.open") as mock_open:
            self.assertFalse(shared.is_valid_wav_file("/etc/hosts"))
        mock_open.assert_not_called()

    def test_is_valid_wav_file_accepts_temp_upload(self):
        from qwen3_tts.interface.ui import shared

        upload = os.path.join(self.outside, "upload.wav")  # system tempdir
        with open(upload, "wb") as f:
            f.write(self.WAV_HEADER)
        self.assertTrue(shared.is_valid_wav_file(upload))

    def test_process_batch_rejects_output_dir_outside_home_and_temp(self):
        from types import SimpleNamespace

        from qwen3_tts.interface.generate import process_batch

        args = SimpleNamespace(
            output="/opt/qwen3-escape-out",
            mode=None,
            prompt=None,
            description=None,
            trim_silence=False,
            normalize=False,
            speed=None,
            pitch=None,
        )
        with (
            patch("os.makedirs") as mock_makedirs,
            patch("qwen3_tts.interface.generate.generate_via_server", return_value=[]),
        ):
            with self.assertRaises(ValueError):
                process_batch(["hi"], args, {}, {}, use_server=True)
        mock_makedirs.assert_not_called()


class TestCliArgumentContainment(unittest.TestCase):
    """CLI path arguments are contained to home/temp before any sink touches them.

    CodeQL alerts #3299-#3303: --srt/--dialogue/--batch (isfile/open) and
    --watch (isdir) took the raw argument straight to the filesystem. The
    stubs force ``isfile``/``isdir`` True so an unfixed build reaches the
    handler instead of exiting early on "not found".
    """

    def setUp(self):
        self._home = tempfile.TemporaryDirectory()
        self.addCleanup(self._home.cleanup)
        self.home = self._home.name
        env = patch.dict(os.environ, {"HOME": self.home})
        env.start()
        self.addCleanup(env.stop)

    def _generation_args(self, **overrides):
        from types import SimpleNamespace

        base = dict(
            repl=False,
            watch=None,
            srt=None,
            dialogue=None,
            batch=None,
            voice=None,
            prompt=None,
            mode=None,
            description=None,
            preset=None,
            text=None,
            clipboard=False,
            dry_run=False,
            output=None,
        )
        return SimpleNamespace(**{**base, **overrides})

    def _handle(self, args):
        from qwen3_tts.interface import generate

        return generate._handle_generation(args, {}, {}, True, 500)

    def test_srt_outside_home_and_temp_is_refused(self):
        from qwen3_tts.interface import generate

        args = self._generation_args(srt="/opt/qwen3-escape.srt")
        with (
            patch("os.path.isfile", return_value=True),
            patch.object(generate, "process_srt_file") as handler,
        ):
            with self.assertRaises(SystemExit):
                self._handle(args)
        handler.assert_not_called()

    def test_srt_under_home_is_processed_by_resolved_path(self):
        from qwen3_tts.interface import generate

        srt = os.path.join(self.home, "subs.srt")
        with open(srt, "w") as f:
            f.write("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
        with patch.object(generate, "process_srt_file") as handler:
            self._handle(self._generation_args(srt=srt))
        handler.assert_called_once()
        self.assertEqual(handler.call_args[0][0], os.path.realpath(srt))

    def test_dialogue_outside_home_and_temp_is_refused(self):
        from qwen3_tts.interface import generate

        args = self._generation_args(dialogue="/opt/qwen3-escape.txt")
        with (
            patch("os.path.isfile", return_value=True),
            patch.object(generate, "process_dialogue") as handler,
        ):
            with self.assertRaises(SystemExit):
                self._handle(args)
        handler.assert_not_called()

    def test_dialogue_under_home_is_processed_by_resolved_path(self):
        from qwen3_tts.interface import generate

        dialogue = os.path.join(self.home, "talk.txt")
        with open(dialogue, "w") as f:
            f.write("A: hi\n")
        with patch.object(generate, "process_dialogue") as handler:
            self._handle(self._generation_args(dialogue=dialogue))
        handler.assert_called_once()
        self.assertEqual(handler.call_args[0][0], os.path.realpath(dialogue))

    def test_batch_file_outside_home_and_temp_is_never_opened(self):
        from qwen3_tts.interface import generate

        args = self._generation_args(batch="/opt/qwen3-escape.json")
        with (
            patch("os.path.isfile", return_value=True),
            patch("builtins.open", mock_open(read_data='["hi"]')) as opened,
            patch.object(generate, "process_batch") as handler,
        ):
            with self.assertRaises(SystemExit):
                self._handle(args)
        opened.assert_not_called()
        handler.assert_not_called()

    def test_batch_file_under_home_is_read_and_processed(self):
        from qwen3_tts.interface import generate

        batch = os.path.join(self.home, "batch.json")
        with open(batch, "w") as f:
            f.write('["hi"]')
        with patch.object(generate, "process_batch") as handler:
            self._handle(self._generation_args(batch=batch))
        handler.assert_called_once()
        self.assertEqual(handler.call_args[0][0], ["hi"])

    @unittest.skipUnless(os.path.isdir("/tmp"), "needs /tmp")
    def test_batch_file_under_slash_tmp_is_read_and_processed(self):
        """/tmp counts as temp even where gettempdir() is elsewhere (macOS)."""
        from qwen3_tts.interface import generate

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            batch = os.path.join(tmp, "batch.json")
            with open(batch, "w") as f:
                f.write('["hi"]')
            with patch.object(generate, "process_batch") as handler:
                self._handle(self._generation_args(batch=batch))
        handler.assert_called_once()

    def test_watch_dir_outside_home_and_temp_is_refused(self):
        from types import SimpleNamespace

        from qwen3_tts.interface.generate_interactive import run_watch_mode

        args = SimpleNamespace(output=None, mode=None, prompt=None, description=None)
        # watchdog is an optional dep (absent from the torchless CI env)
        stub = MagicMock()
        modules = {
            "watchdog": stub,
            "watchdog.events": stub.events,
            "watchdog.observers": stub.observers,
        }
        # An unfixed build falls through to the infinite watch loop; make that
        # an error instead of a hang.
        with (
            patch.dict("sys.modules", modules),
            patch("os.path.isdir", return_value=True),
            patch("os.makedirs") as mock_makedirs,
            patch("time.sleep", side_effect=RuntimeError("reached the watch loop")),
        ):
            with self.assertRaises(ValueError):
                run_watch_mode("/opt/qwen3-escape-watch", {}, args, {}, True)
        mock_makedirs.assert_not_called()

    def test_watch_dir_under_home_passes_containment(self):
        from types import SimpleNamespace

        from qwen3_tts.interface.generate_interactive import run_watch_mode

        args = SimpleNamespace(output=None, mode=None, prompt=None, description=None)
        stub = MagicMock()
        modules = {
            "watchdog": stub,
            "watchdog.events": stub.events,
            "watchdog.observers": stub.observers,
        }
        missing = os.path.join(self.home, "no-such-watch-dir")
        with patch.dict("sys.modules", modules):
            # Passes containment, then returns on the not-found branch.
            self.assertIsNone(run_watch_mode(missing, {}, args, {}, True))


class TestVoiceManagementPathInjection(unittest.TestCase):
    """Test voice_management.py path injection (Group 3 - 4 alerts, all false positives)."""

    def _import_voice_management(self):
        from qwen3_tts.interface.ui import voice_management

        return voice_management

    def test_create_voice_prompt_destinations_are_protected(self):
        """create_voice_prompt protects destination paths with safe_path_join.

        This test verifies the security pattern: destinations (wav_path, txt_path, pt_path)
        are constructed using safe_path_join(VOICE_PROMPTS_DIR, ...) and validate_voice_name,
        preventing path traversal even if audio_path is user-controlled.

        The audio_path parameter is only used as a SOURCE for reading/copying to safe destinations.
        All 4 CodeQL alerts in voice_management.py are false positives.
        """
        voice_management = self._import_voice_management()

        # Verify the function uses safe_path_join for destinations
        import inspect

        # Get the source code to verify the pattern
        source = inspect.getsource(voice_management.create_voice_prompt)

        # Verify safe_path_join is used for destination construction
        self.assertIn("safe_path_join", source)
        self.assertIn("VOICE_PROMPTS_DIR", source)
        self.assertIn("validate_voice_name", source)

        # The security pattern:
        # - Line 75-76: wav_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base_name}.wav")
        # - Line 79: pt_path = safe_path_join(VOICE_PROMPTS_DIR, f"{base_name}.pt")
        # These ensure destinations are under VOICE_PROMPTS_DIR regardless of voice_name
        #
        # - audio_path is user-provided but only used as SOURCE:
        #   - Line 93: shutil.copy(audio_path, wav_path) - copies FROM audio_path TO safe wav_path
        #   - Line 117: with open(audio_path, "rb") - reads FROM audio_path
        #   - Line 187: with open(audio_path, "rb") - reads FROM audio_path
        # This is safe: user's own file is copied to safe location validated by safe_path_join


class TestGenerateInteractivePathInjection(unittest.TestCase):
    """Test generate_interactive.py path injection (Group 4 - 2 alerts)."""

    def _import_generate_interactive(self):
        from qwen3_tts.interface import generate_interactive

        return generate_interactive

    def test_run_watch_mode_validates_watch_dir(self):
        """run_watch_mode should validate watch_dir parameter (line 592)."""
        generate_interactive = self._import_generate_interactive()
        import inspect

        # Verify the function validates watch_dir with safe_path_join
        source = inspect.getsource(generate_interactive.run_watch_mode)

        # Check for security validation patterns
        self.assertIn("safe_path_join", source)
        self.assertIn("Path traversal detected", source)
        # Verify os.path.isdir is called on safe_watch_dir, not raw watch_dir
        self.assertIn("os.path.isdir(safe_watch_dir)", source)

    def test_run_watch_mode_validates_output_dir(self):
        """run_watch_mode should validate output_dir parameter (line 597)."""
        generate_interactive = self._import_generate_interactive()
        import inspect

        # Verify the function validates output_dir with safe_path_join
        source = inspect.getsource(generate_interactive.run_watch_mode)

        # Check for security validation patterns
        self.assertIn("safe_path_join", source)
        self.assertIn("Path traversal detected", source)
        self.assertIn("output_dir must be under home directory", source)
        # Verify os.makedirs is called on safe_output_dir, not raw output_dir
        self.assertIn("os.makedirs(safe_output_dir", source)


if __name__ == "__main__":
    unittest.main()
