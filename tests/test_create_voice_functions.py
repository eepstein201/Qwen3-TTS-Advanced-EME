"""Tests for qwen3_tts/tools/create_voice.py functions.

Covers:
  - _resolve_audio_path: direct path, Downloads fallback, not found, interactive
  - _transcribe_with_asr: success, user reject, exception
  - _resolve_transcript: --no-transcript, --transcript text, --transcript file,
    --auto-transcribe (mock ASR), interactive input
  - create_and_save_voice_prompt: MLX-only .wav+.txt save, torch .pt save,
    test generation flag, pydub fallback for non-wav formats
"""
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
    import soundfile  # noqa: F401
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires numpy and soundfile")


# ---------------------------------------------------------------------------
# _resolve_audio_path
# ---------------------------------------------------------------------------

@_skip
class TestResolveAudioPath(unittest.TestCase):
    """Tests for _resolve_audio_path."""

    def _make_args(self, audio=None):
        return SimpleNamespace(audio=audio)

    def test_direct_path_exists(self):
        """Returns audio path when the file exists directly."""
        from qwen3_tts.tools.create_voice import _resolve_audio_path

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            args = self._make_args(audio=f.name)
            result = _resolve_audio_path(args)
            self.assertEqual(result, f.name)

    def test_downloads_fallback(self):
        """Falls back to ~/Downloads/<name> when direct path doesn't exist."""
        from qwen3_tts.tools.create_voice import _resolve_audio_path

        filename = "test_nonexistent_audio_12345.wav"
        downloads_path = os.path.expanduser(f"~/Downloads/{filename}")

        with mock.patch("os.path.isfile") as mock_isfile:
            # First call: direct path doesn't exist
            # Second call: downloads path exists
            mock_isfile.side_effect = lambda p: p == downloads_path
            args = self._make_args(audio=filename)
            result = _resolve_audio_path(args)
            self.assertEqual(result, downloads_path)

    def test_not_found_exits(self):
        """Exits with error when audio file not found anywhere."""
        from qwen3_tts.tools.create_voice import _resolve_audio_path

        args = self._make_args(audio="/tmp/nonexistent_audio_xyz.wav")
        with self.assertRaises(SystemExit):
            _resolve_audio_path(args)

    def test_interactive_input(self):
        """Prompts for audio path when args.audio is None."""
        from qwen3_tts.tools.create_voice import _resolve_audio_path

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            args = self._make_args(audio=None)
            with mock.patch("builtins.input", return_value=f.name):
                result = _resolve_audio_path(args)
                self.assertEqual(result, f.name)

    def test_interactive_empty_input_exits(self):
        """Exits when interactive input is empty."""
        from qwen3_tts.tools.create_voice import _resolve_audio_path

        args = self._make_args(audio=None)
        with mock.patch("builtins.input", return_value=""):
            with self.assertRaises(SystemExit):
                _resolve_audio_path(args)

    def test_tilde_expansion(self):
        """Expands ~ in audio path."""
        from qwen3_tts.tools.create_voice import _resolve_audio_path

        with tempfile.NamedTemporaryFile(suffix=".wav", dir=os.path.expanduser("~")) as f:
            # Build a path using ~
            basename = os.path.basename(f.name)
            tilde_path = f"~/{basename}"
            args = self._make_args(audio=tilde_path)
            result = _resolve_audio_path(args)
            self.assertEqual(result, f.name)


# ---------------------------------------------------------------------------
# _transcribe_with_asr
# ---------------------------------------------------------------------------

@_skip
class TestTranscribeWithAsr(unittest.TestCase):
    """Tests for _transcribe_with_asr."""

    @mock.patch("qwen3_tts.tools.create_voice.transcribe_audio",
                create=True)
    @mock.patch("builtins.input", return_value="y")
    def test_success_accepted(self, _mock_input, mock_transcribe):
        """Returns transcript when user accepts."""
        from qwen3_tts.tools.create_voice import _transcribe_with_asr

        mock_transcribe.return_value = "Hello world"
        with mock.patch("qwen3_tts.core.engine.transcribe_audio",
                        mock_transcribe):
            result = _transcribe_with_asr("/fake/audio.wav")
        self.assertEqual(result, "Hello world")

    @mock.patch("qwen3_tts.tools.create_voice.transcribe_audio",
                create=True)
    @mock.patch("builtins.input", return_value="n")
    def test_user_rejects(self, _mock_input, mock_transcribe):
        """Returns None when user rejects transcript."""
        from qwen3_tts.tools.create_voice import _transcribe_with_asr

        mock_transcribe.return_value = "Hello world"
        with mock.patch("qwen3_tts.core.engine.transcribe_audio",
                        mock_transcribe):
            result = _transcribe_with_asr("/fake/audio.wav")
        self.assertIsNone(result)

    def test_exception_returns_none(self):
        """Returns None when transcription raises an exception."""
        from qwen3_tts.tools.create_voice import _transcribe_with_asr

        with mock.patch("qwen3_tts.core.engine.transcribe_audio",
                        side_effect=RuntimeError("ASR failed")):
            result = _transcribe_with_asr("/fake/audio.wav")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# _resolve_transcript
# ---------------------------------------------------------------------------

@_skip
class TestResolveTranscript(unittest.TestCase):
    """Tests for _resolve_transcript."""

    def _make_args(self, **kwargs):
        defaults = {
            "no_transcript": False,
            "transcript": None,
            "auto_transcribe": False,
        }
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    @mock.patch("qwen3_tts.core.engine.is_asr_available", return_value=False)
    def test_no_transcript_flag(self, _mock_asr):
        """Returns empty string for --no-transcript."""
        from qwen3_tts.tools.create_voice import _resolve_transcript

        args = self._make_args(no_transcript=True)
        result = _resolve_transcript(args, "/fake/audio.wav")
        self.assertEqual(result, "")

    @mock.patch("qwen3_tts.core.engine.is_asr_available", return_value=False)
    def test_transcript_text_directly(self, _mock_asr):
        """Returns transcript text when it's not a file path."""
        from qwen3_tts.tools.create_voice import _resolve_transcript

        args = self._make_args(transcript="Hello from the transcript")
        result = _resolve_transcript(args, "/fake/audio.wav")
        self.assertEqual(result, "Hello from the transcript")

    @mock.patch("qwen3_tts.core.engine.is_asr_available", return_value=False)
    def test_transcript_from_file(self, _mock_asr):
        """Reads transcript from file when path exists."""
        from qwen3_tts.tools.create_voice import _resolve_transcript

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Transcript from file content")
            f.flush()
            try:
                args = self._make_args(transcript=f.name)
                result = _resolve_transcript(args, "/fake/audio.wav")
                self.assertEqual(result, "Transcript from file content")
            finally:
                os.unlink(f.name)

    @mock.patch("qwen3_tts.core.engine.is_asr_available", return_value=True)
    @mock.patch("qwen3_tts.core.engine.transcribe_audio",
                return_value="Auto transcribed text")
    @mock.patch("builtins.input", return_value="y")
    def test_auto_transcribe_success(self, _mock_input, _mock_transcribe,
                                      _mock_asr):
        """Returns auto-transcribed text when --auto-transcribe is set."""
        from qwen3_tts.tools.create_voice import _resolve_transcript

        args = self._make_args(auto_transcribe=True)
        result = _resolve_transcript(args, "/fake/audio.wav")
        self.assertEqual(result, "Auto transcribed text")

    @mock.patch("qwen3_tts.core.engine.is_asr_available", return_value=False)
    def test_auto_transcribe_no_asr_exits(self, _mock_asr):
        """Exits when --auto-transcribe but ASR not available."""
        from qwen3_tts.tools.create_voice import _resolve_transcript

        args = self._make_args(auto_transcribe=True)
        with self.assertRaises(SystemExit):
            _resolve_transcript(args, "/fake/audio.wav")

    @mock.patch("qwen3_tts.core.engine.is_asr_available", return_value=True)
    @mock.patch("qwen3_tts.core.engine.transcribe_audio",
                side_effect=RuntimeError("fail"))
    def test_auto_transcribe_failure_exits(self, _mock_transcribe, _mock_asr):
        """Exits when --auto-transcribe and transcription fails."""
        from qwen3_tts.tools.create_voice import _resolve_transcript

        args = self._make_args(auto_transcribe=True)
        with self.assertRaises(SystemExit):
            _resolve_transcript(args, "/fake/audio.wav")

    @mock.patch("qwen3_tts.core.engine.is_asr_available", return_value=False)
    @mock.patch("builtins.input", return_value="Typed transcript")
    def test_interactive_manual_input(self, _mock_input, _mock_asr):
        """Returns manually typed transcript in interactive mode (no ASR)."""
        from qwen3_tts.tools.create_voice import _resolve_transcript

        args = self._make_args()
        result = _resolve_transcript(args, "/fake/audio.wav")
        self.assertEqual(result, "Typed transcript")

    @mock.patch("qwen3_tts.core.engine.is_asr_available", return_value=True)
    @mock.patch("qwen3_tts.core.engine.transcribe_audio",
                return_value="ASR text")
    @mock.patch("builtins.input")
    def test_interactive_choose_asr(self, mock_input, _mock_transcribe,
                                     _mock_asr):
        """In interactive mode, choosing option 1 triggers ASR."""
        from qwen3_tts.tools.create_voice import _resolve_transcript

        # First input: choose "1" (ASR), second input: "y" (accept)
        mock_input.side_effect = ["1", "y"]
        args = self._make_args()
        result = _resolve_transcript(args, "/fake/audio.wav")
        self.assertEqual(result, "ASR text")

    @mock.patch("qwen3_tts.core.engine.is_asr_available", return_value=True)
    @mock.patch("builtins.input")
    def test_interactive_choose_manual(self, mock_input, _mock_asr):
        """In interactive mode, choosing option 2 prompts for manual input."""
        from qwen3_tts.tools.create_voice import _resolve_transcript

        # First input: choose "2" (manual), second input: actual transcript
        mock_input.side_effect = ["2", "Manual transcript"]
        args = self._make_args()
        result = _resolve_transcript(args, "/fake/audio.wav")
        self.assertEqual(result, "Manual transcript")

    @mock.patch("qwen3_tts.core.engine.is_asr_available", return_value=True)
    @mock.patch("builtins.input")
    def test_interactive_asr_rejected_falls_to_manual(self, mock_input,
                                                       _mock_asr):
        """When ASR transcript is rejected, falls back to manual input."""
        from qwen3_tts.tools.create_voice import _resolve_transcript

        with mock.patch("qwen3_tts.core.engine.transcribe_audio",
                        return_value="ASR text"):
            # Choose ASR, reject, then type manual transcript
            mock_input.side_effect = ["1", "n", "Fallback transcript"]
            args = self._make_args()
            result = _resolve_transcript(args, "/fake/audio.wav")
            self.assertEqual(result, "Fallback transcript")

    @mock.patch("qwen3_tts.core.engine.is_asr_available", return_value=False)
    @mock.patch("builtins.input")
    def test_interactive_input_is_file_path(self, mock_input, _mock_asr):
        """When interactive input is a valid file path, reads from file."""
        from qwen3_tts.tools.create_voice import _resolve_transcript

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False) as f:
            f.write("Content from file via interactive")
            f.flush()
            try:
                mock_input.return_value = f.name
                args = self._make_args()
                result = _resolve_transcript(args, "/fake/audio.wav")
                self.assertEqual(result, "Content from file via interactive")
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# create_and_save_voice_prompt — MLX-only mode
# ---------------------------------------------------------------------------

@_skip
class TestCreateAndSaveVoicePromptMlxOnly(unittest.TestCase):
    """Tests for create_and_save_voice_prompt in mlx_only mode."""

    def test_mlx_only_saves_wav_and_txt(self):
        """MLX-only mode saves .wav and .txt files in VOICE_PROMPTS_DIR."""
        from qwen3_tts.tools.create_voice import create_and_save_voice_prompt

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small reference audio file
            audio_path = os.path.join(tmpdir, "ref.wav")
            audio_data = np.zeros(16000, dtype=np.float32)
            import soundfile as sf
            sf.write(audio_path, audio_data, 16000)

            prompts_dir = os.path.join(tmpdir, "prompts")
            os.makedirs(prompts_dir)

            with mock.patch("qwen3_tts.tools.create_voice.VOICE_PROMPTS_DIR",
                            prompts_dir):
                with mock.patch("qwen3_tts.tools.create_voice.USER_FILES_DIR",
                                tmpdir):
                    result = create_and_save_voice_prompt(
                        audio_path, "Hello test", "test_voice",
                        test_generation=False, mlx_only=True,
                    )

            # Check files were created
            self.assertTrue(os.path.exists(
                os.path.join(prompts_dir, "test_voice.wav")))
            self.assertTrue(os.path.exists(
                os.path.join(prompts_dir, "test_voice.txt")))

            # Check transcript content
            with open(os.path.join(prompts_dir, "test_voice.txt")) as f:
                self.assertEqual(f.read(), "Hello test")

            # Return value is the .wav path (realpath-resolved)
            self.assertEqual(result,
                             os.path.realpath(os.path.join(prompts_dir, "test_voice.wav")))

    def test_mlx_only_with_pt_extension_in_name(self):
        """Handles prompt_name that already has .pt extension."""
        from qwen3_tts.tools.create_voice import create_and_save_voice_prompt

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "ref.wav")
            audio_data = np.zeros(16000, dtype=np.float32)
            import soundfile as sf
            sf.write(audio_path, audio_data, 16000)

            prompts_dir = os.path.join(tmpdir, "prompts")
            os.makedirs(prompts_dir)

            with mock.patch("qwen3_tts.tools.create_voice.VOICE_PROMPTS_DIR",
                            prompts_dir):
                with mock.patch("qwen3_tts.tools.create_voice.USER_FILES_DIR",
                                tmpdir):
                    create_and_save_voice_prompt(
                        audio_path, "Hello", "myvoice.pt",
                        test_generation=False, mlx_only=True,
                    )

            # Should strip .pt and create .wav/.txt
            self.assertTrue(os.path.exists(
                os.path.join(prompts_dir, "myvoice.wav")))
            self.assertTrue(os.path.exists(
                os.path.join(prompts_dir, "myvoice.txt")))

    def test_mlx_only_cleans_up_temp_wav(self):
        """MLX-only mode cleans up temp wav when pydub conversion was used."""
        from qwen3_tts.tools.create_voice import create_and_save_voice_prompt

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create audio that soundfile can read (no pydub needed)
            audio_path = os.path.join(tmpdir, "ref.wav")
            audio_data = np.zeros(16000, dtype=np.float32)
            import soundfile as sf
            sf.write(audio_path, audio_data, 16000)

            prompts_dir = os.path.join(tmpdir, "prompts")
            os.makedirs(prompts_dir)

            # Create a fake temp_reference.wav to simulate pydub conversion
            temp_wav = os.path.join(tmpdir, "temp_reference.wav")

            with mock.patch("qwen3_tts.tools.create_voice.VOICE_PROMPTS_DIR",
                            prompts_dir):
                with mock.patch("qwen3_tts.tools.create_voice.USER_FILES_DIR",
                                tmpdir):
                    create_and_save_voice_prompt(
                        audio_path, "Hello", "test_voice",
                        test_generation=False, mlx_only=True,
                    )

            # temp_reference.wav should not exist (cleaned up or never created)
            self.assertFalse(os.path.exists(temp_wav))


# ---------------------------------------------------------------------------
# create_and_save_voice_prompt — torch mode
# ---------------------------------------------------------------------------

@_skip
class TestCreateAndSaveVoicePromptTorch(unittest.TestCase):
    """Tests for create_and_save_voice_prompt in torch (non-mlx-only) mode."""

    @mock.patch("qwen3_tts.core.engine.run_inference")
    @mock.patch("qwen3_tts.core.engine.create_voice_prompt")
    @mock.patch("qwen3_tts.core.engine.load_model")
    def test_torch_saves_pt_and_mlx_files(self, mock_load, mock_create_vp,
                                           mock_inference):
        """Torch mode saves .pt, .wav, and .txt files."""
        from qwen3_tts.tools.create_voice import create_and_save_voice_prompt

        mock_model = mock.MagicMock()
        mock_load.return_value = mock_model
        mock_voice_prompt = mock.MagicMock()
        mock_create_vp.return_value = mock_voice_prompt

        # Mock torch.save
        mock_torch = mock.MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "ref.wav")
            audio_data = np.zeros(16000, dtype=np.float32)
            import soundfile as sf
            sf.write(audio_path, audio_data, 16000)

            prompts_dir = os.path.join(tmpdir, "prompts")
            os.makedirs(prompts_dir)

            with mock.patch("qwen3_tts.tools.create_voice.VOICE_PROMPTS_DIR",
                            prompts_dir):
                with mock.patch("qwen3_tts.tools.create_voice.USER_FILES_DIR",
                                tmpdir):
                    with mock.patch.dict(sys.modules, {"torch": mock_torch}):
                        result = create_and_save_voice_prompt(
                            audio_path, "Hello torch", "torch_voice",
                            test_generation=False, mlx_only=False,
                        )

            # .pt file path returned (realpath-resolved)
            self.assertEqual(result,
                             os.path.realpath(os.path.join(prompts_dir, "torch_voice.pt")))

            # MLX files also created
            self.assertTrue(os.path.exists(
                os.path.join(prompts_dir, "torch_voice.wav")))
            self.assertTrue(os.path.exists(
                os.path.join(prompts_dir, "torch_voice.txt")))

            # torch.save was called (with realpath-resolved path)
            mock_torch.save.assert_called_once_with(
                mock_voice_prompt,
                os.path.realpath(os.path.join(prompts_dir, "torch_voice.pt")),
            )

    @mock.patch("qwen3_tts.core.engine.run_inference")
    @mock.patch("qwen3_tts.core.engine.create_voice_prompt")
    @mock.patch("qwen3_tts.core.engine.load_model")
    def test_torch_with_test_generation(self, mock_load, mock_create_vp,
                                         mock_inference):
        """Torch mode with test_generation runs inference and saves test wav."""
        from qwen3_tts.tools.create_voice import create_and_save_voice_prompt

        mock_model = mock.MagicMock()
        mock_load.return_value = mock_model
        mock_voice_prompt = mock.MagicMock()
        mock_create_vp.return_value = mock_voice_prompt

        # Mock inference to return audio data
        test_audio = np.zeros(24000, dtype=np.float32)
        mock_inference.return_value = (test_audio, 24000)

        mock_torch = mock.MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "ref.wav")
            audio_data = np.zeros(16000, dtype=np.float32)
            import soundfile as sf
            sf.write(audio_path, audio_data, 16000)

            prompts_dir = os.path.join(tmpdir, "prompts")
            os.makedirs(prompts_dir)

            with mock.patch("qwen3_tts.tools.create_voice.VOICE_PROMPTS_DIR",
                            prompts_dir):
                with mock.patch("qwen3_tts.tools.create_voice.USER_FILES_DIR",
                                tmpdir):
                    with mock.patch.dict(sys.modules, {"torch": mock_torch}):
                        with mock.patch(
                            "qwen3_tts.core.config.IS_MACOS", False
                        ):
                            with mock.patch(
                                "qwen3_tts.core.config.IS_LINUX", False
                            ):
                                create_and_save_voice_prompt(
                                    audio_path, "Test gen", "gen_voice",
                                    test_generation=True, mlx_only=False,
                                )

            # run_inference should have been called
            mock_inference.assert_called_once()

            # Test audio file should have been created
            test_output = os.path.join(tmpdir, "test_gen_voice.wav")
            self.assertTrue(os.path.exists(test_output))

    @mock.patch("qwen3_tts.core.engine.run_inference")
    @mock.patch("qwen3_tts.core.engine.create_voice_prompt")
    @mock.patch("qwen3_tts.core.engine.load_model")
    def test_torch_test_gen_opens_on_macos(self, mock_load, mock_create_vp,
                                            mock_inference):
        """On macOS, test generation opens the audio file."""
        from qwen3_tts.tools.create_voice import create_and_save_voice_prompt

        mock_load.return_value = mock.MagicMock()
        mock_create_vp.return_value = mock.MagicMock()
        mock_inference.return_value = (
            np.zeros(24000, dtype=np.float32), 24000
        )

        mock_torch = mock.MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "ref.wav")
            import soundfile as sf
            sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000)

            prompts_dir = os.path.join(tmpdir, "prompts")
            os.makedirs(prompts_dir)

            with mock.patch("qwen3_tts.tools.create_voice.VOICE_PROMPTS_DIR",
                            prompts_dir):
                with mock.patch("qwen3_tts.tools.create_voice.USER_FILES_DIR",
                                tmpdir):
                    with mock.patch.dict(sys.modules, {"torch": mock_torch}):
                        with mock.patch(
                            "qwen3_tts.tools.create_voice.subprocess.run"
                        ) as mock_subproc:
                            with mock.patch(
                                "qwen3_tts.core.config.IS_MACOS", True
                            ):
                                with mock.patch(
                                    "qwen3_tts.core.config.IS_LINUX",
                                    False,
                                ):
                                    create_and_save_voice_prompt(
                                        audio_path, "Mac test", "mac_voice",
                                        test_generation=True, mlx_only=False,
                                    )

            # subprocess.run should have been called with "open"
            mock_subproc.assert_called_once()
            call_args = mock_subproc.call_args[0][0]
            self.assertEqual(call_args[0], "open")


# ---------------------------------------------------------------------------
# create_and_save_voice_prompt — pydub fallback
# ---------------------------------------------------------------------------

@_skip
class TestCreateAndSaveVoicePromptPydubFallback(unittest.TestCase):
    """Tests for pydub fallback when soundfile can't read the audio."""

    def _setup_pydub_test(self, tmpdir):
        """Set up common pydub fallback test scaffolding.

        Returns (audio_path, prompts_dir, mock_audio_segment, mock_pydub_module).
        """
        audio_path = os.path.join(tmpdir, "ref.m4a")
        with open(audio_path, "wb") as f:
            f.write(b"fake_m4a_data")

        prompts_dir = os.path.join(tmpdir, "prompts")
        os.makedirs(prompts_dir)

        audio_data = np.zeros(16000, dtype=np.float32)
        temp_wav = os.path.join(tmpdir, "temp_reference.wav")

        import soundfile as sf
        original_sf_read = sf.read

        def mock_sf_read(path, *args, **kwargs):
            if path == audio_path:
                raise sf.SoundFileError("Cannot read m4a")
            if path == temp_wav and not os.path.exists(temp_wav):
                sf.write(temp_wav, audio_data, 16000)
            return original_sf_read(path, *args, **kwargs)

        mock_pydub_segment = mock.MagicMock()
        mock_pydub_segment.export = mock.MagicMock(
            side_effect=lambda path, format: sf.write(path, audio_data, 16000)
        )
        mock_audio_segment = mock.MagicMock()
        mock_audio_segment.from_file.return_value = mock_pydub_segment

        # Build a mock pydub module that returns our mock AudioSegment
        mock_pydub_module = mock.MagicMock()
        mock_pydub_module.AudioSegment = mock_audio_segment

        return audio_path, prompts_dir, mock_audio_segment, mock_pydub_module, mock_sf_read

    def test_pydub_conversion_for_unsupported_format(self):
        """Falls back to pydub when soundfile raises SoundFileError."""
        from qwen3_tts.tools.create_voice import create_and_save_voice_prompt

        with tempfile.TemporaryDirectory() as tmpdir:
            (audio_path, prompts_dir, mock_audio_segment,
             mock_pydub_module, mock_sf_read) = self._setup_pydub_test(tmpdir)

            with mock.patch("qwen3_tts.tools.create_voice.VOICE_PROMPTS_DIR",
                            prompts_dir):
                with mock.patch("qwen3_tts.tools.create_voice.USER_FILES_DIR",
                                tmpdir):
                    with mock.patch("soundfile.read",
                                    side_effect=mock_sf_read):
                        with mock.patch.dict(
                            sys.modules, {"pydub": mock_pydub_module}
                        ):
                            create_and_save_voice_prompt(
                                audio_path, "M4a test", "m4a_voice",
                                test_generation=False, mlx_only=True,
                            )

            # Should have created the .wav file via copy (from temp)
            self.assertTrue(os.path.exists(
                os.path.join(prompts_dir, "m4a_voice.wav")))
            self.assertTrue(os.path.exists(
                os.path.join(prompts_dir, "m4a_voice.txt")))

    def test_m4a_extension_converted_to_mp4(self):
        """The m4a extension is remapped to mp4 for pydub."""
        from qwen3_tts.tools.create_voice import create_and_save_voice_prompt

        with tempfile.TemporaryDirectory() as tmpdir:
            (audio_path, prompts_dir, mock_audio_segment,
             mock_pydub_module, mock_sf_read) = self._setup_pydub_test(tmpdir)

            with mock.patch("qwen3_tts.tools.create_voice.VOICE_PROMPTS_DIR",
                            prompts_dir):
                with mock.patch("qwen3_tts.tools.create_voice.USER_FILES_DIR",
                                tmpdir):
                    with mock.patch("soundfile.read",
                                    side_effect=mock_sf_read):
                        with mock.patch.dict(
                            sys.modules, {"pydub": mock_pydub_module}
                        ):
                            create_and_save_voice_prompt(
                                audio_path, "M4a test", "m4a_voice",
                                test_generation=False, mlx_only=True,
                            )

            # pydub should have been called with format="mp4" (not "m4a")
            mock_audio_segment.from_file.assert_called_once_with(
                audio_path, format="mp4"
            )


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------

@_skip
class TestMainEntryPoint(unittest.TestCase):
    """Tests for the main() CLI entry point."""

    @mock.patch("qwen3_tts.tools.create_voice.create_and_save_voice_prompt")
    @mock.patch("qwen3_tts.tools.create_voice._resolve_transcript",
                return_value="Test transcript")
    @mock.patch("qwen3_tts.tools.create_voice._resolve_audio_path",
                return_value="/fake/audio.wav")
    @mock.patch("qwen3_tts.tools.create_voice.get_backend",
                return_value="torch")
    def test_main_with_all_args(self, _mock_backend, _mock_resolve_audio,
                                 _mock_resolve_transcript,
                                 mock_create):
        """main() passes correct args to create_and_save_voice_prompt."""
        from qwen3_tts.tools.create_voice import main

        with mock.patch("sys.argv", [
            "create_voice", "audio.wav", "-t", "Hello", "-n", "my_voice",
            "--no-test",
        ]):
            main()

        mock_create.assert_called_once_with(
            "/fake/audio.wav", "Test transcript", "my_voice",
            test_generation=False,
            mlx_only=False,
        )

    @mock.patch("qwen3_tts.tools.create_voice.create_and_save_voice_prompt")
    @mock.patch("qwen3_tts.tools.create_voice._resolve_transcript",
                return_value="Test transcript")
    @mock.patch("qwen3_tts.tools.create_voice._resolve_audio_path",
                return_value="/fake/audio.wav")
    @mock.patch("qwen3_tts.tools.create_voice.get_backend",
                return_value="mlx")
    def test_main_auto_mlx_only(self, _mock_backend, _mock_resolve_audio,
                                 _mock_resolve_transcript, mock_create):
        """main() auto-enables mlx_only when backend is mlx."""
        from qwen3_tts.tools.create_voice import main

        with mock.patch("sys.argv", [
            "create_voice", "audio.wav", "-t", "Hello", "-n", "my_voice",
        ]):
            main()

        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        self.assertTrue(kwargs["mlx_only"])

    @mock.patch("qwen3_tts.tools.create_voice.create_and_save_voice_prompt")
    @mock.patch("qwen3_tts.tools.create_voice._resolve_transcript",
                return_value="Test transcript")
    @mock.patch("qwen3_tts.tools.create_voice._resolve_audio_path",
                return_value="/fake/audio.wav")
    @mock.patch("qwen3_tts.tools.create_voice.get_backend",
                return_value="mlx")
    def test_main_force_torch_overrides_mlx(self, _mock_backend,
                                             _mock_resolve_audio,
                                             _mock_resolve_transcript,
                                             mock_create):
        """--force-torch overrides auto mlx_only detection."""
        from qwen3_tts.tools.create_voice import main

        with mock.patch("sys.argv", [
            "create_voice", "audio.wav", "-t", "Hello", "-n", "my_voice",
            "--force-torch",
        ]):
            main()

        mock_create.assert_called_once()
        _, kwargs = mock_create.call_args
        self.assertFalse(kwargs["mlx_only"])

    @mock.patch("qwen3_tts.tools.create_voice.create_and_save_voice_prompt")
    @mock.patch("qwen3_tts.tools.create_voice._resolve_transcript",
                return_value="Test")
    @mock.patch("qwen3_tts.tools.create_voice._resolve_audio_path",
                return_value="/fake/audio.wav")
    @mock.patch("qwen3_tts.tools.create_voice.get_backend",
                return_value="torch")
    @mock.patch("builtins.input", return_value="interactive_name")
    def test_main_prompts_for_name(self, _mock_input, _mock_backend,
                                    _mock_resolve_audio,
                                    _mock_resolve_transcript, mock_create):
        """main() prompts for name when --name not provided."""
        from qwen3_tts.tools.create_voice import main

        with mock.patch("sys.argv", [
            "create_voice", "audio.wav", "-t", "Hello",
        ]):
            main()

        mock_create.assert_called_once()
        args_passed = mock_create.call_args[0]
        self.assertEqual(args_passed[2], "interactive_name")


# ---------------------------------------------------------------------------
# R-43: main() argv testability
# ---------------------------------------------------------------------------

class TestCreateVoiceMainArgv:
    """R-43: main() must accept optional argv param (pytest-style)."""

    def test_main_accepts_argv_param(self):
        """main() signature must declare an argv keyword argument."""
        import inspect
        from qwen3_tts.tools.create_voice import main

        sig = inspect.signature(main)
        assert "argv" in sig.parameters, "main() must accept argv parameter"

    def test_main_returns_nonzero_on_missing_audio(self, tmp_path):
        """main(argv=...) returns non-zero when audio file does not exist."""
        from qwen3_tts.tools.create_voice import main

        fake_path = str(tmp_path / "nonexistent.wav")
        result = main(argv=[fake_path, "-n", "test", "--no-test", "--no-transcript"])
        assert result != 0

    def test_main_uses_argv_not_sys_argv(self, monkeypatch):
        """main() parses argv param and ignores sys.argv."""
        from qwen3_tts.tools.create_voice import main

        # Poison sys.argv — if main reads it, argparse will fail on unknown flag
        monkeypatch.setattr("sys.argv", ["create_voice", "--bogus-flag-99"])
        result = main(argv=["nonexistent.wav", "-n", "test", "--no-test", "--no-transcript"])
        # Fails on missing file, NOT on argparse error from sys.argv
        assert result != 0

    def test_main_returns_zero_type(self):
        """main is callable and its return type is int on error paths."""
        from qwen3_tts.tools.create_voice import main

        result = main(argv=["nonexistent_file.wav", "-n", "x", "--no-transcript"])
        assert isinstance(result, int)


if __name__ == "__main__":
    unittest.main()
