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
import contextlib
import os
import tempfile
import unittest
from unittest.mock import MagicMock, mock_open, patch

try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

try:
    import numpy as np
    import soundfile as sf
    HAS_AUDIO_DEPS = True
except ImportError:
    HAS_AUDIO_DEPS = False

_MOD = "qwen3_tts.interface.ui.voice_management"
_ENGINE = "qwen3_tts.core.engine"
_ENGINE_VOICE_PROMPT = "qwen3_tts.core.engine.voice_prompt"


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestCreateVoicePromptUI(unittest.TestCase):
    """Tests for create_voice_prompt."""

    def test_no_audio_raises(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        with self.assertRaises(gr.Error):
            create_voice_prompt(None, "hello", "my_voice", False)

    def test_no_name_raises(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        with self.assertRaises(gr.Error):
            create_voice_prompt("/tmp/audio.wav", "hello", "", False)

    def test_invalid_name_raises(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        with patch(f"{_MOD}.validate_prompt_name", return_value=[{"error": "bad name"}]):
            with self.assertRaises(gr.Error):
                create_voice_prompt("/tmp/audio.wav", "hello", "my_voice", False)

    def test_mlx_already_exists_raises(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        with patch(f"{_MOD}.validate_prompt_name", return_value=None), \
             patch(f"{_MOD}.load_config", return_value={"advanced": {"backend": "mlx"}}), \
             patch(f"{_MOD}.strip_extension", return_value="voice1"), \
             patch("os.path.exists", return_value=True):
            with self.assertRaises(gr.Error):
                create_voice_prompt("/tmp/audio.wav", "hello", "voice1", False)

    def test_mlx_creates_wav_and_txt(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        with tempfile.TemporaryDirectory() as target, \
             patch(f"{_MOD}.validate_prompt_name", return_value=None), \
             patch(f"{_MOD}.load_config", return_value={"advanced": {"backend": "mlx"}}), \
             patch(f"{_MOD}.strip_extension", return_value="new_voice"), \
             patch(f"{_MOD}.VOICE_PROMPTS_DIR", target), \
             patch(f"{_ENGINE}.save_voice_prompt_mlx", return_value="wav") as mock_writer, \
             patch(f"{_ENGINE}.clear_voice_prompt_cache"), \
             patch(f"{_MOD}.get_voice_prompts", return_value=["new_voice.wav"]), \
             patch(f"{_MOD}.get_default_clone_prompt", return_value="new_voice.wav"):
            status, prompts, default = create_voice_prompt(
                "/tmp/audio.wav", "Hello world", "new_voice", False,
            )
        self.assertIn("MLX", status)
        mock_writer.assert_called_once_with(
            "new_voice", "/tmp/audio.wav", "Hello world"
        )

    def test_mlx_no_transcript_mode(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        with tempfile.TemporaryDirectory() as target, \
             patch(f"{_MOD}.validate_prompt_name", return_value=None), \
             patch(f"{_MOD}.load_config", return_value={"advanced": {"backend": "mlx"}}), \
             patch(f"{_MOD}.strip_extension", return_value="voice"), \
             patch(f"{_MOD}.VOICE_PROMPTS_DIR", target), \
             patch(f"{_ENGINE}.save_voice_prompt_mlx", return_value="wav") as mock_writer, \
             patch(f"{_ENGINE}.clear_voice_prompt_cache"), \
             patch(f"{_MOD}.get_voice_prompts", return_value=["voice.wav"]), \
             patch(f"{_MOD}.get_default_clone_prompt", return_value="voice.wav"):
            status, _, _ = create_voice_prompt(
                "/tmp/audio.wav", None, "voice", True,
            )
        self.assertIn("MLX", status)
        mock_writer.assert_called_once_with("voice", "/tmp/audio.wav", None)

    def test_mlx_no_transcript_no_flag_raises(self):
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt
        with tempfile.TemporaryDirectory() as target, \
             patch(f"{_MOD}.validate_prompt_name", return_value=None), \
             patch(f"{_MOD}.load_config", return_value={"advanced": {"backend": "mlx"}}), \
             patch(f"{_MOD}.strip_extension", return_value="voice"), \
             patch(f"{_MOD}.VOICE_PROMPTS_DIR", target), \
             patch(f"{_ENGINE}.save_voice_prompt_mlx") as mock_writer:
            with self.assertRaises(gr.Error):
                create_voice_prompt("/tmp/audio.wav", "", "voice", False)
        mock_writer.assert_not_called()


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


@unittest.skipUnless(HAS_AUDIO_DEPS, "requires numpy + soundfile")
class TestMlxCreateEngineWriterPins(unittest.TestCase):
    """The UI MLX voice-create must route through the engine writer.

    ``save_voice_prompt_mlx`` is the single guarded store path: reference-
    source containment (home or the system tempdir), a zero-sample refusal,
    the >=24 kHz write-time guarantee, and ``.wav``-removal rollback when the
    transcript write fails. The pre-0G inline branch implemented none of
    those guards, so each pin below fails against it for exactly the reason
    named in its test id. The writer itself runs for real against tmp paths;
    the home/tempdir roles are re-pointed so an "outside" source can exist
    portably (same shape as TestReferenceSourceContainment, batch 3).
    """

    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.fake_home = os.path.join(root.name, "home")
        self.fake_tmp = os.path.join(root.name, "tmp")
        self.outside = os.path.join(root.name, "outside")
        self.target = os.path.join(root.name, "prompts")
        for d in (self.fake_home, self.fake_tmp, self.outside, self.target):
            os.mkdir(d)

    def _env_patches(self, base_name):
        """Patch the create path's environment: mlx backend, tmp target dir,
        home/tempdir roles re-pointed so `self.outside` is outside BOTH
        allowed roots, and the listing helpers stubbed."""
        return [
            patch(f"{_MOD}.validate_prompt_name", return_value=None),
            patch(
                f"{_MOD}.load_config",
                return_value={"advanced": {"backend": "mlx"}},
            ),
            patch(f"{_MOD}.strip_extension", return_value=base_name),
            patch(f"{_MOD}.VOICE_PROMPTS_DIR", self.target),
            patch(f"{_ENGINE_VOICE_PROMPT}.VOICE_PROMPTS_DIR", self.target),
            patch("os.path.expanduser", return_value=self.fake_home),
            patch("tempfile.gettempdir", return_value=self.fake_tmp),
            patch(f"{_MOD}.get_voice_prompts", return_value=[]),
            patch(f"{_MOD}.get_default_clone_prompt", return_value=""),
        ]

    def _write_wav(self, directory, rate):
        path = os.path.join(directory, "clip.wav")
        sf.write(path, np.zeros((rate // 10,), dtype="float32"), rate)
        return path

    def test_outside_home_reference_is_refused_and_writes_nothing(self):
        """Containment: a decodable reference OUTSIDE home and the system
        tempdir must raise gr.Error with the engine's message and write no
        pair anywhere. The inline branch happily copied such a file."""
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt

        src = self._write_wav(self.outside, 24000)
        with contextlib.ExitStack() as stack:
            for p in self._env_patches("evil_voice"):
                stack.enter_context(p)
            with self.assertRaises(gr.Error) as ctx:
                create_voice_prompt(src, "hello", "evil_voice", False)
        self.assertIn("home directory", str(ctx.exception))
        self.assertEqual(
            os.listdir(self.target), [], "no pair may be written"
        )
        self.assertEqual(os.listdir(self.outside), ["clip.wav"])

    def test_transcript_write_failure_rolls_back_the_wav(self):
        """Rollback: when the transcript write fails after the audio is
        stored, no orphan .wav may survive. The .txt target is planted as a
        DIRECTORY so the transcript open fails with IsADirectoryError right
        after the .wav has been stored -- exactly the writer's rollback
        trigger (os.path.exists is taught to ignore the marker so the
        already-exists pre-check does not trip on it). The inline branch
        orphaned the .wav."""
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt

        src = self._write_wav(self.fake_home, 24000)
        txt_marker = os.path.join(self.target, "rollback_voice.txt")
        os.mkdir(txt_marker)

        real_exists = os.path.exists

        def exists_ignoring_marker(path):
            if os.path.realpath(str(path)) == os.path.realpath(txt_marker):
                return False
            return real_exists(path)

        with contextlib.ExitStack() as stack:
            for p in self._env_patches("rollback_voice"):
                stack.enter_context(p)
            stack.enter_context(
                patch("os.path.exists", side_effect=exists_ignoring_marker)
            )
            with self.assertRaises(gr.Error):
                create_voice_prompt(src, "hello", "rollback_voice", False)
        self.assertFalse(
            any(f.endswith(".wav") for f in os.listdir(self.target)),
            "no orphan .wav may survive a failed transcript write",
        )

    def test_none_transcript_reaches_writer_and_is_coerced(self):
        """None-coercion: transcript=None (x_vector flag on) must reach the
        writer RAW -- the engine coerces ``(transcript or "").strip()``; the
        UI must not call .strip() itself (AttributeError on None) -- and the
        stored .txt must be the coerced empty string."""
        import qwen3_tts.core.engine as engine_pkg
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt

        src = self._write_wav(self.fake_home, 24000)
        with contextlib.ExitStack() as stack:
            for p in self._env_patches("coerce_voice"):
                stack.enter_context(p)
            spy = stack.enter_context(
                patch(
                    f"{_ENGINE}.save_voice_prompt_mlx",
                    wraps=engine_pkg.save_voice_prompt_mlx,
                )
            )
            status, _, _ = create_voice_prompt(src, None, "coerce_voice", True)
        self.assertIn("MLX", status)
        spy.assert_called_once()
        self.assertIsNone(
            spy.call_args.args[2],
            "the raw None must reach the writer; the UI must not .strip() it",
        )
        self.assertEqual(
            sorted(os.listdir(self.target)),
            ["coerce_voice.txt", "coerce_voice.wav"],
        )
        with open(os.path.join(self.target, "coerce_voice.txt")) as f:
            self.assertEqual(f.read(), "")

    def test_mlx_create_clears_the_engine_prompt_cache(self):
        """Cache parity: the server's MLX create clears the engine voice-
        prompt cache after storing (app_prompts.py); the rewired UI path
        must not become the only create surface skipping it."""
        from qwen3_tts.interface.ui.voice_management import create_voice_prompt

        src = self._write_wav(self.fake_home, 24000)
        with contextlib.ExitStack() as stack:
            for p in self._env_patches("cached_voice"):
                stack.enter_context(p)
            mock_writer = stack.enter_context(
                patch(f"{_ENGINE}.save_voice_prompt_mlx", return_value="wav")
            )
            mock_clear = stack.enter_context(
                patch(f"{_ENGINE}.clear_voice_prompt_cache")
            )
            create_voice_prompt(src, "hello", "cached_voice", False)
        mock_clear.assert_called_once_with()
        mock_writer.assert_called_once()
