#!/usr/bin/env python3
"""Tests for qwen3_tts.interface.generate_helpers module.

Covers text/audio utilities, SSML/SRT parsing, history, voice alias resolution,
and generation parameter handling. Pure unit tests — no server, GPU, or models.

Run with:
    cd ~/Qwen3-TTS_UserFiles/.worktrees/coverage-pr1 && python -m pytest tests/test_generate_helpers.py -v
"""

import base64
import io
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

    class _DummyMarkerFunc:
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition, **kwargs):
            return lambda f: f

    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()

    class _DummyPytest:
        mark = _DummyMark()

    pytest = _DummyPytest()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =========================================================================
# voice_prompt_exists
# =========================================================================

@pytest.mark.unit
class TestVoicePromptExists(unittest.TestCase):
    """Tests for voice_prompt_exists — backend-aware prompt file check."""

    @mock.patch("qwen3_tts.interface.generate_helpers.get_backend", return_value="mlx")
    @mock.patch("qwen3_tts.interface.generate_helpers.os.path.exists")
    def test_mlx_backend_requires_wav_and_txt(self, mock_exists, _mock_backend):
        """MLX backend returns True only when both .wav and .txt exist."""
        from qwen3_tts.interface.generate_helpers import voice_prompt_exists
        mock_exists.side_effect = lambda p: p.endswith(".wav") or p.endswith(".txt")
        self.assertTrue(voice_prompt_exists("my_voice"))

    @mock.patch("qwen3_tts.interface.generate_helpers.get_backend", return_value="mlx")
    @mock.patch("qwen3_tts.interface.generate_helpers.os.path.exists")
    def test_mlx_backend_missing_txt(self, mock_exists, _mock_backend):
        """MLX backend returns False when .txt is missing."""
        from qwen3_tts.interface.generate_helpers import voice_prompt_exists
        mock_exists.side_effect = lambda p: p.endswith(".wav")
        self.assertFalse(voice_prompt_exists("my_voice"))

    @mock.patch("qwen3_tts.interface.generate_helpers.get_backend", return_value="mlx")
    @mock.patch("qwen3_tts.interface.generate_helpers.os.path.exists")
    def test_mlx_backend_strips_pt_suffix(self, mock_exists, _mock_backend):
        """MLX backend strips .pt suffix before checking .wav/.txt pair."""
        from qwen3_tts.interface.generate_helpers import voice_prompt_exists
        mock_exists.return_value = True
        self.assertTrue(voice_prompt_exists("my_voice.pt"))

    @mock.patch("qwen3_tts.interface.generate_helpers.get_backend", return_value="torch")
    @mock.patch("qwen3_tts.interface.generate_helpers.os.path.exists", return_value=True)
    def test_torch_backend_checks_pt_file(self, mock_exists, _mock_backend):
        """Torch backend returns True when .pt file exists."""
        from qwen3_tts.interface.generate_helpers import voice_prompt_exists
        self.assertTrue(voice_prompt_exists("my_voice.pt"))

    @mock.patch("qwen3_tts.interface.generate_helpers.get_backend", return_value="torch")
    @mock.patch("qwen3_tts.interface.generate_helpers.os.path.exists", return_value=False)
    def test_torch_backend_missing_pt(self, mock_exists, _mock_backend):
        """Torch backend returns False when .pt file is missing."""
        from qwen3_tts.interface.generate_helpers import voice_prompt_exists
        self.assertFalse(voice_prompt_exists("my_voice.pt"))


# =========================================================================
# list_voice_prompts
# =========================================================================

@pytest.mark.unit
class TestListVoicePrompts(unittest.TestCase):
    """Tests for list_voice_prompts — directory listing of prompt files."""

    @mock.patch("qwen3_tts.interface.generate_helpers.os.listdir")
    def test_lists_pt_files(self, mock_listdir):
        """Returns sorted .pt files."""
        from qwen3_tts.interface.generate_helpers import list_voice_prompts
        mock_listdir.return_value = ["beta.pt", "alpha.pt", "readme.md"]
        result = list_voice_prompts()
        self.assertEqual(result, ["alpha.pt", "beta.pt"])

    @mock.patch("qwen3_tts.interface.generate_helpers.os.listdir")
    def test_lists_mlx_wav_txt_pairs(self, mock_listdir):
        """Returns .wav files that have a matching .txt file."""
        from qwen3_tts.interface.generate_helpers import list_voice_prompts
        mock_listdir.return_value = ["voice.wav", "voice.txt", "orphan.wav"]
        result = list_voice_prompts()
        self.assertIn("voice.wav", result)
        self.assertNotIn("orphan.wav", result)

    @mock.patch("qwen3_tts.interface.generate_helpers.os.listdir", return_value=[])
    def test_empty_directory(self, _mock):
        """Returns empty list for empty directory."""
        from qwen3_tts.interface.generate_helpers import list_voice_prompts
        self.assertEqual(list_voice_prompts(), [])

    @mock.patch("qwen3_tts.interface.generate_helpers.os.listdir", side_effect=OSError("not found"))
    def test_oserror_returns_empty(self, _mock):
        """Returns empty list when directory does not exist."""
        from qwen3_tts.interface.generate_helpers import list_voice_prompts
        self.assertEqual(list_voice_prompts(), [])


# =========================================================================
# get_text
# =========================================================================

@pytest.mark.unit
class TestGetText(unittest.TestCase):
    """Tests for get_text — resolves text from string, file, or ~/Downloads."""

    def test_direct_text_passthrough(self):
        """Returns text directly when it is not a file path."""
        from qwen3_tts.interface.generate_helpers import get_text
        self.assertEqual(get_text("Hello world"), "Hello world")

    def test_reads_text_from_file(self):
        """Reads and strips text from an existing file."""
        from qwen3_tts.interface.generate_helpers import get_text
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("  file contents  \n")
            path = f.name
        try:
            self.assertEqual(get_text(path), "file contents")
        finally:
            os.unlink(path)

    @mock.patch("qwen3_tts.interface.generate_helpers.os.path.isfile")
    def test_downloads_fallback(self, mock_isfile):
        """Falls back to ~/Downloads for bare filenames."""
        from qwen3_tts.interface.generate_helpers import get_text
        mock_isfile.side_effect = lambda p: "Downloads" in p
        m = mock.mock_open(read_data="downloaded text")
        with mock.patch("builtins.open", m):
            result = get_text("notes.txt")
        self.assertEqual(result, "downloaded text")

    def test_path_traversal_blocked(self):
        """Blocks path traversal attempts with '..' in filename."""
        from qwen3_tts.interface.generate_helpers import get_text
        result = get_text("../etc/passwd")
        self.assertEqual(result, "../etc/passwd")

    def test_slash_in_name_skips_downloads(self):
        """Skips Downloads fallback when filename contains '/'."""
        from qwen3_tts.interface.generate_helpers import get_text
        result = get_text("sub/file.txt")
        self.assertEqual(result, "sub/file.txt")


# =========================================================================
# get_clipboard_text
# =========================================================================

@pytest.mark.unit
class TestGetClipboardText(unittest.TestCase):
    """Tests for get_clipboard_text — platform-aware clipboard access."""

    @mock.patch("qwen3_tts.interface.generate_helpers.subprocess.run")
    @mock.patch("qwen3_tts.core.config.IS_MACOS", True)
    @mock.patch("qwen3_tts.core.config.IS_LINUX", False)
    @mock.patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_macos_uses_pbpaste(self, mock_run):
        """macOS calls pbpaste and returns trimmed output."""
        from qwen3_tts.interface.generate_helpers import get_clipboard_text
        mock_run.return_value = mock.Mock(stdout="  hello  ")
        result = get_clipboard_text()
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args[0][0], ["pbpaste"])
        self.assertEqual(result, "hello")

    @mock.patch("qwen3_tts.interface.generate_helpers.subprocess.run")
    @mock.patch("qwen3_tts.core.config.IS_MACOS", False)
    @mock.patch("qwen3_tts.core.config.IS_LINUX", True)
    @mock.patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_linux_uses_xclip(self, mock_run):
        """Linux calls xclip and returns trimmed output."""
        from qwen3_tts.interface.generate_helpers import get_clipboard_text
        mock_run.return_value = mock.Mock(stdout="linux text")
        result = get_clipboard_text()
        self.assertIn("xclip", mock_run.call_args[0][0])
        self.assertEqual(result, "linux text")

    @mock.patch("qwen3_tts.core.config.IN_COLAB", True)
    def test_colab_exits(self):
        """Colab environment exits with error."""
        from qwen3_tts.interface.generate_helpers import get_clipboard_text
        with self.assertRaises(SystemExit):
            get_clipboard_text()

    @mock.patch("qwen3_tts.interface.generate_helpers.subprocess.run")
    @mock.patch("qwen3_tts.core.config.IS_MACOS", True)
    @mock.patch("qwen3_tts.core.config.IS_LINUX", False)
    @mock.patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_empty_clipboard_exits(self, mock_run):
        """Exits when clipboard is empty."""
        from qwen3_tts.interface.generate_helpers import get_clipboard_text
        mock_run.return_value = mock.Mock(stdout="   ")
        with self.assertRaises(SystemExit):
            get_clipboard_text()

    @mock.patch("qwen3_tts.interface.generate_helpers.subprocess.run",
                side_effect=FileNotFoundError)
    @mock.patch("qwen3_tts.core.config.IS_MACOS", True)
    @mock.patch("qwen3_tts.core.config.IS_LINUX", False)
    @mock.patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_missing_command_exits(self, _mock_run):
        """Exits when clipboard command is not found."""
        from qwen3_tts.interface.generate_helpers import get_clipboard_text
        with self.assertRaises(SystemExit):
            get_clipboard_text()


# =========================================================================
# auto_increment_filename
# =========================================================================

@pytest.mark.unit
class TestAutoIncrementFilename(unittest.TestCase):
    """Tests for auto_increment_filename — collision-free file naming."""

    @mock.patch("qwen3_tts.interface.generate_helpers.os.path.exists", return_value=False)
    def test_no_collision_returns_original(self, _mock):
        """Returns original path when no collision exists."""
        from qwen3_tts.interface.generate_helpers import auto_increment_filename
        self.assertEqual(auto_increment_filename("/tmp/out.wav"), "/tmp/out.wav")

    @mock.patch("qwen3_tts.interface.generate_helpers.os.path.exists")
    def test_simple_increment(self, mock_exists):
        """Appends _2 when original exists."""
        from qwen3_tts.interface.generate_helpers import auto_increment_filename
        mock_exists.side_effect = lambda p: p == "/tmp/out.wav"
        self.assertEqual(auto_increment_filename("/tmp/out.wav"), "/tmp/out_2.wav")

    @mock.patch("qwen3_tts.interface.generate_helpers.os.path.exists")
    def test_already_numbered_file(self, mock_exists):
        """Increments from existing _2 to _3."""
        from qwen3_tts.interface.generate_helpers import auto_increment_filename
        mock_exists.side_effect = lambda p: p in ("/tmp/out_2.wav", "/tmp/out_3.wav")
        result = auto_increment_filename("/tmp/out_2.wav")
        self.assertEqual(result, "/tmp/out_4.wav")


# =========================================================================
# play_audio
# =========================================================================

@pytest.mark.unit
class TestPlayAudio(unittest.TestCase):
    """Tests for play_audio — platform-aware audio playback."""

    @mock.patch("qwen3_tts.interface.generate_helpers.subprocess.run")
    @mock.patch("qwen3_tts.core.config.IS_MACOS", True)
    @mock.patch("qwen3_tts.core.config.IS_LINUX", False)
    @mock.patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_macos_uses_afplay(self, mock_run):
        """macOS invokes afplay."""
        from qwen3_tts.interface.generate_helpers import play_audio
        play_audio("/tmp/audio.wav")
        self.assertEqual(mock_run.call_args[0][0][0], "afplay")

    @mock.patch("qwen3_tts.interface.generate_helpers.subprocess.run")
    @mock.patch("qwen3_tts.core.config.IS_MACOS", False)
    @mock.patch("qwen3_tts.core.config.IS_LINUX", True)
    @mock.patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_linux_uses_ffplay(self, mock_run):
        """Linux invokes ffplay with -nodisp -autoexit."""
        from qwen3_tts.interface.generate_helpers import play_audio
        play_audio("/tmp/audio.wav")
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "ffplay")
        self.assertIn("-nodisp", cmd)

    @mock.patch("qwen3_tts.core.config.IN_COLAB", True)
    def test_colab_skips_playback(self):
        """Colab environment logs instead of playing."""
        from qwen3_tts.interface.generate_helpers import play_audio
        play_audio("/tmp/audio.wav")  # Should not raise

    @mock.patch("qwen3_tts.core.config.IS_MACOS", False)
    @mock.patch("qwen3_tts.core.config.IS_LINUX", False)
    @mock.patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_unsupported_platform_warns(self):
        """Unsupported platform logs a warning."""
        from qwen3_tts.interface.generate_helpers import play_audio
        play_audio("/tmp/audio.wav")  # Should not raise

    @mock.patch("qwen3_tts.interface.generate_helpers.subprocess.run",
                side_effect=FileNotFoundError)
    @mock.patch("qwen3_tts.core.config.IS_MACOS", True)
    @mock.patch("qwen3_tts.core.config.IS_LINUX", False)
    @mock.patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_missing_player_warns(self, _mock_run):
        """Missing player binary logs a warning instead of crashing."""
        from qwen3_tts.interface.generate_helpers import play_audio
        play_audio("/tmp/audio.wav")  # Should not raise


# =========================================================================
# open_file
# =========================================================================

@pytest.mark.unit
class TestOpenFile(unittest.TestCase):
    """Tests for open_file — open file with system default handler."""

    @mock.patch("qwen3_tts.interface.generate_helpers.subprocess.run")
    @mock.patch("qwen3_tts.core.config.IS_MACOS", True)
    @mock.patch("qwen3_tts.core.config.IS_LINUX", False)
    @mock.patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_macos_uses_open(self, mock_run):
        """macOS calls 'open' command."""
        from qwen3_tts.interface.generate_helpers import open_file
        open_file("/tmp/file.wav")
        self.assertEqual(mock_run.call_args[0][0], ["open", "/tmp/file.wav"])

    @mock.patch("qwen3_tts.interface.generate_helpers.subprocess.run")
    @mock.patch("qwen3_tts.core.config.IS_MACOS", False)
    @mock.patch("qwen3_tts.core.config.IS_LINUX", True)
    @mock.patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_linux_uses_xdg_open(self, mock_run):
        """Linux calls xdg-open."""
        from qwen3_tts.interface.generate_helpers import open_file
        open_file("/tmp/file.wav")
        self.assertEqual(mock_run.call_args[0][0], ["xdg-open", "/tmp/file.wav"])

    @mock.patch("qwen3_tts.core.config.IN_COLAB", True)
    def test_colab_prints_path(self):
        """Colab prints file path instead of opening."""
        from qwen3_tts.interface.generate_helpers import open_file
        open_file("/tmp/file.wav")  # Should not raise

    @mock.patch("qwen3_tts.interface.generate_helpers.subprocess.run",
                side_effect=FileNotFoundError)
    @mock.patch("qwen3_tts.core.config.IS_MACOS", False)
    @mock.patch("qwen3_tts.core.config.IS_LINUX", True)
    @mock.patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_xdg_open_not_found(self, _mock_run):
        """Missing xdg-open logs warning instead of crashing."""
        from qwen3_tts.interface.generate_helpers import open_file
        open_file("/tmp/file.wav")  # Should not raise


# =========================================================================
# log_generation / show_history
# =========================================================================

@pytest.mark.unit
class TestLogGeneration(unittest.TestCase):
    """Tests for log_generation — append history entries."""

    def test_basic_entry(self):
        """Writes a valid JSON line to the history file."""
        from qwen3_tts.interface.generate_helpers import log_generation
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with mock.patch("qwen3_tts.interface.generate_helpers.HISTORY_FILE", path):
                log_generation("Hello", "clone", "voice.pt", "/out.wav", {"temperature": 0.7})
            with open(path) as f:
                entry = json.loads(f.readline())
            self.assertEqual(entry["mode"], "clone")
            self.assertEqual(entry["text"], "Hello")
            self.assertNotIn("duration_sec", entry)
        finally:
            os.unlink(path)

    def test_with_duration(self):
        """Includes rounded duration_sec when provided."""
        from qwen3_tts.interface.generate_helpers import log_generation
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with mock.patch("qwen3_tts.interface.generate_helpers.HISTORY_FILE", path):
                log_generation("Hi", "design", "desc", "/o.wav", {}, duration_sec=3.456)
            with open(path) as f:
                entry = json.loads(f.readline())
            self.assertEqual(entry["duration_sec"], 3.46)
        finally:
            os.unlink(path)

    def test_text_truncation(self):
        """Truncates text longer than 200 characters."""
        from qwen3_tts.interface.generate_helpers import log_generation
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            long_text = "A" * 300
            with mock.patch("qwen3_tts.interface.generate_helpers.HISTORY_FILE", path):
                log_generation(long_text, "clone", "v.pt", "/o.wav", {})
            with open(path) as f:
                entry = json.loads(f.readline())
            self.assertTrue(entry["text"].endswith("..."))
            self.assertEqual(len(entry["text"]), 203)  # 200 + "..."
            self.assertEqual(entry["text_length"], 300)
        finally:
            os.unlink(path)


@pytest.mark.unit
class TestShowHistory(unittest.TestCase):
    """Tests for show_history — display recent generation history."""

    def test_no_file(self):
        """Prints message when history file does not exist."""
        from qwen3_tts.interface.generate_helpers import show_history
        with mock.patch("qwen3_tts.interface.generate_helpers.HISTORY_FILE", "/nonexistent"):
            with mock.patch("builtins.print") as mock_print:
                show_history()
            mock_print.assert_called_with("No generation history found.")

    def test_empty_file(self):
        """Prints message when history file is empty."""
        from qwen3_tts.interface.generate_helpers import show_history
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with mock.patch("qwen3_tts.interface.generate_helpers.HISTORY_FILE", path):
                with mock.patch("builtins.print") as mock_print:
                    show_history()
            mock_print.assert_any_call("No generation history found.")
        finally:
            os.unlink(path)

    def test_entries_displayed(self):
        """Displays formatted entries from history file."""
        from qwen3_tts.interface.generate_helpers import show_history
        entry = {
            "timestamp": "2026-03-20T10:00:00.000",
            "text": "Hello world",
            "mode": "clone",
            "voice": "test.pt",
            "output": "/tmp/out.wav",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(entry) + "\n")
            path = f.name
        try:
            with mock.patch("qwen3_tts.interface.generate_helpers.HISTORY_FILE", path):
                with mock.patch("builtins.print") as mock_print:
                    show_history(count=5)
            output = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("clone", output)
            self.assertIn("test.pt", output)
        finally:
            os.unlink(path)


# =========================================================================
# get_voice_alias
# =========================================================================

@pytest.mark.unit
class TestGetVoiceAlias(unittest.TestCase):
    """Tests for get_voice_alias — resolve alias from config."""

    def test_found_alias(self):
        """Returns alias value when found."""
        from qwen3_tts.interface.generate_helpers import get_voice_alias
        config = {"aliases": {"default": {"prompt": "v.pt"}}}
        self.assertEqual(get_voice_alias("default", config), {"prompt": "v.pt"})

    def test_missing_alias(self):
        """Returns None when alias is not found."""
        from qwen3_tts.interface.generate_helpers import get_voice_alias
        config = {"aliases": {}}
        self.assertIsNone(get_voice_alias("missing", config))


# =========================================================================
# parse_ssml
# =========================================================================

@pytest.mark.unit
class TestParseSSML(unittest.TestCase):
    """Tests for parse_ssml — SSML tag processing."""

    def test_no_ssml(self):
        """Returns text unchanged when no SSML tags are present."""
        from qwen3_tts.interface.generate_helpers import parse_ssml
        text, meta = parse_ssml("Hello world")
        self.assertEqual(text, "Hello world")
        self.assertFalse(meta["has_ssml"])

    def test_break_ms_short(self):
        """<break time='200ms'/> produces '. ' pause marker."""
        from qwen3_tts.interface.generate_helpers import parse_ssml
        text, meta = parse_ssml('Hello<break time="200ms"/> world')
        self.assertIn(". ", text)
        self.assertTrue(meta["has_ssml"])

    def test_break_ms_long(self):
        """<break time='1500ms'/> produces '... ' pause marker."""
        from qwen3_tts.interface.generate_helpers import parse_ssml
        text, meta = parse_ssml('Hello<break time="1500ms"/> world')
        self.assertIn("... ", text)

    def test_break_seconds(self):
        """<break time='2s'/> produces '.... ' pause marker."""
        from qwen3_tts.interface.generate_helpers import parse_ssml
        text, meta = parse_ssml('Hello<break time="2s"/> world')
        self.assertIn(".... ", text)

    def test_sub_tag(self):
        """<sub alias='replacement'>original</sub> replaces content."""
        from qwen3_tts.interface.generate_helpers import parse_ssml
        text, meta = parse_ssml('<sub alias="World Wide Web">WWW</sub>')
        self.assertIn("World Wide Web", text)
        self.assertNotIn("WWW", text)

    def test_say_as_characters(self):
        """<say-as interpret-as='characters'>ABC</say-as> spells out as 'A B C'."""
        from qwen3_tts.interface.generate_helpers import parse_ssml
        text, meta = parse_ssml('<say-as interpret-as="characters">ABC</say-as>')
        self.assertIn("A B C", text)

    def test_emphasis_tag(self):
        """<emphasis> tags are stripped, content preserved."""
        from qwen3_tts.interface.generate_helpers import parse_ssml
        text, meta = parse_ssml('<emphasis>important</emphasis>')
        self.assertIn("important", text)
        self.assertNotIn("<emphasis", text)

    def test_prosody_rate_and_pitch(self):
        """<prosody rate='fast' pitch='low'> sets speed and pitch in metadata."""
        from qwen3_tts.interface.generate_helpers import parse_ssml
        text, meta = parse_ssml('<prosody rate="fast" pitch="low">Hello</prosody>')
        self.assertEqual(meta["prosody"]["speed"], 1.2)
        self.assertEqual(meta["prosody"]["pitch"], -2)


# =========================================================================
# process_ssml_text
# =========================================================================

@pytest.mark.unit
class TestProcessSSMLText(unittest.TestCase):
    """Tests for process_ssml_text — apply SSML prosody to args."""

    def test_prosody_applied_to_args(self):
        """Sets speed/pitch on args from SSML prosody metadata."""
        from qwen3_tts.interface.generate_helpers import process_ssml_text
        args = SimpleNamespace(speed=None, pitch=None)
        result = process_ssml_text('<prosody rate="slow" pitch="high">Hello</prosody>', args)
        self.assertEqual(args.speed, 0.8)
        self.assertEqual(args.pitch, 2)
        self.assertIn("Hello", result)

    def test_no_overwrite_if_args_set(self):
        """Does not overwrite existing speed/pitch on args."""
        from qwen3_tts.interface.generate_helpers import process_ssml_text
        args = SimpleNamespace(speed=1.5, pitch=-3)
        process_ssml_text('<prosody rate="slow" pitch="high">Hello</prosody>', args)
        self.assertEqual(args.speed, 1.5)
        self.assertEqual(args.pitch, -3)


# =========================================================================
# parse_srt / srt_time_to_ms
# =========================================================================

@pytest.mark.unit
class TestParseSrt(unittest.TestCase):
    """Tests for parse_srt — SRT subtitle file parsing."""

    def test_valid_srt_entries(self):
        """Parses standard SRT format into (index, start_ms, end_ms, text) tuples."""
        from qwen3_tts.interface.generate_helpers import parse_srt
        srt_content = (
            "1\n"
            "00:00:01,000 --> 00:00:04,000\n"
            "Hello world\n\n"
            "2\n"
            "00:01:00,500 --> 00:01:05,250\n"
            "Second line\n\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False,
                                         encoding="utf-8") as f:
            f.write(srt_content)
            path = f.name
        try:
            entries = parse_srt(path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0], (1, 1000, 4000, "Hello world"))
            self.assertEqual(entries[1], (2, 60500, 65250, "Second line"))
        finally:
            os.unlink(path)


@pytest.mark.unit
class TestSrtTimeToMs(unittest.TestCase):
    """Tests for srt_time_to_ms — SRT timestamp conversion."""

    def test_conversion(self):
        """Converts HH:MM:SS,mmm to total milliseconds."""
        from qwen3_tts.interface.generate_helpers import srt_time_to_ms
        self.assertEqual(srt_time_to_ms("00:00:01,000"), 1000)
        self.assertEqual(srt_time_to_ms("01:30:00,500"), 5400500)
        self.assertEqual(srt_time_to_ms("00:00:00,000"), 0)


# =========================================================================
# process_audio_args
# =========================================================================

@pytest.mark.unit
class TestProcessAudioArgs(unittest.TestCase):
    """Tests for process_audio_args — conditional audio post-processing."""

    def test_no_processing_needed(self):
        """Returns audio unchanged when no effects are requested."""
        from qwen3_tts.interface.generate_helpers import process_audio_args
        args = SimpleNamespace(trim_silence=False, normalize=False, speed=None, pitch=None)
        audio = [0.1, 0.2, 0.3]
        result = process_audio_args(audio, 24000, args)
        self.assertEqual(result, audio)

    @mock.patch("qwen3_tts.core.engine.process_audio", return_value=[0.5, 0.6])
    def test_with_processing(self, mock_process):
        """Delegates to engine.process_audio when effects are requested."""
        from qwen3_tts.interface.generate_helpers import process_audio_args
        args = SimpleNamespace(trim_silence=True, normalize=False, speed=None, pitch=None)
        result = process_audio_args([0.1, 0.2], 24000, args)
        mock_process.assert_called_once()
        self.assertEqual(result, [0.5, 0.6])


# =========================================================================
# _build_generation_payload
# =========================================================================

@pytest.mark.unit
class TestBuildGenerationPayload(unittest.TestCase):
    """Tests for _build_generation_payload — server request construction."""

    def test_clone_mode(self):
        """Clone mode includes prompt_file."""
        from qwen3_tts.interface.generate_helpers import _build_generation_payload
        payload = _build_generation_payload(
            "clone", {"language": "English"}, {"temperature": 0.7},
            prompt_file="voice.pt",
        )
        self.assertEqual(payload["mode"], "clone")
        self.assertEqual(payload["prompt_file"], "voice.pt")
        self.assertEqual(payload["language"], "English")

    def test_design_mode(self):
        """Design mode includes voice_description."""
        from qwen3_tts.interface.generate_helpers import _build_generation_payload
        payload = _build_generation_payload(
            "design", {"language": "English"}, {},
            voice_description="A calm voice",
        )
        self.assertEqual(payload["voice_description"], "A calm voice")

    def test_custom_mode(self):
        """Custom mode includes speaker and instruct."""
        from qwen3_tts.interface.generate_helpers import _build_generation_payload
        payload = _build_generation_payload(
            "custom", {}, {},
            speaker="Chelsie", instruct="Speak slowly",
        )
        self.assertEqual(payload["speaker"], "Chelsie")
        self.assertEqual(payload["instruct"], "Speak slowly")

    def test_with_max_chunk_chars(self):
        """Includes max_chunk_chars when provided."""
        from qwen3_tts.interface.generate_helpers import _build_generation_payload
        payload = _build_generation_payload("clone", {}, {}, max_chunk_chars=500)
        self.assertEqual(payload["max_chunk_chars"], 500)

    def test_clone_x_vector_only(self):
        """Clone mode with x_vector_only_mode sets flag in payload."""
        from qwen3_tts.interface.generate_helpers import _build_generation_payload
        payload = _build_generation_payload(
            "clone", {}, {}, prompt_file="v.pt", x_vector_only_mode=True,
        )
        self.assertTrue(payload["x_vector_only_mode"])


# =========================================================================
# _decode_base64_result / _save_base64_result
# =========================================================================

@pytest.mark.unit
class TestDecodeBase64Result(unittest.TestCase):
    """Tests for _decode_base64_result — base64 audio decoding roundtrip."""

    def test_roundtrip_decode(self):
        """Decodes base64-encoded WAV to numpy array + sample rate."""
        try:
            import numpy as np
            import soundfile as sf
        except ImportError:
            self.skipTest("requires numpy and soundfile")

        from qwen3_tts.interface.generate_helpers import _decode_base64_result

        # Create a tiny valid WAV in memory
        audio_data = np.zeros(100, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio_data, 24000, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode()

        wav, sr = _decode_base64_result({"audio_base64": b64})
        self.assertEqual(sr, 24000)
        self.assertEqual(len(wav), 100)


@pytest.mark.unit
class TestSaveBase64Result(unittest.TestCase):
    """Tests for _save_base64_result — write base64 audio to file."""

    def test_saves_to_file(self):
        """Decodes and writes raw bytes to output path."""
        from qwen3_tts.interface.generate_helpers import _save_base64_result
        raw_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt "
        b64 = base64.b64encode(raw_bytes).decode()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            _save_base64_result({"audio_base64": b64}, path)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), raw_bytes)
        finally:
            os.unlink(path)


# =========================================================================
# get_generation_params
# =========================================================================

@pytest.mark.unit
class TestGetGenerationParams(unittest.TestCase):
    """Tests for get_generation_params — merge config, preset, and CLI args."""

    def test_defaults_from_config(self):
        """Returns config defaults when no preset or arg overrides."""
        from qwen3_tts.interface.generate_helpers import get_generation_params
        args = SimpleNamespace(
            preset=None, temperature=None, top_k=None, top_p=None,
            repetition_penalty=None, seed=None,
        )
        config = {"generation": {"temperature": 0.7, "top_k": 50}}
        params = get_generation_params(args, config)
        self.assertEqual(params["temperature"], 0.7)
        self.assertEqual(params["top_k"], 50)

    def test_preset_override(self):
        """Preset values override config defaults."""
        from qwen3_tts.interface.generate_helpers import get_generation_params
        args = SimpleNamespace(
            preset="consistent", temperature=None, top_k=None, top_p=None,
            repetition_penalty=None, seed=None,
        )
        config = {
            "generation": {"temperature": 0.7, "top_k": 50},
            "presets": {"consistent": {"temperature": 0.5, "seed": 42}},
        }
        params = get_generation_params(args, config)
        self.assertEqual(params["temperature"], 0.5)
        self.assertEqual(params["seed"], 42)

    def test_arg_override(self):
        """CLI args override both config and preset."""
        from qwen3_tts.interface.generate_helpers import get_generation_params
        args = SimpleNamespace(
            preset="consistent", temperature=0.9, top_k=None, top_p=None,
            repetition_penalty=None, seed=None,
        )
        config = {
            "generation": {"temperature": 0.7},
            "presets": {"consistent": {"temperature": 0.5}},
        }
        params = get_generation_params(args, config)
        self.assertEqual(params["temperature"], 0.9)


if __name__ == "__main__":
    unittest.main()
