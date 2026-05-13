#!/usr/bin/env python3
"""Tests for qwen3_tts.interface.cli.srt module.

Run: python -m pytest tests/test_cli_srt.py -v
No GPU, models, or running server required.
"""
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from qwen3_tts.interface.cli.srt import process_srt_file
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires qwen3_tts")

_MOD = "qwen3_tts.interface.cli.srt"


def _make_args(**overrides):
    defaults = dict(
        output=None, mode=None, prompt=None, description=None,
        trim_silence=False, normalize=False, speed=None, pitch=None,
        play=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _write_srt(tmpdir, content=None):
    """Write a sample SRT file and return its path."""
    if content is None:
        content = (
            "1\n"
            "00:00:01,000 --> 00:00:04,000\n"
            "First subtitle\n\n"
            "2\n"
            "00:00:05,000 --> 00:00:08,000\n"
            "Second subtitle\n\n"
        )
    path = os.path.join(tmpdir, "test.srt")
    with open(path, "w") as f:
        f.write(content)
    return path


@_skip
class TestProcessSrtServerMode(unittest.TestCase):
    """Tests for process_srt_file in server mode."""

    @patch(f"{_MOD}.play_audio")
    @patch("soundfile.write")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w)
    @patch(f"{_MOD}._decode_base64_result")
    @patch(f"{_MOD}.generate_via_server")
    def test_generates_per_subtitle(self, mock_gen, mock_decode, mock_proc,
                                    mock_sf, mock_play):
        """Calls generate_via_server once per subtitle entry."""
        import numpy as np
        mock_gen.return_value = [{"audio_base64": "AA"}]
        mock_decode.return_value = (np.zeros(100, dtype=np.float32), 24000)
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            srt_path = _write_srt(tmpdir)
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            process_srt_file(srt_path, config, args, {}, use_server=True)
        self.assertEqual(mock_gen.call_count, 2)

    @patch(f"{_MOD}.play_audio")
    @patch("soundfile.write")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w)
    @patch(f"{_MOD}._decode_base64_result")
    @patch(f"{_MOD}.generate_via_server")
    def test_writes_individual_and_combined(self, mock_gen, mock_decode,
                                            mock_proc, mock_sf, mock_play):
        """Writes one file per subtitle plus a combined file."""
        import numpy as np
        mock_gen.return_value = [{"audio_base64": "AA"}]
        mock_decode.return_value = (np.zeros(100, dtype=np.float32), 24000)
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            srt_path = _write_srt(tmpdir)
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            process_srt_file(srt_path, config, args, {}, use_server=True)
        # 2 individual + 1 combined = 3 sf.write calls
        self.assertEqual(mock_sf.call_count, 3)

    @patch(f"{_MOD}.play_audio")
    @patch("soundfile.write")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w)
    @patch(f"{_MOD}._decode_base64_result")
    @patch(f"{_MOD}.generate_via_server")
    def test_combined_file_named_correctly(self, mock_gen, mock_decode,
                                           mock_proc, mock_sf, mock_play):
        """Combined file is named {basename}_combined.wav."""
        import numpy as np
        mock_gen.return_value = [{"audio_base64": "AA"}]
        mock_decode.return_value = (np.zeros(100, dtype=np.float32), 24000)
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            srt_path = _write_srt(tmpdir)
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            process_srt_file(srt_path, config, args, {}, use_server=True)
        # Last sf.write call should be the combined file
        combined_path = mock_sf.call_args_list[-1].args[0]
        self.assertTrue(combined_path.endswith("test_combined.wav"))


@_skip
class TestProcessSrtLocalMode(unittest.TestCase):
    """Tests for process_srt_file in local mode."""

    @patch(f"{_MOD}.play_audio")
    @patch("soundfile.write")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w)
    @patch(f"{_MOD}.generate_local")
    def test_local_mode_calls_generate_local(self, mock_gen, mock_proc,
                                             mock_sf, mock_play):
        """Local mode calls generate_local for each subtitle."""
        import numpy as np
        mock_gen.return_value = (np.zeros(100, dtype=np.float32), 24000)
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            srt_path = _write_srt(tmpdir)
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            process_srt_file(srt_path, config, args, {}, use_server=False)
        self.assertEqual(mock_gen.call_count, 2)


@_skip
class TestProcessSrtEdgeCases(unittest.TestCase):
    """Edge case tests for process_srt_file."""

    def test_empty_srt_prints_error(self):
        """Empty SRT file prints error and returns None."""
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            srt_path = _write_srt(tmpdir, content="")
            args = _make_args(output=tmpdir)
            config = {"language": "English"}
            with patch("builtins.print"):
                result = process_srt_file(srt_path, config, args, {}, use_server=True)
            self.assertIsNone(result)

    @patch(f"{_MOD}.play_audio")
    @patch("soundfile.write")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w)
    @patch(f"{_MOD}._decode_base64_result")
    @patch(f"{_MOD}.generate_via_server")
    def test_creates_output_dir(self, mock_gen, mock_decode, mock_proc,
                                mock_sf, mock_play):
        """Missing output directory is created automatically."""
        import numpy as np
        mock_gen.return_value = [{"audio_base64": "AA"}]
        mock_decode.return_value = (np.zeros(100, dtype=np.float32), 24000)
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            srt_path = _write_srt(tmpdir)
            new_dir = os.path.join(tmpdir, "subdir")
            args = _make_args(output=new_dir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            process_srt_file(srt_path, config, args, {}, use_server=True)
            self.assertTrue(os.path.isdir(new_dir))

    @patch(f"{_MOD}.play_audio")
    @patch("soundfile.write")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w)
    @patch(f"{_MOD}._decode_base64_result")
    @patch(f"{_MOD}.generate_via_server")
    def test_play_flag_plays_audio(self, mock_gen, mock_decode, mock_proc,
                                   mock_sf, mock_play):
        """play=True calls play_audio on combined file."""
        import numpy as np
        mock_gen.return_value = [{"audio_base64": "AA"}]
        mock_decode.return_value = (np.zeros(100, dtype=np.float32), 24000)
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            srt_path = _write_srt(tmpdir)
            args = _make_args(output=tmpdir, play=True)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            process_srt_file(srt_path, config, args, {}, use_server=True)
        mock_play.assert_called_once()

    @patch(f"{_MOD}.play_audio")
    @patch("soundfile.write")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w)
    @patch(f"{_MOD}._decode_base64_result")
    @patch(f"{_MOD}.generate_via_server")
    def test_design_mode_passes_description(self, mock_gen, mock_decode,
                                            mock_proc, mock_sf, mock_play):
        """Design mode passes voice_description to server."""
        import numpy as np
        mock_gen.return_value = [{"audio_base64": "AA"}]
        mock_decode.return_value = (np.zeros(100, dtype=np.float32), 24000)
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            srt_path = _write_srt(tmpdir)
            args = _make_args(output=tmpdir, mode="design", description="warm voice")
            config = {"language": "English", "default_voice_description": "default"}
            process_srt_file(srt_path, config, args, {}, use_server=True)
        call_kwargs = mock_gen.call_args
        self.assertEqual(call_kwargs.kwargs.get("voice_description"), "warm voice")


if __name__ == "__main__":
    unittest.main()
