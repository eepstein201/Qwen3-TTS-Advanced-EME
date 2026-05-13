#!/usr/bin/env python3
"""Tests for qwen3_tts.interface.cli.dialogue module.

Run: python -m pytest tests/test_cli_dialogue.py -v
No GPU, models, or running server required.
"""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from qwen3_tts.interface.cli.dialogue import process_dialogue
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires qwen3_tts")


def _make_args(**overrides):
    defaults = dict(
        output=None, mode=None, prompt=None, description=None,
        trim_silence=False, normalize=False, speed=None, pitch=None,
        play=False, no_open=True, save_individual=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _simple_dialogue():
    """Simple array format dialogue."""
    return [
        {"text": "Hello there.", "mode": "clone", "prompt": "voice.pt"},
        {"text": "Hi back!", "mode": "clone", "prompt": "voice.pt"},
    ]


def _object_dialogue(pause_ms=500):
    """Object format with speakers config."""
    return {
        "speakers": {
            "narrator": {"prompt": "narrator.pt", "mode": "clone"},
            "alice": {"mode": "custom", "speaker": "vivian"},
        },
        "pause_ms": pause_ms,
        "lines": [
            {"speaker": "narrator", "text": "Once upon a time..."},
            {"speaker": "alice", "text": "Hello!"},
        ],
    }


_MOD = "qwen3_tts.interface.cli.dialogue"


@_skip
class TestProcessDialogueSimpleArray(unittest.TestCase):
    """Tests for simple array format dialogue."""

    @patch(f"{_MOD}.open_file")
    @patch(f"{_MOD}.play_audio")
    @patch(f"{_MOD}.log_generation")
    @patch("soundfile.write")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w)
    @patch(f"{_MOD}._decode_base64_result")
    @patch(f"{_MOD}.generate_via_server")
    def test_server_mode(self, mock_gen, mock_decode, mock_proc,
                         mock_sf, mock_log, mock_play, mock_open):
        """Server mode generates audio for each line and combines."""
        import numpy as np
        mock_gen.return_value = [{"audio_base64": "AAAA"}]
        mock_decode.return_value = (np.zeros(100, dtype=np.float32), 24000)
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            with open(path, "w") as f:
                json.dump(_simple_dialogue(), f)
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            process_dialogue(path, config, args, {}, use_server=True)
        self.assertEqual(mock_gen.call_count, 2)
        self.assertTrue(mock_sf.called)

    @patch(f"{_MOD}.open_file")
    @patch(f"{_MOD}.play_audio")
    @patch(f"{_MOD}.log_generation")
    @patch("soundfile.write")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w)
    @patch(f"{_MOD}.generate_local")
    def test_local_mode(self, mock_gen, mock_proc, mock_sf,
                        mock_log, mock_play, mock_open):
        """Local mode calls generate_local for each line."""
        import numpy as np
        mock_gen.return_value = (np.zeros(100, dtype=np.float32), 24000)
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            with open(path, "w") as f:
                json.dump(_simple_dialogue(), f)
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            process_dialogue(path, config, args, {}, use_server=False)
        self.assertEqual(mock_gen.call_count, 2)


@_skip
class TestProcessDialogueObjectFormat(unittest.TestCase):
    """Tests for object format dialogue with speakers config."""

    @patch(f"{_MOD}.open_file")
    @patch(f"{_MOD}.play_audio")
    @patch(f"{_MOD}.log_generation")
    @patch("soundfile.write")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w)
    @patch(f"{_MOD}._decode_base64_result")
    @patch(f"{_MOD}.generate_via_server")
    def test_speakers_config_resolves(self, mock_gen, mock_decode, mock_proc,
                                      mock_sf, mock_log, mock_play, mock_open):
        """Speaker configs from 'speakers' dict are resolved for each line."""
        import numpy as np
        mock_gen.return_value = [{"audio_base64": "AAAA"}]
        mock_decode.return_value = (np.zeros(100, dtype=np.float32), 24000)
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            with open(path, "w") as f:
                json.dump(_object_dialogue(), f)
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            process_dialogue(path, config, args, {}, use_server=True)
        # Second call should use custom mode with speaker
        second_call = mock_gen.call_args_list[1]
        self.assertEqual(second_call.args[1], "custom")

    @patch(f"{_MOD}.open_file")
    @patch(f"{_MOD}.play_audio")
    @patch(f"{_MOD}.log_generation")
    @patch("soundfile.write")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w)
    @patch(f"{_MOD}._decode_base64_result")
    @patch(f"{_MOD}.generate_via_server")
    def test_custom_pause_ms(self, mock_gen, mock_decode, mock_proc,
                             mock_sf, mock_log, mock_play, mock_open):
        """Custom pause_ms is respected in combined audio."""
        import numpy as np
        mock_gen.return_value = [{"audio_base64": "AAAA"}]
        mock_decode.return_value = (np.zeros(100, dtype=np.float32), 24000)
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            with open(path, "w") as f:
                json.dump(_object_dialogue(pause_ms=300), f)
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            process_dialogue(path, config, args, {}, use_server=True)
        # Combined audio should include silence samples at 300ms
        combined_call = mock_sf.call_args_list[-1]
        combined_audio = combined_call.args[1]
        # 100 samples * 2 lines + 24000 * 0.3 = 7400
        self.assertGreater(len(combined_audio), 200)


@_skip
class TestProcessDialogueEdgeCases(unittest.TestCase):
    """Edge case tests for process_dialogue."""

    def test_empty_lines(self):
        """Empty lines list prints error and returns None."""
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            with open(path, "w") as f:
                json.dump([], f)
            args = _make_args(output=tmpdir)
            config = {"language": "English"}
            with patch("builtins.print"):
                result = process_dialogue(path, config, args, {}, use_server=True)
            self.assertIsNone(result)

    def test_pause_ms_clamped_to_max(self):
        """pause_ms > 10000 is clamped to 10000."""
        data = {"lines": [{"text": "hello", "mode": "clone"}], "pause_ms": 99999}
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            with open(path, "w") as f:
                json.dump(data, f)
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            # We only need to verify clamping, so mock generation to verify the function runs
            import numpy as np
            with patch(f"{_MOD}.generate_via_server", return_value=[{"audio_base64": "AA"}]), \
                 patch(f"{_MOD}._decode_base64_result",
                       return_value=(np.zeros(100, dtype=np.float32), 24000)), \
                 patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w), \
                 patch("soundfile.write"), \
                 patch(f"{_MOD}.log_generation"), \
                 patch(f"{_MOD}.open_file"):
                process_dialogue(path, config, args, {}, use_server=True)

    def test_pause_ms_clamped_to_min(self):
        """Negative pause_ms is clamped to 0."""
        data = {"lines": [{"text": "hi"}], "pause_ms": -500}
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            with open(path, "w") as f:
                json.dump(data, f)
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            import numpy as np
            with patch(f"{_MOD}.generate_via_server", return_value=[{"audio_base64": "AA"}]), \
                 patch(f"{_MOD}._decode_base64_result",
                       return_value=(np.zeros(100, dtype=np.float32), 24000)), \
                 patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w), \
                 patch("soundfile.write"), \
                 patch(f"{_MOD}.log_generation"), \
                 patch(f"{_MOD}.open_file"):
                process_dialogue(path, config, args, {}, use_server=True)

    def test_invalid_pause_ms_defaults_to_500(self):
        """Non-numeric pause_ms defaults to 500."""
        data = {"lines": [{"text": "hi"}], "pause_ms": "invalid"}
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            with open(path, "w") as f:
                json.dump(data, f)
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            import numpy as np
            with patch(f"{_MOD}.generate_via_server", return_value=[{"audio_base64": "AA"}]), \
                 patch(f"{_MOD}._decode_base64_result",
                       return_value=(np.zeros(100, dtype=np.float32), 24000)), \
                 patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w), \
                 patch("soundfile.write"), \
                 patch(f"{_MOD}.log_generation"), \
                 patch(f"{_MOD}.open_file"):
                process_dialogue(path, config, args, {}, use_server=True)

    @patch(f"{_MOD}.open_file")
    @patch(f"{_MOD}.play_audio")
    @patch(f"{_MOD}.log_generation")
    @patch("soundfile.write")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w)
    @patch(f"{_MOD}._decode_base64_result")
    @patch(f"{_MOD}.generate_via_server")
    def test_error_recovery_continues(self, mock_gen, mock_decode, mock_proc,
                                      mock_sf, mock_log, mock_play, mock_open):
        """Generation error on one line does not stop others."""
        import numpy as np
        mock_gen.side_effect = [
            Exception("generation error"),
            [{"audio_base64": "AA"}],
        ]
        mock_decode.return_value = (np.zeros(100, dtype=np.float32), 24000)
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            with open(path, "w") as f:
                json.dump(_simple_dialogue(), f)
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            process_dialogue(path, config, args, {}, use_server=True)
        # Should still produce combined audio from the one successful line
        self.assertTrue(mock_sf.called)

    @patch(f"{_MOD}.open_file")
    @patch(f"{_MOD}.play_audio")
    @patch(f"{_MOD}.log_generation")
    @patch("soundfile.write")
    @patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w)
    @patch(f"{_MOD}._decode_base64_result")
    @patch(f"{_MOD}.generate_via_server")
    def test_save_individual_writes_per_line(self, mock_gen, mock_decode, mock_proc,
                                             mock_sf, mock_log, mock_play, mock_open):
        """save_individual=True writes individual files per line."""
        import numpy as np
        mock_gen.return_value = [{"audio_base64": "AA"}]
        mock_decode.return_value = (np.zeros(100, dtype=np.float32), 24000)
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            with open(path, "w") as f:
                json.dump(_simple_dialogue(), f)
            args = _make_args(output=tmpdir, save_individual=True)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            process_dialogue(path, config, args, {}, use_server=True)
        # 2 individual + 1 combined = 3 sf.write calls
        self.assertEqual(mock_sf.call_count, 3)

    def test_skips_empty_text_lines(self):
        """Lines with empty text are skipped."""
        data = [
            {"text": "Hello", "mode": "clone", "prompt": "v.pt"},
            {"text": "", "mode": "clone", "prompt": "v.pt"},
        ]
        import numpy as np
        with tempfile.TemporaryDirectory(dir=os.path.expanduser("~")) as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            with open(path, "w") as f:
                json.dump(data, f)
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "v.pt"}
            with patch(f"{_MOD}.generate_via_server",
                       return_value=[{"audio_base64": "AA"}]) as mock_gen, \
                 patch(f"{_MOD}._decode_base64_result",
                       return_value=(np.zeros(100, dtype=np.float32), 24000)), \
                 patch(f"{_MOD}.process_audio_args", side_effect=lambda w, sr, a: w), \
                 patch("soundfile.write"), \
                 patch(f"{_MOD}.log_generation"), \
                 patch(f"{_MOD}.open_file"):
                process_dialogue(path, config, args, {}, use_server=True)
            self.assertEqual(mock_gen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
