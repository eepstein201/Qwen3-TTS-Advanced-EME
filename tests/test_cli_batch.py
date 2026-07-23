#!/usr/bin/env python3
"""Tests for qwen3_tts.interface.cli.batch module.

Run: python -m pytest tests/test_cli_batch.py -v
No GPU, models, or running server required.
"""
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from qwen3_tts.interface.cli.batch import load_batch_file, process_batch
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires qwen3_tts")


def _make_args(**overrides):
    defaults = dict(
        output=None, mode=None, prompt=None, description=None,
        trim_silence=False, normalize=False, speed=None, pitch=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@_skip
class TestLoadBatchFile(unittest.TestCase):
    """Tests for load_batch_file()."""

    def test_valid_json_array(self):
        """Valid JSON array returns list of texts."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["hello", "world"], f)
            path = f.name
        try:
            result = load_batch_file(path)
            self.assertEqual(result, ["hello", "world"])
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        """Missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_batch_file("/tmp/nonexistent_batch_file.json")

    def test_non_array_raises(self):
        """Non-array JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_batch_file(path)
        finally:
            os.unlink(path)

    def test_invalid_json_raises(self):
        """Invalid JSON raises json.JSONDecodeError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json at all")
            path = f.name
        try:
            with self.assertRaises(json.JSONDecodeError):
                load_batch_file(path)
        finally:
            os.unlink(path)

    def test_empty_array(self):
        """Empty JSON array returns empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            path = f.name
        try:
            result = load_batch_file(path)
            self.assertEqual(result, [])
        finally:
            os.unlink(path)

    def test_tilde_expansion(self):
        """Path with ~ is expanded."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                         dir=os.path.expanduser("~")) as f:
            json.dump(["test"], f)
            full_path = f.name
        basename = os.path.basename(full_path)
        try:
            result = load_batch_file(f"~/{basename}")
            self.assertEqual(result, ["test"])
        finally:
            os.unlink(full_path)


@_skip
class TestProcessBatch(unittest.TestCase):
    """Tests for process_batch()."""

    @patch("qwen3_tts.interface.generate.generate_via_server")
    @patch("soundfile.write")
    def test_server_mode_no_processing(self, mock_sf, mock_gen):
        """Server mode without audio processing saves base64 results directly."""
        mock_gen.return_value = [
            {"audio_base64": "AAAA"},
            {"audio_base64": "BBBB"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            with patch("qwen3_tts.interface.generate._save_base64_result") as mock_save:
                result = process_batch(["hello", "world"], args, config, {}, use_server=True)
            self.assertEqual(len(result), 2)
            self.assertEqual(mock_save.call_count, 2)

    @patch("qwen3_tts.interface.generate.generate_via_server")
    @patch("soundfile.write")
    @patch("qwen3_tts.interface.generate._decode_base64_result")
    @patch("qwen3_tts.interface.generate.process_audio_args")
    def test_server_mode_with_processing(self, mock_proc, mock_decode, mock_sf, mock_gen):
        """Server mode with trim_silence triggers audio processing pipeline."""
        import numpy as np
        mock_gen.return_value = [{"audio_base64": "AAAA"}]
        mock_decode.return_value = (np.zeros(100), 24000)
        mock_proc.return_value = np.zeros(100)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(output=tmpdir, trim_silence=True)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            result = process_batch(["hello"], args, config, {}, use_server=True)
            self.assertEqual(len(result), 1)
            mock_decode.assert_called_once()
            mock_proc.assert_called_once()

    @patch("qwen3_tts.interface.generate.generate_local")
    @patch("qwen3_tts.interface.generate.process_audio_args")
    @patch("soundfile.write")
    def test_local_mode(self, mock_sf, mock_proc, mock_gen):
        """Local mode calls generate_local for each text."""
        import numpy as np
        mock_gen.return_value = (np.zeros(100), 24000)
        mock_proc.return_value = np.zeros(100)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(output=tmpdir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            result = process_batch(["a", "b"], args, config, {}, use_server=False)
            self.assertEqual(len(result), 2)
            self.assertEqual(mock_gen.call_count, 2)

    @patch("qwen3_tts.interface.generate.generate_via_server")
    @patch("soundfile.write")
    def test_creates_output_dir(self, mock_sf, mock_gen):
        """Missing output directory is created automatically."""
        mock_gen.return_value = [{"audio_base64": "AAAA"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "new_subdir")
            args = _make_args(output=new_dir)
            config = {"language": "English", "default_clone_prompt": "voice.pt"}
            with patch("qwen3_tts.interface.generate._save_base64_result"):
                process_batch(["hello"], args, config, {}, use_server=True)
            self.assertTrue(os.path.isdir(new_dir))

    @patch("qwen3_tts.interface.generate.generate_via_server")
    @patch("soundfile.write")
    def test_design_mode_passes_description(self, mock_sf, mock_gen):
        """Design mode passes voice_description to server."""
        mock_gen.return_value = [{"audio_base64": "AAAA"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            args = _make_args(output=tmpdir, mode="design", description="warm voice")
            config = {"language": "English", "default_voice_description": "fallback"}
            with patch("qwen3_tts.interface.generate._save_base64_result"):
                process_batch(["hello"], args, config, {}, use_server=True)
            call_kwargs = mock_gen.call_args
            self.assertEqual(call_kwargs.kwargs.get("voice_description"), "warm voice")
            self.assertIsNone(call_kwargs.kwargs.get("prompt_file"))


if __name__ == "__main__":
    unittest.main()
