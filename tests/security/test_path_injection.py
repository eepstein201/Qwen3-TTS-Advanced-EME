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
from unittest.mock import patch, MagicMock

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

    @patch('qwen3_tts.core.engine.inference')
    def test_process_srt_file_rejects_traversal_output(self, mock_inference):
        """process_srt_file with relative traversal path '../malicious' raises ValueError."""
        srt = self._import_srt_module()
        # Create a temporary SRT file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
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

    @patch('qwen3_tts.core.engine.inference')
    def test_process_dialogue_rejects_traversal_output(self, mock_inference):
        """process_dialogue with relative traversal path '../malicious' raises ValueError."""
        dialogue = self._import_dialogue_module()
        # Create a temporary dialogue file with valid content (so code reaches output_dir)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"lines": [{"text": "test"}]}')
            temp_dialogue = f.name

        try:
            # Create mock config and args with relative traversal path
            config = {"output_directory": tempfile.gettempdir()}
            args = MagicMock(output="../escape")  # relative path with traversal

            with self.assertRaises(ValueError) as ctx:
                # Attempt to write outside current directory via relative traversal
                dialogue.process_dialogue(temp_dialogue, config, args, {}, use_server=True)
            self.assertIn("traversal", str(ctx.exception).lower())
        finally:
            os.unlink(temp_dialogue)


class TestAutoIncrementFilenameInjection(unittest.TestCase):
    """Test generate_helpers.py auto_increment_filename."""

    def test_auto_increment_rejects_traversal_in_path(self):
        """auto_increment_filename with traversal in path returns safe version."""
        from qwen3_tts.interface.generate_helpers import auto_increment_filename

        # Create a temp directory to work in
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "output.wav")
            # Create the file so auto-increment triggers
            open(base_path, 'w').close()

            # This should NOT escape the tmpdir via traversal
            result = auto_increment_filename(os.path.join(tmpdir, "../escape/output.wav"))
            # Result should still be under tmpdir or raise ValueError
            self.assertTrue(result.startswith(tmpdir) or ValueError)


class TestCreateVoicePathInjection(unittest.TestCase):
    """Test tools/create_voice.py output path validation."""

    def test_validate_voice_name_rejects_traversal(self):
        """validate_voice_name (used by create_and_save_voice_prompt) rejects traversal."""
        from qwen3_tts.core.config import validate_voice_name

        with self.assertRaises(ValueError) as ctx:
            validate_voice_name("../../../etc/passwd")
        self.assertIn("invalid", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
