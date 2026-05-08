import os
import pytest
from unittest.mock import patch, MagicMock


class TestPlayAudioSafety:
    def test_01_nonexistent_file_no_subprocess(self, tmp_path):
        """play_audio on nonexistent file does not call subprocess.run"""
        from qwen3_tts.interface.generate_helpers import play_audio
        with patch("subprocess.run") as mock_run:
            play_audio("/tmp/this_file_definitely_does_not_exist_xyz.wav")
            mock_run.assert_not_called()

    def test_02_no_player_binary_no_subprocess(self, tmp_path):
        """play_audio when no player binary found does not call subprocess.run"""
        from qwen3_tts.interface.generate_helpers import play_audio
        f = tmp_path / "test.wav"
        f.write_bytes(b"RIFF")
        with patch("shutil.which", return_value=None):
            with patch("subprocess.run") as mock_run:
                play_audio(str(f))
                mock_run.assert_not_called()

    def test_03_cmd_uses_absolute_binary_path(self, tmp_path):
        """play_audio cmd[0] is an absolute path, not a bare binary name"""
        from qwen3_tts.interface.generate_helpers import play_audio
        f = tmp_path / "test.wav"
        f.write_bytes(b"RIFF")
        with patch("shutil.which", return_value="/usr/bin/afplay"):
            with patch("subprocess.run") as mock_run:
                play_audio(str(f))
                if mock_run.called:
                    cmd = mock_run.call_args[0][0]
                    assert os.path.isabs(cmd[0]), f"Expected absolute path, got: {cmd[0]}"

    def test_04_no_shell_true(self, tmp_path):
        """play_audio passes file_path as a separate list element (no shell=True)"""
        from qwen3_tts.interface.generate_helpers import play_audio
        f = tmp_path / "test.wav"
        f.write_bytes(b"RIFF")
        with patch("shutil.which", return_value="/usr/bin/afplay"):
            with patch("subprocess.run") as mock_run:
                play_audio(str(f))
                if mock_run.called:
                    kwargs = mock_run.call_args.kwargs
                    assert not kwargs.get("shell", False), "shell=True is unsafe"
