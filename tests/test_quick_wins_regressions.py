"""Regression tests for Priority 4 quick-wins batch (R-31, R-32, R-39, R-40, R-41, R-47).

Each test is named after the bug it prevents — if it ever fails,
the exact roadmap item that introduced the fix is in the name.
"""
import os
import pathlib
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestR32CleanupPidTokenFileTOCTOU(unittest.TestCase):
    """R-32: cleanup_pid must not race on TOKEN_FILE deletion."""

    def test_r32_cleanup_pid_token_missing_is_noop(self):
        """cleanup_pid must not raise when TOKEN_FILE was already deleted."""
        import qwen3_tts.server.app_lifespan as al

        class FakeState:
            shutdown_timer = None
            shutdown_event = None

        fake_token = pathlib.Path("/tmp/_r32_test_nonexistent_token")
        if fake_token.exists():
            fake_token.unlink()

        with patch.object(al, "TOKEN_FILE", fake_token), \
             patch.object(al, "cleanup_pid_file"), \
             patch.object(al, "cleanup_resources"), \
             patch.object(al, "sys") as mock_sys:
            mock_sys.exit = MagicMock()
            al.cleanup_pid(FakeState())
            # Must not raise FileNotFoundError


class TestR39WritePidFileAtomic(unittest.TestCase):
    """R-39: write_pid_file must be atomic (temp-file + os.replace)."""

    def test_r39_write_pid_file_content_correct(self):
        """PID file must contain the correct PID after write."""
        import tempfile

        import qwen3_tts.core.config as cfg

        with tempfile.TemporaryDirectory() as tmp:
            pid_file = pathlib.Path(tmp) / ".voice_server.pid"
            with patch.object(cfg, "PID_FILE", pid_file):
                cfg.write_pid_file(99999)
                assert pid_file.read_text().strip() == "99999"

    def test_r39_no_temp_files_left_behind(self):
        """No .pid.tmp files should remain after write."""
        import tempfile

        import qwen3_tts.core.config as cfg

        with tempfile.TemporaryDirectory() as tmp:
            pid_file = pathlib.Path(tmp) / ".voice_server.pid"
            with patch.object(cfg, "PID_FILE", pid_file):
                cfg.write_pid_file(12345)
                tmp_files = list(pathlib.Path(tmp).glob("*.pid.tmp"))
                assert tmp_files == [], f"Temp files left behind: {tmp_files}"


class TestR40PreviewPromptServesRealPath(unittest.TestCase):
    """R-40: FileResponse must serve resolved real_path, not original wav_path."""

    def test_r40_preview_prompt_serves_real_path(self):
        """When a symlink exists, FileResponse must use the resolved path."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            # Resolve tmp itself (macOS /var → /private/var)
            tmp = os.path.realpath(tmp)
            # Create a real wav file with valid RIFF header
            real_wav = pathlib.Path(tmp) / "real.wav"
            real_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
            # Create symlink pointing to it (within same dir — valid)
            link = pathlib.Path(tmp) / "link.wav"
            link.symlink_to(real_wav)

            import qwen3_tts.server.app_prompts as ap

            with patch.object(ap, "VOICE_PROMPTS_DIR", tmp), \
                 patch.object(ap, "safe_path_join", side_effect=lambda d, f: os.path.join(str(d), f)):
                resp = ap.handle_preview_prompt("link")
                # FileResponse.path must be the real path, not the symlink
                assert resp.path == str(real_wav), (
                    f"Expected real_path {real_wav}, got {resp.path}"
                )


class TestR31SpeakerValidationCaseInsensitive(unittest.TestCase):
    """R-31: Speaker validation must be case-insensitive in the fallback check."""

    def test_r31_uppercase_speaker_accepted(self):
        """'RYAN' must pass validation just like 'ryan' and 'Ryan'."""
        from qwen3_tts.server.validation import _validate_generation_request

        req = MagicMock()
        req.mode = "custom"
        req.speaker = "RYAN"
        req.text = "hello"
        req.prompt_file = None
        req.voice_description = None
        req.language = "English"
        req.instruct = None
        req.x_vector_only_mode = False

        try:
            _validate_generation_request(req)
        except Exception as e:
            # HTTPException with 400 means validation rejected it
            if hasattr(e, "status_code") and e.status_code == 400 and "Unknown speaker" in str(e.detail):
                self.fail(f"RYAN should be valid but was rejected: {e.detail}")

    def test_r31_mixed_case_speaker_accepted(self):
        """'rYaN' must pass validation (case-insensitive)."""
        from qwen3_tts.server.validation import _validate_generation_request

        req = MagicMock()
        req.mode = "custom"
        req.speaker = "rYaN"
        req.text = "hello"
        req.prompt_file = None
        req.voice_description = None
        req.language = "English"
        req.instruct = None
        req.x_vector_only_mode = False

        try:
            _validate_generation_request(req)
        except Exception as e:
            if hasattr(e, "status_code") and e.status_code == 400 and "Unknown speaker" in str(e.detail):
                self.fail(f"rYaN should be valid but was rejected: {e.detail}")


class TestR47VoicePromptsDirIsPath(unittest.TestCase):
    """R-47: VOICE_PROMPTS_DIR must be pathlib.Path, not str."""

    def test_r47_voice_prompts_dir_is_pathlib_path(self):
        """VOICE_PROMPTS_DIR must be a pathlib.Path for consistency with PID_FILE/LOG_FILE."""
        from qwen3_tts.core.config import VOICE_PROMPTS_DIR
        assert isinstance(VOICE_PROMPTS_DIR, pathlib.Path), (
            f"VOICE_PROMPTS_DIR must be pathlib.Path, got {type(VOICE_PROMPTS_DIR).__name__}"
        )


class TestR41DeletePromptPartialFailure(unittest.TestCase):
    """R-41: delete_prompt must report partial failures per file."""

    def test_r41_delete_prompt_reports_failed_files(self):
        """If one file fails to delete, result should include files_failed."""
        import tempfile

        import qwen3_tts.server.app_prompts as ap

        with tempfile.TemporaryDirectory() as tmp:
            # Create .pt and .wav files
            pt_file = pathlib.Path(tmp) / "test.pt"
            wav_file = pathlib.Path(tmp) / "test.wav"
            pt_file.write_bytes(b"data")
            wav_file.write_bytes(b"data")

            original_remove = os.remove

            def failing_remove(path):
                if str(path).endswith(".pt"):
                    raise OSError("Permission denied")
                original_remove(path)

            fake_state = MagicMock()
            with patch.object(ap, "VOICE_PROMPTS_DIR", str(tmp)), \
                 patch.object(ap, "safe_path_join", side_effect=lambda d, f: os.path.join(str(d), f)), \
                 patch("os.remove", side_effect=failing_remove), \
                 patch.object(ap, "clear_voice_prompt_cache", create=True):
                # Patch the inline import too
                with patch.dict("sys.modules", {"qwen3_tts.core.engine": MagicMock()}):
                    fake_req = MagicMock()
                    fake_req.name = "test"
                    result = ap.handle_delete_prompt(fake_state, fake_req, config_fn=lambda: {})
                    assert "files_failed" in result, "Result must include files_failed on partial failure"
                    assert "test.pt" in result["files_failed"]


if __name__ == "__main__":
    unittest.main()
