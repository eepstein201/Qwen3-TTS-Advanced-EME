#!/usr/bin/env python3
"""Extended tests for qwen3_tts.tools.healthcheck module.

Covers uncovered functions:
  - check_python_version: pass + fail
  - check_backend_availability: various platform combos
  - check_config: exists/valid, exists/invalid, missing
  - check_model_cache: no cache, empty, with models
  - check_voice_prompts: missing dir, empty, with files
  - check_server_status: running, stale PID, not running
  - check_audio_dependencies: ffmpeg found/missing
  - check_disk_space: OK, low, error
  - run_healthcheck: all pass, some fail
  - main: argparse entry point

Run: pytest tests/test_healthcheck_ext.py -v
"""
import os
import sys
import pathlib
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MOD = "qwen3_tts.tools.healthcheck"


class TestCheckPythonVersion(unittest.TestCase):

    def test_pass_current_version(self):
        from qwen3_tts.tools.healthcheck import check_python_version
        status, details = check_python_version()
        # We're running 3.11+ so this should always pass
        self.assertEqual(status, "pass")
        self.assertIn("Python", details)

    def test_fail_old_version(self):
        from qwen3_tts.tools.healthcheck import check_python_version
        mock_version = MagicMock()
        mock_version.major = 3
        mock_version.minor = 8
        mock_version.micro = 5
        with patch("sys.version_info", mock_version):
            status, details = check_python_version()
        self.assertEqual(status, "fail")
        self.assertIn("3.10+", details)


class TestCheckBackendAvailability(unittest.TestCase):

    def test_mlx_available(self):
        from qwen3_tts.tools.healthcheck import check_backend_availability
        with patch(f"{_MOD}.IS_MACOS", True), \
             patch(f"{_MOD}.IS_LINUX", False), \
             patch("platform.machine", return_value="arm64"):
            status, details = check_backend_availability()
        self.assertEqual(status, "pass")
        self.assertIn("mlx", details)

    def test_no_backends(self):
        from qwen3_tts.tools.healthcheck import check_backend_availability
        with patch(f"{_MOD}.IS_MACOS", False), \
             patch(f"{_MOD}.IS_LINUX", False), \
             patch.dict("sys.modules", {"torch": None}):
            status, details = check_backend_availability()
        # torch import raises since we set it to None
        self.assertIn("No backends", details)


class TestCheckConfig(unittest.TestCase):

    def test_config_exists_valid(self):
        from qwen3_tts.tools.healthcheck import check_config
        mock_config = {"advanced": {"backend": "mlx", "model_size": "1.7B"}}
        with patch(f"{_MOD}.CONFIG_PATH", "/tmp/fake_config.json"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("qwen3_tts.core.config.load_config", return_value=mock_config), \
             patch("qwen3_tts.core.config.validate_config", return_value=[]):
            status, details = check_config()
        self.assertEqual(status, "pass")
        self.assertIn("mlx", details)

    def test_config_exists_with_issues(self):
        from qwen3_tts.tools.healthcheck import check_config
        with patch(f"{_MOD}.CONFIG_PATH", "/tmp/fake_config.json"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("qwen3_tts.core.config.load_config", return_value={}), \
             patch("qwen3_tts.core.config.validate_config", return_value=["issue1", "issue2"]):
            status, details = check_config()
        self.assertEqual(status, "warn")
        self.assertIn("2 validation", details)

    def test_config_missing(self):
        from qwen3_tts.tools.healthcheck import check_config
        with patch(f"{_MOD}.CONFIG_PATH", "/nonexistent/config.json"), \
             patch("pathlib.Path.exists", return_value=False):
            status, details = check_config()
        self.assertEqual(status, "warn")
        self.assertIn("not found", details)

    def test_config_load_error(self):
        from qwen3_tts.tools.healthcheck import check_config
        with patch(f"{_MOD}.CONFIG_PATH", "/tmp/fake.json"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("qwen3_tts.core.config.load_config", side_effect=ValueError("bad json")), \
             patch("qwen3_tts.core.config.validate_config", return_value=[]):
            status, details = check_config()
        self.assertEqual(status, "fail")
        self.assertIn("error", details.lower())


class TestCheckModelCache(unittest.TestCase):

    def test_no_cache_dir(self):
        from qwen3_tts.tools.healthcheck import check_model_cache
        with patch(f"{_MOD}.HF_CACHE", pathlib.Path("/nonexistent_cache_xyz")):
            status, details = check_model_cache()
        self.assertEqual(status, "warn")

    def test_empty_cache(self, ):
        from qwen3_tts.tools.healthcheck import check_model_cache
        with patch(f"{_MOD}.HF_CACHE") as mock_cache:
            mock_cache.exists.return_value = True
            mock_cache.iterdir.return_value = []
            status, details = check_model_cache()
        self.assertEqual(status, "info")
        self.assertIn("No models", details)

    def test_cache_with_models(self, ):
        from qwen3_tts.tools.healthcheck import check_model_cache
        mock_dir = MagicMock()
        mock_dir.is_dir.return_value = True
        mock_dir.name = "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.stat.return_value = MagicMock(st_size=1024 * 1024 * 100)
        mock_dir.rglob.return_value = [mock_file]

        with patch(f"{_MOD}.HF_CACHE") as mock_cache:
            mock_cache.exists.return_value = True
            mock_cache.iterdir.return_value = [mock_dir]
            status, details = check_model_cache()
        self.assertEqual(status, "pass")
        self.assertIn("1 model", details)


class TestCheckVoicePrompts(unittest.TestCase):

    def test_missing_dir(self):
        from qwen3_tts.tools.healthcheck import check_voice_prompts
        with patch(f"{_MOD}.VOICE_PROMPTS_DIR", "/nonexistent_xyz"):
            status, details = check_voice_prompts()
        self.assertEqual(status, "warn")

    def test_empty_dir(self):
        from qwen3_tts.tools.healthcheck import check_voice_prompts
        with patch(f"{_MOD}.VOICE_PROMPTS_DIR"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.glob", return_value=[]):
            status, details = check_voice_prompts()
        self.assertEqual(status, "info")

    def test_with_files(self):
        import tempfile
        import shutil
        from qwen3_tts.tools.healthcheck import check_voice_prompts
        tmp_path = pathlib.Path(tempfile.mkdtemp())
        try:
            (tmp_path / "voice1.pt").write_text("data")
            (tmp_path / "voice1.wav").write_text("data")
            with patch(f"{_MOD}.VOICE_PROMPTS_DIR", str(tmp_path)):
                status, details = check_voice_prompts()
            self.assertEqual(status, "pass")
            self.assertIn("1 .pt", details)
            self.assertIn("1 .wav", details)
        finally:
            shutil.rmtree(tmp_path)


class TestCheckServerStatus(unittest.TestCase):

    def test_running(self):
        from qwen3_tts.tools.healthcheck import check_server_status
        state = {"running": True, "health_ok": True, "pid": 12345, "stale_pid": False}
        with patch(f"{_MOD}.detect_server_state", return_value=state):
            status, details = check_server_status()
        self.assertEqual(status, "pass")
        self.assertIn("12345", details)

    def test_stale_pid(self):
        from qwen3_tts.tools.healthcheck import check_server_status
        state = {"running": False, "health_ok": False, "pid": 99999, "stale_pid": True}
        with patch(f"{_MOD}.detect_server_state", return_value=state):
            status, details = check_server_status()
        self.assertEqual(status, "warn")
        self.assertIn("stale", details)

    def test_not_running(self):
        from qwen3_tts.tools.healthcheck import check_server_status
        state = {"running": False, "health_ok": False, "pid": None, "stale_pid": False}
        with patch(f"{_MOD}.detect_server_state", return_value=state):
            status, details = check_server_status()
        self.assertEqual(status, "info")


class TestCheckAudioDeps(unittest.TestCase):

    def test_all_found(self):
        from qwen3_tts.tools.healthcheck import check_audio_dependencies
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            status, details = check_audio_dependencies()
        self.assertEqual(status, "pass")

    def test_ffmpeg_missing(self):
        from qwen3_tts.tools.healthcheck import check_audio_dependencies
        with patch("subprocess.run", side_effect=FileNotFoundError):
            status, details = check_audio_dependencies()
        self.assertEqual(status, "warn")
        self.assertIn("ffmpeg", details.lower())


class TestCheckDiskSpace(unittest.TestCase):

    def test_plenty_of_space(self):
        from qwen3_tts.tools.healthcheck import check_disk_space
        mock_usage = MagicMock()
        mock_usage.free = 50 * (1024**3)
        mock_usage.total = 500 * (1024**3)
        with patch("shutil.disk_usage", return_value=mock_usage):
            status, details = check_disk_space()
        self.assertEqual(status, "pass")

    def test_low_space(self):
        from qwen3_tts.tools.healthcheck import check_disk_space
        mock_usage = MagicMock()
        mock_usage.free = 3 * (1024**3)
        mock_usage.total = 100 * (1024**3)
        with patch("shutil.disk_usage", return_value=mock_usage):
            status, details = check_disk_space()
        self.assertEqual(status, "warn")
        self.assertIn("Low", details)

    def test_moderate_space(self):
        from qwen3_tts.tools.healthcheck import check_disk_space
        mock_usage = MagicMock()
        mock_usage.free = 10 * (1024**3)
        mock_usage.total = 100 * (1024**3)
        with patch("shutil.disk_usage", return_value=mock_usage):
            status, details = check_disk_space()
        self.assertEqual(status, "warn")
        self.assertIn("Moderate", details)

    def test_error(self):
        from qwen3_tts.tools.healthcheck import check_disk_space
        with patch("shutil.disk_usage", side_effect=OSError("no access")):
            status, details = check_disk_space()
        self.assertEqual(status, "warn")


class TestRunHealthcheck(unittest.TestCase):

    def test_all_pass(self):
        from qwen3_tts.tools.healthcheck import run_healthcheck
        with patch(f"{_MOD}.check_python_version", return_value=("pass", "3.11")), \
             patch(f"{_MOD}.check_backend_availability", return_value=("pass", "mlx")), \
             patch(f"{_MOD}.check_config", return_value=("pass", "OK")), \
             patch(f"{_MOD}.check_model_cache", return_value=("pass", "1 model")), \
             patch(f"{_MOD}.check_voice_prompts", return_value=("pass", "2 files")), \
             patch(f"{_MOD}.check_server_status", return_value=("pass", "running")), \
             patch(f"{_MOD}.check_audio_dependencies", return_value=("pass", "OK")), \
             patch(f"{_MOD}.check_disk_space", return_value=("pass", "50GB")), \
             patch("builtins.print"):
            result = run_healthcheck()
        self.assertEqual(result, 0)

    def test_some_fail(self):
        from qwen3_tts.tools.healthcheck import run_healthcheck
        with patch(f"{_MOD}.check_python_version", return_value=("fail", "3.8")), \
             patch(f"{_MOD}.check_backend_availability", return_value=("pass", "mlx")), \
             patch(f"{_MOD}.check_config", return_value=("pass", "OK")), \
             patch(f"{_MOD}.check_model_cache", return_value=("pass", "1 model")), \
             patch(f"{_MOD}.check_voice_prompts", return_value=("pass", "2 files")), \
             patch(f"{_MOD}.check_server_status", return_value=("pass", "running")), \
             patch(f"{_MOD}.check_audio_dependencies", return_value=("pass", "OK")), \
             patch(f"{_MOD}.check_disk_space", return_value=("pass", "50GB")), \
             patch("builtins.print"):
            result = run_healthcheck()
        self.assertEqual(result, 1)


class TestMain(unittest.TestCase):

    def test_main_runs(self):
        from qwen3_tts.tools.healthcheck import main
        with patch(f"{_MOD}.run_healthcheck", return_value=0), \
             patch("sys.argv", ["healthcheck"]):
            result = main()
        self.assertEqual(result, 0)


class TestPrintHelpers(unittest.TestCase):

    def test_print_header(self):
        from qwen3_tts.tools.healthcheck import _print_header
        with patch("builtins.print") as mock_print:
            _print_header("Test")
        self.assertEqual(mock_print.call_count, 2)

    def test_print_check_pass(self):
        from qwen3_tts.tools.healthcheck import _print_check
        with patch("builtins.print") as mock_print:
            _print_check("Test", "pass", "details")
        self.assertEqual(mock_print.call_count, 2)

    def test_print_check_warn(self):
        from qwen3_tts.tools.healthcheck import _print_check
        with patch("builtins.print") as mock_print:
            _print_check("Test", "warn")
        self.assertEqual(mock_print.call_count, 1)

    def test_print_check_fail(self):
        from qwen3_tts.tools.healthcheck import _print_check
        with patch("builtins.print") as mock_print:
            _print_check("Test", "fail")
        self.assertEqual(mock_print.call_count, 1)

    def test_print_info(self):
        from qwen3_tts.tools.healthcheck import _print_info
        with patch("builtins.print") as mock_print:
            _print_info("Label", "details")
        self.assertEqual(mock_print.call_count, 2)

    def test_print_info_no_details(self):
        from qwen3_tts.tools.healthcheck import _print_info
        with patch("builtins.print") as mock_print:
            _print_info("Label", "")
        self.assertEqual(mock_print.call_count, 1)


if __name__ == "__main__":
    unittest.main()
