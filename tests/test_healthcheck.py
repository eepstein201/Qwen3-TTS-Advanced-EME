"""Tests for healthcheck.py module."""
import sys
import unittest
from io import StringIO
from unittest import mock


class TestCheckPythonVersion(unittest.TestCase):
    """Tests for check_python_version function."""

    def test_pass_for_current_python(self):
        """Python 3.10+ should pass."""
        from qwen3_tts.tools.healthcheck import check_python_version
        status, details = check_python_version()
        # Current Python should be 3.10+
        self.assertEqual(status, "pass")

    def test_details_contains_version(self):
        """Details should contain version info."""
        from qwen3_tts.tools.healthcheck import check_python_version
        status, details = check_python_version()
        self.assertIn("Python", details)


class TestCheckBackendAvailability(unittest.TestCase):
    """Tests for check_backend_availability function."""

    def test_includes_torch_when_available(self):
        """Returns torch in available backends."""
        from qwen3_tts.tools.healthcheck import check_backend_availability

        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        with mock.patch.dict(sys.modules, {'torch': mock_torch}):
            with mock.patch('qwen3_tts.tools.healthcheck.IS_MACOS', False):
                with mock.patch('qwen3_tts.tools.healthcheck.IS_LINUX', False):
                    status, details = check_backend_availability()
                    self.assertIn("torch", details)


class TestCheckConfig(unittest.TestCase):
    """Tests for check_config function."""

    def test_returns_warn_when_config_missing(self):
        """Returns warn when config file doesn't exist."""
        from qwen3_tts.tools.healthcheck import check_config

        with mock.patch('qwen3_tts.tools.healthcheck.CONFIG_PATH') as mock_path:
            mock_path.exists.return_value = False
            status, details = check_config()
            self.assertEqual(status, "warn")
            self.assertIn("not found", details.lower())


class TestCheckModelCache(unittest.TestCase):
    """Tests for check_model_cache function."""

    def test_returns_warn_when_cache_missing(self):
        """Returns warn when HuggingFace cache doesn't exist."""
        from qwen3_tts.tools.healthcheck import check_model_cache

        with mock.patch('qwen3_tts.tools.healthcheck.HF_CACHE') as mock_cache:
            mock_cache.exists.return_value = False
            status, details = check_model_cache()
            self.assertEqual(status, "warn")
            self.assertIn("not found", details.lower())

    def test_returns_info_when_no_models(self):
        """Returns info when cache exists but no TTS models."""
        from qwen3_tts.tools.healthcheck import check_model_cache

        with mock.patch('qwen3_tts.tools.healthcheck.HF_CACHE') as mock_cache:
            mock_cache.exists.return_value = True
            mock_cache.iterdir.return_value = []
            status, details = check_model_cache()
            self.assertEqual(status, "info")
            self.assertIn("No models cached", details)


class TestCheckVoicePrompts(unittest.TestCase):
    """Tests for check_voice_prompts function."""

    def test_returns_warn_when_dir_missing(self):
        """Returns warn when voice prompts directory doesn't exist."""
        from qwen3_tts.tools.healthcheck import check_voice_prompts

        mock_path = mock.MagicMock()
        mock_path.exists.return_value = False
        with mock.patch('qwen3_tts.tools.healthcheck.pathlib.Path', return_value=mock_path):
            status, details = check_voice_prompts()
            self.assertEqual(status, "warn")
            self.assertIn("not found", details.lower())

    def test_returns_info_when_empty(self):
        """Returns info when directory exists but is empty."""
        from qwen3_tts.tools.healthcheck import check_voice_prompts

        mock_path = mock.MagicMock()
        mock_path.exists.return_value = True
        mock_path.glob.return_value = []
        with mock.patch('qwen3_tts.tools.healthcheck.pathlib.Path', return_value=mock_path):
            status, details = check_voice_prompts()
            self.assertEqual(status, "info")
            self.assertIn("No voice prompts", details)


class TestCheckServerStatus(unittest.TestCase):
    """Tests for check_server_status function."""

    def test_returns_info_when_no_pid_file(self):
        """Returns info when nothing is running."""
        from qwen3_tts.tools.healthcheck import check_server_status

        mock_state = {
            "running": False, "health_ok": False,
            "pid": None, "pid_alive": False, "stale_pid": False,
        }
        with mock.patch('qwen3_tts.tools.healthcheck.detect_server_state', return_value=mock_state):
            status, details = check_server_status()
            self.assertEqual(status, "info")
            self.assertIn("not running", details.lower())

    def test_returns_warn_when_stale_pid(self):
        """Returns warn when PID file exists but process not running."""
        from qwen3_tts.tools.healthcheck import check_server_status

        mock_state = {
            "running": False, "health_ok": False,
            "pid": 99999, "pid_alive": False, "stale_pid": True,
        }
        with mock.patch('qwen3_tts.tools.healthcheck.detect_server_state', return_value=mock_state):
            status, details = check_server_status()
            self.assertEqual(status, "warn")
            self.assertIn("stale", details.lower())

    def test_returns_pass_when_running(self):
        """Returns pass when server is running."""
        from qwen3_tts.tools.healthcheck import check_server_status

        mock_state = {
            "running": True, "health_ok": True,
            "pid": 12345, "pid_alive": True, "stale_pid": False,
        }
        with mock.patch('qwen3_tts.tools.healthcheck.detect_server_state', return_value=mock_state):
            status, details = check_server_status()
            self.assertEqual(status, "pass")
            self.assertIn("running", details.lower())


class TestCheckAudioDependencies(unittest.TestCase):
    """Tests for check_audio_dependencies function."""

    def test_returns_pass_with_ffmpeg(self):
        """Returns pass when ffmpeg is available."""
        from qwen3_tts.tools.healthcheck import check_audio_dependencies

        mock_result = mock.MagicMock()
        mock_result.returncode = 0

        with mock.patch('subprocess.run', return_value=mock_result):
            status, details = check_audio_dependencies()
            self.assertEqual(status, "pass")
            self.assertIn("ffmpeg", details.lower())

    def test_returns_warn_without_ffmpeg(self):
        """Returns warn when ffmpeg is not available."""
        from qwen3_tts.tools.healthcheck import check_audio_dependencies

        with mock.patch('subprocess.run', side_effect=FileNotFoundError()):
            status, details = check_audio_dependencies()
            self.assertEqual(status, "warn")
            self.assertIn("ffmpeg not found", details)


class TestCheckDiskSpace(unittest.TestCase):
    """Tests for check_disk_space function."""

    def test_returns_pass_with_sufficient_space(self):
        """Returns pass when plenty of disk space."""
        from qwen3_tts.tools.healthcheck import check_disk_space

        mock_stat = mock.MagicMock()
        mock_stat.free = 50 * (1024**3)  # 50GB
        mock_stat.total = 100 * (1024**3)  # 100GB

        with mock.patch('shutil.disk_usage', return_value=mock_stat):
            status, details = check_disk_space()
            self.assertEqual(status, "pass")
            self.assertIn("50", details)

    def test_returns_warn_with_low_space(self):
        """Returns warn when disk space is low."""
        from qwen3_tts.tools.healthcheck import check_disk_space

        mock_stat = mock.MagicMock()
        mock_stat.free = 3 * (1024**3)  # 3GB
        mock_stat.total = 100 * (1024**3)

        with mock.patch('shutil.disk_usage', return_value=mock_stat):
            status, details = check_disk_space()
            self.assertEqual(status, "warn")
            self.assertIn("Low disk space", details)


class TestRunHealthcheck(unittest.TestCase):
    """Tests for run_healthcheck function."""

    def test_returns_zero_on_all_pass(self):
        """Returns 0 when all checks pass."""
        from qwen3_tts.tools.healthcheck import run_healthcheck

        with mock.patch('qwen3_tts.tools.healthcheck.check_python_version', return_value=("pass", "ok")):
            with mock.patch('qwen3_tts.tools.healthcheck.check_backend_availability', return_value=("pass", "ok")):
                with mock.patch('qwen3_tts.tools.healthcheck.check_config', return_value=("pass", "ok")):
                    with mock.patch('qwen3_tts.tools.healthcheck.check_model_cache', return_value=("pass", "ok")):
                        with mock.patch('qwen3_tts.tools.healthcheck.check_voice_prompts', return_value=("pass", "ok")):
                            with mock.patch('qwen3_tts.tools.healthcheck.check_server_status', return_value=("pass", "ok")):
                                with mock.patch('qwen3_tts.tools.healthcheck.check_audio_dependencies', return_value=("pass", "ok")):
                                    with mock.patch('qwen3_tts.tools.healthcheck.check_disk_space', return_value=("pass", "ok")):
                                        with mock.patch('builtins.print'):
                                            result = run_healthcheck()
                                            self.assertEqual(result, 0)

    def test_returns_one_on_failure(self):
        """Returns 1 when any check fails."""
        from qwen3_tts.tools.healthcheck import run_healthcheck

        with mock.patch('qwen3_tts.tools.healthcheck.check_python_version', return_value=("fail", "bad")):
            with mock.patch('qwen3_tts.tools.healthcheck.check_backend_availability', return_value=("pass", "ok")):
                with mock.patch('qwen3_tts.tools.healthcheck.check_config', return_value=("pass", "ok")):
                    with mock.patch('qwen3_tts.tools.healthcheck.check_model_cache', return_value=("pass", "ok")):
                        with mock.patch('qwen3_tts.tools.healthcheck.check_voice_prompts', return_value=("pass", "ok")):
                            with mock.patch('qwen3_tts.tools.healthcheck.check_server_status', return_value=("pass", "ok")):
                                with mock.patch('qwen3_tts.tools.healthcheck.check_audio_dependencies', return_value=("pass", "ok")):
                                    with mock.patch('qwen3_tts.tools.healthcheck.check_disk_space', return_value=("pass", "ok")):
                                        with mock.patch('builtins.print'):
                                            result = run_healthcheck()
                                            self.assertEqual(result, 1)


class TestPrintHelpers(unittest.TestCase):
    """Tests for print helper functions."""

    def test_print_header(self):
        """_print_header prints formatted header."""
        from qwen3_tts.tools.healthcheck import _print_header

        with mock.patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            _print_header("Test Section")
            output = mock_stdout.getvalue()
            self.assertIn("Test Section", output)

    def test_print_check_pass(self):
        """_print_check prints pass indicator."""
        from qwen3_tts.tools.healthcheck import _print_check

        with mock.patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            _print_check("Test check", "pass", "Details here")
            output = mock_stdout.getvalue()
            self.assertIn("Test check", output)
            self.assertIn("Details here", output)

    def test_print_info(self):
        """_print_info prints info item."""
        from qwen3_tts.tools.healthcheck import _print_info

        with mock.patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            _print_info("Info label", "Info details")
            output = mock_stdout.getvalue()
            self.assertIn("Info label", output)
            self.assertIn("Info details", output)
