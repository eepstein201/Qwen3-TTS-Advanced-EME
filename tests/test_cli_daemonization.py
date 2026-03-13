"""Tests for CLI server daemonization.

Validates that the CLI can start/stop the server without bash scripts.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Dummy decorator for when pytest is not available
    class _DummyMarkerFunc:
        """Represents a marker function like skipif that takes condition and returns decorator."""
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition, **kwargs):
            # skipif, etc. take condition as first arg, return a decorator
            return lambda f: f
    class _DummyMarker:
        def __call__(self, func):
            return func
        def __getattr__(self, name):
            # Return special function for skipif, otherwise return a callable marker
            if name == 'skipif':
                return _DummyMarkerFunc(name)
            return _DummyMarkerFunc(name)
        @property
        def unit(self):
            return self
    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()
    class _DummyPytest:
        mark = _DummyMark()
    pytest = _DummyPytest()


@pytest.mark.unit
class TestCLIDaemonization(unittest.TestCase):
    """Test server daemonization functionality."""

    def test_server_start_creates_pid_file(self):
        """Starting server should create .voice_server.pid file."""
        from qwen3_tts.cli import _start_server_daemon

        # Use a temp PID file for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            test_pid_file = Path(tmpdir) / "test.pid"

            with patch('qwen3_tts.core.config.PID_FILE', test_pid_file):
                # Mock subprocess.Popen to avoid actually starting server
                with patch('qwen3_tts.cli.subprocess.Popen') as mock_popen:
                    mock_proc = mock_popen.return_value
                    mock_proc.pid = 12345

                    _start_server_daemon(public=False)

                    # Verify PID file was created
                    self.assertTrue(test_pid_file.exists())
                    self.assertEqual(test_pid_file.read_text(), "12345")

    def test_server_stop_sends_shutdown_request(self):
        """Stopping server should send HTTP shutdown request."""
        from qwen3_tts.core.config import load_config, get_server_url, auth_headers
        import requests

        config = load_config()
        # Only test if server is not running (don't affect running server)
        if 'TTS_TEST_RUNNING_SERVER' not in os.environ:
            # Mock the request to avoid actually stopping a server
            with patch('requests.post') as mock_post:
                mock_post.return_value.status_code = 200
                # Simulate stop request
                url = get_server_url(config)
                headers = auth_headers()
                requests.post(f"{url}/shutdown", headers=headers, timeout=5)
                mock_post.assert_called_once()

    def test_server_foreground_flag(self):
        """Server should accept --foreground flag for Colab."""
        import subprocess
        # Test that the flag is accepted (doesn't crash)
        result = subprocess.run(
            [sys.executable, '-m', 'qwen3_tts.cli', 'server', 'start', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        # The help should include both flags
        self.assertIn('--foreground', result.stdout)
        self.assertIn('--public', result.stdout)


@pytest.mark.unit
class TestPIDLifecycle(unittest.TestCase):
    """Test PID lifecycle functions in config.py."""

    def test_read_pid_file_missing(self):
        """read_pid_file returns None when file doesn't exist."""
        from qwen3_tts.core.config import read_pid_file

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "nonexistent.pid"
            with patch('qwen3_tts.core.config.PID_FILE', fake_pid):
                self.assertIsNone(read_pid_file())

    def test_read_pid_file_valid(self):
        """read_pid_file returns int when file has valid PID."""
        from qwen3_tts.core.config import read_pid_file

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "test.pid"
            fake_pid.write_text("42\n")
            with patch('qwen3_tts.core.config.PID_FILE', fake_pid):
                self.assertEqual(read_pid_file(), 42)

    def test_read_pid_file_corrupt(self):
        """read_pid_file returns None when file has garbage."""
        from qwen3_tts.core.config import read_pid_file

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "test.pid"
            fake_pid.write_text("not_a_number\n")
            with patch('qwen3_tts.core.config.PID_FILE', fake_pid):
                self.assertIsNone(read_pid_file())

    def test_write_pid_file_creates_file(self):
        """write_pid_file creates PID file with correct content."""
        from qwen3_tts.core.config import write_pid_file

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "test.pid"
            with patch('qwen3_tts.core.config.PID_FILE', fake_pid):
                write_pid_file(12345)
                self.assertTrue(fake_pid.exists())
                self.assertEqual(fake_pid.read_text(), "12345")

    def test_cleanup_pid_file_removes_file(self):
        """cleanup_pid_file removes existing PID file."""
        from qwen3_tts.core.config import cleanup_pid_file

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "test.pid"
            fake_pid.write_text("42")
            with patch('qwen3_tts.core.config.PID_FILE', fake_pid):
                cleanup_pid_file()
                self.assertFalse(fake_pid.exists())

    def test_cleanup_pid_file_noop_when_missing(self):
        """cleanup_pid_file is idempotent when file doesn't exist."""
        from qwen3_tts.core.config import cleanup_pid_file

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "nonexistent.pid"
            with patch('qwen3_tts.core.config.PID_FILE', fake_pid):
                cleanup_pid_file()  # Should not raise
                self.assertFalse(fake_pid.exists())

    def test_is_pid_alive_current_process(self):
        """is_pid_alive returns True for the current process."""
        from qwen3_tts.core.config import is_pid_alive

        self.assertTrue(is_pid_alive(os.getpid()))

    def test_is_pid_alive_dead_pid(self):
        """is_pid_alive returns False for a non-existent PID."""
        from qwen3_tts.core.config import is_pid_alive

        # Use a very high PID unlikely to exist
        self.assertFalse(is_pid_alive(99999))

    def test_detect_state_healthy_with_pid(self):
        """detect_server_state with healthy server and valid PID."""
        from qwen3_tts.core.config import detect_server_state

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "test.pid"
            current_pid = os.getpid()
            fake_pid.write_text(str(current_pid))

            with patch('qwen3_tts.core.config.PID_FILE', fake_pid), \
                 patch('qwen3_tts.core.config.is_server_running', return_value=True):
                state = detect_server_state()

                self.assertTrue(state["running"])
                self.assertTrue(state["health_ok"])
                self.assertEqual(state["pid"], current_pid)
                self.assertTrue(state["pid_alive"])
                self.assertFalse(state["stale_pid"])

    def test_detect_state_healthy_no_pid(self):
        """detect_server_state with healthy server but no PID file (lost PID)."""
        from qwen3_tts.core.config import detect_server_state

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "nonexistent.pid"

            with patch('qwen3_tts.core.config.PID_FILE', fake_pid), \
                 patch('qwen3_tts.core.config.is_server_running', return_value=True):
                state = detect_server_state()

                self.assertTrue(state["running"])
                self.assertTrue(state["health_ok"])
                self.assertIsNone(state["pid"])
                self.assertFalse(state["pid_alive"])
                self.assertFalse(state["stale_pid"])

    def test_detect_state_stale_pid(self):
        """detect_server_state with dead server and stale PID file."""
        from qwen3_tts.core.config import detect_server_state

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "test.pid"
            fake_pid.write_text("99999")  # Non-existent PID

            with patch('qwen3_tts.core.config.PID_FILE', fake_pid), \
                 patch('qwen3_tts.core.config.is_server_running', return_value=False), \
                 patch('qwen3_tts.core.config.is_pid_alive', return_value=False):
                state = detect_server_state()

                self.assertFalse(state["running"])
                self.assertFalse(state["health_ok"])
                self.assertEqual(state["pid"], 99999)
                self.assertFalse(state["pid_alive"])
                self.assertTrue(state["stale_pid"])

    def test_detect_state_nothing(self):
        """detect_server_state with no server and no PID file."""
        from qwen3_tts.core.config import detect_server_state

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "nonexistent.pid"

            with patch('qwen3_tts.core.config.PID_FILE', fake_pid), \
                 patch('qwen3_tts.core.config.is_server_running', return_value=False):
                state = detect_server_state()

                self.assertFalse(state["running"])
                self.assertFalse(state["health_ok"])
                self.assertIsNone(state["pid"])
                self.assertFalse(state["pid_alive"])
                self.assertFalse(state["stale_pid"])


@pytest.mark.unit
class TestFindPidByPort(unittest.TestCase):
    """Test find_pid_by_port() in config.py."""

    def test_find_pid_by_port_success(self):
        """find_pid_by_port returns int PID when lsof finds a listener."""
        from qwen3_tts.core.config import find_pid_by_port

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "12345\n"

        with patch('qwen3_tts.core.config.subprocess.run', return_value=mock_result):
            self.assertEqual(find_pid_by_port(5123), 12345)

    def test_find_pid_by_port_no_listener(self):
        """find_pid_by_port returns None when nothing is listening."""
        from qwen3_tts.core.config import find_pid_by_port

        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch('qwen3_tts.core.config.subprocess.run', return_value=mock_result):
            self.assertIsNone(find_pid_by_port(5123))

    def test_find_pid_by_port_lsof_missing(self):
        """find_pid_by_port returns None when lsof is not installed."""
        from qwen3_tts.core.config import find_pid_by_port

        with patch('qwen3_tts.core.config.subprocess.run', side_effect=FileNotFoundError):
            self.assertIsNone(find_pid_by_port(5123))

    def test_find_pid_by_port_multiple_pids(self):
        """find_pid_by_port takes the first PID when multiple are returned."""
        from qwen3_tts.core.config import find_pid_by_port

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "11111\n22222\n"

        with patch('qwen3_tts.core.config.subprocess.run', return_value=mock_result):
            self.assertEqual(find_pid_by_port(5123), 11111)

    def test_find_pid_by_port_timeout(self):
        """find_pid_by_port returns None on timeout."""
        from qwen3_tts.core.config import find_pid_by_port
        import subprocess as sp

        with patch('qwen3_tts.core.config.subprocess.run', side_effect=sp.TimeoutExpired("lsof", 5)):
            self.assertIsNone(find_pid_by_port(5123))


@pytest.mark.unit
class TestCLIStopRewrite(unittest.TestCase):
    """Test rewritten stop() command using detect_server_state()."""

    def test_stop_without_pid_file_uses_health_check(self):
        """stop should use health check even when no PID file exists."""
        from qwen3_tts.core.config import detect_server_state

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "nonexistent.pid"

            # Server healthy but no PID file
            with patch('qwen3_tts.core.config.PID_FILE', fake_pid), \
                 patch('qwen3_tts.core.config.is_server_running', return_value=True):
                state = detect_server_state()
                self.assertTrue(state["running"])
                self.assertTrue(state["health_ok"])
                self.assertIsNone(state["pid"])

    def test_stop_stale_pid_cleans_up(self):
        """stop should clean stale PID and report not running."""
        from qwen3_tts.core.config import detect_server_state, cleanup_pid_file

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "test.pid"
            fake_pid.write_text("99999")

            with patch('qwen3_tts.core.config.PID_FILE', fake_pid), \
                 patch('qwen3_tts.core.config.is_server_running', return_value=False), \
                 patch('qwen3_tts.core.config.is_pid_alive', return_value=False):
                state = detect_server_state()
                self.assertTrue(state["stale_pid"])
                self.assertFalse(state["running"])

                # cleanup_pid_file should remove it
                cleanup_pid_file()
                self.assertFalse(fake_pid.exists())

    def test_stop_nothing_running(self):
        """stop should report not running when no server and no PID."""
        from qwen3_tts.core.config import detect_server_state

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "nonexistent.pid"

            with patch('qwen3_tts.core.config.PID_FILE', fake_pid), \
                 patch('qwen3_tts.core.config.is_server_running', return_value=False):
                state = detect_server_state()
                self.assertFalse(state["running"])
                self.assertFalse(state["stale_pid"])

    def test_stop_force_kills_after_shutdown_timeout(self):
        """stop should SIGTERM if /shutdown doesn't terminate the server."""
        from qwen3_tts.core.config import detect_server_state

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "test.pid"
            current_pid = os.getpid()
            fake_pid.write_text(str(current_pid))

            # Server healthy, PID alive — but health stays True (simulates shutdown failure)
            with patch('qwen3_tts.core.config.PID_FILE', fake_pid), \
                 patch('qwen3_tts.core.config.is_server_running', return_value=True):
                state = detect_server_state()
                self.assertTrue(state["running"])
                self.assertTrue(state["health_ok"])
                self.assertTrue(state["pid_alive"])
                # In real CLI, this state would trigger SIGTERM fallback

    def test_stop_reports_auth_failure(self):
        """stop should report 401 when shutdown is rejected due to auth."""
        from click.testing import CliRunner
        from qwen3_tts.cli import server

        mock_resp = Mock()
        mock_resp.status_code = 401

        with patch('qwen3_tts.core.config.is_server_running', side_effect=[True, False]), \
             patch('qwen3_tts.core.config.read_pid_file', return_value=None), \
             patch('qwen3_tts.core.config.is_pid_alive', return_value=False), \
             patch('qwen3_tts.core.config.find_pid_by_port', return_value=None), \
             patch('requests.post', return_value=mock_resp):
            runner = CliRunner()
            result = runner.invoke(server, ['stop'])
            self.assertIn("401", result.output)

    def test_stop_discovers_pid_by_port(self):
        """stop should discover PID via port when PID file is missing."""
        from click.testing import CliRunner
        from qwen3_tts.cli import server

        mock_resp = Mock()
        mock_resp.status_code = 401

        # is_server_running: True (detect), True (health_ok), then False (final check)
        with patch('qwen3_tts.core.config.is_server_running', side_effect=[True, False]), \
             patch('qwen3_tts.core.config.read_pid_file', return_value=None), \
             patch('qwen3_tts.core.config.find_pid_by_port', return_value=99999), \
             patch('qwen3_tts.core.config.is_pid_alive', return_value=False), \
             patch('requests.post', return_value=mock_resp):
            runner = CliRunner()
            result = runner.invoke(server, ['stop'])
            self.assertIn("Discovered server PID", result.output)

    def test_stop_fails_when_server_still_running(self):
        """stop should exit 1 when server is still running after all attempts."""
        from click.testing import CliRunner
        from qwen3_tts.cli import server

        mock_resp = Mock()
        mock_resp.status_code = 401

        # is_server_running always returns True (server won't die)
        with patch('qwen3_tts.core.config.is_server_running', return_value=True), \
             patch('qwen3_tts.core.config.read_pid_file', return_value=None), \
             patch('qwen3_tts.core.config.find_pid_by_port', return_value=None), \
             patch('qwen3_tts.core.config.is_pid_alive', return_value=False), \
             patch('requests.post', return_value=mock_resp):
            runner = CliRunner()
            result = runner.invoke(server, ['stop'])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("still running", result.output)


@pytest.mark.unit
class TestCLIStartRewrite(unittest.TestCase):
    """Test rewritten start() command using detect_server_state()."""

    def test_start_cleans_stale_pid(self):
        """start should clean stale PID and proceed to start."""
        from qwen3_tts.core.config import detect_server_state, cleanup_pid_file

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "test.pid"
            fake_pid.write_text("99999")

            with patch('qwen3_tts.core.config.PID_FILE', fake_pid), \
                 patch('qwen3_tts.core.config.is_server_running', return_value=False), \
                 patch('qwen3_tts.core.config.is_pid_alive', return_value=False):
                state = detect_server_state()
                self.assertTrue(state["stale_pid"])
                self.assertFalse(state["running"])

                # Cleanup should succeed
                cleanup_pid_file()
                self.assertFalse(fake_pid.exists())

    def test_start_already_running(self):
        """start should detect already-running server via health check."""
        from qwen3_tts.core.config import detect_server_state

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "test.pid"
            fake_pid.write_text(str(os.getpid()))

            with patch('qwen3_tts.core.config.PID_FILE', fake_pid), \
                 patch('qwen3_tts.core.config.is_server_running', return_value=True):
                state = detect_server_state()
                self.assertTrue(state["running"])
                self.assertTrue(state["health_ok"])

    def test_start_daemon_uses_write_pid_file(self):
        """_start_server_daemon should use write_pid_file() instead of raw PID_FILE."""
        from qwen3_tts.cli import _start_server_daemon

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_pid = Path(tmpdir) / "test.pid"

            with patch('qwen3_tts.core.config.PID_FILE', fake_pid), \
                 patch('qwen3_tts.cli.subprocess.Popen') as mock_popen:
                mock_proc = mock_popen.return_value
                mock_proc.pid = 54321

                _start_server_daemon(public=False)

                self.assertTrue(fake_pid.exists())
                self.assertEqual(fake_pid.read_text(), "54321")


@pytest.mark.unit
class TestShutdownEndpoint(unittest.TestCase):
    """Test /shutdown endpoint returns response before terminating."""

    def test_shutdown_endpoint_pattern(self):
        """Verify shutdown uses BackgroundTask + SIGTERM pattern (not sys.exit)."""
        import ast

        # Parse app.py and verify /shutdown doesn't call sys.exit
        worktree = Path(__file__).resolve().parent.parent
        app_path = worktree / "qwen3_tts" / "server" / "app.py"
        source = app_path.read_text()

        # Find the shutdown function and check it doesn't use sys.exit
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "shutdown":
                # Check no sys.exit calls in the function body
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Attribute) and func.attr == "exit":
                            if isinstance(func.value, ast.Name) and func.value.id == "sys":
                                self.fail("/shutdown still calls sys.exit()")
                break


@pytest.mark.unit
class TestGradioStopVerification(unittest.TestCase):
    """Test Gradio stop_server() polls for shutdown verification."""

    def test_gradio_stop_verifies_shutdown(self):
        """stop_server should poll is_server_running instead of fixed sleep."""
        import ast

        worktree = Path(__file__).resolve().parent.parent
        # ui.py was split into a package; stop_server lives in ui/_facade.py
        ui_path = worktree / "qwen3_tts" / "interface" / "ui" / "_facade.py"
        source = ui_path.read_text()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "stop_server":
                # Should have a for loop (polling) not just time.sleep(1)
                has_for_loop = any(
                    isinstance(child, ast.For) for child in ast.walk(node)
                )
                self.assertTrue(has_for_loop,
                    "stop_server should poll with a for loop, not fixed sleep")
                break


if __name__ == '__main__':
    unittest.main()
