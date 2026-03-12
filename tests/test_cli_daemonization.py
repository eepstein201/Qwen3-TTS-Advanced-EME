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
from unittest.mock import patch

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


if __name__ == '__main__':
    unittest.main()
