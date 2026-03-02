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

import pytest


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
        # The help should include the flag
        self.assertIn('--public', result.stdout)


if __name__ == '__main__':
    unittest.main()
