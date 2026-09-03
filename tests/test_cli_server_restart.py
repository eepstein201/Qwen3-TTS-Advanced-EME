"""Tests for tts server restart command."""

import subprocess
import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from qwen3_tts.cli_server import server


class TestServerRestart(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @patch("qwen3_tts.cli_server._start_server_daemon")
    @patch("qwen3_tts.cli_server.load_config", create=True)
    def test_restart_when_not_running_starts_server(self, mock_load, mock_start):
        mock_load.return_value = {"server": {"port": 5123}}
        mock_proc = MagicMock()
        mock_proc.pid = 9999
        mock_start.return_value = mock_proc

        with patch(
            "qwen3_tts.core.config.is_server_running", return_value=False
        ), patch(
            "qwen3_tts.core.config.pm2_owner_of_port", return_value=None,
        ), patch(
            "qwen3_tts.core.config.detect_server_state",
            return_value={"running": False, "stale_pid": False, "pid": None,
                          "health_ok": False},
        ), patch("qwen3_tts.core.config.load_config", return_value={"server": {"port": 5123}}):
            result = self.runner.invoke(server, ["restart"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Server was not running", result.output)
        self.assertIn("9999", result.output)
        mock_start.assert_called_once()

    @patch("qwen3_tts.cli_server._start_server_daemon")
    def test_restart_when_running_stops_then_starts(self, mock_start):
        mock_proc = MagicMock()
        mock_proc.pid = 8888
        mock_start.return_value = mock_proc

        # First call (restart check): running. Second (stop verify): not running.
        # Third+ (restart post-stop poll): not running.
        running_returns = iter([True, False, False, False, False])

        with patch(
            "qwen3_tts.core.config.is_server_running",
            side_effect=lambda _c: next(running_returns, False),
        ), patch(
            "qwen3_tts.core.config.load_config",
            return_value={"server": {"port": 5123}},
        ), patch(
            "qwen3_tts.core.config.pm2_owner_of_port", return_value=None,
        ), patch(
            "qwen3_tts.core.config.detect_server_state",
            return_value={"running": True, "stale_pid": False, "pid": 1234,
                          "health_ok": True},
        ), patch(
            "qwen3_tts.core.config.cleanup_pid_file",
        ), patch(
            "qwen3_tts.core.config.find_pid_by_port", return_value=None,
        ), patch(
            "qwen3_tts.core.config.is_pid_alive", return_value=False,
        ), patch(
            "qwen3_tts.core.http_client.server_request",
        ) as mock_req, patch(
            "qwen3_tts.cli_server.time.sleep",
        ):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_req.return_value = mock_resp

            result = self.runner.invoke(server, ["restart"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("8888", result.output)
        mock_start.assert_called_once()

    @patch("qwen3_tts.cli_server._start_server_daemon")
    def test_restart_public_flag_passed(self, mock_start):
        mock_proc = MagicMock()
        mock_proc.pid = 7777
        mock_start.return_value = mock_proc

        with patch(
            "qwen3_tts.core.config.is_server_running", return_value=False,
        ), patch(
            "qwen3_tts.core.config.pm2_owner_of_port", return_value=None,
        ), patch(
            "qwen3_tts.core.config.load_config",
            return_value={"server": {"port": 5123}},
        ), patch(
            "qwen3_tts.core.config.detect_server_state",
            return_value={"running": False, "stale_pid": False, "pid": None,
                          "health_ok": False},
        ):
            result = self.runner.invoke(server, ["restart", "--public"])

        self.assertEqual(result.exit_code, 0, result.output)
        mock_start.assert_called_once_with(public=True)

    def test_restart_fails_if_server_wont_stop(self):
        with patch(
            "qwen3_tts.core.config.is_server_running", return_value=True,
        ), patch(
            "qwen3_tts.core.config.pm2_owner_of_port", return_value=None,
        ), patch(
            "qwen3_tts.core.config.load_config",
            return_value={"server": {"port": 5123}},
        ), patch(
            "qwen3_tts.core.config.detect_server_state",
            return_value={"running": True, "stale_pid": False, "pid": 1234,
                          "health_ok": True},
        ), patch(
            "qwen3_tts.core.config.cleanup_pid_file",
        ), patch(
            "qwen3_tts.core.config.find_pid_by_port", return_value=None,
        ), patch(
            "qwen3_tts.core.config.is_pid_alive", return_value=False,
        ), patch(
            "qwen3_tts.core.http_client.server_request",
        ) as mock_req, patch(
            "qwen3_tts.cli_server.time.sleep",
        ):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_req.return_value = mock_resp

            result = self.runner.invoke(server, ["restart"])

        self.assertNotEqual(result.exit_code, 0)
        # Stop command itself reports the port error, or restart catches the exit
        self.assertTrue(
            "did not stop" in result.output
            or "still running" in result.output
            or "failed to stop" in result.output,
            f"Unexpected output: {result.output}",
        )


class TestServerRestartPM2Delegation(unittest.TestCase):
    """restart() must delegate to `pm2 restart <name>` when PM2 owns the
    process, rather than stopping it (which itself delegates to `pm2
    stop`) and then spawning a *second*, PM2-untracked daemon via
    `_start_server_daemon()` on the same port."""

    def setUp(self):
        self.runner = CliRunner()

    @patch("qwen3_tts.cli_server._start_server_daemon")
    def test_restart_delegates_to_pm2_when_pm2_managed(self, mock_start):
        pm2_result = subprocess.CompletedProcess(
            args=["pm2", "restart", "tts-server-5123"],
            returncode=0, stdout="", stderr="",
        )
        with patch(
            "qwen3_tts.core.config.load_config",
            return_value={"server": {"port": 5123}},
        ), patch(
            "qwen3_tts.core.config.pm2_owner_of_port", return_value="tts-server-5123",
        ), patch(
            "qwen3_tts.core.config.is_server_running", return_value=True,
        ), patch(
            "qwen3_tts.cli_server.subprocess.run", return_value=pm2_result,
        ) as mock_run, patch(
            "qwen3_tts.cli_server.time.sleep",
        ):
            result = self.runner.invoke(server, ["restart"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("tts-server-5123", result.output)
        mock_run.assert_called_once_with(
            ["pm2", "restart", "tts-server-5123"],
            capture_output=True, text=True, timeout=30,
        )
        # Must never spawn a second, PM2-untracked daemon.
        mock_start.assert_not_called()

    @patch("qwen3_tts.cli_server._start_server_daemon")
    def test_restart_warns_public_flag_ignored_under_pm2(self, mock_start):
        pm2_result = subprocess.CompletedProcess(
            args=["pm2", "restart", "tts-server-5123"],
            returncode=0, stdout="", stderr="",
        )
        with patch(
            "qwen3_tts.core.config.load_config",
            return_value={"server": {"port": 5123}},
        ), patch(
            "qwen3_tts.core.config.pm2_owner_of_port", return_value="tts-server-5123",
        ), patch(
            "qwen3_tts.core.config.is_server_running", return_value=True,
        ), patch(
            "qwen3_tts.cli_server.subprocess.run", return_value=pm2_result,
        ), patch(
            "qwen3_tts.cli_server.time.sleep",
        ):
            result = self.runner.invoke(server, ["restart", "--public"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--public", result.output)
        self.assertIn("ignored", result.output)
        mock_start.assert_not_called()

    @patch("qwen3_tts.cli_server._start_server_daemon")
    def test_restart_reports_pm2_command_failure(self, mock_start):
        pm2_result = subprocess.CompletedProcess(
            args=["pm2", "restart", "tts-server-5123"],
            returncode=1, stdout="", stderr="process not found",
        )
        with patch(
            "qwen3_tts.core.config.load_config",
            return_value={"server": {"port": 5123}},
        ), patch(
            "qwen3_tts.core.config.pm2_owner_of_port", return_value="tts-server-5123",
        ), patch(
            "qwen3_tts.core.config.is_server_running", return_value=True,
        ), patch(
            "qwen3_tts.cli_server.subprocess.run", return_value=pm2_result,
        ):
            result = self.runner.invoke(server, ["restart"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("process not found", result.output)
        mock_start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
