"""Tests for tts server restart command."""

import signal
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from qwen3_tts.cli_server import _kill_server_process, server


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

        with (
            patch("qwen3_tts.core.config.is_server_running", return_value=False),
            patch(
                "qwen3_tts.core.config.pm2_owner_of_port",
                return_value=None,
            ),
            patch(
                "qwen3_tts.core.config.detect_server_state",
                return_value={
                    "running": False,
                    "stale_pid": False,
                    "pid": None,
                    "health_ok": False,
                },
            ),
            patch(
                "qwen3_tts.core.config.load_config",
                return_value={"server": {"port": 5123}},
            ),
        ):
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

        with (
            patch(
                "qwen3_tts.core.config.is_server_running",
                side_effect=lambda _c: next(running_returns, False),
            ),
            patch(
                "qwen3_tts.core.config.load_config",
                return_value={"server": {"port": 5123}},
            ),
            patch(
                "qwen3_tts.core.config.pm2_owner_of_port",
                return_value=None,
            ),
            patch(
                "qwen3_tts.core.config.detect_server_state",
                return_value={
                    "running": True,
                    "stale_pid": False,
                    "pid": 1234,
                    "health_ok": True,
                },
            ),
            patch(
                "qwen3_tts.core.config.cleanup_pid_file",
            ),
            patch(
                "qwen3_tts.core.config.find_pid_by_port",
                return_value=None,
            ),
            patch(
                "qwen3_tts.core.config.is_pid_alive",
                return_value=False,
            ),
            patch(
                "qwen3_tts.core.http_client.server_request",
            ) as mock_req,
            patch(
                "qwen3_tts.cli_server.time.sleep",
            ),
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

        with (
            patch(
                "qwen3_tts.core.config.is_server_running",
                return_value=False,
            ),
            patch(
                "qwen3_tts.core.config.pm2_owner_of_port",
                return_value=None,
            ),
            patch(
                "qwen3_tts.core.config.load_config",
                return_value={"server": {"port": 5123}},
            ),
            patch(
                "qwen3_tts.core.config.detect_server_state",
                return_value={
                    "running": False,
                    "stale_pid": False,
                    "pid": None,
                    "health_ok": False,
                },
            ),
        ):
            result = self.runner.invoke(server, ["restart", "--public"])

        self.assertEqual(result.exit_code, 0, result.output)
        mock_start.assert_called_once_with(public=True)

    def test_restart_fails_if_server_wont_stop(self):
        with (
            patch(
                "qwen3_tts.core.config.is_server_running",
                return_value=True,
            ),
            patch(
                "qwen3_tts.core.config.pm2_owner_of_port",
                return_value=None,
            ),
            patch(
                "qwen3_tts.core.config.load_config",
                return_value={"server": {"port": 5123}},
            ),
            patch(
                "qwen3_tts.core.config.detect_server_state",
                return_value={
                    "running": True,
                    "stale_pid": False,
                    "pid": 1234,
                    "health_ok": True,
                },
            ),
            patch(
                "qwen3_tts.core.config.cleanup_pid_file",
            ),
            patch(
                "qwen3_tts.core.config.find_pid_by_port",
                return_value=None,
            ),
            patch(
                "qwen3_tts.core.config.is_pid_alive",
                return_value=False,
            ),
            patch(
                "qwen3_tts.core.http_client.server_request",
            ) as mock_req,
            patch(
                "qwen3_tts.cli_server.time.sleep",
            ),
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
            returncode=0,
            stdout="",
            stderr="",
        )
        with (
            patch(
                "qwen3_tts.core.config.load_config",
                return_value={"server": {"port": 5123}},
            ),
            patch(
                "qwen3_tts.core.config.pm2_owner_of_port",
                return_value="tts-server-5123",
            ),
            patch(
                "qwen3_tts.core.config.is_server_running",
                return_value=True,
            ),
            patch(
                "qwen3_tts.cli_server.subprocess.run",
                return_value=pm2_result,
            ) as mock_run,
            patch(
                "qwen3_tts.cli_server.time.sleep",
            ),
        ):
            result = self.runner.invoke(server, ["restart"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("tts-server-5123", result.output)
        mock_run.assert_called_once_with(
            ["pm2", "restart", "tts-server-5123"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Must never spawn a second, PM2-untracked daemon.
        mock_start.assert_not_called()

    @patch("qwen3_tts.cli_server._start_server_daemon")
    def test_restart_warns_public_flag_ignored_under_pm2(self, mock_start):
        pm2_result = subprocess.CompletedProcess(
            args=["pm2", "restart", "tts-server-5123"],
            returncode=0,
            stdout="",
            stderr="",
        )
        with (
            patch(
                "qwen3_tts.core.config.load_config",
                return_value={"server": {"port": 5123}},
            ),
            patch(
                "qwen3_tts.core.config.pm2_owner_of_port",
                return_value="tts-server-5123",
            ),
            patch(
                "qwen3_tts.core.config.is_server_running",
                return_value=True,
            ),
            patch(
                "qwen3_tts.cli_server.subprocess.run",
                return_value=pm2_result,
            ),
            patch(
                "qwen3_tts.cli_server.time.sleep",
            ),
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
            returncode=1,
            stdout="",
            stderr="process not found",
        )
        with (
            patch(
                "qwen3_tts.core.config.load_config",
                return_value={"server": {"port": 5123}},
            ),
            patch(
                "qwen3_tts.core.config.pm2_owner_of_port",
                return_value="tts-server-5123",
            ),
            patch(
                "qwen3_tts.core.config.is_server_running",
                return_value=True,
            ),
            patch(
                "qwen3_tts.cli_server.subprocess.run",
                return_value=pm2_result,
            ),
        ):
            result = self.runner.invoke(server, ["restart"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("process not found", result.output)
        mock_start.assert_not_called()


class TestKillServerProcess(unittest.TestCase):
    """Direct coverage of the stop fallback `_kill_server_process`:
    SIGTERM -> bounded poll -> SIGKILL escalation, with `cleanup_pid_file`
    always reached. Only exercised when the graceful /shutdown poll fails,
    so the CLI `stop`/`restart` paths above mock past it."""

    def _run_kill(self, state, port, *, find_pid=None, alive=None, kill=None):
        """Invoke the fallback with the config/os/time seams patched.

        Returns (find, kill, cleanup, sleep, echo) mocks for assertions.
        """
        kill_kwargs = {} if kill is None else {"side_effect": kill}
        with (
            patch(
                "qwen3_tts.core.config.find_pid_by_port",
                return_value=find_pid,
            ) as mock_find,
            patch(
                "qwen3_tts.core.config.is_pid_alive",
                side_effect=alive,
            ),
            patch(
                "qwen3_tts.core.config.cleanup_pid_file",
            ) as mock_cleanup,
            patch(
                "qwen3_tts.cli_server.os.kill",
                **kill_kwargs,
            ) as mock_kill,
            patch(
                "qwen3_tts.cli_server.time.sleep",
            ) as mock_sleep,
            patch(
                "qwen3_tts.cli_server.click.echo",
            ) as mock_echo,
        ):
            _kill_server_process(state, port)
        return mock_find, mock_kill, mock_cleanup, mock_sleep, mock_echo

    def test_sigterm_stops_process_and_skips_sigkill(self):
        # alive gate True, poll iteration 1 False (break), SIGKILL gate False
        _, mock_kill, mock_cleanup, _, _ = self._run_kill(
            {"pid": 1234}, 5123, alive=[True, False, False]
        )
        mock_kill.assert_called_once_with(1234, signal.SIGTERM)
        mock_cleanup.assert_called_once()

    def test_discovers_pid_by_port_when_pid_file_missing(self):
        mock_find, mock_kill, mock_cleanup, _, mock_echo = self._run_kill(
            {"pid": None}, 5123, find_pid=777, alive=[True, False, False]
        )
        mock_find.assert_called_once_with(5123)
        mock_kill.assert_called_once_with(777, signal.SIGTERM)
        echoed = [str(c) for c in mock_echo.call_args_list]
        self.assertTrue(any("Discovered server PID 777" in line for line in echoed))
        mock_cleanup.assert_called_once()

    def test_discovery_finds_nothing_skips_kill_but_cleans(self):
        _, mock_kill, mock_cleanup, _, _ = self._run_kill(
            {"pid": None}, 5123, find_pid=None, alive=[]
        )
        mock_kill.assert_not_called()
        mock_cleanup.assert_called_once()

    def test_dead_pid_skips_kill_but_cleans(self):
        _, mock_kill, mock_cleanup, _, _ = self._run_kill(
            {"pid": 1234}, 5123, alive=[False]
        )
        mock_kill.assert_not_called()
        mock_cleanup.assert_called_once()

    def test_sigterm_failure_escalates_to_sigkill(self):
        # alive stays True: gate + 6 poll iterations + SIGKILL gate = 8 calls
        _, mock_kill, mock_cleanup, _, mock_echo = self._run_kill(
            {"pid": 1234}, 5123, alive=[True] * 8
        )
        self.assertEqual(
            [c.args for c in mock_kill.call_args_list],
            [(1234, signal.SIGTERM), (1234, signal.SIGKILL)],
        )
        echoed = [str(c) for c in mock_echo.call_args_list]
        self.assertTrue(any("SIGTERM failed" in line for line in echoed))
        mock_cleanup.assert_called_once()

    def test_sigterm_oserror_swallowed_cleanup_still_runs(self):
        # The SIGTERM os.kill raises; the poll then shows the process dead,
        # so no SIGKILL is attempted and cleanup still happens.
        _, mock_kill, mock_cleanup, _, _ = self._run_kill(
            {"pid": 1234}, 5123, alive=[True, False, False], kill=[OSError("nope")]
        )
        mock_kill.assert_called_once_with(1234, signal.SIGTERM)
        mock_cleanup.assert_called_once()

    def test_sigkill_oserror_swallowed_cleanup_still_runs(self):
        _, mock_kill, mock_cleanup, _, _ = self._run_kill(
            {"pid": 1234}, 5123, alive=[True] * 8, kill=[None, OSError("nope")]
        )
        self.assertEqual(mock_kill.call_count, 2)
        mock_cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
