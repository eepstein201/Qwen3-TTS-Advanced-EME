"""Tests for PM2 process-supervision detection (qwen3_tts.core.config.pm2).

`tts server stop`/`restart` need to detect when the server process is
supervised by PM2 (ecosystem.config.cjs, autorestart: true) so they can
delegate to `pm2 stop`/`pm2 restart` instead of killing the process directly
-- a direct kill is immediately undone by PM2's autorestart, which makes
`tts server stop` appear to silently fail.
"""

import json
import subprocess
import unittest
from unittest.mock import patch

from qwen3_tts.core.config.pm2 import pm2_owner_of_port


class TestPm2OwnerOfPort(unittest.TestCase):
    def test_returns_none_when_nothing_listens_on_port(self):
        with patch("qwen3_tts.core.config.find_pid_by_port", return_value=None):
            self.assertIsNone(pm2_owner_of_port(5123))

    def test_returns_none_when_pm2_binary_missing(self):
        with patch(
            "qwen3_tts.core.config.find_pid_by_port", return_value=79974
        ), patch(
            "qwen3_tts.core.config.pm2.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            self.assertIsNone(pm2_owner_of_port(5123))

    def test_returns_none_when_pm2_jlist_nonzero_exit(self):
        proc = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
        with patch(
            "qwen3_tts.core.config.find_pid_by_port", return_value=79974
        ), patch("qwen3_tts.core.config.pm2.subprocess.run", return_value=proc):
            self.assertIsNone(pm2_owner_of_port(5123))

    def test_returns_none_when_pm2_jlist_not_json(self):
        proc = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not json", stderr=""
        )
        with patch(
            "qwen3_tts.core.config.find_pid_by_port", return_value=79974
        ), patch("qwen3_tts.core.config.pm2.subprocess.run", return_value=proc):
            self.assertIsNone(pm2_owner_of_port(5123))

    def test_returns_none_when_no_online_apps(self):
        apps = [
            {"name": "tts-server-5123", "pid": 79939, "pm2_env": {"status": "stopped"}}
        ]
        proc = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(apps), stderr=""
        )
        with patch(
            "qwen3_tts.core.config.find_pid_by_port", return_value=79974
        ), patch("qwen3_tts.core.config.pm2.subprocess.run", return_value=proc):
            self.assertIsNone(pm2_owner_of_port(5123))

    def test_returns_app_name_when_leaf_pid_matches_directly(self):
        apps = [
            {"name": "tts-server-5123", "pid": 79974, "pm2_env": {"status": "online"}}
        ]
        proc = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(apps), stderr=""
        )
        with patch(
            "qwen3_tts.core.config.find_pid_by_port", return_value=79974
        ), patch("qwen3_tts.core.config.pm2.subprocess.run", return_value=proc):
            self.assertEqual(pm2_owner_of_port(5123), "tts-server-5123")

    def test_returns_app_name_via_ancestor_walk(self):
        # port PID (python, leaf) -> zsh -> node (the pm2-tracked pid)
        apps = [
            {"name": "tts-server-5123", "pid": 79939, "pm2_env": {"status": "online"}}
        ]
        proc = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(apps), stderr=""
        )
        ppid_chain = {79974: 79950, 79950: 79939, 79939: 1}
        with patch(
            "qwen3_tts.core.config.find_pid_by_port", return_value=79974
        ), patch(
            "qwen3_tts.core.config.pm2.subprocess.run", return_value=proc
        ), patch(
            "qwen3_tts.core.config.pm2._parent_pid",
            side_effect=lambda pid: ppid_chain.get(pid),
        ):
            self.assertEqual(pm2_owner_of_port(5123), "tts-server-5123")

    def test_returns_none_when_ancestor_chain_exhausted(self):
        apps = [
            {"name": "tts-server-5123", "pid": 79939, "pm2_env": {"status": "online"}}
        ]
        proc = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(apps), stderr=""
        )
        # Unrelated process tree -- never reaches the pm2-tracked pid.
        ppid_chain = {79974: 100, 100: 1}
        with patch(
            "qwen3_tts.core.config.find_pid_by_port", return_value=79974
        ), patch(
            "qwen3_tts.core.config.pm2.subprocess.run", return_value=proc
        ), patch(
            "qwen3_tts.core.config.pm2._parent_pid",
            side_effect=lambda pid: ppid_chain.get(pid),
        ):
            self.assertIsNone(pm2_owner_of_port(5123))

    def test_returns_none_on_pm2_jlist_timeout(self):
        with patch(
            "qwen3_tts.core.config.find_pid_by_port", return_value=79974
        ), patch(
            "qwen3_tts.core.config.pm2.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="pm2", timeout=5),
        ):
            self.assertIsNone(pm2_owner_of_port(5123))


class TestParentPid(unittest.TestCase):
    def test_returns_ppid_from_ps_output(self):
        from qwen3_tts.core.config.pm2 import _parent_pid

        proc = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=" 79950\n", stderr=""
        )
        with patch("qwen3_tts.core.config.pm2.subprocess.run", return_value=proc):
            self.assertEqual(_parent_pid(79974), 79950)

    def test_returns_none_when_ps_fails(self):
        from qwen3_tts.core.config.pm2 import _parent_pid

        proc = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
        with patch("qwen3_tts.core.config.pm2.subprocess.run", return_value=proc):
            self.assertIsNone(_parent_pid(79974))

    def test_returns_none_on_non_numeric_output(self):
        from qwen3_tts.core.config.pm2 import _parent_pid

        proc = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="garbage\n", stderr=""
        )
        with patch("qwen3_tts.core.config.pm2.subprocess.run", return_value=proc):
            self.assertIsNone(_parent_pid(79974))


if __name__ == "__main__":
    unittest.main()
