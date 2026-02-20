#!/usr/bin/env python3
"""Tests for ensure_server_running() Popen fallback PYTHONPATH fix.

Tests that when ~/bin/tts is absent, the Popen fallback passes
PYTHONPATH in env so the qwen3_tts package is importable in the daemon.

Without env=, the spawned process inherits no PYTHONPATH and crashes with
ModuleNotFoundError: No module named 'qwen3_tts'.

Run: python -m pytest tests/test_generate_server_fallback.py -v
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qwen3_tts.interface.generate import ensure_server_running
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires qwen3_tts.interface.generate")

_CONFIG = {"server": {"host": "127.0.0.1", "port": 5123}}


@_skip
class TestEnsureServerRunningFallbackPythonpath(unittest.TestCase):
    """Popen fallback must pass PYTHONPATH so qwen3_tts is importable in daemon."""

    def _run_with_fallback(self, extra_env=None):
        """Invoke ensure_server_running with ~/bin/tts absent.

        Mocks out all I/O and returns the captured Popen mock call args.
        """
        mock_popen = MagicMock()
        # First call: server not running. Second call: server up after start.
        env_patch = extra_env or {}
        with patch("qwen3_tts.interface.generate.is_server_running",
                   side_effect=[False, True]), \
             patch("qwen3_tts.interface.generate.os.path.exists",
                   return_value=False), \
             patch("builtins.open", mock_open()), \
             patch("qwen3_tts.interface.generate.subprocess.Popen", mock_popen), \
             patch("qwen3_tts.interface.generate.time.sleep", return_value=None), \
             patch.dict(os.environ, env_patch):
            ensure_server_running(_CONFIG)

        self.assertTrue(mock_popen.called,
                        "Popen must be called when ~/bin/tts is absent")
        return mock_popen.call_args

    def test_popen_receives_env_keyword(self):
        """Popen must be called with env= so the daemon gets a custom environment."""
        call = self._run_with_fallback()
        self.assertIn("env", call.kwargs,
                      "Popen must be called with env= keyword argument")

    def test_popen_env_contains_pythonpath(self):
        """env passed to Popen must include a PYTHONPATH key."""
        call = self._run_with_fallback()
        env = call.kwargs["env"]
        self.assertIn("PYTHONPATH", env,
                      "PYTHONPATH must be present in the env passed to Popen")

    def test_popen_pythonpath_includes_user_files_dir(self):
        """PYTHONPATH must contain ~/Qwen3-TTS_UserFiles so qwen3_tts is importable."""
        call = self._run_with_fallback()
        env = call.kwargs["env"]
        user_files = os.path.expanduser("~/Qwen3-TTS_UserFiles")
        self.assertIn(user_files, env["PYTHONPATH"],
                      f"PYTHONPATH must include {user_files}")

    def test_popen_pythonpath_preserves_existing_pythonpath(self):
        """Pre-existing PYTHONPATH entries must not be discarded."""
        call = self._run_with_fallback(extra_env={"PYTHONPATH": "/some/other/path"})
        env = call.kwargs["env"]
        self.assertIn("/some/other/path", env["PYTHONPATH"],
                      "Pre-existing PYTHONPATH entries must be preserved in Popen env")


if __name__ == "__main__":
    unittest.main()
