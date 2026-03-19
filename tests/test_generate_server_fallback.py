#!/usr/bin/env python3
"""Tests for ensure_server_running() fallback server startup.

Tests that when `tts` CLI is not on PATH, the fallback uses
`python -m qwen3_tts.server.app` to start the server.

Run: python -m pytest tests/test_generate_server_fallback.py -v
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    class _DummyMarkerFunc:
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition, **kwargs):
            return lambda f: f
    class _DummyMarker:
        def __call__(self, func):
            return func
        def __getattr__(self, name):
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qwen3_tts.interface.generate import ensure_server_running
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires qwen3_tts.interface.generate")

_CONFIG = {"server": {"host": "127.0.0.1", "port": 5123}}


@pytest.mark.unit
@_skip
class TestEnsureServerRunningFallback(unittest.TestCase):
    """Fallback must use python -m qwen3_tts.server.app when tts CLI not found."""

    def _run_with_fallback(self):
        """Invoke ensure_server_running with tts not on PATH.

        Mocks out all I/O and returns the captured Popen mock call args.
        """
        mock_popen = MagicMock()
        # First call: server not running. Second call: server up after start.
        with patch("qwen3_tts.interface.generate.is_server_running",
                   side_effect=[False, True]), \
             patch("shutil.which",
                   return_value=None), \
             patch("builtins.open", mock_open()), \
             patch("qwen3_tts.interface.generate.subprocess.Popen", mock_popen), \
             patch("qwen3_tts.interface.generate.time.sleep", return_value=None):
            ensure_server_running(_CONFIG)

        self.assertTrue(mock_popen.called,
                        "Popen must be called when tts is not on PATH")
        return mock_popen.call_args

    def test_popen_uses_module_invocation(self):
        """Popen must use -m qwen3_tts.server.app."""
        call = self._run_with_fallback()
        cmd = call.args[0] if call.args else call.kwargs.get("args", [])
        self.assertIn("-m", cmd, "Popen must use -m flag for module invocation")
        self.assertIn("qwen3_tts.server.app", cmd,
                      "Popen must reference qwen3_tts.server.app module")

    def test_popen_uses_sys_executable(self):
        """Popen must use sys.executable as the Python interpreter."""
        call = self._run_with_fallback()
        cmd = call.args[0] if call.args else call.kwargs.get("args", [])
        self.assertEqual(cmd[0], sys.executable,
                         "Popen must use sys.executable")

    def test_popen_starts_new_session(self):
        """Popen must set start_new_session=True for daemon behavior."""
        call = self._run_with_fallback()
        self.assertTrue(call.kwargs.get("start_new_session", False),
                        "Popen must use start_new_session=True")

    def test_uses_shutil_which_for_tts(self):
        """Should use shutil.which('tts') to find the CLI."""
        mock_popen = MagicMock()
        mock_which = MagicMock(return_value=None)
        with patch("qwen3_tts.interface.generate.is_server_running",
                   side_effect=[False, True]), \
             patch("shutil.which", mock_which), \
             patch("builtins.open", mock_open()), \
             patch("qwen3_tts.interface.generate.subprocess.Popen", mock_popen), \
             patch("qwen3_tts.interface.generate.time.sleep", return_value=None):
            ensure_server_running(_CONFIG)
        mock_which.assert_called_once_with("tts")


if __name__ == "__main__":
    unittest.main()
