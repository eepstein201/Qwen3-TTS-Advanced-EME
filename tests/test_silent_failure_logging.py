"""Previously-swallowed errors must be logged (observability).

Several fallback paths caught exceptions and silently degraded (``except ...:
pass`` / ``return None``) with no log record, making real filesystem/permission
failures invisible in production. These tests assert a warning is emitted while
the graceful fallback behavior is preserved.

See docs/reviews/e2e-review-2026-07-01.md (Phase 5 silent-failure findings).
"""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests

from qwen3_tts.core import config as config_mod
from qwen3_tts.server.client.voices import VoiceManagerMixin


class TestGetDefaultClonePromptLogging(unittest.TestCase):
    """config.get_default_clone_prompt must log when the prompt-dir scan fails."""

    def test_listdir_oserror_logs_warning_and_returns_none(self):
        # config={} → no configured default → falls through to the dir scan,
        # which we force to raise OSError (e.g. unreadable voice_prompts dir).
        with patch.object(config_mod.os, "listdir", side_effect=OSError("boom")):
            with self.assertLogs("tts.config", level="WARNING") as cm:
                result = config_mod.get_default_clone_prompt(config={})
        self.assertIsNone(result)
        self.assertTrue(
            any("boom" in line or "voice_prompts" in line.lower() for line in cm.output),
            f"expected a warning mentioning the failure; got {cm.output}",
        )


class _StubVoiceClient(VoiceManagerMixin):
    """Minimal host for the mixin: server 'running' but /prompts fails."""

    def __init__(self, session, prompts_dir):
        self._session = session
        self.server_url = "http://127.0.0.1:5123"
        self.voice_prompts_dir = prompts_dir

    def is_server_running(self):
        return True


class TestListPromptsFallbackLogging(unittest.TestCase):
    """list_prompts must log when the server call fails before FS fallback."""

    def test_server_error_logs_warning_and_falls_back(self):
        session = MagicMock()
        session.get.side_effect = requests.RequestException("connection refused")
        with tempfile.TemporaryDirectory() as tmp:
            client = _StubVoiceClient(session, tmp)
            with patch(
                "qwen3_tts.server.client.voices.auth_headers", return_value={}
            ):
                with self.assertLogs("tts.client.voices", level="WARNING") as cm:
                    result = client.list_prompts()
        self.assertEqual(result, [])  # empty dir → graceful fallback
        self.assertTrue(
            any("connection refused" in line or "fall" in line.lower() for line in cm.output),
            f"expected a warning about the server-listing failure; got {cm.output}",
        )


if __name__ == "__main__":
    unittest.main()
