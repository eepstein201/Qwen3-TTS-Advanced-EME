#!/usr/bin/env python3
"""Tests for TTSClient in qwen3_tts/server/client.py.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m unittest tests.test_client -v

No running server required — all HTTP calls are mocked.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.unit
class TestTTSClientInit(unittest.TestCase):
    """Test TTSClient initialization."""

    def _make_config(self, data=None):
        """Create a temp config file and return its path."""
        if data is None:
            data = {
                "server": {"host": "127.0.0.1", "port": 5123},
                "presets": {"consistent": {"temperature": 0.5}},
                "aliases": {"default": {"prompt": "voice.pt"}},
            }
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(data, f)
        f.close()
        return f.name

    def test_init_creates_session(self):
        """TTSClient creates a requests session."""
        from qwen3_tts.server.client import TTSClient
        tmp = self._make_config()
        try:
            client = TTSClient(config_path=tmp)
            self.assertIsNotNone(client._session)
            client.close()
        finally:
            os.unlink(tmp)

    def test_server_url_from_config(self):
        """TTSClient.server_url reads from config."""
        from qwen3_tts.server.client import TTSClient
        tmp = self._make_config({"server": {"host": "10.0.0.1", "port": 9999}})
        try:
            client = TTSClient(config_path=tmp)
            self.assertEqual(client.server_url, "http://10.0.0.1:9999")
            client.close()
        finally:
            os.unlink(tmp)

    def test_is_server_running_healthy(self):
        """is_server_running returns True when health check succeeds."""
        from qwen3_tts.server.client import TTSClient
        tmp = self._make_config()
        try:
            client = TTSClient(config_path=tmp)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"status": "ok"}
            with patch("requests.get", return_value=mock_resp):
                result = client.is_server_running()
                self.assertTrue(result)
            client.close()
        finally:
            os.unlink(tmp)

    def test_is_server_running_down(self):
        """is_server_running returns False when connection fails."""
        from qwen3_tts.server.client import TTSClient
        import requests
        tmp = self._make_config()
        try:
            client = TTSClient(config_path=tmp)
            with patch("requests.get", side_effect=requests.ConnectionError):
                result = client.is_server_running()
                self.assertFalse(result)
            client.close()
        finally:
            os.unlink(tmp)

    def test_list_presets(self):
        """list_presets returns presets from config."""
        from qwen3_tts.server.client import TTSClient
        tmp = self._make_config()
        try:
            client = TTSClient(config_path=tmp)
            presets = client.list_presets()
            self.assertIsInstance(presets, dict)
            self.assertIn("consistent", presets)
            client.close()
        finally:
            os.unlink(tmp)

    def test_list_aliases(self):
        """list_aliases returns aliases from config."""
        from qwen3_tts.server.client import TTSClient
        tmp = self._make_config()
        try:
            client = TTSClient(config_path=tmp)
            aliases = client.list_aliases()
            self.assertIsInstance(aliases, dict)
            self.assertIn("default", aliases)
            client.close()
        finally:
            os.unlink(tmp)

    def test_resolve_alias_found(self):
        """resolve_alias returns settings for existing alias."""
        from qwen3_tts.server.client import TTSClient
        tmp = self._make_config()
        try:
            client = TTSClient(config_path=tmp)
            result = client.resolve_alias("default")
            self.assertIsNotNone(result)
            self.assertEqual(result["prompt"], "voice.pt")
            client.close()
        finally:
            os.unlink(tmp)

    def test_resolve_alias_not_found(self):
        """resolve_alias returns None for missing alias."""
        from qwen3_tts.server.client import TTSClient
        tmp = self._make_config()
        try:
            client = TTSClient(config_path=tmp)
            result = client.resolve_alias("nonexistent")
            self.assertIsNone(result)
            client.close()
        finally:
            os.unlink(tmp)

    def test_context_manager(self):
        """TTSClient works as context manager."""
        from qwen3_tts.server.client import TTSClient
        tmp = self._make_config()
        try:
            with TTSClient(config_path=tmp) as client:
                self.assertIsNotNone(client._session)
        finally:
            os.unlink(tmp)


if __name__ == "__main__":
    unittest.main()
