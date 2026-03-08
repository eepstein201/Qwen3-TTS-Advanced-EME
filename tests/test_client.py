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


@pytest.mark.unit
class TestClientHelpers(unittest.TestCase):
    """Test internal helper functions for code reuse."""

    def _make_config(self, data=None):
        """Create a temp config file and return its path."""
        if data is None:
            data = {
                "server": {"host": "127.0.0.1", "port": 5123},
                "presets": {"consistent": {"temperature": 0.5}},
                "aliases": {"default": {"prompt": "voice.pt", "preset": "consistent"}},
                "generation": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            }
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(data, f)
        f.close()
        return f.name

    def test_resolve_voice_alias_helper_exists(self):
        """_resolve_voice_alias helper should exist."""
        from qwen3_tts.server import client
        self.assertTrue(
            hasattr(client, '_resolve_voice_alias'),
            "_resolve_voice_alias helper should exist"
        )

    def test_resolve_voice_alias_returns_updated_params(self):
        """_resolve_voice_alias should return updated parameters."""
        from qwen3_tts.server.client import TTSClient, _resolve_voice_alias
        tmp = self._make_config({
            "server": {"host": "127.0.0.1", "port": 5123},
            "aliases": {"my_voice": {"prompt": "custom.pt", "mode": "clone"}},
        })
        try:
            client = TTSClient(config_path=tmp)
            alias = client.resolve_alias("my_voice")
            result = _resolve_voice_alias(
                alias=alias,
                prompt=None,
                mode=None,
                description=None,
                speaker=None,
                instruct=None,
                preset=None,
            )
            self.assertEqual(result["prompt"], "custom.pt")
            self.assertEqual(result["mode"], "clone")
            client.close()
        finally:
            os.unlink(tmp)

    def test_resolve_voice_alias_preserves_user_overrides(self):
        """_resolve_voice_alias should not override user-provided values."""
        from qwen3_tts.server.client import TTSClient, _resolve_voice_alias
        tmp = self._make_config({
            "server": {"host": "127.0.0.1", "port": 5123},
            "aliases": {"my_voice": {"prompt": "alias.pt", "mode": "clone"}},
        })
        try:
            client = TTSClient(config_path=tmp)
            alias = client.resolve_alias("my_voice")
            # User provides their own prompt - should NOT be overridden
            result = _resolve_voice_alias(
                alias=alias,
                prompt="user.pt",
                mode=None,
                description=None,
                speaker=None,
                instruct=None,
                preset=None,
            )
            self.assertEqual(result["prompt"], "user.pt")  # User value preserved
            client.close()
        finally:
            os.unlink(tmp)

    def test_build_gen_params_helper_exists(self):
        """_build_gen_params helper should exist."""
        from qwen3_tts.server import client
        self.assertTrue(
            hasattr(client, '_build_gen_params'),
            "_build_gen_params helper should exist"
        )

    def test_build_gen_params_uses_config_defaults(self):
        """_build_gen_params should use config defaults."""
        from qwen3_tts.server.client import TTSClient, _build_gen_params
        tmp = self._make_config({
            "server": {"host": "127.0.0.1", "port": 5123},
            "generation": {"temperature": 0.8, "top_k": 40},
        })
        try:
            client = TTSClient(config_path=tmp)
            result = _build_gen_params(
                config=client.config,
                temperature=None,
                top_k=None,
                top_p=None,
                repetition_penalty=None,
                max_new_tokens=None,
                seed=None,
            )
            self.assertEqual(result["temperature"], 0.8)
            self.assertEqual(result["top_k"], 40)
            client.close()
        finally:
            os.unlink(tmp)

    def test_build_gen_params_overrides_with_user_values(self):
        """_build_gen_params should use user-provided values over config."""
        from qwen3_tts.server.client import TTSClient, _build_gen_params
        tmp = self._make_config({
            "server": {"host": "127.0.0.1", "port": 5123},
            "generation": {"temperature": 0.8, "top_k": 40},
        })
        try:
            client = TTSClient(config_path=tmp)
            result = _build_gen_params(
                config=client.config,
                temperature=0.5,  # User override
                top_k=None,
                top_p=None,
                repetition_penalty=None,
                max_new_tokens=None,
                seed=None,
            )
            self.assertEqual(result["temperature"], 0.5)  # User value
            self.assertEqual(result["top_k"], 40)  # Config default
            client.close()
        finally:
            os.unlink(tmp)


if __name__ == "__main__":
    unittest.main()
