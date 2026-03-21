"""Tests for qwen3_tts/server/client/voices.py.

Covers: list_prompts(), delete_prompt(), rename_prompt(),
preview_prompt(), get_prompt_details() — all with mocked HTTP calls.
No running server or GPU required.

Run with:
    python -m pytest tests/test_client_voices.py -v --tb=short
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(data=None):
    """Create a temp config file and return its path."""
    if data is None:
        data = {
            "server": {"host": "127.0.0.1", "port": 5123},
            "presets": {},
            "aliases": {},
            "generation": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
            "output_directory": "~/Downloads",
            "default_clone_prompt": "default.pt",
            "default_voice_description": "neutral voice",
            "default_speaker": "ryan",
            "language": "English",
            "prosody_presets": {},
        }
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return f.name


def _mock_error_response(status=500, message="something went wrong"):
    """Build a mock error response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"error": message}
    return resp


def _client_with_server(config_path):
    """Return (client, session_mock) with session pre-wired."""
    from qwen3_tts.server.client import TTSClient

    client = TTSClient(config_path=config_path)
    session = MagicMock()
    client._session = session
    return client, session


# ============================================================================
# list_prompts()
# ============================================================================


class TestListPrompts(unittest.TestCase):
    """list_prompts() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_list_prompts_via_server(self):
        """list_prompts uses server endpoint when available."""
        client, session = _client_with_server(self.cfg)
        session.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"prompts": ["voice1.pt", "voice2.wav"]}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.list_prompts()
        self.assertEqual(result, ["voice1.pt", "voice2.wav"])

    def test_list_prompts_filesystem_fallback(self):
        """list_prompts falls back to filesystem when server is down."""
        client, _ = _client_with_server(self.cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            client.voice_prompts_dir = tmpdir
            # Create .pt file
            open(os.path.join(tmpdir, "torch_voice.pt"), "w").close()
            # Create .wav + .txt pair (MLX format)
            open(os.path.join(tmpdir, "mlx_voice.wav"), "w").close()
            open(os.path.join(tmpdir, "mlx_voice.txt"), "w").close()
            # Create .wav without .txt (should be excluded)
            open(os.path.join(tmpdir, "orphan.wav"), "w").close()

            with patch.object(client, "is_server_running", return_value=False):
                result = client.list_prompts()

        self.assertIn("torch_voice.pt", result)
        self.assertIn("mlx_voice.wav", result)
        self.assertNotIn("orphan.wav", result)

    def test_list_prompts_missing_dir_returns_empty(self):
        """list_prompts returns empty list when voice dir doesn't exist."""
        client, _ = _client_with_server(self.cfg)
        client.voice_prompts_dir = "/nonexistent/path"

        with patch.object(client, "is_server_running", return_value=False):
            result = client.list_prompts()
        self.assertEqual(result, [])


# ============================================================================
# delete_prompt()
# ============================================================================


class TestDeletePrompt(unittest.TestCase):
    """delete_prompt() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_delete_prompt_success(self):
        """delete_prompt returns response on success."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "deleted", "files_removed": ["v.pt"]}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.delete_prompt("v.pt")
        self.assertEqual(result["status"], "deleted")

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["name"], "v.pt")

    def test_delete_prompt_error_raises(self):
        """delete_prompt raises VoicePromptError on failure."""
        from qwen3_tts.core.config import VoicePromptError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(404, "not found")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(VoicePromptError):
                client.delete_prompt("missing.pt")


# ============================================================================
# rename_prompt()
# ============================================================================


class TestRenamePrompt(unittest.TestCase):
    """rename_prompt() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_rename_prompt_success(self):
        """rename_prompt sends old_name and new_name."""
        client, session = _client_with_server(self.cfg)
        session.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"status": "renamed"}),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.rename_prompt("old.pt", "new.pt")
        self.assertEqual(result["status"], "renamed")

        payload = session.post.call_args[1]["json"]
        self.assertEqual(payload["old_name"], "old.pt")
        self.assertEqual(payload["new_name"], "new.pt")

    def test_rename_prompt_error_raises(self):
        """rename_prompt raises VoicePromptError on failure."""
        from qwen3_tts.core.config import VoicePromptError

        client, session = _client_with_server(self.cfg)
        session.post.return_value = _mock_error_response(404, "prompt not found")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(VoicePromptError):
                client.rename_prompt("missing.pt", "new.pt")


# ============================================================================
# preview_prompt()
# ============================================================================


class TestPreviewPrompt(unittest.TestCase):
    """preview_prompt() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_preview_prompt_returns_bytes(self):
        """preview_prompt returns raw audio bytes."""
        client, session = _client_with_server(self.cfg)
        wav_bytes = b"\x00\x01\x02\x03"
        session.get.return_value = MagicMock(
            status_code=200,
            content=wav_bytes,
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.preview_prompt("voice.pt")
        self.assertEqual(result, wav_bytes)

        params = session.get.call_args[1]["params"]
        self.assertEqual(params["name"], "voice.pt")

    def test_preview_prompt_error_raises(self):
        """preview_prompt raises VoicePromptError on failure."""
        from qwen3_tts.core.config import VoicePromptError

        client, session = _client_with_server(self.cfg)
        session.get.return_value = _mock_error_response(404, "not found")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(VoicePromptError):
                client.preview_prompt("missing.pt")


# ============================================================================
# get_prompt_details()
# ============================================================================


class TestGetPromptDetails(unittest.TestCase):
    """get_prompt_details() method."""

    def setUp(self):
        self.cfg = _make_config()

    def tearDown(self):
        os.unlink(self.cfg)

    def test_get_single_prompt_details(self):
        """get_prompt_details with name returns single prompt metadata."""
        client, session = _client_with_server(self.cfg)
        expected = {"name": "v.pt", "size": 1024, "duration": 3.5}
        session.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=expected),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.get_prompt_details("v.pt")
        self.assertEqual(result["duration"], 3.5)

        params = session.get.call_args[1]["params"]
        self.assertEqual(params["name"], "v.pt")

    def test_get_all_prompt_details(self):
        """get_prompt_details without name returns all prompts."""
        client, session = _client_with_server(self.cfg)
        expected = {"prompts": [{"name": "a.pt"}, {"name": "b.pt"}]}
        session.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=expected),
        )

        with patch.object(client, "is_server_running", return_value=True):
            result = client.get_prompt_details()
        self.assertEqual(len(result["prompts"]), 2)

        # No name param when querying all
        params = session.get.call_args[1]["params"]
        self.assertEqual(params, {})

    def test_get_prompt_details_error_raises(self):
        """get_prompt_details raises VoicePromptError on failure."""
        from qwen3_tts.core.config import VoicePromptError

        client, session = _client_with_server(self.cfg)
        session.get.return_value = _mock_error_response(500, "internal")

        with patch.object(client, "is_server_running", return_value=True):
            with self.assertRaises(VoicePromptError):
                client.get_prompt_details("v.pt")


if __name__ == "__main__":
    unittest.main()
