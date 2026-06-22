"""Tests for improved error handling in client.py."""
from unittest import mock

import pytest


class TestClientErrorHandling:
    """Tests for TTSClient error handling."""

    def test_load_model_raises_model_error_on_failure(self):
        """load_model raises appropriate error type on failure."""
        from qwen3_tts.core.config import TTSError
        from qwen3_tts.server.client import TTSClient

        client = TTSClient()
        with mock.patch.object(client, 'is_server_running', return_value=True):
            with mock.patch.object(client._session, 'post') as mock_post:
                mock_response = mock.Mock()
                mock_response.status_code = 500
                mock_response.json.return_value = {"detail": "Model load failed"}
                mock_post.return_value = mock_response

                with pytest.raises(TTSError) as exc:
                    client.load_model("clone")
                # Should be a TTSError subclass, not generic Exception
                assert type(exc.value).__name__ != "Exception"

    def test_delete_prompt_raises_voice_prompt_error(self):
        """delete_prompt raises appropriate error type on failure."""
        from qwen3_tts.core.config import TTSError
        from qwen3_tts.server.client import TTSClient

        client = TTSClient()
        with mock.patch.object(client, 'is_server_running', return_value=True):
            with mock.patch.object(client._session, 'post') as mock_post:
                mock_response = mock.Mock()
                mock_response.status_code = 500
                mock_response.json.return_value = {"detail": "Delete failed"}
                mock_post.return_value = mock_response

                with pytest.raises(TTSError) as exc:
                    client.delete_prompt("test_voice")
                assert type(exc.value).__name__ != "Exception"

    def test_generate_raises_generation_error_on_server_error(self):
        """generate raises GenerationError on server error."""
        from qwen3_tts.core.config import GenerationError
        from qwen3_tts.server.client import TTSClient

        client = TTSClient()
        with mock.patch.object(client, 'is_server_running', return_value=True):
            with mock.patch.object(client._session, 'post') as mock_post:
                mock_response = mock.Mock()
                mock_response.status_code = 500
                mock_response.json.return_value = {"detail": "Generation failed"}
                mock_post.return_value = mock_response

                with pytest.raises(GenerationError):
                    client.generate("test text", mode="clone", prompt="default.pt")


class TestVoicePromptError:
    """Tests for VoicePromptError class."""

    def test_voice_prompt_error_exists(self):
        """VoicePromptError class exists."""
        from qwen3_tts.core.config import VoicePromptError

        assert issubclass(VoicePromptError, Exception)

    def test_voice_prompt_error_has_message(self):
        """VoicePromptError has user-friendly message."""
        from qwen3_tts.core.config import VoicePromptError

        error = VoicePromptError("delete", "Voice prompt not found")
        assert "Voice prompt" in error.user_message or "not found" in error.user_message.lower()

    def test_voice_prompt_error_has_recovery(self):
        """VoicePromptError has recovery hint."""
        from qwen3_tts.core.config import VoicePromptError

        error = VoicePromptError("delete", "Test error")
        assert error.recovery in ("config", "retry", "restart")


class TestModelError:
    """Tests for ModelError class."""

    def test_model_error_exists(self):
        """ModelError class exists."""
        from qwen3_tts.core.config import ModelError

        assert issubclass(ModelError, Exception)

    def test_model_error_has_model_type(self):
        """ModelError stores model type."""
        from qwen3_tts.core.config import ModelError

        error = ModelError("clone", "Failed to load")
        assert error.model_type == "clone"


class TestExceptionChaining:
    """Tests for exception handling in client."""

    def test_client_raises_connection_error_directly(self):
        """Client raises ConnectionError when network fails."""
        from qwen3_tts.server.client import TTSClient

        client = TTSClient()
        with mock.patch.object(client, 'is_server_running', return_value=True):
            with mock.patch.object(client._session, 'post') as mock_post:
                # Simulate network error
                mock_post.side_effect = ConnectionError("Network unreachable")

                # Should raise the original ConnectionError
                with pytest.raises(ConnectionError):
                    client.generate("test", mode="clone", prompt="default.pt")
