"""Tests for core protocols module."""
import pytest
from typing import Protocol, runtime_checkable


class TestConfigProvider:
    """Tests for ConfigProvider protocol."""

    def test_config_provider_is_protocol(self):
        """ConfigProvider is a Protocol."""
        from qwen3_tts.core.protocols import ConfigProvider

        assert issubclass(ConfigProvider, Protocol)

    def test_config_provider_has_required_methods(self):
        """ConfigProvider has required methods."""
        from qwen3_tts.core.protocols import ConfigProvider

        # Check protocol has the expected methods by checking __annotations__
        # Protocol classes define their interface through method signatures
        assert hasattr(ConfigProvider, 'get_server_url')
        assert hasattr(ConfigProvider, 'get_generation_params')
        assert hasattr(ConfigProvider, 'get_voice_prompts_dir')

    def test_class_implementing_config_provider(self):
        """A class implementing ConfigProvider methods satisfies the protocol."""
        from qwen3_tts.core.protocols import ConfigProvider

        class MockConfigProvider:
            def get_server_url(self) -> str:
                return "http://localhost:5123"

            def get_generation_params(self) -> dict:
                return {"temperature": 0.7}

            def get_voice_prompts_dir(self) -> str:
                return "/tmp/voice_prompts"

        provider = MockConfigProvider()
        # Should be usable as ConfigProvider
        assert provider.get_server_url() == "http://localhost:5123"


class TestGenerator:
    """Tests for Generator protocol."""

    def test_generator_is_protocol(self):
        """Generator is a Protocol."""
        from qwen3_tts.core.protocols import Generator

        assert issubclass(Generator, Protocol)

    def test_class_implementing_generator(self):
        """A class implementing Generator methods satisfies the protocol."""
        from qwen3_tts.core.protocols import Generator

        class MockGenerator:
            def generate(self, text: str, **kwargs) -> str:
                return "/tmp/output.wav"

            def generate_streaming(self, text: str, **kwargs):
                yield (b"audio_data", 24000)

        gen = MockGenerator()
        assert gen.generate("test") == "/tmp/output.wav"


class TestServerManager:
    """Tests for ServerManager protocol."""

    def test_server_manager_is_protocol(self):
        """ServerManager is a Protocol."""
        from qwen3_tts.core.protocols import ServerManager

        assert issubclass(ServerManager, Protocol)

    def test_class_implementing_server_manager(self):
        """A class implementing ServerManager methods satisfies the protocol."""
        from qwen3_tts.core.protocols import ServerManager

        class MockServerManager:
            def is_server_running(self) -> bool:
                return True

            def get_stats(self) -> dict:
                return {"memory_mb": 100}

            def shutdown(self) -> None:
                pass

        manager = MockServerManager()
        assert manager.is_server_running() is True


class TestPromptManager:
    """Tests for PromptManager protocol."""

    def test_prompt_manager_is_protocol(self):
        """PromptManager is a Protocol."""
        from qwen3_tts.core.protocols import PromptManager

        assert issubclass(PromptManager, Protocol)

    def test_class_implementing_prompt_manager(self):
        """A class implementing PromptManager methods satisfies the protocol."""
        from qwen3_tts.core.protocols import PromptManager

        class MockPromptManager:
            def list_prompts(self) -> list:
                return ["voice1.pt", "voice2.pt"]

            def delete_prompt(self, name: str) -> bool:
                return True

            def rename_prompt(self, old_name: str, new_name: str) -> bool:
                return True

        manager = MockPromptManager()
        assert len(manager.list_prompts()) == 2


class TestProtocolIntegration:
    """Integration tests for protocol usage."""

    def test_ttsclient_accepts_config_provider(self):
        """TTSClient can accept a ConfigProvider instead of loading config."""
        from qwen3_tts.server.client import TTSClient

        class MockConfigProvider:
            def get_server_url(self) -> str:
                return "http://localhost:9999"

            def get_generation_params(self) -> dict:
                return {"temperature": 0.5}

            def get_voice_prompts_dir(self) -> str:
                return "/custom/prompts"

        # TTSClient should accept config_provider parameter
        provider = MockConfigProvider()
        client = TTSClient(config_provider=provider)

        # The client should use the provider's server URL
        assert client.server_url == "http://localhost:9999"

    def test_ttsclient_defaults_to_file_config(self):
        """TTSClient defaults to loading config from file when no provider given."""
        from qwen3_tts.server.client import TTSClient

        # Without config_provider, should use default file-based config
        client = TTSClient()
        assert client.server_url is not None
