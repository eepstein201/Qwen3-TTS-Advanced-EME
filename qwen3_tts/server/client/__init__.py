"""
TTS Client package - HTTP-only Python API for Qwen3-TTS generation.

This package NEVER imports torch or qwen3_tts.core.engine — it communicates
exclusively over HTTP to the TTS server.

Usage:
    from qwen3_tts.server.client import TTSClient

    client = TTSClient()

    # Generate with default voice
    audio_path = client.generate("Hello world", output="greeting.wav")

    # Generate with specific voice prompt
    audio_path = client.generate("Hello", prompt="narrator.pt")

    # Generate with voice alias
    audio_path = client.generate("Hello", voice="narrator")

    # Generate with premium speaker (custom mode)
    audio_path = client.generate(
        "Hello",
        mode="custom",
        speaker="Ryan",
        instruct="Speak with enthusiasm"
    )

    # Check server status
    if client.is_server_running():
        stats = client.get_stats()
        print(f"Memory: {stats['mps_memory_allocated_mb']}MB")
"""

from qwen3_tts.server.client._base import _ClientBase
from qwen3_tts.server.client.generator import GeneratorMixin
from qwen3_tts.server.client.models import ModelManagerMixin
from qwen3_tts.server.client.voices import VoiceManagerMixin
from qwen3_tts.server.client.config_fetcher import ConfigFetcherMixin


class TTSClient(GeneratorMixin, ModelManagerMixin, VoiceManagerMixin, ConfigFetcherMixin, _ClientBase):
    """HTTP-only client for Qwen3-TTS generation."""
    pass


__all__ = ["TTSClient", "generate"]


def generate(text, **kwargs):
    """Generate speech from text using default settings.

    This is a convenience function that creates a TTSClient and calls generate().
    For repeated use, create a TTSClient instance instead.

    Args:
        text: Text to synthesize
        **kwargs: Additional arguments passed to TTSClient.generate()

    Returns:
        Path to the generated audio file
    """
    client = TTSClient()
    try:
        return client.generate(text, **kwargs)
    finally:
        client.close()


if __name__ == "__main__":
    # Example usage
    client = TTSClient()
    print("TTS Client Library")
    print(f"Server running: {client.is_server_running()}")
    print(f"Available prompts: {client.list_prompts()}")
    print(f"Available presets: {list(client.list_presets().keys())}")
    print(f"Available aliases: {list(client.list_aliases().keys())}")
