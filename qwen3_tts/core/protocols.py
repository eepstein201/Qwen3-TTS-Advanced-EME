"""Protocol definitions for dependency inversion.

This module defines Protocol classes that specify interfaces for:
- Configuration access (ConfigProvider)
- Audio generation (Generator)
- Server management (ServerManager)
- Prompt management (PromptManager)
- TTS server backends (TTSServerProtocol, VLLMBackendProtocol)

Using protocols enables:
- Testability via mock implementations
- Flexibility in implementation (file-based, database, API, etc.)
- Clear interface contracts between components
- Decoupling from specific backend implementations
"""

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol for configuration access.

    Implementations can provide config from:
    - JSON file (default)
    - Environment variables
    - Database
    - Remote API
    """

    def get_server_url(self) -> str:
        """Get the TTS server URL.

        Returns:
            Server URL string (e.g., "http://localhost:5123")
        """
        ...

    def get_generation_params(self) -> dict[str, Any]:
        """Get default generation parameters.

        Returns:
            Dict with temperature, top_k, top_p, etc.
        """
        ...

    def get_voice_prompts_dir(self) -> str:
        """Get the voice prompts directory path.

        Returns:
            Absolute path to voice prompts directory
        """
        ...

    def get_output_directory(self) -> str:
        """Get the output directory for generated audio.

        Returns:
            Absolute path to output directory
        """
        ...


@runtime_checkable
class Generator(Protocol):
    """Protocol for TTS generation.

    Implementations can:
    - Call remote server (default)
    - Use local model directly
    - Mock for testing
    """

    def generate(
        self,
        text: str,
        mode: str = "clone",
        **kwargs,
    ) -> str:
        """Generate audio from text.

        Args:
            text: Text to synthesize
            mode: Generation mode (clone/design/custom)
            **kwargs: Mode-specific parameters

        Returns:
            Path to generated audio file

        Raises:
            GenerationError: If generation fails
        """
        ...

    def generate_streaming(
        self,
        text: str,
        mode: str = "clone",
        **kwargs,
    ) -> Iterator[tuple]:
        """Generate audio with streaming.

        Args:
            text: Text to synthesize
            mode: Generation mode
            **kwargs: Mode-specific parameters

        Yields:
            Tuple of (audio_chunk, sample_rate)
        """
        ...


@runtime_checkable
class ServerManager(Protocol):
    """Protocol for server lifecycle management.

    Implementations can:
    - Manage local subprocess (default)
    - Connect to remote server
    - Mock for testing
    """

    def is_server_running(self) -> bool:
        """Check if server is running and healthy.

        Returns:
            True if server is running and responding
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get server statistics.

        Returns:
            Dict with memory, cache stats, etc.
        """
        ...

    def shutdown(self) -> None:
        """Gracefully shut down the server."""
        ...


@runtime_checkable
class PromptManager(Protocol):
    """Protocol for voice prompt management.

    Implementations can:
    - Manage local files (default)
    - Use database storage
    - Mock for testing
    """

    def list_prompts(self) -> list[str]:
        """List available voice prompts.

        Returns:
            List of prompt names
        """
        ...

    def delete_prompt(self, name: str) -> bool:
        """Delete a voice prompt.

        Args:
            name: Prompt name to delete

        Returns:
            True if deleted successfully
        """
        ...

    def rename_prompt(self, old_name: str, new_name: str) -> bool:
        """Rename a voice prompt.

        Args:
            old_name: Current prompt name
            new_name: New prompt name

        Returns:
            True if renamed successfully
        """
        ...

    def preview_prompt(self, name: str) -> bytes:
        """Get preview audio for a prompt.

        Args:
            name: Prompt name

        Returns:
            Audio bytes (WAV format)
        """
        ...


@runtime_checkable
class TTSServerProtocol(Protocol):
    """Protocol for TTS server backends (torch, MLX, vLLM).

    Implementations can:
    - Use local model (torch/MLX)
    - Connect to remote vLLM server
    - Mock for testing
    """

    async def generate(
        self,
        text: str,
        mode: str = "clone",
        prompt_audio: bytes | None = None,
        voice_description: str | None = None,
        speaker: str | None = None,
        **kwargs,
    ) -> tuple[int, Any]:
        """Generate audio from text asynchronously.

        Args:
            text: Text to synthesize
            mode: Generation mode (clone/design/custom)
            prompt_audio: Reference audio bytes (clone mode only)
            voice_description: Voice description (design mode only)
            speaker: Speaker name (custom mode only)
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (sample_rate, audio_array) where audio_array is float32 numpy array

        Raises:
            RuntimeError: If generation fails
        """
        ...

    async def health_check(self) -> bool:
        """Check if backend is healthy and ready.

        Returns:
            True if backend is ready to generate
        """
        ...

    async def load_model(self, model_type: str) -> None:
        """Load a model into memory.

        Args:
            model_type: Model type to load ("clone", "design", "custom")

        Raises:
            RuntimeError: If model loading fails
        """
        ...

    async def unload_model(self, model_type: str) -> None:
        """Unload a model from memory.

        Args:
            model_type: Model type to unload ("clone", "design", "custom")
        """
        ...


@runtime_checkable
class VLLMBackendProtocol(TTSServerProtocol, Protocol):
    """Extended protocol for vLLM-specific backends.

    Adds vLLM-specific methods like:
    - Circuit breaker management
    - Async HTTP client lifecycle
    - Fallback to torch/MLX on failure
    """

    async def is_ready(self) -> bool:
        """Check if vLLM server is ready.

        Returns:
            True if vLLM server is ready
        """
        ...

    @property
    def base_url(self) -> str:
        """Return the base URL for vLLM server.

        Returns:
            Base URL string (e.g., "http://127.0.0.1:5123")

        Raises:
            RuntimeError: If server not started
        """
        ...

    async def start(self) -> None:
        """Start the vLLM server subprocess.

        Raises:
            RuntimeError: If server fails to start
        """
        ...

    def stop(self) -> None:
        """Stop the vLLM server subprocess."""
        ...
