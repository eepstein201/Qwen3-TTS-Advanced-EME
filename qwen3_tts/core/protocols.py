"""Protocol definitions for dependency inversion.

This module defines Protocol classes that specify interfaces for:
- Configuration access (ConfigProvider)
- Audio generation (Generator)
- Server management (ServerManager)
- Prompt management (PromptManager)

Using protocols enables:
- Testability via mock implementations
- Flexibility in implementation (file-based, database, API, etc.)
- Clear interface contracts between components
"""
from typing import Protocol, runtime_checkable, Iterator, List, Dict, Any, Optional


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

    def get_generation_params(self) -> Dict[str, Any]:
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

    def get_stats(self) -> Dict[str, Any]:
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

    def list_prompts(self) -> List[str]:
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


# =============================================================================
# Default implementations for backward compatibility
# =============================================================================

class FileConfigProvider:
    """File-based configuration provider (default implementation)."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional config path.

        Args:
            config_path: Path to config.json. Uses default if None.
        """
        from qwen3_tts.core.config import CONFIG_PATH, load_config
        self._config_path = config_path or CONFIG_PATH
        self._load_config = load_config
        self._config_cache = None

    @property
    def _config(self) -> dict:
        """Lazy-load config."""
        if self._config_cache is None:
            self._config_cache = self._load_config()
        return self._config_cache

    def get_server_url(self) -> str:
        """Get server URL from config."""
        from qwen3_tts.core.config import get_server_url
        return get_server_url(self._config)

    def get_generation_params(self) -> Dict[str, Any]:
        """Get generation params from config."""
        return self._config.get("generation", {})

    def get_voice_prompts_dir(self) -> str:
        """Get voice prompts directory."""
        from qwen3_tts.core.config import VOICE_PROMPTS_DIR
        return str(VOICE_PROMPTS_DIR)

    def get_output_directory(self) -> str:
        """Get output directory from config."""
        return self._config.get("output_directory", "~/Downloads")


class DefaultPromptManager:
    """Default prompt manager using HTTP client."""

    def __init__(self, server_url: str, auth_token: str):
        """Initialize with server connection.

        Args:
            server_url: TTS server URL
            auth_token: Authentication token
        """
        self._server_url = server_url
        self._auth_token = auth_token
        self._session = None

    def _get_session(self):
        """Lazy-create HTTP session."""
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def list_prompts(self) -> List[str]:
        """List prompts via HTTP."""
        resp = self._get_session().get(
            f"{self._server_url}/prompts",
            headers={"Authorization": f"Bearer {self._auth_token}"}
        )
        resp.raise_for_status()
        return [p["name"] for p in resp.json().get("prompts", [])]

    def delete_prompt(self, name: str) -> bool:
        """Delete prompt via HTTP."""
        resp = self._get_session().post(
            f"{self._server_url}/delete-prompt",
            json={"name": name},
            headers={"Authorization": f"Bearer {self._auth_token}"}
        )
        return resp.status_code == 200

    def rename_prompt(self, old_name: str, new_name: str) -> bool:
        """Rename prompt via HTTP."""
        resp = self._get_session().post(
            f"{self._server_url}/rename-prompt",
            json={"old_name": old_name, "new_name": new_name},
            headers={"Authorization": f"Bearer {self._auth_token}"}
        )
        return resp.status_code == 200

    def preview_prompt(self, name: str) -> bytes:
        """Get preview audio via HTTP."""
        resp = self._get_session().get(
            f"{self._server_url}/preview-prompt",
            params={"name": name},
            headers={"Authorization": f"Bearer {self._auth_token}"}
        )
        resp.raise_for_status()
        return resp.content
