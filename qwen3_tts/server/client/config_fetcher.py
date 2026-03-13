"""
ConfigFetcherMixin — config and stats retrieval methods for TTSClient.

Handles list_presets(), list_aliases(), resolve_alias(), get_stats(),
and get_health().

This module NEVER imports torch or qwen3_tts.core.engine at module scope.
"""

import requests

from qwen3_tts.server.client._base import _require_server
from qwen3_tts.core.config import auth_headers


class ConfigFetcherMixin:
    """Mixin providing config and stats retrieval capabilities."""

    def list_presets(self):
        """List available presets."""
        return self.config.get("presets", {})

    def list_aliases(self):
        """List available voice aliases."""
        return self.config.get("aliases", {})

    def resolve_alias(self, alias_name):
        """Resolve a voice alias to its settings."""
        aliases = self.config.get("aliases", {})
        return aliases.get(alias_name)

    @_require_server
    def get_stats(self):
        """Get server statistics."""
        resp = self._session.get(f"{self.server_url}/stats", timeout=5, headers=auth_headers())
        return resp.json()

    @_require_server
    def get_health(self):
        """Get server health info including loaded models and backend."""
        resp = self._session.get(f"{self.server_url}/health", timeout=5)
        return resp.json()
