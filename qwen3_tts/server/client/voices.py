"""
VoiceManagerMixin — voice prompt management methods for TTSClient.

Handles list_prompts(), delete_prompt(), rename_prompt(), preview_prompt(),
and get_prompt_details().

This module NEVER imports torch or qwen3_tts.core.engine at module scope.
"""

import os

import requests

from qwen3_tts.server.client._base import _require_server
from qwen3_tts.core.config import auth_headers, VoicePromptError, VOICE_PROMPTS_DIR


class VoiceManagerMixin:
    """Mixin providing voice prompt management capabilities."""

    def list_prompts(self):
        """List available voice prompts.

        Uses the server /prompts endpoint when running (returns backend-aware list),
        falls back to local filesystem listing.
        """
        if self.is_server_running():
            try:
                resp = self._session.get(f"{self.server_url}/prompts", timeout=5, headers=auth_headers())
                if resp.status_code == 200:
                    return resp.json().get("prompts", [])
            except Exception:  # nosec B110
                pass
        # Fallback to local filesystem (.pt for torch, .wav+.txt for MLX)
        try:
            files = os.listdir(self.voice_prompts_dir)
        except OSError:
            return []
        pt_prompts = {f for f in files if f.endswith('.pt')}
        txt_bases = {f[:-4] for f in files if f.endswith('.txt')}
        mlx_prompts = {f for f in files if f.endswith('.wav') and f[:-4] in txt_bases}
        return sorted(pt_prompts | mlx_prompts)

    @_require_server
    def delete_prompt(self, name):
        """Delete a voice prompt and all its format files.

        Args:
            name: Voice prompt name (with or without extension)

        Returns:
            Response dict with status and files_removed list
        """
        resp = self._session.post(
            f"{self.server_url}/delete-prompt",
            json={"name": name},
            timeout=10,
            headers=auth_headers(),
        )
        if resp.status_code != 200:
            error_msg = resp.json().get("error", "Unknown error")
            raise VoicePromptError("delete", error_msg)
        return resp.json()

    @_require_server
    def rename_prompt(self, old_name, new_name):
        """Rename a voice prompt (all format files).

        Args:
            old_name: Current prompt name
            new_name: New prompt name

        Returns:
            Response dict with status and files_renamed list
        """
        resp = self._session.post(
            f"{self.server_url}/rename-prompt",
            json={"old_name": old_name, "new_name": new_name},
            timeout=10,
            headers=auth_headers(),
        )
        if resp.status_code != 200:
            error_msg = resp.json().get("error", "Unknown error")
            raise VoicePromptError("rename", error_msg)
        return resp.json()

    @_require_server
    def preview_prompt(self, name):
        """Get the .wav audio data for a voice prompt.

        Args:
            name: Voice prompt name

        Returns:
            Raw bytes of the .wav file
        """
        resp = self._session.get(
            f"{self.server_url}/preview-prompt",
            params={"name": name},
            timeout=10,
            headers=auth_headers(),
        )
        if resp.status_code != 200:
            try:
                error_msg = resp.json().get("error", "Unknown error")
            except (ValueError, requests.exceptions.JSONDecodeError):
                error_msg = f"HTTP {resp.status_code}"
            raise VoicePromptError("preview", error_msg)
        return resp.content

    @_require_server
    def get_prompt_details(self, name=None):
        """Get metadata for voice prompts.

        Args:
            name: Prompt name for single prompt details, or None for all prompts

        Returns:
            Dict with prompt metadata (single) or {"prompts": [...]} (all)
        """
        params = {"name": name} if name else {}
        resp = self._session.get(
            f"{self.server_url}/prompt-details",
            params=params,
            timeout=10,
            headers=auth_headers(),
        )
        if resp.status_code != 200:
            error_msg = resp.json().get("error", "Unknown error")
            raise VoicePromptError("details", error_msg)
        return resp.json()
