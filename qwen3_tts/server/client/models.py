"""
ModelManagerMixin — model lifecycle management methods for TTSClient.

Handles load_model(), unload_model(), get_models(), update_model_config(),
and update_startup_config().

This module NEVER imports torch or qwen3_tts.core.engine at module scope.
"""

from qwen3_tts.core.config import ModelError, auth_headers
from qwen3_tts.core.http_client import LOAD_MODEL_TIMEOUT_SEC
from qwen3_tts.server.client._base import _extract_error_message, _require_server


class ModelManagerMixin:
    """Mixin providing model lifecycle management capabilities."""

    @_require_server
    def load_model(self, mode):
        """Request the server to load a model on demand.

        Args:
            mode: Model type — "clone", "design", or "custom".

        Returns:
            Response dict with "status" key ("loaded" or "already_loaded").
        """
        resp = self._session.post(
            f"{self.server_url}/load-model",
            json={"model_type": mode},
            timeout=LOAD_MODEL_TIMEOUT_SEC,
            headers=auth_headers(),
        )
        if resp.status_code != 200:
            raise ModelError(mode, "load", _extract_error_message(resp))
        return resp.json()

    @_require_server
    def unload_model(self, mode):
        """Unload a model to free memory.

        Args:
            mode: Model type — "clone", "design", or "custom".

        Returns:
            Response dict with "status" key ("unloaded" or "already_unloaded").
        """
        resp = self._session.post(
            f"{self.server_url}/unload-model",
            json={"model_type": mode},
            timeout=10,
            headers=auth_headers(),
        )
        if resp.status_code not in (200, 409):
            raise ModelError(mode, "unload", _extract_error_message(resp))
        return resp.json()

    @_require_server
    def get_models(self):
        """Get information about available models and their load status.

        Returns:
            Response dict with "models", "backend", "model_size" keys.
        """
        resp = self._session.get(
            f"{self.server_url}/models",
            timeout=5,
            headers=auth_headers(),
        )
        return resp.json()

    @_require_server
    def update_model_config(self, model_size=None, mlx_quantization=None):
        """Update model size and/or quantization settings.

        Args:
            model_size: "1.7B" or "0.6B" (optional).
            mlx_quantization: "4bit", "5bit", "6bit", "8bit", or "bf16" (optional).

        Returns:
            Response dict with "status", "changes", "models_unloaded" keys.

        The server will unload current models and load the new variant
        on the next generation request.
        """
        data = {}
        if model_size:
            data["model_size"] = model_size
        if mlx_quantization:
            data["mlx_quantization"] = mlx_quantization

        if not data:
            raise ValueError("At least one of model_size or mlx_quantization required")

        resp = self._session.post(
            f"{self.server_url}/update-model-config",
            json=data,
            timeout=10,
            headers=auth_headers(),
        )
        if resp.status_code != 200:
            raise ModelError("config", "update", _extract_error_message(resp))
        return resp.json()

    @_require_server
    def update_startup_config(self, clone=None, design=None, custom=None):
        """Update which models load at server startup.

        Args:
            clone: True/False to enable/disable clone model at startup (optional).
            design: True/False to enable/disable design model at startup (optional).
            custom: True/False to enable/disable custom model at startup (optional).

        Returns:
            Response dict with "status" and "changes" keys.
        """
        data = {}
        if clone is not None:
            data["clone"] = clone
        if design is not None:
            data["design"] = design
        if custom is not None:
            data["custom"] = custom
        if not data:
            raise ValueError("At least one model type required")
        resp = self._session.post(
            f"{self.server_url}/update-startup-config",
            json=data,
            timeout=10,
            headers=auth_headers(),
        )
        if resp.status_code != 200:
            raise ModelError("startup", "update", _extract_error_message(resp))
        return resp.json()
