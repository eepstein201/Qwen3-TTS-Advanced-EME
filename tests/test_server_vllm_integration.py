"""Tests for VLLMAdapter integration with FastAPI server (HIGH-2)."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestVLLMLifecycleFunctions:
    """HIGH-2: lifespan helpers for starting/stopping VLLMAdapter."""

    def test_maybe_start_vllm_adapter_exists(self):
        """_maybe_start_vllm_adapter must be importable from app_lifespan."""
        from qwen3_tts.server.app_lifespan import _maybe_start_vllm_adapter
        assert callable(_maybe_start_vllm_adapter)

    def test_maybe_stop_vllm_adapter_exists(self):
        """_maybe_stop_vllm_adapter must be importable from app_lifespan."""
        from qwen3_tts.server.app_lifespan import _maybe_stop_vllm_adapter
        assert callable(_maybe_stop_vllm_adapter)

    @pytest.mark.asyncio
    async def test_start_skips_when_backend_not_vllm(self, monkeypatch):
        """_maybe_start_vllm_adapter does nothing when backend != vllm."""
        from qwen3_tts.server.app_lifespan import _maybe_start_vllm_adapter
        monkeypatch.setattr(
            "qwen3_tts.server.app_lifespan.get_backend", lambda: "mlx"
        )
        state = SimpleNamespace(vllm_adapter=None)
        await _maybe_start_vllm_adapter(state)
        assert state.vllm_adapter is None

    @pytest.mark.asyncio
    async def test_start_creates_adapter_when_backend_is_vllm(self, monkeypatch):
        """_maybe_start_vllm_adapter creates and starts adapter when backend=vllm."""
        from qwen3_tts.server.app_lifespan import _maybe_start_vllm_adapter

        # Mock load_config to return config with vLLM enabled
        mock_config = {
            "vllm": {
                "enabled": True,
                "fallback_to_torch": True,
            }
        }
        monkeypatch.setattr(
            "qwen3_tts.server.app_lifespan.load_config", lambda: mock_config
        )

        mock_adapter = MagicMock()
        mock_adapter.start = AsyncMock()
        mock_adapter.port = 5124  # Mock port for client creation
        mock_cls = MagicMock(return_value=mock_adapter)
        # Patch where the lazy import resolves, not where it's referenced
        monkeypatch.setattr(
            "qwen3_tts.core.engine_vllm.VLLMAdapter", mock_cls
        )

        state = SimpleNamespace(vllm_adapter=None)
        await _maybe_start_vllm_adapter(state)

        mock_adapter.start.assert_awaited_once()
        assert state.vllm_adapter is mock_adapter

    @pytest.mark.asyncio
    async def test_stop_calls_adapter_stop(self):
        """_maybe_stop_vllm_adapter calls stop() and sets vllm_adapter to None."""
        from qwen3_tts.server.app_lifespan import _maybe_stop_vllm_adapter

        mock_adapter = MagicMock()
        state = SimpleNamespace(vllm_adapter=mock_adapter)
        await _maybe_stop_vllm_adapter(state)

        mock_adapter.stop.assert_called_once()
        assert state.vllm_adapter is None

    @pytest.mark.asyncio
    async def test_stop_noop_when_no_adapter(self):
        """_maybe_stop_vllm_adapter does nothing when vllm_adapter is None."""
        from qwen3_tts.server.app_lifespan import _maybe_stop_vllm_adapter

        state = SimpleNamespace(vllm_adapter=None)
        await _maybe_stop_vllm_adapter(state)  # Must not raise
        assert state.vllm_adapter is None


class TestVLLMStateInitialization:
    """HIGH-2: app.state.vllm_adapter must be initialized in lifespan."""

    def test_lifespan_initializes_vllm_adapter_to_none(self):
        """The lifespan function sets app.state.vllm_adapter = None at startup."""
        import inspect

        from qwen3_tts.server import app_lifespan

        source = inspect.getsource(app_lifespan.lifespan)
        assert "vllm_adapter" in source, (
            "lifespan() must initialize app.state.vllm_adapter"
        )
