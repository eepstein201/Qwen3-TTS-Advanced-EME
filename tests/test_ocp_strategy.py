"""Tests for OCP strategy pattern in inference module."""
import pytest
from unittest import mock
import numpy as np


class TestInferenceStrategyRegistry:
    """Tests for inference strategy registry."""

    def test_registry_exists(self):
        """Inference strategy registry exists."""
        from qwen3_tts.core.engine.inference import _INFERENCE_STRATEGIES

        assert isinstance(_INFERENCE_STRATEGIES, dict)

    def test_registry_has_mlx_strategy(self):
        """Registry has MLX strategy."""
        from qwen3_tts.core.engine.inference import _INFERENCE_STRATEGIES

        assert "mlx" in _INFERENCE_STRATEGIES
        assert callable(_INFERENCE_STRATEGIES["mlx"])

    def test_registry_has_torch_strategy(self):
        """Registry has torch strategy."""
        from qwen3_tts.core.engine.inference import _INFERENCE_STRATEGIES

        assert "torch" in _INFERENCE_STRATEGIES
        assert callable(_INFERENCE_STRATEGIES["torch"])


class TestRegisterBackend:
    """Tests for register_backend function."""

    def test_register_backend_exists(self):
        """register_backend function exists."""
        from qwen3_tts.core.engine.inference import register_backend

        assert callable(register_backend)

    def test_register_custom_backend(self):
        """Can register a custom backend."""
        from qwen3_tts.core.engine.inference import register_backend, _INFERENCE_STRATEGIES

        def my_custom_backend(model, text, mode, **kwargs):
            return ([np.zeros(1000, dtype=np.float32)], 24000)

        register_backend("custom_test", my_custom_backend)
        assert "custom_test" in _INFERENCE_STRATEGIES

    def test_registered_backend_is_callable(self):
        """Registered backend is callable."""
        from qwen3_tts.core.engine.inference import register_backend, _INFERENCE_STRATEGIES

        def my_backend(model, text, mode, **kwargs):
            return ([np.zeros(1000, dtype=np.float32)], 24000)

        register_backend("test_callable", my_backend)
        assert callable(_INFERENCE_STRATEGIES["test_callable"])


class TestModeStrategyRegistry:
    """Tests for mode strategy registry."""

    def test_mode_registry_exists(self):
        """Mode strategy registry exists."""
        from qwen3_tts.core.engine.inference import _MODE_STRATEGIES_TORCH

        assert isinstance(_MODE_STRATEGIES_TORCH, dict)

    def test_mode_registry_has_clone(self):
        """Mode registry has clone strategy."""
        from qwen3_tts.core.engine.inference import _MODE_STRATEGIES_TORCH

        assert "clone" in _MODE_STRATEGIES_TORCH
        # Value is a method name (string) or callable
        value = _MODE_STRATEGIES_TORCH["clone"]
        assert isinstance(value, str) or callable(value)

    def test_mode_registry_has_design(self):
        """Mode registry has design strategy."""
        from qwen3_tts.core.engine.inference import _MODE_STRATEGIES_TORCH

        assert "design" in _MODE_STRATEGIES_TORCH
        value = _MODE_STRATEGIES_TORCH["design"]
        assert isinstance(value, str) or callable(value)

    def test_mode_registry_has_custom(self):
        """Mode registry has custom strategy."""
        from qwen3_tts.core.engine.inference import _MODE_STRATEGIES_TORCH

        assert "custom" in _MODE_STRATEGIES_TORCH
        value = _MODE_STRATEGIES_TORCH["custom"]
        assert isinstance(value, str) or callable(value)


class TestStrategyDispatch:
    """Tests for strategy-based dispatch."""

    def test_backend_dispatch_uses_registry(self):
        """_run_inference_single dispatches via registry."""
        from qwen3_tts.core.engine.inference import _INFERENCE_STRATEGIES

        # The registry should be used for dispatch
        # Verify the strategies have the right signature
        for backend, strategy in _INFERENCE_STRATEGIES.items():
            # Strategies should accept model, text, mode, and kwargs
            import inspect
            sig = inspect.signature(strategy)
            params = list(sig.parameters.keys())
            assert "model" in params or len(params) >= 1

    def test_can_add_new_backend_without_modifying_code(self):
        """Can add new backend without modifying existing code (OCP)."""
        from qwen3_tts.core.engine.inference import register_backend, _INFERENCE_STRATEGIES

        initial_count = len(_INFERENCE_STRATEGIES)

        # Register a mock backend
        def mock_backend(model, text, mode, **kwargs):
            return ([np.zeros(100, dtype=np.float32)], 24000)

        register_backend("mock_ocp_test", mock_backend)

        # Should have one more backend
        assert len(_INFERENCE_STRATEGIES) == initial_count + 1

        # Clean up
        del _INFERENCE_STRATEGIES["mock_ocp_test"]
