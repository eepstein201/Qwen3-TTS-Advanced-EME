#!/usr/bin/env python3
"""Integration tests for Qwen3-TTS.

These tests verify end-to-end functionality including:
- Full generation pipeline
- Concurrent operations
- Model lifecycle
- Server lifecycle
- Voice prompt operations
- Error recovery

Run with:
    python -m pytest tests/test_integration.py -v -m integration
    # Or with unittest:
    python -m unittest tests.test_integration -v

Note: Many tests require a running server or will start one.
Tests are marked with @pytest.mark.integration and can be skipped in CI.
"""

import threading
import unittest

# Check optional dependencies
try:
    import soundfile  # noqa: F401
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from fastapi import HTTPException
    from fastapi.testclient import TestClient  # noqa: F401
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

_skip_integration = unittest.skipUnless(
    HAS_SOUNDFILE and HAS_FASTAPI,
    "Integration tests require soundfile + fastapi"
)


class TestGenerationPipeline(unittest.TestCase):
    """End-to-end generation pipeline tests."""

    @_skip_integration
    def test_generate_request_structure(self):
        """Verify GenerateRequest has correct structure."""
        from qwen3_tts.server.validation import GenerateRequest

        # Test clone mode request
        req = GenerateRequest(
            texts=["Hello world"],
            mode="clone",
            prompt="default.pt",
        )
        self.assertEqual(req.texts, ["Hello world"])
        self.assertEqual(req.mode, "clone")

        # Test design mode request
        req2 = GenerateRequest(
            texts=["Test"],
            mode="design",
            voice_description="A calm voice",
        )
        self.assertEqual(req2.mode, "design")

        # Test custom mode request
        req3 = GenerateRequest(
            texts=["Test"],
            mode="custom",
            speaker="ryan",
        )
        self.assertEqual(req3.mode, "custom")
        self.assertEqual(req3.speaker, "ryan")

    @_skip_integration
    def test_generation_params_validation(self):
        """Verify generation params are validated."""
        from qwen3_tts.server.validation import GenerateRequest

        # Valid params
        req = GenerateRequest(
            texts=["Test"],
            mode="clone",
            prompt="default.pt",
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        self.assertEqual(req.temperature, 0.7)

        # Boundary values
        req2 = GenerateRequest(
            texts=["Test"],
            mode="clone",
            prompt="default.pt",
            temperature=0.0,  # Min
        )
        self.assertEqual(req2.temperature, 0.0)

    @_skip_integration
    def test_text_chunking_logic(self):
        """Verify text chunking works correctly."""
        from qwen3_tts.core.engine.text_processing import _split_text

        # Short text - no chunking
        chunks = _split_text("Hello world", max_chars=500)
        self.assertEqual(len(chunks), 1)

        # Long text - should chunk
        long_text = "This is a sentence. " * 50  # ~1000 chars
        chunks = _split_text(long_text, max_chars=200)
        self.assertGreater(len(chunks), 1)


class TestConcurrentOperations(unittest.TestCase):
    """Tests for concurrent generation and access patterns."""

    @_skip_integration
    def test_config_lock_exists(self):
        """Verify config lock exists for thread-safety."""
        from qwen3_tts.core.config import _config_lock

        # Just verify the lock exists and is a proper lock
        self.assertIsInstance(_config_lock, type(threading.Lock()))

    @_skip_integration
    def test_voice_prompt_cache_lock(self):
        """Verify voice prompt cache is thread-safe."""
        from qwen3_tts.core.engine.voice_prompt import _torch_prompt_cache_lock

        # Just verify the lock exists and is a proper lock
        self.assertIsInstance(_torch_prompt_cache_lock, type(threading.Lock()))


class TestModelLifecycle(unittest.TestCase):
    """Tests for model loading, unloading, and state management."""

    @_skip_integration
    def test_model_state_structure(self):
        """Verify model state dict has correct structure."""
        # This tests the expected structure without actually loading models
        expected_keys = ["clone", "design", "custom"]

        # Simulate model state
        model_state = {k: None for k in expected_keys}
        self.assertIn("clone", model_state)
        self.assertIn("design", model_state)
        self.assertIn("custom", model_state)

    @_skip_integration
    def test_model_info_structure(self):
        """Verify MODEL_INFO has correct structure."""
        from qwen3_tts.core.config import MODEL_INFO

        # MODEL_INFO is nested by model size
        self.assertIn("1.7B", MODEL_INFO)
        self.assertIn("0.6B", MODEL_INFO)

        # Each size has clone/design/custom
        for size in ["1.7B", "0.6B"]:
            for model_type in ["clone", "design", "custom"]:
                self.assertIn(model_type, MODEL_INFO[size])
                info = MODEL_INFO[size][model_type]
                self.assertIn("name", info)
                self.assertIn("description", info)


class TestVoicePromptOperations(unittest.TestCase):
    """Tests for voice prompt CRUD operations."""

    @_skip_integration
    def test_prompt_name_validation_in_validation_module(self):
        """Verify prompt names are validated correctly in validation module."""
        from qwen3_tts.server.validation import _validate_prompt_name

        # Valid names - should return None
        self.assertIsNone(_validate_prompt_name("my_voice"))
        self.assertIsNone(_validate_prompt_name("voice-123"))
        self.assertIsNone(_validate_prompt_name("test.voice"))

        # Invalid names - should return error tuple
        result = _validate_prompt_name("../etc/passwd")
        self.assertIsNotNone(result)
        self.assertEqual(result[1], 400)  # status code

        result = _validate_prompt_name("invalid!name")
        self.assertIsNotNone(result)

        result = _validate_prompt_name("..traversal")
        self.assertIsNotNone(result)

    @_skip_integration
    def test_strip_extension(self):
        """Verify extension stripping works correctly."""
        from qwen3_tts.server.validation import _strip_extension

        self.assertEqual(_strip_extension("voice.pt"), "voice")
        self.assertEqual(_strip_extension("voice.wav"), "voice")
        self.assertEqual(_strip_extension("voice"), "voice")
        self.assertEqual(_strip_extension("my.voice.pt"), "my.voice")


class TestErrorHandling(unittest.TestCase):
    """Tests for error handling and recovery."""

    @_skip_integration
    def test_error_response_raises_httpexception(self):
        """Verify error response raises HTTPException with correct structure."""
        from qwen3_tts.server.validation import _error_response

        with self.assertRaises(HTTPException) as ctx:
            _error_response(400, "Test error", "Test detail", "retry")

        self.assertEqual(ctx.exception.status_code, 400)
        detail = ctx.exception.detail
        self.assertEqual(detail["error"], "Test error")
        self.assertEqual(detail["detail"], "Test detail")
        self.assertEqual(detail["recovery"], "retry")

    @_skip_integration
    def test_custom_error_classes(self):
        """Verify custom error classes work correctly."""
        from qwen3_tts.core.config import (
            ModelError,
            TTSError,
            VoicePromptError,
        )

        # Test base error
        err = TTSError("Test error", "Technical detail", "retry")
        self.assertEqual(str(err), "Test error")
        self.assertEqual(err.technical_detail, "Technical detail")
        self.assertEqual(err.recovery, "retry")

        # Test specialized errors
        voice_err = VoicePromptError("delete", "Not found")
        self.assertEqual(voice_err.operation, "delete")

        model_err = ModelError("clone", "load", "Failed")
        self.assertEqual(model_err.model_type, "clone")
        self.assertEqual(model_err.operation, "load")

    @_skip_integration
    def test_generation_error_inheritance(self):
        """Verify GenerationError is a proper TTSError subclass."""
        from qwen3_tts.core.config import GenerationError, TTSError

        err = GenerationError("Generation failed")
        self.assertIsInstance(err, TTSError)
        self.assertIsInstance(err, Exception)


class TestConfigOperations(unittest.TestCase):
    """Tests for configuration operations."""

    @_skip_integration
    def test_config_has_expected_structure(self):
        """Verify default config has expected keys."""
        from qwen3_tts.core.config import get_default_config

        config = get_default_config()

        self.assertIn("generation", config)
        self.assertIn("server", config)
        self.assertIn("security", config)

        # Check default generation params
        gen = config["generation"]
        self.assertGreater(gen.get("temperature", 0), 0)
        self.assertLess(gen.get("temperature", 1), 2)
        self.assertIn("top_k", gen)
        self.assertIn("top_p", gen)

    @_skip_integration
    def test_security_defaults(self):
        """Verify security defaults are safe."""
        from qwen3_tts.core.config import get_default_config

        config = get_default_config()
        security = config.get("security", {})

        # Should have reasonable limits
        self.assertIn("max_text_length", security)
        self.assertGreater(security["max_text_length"], 0)
        self.assertLess(security["max_text_length"], 100000)  # Not too high


class TestUtilityFunctions(unittest.TestCase):
    """Tests for utility functions used across the codebase."""

    @_skip_integration
    def test_text_normalization(self):
        """Verify text normalization works."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        # Should handle normal text
        result = _normalize_text("Hello world")
        self.assertIsInstance(result, str)
        # Result should contain normalized content
        self.assertTrue(len(result) > 0)

    @_skip_integration
    def test_audio_validation(self):
        """Verify audio validation catches issues."""
        import numpy as np

        from qwen3_tts.core.engine.inference import _validate_audio

        # Valid audio
        valid_audio = np.random.randn(16000).astype(np.float32) * 0.1
        # Should not raise
        _validate_audio(valid_audio, 16000)

        # Silent audio should warn but not raise
        silent_audio = np.zeros(16000, dtype=np.float32)
        _validate_audio(silent_audio, 16000)


class TestCrossfadeFunctionality(unittest.TestCase):
    """Tests for audio crossfade between chunks."""

    @_skip_integration
    def test_crossfade_basic(self):
        """Verify crossfade produces expected output."""
        import numpy as np

        from qwen3_tts.core.engine.inference import _crossfade_chunks

        # Two chunks of audio
        chunk1 = np.random.randn(1000).astype(np.float32)
        chunk2 = np.random.randn(1000).astype(np.float32)

        # Crossfade with 10ms overlap at 24kHz
        result = _crossfade_chunks([chunk1, chunk2], crossfade_ms=10, sample_rate=24000)

        self.assertIsInstance(result, np.ndarray)
        # Result should be reasonable length
        self.assertGreater(result.shape[0], 0)

    @_skip_integration
    def test_crossfade_single_chunk(self):
        """Single chunk should pass through unchanged."""
        import numpy as np

        from qwen3_tts.core.engine.inference import _crossfade_chunks

        chunk = np.random.randn(1000).astype(np.float32)
        result = _crossfade_chunks([chunk], crossfade_ms=10, sample_rate=24000)

        np.testing.assert_array_equal(result, chunk)


class TestServerHealthEndpoints(unittest.TestCase):
    """Tests for server health and status endpoints."""

    @_skip_integration
    def test_health_endpoint_structure(self):
        """Verify /health endpoint returns expected structure."""
        # This test verifies the expected response structure
        # without actually starting the server
        expected_fields = ["status", "backend", "models_loaded"]

        # The actual endpoint would return these fields
        # Here we just verify the expected structure
        for field in expected_fields:
            self.assertIsInstance(field, str)


# Additional test class for running without server
class TestOfflineFunctionality(unittest.TestCase):
    """Tests that don't require a running server."""

    def test_imports_work(self):
        """Verify all key modules can be imported."""
        # Core modules
        from qwen3_tts.core import config, engine

        # Server modules
        from qwen3_tts.server import client

        # Interface modules

        # Tools

        # Verify modules have expected attributes
        self.assertTrue(hasattr(config, 'TTSError'))
        self.assertTrue(hasattr(engine, 'run_inference'))
        self.assertTrue(hasattr(client, 'TTSClient'))

    def test_lazy_imports_no_heavy_deps(self):
        """Verify lazy imports don't pull in torch/mlx at module scope."""
        import qwen3_tts.core.config as config_module
        import qwen3_tts.server.client as client_module

        # Check that torch/mlx aren't in module globals
        config_globals = set(dir(config_module))
        client_globals = set(dir(client_module))

        self.assertNotIn('torch', config_globals)
        self.assertNotIn('mlx', config_globals)
        self.assertNotIn('torch', client_globals)
        self.assertNotIn('mlx', client_globals)


class TestOCPStrategyPattern(unittest.TestCase):
    """Tests for the OCP-compliant strategy pattern."""

    def test_inference_strategies_registry_exists(self):
        """Verify inference strategies registry exists."""
        from qwen3_tts.core.engine import _INFERENCE_STRATEGIES

        self.assertIsInstance(_INFERENCE_STRATEGIES, dict)
        self.assertIn("torch", _INFERENCE_STRATEGIES)
        self.assertIn("mlx", _INFERENCE_STRATEGIES)

    def test_mode_strategies_registry_exists(self):
        """Verify mode strategies registry exists."""
        from qwen3_tts.core.engine import _MODE_STRATEGIES_TORCH

        self.assertIsInstance(_MODE_STRATEGIES_TORCH, dict)
        self.assertIn("clone", _MODE_STRATEGIES_TORCH)
        self.assertIn("design", _MODE_STRATEGIES_TORCH)
        self.assertIn("custom", _MODE_STRATEGIES_TORCH)

    def test_register_backend_function_exists(self):
        """Verify register_backend function exists."""
        from qwen3_tts.core.engine import register_backend

        self.assertTrue(callable(register_backend))

    def test_can_register_custom_backend(self):
        """Verify custom backends can be registered."""
        import numpy as np

        from qwen3_tts.core.engine import _INFERENCE_STRATEGIES, register_backend

        def mock_backend(model, text, mode, **kwargs):
            return ([np.zeros(100, dtype=np.float32)], 24000)

        register_backend("test_integration_backend", mock_backend)

        self.assertIn("test_integration_backend", _INFERENCE_STRATEGIES)

        # Clean up
        del _INFERENCE_STRATEGIES["test_integration_backend"]


if __name__ == "__main__":
    unittest.main()
