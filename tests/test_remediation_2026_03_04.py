#!/usr/bin/env python3
"""Tests for code review remediation (R-1 through R-12 features).

Covers: _validate_audio, _validate_generation_request, _error_response,
voice prompt cache thread safety, cache info, Pydantic response models.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_remediation_2026_03_04.py -v

No GPU, models, or running server required.
"""

import os
import sys
import threading
import unittest
from unittest.mock import patch, MagicMock

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    class _DummyMarkerFunc:
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition, **kwargs):
            return lambda f: f
    class _DummyMarker:
        def __call__(self, func):
            return func
        def __getattr__(self, name):
            if name == 'skipif':
                return _DummyMarkerFunc(name)
            return _DummyMarkerFunc(name)
        @property
        def unit(self):
            return self
    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()
    class _DummyPytest:
        mark = _DummyMark()
    pytest = _DummyPytest()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import fastapi  # noqa: F401
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

try:
    from pydantic import BaseModel  # noqa: F401
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

_skip_numpy = unittest.skipUnless(HAS_NUMPY, "requires numpy")
_skip_fastapi = unittest.skipUnless(HAS_FASTAPI, "requires fastapi")
_skip_pydantic = unittest.skipUnless(HAS_PYDANTIC, "requires pydantic")


# =========================================================================
# Audio Validation Tests (R-11)
# =========================================================================

@_skip_numpy
@pytest.mark.unit
class TestValidateAudio(unittest.TestCase):
    """Test _validate_audio() from engine.py."""

    def _get_validate_audio(self):
        from qwen3_tts.core.engine import _validate_audio
        return _validate_audio

    def test_normal_audio_unchanged(self):
        """Normal audio passes through without modification."""
        validate = self._get_validate_audio()
        audio = np.array([0.1, -0.2, 0.5, -0.3], dtype=np.float32)
        result = validate(audio, 24000)
        np.testing.assert_array_equal(result, audio)

    def test_nan_replaced_with_zeros(self):
        """NaN values are replaced with zeros."""
        validate = self._get_validate_audio()
        audio = np.array([0.1, float('nan'), 0.5, float('nan')], dtype=np.float32)
        result = validate(audio, 24000)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertEqual(result[1], 0.0)
        self.assertEqual(result[3], 0.0)

    def test_clipping_normalized(self):
        """Audio exceeding [-1, 1] is normalized."""
        validate = self._get_validate_audio()
        audio = np.array([2.0, -3.0, 1.5], dtype=np.float32)
        result = validate(audio, 24000)
        self.assertLessEqual(np.max(np.abs(result)), 1.0)

    def test_silence_passes_through(self):
        """All-zeros audio passes through (warning only, no modification)."""
        validate = self._get_validate_audio()
        audio = np.zeros(1000, dtype=np.float32)
        result = validate(audio, 24000)
        np.testing.assert_array_equal(result, audio)

    def test_empty_audio_passes_through(self):
        """Empty array passes through without error."""
        validate = self._get_validate_audio()
        audio = np.array([], dtype=np.float32)
        result = validate(audio, 24000)
        self.assertEqual(len(result), 0)


# =========================================================================
# Generation Request Validation Tests (R-7)
# =========================================================================

@_skip_fastapi
@pytest.mark.unit
class TestValidateGenerationRequest(unittest.TestCase):
    """Test _validate_generation_request() from app.py."""

    def _get_validator_and_request_class(self):
        from qwen3_tts.server.app import _validate_generation_request, GenerateRequest
        return _validate_generation_request, GenerateRequest

    def test_valid_clone_request(self):
        """Valid clone request passes validation."""
        validate, Request = self._get_validator_and_request_class()
        req = Request(text="Hello", mode="clone")
        # Should not raise
        validate(req, {})

    def test_valid_design_request(self):
        """Valid design request passes validation."""
        validate, Request = self._get_validator_and_request_class()
        req = Request(text="Hello", mode="design")
        validate(req, {})

    def test_invalid_mode(self):
        """Invalid mode raises HTTPException."""
        from fastapi import HTTPException
        validate, Request = self._get_validator_and_request_class()
        req = Request(text="Hello", mode="invalid")
        with self.assertRaises(HTTPException) as ctx:
            validate(req, {})
        self.assertEqual(ctx.exception.status_code, 400)

    def test_path_traversal_blocked(self):
        """Path traversal in prompt_file raises HTTPException."""
        from fastapi import HTTPException
        validate, Request = self._get_validator_and_request_class()
        req = Request(text="Hello", mode="clone", prompt_file="../etc/passwd")
        with self.assertRaises(HTTPException) as ctx:
            validate(req, {})
        self.assertEqual(ctx.exception.status_code, 400)

    def test_path_traversal_slash_blocked(self):
        """Forward slash in prompt_file raises HTTPException."""
        from fastapi import HTTPException
        validate, Request = self._get_validator_and_request_class()
        req = Request(text="Hello", mode="clone", prompt_file="foo/bar.pt")
        with self.assertRaises(HTTPException) as ctx:
            validate(req, {})
        self.assertEqual(ctx.exception.status_code, 400)

    def test_invalid_speaker_blocked(self):
        """Invalid speaker for custom mode raises HTTPException."""
        from fastapi import HTTPException
        validate, Request = self._get_validator_and_request_class()
        req = Request(text="Hello", mode="custom", speaker="nonexistent_speaker_xyz")
        with self.assertRaises(HTTPException) as ctx:
            validate(req, {})
        self.assertEqual(ctx.exception.status_code, 400)


# =========================================================================
# Error Response Helper Tests (R-8)
# =========================================================================

@_skip_fastapi
@pytest.mark.unit
class TestErrorResponse(unittest.TestCase):
    """Test _error_response() helper from app.py."""

    def _get_error_response(self):
        from qwen3_tts.server.app import _error_response
        return _error_response

    def test_raises_http_exception(self):
        """_error_response raises HTTPException."""
        from fastapi import HTTPException
        error_response = self._get_error_response()
        with self.assertRaises(HTTPException) as ctx:
            error_response(500, "test_error", "test detail", "retry")
        self.assertEqual(ctx.exception.status_code, 500)

    def test_structured_detail(self):
        """HTTPException detail is a dict with error/detail/recovery."""
        from fastapi import HTTPException
        error_response = self._get_error_response()
        with self.assertRaises(HTTPException) as ctx:
            error_response(400, "bad_input", "missing field", "config")
        detail = ctx.exception.detail
        self.assertIsInstance(detail, dict)
        self.assertEqual(detail["error"], "bad_input")
        self.assertEqual(detail["detail"], "missing field")
        self.assertEqual(detail["recovery"], "config")

    def test_default_recovery(self):
        """Default recovery is 'retry'."""
        from fastapi import HTTPException
        error_response = self._get_error_response()
        with self.assertRaises(HTTPException) as ctx:
            error_response(500, "oops")
        self.assertEqual(ctx.exception.detail["recovery"], "retry")


# =========================================================================
# Voice Prompt Cache Tests (R-4, I-5, I-2)
# =========================================================================

@pytest.mark.unit
class TestVoicePromptCacheInfo(unittest.TestCase):
    """Test voice_prompt_cache_info() returns correct structure."""

    @patch("qwen3_tts.core.engine.get_backend", return_value="torch")
    def test_cache_info_structure(self, _mock_backend):
        """Cache info has currsize, hits, misses, maxsize fields."""
        from qwen3_tts.core.engine import voice_prompt_cache_info
        info = voice_prompt_cache_info()
        self.assertTrue(hasattr(info, "currsize"))
        self.assertTrue(hasattr(info, "hits"))
        self.assertTrue(hasattr(info, "misses"))
        self.assertTrue(hasattr(info, "maxsize"))

    @patch("qwen3_tts.core.engine.get_backend", return_value="torch")
    def test_cache_info_types(self, _mock_backend):
        """Cache info values are integers."""
        from qwen3_tts.core.engine import voice_prompt_cache_info
        info = voice_prompt_cache_info()
        self.assertIsInstance(info.currsize, int)
        self.assertIsInstance(info.hits, int)
        self.assertIsInstance(info.misses, int)
        self.assertIsInstance(info.maxsize, int)


@pytest.mark.unit
class TestVoicePromptCacheThreadSafety(unittest.TestCase):
    """Test that concurrent cache access doesn't crash."""

    def test_concurrent_clear_and_info(self):
        """Concurrent clear + info calls don't raise."""
        from qwen3_tts.core.engine import clear_voice_prompt_cache, voice_prompt_cache_info
        errors = []

        def worker(fn, n=50):
            try:
                for _ in range(n):
                    fn()
            except Exception as e:
                errors.append(e)

        with patch("qwen3_tts.core.engine.get_backend", return_value="torch"):
            threads = [
                threading.Thread(target=worker, args=(clear_voice_prompt_cache,)),
                threading.Thread(target=worker, args=(voice_prompt_cache_info,)),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        self.assertEqual(errors, [], f"Thread errors: {errors}")


# =========================================================================
# Pydantic Response Model Tests (R-9)
# =========================================================================

@_skip_pydantic
@_skip_fastapi
@pytest.mark.unit
class TestPydanticModels(unittest.TestCase):
    """Test Pydantic response models serialize correctly."""

    def test_health_response_full(self):
        """HealthResponse with all fields serializes."""
        from qwen3_tts.server.app import HealthResponse
        resp = HealthResponse(
            status="ready",
            backend="torch",
            model_size="1.7B",
            clone_model_loaded=True,
            design_model_loaded=False,
            custom_model_loaded=False,
            model_load_times={"clone": 5.2},
            model_load_errors={},
            mlx_quantization=None,
            dtype="bfloat16",
        )
        data = resp.model_dump()
        self.assertEqual(data["status"], "ready")
        self.assertTrue(data["clone_model_loaded"])

    def test_health_response_loading(self):
        """HealthResponse with minimal fields (loading state) serializes."""
        from qwen3_tts.server.app import HealthResponse
        resp = HealthResponse(
            status="loading",
            model_load_errors={"clone": "timeout"},
        )
        data = resp.model_dump()
        self.assertEqual(data["status"], "loading")
        self.assertEqual(data["model_load_errors"]["clone"], "timeout")

    def test_generate_response(self):
        """GenerateResponse serializes with results."""
        from qwen3_tts.server.app import GenerateResponse, GenerateResult
        result = GenerateResult(index=0, audio_base64="AAAA", sample_rate=24000)
        resp = GenerateResponse(results=[result])
        data = resp.model_dump()
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["sample_rate"], 24000)

    def test_error_response_model(self):
        """ErrorResponse model serializes."""
        from qwen3_tts.server.app import ErrorResponse
        resp = ErrorResponse(error="test", detail="details", recovery="retry")
        data = resp.model_dump()
        self.assertEqual(data["error"], "test")
        self.assertEqual(data["recovery"], "retry")


# =========================================================================
# Signal Handler Tests (R-1)
# =========================================================================

@pytest.mark.unit
class TestSignalHandlerPattern(unittest.TestCase):
    """Verify signal handler resets to SIG_DFL before re-sending."""

    def test_signal_handler_resets_before_kill(self):
        """_signal_handler in run_server resets handlers to prevent recursion."""
        import inspect
        from qwen3_tts.server.app import run_server
        source = inspect.getsource(run_server)
        # Handler must reset SIGTERM before sending
        self.assertIn("signal.SIG_DFL", source)
        # Must appear before os.kill in the handler
        sig_dfl_pos = source.index("signal.SIG_DFL")
        os_kill_pos = source.index("os.kill", sig_dfl_pos)
        self.assertLess(sig_dfl_pos, os_kill_pos)


if __name__ == "__main__":
    unittest.main()
