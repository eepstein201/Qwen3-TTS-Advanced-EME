#!/usr/bin/env python3
"""Tests for P3/P4 code review remediation.

Phase 2: Text processing fixes (R-21, R-22, _normalize_text, _expand_currency)
Phase 3: Engine fixes (R-14, R-16, R-17, R-18, R-27, torch.load)
Phase 4: Server fixes (R-13, R-19, R-20, R-24, R-26)
Phase 5: Documentation and optional features (R-23, R-25)
"""

import inspect
import unittest
import unittest.mock

import numpy as np


# ---------------------------------------------------------------------------
# Task 2: _normalize_text bare try/except:pass → logged warnings
# ---------------------------------------------------------------------------


class TestNormalizeTextLogging(unittest.TestCase):
    """Verify _normalize_text logs warnings instead of silently swallowing errors."""

    @unittest.mock.patch("qwen3_tts.core.engine.text_processing.logger")
    def test_normalization_step_failure_logged(self, mock_logger):
        """If a normalization step fails, a warning should be logged."""
        import qwen3_tts.core.engine.text_processing as tp

        original = tp._EMAIL_RE
        tp._EMAIL_RE = None  # Will cause AttributeError on .sub()
        try:
            result = tp._normalize_text("user@test.com hello")
            # Should still return a string (graceful degradation)
            self.assertIsInstance(result, str)
            # Should have logged a warning
            mock_logger.warning.assert_called()
            # Verify the warning mentions "email" in the step name arg
            call_args = mock_logger.warning.call_args_list[0]
            self.assertIn("email", call_args[0][1])
        finally:
            tp._EMAIL_RE = original

    @unittest.mock.patch("qwen3_tts.core.engine.text_processing.logger")
    def test_url_step_failure_logged(self, mock_logger):
        """URL normalization failure should be logged."""
        import qwen3_tts.core.engine.text_processing as tp

        original = tp._URL_RE
        tp._URL_RE = None
        try:
            result = tp._normalize_text("Visit https://example.com")
            self.assertIsInstance(result, str)
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args_list[0]
            self.assertIn("url", call_args[0][1])
        finally:
            tp._URL_RE = original

    @unittest.mock.patch("qwen3_tts.core.engine.text_processing.logger")
    def test_phone_step_failure_logged(self, mock_logger):
        """Phone normalization failure should be logged."""
        import qwen3_tts.core.engine.text_processing as tp

        original = tp._PHONE_RE
        tp._PHONE_RE = None
        try:
            result = tp._normalize_text("Call (800) 555-1234")
            self.assertIsInstance(result, str)
            mock_logger.warning.assert_called()
            call_args = mock_logger.warning.call_args_list[0]
            self.assertIn("phone", call_args[0][1])
        finally:
            tp._PHONE_RE = original

    def test_normal_text_no_warnings(self):
        """Normal text should not trigger any warnings."""
        import qwen3_tts.core.engine.text_processing as tp

        with unittest.mock.patch.object(tp.logger, "warning") as mock_warn:
            tp._normalize_text("Hello world, this is a test.")
            mock_warn.assert_not_called()


# ---------------------------------------------------------------------------
# Task 3: _expand_currency decimal handling
# ---------------------------------------------------------------------------


class TestCurrencyExpansion(unittest.TestCase):
    """Verify _expand_currency handles decimals like $5.99."""

    def test_decimal_currency(self):
        """$5.99 should expand to include cents."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        result = _normalize_text("The price is $5.99")
        self.assertIn("ninety", result.lower())
        self.assertIn("cent", result.lower())

    def test_whole_dollar(self):
        """$5 should still work as before."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        result = _normalize_text("It costs $5")
        self.assertIn("five", result.lower())
        self.assertIn("dollar", result.lower())

    def test_one_dollar(self):
        """$1 should use singular 'dollar'."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        result = _normalize_text("It costs $1")
        self.assertIn("one", result.lower())
        self.assertNotIn("dollars", result.lower())

    def test_one_cent(self):
        """$0.01 should say 'one cent'."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        result = _normalize_text("It costs $0.01")
        self.assertIn("cent", result.lower())
        self.assertNotIn("cents", result.lower())

    def test_euro_decimal(self):
        """€3.50 should handle cents."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        result = _normalize_text("It costs €3.50")
        self.assertIn("three", result.lower())
        self.assertIn("euro", result.lower())

    def test_pound_decimal(self):
        """£2.50 should use pence."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        result = _normalize_text("It costs £2.50")
        self.assertIn("two", result.lower())
        self.assertIn("pound", result.lower())
        self.assertIn("pence", result.lower())

    def test_zero_cents_no_subunit(self):
        """$5.00 should not include 'cent' text."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        result = _normalize_text("It costs $5.00")
        self.assertIn("five", result.lower())
        self.assertIn("dollar", result.lower())
        self.assertNotIn("cent", result.lower())


# ---------------------------------------------------------------------------
# Task 4: Cache pysbd.Segmenter per language (R-21)
# ---------------------------------------------------------------------------


class TestPysdbCache(unittest.TestCase):
    """Verify pysbd.Segmenter is cached per language code."""

    def test_segmenter_cached_per_language(self):
        """Second call with same language should reuse cached segmenter."""
        from qwen3_tts.core.engine.text_processing import _split_text, _SEGMENTER_CACHE

        _SEGMENTER_CACHE.clear()
        _split_text(
            "Hello world. This is a test. Another one.",
            max_chars=15,
            language="English",
        )
        self.assertIn("en", _SEGMENTER_CACHE)

    def test_different_languages_cached_separately(self):
        """Different languages should have separate cache entries."""
        from qwen3_tts.core.engine.text_processing import _split_text, _SEGMENTER_CACHE

        _SEGMENTER_CACHE.clear()
        _split_text("Hello world. This is a test.", max_chars=15, language="English")
        _split_text("Hola mundo. Esta es una prueba.", max_chars=15, language="Spanish")
        self.assertIn("en", _SEGMENTER_CACHE)
        self.assertIn("es", _SEGMENTER_CACHE)

    def test_cache_reuse_same_object(self):
        """Same language should return the same segmenter object."""
        from qwen3_tts.core.engine.text_processing import _split_text, _SEGMENTER_CACHE

        _SEGMENTER_CACHE.clear()
        _split_text("Hello world. This is a test.", max_chars=15, language="English")
        seg1 = _SEGMENTER_CACHE.get("en")
        _split_text("Another sentence. And one more.", max_chars=15, language="English")
        seg2 = _SEGMENTER_CACHE.get("en")
        self.assertIs(seg1, seg2)


# ---------------------------------------------------------------------------
# Task 5: Cache num2words import at module level (R-22)
# ---------------------------------------------------------------------------


class TestNum2wordsCache(unittest.TestCase):
    """Verify num2words is cached at module level after first use."""

    def test_num2words_cached_after_first_call(self):
        """num2words should be cached at module level after first use."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        _normalize_text("123")
        from qwen3_tts.core.engine import text_processing as tp

        self.assertTrue(tp._n2w_loaded)

    def test_num2words_function_cached(self):
        """_n2w_cached should hold the num2words function after first call."""
        from qwen3_tts.core.engine.text_processing import _normalize_text

        _normalize_text("456")
        from qwen3_tts.core.engine import text_processing as tp

        # num2words is installed in test env, so _n2w_cached should be set
        self.assertIsNotNone(tp._n2w_cached)


# ===========================================================================
# Phase 3: Engine fixes (R-14, R-16, R-17, R-18, R-27, torch.load)
# ===========================================================================


# ---------------------------------------------------------------------------
# Task 6: Crossfade between chunks (R-14) + silence gap (R-27)
# ---------------------------------------------------------------------------


class TestCrossfade(unittest.TestCase):
    """Verify _crossfade_chunks produces smooth transitions."""

    def test_crossfade_smooth_transition(self):
        """Crossfade should produce smooth transition (no discontinuity)."""
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        sr = 24000
        chunk1 = np.ones(sr, dtype=np.float32) * 0.5
        chunk2 = np.ones(sr, dtype=np.float32) * -0.5
        result = _crossfade_chunks([chunk1, chunk2], sr, crossfade_ms=50)
        mid = len(chunk1)
        transition = result[mid - 100 : mid + 100]
        max_jump = np.max(np.abs(np.diff(transition)))
        self.assertLess(max_jump, 0.1, "Crossfade should smooth the transition")

    def test_single_chunk_passthrough(self):
        """Single chunk should be returned unchanged."""
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        chunk = np.ones(1000, dtype=np.float32)
        result = _crossfade_chunks([chunk], 24000)
        np.testing.assert_array_equal(result, chunk)

    def test_empty_chunks(self):
        """Empty list should return empty array."""
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        result = _crossfade_chunks([], 24000)
        self.assertEqual(len(result), 0)

    def test_silence_gap_mode(self):
        """Silence gap should insert zeros between chunks."""
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        sr = 24000
        chunk1 = np.ones(sr, dtype=np.float32)
        chunk2 = np.ones(sr, dtype=np.float32)
        result = _crossfade_chunks(
            [chunk1, chunk2], sr, crossfade_ms=0, silence_gap_s=0.5
        )
        expected_len = 2 * sr + int(0.5 * sr)
        self.assertEqual(len(result), expected_len)

    def test_silence_gap_zeros(self):
        """The gap between chunks should be all zeros."""
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        sr = 24000
        chunk1 = np.ones(1000, dtype=np.float32)
        chunk2 = np.ones(1000, dtype=np.float32) * 2.0
        gap_samples = int(sr * 0.1)
        result = _crossfade_chunks(
            [chunk1, chunk2], sr, crossfade_ms=0, silence_gap_s=0.1
        )
        gap = result[1000 : 1000 + gap_samples]
        np.testing.assert_array_equal(gap, np.zeros(gap_samples, dtype=np.float32))

    def test_crossfade_disabled(self):
        """crossfade_ms=0 with no silence_gap should just concatenate."""
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        chunk1 = np.ones(100, dtype=np.float32)
        chunk2 = np.ones(200, dtype=np.float32) * 2.0
        result = _crossfade_chunks([chunk1, chunk2], 24000, crossfade_ms=0)
        self.assertEqual(len(result), 300)
        np.testing.assert_array_equal(result[:100], chunk1)
        np.testing.assert_array_equal(result[100:], chunk2)

    def test_three_chunks(self):
        """Should handle 3+ chunks correctly."""
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        sr = 24000
        chunks = [np.ones(500, dtype=np.float32) * i for i in range(1, 4)]
        result = _crossfade_chunks(chunks, sr, crossfade_ms=10)
        # Should be shorter than concatenation due to overlaps
        self.assertLess(len(result), 1500)
        self.assertGreater(len(result), 0)


# ---------------------------------------------------------------------------
# Task 7: Model warm-up (R-16)
# ---------------------------------------------------------------------------


class TestModelWarmup(unittest.TestCase):
    """Verify _warmup_model exists and is callable."""

    def test_warmup_function_exists(self):
        from qwen3_tts.core.engine.model_loader import _warmup_model

        self.assertTrue(callable(_warmup_model))

    def test_warmup_called_in_load_model(self):
        """load_model should call _warmup_model after loading."""
        source = inspect.getsource(
            __import__(
                "qwen3_tts.core.engine.model_loader", fromlist=["load_model"]
            ).load_model
        )
        self.assertIn("_warmup_model", source)

    def test_warmup_nonfatal(self):
        """_warmup_model should not raise even if model fails."""
        from qwen3_tts.core.engine.model_loader import _warmup_model

        mock_model = unittest.mock.MagicMock()
        mock_model.generate_voice_design.side_effect = RuntimeError("test failure")
        # Should not raise
        _warmup_model(mock_model, "design", "torch")


# ---------------------------------------------------------------------------
# Task 8: Temperature consistency (R-17)
# ---------------------------------------------------------------------------


class TestTemperatureConsistency(unittest.TestCase):
    """MLX inference should not hardcode 0.9 temperature."""

    def test_mlx_inference_no_hardcoded_temperature(self):
        """MLX inference should read temperature from config, not hardcode 0.9."""
        from qwen3_tts.core.engine.inference import _run_inference_mlx

        source = inspect.getsource(_run_inference_mlx)
        # Normalize whitespace and quotes for reliable matching
        normalized = source.replace(" ", "").replace("'", '"')
        self.assertNotIn('"temperature",0.9', normalized)

    def test_mlx_streaming_no_hardcoded_temperature(self):
        """MLX streaming should read temperature from config, not hardcode 0.9."""
        from qwen3_tts.core.engine.inference import _run_inference_mlx_streaming

        source = inspect.getsource(_run_inference_mlx_streaming)
        normalized = source.replace(" ", "").replace("'", '"')
        self.assertNotIn('"temperature",0.9', normalized)

    def test_mlx_reads_config_for_defaults(self):
        """Both MLX functions should reference load_config for defaults."""
        from qwen3_tts.core.engine.inference import (
            _run_inference_mlx,
            _run_inference_mlx_streaming,
        )

        for fn in (_run_inference_mlx, _run_inference_mlx_streaming):
            source = inspect.getsource(fn)
            self.assertIn(
                "load_config", source, f"{fn.__name__} should use load_config"
            )


# ---------------------------------------------------------------------------
# Task 9: Turing GPU quantization override (R-18)
# ---------------------------------------------------------------------------


class TestTuringQuantizationOverride(unittest.TestCase):
    """Turing GPU auto-8bit should respect explicit torch_quantization."""

    def test_explicit_quantization_respected(self):
        """When user explicitly sets torch_quantization, don't auto-override."""
        # Logic extracted to _resolve_load_kwargs in Phase 5 refactor
        from qwen3_tts.core.engine.model_loader import _resolve_load_kwargs

        source = inspect.getsource(_resolve_load_kwargs)
        self.assertIn("explicitly", source.lower())

    def test_explicitly_set_check_in_code(self):
        """Code should check if torch_quantization is in config."""
        # Logic extracted to _resolve_load_kwargs in Phase 5 refactor
        from qwen3_tts.core.engine.model_loader import _resolve_load_kwargs

        source = inspect.getsource(_resolve_load_kwargs)
        self.assertIn("explicitly_set", source)
        self.assertIn("torch_quantization", source)


# ---------------------------------------------------------------------------
# Task 10: torch.load security (deprecation warning)
# ---------------------------------------------------------------------------


class TestTorchLoadSecurity(unittest.TestCase):
    """Verify security measures for torch.load in voice_prompt.py."""

    def test_path_traversal_check_exists(self):
        """_load_pt_safe should check realpath for path traversal."""
        from qwen3_tts.core.engine.voice_prompt import _load_pt_safe

        source = inspect.getsource(_load_pt_safe)
        self.assertIn("realpath", source)

    def test_no_unsafe_deserialization_escape_hatch(self):
        """Unsafe deserialization env var must not exist in _load_pt_safe."""
        from qwen3_tts.core.engine.voice_prompt import _load_pt_safe

        source = inspect.getsource(_load_pt_safe)
        self.assertNotIn("ALLOW_UNSAFE", source)
        self.assertNotIn("weights_only=False", source)

    def test_refuses_path_outside_prompts_dir(self):
        """Should refuse to load files outside VOICE_PROMPTS_DIR."""
        # Logic extracted to _load_pt_safe in Phase 5 refactor
        from qwen3_tts.core.engine.voice_prompt import _load_pt_safe

        source = inspect.getsource(_load_pt_safe)
        self.assertIn("Refusing to load", source)


# ===========================================================================
# Phase 4: Server fixes (R-13, R-19, R-20, R-24, R-26)
# ===========================================================================


# ---------------------------------------------------------------------------
# Task 11: Rate limiting with real IP resolution (R-13)
# ---------------------------------------------------------------------------


class TestRateLimiting(unittest.TestCase):
    """Verify rate limiting infrastructure."""

    def test_real_ip_resolver_ignores_xff_for_loopback(self):
        """S2 fix: XFF is NOT trusted when client connects from loopback (spoofing prevention)."""
        from qwen3_tts.server.app import _get_real_client_ip

        mock_request = unittest.mock.MagicMock()
        mock_request.headers = {"X-Forwarded-For": "1.2.3.4, 10.0.0.1"}
        mock_request.client.host = "127.0.0.1"
        self.assertEqual(_get_real_client_ip(mock_request), "127.0.0.1")

    def test_real_ip_resolver_trusts_xff_for_non_loopback(self):
        """XFF is trusted when client connects from a non-loopback address (behind proxy)."""
        from qwen3_tts.server.app import _get_real_client_ip

        mock_request = unittest.mock.MagicMock()
        mock_request.headers = {"X-Forwarded-For": "1.2.3.4, 10.0.0.1"}
        mock_request.client.host = "10.0.0.5"
        self.assertEqual(_get_real_client_ip(mock_request), "1.2.3.4")

    def test_real_ip_falls_back_to_client_host(self):
        """Without X-Forwarded-For, should use client.host."""
        from qwen3_tts.server.app import _get_real_client_ip

        mock_request = unittest.mock.MagicMock()
        mock_request.headers = {}
        mock_request.client.host = "192.168.1.100"
        self.assertEqual(_get_real_client_ip(mock_request), "192.168.1.100")

    def test_real_ip_no_client(self):
        """If request.client is None, should return 127.0.0.1."""
        from qwen3_tts.server.app import _get_real_client_ip

        mock_request = unittest.mock.MagicMock()
        mock_request.headers = {}
        mock_request.client = None
        self.assertEqual(_get_real_client_ip(mock_request), "127.0.0.1")

    def test_slowapi_flag_exists(self):
        """_HAS_SLOWAPI flag should exist in module."""
        from qwen3_tts.server import app as app_module

        self.assertIsInstance(app_module._HAS_SLOWAPI, bool)

    def test_rate_limit_decorator_exists(self):
        """_rate_limit helper should be callable."""
        from qwen3_tts.server.app import _rate_limit

        self.assertTrue(callable(_rate_limit))

    def test_rate_limit_decorators_on_endpoints(self):
        """Key endpoints should have rate limit decorators in source."""
        source = inspect.getsource(__import__("qwen3_tts.server.app", fromlist=["app"]))
        # Check that _rate_limit is applied near the endpoint definitions
        self.assertIn("@_rate_limit(_generate_limit)", source)
        self.assertIn("@_rate_limit(_model_limit)", source)


# ---------------------------------------------------------------------------
# Task 12: Thread-safe request_queue (R-19)
# ---------------------------------------------------------------------------


class TestRequestQueueThreadSafety(unittest.TestCase):
    """Verify request_queue uses a lock."""

    def test_request_queue_lock_in_source(self):
        """Source should reference request_queue_lock."""
        source = inspect.getsource(
            __import__("qwen3_tts.server.app_lifespan", fromlist=["_get_queue_size"])
        )
        self.assertIn("request_queue_lock", source)

    def test_get_queue_size_function_exists(self):
        """_get_queue_size helper should exist."""
        from qwen3_tts.server.app import _get_queue_size

        self.assertTrue(callable(_get_queue_size))


# ---------------------------------------------------------------------------
# Task 13: Symlink resolution in /preview-prompt (R-20)
# ---------------------------------------------------------------------------


class TestPreviewPromptSymlink(unittest.TestCase):
    """Verify /preview-prompt resolves symlinks."""

    def test_realpath_validation_in_endpoint(self):
        """preview_prompt handler should check realpath."""
        from qwen3_tts.server.app_prompts import handle_preview_prompt

        source = inspect.getsource(handle_preview_prompt)
        self.assertIn("realpath", source)

    def test_rejects_path_outside_dir(self):
        """preview_prompt handler should reject paths resolving outside VOICE_PROMPTS_DIR."""
        from qwen3_tts.server.app_prompts import handle_preview_prompt

        source = inspect.getsource(handle_preview_prompt)
        self.assertIn("Access denied", source)


# ---------------------------------------------------------------------------
# Task 14: Pagination for /prompts (R-24)
# ---------------------------------------------------------------------------


class TestPromptsPagination(unittest.TestCase):
    """Verify /prompts supports pagination."""

    def test_pagination_params_in_source(self):
        """list_prompts handler should reference offset and limit."""
        from qwen3_tts.server.app_prompts import handle_list_prompts

        source = inspect.getsource(handle_list_prompts)
        self.assertIn("total", source)
        self.assertIn("offset", source)
        self.assertIn("limit", source)


# ---------------------------------------------------------------------------
# Task 15: Audit logging for auth failures (R-26)
# ---------------------------------------------------------------------------


class TestAuditLogging(unittest.TestCase):
    """Verify auth failures are logged with client IP."""

    def test_auth_failure_logs_warning(self):
        """verify_auth should log client IP and failure reason on failure (R-26)."""
        from qwen3_tts.server.app import verify_auth

        source = inspect.getsource(verify_auth)
        self.assertIn("logger.warning", source)
        self.assertIn("Auth failure", source)
        # R-26: Enhanced audit logging includes failure reason
        self.assertIn("failure_reason", source)
        self.assertIn("missing_token", source)
        self.assertIn("invalid_token", source)

    def test_auth_uses_real_client_ip(self):
        """verify_auth should use _get_real_client_ip for audit logging."""
        from qwen3_tts.server.app import verify_auth

        source = inspect.getsource(verify_auth)
        self.assertIn("_get_real_client_ip", source)


# ===========================================================================
# Phase 5: Documentation and optional features (R-23, R-25)
# ===========================================================================


# ---------------------------------------------------------------------------
# Task 16: Streaming wire format documentation (R-25)
# ---------------------------------------------------------------------------


class TestStreamingDocs(unittest.TestCase):
    """Verify generate_stream has wire format documentation."""

    def test_wire_format_documented(self):
        """generate_stream docstring should describe the wire format."""
        from qwen3_tts.server.app import generate_stream

        doc = generate_stream.__doc__ or ""
        self.assertIn("float32", doc)
        self.assertIn("little-endian", doc)

    def test_python_consumer_example(self):
        """Docstring should include Python consumer example."""
        from qwen3_tts.server.app import generate_stream

        doc = generate_stream.__doc__ or ""
        self.assertIn("httpx", doc)

    def test_javascript_consumer_example(self):
        """Docstring should include JavaScript consumer example."""
        from qwen3_tts.server.app import generate_stream

        doc = generate_stream.__doc__ or ""
        self.assertIn("Float32Array", doc)


# ---------------------------------------------------------------------------
# Task 17: LUFS normalization (R-23)
# ---------------------------------------------------------------------------


class TestLUFSNormalization(unittest.TestCase):
    """Verify LUFS normalization function exists and integrates with process_audio."""

    def test_normalize_lufs_function_exists(self):
        """normalize_lufs should be importable from audio_processing."""
        from qwen3_tts.core.engine.audio_processing import normalize_lufs

        self.assertTrue(callable(normalize_lufs))

    def test_normalize_lufs_in_facade(self):
        """normalize_lufs should be exported from engine facade."""
        from qwen3_tts.core.engine import normalize_lufs

        self.assertTrue(callable(normalize_lufs))

    def test_process_audio_accepts_lufs_target(self):
        """process_audio should accept lufs_target parameter."""
        sig = inspect.signature(
            __import__(
                "qwen3_tts.core.engine.audio_processing", fromlist=["process_audio"]
            ).process_audio
        )
        self.assertIn("lufs_target", sig.parameters)

    def test_process_audio_lufs_none_is_noop(self):
        """lufs_target=None should not change audio."""
        from qwen3_tts.core.engine.audio_processing import process_audio

        audio = np.ones(1000, dtype=np.float32) * 0.5
        result = process_audio(audio, 24000, lufs_target=None)
        np.testing.assert_array_equal(result, audio)


if __name__ == "__main__":
    unittest.main()
