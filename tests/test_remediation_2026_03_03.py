#!/usr/bin/env python3
"""Tests for 2026-03-03 codebase remediation: 5 core logic fixes.

Fix 1: Zombie GPU cleanup in engine_vllm.py
Fix 2: Port race retry in engine_vllm.py
Fix 3: Pre-compiled regexes in engine.py
Fix 4: Metal retry depth limit in engine.py
Fix 5: Config validation enforcement in config.py
"""

import asyncio
import platform
import re
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

# ---------------------------------------------------------------------------
# Fix 5: Config validation enforcement
# ---------------------------------------------------------------------------


class TestConfigValidationEnforcement(unittest.TestCase):
    """validate_config() should correct invalid values in-memory."""

    def _validate(self, config):
        from qwen3_tts.core.config import validate_config
        return validate_config(config)

    def test_invalid_backend_corrected_to_platform_default(self):
        config = {"advanced": {"backend": "invalid_backend"}}
        issues = self._validate(config)
        self.assertIn(config["advanced"]["backend"], ("torch", "mlx"))
        self.assertTrue(any("corrected" in i and "backend" in i for i in issues))

    def test_invalid_model_size_corrected_to_1_7B(self):
        config = {"advanced": {"model_size": "99B"}}
        issues = self._validate(config)
        self.assertEqual(config["advanced"]["model_size"], "1.7B")
        self.assertTrue(any("corrected" in i and "model_size" in i for i in issues))

    def test_temperature_too_high_corrected(self):
        config = {"generation": {"temperature": 5.0}}
        issues = self._validate(config)
        self.assertEqual(config["generation"]["temperature"], 0.7)
        self.assertTrue(any("corrected" in i and "temperature" in i for i in issues))

    def test_temperature_negative_corrected(self):
        config = {"generation": {"temperature": -1.0}}
        issues = self._validate(config)
        self.assertEqual(config["generation"]["temperature"], 0.7)

    def test_temperature_valid_not_corrected(self):
        config = {"generation": {"temperature": 1.5}}
        issues = self._validate(config)
        self.assertEqual(config["generation"]["temperature"], 1.5)
        self.assertFalse(any("corrected" in i and "temperature" in i for i in issues))

    def test_max_text_length_invalid_corrected(self):
        config = {"security": {"max_text_length": -5}}
        issues = self._validate(config)
        self.assertEqual(config["security"]["max_text_length"], 10000)
        self.assertTrue(any("corrected" in i and "max_text_length" in i for i in issues))

    def test_max_text_length_non_int_corrected(self):
        config = {"security": {"max_text_length": "abc"}}
        issues = self._validate(config)
        self.assertEqual(config["security"]["max_text_length"], 10000)

    def test_vllm_gpu_utilization_too_high_corrected(self):
        config = {"advanced": {"vllm_gpu_memory_utilization": 1.5}}
        issues = self._validate(config)
        self.assertEqual(config["advanced"]["vllm_gpu_memory_utilization"], 0.7)
        self.assertTrue(any("corrected" in i and "vllm_gpu_memory_utilization" in i for i in issues))

    def test_vllm_gpu_utilization_zero_corrected(self):
        config = {"advanced": {"vllm_gpu_memory_utilization": 0.0}}
        issues = self._validate(config)
        self.assertEqual(config["advanced"]["vllm_gpu_memory_utilization"], 0.7)

    def test_vllm_port_out_of_range_corrected(self):
        config = {"advanced": {"vllm_port": 80}}
        issues = self._validate(config)
        self.assertIsNone(config["advanced"]["vllm_port"])
        self.assertTrue(any("corrected" in i and "vllm_port" in i for i in issues))

    def test_vllm_port_too_high_corrected(self):
        config = {"advanced": {"vllm_port": 99999}}
        issues = self._validate(config)
        self.assertIsNone(config["advanced"]["vllm_port"])

    def test_empty_config_no_keys_added(self):
        """Empty config {} should not gain any new keys."""
        config = {}
        issues = self._validate(config)
        self.assertEqual(config, {})
        self.assertEqual(issues, [])

    def test_valid_config_unchanged(self):
        config = {
            "advanced": {"backend": "torch", "model_size": "0.6B",
                         "vllm_gpu_memory_utilization": 0.9, "vllm_port": 8100},
            "generation": {"temperature": 1.0},
            "security": {"max_text_length": 5000},
        }
        issues = self._validate(config)
        self.assertEqual(config["advanced"]["backend"], "torch")
        self.assertEqual(config["generation"]["temperature"], 1.0)
        self.assertFalse(any("corrected" in i for i in issues))

    def test_missing_subsection_no_bloat(self):
        """Config with only 'generation' should not gain 'advanced' or 'security'."""
        config = {"generation": {"temperature": 0.5}}
        self._validate(config)
        self.assertNotIn("advanced", config)
        self.assertNotIn("security", config)


# ---------------------------------------------------------------------------
# Fix 3: Pre-compiled regex constants
# ---------------------------------------------------------------------------


class TestPreCompiledRegexes(unittest.TestCase):
    """Verify compiled regex constants exist and _normalize_text() still works."""

    def test_compiled_constants_exist(self):
        from qwen3_tts.core.engine import text_processing
        expected = [
            "_EMAIL_RE", "_URL_RE", "_URL_PROTO_RE", "_URL_WWW_RE",
            "_PHONE_RE", "_PHONE_NONDIGIT_RE", "_ORDINAL_RE",
            "_ISO_DATE_RE", "_US_DATE_RE", "_CARDINAL_RE",
            "_ABBREV_TABLE_COMPILED", "_CURRENCY_RE",
        ]
        for name in expected:
            obj = getattr(text_processing, name, None)
            self.assertIsNotNone(obj, f"Missing compiled constant: {name}")
            if name == "_ABBREV_TABLE_COMPILED":
                self.assertIsInstance(obj, list)
                for pat, repl in obj:
                    self.assertIsInstance(pat, re.Pattern)
            else:
                self.assertIsInstance(obj, re.Pattern, f"{name} is not re.Pattern")

    def test_normalize_text_email(self):
        from qwen3_tts.core.engine.text_processing import _normalize_text
        result = _normalize_text("Email test@example.com")
        self.assertIn("at", result)
        self.assertIn("dot", result)

    def test_normalize_text_url(self):
        from qwen3_tts.core.engine.text_processing import _normalize_text
        result = _normalize_text("Visit https://example.com")
        self.assertNotIn("https://", result)
        self.assertIn("dot", result)

    def test_normalize_text_phone(self):
        from qwen3_tts.core.engine.text_processing import _normalize_text
        result = _normalize_text("Call (800) 555-1234")
        # Phone digits get expanded, then num2words may convert them to words
        self.assertNotIn("(800)", result)
        self.assertNotIn("555-1234", result)

    def test_normalize_text_abbreviation(self):
        from qwen3_tts.core.engine.text_processing import _normalize_text
        result = _normalize_text("Dr. Smith")
        self.assertIn("Doctor", result)


# ---------------------------------------------------------------------------
# Fix 4: Metal retry depth limit
# ---------------------------------------------------------------------------


class TestMetalRetryDepthLimit(unittest.TestCase):
    """_run_inference_single should limit Metal retry recursion depth."""

    def _make_metal_error(self):
        return RuntimeError("Metal command buffer execution failed")

    @patch("qwen3_tts.core.engine.inference.get_backend", return_value="mlx")
    def test_depth_0_retries_on_metal_crash(self, mock_backend):
        """At depth 0 with text > 100 chars, Metal crash triggers split retry."""
        from qwen3_tts.core.engine import _run_inference_single, _INFERENCE_STRATEGIES
        import numpy as _np

        long_text = "A " * 110  # > 100 chars, has spaces for splitting
        sr = 24000
        wav = _np.zeros(1000, dtype="float32")

        # Create a mock strategy that raises Metal error first, then succeeds
        call_count = [0]
        original_mlx = _INFERENCE_STRATEGIES.get("mlx")

        def mock_strategy(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise self._make_metal_error()
            return (wav, sr)

        with patch.dict(_INFERENCE_STRATEGIES, {"mlx": mock_strategy}):
            result_wav, result_sr = _run_inference_single(
                MagicMock(), long_text, "clone", {}, voice_prompt={"ref_audio": MagicMock(), "ref_text": "test"}, _metal_retry_depth=0
            )
            self.assertEqual(result_sr, sr)
            # Should have called strategy 3 times (1 fail + 2 sub-chunks)
            self.assertEqual(call_count[0], 3)

    @patch("qwen3_tts.core.engine.inference.get_backend", return_value="mlx")
    def test_depth_2_raises_immediately(self, mock_backend):
        """At depth 2, Metal crash re-raises without recursion."""
        from qwen3_tts.core.engine import _run_inference_single, _INFERENCE_STRATEGIES

        def mock_strategy(*args, **kwargs):
            raise self._make_metal_error()

        with patch.dict(_INFERENCE_STRATEGIES, {"mlx": mock_strategy}):
            with self.assertRaises(RuntimeError) as ctx:
                _run_inference_single(
                    MagicMock(), "A " * 110, "clone", {}, voice_prompt={"ref_audio": MagicMock(), "ref_text": "test"}, _metal_retry_depth=2
                )
            self.assertIn("Metal command buffer", str(ctx.exception))

    @patch("qwen3_tts.core.engine.inference.get_backend", return_value="mlx")
    def test_short_text_raises_immediately(self, mock_backend):
        """Short text (<=100 chars) should not trigger retry regardless of depth."""
        from qwen3_tts.core.engine import _run_inference_single, _INFERENCE_STRATEGIES

        call_count = [0]

        def mock_strategy(*args, **kwargs):
            call_count[0] += 1
            raise self._make_metal_error()

        with patch.dict(_INFERENCE_STRATEGIES, {"mlx": mock_strategy}):
            with self.assertRaises(RuntimeError):
                _run_inference_single(
                    MagicMock(), "Short text", "clone", {}, voice_prompt={"ref_audio": MagicMock(), "ref_text": "test"}, _metal_retry_depth=0
                )
            self.assertEqual(call_count[0], 1)


# ---------------------------------------------------------------------------
# Fix 1: Zombie GPU cleanup
# ---------------------------------------------------------------------------


class TestZombieGPUCleanup(unittest.TestCase):
    """start() should call stop() if _wait_until_ready() fails."""

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_timeout_error_calls_stop(self):
        from qwen3_tts.core.engine_vllm import VLLMAdapter
        adapter = VLLMAdapter.__new__(VLLMAdapter)
        adapter._process = None
        adapter._client = None
        adapter._ready_event = asyncio.Event()
        adapter._cancellation_callback = None
        adapter.port = None
        adapter.log_file = "/tmp/test.log"
        adapter.model_name = "test"
        adapter.gpu_memory_utilization = 0.7
        adapter._auto_port = True

        adapter._start = MagicMock()
        adapter._wait_until_ready = AsyncMock(side_effect=TimeoutError("timeout"))
        adapter.stop = MagicMock()

        with self.assertRaises(TimeoutError):
            self._run_async(adapter.start())
        adapter.stop.assert_called()

    def test_runtime_error_calls_stop(self):
        from qwen3_tts.core.engine_vllm import VLLMAdapter
        adapter = VLLMAdapter.__new__(VLLMAdapter)
        adapter._process = None
        adapter._client = None
        adapter._ready_event = asyncio.Event()
        adapter._cancellation_callback = None
        adapter.port = None
        adapter.log_file = "/tmp/test.log"
        adapter.model_name = "test"
        adapter.gpu_memory_utilization = 0.7
        adapter._auto_port = True

        adapter._start = MagicMock()
        adapter._wait_until_ready = AsyncMock(side_effect=RuntimeError("process exited"))
        adapter.stop = MagicMock()

        with self.assertRaises(RuntimeError):
            self._run_async(adapter.start())
        adapter.stop.assert_called()


# ---------------------------------------------------------------------------
# Fix 2: Port race retry
# ---------------------------------------------------------------------------


class TestPortRaceRetry(unittest.TestCase):
    """start() should retry with new port on failure when using auto-port."""

    def _run_async(self, coro):
        return asyncio.run(coro)

    def test_retry_succeeds_on_third_attempt_auto_port(self):
        from qwen3_tts.core.engine_vllm import VLLMAdapter
        adapter = VLLMAdapter.__new__(VLLMAdapter)
        adapter._process = None
        adapter._client = None
        adapter._ready_event = asyncio.Event()
        adapter._cancellation_callback = None
        adapter.port = None
        adapter.log_file = "/tmp/test.log"
        adapter.model_name = "test"
        adapter.gpu_memory_utilization = 0.7
        adapter._auto_port = True

        call_count = [0]
        async def mock_wait():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("port in use")

        adapter._start = MagicMock()
        adapter._wait_until_ready = mock_wait
        adapter.stop = MagicMock()

        self._run_async(adapter.start())
        # stop() called for the 2 failures
        self.assertEqual(adapter.stop.call_count, 2)
        # _start() called 3 times
        self.assertEqual(adapter._start.call_count, 3)

    def test_fixed_port_unchanged_across_retries(self):
        from qwen3_tts.core.engine_vllm import VLLMAdapter
        adapter = VLLMAdapter.__new__(VLLMAdapter)
        adapter._process = None
        adapter._client = None
        adapter._ready_event = asyncio.Event()
        adapter._cancellation_callback = None
        adapter.port = 8200
        adapter.log_file = "/tmp/test.log"
        adapter.model_name = "test"
        adapter.gpu_memory_utilization = 0.7
        adapter._auto_port = False

        call_count = [0]
        ports_seen = []
        original_start = adapter._start

        def mock_start():
            ports_seen.append(adapter.port)

        async def mock_wait():
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("not ready")

        adapter._start = mock_start
        adapter._wait_until_ready = mock_wait
        adapter.stop = MagicMock()

        self._run_async(adapter.start())
        # Port should stay 8200 across all attempts
        self.assertTrue(all(p == 8200 for p in ports_seen))

    def test_all_retries_exhausted_raises(self):
        from qwen3_tts.core.engine_vllm import VLLMAdapter
        adapter = VLLMAdapter.__new__(VLLMAdapter)
        adapter._process = None
        adapter._client = None
        adapter._ready_event = asyncio.Event()
        adapter._cancellation_callback = None
        adapter.port = None
        adapter.log_file = "/tmp/test.log"
        adapter.model_name = "test"
        adapter.gpu_memory_utilization = 0.7
        adapter._auto_port = True

        adapter._start = MagicMock()
        adapter._wait_until_ready = AsyncMock(side_effect=RuntimeError("always fails"))
        adapter.stop = MagicMock()

        with self.assertRaises(RuntimeError):
            self._run_async(adapter.start())
        self.assertEqual(adapter.stop.call_count, 3)


if __name__ == "__main__":
    unittest.main()
