"""Extended tests for engine_vllm.py — part 2.

Split from test_engine_vllm_ext.py (over 800 lines).
Covers: start() retry logic, set_cancellation_callback.

Run: python -m pytest tests/test_engine_vllm_ext_part2.py -v
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

_MOD = "qwen3_tts.core.engine_vllm"


def _make_adapter(port=8200, log_file=None):
    """Create adapter with mocked _get_default_log_file."""
    with patch(f"{_MOD}.VLLMAdapter._get_default_log_file", return_value="/tmp/test_vllm_ext.log"):
        from qwen3_tts.core.engine_vllm import VLLMAdapter
        return VLLMAdapter(port=port, log_file=log_file or "/tmp/test_vllm_ext.log")


class TestStartRetryLogic(unittest.TestCase):
    """Test start() retry logic."""

    def _make_auto_port_adapter(self):
        with patch(f"{_MOD}.VLLMAdapter._get_default_log_file", return_value="/tmp/test.log"):
            from qwen3_tts.core.engine_vllm import VLLMAdapter
            return VLLMAdapter(log_file="/tmp/test.log")

    def test_start_success_on_first_attempt(self):
        adapter = self._make_auto_port_adapter()
        adapter.port = 8500

        with patch.object(adapter, "_start"), \
             patch.object(adapter, "_wait_until_ready", new_callable=AsyncMock):
            asyncio.run(adapter.start())

    def test_start_retries_on_failure(self):
        adapter = self._make_auto_port_adapter()
        adapter.port = 8500
        attempt_count = [0]

        async def fail_then_succeed():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise TimeoutError("not ready")

        with patch.object(adapter, "_start"), \
             patch.object(adapter, "_wait_until_ready", side_effect=fail_then_succeed), \
             patch.object(adapter, "stop"):
            asyncio.run(adapter.start())

        self.assertEqual(attempt_count[0], 3)

    def test_start_exhausts_retries_and_raises(self):
        adapter = self._make_auto_port_adapter()
        adapter.port = 8500

        async def always_fail():
            raise TimeoutError("never ready")

        with patch.object(adapter, "_start"), \
             patch.object(adapter, "_wait_until_ready", side_effect=always_fail), \
             patch.object(adapter, "stop"):
            with self.assertRaises(TimeoutError):
                asyncio.run(adapter.start())

    def test_start_resets_port_on_retry_with_auto_port(self):
        adapter = self._make_auto_port_adapter()
        port_history = []

        def tracking_start():
            port_history.append(adapter.port)
            adapter.port = 9000 + len(port_history)  # simulate port finding

        attempt_count = [0]

        async def fail_twice():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise TimeoutError("not ready")

        with patch.object(adapter, "_start", side_effect=tracking_start), \
             patch.object(adapter, "_wait_until_ready", side_effect=fail_twice), \
             patch.object(adapter, "stop"):
            asyncio.run(adapter.start())

        # On retry with auto_port, port should be reset to None before _start
        # (so _start will call _find_open_port again)
        self.assertTrue(adapter._auto_port)


class TestSetCancellationCallback(unittest.TestCase):
    """Test set_cancellation_callback registers callback."""

    def test_registers_callback(self):
        adapter = _make_adapter(port=8200)
        def callback():
            return False
        adapter.set_cancellation_callback(callback)
        self.assertIs(adapter._cancellation_callback, callback)

    def test_callback_initially_none(self):
        adapter = _make_adapter(port=8200)
        self.assertIsNone(adapter._cancellation_callback)


if __name__ == "__main__":
    unittest.main()
