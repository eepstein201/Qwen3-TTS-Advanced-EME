"""Characterization tests for engine_vllm.py VLLMAdapter.

Captures current I/O contracts:
- Constructor parameters and defaults
- Mode validation (clone/design/custom)
- Readiness guards
- Port finding behavior
- Stop idempotency
- Property behavior

These tests use mocking to avoid requiring vLLM, GPU, or a running server.
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVLLMAdapterInit(unittest.TestCase):
    """Characterize __init__ parameter contracts."""

    def test_default_model_name(self):
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        import inspect

        sig = inspect.signature(VLLMAdapter.__init__)
        self.assertEqual(
            sig.parameters["model_name"].default,
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        )

    def test_default_gpu_memory_utilization(self):
        from qwen3_tts.core.engine_vllm import VLLMAdapter
        import inspect

        sig = inspect.signature(VLLMAdapter.__init__)
        self.assertEqual(sig.parameters["gpu_memory_utilization"].default, 0.7)

    def test_default_port_is_none(self):
        from qwen3_tts.core.engine_vllm import VLLMAdapter
        import inspect

        sig = inspect.signature(VLLMAdapter.__init__)
        self.assertIsNone(sig.parameters["port"].default)

    def test_default_log_file_is_none(self):
        from qwen3_tts.core.engine_vllm import VLLMAdapter
        import inspect

        sig = inspect.signature(VLLMAdapter.__init__)
        self.assertIsNone(sig.parameters["log_file"].default)

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_init_stores_model_name(self, mock_log):
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter(model_name="test/model")
        self.assertEqual(adapter.model_name, "test/model")

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_init_stores_gpu_memory(self, mock_log):
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter(gpu_memory_utilization=0.9)
        self.assertEqual(adapter.gpu_memory_utilization, 0.9)

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_init_auto_port_flag(self, mock_log):
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        # port=None → auto_port=True
        adapter = VLLMAdapter()
        self.assertTrue(adapter._auto_port)

        # port=8100 → auto_port=False
        adapter2 = VLLMAdapter(port=8100)
        self.assertFalse(adapter2._auto_port)

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_init_process_is_none(self, mock_log):
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter()
        self.assertIsNone(adapter._process)

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_init_client_is_none(self, mock_log):
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter()
        self.assertIsNone(adapter._client)

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_init_not_ready(self, mock_log):
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter()
        self.assertFalse(adapter.is_ready())

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_init_custom_log_file(self, mock_log):
        mock_log.return_value = "/tmp/default.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter(log_file="/custom/path.log")
        self.assertEqual(adapter.log_file, "/custom/path.log")


class TestVLLMAdapterIsReady(unittest.TestCase):
    """Characterize is_ready() contract."""

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_is_ready_returns_bool(self, mock_log):
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter()
        result = adapter.is_ready()
        self.assertIsInstance(result, bool)
        self.assertFalse(result)

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_is_ready_true_after_event_set(self, mock_log):
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter()
        adapter._ready_event.set()
        self.assertTrue(adapter.is_ready())


class TestVLLMAdapterBaseUrl(unittest.TestCase):
    """Characterize base_url property contract."""

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_base_url_raises_when_no_port(self, mock_log):
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter()  # port=None
        with self.assertRaises(RuntimeError):
            _ = adapter.base_url

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_base_url_format(self, mock_log):
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter(port=8200)
        self.assertEqual(adapter.base_url, "http://127.0.0.1:8200")


class TestVLLMAdapterStop(unittest.TestCase):
    """Characterize stop() idempotency contract."""

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_stop_idempotent_no_process(self, mock_log):
        """stop() on adapter with no process should not raise."""
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter()
        # Should not raise
        adapter.stop()

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_stop_clears_process(self, mock_log):
        """stop() should set _process to None."""
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter(port=8100)
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.wait.return_value = 0
        adapter._process = mock_proc

        with patch("os.getpgid", return_value=12345), \
             patch("os.killpg"):
            adapter.stop()

        self.assertIsNone(adapter._process)

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_stop_clears_ready_event(self, mock_log):
        """stop() should clear the ready event."""
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter(port=8100)
        adapter._ready_event.set()

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.wait.return_value = 0
        adapter._process = mock_proc

        with patch("os.getpgid", return_value=12345), \
             patch("os.killpg"):
            adapter.stop()

        self.assertFalse(adapter.is_ready())


class TestVLLMAdapterFindPort(unittest.TestCase):
    """Characterize _find_open_port() contract."""

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_find_open_port_returns_int(self, mock_log):
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter()
        port = adapter._find_open_port(start_port=49152)
        self.assertIsInstance(port, int)
        self.assertGreaterEqual(port, 49152)
        self.assertLess(port, 49252)  # start_port + 100

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    def test_find_open_port_default_start(self, mock_log):
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter()
        import inspect

        sig = inspect.signature(adapter._find_open_port)
        self.assertEqual(sig.parameters["start_port"].default, 8100)

    @patch("qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file")
    @patch("socket.socket")
    def test_find_open_port_raises_when_all_busy(self, mock_socket_cls, mock_log):
        """Raises RuntimeError when no port available in range."""
        mock_log.return_value = "/tmp/test.log"
        from qwen3_tts.core.engine_vllm import VLLMAdapter

        adapter = VLLMAdapter()

        # Make all bind attempts fail
        mock_sock = MagicMock()
        mock_sock.__enter__ = MagicMock(return_value=mock_sock)
        mock_sock.__exit__ = MagicMock(return_value=False)
        mock_sock.bind.side_effect = OSError("Address in use")
        mock_socket_cls.return_value = mock_sock

        with self.assertRaises(RuntimeError) as ctx:
            adapter._find_open_port(start_port=8100)
        self.assertIn("Could not find open port", str(ctx.exception))


class TestVLLMAdapterGenerateValidation(unittest.TestCase):
    """Characterize generate() input validation contracts."""

    def _make_adapter(self):
        with patch(
            "qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file",
            return_value="/tmp/test.log",
        ):
            from qwen3_tts.core.engine_vllm import VLLMAdapter

            adapter = VLLMAdapter(port=8100)
        return adapter

    def test_generate_raises_when_not_ready(self):
        """generate() must raise RuntimeError if server not ready."""
        adapter = self._make_adapter()
        self.assertFalse(adapter.is_ready())

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(
                adapter.generate("hello", mode="clone", prompt_audio=b"fake")
            )
        self.assertIn("not ready", str(ctx.exception))

    def test_generate_clone_requires_prompt_audio(self):
        """clone mode without prompt_audio raises ValueError."""
        adapter = self._make_adapter()
        adapter._ready_event.set()

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(
                adapter.generate("hello", mode="clone")
            )
        self.assertIn("prompt_audio", str(ctx.exception))

    def test_generate_design_requires_voice_description(self):
        """design mode without voice_description raises ValueError."""
        adapter = self._make_adapter()
        adapter._ready_event.set()

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(
                adapter.generate("hello", mode="design")
            )
        self.assertIn("voice_description", str(ctx.exception))

    def test_generate_custom_requires_speaker(self):
        """custom mode without speaker raises ValueError."""
        adapter = self._make_adapter()
        adapter._ready_event.set()

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(
                adapter.generate("hello", mode="custom")
            )
        self.assertIn("speaker", str(ctx.exception))

    def test_generate_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        adapter = self._make_adapter()
        adapter._ready_event.set()

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(
                adapter.generate("hello", mode="invalid_mode")
            )
        self.assertIn("Invalid mode", str(ctx.exception))

    def test_generate_return_type_annotation(self):
        """generate() return type is tuple[int, np.ndarray]."""
        import inspect

        from qwen3_tts.core.engine_vllm import VLLMAdapter

        sig = inspect.signature(VLLMAdapter.generate)
        ret = sig.return_annotation
        self.assertIn("tuple", str(ret).lower())


class TestVLLMAdapterGenerateStreamValidation(unittest.TestCase):
    """Characterize generate_stream() input validation contracts."""

    def _make_adapter(self):
        with patch(
            "qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file",
            return_value="/tmp/test.log",
        ):
            from qwen3_tts.core.engine_vllm import VLLMAdapter

            adapter = VLLMAdapter(port=8100)
        return adapter

    def test_stream_raises_when_not_ready(self):
        """generate_stream() must raise RuntimeError if server not ready."""
        adapter = self._make_adapter()

        async def run():
            async for _ in adapter.generate_stream("hello", mode="clone", prompt_audio=b"fake"):
                pass

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(run())
        self.assertIn("not ready", str(ctx.exception))

    def test_stream_clone_requires_prompt_audio(self):
        """clone mode without prompt_audio raises ValueError."""
        adapter = self._make_adapter()
        adapter._ready_event.set()

        async def run():
            async for _ in adapter.generate_stream("hello", mode="clone"):
                pass

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(run())
        self.assertIn("prompt_audio", str(ctx.exception))

    def test_stream_design_requires_voice_description(self):
        """design mode without voice_description raises ValueError."""
        adapter = self._make_adapter()
        adapter._ready_event.set()

        async def run():
            async for _ in adapter.generate_stream("hello", mode="design"):
                pass

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(run())
        self.assertIn("voice_description", str(ctx.exception))

    def test_stream_custom_requires_speaker(self):
        """custom mode without speaker raises ValueError."""
        adapter = self._make_adapter()
        adapter._ready_event.set()

        async def run():
            async for _ in adapter.generate_stream("hello", mode="custom"):
                pass

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(run())
        self.assertIn("speaker", str(ctx.exception))

    def test_stream_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        adapter = self._make_adapter()
        adapter._ready_event.set()

        async def run():
            async for _ in adapter.generate_stream("hello", mode="bogus"):
                pass

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(run())
        self.assertIn("Invalid mode", str(ctx.exception))


class TestVLLMAdapterCancellation(unittest.TestCase):
    """Characterize cancellation callback contract."""

    def _make_adapter(self):
        with patch(
            "qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file",
            return_value="/tmp/test.log",
        ):
            from qwen3_tts.core.engine_vllm import VLLMAdapter

            adapter = VLLMAdapter(port=8100)
        return adapter

    def test_cancellation_callback_initially_none(self):
        adapter = self._make_adapter()
        self.assertIsNone(adapter._cancellation_callback)

    def test_set_cancellation_callback_stores_callable(self):
        adapter = self._make_adapter()
        def _noop():
            return False

        adapter.set_cancellation_callback(_noop)
        self.assertIs(adapter._cancellation_callback, _noop)


class TestVLLMAdapterStartGuard(unittest.TestCase):
    """Characterize _start() guard against double-start."""

    def _make_adapter(self):
        with patch(
            "qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file",
            return_value="/tmp/test.log",
        ):
            from qwen3_tts.core.engine_vllm import VLLMAdapter

            adapter = VLLMAdapter(port=8100)
        return adapter

    def test_start_raises_if_process_already_running(self):
        """_start() raises RuntimeError if _process is not None."""
        adapter = self._make_adapter()
        adapter._process = MagicMock()  # Simulate running process

        with self.assertRaises(RuntimeError) as ctx:
            adapter._start()
        self.assertIn("already running", str(ctx.exception))


class TestVLLMAdapterGetClient(unittest.TestCase):
    """Characterize _get_client() HTTP client creation."""

    def _make_adapter(self):
        with patch(
            "qwen3_tts.core.engine_vllm.VLLMAdapter._get_default_log_file",
            return_value="/tmp/test.log",
        ):
            from qwen3_tts.core.engine_vllm import VLLMAdapter

            adapter = VLLMAdapter(port=8100)
        return adapter

    def test_get_client_creates_httpx_client(self):
        import httpx

        adapter = self._make_adapter()
        client = adapter._get_client()
        self.assertIsInstance(client, httpx.AsyncClient)

    def test_get_client_reuses_instance(self):
        """Subsequent calls return the same client."""
        adapter = self._make_adapter()
        c1 = adapter._get_client()
        c2 = adapter._get_client()
        self.assertIs(c1, c2)

    def test_get_client_timeout_config(self):
        """Client should have extended timeout for long generations."""
        adapter = self._make_adapter()
        client = adapter._get_client()
        # 600s overall timeout
        self.assertEqual(client.timeout.read, 600.0)
        self.assertEqual(client.timeout.connect, 60.0)


if __name__ == "__main__":
    unittest.main()
