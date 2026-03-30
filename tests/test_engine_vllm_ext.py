"""Extended tests for engine_vllm.py — covers async paths, subprocess lifecycle,
HTTP client mocking, and streaming.

Targets uncovered lines:
  90-94   _get_default_log_file
  131-167 _start() subprocess launch
  184-218 _wait_until_ready()
  264-265 stop() ProcessLookupError in killpg
  271-278 stop() TimeoutExpired + SIGKILL
  286-290 stop() close log file handle
  294-298 stop() close HTTP client
  345-355 _build_request() clone temp file path
  363     _build_request() design
  371     _build_request() custom
  379-380 _build_request() kwargs merge + return
  424-445 generate() HTTP call + response parsing
  478-518 generate_stream() HTTP streaming
"""

import asyncio
import io
import json
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

_MOD = "qwen3_tts.core.engine_vllm"


def _make_adapter(port=8200, log_file=None):
    """Create adapter with mocked _get_default_log_file."""
    with patch(f"{_MOD}.VLLMAdapter._get_default_log_file", return_value="/tmp/test_vllm_ext.log"):
        from qwen3_tts.core.engine_vllm import VLLMAdapter
        return VLLMAdapter(port=port, log_file=log_file or "/tmp/test_vllm_ext.log")


class TestGetDefaultLogFile(unittest.TestCase):
    """Test _get_default_log_file creates directory and returns path."""

    def test_returns_string_path(self):
        from qwen3_tts.core.engine_vllm import VLLMAdapter
        with patch("qwen3_tts.core.config.USER_FILES_DIR", "/tmp/fake_user_files"), \
             patch.object(Path, "mkdir"):
            adapter = VLLMAdapter.__new__(VLLMAdapter)
            result = adapter._get_default_log_file()
        self.assertIsInstance(result, str)
        self.assertIn("vllm_server.log", result)

    def test_creates_log_directory(self):
        from qwen3_tts.core.engine_vllm import VLLMAdapter
        with patch("qwen3_tts.core.config.USER_FILES_DIR", "/tmp/fake_user_files"), \
             patch.object(Path, "mkdir") as mock_mkdir:
            adapter = VLLMAdapter.__new__(VLLMAdapter)
            adapter._get_default_log_file()
        mock_mkdir.assert_called_once_with(exist_ok=True)


class TestStartSubprocess(unittest.TestCase):
    """Test _start() spawns subprocess correctly."""

    def test_start_with_explicit_port(self):
        """_start() uses existing port without calling _find_open_port."""
        adapter = _make_adapter(port=8300)
        mock_proc = MagicMock()
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("builtins.open", MagicMock()):
            adapter._start()

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        self.assertIn("8300", cmd)
        self.assertEqual(adapter._process, mock_proc)

    def test_start_auto_port_calls_find_open_port(self):
        """_start() with port=None calls _find_open_port."""
        with patch(f"{_MOD}.VLLMAdapter._get_default_log_file", return_value="/tmp/test.log"):
            from qwen3_tts.core.engine_vllm import VLLMAdapter
            adapter = VLLMAdapter(log_file="/tmp/test.log")  # port=None → auto
        self.assertIsNone(adapter.port)

        mock_proc = MagicMock()
        with patch.object(adapter, "_find_open_port", return_value=9999) as mock_find, \
             patch("subprocess.Popen", return_value=mock_proc), \
             patch("builtins.open", MagicMock()):
            adapter._start()

        mock_find.assert_called_once()
        self.assertEqual(adapter.port, 9999)

    def test_start_includes_model_name_in_cmd(self):
        adapter = _make_adapter(port=8301)
        adapter.model_name = "TestOrg/TestModel"
        mock_proc = MagicMock()
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("builtins.open", MagicMock()):
            adapter._start()

        cmd = mock_popen.call_args[0][0]
        self.assertIn("TestOrg/TestModel", cmd)

    def test_start_raises_when_already_running(self):
        adapter = _make_adapter(port=8302)
        adapter._process = MagicMock()
        with self.assertRaises(RuntimeError) as ctx:
            adapter._start()
        self.assertIn("already running", str(ctx.exception))

    def test_start_sets_log_file_handle(self):
        adapter = _make_adapter(port=8303)
        mock_proc = MagicMock()
        mock_fh = MagicMock()
        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("builtins.open", return_value=mock_fh):
            adapter._start()
        self.assertIs(adapter._log_fh, mock_fh)


class TestWaitUntilReady(unittest.TestCase):
    """Test _wait_until_ready() async polling."""

    def test_ready_on_first_poll(self):
        adapter = _make_adapter(port=8200)
        adapter._process = None  # No process check

        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            asyncio.run(adapter._wait_until_ready(timeout=5.0))

        self.assertTrue(adapter._ready_event.is_set())

    def test_timeout_raises_TimeoutError(self):
        """_wait_until_ready raises TimeoutError if server never responds."""
        adapter = _make_adapter(port=8201)
        adapter._process = None

        import httpx
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(f"{_MOD}._HEALTH_CHECK_POLL_SECS", 0.01):
            with self.assertRaises(TimeoutError):
                asyncio.run(adapter._wait_until_ready(timeout=0.05))

    def test_process_exit_raises_RuntimeError(self):
        """_wait_until_ready raises RuntimeError if process exits unexpectedly."""
        adapter = _make_adapter(port=8202)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # non-None → process exited
        mock_proc.returncode = 1
        adapter._process = mock_proc

        import httpx
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("conn refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(adapter._wait_until_ready(timeout=5.0))
        self.assertIn("exited with code", str(ctx.exception))

    def test_http_status_error_retries(self):
        """HTTPStatusError is caught and treated as not-ready."""
        adapter = _make_adapter(port=8203)
        adapter._process = None

        call_count = 0

        import httpx
        async def mock_get(url):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
            mock_response = MagicMock()
            mock_response.status_code = 200
            return mock_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(f"{_MOD}._HEALTH_CHECK_POLL_SECS", 0.01):
            asyncio.run(adapter._wait_until_ready(timeout=5.0))

        self.assertTrue(adapter.is_ready())


class TestStopSubprocess(unittest.TestCase):
    """Test stop() lifecycle including edge cases."""

    def test_stop_no_process_logs_warning(self):
        """stop() with no process just logs warning and returns."""
        adapter = _make_adapter(port=8200)
        # Should not raise
        adapter.stop()
        self.assertIsNone(adapter._process)

    def test_stop_process_lookup_error_in_killpg(self):
        """ProcessLookupError in killpg is swallowed."""
        adapter = _make_adapter(port=8200)
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.wait.return_value = 0
        adapter._process = mock_proc

        with patch("os.getpgid", return_value=99999), \
             patch("os.killpg", side_effect=ProcessLookupError("not found")):
            adapter.stop()  # Should not raise

        self.assertIsNone(adapter._process)
        self.assertFalse(adapter.is_ready())

    def test_stop_timeout_expired_forces_sigkill(self):
        """TimeoutExpired causes SIGKILL to process group."""
        import subprocess
        import signal

        adapter = _make_adapter(port=8200)
        mock_proc = MagicMock()
        mock_proc.pid = 88888
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="vllm", timeout=10)]
        adapter._process = mock_proc

        killpg_calls = []
        def fake_killpg(pgid, sig):
            killpg_calls.append(sig)
            if sig == signal.SIGKILL:
                mock_proc.wait.side_effect = None
                mock_proc.wait.return_value = -9

        with patch("os.getpgid", return_value=88888), \
             patch("os.killpg", side_effect=fake_killpg):
            adapter.stop()

        self.assertIn(signal.SIGKILL, killpg_calls)
        self.assertIsNone(adapter._process)

    def test_stop_closes_log_file_handle(self):
        adapter = _make_adapter(port=8200)
        mock_proc = MagicMock()
        mock_proc.pid = 77777
        mock_proc.wait.return_value = 0
        adapter._process = mock_proc

        mock_fh = MagicMock()
        adapter._log_fh = mock_fh

        with patch("os.getpgid", return_value=77777), \
             patch("os.killpg"):
            adapter.stop()

        mock_fh.close.assert_called_once()
        self.assertIsNone(adapter._log_fh)

    def test_stop_log_file_close_exception_suppressed(self):
        """Exception when closing log file is caught and logged."""
        adapter = _make_adapter(port=8200)
        mock_proc = MagicMock()
        mock_proc.pid = 66666
        mock_proc.wait.return_value = 0
        adapter._process = mock_proc

        mock_fh = MagicMock()
        mock_fh.close.side_effect = IOError("disk error")
        adapter._log_fh = mock_fh

        with patch("os.getpgid", return_value=66666), \
             patch("os.killpg"):
            adapter.stop()  # Should not raise

        self.assertIsNone(adapter._log_fh)

    def test_stop_closes_http_client(self):
        """stop() closes HTTP client when it exists."""
        adapter = _make_adapter(port=8200)
        mock_proc = MagicMock()
        mock_proc.pid = 55555
        mock_proc.wait.return_value = 0
        adapter._process = mock_proc

        mock_client = MagicMock()
        adapter._client = mock_client

        async def mock_aclose():
            pass
        mock_client.aclose = mock_aclose

        with patch("os.getpgid", return_value=55555), \
             patch("os.killpg"):
            asyncio.run(asyncio.sleep(0))  # ensure event loop exists
            adapter.stop()

        self.assertIsNone(adapter._client)

    def test_stop_http_client_close_exception_suppressed(self):
        """Exception when closing HTTP client is caught and logged."""
        adapter = _make_adapter(port=8200)
        mock_proc = MagicMock()
        mock_proc.pid = 44444
        mock_proc.wait.return_value = 0
        adapter._process = mock_proc

        mock_client = MagicMock()
        mock_client.aclose.side_effect = RuntimeError("no event loop")
        adapter._client = mock_client

        # No running event loop in this thread, so create_task will fail
        with patch("os.getpgid", return_value=44444), \
             patch("os.killpg"), \
             patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")):
            adapter.stop()  # Should not raise

        self.assertIsNone(adapter._client)


class TestBuildRequest(unittest.TestCase):
    """Test _build_request() for all modes."""

    def setUp(self):
        self.adapter = _make_adapter(port=8200)

    def test_clone_mode_creates_temp_file(self):
        audio_bytes = b"\x00\x01\x02\x03" * 100
        request = self.adapter._build_request(
            "hello", "clone", prompt_audio=audio_bytes,
            voice_description=None, speaker=None,
        )
        self.assertEqual(request["input"]["mode"], "clone")
        prompt_path = request["input"]["prompt_audio"]
        self.assertTrue(os.path.exists(prompt_path))
        with open(prompt_path, "rb") as f:
            self.assertEqual(f.read(), audio_bytes)
        os.unlink(prompt_path)  # cleanup

    def test_clone_write_exception_cleans_up_temp_file(self):
        """If temp file write fails, temp file is cleaned up."""
        adapter = _make_adapter(port=8200)
        with patch("tempfile.NamedTemporaryFile") as mock_ntf:
            mock_fh = MagicMock()
            mock_fh.name = "/tmp/fake_vllm_test.wav"
            mock_fh.write.side_effect = OSError("disk full")
            mock_ntf.return_value = mock_fh

            with patch("os.unlink") as mock_unlink:
                with self.assertRaises(OSError):
                    adapter._build_request("hello", "clone", prompt_audio=b"audio",
                                           voice_description=None, speaker=None)
            mock_unlink.assert_called_once_with("/tmp/fake_vllm_test.wav")

    def test_design_mode_request(self):
        request = self.adapter._build_request(
            "hello", "design", prompt_audio=None,
            voice_description="warm friendly voice", speaker=None,
        )
        self.assertEqual(request["input"]["mode"], "design")
        self.assertEqual(request["input"]["voice_description"], "warm friendly voice")
        self.assertNotIn("prompt_audio", request["input"])

    def test_custom_mode_request(self):
        request = self.adapter._build_request(
            "hello", "custom", prompt_audio=None,
            voice_description=None, speaker="Bob",
        )
        self.assertEqual(request["input"]["mode"], "custom")
        self.assertEqual(request["input"]["speaker"], "Bob")

    def test_kwargs_merged_into_request(self):
        request = self.adapter._build_request(
            "hello", "design", prompt_audio=None,
            voice_description="test voice", speaker=None,
            language="en", temperature=0.8,
        )
        self.assertEqual(request["input"]["language"], "en")
        self.assertEqual(request["input"]["temperature"], 0.8)

    def test_model_name_in_request(self):
        self.adapter.model_name = "Org/Model-1.7B"
        request = self.adapter._build_request(
            "hello", "design", prompt_audio=None,
            voice_description="test", speaker=None,
        )
        self.assertEqual(request["model"], "Org/Model-1.7B")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.adapter._build_request("hello", "invalid", None, None, None)
        self.assertIn("Invalid mode", str(ctx.exception))


class TestGenerateHTTP(unittest.TestCase):
    """Test generate() HTTP call and response parsing."""

    def setUp(self):
        self.adapter = _make_adapter(port=8200)
        self.adapter._ready_event.set()

    def _make_mock_response(self, audio_data: bytes):
        """Build a mock httpx response with base64 audio."""
        import base64
        audio_b64 = base64.b64encode(audio_data).decode()
        resp_data = {"data": [{"audio": audio_b64}]}
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=resp_data)
        return mock_response

    def test_generate_design_success(self):
        """generate() succeeds with design mode, returns (sr, audio)."""
        import numpy as np
        fake_audio = np.zeros(1000, dtype=np.float32)
        fake_sr = 24000

        buf = io.BytesIO()
        import soundfile as sf
        sf.write(buf, fake_audio, fake_sr, format="WAV")
        wav_bytes = buf.getvalue()

        mock_response = self._make_mock_response(wav_bytes)
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        self.adapter._client = mock_client

        sr, audio = asyncio.run(
            self.adapter.generate("hello", mode="design", voice_description="test voice")
        )
        self.assertEqual(sr, fake_sr)
        self.assertEqual(len(audio), len(fake_audio))

    def test_generate_custom_success(self):
        """generate() succeeds with custom mode."""
        import numpy as np
        fake_audio = np.zeros(500, dtype=np.float32)
        buf = io.BytesIO()
        import soundfile as sf
        sf.write(buf, fake_audio, 22050, format="WAV")

        mock_response = self._make_mock_response(buf.getvalue())
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        self.adapter._client = mock_client

        sr, audio = asyncio.run(
            self.adapter.generate("hello", mode="custom", speaker="Alice")
        )
        self.assertEqual(sr, 22050)

    def test_generate_http_status_error_raises_RuntimeError(self):
        """HTTPStatusError is wrapped as RuntimeError."""
        import httpx

        mock_request = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Internal Server Error"
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError("500", request=mock_request, response=mock_resp)
        )
        self.adapter._client = mock_client

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(self.adapter.generate("hello", mode="design", voice_description="v"))
        self.assertIn("vLLM generation failed", str(ctx.exception))

    def test_generate_generic_exception_reraises(self):
        """Generic exceptions are re-raised as-is."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ValueError("unexpected"))
        self.adapter._client = mock_client

        with self.assertRaises(ValueError):
            asyncio.run(self.adapter.generate("hello", mode="design", voice_description="v"))

    def test_generate_clone_success(self):
        """generate() with clone mode writes temp file and succeeds."""
        import numpy as np
        fake_audio = np.zeros(800, dtype=np.float32)
        buf = io.BytesIO()
        import soundfile as sf
        sf.write(buf, fake_audio, 16000, format="WAV")

        mock_response = self._make_mock_response(buf.getvalue())
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        self.adapter._client = mock_client

        sr, audio = asyncio.run(
            self.adapter.generate("hello", mode="clone", prompt_audio=b"\x00\x01" * 100)
        )
        self.assertEqual(sr, 16000)


class TestGenerateStreamHTTP(unittest.TestCase):
    """Test generate_stream() SSE streaming."""

    def setUp(self):
        self.adapter = _make_adapter(port=8200)
        self.adapter._ready_event.set()

    def _make_stream_context(self, sse_lines: list[str]):
        """Build mock async context manager for client.stream()."""

        async def fake_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_lines = fake_aiter_lines
        mock_response.close = MagicMock()

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        return mock_context, mock_response

    def _wav_bytes(self, samples=500, sr=24000):
        import numpy as np
        import soundfile as sf
        audio = np.zeros(samples, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return buf.getvalue()

    def _sse_line(self, audio_bytes: bytes) -> str:
        """Build an SSE data line with base64 audio."""
        import base64
        payload = {"audio": base64.b64encode(audio_bytes).decode()}
        return f"data: {json.dumps(payload)}"

    def test_stream_yields_chunks(self):
        """Streaming with valid SSE lines yields audio chunks."""
        wav1 = self._wav_bytes(500)
        wav2 = self._wav_bytes(300)
        sse_lines = [
            self._sse_line(wav1),
            "",  # empty lines are skipped
            self._sse_line(wav2),
            "data: [DONE]",
        ]

        mock_ctx, _ = self._make_stream_context(sse_lines)

        async def run():
            async with mock_ctx:
                pass  # set up context

        # Test via actual streaming
        async def collect():
            chunks = []
            with patch.object(self.adapter._get_client(), "stream", return_value=mock_ctx):
                async for sr, audio in self.adapter.generate_stream(
                    "hello", mode="design", voice_description="warm"
                ):
                    chunks.append((sr, audio))
            return chunks

        self.adapter._client = self.adapter._get_client()
        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)
        self.adapter._client = mock_client

        chunks = asyncio.run(collect())
        self.assertEqual(len(chunks), 2)

    def test_stream_skips_empty_lines(self):
        """Empty lines in SSE stream are ignored."""
        wav = self._wav_bytes(200)
        sse_lines = ["", "  ", self._sse_line(wav), "data: [DONE]"]
        mock_ctx, _ = self._make_stream_context(sse_lines)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)
        self.adapter._client = mock_client

        async def collect():
            chunks = []
            async for sr, audio in self.adapter.generate_stream(
                "hello", mode="design", voice_description="v"
            ):
                chunks.append(sr)
            return chunks

        chunks = asyncio.run(collect())
        self.assertEqual(len(chunks), 1)

    def test_stream_done_sentinel_stops_iteration(self):
        """[DONE] sentinel ends streaming."""
        sse_lines = ["data: [DONE]", self._sse_line(self._wav_bytes(100))]
        mock_ctx, _ = self._make_stream_context(sse_lines)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)
        self.adapter._client = mock_client

        async def collect():
            chunks = []
            async for sr, audio in self.adapter.generate_stream(
                "hello", mode="design", voice_description="v"
            ):
                chunks.append(sr)
            return chunks

        chunks = asyncio.run(collect())
        self.assertEqual(len(chunks), 0)  # [DONE] breaks before the audio line

    def test_stream_invalid_json_skipped(self):
        """Invalid JSON in SSE line is skipped with warning."""
        wav = self._wav_bytes(100)
        sse_lines = [
            "data: {not valid json",
            self._sse_line(wav),
            "data: [DONE]",
        ]
        mock_ctx, _ = self._make_stream_context(sse_lines)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)
        self.adapter._client = mock_client

        async def collect():
            chunks = []
            async for sr, audio in self.adapter.generate_stream(
                "hello", mode="design", voice_description="v"
            ):
                chunks.append(sr)
            return chunks

        chunks = asyncio.run(collect())
        self.assertEqual(len(chunks), 1)

    def test_stream_cancellation_callback_triggers(self):
        """Cancellation callback causes RuntimeError('Generation cancelled')."""
        cancel_called = []

        def cancel_cb():
            cancel_called.append(True)
            return True

        self.adapter.set_cancellation_callback(cancel_cb)

        wav = self._wav_bytes(100)
        sse_lines = [self._sse_line(wav), "data: [DONE]"]
        mock_ctx, mock_resp = self._make_stream_context(sse_lines)
        mock_resp.close = MagicMock()

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)
        self.adapter._client = mock_client

        async def collect():
            async for _ in self.adapter.generate_stream(
                "hello", mode="design", voice_description="v"
            ):
                pass

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(collect())
        self.assertIn("cancelled", str(ctx.exception))

    def test_stream_http_status_error_raises_RuntimeError(self):
        """HTTPStatusError is wrapped as RuntimeError."""
        import httpx

        mock_resp = MagicMock()
        mock_resp.text = "503 Service Unavailable"
        err = httpx.HTTPStatusError("503", request=MagicMock(), response=mock_resp)

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=err)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)
        self.adapter._client = mock_client

        async def collect():
            async for _ in self.adapter.generate_stream(
                "hello", mode="design", voice_description="v"
            ):
                pass

        with self.assertRaises(RuntimeError) as ctx:
            asyncio.run(collect())
        self.assertIn("vLLM streaming failed", str(ctx.exception))

    def test_stream_generic_exception_reraises(self):
        """Generic exceptions propagate from streaming."""
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=ConnectionError("refused"))
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_ctx)
        self.adapter._client = mock_client

        async def collect():
            async for _ in self.adapter.generate_stream(
                "hello", mode="design", voice_description="v"
            ):
                pass

        with self.assertRaises(ConnectionError):
            asyncio.run(collect())




class TestVLLMStartCmdParams(unittest.TestCase):
    """HIGH-1/MED-2: _start() command must include multimodal and perf params."""

    def _get_start_cmd(self, **adapter_kwargs):
        """Instantiate adapter, call _start(), return the cmd list."""
        with patch(f"{_MOD}.VLLMAdapter._get_default_log_file", return_value="/tmp/test_vllm.log"):
            from qwen3_tts.core.engine_vllm import VLLMAdapter
            adapter = VLLMAdapter(port=9999, log_file="/tmp/test_vllm.log", **adapter_kwargs)

        mock_proc = MagicMock()
        with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, \
             patch("builtins.open", MagicMock()):
            adapter._start()
        return mock_popen.call_args[0][0]

    def test_cmd_includes_limit_mm_per_prompt_audio_1(self):
        """_start() cmd must include --limit-mm-per-prompt audio=1."""
        cmd = self._get_start_cmd()
        self.assertIn("--limit-mm-per-prompt", cmd)
        idx = cmd.index("--limit-mm-per-prompt")
        self.assertEqual(cmd[idx + 1], "audio=1")

    def test_cmd_includes_chunked_prefill(self):
        """_start() cmd must include --enable-chunked-prefill."""
        cmd = self._get_start_cmd()
        self.assertIn("--enable-chunked-prefill", cmd)

    def test_cmd_includes_dtype_bfloat16_default(self):
        """_start() cmd must include --dtype bfloat16 by default."""
        cmd = self._get_start_cmd()
        self.assertIn("--dtype", cmd)
        idx = cmd.index("--dtype")
        self.assertEqual(cmd[idx + 1], "bfloat16")

    def test_cmd_includes_max_model_len(self):
        """_start() cmd must include --max-model-len."""
        cmd = self._get_start_cmd()
        self.assertIn("--max-model-len", cmd)

    def test_dtype_is_configurable(self):
        """dtype kwarg to __init__ is used in the launch command."""
        cmd = self._get_start_cmd(dtype="float16")
        idx = cmd.index("--dtype")
        self.assertEqual(cmd[idx + 1], "float16")

    def test_max_model_len_is_configurable(self):
        """max_model_len kwarg to __init__ is used in the launch command."""
        cmd = self._get_start_cmd(max_model_len=8192)
        idx = cmd.index("--max-model-len")
        self.assertEqual(cmd[idx + 1], "8192")

    def test_existing_params_still_present(self):
        """All pre-existing params remain in the command."""
        cmd = self._get_start_cmd()
        for flag in ("--model", "--gpu-memory-utilization", "--port", "--disable-log-requests"):
            self.assertIn(flag, cmd)


if __name__ == "__main__":
    unittest.main()
