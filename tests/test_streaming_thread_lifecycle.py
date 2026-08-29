"""Tests for streaming inference thread lifecycle — H2 (A2).

Verifies that the GPU ``inference_lock`` is NOT released until the streaming
inference thread has actually stopped, so an in-flight ``model.generate()``
cannot race the next request's GPU access.

Both the HTTP (``/generate-stream``) and WebSocket (``/ws``) paths are tested.
The mechanism: mock ``run_inference_streaming`` as a slow generator (yields one
chunk then blocks), trigger an early consumer exit (aclose / cancel) while the
thread is still mid-generation, and assert that a second lock acquisition does
not complete until the inference thread has signalled it is done.

Run: python -m unittest tests.test_streaming_thread_lifecycle -v
"""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

try:
    import numpy as np

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires numpy")


def _make_state():
    """Create a minimal app.state with all attributes the streaming paths need.

    Mirrors ``_setup_fastapi_app_state`` from voice_test_helpers but uses a
    plain SimpleNamespace so the test is self-contained (conftest autouse
    fixtures do NOT fire under the batch runner).
    """
    state = SimpleNamespace()
    state.auth_token = "test_token"  # nosec B105
    state.models = {"clone": None, "design": None, "custom": MagicMock()}
    state.model_load_times = {}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
    state.generation_state = {
        "active": False,
        "start_time": 0.0,
        "text_length": 0,
        "mode": "",
        "batch_index": 0,
        "batch_total": 0,
        "chunk_index": 0,
        "chunk_total": 0,
        "generation_id": None,
        "cancelled": False,
    }
    state.request_queue = set()
    state.request_queue_lock = threading.Lock()
    state.pending_requests = []
    state.pending_lock = asyncio.Lock()
    state.last_activity = 0
    state.models_loaded = threading.Event()
    state.models_loaded.set()
    state.gen_cache = {}
    state.gen_cache_lock = threading.Lock()
    state.inference_lock = asyncio.Lock()
    # _stream_generation now stamps generation_state under generation_lock (P3).
    state.generation_lock = asyncio.Lock()
    state.eta_cache = {"median_rate": None, "last_updated": 0}
    state.eta_cache_lock = threading.Lock()
    state.shutdown_timer = None
    state.server_config = {
        "security": {"max_text_length": 50000, "max_batch_size": 20},
        "auto_shutdown_minutes": 0,
    }
    return state


async def _drain_streaming_response(response):
    """Iterate a StreamingResponse body iterator to completion or first error."""
    async for _ in response.body_iterator:
        pass


async def _collect_streaming_response(response):
    """Return the full streamed body so terminal frames can be inspected."""
    body = b""
    async for part in response.body_iterator:
        body += part
    return body


def _terminal_error_from(body):
    """Decode the in-band terminal error frame (WS2 2.5), or None if absent.

    Frames are [sample_rate:4][length:4][payload:length]; a real audio chunk
    never has sample_rate 0, so the sentinel identifies the error frame.
    """
    import json
    import struct

    from qwen3_tts.server.app_generation import STREAM_ERROR_SENTINEL_SR

    offset = 0
    while offset + 8 <= len(body):
        sr, length = struct.unpack("<II", body[offset : offset + 8])
        payload = body[offset + 8 : offset + 8 + length]
        if sr == STREAM_ERROR_SENTINEL_SR:
            return json.loads(payload.decode("utf-8"))
        offset += 8 + length
    return None


def _reset_state(state):
    """Reset mutable state attributes between tests to avoid cross-pollution."""
    state.inference_lock = asyncio.Lock()
    state.generation_lock = asyncio.Lock()
    state.generation_state.update(
        {
            "active": False,
            "start_time": 0.0,
            "text_length": 0,
            "mode": "",
            "generation_id": None,
            "cancelled": False,
        }
    )


@_skip
class TestStreamingThreadLifecycle(unittest.IsolatedAsyncioTestCase):
    """The inference_lock must be held until the inference thread signals done."""

    async def test_http_inference_thread_joined_before_lock_release(self):
        """HTTP /generate-stream: closing the body generator (simulates client
        disconnect) must NOT release inference_lock until the inference thread
        has actually stopped.

        Pre-fix the consumer finally only sets stop_event and resets state,
        releasing the lock while the thread may still be mid-generation. Post-fix
        the finally awaits ``_await_inference_thread_done(done)`` before the
        ``async with inference_lock`` block exits.
        """
        from qwen3_tts.server.app_generation import handle_generate_stream
        from qwen3_tts.server.validation import GenerateRequest

        state = _make_state()
        release = threading.Event()
        thread_done_signal = threading.Event()

        def slow_inference(**kwargs):
            """Yield one chunk immediately, then block (mid-generation)."""
            chunk = np.zeros(100, dtype=np.float32)
            yield (chunk, 24000)
            release.wait(timeout=5.0)
            thread_done_signal.set()

        req = GenerateRequest(text="Hello world", mode="custom")

        try:
            with patch(
                "qwen3_tts.core.engine.run_inference_streaming",
                side_effect=slow_inference,
            ), patch(
                "qwen3_tts.server.validation._validate_generation_request"
            ), patch(
                "qwen3_tts.server.app_generation._check_memory_available",
                return_value=(True, 4096),
            ):
                response = await handle_generate_stream(
                    request=MagicMock(),
                    state=state,
                    req=req,
                    security={"max_text_length": 50000},
                    config_provider=None,
                )

                body = response.body_iterator

                # Consume first chunk — thread is now blocked in release.wait()
                chunk1 = await body.__anext__()
                self.assertIsNotNone(chunk1)

                # Second coroutine tries to acquire the lock
                second_acquired = asyncio.Event()

                async def try_acquire():
                    async with state.inference_lock:
                        second_acquired.set()

                acquire_task = asyncio.create_task(try_acquire())

                # Should NOT acquire yet — consumer still holds the lock
                await asyncio.sleep(0.05)
                self.assertFalse(
                    second_acquired.is_set(),
                    "Lock acquired while first generation still holds it",
                )

                # Schedule thread release after a short delay
                async def release_after_delay():
                    await asyncio.sleep(0.3)
                    release.set()

                release_task = asyncio.create_task(release_after_delay())

                # Close the generator (simulates client disconnect).
                # Post-fix: the finally awaits the inference thread's done
                # event, blocking until release fires (~0.3 s).
                # Pre-fix: the finally returns immediately, releasing the lock
                # while the thread is still blocked.
                await body.aclose()

                # Wait for the second acquisition to complete
                await asyncio.wait_for(acquire_task, timeout=5.0)

                # The thread MUST have finished before the lock was released.
                self.assertTrue(
                    thread_done_signal.is_set(),
                    "inference_lock was released before the inference "
                    "thread finished",
                )

                release_task.cancel()
                try:
                    await release_task
                except asyncio.CancelledError:
                    pass
        finally:
            _reset_state(state)

    async def test_ws_inference_thread_joined_before_lock_release(self):
        """WebSocket /ws: cancelling the generation task (simulates client
        disconnect) must NOT release inference_lock until the inference thread
        has actually stopped.
        """
        from qwen3_tts.server.websocket import _stream_generation

        state = _make_state()
        release = threading.Event()
        thread_done_signal = threading.Event()

        def slow_inference(**kwargs):
            chunk = np.zeros(100, dtype=np.float32)
            yield (chunk, 24000)
            release.wait(timeout=5.0)
            thread_done_signal.set()

        first_chunk_sent = threading.Event()
        ws = MagicMock()

        async def mock_send_bytes(data):
            first_chunk_sent.set()

        ws.send_bytes = mock_send_bytes
        ws.send_json = AsyncMock()

        stop_event = threading.Event()

        try:
            with patch(
                "qwen3_tts.core.engine.run_inference_streaming",
                side_effect=slow_inference,
            ), patch(
                "qwen3_tts.server.validation._validate_generation_request"
            ), patch(
                "qwen3_tts.server.app_lifespan._check_memory_available",
                return_value=(True, 4096),
            ):
                gen_task = asyncio.create_task(
                    _stream_generation(
                        websocket=ws,
                        app_state=state,
                        text="Hello world",
                        mode="custom",
                        data={"text": "Hello world", "mode": "custom"},
                        stop_event=stop_event,
                        disconnect_event=threading.Event(),
                    )
                )

                # Wait for the first chunk to be produced
                for _ in range(100):
                    if first_chunk_sent.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(
                    first_chunk_sent.is_set(), "First chunk was not sent"
                )

                # Second coroutine tries to acquire the lock
                second_acquired = asyncio.Event()

                async def try_acquire():
                    async with state.inference_lock:
                        second_acquired.set()

                acquire_task = asyncio.create_task(try_acquire())
                await asyncio.sleep(0.05)
                self.assertFalse(
                    second_acquired.is_set(),
                    "Lock acquired while first generation still holds it",
                )

                # Schedule thread release after a short delay
                async def release_after_delay():
                    await asyncio.sleep(0.3)
                    release.set()

                release_task = asyncio.create_task(release_after_delay())

                # Cancel the generation task (simulates client disconnect).
                # Post-fix: the consumer finally awaits done, blocking until
                # release fires (~0.3 s) and the thread finishes.
                # Pre-fix: no finally → CancelledError propagates immediately
                # → lock released while thread is still blocked.
                gen_task.cancel()
                try:
                    await gen_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

                await asyncio.wait_for(acquire_task, timeout=5.0)

                self.assertTrue(
                    thread_done_signal.is_set(),
                    "inference_lock was released before the inference "
                    "thread finished",
                )

                release_task.cancel()
                try:
                    await release_task
                except asyncio.CancelledError:
                    pass
        finally:
            _reset_state(state)


    async def test_http_stream_unknown_exception_does_not_deadlock(self):
        """H3: an exception outside the old narrow catch tuple (e.g.
        ``AttributeError``) must not deadlock the streaming response.

        Pre-fix the uncaught exception kills the inference thread WITHOUT
        queueing the ``None`` completion sentinel, so the consumer blocks
        forever on ``await queue.get()`` (deadlock).
        Post-fix the broad ``except Exception`` captures it and the ``finally``
        always sends the sentinel, so the response completes — the
        false-success guard raises ``RuntimeError`` because zero chunks were
        delivered.
        """
        from qwen3_tts.server.app_generation import handle_generate_stream
        from qwen3_tts.server.validation import GenerateRequest

        state = _make_state()

        def raising_inference(**kwargs):
            raise AttributeError("simulated unknown exception")

        req = GenerateRequest(text="Hello world", mode="custom")

        try:
            with patch(
                "qwen3_tts.core.engine.run_inference_streaming",
                side_effect=raising_inference,
            ), patch(
                "qwen3_tts.server.validation._validate_generation_request"
            ), patch(
                "qwen3_tts.server.app_generation._check_memory_available",
                return_value=(True, 4096),
            ):
                response = await handle_generate_stream(
                    request=MagicMock(),
                    state=state,
                    req=req,
                    security={"max_text_length": 50000},
                    config_provider=None,
                )

                # Pre-fix: consumer deadlocks on queue.get() -> wait_for times
                # out = RED. Post-fix: thread captures the error, sends the
                # sentinel, the consumer breaks and emits a terminal error
                # frame (WS2 2.5 — raising here would only truncate the
                # connection, which the client cannot tell from a network drop).
                body = await asyncio.wait_for(
                    _collect_streaming_response(response), timeout=5
                )
                self.assertIsNotNone(
                    _terminal_error_from(body),
                    "stream ended without a terminal error frame",
                )
        finally:
            _reset_state(state)

    async def test_http_stream_pre_chunk_error_aborts_response(self):
        """H3: a caught-tuple error before any chunk must NOT produce a silent
        empty 200 response.

        Pre-fix the narrow ``except`` captures the error and sends the ``None``
        sentinel, but the consumer completes cleanly with zero chunks —
        Starlette has already committed 200 status headers, so the client sees
        a silent empty 200.
        Post-fix the false-success guard raises ``RuntimeError`` so the client
        sees a broken connection instead.
        """
        from qwen3_tts.server.app_generation import handle_generate_stream
        from qwen3_tts.server.validation import GenerateRequest

        state = _make_state()

        def raising_inference(**kwargs):
            raise RuntimeError("boom")

        req = GenerateRequest(text="Hello world", mode="custom")

        try:
            with patch(
                "qwen3_tts.core.engine.run_inference_streaming",
                side_effect=raising_inference,
            ), patch(
                "qwen3_tts.server.validation._validate_generation_request"
            ), patch(
                "qwen3_tts.server.app_generation._check_memory_available",
                return_value=(True, 4096),
            ):
                response = await handle_generate_stream(
                    request=MagicMock(),
                    state=state,
                    req=req,
                    security={"max_text_length": 50000},
                    config_provider=None,
                )

                # Pre-fix: body drains cleanly (silent empty 200) = RED.
                # Post-fix: a terminal error frame carries the failure in band,
                # so the client gets the reason instead of a bare truncated
                # connection (WS2 2.5 superseded the raise-based guard).
                body = await asyncio.wait_for(
                    _collect_streaming_response(response), timeout=5
                )
                error = _terminal_error_from(body)
                self.assertIsNotNone(
                    error, "empty stream ended without a terminal error frame"
                )
                self.assertIn("boom", error["error"])
        finally:
            _reset_state(state)


@_skip
class TestWsStreamJoinTimeout(unittest.IsolatedAsyncioTestCase):
    """H1: the /ws consumer must scale its inference-thread join with the text
    length and configured chunk size, exactly like the HTTP path.

    Pre-fix ``websocket.py`` called ``_await_inference_thread_done(done)`` with
    no timeout, so the wait fell back to the flat 90 s floor. The join must
    cover ONE chunk's generation; once ``max_chunk_chars`` is raised above the
    500-char default a 90 s wait expires mid-generation and the consumer
    releases ``inference_lock`` while ``model.generate()`` is still on the GPU
    — the precise race the join exists to prevent. Only the HTTP call site and
    the helper itself were pinned; nothing pinned /ws.
    """

    async def _run_ws_generation(self, text, max_chunk_chars, join_stub):
        """Drive ``_stream_generation`` to completion with the join call stubbed.

        Returns the recorded call kwargs of ``_await_inference_thread_done``.
        """
        from qwen3_tts.server.websocket import _stream_generation

        state = _make_state()

        def quick_inference(**kwargs):
            yield (np.zeros(100, dtype=np.float32), 24000)

        ws = MagicMock()
        ws.send_bytes = AsyncMock()
        ws.send_json = AsyncMock()

        try:
            with patch(
                "qwen3_tts.core.engine.run_inference_streaming",
                side_effect=quick_inference,
            ), patch(
                "qwen3_tts.server.validation._validate_generation_request"
            ), patch(
                "qwen3_tts.server.app_lifespan._check_memory_available",
                return_value=(True, 4096),
            ), patch(
                "qwen3_tts.server.websocket._await_inference_thread_done",
                side_effect=join_stub,
            ) as join_mock:
                await asyncio.wait_for(
                    _stream_generation(
                        websocket=ws,
                        app_state=state,
                        text=text,
                        mode="custom",
                        data={
                            "text": text,
                            "mode": "custom",
                            "max_chunk_chars": max_chunk_chars,
                        },
                        stop_event=threading.Event(),
                        disconnect_event=threading.Event(),
                    ),
                    timeout=10,
                )
                return join_mock
        finally:
            _reset_state(state)

    async def test_ws_join_timeout_scales_with_text_and_chunk_size(self):
        """/ws must pass ``_stream_thread_join_timeout(len(text),
        req.max_chunk_chars)`` — not the flat floor."""
        from qwen3_tts.server.app_generation import (
            _STREAM_THREAD_JOIN_FLOOR_SEC,
            _stream_thread_join_timeout,
        )

        text = "a" * 2000
        max_chunk_chars = 1500

        async def join_stub(done, timeout=None):
            return True

        join_mock = await self._run_ws_generation(text, max_chunk_chars, join_stub)

        self.assertEqual(
            join_mock.call_count, 1, "consumer did not await the inference thread"
        )
        passed = join_mock.call_args.kwargs.get("timeout")
        expected = _stream_thread_join_timeout(len(text), max_chunk_chars)
        self.assertEqual(
            passed,
            expected,
            "/ws join timeout must be derived from the text length and "
            f"max_chunk_chars (expected {expected}s, got {passed})",
        )
        # Guards against a regression that silently reinstates the constant:
        # this request is large enough that the derived value clears the floor.
        self.assertGreater(
            expected,
            _STREAM_THREAD_JOIN_FLOOR_SEC,
            "test fixture too small to distinguish the derived value "
            "from the floor",
        )

    async def test_ws_logs_when_inference_thread_outlives_the_join(self):
        """A join that times out releases ``inference_lock`` with the thread
        still running; that must be logged, not silent (HTTP path parity)."""

        async def timing_out_join(done, timeout=None):
            return False

        with self.assertLogs("tts.server.websocket", level="ERROR") as captured:
            await self._run_ws_generation("hello world", 500, timing_out_join)

        self.assertTrue(
            any(
                "did not stop" in message
                for message in captured.output
            ),
            f"no timeout warning logged; got {captured.output}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
