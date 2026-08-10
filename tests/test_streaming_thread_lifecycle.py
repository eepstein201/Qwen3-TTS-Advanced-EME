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
                # out -> asyncio.TimeoutError (not RuntimeError) = RED.
                # Post-fix: thread captures error, sends sentinel, consumer
                # breaks, false-success guard raises RuntimeError = GREEN.
                with self.assertRaises(RuntimeError):
                    await asyncio.wait_for(
                        _drain_streaming_response(response), timeout=5
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

                # Pre-fix: body drains cleanly (empty 200) -> assertRaises
                # fails because no RuntimeError is raised = RED.
                # Post-fix: false-success guard raises RuntimeError = GREEN.
                with self.assertRaises(RuntimeError):
                    await asyncio.wait_for(
                        _drain_streaming_response(response), timeout=5
                    )
        finally:
            _reset_state(state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
