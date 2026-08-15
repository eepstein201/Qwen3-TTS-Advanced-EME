"""Test vLLM non-blocking async behavior.

Tests that vLLM generation calls don't block the event loop,
allowing other requests to be processed concurrently.

The tests below call the REAL ``AsyncVLLMClient.generate()`` — only the
HTTP transport is faked (an async ``post()`` that yields like real async
I/O). Earlier versions of this module replaced ``client.generate`` with a
local mock, so a production regression to a synchronous blocking call
(e.g. ``time.sleep`` instead of ``await``) left every test green.
"""

import asyncio
import contextlib
import time
import unittest
from unittest.mock import patch

from qwen3_tts.server.vllm_client import AsyncVLLMClient

# A single faked HTTP round-trip takes this long. Long enough that a
# heartbeat task can tick many times during a truly async call, and that
# N serialized calls blow the wall-clock bound.
_REQUEST_DELAY_SECS = 0.2
# Heartbeat cadence used to detect event-loop starvation.
_HEARTBEAT_INTERVAL_SECS = 0.01
# A genuinely async generate() yields the loop for the full request delay,
# producing ~_REQUEST_DELAY_SECS/_HEARTBEAT_INTERVAL_SECS ticks. A blocking
# generate() yields almost nothing. Half the expected count separates the
# two regimes with margin for scheduler jitter.
_MIN_HEARTBEATS_DURING_GENERATE = 10


class _FakeResponse:
    def __init__(self):
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": [{"audio": ""}]}


class _FakeAsyncHttpClient:
    """Stand-in for httpx.AsyncClient whose post() yields like async I/O."""

    def __init__(self, delay=_REQUEST_DELAY_SECS):
        self._delay = delay
        self.post_calls = []

    async def post(self, url, json=None):
        self.post_calls.append(json)
        await asyncio.sleep(self._delay)
        return _FakeResponse()


class TestVLLMNonBlocking(unittest.TestCase):
    def test_generate_is_non_blocking(self):
        """A heartbeat task must keep ticking while the REAL generate() runs.

        If generate() regresses to a synchronous blocking call (e.g.
        time.sleep instead of await), the heartbeat starves and this fails.
        """

        async def scenario():
            client = AsyncVLLMClient(base_url="http://localhost:8100")
            client._client = _FakeAsyncHttpClient()

            ticks = {"count": 0}

            async def heartbeat():
                while True:
                    ticks["count"] += 1
                    await asyncio.sleep(_HEARTBEAT_INTERVAL_SECS)

            beat = asyncio.create_task(heartbeat())
            try:
                with patch.object(
                    AsyncVLLMClient,
                    "_decode_audio",
                    staticmethod(lambda audio_base64: (24000, None)),
                ):
                    await client.generate(text="Hello")
            finally:
                beat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await beat
            return ticks["count"]

        ticks = asyncio.run(scenario())
        self.assertGreaterEqual(
            ticks,
            _MIN_HEARTBEATS_DURING_GENERATE,
            "Event loop was starved during generate() — "
            f"only {ticks} heartbeat ticks ran in "
            f"{_REQUEST_DELAY_SECS}s (blocking call?)",
        )

    def test_circuit_breaker_prevents_blocking(self):
        """The circuit breaker must start CLOSED and expose its state."""

        async def scenario():
            client = AsyncVLLMClient(base_url="http://localhost:8100")
            state = await asyncio.create_task(
                asyncio.to_thread(lambda: client.circuit_breaker.state)
            )
            return state

        state = asyncio.run(scenario())
        self.assertEqual(state, "CLOSED")

    def test_multiple_concurrent_requests(self):
        """Ten concurrent REAL generate() calls must overlap, not serialize.

        Each faked round-trip takes _REQUEST_DELAY_SECS; if the calls run
        concurrently the batch finishes in ~one delay, while a serialized
        (blocking) implementation takes >= 10x that.
        """
        num_requests = 10

        async def scenario():
            client = AsyncVLLMClient(base_url="http://localhost:8100")
            fake = _FakeAsyncHttpClient()
            client._client = fake
            with patch.object(
                AsyncVLLMClient,
                "_decode_audio",
                staticmethod(lambda audio_base64: (24000, None)),
            ):
                start = time.monotonic()
                results = await asyncio.gather(
                    *[client.generate(text=f"Request {i}") for i in range(num_requests)]
                )
                elapsed = time.monotonic() - start
            return results, elapsed, len(fake.post_calls)

        results, elapsed, post_calls = asyncio.run(scenario())
        self.assertEqual(len(results), num_requests)
        self.assertEqual(post_calls, num_requests, "every request must hit the server")
        self.assertLess(
            elapsed,
            num_requests * _REQUEST_DELAY_SECS,
            f"{num_requests} requests took {elapsed:.2f}s — they were serialized, "
            "not run concurrently",
        )


if __name__ == "__main__":
    unittest.main()
