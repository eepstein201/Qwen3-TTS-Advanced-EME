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


class _SeqFakeClient:
    """Async post() stand-in returning a scripted sequence.

    Items are either httpx.Response objects (returned to generate(), whose
    raise_for_status() then behaves per the status code) or exceptions
    (raised directly).
    """

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    async def post(self, url, json=None):
        self.calls += 1
        item = self._outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def get(self, url, timeout=None):
        return self._outcomes.pop(0)


def _response(status):
    import httpx

    return httpx.Response(
        status, request=httpx.Request("POST", "http://localhost:8100/x")
    )


def _ok_response():
    resp = _response(200)
    resp.json = lambda: {"data": [{"audio": ""}]}
    return resp


class TestCircuitBreakerStates(unittest.TestCase):
    """4B.3 item 13: circuit-breaker state transitions (missed 99-103,
    108, 120, 132)."""

    def test_open_transitions_to_half_open_after_cooldown(self):
        from qwen3_tts.server.vllm_client import CircuitBreaker

        async def scenario():
            cb = CircuitBreaker(failure_threshold=2, cooldown_secs=60)
            cb._state = "OPEN"
            cb._last_failure_time = time.time() - 120
            async with cb:
                return cb._state

        self.assertEqual(asyncio.run(scenario()), "HALF_OPEN")

    def test_open_within_cooldown_raises(self):
        from qwen3_tts.server.vllm_client import CircuitBreaker

        async def scenario():
            cb = CircuitBreaker(failure_threshold=2, cooldown_secs=60)
            cb._state = "OPEN"
            cb._last_failure_time = time.time()
            async with cb:
                return cb._state

        with self.assertRaisesRegex(RuntimeError, "OPEN"):
            asyncio.run(scenario())

    def test_half_open_success_closes_circuit(self):
        from qwen3_tts.server.vllm_client import CircuitBreaker

        async def scenario():
            cb = CircuitBreaker(failure_threshold=2)
            cb._state = "HALF_OPEN"
            async with cb:
                pass
            return cb._state

        self.assertEqual(asyncio.run(scenario()), "CLOSED")

    def test_failures_reaching_threshold_trip_open(self):
        from qwen3_tts.server.vllm_client import CircuitBreaker

        async def scenario():
            cb = CircuitBreaker(failure_threshold=2)
            for _ in range(2):
                try:
                    async with cb:
                        raise ValueError("boom")
                except ValueError:
                    pass
            return cb._state, cb._failure_count

        state, count = asyncio.run(scenario())
        self.assertEqual(state, "OPEN")
        self.assertEqual(count, 2)


class TestVLLMClientLifecycle(unittest.TestCase):
    """4B.3 item 13: close(), _decode_audio, health_check, circuit_state,
    and generate() payload/error arms (missed 196-198, 203-207, 256, 259,
    283-294, 304-305, 320, 332)."""

    def test_close_awaits_aclose_and_clears_client(self):
        from unittest.mock import AsyncMock, MagicMock

        from qwen3_tts.server.vllm_client import AsyncVLLMClient

        client = AsyncVLLMClient(base_url="http://localhost:8100")
        fake = MagicMock()
        fake.aclose = AsyncMock()
        client._client = fake
        asyncio.run(client.close())
        fake.aclose.assert_awaited_once()
        self.assertIsNone(client._client)

    def test_close_with_no_client_is_a_noop(self):
        from qwen3_tts.server.vllm_client import AsyncVLLMClient

        client = AsyncVLLMClient(base_url="http://localhost:8100")
        asyncio.run(client.close())
        self.assertIsNone(client._client)

    def test_decode_audio_decodes_base64_via_soundfile(self):
        import base64 as b64
        from unittest.mock import MagicMock

        from qwen3_tts.server.vllm_client import AsyncVLLMClient

        fake_sf = MagicMock()
        fake_sf.read.return_value = ("audio_sentinel", 24000)
        payload = b64.b64encode(b"RIFF....").decode()
        import sys

        with patch.dict(sys.modules, {"soundfile": fake_sf}):
            sr, audio = AsyncVLLMClient._decode_audio(payload)
        self.assertEqual(sr, 24000)
        self.assertEqual(audio, "audio_sentinel")
        (stream,), _ = fake_sf.read.call_args
        self.assertEqual(stream.getvalue(), b"RIFF....")

    def test_circuit_state_property_exposes_breaker_state(self):
        from qwen3_tts.server.vllm_client import AsyncVLLMClient

        client = AsyncVLLMClient(base_url="http://localhost:8100")
        self.assertEqual(client.circuit_state, "CLOSED")

    def test_health_check_true_on_200(self):
        from qwen3_tts.server.vllm_client import AsyncVLLMClient

        client = AsyncVLLMClient(base_url="http://localhost:8100")
        client._client = _SeqFakeClient([_response(200)])
        self.assertTrue(asyncio.run(client.health_check()))

    def test_health_check_false_on_non_200(self):
        from qwen3_tts.server.vllm_client import AsyncVLLMClient

        client = AsyncVLLMClient(base_url="http://localhost:8100")
        client._client = _SeqFakeClient([_response(503)])
        self.assertFalse(asyncio.run(client.health_check()))


class TestVLLMGeneratePayloadAndRetries(unittest.TestCase):
    """4B.3 item 13: generate() request-payload arms and retry/raise
    behavior (missed 256, 259, 283-294, 304-305)."""

    def _client_with(self, fake):
        from qwen3_tts.server.vllm_client import AsyncVLLMClient

        client = AsyncVLLMClient(base_url="http://localhost:8100")
        client._client = fake
        return client

    def _decode_patch(self):
        return patch.object(
            AsyncVLLMClient,
            "_decode_audio",
            staticmethod(lambda audio_base64: (24000, None)),
        )

    def _sleep_patch(self):
        from unittest.mock import AsyncMock

        return patch(
            "qwen3_tts.server.vllm_client.asyncio.sleep", new_callable=AsyncMock
        )

    def test_design_and_custom_payload_fields_reach_request(self):
        async def scenario():
            fake = _FakeAsyncHttpClient(delay=0.0)
            client = self._client_with(fake)
            with self._decode_patch():
                await client.generate(
                    text="hi", mode="design", voice_description="deep voice"
                )
                await client.generate(text="hi", mode="custom", speaker="ryan")
            return fake.post_calls

        calls = asyncio.run(scenario())
        self.assertEqual(calls[0]["input"]["voice_description"], "deep voice")
        self.assertEqual(calls[1]["input"]["speaker"], "ryan")

    def test_5xx_retries_then_succeeds(self):
        async def scenario():
            fake = _SeqFakeClient([_response(500), _ok_response()])
            client = self._client_with(fake)
            with self._decode_patch(), self._sleep_patch():
                result = await client.generate(text="hi")
            return result, fake.calls

        result, calls = asyncio.run(scenario())
        self.assertEqual(result, (24000, None))
        self.assertEqual(calls, 2)

    def test_4xx_raises_without_retry(self):
        async def scenario():
            fake = _SeqFakeClient([_response(404)])
            client = self._client_with(fake)
            with self._decode_patch(), self._sleep_patch() as mock_sleep:
                with self.assertRaisesRegex(RuntimeError, "vLLM generation failed"):
                    await client.generate(text="hi")
            return fake.calls, mock_sleep

        calls, mock_sleep = asyncio.run(scenario())
        self.assertEqual(calls, 1)
        mock_sleep.assert_not_awaited()

    def test_5xx_exhausts_retries_and_raises(self):
        async def scenario():
            fake = _SeqFakeClient([_response(500)] * 3)
            client = self._client_with(fake)
            with self._decode_patch(), self._sleep_patch():
                await client.generate(text="hi")
            return fake.calls

        with self.assertRaisesRegex(RuntimeError, "vLLM generation failed"):
            asyncio.run(scenario())

    def test_clone_temp_file_unlink_failure_warns_but_returns(self):
        from unittest.mock import MagicMock

        async def scenario():
            fake = _SeqFakeClient([_ok_response()])
            client = self._client_with(fake)
            fake_tmp = MagicMock()
            fake_tmp.__enter__.return_value = fake_tmp
            fake_tmp.name = "/nonexistent/fake_prompt.wav"
            logs_cm = self.assertLogs("tts.server.vllm_client", level="WARNING")
            with (
                self._decode_patch(),
                patch(
                    "tempfile.NamedTemporaryFile",
                    return_value=fake_tmp,
                ),
                patch(
                    "qwen3_tts.server.vllm_client.os.path.exists",
                    return_value=True,
                ),
                patch(
                    "qwen3_tts.server.vllm_client.os.unlink",
                    side_effect=OSError("busy"),
                ),
                logs_cm as logs,
            ):
                result = await client.generate(
                    text="hi", mode="clone", prompt_audio=b"WAVDATA"
                )
            return result, fake_tmp, logs.output

        result, fake_tmp, log_lines = asyncio.run(scenario())
        self.assertEqual(result, (24000, None))
        fake_tmp.write.assert_called_once_with(b"WAVDATA")
        self.assertTrue(
            any(
                "Failed to remove temp prompt-audio file" in line for line in log_lines
            ),
            log_lines,
        )


if __name__ == "__main__":
    unittest.main()
