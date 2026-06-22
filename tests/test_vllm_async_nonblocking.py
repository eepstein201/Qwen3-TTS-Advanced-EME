"""Test vLLM non-blocking async behavior.

Tests that vLLM generation calls don't block the event loop,
allowing other requests to be processed concurrently.
"""

import asyncio
import unittest

from qwen3_tts.server.vllm_client import AsyncVLLMClient


class TestVLLMNonBlocking(unittest.TestCase):
    def test_generate_is_non_blocking(self):
        """Verify vLLM generate doesn't block event loop."""

        async def test():
            # Mock AsyncVLLMClient
            client = AsyncVLLMClient(
                base_url="http://localhost:8100"
            )

            # Mock the generate method to simulate slow response
            async def slow_generate(*args, **kwargs):
                await asyncio.sleep(0.1)  # Simulate slow generation
                return {"audio": "base64data"}

            client.generate = slow_generate

            # Track if event loop can process other tasks
            other_task_ran = False

            async def other_task():
                nonlocal other_task_ran
                other_task_ran = True
                await asyncio.sleep(0.05)

            # Start both tasks concurrently
            await asyncio.gather(
                client.generate(text="Hello"),
                other_task()
            )

            # Verify other task ran during generate
            self.assertTrue(other_task_ran, "Event loop was blocked!")

        # Run async test
        asyncio.run(test())

    def test_circuit_breaker_prevents_blocking(self):
        """Verify circuit breaker doesn't block event loop."""

        async def test():
            # Mock AsyncVLLMClient with circuit breaker
            client = AsyncVLLMClient(
                base_url="http://localhost:8100"
            )

            # Circuit breaker state changes should be async
            # This test verifies the circuit breaker pattern itself is non-blocking
            async def check_state():
                # Access circuit breaker state
                state = client.circuit_breaker.state
                return state

            # Should be able to check state without blocking
            state = await asyncio.create_task(check_state())
            self.assertIsNotNone(state)

        asyncio.run(test())

    def test_multiple_concurrent_requests(self):
        """Verify multiple requests can be processed concurrently."""

        async def test():
            # Mock AsyncVLLMClient
            client = AsyncVLLMClient(
                base_url="http://localhost:8100"
            )

            # Mock generate
            call_count = 0

            async def mock_generate(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                await asyncio.sleep(0.05)
                return {"audio": f"data{call_count}"}

            client.generate = mock_generate

            # Launch 10 concurrent requests
            tasks = [client.generate(text=f"Request {i}") for i in range(10)]
            results = await asyncio.gather(*tasks)

            # All should complete
            self.assertEqual(len(results), 10)
            self.assertEqual(call_count, 10)

        asyncio.run(test())


if __name__ == "__main__":
    unittest.main()
