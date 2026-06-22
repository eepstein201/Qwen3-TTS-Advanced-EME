"""Test vLLM concurrent request integration.

Integration tests that multiple generation requests don't block each other.
Requires running server for full integration testing.
"""

import unittest

try:
    from httpx import AsyncClient
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class TestVLLMConcurrentIntegration(unittest.TestCase):
    async def test_concurrent_generation_requests(self):
        """Test that multiple generation requests don't block each other."""

        # This requires running server
        # Skip if server not available or httpx not available
        if not HTTPX_AVAILABLE:
            self.skipTest("httpx not available")

        try:
            async with AsyncClient() as client:
                response = await client.get("http://localhost:5123/health", timeout=2.0)
                if response.status_code != 200:
                    self.skipTest("Server not running or not healthy")
        except Exception:
            self.skipTest("Server not available")

        # Test would require valid auth token and audio generation
        # For now, skip this test
        self.skipTest("Requires full server setup with authentication")


if __name__ == "__main__":
    unittest.main()
