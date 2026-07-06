"""Tests for vLLM client temp-file cleanup (no leak on error or success).

The clone branch of AsyncVLLMClient.generate writes the reference audio to a
NamedTemporaryFile(delete=False). If that file is never unlinked it leaks. These
tests assert the temp file is removed even when the HTTP request raises.
"""

import asyncio
import os
import unittest

from qwen3_tts.server.vllm_client import AsyncVLLMClient


class _RaisingClient:
    """Fake httpx client whose post() captures the temp path then raises."""

    def __init__(self, captured: dict):
        self._captured = captured

    async def post(self, url, json=None):
        # Record the temp path the request references; it should exist right now.
        self._captured["path"] = json["input"]["prompt_audio"]
        self._captured["existed_during_request"] = os.path.exists(
            self._captured["path"]
        )
        raise RuntimeError("simulated vLLM failure")


class TestVLLMTempFileCleanup(unittest.TestCase):
    def test_temp_file_removed_when_request_raises(self):
        """Temp prompt-audio file must be unlinked even if the request raises."""

        async def run():
            client = AsyncVLLMClient(base_url="http://localhost:8100")
            captured: dict = {}
            client._client = _RaisingClient(captured)

            with self.assertRaises(Exception):
                await client.generate(
                    text="hello",
                    mode="clone",
                    prompt_audio=b"RIFFfakeaudio",
                )

            # The file was created and referenced during the request...
            self.assertIn("path", captured)
            self.assertTrue(captured["existed_during_request"])
            # ...but must be gone once generate() returns/raises.
            self.assertFalse(
                os.path.exists(captured["path"]),
                "Temp prompt_audio file leaked after request failure",
            )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
