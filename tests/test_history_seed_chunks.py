"""Regression tests: history table 'seed' and 'chunks' columns must fill reliably.

Two root causes fixed here:

1. SEED — the server never returned the seed it used. Blank (random) seeds
   showed "-" in the history table and could not be reused. Fix: the server
   resolves an explicit seed (random when none supplied), applies it, and
   returns it in each result. The auto-generated seed is kept OUT of the
   generation-cache key so repeated identical requests still cache-hit.

2. CHUNKS — cache-hit result dicts omitted the "chunks" key, so a repeated
   (cached) generation reported 0 chunks. Fix: chunk count and seed are stored
   in the cache entry and echoed on cache hits.
"""

import inspect
import unittest
from unittest.mock import MagicMock, patch

from qwen3_tts.server import app_generation


class TestGenerateResultModel(unittest.TestCase):
    """GenerateResult must carry a seed field (else response_model strips it)."""

    def test_seed_field_present(self):
        from qwen3_tts.server.validation import GenerateResult

        r = GenerateResult(index=0, sample_rate=24000, seed=12345, chunks=3)
        self.assertEqual(r.seed, 12345)
        self.assertEqual(r.chunks, 3)

    def test_seed_defaults_none(self):
        from qwen3_tts.server.validation import GenerateResult

        r = GenerateResult(index=0, sample_rate=24000)
        self.assertIsNone(r.seed)


class TestResolveGenerationSeed(unittest.TestCase):
    """_resolve_generation_seed returns the user seed, or a random int when None."""

    def test_returns_user_seed_unchanged(self):
        self.assertEqual(app_generation._resolve_generation_seed(42), 42)

    def test_generates_int_when_none(self):
        seed = app_generation._resolve_generation_seed(None)
        self.assertIsInstance(seed, int)
        self.assertGreaterEqual(seed, 0)

    def test_generated_seed_within_int32(self):
        for _ in range(50):
            seed = app_generation._resolve_generation_seed(None)
            self.assertLessEqual(seed, 2**31 - 1)

    def test_zero_is_respected_not_treated_as_missing(self):
        # seed=0 is a valid explicit seed, must not be replaced with a random one.
        self.assertEqual(app_generation._resolve_generation_seed(0), 0)


class TestCacheEntryCarriesSeedAndChunks(unittest.TestCase):
    """Source-level: cache writes and cache-hit results include seed + chunks.

    Driving the full generation pipeline needs a real model, so this follows the
    source-inspection convention already used for app_generation
    (see tests/test_generation_offload.py).
    """

    def setUp(self):
        self.src = inspect.getsource(app_generation)

    def test_cache_hit_results_include_chunks(self):
        # Both pre-lock and post-lock cache-hit result dicts must carry chunks.
        self.assertGreaterEqual(
            self.src.count('"chunks": entry.get("chunks"'),
            2,
            "cache-hit result dicts should echo chunks from the cache entry",
        )

    def test_cache_hit_results_include_seed(self):
        self.assertGreaterEqual(
            self.src.count('"seed": entry.get("seed")'),
            2,
            "cache-hit result dicts should echo seed from the cache entry",
        )

    def test_cache_entry_write_stores_seed_and_chunks(self):
        self.assertIn('"chunks":', self.src)
        self.assertIn('"seed":', self.src)
        # The cache entry (keyed by main_file) must persist both.
        self.assertRegex(
            self.src,
            r"main_file.*\n(?:.*\n)*?.*\"seed\":",
        )


class TestStreamingPathReportsSeed(unittest.TestCase):
    """/generate-stream must resolve, apply, and report the seed via X-Seed."""

    def setUp(self):
        self.src = inspect.getsource(app_generation)

    def test_stream_resolves_seed(self):
        # handle_generate_stream must call the shared resolver.
        self.assertIn("_resolve_generation_seed(req.seed)", self.src)

    def test_stream_applies_seeded_params(self):
        # The streaming inference call must use seeded_params, not raw gen_params.
        self.assertIn("gen_params=seeded_params", self.src)

    def test_stream_sets_x_seed_header(self):
        self.assertIn('"X-Seed": str(used_seed)', self.src)

    def test_client_streaming_reads_x_seed(self):
        from qwen3_tts.server.client import generator

        gsrc = inspect.getsource(generator)
        self.assertIn('resp.headers.get("X-Seed")', gsrc)


class TestWebSocketPathReportsSeed(unittest.TestCase):
    """/ws must resolve, apply, and report the seed in the completion message."""

    def setUp(self):
        from qwen3_tts.server import websocket

        self.src = inspect.getsource(websocket)

    def test_ws_resolves_seed(self):
        self.assertIn("_resolve_generation_seed(data.get(\"seed\"))", self.src)

    def test_ws_applies_seed_to_gen_params(self):
        self.assertIn('"seed": used_seed', self.src)

    def test_ws_completion_reports_seed(self):
        # The "complete" message dict must include the seed.
        self.assertRegex(
            self.src,
            r'"status":\s*"complete",\s*\n\s*"chunks":\s*chunk_count,\s*\n\s*"seed":\s*used_seed',
        )


class TestClientCapturesSeed(unittest.TestCase):
    """TTSClient must expose last_seed / last_chunk_count and populate them."""

    def _make_client(self):
        from qwen3_tts.server.client import TTSClient

        return TTSClient()

    def test_last_seed_initialized(self):
        client = self._make_client()
        self.assertIsNone(client.last_seed)

    def test_last_chunk_count_initialized(self):
        client = self._make_client()
        self.assertEqual(client.last_chunk_count, 0)

    def test_generate_via_server_sets_last_seed(self):
        import numpy as np

        client = self._make_client()

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = {
            "results": [
                {
                    "index": 0,
                    "audio_base64": "",
                    "sample_rate": 24000,
                    "chunks": 4,
                    "seed": 987654,
                }
            ]
        }

        with patch.object(client._session, "post", return_value=fake_resp), patch(
            "soundfile.read", return_value=(np.zeros(10, dtype=np.float32), 24000)
        ), patch("base64.b64decode", return_value=b""):
            client._generate_via_server(
                text="hello",
                mode="clone",
                prompt="lsmith.pt",
                description=None,
                speaker=None,
                instruct=None,
                gen_params={},
            )

        self.assertEqual(client.last_seed, 987654)
        self.assertEqual(client.last_chunk_count, 4)


if __name__ == "__main__":
    unittest.main()
