#!/usr/bin/env python3
"""E2E: /generate-stream wire format + X-Seed header against the LIVE server.

Closes 4D gap 3: the stream wire format (length-prefixed frames, terminal
sentinel error frame) and the ``X-Seed`` header were unit-tested only, against
synthetic bytes — nothing ever parsed a REAL ``/generate-stream`` response.
These tests drive a live generation and parse the response through the ONE
canonical parser, ``core/stream_protocol.iter_stream_chunks`` (never re-fork
it — the old duplicate drifted and decoded JSON error payloads as float32).

Contract under test (``app_generation.py`` + ``core/stream_protocol.py``):

  POST /generate-stream (Bearer auth, JSON body, clone mode needs prompt_file)
    response headers include ``X-Seed``: the seed ACTUALLY used — the
    caller-supplied one verbatim, or the server-generated one when the request
    carries none (seed is kept out of the cache key and echoed on hits).
  body = ``[sample_rate:4 LE u32][length:4 LE u32][float32 LE audio]`` frames,
    block boundaries irrelevant; ``sample_rate == 0`` is the terminal ERROR
    sentinel (JSON payload) and must never appear on a happy-path stream.

The sentinel frame itself stays unit-covered (``tests/test_stream_error_frame.py``)
by design: it fires only on a genuine mid-stream inference failure, which
cannot be triggered against a live healthy server without fault injection.
Cancel, by contrast, truncates the stream cleanly (no sentinel) — that shape
is pinned in ``tests/test_streaming_cancel.py``.

Read timeout is BETWEEN chunks (``requests`` with ``stream=True``), not total
time — the documented semantics of both streaming clients.

Prerequisites: live server on :5123 with clone loaded (auto-load only applies
to /generate's handler — the streaming handler requires the slot non-empty
too, but /generate's on-demand path does NOT run here).

Usage:  pytest tests/test_e2e_generate_stream.py -m e2e
"""

import json
import os
import unittest
import urllib.request

try:
    import pytest
    pytestmark = pytest.mark.e2e
except ImportError:
    pass

SERVER_URL = "http://127.0.0.1:5123"
TOKEN_FILE = os.path.expanduser("~/.config/qwen3-tts/.voice_server_token")

# requests read timeout = inter-chunk GAP (stream=True), not total runtime.
CONNECT_TIMEOUT = 10
INTER_CHUNK_TIMEOUT = 180

_TEXT = "Hello from the /generate-stream end-to-end test."


def _read_token():
    with open(TOKEN_FILE) as f:
        return f.read().strip()


def _pick_prompt():
    """First non-'e2e' prompt, bare-named (server lists suffixed names)."""
    req = urllib.request.Request(
        f"{SERVER_URL}/prompts",
        headers={"Authorization": f"Bearer {_read_token()}"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        listed = json.loads(resp.read().decode())["prompts"]
    for name in listed:
        if not name.startswith("e2e"):
            return name.removesuffix(".wav").removesuffix(".pt")
    raise unittest.SkipTest("no non-e2e voice prompt available for clone mode")


class TestE2EGenerateStream(unittest.TestCase):
    """Live /generate-stream: canonical frame parse + X-Seed echo contract."""

    @classmethod
    def setUpClass(cls):
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3) as r:
                health = json.loads(r.read().decode())
        except Exception:
            raise unittest.SkipTest("TTS server not running on port 5123")
        if not health.get("clone_model_loaded"):
            raise unittest.SkipTest("clone model not loaded — load it before running")
        cls.token = _read_token()
        cls.prompt = _pick_prompt()

    # --- helpers ---------------------------------------------------------

    def _stream(self, body):
        """POST /generate-stream and return (response, header_seed_str)."""
        import requests

        resp = requests.post(
            f"{SERVER_URL}/generate-stream",
            json=body,
            headers={"Authorization": f"Bearer {self.token}"},
            stream=True,
            timeout=(CONNECT_TIMEOUT, INTER_CHUNK_TIMEOUT),
        )
        self.assertEqual(resp.status_code, 200, resp.text[:200])
        self.assertIn("X-Seed", resp.headers, "X-Seed header missing")
        return resp, resp.headers["X-Seed"]

    def _parse_chunks(self, resp):
        """Run the live response through the canonical parser."""
        from qwen3_tts.core.stream_protocol import iter_stream_chunks

        chunks = list(iter_stream_chunks(resp.iter_content(chunk_size=4096)))
        resp.close()
        return chunks

    # --- tests -----------------------------------------------------------

    def test_01_supplied_seed_is_echoed_in_x_seed_header(self):
        """Client seed → X-Seed echoes it verbatim; frames parse as float32."""
        seed = 4321
        resp, header_seed = self._stream({
            "text": _TEXT,
            "mode": "clone",
            "prompt_file": self.prompt,
            "seed": seed,
        })
        self.assertEqual(header_seed, str(seed))

        chunks = self._parse_chunks(resp)
        self.assertGreaterEqual(len(chunks), 1, "no audio frames parsed")
        total_samples = 0
        for samples, sr in chunks:
            self.assertGreater(sr, 0, "sample_rate 0 is the error sentinel — "
                                      "must never appear on a happy-path stream")
            self.assertLessEqual(sr, 192_000, f"implausible sample rate: {sr}")
            self.assertEqual(samples.dtype.name, "float32")
            total_samples += len(samples)
        self.assertGreater(total_samples, 0, "parsed zero audio samples")

    def test_02_omitted_seed_server_generates_one_and_reports_it(self):
        """No seed in the request → server generates one and reports it via X-Seed."""
        resp, header_seed = self._stream({
            "text": _TEXT,
            "mode": "clone",
            "prompt_file": self.prompt,
        })
        self.assertTrue(header_seed.isdigit(),
                        f"X-Seed must be a non-negative integer, got {header_seed!r}")

        chunks = self._parse_chunks(resp)
        self.assertGreaterEqual(len(chunks), 1, "no audio frames parsed")
        for samples, sr in chunks:
            self.assertGreater(sr, 0)
            self.assertEqual(samples.dtype.name, "float32")


if __name__ == "__main__":
    unittest.main()
