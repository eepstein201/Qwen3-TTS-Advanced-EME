#!/usr/bin/env python3
"""E2E: /cancel-generation contract against the LIVE server.

Closes 4D gap 4: the playwright module's test_06 clicks the UI Stop button but
tolerates every outcome, and the API contract CLAUDE.md warns about — top-level
``cancelled: true`` with a ``results`` list possibly SHORTER than ``texts`` —
had no E2E guard (Step 1A proved the truncation shape only via a one-off curl).

Contract under test (``app.py`` ``cancel_generation`` + ``app_generation.py``
batch loop):

  POST /cancel-generation, nothing active  → {"status": "no_active_generation"}
  POST /cancel-generation, batch mid-flight → {"status": "cancellation_requested",
                                               "generation_id": …}
  the interrupted POST /generate returns 200 with ``cancelled: true`` and
  ``results`` holding only the COMPLETED items — never indexed blindly: a
  caller that assumes len(results) == len(texts) silently processes fewer
  items (generate_server.py raises TTSGenericError on exactly that mismatch).

Each item text is ~2 chunks (700 chars vs the 500 default) so a cancel
observed at item 2's start leaves item 3 (~15 s warm) unable to complete —
``len(results) < len(texts)`` is then guaranteed, not racy.

Runs against the STANDARD server profile (unlike test_e2e_queueing.py, which
needs TTS_DISABLE_RATE_LIMITING=1): one /generate + one /cancel-generation +
public /generation-status polls stay far under the limits.

Prerequisites: live, idle server on :5123 with clone loaded.

Usage:  pytest tests/test_e2e_cancel_generation.py -m e2e
"""

import json
import os
import threading
import time
import unittest
import urllib.request

try:
    import pytest

    pytestmark = pytest.mark.e2e
except ImportError:
    pass

SERVER_URL = "http://127.0.0.1:5123"
TOKEN_FILE = os.path.expanduser("~/.config/qwen3-tts/.voice_server_token")

_TEXT_BASES = [
    "The first item of a cancellable batch speaks a couple of sentences so it "
    "spans roughly two chunks of five hundred characters, ensuring the batch "
    "takes long enough that a cancel observed at the second item still leaves "
    "the third item unfinished. " * 2,
    "The second item of a cancellable batch speaks a couple of sentences so it "
    "spans roughly two chunks of five hundred characters, ensuring the batch "
    "takes long enough that a cancel observed at the second item still leaves "
    "the third item unfinished. " * 2,
    "The third item of a cancellable batch speaks a couple of sentences so it "
    "spans roughly two chunks of five hundred characters, ensuring the batch "
    "takes long enough that a cancel observed at the second item still leaves "
    "the third item unfinished. " * 2,
]


def _unique_texts():
    """Cache-cold texts: the generation cache keys on the exact text (the seed
    is deliberately outside the key), so constant texts made re-runs resolve
    every item as an instant cache hit — the batch finished before the poll
    ever saw ``active``, and a cache-hit item 3 could complete past the cancel,
    breaking the short-results assertion. A per-run stamp defeats both."""
    stamp = f"{int(time.time() * 1000) % 100_000_000:08d}"
    return [f"Run {stamp}, item {i + 1}. {base}" for i, base in enumerate(_TEXT_BASES)]


def _read_token():
    with open(TOKEN_FILE) as f:
        return f.read().strip()


def _request(method, endpoint, body=None, timeout=30):
    req = urllib.request.Request(
        f"{SERVER_URL}{endpoint}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {_read_token()}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode())


def _pick_prompt():
    _s, body = _request("GET", "/prompts")
    for name in body["prompts"]:
        if not name.startswith("e2e"):
            return name.removesuffix(".wav").removesuffix(".pt")
    raise unittest.SkipTest("no non-e2e voice prompt available for clone mode")


def _public_generation_status():
    with urllib.request.urlopen(f"{SERVER_URL}/generation-status", timeout=5) as r:
        return json.loads(r.read().decode())


class TestE2ECancelGeneration(unittest.TestCase):
    """Live /cancel-generation: idle shape + mid-batch truncation contract."""

    @classmethod
    def setUpClass(cls):
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3) as r:
                health = json.loads(r.read().decode())
        except Exception as e:
            raise unittest.SkipTest("TTS server not running on port 5123") from e
        if not health.get("clone_model_loaded"):
            raise unittest.SkipTest("clone model not loaded — load it before running")
        cls.prompt = _pick_prompt()

    def test_01_idle_cancel_reports_no_active_generation(self):
        """Nothing running → the explicit idle shape, not an error."""
        status, body = _request("POST", "/cancel-generation")
        self.assertEqual(status, 200)
        self.assertEqual(
            body.get("status"),
            "no_active_generation",
            f"server must be idle for this test; got {body}",
        )

    def test_02_batch_cancel_returns_short_results_with_flag(self):
        """Cancel mid-batch → 200, cancelled=true, results strictly shorter."""
        outcome = {}

        def _generate():
            import requests

            try:
                resp = requests.post(
                    f"{SERVER_URL}/generate",
                    json={
                        "texts": _unique_texts(),
                        "mode": "clone",
                        "prompt_file": self.prompt,
                    },
                    headers={"Authorization": f"Bearer {_read_token()}"},
                    timeout=600,
                )
                outcome["status"] = resp.status_code
                outcome["body"] = resp.json()
            except Exception as exc:  # surfaced via the join assert below
                outcome["error"] = str(exc)

        worker = threading.Thread(target=_generate)
        worker.start()
        try:
            # Wait until item 2 has started (coarse public batch_index), then
            # cancel — item 3's ~15 s of remaining work guarantees truncation.
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                snap = _public_generation_status()
                if snap.get("active") and snap.get("batch_index", 0) >= 1:
                    break
                time.sleep(1.0)
            else:
                self.fail(
                    f"generation never reached batch item 2 (last status: {snap})"
                )

            status, body = _request("POST", "/cancel-generation")
            self.assertEqual(status, 200)
            self.assertEqual(body.get("status"), "cancellation_requested")
            self.assertIn("generation_id", body)
        finally:
            worker.join(timeout=240)
        self.assertFalse(
            worker.is_alive(), "generate thread still running after cancel"
        )
        self.assertNotIn("error", outcome, f"generate request failed: {outcome}")

        self.assertEqual(outcome["status"], 200)
        gen_body = outcome["body"]
        self.assertIs(
            gen_body.get("cancelled"),
            True,
            f"cancelled flag must be top-level true: {gen_body}",
        )
        results = gen_body.get("results")
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1, "item 1 completed before the cancel")
        self.assertLess(
            len(results),
            3,
            "results must be SHORTER than texts on a cancelled batch — "
            "a full-length results list means the cancel didn't take",
        )
        for item in results:
            self.assertIn(
                "audio_base64",
                item,
                "each completed result must carry audio (never a stub)",
            )


if __name__ == "__main__":
    unittest.main()
