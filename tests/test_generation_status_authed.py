"""Task 1.0 (Step 7A): auth-gated progress details on /generation-status.

The public payload must stay byte-identical: unauthenticated callers never see
``batch_total``/``chunk_total``/``progress_pct``/``eta_sec`` (they reveal the
in-flight request's size). A caller carrying the server token gets them.
A wrong token must behave exactly like an absent one.

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_generation_status_authed.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

try:
    from fastapi.testclient import TestClient

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi")

_PUBLIC_IDLE_KEYS = {"active", "batch_index", "chunk_index", "cancelled"}
_AUTHED_KEYS = {"batch_total", "chunk_total", "progress_pct", "eta_sec"}
_NOW = 1_000_000.0


class TestChunkProgressPct(unittest.TestCase):
    def _pct(self, completed, total):
        from qwen3_tts.server.app_lifespan import chunk_progress_pct

        return chunk_progress_pct(completed, total)

    def test_unknown_total_is_none(self):
        self.assertIsNone(self._pct(0, 0))
        self.assertIsNone(self._pct(3, 0))

    def test_single_chunk_is_none(self):
        self.assertIsNone(self._pct(0, 1))

    def test_fraction_of_completed_chunks(self):
        self.assertEqual(self._pct(0, 4), 0.0)
        self.assertEqual(self._pct(1, 4), 25.0)

    def test_never_reaches_100(self):
        # Completion is signalled by active=False; a saturated bar would race it.
        for completed in (4, 5, 100):
            self.assertLess(self._pct(completed, 4), 100.0)


class TestEstimateEtaSec(unittest.TestCase):
    def _eta(self, **state):
        from qwen3_tts.server.app_lifespan import estimate_eta_sec

        gen_state = {"start_time": _NOW - 10.0, "chunk_index": 0, "chunk_total": 0}
        gen_state.update(state)
        return estimate_eta_sec(gen_state, now=_NOW)

    def test_zero_completed_is_none(self):
        self.assertIsNone(self._eta(chunk_index=0, chunk_total=5))

    def test_unknown_total_is_none(self):
        self.assertIsNone(self._eta(chunk_index=2, chunk_total=0))

    def test_known_rate(self):
        # 2 chunks in 10 s -> 5 s/chunk, 3 chunks left -> 15 s.
        self.assertEqual(self._eta(chunk_index=2, chunk_total=5), 15.0)


@_skip
class TestGenerationStatusAuthGating(unittest.TestCase):
    def setUp(self):
        from qwen3_tts.server.app import app
        from tests.conftest import _init_app_state, _restore_app_state, _save_app_state

        original = _save_app_state(app)
        self.addCleanup(_restore_app_state, app, original)
        _init_app_state(app, auth_token="test_token")
        app.state.models_loaded.set()
        self.app = app
        self.client = TestClient(app, raise_server_exceptions=False)
        for attr in (
            "limiter",
            "limiter_global",
            "limiter_hybrid",
            "limiter_ip",
            "limiter_token",
        ):
            limiter = getattr(app.state, attr, None)
            if limiter is not None and hasattr(limiter, "reset"):
                limiter.reset()

    def _set_active(self, chunk_index=2, chunk_total=5, batch_total=3):
        self.app.state.generation_state.update(
            {
                "active": True,
                "start_time": _NOW - 10.0,
                "batch_index": 1,
                "batch_total": batch_total,
                "chunk_index": chunk_index,
                "chunk_total": chunk_total,
            }
        )

    def _get(self, headers=None):
        with patch("qwen3_tts.server.app.time.time", return_value=_NOW):
            resp = self.client.get("/generation-status", headers=headers or {})
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _authed(self):
        return self._get({"Authorization": "Bearer test_token"})

    def test_unauthed_idle_has_exactly_the_public_keys(self):
        self.assertEqual(set(self._get()), _PUBLIC_IDLE_KEYS)

    def test_unauthed_active_adds_only_elapsed(self):
        self._set_active()
        body = self._get()
        self.assertEqual(set(body), _PUBLIC_IDLE_KEYS | {"elapsed_sec"})
        self.assertEqual(body["elapsed_sec"], 10.0)

    def test_authed_active_adds_the_four_details(self):
        self._set_active(chunk_index=2, chunk_total=5, batch_total=3)
        body = self._authed()
        self.assertEqual(set(body), _PUBLIC_IDLE_KEYS | {"elapsed_sec"} | _AUTHED_KEYS)
        self.assertEqual(body["batch_total"], 3)
        self.assertEqual(body["chunk_total"], 5)
        self.assertEqual(body["progress_pct"], 40.0)
        self.assertEqual(body["eta_sec"], 15.0)

    def test_authed_single_chunk_omits_percent_and_eta(self):
        self._set_active(chunk_index=0, chunk_total=1)
        body = self._authed()
        self.assertEqual(body["chunk_total"], 1)
        self.assertNotIn("progress_pct", body)
        self.assertNotIn("eta_sec", body)

    def test_authed_progress_never_reaches_100(self):
        self._set_active(chunk_index=5, chunk_total=5)
        self.assertLess(self._authed()["progress_pct"], 100.0)

    def test_authed_idle_carries_totals_but_no_estimates(self):
        body = self._authed()
        self.assertEqual(set(body), _PUBLIC_IDLE_KEYS | {"batch_total", "chunk_total"})

    def test_wrong_token_is_identical_to_absent_token(self):
        self._set_active()
        absent = self._get()
        wrong = self._get({"Authorization": "Bearer not-the-token"})
        self.assertEqual(wrong, absent)

    def test_wrong_and_non_ascii_tokens_never_log_or_raise(self):
        self._set_active()
        with self.assertNoLogs(level="WARNING"):
            self._get({"Authorization": "Bearer not-the-token"})
        # httpx refuses non-ASCII header values, so drive the predicate directly.
        from qwen3_tts.server.app import request_is_authed

        request = MagicMock()
        request.headers = {"Authorization": "Bearer töken\udcff"}
        request.app.state.auth_token = "test_token"
        self.assertFalse(request_is_authed(request))


if __name__ == "__main__":
    unittest.main()
