"""GEN-2: Pydantic response contracts for server routes.

Each test hits a route through the FastAPI TestClient and validates the JSON
body against the route's ``response_model``. ``response_model`` FILTERS any
field the model does not declare, so these tests exist to catch a model that
silently drops a field the handler emits (the assertion on key fields plus
``model_validate`` catches both shape drift and filtering).

Deliberately untyped routes (see validation.py GEN-2 block): /generate-stream
and /ws (binary/WebSocket frames), /preview-prompt and /shutdown (non-JSON
Response objects).

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_response_contracts.py -v
"""

import os
import unittest
from unittest.mock import MagicMock, patch

try:
    from fastapi.testclient import TestClient

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi")


@_skip
class _ContractTestBase(unittest.TestCase):
    """Shared TestClient harness with app.state snapshot/restore."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        from tests.conftest import _init_app_state

        _init_app_state(app, auth_token="test_token")
        app.state.models_loaded.set()  # simulate ready
        app.state.server_config = {
            "security": {"max_text_length": 50000, "max_batch_size": 20},
            "auto_shutdown_minutes": 0,
        }
        cls.app = app
        cls.client = TestClient(app, raise_server_exceptions=False)
        cls.auth = {"Authorization": "Bearer test_token"}

    def _reset_limiters(self):
        """Reset in-process rate limiters (house pattern; unittest ignores the
        pytest autouse fixture in conftest.py)."""
        for attr in ("limiter", "limiter_global", "limiter_hybrid", "limiter_ip", "limiter_token"):
            limiter = getattr(self.app.state, attr, None)
            if limiter is not None and hasattr(limiter, "reset"):
                limiter.reset()


@_skip
class TestPublicStatusContracts(_ContractTestBase):
    """G1: /ready, /queue-status, /generation-status, /stats, /models."""

    def test_ready_matches_contract(self):
        from qwen3_tts.server.validation import ReadyResponse

        resp = self.client.get("/ready")
        self.assertEqual(resp.status_code, 200, resp.text)
        model = ReadyResponse.model_validate(resp.json())
        self.assertEqual(model.status, "ready")

    def test_queue_status_matches_contract(self):
        from qwen3_tts.server.validation import QueueStatusResponse

        resp = self.client.get("/queue-status")
        self.assertEqual(resp.status_code, 200, resp.text)
        model = QueueStatusResponse.model_validate(resp.json())
        self.assertIsInstance(model.queue_length, int)
        self.assertIsInstance(model.active, bool)

    def test_generation_status_matches_contract(self):
        from qwen3_tts.server.validation import GenerationStatusResponse

        resp = self.client.get("/generation-status")
        self.assertEqual(resp.status_code, 200, resp.text)
        model = GenerationStatusResponse.model_validate(resp.json())
        self.assertFalse(model.active)
        self.assertFalse(model.cancelled)

    def test_generation_status_active_matches_contract(self):
        import time as _time

        from qwen3_tts.server.validation import GenerationStatusResponse

        gs = self.app.state.generation_state
        gs["active"] = True
        gs["start_time"] = _time.time() - 5
        try:
            resp = self.client.get("/generation-status")
            model = GenerationStatusResponse.model_validate(resp.json())
            self.assertTrue(model.active)
            self.assertIsNotNone(model.elapsed_sec)
        finally:
            gs["active"] = False
            gs["start_time"] = 0.0

    def test_stats_matches_contract(self):
        from qwen3_tts.server.validation import StatsResponse

        resp = self.client.get("/stats", headers=self.auth)
        self.assertEqual(resp.status_code, 200, resp.text)
        model = StatsResponse.model_validate(resp.json())
        self.assertEqual(model.status, "ok")
        self.assertFalse(model.generation_health.degraded)
        # Backend-conditional fields stay present-but-null on the other backend.
        self.assertIn("backend", resp.json())

    def test_models_matches_contract(self):
        from qwen3_tts.server.validation import ModelsResponse

        resp = self.client.get("/models", headers=self.auth)
        self.assertEqual(resp.status_code, 200, resp.text)
        model = ModelsResponse.model_validate(resp.json())
        self.assertIn("clone", model.models)
        entry = model.models["clone"]
        # Key fields must survive response_model filtering (the silent-data-loss
        # failure mode these tests exist to catch).
        for field in ("loaded", "loading", "description", "memory_mb", "repo_id",
                      "load_at_startup", "load_time_sec"):
            self.assertIn(field, resp.json()["models"]["clone"])
        self.assertFalse(entry.loaded)


if __name__ == "__main__":
    unittest.main()
