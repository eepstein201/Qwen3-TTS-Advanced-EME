#!/usr/bin/env python3
"""TDD tests for async model loading in TTS server.

Tests that:
  - _models_loaded is a threading.Event at module scope
  - /health returns 503 {"status":"loading"} while models are loading
  - /health returns 200 {"status":"ok"} after _models_loaded is set

No GPU, models, or running server required. Tests mock model state.

Run: python -m pytest tests/test_server_startup.py -v
"""
import threading
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import soundfile  # noqa: F401
    import flask  # noqa: F401
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires soundfile and flask")


@_skip
class TestModelsLoadedEvent(unittest.TestCase):
    """_models_loaded must be a threading.Event at module scope."""

    def test_event_exists_at_module_scope(self):
        import qwen3_tts.server.app as srv
        self.assertTrue(hasattr(srv, "_models_loaded"),
                        "_models_loaded must exist as a module-level attribute")

    def test_event_is_threading_event(self):
        import qwen3_tts.server.app as srv
        self.assertIsInstance(srv._models_loaded, type(threading.Event()),
                              "_models_loaded must be a threading.Event instance")

    def test_event_has_is_set_method(self):
        import qwen3_tts.server.app as srv
        self.assertTrue(callable(srv._models_loaded.is_set))


@_skip
class TestHealthEndpointDuringLoading(unittest.TestCase):
    """Health endpoint must reflect loading state via _models_loaded."""

    @classmethod
    def setUpClass(cls):
        import qwen3_tts.server.app as srv
        # Minimal config so app doesn't crash
        srv.auth_token = "test_token_startup"  # nosec B105
        srv.server_config = {
            "security": {"max_text_length": 10000, "max_batch_size": 20},
            "auto_shutdown_minutes": 0,
        }
        srv.clone_model = None
        srv.design_model = None
        srv.custom_model = None
        # Clear event so tests start in "loading" state
        srv._models_loaded.clear()
        cls.srv = srv
        srv.app.testing = True
        cls.client = srv.app.test_client()

    def test_health_returns_503_while_loading(self):
        """/health must return 503 with status='loading' when _models_loaded not set."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 503,
                         f"Expected 503, got {resp.status_code}: {resp.get_json()}")
        data = resp.get_json()
        self.assertEqual(data["status"], "loading")

    def test_health_returns_200_after_loading(self):
        """/health must return 200 with status='ok' once _models_loaded is set."""
        self.srv._models_loaded.set()
        try:
            resp = self.client.get("/health")
            self.assertEqual(resp.status_code, 200,
                             f"Expected 200, got {resp.status_code}: {resp.get_json()}")
            data = resp.get_json()
            self.assertEqual(data["status"], "ok")
        finally:
            self.srv._models_loaded.clear()

    def test_health_loading_response_is_minimal(self):
        """503 loading response has status key but not model-specific fields."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 503)
        data = resp.get_json()
        self.assertIn("status", data)
        self.assertNotIn("clone_model_loaded", data)
        self.assertNotIn("backend", data)

    def test_health_is_accessible_without_auth(self):
        """/health must be reachable without Bearer token (public endpoint)."""
        resp = self.client.get("/health")
        # 503 is fine — the point is it does NOT return 401
        self.assertNotEqual(resp.status_code, 401)

    @classmethod
    def tearDownClass(cls):
        # Leave event set so later tests that need a "ready" server work
        cls.srv._models_loaded.set()


if __name__ == "__main__":
    unittest.main()
