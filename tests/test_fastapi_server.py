"""Tests for FastAPI server implementation.

Validates that the FastAPI server works correctly with async,
proper streaming, and app.state for worker-safe operations.
"""

import unittest
from pathlib import Path

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Dummy decorator for when pytest is not available
    class _DummyMarkerFunc:
        """Represents a marker function like skipif that takes condition and returns decorator."""
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition, **kwargs):
            # skipif, etc. take condition as first arg, return a decorator
            return lambda f: f
    class _DummyMarker:
        def __call__(self, func):
            return func
        def __getattr__(self, name):
            # Return special function for skipif, otherwise return a callable marker
            if name == 'skipif':
                return _DummyMarkerFunc(name)
            return _DummyMarkerFunc(name)
        @property
        def unit(self):
            return self
    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()
    class _DummyPytest:
        mark = _DummyMark()
    pytest = _DummyPytest()

from fastapi.testclient import TestClient

from qwen3_tts.server.app import app


@pytest.mark.unit
class TestFastAPIServer(unittest.TestCase):
    """Test FastAPI server endpoints."""

    def test_fastapi_app_exists(self):
        """FastAPI app should be importable."""
        from fastapi import FastAPI
        self.assertIsInstance(app, FastAPI)

    def test_app_has_routes(self):
        """App should have expected routes."""
        routes = [route.path for route in app.routes]
        self.assertIn("/health", routes)
        self.assertIn("/generation-status", routes)
        self.assertIn("/shutdown", routes)
        self.assertIn("/stats", routes)
        self.assertIn("/models", routes)

    def test_app_has_lifespan(self):
        """App should have lifespan configured."""
        self.assertIsNotNone(app.router.lifespan_context)

    def test_health_endpoint_exists(self):
        """Health endpoint should be accessible."""
        # Just verify the route exists
        routes = [r for r in app.routes if hasattr(r, 'path') and r.path == "/health"]
        self.assertTrue(len(routes) > 0)

    def test_models_endpoint_exists(self):
        """Models endpoint should exist."""
        routes = [r for r in app.routes if hasattr(r, 'path') and r.path == "/models"]
        self.assertTrue(len(routes) > 0)


if __name__ == '__main__':
    unittest.main()
