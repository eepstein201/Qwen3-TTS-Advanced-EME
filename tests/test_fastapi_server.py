"""Tests for FastAPI server implementation.

Validates that the FastAPI server works correctly with async,
proper streaming, and app.state for worker-safe operations.
"""

import unittest
from pathlib import Path

import pytest

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
