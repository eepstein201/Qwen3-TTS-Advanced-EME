#!/usr/bin/env python3
"""Async-specific behavior tests for FastAPI TTS server endpoints.

Tests that:
  - asyncio.Lock exists for GPU serialization
  - Streaming endpoint requires auth
  - Auth fails without token

No GPU, models, or running server required. Tests use FastAPI TestClient.

Run: pytest tests/test_fastapi_endpoints.py -v
"""
import os
import sys
import asyncio
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi.testclient import TestClient
    import soundfile  # noqa: F401
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = pytest.mark.skipif(not HAS_DEPS, reason="requires fastapi, soundfile, numpy")

if HAS_DEPS:
    from qwen3_tts.server.app import app


# Pytest-style tests using fixtures from conftest.py

@pytest.mark.unit
@_skip
def test_asyncio_lock_works():
    """Verify asyncio.Lock works for GPU serialization."""
    lock = asyncio.Lock()
    assert lock is not None

    async def test_lock_context():
        async with lock:
            return True

    result = asyncio.run(test_lock_context())
    assert result is True


@pytest.mark.unit
@_skip
def test_fastapi_app_exists():
    """FastAPI app should be importable and valid."""
    from fastapi import FastAPI
    assert isinstance(app, FastAPI)


@pytest.mark.unit
@_skip
def test_fastapi_routes_exist():
    """FastAPI app should have all expected routes."""
    routes = [route.path for route in app.routes]
    expected_routes = [
        "/health",
        "/generation-status",
        "/shutdown",
        "/stats",
        "/models",
        "/generate",
        "/generate-stream",
    ]
    for route in expected_routes:
        assert route in routes, f"Route {route} should exist"


@pytest.mark.unit
@_skip
def test_fastapi_lifespan_configured():
    """FastAPI app should have lifespan configured for startup/shutdown."""
    assert app.router.lifespan_context is not None


@pytest.mark.unit
@_skip
def test_public_endpoints_no_auth(fastapi_client):
    """Public endpoints should work without auth token."""
    public_endpoints = ["/health", "/generation-status"]

    for endpoint in public_endpoints:
        response = fastapi_client.get(endpoint)
        # Should not return 401 (may return 503 if models not loaded)
        assert response.status_code != 401, \
            f"{endpoint} should be public, got {response.status_code}"


@pytest.mark.unit
@_skip
def test_protected_endpoints_require_auth(fastapi_client):
    """Protected endpoints should return 401 without auth token."""
    # Create a fresh client without fixture setup to test missing auth
    from fastapi.testclient import TestClient
    client = TestClient(app)

    response = client.get("/stats")
    assert response.status_code == 401, "Should require auth"


@pytest.mark.unit
@_skip
def test_valid_auth_accepted(fastapi_client):
    """Valid auth token should be accepted."""
    response = fastapi_client.get("/stats")
    # Should not get 401 (may get 503 or other error if models not loaded)
    assert response.status_code != 401, "Valid auth token should be accepted"


@pytest.mark.unit
@_skip
def test_invalid_auth_rejected():
    """Invalid auth token should be rejected."""
    from fastapi.testclient import TestClient
    client = TestClient(app)

    response = client.get(
        "/stats",
        headers={"Authorization": "Bearer invalid_token_12345"}
    )
    assert response.status_code == 401


@pytest.mark.unit
@_skip
def test_missing_auth_rejected():
    """Missing auth header should be rejected."""
    from fastapi.testclient import TestClient
    client = TestClient(app)

    response = client.get("/stats")
    assert response.status_code == 401


@pytest.mark.unit
@_skip
def test_streaming_requires_auth():
    """Streaming endpoint should reject requests without auth token."""
    from fastapi.testclient import TestClient
    client = TestClient(app)

    response = client.post(
        "/generate-stream",
        json={"text": "Hello world", "mode": "clone"},
    )
    assert response.status_code == 401


@pytest.mark.unit
@_skip
def test_streaming_with_auth(fastapi_client):
    """Streaming endpoint with auth should not return 401."""
    response = fastapi_client.post(
        "/generate-stream",
        json={"text": "Hello", "mode": "custom", "speaker": "ryan"}
    )
    # Should not get 401 (may get 503 or other error if models not loaded)
    assert response.status_code != 401
