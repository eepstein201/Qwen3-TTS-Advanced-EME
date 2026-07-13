#!/usr/bin/env python3
"""Async-specific behavior tests for FastAPI TTS server endpoints.

Tests that:
  - asyncio.Lock exists for GPU serialization
  - Streaming endpoint requires auth
  - Auth fails without token

No GPU, models, or running server required. Tests use FastAPI TestClient.

Run: pytest tests/test_fastapi_endpoints.py -v
"""
import asyncio
from unittest.mock import MagicMock, patch

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

try:
    import numpy as np  # noqa: F401
    import soundfile  # noqa: F401
    from fastapi.testclient import TestClient  # noqa: F401
    HAS_FASTAPI_DEPS = True
except ImportError:
    HAS_FASTAPI_DEPS = False

try:
    import slowapi  # noqa: F401  (server.app imports slowapi unconditionally)
    HAS_SLOWAPI = True
except ImportError:
    HAS_SLOWAPI = False

HAS_DEPS = HAS_FASTAPI_DEPS and HAS_SLOWAPI

# server.app cannot be imported without these (it imports slowapi unconditionally).
# Skip the whole module cleanly — not a collection/setup error — when any are
# missing, e.g. an env installed without the server/test extra. A module-level
# skip also covers in-body `from qwen3_tts.server.app import ...` calls that a
# per-test decorator would miss.
if HAS_PYTEST and not HAS_DEPS:
    pytest.skip(
        "requires fastapi, soundfile, numpy, slowapi", allow_module_level=True
    )

if HAS_DEPS:
    _skip = pytest.mark.skipif(not HAS_DEPS, reason="requires fastapi, soundfile, numpy, slowapi")
else:
    def _skip(f):
        return f

if HAS_DEPS:
    from qwen3_tts.server.app import app


if HAS_PYTEST and HAS_DEPS:
    @pytest.fixture(autouse=True)
    def _healthy_memory_guard():
        """Pin the memory guard to healthy so tests never depend on host RAM.

        Without this, validation tests expecting 400 flake to 503 whenever the
        host dips below the guard's 1 GB threshold mid-suite. Tests that
        exercise the guard itself apply their own inner patch, which overrides
        this one.
        """
        with patch(
            "qwen3_tts.server.app_generation._check_memory_available",
            return_value=(True, 8192),
        ):
            yield


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


# ---------------------------------------------------------------------------
# Phase 4A: Generation & model management endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
def test_generate_no_text(fastapi_client):
    """POST /generate with no text returns 400."""
    response = fastapi_client.post("/generate", json={"mode": "clone"})
    assert response.status_code == 400


@pytest.mark.unit
@_skip
def test_generate_empty_text(fastapi_client):
    """POST /generate with empty string returns 400."""
    response = fastapi_client.post(
        "/generate", json={"text": "", "mode": "clone"}
    )
    assert response.status_code == 400


@pytest.mark.unit
@_skip
def test_generate_model_not_loaded(fastapi_client):
    """POST /generate when model is None returns 503 model_not_loaded."""
    app.state.models_loaded.set()
    response = fastapi_client.post(
        "/generate",
        json={"text": "Hello world", "mode": "clone", "prompt_file": "v.wav"},
    )
    assert response.status_code == 503
    data = response.json()
    assert data["detail"]["error"] == "model_not_loaded"


@pytest.mark.unit
@_skip
def test_generate_batch_size_exceeded(fastapi_client):
    """POST /generate with too many texts returns 400."""
    app.state.models_loaded.set()
    texts = [f"text {i}" for i in range(25)]
    response = fastapi_client.post(
        "/generate", json={"texts": texts, "mode": "clone"}
    )
    assert response.status_code == 400
    assert "exceeds limit" in response.json()["detail"]


@pytest.mark.unit
@_skip
def test_generate_text_too_long(fastapi_client):
    """POST /generate with text exceeding max_text_length returns 400."""
    app.state.models_loaded.set()
    app.state.models["clone"] = MagicMock()
    try:
        long_text = "x" * 10001
        response = fastapi_client.post(
            "/generate",
            json={"text": long_text, "mode": "clone", "prompt_file": "v.wav"},
        )
        assert response.status_code == 400
        assert "character limit" in response.json()["detail"]
    finally:
        app.state.models["clone"] = None


@pytest.mark.unit
@_skip
def test_generate_empty_text_in_batch(fastapi_client):
    """POST /generate with empty text in batch returns 400."""
    app.state.models_loaded.set()
    response = fastapi_client.post(
        "/generate", json={"texts": ["hello", ""], "mode": "clone"}
    )
    assert response.status_code == 400
    assert "empty or invalid" in response.json()["detail"]


@pytest.mark.unit
@_skip
def test_generate_memory_guard(fastapi_client):
    """POST /generate returns 503 when memory is low."""
    from unittest.mock import patch
    app.state.models_loaded.set()
    with patch("qwen3_tts.server.app_generation._check_memory_available", return_value=(False, 500)):
        response = fastapi_client.post(
            "/generate",
            json={"text": "Hello", "mode": "clone", "prompt_file": "v.wav"},
        )
    assert response.status_code == 503
    assert "insufficient_memory" in str(response.json()["detail"])


@pytest.mark.unit
@_skip
def test_generate_stream_no_text(fastapi_client):
    """POST /generate-stream with no text returns 400."""
    response = fastapi_client.post(
        "/generate-stream", json={"mode": "clone"}
    )
    assert response.status_code == 400


@pytest.mark.unit
@_skip
def test_generate_stream_empty_text(fastapi_client):
    """POST /generate-stream with empty text returns 400."""
    response = fastapi_client.post(
        "/generate-stream", json={"text": "", "mode": "clone"}
    )
    assert response.status_code == 400


@pytest.mark.unit
@_skip
def test_generate_stream_model_not_loaded(fastapi_client):
    """POST /generate-stream when model is None returns 503."""
    app.state.models_loaded.set()
    response = fastapi_client.post(
        "/generate-stream",
        json={"text": "Hello", "mode": "clone", "prompt_file": "v.wav"},
    )
    assert response.status_code == 503


@pytest.mark.unit
@_skip
def test_generate_stream_memory_guard(fastapi_client):
    """POST /generate-stream returns 503 when memory is low."""
    from unittest.mock import patch
    app.state.models_loaded.set()
    with patch("qwen3_tts.server.app_generation._check_memory_available", return_value=(False, 300)):
        response = fastapi_client.post(
            "/generate-stream",
            json={"text": "Hello", "mode": "clone", "prompt_file": "v.wav"},
        )
    assert response.status_code == 503


@pytest.mark.unit
@_skip
def test_load_model_invalid_type(fastapi_client):
    """POST /load-model with invalid model_type returns 400."""
    response = fastapi_client.post(
        "/load-model", json={"model_type": "invalid"}
    )
    assert response.status_code == 400
    assert "Unknown model type" in response.json()["detail"]


@pytest.mark.unit
@_skip
def test_load_model_already_loaded(fastapi_client):
    """POST /load-model when model is already loaded returns already_loaded."""
    app.state.models["clone"] = MagicMock()
    try:
        response = fastapi_client.post(
            "/load-model", json={"model_type": "clone"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "already_loaded"
    finally:
        app.state.models["clone"] = None


@pytest.mark.unit
@_skip
def test_unload_model_invalid_type(fastapi_client):
    """POST /unload-model with invalid model_type returns 400."""
    response = fastapi_client.post(
        "/unload-model", json={"model_type": "bad"}
    )
    assert response.status_code == 400


@pytest.mark.unit
@_skip
def test_unload_model_already_unloaded(fastapi_client):
    """POST /unload-model when model is None returns already_unloaded."""
    response = fastapi_client.post(
        "/unload-model", json={"model_type": "design"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "already_unloaded"


@pytest.mark.unit
@_skip
def test_unload_model_active_generation_conflict(fastapi_client):
    """POST /unload-model during active generation on that mode returns 409."""
    app.state.models["clone"] = MagicMock()
    app.state.generation_state["active"] = True
    app.state.generation_state["mode"] = "clone"
    try:
        response = fastapi_client.post(
            "/unload-model", json={"model_type": "clone"}
        )
        assert response.status_code == 409
        assert "active" in response.json()["detail"]
    finally:
        app.state.models["clone"] = None
        app.state.generation_state["active"] = False
        app.state.generation_state["mode"] = ""


@pytest.mark.unit
@_skip
def test_update_model_config_no_params(fastapi_client):
    """POST /update-model-config with no params returns 400."""
    response = fastapi_client.post(
        "/update-model-config", json={}
    )
    assert response.status_code == 400
    assert "At least one" in response.json()["detail"]


@pytest.mark.unit
@_skip
def test_update_model_config_invalid_size(fastapi_client):
    """POST /update-model-config with bad model_size returns 400."""
    response = fastapi_client.post(
        "/update-model-config", json={"model_size": "99B"}
    )
    assert response.status_code == 400
    assert "Invalid model_size" in response.json()["detail"]


@pytest.mark.unit
@_skip
def test_update_model_config_invalid_quant(fastapi_client):
    """POST /update-model-config with bad mlx_quantization returns 400."""
    response = fastapi_client.post(
        "/update-model-config", json={"mlx_quantization": "2bit"}
    )
    assert response.status_code == 400
    assert "Invalid mlx_quantization" in response.json()["detail"]


@pytest.mark.unit
@_skip
def test_update_model_config_valid(fastapi_client):
    """POST /update-model-config with valid params succeeds."""
    from unittest.mock import patch
    with patch("qwen3_tts.server.app_models.save_config"), \
         patch("qwen3_tts.server.app._get_app_config", return_value={"advanced": {}}):
        response = fastapi_client.post(
            "/update-model-config", json={"model_size": "0.6B"}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "config_updated"
    assert "model_size=0.6B" in data["changes"]


@pytest.mark.unit
@_skip
def test_update_startup_config_no_types(fastapi_client):
    """POST /update-startup-config with no model types returns 400."""
    response = fastapi_client.post(
        "/update-startup-config", json={}
    )
    assert response.status_code == 400


@pytest.mark.unit
@_skip
def test_update_startup_config_valid(fastapi_client):
    """POST /update-startup-config saves config correctly."""
    from unittest.mock import patch
    with patch("qwen3_tts.server.app_models.save_config") as mock_save, \
         patch("qwen3_tts.server.app._get_app_config",
               return_value={"models": {"clone": {}, "design": {}, "custom": {}}}):
        response = fastapi_client.post(
            "/update-startup-config",
            json={"clone": True, "design": False},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "updated"
    assert "clone=on" in data["changes"]
    assert "design=off" in data["changes"]
    mock_save.assert_called_once()


@pytest.mark.unit
@_skip
def test_generate_stream_text_too_long(fastapi_client):
    """POST /generate-stream with text exceeding limit returns 400."""
    app.state.models_loaded.set()
    app.state.models["clone"] = MagicMock()
    try:
        long_text = "y" * 10001
        response = fastapi_client.post(
            "/generate-stream",
            json={"text": long_text, "mode": "clone", "prompt_file": "v.wav"},
        )
        assert response.status_code == 400
        assert "character limit" in response.json()["detail"]
    finally:
        app.state.models["clone"] = None
