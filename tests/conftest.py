#!/usr/bin/env python3
"""Shared pytest fixtures for qwen3-tts test suite.

Provides common test utilities:
- unused_port: OS-assigned unused port for parallel test execution
- fastapi_client: FastAPI TestClient with mocked auth
- tmp_config: Temporary config file for isolated testing
- mock_engine: Mocked engine module for testing without models

Usage:
    from tests.conftest import fastapi_client, tmp_config

    def test_something(fastapi_client):
        response = fastapi_client.get("/health")
        assert response.status_code == 200
"""
import json
import os
import socket
import tempfile
from unittest.mock import AsyncMock, MagicMock

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
    import soundfile  # noqa: F401
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# qwen3_tts.server.app imports slowapi unconditionally, so any fixture that
# imports server.app needs slowapi too — gate those on HAS_FASTAPI and HAS_SLOWAPI.
try:
    import slowapi  # noqa: F401
    HAS_SLOWAPI = True
except ImportError:
    HAS_SLOWAPI = False


@pytest.fixture
def unused_port():
    """Get an OS-assigned unused port for this test.

    This fixture enables parallel test execution with pytest-xdist
    by ensuring each worker gets its own unique port. When multiple
    workers run tests simultaneously, they won't collide on port 5123.

    Example:
        def test_with_port(unused_port):
            port = unused_port
            # Use this port for test server or client

    The port is guaranteed to be free at fixture yield time.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        port = s.getsockname()[1]
    yield port


def _init_app_state(app, auth_token="test_token"):
    """Initialize app.state with minimal required attributes for testing.

    Shared by both the autouse xdist fixture and the fastapi_client fixture.
    """
    import asyncio
    import threading

    app.state.auth_token = auth_token  # nosec B105
    app.state.models = {"clone": None, "design": None, "custom": None}
    app.state.model_load_times = {}
    mock_lock = AsyncMock()
    mock_lock.__aenter__.return_value = None
    mock_lock.__aexit__.return_value = None
    app.state.generation_lock = mock_lock
    app.state.generation_state = {
        "active": False,
        "start_time": 0.0,
        "text_length": 0,
        "mode": "",
        "batch_index": 0,
        "batch_total": 0,
        "chunk_index": 0,
        "chunk_total": 0,
        "generation_id": None,
        "cancelled": False,
    }
    app.state.request_queue = set()
    app.state.request_queue_lock = threading.Lock()
    app.state.last_activity = 0
    app.state.models_loaded = threading.Event()
    app.state.gen_cache = {}
    app.state.gen_cache_lock = threading.Lock()
    app.state.inference_lock = asyncio.Lock()
    app.state.eta_cache = {"median_rate": None, "last_updated": 0}
    app.state.eta_cache_lock = threading.Lock()
    app.state.model_load_errors = {"clone": None, "design": None, "custom": None}
    app.state.shutdown_timer = None
    app.state.pending_lock = asyncio.Lock()
    app.state.pending_requests = []


def _save_app_state(app):
    """Snapshot non-private app.state attributes for later restoration."""
    original = {}
    for key in dir(app.state):
        if not key.startswith('_'):
            original[key] = getattr(app.state, key, None)
    return original


def _restore_app_state(app, original):
    """Restore app.state from a snapshot."""
    for key, value in original.items():
        if value is None:
            try:
                delattr(app.state, key)
            except AttributeError:
                pass
        else:
            setattr(app.state, key, value)


@pytest.fixture(autouse=True)
def initialize_app_state_for_xdist():
    """Auto-initialize app.state for all tests to support xdist parallel execution.

    With pytest-xdist, each worker has its own app instance. Tests that create
    their own TestClient(app) directly need app.state initialized. This fixture
    runs automatically for all tests to ensure app.state has minimal required
    attributes.

    This is autouse=True so it applies to all tests without needing to request it.
    """
    if not (HAS_FASTAPI and HAS_SLOWAPI):
        yield
        return

    from qwen3_tts.server.app import app

    # Check if already initialized (another test in same worker may have set it up)
    if hasattr(app.state, '_xdist_initialized'):
        yield
        return

    # Skip if app.state was already initialized externally — e.g. by a unittest
    # TestCase's setUpClass calling _make_test_client. Clobbering would replace
    # auth_token with "xdist_test_token" and reset the models_loaded Event,
    # breaking auth and readiness checks for those tests.
    if hasattr(app.state, 'auth_token'):
        yield
        return

    original_state = _save_app_state(app)

    try:
        _init_app_state(app, auth_token="xdist_test_token")
        app.state._xdist_initialized = True
        app.state.test_port = 5123  # Default port for bare TestClient tests

        yield

    finally:
        app.state._xdist_initialized = False
        _restore_app_state(app, original_state)


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset slowapi rate limiter state before each test.

    Without this, rate limit counters accumulate across tests within a suite
    run, causing later tests to receive unexpected 429 responses.
    """
    if not (HAS_FASTAPI and HAS_SLOWAPI):
        yield
        return

    from qwen3_tts.server.app import app

    for attr in ("limiter", "limiter_hybrid", "limiter_ip", "limiter_token"):
        limiter = getattr(app.state, attr, None)
        if limiter is not None and hasattr(limiter, "reset"):
            limiter.reset()

    yield


@pytest.fixture
def tmp_config():
    """Create a temporary config file for isolated testing.

    Returns a dict with the config data and temp file path.
    Automatically cleans up the temp file after the test.

    Example:
        def test_with_config(tmp_config):
            cfg = tmp_config["data"]
            cfg["output_directory"] = "/tmp/test_output"
            # ... test code using tmp_config["path"]
    """
    config_data = {
        "default_voice_description": "A calm, soothing voice",
        "default_clone_prompt": "test_voice.pt",
        "output_directory": "/tmp/test_output",
        "language": "English",
        "server": {
            "host": "127.0.0.1",
            "port": 5123,
            "auto_shutdown_minutes": 0,
        },
        "models": {
            "clone": {"load_at_startup": False},
            "design": {"load_at_startup": False},
            "custom": {"load_at_startup": False},
        },
        "security": {
            "max_text_length": 10000,
            "max_batch_size": 20,
        },
        "generation": {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
            "seed": None,
            "max_chunk_chars": 500,
            "max_new_tokens": 2048,
        },
        "advanced": {
            "backend": "torch",
            "model_size": "1.7B",
            "dtype": "float16",
            "audio_loader": "torchaudio",
        },
        "presets": {
            "consistent": {"temperature": 0.5, "top_k": 30, "seed": 42},
            "creative": {"temperature": 0.9, "top_p": 0.98},
        },
        "aliases": {
            "default": {
                "prompt": "test_voice.pt",
                "preset": "consistent",
            }
        },
        "ui": {"port": 7860},
    }

    fd, path = tempfile.mkstemp(suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(config_data, f)
        yield {"data": config_data, "path": path}
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.fixture
def mock_engine():
    """Provide a mocked engine module for testing without loading models.

    Mocks the heavy engine imports and provides fake model objects.
    Use this when testing server endpoints that would normally require
    loaded TTS models.

    Example:
        def test_generate_endpoint(mock_engine):
            # mock_engine provides fake models and inference mocks
            response = app_client.post("/generate", json={...})
            assert response.status_code == 200
    """
    mock = MagicMock()

    # Fake model objects
    mock.clone_model = MagicMock()
    mock.clone_model.__class__.__name__ = "Qwen3TTSModel"
    mock.design_model = MagicMock()
    mock.custom_model = MagicMock()

    # Mock inference functions
    mock.run_inference = MagicMock(return_value=(
        bytearray(b'mock_audio_data'),  # audio bytes
        24000,  # sample rate
    ))

    # Mock streaming inference
    def mock_streaming(*args, **kwargs):
        """Yield fake audio chunks."""
        import numpy as np
        for i in range(3):
            chunk = np.random.rand(2400).astype(np.float32)  # 0.1s chunks
            yield chunk, 24000

    mock.run_inference_streaming = MagicMock(side_effect=mock_streaming)

    # Mock voice prompt loading
    mock.load_voice_prompt = MagicMock(return_value=mock.clone_model)

    # Mock voice prompt cache
    cache_info = MagicMock()
    cache_info.currsize = 0
    cache_info.hits = 0
    mock.voice_prompt_cache_info = MagicMock(return_value=cache_info)

    return mock


@pytest.fixture
def fastapi_client(tmp_config, unused_port):
    """Create a FastAPI TestClient with mocked auth and minimal setup.

    This fixture:
    1. Uses an OS-assigned unique port for parallel test execution (xdist)
    2. Sets up a minimal test config
    3. Initializes app.state with required attributes (mimics lifespan)
    4. Mocks the auth token
    5. Stores the port in app.state for endpoint construction
    6. Returns a wrapper client that automatically adds auth headers

    Skips if FastAPI or soundfile are not installed.

    Example:
        def test_health_endpoint(fastapi_client):
            response = fastapi_client.get("/health")
            assert response.status_code in (200, 503)  # loading or ready
    """
    if not (HAS_FASTAPI and HAS_SLOWAPI):
        pytest.skip("requires fastapi, soundfile, slowapi")

    from qwen3_tts.server.app import app

    test_token = "test_token_fixture"  # nosec B105
    original_state = _save_app_state(app)

    try:
        _init_app_state(app, auth_token=test_token)
        app.state.test_port = unused_port  # Store dynamic port for any URL construction

        # Update server_config with the dynamic port
        test_config = tmp_config["data"].copy()
        test_config["server"] = test_config["server"].copy()
        test_config["server"]["port"] = unused_port
        app.state.server_config = test_config

        # Create TestClient with app state override
        raw_client = TestClient(app)

        # Return a wrapper that automatically adds auth headers
        class AuthenticatedTestClient:
            def __init__(self, client, token, port):
                self._client = client
                self._token = token
                self.port = port  # Expose the dynamic port for tests

            def get(self, path, **kwargs):
                headers = kwargs.pop('headers', {})
                headers['Authorization'] = f'Bearer {self._token}'
                return self._client.get(path, headers=headers, **kwargs)

            def post(self, path, **kwargs):
                headers = kwargs.pop('headers', {})
                headers['Authorization'] = f'Bearer {self._token}'
                return self._client.post(path, headers=headers, **kwargs)

            def put(self, path, **kwargs):
                headers = kwargs.pop('headers', {})
                headers['Authorization'] = f'Bearer {self._token}'
                return self._client.put(path, headers=headers, **kwargs)

            def delete(self, path, **kwargs):
                headers = kwargs.pop('headers', {})
                headers['Authorization'] = f'Bearer {self._token}'
                return self._client.delete(path, headers=headers, **kwargs)

        yield AuthenticatedTestClient(raw_client, test_token, unused_port)

    finally:
        _restore_app_state(app, original_state)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "requires_server: marks tests that need a running server"
    )
