#!/usr/bin/env python3
"""Shared pytest fixtures for qwen3-tts test suite.

Provides common test utilities:
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
import sys
import tempfile
from unittest.mock import MagicMock

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi.testclient import TestClient
    import soundfile  # noqa: F401
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


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
def fastapi_client(tmp_config):
    """Create a FastAPI TestClient with mocked auth and minimal setup.

    This fixture:
    1. Sets up a minimal test config
    2. Initializes app.state with required attributes (mimics lifespan)
    3. Mocks the auth token
    4. Returns a wrapper client that automatically adds auth headers

    Skips if FastAPI or soundfile are not installed.

    Example:
        def test_health_endpoint(fastapi_client):
            response = fastapi_client.get("/health")
            assert response.status_code in (200, 503)  # loading or ready
    """
    if not HAS_FASTAPI:
        pytest.skip("requires fastapi and soundfile")

    import asyncio
    import threading
    from qwen3_tts.server.app import app

    # Store original state for cleanup
    original_state = {}
    for key in dir(app.state):
        if not key.startswith('_'):
            original_state[key] = getattr(app.state, key, None)

    try:
        # Initialize app.state with minimal required attributes (mimics lifespan)
        test_token = "test_token_fixture"  # nosec B105
        app.state.auth_token = test_token
        app.state.models = {"clone": None, "design": None, "custom": None}
        app.state.model_load_times = {}
        app.state.generation_lock = threading.Lock()
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
        app.state.last_activity = 0
        app.state.models_loaded = threading.Event()
        app.state.gen_cache = {}
        app.state.gen_cache_lock = threading.Lock()
        app.state.inference_lock = asyncio.Lock()
        app.state.eta_cache = {"median_rate": None, "last_updated": 0}
        app.state.shutdown_timer = None
        app.state.server_config = tmp_config["data"]

        # Create TestClient with app state override
        raw_client = TestClient(app)

        # Return a wrapper that automatically adds auth headers
        class AuthenticatedTestClient:
            def __init__(self, client, token):
                self._client = client
                self._token = token

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

        yield AuthenticatedTestClient(raw_client, test_token)

    finally:
        # Restore original state
        for key, value in original_state.items():
            if value is None:
                # Remove attributes that weren't originally present
                try:
                    delattr(app.state, key)
                except AttributeError:
                    pass
            else:
                setattr(app.state, key, value)


@pytest.fixture
def authenticated_fastapi_client(fastapi_client):
    """Create a FastAPI TestClient with valid auth headers pre-configured.

    Returns a client helper that automatically adds auth headers.

    Example:
        def test_protected_endpoint(authenticated_fastapi_client):
            response = authenticated_fastapi_client.get("/models")
            assert response.status_code == 200
    """
    class AuthenticatedClient:
        def __init__(self, client, token):
            self.client = client
            self.token = token

        def get(self, path, **kwargs):
            headers = kwargs.pop('headers', {})
            headers['Authorization'] = f'Bearer {self.token}'
            return self.client.get(path, headers=headers, **kwargs)

        def post(self, path, **kwargs):
            headers = kwargs.pop('headers', {})
            headers['Authorization'] = f'Bearer {self.token}'
            return self.client.post(path, headers=headers, **kwargs)

        def put(self, path, **kwargs):
            headers = kwargs.pop('headers', {})
            headers['Authorization'] = f'Bearer {self.token}'
            return self.client.put(path, headers=headers, **kwargs)

        def delete(self, path, **kwargs):
            headers = kwargs.pop('headers', {})
            headers['Authorization'] = f'Bearer {self.token}'
            return self.client.delete(path, headers=headers, **kwargs)

    return AuthenticatedClient(fastapi_client, fastapi_client.auth_token)


@pytest.fixture
def loaded_models(fastapi_client):
    """Context manager that temporarily marks models as loaded.

    Useful for testing endpoints that require models to be loaded
    without actually loading them.

    Example:
        def test_generate_with_loaded_models(loaded_models):
            # Models are "loaded" for this test
            response = fastapi_client.post("/generate", json={...})
    """
    from qwen3_tts.server.app import app

    # Store original state
    original_clone = app.state.models.get("clone")
    original_design = app.state.models.get("design")
    original_custom = app.state.models.get("custom")

    try:
        # Set fake loaded models
        app.state.models["clone"] = MagicMock()
        app.state.models["design"] = MagicMock()
        app.state.models["custom"] = MagicMock()

        yield

    finally:
        # Restore
        app.state.models["clone"] = original_clone
        app.state.models["design"] = original_design
        app.state.models["custom"] = original_custom


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
