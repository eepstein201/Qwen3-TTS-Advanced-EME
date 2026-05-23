"""Shared test infrastructure for decomposed test_voice_*.py modules.

Provides dependency checks, skip decorators, and FastAPI test helpers
that were originally at the top of test_voice.py.
"""

import unittest
from contextlib import asynccontextmanager

from unittest.mock import patch, MagicMock  # noqa: F401 — re-exported for test files

# Check optional dependencies — tests that need these are skipped when missing
try:
    import soundfile  # noqa: F401
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import gradio  # noqa: F401
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

try:
    from fastapi.testclient import TestClient  # noqa: F401
    import soundfile  # noqa: F401
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# Convenience booleans
_server_deps = HAS_SOUNDFILE and HAS_FASTAPI
_client_deps = HAS_SOUNDFILE
_ui_deps = HAS_GRADIO

# Skip decorators
_skip_server = unittest.skipUnless(_server_deps, "requires soundfile + fastapi")
_skip_client = unittest.skipUnless(_client_deps, "requires soundfile")
_skip_ui = unittest.skipUnless(_ui_deps, "requires gradio")
_skip_generate = unittest.skipUnless(HAS_SOUNDFILE, "requires soundfile (voice_generate)")


# =============================================================================
# Helper functions for FastAPI test setup
# =============================================================================

def _setup_fastapi_app_state(app, server_config=None):
    """Initialize app.state with minimal required attributes for FastAPI tests."""
    import threading
    import asyncio

    from unittest.mock import AsyncMock
    app.state.auth_token = "test_token"  # nosec B105
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
    if server_config:
        app.state.server_config = server_config
    else:
        app.state.server_config = {
            "security": {"max_text_length": 10000, "max_batch_size": 20},
            "auto_shutdown_minutes": 0,
        }


@asynccontextmanager
async def _null_lifespan(app):
    """No-op lifespan to prevent real model loading during tests."""
    yield

def _make_test_client(app, server_config=None):
    """Create TestClient without triggering real lifespan model loading."""
    from fastapi.testclient import TestClient
    from unittest.mock import patch

    _setup_fastapi_app_state(app, server_config)
    # Mock memory check to avoid 503 due to low memory during tests
    with patch("qwen3_tts.server.app_lifespan._check_memory_available", return_value=(True, 10000)):
        original = app.router.lifespan_context
        app.router.lifespan_context = _null_lifespan
        client = TestClient(app)
        app.router.lifespan_context = original
        return client
