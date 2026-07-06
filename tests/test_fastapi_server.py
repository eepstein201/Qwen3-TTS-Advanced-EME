#!/usr/bin/env python3
"""Tests for FastAPI server — prompt endpoints, admin endpoints, and helpers.

Covers:
- Helper functions: _sanitize_error, _get_real_client_ip, _check_memory_available,
  _estimate_eta, reset_activity_timer, cleanup_resources
- Public endpoints: /health, /ready, /generation-status, /queue-status
- Admin endpoints: /stats, /cancel-generation
- Prompt endpoints: /prompts, /delete-prompt, /rename-prompt, /preview-prompt,
  /prompt-details

No GPU, models, or running server required. Tests use FastAPI TestClient.

Run: pytest tests/test_fastapi_server.py -v
"""
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

    class _DummyMarkerFunc:
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition, **kwargs):
            return lambda f: f

    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()

    class _DummyPytest:
        mark = _DummyMark()
    pytest = _DummyPytest()

try:
    import slowapi  # noqa: F401  (server.app imports slowapi unconditionally)
    import soundfile  # noqa: F401
    from fastapi.testclient import TestClient  # noqa: F401
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

# server.app cannot be imported without these (it imports slowapi unconditionally).
# Skip the whole module cleanly — not a collection/setup error — when any are
# missing, e.g. an env installed without the server/test extra. A module-level
# skip also covers in-body `from qwen3_tts.server.app import ...` calls that a
# per-test decorator would miss.
if HAS_PYTEST and not HAS_DEPS:
    pytest.skip(
        "requires fastapi, soundfile, slowapi", allow_module_level=True
    )

if HAS_DEPS:
    _skip = pytest.mark.skipif(not HAS_DEPS, reason="requires fastapi, soundfile, slowapi")
else:
    def _skip(f):
        return f

if HAS_DEPS:
    from qwen3_tts.server.app import app


# ---------------------------------------------------------------------------
# Helper function tests (import directly, no HTTP needed)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
def test_sanitize_error_strips_unix_paths():
    """_sanitize_error replaces Unix absolute paths."""
    from qwen3_tts.server.app import _sanitize_error
    result = _sanitize_error("File not found: /Users/foo/bar/voice.pt")
    assert "/Users" not in result
    assert "<path>" in result


@pytest.mark.unit
@_skip
def test_sanitize_error_strips_windows_paths():
    """_sanitize_error replaces Windows paths."""
    from qwen3_tts.server.app import _sanitize_error
    result = _sanitize_error("File not found: C:\\Users\\foo\\voice.pt")
    assert "C:\\" not in result
    assert "<path>" in result


@pytest.mark.unit
@_skip
def test_sanitize_error_caps_length():
    """_sanitize_error caps output at 200 chars."""
    from qwen3_tts.server.app import _sanitize_error
    result = _sanitize_error("x" * 500)
    assert len(result) <= 200


@pytest.mark.unit
@_skip
def test_get_real_client_ip_loopback_trusts_xff():
    """_get_real_client_ip honors X-Forwarded-For from a loopback proxy (Colab tunnel)."""
    from qwen3_tts.server.app import _get_real_client_ip
    mock_request = MagicMock()
    mock_request.client.host = "127.0.0.1"
    mock_request.headers.get.return_value = "10.0.0.1, 10.0.0.2"
    result = _get_real_client_ip(mock_request)
    assert result == "10.0.0.1"


@pytest.mark.unit
@_skip
def test_get_real_client_ip_untrusted_peer_ignores_xff():
    """_get_real_client_ip ignores X-Forwarded-For from an untrusted (direct) peer."""
    from qwen3_tts.server.app import _get_real_client_ip
    mock_request = MagicMock()
    mock_request.client.host = "10.0.0.5"
    mock_request.headers.get.return_value = "192.168.1.1, 10.0.0.2"
    result = _get_real_client_ip(mock_request)
    assert result == "10.0.0.5"


@pytest.mark.unit
@_skip
def test_get_real_client_ip_no_client():
    """_get_real_client_ip returns 127.0.0.1 when client is None (and no XFF)."""
    from qwen3_tts.server.app import _get_real_client_ip
    mock_request = MagicMock()
    mock_request.client = None
    mock_request.headers.get.return_value = None
    result = _get_real_client_ip(mock_request)
    assert result == "127.0.0.1"


@pytest.mark.unit
@_skip
def test_check_memory_available_no_psutil():
    """_check_memory_available returns (True, 0) when psutil missing."""
    from qwen3_tts.server.app import _check_memory_available
    with patch("qwen3_tts.server.app_lifespan._HAS_PSUTIL", False):
        ok, mb = _check_memory_available()
    assert ok is True
    assert mb == 0


@pytest.mark.unit
@_skip
def test_check_memory_available_low():
    """_check_memory_available returns False when memory below threshold."""
    import qwen3_tts.server.app_lifespan as _app_mod
    from qwen3_tts.server.app import _check_memory_available
    mock_mem = MagicMock()
    mock_mem.available = 500 * 1024 * 1024  # 500 MB
    mock_psutil = MagicMock()
    mock_psutil.virtual_memory.return_value = mock_mem
    orig_psutil = getattr(_app_mod, 'psutil', None)
    orig_flag = _app_mod._HAS_PSUTIL
    try:
        _app_mod.psutil = mock_psutil
        _app_mod._HAS_PSUTIL = True
        ok, mb = _check_memory_available()
    finally:
        _app_mod._HAS_PSUTIL = orig_flag
        if orig_psutil is not None:
            _app_mod.psutil = orig_psutil
        elif hasattr(_app_mod, 'psutil'):
            delattr(_app_mod, 'psutil')
    assert ok is False
    assert mb == 500


@pytest.mark.unit
@_skip
def test_check_memory_available_ok():
    """_check_memory_available returns True when memory sufficient."""
    import qwen3_tts.server.app_lifespan as _app_mod
    from qwen3_tts.server.app import _check_memory_available
    mock_mem = MagicMock()
    mock_mem.available = 4 * 1024 * 1024 * 1024  # 4 GB
    mock_psutil = MagicMock()
    mock_psutil.virtual_memory.return_value = mock_mem
    orig_psutil = getattr(_app_mod, 'psutil', None)
    orig_flag = _app_mod._HAS_PSUTIL
    try:
        _app_mod.psutil = mock_psutil
        _app_mod._HAS_PSUTIL = True
        ok, mb = _check_memory_available()
    finally:
        _app_mod._HAS_PSUTIL = orig_flag
        if orig_psutil is not None:
            _app_mod.psutil = orig_psutil
        elif hasattr(_app_mod, 'psutil'):
            delattr(_app_mod, 'psutil')
    assert ok is True
    assert mb == 4096


@pytest.mark.unit
@_skip
def test_estimate_eta_zero_median_rate():
    """_estimate_eta returns None when median_rate is 0."""
    from qwen3_tts.server.app import _estimate_eta
    state = SimpleNamespace(
        eta_cache={"median_rate": 0.0, "last_updated": time.time() + 9999},
        eta_cache_lock=threading.Lock(),
    )
    result = _estimate_eta(state, text_length=100, elapsed_sec=5.0)
    assert result is None


@pytest.mark.unit
@_skip
def test_estimate_eta_none_median_rate():
    """_estimate_eta returns None when median_rate is None."""
    from qwen3_tts.server.app import _estimate_eta
    state = SimpleNamespace(
        eta_cache={"median_rate": None, "last_updated": time.time() + 9999},
        eta_cache_lock=threading.Lock(),
    )
    result = _estimate_eta(state, text_length=100, elapsed_sec=5.0)
    assert result is None


@pytest.mark.unit
@_skip
def test_estimate_eta_positive_rate():
    """_estimate_eta returns remaining seconds with valid rate."""
    from qwen3_tts.server.app import _estimate_eta
    state = SimpleNamespace(
        eta_cache={"median_rate": 10.0, "last_updated": time.time() + 9999},
        eta_cache_lock=threading.Lock(),
    )
    # text_length=100, rate=10 chars/sec → estimated_total=10s, elapsed=2s → remaining=8s
    result = _estimate_eta(state, text_length=100, elapsed_sec=2.0)
    assert result == 8.0


@pytest.mark.unit
@_skip
def test_reset_activity_timer_no_auto_shutdown():
    """reset_activity_timer sets last_activity but skips timer when auto_shutdown=0."""
    from qwen3_tts.server.app import reset_activity_timer
    state = SimpleNamespace(
        last_activity=0,
        server_config={"auto_shutdown_minutes": 0},
        shutdown_timer=None,
    )
    reset_activity_timer(state)
    assert state.last_activity > 0
    assert state.shutdown_timer is None


@pytest.mark.unit
@_skip
def test_cleanup_resources_cleans_models():
    """cleanup_resources sets models to None."""
    from qwen3_tts.server.app import cleanup_resources
    state = SimpleNamespace(
        shutdown_timer=None,
        models={"clone": MagicMock(), "design": None, "custom": None},
        gen_cache={},
    )
    with patch("qwen3_tts.server.app.cleanup_pid_file"):
        cleanup_resources(state)
    assert state.models["clone"] is None


@pytest.mark.unit
@_skip
def test_cleanup_resources_clears_gen_cache(tmp_path):
    """cleanup_resources removes gen_cache temp files."""
    from qwen3_tts.server.app import cleanup_resources
    fake_file = tmp_path / "cached.wav"
    fake_file.write_bytes(b"fake audio")
    state = SimpleNamespace(
        shutdown_timer=None,
        models={"clone": None, "design": None, "custom": None},
        gen_cache={"key1": {"main_file": str(fake_file), "sample_rate": 24000}},
    )
    with patch("qwen3_tts.server.app.cleanup_pid_file"):
        cleanup_resources(state)
    assert not fake_file.exists()
    assert len(state.gen_cache) == 0


# ---------------------------------------------------------------------------
# Public endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
def test_health_models_loading(fastapi_client):
    """GET /health returns 503 when models not yet loaded."""
    app.state.models_loaded.clear()
    response = fastapi_client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "loading"


@pytest.mark.unit
@_skip
def test_health_models_loaded(fastapi_client):
    """GET /health returns 200 when models loaded."""
    app.state.models_loaded.set()
    response = fastapi_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "backend" in data


@pytest.mark.unit
@_skip
def test_ready_not_loaded(fastapi_client):
    """GET /ready returns 503 when models not loaded."""
    app.state.models_loaded.clear()
    response = fastapi_client.get("/ready")
    assert response.status_code == 503


@pytest.mark.unit
@_skip
def test_ready_loaded(fastapi_client):
    """GET /ready returns 200 when models loaded."""
    app.state.models_loaded.set()
    response = fastapi_client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.unit
@_skip
def test_generation_status_inactive(fastapi_client):
    """GET /generation-status shows inactive when no generation."""
    response = fastapi_client.get("/generation-status")
    assert response.status_code == 200
    assert response.json()["active"] is False


@pytest.mark.unit
@_skip
def test_generation_status_active(fastapi_client):
    """GET /generation-status shows elapsed_sec when active."""
    app.state.generation_state["active"] = True
    app.state.generation_state["start_time"] = time.time() - 5
    app.state.generation_state["text_length"] = 100
    app.state.generation_state["mode"] = "clone"
    try:
        response = fastapi_client.get("/generation-status")
        data = response.json()
        assert data["active"] is True
        assert data["elapsed_sec"] >= 4  # at least 4 seconds
    finally:
        app.state.generation_state["active"] = False
        app.state.generation_state["start_time"] = 0.0


@pytest.mark.unit
@_skip
def test_queue_status_empty(fastapi_client):
    """GET /queue-status returns empty queue."""
    import asyncio
    if not hasattr(app.state, 'pending_lock'):
        app.state.pending_lock = asyncio.Lock()
    if not hasattr(app.state, 'pending_requests'):
        app.state.pending_requests = []
    response = fastapi_client.get("/queue-status")
    assert response.status_code == 200
    data = response.json()
    assert data["queue_length"] == 0
    assert data["active"] is False


# ---------------------------------------------------------------------------
# Admin endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
def test_stats_returns_expected_keys(fastapi_client):
    """GET /stats returns expected fields."""
    app.state.models_loaded.set()
    cache_info = MagicMock()
    cache_info.currsize = 3
    cache_info.hits = 10
    with patch("qwen3_tts.core.engine.voice_prompt_cache_info", return_value=cache_info):
        response = fastapi_client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["voice_prompts_cached"] == 3
    assert data["voice_prompts_cache_hits"] == 10
    assert "idle_seconds" in data


@pytest.mark.unit
@_skip
def test_cancel_generation_no_active(fastapi_client):
    """POST /cancel-generation when no active generation."""
    response = fastapi_client.post("/cancel-generation")
    assert response.status_code == 200
    assert response.json()["status"] == "no_active_generation"


@pytest.mark.unit
@_skip
def test_cancel_generation_active(fastapi_client):
    """POST /cancel-generation when generation is active."""
    import asyncio
    app.state.generation_lock = asyncio.Lock()
    app.state.generation_state["active"] = True
    app.state.generation_state["generation_id"] = "test123"
    try:
        response = fastapi_client.post("/cancel-generation")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancellation_requested"
        assert data["generation_id"] == "test123"
        assert app.state.generation_state["cancelled"] is True
    finally:
        app.state.generation_state["active"] = False
        app.state.generation_state["cancelled"] = False
        app.state.generation_state["generation_id"] = None


# ---------------------------------------------------------------------------
# Prompt endpoint tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
def test_prompts_mlx_backend(fastapi_client):
    """GET /prompts lists MLX-style prompts (wav+txt pairs)."""
    with patch("qwen3_tts.server.app.get_backend", return_value="mlx"), \
         patch("qwen3_tts.server.app_prompts.os.listdir",
               return_value=["alice.wav", "alice.txt", "bob.wav", "bob.txt", "orphan.wav"]):
        response = fastapi_client.get("/prompts")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert "alice.wav" in data["prompts"]
    assert "bob.wav" in data["prompts"]


@pytest.mark.unit
@_skip
def test_prompts_torch_backend(fastapi_client):
    """GET /prompts lists torch-style prompts (.pt files)."""
    with patch("qwen3_tts.server.app.get_backend", return_value="torch"), \
         patch("qwen3_tts.server.app_prompts.os.listdir",
               return_value=["alice.pt", "bob.pt", "readme.txt"]):
        response = fastapi_client.get("/prompts")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert "alice.pt" in data["prompts"]
    assert "bob.pt" in data["prompts"]


@pytest.mark.unit
@_skip
def test_prompts_pagination(fastapi_client):
    """GET /prompts supports offset and limit."""
    with patch("qwen3_tts.server.app.get_backend", return_value="torch"), \
         patch("qwen3_tts.server.app_prompts.os.listdir",
               return_value=["a.pt", "b.pt", "c.pt", "d.pt"]):
        response = fastapi_client.get("/prompts?offset=1&limit=2")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 4
    assert len(data["prompts"]) == 2


@pytest.mark.unit
@_skip
def test_prompts_empty_dir(fastapi_client):
    """GET /prompts returns empty when VOICE_PROMPTS_DIR fails."""
    with patch("qwen3_tts.server.app_prompts.os.listdir", side_effect=OSError("no dir")):
        response = fastapi_client.get("/prompts")
    assert response.status_code == 200
    assert response.json()["prompts"] == []


@pytest.mark.unit
@_skip
def test_delete_prompt_success(fastapi_client):
    """POST /delete-prompt removes matching files."""
    with patch("qwen3_tts.server.app_prompts.os.path.exists", side_effect=lambda p: p.endswith(".wav") or p.endswith(".txt")), \
         patch("qwen3_tts.server.app_prompts.os.remove") as mock_rm, \
         patch("qwen3_tts.server.app._get_app_config", return_value={"default_clone_prompt": ""}), \
         patch("qwen3_tts.core.engine.clear_voice_prompt_cache"):
        response = fastapi_client.post(
            "/delete-prompt", json={"name": "alice"}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    assert mock_rm.call_count == 2  # .wav and .txt


@pytest.mark.unit
@_skip
def test_delete_prompt_not_found(fastapi_client):
    """POST /delete-prompt returns 404 for missing prompt."""
    with patch("qwen3_tts.server.app_prompts.os.path.exists", return_value=False):
        response = fastapi_client.post(
            "/delete-prompt", json={"name": "nonexistent"}
        )
    assert response.status_code == 404


@pytest.mark.unit
@_skip
def test_delete_prompt_invalid_name(fastapi_client):
    """POST /delete-prompt rejects empty name."""
    response = fastapi_client.post(
        "/delete-prompt", json={"name": ""}
    )
    assert response.status_code in (400, 422)


@pytest.mark.unit
@_skip
def test_rename_prompt_same_name(fastapi_client):
    """POST /rename-prompt rejects when old==new."""
    response = fastapi_client.post(
        "/rename-prompt", json={"old_name": "alice", "new_name": "alice"}
    )
    assert response.status_code == 400
    assert "same" in response.json()["detail"].lower()


@pytest.mark.unit
@_skip
def test_rename_prompt_collision(fastapi_client):
    """POST /rename-prompt returns 409 when new name exists."""
    with patch("qwen3_tts.server.app_prompts.os.path.exists", return_value=True):
        response = fastapi_client.post(
            "/rename-prompt", json={"old_name": "alice", "new_name": "bob"}
        )
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


@pytest.mark.unit
@_skip
def test_rename_prompt_not_found(fastapi_client):
    """POST /rename-prompt returns 404 when old prompt not found."""
    # First call (collision check for .pt): False, .wav: False, .txt: False
    # Then old_exists check: all False
    with patch("qwen3_tts.server.app_prompts.os.path.exists", return_value=False):
        response = fastapi_client.post(
            "/rename-prompt", json={"old_name": "gone", "new_name": "newname"}
        )
    assert response.status_code == 404


@pytest.mark.unit
@_skip
def test_preview_prompt_not_found(fastapi_client):
    """GET /preview-prompt returns 404 when .wav not found."""
    with patch("qwen3_tts.server.app_prompts.os.path.exists", return_value=False), \
         patch("qwen3_tts.server.app_prompts.os.path.realpath", side_effect=lambda p: p):
        response = fastapi_client.get("/preview-prompt?name=missing")
    assert response.status_code == 404


@pytest.mark.unit
@_skip
def test_preview_prompt_symlink_traversal(fastapi_client):
    """GET /preview-prompt rejects symlinks outside prompts dir."""
    with patch("qwen3_tts.server.app_prompts.os.path.realpath",
               side_effect=lambda p: "/etc/passwd" if "voice" in str(p) else p), \
         patch("qwen3_tts.server.app_prompts.os.path.exists", return_value=True):
        response = fastapi_client.get("/preview-prompt?name=evil")
    assert response.status_code == 403


@pytest.mark.unit
@_skip
def test_prompt_details_not_found(fastapi_client):
    """GET /prompt-details returns 404 for missing prompt."""
    with patch("qwen3_tts.server.app_prompts.os.path.exists", return_value=False), \
         patch("qwen3_tts.server.app_prompts.os.path.getsize", return_value=0), \
         patch("qwen3_tts.server.app_prompts.os.path.getmtime", return_value=0):
        response = fastapi_client.get("/prompt-details?name=missing")
    assert response.status_code == 404


@pytest.mark.unit
@_skip
def test_prompt_details_all_prompts(fastapi_client):
    """GET /prompt-details without name returns all prompts."""
    fake_files = ["alice.pt", "alice.wav", "alice.txt", "bob.pt"]
    with patch("qwen3_tts.server.app_prompts.os.listdir", return_value=fake_files), \
         patch("qwen3_tts.server.app_prompts.os.path.exists", return_value=True), \
         patch("qwen3_tts.server.app_prompts.os.path.getsize", return_value=1000), \
         patch("qwen3_tts.server.app_prompts.os.path.getmtime", return_value=1000000.0), \
         patch("qwen3_tts.server.app_prompts.get_default_clone_prompt", return_value="alice.pt"):
        response = fastapi_client.get("/prompt-details")
    assert response.status_code == 200
    data = response.json()
    assert len(data["prompts"]) == 2  # alice and bob


# ---------------------------------------------------------------------------
# Route existence (kept from original)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@_skip
def test_app_exists():
    """FastAPI app should be importable."""
    from fastapi import FastAPI
    assert isinstance(app, FastAPI)


@pytest.mark.unit
@_skip
def test_app_has_routes():
    """App should have expected routes."""
    routes = [route.path for route in app.routes]
    assert "/health" in routes
    assert "/generation-status" in routes
    assert "/shutdown" in routes
    assert "/stats" in routes
    assert "/models" in routes


@pytest.mark.unit
@_skip
def test_app_has_lifespan():
    """App should have lifespan configured."""
    assert app.router.lifespan_context is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
