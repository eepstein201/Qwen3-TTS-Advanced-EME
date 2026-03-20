#!/usr/bin/env python3
"""Extended FastAPI server tests covering success paths.

Covers:
  - /load-model success + error variants (ImportError, RuntimeError, etc.)
  - /unload-model success path (cache cleanup, model cleanup)
  - /generate success path (mock inference, cache hit, binary WAV response)
  - /shutdown endpoint
  - /rename-prompt success + rollback + default update
  - /preview-prompt success
  - /prompt-details single + all
  - _background_load function
  - run_server function basics

Run: pytest tests/test_fastapi_app_ext.py -v
"""
import asyncio
import os
import sys
import time
import threading
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

_APP = "qwen3_tts.server.app"

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="requires fastapi")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def fastapi_client(tmp_path):
    """Minimal FastAPI test client with auth, using tmp_path for voice_prompts."""
    from qwen3_tts.server.app import app

    token = "test_tok"
    app.state.auth_token = token
    app.state.models = {"clone": None, "design": None, "custom": None}
    app.state.model_load_times = {}
    app.state.generation_lock = AsyncMock()
    app.state.generation_lock.__aenter__.return_value = None
    app.state.generation_lock.__aexit__.return_value = None
    app.state.generation_state = {
        "active": False, "start_time": 0.0, "text_length": 0,
        "mode": "", "batch_index": 0, "batch_total": 0,
        "chunk_index": 0, "chunk_total": 0,
        "generation_id": None, "cancelled": False,
    }
    app.state.request_queue = set()
    app.state.request_queue_lock = threading.Lock()
    app.state.last_activity = 0
    app.state.models_loaded = threading.Event()
    app.state.gen_cache = {}
    app.state.gen_cache_lock = threading.Lock()
    app.state.inference_lock = asyncio.Lock()
    app.state.eta_cache = {"median_rate": None, "last_updated": 0}
    app.state.model_load_errors = {"clone": None, "design": None, "custom": None}
    app.state.shutdown_timer = None
    app.state.pending_lock = asyncio.Lock()
    app.state.pending_requests = []
    app.state.server_config = {
        "security": {"max_text_length": 10000, "max_batch_size": 20},
        "models": {},
    }

    raw = TestClient(app)

    class _AuthClient:
        def __init__(self, c, t):
            self._c, self._t = c, t

        def get(self, path, **kw):
            h = kw.pop("headers", {})
            h["Authorization"] = f"Bearer {self._t}"
            return self._c.get(path, headers=h, **kw)

        def post(self, path, **kw):
            h = kw.pop("headers", {})
            h["Authorization"] = f"Bearer {self._t}"
            return self._c.post(path, headers=h, **kw)

    yield _AuthClient(raw, token), app, tmp_path


# ---------------------------------------------------------------------------
# /load-model success + error paths
# ---------------------------------------------------------------------------

class TestLoadModelSuccess:

    def test_load_success(self, fastapi_client):
        client, app, _ = fastapi_client
        mock_model = MagicMock()
        info = {"name": "TestModel", "description": "Test"}
        with patch("qwen3_tts.core.engine.load_model", return_value=mock_model), \
             patch("qwen3_tts.core.config.get_model_info", return_value=info):
            resp = client.post("/load-model", json={"model_type": "clone"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "loaded"
        assert app.state.models["clone"] is mock_model

    def test_load_import_error(self, fastapi_client):
        client, app, _ = fastapi_client
        info = {"name": "TestModel", "description": "Test"}
        with patch("qwen3_tts.core.engine.load_model", side_effect=ImportError("no mlx")), \
             patch("qwen3_tts.core.config.get_model_info", return_value=info):
            resp = client.post("/load-model", json={"model_type": "design"})
        # _error_response raises HTTPException → 500
        assert resp.status_code == 500
        assert app.state.model_load_errors["design"] is not None

    def test_load_runtime_error(self, fastapi_client):
        client, app, _ = fastapi_client
        info = {"name": "TestModel", "description": "Test"}
        with patch("qwen3_tts.core.engine.load_model", side_effect=RuntimeError("OOM")), \
             patch("qwen3_tts.core.config.get_model_info", return_value=info):
            resp = client.post("/load-model", json={"model_type": "custom"})
        assert resp.status_code == 500
        assert app.state.model_load_errors["custom"] is not None

    def test_load_unexpected_error(self, fastapi_client):
        client, app, _ = fastapi_client
        info = {"name": "TestModel", "description": "Test"}
        with patch("qwen3_tts.core.engine.load_model", side_effect=TypeError("weird")), \
             patch("qwen3_tts.core.config.get_model_info", return_value=info):
            resp = client.post("/load-model", json={"model_type": "clone"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# /unload-model success path
# ---------------------------------------------------------------------------

class TestUnloadModelSuccess:

    def test_unload_success(self, fastapi_client):
        client, app, _ = fastapi_client
        app.state.models["clone"] = MagicMock()
        app.state.model_load_times["clone"] = 5.0
        with patch("qwen3_tts.core.engine.unload_model_cleanup"):
            resp = client.post("/unload-model", json={"model_type": "clone"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"
        assert app.state.models["clone"] is None
        assert "clone" not in app.state.model_load_times

    def test_unload_clears_gen_cache(self, fastapi_client, tmp_path):
        client, app, _ = fastapi_client
        app.state.models["design"] = MagicMock()
        # Create a cache file
        cache_file = tmp_path / "cached.wav"
        cache_file.write_text("data")
        app.state.gen_cache = {"key1": {"main_file": str(cache_file), "sample_rate": 24000}}
        with patch("qwen3_tts.core.engine.unload_model_cleanup"):
            resp = client.post("/unload-model", json={"model_type": "design"})
        assert resp.status_code == 200
        assert len(app.state.gen_cache) == 0
        assert not cache_file.exists()


# ---------------------------------------------------------------------------
# /generate success path
# ---------------------------------------------------------------------------

class TestGenerateSuccess:

    def test_generate_clone_success(self, fastapi_client):
        """Test the full generate success path with mocked inference."""
        client, app, _ = fastapi_client
        import numpy as np
        mock_model = MagicMock()
        app.state.models["clone"] = mock_model
        app.state.models_loaded.set()

        # Create a small WAV-like array
        wav = np.zeros(4800, dtype=np.float32)
        sr = 24000

        with patch(f"{_APP}._check_memory_available", return_value=(True, 4000)), \
             patch("qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock()), \
             patch("qwen3_tts.core.engine.run_inference", return_value=(wav, sr)), \
             patch(f"{_APP}._gen_cache_key", return_value="test_key"), \
             patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test_cache.wav"
            mock_tmp.return_value = mock_file
            with patch("os.chmod"), \
                 patch("soundfile.write"):
                resp = client.post("/generate", json={
                    "text": "Hello world",
                    "mode": "clone",
                    "prompt_file": "voice.wav",
                })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert "audio_base64" in data["results"][0]

    def test_generate_cache_hit_pre_lock(self, fastapi_client, tmp_path):
        """Test full cache hit (pre-lock) skips inference entirely."""
        client, app, _ = fastapi_client
        mock_model = MagicMock()
        app.state.models["clone"] = mock_model

        # Put a file in gen_cache
        cache_file = tmp_path / "cached.wav"
        cache_file.write_bytes(b"RIFF" + b"\x00" * 100)
        app.state.gen_cache = {"test_key": {
            "main_file": str(cache_file),
            "sample_rate": 24000,
            "timestamp": time.time(),
        }}

        with patch(f"{_APP}._check_memory_available", return_value=(True, 4000)), \
             patch(f"{_APP}._gen_cache_key", return_value="test_key"):
            resp = client.post("/generate", json={
                "text": "Hello cached",
                "mode": "clone",
                "prompt_file": "voice.wav",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1
        assert "audio_base64" in data["results"][0]

    def test_generate_design_mode(self, fastapi_client):
        """Test design mode generation (no prompt_file needed)."""
        client, app, _ = fastapi_client
        import numpy as np
        mock_model = MagicMock()
        app.state.models["design"] = mock_model

        wav = np.zeros(4800, dtype=np.float32)
        sr = 24000

        with patch(f"{_APP}._check_memory_available", return_value=(True, 4000)), \
             patch("qwen3_tts.core.engine.run_inference", return_value=(wav, sr)), \
             patch(f"{_APP}._gen_cache_key", return_value="design_key"), \
             patch("tempfile.NamedTemporaryFile") as mock_tmp:
            mock_file = MagicMock()
            mock_file.name = "/tmp/design_cache.wav"
            mock_tmp.return_value = mock_file
            with patch("os.chmod"), patch("soundfile.write"):
                resp = client.post("/generate", json={
                    "text": "Hello world",
                    "mode": "design",
                    "voice_description": "warm female voice",
                })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1

    def test_generate_inference_error(self, fastapi_client):
        """Test error during inference returns 500."""
        client, app, _ = fastapi_client
        app.state.models["clone"] = MagicMock()

        with patch(f"{_APP}._check_memory_available", return_value=(True, 4000)), \
             patch("qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock()), \
             patch("qwen3_tts.core.engine.run_inference", side_effect=RuntimeError("CUDA OOM")), \
             patch(f"{_APP}._gen_cache_key", return_value="err_key"):
            resp = client.post("/generate", json={
                "text": "Hello",
                "mode": "clone",
                "prompt_file": "voice.wav",
            })
        assert resp.status_code == 500

    def test_generate_clone_no_prompt(self, fastapi_client):
        """Test clone mode without prompt_file raises 400."""
        client, app, _ = fastapi_client
        app.state.models["clone"] = MagicMock()

        with patch(f"{_APP}._check_memory_available", return_value=(True, 4000)), \
             patch(f"{_APP}._gen_cache_key", return_value="nope"):
            resp = client.post("/generate", json={
                "text": "Hello",
                "mode": "clone",
            })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /shutdown endpoint
# ---------------------------------------------------------------------------

class TestShutdownEndpoint:

    def test_shutdown_returns_json(self, fastapi_client):
        client, app, _ = fastapi_client
        app.state.shutdown_timer = None
        # Prevent the background task from actually sending SIGTERM
        with patch("os.kill"), \
             patch(f"{_APP}.cleanup_pid_file"), \
             patch(f"{_APP}.cleanup_resources"), \
             patch("os.path.exists", return_value=False):
            resp = client.post("/shutdown")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "shutting_down"

    def test_shutdown_cancels_timer(self, fastapi_client):
        client, app, _ = fastapi_client
        mock_timer = MagicMock()
        app.state.shutdown_timer = mock_timer
        with patch("os.kill"), \
             patch(f"{_APP}.cleanup_pid_file"), \
             patch(f"{_APP}.cleanup_resources"), \
             patch("os.path.exists", return_value=False):
            resp = client.post("/shutdown")
        assert resp.status_code == 200
        mock_timer.cancel.assert_called_once()


# ---------------------------------------------------------------------------
# /rename-prompt success + rollback + default update
# ---------------------------------------------------------------------------

class TestRenamePromptSuccess:

    def test_rename_success(self, fastapi_client):
        client, app, tmp_path = fastapi_client
        # Create prompt files
        wav_path = tmp_path / "old_voice.wav"
        txt_path = tmp_path / "old_voice.txt"
        wav_path.write_text("audio")
        txt_path.write_text("transcript")

        with patch(f"{_APP}.VOICE_PROMPTS_DIR", str(tmp_path)), \
             patch("qwen3_tts.core.engine.clear_voice_prompt_cache"), \
             patch(f"{_APP}._get_app_config", return_value={"default_clone_prompt": ""}), \
             patch(f"{_APP}.save_config"):
            resp = client.post("/rename-prompt", json={
                "old_name": "old_voice",
                "new_name": "new_voice",
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "renamed"
        assert (tmp_path / "new_voice.wav").exists()
        assert (tmp_path / "new_voice.txt").exists()
        assert not wav_path.exists()

    def test_rename_updates_default(self, fastapi_client):
        """When the renamed prompt was the default, config is updated."""
        client, app, tmp_path = fastapi_client
        (tmp_path / "my_voice.wav").write_text("audio")
        (tmp_path / "my_voice.txt").write_text("text")
        saved_config = {}

        def _save(cfg):
            saved_config.update(cfg)

        with patch(f"{_APP}.VOICE_PROMPTS_DIR", str(tmp_path)), \
             patch("qwen3_tts.core.engine.clear_voice_prompt_cache"), \
             patch(f"{_APP}._get_app_config", return_value={"default_clone_prompt": "my_voice"}), \
             patch(f"{_APP}.save_config", side_effect=_save):
            resp = client.post("/rename-prompt", json={
                "old_name": "my_voice",
                "new_name": "renamed_voice",
            })
        assert resp.status_code == 200
        assert saved_config.get("default_clone_prompt") == "renamed_voice"

    def test_rename_rollback_on_failure(self, fastapi_client):
        """If rename fails mid-way, already-renamed files are rolled back."""
        client, app, tmp_path = fastapi_client
        (tmp_path / "voice.wav").write_text("audio")
        (tmp_path / "voice.txt").write_text("text")

        call_count = [0]
        original_rename = os.rename

        def _failing_rename(src, dst):
            call_count[0] += 1
            if call_count[0] == 2:
                raise OSError("disk full")
            original_rename(src, dst)

        with patch(f"{_APP}.VOICE_PROMPTS_DIR", str(tmp_path)), \
             patch("os.rename", side_effect=_failing_rename):
            resp = client.post("/rename-prompt", json={
                "old_name": "voice",
                "new_name": "voice2",
            })
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# /preview-prompt success
# ---------------------------------------------------------------------------

class TestPreviewPromptSuccess:

    def test_preview_returns_wav(self, fastapi_client):
        client, app, tmp_path = fastapi_client
        wav_path = tmp_path / "my_voice.wav"
        wav_path.write_bytes(b"RIFF" + b"\x00" * 40)

        with patch(f"{_APP}.VOICE_PROMPTS_DIR", str(tmp_path)):
            resp = client.get("/preview-prompt?name=my_voice")
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("audio/wav")


# ---------------------------------------------------------------------------
# /prompt-details success
# ---------------------------------------------------------------------------

class TestPromptDetailsSuccess:

    def test_single_prompt_details(self, fastapi_client):
        client, app, tmp_path = fastapi_client
        wav = tmp_path / "test_voice.wav"
        txt = tmp_path / "test_voice.txt"
        wav.write_bytes(b"RIFF" + b"\x00" * 40)
        txt.write_text("transcript")

        with patch(f"{_APP}.VOICE_PROMPTS_DIR", str(tmp_path)), \
             patch(f"{_APP}.get_default_clone_prompt", return_value="test_voice"):
            resp = client.get("/prompt-details?name=test_voice")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test_voice"
        assert ".wav" in data["formats"]
        assert ".txt" in data["formats"]
        assert data["is_default"] is True

    def test_all_prompt_details(self, fastapi_client):
        client, app, tmp_path = fastapi_client
        (tmp_path / "a.wav").write_text("audio")
        (tmp_path / "a.txt").write_text("text")
        (tmp_path / "b.pt").write_text("model")

        with patch(f"{_APP}.VOICE_PROMPTS_DIR", str(tmp_path)), \
             patch(f"{_APP}.get_default_clone_prompt", return_value="a"):
            resp = client.get("/prompt-details")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["prompts"]) == 2

    def test_prompt_details_oserror(self, fastapi_client):
        client, _, _ = fastapi_client
        with patch(f"{_APP}.VOICE_PROMPTS_DIR", "/nonexistent_dir_xyz"), \
             patch(f"{_APP}.get_default_clone_prompt", return_value=""):
            resp = client.get("/prompt-details")
        assert resp.status_code == 200
        assert resp.json()["prompts"] == []


# ---------------------------------------------------------------------------
# _background_load
# ---------------------------------------------------------------------------

class TestBackgroundLoad:

    def test_loads_configured_models(self):
        from qwen3_tts.server.app import _background_load

        app_state = MagicMock()
        app_state.server_config = {
            "models": {
                "clone": {"load_at_startup": True},
                "design": {"load_at_startup": False},
            }
        }
        app_state.models = {"clone": None, "design": None, "custom": None}
        app_state.model_load_times = {}
        app_state.model_load_errors = {"clone": None, "design": None, "custom": None}

        mock_model = MagicMock()
        info = {"name": "TestModel"}
        with patch("qwen3_tts.core.engine.load_model", return_value=mock_model), \
             patch("qwen3_tts.core.config.get_model_info", return_value=info), \
             patch(f"{_APP}.get_backend", return_value="mlx"), \
             patch("qwen3_tts.core.engine.migrate_orphan_mlx_prompts"):
            _background_load(app_state)

        assert app_state.models["clone"] is mock_model
        assert app_state.models["design"] is None
        app_state.models_loaded.set.assert_called_once()

    def test_handles_load_error(self):
        from qwen3_tts.server.app import _background_load

        app_state = MagicMock()
        app_state.server_config = {
            "models": {"clone": {"load_at_startup": True}}
        }
        app_state.models = {"clone": None, "design": None, "custom": None}
        app_state.model_load_times = {}
        app_state.model_load_errors = {"clone": None, "design": None, "custom": None}

        with patch("qwen3_tts.core.engine.load_model", side_effect=RuntimeError("OOM")), \
             patch("qwen3_tts.core.config.get_model_info", return_value={"name": "M"}), \
             patch(f"{_APP}.get_backend", return_value="mlx"), \
             patch("qwen3_tts.core.engine.migrate_orphan_mlx_prompts"):
            _background_load(app_state)

        assert app_state.model_load_errors["clone"] is not None
        app_state.models_loaded.set.assert_called_once()

    def test_no_models_configured(self):
        from qwen3_tts.server.app import _background_load

        app_state = MagicMock()
        app_state.server_config = {"models": {}}
        app_state.models = {"clone": None, "design": None, "custom": None}
        app_state.model_load_times = {}
        app_state.model_load_errors = {"clone": None, "design": None, "custom": None}

        with patch(f"{_APP}.get_backend", return_value="mlx"), \
             patch("qwen3_tts.core.engine.migrate_orphan_mlx_prompts"):
            _background_load(app_state)

        app_state.models_loaded.set.assert_called_once()

    def test_torch_backend_migration(self):
        from qwen3_tts.server.app import _background_load

        app_state = MagicMock()
        app_state.server_config = {"models": {}}
        app_state.models = {"clone": None, "design": None, "custom": None}
        app_state.model_load_times = {}
        app_state.model_load_errors = {"clone": None, "design": None, "custom": None}

        with patch(f"{_APP}.get_backend", return_value="torch"), \
             patch("qwen3_tts.core.engine.migrate_orphan_mlx_prompts") as mock_migrate:
            _background_load(app_state)

        mock_migrate.assert_called_once()


# ---------------------------------------------------------------------------
# run_server basics
# ---------------------------------------------------------------------------

class TestRunServer:

    def test_run_server_public_binds_all(self):
        from qwen3_tts.server.app import run_server
        with patch(f"{_APP}.uvicorn") as mock_uv, \
             patch(f"{_APP}.IN_COLAB", False), \
             patch("signal.signal"), \
             patch("builtins.print"):
            run_server(host="127.0.0.1", port=5123, public=True)
        call_kw = mock_uv.run.call_args
        assert call_kw[1]["host"] == "0.0.0.0"

    def test_run_server_colab_binds_all(self):
        from qwen3_tts.server.app import run_server
        with patch(f"{_APP}.uvicorn") as mock_uv, \
             patch(f"{_APP}.IN_COLAB", True), \
             patch("signal.signal"), \
             patch("builtins.print"):
            run_server(host="127.0.0.1", port=5123, public=False)
        call_kw = mock_uv.run.call_args
        assert call_kw[1]["host"] == "0.0.0.0"


# ---------------------------------------------------------------------------
# /delete-prompt clears default config
# ---------------------------------------------------------------------------

class TestDeletePromptDefaultClear:

    def test_delete_clears_default(self, fastapi_client):
        """When deleted prompt was the default, config.default_clone_prompt is cleared."""
        client, app, tmp_path = fastapi_client
        (tmp_path / "def_voice.wav").write_text("audio")
        (tmp_path / "def_voice.txt").write_text("text")
        saved_config = {}

        def _save(cfg):
            saved_config.update(cfg)

        with patch(f"{_APP}.VOICE_PROMPTS_DIR", str(tmp_path)), \
             patch("qwen3_tts.core.engine.clear_voice_prompt_cache"), \
             patch(f"{_APP}._get_app_config", return_value={"default_clone_prompt": "def_voice"}), \
             patch(f"{_APP}.save_config", side_effect=_save):
            resp = client.post("/delete-prompt", json={"name": "def_voice"})
        assert resp.status_code == 200
        assert saved_config.get("default_clone_prompt") == ""


# ---------------------------------------------------------------------------
# /list-models (stats endpoint with model info)
# ---------------------------------------------------------------------------

class TestListModels:

    def test_list_models_with_loaded(self, fastapi_client):
        client, app, _ = fastapi_client
        app.state.models["clone"] = MagicMock()
        app.state.model_load_times["clone"] = 3.5
        app.state.models_loaded.set()

        info = {"name": "TestModel", "description": "Test", "memory_mb": 2500}
        with patch("qwen3_tts.core.config.get_model_info", return_value=info), \
             patch(f"{_APP}.get_backend", return_value="mlx"):
            resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
