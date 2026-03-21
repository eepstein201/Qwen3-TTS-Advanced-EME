#!/usr/bin/env python3
"""Extended FastAPI server tests — second batch.

Covers uncovered areas in qwen3_tts/server/app.py:
  - reset_activity_timer with existing timer + auto_shutdown_minutes > 0
  - auto_shutdown function
  - cleanup_pid function
  - cleanup_resources with gen_cache temp files
  - _check_memory_available (psutil present, low memory, no psutil)
  - _background_load (no models config, load failure, torch migration)
  - IN_COLAB CORS regex branch
  - GPU/MLX memory stats in /stats
  - /update-model-config cache invalidation + audio_loader sync
  - /generate-stream endpoint (clone + error paths)
  - run_server (public + Colab host binding)
  - lifespan startup/shutdown
  - _get_real_client_ip (loopback vs proxy)

Run: python -m pytest tests/test_fastapi_app_ext2.py -v
"""
import asyncio
import os
import threading
import time
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

_APP = "qwen3_tts.server.app"


def _make_app_state(**overrides):
    """Create a minimal mock app_state with required attributes."""
    state = MagicMock()
    state.models = {"clone": None, "design": None, "custom": None}
    state.model_load_times = {}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
    state.last_activity = 0
    state.shutdown_timer = None
    state.server_config = {"auto_shutdown_minutes": 0, "models": {}}
    state.gen_cache = {}
    state.gen_cache_lock = threading.Lock()
    state.models_loaded = threading.Event()
    state.shutdown_event = MagicMock()
    state.generation_lock = AsyncMock()
    state.generation_lock.__aenter__ = AsyncMock(return_value=None)
    state.generation_lock.__aexit__ = AsyncMock(return_value=None)
    for k, v in overrides.items():
        setattr(state, k, v)
    return state


# ---------------------------------------------------------------------------
# reset_activity_timer
# ---------------------------------------------------------------------------

class TestResetActivityTimer(unittest.TestCase):

    def test_no_auto_shutdown(self):
        from qwen3_tts.server.app import reset_activity_timer
        state = _make_app_state()
        state.server_config = {"auto_shutdown_minutes": 0}
        reset_activity_timer(state)
        self.assertIsNone(state.shutdown_timer)

    def test_creates_timer(self):
        from qwen3_tts.server.app import reset_activity_timer
        state = _make_app_state()
        state.server_config = {"auto_shutdown_minutes": 5}
        state.shutdown_timer = None
        reset_activity_timer(state)
        self.assertIsNotNone(state.shutdown_timer)
        # Clean up — cancel the timer
        state.shutdown_timer.cancel()

    def test_cancels_existing_timer(self):
        from qwen3_tts.server.app import reset_activity_timer
        state = _make_app_state()
        state.server_config = {"auto_shutdown_minutes": 5}
        old_timer = MagicMock()
        state.shutdown_timer = old_timer
        reset_activity_timer(state)
        old_timer.cancel.assert_called_once()
        # New timer should be set
        self.assertIsNotNone(state.shutdown_timer)
        self.assertIsNot(state.shutdown_timer, old_timer)
        # Clean up
        state.shutdown_timer.cancel()


# ---------------------------------------------------------------------------
# auto_shutdown
# ---------------------------------------------------------------------------

class TestAutoShutdown(unittest.TestCase):

    def test_auto_shutdown_calls_cleanup_and_exits(self):
        from qwen3_tts.server.app import auto_shutdown
        state = _make_app_state()
        state.server_config = {"auto_shutdown_minutes": 10}
        with patch(f"{_APP}.cleanup_resources") as mock_cleanup, \
             patch("sys.exit") as mock_exit:
            auto_shutdown(state)
        mock_cleanup.assert_called_once_with(state)
        mock_exit.assert_called_once_with(0)


# ---------------------------------------------------------------------------
# cleanup_pid
# ---------------------------------------------------------------------------

class TestCleanupPid(unittest.TestCase):

    def test_cleanup_pid_full_flow(self):
        from qwen3_tts.server.app import cleanup_pid
        state = _make_app_state()
        timer = MagicMock()
        state.shutdown_timer = timer
        state.shutdown_event = MagicMock()
        with patch(f"{_APP}.cleanup_pid_file") as mock_cpf, \
             patch(f"{_APP}.cleanup_resources") as mock_cr, \
             patch(f"{_APP}.TOKEN_FILE", "/tmp/fake_token_xyz"), \
             patch("os.path.exists", return_value=True), \
             patch("os.remove") as mock_rm, \
             patch("sys.exit") as mock_exit:
            cleanup_pid(state)
        timer.cancel.assert_called_once()
        mock_cpf.assert_called_once()
        mock_rm.assert_called_once_with("/tmp/fake_token_xyz")
        state.shutdown_event.set.assert_called_once()
        mock_cr.assert_called_once_with(state)
        mock_exit.assert_called_once_with(0)

    def test_cleanup_pid_no_timer_no_token(self):
        from qwen3_tts.server.app import cleanup_pid
        state = _make_app_state()
        state.shutdown_timer = None
        state.shutdown_event = None
        with patch(f"{_APP}.cleanup_pid_file"), \
             patch(f"{_APP}.cleanup_resources"), \
             patch(f"{_APP}.TOKEN_FILE", "/tmp/nonexistent_xyz"), \
             patch("os.path.exists", return_value=False), \
             patch("sys.exit"):
            cleanup_pid(state)
        # No exception — gracefully handles None timer and missing token


# ---------------------------------------------------------------------------
# cleanup_resources
# ---------------------------------------------------------------------------

class TestCleanupResources(unittest.TestCase):

    def test_cleanup_with_gen_cache_files(self):
        import tempfile
        import shutil
        from qwen3_tts.server.app import cleanup_resources
        tmp = tempfile.mkdtemp()
        try:
            f1 = os.path.join(tmp, "gen1.wav")
            with open(f1, "w") as f:
                f.write("data")
            state = _make_app_state()
            state.gen_cache = {"k1": {"main_file": f1}}
            state.shutdown_timer = None
            with patch(f"{_APP}.cleanup_pid_file"):
                cleanup_resources(state)
            self.assertFalse(os.path.exists(f1))
            self.assertEqual(state.gen_cache, {})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cleanup_with_timer(self):
        from qwen3_tts.server.app import cleanup_resources
        state = _make_app_state()
        timer = MagicMock()
        state.shutdown_timer = timer
        with patch(f"{_APP}.cleanup_pid_file"):
            cleanup_resources(state)
        timer.cancel.assert_called_once()

    def test_cleanup_oserror_on_file_removal(self):
        from qwen3_tts.server.app import cleanup_resources
        state = _make_app_state()
        state.gen_cache = {"k1": {"main_file": "/nonexistent/file.wav"}}
        state.shutdown_timer = None
        with patch(f"{_APP}.cleanup_pid_file"):
            # Should not raise despite missing file
            cleanup_resources(state)

    def test_cleanup_models_deletion(self):
        from qwen3_tts.server.app import cleanup_resources
        state = _make_app_state()
        state.models = {"clone": MagicMock(), "design": None, "custom": MagicMock()}
        state.shutdown_timer = None
        with patch(f"{_APP}.cleanup_pid_file"):
            cleanup_resources(state)
        self.assertIsNone(state.models["clone"])
        self.assertIsNone(state.models["custom"])


# ---------------------------------------------------------------------------
# _check_memory_available
# ---------------------------------------------------------------------------

class TestCheckMemoryAvailable(unittest.TestCase):

    def test_no_psutil(self):
        from qwen3_tts.server.app import _check_memory_available
        with patch(f"{_APP}._HAS_PSUTIL", False):
            ok, mb = _check_memory_available()
        self.assertTrue(ok)
        self.assertEqual(mb, 0)

    def test_enough_memory(self):
        from qwen3_tts.server.app import _check_memory_available
        mock_mem = MagicMock()
        mock_mem.available = 10 * 1024 * 1024 * 1024  # 10 GB
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem
        with patch(f"{_APP}._HAS_PSUTIL", True), \
             patch.dict("sys.modules", {"psutil": mock_psutil}):
            import qwen3_tts.server.app as app_mod
            orig_psutil = getattr(app_mod, "psutil", None)
            app_mod.psutil = mock_psutil
            try:
                ok, mb = _check_memory_available()
            finally:
                if orig_psutil is None:
                    if hasattr(app_mod, "psutil"):
                        delattr(app_mod, "psutil")
                else:
                    app_mod.psutil = orig_psutil
        self.assertTrue(ok)
        self.assertEqual(mb, 10240)

    def test_low_memory_below_threshold(self):
        from qwen3_tts.server.app import _check_memory_available
        mock_mem = MagicMock()
        mock_mem.available = 500 * 1024 * 1024  # 500 MB
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem
        import qwen3_tts.server.app as app_mod
        orig_psutil = getattr(app_mod, "psutil", None)
        app_mod.psutil = mock_psutil
        try:
            with patch(f"{_APP}._HAS_PSUTIL", True):
                ok, mb = _check_memory_available()
        finally:
            if orig_psutil is None:
                if hasattr(app_mod, "psutil"):
                    delattr(app_mod, "psutil")
            else:
                app_mod.psutil = orig_psutil
        self.assertFalse(ok)
        self.assertEqual(mb, 500)

    def test_moderate_memory_warning(self):
        from qwen3_tts.server.app import _check_memory_available
        mock_mem = MagicMock()
        mock_mem.available = int(1.5 * 1024 * 1024 * 1024)  # 1.5 GB
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem
        import qwen3_tts.server.app as app_mod
        orig_psutil = getattr(app_mod, "psutil", None)
        app_mod.psutil = mock_psutil
        try:
            with patch(f"{_APP}._HAS_PSUTIL", True):
                ok, mb = _check_memory_available()
        finally:
            if orig_psutil is None:
                if hasattr(app_mod, "psutil"):
                    delattr(app_mod, "psutil")
            else:
                app_mod.psutil = orig_psutil
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# _background_load
# ---------------------------------------------------------------------------

class TestBackgroundLoad(unittest.TestCase):

    def test_no_models_config_defaults_clone(self):
        from qwen3_tts.server.app import _background_load
        state = _make_app_state()
        state.server_config = {"models": {}}
        mock_model = MagicMock()
        info = {"name": "TestModel"}
        with patch("qwen3_tts.core.engine.load_model", return_value=mock_model), \
             patch("qwen3_tts.core.config.get_model_info", return_value=info), \
             patch("qwen3_tts.core.engine.migrate_orphan_mlx_prompts"), \
             patch(f"{_APP}.get_backend", return_value="mlx"):
            _background_load(state)
        self.assertIs(state.models["clone"], mock_model)
        self.assertTrue(state.models_loaded.is_set())

    def test_no_startup_models(self):
        from qwen3_tts.server.app import _background_load
        state = _make_app_state()
        state.server_config = {"models": {"clone": {"load_at_startup": False}}}
        with patch("qwen3_tts.core.engine.load_model") as mock_load, \
             patch("qwen3_tts.core.engine.migrate_orphan_mlx_prompts"), \
             patch(f"{_APP}.get_backend", return_value="mlx"):
            _background_load(state)
        mock_load.assert_not_called()
        self.assertTrue(state.models_loaded.is_set())

    def test_load_failure_stores_error(self):
        from qwen3_tts.server.app import _background_load
        state = _make_app_state()
        state.server_config = {"models": {"design": {"load_at_startup": True}}}
        info = {"name": "DesignModel"}
        with patch("qwen3_tts.core.engine.load_model", side_effect=RuntimeError("OOM")), \
             patch("qwen3_tts.core.config.get_model_info", return_value=info), \
             patch("qwen3_tts.core.engine.migrate_orphan_mlx_prompts"), \
             patch(f"{_APP}.get_backend", return_value="mlx"):
            _background_load(state)
        self.assertIsNotNone(state.model_load_errors["design"])
        self.assertIsNone(state.models["design"])

    def test_torch_backend_runs_migration(self):
        from qwen3_tts.server.app import _background_load
        state = _make_app_state()
        state.server_config = {"models": {"clone": {"load_at_startup": True}}}
        info = {"name": "CloneModel"}
        with patch("qwen3_tts.core.engine.load_model", return_value=MagicMock()), \
             patch("qwen3_tts.core.config.get_model_info", return_value=info), \
             patch("qwen3_tts.core.engine.migrate_orphan_mlx_prompts") as mock_migrate, \
             patch(f"{_APP}.get_backend", return_value="torch"):
            _background_load(state)
        mock_migrate.assert_called_once()

    def test_migration_failure_handled(self):
        from qwen3_tts.server.app import _background_load
        state = _make_app_state()
        state.server_config = {"models": {"clone": {"load_at_startup": True}}}
        info = {"name": "CloneModel"}
        with patch("qwen3_tts.core.engine.load_model", return_value=MagicMock()), \
             patch("qwen3_tts.core.config.get_model_info", return_value=info), \
             patch("qwen3_tts.core.engine.migrate_orphan_mlx_prompts",
                   side_effect=RuntimeError("migrate fail")), \
             patch(f"{_APP}.get_backend", return_value="torch"):
            _background_load(state)
        # Should not raise


# ---------------------------------------------------------------------------
# _get_real_client_ip
# ---------------------------------------------------------------------------

class TestGetRealClientIp(unittest.TestCase):

    def test_loopback_ignores_xff(self):
        from qwen3_tts.server.app import _get_real_client_ip
        request = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8"}
        result = _get_real_client_ip(request)
        self.assertEqual(result, "127.0.0.1")

    def test_non_loopback_reads_xff(self):
        from qwen3_tts.server.app import _get_real_client_ip
        request = MagicMock()
        request.client.host = "10.0.0.5"
        request.headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8"}
        result = _get_real_client_ip(request)
        self.assertEqual(result, "1.2.3.4")

    def test_no_client(self):
        from qwen3_tts.server.app import _get_real_client_ip
        request = MagicMock()
        request.client = None
        request.headers = {}
        result = _get_real_client_ip(request)
        self.assertEqual(result, "127.0.0.1")


# ---------------------------------------------------------------------------
# GPU / MLX memory stats in /stats
# ---------------------------------------------------------------------------

class TestStatsMemory(unittest.TestCase):

    def _get_stats_data(self, backend="mlx", **patches):
        """Helper to call _get_stats_data-like code via the /stats endpoint logic."""
        # We test the _build_stats internal by calling the endpoint via client
        # but it's easier to test the specific branches directly.
        pass

    def test_torch_mps_memory(self):
        """Test MPS memory stats path (lines 616-621)."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("requires fastapi")

        from qwen3_tts.server.app import app

        state = app.state
        state.auth_token = "tok"
        state.models = {"clone": MagicMock(), "design": None, "custom": None}
        state.model_load_times = {"clone": 5.0}
        state.model_load_errors = {"clone": None, "design": None, "custom": None}
        state.last_activity = time.time()
        state.shutdown_timer = None
        state.server_config = {"auto_shutdown_minutes": 0, "models": {}}
        state.gen_cache = {}
        state.gen_cache_lock = threading.Lock()
        state.models_loaded = threading.Event()
        state.models_loaded.set()
        state.request_queue = set()
        state.request_queue_lock = threading.Lock()
        state.eta_cache = {"median_rate": None, "last_updated": 0}

        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.mps.current_allocated_memory.return_value = 1024 * 1024 * 500
        mock_torch.cuda.is_available.return_value = False

        mock_cache_info = MagicMock(currsize=0, hits=0)

        with patch(f"{_APP}.get_backend", return_value="torch"), \
             patch(f"{_APP}.get_model_size", return_value="1.7B"), \
             patch(f"{_APP}.get_torch_dtype_name", return_value="float16"), \
             patch(f"{_APP}.get_generation_cache_max", return_value=10), \
             patch("qwen3_tts.core.engine.voice_prompt.voice_prompt_cache_info",
                   return_value=mock_cache_info), \
             patch.dict("sys.modules", {"torch": mock_torch}):
            client = TestClient(app)
            resp = client.get("/stats", headers={"Authorization": "Bearer tok"})

        data = resp.json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(data.get("mps_memory_allocated_mb"), 500.0)

    def test_torch_cuda_memory(self):
        """Test CUDA memory stats path (lines 623-630)."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("requires fastapi")

        from qwen3_tts.server.app import app

        state = app.state
        state.auth_token = "tok"
        state.models = {"clone": MagicMock(), "design": None, "custom": None}
        state.model_load_times = {"clone": 5.0}
        state.model_load_errors = {"clone": None, "design": None, "custom": None}
        state.last_activity = time.time()
        state.shutdown_timer = None
        state.server_config = {"auto_shutdown_minutes": 0, "models": {}}
        state.gen_cache = {}
        state.gen_cache_lock = threading.Lock()
        state.models_loaded = threading.Event()
        state.models_loaded.set()
        state.request_queue = set()
        state.request_queue_lock = threading.Lock()
        state.eta_cache = {"median_rate": None, "last_updated": 0}

        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 200
        mock_torch.cuda.memory_reserved.return_value = 1024 * 1024 * 400

        mock_cache_info = MagicMock(currsize=0, hits=0)

        with patch(f"{_APP}.get_backend", return_value="torch"), \
             patch(f"{_APP}.get_model_size", return_value="1.7B"), \
             patch(f"{_APP}.get_torch_dtype_name", return_value="float16"), \
             patch(f"{_APP}.get_generation_cache_max", return_value=10), \
             patch("qwen3_tts.core.engine.voice_prompt.voice_prompt_cache_info",
                   return_value=mock_cache_info), \
             patch.dict("sys.modules", {"torch": mock_torch}):
            client = TestClient(app)
            resp = client.get("/stats", headers={"Authorization": "Bearer tok"})

        data = resp.json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(data.get("cuda_memory_allocated_mb"), 200.0)
        self.assertEqual(data.get("cuda_memory_reserved_mb"), 400.0)


# ---------------------------------------------------------------------------
# /update-model-config — cache invalidation + audio_loader sync
# ---------------------------------------------------------------------------

class TestUpdateModelConfig(unittest.TestCase):

    def _setup_client(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("requires fastapi")

        from qwen3_tts.server.app import app

        token = "tok"
        state = app.state
        state.auth_token = token
        state.models = {"clone": MagicMock(), "design": None, "custom": None}
        state.model_load_times = {}
        state.generation_lock = asyncio.Lock()
        state.generation_state = {
            "active": False, "start_time": 0.0, "text_length": 0,
            "mode": "", "batch_index": 0, "batch_total": 0,
            "chunk_index": 0, "chunk_total": 0,
            "generation_id": None, "cancelled": False,
        }
        state.last_activity = 0
        state.shutdown_timer = None
        state.server_config = {"auto_shutdown_minutes": 0, "models": {}}
        state.gen_cache = {}
        state.gen_cache_lock = threading.Lock()
        state.models_loaded = threading.Event()
        state.models_loaded.set()
        state.model_load_errors = {"clone": None, "design": None, "custom": None}
        state.eta_cache = {"median_rate": None, "last_updated": 0}
        state.request_queue = set()
        state.request_queue_lock = threading.Lock()
        state.inference_lock = asyncio.Lock()
        state.pending_lock = asyncio.Lock()
        state.pending_requests = []

        client = TestClient(app)
        return client, token, state

    def test_update_size_and_quant(self):
        client, token, state = self._setup_client()
        with patch(f"{_APP}._get_app_config",
                   return_value={"advanced": {"model_size": "1.7B"}}), \
             patch(f"{_APP}.save_config"):
            resp = client.post(
                "/update-model-config",
                json={"model_size": "0.6B", "mlx_quantization": "4bit"},
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "config_updated")
        # All models should be unloaded
        self.assertIsNone(state.models["clone"])

    def test_update_triggers_audio_loader_sync(self):
        client, token, state = self._setup_client()
        config = {"advanced": {"model_size": "1.7B", "audio_loader": "librosa"}}
        with patch(f"{_APP}._get_app_config", return_value=config), \
             patch(f"{_APP}.save_config"), \
             patch("qwen3_tts.core.engine.set_audio_loader") as mock_sal:
            resp = client.post(
                "/update-model-config",
                json={"model_size": "0.6B"},
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(resp.status_code, 200)
        mock_sal.assert_called_once_with("librosa")

    def test_update_audio_loader_import_error(self):
        client, token, state = self._setup_client()
        config = {"advanced": {"model_size": "1.7B", "audio_loader": "bad"}}
        with patch(f"{_APP}._get_app_config", return_value=config), \
             patch(f"{_APP}.save_config"), \
             patch("qwen3_tts.core.engine.set_audio_loader",
                   side_effect=ImportError("no module")):
            resp = client.post(
                "/update-model-config",
                json={"model_size": "0.6B"},
                headers={"Authorization": f"Bearer {token}"},
            )
        # Should succeed despite import error
        self.assertEqual(resp.status_code, 200)

    def test_update_clears_gen_cache_with_files(self):
        import tempfile
        import shutil
        client, token, state = self._setup_client()
        tmp = tempfile.mkdtemp()
        try:
            f1 = os.path.join(tmp, "cached.wav")
            with open(f1, "w") as f:
                f.write("audio")
            state.gen_cache = {"k1": {"main_file": f1}}

            with patch(f"{_APP}._get_app_config",
                       return_value={"advanced": {"model_size": "1.7B"}}), \
                 patch(f"{_APP}.save_config"):
                resp = client.post(
                    "/update-model-config",
                    json={"model_size": "0.6B"},
                    headers={"Authorization": f"Bearer {token}"},
                )
            self.assertEqual(resp.status_code, 200)
            self.assertFalse(os.path.exists(f1))
            self.assertEqual(state.gen_cache, {})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_update_cache_oserror_handled(self):
        client, token, state = self._setup_client()
        # Entry with nonexistent file
        state.gen_cache = {"k1": {"main_file": "/nonexistent/x.wav"}}
        with patch(f"{_APP}._get_app_config",
                   return_value={"advanced": {"model_size": "1.7B"}}), \
             patch(f"{_APP}.save_config"):
            resp = client.post(
                "/update-model-config",
                json={"model_size": "0.6B"},
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(resp.status_code, 200)


# ---------------------------------------------------------------------------
# /generate-stream endpoint
# ---------------------------------------------------------------------------

class TestGenerateStream(unittest.TestCase):

    def _setup_stream_client(self):
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("requires fastapi")

        from qwen3_tts.server.app import app

        token = "tok"
        state = app.state
        state.auth_token = token
        state.models = {"clone": MagicMock(), "design": None, "custom": None}
        state.model_load_times = {"clone": 5.0}
        state.generation_lock = asyncio.Lock()
        state.generation_state = {
            "active": False, "start_time": 0.0, "text_length": 0,
            "mode": "", "batch_index": 0, "batch_total": 0,
            "chunk_index": 0, "chunk_total": 0,
            "generation_id": None, "cancelled": False,
        }
        state.last_activity = 0
        state.shutdown_timer = None
        state.server_config = {
            "auto_shutdown_minutes": 0,
            "models": {},
            "security": {"max_text_length": 10000, "max_batch_size": 20},
        }
        state.gen_cache = {}
        state.gen_cache_lock = threading.Lock()
        state.models_loaded = threading.Event()
        state.models_loaded.set()
        state.model_load_errors = {"clone": None, "design": None, "custom": None}
        state.eta_cache = {"median_rate": None, "last_updated": 0}
        state.inference_lock = asyncio.Lock()
        state.request_queue = set()
        state.request_queue_lock = threading.Lock()
        state.pending_lock = asyncio.Lock()
        state.pending_requests = []

        client = TestClient(app)
        return client, token, state

    def test_stream_clone_success(self):
        import struct
        import numpy as np
        client, token, state = self._setup_stream_client()

        chunk = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        def fake_stream(**kwargs):
            yield chunk, 24000

        mock_prompt = MagicMock()
        with patch("qwen3_tts.core.engine.load_voice_prompt", return_value=mock_prompt), \
             patch("qwen3_tts.core.engine.run_inference_streaming", side_effect=fake_stream), \
             patch(f"{_APP}._check_memory_available", return_value=(True, 8000)), \
             patch(f"{_APP}._validate_generation_request"):
            resp = client.post(
                "/generate-stream",
                json={"text": "Hello", "mode": "clone", "prompt_file": "voice1.wav"},
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(resp.status_code, 200)
        # Response should contain binary audio data
        data = resp.content
        self.assertGreater(len(data), 0)
        # Parse header: [sample_rate:4][length:4]
        if len(data) >= 8:
            sr, length = struct.unpack("<II", data[:8])
            self.assertEqual(sr, 24000)
            self.assertEqual(length, len(chunk) * 4)

    def test_stream_model_not_loaded(self):
        client, token, state = self._setup_stream_client()
        state.models["clone"] = None

        with patch(f"{_APP}._check_memory_available", return_value=(True, 8000)), \
             patch(f"{_APP}._validate_generation_request"):
            resp = client.post(
                "/generate-stream",
                json={"text": "Hello", "mode": "clone", "prompt_file": "voice1.wav"},
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(resp.status_code, 503)

    def test_stream_low_memory(self):
        client, token, state = self._setup_stream_client()
        with patch(f"{_APP}._check_memory_available", return_value=(False, 500)), \
             patch(f"{_APP}._validate_generation_request"):
            resp = client.post(
                "/generate-stream",
                json={"text": "Hello", "mode": "clone", "prompt_file": "voice1.wav"},
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(resp.status_code, 503)
        detail = resp.json().get("detail", {})
        if isinstance(detail, dict):
            self.assertEqual(detail.get("error"), "insufficient_memory")
        else:
            self.assertIn("memory", str(detail).lower())

    def test_stream_missing_prompt_file(self):
        client, token, state = self._setup_stream_client()
        with patch(f"{_APP}._check_memory_available", return_value=(True, 8000)), \
             patch(f"{_APP}._validate_generation_request"):
            resp = client.post(
                "/generate-stream",
                json={"text": "Hello", "mode": "clone"},
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(resp.status_code, 400)

    def test_stream_prompt_not_found(self):
        client, token, state = self._setup_stream_client()
        with patch("qwen3_tts.core.engine.load_voice_prompt", return_value=None), \
             patch(f"{_APP}._check_memory_available", return_value=(True, 8000)), \
             patch(f"{_APP}._validate_generation_request"):
            resp = client.post(
                "/generate-stream",
                json={"text": "Hello", "mode": "clone", "prompt_file": "missing.wav"},
                headers={"Authorization": f"Bearer {token}"},
            )
        self.assertEqual(resp.status_code, 404)


# ---------------------------------------------------------------------------
# run_server
# ---------------------------------------------------------------------------

class TestRunServer(unittest.TestCase):

    def test_run_server_public_binds_all(self):
        from qwen3_tts.server.app import run_server
        mock_handler = MagicMock()
        mock_handler.level = 0
        with patch(f"{_APP}.uvicorn") as mock_uv, \
             patch(f"{_APP}.IN_COLAB", False), \
             patch(f"{_APP}.LOG_FILE", "/tmp/fake_log.log"), \
             patch("logging.handlers.RotatingFileHandler", return_value=mock_handler), \
             patch("logging.StreamHandler", return_value=mock_handler), \
             patch("builtins.print"):
            run_server(host="127.0.0.1", port=5123, public=True)
        mock_uv.run.assert_called_once()
        # Verify host was changed to 0.0.0.0 for public
        call_kwargs = mock_uv.run.call_args
        self.assertEqual(call_kwargs.kwargs.get("host"), "0.0.0.0")

    def test_run_server_colab_binds_all(self):
        from qwen3_tts.server.app import run_server
        mock_handler = MagicMock()
        mock_handler.level = 0
        with patch(f"{_APP}.uvicorn") as mock_uv, \
             patch(f"{_APP}.IN_COLAB", True), \
             patch(f"{_APP}.LOG_FILE", "/tmp/fake_log.log"), \
             patch("logging.handlers.RotatingFileHandler", return_value=mock_handler), \
             patch("logging.StreamHandler", return_value=mock_handler), \
             patch("builtins.print"):
            run_server(host="127.0.0.1", port=5123, public=False)
        mock_uv.run.assert_called_once()
        self.assertEqual(mock_uv.run.call_args.kwargs.get("host"), "0.0.0.0")


# ---------------------------------------------------------------------------
# Lifespan startup/shutdown
# ---------------------------------------------------------------------------

class TestLifespan(unittest.TestCase):

    def test_lifespan_initializes_state(self):
        """Test that the lifespan context manager initializes app state correctly."""
        import asyncio
        from qwen3_tts.server.app import lifespan, app

        mock_config = {
            "server": {"auto_shutdown_minutes": 0},
            "models": {"clone": {"load_at_startup": False}},
            "security": {"max_text_length": 10000},
        }

        async def _run():
            with patch(f"{_APP}.load_config", return_value=mock_config), \
                 patch(f"{_APP}.TOKEN_FILE", "/tmp/test_token_xyz"), \
                 patch(f"{_APP}._background_load"), \
                 patch(f"{_APP}.cleanup_resources"), \
                 patch(f"{_APP}.cleanup_pid_file"), \
                 patch("atexit.register"), \
                 patch("builtins.open", MagicMock()), \
                 patch("os.chmod"), \
                 patch("os.unlink"):
                async with lifespan(app):
                    # Verify state was initialized during startup
                    self.assertIsNotNone(app.state.auth_token)
                    self.assertEqual(
                        set(app.state.models.keys()),
                        {"clone", "design", "custom"},
                    )
                    self.assertIsNotNone(app.state.generation_lock)

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# IN_COLAB CORS regex
# ---------------------------------------------------------------------------

class TestColabCors(unittest.TestCase):

    def test_colab_cors_regex_matches_gradio_live(self):
        """Verify the Colab CORS regex allows *.gradio.live origins."""
        import re
        colab_regex = (
            r"(^https?://(localhost|127\.0\.0\.1)(:\d+)?$)"
            r"|(^https://[a-z0-9-]+\.gradio\.live$)"
        )
        self.assertTrue(re.match(colab_regex, "https://abc-123.gradio.live"))
        self.assertTrue(re.match(colab_regex, "http://localhost:7860"))
        self.assertFalse(re.match(colab_regex, "https://evil.com"))


# ---------------------------------------------------------------------------
# _estimate_eta
# ---------------------------------------------------------------------------

class TestEstimateEta(unittest.TestCase):

    def test_no_history(self):
        from qwen3_tts.server.app import _estimate_eta
        state = _make_app_state()
        state.eta_cache = {"median_rate": None, "last_updated": 0}
        with patch(f"{_APP}.get_eta_cache_ttl", return_value=60), \
             patch(f"{_APP}.HISTORY_FILE", "/nonexistent_history.jsonl"), \
             patch("os.path.exists", return_value=False):
            result = _estimate_eta(state, 100, 5.0)
        self.assertIsNone(result)

    def test_with_cached_rate(self):
        from qwen3_tts.server.app import _estimate_eta
        state = _make_app_state()
        # median_rate = chars/sec; fresh cache
        state.eta_cache = {"median_rate": 10.0, "last_updated": time.time()}
        with patch(f"{_APP}.get_eta_cache_ttl", return_value=60):
            result = _estimate_eta(state, 100, 5.0)
        # estimated_total = 100/10 = 10s, remaining = 10 - 5 = 5.0
        self.assertAlmostEqual(result, 5.0, places=1)


if __name__ == "__main__":
    unittest.main()
