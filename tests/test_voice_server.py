"""Server endpoint tests extracted from test_voice.py."""

import time
import unittest
from unittest.mock import patch

from tests.voice_test_helpers import (
    _make_test_client,
    _skip_server,
)


@_skip_server
class TestServerValidation(unittest.TestCase):
    """Test server input validation without loading any models."""

    @classmethod
    def setUpClass(cls):
        """Set up FastAPI TestClient with mocked models."""
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        })
        app.state.models_loaded.set()  # simulate models ready
        cls.auth = {"Authorization": "Bearer test_token"}

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")

    def test_generate_empty_texts(self):
        resp = self.client.post("/generate", json={"texts": []}, headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("no text", resp.json()["detail"].lower())

    def test_generate_batch_too_large(self):
        texts = ["hello"] * 5  # max is 3 in test config
        resp = self.client.post("/generate", json={"texts": texts}, headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("exceeds limit", resp.json()["detail"])

    def test_generate_text_too_long(self):
        resp = self.client.post("/generate", json={"texts": ["x" * 200]}, headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("character limit", resp.json()["detail"])

    def test_generate_invalid_mode(self):
        resp = self.client.post("/generate", json={"texts": ["hello"], "mode": "invalid"}, headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid mode", resp.json()["detail"])

    def test_generate_path_traversal_prompt(self):
        resp = self.client.post("/generate", json={
            "texts": ["hello"],
            "mode": "clone",
            "prompt_file": "../../../etc/passwd",
        }, headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("path traversal", resp.json()["detail"])

    def test_generate_invalid_speaker(self):
        resp = self.client.post("/generate", json={
            "texts": ["hello"],
            "mode": "custom",
            "speaker": "nonexistent_speaker",
        }, headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unknown speaker", resp.json()["detail"])

    def test_generate_valid_speaker_accepted(self):
        # This will fail with 503 (model not loaded) rather than 400 (validation error)
        resp = self.client.post("/generate", json={
            "texts": ["hello"],
            "mode": "custom",
            "speaker": "Ryan",
        }, headers=self.auth)
        # Should pass validation (400) and hit model-not-loaded (503)
        self.assertIn(resp.status_code, [200, 503])

    def test_error_response_has_detail_field(self):
        """All error responses should include a detail field (FastAPI format)."""
        # Validation error
        resp = self.client.post("/generate", json={"texts": []}, headers=self.auth)
        data = resp.json()
        self.assertIn("detail", data)

        # Model not loaded
        resp = self.client.post("/generate", json={
            "texts": ["hello"], "mode": "clone", "prompt_file": "test.pt"
        })
        data = resp.json()
        self.assertIn("detail", data)

    def test_generate_generic_exception_returns_sanitized_detail(self):
        """Generic exceptions in /generate must not expose raw exception messages."""
        # Send a request with invalid parameters that triggers server-side error
        # FastAPI validation catches this and returns a sanitized error
        resp = self.client.post(
            "/generate",
            json={"texts": ["hello"], "mode": "invalid_mode_that_triggers_error"},
            headers=self.auth,
        )
        data = resp.json()
        # Should get a validation error (400) or server error (500)
        self.assertIn(resp.status_code, (400, 422, 500))
        # FastAPI errors use "detail" field and sanitize messages
        self.assertIn("detail", data)
        # Verify no sensitive paths are leaked
        detail_str = str(data.get("detail", ""))
        self.assertNotIn("/home/user", detail_str.lower(),
                         "Error response must not expose home directory paths")
        self.assertNotIn(".ssh", detail_str.lower(),
                         "Error response must not expose .ssh directory")

    def test_generate_missing_prompt_mlx_raise_returns_404(self):
        """MLX loader RAISES FileNotFoundError for a missing prompt (torch
        returns None) — /generate must map it to 404, not an unhandled 500.
        Audit follow-up (2026-08-15): load_voice_prompt_mlx raises while the
        server only mapped the torch None case."""
        from qwen3_tts.server.app import app as _app

        _app.state.models["clone"] = object()  # pass the model-loaded gate
        try:
            with patch(
                "qwen3_tts.core.engine.load_voice_prompt",
                side_effect=FileNotFoundError(
                    "Voice prompt not found: ghost_prompt"
                ),
            ):
                resp = self.client.post("/generate", json={
                    "texts": ["hello"],
                    "mode": "clone",
                    "prompt_file": "ghost_prompt",
                }, headers=self.auth)
        finally:
            _app.state.models["clone"] = None
        self.assertEqual(resp.status_code, 404)
        self.assertIn("not found", resp.json()["detail"].lower())

    def test_generate_stream_missing_prompt_mlx_raise_returns_404(self):
        """/generate-stream must map the MLX FileNotFoundError to 404 too."""
        from qwen3_tts.server.app import app as _app

        _app.state.models["clone"] = object()  # pass the model-loaded gate
        try:
            with patch(
                "qwen3_tts.core.engine.load_voice_prompt",
                side_effect=FileNotFoundError(
                    "Voice prompt not found: ghost_prompt"
                ),
            ):
                resp = self.client.post("/generate-stream", json={
                    "text": "hello",
                    "mode": "clone",
                    "prompt_file": "ghost_prompt",
                }, headers=self.auth)
        finally:
            _app.state.models["clone"] = None
        self.assertEqual(resp.status_code, 404)
        self.assertIn("not found", resp.json()["detail"].lower())

    def test_load_model_exception_returns_sanitized_detail(self):
        """Exceptions in /load-model must not expose raw exception messages."""
        # Send a request with invalid model_type to trigger validation error
        # FastAPI returns a sanitized HTTPException
        resp = self.client.post(
            "/load-model",
            json={"model_type": "invalid_model_type_xyz"},
            headers=self.auth,
        )
        data = resp.json()
        # Should get a validation error (400) with sanitized message
        self.assertIn(resp.status_code, (400, 422, 500))
        # FastAPI errors use "detail" field and sanitize messages
        self.assertIn("detail", data)
        # Verify no sensitive paths are leaked
        detail_str = str(data.get("detail", ""))
        self.assertNotIn("/home/user/lib", detail_str.lower(),
                         "Error response must not expose library paths")
        self.assertNotIn(".py", detail_str.lower(),
                         "Error response must not expose Python file paths")

    def test_rename_prompt_oserror_returns_sanitized_detail(self):
        """OSError in /rename-prompt must not expose internal file paths."""
        from unittest.mock import patch
        secret_path = "/home/user/.ssh/voice_prompts/secret_file.pt"  # nosec B105

        def mock_exists(path):
            # Old file exists as .pt; new file does not exist (no collision)
            return "existing.pt" in path and "new_name" not in path

        with patch('qwen3_tts.server.app_prompts.os.path.exists', side_effect=mock_exists), \
             patch('qwen3_tts.server.app_prompts.os.rename', side_effect=OSError(secret_path)):
            resp = self.client.post(
                "/rename-prompt",
                json={"old_name": "existing.pt", "new_name": "new_name.pt"},
                headers=self.auth,
            )
        data = resp.json()
        self.assertEqual(resp.status_code, 500)
        self.assertNotIn(secret_path, str(data))


@_skip_server
class TestServerAuth(unittest.TestCase):
    """Test server authentication."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {},
            "auto_shutdown_minutes": 0,
        })
        app.state.auth_token = "test_secret_token"  # nosec B105
        app.state.models_loaded.set()  # simulate models ready

    @classmethod
    def tearDownClass(cls):
        from qwen3_tts.server.app import app
        app.state.auth_token = "test_token"  # nosec B105

    def test_health_no_auth_required(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)

    def test_stats_requires_auth(self):
        resp = self.client.get("/stats")
        self.assertEqual(resp.status_code, 401)

    def test_stats_with_valid_auth(self):
        resp = self.client.get("/stats", headers={
            "Authorization": "Bearer test_secret_token"
        })
        self.assertEqual(resp.status_code, 200)

    def test_generate_requires_auth(self):
        resp = self.client.post("/generate", json={"texts": ["hello"]})
        self.assertEqual(resp.status_code, 401)

    def test_generate_wrong_token(self):
        resp = self.client.post("/generate",
            json={"texts": ["hello"]},
            headers={"Authorization": "Bearer wrong_token"})
        self.assertEqual(resp.status_code, 401)

    def test_models_requires_auth(self):
        resp = self.client.get("/models")
        self.assertEqual(resp.status_code, 401)

    def test_models_with_auth(self):
        resp = self.client.get("/models", headers={
            "Authorization": "Bearer test_secret_token"
        })
        self.assertEqual(resp.status_code, 200)

    def test_generation_status_no_auth_required(self):
        resp = self.client.get("/generation-status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data["active"])


@_skip_server
class TestHealthEndpointInfo(unittest.TestCase):
    """Test /health endpoint returns expected info fields."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={"security": {}, "auto_shutdown_minutes": 0})
        app.state.models_loaded.set()

    def test_health_returns_backend(self):
        """/health returns backend field."""
        resp = self.client.get("/health")
        data = resp.json()
        self.assertIn("backend", data)
        self.assertIn(data["backend"], ["torch", "mlx"])

    def test_health_returns_model_size(self):
        """/health returns model_size field."""
        resp = self.client.get("/health")
        data = resp.json()
        self.assertIn("model_size", data)
        self.assertIn(data["model_size"], ["1.7B", "0.6B"])

    def test_health_returns_model_loaded_fields(self):
        """/health returns individual model loaded fields."""
        resp = self.client.get("/health")
        data = resp.json()
        # Check for individual model loaded fields
        self.assertIn("clone_model_loaded", data)
        self.assertIn("design_model_loaded", data)
        self.assertIn("custom_model_loaded", data)
        self.assertIsInstance(data["clone_model_loaded"], bool)


@_skip_server
class TestGenerationStatus(unittest.TestCase):
    """Test /generation-status endpoint and chunk progress tracking."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {},
            "auto_shutdown_minutes": 0,
        })
        app.state.models_loaded.set()  # simulate models ready for tests that need a live server

    def test_generation_status_no_auth_required(self):
        """/generation-status is public."""
        resp = self.client.get("/generation-status")
        self.assertEqual(resp.status_code, 200)

    def test_generation_status_returns_active(self):
        """/generation-status returns active field."""
        resp = self.client.get("/generation-status")
        data = resp.json()
        self.assertIn("active", data)
        self.assertIsInstance(data["active"], bool)

    def test_generation_status_when_inactive(self):
        """When no generation active, returns minimal info."""
        resp = self.client.get("/generation-status")
        data = resp.json()
        self.assertFalse(data["active"])

    def test_generation_status_does_not_leak_input_size(self):
        """Public endpoint must not reveal text length or batch size.

        eta_sec is derived from text_length, batch_total reveals batch size, and
        chunk_total reveals the number of chunks (≈ text length). None of these
        may appear in the unauthenticated response, even mid-generation.
        """
        from qwen3_tts.server.app import app

        app.state.generation_state["active"] = True
        app.state.generation_state["start_time"] = time.time() - 5
        app.state.generation_state["text_length"] = 12345
        app.state.generation_state["batch_total"] = 7
        app.state.generation_state["chunk_total"] = 42
        try:
            resp = self.client.get("/generation-status")
            data = resp.json()
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(data["active"])
            self.assertNotIn("eta_sec", data)
            self.assertNotIn("batch_total", data)
            self.assertNotIn("chunk_total", data)
            self.assertNotIn("text_length", data)
            # Coarse position + time fields remain available.
            self.assertIn("cancelled", data)
            self.assertIn("elapsed_sec", data)
        finally:
            app.state.generation_state["active"] = False
            app.state.generation_state["start_time"] = 0.0
            app.state.generation_state["batch_total"] = 0
            app.state.generation_state["chunk_total"] = 0


@_skip_server
class TestLoadModelEndpoint(unittest.TestCase):
    """Test /load-model endpoint validation."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={"security": {}, "auto_shutdown_minutes": 0})
        app.state.models_loaded.set()  # simulate models ready for tests that need a live server
        # Prevent real model loads: with a backend + cached models present,
        # /load-model would otherwise pull ~7.5GB into the test process and
        # trip the /generate memory guard for later test classes (503 != 400).
        cls._load_patcher = patch(
            "qwen3_tts.core.engine.load_model", return_value=object()
        )
        cls._load_patcher.start()

    @classmethod
    def tearDownClass(cls):
        from qwen3_tts.server.app import app
        cls._load_patcher.stop()
        # Drop stub models so later test classes see a clean not-loaded state.
        for model_type in ("clone", "design", "custom"):
            app.state.models[model_type] = None
        app.state.auth_token = "test_token"  # nosec B105

    def test_load_model_requires_auth(self):
        """POST /load-model requires authentication."""
        resp = self.client.post("/load-model", json={"model_type": "clone"})
        self.assertEqual(resp.status_code, 401)

    def test_load_model_validates_type(self):
        """POST /load-model validates model_type."""
        resp = self.client.post("/load-model",
            json={"model_type": "invalid_type"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unknown model type", resp.json()["detail"])

    def test_load_model_accepts_valid_types(self):
        """POST /load-model accepts clone, design, custom."""
        for model_type in ["clone", "design", "custom"]:
            resp = self.client.post("/load-model",
                json={"model_type": model_type},
                headers={"Authorization": "Bearer test_token"})
            # Should either succeed (200), fail because model not available (503),
            # or fail because backend library not installed (500)
            # but NOT validation error (400)
            self.assertIn(resp.status_code, [200, 500, 503],
                f"model_type '{model_type}' should be valid")

    def test_health_model_load_errors_sanitized(self):
        """/health must not expose raw model-load error paths (CWE-209).

        The on-demand loader stores the failure into state.model_load_errors,
        which the UNAUTHENTICATED /health endpoint returns verbatim. The raw
        exception routinely contains absolute HuggingFace cache paths and repo
        IDs, so it must be sanitized at storage time (the background loader
        already does this — app_lifespan.py:476).
        """
        from qwen3_tts.server.app import app

        # Deterministic regardless of test order: ensure clone is unloaded and
        # no prior error is recorded so /load-model actually invokes load_model.
        app.state.models["clone"] = None
        app.state.model_load_errors["clone"] = None
        secret = (
            "/Users/secret/.cache/huggingface/hub/"
            "models--x/snapshots/abc missing weight"  # nosec B105 -- test fixture
        )
        try:
            with patch(
                "qwen3_tts.core.engine.load_model", side_effect=RuntimeError(secret)
            ):
                resp = self.client.post(
                    "/load-model",
                    json={"model_type": "clone"},
                    headers={"Authorization": "Bearer test_token"},
                )
            # The authenticated caller gets a sanitized 500; the error is also
            # recorded in state for /health.
            self.assertEqual(resp.status_code, 500)

            # /health is unauthenticated and echoes model_load_errors verbatim.
            health = self.client.get("/health").json()
            err = health.get("model_load_errors", {}).get("clone")
            self.assertIsNotNone(err, "expected a load error to be recorded")
            self.assertIn("<path>", err, "absolute paths must be redacted")
            for needle in ("/Users", ".cache", "snapshots"):
                self.assertNotIn(needle, err)
        finally:
            app.state.models["clone"] = None
            app.state.model_load_errors["clone"] = None


@_skip_server
class TestCancelGenerationEndpoint(unittest.TestCase):
    """Test /cancel-generation endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {},
            "auto_shutdown_minutes": 0,
        })
        app.state.models_loaded.set()

    @classmethod
    def tearDownClass(cls):
        from qwen3_tts.server.app import app
        app.state.auth_token = "test_token"  # nosec B105

    def test_cancel_requires_auth(self):
        """POST /cancel-generation requires authentication."""
        resp = self.client.post("/cancel-generation")
        self.assertEqual(resp.status_code, 401)

    def test_cancel_when_no_active_generation(self):
        """Cancel returns no_active_generation when nothing running."""
        from qwen3_tts.server.app import app
        app.state.generation_state["active"] = False
        resp = self.client.post("/cancel-generation",
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "no_active_generation")

    def test_cancel_sets_cancelled_flag(self):
        """Cancel sets the cancelled flag in generation_state."""
        from qwen3_tts.server.app import app
        app.state.generation_state.update({
            "active": True,
            "cancelled": False,
            "generation_id": "test123",
        })
        resp = self.client.post("/cancel-generation",
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "cancellation_requested")
        from qwen3_tts.server.app import app
        self.assertTrue(app.state.generation_state["cancelled"])
        # Reset
        app.state.generation_state.update({
            "active": False,
            "cancelled": False,
            "generation_id": None,
        })

    def test_cancel_returns_generation_id(self):
        """Cancel returns the generation_id."""
        from qwen3_tts.server.app import app
        app.state.generation_state.update({
            "active": True,
            "cancelled": False,
            "generation_id": "abc12345",
        })
        resp = self.client.post("/cancel-generation",
            headers={"Authorization": "Bearer test_token"})
        data = resp.json()
        self.assertEqual(data["generation_id"], "abc12345")
        # Reset
        app.state.generation_state.update({
            "active": False,
            "cancelled": False,
            "generation_id": None,
        })


@_skip_server
class TestGenerationStateFields(unittest.TestCase):
    """Test generation_state has required fields for cancellation."""

    def test_generation_state_has_cancelled_field(self):
        """generation_state dict has cancelled field."""
        from qwen3_tts.server.app import app
        self.assertIn("cancelled", app.state.generation_state)

    def test_generation_state_has_generation_id(self):
        """generation_state dict has generation_id field."""
        from qwen3_tts.server.app import app
        self.assertIn("generation_id", app.state.generation_state)

    def test_generation_state_initial_values(self):
        """generation_state has correct initial values."""
        from qwen3_tts.server.app import app
        # These should be the default/initial values
        state = app.state.generation_state
        self.assertIn("active", state)
        self.assertIn("start_time", state)
        self.assertIn("text_length", state)
        self.assertIn("mode", state)
        self.assertIn("chunk_index", state)
        self.assertIn("chunk_total", state)


@_skip_server
class TestUpdateModelConfigEndpoint(unittest.TestCase):
    """Test /update-model-config server endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {"max_text_length": 10000},
            "auto_shutdown_minutes": 0,
        })
        app.state.models_loaded.set()  # simulate models ready for tests that need a live server

    @classmethod
    def tearDownClass(cls):
        from qwen3_tts.server.app import app
        app.state.auth_token = "test_token"  # nosec B105

    def test_update_model_config_requires_auth(self):
        """POST /update-model-config requires authentication."""
        resp = self.client.post("/update-model-config",
            json={"model_size": "0.6B"})
        self.assertEqual(resp.status_code, 401)

    def test_update_model_config_validates_model_size(self):
        """POST /update-model-config validates model_size."""
        resp = self.client.post("/update-model-config",
            json={"model_size": "invalid"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid model_size", resp.json()["detail"])

    def test_update_model_config_validates_mlx_quantization(self):
        """POST /update-model-config validates mlx_quantization."""
        resp = self.client.post("/update-model-config",
            json={"mlx_quantization": "invalid"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid mlx_quantization", resp.json()["detail"])


@_skip_server
class TestStreamingEndpointStructure(unittest.TestCase):
    """Test /generate-stream endpoint structure."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {"max_text_length": 10000},
            "auto_shutdown_minutes": 0,
        })
        app.state.models_loaded.set()

    @classmethod
    def tearDownClass(cls):
        from qwen3_tts.server.app import app
        app.state.auth_token = "test_token"  # nosec B105

    def test_generate_stream_requires_auth(self):
        """POST /generate-stream requires authentication."""
        resp = self.client.post("/generate-stream",
            json={"text": "Hello", "mode": "clone"})
        self.assertEqual(resp.status_code, 401)

    def test_generate_stream_requires_text(self):
        """POST /generate-stream requires text."""
        resp = self.client.post("/generate-stream",
            json={"mode": "clone"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No text provided", resp.json()["detail"])

    def test_generate_stream_validates_mode(self):
        """POST /generate-stream validates mode."""
        resp = self.client.post("/generate-stream",
            json={"text": "Hello", "mode": "invalid"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid mode", resp.json()["detail"])


@_skip_server
class TestGenerateStreamIdCheck(unittest.TestCase):
    """Test generate_stream generation_id race condition fix."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app)
        app.state.models_loaded.set()
        cls.auth = {"Authorization": "Bearer test_token"}

    def test_generate_stream_checks_generation_id(self):
        """generate_stream only resets state if generation_id matches."""
        import inspect

        from qwen3_tts.server import app_generation
        source = inspect.getsource(app_generation)
        # Should check generation_id before resetting
        self.assertIn('if state.generation_state.get("generation_id") == gen_id', source)

    def test_generation_state_has_generation_id(self):
        """generation_state includes generation_id field."""
        from qwen3_tts.server.app import app
        self.assertIn("generation_id", app.state.generation_state)

    def test_generation_state_has_cancelled(self):
        """generation_state includes cancelled field."""
        from qwen3_tts.server.app import app
        self.assertIn("cancelled", app.state.generation_state)


@_skip_server
class TestUnloadModelEndpoint(unittest.TestCase):
    """Test /unload-model endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {},
            "auto_shutdown_minutes": 0,
            "models": {"clone": {"load_at_startup": True}},
        })
        app.state.models_loaded.set()

    @classmethod
    def tearDownClass(cls):
        from qwen3_tts.server.app import app
        app.state.auth_token = "test_token"  # nosec B105

    def test_unload_requires_auth(self):
        """POST /unload-model requires authentication."""
        resp = self.client.post("/unload-model", json={"model_type": "clone"})
        self.assertEqual(resp.status_code, 401)

    def test_unload_validates_type(self):
        """POST /unload-model validates model_type."""
        resp = self.client.post("/unload-model",
            json={"model_type": "invalid"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)

    def test_unload_requires_model_type(self):
        """POST /unload-model requires model_type field."""
        resp = self.client.post("/unload-model",
            json={},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 422)  # FastAPI validation error

    def test_unload_already_unloaded(self):
        """POST /unload-model returns already_unloaded when model not loaded."""
        from qwen3_tts.server.app import app
        app.state.models["clone"] = None
        resp = self.client.post("/unload-model",
            json={"model_type": "clone"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "already_unloaded")

    def test_unload_rejects_during_generation(self):
        """POST /unload-model returns 409 when generation active for that mode."""
        from qwen3_tts.server.app import app
        app.state.generation_state["active"] = True
        app.state.generation_state["mode"] = "clone"
        try:
            resp = self.client.post("/unload-model",
                json={"model_type": "clone"},
                headers={"Authorization": "Bearer test_token"})
            self.assertEqual(resp.status_code, 409)
        finally:
            app.state.generation_state["active"] = False
            app.state.generation_state["mode"] = ""


@_skip_server
class TestUpdateStartupConfigEndpoint(unittest.TestCase):
    """Test /update-startup-config endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {},
            "auto_shutdown_minutes": 0,
        })
        app.state.models_loaded.set()

    @classmethod
    def tearDownClass(cls):
        from qwen3_tts.server.app import app
        app.state.auth_token = "test_token"  # nosec B105

    def setUp(self):
        # Reset rate limiters before each test — under python -m unittest the
        # pytest autouse fixture (conftest.reset_rate_limiters) never fires, so
        # counters accumulate and endpoints return 429 after a few calls.
        from qwen3_tts.server.app import app
        for attr in ("limiter", "limiter_hybrid", "limiter_ip", "limiter_token"):
            limiter = getattr(app.state, attr, None)
            if limiter is not None and hasattr(limiter, "reset"):
                limiter.reset()

    def test_startup_config_requires_auth(self):
        """POST /update-startup-config requires authentication."""
        resp = self.client.post("/update-startup-config", json={"clone": True})
        self.assertEqual(resp.status_code, 401)

    def test_startup_config_empty_body(self):
        """POST /update-startup-config rejects empty body."""
        resp = self.client.post("/update-startup-config",
            json={},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)

    @patch("qwen3_tts.server.app_models.save_config")
    @patch("qwen3_tts.core.config.load_config")
    def test_startup_config_saves(self, mock_load, mock_save):
        """POST /update-startup-config saves to config."""
        mock_load.return_value = {"models": {"clone": {}, "design": {}, "custom": {}}}
        resp = self.client.post("/update-startup-config",
            json={"clone": True, "design": False},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "updated")
        self.assertTrue(mock_save.called)

    @patch("qwen3_tts.server.app_models.save_config")
    @patch("qwen3_tts.core.config.load_config")
    def test_startup_config_partial_update(self, mock_load, mock_save):
        """POST /update-startup-config accepts partial updates."""
        mock_load.return_value = {"models": {"clone": {"load_at_startup": True}}}
        resp = self.client.post("/update-startup-config",
            json={"design": True},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 200)
        changes = resp.json()["changes"]
        self.assertEqual(len(changes), 1)
        self.assertIn("design=on", changes[0])


@_skip_server
class TestModelsEndpointEnhanced(unittest.TestCase):
    """Test /models endpoint includes load_at_startup and load_time_sec."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {},
            "auto_shutdown_minutes": 0,
            "models": {
                "clone": {"load_at_startup": True},
                "design": {"load_at_startup": False},
                "custom": {"load_at_startup": False},
            },
        })
        app.state.model_load_times = {"clone": 5.2}  # Override for testing
        app.state.models_loaded.set()  # simulate models ready

    @classmethod
    def tearDownClass(cls):
        from qwen3_tts.server.app import app
        app.state.auth_token = "test_token"  # nosec B105
        app.state.model_load_times = {}

    def test_models_has_load_at_startup(self):
        """GET /models includes load_at_startup field."""
        resp = self.client.get("/models",
            headers={"Authorization": "Bearer test_token"})
        data = resp.json()
        clone_info = data["models"]["clone"]
        self.assertIn("load_at_startup", clone_info)
        self.assertTrue(clone_info["load_at_startup"])

    def test_models_has_load_time(self):
        """GET /models includes load_time_sec field."""
        resp = self.client.get("/models",
            headers={"Authorization": "Bearer test_token"})
        data = resp.json()
        clone_info = data["models"]["clone"]
        self.assertIn("load_time_sec", clone_info)
        self.assertEqual(clone_info["load_time_sec"], 5.2)

    def test_health_includes_load_times(self):
        """GET /health includes model_load_times."""
        resp = self.client.get("/health")
        data = resp.json()
        self.assertIn("model_load_times", data)


class TestUpdateModelConfigAcceptsAll5MlxQuants:
    """Server /update-model-config must accept all VALID_MLX_QUANTIZATIONS values."""

    def test_5bit_accepted(self):
        from qwen3_tts.core.config import VALID_MLX_QUANTIZATIONS
        from qwen3_tts.server import app_models
        assert "5bit" in VALID_MLX_QUANTIZATIONS
        # The handler must not have a stale local tuple
        import inspect
        src = inspect.getsource(app_models.handle_update_model_config)
        assert '("4bit", "8bit", "bf16")' not in src, (
            "app_models.py still has the stale 3-element valid_quants tuple"
        )
        assert "VALID_MLX_QUANTIZATIONS" in src, (
            "app_models.py should read from the canonical VALID_MLX_QUANTIZATIONS"
        )


@_skip_server
class TestClonePromptLoadLockScope(unittest.TestCase):
    """P5: clone-mode voice-prompt load must run OUTSIDE inference_lock.

    Disk I/O + tensor deserialize is not GPU work; holding the GPU-serialization
    lock during it needlessly blocks every other generation. The streaming and
    WebSocket sibling paths load the prompt before the lock; the batch /generate
    path now matches them.
    """

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app

        cls.client = _make_test_client(
            app,
            server_config={"security": {"max_text_length": 10000}, "auto_shutdown_minutes": 0},
        )
        app.state.models_loaded.set()

    def test_clone_prompt_load_runs_outside_inference_lock(self):
        from unittest.mock import MagicMock, patch

        import numpy as np

        from qwen3_tts.server.app import app

        app.state.models["clone"] = MagicMock()
        # Ensure no cache hit so load_voice_prompt actually runs.
        with app.state.gen_cache_lock:
            app.state.gen_cache.clear()

        lock_state_during_load = []

        def spy_load(*args, **kwargs):
            # Runs in the to_thread worker. On CPython the GIL makes this bool
            # read atomic, and a held inference_lock stays held for the whole
            # call (the owning coroutine is suspended at the await). So this
            # reliably distinguishes load-inside-lock from load-outside-lock.
            lock_state_during_load.append(app.state.inference_lock.locked())
            return MagicMock()

        fake_wav = np.zeros(100, dtype="float32")
        with patch("qwen3_tts.core.engine.load_voice_prompt", side_effect=spy_load), patch(
            "qwen3_tts.core.engine.run_inference", return_value=(fake_wav, 24000)
        ), patch(
            "qwen3_tts.server.app_lifespan._check_memory_available",
            return_value=(True, 10000),
        ):
            resp = self.client.post(
                "/generate",
                json={"texts": ["hello"], "mode": "clone", "prompt_file": "voice.pt"},
                headers={"Authorization": "Bearer test_token"},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(
            lock_state_during_load,
            [False],
            f"clone prompt load must run OUTSIDE inference_lock (got {lock_state_during_load})",
        )


if __name__ == "__main__":
    unittest.main()
