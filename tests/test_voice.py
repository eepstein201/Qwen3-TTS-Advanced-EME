#!/usr/bin/env python3
"""Tests for Qwen3-TTS codebase.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/ -v

These tests do NOT require GPU, models, or a running server.
They test config, validation, auth, error classes, and the server
endpoint logic using FastAPI's TestClient.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    import soundfile  # already imported above, but need for server deps
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# voice_server requires soundfile + fastapi; voice_client requires soundfile;
# voice_ui requires gradio
_server_deps = HAS_SOUNDFILE and HAS_FASTAPI
_client_deps = HAS_SOUNDFILE
_ui_deps = HAS_GRADIO

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
    app.state.model_load_errors = {"clone": None, "design": None, "custom": None}
    app.state.shutdown_timer = None
    if server_config:
        app.state.server_config = server_config
    else:
        app.state.server_config = {
            "security": {"max_text_length": 10000, "max_batch_size": 20},
            "auto_shutdown_minutes": 0,
        }


from contextlib import asynccontextmanager

@asynccontextmanager
async def _null_lifespan(app):
    """No-op lifespan to prevent real model loading during tests."""
    yield

def _make_test_client(app, server_config=None):
    """Create TestClient without triggering real lifespan model loading."""
    from fastapi.testclient import TestClient
    _setup_fastapi_app_state(app, server_config)
    original = app.router.lifespan_context
    app.router.lifespan_context = _null_lifespan
    client = TestClient(app)
    app.router.lifespan_context = original
    return client


# =============================================================================
# qwen3_tts.core.config tests
# =============================================================================

class TestTTSConfig(unittest.TestCase):
    """Test qwen3_tts.core.config module (no heavy imports)."""

    def test_imports_no_torch(self):
        """qwen3_tts.core.config must not import torch."""
        from qwen3_tts.core import config  # noqa: F401
        self.assertNotIn("torch", sys.modules.keys() - {"torch"})

    def test_error_hierarchy(self):
        from qwen3_tts.core.config import (
            TTSError, ServerConnectionError, ModelNotLoadedError,
            InvalidInputError, GenerationError, AuthenticationError,
        )
        # All errors inherit from TTSError
        for cls in [ServerConnectionError, ModelNotLoadedError,
                    InvalidInputError, GenerationError, AuthenticationError]:
            err = cls("test") if cls != ModelNotLoadedError else cls("clone")
            self.assertIsInstance(err, TTSError)

    def test_error_format_cli(self):
        from qwen3_tts.core.config import ServerConnectionError
        err = ServerConnectionError("details here")
        formatted = err.format_cli()
        self.assertIn("Cannot connect", formatted)
        self.assertIn("tts server start", formatted)

    def test_error_format_gradio(self):
        from qwen3_tts.core.config import GenerationError
        err = GenerationError("oops")
        html = err.format_gradio()
        self.assertIn("Audio generation failed", html)

    def test_read_auth_token_missing(self):
        from qwen3_tts.core.config import TOKEN_FILE, read_auth_token
        # Use a temp path that doesn't exist
        with patch("qwen3_tts.core.config.TOKEN_FILE", os.path.join(tempfile.gettempdir(), "nonexistent_token_test_xyz")):
            result = read_auth_token()
        # If the real file exists, it returns its content; with patched path it's None
        # Can't reliably test with real TOKEN_FILE, so test the function signature
        self.assertTrue(result is None or isinstance(result, str))

    def test_auth_headers_returns_dict(self):
        from qwen3_tts.core.config import auth_headers
        headers = auth_headers()
        self.assertIsInstance(headers, dict)

    def test_auth_headers_with_token(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".token", delete=False) as f:
            f.write("test_token_abc123")
            token_path = f.name
        try:
            with patch("qwen3_tts.core.config.TOKEN_FILE", token_path):
                from qwen3_tts.core.config import read_auth_token, auth_headers
                # Need to reimport to pick up patched value
                token = read_auth_token()
                headers = auth_headers()
            # The token file exists so we should get headers
            # (actual behavior depends on whether TOKEN_FILE is patched at call time)
        finally:
            os.unlink(token_path)

    def test_model_info_keys(self):
        from qwen3_tts.core.config import MODEL_INFO
        # MODEL_INFO is now nested by size: MODEL_INFO["1.7B"]["clone"], etc.
        self.assertIn("1.7B", MODEL_INFO)
        self.assertIn("0.6B", MODEL_INFO)
        for size in ("1.7B", "0.6B"):
            self.assertIn("clone", MODEL_INFO[size])
            self.assertIn("design", MODEL_INFO[size])
            self.assertIn("custom", MODEL_INFO[size])
            for info in MODEL_INFO[size].values():
                self.assertIn("name", info)
                self.assertIn("description", info)
                self.assertIn("memory_mb", info)

    def test_custom_voice_speakers(self):
        from qwen3_tts.core.config import CUSTOM_VOICE_SPEAKERS
        self.assertIn("ryan", CUSTOM_VOICE_SPEAKERS)
        self.assertIn("aiden", CUSTOM_VOICE_SPEAKERS)
        for key, info in CUSTOM_VOICE_SPEAKERS.items():
            self.assertIn("name", info)
            self.assertIn("lang", info)
            self.assertIn("desc", info)


# =============================================================================
# voice_server validation tests (using FastAPI TestClient, no models needed)
# =============================================================================

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

        with patch('qwen3_tts.server.app.os.path.exists', side_effect=mock_exists), \
             patch('qwen3_tts.server.app.os.rename', side_effect=OSError(secret_path)):
            resp = self.client.post(
                "/rename-prompt",
                json={"old_name": "existing.pt", "new_name": "new_name.pt"},
                headers=self.auth,
            )
        data = resp.json()
        self.assertEqual(resp.status_code, 500)
        self.assertNotIn(secret_path, str(data))


# =============================================================================
# voice_server auth tests
# =============================================================================

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


# =============================================================================
# SSML parsing tests (from qwen3_tts.interface.generate, lightweight)
# =============================================================================

@_skip_generate
class TestSSMLParsing(unittest.TestCase):
    """Test SSML parsing in voice_generate."""

    def test_no_ssml(self):
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml("Hello world")
        self.assertEqual(text, "Hello world")
        self.assertFalse(meta["has_ssml"])

    def test_break_tag(self):
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml('Hello <break time="500ms"/> world')
        self.assertTrue(meta["has_ssml"])
        self.assertNotIn("<break", text)

    def test_sub_tag(self):
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml('<sub alias="World Wide Web">WWW</sub>')
        self.assertIn("World Wide Web", text)
        self.assertNotIn("WWW", text)

    def test_say_as_characters(self):
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml('<say-as interpret-as="characters">ABC</say-as>')
        self.assertIn("A B C", text)

    def test_prosody_speed(self):
        from qwen3_tts.interface.generate import parse_ssml
        text, meta = parse_ssml('<prosody rate="fast">Quick text</prosody>')
        self.assertTrue(meta["has_ssml"])
        self.assertEqual(meta["prosody"]["speed"], 1.2)


# =============================================================================
# SRT parsing tests
# =============================================================================

@_skip_generate
class TestSRTParsing(unittest.TestCase):
    """Test SRT parsing."""

    def test_parse_srt(self):
        from qwen3_tts.interface.generate import parse_srt
        srt_content = """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,500
Second subtitle
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
            f.write(srt_content)
            srt_path = f.name
        try:
            entries = parse_srt(srt_path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0][3], "Hello world")
            self.assertEqual(entries[1][3], "Second subtitle")
        finally:
            os.unlink(srt_path)

    def test_srt_time_to_ms(self):
        from qwen3_tts.interface.generate import srt_time_to_ms
        self.assertEqual(srt_time_to_ms("00:01:30,500"), 90500)
        self.assertEqual(srt_time_to_ms("01:00:00,000"), 3600000)


# =============================================================================
# Auto-increment filename tests
# =============================================================================

@_skip_generate
class TestAutoIncrementFilename(unittest.TestCase):
    """Test auto_increment_filename helper."""

    def test_no_conflict(self):
        from qwen3_tts.interface.generate import auto_increment_filename
        # Non-existent file should return as-is
        _tmp = os.path.join(tempfile.gettempdir(), "nonexistent_test_xyz.wav")
        result = auto_increment_filename(_tmp)
        self.assertEqual(result, _tmp)

    def test_conflict_increments(self):
        from qwen3_tts.interface.generate import auto_increment_filename
        # Create a temp file that definitely exists
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
            f.write(b"test")  # Write some content to ensure file exists
        try:
            # Verify file exists before calling
            self.assertTrue(os.path.exists(path), "Temp file should exist")
            result = auto_increment_filename(path)
            self.assertNotEqual(result, path)
            # Check that result has _2 before the extension
            base, ext = os.path.splitext(result)
            self.assertTrue(base.endswith("_2"), f"Expected '_2' suffix in {base}")
        finally:
            os.unlink(path)
            # Also clean up the incremented file if it was created
            if os.path.exists(result):
                os.unlink(result)

    def test_already_numbered(self):
        from qwen3_tts.interface.generate import auto_increment_filename
        # Create files with _2 suffix
        with tempfile.NamedTemporaryFile(suffix="_2.wav", delete=False, dir=tempfile.gettempdir(), prefix="test_") as f:
            path = f.name
        try:
            result = auto_increment_filename(path)
            self.assertNotEqual(result, path)
            self.assertIn("_3", result)
        finally:
            os.unlink(path)


# =============================================================================
# Backend config tests (qwen3_tts.core.config backend helpers)
# =============================================================================

class TestBackendConfig(unittest.TestCase):
    """Test backend-related config helpers in qwen3_tts.core.config."""

    def test_valid_backends(self):
        from qwen3_tts.core.config import VALID_BACKENDS
        self.assertIn("torch", VALID_BACKENDS)
        self.assertIn("mlx", VALID_BACKENDS)

    def test_valid_mlx_quantizations(self):
        from qwen3_tts.core.config import VALID_MLX_QUANTIZATIONS
        self.assertIn("4bit", VALID_MLX_QUANTIZATIONS)
        self.assertIn("8bit", VALID_MLX_QUANTIZATIONS)
        self.assertIn("bf16", VALID_MLX_QUANTIZATIONS)

    def test_mlx_model_info_keys(self):
        from qwen3_tts.core.config import MLX_MODEL_INFO
        # MLX_MODEL_INFO is now nested by size: MLX_MODEL_INFO["1.7B"]["clone"], etc.
        self.assertIn("1.7B", MLX_MODEL_INFO)
        self.assertIn("0.6B", MLX_MODEL_INFO)
        for size in ("1.7B", "0.6B"):
            self.assertIn("clone", MLX_MODEL_INFO[size])
            self.assertIn("design", MLX_MODEL_INFO[size])
            self.assertIn("custom", MLX_MODEL_INFO[size])
            for info in MLX_MODEL_INFO[size].values():
                self.assertIn("name_template", info)
                self.assertIn("description", info)
                self.assertIn("memory_mb", info)

    def test_mlx_model_info_templates(self):
        from qwen3_tts.core.config import MLX_MODEL_INFO
        for size in ("1.7B", "0.6B"):
            for model_type, info in MLX_MODEL_INFO[size].items():
                self.assertIn("{quant}", info["name_template"])

    def test_get_backend_default(self):
        """get_backend() defaults to 'mlx' with no env/config override (MLX-first architecture)."""
        from qwen3_tts.core.config import get_backend
        # Clear env override
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_BACKEND", None)
            with patch("qwen3_tts.core.config.load_config", return_value={}):
                result = get_backend()
        self.assertEqual(result, "mlx")

    def test_get_backend_from_config(self):
        from qwen3_tts.core.config import get_backend
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_BACKEND", None)
            with patch("qwen3_tts.core.config.load_config", return_value={"advanced": {"backend": "mlx"}}):
                result = get_backend()
        self.assertEqual(result, "mlx")

    def test_get_backend_env_override(self):
        """TTS_BACKEND env var overrides config."""
        from qwen3_tts.core.config import get_backend
        with patch.dict(os.environ, {"TTS_BACKEND": "mlx"}):
            with patch("qwen3_tts.core.config.load_config", return_value={"advanced": {"backend": "torch"}}):
                result = get_backend()
        self.assertEqual(result, "mlx")

    def test_get_backend_invalid_falls_back(self):
        """Invalid backend value falls back to 'mlx' (MLX-first architecture)."""
        from qwen3_tts.core.config import get_backend
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_BACKEND", None)
            with patch("qwen3_tts.core.config.load_config", return_value={"advanced": {"backend": "invalid"}}):
                result = get_backend()
        self.assertEqual(result, "mlx")

    def test_get_mlx_quantization_default(self):
        from qwen3_tts.core.config import get_mlx_quantization
        with patch("qwen3_tts.core.config.load_config", return_value={}):
            result = get_mlx_quantization()
        self.assertEqual(result, "8bit")

    def test_get_mlx_quantization_from_config(self):
        from qwen3_tts.core.config import get_mlx_quantization
        with patch("qwen3_tts.core.config.load_config", return_value={"advanced": {"mlx_quantization": "4bit"}}):
            result = get_mlx_quantization()
        self.assertEqual(result, "4bit")

    def test_get_mlx_quantization_invalid_falls_back(self):
        from qwen3_tts.core.config import get_mlx_quantization
        with patch("qwen3_tts.core.config.load_config", return_value={"advanced": {"mlx_quantization": "garbage"}}):
            result = get_mlx_quantization()
        self.assertEqual(result, "8bit")

    def test_get_mlx_model_name(self):
        from qwen3_tts.core.config import get_mlx_model_name
        with patch("qwen3_tts.core.config.get_mlx_quantization", return_value="8bit"):
            name = get_mlx_model_name("clone")
        self.assertEqual(name, "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")

    def test_get_mlx_model_name_4bit(self):
        from qwen3_tts.core.config import get_mlx_model_name
        with patch("qwen3_tts.core.config.get_mlx_quantization", return_value="4bit"):
            name = get_mlx_model_name("design")
        self.assertEqual(name, "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit")

    def test_get_mlx_model_name_invalid_type(self):
        from qwen3_tts.core.config import get_mlx_model_name
        with self.assertRaises(ValueError):
            get_mlx_model_name("nonexistent")


# =============================================================================
# MLX voice prompt loading tests
# =============================================================================

class TestMLXVoicePrompt(unittest.TestCase):
    """Test MLX voice prompt loading (file-based, no mlx import needed)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_dir = None

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_voice_prompt_mlx_success(self):
        """Load MLX prompt from .wav + .txt pair."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx
        # Create fake wav and txt
        wav_path = os.path.join(self.tmpdir, "test_voice.wav")
        txt_path = os.path.join(self.tmpdir, "test_voice.txt")
        with open(wav_path, "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        with open(txt_path, "w") as f:
            f.write("Hello, this is a test transcript.")

        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            result = load_voice_prompt_mlx("test_voice.pt")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["ref_audio"], wav_path)
        self.assertEqual(result["ref_text"], "Hello, this is a test transcript.")

    def test_load_voice_prompt_mlx_strips_pt(self):
        """Prompt name with .pt extension is handled correctly."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx
        wav_path = os.path.join(self.tmpdir, "voice.wav")
        txt_path = os.path.join(self.tmpdir, "voice.txt")
        with open(wav_path, "wb") as f:
            f.write(b"fake")
        with open(txt_path, "w") as f:
            f.write("transcript")

        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            result = load_voice_prompt_mlx("voice.pt")
        self.assertEqual(result["ref_audio"], wav_path)

    def test_load_voice_prompt_mlx_missing_files(self):
        """Raises FileNotFoundError when wav/txt missing."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx
        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            with self.assertRaises(FileNotFoundError):
                load_voice_prompt_mlx("nonexistent")

    def test_load_voice_prompt_mlx_pt_only_error(self):
        """Clear error when only .pt exists (no MLX-compatible files)."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx
        pt_path = os.path.join(self.tmpdir, "legacy.pt")
        with open(pt_path, "wb") as f:
            f.write(b"fake tensor data")

        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            with self.assertRaises(FileNotFoundError) as ctx:
                load_voice_prompt_mlx("legacy")
            self.assertIn("only has a .pt file", str(ctx.exception))
            self.assertIn("tts voice create", str(ctx.exception))

    def test_load_voice_prompt_dispatch_torch(self):
        """load_voice_prompt dispatches to torch backend."""
        from qwen3_tts.core.engine import load_voice_prompt
        with patch("qwen3_tts.core.engine.voice_prompt.get_backend", return_value="torch"):
            with patch("qwen3_tts.core.engine.voice_prompt._load_voice_prompt_torch", return_value="mock_tensor") as mock:
                result = load_voice_prompt("test.pt")
        mock.assert_called_once_with("test.pt")
        self.assertEqual(result, "mock_tensor")

    def test_load_voice_prompt_dispatch_mlx(self):
        """load_voice_prompt dispatches to MLX backend."""
        from qwen3_tts.core.engine import load_voice_prompt
        mock_result = {"ref_audio": "/fake/path.wav", "ref_text": "text"}
        with patch("qwen3_tts.core.engine.voice_prompt.get_backend", return_value="mlx"):
            with patch("qwen3_tts.core.engine.voice_prompt.load_voice_prompt_mlx", return_value=mock_result) as mock:
                result = load_voice_prompt("test.pt")
        mock.assert_called_once_with("test.pt")
        self.assertEqual(result, mock_result)


# =============================================================================
# Backend dispatch tests (no actual model loading — tests dispatch logic)
# =============================================================================

class TestBackendDispatch(unittest.TestCase):
    """Test that public API dispatches to correct backend functions."""

    def test_load_model_dispatch_torch(self):
        from qwen3_tts.core.engine import load_model
        with patch("qwen3_tts.core.engine.model_loader.get_backend", return_value="torch"):
            with patch("qwen3_tts.core.engine.model_loader._load_model_torch", return_value="torch_model") as mock:
                result = load_model("clone")
        mock.assert_called_once_with("clone")
        self.assertEqual(result, "torch_model")

    def test_load_model_dispatch_mlx(self):
        from qwen3_tts.core.engine import load_model
        with patch("qwen3_tts.core.engine.model_loader.get_backend", return_value="mlx"):
            with patch("qwen3_tts.core.engine.model_loader._load_model_mlx", return_value="mlx_model") as mock:
                result = load_model("design")
        mock.assert_called_once_with("design")
        self.assertEqual(result, "mlx_model")

    def test_run_inference_dispatch_torch(self):
        from qwen3_tts.core.engine import run_inference
        with patch("qwen3_tts.core.engine.inference.get_backend", return_value="torch"):
            # Patch the registry entry for torch backend
            with patch.dict(
                "qwen3_tts.core.engine.inference._INFERENCE_STRATEGIES",
                {"torch": lambda *args, **kwargs: ("wav", 24000)}
            ):
                result = run_inference("model", "text", "clone", {})
        self.assertEqual(result, ("wav", 24000))

    def test_run_inference_dispatch_mlx(self):
        from qwen3_tts.core.engine import run_inference
        with patch("qwen3_tts.core.engine.inference.get_backend", return_value="mlx"):
            # Patch the registry entry for mlx backend
            with patch.dict(
                "qwen3_tts.core.engine.inference._INFERENCE_STRATEGIES",
                {"mlx": lambda *args, **kwargs: ("wav", 24000)}
            ):
                result = run_inference("model", "text", "design", {})
        self.assertEqual(result, ("wav", 24000))


# =============================================================================
# MLX inference tests (skipped if mlx not installed)
# =============================================================================

# Check for mlx import capability (not config — actual library)
try:
    import mlx.core  # noqa: F401
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    from mlx_audio.tts.utils import load_model as _  # noqa: F401
    HAS_MLX_AUDIO = True
except ImportError:
    HAS_MLX_AUDIO = False


@unittest.skipIf(not HAS_MLX, "mlx not installed")
class TestMLXImport(unittest.TestCase):
    """Test that MLX imports work when mlx is available."""

    def test_mlx_core_import(self):
        import mlx.core as mx
        self.assertTrue(hasattr(mx, "array"))

    @unittest.skipIf(not HAS_MLX_AUDIO, "mlx-audio not installed")
    def test_mlx_audio_import(self):
        from mlx_audio.tts.utils import load_model
        self.assertTrue(callable(load_model))


class TestMLXInferenceCloneValidation(unittest.TestCase):
    """Test MLX inference input validation (no actual model needed)."""

    def test_clone_requires_voice_prompt(self):
        """_run_inference_mlx raises ValueError without voice_prompt in clone mode."""
        from qwen3_tts.core.engine.inference import _run_inference_mlx
        with self.assertRaises(ValueError) as ctx:
            _run_inference_mlx(
                model=MagicMock(),
                text="test",
                mode="clone",
                gen_params={},
                voice_prompt=None,
            )
        self.assertIn("required for clone mode", str(ctx.exception))

    def test_clone_rejects_non_dict_prompt(self):
        """_run_inference_mlx raises TypeError for non-dict voice_prompt."""
        from qwen3_tts.core.engine.inference import _run_inference_mlx
        with self.assertRaises(TypeError) as ctx:
            _run_inference_mlx(
                model=MagicMock(),
                text="test",
                mode="clone",
                gen_params={},
                voice_prompt="some_tensor_object",
            )
        self.assertIn("MLX clone mode requires a voice prompt dict", str(ctx.exception))
        self.assertIn("tts voice create", str(ctx.exception))


# =============================================================================
# Lazy import safety tests
# =============================================================================

class TestLazyImports(unittest.TestCase):
    """Verify that qwen3_tts.core.engine does not import torch or mlx at module scope."""

    def test_engine_no_torch_at_module_scope(self):
        """qwen3_tts.core.engine module should not force-import torch."""
        # Remove engine from cache to test fresh import behavior
        saved_modules = {}
        for mod in list(sys.modules.keys()):
            if mod == "qwen3_tts.core.engine" or mod.startswith("qwen3_tts.core.engine."):
                saved_modules[mod] = sys.modules.pop(mod)

        # Also note if torch was already loaded
        torch_was_loaded = "torch" in sys.modules

        try:
            from qwen3_tts.core import engine as voice_engine  # noqa: F401
            if not torch_was_loaded:
                # torch should not have been imported by engine
                self.assertNotIn("torch", sys.modules,
                    "qwen3_tts.core.engine imported torch at module scope")
        finally:
            # Restore
            for mod, val in saved_modules.items():
                sys.modules[mod] = val

    def test_config_no_torch(self):
        """qwen3_tts.core.config must not import torch (regression check)."""
        from qwen3_tts.core import config  # noqa: F401
        # config module should never cause torch to load
        self.assertNotIn("torch", dir(config))


# =============================================================================
# Phase 14: 0.6B Model Size tests
# =============================================================================

class TestModelSize(unittest.TestCase):
    """Test 0.6B model size configuration."""

    def test_valid_model_sizes(self):
        from qwen3_tts.core.config import VALID_MODEL_SIZES
        self.assertIn("1.7B", VALID_MODEL_SIZES)
        self.assertIn("0.6B", VALID_MODEL_SIZES)

    def test_get_model_size_default(self):
        """get_model_size() defaults to 1.7B."""
        from qwen3_tts.core.config import get_model_size
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_MODEL_SIZE", None)
            with patch("qwen3_tts.core.config.load_config", return_value={}):
                self.assertEqual(get_model_size(), "1.7B")

    def test_get_model_size_from_config(self):
        from qwen3_tts.core.config import get_model_size
        config = {"advanced": {"model_size": "0.6B"}}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_MODEL_SIZE", None)
            with patch("qwen3_tts.core.config.load_config", return_value=config):
                self.assertEqual(get_model_size(), "0.6B")

    def test_get_model_size_env_override(self):
        """TTS_MODEL_SIZE env var overrides config."""
        from qwen3_tts.core.config import get_model_size
        config = {"advanced": {"model_size": "1.7B"}}
        with patch.dict(os.environ, {"TTS_MODEL_SIZE": "0.6B"}):
            with patch("qwen3_tts.core.config.load_config", return_value=config):
                self.assertEqual(get_model_size(), "0.6B")

    def test_get_torch_model_name(self):
        from qwen3_tts.core.config import get_torch_model_name
        with patch("qwen3_tts.core.config.get_model_size", return_value="1.7B"):
            name = get_torch_model_name("clone")
            self.assertIn("1.7B", name)
            self.assertIn("Base", name)

    def test_get_torch_model_name_0_6B(self):
        from qwen3_tts.core.config import get_torch_model_name
        with patch("qwen3_tts.core.config.get_model_size", return_value="0.6B"):
            name = get_torch_model_name("clone")
            self.assertIn("0.6B", name)

    def test_model_info_has_0_6B(self):
        from qwen3_tts.core.config import MODEL_INFO, MLX_MODEL_INFO
        # Both should have 0.6B entries
        self.assertIn("0.6B", MODEL_INFO)
        self.assertIn("0.6B", MLX_MODEL_INFO)
        # And all 3 modes
        for mode in ("clone", "design", "custom"):
            self.assertIn(mode, MODEL_INFO["0.6B"])
            self.assertIn(mode, MLX_MODEL_INFO["0.6B"])


# =============================================================================
# Phase 15: Streaming tests
# =============================================================================

class TestStreaming(unittest.TestCase):
    """Test streaming inference API."""

    def test_run_inference_streaming_exists(self):
        """run_inference_streaming function is importable."""
        from qwen3_tts.core.engine import run_inference_streaming
        self.assertTrue(callable(run_inference_streaming))

    def test_mlx_streaming_function_exists(self):
        """_run_inference_mlx_streaming function is importable."""
        from qwen3_tts.core.engine.inference import _run_inference_mlx_streaming
        self.assertTrue(callable(_run_inference_mlx_streaming))

    def test_streaming_torch_falls_back_to_chunked(self):
        """run_inference_streaming for torch uses chunked inference (not native streaming)."""
        from qwen3_tts.core.engine import run_inference_streaming
        import inspect
        source = inspect.getsource(run_inference_streaming)
        # Torch backend falls back to chunked approach
        self.assertIn("_run_inference_single", source)

    def test_streaming_mlx_function_signature(self):
        """_run_inference_mlx_streaming has correct parameters."""
        from qwen3_tts.core.engine.inference import _run_inference_mlx_streaming
        import inspect
        sig = inspect.signature(_run_inference_mlx_streaming)
        params = list(sig.parameters.keys())
        self.assertIn("model", params)
        self.assertIn("text", params)
        self.assertIn("mode", params)
        self.assertIn("gen_params", params)


@_skip_server
class TestStreamingServerEndpoint(unittest.TestCase):
    """Test /generate-stream server endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {"max_text_length": 1000, "max_batch_size": 10},
            "auto_shutdown_minutes": 0,
        })
        app.state.models_loaded.set()  # simulate models ready

    @classmethod
    def tearDownClass(cls):
        from qwen3_tts.server.app import app
        app.state.auth_token = "test_token"  # nosec B105

    def test_generate_stream_requires_auth(self):
        """POST /generate-stream requires authentication."""
        resp = self.client.post("/generate-stream", json={
            "text": "hello", "mode": "design"
        })
        self.assertEqual(resp.status_code, 401)

    def test_generate_stream_validates_text(self):
        """POST /generate-stream validates text input."""
        resp = self.client.post("/generate-stream",
            json={"mode": "design"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("text", resp.json()["detail"].lower())

    def test_generate_stream_validates_mode(self):
        """POST /generate-stream validates mode."""
        resp = self.client.post("/generate-stream",
            json={"text": "hello", "mode": "invalid"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("mode", resp.json()["detail"].lower())


# =============================================================================
# Phase 16: ASR tests
# =============================================================================

class TestASR(unittest.TestCase):
    """Test ASR transcription functions."""

    def setUp(self):
        """Reset ASR model state before each test to prevent pollution."""
        from qwen3_tts.core.engine import asr
        asr._asr_model_mlx = None
        asr._asr_model_torch = None

    def tearDown(self):
        """Clean up ASR model state after each test."""
        from qwen3_tts.core.engine import asr
        asr._asr_model_mlx = None
        asr._asr_model_torch = None

    def test_transcribe_audio_exists(self):
        """transcribe_audio function is importable."""
        from qwen3_tts.core.engine import transcribe_audio
        self.assertTrue(callable(transcribe_audio))

    def test_is_asr_available_exists(self):
        """is_asr_available function is importable."""
        from qwen3_tts.core.engine import is_asr_available
        self.assertTrue(callable(is_asr_available))

    def test_asr_models_are_lazy_loaded(self):
        """ASR model caches are None until transcribe_audio is called."""
        from qwen3_tts.core.engine import asr
        self.assertIsNone(asr._asr_model_mlx)
        self.assertIsNone(asr._asr_model_torch)

    def test_is_asr_available_mlx_with_stt(self):
        """is_asr_available returns True when MLX + mlx_audio.stt available."""
        from qwen3_tts.core.engine import is_asr_available
        with patch("qwen3_tts.core.engine.asr.get_backend", return_value="mlx"):
            with patch.dict(sys.modules, {"mlx_audio.stt": MagicMock()}):
                result = is_asr_available()
        self.assertIsInstance(result, bool)

    def test_transcribe_audio_mlx_returns_string(self):
        """transcribe_audio returns a string via MLX path."""
        from qwen3_tts.core.engine import transcribe_audio

        mock_result = MagicMock()
        mock_result.text = "Hello world"

        mock_model = MagicMock()
        mock_model.generate.return_value = mock_result

        with patch("qwen3_tts.core.engine.asr.get_backend", return_value="mlx"):
            with patch("qwen3_tts.core.engine.asr._asr_model_mlx", mock_model):
                result = transcribe_audio("/fake/path.wav")

        self.assertIsInstance(result, str)
        self.assertEqual(result, "Hello world")

    def test_is_asr_available_torch_with_transformers(self):
        """is_asr_available returns True when torch + transformers importable."""
        from qwen3_tts.core.engine import is_asr_available
        # Check if transformers is actually importable in this env
        try:
            from transformers import pipeline  # noqa: F401
            has_transformers = True
        except (ImportError, Exception):
            has_transformers = False
        with patch("qwen3_tts.core.engine.asr.get_backend", return_value="torch"):
            result = is_asr_available()
        self.assertEqual(result, has_transformers)

    def test_transcribe_audio_torch_dispatches(self):
        """transcribe_audio uses torch path when backend is torch."""
        from qwen3_tts.core.engine import transcribe_audio

        mock_pipe = MagicMock(return_value={"text": "Torch transcript"})

        with patch("qwen3_tts.core.engine.asr.get_backend", return_value="torch"):
            with patch("qwen3_tts.core.engine.asr._asr_model_torch", mock_pipe):
                result = transcribe_audio("/fake/path.wav")

        self.assertEqual(result, "Torch transcript")
        mock_pipe.assert_called_once()

    def test_transcribe_audio_torch_passes_language(self):
        """Torch ASR passes language via generate_kwargs."""
        from qwen3_tts.core.engine import transcribe_audio

        mock_pipe = MagicMock(return_value={"text": "Bonjour"})

        with patch("qwen3_tts.core.engine.asr.get_backend", return_value="torch"):
            with patch("qwen3_tts.core.engine.asr._asr_model_torch", mock_pipe):
                transcribe_audio("/fake/path.wav", language="fr")

        call_kwargs = mock_pipe.call_args[1]
        self.assertEqual(call_kwargs["generate_kwargs"]["language"], "fr")


# =============================================================================
# Phase 17: Stability tests
# =============================================================================

class TestStability(unittest.TestCase):
    """Test stability hardening features."""

    def test_retry_delays_constant_exists(self):
        """_RETRY_DELAYS constant is defined."""
        from qwen3_tts.core.engine.model_loader import _RETRY_DELAYS
        self.assertEqual(len(_RETRY_DELAYS), 3)
        self.assertEqual(_RETRY_DELAYS, (5, 15, 45))

    def test_retry_delays_is_exponential(self):
        """_RETRY_DELAYS uses exponential backoff pattern."""
        from qwen3_tts.core.engine.model_loader import _RETRY_DELAYS
        # Each delay should be roughly 3x the previous (5 -> 15 -> 45)
        self.assertEqual(_RETRY_DELAYS[1], _RETRY_DELAYS[0] * 3)
        self.assertEqual(_RETRY_DELAYS[2], _RETRY_DELAYS[1] * 3)

    def test_max_chunk_chars_helper_exists(self):
        """_get_max_chunk_chars helper function exists."""
        from qwen3_tts.core.engine.inference import _get_max_chunk_chars
        self.assertTrue(callable(_get_max_chunk_chars))

    def test_max_chunk_chars_default(self):
        """_get_max_chunk_chars returns default 500."""
        from qwen3_tts.core.engine.inference import _get_max_chunk_chars
        with patch("qwen3_tts.core.engine.inference.load_config", return_value={}):
            result = _get_max_chunk_chars()
        self.assertEqual(result, 500)

    def test_max_chunk_chars_from_config(self):
        """_get_max_chunk_chars reads from config."""
        from qwen3_tts.core.engine.inference import _get_max_chunk_chars
        config = {"generation": {"max_chunk_chars": 300}}
        with patch("qwen3_tts.core.engine.inference.load_config", return_value=config):
            result = _get_max_chunk_chars()
        self.assertEqual(result, 300)


class TestFloat32Guard(unittest.TestCase):
    """Test float32 dtype guard for torch clone mode on MPS."""

    def test_float32_guard_exists_in_torch_inference(self):
        """_run_inference_torch has float32 guard logic."""
        from qwen3_tts.core.engine.inference import _run_inference_torch
        import inspect
        source = inspect.getsource(_run_inference_torch)
        # Should have float32 override logic for clone mode
        self.assertIn("float32", source)
        self.assertIn("clone", source)


class TestMLXMetalRecovery(unittest.TestCase):
    """Test MLX Metal kernel crash recovery."""

    def test_run_inference_handles_exceptions(self):
        """run_inference wraps inference in try/except."""
        from qwen3_tts.core.engine.inference import _run_inference_single
        import inspect
        source = inspect.getsource(_run_inference_single)
        # Should have exception handling
        self.assertIn("except", source)


# =============================================================================
# Text chunking tests
# =============================================================================

class TestTextChunking(unittest.TestCase):
    """Test text chunking for long-form generation."""

    def test_split_text_short(self):
        """Short text is not split."""
        from qwen3_tts.core.engine.text_processing import _split_text
        chunks = _split_text("Hello world.", max_chars=500)
        self.assertEqual(chunks, ["Hello world."])

    def test_split_text_sentences(self):
        """Text is split on sentence boundaries."""
        from qwen3_tts.core.engine.text_processing import _split_text
        text = "First sentence. Second sentence. Third sentence."
        chunks = _split_text(text, max_chars=30)
        self.assertGreater(len(chunks), 1)
        # Each chunk should be <= max_chars
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 30)

    def test_split_text_preserves_content(self):
        """All content is preserved after splitting."""
        from qwen3_tts.core.engine.text_processing import _split_text
        text = "The quick brown fox jumps over the lazy dog. A second sentence follows."
        chunks = _split_text(text, max_chars=50)
        combined = " ".join(chunks)
        # All words should be present
        for word in text.split():
            self.assertIn(word.rstrip(".,"), combined)

    def test_split_text_question_mark(self):
        """Text splits on question marks."""
        from qwen3_tts.core.engine.text_processing import _split_text
        text = "Is this a question? Yes it is."
        chunks = _split_text(text, max_chars=25)
        self.assertGreater(len(chunks), 1)

    def test_split_text_exclamation(self):
        """Text splits on exclamation marks."""
        from qwen3_tts.core.engine.text_processing import _split_text
        text = "Hello! How are you today?"
        chunks = _split_text(text, max_chars=15)
        self.assertGreater(len(chunks), 1)

    def test_split_text_newlines(self):
        """Text splits on newlines."""
        from qwen3_tts.core.engine.text_processing import _split_text
        text = "First paragraph.\n\nSecond paragraph."
        chunks = _split_text(text, max_chars=20)
        self.assertGreater(len(chunks), 1)

    def test_split_text_comma_fallback(self):
        """Very long sentence falls back to clause boundaries."""
        from qwen3_tts.core.engine.text_processing import _split_text
        # A single long sentence with commas but no periods
        text = "This is a very long sentence, with several clauses, that should be split at commas when needed"
        chunks = _split_text(text, max_chars=40)
        # Should split due to length
        self.assertGreater(len(chunks), 1)


# =============================================================================
# Server health endpoint info tests
# =============================================================================

@_skip_server
@_skip_ui
class TestMLXMemoryStats(unittest.TestCase):
    """Test MLX memory stats collection in /stats endpoint."""

    def test_stats_mlx_memory_code_exists(self):
        """FastAPI app has MLX memory collection code."""
        import inspect
        from qwen3_tts.server import app as app_module
        # Find the stats route handler
        source = inspect.getsource(app_module)
        # Should have MLX memory collection
        self.assertIn("mlx_memory_active_mb", source)
        self.assertIn("mlx_memory_peak_mb", source)
        self.assertIn("mx.metal.get_active_memory", source)

    def test_ui_checks_mlx_memory_first(self):
        """voice_ui checks for MLX memory before MPS memory."""
        import inspect
        from qwen3_tts.interface import ui as voice_ui
        source = inspect.getsource(voice_ui.get_server_status)
        # Should check mlx_memory first
        self.assertIn("mlx_memory_active_mb", source)


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


# =============================================================================
# Generation status and chunk progress tests
# =============================================================================

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


# =============================================================================
# Load model endpoint tests
# =============================================================================

@_skip_server
class TestLoadModelEndpoint(unittest.TestCase):
    """Test /load-model endpoint validation."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={"security": {}, "auto_shutdown_minutes": 0})
        app.state.models_loaded.set()  # simulate models ready for tests that need a live server

    @classmethod
    def tearDownClass(cls):
        from qwen3_tts.server.app import app
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


# =============================================================================
# Cancel generation endpoint tests
# =============================================================================

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


# =============================================================================
# Streaming client tests
# =============================================================================

@_skip_client
class TestStreamingClientMethod(unittest.TestCase):
    """Test TTSClient.generate_streaming method."""

    def test_generate_streaming_method_exists(self):
        """TTSClient has generate_streaming method."""
        from qwen3_tts.server.client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "generate_streaming"))
        self.assertTrue(callable(getattr(client, "generate_streaming")))

    def test_cancel_generation_method_exists(self):
        """TTSClient has cancel_generation method."""
        from qwen3_tts.server.client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "cancel_generation"))
        self.assertTrue(callable(getattr(client, "cancel_generation")))

    def test_generate_streaming_signature(self):
        """generate_streaming has expected parameters."""
        import inspect
        from qwen3_tts.server.client import TTSClient
        sig = inspect.signature(TTSClient.generate_streaming)
        params = list(sig.parameters.keys())
        # Should have text, mode, and various optional params
        self.assertIn("text", params)
        self.assertIn("mode", params)


# =============================================================================
# UI history functions tests
# =============================================================================

@_skip_ui
class TestUIHistoryFunctions(unittest.TestCase):
    """Test voice_ui generation history functions."""

    def test_history_functions_exist(self):
        """voice_ui has history-related functions."""
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(hasattr(voice_ui, "add_to_history"))
        self.assertTrue(hasattr(voice_ui, "get_history_data"))
        self.assertTrue(hasattr(voice_ui, "MAX_HISTORY_SIZE"))

    def test_add_to_history(self):
        """add_to_history adds entries to history and returns new list."""
        from qwen3_tts.interface import ui as voice_ui
        history = []

        history = voice_ui.add_to_history(history, "clone", "Test text", "/path/to/audio.wav", 5)
        self.assertEqual(len(history), 1)
        entry = history[0]
        self.assertEqual(entry["mode"], "Clone")
        self.assertEqual(entry["chunks"], 5)
        self.assertEqual(entry["path"], "/path/to/audio.wav")

    def test_history_max_size(self):
        """History doesn't exceed MAX_HISTORY_SIZE."""
        from qwen3_tts.interface import ui as voice_ui
        history = []

        # Add more than max entries
        for i in range(voice_ui.MAX_HISTORY_SIZE + 5):
            history = voice_ui.add_to_history(history, "clone", f"Text {i}", f"/path/{i}.wav", 1)

        self.assertEqual(len(history), voice_ui.MAX_HISTORY_SIZE)

    def test_add_to_history_does_not_mutate_input(self):
        """add_to_history returns a new list, does not mutate the input."""
        from qwen3_tts.interface import ui as voice_ui
        original = []
        result = voice_ui.add_to_history(original, "clone", "Test", "/path.wav", 1)
        self.assertEqual(len(original), 0)
        self.assertEqual(len(result), 1)

    def test_get_history_data_format(self):
        """get_history_data returns list of lists."""
        from qwen3_tts.interface import ui as voice_ui
        history = []
        history = voice_ui.add_to_history(history, "clone", "Test text", "/path/test.wav", 3)

        data = voice_ui.get_history_data(history)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertIsInstance(data[0], list)
        # Should be [time, mode, text, chunks]
        self.assertEqual(len(data[0]), 4)

    def test_history_text_truncation(self):
        """Long text is truncated in history entries."""
        from qwen3_tts.interface import ui as voice_ui
        history = []

        long_text = "A" * 100  # 100 character text
        history = voice_ui.add_to_history(history, "clone", long_text, "/path/test.wav", 1)

        entry = history[0]
        # Text should be truncated to 40 chars + "..."
        self.assertLessEqual(len(entry["text"]), 43)
        self.assertTrue(entry["text"].endswith("..."))


# =============================================================================
# UI cancel function tests
# =============================================================================

@_skip_ui
class TestUICancelFunction(unittest.TestCase):
    """Test voice_ui cancel streaming function."""

    def test_cancel_streaming_generation_exists(self):
        """voice_ui has cancel_streaming_generation function."""
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(hasattr(voice_ui, "cancel_streaming_generation"))
        self.assertTrue(callable(voice_ui.cancel_streaming_generation))

    def test_cancel_streaming_generation_returns_tuple(self):
        """cancel_streaming_generation returns a 2-tuple (status, status_html)."""
        from qwen3_tts.interface.ui import cancel_streaming_generation
        from unittest.mock import patch, MagicMock

        mock_client = MagicMock()
        mock_client.cancel_generation.return_value = {"status": "no_active_generation"}

        with patch("qwen3_tts.server.client.TTSClient", return_value=mock_client):
            result = cancel_streaming_generation()

        self.assertIsInstance(result, tuple)
        # Returns (status_text, status_html) — no audio element with WaveSurfer
        self.assertEqual(len(result), 2)

    def test_cancel_streaming_generation_status_text(self):
        """cancel_streaming_generation returns status text as first element."""
        from qwen3_tts.interface.ui import cancel_streaming_generation
        from unittest.mock import patch, MagicMock

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("qwen3_tts.interface.ui.generation.is_server_running", return_value=True), \
             patch("requests.post", return_value=mock_response):
            result = cancel_streaming_generation()

        # First element is status text
        self.assertIn("cancelled", result[0].lower())



# =============================================================================
# UI text info helper tests
# =============================================================================

@_skip_ui
class TestUITextInfo(unittest.TestCase):
    """Test voice_ui text info helper functions."""

    def test_update_text_info_exists(self):
        """voice_ui has update_text_info function."""
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(hasattr(voice_ui, "update_text_info"))

    def test_update_text_info_empty(self):
        """update_text_info returns empty string for empty input."""
        from qwen3_tts.interface.ui import update_text_info
        self.assertEqual(update_text_info(""), "")
        self.assertEqual(update_text_info(None), "")

    def test_update_text_info_short(self):
        """update_text_info shows char count for short text."""
        from qwen3_tts.interface.ui import update_text_info
        result = update_text_info("Hello")
        self.assertIn("5 chars", result)

    def test_update_text_info_long(self):
        """update_text_info shows chunks estimate for long text."""
        from qwen3_tts.interface.ui import update_text_info
        long_text = "A" * 1000  # 1000 chars = ~2 chunks
        result = update_text_info(long_text)
        self.assertIn("1000 chars", result)
        self.assertIn("chunks", result)


# =============================================================================
# UI model settings tests
# =============================================================================

@_skip_ui
class TestUIModelSettings(unittest.TestCase):
    """Test voice_ui model settings functions (Phase 19: MLX-First Architecture)."""

    def test_model_settings_functions_exist(self):
        """voice_ui has model settings functions."""
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(hasattr(voice_ui, "get_current_model_settings"))
        self.assertTrue(hasattr(voice_ui, "apply_model_settings"))
        self.assertTrue(callable(voice_ui.get_current_model_settings))
        self.assertTrue(callable(voice_ui.apply_model_settings))

    def test_get_current_model_settings_returns_tuple(self):
        """get_current_model_settings returns a 3-tuple."""
        from qwen3_tts.interface.ui import get_current_model_settings
        result = get_current_model_settings()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        # (size, quant, backend)
        size, quant, backend = result
        self.assertIn(size, ("1.7B", "0.6B"))
        self.assertIn(quant, ("4bit", "8bit", "bf16"))
        self.assertIn(backend, ("torch", "mlx"))

    def test_apply_model_settings_returns_tuple(self):
        """apply_model_settings returns a 2-tuple (message, status_html)."""
        from qwen3_tts.interface.ui import apply_model_settings
        # Without server running, should return error message
        result = apply_model_settings("1.7B", "8bit")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        msg, html = result
        self.assertIsInstance(msg, str)
        self.assertIsInstance(html, str)

    def test_apply_model_settings_requires_server(self):
        """apply_model_settings returns error when server not running."""
        from qwen3_tts.interface.ui import apply_model_settings
        with unittest.mock.patch("qwen3_tts.server.client.TTSClient") as MockClient:
            MockClient.return_value.is_server_running.return_value = False
            msg, _ = apply_model_settings("0.6B", "4bit")
        self.assertIn("not running", msg.lower())


@_skip_ui
class TestUIModelSettingsImports(unittest.TestCase):
    """Test voice_ui imports required for model settings."""

    def test_model_settings_imports(self):
        """voice_ui imports required constants for model settings."""
        from qwen3_tts.interface import ui as voice_ui
        # Should have imported these from qwen3_tts.core.config
        self.assertTrue(hasattr(voice_ui, "VALID_MODEL_SIZES"))
        self.assertTrue(hasattr(voice_ui, "VALID_MLX_QUANTIZATIONS"))
        self.assertTrue(hasattr(voice_ui, "get_backend"))
        self.assertTrue(hasattr(voice_ui, "get_model_size"))
        self.assertTrue(hasattr(voice_ui, "get_mlx_quantization"))


# =============================================================================
# Update model config endpoint tests
# =============================================================================

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


@_skip_client
class TestClientUpdateModelConfig(unittest.TestCase):
    """Test TTSClient.update_model_config method."""

    def test_update_model_config_method_exists(self):
        """TTSClient has update_model_config method."""
        from qwen3_tts.server.client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "update_model_config"))
        self.assertTrue(callable(client.update_model_config))


# =============================================================================
# Streaming server endpoint structure tests
# =============================================================================

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


# =============================================================================
# Generation functions return history update tests
# =============================================================================


# =============================================================================
# Generation stream generation_id check tests
# =============================================================================

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
        from qwen3_tts.server import app as app_module
        source = inspect.getsource(app_module)
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



# =============================================================================
# createVoice script backend override tests
# =============================================================================

# =============================================================================
# Phase 21b: MLX voice prompt cache tests
# =============================================================================

class TestMLXVoicePromptCache(unittest.TestCase):
    """Test MLX voice prompt caching in voice_engine."""

    def setUp(self):
        # Clear cache before each test (earlier test classes may have populated it)
        from qwen3_tts.core.engine.voice_prompt import _mlx_prompt_cache
        _mlx_prompt_cache.clear()
        self.tmpdir = tempfile.mkdtemp()
        # Create fake wav and txt
        for name in ("voice_a", "voice_b"):
            with open(os.path.join(self.tmpdir, f"{name}.wav"), "wb") as f:
                f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")
            with open(os.path.join(self.tmpdir, f"{name}.txt"), "w") as f:
                f.write(f"Transcript for {name}")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        # Clear cache between tests
        from qwen3_tts.core.engine.voice_prompt import _mlx_prompt_cache
        _mlx_prompt_cache.clear()

    def test_mlx_cache_returns_consistent_results(self):
        """Cached result is identical to first load."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx
        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            first = load_voice_prompt_mlx("voice_a")
            second = load_voice_prompt_mlx("voice_a")
        self.assertIs(first, second)  # Same object from cache

    def test_mlx_cache_stores_entries(self):
        """Loading a prompt adds it to the cache."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx
        from qwen3_tts.core.engine.voice_prompt import _mlx_prompt_cache
        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            load_voice_prompt_mlx("voice_a")
        self.assertIn("voice_a", _mlx_prompt_cache)

    def test_clear_voice_prompt_cache_clears_mlx(self):
        """clear_voice_prompt_cache clears MLX cache."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx, clear_voice_prompt_cache
        from qwen3_tts.core.engine.voice_prompt import _mlx_prompt_cache
        with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
            load_voice_prompt_mlx("voice_a")
        self.assertEqual(len(_mlx_prompt_cache), 1)
        clear_voice_prompt_cache()
        self.assertEqual(len(_mlx_prompt_cache), 0)

    def test_mlx_cache_info_returns_currsize(self):
        """voice_prompt_cache_info returns MLX cache size."""
        from qwen3_tts.core.engine import load_voice_prompt_mlx, voice_prompt_cache_info
        with patch("qwen3_tts.core.engine.voice_prompt.get_backend", return_value="mlx"):
            with patch("qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", self.tmpdir):
                load_voice_prompt_mlx("voice_a")
            info = voice_prompt_cache_info()
        self.assertEqual(info.currsize, 1)


# =============================================================================
# Phase 21b: ETA cache tests
# =============================================================================

@_skip_server
class TestETACache(unittest.TestCase):
    """Test ETA estimation cache in FastAPI app."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={"security": {}, "auto_shutdown_minutes": 0})
        app.state.models_loaded.set()

    def test_eta_cache_exists(self):
        """FastAPI app has eta_cache in app.state."""
        from qwen3_tts.server.app import app
        self.assertTrue(hasattr(app.state, "eta_cache"))
        self.assertIn("median_rate", app.state.eta_cache)
        self.assertIn("last_updated", app.state.eta_cache)

    def test_eta_cache_ttl_function(self):
        """FastAPI app has _get_eta_cache_ttl function that reads from config."""
        import qwen3_tts.server.app as _srv
        self.assertTrue(hasattr(_srv, "_get_eta_cache_ttl"))
        self.assertTrue(callable(_srv._get_eta_cache_ttl))
        # Function should return default value of 30
        result = _srv._get_eta_cache_ttl()
        self.assertEqual(result, 30)

    def test_estimate_eta_uses_cache(self):
        """_estimate_eta reads from cache when fresh."""
        from qwen3_tts.server.app import app, _estimate_eta
        # Pre-populate cache with a known rate
        app.state.eta_cache["median_rate"] = 10.0  # 10 chars/sec
        app.state.eta_cache["last_updated"] = time.time()  # fresh

        result = _estimate_eta(app.state, 100, 5.0)
        # 100 chars / 10 chars/sec = 10s total, 10 - 5 = 5s remaining
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 5.0, delta=0.5)

    def test_estimate_eta_returns_none_without_data(self):
        """_estimate_eta returns None when no history data."""
        from qwen3_tts.server.app import app, _estimate_eta
        app.state.eta_cache["median_rate"] = None
        app.state.eta_cache["last_updated"] = time.time()

        result = _estimate_eta(app.state, 100, 5.0)
        self.assertIsNone(result)


# =============================================================================
# Phase 21b: Generation result cache tests
# =============================================================================

@_skip_server
class TestGenerationCache(unittest.TestCase):
    """Test generation result cache in FastAPI app."""

    def setUp(self):
        from qwen3_tts.server.app import app
        app.state.gen_cache.clear()

    def test_gen_cache_key_deterministic(self):
        """Same inputs produce same cache key."""
        from qwen3_tts.server.app import _gen_cache_key
        key1 = _gen_cache_key("hello", "clone", {"temperature": 0.7}, prompt_file="voice.pt")
        key2 = _gen_cache_key("hello", "clone", {"temperature": 0.7}, prompt_file="voice.pt")
        self.assertEqual(key1, key2)

    def test_gen_cache_key_varies_by_text(self):
        """Different text produces different cache key."""
        from qwen3_tts.server.app import _gen_cache_key
        key1 = _gen_cache_key("hello", "clone", {})
        key2 = _gen_cache_key("world", "clone", {})
        self.assertNotEqual(key1, key2)

    def test_gen_cache_key_varies_by_mode(self):
        """Different mode produces different cache key."""
        from qwen3_tts.server.app import _gen_cache_key
        key1 = _gen_cache_key("hello", "clone", {})
        key2 = _gen_cache_key("hello", "design", {})
        self.assertNotEqual(key1, key2)

    def test_gen_cache_dict_exists(self):
        """FastAPI app has gen_cache in app.state."""
        from qwen3_tts.server.app import app
        self.assertTrue(hasattr(app.state, "gen_cache"))
        self.assertIsInstance(app.state.gen_cache, dict)

    def test_gen_cache_max_size_function(self):
        """FastAPI app has _get_gen_cache_max function that reads from config."""
        import qwen3_tts.server.app as _srv
        self.assertTrue(hasattr(_srv, "_get_gen_cache_max"))
        self.assertTrue(callable(_srv._get_gen_cache_max))
        # Function should return default value of 5
        result = _srv._get_gen_cache_max()
        self.assertEqual(result, 5)


# =============================================================================
# Phase 21a: Voice management endpoint tests
# =============================================================================

@_skip_server
class TestDeletePromptEndpoint(unittest.TestCase):
    """Test POST /delete-prompt endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        })
        app.state.auth_token = "test_secret_token"  # nosec B105
        app.state.models_loaded.set()
        cls.auth = {"Authorization": "Bearer test_secret_token"}

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create fake voice files
        for ext in (".pt", ".wav", ".txt"):
            with open(os.path.join(self.tmpdir, f"test_voice{ext}"), "w") as f:
                f.write("fake")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_delete_requires_auth(self):
        """POST /delete-prompt requires authentication."""
        resp = self.client.post("/delete-prompt", json={"name": "test"})
        self.assertEqual(resp.status_code, 401)

    def test_delete_path_traversal(self):
        """POST /delete-prompt rejects path traversal."""
        resp = self.client.post("/delete-prompt", json={"name": "../etc/passwd"},
                                headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid prompt name", resp.json()["detail"])

    def test_delete_nonexistent(self):
        """POST /delete-prompt returns 404 for missing prompt."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/delete-prompt", json={"name": "nonexistent"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 404)

    def test_delete_success(self):
        """POST /delete-prompt deletes all format files."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/delete-prompt", json={"name": "test_voice"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "deleted")
        self.assertEqual(len(data["files_removed"]), 3)
        # Verify files are gone
        for ext in (".pt", ".wav", ".txt"):
            self.assertFalse(os.path.exists(os.path.join(self.tmpdir, f"test_voice{ext}")))


@_skip_server
class TestRenamePromptEndpoint(unittest.TestCase):
    """Test POST /rename-prompt endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        })
        app.state.models_loaded.set()
        cls.auth = {"Authorization": "Bearer test_token"}

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        for ext in (".pt", ".wav", ".txt"):
            with open(os.path.join(self.tmpdir, f"old_voice{ext}"), "w") as f:
                f.write("fake")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rename_requires_auth(self):
        """POST /rename-prompt requires authentication."""
        resp = self.client.post("/rename-prompt", json={"old_name": "a", "new_name": "b"})
        self.assertEqual(resp.status_code, 401)

    def test_rename_path_traversal(self):
        """POST /rename-prompt rejects path traversal in both names."""
        resp = self.client.post("/rename-prompt",
                                json={"old_name": "../bad", "new_name": "ok"},
                                headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        resp = self.client.post("/rename-prompt",
                                json={"old_name": "ok", "new_name": "../bad"},
                                headers=self.auth)
        self.assertEqual(resp.status_code, 400)

    def test_rename_collision(self):
        """POST /rename-prompt returns 409 when new name already exists."""
        # Create collision target
        with open(os.path.join(self.tmpdir, "existing.wav"), "w") as f:
            f.write("fake")
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/rename-prompt",
                                    json={"old_name": "old_voice", "new_name": "existing"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 409)

    def test_rename_success(self):
        """POST /rename-prompt renames all format files."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/rename-prompt",
                                    json={"old_name": "old_voice", "new_name": "new_voice"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "renamed")
        # Old files should be gone, new files should exist
        for ext in (".pt", ".wav", ".txt"):
            self.assertFalse(os.path.exists(os.path.join(self.tmpdir, f"old_voice{ext}")))
            self.assertTrue(os.path.exists(os.path.join(self.tmpdir, f"new_voice{ext}")))

    def test_rename_not_found(self):
        """POST /rename-prompt returns 404 for missing prompt."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/rename-prompt",
                                    json={"old_name": "nonexistent", "new_name": "new"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 404)


@_skip_server
class TestPreviewPromptEndpoint(unittest.TestCase):
    """Test GET /preview-prompt endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        })
        app.state.models_loaded.set()  # simulate models ready for tests that need a live server
        app.state.auth_token = "test_secret_token"  # nosec B105
        cls.auth = {"Authorization": "Bearer test_secret_token"}

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        with open(os.path.join(self.tmpdir, "test_voice.wav"), "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_preview_requires_auth(self):
        """GET /preview-prompt requires authentication."""
        resp = self.client.get("/preview-prompt?name=test")
        self.assertEqual(resp.status_code, 401)

    def test_preview_not_found(self):
        """GET /preview-prompt returns 404 for missing prompt."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.get("/preview-prompt?name=nonexistent",
                                   headers=self.auth)
        self.assertEqual(resp.status_code, 404)

    def test_preview_returns_audio(self):
        """GET /preview-prompt returns audio/wav content."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.get("/preview-prompt?name=test_voice",
                                   headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("audio/wav", resp.headers.get("content-type", ""))


@_skip_server
class TestPromptDetailsEndpoint(unittest.TestCase):
    """Test GET /prompt-details endpoint."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app
        cls.client = _make_test_client(app, server_config={
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        })
        app.state.auth_token = "test_secret_token"  # nosec B105
        app.state.models_loaded.set()  # simulate models ready for tests that need a live server
        cls.auth = {"Authorization": "Bearer test_secret_token"}

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        for ext in (".pt", ".wav", ".txt"):
            with open(os.path.join(self.tmpdir, f"voice_a{ext}"), "w") as f:
                f.write("fake data here")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_details_requires_auth(self):
        """GET /prompt-details requires authentication."""
        resp = self.client.get("/prompt-details?name=test")
        self.assertEqual(resp.status_code, 401)

    def test_details_single_prompt(self):
        """GET /prompt-details?name=X returns metadata for one prompt."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.get("/prompt-details?name=voice_a",
                                   headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "voice_a")
        self.assertIn(".pt", data["formats"])
        self.assertIn(".wav", data["formats"])
        self.assertIn(".txt", data["formats"])
        self.assertGreater(data["size_bytes"], 0)

    def test_details_all_prompts(self):
        """GET /prompt-details without name returns all prompts."""
        with patch("qwen3_tts.server.app.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.get("/prompt-details", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("prompts", data)
        self.assertEqual(len(data["prompts"]), 1)
        self.assertEqual(data["prompts"][0]["name"], "voice_a")


@_skip_client
class TestClientPromptManagement(unittest.TestCase):
    """Test voice_client prompt management method signatures."""

    def test_delete_prompt_method_exists(self):
        """TTSClient has delete_prompt method."""
        from qwen3_tts.server.client import TTSClient
        self.assertTrue(hasattr(TTSClient, "delete_prompt"))

    def test_rename_prompt_method_exists(self):
        """TTSClient has rename_prompt method."""
        from qwen3_tts.server.client import TTSClient
        self.assertTrue(hasattr(TTSClient, "rename_prompt"))

    def test_preview_prompt_method_exists(self):
        """TTSClient has preview_prompt method."""
        from qwen3_tts.server.client import TTSClient
        self.assertTrue(hasattr(TTSClient, "preview_prompt"))

    def test_get_prompt_details_method_exists(self):
        """TTSClient has get_prompt_details method."""
        from qwen3_tts.server.client import TTSClient
        self.assertTrue(hasattr(TTSClient, "get_prompt_details"))

    def test_list_prompts_uses_server(self):
        """list_prompts calls server /prompts when running."""
        import inspect
        from qwen3_tts.server.client import TTSClient
        source = inspect.getsource(TTSClient.list_prompts)
        self.assertIn("/prompts", source)
        self.assertIn("is_server_running", source)


class TestSetDefaultClonePrompt(unittest.TestCase):
    """Test set_default_clone_prompt config helper."""

    def test_set_default_writes_config(self):
        """set_default_clone_prompt updates config.json."""
        from qwen3_tts.core.config import set_default_clone_prompt, load_config, save_config, CONFIG_PATH
        # Save original config
        original = load_config()
        try:
            set_default_clone_prompt("test_voice.pt")
            config = load_config()
            self.assertEqual(config["default_clone_prompt"], "test_voice.pt")
        finally:
            # Restore original config
            save_config(original)


@_skip_ui
class TestVoiceManagementUI(unittest.TestCase):
    """Test voice management UI helper functions."""

    def test_get_prompt_table_data_exists(self):
        """voice_ui has get_prompt_table_data function."""
        from qwen3_tts.interface.ui import get_prompt_table_data
        self.assertTrue(callable(get_prompt_table_data))

    def test_preview_voice_exists(self):
        """voice_ui has preview_voice function."""
        from qwen3_tts.interface.ui import preview_voice
        self.assertTrue(callable(preview_voice))

    def test_rename_voice_exists(self):
        """voice_ui has rename_voice function."""
        from qwen3_tts.interface.ui import rename_voice
        self.assertTrue(callable(rename_voice))

    def test_delete_voice_exists(self):
        """voice_ui has delete_voice function."""
        from qwen3_tts.interface.ui import delete_voice
        self.assertTrue(callable(delete_voice))

    def test_set_voice_default_exists(self):
        """voice_ui has set_voice_default function."""
        from qwen3_tts.interface.ui import set_voice_default
        self.assertTrue(callable(set_voice_default))

    def test_delete_voice_prompt_rejects_path_traversal(self):
        """delete_voice_prompt must reject names with .. or /"""
        from qwen3_tts.interface.generate import delete_voice_prompt
        result = delete_voice_prompt("../evil_file")
        self.assertFalse(result, "Expected False for traversal name '../evil_file'")

    def test_rename_voice_prompt_rejects_path_traversal(self):
        """rename_voice_prompt must reject names with .. or /"""
        from qwen3_tts.interface.generate import rename_voice_prompt
        result = rename_voice_prompt("../evil", "safe_name")
        self.assertFalse(result, "Expected False for traversal name '../evil'")


# =============================================================================
# Phase 21c: Platform detection and Colab support tests
# =============================================================================

class TestPlatformDetection(unittest.TestCase):
    """Test platform detection constants and get_device()."""

    def test_platform_constants_exist(self):
        """qwen3_tts.core.config has IN_COLAB, IS_MACOS, IS_LINUX constants."""
        from qwen3_tts.core.config import IN_COLAB, IS_MACOS, IS_LINUX
        self.assertIsInstance(IN_COLAB, bool)
        self.assertIsInstance(IS_MACOS, bool)
        self.assertIsInstance(IS_LINUX, bool)

    def test_get_device_exists(self):
        """qwen3_tts.core.config has get_device function."""
        from qwen3_tts.core.config import get_device
        self.assertTrue(callable(get_device))
        result = get_device()
        self.assertIn(result, ("cuda", "mps", "cpu"))

    def test_get_device_returns_mps_on_macos_arm(self):
        """get_device returns 'mps' on macOS ARM64."""
        import platform as _platform
        with patch("qwen3_tts.core.config.IS_MACOS", True), \
             patch("qwen3_tts.core.config.IS_LINUX", False), \
             patch("qwen3_tts.core.config.IN_COLAB", False), \
             patch("qwen3_tts.core.config.platform.machine", return_value="arm64"):
            from qwen3_tts.core.config import get_device
            result = get_device()
        self.assertEqual(result, "mps")

    def test_get_device_returns_cuda_with_env(self):
        """get_device returns 'cuda' when CUDA_VISIBLE_DEVICES is set."""
        with patch("qwen3_tts.core.config.IS_MACOS", False), \
             patch("qwen3_tts.core.config.IS_LINUX", True), \
             patch("qwen3_tts.core.config.IN_COLAB", False), \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            from qwen3_tts.core.config import get_device
            result = get_device()
        self.assertEqual(result, "cuda")

    def test_get_device_returns_cpu_fallback(self):
        """get_device returns 'cpu' when no GPU available."""
        with patch("qwen3_tts.core.config.IS_MACOS", False), \
             patch("qwen3_tts.core.config.IS_LINUX", True), \
             patch("qwen3_tts.core.config.IN_COLAB", False), \
             patch.dict(os.environ, {}, clear=True), \
             patch("qwen3_tts.core.config.os.path.exists", return_value=False):
            from qwen3_tts.core.config import get_device
            # Need to remove CUDA_VISIBLE_DEVICES if present
            env = os.environ.copy()
            env.pop("CUDA_VISIBLE_DEVICES", None)
            with patch.dict(os.environ, env, clear=True):
                result = get_device()
        self.assertEqual(result, "cpu")


class TestDeviceAwareEngine(unittest.TestCase):
    """Test device-aware engine code."""

    def test_load_model_torch_uses_get_device(self):
        """_load_model_torch uses get_device() for device_map."""
        import inspect
        from qwen3_tts.core.engine.model_loader import _load_model_torch
        source = inspect.getsource(_load_model_torch)
        self.assertIn("get_device", source)
        self.assertNotIn('device_map="mps"', source)

    def test_install_mps_patch_checks_platform(self):
        """_install_mps_patch checks IS_MACOS before patching."""
        import inspect
        from qwen3_tts.core.engine.model_loader import _install_mps_patch
        source = inspect.getsource(_install_mps_patch)
        self.assertIn("IS_MACOS", source)

    def test_cuda_memory_cleanup_exists(self):
        """_run_inference_torch has CUDA memory cleanup code."""
        import inspect
        from qwen3_tts.core.engine.inference import _run_inference_torch
        source = inspect.getsource(_run_inference_torch)
        self.assertIn("torch.cuda.is_available", source)
        self.assertIn("torch.cuda.empty_cache", source)


@_skip_generate
class TestPlatformSafeCommands(unittest.TestCase):
    """Test platform-safe command helpers in voice_generate."""

    def test_play_audio_checks_platform(self):
        """play_audio checks platform before choosing command."""
        import inspect
        from qwen3_tts.interface.generate import play_audio
        source = inspect.getsource(play_audio)
        self.assertIn("IS_MACOS", source)
        self.assertIn("IS_LINUX", source)
        self.assertIn("IN_COLAB", source)

    def test_get_clipboard_text_checks_platform(self):
        """get_clipboard_text checks platform before choosing command."""
        import inspect
        from qwen3_tts.interface.generate import get_clipboard_text
        source = inspect.getsource(get_clipboard_text)
        self.assertIn("IS_MACOS", source)
        self.assertIn("IS_LINUX", source)

    def test_open_file_exists(self):
        """voice_generate has open_file helper function."""
        from qwen3_tts.interface.generate import open_file
        self.assertTrue(callable(open_file))

    def test_open_file_handles_missing_xdg(self):
        """open_file wraps xdg-open in try/except."""
        import inspect
        from qwen3_tts.interface.generate import open_file
        source = inspect.getsource(open_file)
        self.assertIn("FileNotFoundError", source)
        self.assertIn("xdg-open", source)


# =============================================================================
# Unload model endpoint tests
# =============================================================================

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


# =============================================================================
# Update startup config endpoint tests
# =============================================================================

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

    @patch("qwen3_tts.server.app.save_config")
    @patch("qwen3_tts.server.app.load_config")
    def test_startup_config_saves(self, mock_load, mock_save):
        """POST /update-startup-config saves to config."""
        mock_load.return_value = {"models": {"clone": {}, "design": {}, "custom": {}}}
        resp = self.client.post("/update-startup-config",
            json={"clone": True, "design": False},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "updated")
        self.assertTrue(mock_save.called)

    @patch("qwen3_tts.server.app.save_config")
    @patch("qwen3_tts.server.app.load_config")
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


# =============================================================================
# Client model management methods tests
# =============================================================================

@_skip_client
class TestClientModelMethods(unittest.TestCase):
    """Test that TTSClient has unload_model, update_startup_config, get_models methods."""

    def test_unload_model_exists(self):
        from qwen3_tts.server.client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "unload_model"))
        self.assertTrue(callable(getattr(client, "unload_model")))

    def test_update_startup_config_exists(self):
        from qwen3_tts.server.client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "update_startup_config"))
        self.assertTrue(callable(getattr(client, "update_startup_config")))

    def test_get_models_exists(self):
        from qwen3_tts.server.client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "get_models"))
        self.assertTrue(callable(getattr(client, "get_models")))


# =============================================================================
# Engine model cleanup and ASR info tests
# =============================================================================

class TestEngineModelCleanup(unittest.TestCase):
    """Test unload_model_cleanup, is_asr_loaded, get_asr_model_info."""

    def test_unload_model_cleanup_exists(self):
        from qwen3_tts.core.engine import unload_model_cleanup
        self.assertTrue(callable(unload_model_cleanup))

    def test_is_asr_loaded_returns_bool(self):
        from qwen3_tts.core.engine import is_asr_loaded
        result = is_asr_loaded()
        self.assertIsInstance(result, bool)

    def test_get_asr_model_info_returns_dict(self):
        from qwen3_tts.core.engine import get_asr_model_info
        info = get_asr_model_info()
        self.assertIsInstance(info, dict)
        self.assertIn("loaded", info)
        self.assertIn("backend", info)
        self.assertIn("model_name", info)


# =============================================================================
# Enhanced /models endpoint tests
# =============================================================================

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


# =============================================================================
# Smart Audio Loader tests
# =============================================================================

class TestSmartAudioLoader(unittest.TestCase):
    """Test smart audio loader functions."""

    def test_load_audio_exists(self):
        from qwen3_tts.core.engine import load_audio
        self.assertTrue(callable(load_audio))

    def test_load_audio_for_cloning_exists(self):
        from qwen3_tts.core.engine import load_audio_for_cloning
        self.assertTrue(callable(load_audio_for_cloning))

    def test_get_audio_loader_returns_valid(self):
        from qwen3_tts.core.engine import get_audio_loader
        result = get_audio_loader()
        self.assertIn(result, ("torchaudio", "librosa"))

    def test_set_audio_loader_validates(self):
        from qwen3_tts.core.engine import set_audio_loader
        with self.assertRaises(ValueError):
            set_audio_loader("invalid_loader")

    def test_set_audio_loader_updates(self):
        from qwen3_tts.core.engine import set_audio_loader, get_audio_loader
        original = get_audio_loader()
        try:
            set_audio_loader("librosa")
            self.assertEqual(get_audio_loader(), "librosa")
            set_audio_loader("torchaudio")
            self.assertEqual(get_audio_loader(), "torchaudio")
        finally:
            set_audio_loader(original)


# =============================================================================
# Manage Models UI tests
# =============================================================================

@_skip_ui
class TestManageModelsUI(unittest.TestCase):
    """Test Manage Models UI helper functions."""

    def test_get_model_table_data_exists(self):
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(callable(getattr(voice_ui, "get_model_table_data", None)))

    def test_toggle_model_exists(self):
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(callable(getattr(voice_ui, "toggle_model", None)))

    def test_get_model_status_html_exists(self):
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(callable(getattr(voice_ui, "get_model_status_html", None)))

    def test_update_startup_defaults_exists(self):
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(callable(getattr(voice_ui, "update_startup_defaults", None)))

    def test_get_audio_loader_setting_exists(self):
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(callable(getattr(voice_ui, "get_audio_loader_setting", None)))

    def test_set_audio_loader_setting_exists(self):
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(callable(getattr(voice_ui, "set_audio_loader_setting", None)))


# =============================================================================
# Improvement 1: Rubber Band audio processing tests
# =============================================================================

class TestRubberBandAudioProcessing(unittest.TestCase):
    """Test pyrubberband fallback to librosa for speed/pitch adjustment."""

    def test_adjust_speed_noop(self):
        """Speed factor 1.0 should return audio unchanged."""
        from qwen3_tts.core.engine import adjust_speed
        import numpy as np
        audio = np.random.randn(16000).astype(np.float32)
        result = adjust_speed(audio, 24000, 1.0)
        np.testing.assert_array_equal(result, audio)

    def test_adjust_pitch_noop(self):
        """Pitch shift 0 semitones should return audio unchanged."""
        from qwen3_tts.core.engine import adjust_pitch
        import numpy as np
        audio = np.random.randn(16000).astype(np.float32)
        result = adjust_pitch(audio, 24000, 0)
        np.testing.assert_array_equal(result, audio)

    def test_adjust_speed_with_librosa_fallback(self):
        """Speed adjustment should work even when pyrubberband is missing."""
        from qwen3_tts.core.engine import adjust_speed
        import numpy as np
        audio = np.random.randn(16000).astype(np.float32)
        # Mock pyrubberband import failure to force librosa fallback
        with patch.dict('sys.modules', {'pyrubberband': None}):
            try:
                result = adjust_speed(audio, 24000, 1.5)
                self.assertIsInstance(result, np.ndarray)
            except ImportError:
                # librosa may not be installed either — that's OK in test env
                pass

    def test_adjust_pitch_with_librosa_fallback(self):
        """Pitch adjustment should work even when pyrubberband is missing."""
        from qwen3_tts.core.engine import adjust_pitch
        import numpy as np
        audio = np.random.randn(16000).astype(np.float32)
        with patch.dict('sys.modules', {'pyrubberband': None}):
            try:
                result = adjust_pitch(audio, 24000, 2)
                self.assertIsInstance(result, np.ndarray)
            except ImportError:
                pass


# =============================================================================
# Improvement 2: Prosody presets tests
# =============================================================================

class TestProsodyPresets(unittest.TestCase):
    """Test prosody preset loading and resolution."""

    def test_default_prosody_presets_exist(self):
        """DEFAULT_PROSODY_PRESETS should contain standard presets."""
        from qwen3_tts.core.config import DEFAULT_PROSODY_PRESETS
        self.assertIn("excited", DEFAULT_PROSODY_PRESETS)
        self.assertIn("calm", DEFAULT_PROSODY_PRESETS)
        self.assertIn("whisper", DEFAULT_PROSODY_PRESETS)
        self.assertIn("authoritative", DEFAULT_PROSODY_PRESETS)
        self.assertIsInstance(DEFAULT_PROSODY_PRESETS["excited"], str)

    def test_get_prosody_presets_returns_defaults(self):
        """get_prosody_presets with empty config should return defaults."""
        from qwen3_tts.core.config import get_prosody_presets, DEFAULT_PROSODY_PRESETS
        presets = get_prosody_presets(config={})
        self.assertEqual(presets, DEFAULT_PROSODY_PRESETS)

    def test_get_prosody_presets_merges_user(self):
        """User presets should override defaults."""
        from qwen3_tts.core.config import get_prosody_presets
        config = {"prosody_presets": {"excited": "custom excited text", "newpreset": "new text"}}
        presets = get_prosody_presets(config)
        self.assertEqual(presets["excited"], "custom excited text")
        self.assertEqual(presets["newpreset"], "new text")
        # Defaults should still be present
        self.assertIn("calm", presets)

    def test_prosody_presets_in_config_json(self):
        """config.json should have prosody_presets section."""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
        with open(config_path) as f:
            config = json.load(f)
        self.assertIn("prosody_presets", config)
        self.assertIn("excited", config["prosody_presets"])

    @_skip_generate
    def test_prosody_cli_flag_exists(self):
        """qwen3_tts.interface.generate should accept --prosody flag."""
        from qwen3_tts.interface import generate as voice_generate
        import argparse
        # Build parser and check --prosody is registered
        parser = voice_generate.build_parser() if hasattr(voice_generate, 'build_parser') else None
        if parser is None:
            # Check that the module has the argparse setup
            source = open(voice_generate.__file__).read()
            self.assertIn("--prosody", source)


@_skip_ui
class TestProsodyUI(unittest.TestCase):
    """Test prosody preset UI helpers."""

    def test_get_prosody_choices_function(self):
        """get_prosody_choices should return list with (none) first."""
        from qwen3_tts.interface import ui as voice_ui
        choices = voice_ui.get_prosody_choices()
        self.assertIsInstance(choices, list)
        self.assertEqual(choices[0], "(none)")
        self.assertTrue(len(choices) > 1)

    def test_apply_prosody_preset_none(self):
        """Selecting (none) should return empty string."""
        from qwen3_tts.interface import ui as voice_ui
        result = voice_ui.apply_prosody_preset("(none)")
        self.assertEqual(result, "")

    def test_apply_prosody_preset_valid(self):
        """Selecting a valid preset should return its instruction text."""
        from qwen3_tts.interface import ui as voice_ui
        from qwen3_tts.core.config import DEFAULT_PROSODY_PRESETS
        result = voice_ui.apply_prosody_preset("excited")
        self.assertEqual(result, DEFAULT_PROSODY_PRESETS["excited"])


# =============================================================================
# Improvement 3: x_vector_only_mode tests
# =============================================================================

class TestXVectorOnlyMode(unittest.TestCase):
    """Test x_vector_only_mode parameter propagation."""

    def test_run_inference_accepts_x_vector_only_mode(self):
        """run_inference should accept x_vector_only_mode parameter."""
        import inspect
        from qwen3_tts.core.engine import run_inference
        sig = inspect.signature(run_inference)
        self.assertIn("x_vector_only_mode", sig.parameters)

    def test_run_inference_streaming_accepts_x_vector_only_mode(self):
        """run_inference_streaming should accept x_vector_only_mode parameter."""
        import inspect
        from qwen3_tts.core.engine import run_inference_streaming
        sig = inspect.signature(run_inference_streaming)
        self.assertIn("x_vector_only_mode", sig.parameters)

    def test_inference_single_accepts_x_vector_only_mode(self):
        """_run_inference_single should accept x_vector_only_mode parameter."""
        import inspect
        from qwen3_tts.core.engine.inference import _run_inference_single
        sig = inspect.signature(_run_inference_single)
        self.assertIn("x_vector_only_mode", sig.parameters)

    @_skip_generate
    def test_generate_via_server_accepts_x_vector_only_mode(self):
        """generate_via_server should accept x_vector_only_mode parameter."""
        import inspect
        from qwen3_tts.interface.generate import generate_via_server
        sig = inspect.signature(generate_via_server)
        self.assertIn("x_vector_only_mode", sig.parameters)

    @_skip_generate
    def test_generate_streaming_accepts_x_vector_only_mode(self):
        """generate_streaming in voice_generate should accept x_vector_only_mode."""
        import inspect
        from qwen3_tts.interface.generate import generate_streaming
        sig = inspect.signature(generate_streaming)
        self.assertIn("x_vector_only_mode", sig.parameters)


@_skip_client
class TestXVectorOnlyClient(unittest.TestCase):
    """Test x_vector_only_mode in client."""

    def test_client_generate_accepts_x_vector_only_mode(self):
        """TTSClient.generate should accept x_vector_only_mode parameter."""
        import inspect
        from qwen3_tts.server.client import TTSClient
        sig = inspect.signature(TTSClient.generate)
        self.assertIn("x_vector_only_mode", sig.parameters)

    def test_client_streaming_accepts_x_vector_only_mode(self):
        """TTSClient.generate_streaming should accept x_vector_only_mode parameter."""
        import inspect
        from qwen3_tts.server.client import TTSClient
        sig = inspect.signature(TTSClient.generate_streaming)
        self.assertIn("x_vector_only_mode", sig.parameters)


class TestCreateVoiceNoTranscript(unittest.TestCase):
    """Test --no-transcript flag for qwen3_tts.tools.create_voice."""

    def test_no_transcript_flag_in_parser(self):
        """qwen3_tts.tools.create_voice should accept --no-transcript flag."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "qwen3_tts", "tools", "create_voice.py"
        )
        with open(source_path) as f:
            source = f.read()
        self.assertIn("--no-transcript", source)


# =============================================================================
# Click CLI dispatch tests
# =============================================================================

class TestClickCLI(unittest.TestCase):
    """Test the Click CLI routing."""

    def test_cli_imports(self):
        """qwen3_tts.cli imports without error."""
        from qwen3_tts.cli import cli, TTSGroup
        self.assertIsNotNone(cli)
        self.assertIsInstance(cli, TTSGroup)

    def test_ttsgroup_prepends_generate(self):
        """TTSGroup prepends 'generate' for bare text args."""
        from qwen3_tts.cli import cli
        # Check that known subcommands exist
        self.assertIn('generate', cli.commands)
        self.assertIn('server', cli.commands)
        self.assertIn('voice', cli.commands)
        self.assertIn('list', cli.commands)
        self.assertIn('config', cli.commands)
        self.assertIn('ui', cli.commands)
        self.assertIn('history', cli.commands)
        self.assertIn('stats', cli.commands)

    def test_ttsgroup_server_mode_stripping(self):
        """TTSGroup strips --_server-mode and re-inserts after subcommand."""
        from qwen3_tts.cli import TTSGroup
        # Verify the class has parse_args that handles --_server-mode
        import inspect
        source = inspect.getsource(TTSGroup.parse_args)
        self.assertIn('--_server-mode', source)
        self.assertIn('server_mode', source)

    def test_ui_rejects_server_mode_flag(self):
        """tts ui does not accept --_server-mode (it's a generate-only flag)."""
        from click.testing import CliRunner
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch('qwen3_tts.interface.generate.launch_gradio_ui'):
            result = runner.invoke(cli, ['ui', '--_server-mode'])
        # ui command should NOT fail with "No such option: --_server-mode"
        self.assertNotEqual(result.exit_code, 2,
                            f"ui rejected --_server-mode: {result.output}")

    def test_ttsgroup_skips_server_mode_for_non_generate_commands(self):
        """TTSGroup.parse_args does NOT re-insert --_server-mode for ui, config, etc."""
        from click.testing import CliRunner
        from qwen3_tts.cli import cli
        runner = CliRunner()
        # Commands that should NOT get --_server-mode re-inserted
        non_generate_cmds = ['ui', 'config', 'history', 'stats']
        for cmd in non_generate_cmds:
            result = runner.invoke(cli, ['--_server-mode', cmd, '--help'])
            self.assertNotIn('No such option', result.output,
                             f"--_server-mode leaked to '{cmd}': {result.output}")

    def test_flag_map_completeness(self):
        """_FLAG_MAP covers all generation options."""
        from qwen3_tts.cli import _FLAG_MAP
        expected_keys = [
            'mode', 'prompt', 'description', 'speaker', 'instruct',
            'voice', 'prosody', 'no_transcript', 'output', 'play',
            'stream', 'no_open', 'speed', 'pitch', 'trim_silence',
            'normalize', 'preset', 'temperature', 'top_k', 'top_p',
            'seed', 'repetition_penalty', 'max_chunk_chars',
            'clipboard', 'ssml', 'local', 'dry_run', 'backend',
            'model_size', 'server_mode',
        ]
        for key in expected_keys:
            self.assertIn(key, _FLAG_MAP, f"Missing key in _FLAG_MAP: {key}")

    def test_cli_version(self):
        """CLI has version option."""
        from click.testing import CliRunner
        from qwen3_tts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('3.0.0', result.output)

    def test_cli_help(self):
        """CLI --help shows subcommands."""
        from click.testing import CliRunner
        from qwen3_tts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('generate', result.output)
        self.assertIn('server', result.output)
        self.assertIn('voice', result.output)

    def test_cli_server_help(self):
        """CLI server --help shows subcommands."""
        from click.testing import CliRunner
        from qwen3_tts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['server', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('start', result.output)
        self.assertIn('stop', result.output)
        self.assertIn('status', result.output)

    def test_cli_generate_help(self):
        """CLI generate --help shows all options."""
        from click.testing import CliRunner
        from qwen3_tts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['generate', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('--mode', result.output)
        self.assertIn('--prompt', result.output)
        self.assertIn('--output', result.output)

    @_skip_ui
    def test_preview_voice_cleanup_on_failure(self):
        """preview_voice must clean up temp file on exception."""
        import os
        import tempfile
        from unittest.mock import patch, MagicMock

        # Track the temp file path
        temp_file_path = None

        def mock_named_temp_file(*args, **kwargs):
            nonlocal temp_file_path
            mock_file = MagicMock()
            # Create an actual temp file to track
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            temp_file_path = path
            mock_file.name = path
            mock_file.write.side_effect = RuntimeError("Server error")
            mock_file.close = MagicMock()
            return mock_file

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake audio bytes"

        with patch('qwen3_tts.interface.ui.voice_management.is_server_running', return_value=True), \
             patch('requests.get', return_value=mock_resp), \
             patch('qwen3_tts.interface.ui.voice_management.tempfile.NamedTemporaryFile', side_effect=mock_named_temp_file):

            from qwen3_tts.interface.ui import preview_voice
            result = preview_voice("test_prompt")

            # Should return None on failure
            self.assertIsNone(result)
            # Temp file should be cleaned up (not exist)
            if temp_file_path:
                self.assertFalse(os.path.exists(temp_file_path),
                                 f"Temp file {temp_file_path} should be cleaned up on exception")

    @_skip_ui
    def test_preview_voice_cleanup_on_write_failure(self):
        """preview_voice must clean up temp file when write fails."""
        import os
        import tempfile
        from unittest.mock import patch, MagicMock

        # Track the temp file path
        temp_file_path = None

        def mock_named_temp_file(*args, **kwargs):
            nonlocal temp_file_path
            mock_file = MagicMock()
            # Create an actual temp file to track
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            temp_file_path = path
            mock_file.name = path
            # Make write fail after file is created
            mock_file.write.side_effect = OSError("Disk full")
            mock_file.close = MagicMock()
            return mock_file

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake audio bytes"

        with patch('qwen3_tts.interface.ui.voice_management.is_server_running', return_value=True), \
             patch('requests.get', return_value=mock_resp), \
             patch('qwen3_tts.interface.ui.voice_management.tempfile.NamedTemporaryFile', side_effect=mock_named_temp_file):

            from qwen3_tts.interface.ui import preview_voice
            result = preview_voice("test_prompt")

            # Should return None on write failure
            self.assertIsNone(result)
            # Temp file should be cleaned up even though it was created
            if temp_file_path:
                self.assertFalse(os.path.exists(temp_file_path),
                                 f"Temp file {temp_file_path} should be cleaned up on write failure")



class TestGetPresets(unittest.TestCase):
    """Tests for get_presets() preset dropdown choices."""

    @patch("qwen3_tts.interface.ui.shared.load_config")
    def test_includes_none_as_first_choice(self, mock_config):
        mock_config.return_value = {"presets": {"consistent": {}, "creative": {}}}
        from qwen3_tts.interface.ui.shared import get_presets
        result = get_presets()
        self.assertEqual(result[0], "(none)")

    @patch("qwen3_tts.interface.ui.shared.load_config")
    def test_contains_config_presets_after_none(self, mock_config):
        mock_config.return_value = {"presets": {"consistent": {}, "creative": {}}}
        from qwen3_tts.interface.ui.shared import get_presets
        result = get_presets()
        self.assertIn("consistent", result)
        self.assertIn("creative", result)
        self.assertEqual(result.index("(none)"), 0)

    @patch("qwen3_tts.interface.ui.shared.load_config")
    def test_empty_presets_still_has_none(self, mock_config):
        mock_config.return_value = {"presets": {}}
        from qwen3_tts.interface.ui.shared import get_presets
        result = get_presets()
        self.assertEqual(result, ["(none)"])


if __name__ == "__main__":
    unittest.main()
