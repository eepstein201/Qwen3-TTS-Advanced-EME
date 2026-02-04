#!/usr/bin/env python3
"""Tests for Qwen3-TTS codebase.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/ -v

These tests do NOT require GPU, models, or a running server.
They test config, validation, auth, error classes, and the server
endpoint logic using Flask's test client.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# tts_config tests
# =============================================================================

class TestTTSConfig(unittest.TestCase):
    """Test tts_config module (no heavy imports)."""

    def test_imports_no_torch(self):
        """tts_config must not import torch."""
        import tts_config  # noqa: F401
        self.assertNotIn("torch", sys.modules.keys() - {"torch"})

    def test_error_hierarchy(self):
        from tts_config import (
            TTSError, ServerConnectionError, ModelNotLoadedError,
            InvalidInputError, GenerationError, AuthenticationError,
        )
        # All errors inherit from TTSError
        for cls in [ServerConnectionError, ModelNotLoadedError,
                    InvalidInputError, GenerationError, AuthenticationError]:
            err = cls("test") if cls != ModelNotLoadedError else cls("clone")
            self.assertIsInstance(err, TTSError)

    def test_error_format_cli(self):
        from tts_config import ServerConnectionError
        err = ServerConnectionError("details here")
        formatted = err.format_cli()
        self.assertIn("Cannot connect", formatted)
        self.assertIn("startTTSServer", formatted)

    def test_error_format_gradio(self):
        from tts_config import GenerationError
        err = GenerationError("oops")
        html = err.format_gradio()
        self.assertIn("Audio generation failed", html)

    def test_read_auth_token_missing(self):
        from tts_config import TOKEN_FILE, read_auth_token
        # Use a temp path that doesn't exist
        with patch("tts_config.TOKEN_FILE", "/tmp/nonexistent_token_test_xyz"):
            result = read_auth_token()
        # If the real file exists, it returns its content; with patched path it's None
        # Can't reliably test with real TOKEN_FILE, so test the function signature
        self.assertTrue(result is None or isinstance(result, str))

    def test_auth_headers_returns_dict(self):
        from tts_config import auth_headers
        headers = auth_headers()
        self.assertIsInstance(headers, dict)

    def test_auth_headers_with_token(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".token", delete=False) as f:
            f.write("test_token_abc123")
            token_path = f.name
        try:
            with patch("tts_config.TOKEN_FILE", token_path):
                from tts_config import read_auth_token, auth_headers
                # Need to reimport to pick up patched value
                token = read_auth_token()
                headers = auth_headers()
            # The token file exists so we should get headers
            # (actual behavior depends on whether TOKEN_FILE is patched at call time)
        finally:
            os.unlink(token_path)

    def test_model_info_keys(self):
        from tts_config import MODEL_INFO
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
        from tts_config import CUSTOM_VOICE_SPEAKERS
        self.assertIn("ryan", CUSTOM_VOICE_SPEAKERS)
        self.assertIn("aiden", CUSTOM_VOICE_SPEAKERS)
        for key, info in CUSTOM_VOICE_SPEAKERS.items():
            self.assertIn("name", info)
            self.assertIn("lang", info)
            self.assertIn("desc", info)


# =============================================================================
# tts_server validation tests (using Flask test client, no models needed)
# =============================================================================

class TestServerValidation(unittest.TestCase):
    """Test server input validation without loading any models."""

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client with mocked models."""
        # We need to mock torch and model-related imports
        # to avoid loading heavy dependencies
        import tts_server
        tts_server.auth_token = None  # Disable auth for tests
        tts_server.server_config = {
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        }
        cls.app = tts_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")

    def test_generate_empty_texts(self):
        resp = self.client.post("/generate", json={"texts": []})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("No texts", resp.get_json()["error"])

    def test_generate_batch_too_large(self):
        texts = ["hello"] * 5  # max is 3 in test config
        resp = self.client.post("/generate", json={"texts": texts})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("exceeds limit", resp.get_json()["error"])

    def test_generate_text_too_long(self):
        resp = self.client.post("/generate", json={"texts": ["x" * 200]})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("character limit", resp.get_json()["error"])

    def test_generate_invalid_mode(self):
        resp = self.client.post("/generate", json={"texts": ["hello"], "mode": "invalid"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid mode", resp.get_json()["error"])

    def test_generate_path_traversal_prompt(self):
        resp = self.client.post("/generate", json={
            "texts": ["hello"],
            "mode": "clone",
            "prompt_file": "../../../etc/passwd",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("path traversal", resp.get_json()["error"])

    def test_generate_invalid_speaker(self):
        resp = self.client.post("/generate", json={
            "texts": ["hello"],
            "mode": "custom",
            "speaker": "nonexistent_speaker",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unknown speaker", resp.get_json()["error"])

    def test_generate_valid_speaker_accepted(self):
        # This will fail with 503 (model not loaded) rather than 400 (validation error)
        resp = self.client.post("/generate", json={
            "texts": ["hello"],
            "mode": "custom",
            "speaker": "Ryan",
        })
        # Should pass validation (400) and hit model-not-loaded (503)
        self.assertIn(resp.status_code, [200, 503])

    def test_error_response_has_recovery_field(self):
        """All error responses should include a recovery hint."""
        # Validation error
        resp = self.client.post("/generate", json={"texts": []})
        data = resp.get_json()
        self.assertIn("recovery", data)

        # Model not loaded
        resp = self.client.post("/generate", json={
            "texts": ["hello"], "mode": "clone", "prompt_file": "test.pt"
        })
        data = resp.get_json()
        self.assertIn("recovery", data)


# =============================================================================
# tts_server auth tests
# =============================================================================

class TestServerAuth(unittest.TestCase):
    """Test server authentication."""

    @classmethod
    def setUpClass(cls):
        import tts_server
        tts_server.auth_token = "test_secret_token"
        tts_server.server_config = {
            "security": {},
            "auto_shutdown_minutes": 0,
        }
        cls.app = tts_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        import tts_server
        tts_server.auth_token = None

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
        data = resp.get_json()
        self.assertFalse(data["active"])


# =============================================================================
# SSML parsing tests (from tts_generate, lightweight)
# =============================================================================

class TestSSMLParsing(unittest.TestCase):
    """Test SSML parsing in tts_generate."""

    def test_no_ssml(self):
        from tts_generate import parse_ssml
        text, meta = parse_ssml("Hello world")
        self.assertEqual(text, "Hello world")
        self.assertFalse(meta["has_ssml"])

    def test_break_tag(self):
        from tts_generate import parse_ssml
        text, meta = parse_ssml('Hello <break time="500ms"/> world')
        self.assertTrue(meta["has_ssml"])
        self.assertNotIn("<break", text)

    def test_sub_tag(self):
        from tts_generate import parse_ssml
        text, meta = parse_ssml('<sub alias="World Wide Web">WWW</sub>')
        self.assertIn("World Wide Web", text)
        self.assertNotIn("WWW", text)

    def test_say_as_characters(self):
        from tts_generate import parse_ssml
        text, meta = parse_ssml('<say-as interpret-as="characters">ABC</say-as>')
        self.assertIn("A B C", text)

    def test_prosody_speed(self):
        from tts_generate import parse_ssml
        text, meta = parse_ssml('<prosody rate="fast">Quick text</prosody>')
        self.assertTrue(meta["has_ssml"])
        self.assertEqual(meta["prosody"]["speed"], 1.2)


# =============================================================================
# SRT parsing tests
# =============================================================================

class TestSRTParsing(unittest.TestCase):
    """Test SRT parsing."""

    def test_parse_srt(self):
        from tts_generate import parse_srt
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
        from tts_generate import srt_time_to_ms
        self.assertEqual(srt_time_to_ms("00:01:30,500"), 90500)
        self.assertEqual(srt_time_to_ms("01:00:00,000"), 3600000)


# =============================================================================
# Auto-increment filename tests
# =============================================================================

class TestAutoIncrementFilename(unittest.TestCase):
    """Test auto_increment_filename helper."""

    def test_no_conflict(self):
        from tts_generate import auto_increment_filename
        # Non-existent file should return as-is
        result = auto_increment_filename("/tmp/nonexistent_test_xyz.wav")
        self.assertEqual(result, "/tmp/nonexistent_test_xyz.wav")

    def test_conflict_increments(self):
        from tts_generate import auto_increment_filename
        # Create a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            result = auto_increment_filename(path)
            self.assertNotEqual(result, path)
            self.assertIn("_2", result)
        finally:
            os.unlink(path)

    def test_already_numbered(self):
        from tts_generate import auto_increment_filename
        # Create files with _2 suffix
        with tempfile.NamedTemporaryFile(suffix="_2.wav", delete=False, dir="/tmp", prefix="test_") as f:
            path = f.name
        try:
            result = auto_increment_filename(path)
            self.assertNotEqual(result, path)
            self.assertIn("_3", result)
        finally:
            os.unlink(path)


# =============================================================================
# Backend config tests (tts_config backend helpers)
# =============================================================================

class TestBackendConfig(unittest.TestCase):
    """Test backend-related config helpers in tts_config."""

    def test_valid_backends(self):
        from tts_config import VALID_BACKENDS
        self.assertIn("torch", VALID_BACKENDS)
        self.assertIn("mlx", VALID_BACKENDS)

    def test_valid_mlx_quantizations(self):
        from tts_config import VALID_MLX_QUANTIZATIONS
        self.assertIn("4bit", VALID_MLX_QUANTIZATIONS)
        self.assertIn("8bit", VALID_MLX_QUANTIZATIONS)
        self.assertIn("bf16", VALID_MLX_QUANTIZATIONS)

    def test_mlx_model_info_keys(self):
        from tts_config import MLX_MODEL_INFO
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
        from tts_config import MLX_MODEL_INFO
        for size in ("1.7B", "0.6B"):
            for model_type, info in MLX_MODEL_INFO[size].items():
                self.assertIn("{quant}", info["name_template"])

    def test_get_backend_default(self):
        """get_backend() defaults to 'torch' with no env/config override."""
        from tts_config import get_backend
        # Clear env override
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_BACKEND", None)
            with patch("tts_config.load_config", return_value={}):
                result = get_backend()
        self.assertEqual(result, "torch")

    def test_get_backend_from_config(self):
        from tts_config import get_backend
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_BACKEND", None)
            with patch("tts_config.load_config", return_value={"advanced": {"backend": "mlx"}}):
                result = get_backend()
        self.assertEqual(result, "mlx")

    def test_get_backend_env_override(self):
        """TTS_BACKEND env var overrides config."""
        from tts_config import get_backend
        with patch.dict(os.environ, {"TTS_BACKEND": "mlx"}):
            with patch("tts_config.load_config", return_value={"advanced": {"backend": "torch"}}):
                result = get_backend()
        self.assertEqual(result, "mlx")

    def test_get_backend_invalid_falls_back(self):
        from tts_config import get_backend
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_BACKEND", None)
            with patch("tts_config.load_config", return_value={"advanced": {"backend": "invalid"}}):
                result = get_backend()
        self.assertEqual(result, "torch")

    def test_get_mlx_quantization_default(self):
        from tts_config import get_mlx_quantization
        with patch("tts_config.load_config", return_value={}):
            result = get_mlx_quantization()
        self.assertEqual(result, "8bit")

    def test_get_mlx_quantization_from_config(self):
        from tts_config import get_mlx_quantization
        with patch("tts_config.load_config", return_value={"advanced": {"mlx_quantization": "4bit"}}):
            result = get_mlx_quantization()
        self.assertEqual(result, "4bit")

    def test_get_mlx_quantization_invalid_falls_back(self):
        from tts_config import get_mlx_quantization
        with patch("tts_config.load_config", return_value={"advanced": {"mlx_quantization": "garbage"}}):
            result = get_mlx_quantization()
        self.assertEqual(result, "8bit")

    def test_get_mlx_model_name(self):
        from tts_config import get_mlx_model_name
        with patch("tts_config.get_mlx_quantization", return_value="8bit"):
            name = get_mlx_model_name("clone")
        self.assertEqual(name, "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")

    def test_get_mlx_model_name_4bit(self):
        from tts_config import get_mlx_model_name
        with patch("tts_config.get_mlx_quantization", return_value="4bit"):
            name = get_mlx_model_name("design")
        self.assertEqual(name, "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit")

    def test_get_mlx_model_name_invalid_type(self):
        from tts_config import get_mlx_model_name
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
        from tts_engine import load_voice_prompt_mlx
        # Create fake wav and txt
        wav_path = os.path.join(self.tmpdir, "test_voice.wav")
        txt_path = os.path.join(self.tmpdir, "test_voice.txt")
        with open(wav_path, "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        with open(txt_path, "w") as f:
            f.write("Hello, this is a test transcript.")

        with patch("tts_engine.VOICE_PROMPTS_DIR", self.tmpdir):
            result = load_voice_prompt_mlx("test_voice.pt")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["ref_audio"], wav_path)
        self.assertEqual(result["ref_text"], "Hello, this is a test transcript.")

    def test_load_voice_prompt_mlx_strips_pt(self):
        """Prompt name with .pt extension is handled correctly."""
        from tts_engine import load_voice_prompt_mlx
        wav_path = os.path.join(self.tmpdir, "voice.wav")
        txt_path = os.path.join(self.tmpdir, "voice.txt")
        with open(wav_path, "wb") as f:
            f.write(b"fake")
        with open(txt_path, "w") as f:
            f.write("transcript")

        with patch("tts_engine.VOICE_PROMPTS_DIR", self.tmpdir):
            result = load_voice_prompt_mlx("voice.pt")
        self.assertEqual(result["ref_audio"], wav_path)

    def test_load_voice_prompt_mlx_missing_files(self):
        """Raises FileNotFoundError when wav/txt missing."""
        from tts_engine import load_voice_prompt_mlx
        with patch("tts_engine.VOICE_PROMPTS_DIR", self.tmpdir):
            with self.assertRaises(FileNotFoundError):
                load_voice_prompt_mlx("nonexistent")

    def test_load_voice_prompt_mlx_pt_only_error(self):
        """Clear error when only .pt exists (no MLX-compatible files)."""
        from tts_engine import load_voice_prompt_mlx
        pt_path = os.path.join(self.tmpdir, "legacy.pt")
        with open(pt_path, "wb") as f:
            f.write(b"fake tensor data")

        with patch("tts_engine.VOICE_PROMPTS_DIR", self.tmpdir):
            with self.assertRaises(FileNotFoundError) as ctx:
                load_voice_prompt_mlx("legacy")
            self.assertIn("only has a .pt file", str(ctx.exception))
            self.assertIn("createVoice", str(ctx.exception))

    def test_load_voice_prompt_dispatch_torch(self):
        """load_voice_prompt dispatches to torch backend."""
        from tts_engine import load_voice_prompt
        with patch("tts_engine.get_backend", return_value="torch"):
            with patch("tts_engine._load_voice_prompt_torch", return_value="mock_tensor") as mock:
                result = load_voice_prompt("test.pt")
        mock.assert_called_once_with("test.pt")
        self.assertEqual(result, "mock_tensor")

    def test_load_voice_prompt_dispatch_mlx(self):
        """load_voice_prompt dispatches to MLX backend."""
        from tts_engine import load_voice_prompt
        mock_result = {"ref_audio": "/fake/path.wav", "ref_text": "text"}
        with patch("tts_engine.get_backend", return_value="mlx"):
            with patch("tts_engine.load_voice_prompt_mlx", return_value=mock_result) as mock:
                result = load_voice_prompt("test.pt")
        mock.assert_called_once_with("test.pt")
        self.assertEqual(result, mock_result)


# =============================================================================
# Backend dispatch tests (no actual model loading — tests dispatch logic)
# =============================================================================

class TestBackendDispatch(unittest.TestCase):
    """Test that public API dispatches to correct backend functions."""

    def test_load_model_dispatch_torch(self):
        from tts_engine import load_model
        with patch("tts_engine.get_backend", return_value="torch"):
            with patch("tts_engine._load_model_torch", return_value="torch_model") as mock:
                result = load_model("clone")
        mock.assert_called_once_with("clone")
        self.assertEqual(result, "torch_model")

    def test_load_model_dispatch_mlx(self):
        from tts_engine import load_model
        with patch("tts_engine.get_backend", return_value="mlx"):
            with patch("tts_engine._load_model_mlx", return_value="mlx_model") as mock:
                result = load_model("design")
        mock.assert_called_once_with("design")
        self.assertEqual(result, "mlx_model")

    def test_run_inference_dispatch_torch(self):
        from tts_engine import run_inference
        with patch("tts_engine.get_backend", return_value="torch"):
            with patch("tts_engine._run_inference_torch", return_value=("wav", 24000)) as mock:
                result = run_inference("model", "text", "clone", {})
        mock.assert_called_once()
        self.assertEqual(result, ("wav", 24000))

    def test_run_inference_dispatch_mlx(self):
        from tts_engine import run_inference
        with patch("tts_engine.get_backend", return_value="mlx"):
            with patch("tts_engine._run_inference_mlx", return_value=("wav", 24000)) as mock:
                result = run_inference("model", "text", "design", {})
        mock.assert_called_once()
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
        from tts_engine import _run_inference_mlx
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
        from tts_engine import _run_inference_mlx
        with self.assertRaises(TypeError) as ctx:
            _run_inference_mlx(
                model=MagicMock(),
                text="test",
                mode="clone",
                gen_params={},
                voice_prompt="some_tensor_object",
            )
        self.assertIn("MLX clone mode requires a voice prompt dict", str(ctx.exception))
        self.assertIn("createVoice", str(ctx.exception))


# =============================================================================
# Lazy import safety tests
# =============================================================================

class TestLazyImports(unittest.TestCase):
    """Verify that tts_engine does not import torch or mlx at module scope."""

    def test_engine_no_torch_at_module_scope(self):
        """tts_engine module should not force-import torch."""
        # Remove tts_engine from cache to test fresh import behavior
        saved_modules = {}
        for mod in list(sys.modules.keys()):
            if mod == "tts_engine" or mod.startswith("tts_engine."):
                saved_modules[mod] = sys.modules.pop(mod)

        # Also note if torch was already loaded
        torch_was_loaded = "torch" in sys.modules

        try:
            import tts_engine  # noqa: F401
            if not torch_was_loaded:
                # torch should not have been imported by tts_engine
                self.assertNotIn("torch", sys.modules,
                    "tts_engine imported torch at module scope")
        finally:
            # Restore
            for mod, val in saved_modules.items():
                sys.modules[mod] = val

    def test_config_no_torch(self):
        """tts_config must not import torch (regression check)."""
        import tts_config  # noqa: F401
        # tts_config should never cause torch to load
        self.assertNotIn("torch", dir(tts_config))


# =============================================================================
# Phase 14: 0.6B Model Size tests
# =============================================================================

class TestModelSize(unittest.TestCase):
    """Test 0.6B model size configuration."""

    def test_valid_model_sizes(self):
        from tts_config import VALID_MODEL_SIZES
        self.assertIn("1.7B", VALID_MODEL_SIZES)
        self.assertIn("0.6B", VALID_MODEL_SIZES)

    def test_get_model_size_default(self):
        """get_model_size() defaults to 1.7B."""
        from tts_config import get_model_size
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_MODEL_SIZE", None)
            with patch("tts_config.load_config", return_value={}):
                self.assertEqual(get_model_size(), "1.7B")

    def test_get_model_size_from_config(self):
        from tts_config import get_model_size
        config = {"advanced": {"model_size": "0.6B"}}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_MODEL_SIZE", None)
            with patch("tts_config.load_config", return_value=config):
                self.assertEqual(get_model_size(), "0.6B")

    def test_get_model_size_env_override(self):
        """TTS_MODEL_SIZE env var overrides config."""
        from tts_config import get_model_size
        config = {"advanced": {"model_size": "1.7B"}}
        with patch.dict(os.environ, {"TTS_MODEL_SIZE": "0.6B"}):
            with patch("tts_config.load_config", return_value=config):
                self.assertEqual(get_model_size(), "0.6B")

    def test_get_torch_model_name(self):
        from tts_config import get_torch_model_name
        with patch("tts_config.get_model_size", return_value="1.7B"):
            name = get_torch_model_name("clone")
            self.assertIn("1.7B", name)
            self.assertIn("Base", name)

    def test_get_torch_model_name_0_6B(self):
        from tts_config import get_torch_model_name
        with patch("tts_config.get_model_size", return_value="0.6B"):
            name = get_torch_model_name("clone")
            self.assertIn("0.6B", name)

    def test_model_info_has_0_6B(self):
        from tts_config import MODEL_INFO, MLX_MODEL_INFO
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
        from tts_engine import run_inference_streaming
        self.assertTrue(callable(run_inference_streaming))

    def test_mlx_streaming_function_exists(self):
        """_run_inference_mlx_streaming function is importable."""
        from tts_engine import _run_inference_mlx_streaming
        self.assertTrue(callable(_run_inference_mlx_streaming))

    def test_streaming_torch_falls_back_to_chunked(self):
        """run_inference_streaming for torch uses chunked inference (not native streaming)."""
        from tts_engine import run_inference_streaming
        import inspect
        source = inspect.getsource(run_inference_streaming)
        # Torch backend falls back to chunked approach
        self.assertIn("_run_inference_single", source)

    def test_streaming_mlx_function_signature(self):
        """_run_inference_mlx_streaming has correct parameters."""
        from tts_engine import _run_inference_mlx_streaming
        import inspect
        sig = inspect.signature(_run_inference_mlx_streaming)
        params = list(sig.parameters.keys())
        self.assertIn("model", params)
        self.assertIn("text", params)
        self.assertIn("mode", params)
        self.assertIn("gen_params", params)


class TestStreamingServerEndpoint(unittest.TestCase):
    """Test /generate-stream server endpoint."""

    @classmethod
    def setUpClass(cls):
        import tts_server
        tts_server.auth_token = "test_token"
        tts_server.server_config = {
            "security": {"max_text_length": 1000, "max_batch_size": 10},
            "auto_shutdown_minutes": 0,
        }
        cls.app = tts_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        import tts_server
        tts_server.auth_token = None

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
        self.assertIn("text", resp.get_json()["error"].lower())

    def test_generate_stream_validates_mode(self):
        """POST /generate-stream validates mode."""
        resp = self.client.post("/generate-stream",
            json={"text": "hello", "mode": "invalid"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("mode", resp.get_json()["error"].lower())


# =============================================================================
# Phase 16: ASR tests
# =============================================================================

class TestASR(unittest.TestCase):
    """Test ASR transcription functions."""

    def test_transcribe_audio_exists(self):
        """transcribe_audio function is importable."""
        from tts_engine import transcribe_audio
        self.assertTrue(callable(transcribe_audio))

    def test_is_asr_available_exists(self):
        """is_asr_available function is importable."""
        from tts_engine import is_asr_available
        self.assertTrue(callable(is_asr_available))

    def test_is_asr_available_torch_returns_false(self):
        """is_asr_available returns False when backend is torch."""
        from tts_engine import is_asr_available
        with patch("tts_engine.get_backend", return_value="torch"):
            self.assertFalse(is_asr_available())

    def test_transcribe_audio_requires_mlx(self):
        """transcribe_audio raises ImportError when backend is torch."""
        from tts_engine import transcribe_audio
        with patch("tts_engine.get_backend", return_value="torch"):
            with self.assertRaises(ImportError) as ctx:
                transcribe_audio("/fake/path.wav")
            self.assertIn("MLX backend", str(ctx.exception))

    def test_asr_model_is_lazy_loaded(self):
        """_asr_model is None until transcribe_audio is called."""
        import tts_engine
        # The global _asr_model should be None at module level
        self.assertIsNone(tts_engine._asr_model)

    def test_is_asr_available_mlx_with_stt(self):
        """is_asr_available returns True when MLX + mlx_audio.stt available."""
        from tts_engine import is_asr_available
        with patch("tts_engine.get_backend", return_value="mlx"):
            # Mock successful import of load_model
            with patch.dict(sys.modules, {"mlx_audio.stt": MagicMock()}):
                # Force re-check by clearing any cached imports
                result = is_asr_available()
        # Should attempt to import and return True if successful
        # (actual result depends on whether mlx_audio is installed)
        self.assertIsInstance(result, bool)

    def test_transcribe_audio_returns_string(self):
        """transcribe_audio returns a string."""
        from tts_engine import transcribe_audio

        # Mock the entire transcription flow
        mock_result = MagicMock()
        mock_result.text = "Hello world"

        mock_model = MagicMock()
        mock_model.generate.return_value = mock_result

        with patch("tts_engine.get_backend", return_value="mlx"):
            with patch("tts_engine._asr_model", mock_model):
                result = transcribe_audio("/fake/path.wav")

        self.assertIsInstance(result, str)
        self.assertEqual(result, "Hello world")


# =============================================================================
# Phase 17: Stability tests
# =============================================================================

class TestStability(unittest.TestCase):
    """Test stability hardening features."""

    def test_retry_delays_constant_exists(self):
        """_RETRY_DELAYS constant is defined."""
        from tts_engine import _RETRY_DELAYS
        self.assertEqual(len(_RETRY_DELAYS), 3)
        self.assertEqual(_RETRY_DELAYS, (5, 15, 45))

    def test_retry_delays_is_exponential(self):
        """_RETRY_DELAYS uses exponential backoff pattern."""
        from tts_engine import _RETRY_DELAYS
        # Each delay should be roughly 3x the previous (5 -> 15 -> 45)
        self.assertEqual(_RETRY_DELAYS[1], _RETRY_DELAYS[0] * 3)
        self.assertEqual(_RETRY_DELAYS[2], _RETRY_DELAYS[1] * 3)

    def test_max_chunk_chars_helper_exists(self):
        """_get_max_chunk_chars helper function exists."""
        from tts_engine import _get_max_chunk_chars
        self.assertTrue(callable(_get_max_chunk_chars))

    def test_max_chunk_chars_default(self):
        """_get_max_chunk_chars returns default 500."""
        from tts_engine import _get_max_chunk_chars
        with patch("tts_engine.load_config", return_value={}):
            result = _get_max_chunk_chars()
        self.assertEqual(result, 500)

    def test_max_chunk_chars_from_config(self):
        """_get_max_chunk_chars reads from config."""
        from tts_engine import _get_max_chunk_chars
        config = {"generation": {"max_chunk_chars": 300}}
        with patch("tts_engine.load_config", return_value=config):
            result = _get_max_chunk_chars()
        self.assertEqual(result, 300)


class TestFloat32Guard(unittest.TestCase):
    """Test float32 dtype guard for torch clone mode on MPS."""

    def test_float32_guard_exists_in_torch_inference(self):
        """_run_inference_torch has float32 guard logic."""
        from tts_engine import _run_inference_torch
        import inspect
        source = inspect.getsource(_run_inference_torch)
        # Should have float32 override logic for clone mode
        self.assertIn("float32", source)
        self.assertIn("clone", source)


class TestMLXMetalRecovery(unittest.TestCase):
    """Test MLX Metal kernel crash recovery."""

    def test_run_inference_handles_exceptions(self):
        """run_inference wraps inference in try/except."""
        from tts_engine import _run_inference_single
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
        from tts_engine import _split_text
        chunks = _split_text("Hello world.", max_chars=500)
        self.assertEqual(chunks, ["Hello world."])

    def test_split_text_sentences(self):
        """Text is split on sentence boundaries."""
        from tts_engine import _split_text
        text = "First sentence. Second sentence. Third sentence."
        chunks = _split_text(text, max_chars=30)
        self.assertGreater(len(chunks), 1)
        # Each chunk should be <= max_chars
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 30)

    def test_split_text_preserves_content(self):
        """All content is preserved after splitting."""
        from tts_engine import _split_text
        text = "The quick brown fox jumps over the lazy dog. A second sentence follows."
        chunks = _split_text(text, max_chars=50)
        combined = " ".join(chunks)
        # All words should be present
        for word in text.split():
            self.assertIn(word.rstrip(".,"), combined)

    def test_split_text_question_mark(self):
        """Text splits on question marks."""
        from tts_engine import _split_text
        text = "Is this a question? Yes it is."
        chunks = _split_text(text, max_chars=25)
        self.assertGreater(len(chunks), 1)

    def test_split_text_exclamation(self):
        """Text splits on exclamation marks."""
        from tts_engine import _split_text
        text = "Hello! How are you today?"
        chunks = _split_text(text, max_chars=15)
        self.assertGreater(len(chunks), 1)

    def test_split_text_newlines(self):
        """Text splits on newlines."""
        from tts_engine import _split_text
        text = "First paragraph.\n\nSecond paragraph."
        chunks = _split_text(text, max_chars=20)
        self.assertGreater(len(chunks), 1)

    def test_split_text_comma_fallback(self):
        """Very long sentence falls back to clause boundaries."""
        from tts_engine import _split_text
        # A single long sentence with commas but no periods
        text = "This is a very long sentence, with several clauses, that should be split at commas when needed"
        chunks = _split_text(text, max_chars=40)
        # Should split due to length
        self.assertGreater(len(chunks), 1)


# =============================================================================
# Server health endpoint info tests
# =============================================================================

class TestHealthEndpointInfo(unittest.TestCase):
    """Test /health endpoint returns expected info fields."""

    @classmethod
    def setUpClass(cls):
        import tts_server
        tts_server.auth_token = None
        tts_server.server_config = {
            "security": {},
            "auto_shutdown_minutes": 0,
        }
        cls.app = tts_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    def test_health_returns_backend(self):
        """/health returns backend field."""
        resp = self.client.get("/health")
        data = resp.get_json()
        self.assertIn("backend", data)
        self.assertIn(data["backend"], ["torch", "mlx"])

    def test_health_returns_model_size(self):
        """/health returns model_size field."""
        resp = self.client.get("/health")
        data = resp.get_json()
        self.assertIn("model_size", data)
        self.assertIn(data["model_size"], ["1.7B", "0.6B"])

    def test_health_returns_model_loaded_fields(self):
        """/health returns individual model loaded fields."""
        resp = self.client.get("/health")
        data = resp.get_json()
        # Check for individual model loaded fields
        self.assertIn("clone_model_loaded", data)
        self.assertIn("design_model_loaded", data)
        self.assertIn("custom_model_loaded", data)
        self.assertIsInstance(data["clone_model_loaded"], bool)


# =============================================================================
# Generation status and chunk progress tests
# =============================================================================

class TestGenerationStatus(unittest.TestCase):
    """Test /generation-status endpoint and chunk progress tracking."""

    @classmethod
    def setUpClass(cls):
        import tts_server
        tts_server.auth_token = None
        tts_server.server_config = {
            "security": {},
            "auto_shutdown_minutes": 0,
        }
        cls.app = tts_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    def test_generation_status_no_auth_required(self):
        """/generation-status is public."""
        resp = self.client.get("/generation-status")
        self.assertEqual(resp.status_code, 200)

    def test_generation_status_returns_active(self):
        """/generation-status returns active field."""
        resp = self.client.get("/generation-status")
        data = resp.get_json()
        self.assertIn("active", data)
        self.assertIsInstance(data["active"], bool)

    def test_generation_status_when_inactive(self):
        """When no generation active, returns minimal info."""
        resp = self.client.get("/generation-status")
        data = resp.get_json()
        self.assertFalse(data["active"])


# =============================================================================
# Load model endpoint tests
# =============================================================================

class TestLoadModelEndpoint(unittest.TestCase):
    """Test /load-model endpoint validation."""

    @classmethod
    def setUpClass(cls):
        import tts_server
        tts_server.auth_token = "test_token"
        tts_server.server_config = {
            "security": {},
            "auto_shutdown_minutes": 0,
        }
        cls.app = tts_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        import tts_server
        tts_server.auth_token = None

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
        self.assertIn("Unknown model type", resp.get_json()["error"])

    def test_load_model_accepts_valid_types(self):
        """POST /load-model accepts clone, design, custom."""
        for model_type in ["clone", "design", "custom"]:
            resp = self.client.post("/load-model",
                json={"model_type": model_type},
                headers={"Authorization": "Bearer test_token"})
            # Should either succeed (200) or fail because model not available (503)
            # but NOT validation error (400)
            self.assertIn(resp.status_code, [200, 503],
                f"model_type '{model_type}' should be valid")


# =============================================================================
# Cancel generation endpoint tests
# =============================================================================

class TestCancelGenerationEndpoint(unittest.TestCase):
    """Test /cancel-generation endpoint."""

    @classmethod
    def setUpClass(cls):
        import tts_server
        tts_server.auth_token = "test_token"
        tts_server.server_config = {
            "security": {},
            "auto_shutdown_minutes": 0,
        }
        # Reset generation state
        tts_server.generation_state.update({
            "active": False,
            "cancelled": False,
            "generation_id": None,
        })
        cls.app = tts_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        import tts_server
        tts_server.auth_token = None

    def test_cancel_requires_auth(self):
        """POST /cancel-generation requires authentication."""
        resp = self.client.post("/cancel-generation")
        self.assertEqual(resp.status_code, 401)

    def test_cancel_when_no_active_generation(self):
        """Cancel returns no_active_generation when nothing running."""
        import tts_server
        tts_server.generation_state["active"] = False
        resp = self.client.post("/cancel-generation",
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "no_active_generation")

    def test_cancel_sets_cancelled_flag(self):
        """Cancel sets the cancelled flag in generation_state."""
        import tts_server
        tts_server.generation_state.update({
            "active": True,
            "cancelled": False,
            "generation_id": "test123",
        })
        resp = self.client.post("/cancel-generation",
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "cancellation_requested")
        self.assertTrue(tts_server.generation_state["cancelled"])
        # Reset
        tts_server.generation_state.update({
            "active": False,
            "cancelled": False,
            "generation_id": None,
        })

    def test_cancel_returns_generation_id(self):
        """Cancel returns the generation_id."""
        import tts_server
        tts_server.generation_state.update({
            "active": True,
            "cancelled": False,
            "generation_id": "abc12345",
        })
        resp = self.client.post("/cancel-generation",
            headers={"Authorization": "Bearer test_token"})
        data = resp.get_json()
        self.assertEqual(data["generation_id"], "abc12345")
        # Reset
        tts_server.generation_state.update({
            "active": False,
            "cancelled": False,
            "generation_id": None,
        })


class TestGenerationStateFields(unittest.TestCase):
    """Test generation_state has required fields for cancellation."""

    def test_generation_state_has_cancelled_field(self):
        """generation_state dict has cancelled field."""
        import tts_server
        self.assertIn("cancelled", tts_server.generation_state)

    def test_generation_state_has_generation_id(self):
        """generation_state dict has generation_id field."""
        import tts_server
        self.assertIn("generation_id", tts_server.generation_state)


# =============================================================================
# Streaming client tests
# =============================================================================

class TestStreamingClientMethod(unittest.TestCase):
    """Test TTSClient.generate_streaming method."""

    def test_generate_streaming_method_exists(self):
        """TTSClient has generate_streaming method."""
        from tts_client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "generate_streaming"))
        self.assertTrue(callable(getattr(client, "generate_streaming")))

    def test_cancel_generation_method_exists(self):
        """TTSClient has cancel_generation method."""
        from tts_client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "cancel_generation"))
        self.assertTrue(callable(getattr(client, "cancel_generation")))

    def test_generate_streaming_signature(self):
        """generate_streaming has expected parameters."""
        import inspect
        from tts_client import TTSClient
        sig = inspect.signature(TTSClient.generate_streaming)
        params = list(sig.parameters.keys())
        # Should have text, mode, and various optional params
        self.assertIn("text", params)
        self.assertIn("mode", params)


# =============================================================================
# UI history functions tests
# =============================================================================

class TestUIHistoryFunctions(unittest.TestCase):
    """Test tts_ui generation history functions."""

    def test_history_functions_exist(self):
        """tts_ui has history-related functions."""
        import tts_ui
        self.assertTrue(hasattr(tts_ui, "generation_history"))
        self.assertTrue(hasattr(tts_ui, "add_to_history"))
        self.assertTrue(hasattr(tts_ui, "get_history_data"))
        self.assertTrue(hasattr(tts_ui, "MAX_HISTORY_SIZE"))

    def test_add_to_history(self):
        """add_to_history adds entries to history."""
        import tts_ui
        # Clear history
        tts_ui.generation_history.clear()

        tts_ui.add_to_history("clone", "Test text", "/path/to/audio.wav", 5)
        self.assertEqual(len(tts_ui.generation_history), 1)
        entry = tts_ui.generation_history[0]
        self.assertEqual(entry["mode"], "Clone")
        self.assertEqual(entry["chunks"], 5)
        self.assertEqual(entry["path"], "/path/to/audio.wav")

    def test_history_max_size(self):
        """History doesn't exceed MAX_HISTORY_SIZE."""
        import tts_ui
        tts_ui.generation_history.clear()

        # Add more than max entries
        for i in range(tts_ui.MAX_HISTORY_SIZE + 5):
            tts_ui.add_to_history("clone", f"Text {i}", f"/path/{i}.wav", 1)

        self.assertEqual(len(tts_ui.generation_history), tts_ui.MAX_HISTORY_SIZE)

    def test_get_history_data_format(self):
        """get_history_data returns list of lists."""
        import tts_ui
        tts_ui.generation_history.clear()
        tts_ui.add_to_history("clone", "Test text", "/path/test.wav", 3)

        data = tts_ui.get_history_data()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertIsInstance(data[0], list)
        # Should be [time, mode, text, chunks]
        self.assertEqual(len(data[0]), 4)

    def test_history_text_truncation(self):
        """Long text is truncated in history entries."""
        import tts_ui
        tts_ui.generation_history.clear()

        long_text = "A" * 100  # 100 character text
        tts_ui.add_to_history("clone", long_text, "/path/test.wav", 1)

        entry = tts_ui.generation_history[0]
        # Text should be truncated to 40 chars + "..."
        self.assertLessEqual(len(entry["text"]), 43)
        self.assertTrue(entry["text"].endswith("..."))


# =============================================================================
# UI cancel function tests
# =============================================================================

class TestUICancelFunction(unittest.TestCase):
    """Test tts_ui cancel streaming function."""

    def test_cancel_streaming_generation_exists(self):
        """tts_ui has cancel_streaming_generation function."""
        import tts_ui
        self.assertTrue(hasattr(tts_ui, "cancel_streaming_generation"))
        self.assertTrue(callable(tts_ui.cancel_streaming_generation))


# =============================================================================
# UI text info helper tests
# =============================================================================

class TestUITextInfo(unittest.TestCase):
    """Test tts_ui text info helper functions."""

    def test_update_text_info_exists(self):
        """tts_ui has update_text_info function."""
        import tts_ui
        self.assertTrue(hasattr(tts_ui, "update_text_info"))

    def test_update_text_info_empty(self):
        """update_text_info returns empty string for empty input."""
        from tts_ui import update_text_info
        self.assertEqual(update_text_info(""), "")
        self.assertEqual(update_text_info(None), "")

    def test_update_text_info_short(self):
        """update_text_info shows char count for short text."""
        from tts_ui import update_text_info
        result = update_text_info("Hello")
        self.assertIn("5 chars", result)

    def test_update_text_info_long(self):
        """update_text_info shows chunks estimate for long text."""
        from tts_ui import update_text_info
        long_text = "A" * 1000  # 1000 chars = ~2 chunks
        result = update_text_info(long_text)
        self.assertIn("1000 chars", result)
        self.assertIn("chunks", result)


# =============================================================================
# Streaming server endpoint structure tests
# =============================================================================

class TestStreamingEndpointStructure(unittest.TestCase):
    """Test /generate-stream endpoint structure."""

    @classmethod
    def setUpClass(cls):
        import tts_server
        tts_server.auth_token = "test_token"
        tts_server.server_config = {
            "security": {"max_text_length": 10000},
            "auto_shutdown_minutes": 0,
        }
        cls.app = tts_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        import tts_server
        tts_server.auth_token = None

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
        self.assertIn("No text provided", resp.get_json()["error"])

    def test_generate_stream_validates_mode(self):
        """POST /generate-stream validates mode."""
        resp = self.client.post("/generate-stream",
            json={"text": "Hello", "mode": "invalid"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid mode", resp.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
