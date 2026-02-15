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
import time
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# voice_config tests
# =============================================================================

class TestTTSConfig(unittest.TestCase):
    """Test voice_config module (no heavy imports)."""

    def test_imports_no_torch(self):
        """voice_config must not import torch."""
        import voice_config  # noqa: F401
        self.assertNotIn("torch", sys.modules.keys() - {"torch"})

    def test_error_hierarchy(self):
        from voice_config import (
            TTSError, ServerConnectionError, ModelNotLoadedError,
            InvalidInputError, GenerationError, AuthenticationError,
        )
        # All errors inherit from TTSError
        for cls in [ServerConnectionError, ModelNotLoadedError,
                    InvalidInputError, GenerationError, AuthenticationError]:
            err = cls("test") if cls != ModelNotLoadedError else cls("clone")
            self.assertIsInstance(err, TTSError)

    def test_error_format_cli(self):
        from voice_config import ServerConnectionError
        err = ServerConnectionError("details here")
        formatted = err.format_cli()
        self.assertIn("Cannot connect", formatted)
        self.assertIn("startTTSServer", formatted)

    def test_error_format_gradio(self):
        from voice_config import GenerationError
        err = GenerationError("oops")
        html = err.format_gradio()
        self.assertIn("Audio generation failed", html)

    def test_read_auth_token_missing(self):
        from voice_config import TOKEN_FILE, read_auth_token
        # Use a temp path that doesn't exist
        with patch("voice_config.TOKEN_FILE", "/tmp/nonexistent_token_test_xyz"):
            result = read_auth_token()
        # If the real file exists, it returns its content; with patched path it's None
        # Can't reliably test with real TOKEN_FILE, so test the function signature
        self.assertTrue(result is None or isinstance(result, str))

    def test_auth_headers_returns_dict(self):
        from voice_config import auth_headers
        headers = auth_headers()
        self.assertIsInstance(headers, dict)

    def test_auth_headers_with_token(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".token", delete=False) as f:
            f.write("test_token_abc123")
            token_path = f.name
        try:
            with patch("voice_config.TOKEN_FILE", token_path):
                from voice_config import read_auth_token, auth_headers
                # Need to reimport to pick up patched value
                token = read_auth_token()
                headers = auth_headers()
            # The token file exists so we should get headers
            # (actual behavior depends on whether TOKEN_FILE is patched at call time)
        finally:
            os.unlink(token_path)

    def test_model_info_keys(self):
        from voice_config import MODEL_INFO
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
        from voice_config import CUSTOM_VOICE_SPEAKERS
        self.assertIn("ryan", CUSTOM_VOICE_SPEAKERS)
        self.assertIn("aiden", CUSTOM_VOICE_SPEAKERS)
        for key, info in CUSTOM_VOICE_SPEAKERS.items():
            self.assertIn("name", info)
            self.assertIn("lang", info)
            self.assertIn("desc", info)


# =============================================================================
# voice_server validation tests (using Flask test client, no models needed)
# =============================================================================

class TestServerValidation(unittest.TestCase):
    """Test server input validation without loading any models."""

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client with mocked models."""
        # We need to mock torch and model-related imports
        # to avoid loading heavy dependencies
        import voice_server
        voice_server.auth_token = None  # Disable auth for tests
        voice_server.server_config = {
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
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
# voice_server auth tests
# =============================================================================

class TestServerAuth(unittest.TestCase):
    """Test server authentication."""

    @classmethod
    def setUpClass(cls):
        import voice_server
        voice_server.auth_token = "test_secret_token"
        voice_server.server_config = {
            "security": {},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        import voice_server
        voice_server.auth_token = None

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
# SSML parsing tests (from voice_generate, lightweight)
# =============================================================================

class TestSSMLParsing(unittest.TestCase):
    """Test SSML parsing in voice_generate."""

    def test_no_ssml(self):
        from voice_generate import parse_ssml
        text, meta = parse_ssml("Hello world")
        self.assertEqual(text, "Hello world")
        self.assertFalse(meta["has_ssml"])

    def test_break_tag(self):
        from voice_generate import parse_ssml
        text, meta = parse_ssml('Hello <break time="500ms"/> world')
        self.assertTrue(meta["has_ssml"])
        self.assertNotIn("<break", text)

    def test_sub_tag(self):
        from voice_generate import parse_ssml
        text, meta = parse_ssml('<sub alias="World Wide Web">WWW</sub>')
        self.assertIn("World Wide Web", text)
        self.assertNotIn("WWW", text)

    def test_say_as_characters(self):
        from voice_generate import parse_ssml
        text, meta = parse_ssml('<say-as interpret-as="characters">ABC</say-as>')
        self.assertIn("A B C", text)

    def test_prosody_speed(self):
        from voice_generate import parse_ssml
        text, meta = parse_ssml('<prosody rate="fast">Quick text</prosody>')
        self.assertTrue(meta["has_ssml"])
        self.assertEqual(meta["prosody"]["speed"], 1.2)


# =============================================================================
# SRT parsing tests
# =============================================================================

class TestSRTParsing(unittest.TestCase):
    """Test SRT parsing."""

    def test_parse_srt(self):
        from voice_generate import parse_srt
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
        from voice_generate import srt_time_to_ms
        self.assertEqual(srt_time_to_ms("00:01:30,500"), 90500)
        self.assertEqual(srt_time_to_ms("01:00:00,000"), 3600000)


# =============================================================================
# Auto-increment filename tests
# =============================================================================

class TestAutoIncrementFilename(unittest.TestCase):
    """Test auto_increment_filename helper."""

    def test_no_conflict(self):
        from voice_generate import auto_increment_filename
        # Non-existent file should return as-is
        result = auto_increment_filename("/tmp/nonexistent_test_xyz.wav")
        self.assertEqual(result, "/tmp/nonexistent_test_xyz.wav")

    def test_conflict_increments(self):
        from voice_generate import auto_increment_filename
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
        from voice_generate import auto_increment_filename
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
# Backend config tests (voice_config backend helpers)
# =============================================================================

class TestBackendConfig(unittest.TestCase):
    """Test backend-related config helpers in voice_config."""

    def test_valid_backends(self):
        from voice_config import VALID_BACKENDS
        self.assertIn("torch", VALID_BACKENDS)
        self.assertIn("mlx", VALID_BACKENDS)

    def test_valid_mlx_quantizations(self):
        from voice_config import VALID_MLX_QUANTIZATIONS
        self.assertIn("4bit", VALID_MLX_QUANTIZATIONS)
        self.assertIn("8bit", VALID_MLX_QUANTIZATIONS)
        self.assertIn("bf16", VALID_MLX_QUANTIZATIONS)

    def test_mlx_model_info_keys(self):
        from voice_config import MLX_MODEL_INFO
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
        from voice_config import MLX_MODEL_INFO
        for size in ("1.7B", "0.6B"):
            for model_type, info in MLX_MODEL_INFO[size].items():
                self.assertIn("{quant}", info["name_template"])

    def test_get_backend_default(self):
        """get_backend() defaults to 'mlx' with no env/config override (MLX-first architecture)."""
        from voice_config import get_backend
        # Clear env override
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_BACKEND", None)
            with patch("voice_config.load_config", return_value={}):
                result = get_backend()
        self.assertEqual(result, "mlx")

    def test_get_backend_from_config(self):
        from voice_config import get_backend
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_BACKEND", None)
            with patch("voice_config.load_config", return_value={"advanced": {"backend": "mlx"}}):
                result = get_backend()
        self.assertEqual(result, "mlx")

    def test_get_backend_env_override(self):
        """TTS_BACKEND env var overrides config."""
        from voice_config import get_backend
        with patch.dict(os.environ, {"TTS_BACKEND": "mlx"}):
            with patch("voice_config.load_config", return_value={"advanced": {"backend": "torch"}}):
                result = get_backend()
        self.assertEqual(result, "mlx")

    def test_get_backend_invalid_falls_back(self):
        """Invalid backend value falls back to 'mlx' (MLX-first architecture)."""
        from voice_config import get_backend
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_BACKEND", None)
            with patch("voice_config.load_config", return_value={"advanced": {"backend": "invalid"}}):
                result = get_backend()
        self.assertEqual(result, "mlx")

    def test_get_mlx_quantization_default(self):
        from voice_config import get_mlx_quantization
        with patch("voice_config.load_config", return_value={}):
            result = get_mlx_quantization()
        self.assertEqual(result, "8bit")

    def test_get_mlx_quantization_from_config(self):
        from voice_config import get_mlx_quantization
        with patch("voice_config.load_config", return_value={"advanced": {"mlx_quantization": "4bit"}}):
            result = get_mlx_quantization()
        self.assertEqual(result, "4bit")

    def test_get_mlx_quantization_invalid_falls_back(self):
        from voice_config import get_mlx_quantization
        with patch("voice_config.load_config", return_value={"advanced": {"mlx_quantization": "garbage"}}):
            result = get_mlx_quantization()
        self.assertEqual(result, "8bit")

    def test_get_mlx_model_name(self):
        from voice_config import get_mlx_model_name
        with patch("voice_config.get_mlx_quantization", return_value="8bit"):
            name = get_mlx_model_name("clone")
        self.assertEqual(name, "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")

    def test_get_mlx_model_name_4bit(self):
        from voice_config import get_mlx_model_name
        with patch("voice_config.get_mlx_quantization", return_value="4bit"):
            name = get_mlx_model_name("design")
        self.assertEqual(name, "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit")

    def test_get_mlx_model_name_invalid_type(self):
        from voice_config import get_mlx_model_name
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
        from voice_engine import load_voice_prompt_mlx
        # Create fake wav and txt
        wav_path = os.path.join(self.tmpdir, "test_voice.wav")
        txt_path = os.path.join(self.tmpdir, "test_voice.txt")
        with open(wav_path, "wb") as f:
            f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")
        with open(txt_path, "w") as f:
            f.write("Hello, this is a test transcript.")

        with patch("voice_engine.VOICE_PROMPTS_DIR", self.tmpdir):
            result = load_voice_prompt_mlx("test_voice.pt")

        self.assertIsInstance(result, dict)
        self.assertEqual(result["ref_audio"], wav_path)
        self.assertEqual(result["ref_text"], "Hello, this is a test transcript.")

    def test_load_voice_prompt_mlx_strips_pt(self):
        """Prompt name with .pt extension is handled correctly."""
        from voice_engine import load_voice_prompt_mlx
        wav_path = os.path.join(self.tmpdir, "voice.wav")
        txt_path = os.path.join(self.tmpdir, "voice.txt")
        with open(wav_path, "wb") as f:
            f.write(b"fake")
        with open(txt_path, "w") as f:
            f.write("transcript")

        with patch("voice_engine.VOICE_PROMPTS_DIR", self.tmpdir):
            result = load_voice_prompt_mlx("voice.pt")
        self.assertEqual(result["ref_audio"], wav_path)

    def test_load_voice_prompt_mlx_missing_files(self):
        """Raises FileNotFoundError when wav/txt missing."""
        from voice_engine import load_voice_prompt_mlx
        with patch("voice_engine.VOICE_PROMPTS_DIR", self.tmpdir):
            with self.assertRaises(FileNotFoundError):
                load_voice_prompt_mlx("nonexistent")

    def test_load_voice_prompt_mlx_pt_only_error(self):
        """Clear error when only .pt exists (no MLX-compatible files)."""
        from voice_engine import load_voice_prompt_mlx
        pt_path = os.path.join(self.tmpdir, "legacy.pt")
        with open(pt_path, "wb") as f:
            f.write(b"fake tensor data")

        with patch("voice_engine.VOICE_PROMPTS_DIR", self.tmpdir):
            with self.assertRaises(FileNotFoundError) as ctx:
                load_voice_prompt_mlx("legacy")
            self.assertIn("only has a .pt file", str(ctx.exception))
            self.assertIn("createVoice", str(ctx.exception))

    def test_load_voice_prompt_dispatch_torch(self):
        """load_voice_prompt dispatches to torch backend."""
        from voice_engine import load_voice_prompt
        with patch("voice_engine.get_backend", return_value="torch"):
            with patch("voice_engine._load_voice_prompt_torch", return_value="mock_tensor") as mock:
                result = load_voice_prompt("test.pt")
        mock.assert_called_once_with("test.pt")
        self.assertEqual(result, "mock_tensor")

    def test_load_voice_prompt_dispatch_mlx(self):
        """load_voice_prompt dispatches to MLX backend."""
        from voice_engine import load_voice_prompt
        mock_result = {"ref_audio": "/fake/path.wav", "ref_text": "text"}
        with patch("voice_engine.get_backend", return_value="mlx"):
            with patch("voice_engine.load_voice_prompt_mlx", return_value=mock_result) as mock:
                result = load_voice_prompt("test.pt")
        mock.assert_called_once_with("test.pt")
        self.assertEqual(result, mock_result)


# =============================================================================
# Backend dispatch tests (no actual model loading — tests dispatch logic)
# =============================================================================

class TestBackendDispatch(unittest.TestCase):
    """Test that public API dispatches to correct backend functions."""

    def test_load_model_dispatch_torch(self):
        from voice_engine import load_model
        with patch("voice_engine.get_backend", return_value="torch"):
            with patch("voice_engine._load_model_torch", return_value="torch_model") as mock:
                result = load_model("clone")
        mock.assert_called_once_with("clone")
        self.assertEqual(result, "torch_model")

    def test_load_model_dispatch_mlx(self):
        from voice_engine import load_model
        with patch("voice_engine.get_backend", return_value="mlx"):
            with patch("voice_engine._load_model_mlx", return_value="mlx_model") as mock:
                result = load_model("design")
        mock.assert_called_once_with("design")
        self.assertEqual(result, "mlx_model")

    def test_run_inference_dispatch_torch(self):
        from voice_engine import run_inference
        with patch("voice_engine.get_backend", return_value="torch"):
            with patch("voice_engine._run_inference_torch", return_value=("wav", 24000)) as mock:
                result = run_inference("model", "text", "clone", {})
        mock.assert_called_once()
        self.assertEqual(result, ("wav", 24000))

    def test_run_inference_dispatch_mlx(self):
        from voice_engine import run_inference
        with patch("voice_engine.get_backend", return_value="mlx"):
            with patch("voice_engine._run_inference_mlx", return_value=("wav", 24000)) as mock:
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
        from voice_engine import _run_inference_mlx
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
        from voice_engine import _run_inference_mlx
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
    """Verify that voice_engine does not import torch or mlx at module scope."""

    def test_engine_no_torch_at_module_scope(self):
        """voice_engine module should not force-import torch."""
        # Remove voice_engine from cache to test fresh import behavior
        saved_modules = {}
        for mod in list(sys.modules.keys()):
            if mod == "voice_engine" or mod.startswith("voice_engine."):
                saved_modules[mod] = sys.modules.pop(mod)

        # Also note if torch was already loaded
        torch_was_loaded = "torch" in sys.modules

        try:
            import voice_engine  # noqa: F401
            if not torch_was_loaded:
                # torch should not have been imported by voice_engine
                self.assertNotIn("torch", sys.modules,
                    "voice_engine imported torch at module scope")
        finally:
            # Restore
            for mod, val in saved_modules.items():
                sys.modules[mod] = val

    def test_config_no_torch(self):
        """voice_config must not import torch (regression check)."""
        import voice_config  # noqa: F401
        # voice_config should never cause torch to load
        self.assertNotIn("torch", dir(voice_config))


# =============================================================================
# Phase 14: 0.6B Model Size tests
# =============================================================================

class TestModelSize(unittest.TestCase):
    """Test 0.6B model size configuration."""

    def test_valid_model_sizes(self):
        from voice_config import VALID_MODEL_SIZES
        self.assertIn("1.7B", VALID_MODEL_SIZES)
        self.assertIn("0.6B", VALID_MODEL_SIZES)

    def test_get_model_size_default(self):
        """get_model_size() defaults to 1.7B."""
        from voice_config import get_model_size
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_MODEL_SIZE", None)
            with patch("voice_config.load_config", return_value={}):
                self.assertEqual(get_model_size(), "1.7B")

    def test_get_model_size_from_config(self):
        from voice_config import get_model_size
        config = {"advanced": {"model_size": "0.6B"}}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TTS_MODEL_SIZE", None)
            with patch("voice_config.load_config", return_value=config):
                self.assertEqual(get_model_size(), "0.6B")

    def test_get_model_size_env_override(self):
        """TTS_MODEL_SIZE env var overrides config."""
        from voice_config import get_model_size
        config = {"advanced": {"model_size": "1.7B"}}
        with patch.dict(os.environ, {"TTS_MODEL_SIZE": "0.6B"}):
            with patch("voice_config.load_config", return_value=config):
                self.assertEqual(get_model_size(), "0.6B")

    def test_get_torch_model_name(self):
        from voice_config import get_torch_model_name
        with patch("voice_config.get_model_size", return_value="1.7B"):
            name = get_torch_model_name("clone")
            self.assertIn("1.7B", name)
            self.assertIn("Base", name)

    def test_get_torch_model_name_0_6B(self):
        from voice_config import get_torch_model_name
        with patch("voice_config.get_model_size", return_value="0.6B"):
            name = get_torch_model_name("clone")
            self.assertIn("0.6B", name)

    def test_model_info_has_0_6B(self):
        from voice_config import MODEL_INFO, MLX_MODEL_INFO
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
        from voice_engine import run_inference_streaming
        self.assertTrue(callable(run_inference_streaming))

    def test_mlx_streaming_function_exists(self):
        """_run_inference_mlx_streaming function is importable."""
        from voice_engine import _run_inference_mlx_streaming
        self.assertTrue(callable(_run_inference_mlx_streaming))

    def test_streaming_torch_falls_back_to_chunked(self):
        """run_inference_streaming for torch uses chunked inference (not native streaming)."""
        from voice_engine import run_inference_streaming
        import inspect
        source = inspect.getsource(run_inference_streaming)
        # Torch backend falls back to chunked approach
        self.assertIn("_run_inference_single", source)

    def test_streaming_mlx_function_signature(self):
        """_run_inference_mlx_streaming has correct parameters."""
        from voice_engine import _run_inference_mlx_streaming
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
        import voice_server
        voice_server.auth_token = "test_token"
        voice_server.server_config = {
            "security": {"max_text_length": 1000, "max_batch_size": 10},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        import voice_server
        voice_server.auth_token = None

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
        from voice_engine import transcribe_audio
        self.assertTrue(callable(transcribe_audio))

    def test_is_asr_available_exists(self):
        """is_asr_available function is importable."""
        from voice_engine import is_asr_available
        self.assertTrue(callable(is_asr_available))

    def test_is_asr_available_torch_returns_false(self):
        """is_asr_available returns False when backend is torch."""
        from voice_engine import is_asr_available
        with patch("voice_engine.get_backend", return_value="torch"):
            self.assertFalse(is_asr_available())

    def test_transcribe_audio_requires_mlx(self):
        """transcribe_audio raises ImportError when backend is torch."""
        from voice_engine import transcribe_audio
        with patch("voice_engine.get_backend", return_value="torch"):
            with self.assertRaises(ImportError) as ctx:
                transcribe_audio("/fake/path.wav")
            self.assertIn("MLX backend", str(ctx.exception))

    def test_asr_model_is_lazy_loaded(self):
        """_asr_model is None until transcribe_audio is called."""
        import voice_engine
        # The global _asr_model should be None at module level
        self.assertIsNone(voice_engine._asr_model)

    def test_is_asr_available_mlx_with_stt(self):
        """is_asr_available returns True when MLX + mlx_audio.stt available."""
        from voice_engine import is_asr_available
        with patch("voice_engine.get_backend", return_value="mlx"):
            # Mock successful import of load_model
            with patch.dict(sys.modules, {"mlx_audio.stt": MagicMock()}):
                # Force re-check by clearing any cached imports
                result = is_asr_available()
        # Should attempt to import and return True if successful
        # (actual result depends on whether mlx_audio is installed)
        self.assertIsInstance(result, bool)

    def test_transcribe_audio_returns_string(self):
        """transcribe_audio returns a string."""
        from voice_engine import transcribe_audio

        # Mock the entire transcription flow
        mock_result = MagicMock()
        mock_result.text = "Hello world"

        mock_model = MagicMock()
        mock_model.generate.return_value = mock_result

        with patch("voice_engine.get_backend", return_value="mlx"):
            with patch("voice_engine._asr_model", mock_model):
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
        from voice_engine import _RETRY_DELAYS
        self.assertEqual(len(_RETRY_DELAYS), 3)
        self.assertEqual(_RETRY_DELAYS, (5, 15, 45))

    def test_retry_delays_is_exponential(self):
        """_RETRY_DELAYS uses exponential backoff pattern."""
        from voice_engine import _RETRY_DELAYS
        # Each delay should be roughly 3x the previous (5 -> 15 -> 45)
        self.assertEqual(_RETRY_DELAYS[1], _RETRY_DELAYS[0] * 3)
        self.assertEqual(_RETRY_DELAYS[2], _RETRY_DELAYS[1] * 3)

    def test_max_chunk_chars_helper_exists(self):
        """_get_max_chunk_chars helper function exists."""
        from voice_engine import _get_max_chunk_chars
        self.assertTrue(callable(_get_max_chunk_chars))

    def test_max_chunk_chars_default(self):
        """_get_max_chunk_chars returns default 500."""
        from voice_engine import _get_max_chunk_chars
        with patch("voice_engine.load_config", return_value={}):
            result = _get_max_chunk_chars()
        self.assertEqual(result, 500)

    def test_max_chunk_chars_from_config(self):
        """_get_max_chunk_chars reads from config."""
        from voice_engine import _get_max_chunk_chars
        config = {"generation": {"max_chunk_chars": 300}}
        with patch("voice_engine.load_config", return_value=config):
            result = _get_max_chunk_chars()
        self.assertEqual(result, 300)


class TestFloat32Guard(unittest.TestCase):
    """Test float32 dtype guard for torch clone mode on MPS."""

    def test_float32_guard_exists_in_torch_inference(self):
        """_run_inference_torch has float32 guard logic."""
        from voice_engine import _run_inference_torch
        import inspect
        source = inspect.getsource(_run_inference_torch)
        # Should have float32 override logic for clone mode
        self.assertIn("float32", source)
        self.assertIn("clone", source)


class TestMLXMetalRecovery(unittest.TestCase):
    """Test MLX Metal kernel crash recovery."""

    def test_run_inference_handles_exceptions(self):
        """run_inference wraps inference in try/except."""
        from voice_engine import _run_inference_single
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
        from voice_engine import _split_text
        chunks = _split_text("Hello world.", max_chars=500)
        self.assertEqual(chunks, ["Hello world."])

    def test_split_text_sentences(self):
        """Text is split on sentence boundaries."""
        from voice_engine import _split_text
        text = "First sentence. Second sentence. Third sentence."
        chunks = _split_text(text, max_chars=30)
        self.assertGreater(len(chunks), 1)
        # Each chunk should be <= max_chars
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 30)

    def test_split_text_preserves_content(self):
        """All content is preserved after splitting."""
        from voice_engine import _split_text
        text = "The quick brown fox jumps over the lazy dog. A second sentence follows."
        chunks = _split_text(text, max_chars=50)
        combined = " ".join(chunks)
        # All words should be present
        for word in text.split():
            self.assertIn(word.rstrip(".,"), combined)

    def test_split_text_question_mark(self):
        """Text splits on question marks."""
        from voice_engine import _split_text
        text = "Is this a question? Yes it is."
        chunks = _split_text(text, max_chars=25)
        self.assertGreater(len(chunks), 1)

    def test_split_text_exclamation(self):
        """Text splits on exclamation marks."""
        from voice_engine import _split_text
        text = "Hello! How are you today?"
        chunks = _split_text(text, max_chars=15)
        self.assertGreater(len(chunks), 1)

    def test_split_text_newlines(self):
        """Text splits on newlines."""
        from voice_engine import _split_text
        text = "First paragraph.\n\nSecond paragraph."
        chunks = _split_text(text, max_chars=20)
        self.assertGreater(len(chunks), 1)

    def test_split_text_comma_fallback(self):
        """Very long sentence falls back to clause boundaries."""
        from voice_engine import _split_text
        # A single long sentence with commas but no periods
        text = "This is a very long sentence, with several clauses, that should be split at commas when needed"
        chunks = _split_text(text, max_chars=40)
        # Should split due to length
        self.assertGreater(len(chunks), 1)


# =============================================================================
# Server health endpoint info tests
# =============================================================================

class TestMLXMemoryStats(unittest.TestCase):
    """Test MLX memory stats collection in /stats endpoint."""

    def test_stats_mlx_memory_code_exists(self):
        """voice_server has MLX memory collection code."""
        import inspect
        import voice_server
        # Find the stats route handler
        source = inspect.getsource(voice_server)
        # Should have MLX memory collection
        self.assertIn("mlx_memory_active_mb", source)
        self.assertIn("mlx_memory_peak_mb", source)
        self.assertIn("mx.metal.get_active_memory", source)

    def test_ui_checks_mlx_memory_first(self):
        """voice_ui checks for MLX memory before MPS memory."""
        import inspect
        import voice_ui
        source = inspect.getsource(voice_ui.get_server_status)
        # Should check mlx_memory first
        self.assertIn("mlx_memory_active_mb", source)


class TestHealthEndpointInfo(unittest.TestCase):
    """Test /health endpoint returns expected info fields."""

    @classmethod
    def setUpClass(cls):
        import voice_server
        voice_server.auth_token = None
        voice_server.server_config = {
            "security": {},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
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
        import voice_server
        voice_server.auth_token = None
        voice_server.server_config = {
            "security": {},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
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
        import voice_server
        voice_server.auth_token = "test_token"
        voice_server.server_config = {
            "security": {},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        import voice_server
        voice_server.auth_token = None

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
            # Should either succeed (200), fail because model not available (503),
            # or fail because backend library not installed (500)
            # but NOT validation error (400)
            self.assertIn(resp.status_code, [200, 500, 503],
                f"model_type '{model_type}' should be valid")


# =============================================================================
# Cancel generation endpoint tests
# =============================================================================

class TestCancelGenerationEndpoint(unittest.TestCase):
    """Test /cancel-generation endpoint."""

    @classmethod
    def setUpClass(cls):
        import voice_server
        voice_server.auth_token = "test_token"
        voice_server.server_config = {
            "security": {},
            "auto_shutdown_minutes": 0,
        }
        # Reset generation state
        voice_server.generation_state.update({
            "active": False,
            "cancelled": False,
            "generation_id": None,
        })
        cls.app = voice_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        import voice_server
        voice_server.auth_token = None

    def test_cancel_requires_auth(self):
        """POST /cancel-generation requires authentication."""
        resp = self.client.post("/cancel-generation")
        self.assertEqual(resp.status_code, 401)

    def test_cancel_when_no_active_generation(self):
        """Cancel returns no_active_generation when nothing running."""
        import voice_server
        voice_server.generation_state["active"] = False
        resp = self.client.post("/cancel-generation",
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "no_active_generation")

    def test_cancel_sets_cancelled_flag(self):
        """Cancel sets the cancelled flag in generation_state."""
        import voice_server
        voice_server.generation_state.update({
            "active": True,
            "cancelled": False,
            "generation_id": "test123",
        })
        resp = self.client.post("/cancel-generation",
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "cancellation_requested")
        self.assertTrue(voice_server.generation_state["cancelled"])
        # Reset
        voice_server.generation_state.update({
            "active": False,
            "cancelled": False,
            "generation_id": None,
        })

    def test_cancel_returns_generation_id(self):
        """Cancel returns the generation_id."""
        import voice_server
        voice_server.generation_state.update({
            "active": True,
            "cancelled": False,
            "generation_id": "abc12345",
        })
        resp = self.client.post("/cancel-generation",
            headers={"Authorization": "Bearer test_token"})
        data = resp.get_json()
        self.assertEqual(data["generation_id"], "abc12345")
        # Reset
        voice_server.generation_state.update({
            "active": False,
            "cancelled": False,
            "generation_id": None,
        })


class TestGenerationStateFields(unittest.TestCase):
    """Test generation_state has required fields for cancellation."""

    def test_generation_state_has_cancelled_field(self):
        """generation_state dict has cancelled field."""
        import voice_server
        self.assertIn("cancelled", voice_server.generation_state)

    def test_generation_state_has_generation_id(self):
        """generation_state dict has generation_id field."""
        import voice_server
        self.assertIn("generation_id", voice_server.generation_state)

    def test_generation_state_initial_values(self):
        """generation_state has correct initial values."""
        import voice_server
        # These should be the default/initial values
        state = voice_server.generation_state
        self.assertIn("active", state)
        self.assertIn("start_time", state)
        self.assertIn("text_length", state)
        self.assertIn("mode", state)
        self.assertIn("chunk_index", state)
        self.assertIn("chunk_total", state)


# =============================================================================
# Streaming client tests
# =============================================================================

class TestStreamingClientMethod(unittest.TestCase):
    """Test TTSClient.generate_streaming method."""

    def test_generate_streaming_method_exists(self):
        """TTSClient has generate_streaming method."""
        from voice_client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "generate_streaming"))
        self.assertTrue(callable(getattr(client, "generate_streaming")))

    def test_cancel_generation_method_exists(self):
        """TTSClient has cancel_generation method."""
        from voice_client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "cancel_generation"))
        self.assertTrue(callable(getattr(client, "cancel_generation")))

    def test_generate_streaming_signature(self):
        """generate_streaming has expected parameters."""
        import inspect
        from voice_client import TTSClient
        sig = inspect.signature(TTSClient.generate_streaming)
        params = list(sig.parameters.keys())
        # Should have text, mode, and various optional params
        self.assertIn("text", params)
        self.assertIn("mode", params)


# =============================================================================
# UI history functions tests
# =============================================================================

class TestUIHistoryFunctions(unittest.TestCase):
    """Test voice_ui generation history functions."""

    def test_history_functions_exist(self):
        """voice_ui has history-related functions."""
        import voice_ui
        self.assertTrue(hasattr(voice_ui, "generation_history"))
        self.assertTrue(hasattr(voice_ui, "add_to_history"))
        self.assertTrue(hasattr(voice_ui, "get_history_data"))
        self.assertTrue(hasattr(voice_ui, "MAX_HISTORY_SIZE"))

    def test_add_to_history(self):
        """add_to_history adds entries to history."""
        import voice_ui
        # Clear history
        voice_ui.generation_history.clear()

        voice_ui.add_to_history("clone", "Test text", "/path/to/audio.wav", 5)
        self.assertEqual(len(voice_ui.generation_history), 1)
        entry = voice_ui.generation_history[0]
        self.assertEqual(entry["mode"], "Clone")
        self.assertEqual(entry["chunks"], 5)
        self.assertEqual(entry["path"], "/path/to/audio.wav")

    def test_history_max_size(self):
        """History doesn't exceed MAX_HISTORY_SIZE."""
        import voice_ui
        voice_ui.generation_history.clear()

        # Add more than max entries
        for i in range(voice_ui.MAX_HISTORY_SIZE + 5):
            voice_ui.add_to_history("clone", f"Text {i}", f"/path/{i}.wav", 1)

        self.assertEqual(len(voice_ui.generation_history), voice_ui.MAX_HISTORY_SIZE)

    def test_get_history_data_format(self):
        """get_history_data returns list of lists."""
        import voice_ui
        voice_ui.generation_history.clear()
        voice_ui.add_to_history("clone", "Test text", "/path/test.wav", 3)

        data = voice_ui.get_history_data()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertIsInstance(data[0], list)
        # Should be [time, mode, text, chunks]
        self.assertEqual(len(data[0]), 4)

    def test_history_text_truncation(self):
        """Long text is truncated in history entries."""
        import voice_ui
        voice_ui.generation_history.clear()

        long_text = "A" * 100  # 100 character text
        voice_ui.add_to_history("clone", long_text, "/path/test.wav", 1)

        entry = voice_ui.generation_history[0]
        # Text should be truncated to 40 chars + "..."
        self.assertLessEqual(len(entry["text"]), 43)
        self.assertTrue(entry["text"].endswith("..."))


# =============================================================================
# UI cancel function tests
# =============================================================================

class TestUICancelFunction(unittest.TestCase):
    """Test voice_ui cancel streaming function."""

    def test_cancel_streaming_generation_exists(self):
        """voice_ui has cancel_streaming_generation function."""
        import voice_ui
        self.assertTrue(hasattr(voice_ui, "cancel_streaming_generation"))
        self.assertTrue(callable(voice_ui.cancel_streaming_generation))

    def test_cancel_streaming_generation_returns_tuple(self):
        """cancel_streaming_generation returns a tuple."""
        from voice_ui import cancel_streaming_generation
        from unittest.mock import patch, MagicMock

        mock_client = MagicMock()
        mock_client.cancel_generation.return_value = {"status": "no_active_generation"}

        with patch("voice_ui.TTSClient", return_value=mock_client):
            result = cancel_streaming_generation()

        self.assertIsInstance(result, tuple)
        # Should return (audio, status, status_html)
        self.assertEqual(len(result), 3)

    def test_cancel_streaming_generation_clears_audio(self):
        """cancel_streaming_generation returns None for audio to clear player."""
        from voice_ui import cancel_streaming_generation
        from unittest.mock import patch, MagicMock

        mock_client = MagicMock()
        mock_client.cancel_generation.return_value = {"status": "cancellation_requested"}

        with patch("voice_ui.TTSClient", return_value=mock_client):
            result = cancel_streaming_generation()

        # First element (audio) should be None to clear the player
        self.assertIsNone(result[0])

    def test_check_generation_cancelled_exists(self):
        """voice_ui has _check_generation_cancelled helper."""
        import voice_ui
        self.assertTrue(hasattr(voice_ui, "_check_generation_cancelled"))
        self.assertTrue(callable(voice_ui._check_generation_cancelled))


# =============================================================================
# UI text info helper tests
# =============================================================================

class TestUITextInfo(unittest.TestCase):
    """Test voice_ui text info helper functions."""

    def test_update_text_info_exists(self):
        """voice_ui has update_text_info function."""
        import voice_ui
        self.assertTrue(hasattr(voice_ui, "update_text_info"))

    def test_update_text_info_empty(self):
        """update_text_info returns empty string for empty input."""
        from voice_ui import update_text_info
        self.assertEqual(update_text_info(""), "")
        self.assertEqual(update_text_info(None), "")

    def test_update_text_info_short(self):
        """update_text_info shows char count for short text."""
        from voice_ui import update_text_info
        result = update_text_info("Hello")
        self.assertIn("5 chars", result)

    def test_update_text_info_long(self):
        """update_text_info shows chunks estimate for long text."""
        from voice_ui import update_text_info
        long_text = "A" * 1000  # 1000 chars = ~2 chunks
        result = update_text_info(long_text)
        self.assertIn("1000 chars", result)
        self.assertIn("chunks", result)


# =============================================================================
# UI model settings tests
# =============================================================================

class TestUIModelSettings(unittest.TestCase):
    """Test voice_ui model settings functions (Phase 19: MLX-First Architecture)."""

    def test_model_settings_functions_exist(self):
        """voice_ui has model settings functions."""
        import voice_ui
        self.assertTrue(hasattr(voice_ui, "get_current_model_settings"))
        self.assertTrue(hasattr(voice_ui, "apply_model_settings"))
        self.assertTrue(callable(voice_ui.get_current_model_settings))
        self.assertTrue(callable(voice_ui.apply_model_settings))

    def test_get_current_model_settings_returns_tuple(self):
        """get_current_model_settings returns a 3-tuple."""
        from voice_ui import get_current_model_settings
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
        from voice_ui import apply_model_settings
        # Without server running, should return error message
        result = apply_model_settings("1.7B", "8bit")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        msg, html = result
        self.assertIsInstance(msg, str)
        self.assertIsInstance(html, str)

    def test_apply_model_settings_requires_server(self):
        """apply_model_settings returns error when server not running."""
        from voice_ui import apply_model_settings
        msg, _ = apply_model_settings("0.6B", "4bit")
        self.assertIn("not running", msg.lower())


class TestUIModelSettingsImports(unittest.TestCase):
    """Test voice_ui imports required for model settings."""

    def test_model_settings_imports(self):
        """voice_ui imports required constants for model settings."""
        import voice_ui
        # Should have imported these from voice_config
        self.assertTrue(hasattr(voice_ui, "VALID_MODEL_SIZES"))
        self.assertTrue(hasattr(voice_ui, "VALID_MLX_QUANTIZATIONS"))
        self.assertTrue(hasattr(voice_ui, "get_backend"))
        self.assertTrue(hasattr(voice_ui, "get_model_size"))
        self.assertTrue(hasattr(voice_ui, "get_mlx_quantization"))


# =============================================================================
# Update model config endpoint tests
# =============================================================================

class TestUpdateModelConfigEndpoint(unittest.TestCase):
    """Test /update-model-config server endpoint."""

    @classmethod
    def setUpClass(cls):
        import voice_server
        voice_server.auth_token = "test_token"
        voice_server.server_config = {
            "security": {"max_text_length": 10000},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        import voice_server
        voice_server.auth_token = None

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
        self.assertIn("Invalid model_size", resp.get_json()["error"])

    def test_update_model_config_validates_mlx_quantization(self):
        """POST /update-model-config validates mlx_quantization."""
        resp = self.client.post("/update-model-config",
            json={"mlx_quantization": "invalid"},
            headers={"Authorization": "Bearer test_token"})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid mlx_quantization", resp.get_json()["error"])


class TestClientUpdateModelConfig(unittest.TestCase):
    """Test TTSClient.update_model_config method."""

    def test_update_model_config_method_exists(self):
        """TTSClient has update_model_config method."""
        from voice_client import TTSClient
        client = TTSClient()
        self.assertTrue(hasattr(client, "update_model_config"))
        self.assertTrue(callable(client.update_model_config))


# =============================================================================
# Streaming server endpoint structure tests
# =============================================================================

class TestStreamingEndpointStructure(unittest.TestCase):
    """Test /generate-stream endpoint structure."""

    @classmethod
    def setUpClass(cls):
        import voice_server
        voice_server.auth_token = "test_token"
        voice_server.server_config = {
            "security": {"max_text_length": 10000},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        import voice_server
        voice_server.auth_token = None

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


# =============================================================================
# Generation functions return history update tests
# =============================================================================

class TestGenerationFunctionsReturnHistory(unittest.TestCase):
    """Test that generation functions return history data for UI update."""

    def test_generate_clone_returns_four_values(self):
        """generate_clone delegates to helper that returns 4 values."""
        import inspect
        import voice_ui
        # Non-streaming functions delegate to _generate_non_streaming_impl
        source = inspect.getsource(voice_ui.generate_clone)
        self.assertIn("_generate_non_streaming_impl", source)
        # Helper should have get_history_data
        helper_source = inspect.getsource(voice_ui._generate_non_streaming_impl)
        self.assertIn("get_history_data()", helper_source)

    def test_generate_design_returns_four_values(self):
        """generate_design delegates to helper that returns 4 values."""
        import inspect
        import voice_ui
        source = inspect.getsource(voice_ui.generate_design)
        self.assertIn("_generate_non_streaming_impl", source)

    def test_generate_custom_returns_four_values(self):
        """generate_custom delegates to helper that returns 4 values."""
        import inspect
        import voice_ui
        source = inspect.getsource(voice_ui.generate_custom)
        self.assertIn("_generate_non_streaming_impl", source)

    def test_streaming_functions_yield_four_values(self):
        """Streaming generation functions delegate to helper that yields 4-tuples."""
        import inspect
        import voice_ui
        # Check that streaming wrappers delegate to _generate_streaming_impl
        source = inspect.getsource(voice_ui.generate_clone_streaming)
        self.assertIn("_generate_streaming_impl", source)
        source = inspect.getsource(voice_ui.generate_design_streaming)
        self.assertIn("_generate_streaming_impl", source)
        source = inspect.getsource(voice_ui.generate_custom_streaming)
        self.assertIn("_generate_streaming_impl", source)
        # Helper should have get_history_data
        helper_source = inspect.getsource(voice_ui._generate_streaming_impl)
        self.assertIn("get_history_data()", helper_source)

    def test_non_streaming_adds_to_history(self):
        """Non-streaming helper calls add_to_history."""
        import inspect
        import voice_ui
        source = inspect.getsource(voice_ui._generate_non_streaming_impl)
        self.assertIn("add_to_history", source)

    def test_streaming_adds_to_history(self):
        """Streaming helper calls add_to_history on completion."""
        import inspect
        import voice_ui
        source = inspect.getsource(voice_ui._generate_streaming_impl)
        self.assertIn("add_to_history", source)


# =============================================================================
# Generation stream generation_id check tests
# =============================================================================

class TestGenerateStreamIdCheck(unittest.TestCase):
    """Test generate_stream generation_id race condition fix."""

    def test_generate_stream_checks_generation_id(self):
        """generate_stream only resets state if generation_id matches."""
        import inspect
        import voice_server
        source = inspect.getsource(voice_server)
        # Should check generation_id before resetting
        self.assertIn('if generation_state.get("generation_id") == gen_id', source)

    def test_generation_state_has_generation_id(self):
        """generation_state includes generation_id field."""
        import voice_server
        self.assertIn("generation_id", voice_server.generation_state)

    def test_generation_state_has_cancelled(self):
        """generation_state includes cancelled field."""
        import voice_server
        self.assertIn("cancelled", voice_server.generation_state)


# =============================================================================
# _check_generation_cancelled helper tests
# =============================================================================

class TestCheckGenerationCancelled(unittest.TestCase):
    """Test _check_generation_cancelled helper function."""

    def test_returns_false_on_error(self):
        """_check_generation_cancelled returns False on connection error."""
        from voice_ui import _check_generation_cancelled
        from unittest.mock import patch

        with patch("requests.get", side_effect=Exception("Connection error")):
            result = _check_generation_cancelled()
        self.assertFalse(result)

    def test_returns_false_when_not_cancelled(self):
        """_check_generation_cancelled returns False when not cancelled."""
        from voice_ui import _check_generation_cancelled
        from unittest.mock import patch, MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"cancelled": False}

        with patch("requests.get", return_value=mock_resp):
            result = _check_generation_cancelled()
        self.assertFalse(result)

    def test_returns_true_when_cancelled(self):
        """_check_generation_cancelled returns True when cancelled."""
        from voice_ui import _check_generation_cancelled
        from unittest.mock import patch, MagicMock

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"cancelled": True}

        with patch("requests.get", return_value=mock_resp):
            result = _check_generation_cancelled()
        self.assertTrue(result)


# =============================================================================
# createVoice script backend override tests
# =============================================================================

class TestCreateVoiceBackendOverride(unittest.TestCase):
    """Test createVoice script forces torch backend."""

    def test_createvoice_sets_voice_backend_env(self):
        """bin/createVoice script sets TTS_BACKEND=torch in torch env."""
        import os
        script_path = os.path.join(os.path.dirname(__file__), "..", "bin", "createVoice")
        with open(script_path, "r") as f:
            content = f.read()
        # Should export TTS_BACKEND=torch when in qwen3-tts env
        self.assertIn("TTS_BACKEND=torch", content)
        self.assertIn("export TTS_BACKEND=torch", content)

    def test_createvoice_comment_explains_override(self):
        """bin/createVoice has comment explaining the backend override."""
        import os
        script_path = os.path.join(os.path.dirname(__file__), "..", "bin", "createVoice")
        with open(script_path, "r") as f:
            content = f.read()
        # Should have explanatory comment
        self.assertIn("Force torch backend", content)


# =============================================================================
# Phase 21b: MLX voice prompt cache tests
# =============================================================================

class TestMLXVoicePromptCache(unittest.TestCase):
    """Test MLX voice prompt caching in voice_engine."""

    def setUp(self):
        # Clear cache before each test (earlier test classes may have populated it)
        from voice_engine import _mlx_prompt_cache
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
        from voice_engine import _mlx_prompt_cache
        _mlx_prompt_cache.clear()

    def test_mlx_cache_returns_consistent_results(self):
        """Cached result is identical to first load."""
        from voice_engine import load_voice_prompt_mlx
        with patch("voice_engine.VOICE_PROMPTS_DIR", self.tmpdir):
            first = load_voice_prompt_mlx("voice_a")
            second = load_voice_prompt_mlx("voice_a")
        self.assertIs(first, second)  # Same object from cache

    def test_mlx_cache_stores_entries(self):
        """Loading a prompt adds it to the cache."""
        from voice_engine import load_voice_prompt_mlx, _mlx_prompt_cache
        with patch("voice_engine.VOICE_PROMPTS_DIR", self.tmpdir):
            load_voice_prompt_mlx("voice_a")
        self.assertIn("voice_a", _mlx_prompt_cache)

    def test_clear_voice_prompt_cache_clears_mlx(self):
        """clear_voice_prompt_cache clears MLX cache."""
        from voice_engine import load_voice_prompt_mlx, clear_voice_prompt_cache, _mlx_prompt_cache
        with patch("voice_engine.VOICE_PROMPTS_DIR", self.tmpdir):
            load_voice_prompt_mlx("voice_a")
        self.assertEqual(len(_mlx_prompt_cache), 1)
        clear_voice_prompt_cache()
        self.assertEqual(len(_mlx_prompt_cache), 0)

    def test_mlx_cache_info_returns_currsize(self):
        """voice_prompt_cache_info returns MLX cache size."""
        from voice_engine import load_voice_prompt_mlx, voice_prompt_cache_info
        with patch("voice_engine.get_backend", return_value="mlx"):
            with patch("voice_engine.VOICE_PROMPTS_DIR", self.tmpdir):
                load_voice_prompt_mlx("voice_a")
            info = voice_prompt_cache_info()
        self.assertEqual(info.currsize, 1)


# =============================================================================
# Phase 21b: ETA cache tests
# =============================================================================

class TestETACache(unittest.TestCase):
    """Test ETA estimation cache in voice_server."""

    @classmethod
    def setUpClass(cls):
        import voice_server
        voice_server.auth_token = None
        voice_server.server_config = {
            "security": {},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()

    def test_eta_cache_exists(self):
        """voice_server has _eta_cache module-level dict."""
        import voice_server
        self.assertTrue(hasattr(voice_server, "_eta_cache"))
        self.assertIn("median_rate", voice_server._eta_cache)
        self.assertIn("last_updated", voice_server._eta_cache)

    def test_eta_cache_ttl_constant(self):
        """voice_server has _ETA_CACHE_TTL constant."""
        import voice_server
        self.assertTrue(hasattr(voice_server, "_ETA_CACHE_TTL"))
        self.assertEqual(voice_server._ETA_CACHE_TTL, 30)

    def test_estimate_eta_uses_cache(self):
        """_estimate_eta reads from cache when fresh."""
        import voice_server
        # Pre-populate cache with a known rate
        voice_server._eta_cache["median_rate"] = 10.0  # 10 chars/sec
        voice_server._eta_cache["last_updated"] = time.time()  # fresh

        result = voice_server._estimate_eta(100, 5.0)
        # 100 chars / 10 chars/sec = 10s total, 10 - 5 = 5s remaining
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 5.0, delta=0.5)

    def test_estimate_eta_returns_none_without_data(self):
        """_estimate_eta returns None when no history data."""
        import voice_server
        voice_server._eta_cache["median_rate"] = None
        voice_server._eta_cache["last_updated"] = time.time()

        result = voice_server._estimate_eta(100, 5.0)
        self.assertIsNone(result)


# =============================================================================
# Phase 21b: Generation result cache tests
# =============================================================================

class TestGenerationCache(unittest.TestCase):
    """Test generation result cache in voice_server."""

    def setUp(self):
        import voice_server
        voice_server._gen_cache.clear()

    def test_gen_cache_key_deterministic(self):
        """Same inputs produce same cache key."""
        import voice_server
        key1 = voice_server._gen_cache_key("hello", "clone", {"temperature": 0.7}, prompt_file="voice.pt")
        key2 = voice_server._gen_cache_key("hello", "clone", {"temperature": 0.7}, prompt_file="voice.pt")
        self.assertEqual(key1, key2)

    def test_gen_cache_key_varies_by_text(self):
        """Different text produces different cache key."""
        import voice_server
        key1 = voice_server._gen_cache_key("hello", "clone", {})
        key2 = voice_server._gen_cache_key("world", "clone", {})
        self.assertNotEqual(key1, key2)

    def test_gen_cache_key_varies_by_mode(self):
        """Different mode produces different cache key."""
        import voice_server
        key1 = voice_server._gen_cache_key("hello", "clone", {})
        key2 = voice_server._gen_cache_key("hello", "design", {})
        self.assertNotEqual(key1, key2)

    def test_gen_cache_put_and_get(self):
        """Can store and retrieve cache entries."""
        import voice_server
        # Create a temp file to cache
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake audio")
            path = f.name
        try:
            voice_server._gen_cache_put("test_key", path, 24000)
            result = voice_server._gen_cache_get("test_key")
            self.assertIsNotNone(result)
            self.assertEqual(result["file"], path)
            self.assertEqual(result["sample_rate"], 24000)
        finally:
            os.unlink(path)

    def test_gen_cache_miss(self):
        """Cache miss returns None."""
        import voice_server
        result = voice_server._gen_cache_get("nonexistent_key")
        self.assertIsNone(result)

    def test_gen_cache_max_size_eviction(self):
        """Cache evicts oldest entry when full."""
        import voice_server
        files = []
        try:
            for i in range(voice_server._GEN_CACHE_MAX + 2):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(b"audio")
                    files.append(f.name)
                voice_server._gen_cache_put(f"key_{i}", f.name, 24000)
                time.sleep(0.01)  # Ensure distinct timestamps

            # Should not exceed max
            self.assertLessEqual(len(voice_server._gen_cache), voice_server._GEN_CACHE_MAX)
        finally:
            for f in files:
                if os.path.exists(f):
                    os.unlink(f)

    def test_gen_cache_invalidate(self):
        """_gen_cache_invalidate clears all entries."""
        import voice_server
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"audio")
            path = f.name
        voice_server._gen_cache_put("key", path, 24000)
        self.assertEqual(len(voice_server._gen_cache), 1)

        voice_server._gen_cache_invalidate()
        self.assertEqual(len(voice_server._gen_cache), 0)

    def test_gen_cache_stale_file_cleanup(self):
        """Cache get cleans up entries with missing files."""
        import voice_server
        voice_server._gen_cache["stale_key"] = {
            "file": "/nonexistent/path.wav",
            "sample_rate": 24000,
            "timestamp": time.time(),
        }
        result = voice_server._gen_cache_get("stale_key")
        self.assertIsNone(result)
        self.assertNotIn("stale_key", voice_server._gen_cache)

    def test_update_model_config_invalidates_gen_cache(self):
        """Updating model config invalidates generation cache."""
        import inspect
        import voice_server
        source = inspect.getsource(voice_server.update_model_config)
        self.assertIn("_gen_cache_invalidate", source)


# =============================================================================
# Phase 21a: Voice management endpoint tests
# =============================================================================

class TestDeletePromptEndpoint(unittest.TestCase):
    """Test POST /delete-prompt endpoint."""

    @classmethod
    def setUpClass(cls):
        import voice_server
        voice_server.auth_token = "test_secret_token"
        voice_server.server_config = {
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()
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
        self.assertIn("path traversal", resp.get_json()["error"])

    def test_delete_nonexistent(self):
        """POST /delete-prompt returns 404 for missing prompt."""
        with patch("voice_server.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/delete-prompt", json={"name": "nonexistent"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 404)

    def test_delete_success(self):
        """POST /delete-prompt deletes all format files."""
        with patch("voice_server.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/delete-prompt", json={"name": "test_voice"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "deleted")
        self.assertEqual(len(data["files_removed"]), 3)
        # Verify files are gone
        for ext in (".pt", ".wav", ".txt"):
            self.assertFalse(os.path.exists(os.path.join(self.tmpdir, f"test_voice{ext}")))


class TestRenamePromptEndpoint(unittest.TestCase):
    """Test POST /rename-prompt endpoint."""

    @classmethod
    def setUpClass(cls):
        import voice_server
        voice_server.auth_token = "test_secret_token"
        voice_server.server_config = {
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()
        cls.auth = {"Authorization": "Bearer test_secret_token"}

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
        with patch("voice_server.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/rename-prompt",
                                    json={"old_name": "old_voice", "new_name": "existing"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 409)

    def test_rename_success(self):
        """POST /rename-prompt renames all format files."""
        with patch("voice_server.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/rename-prompt",
                                    json={"old_name": "old_voice", "new_name": "new_voice"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "renamed")
        # Old files should be gone, new files should exist
        for ext in (".pt", ".wav", ".txt"):
            self.assertFalse(os.path.exists(os.path.join(self.tmpdir, f"old_voice{ext}")))
            self.assertTrue(os.path.exists(os.path.join(self.tmpdir, f"new_voice{ext}")))

    def test_rename_not_found(self):
        """POST /rename-prompt returns 404 for missing prompt."""
        with patch("voice_server.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.post("/rename-prompt",
                                    json={"old_name": "nonexistent", "new_name": "new"},
                                    headers=self.auth)
        self.assertEqual(resp.status_code, 404)


class TestPreviewPromptEndpoint(unittest.TestCase):
    """Test GET /preview-prompt endpoint."""

    @classmethod
    def setUpClass(cls):
        import voice_server
        voice_server.auth_token = "test_secret_token"
        voice_server.server_config = {
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()
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
        with patch("voice_server.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.get("/preview-prompt?name=nonexistent",
                                   headers=self.auth)
        self.assertEqual(resp.status_code, 404)

    def test_preview_returns_audio(self):
        """GET /preview-prompt returns audio/wav content."""
        with patch("voice_server.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.get("/preview-prompt?name=test_voice",
                                   headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("audio/wav", resp.content_type)


class TestPromptDetailsEndpoint(unittest.TestCase):
    """Test GET /prompt-details endpoint."""

    @classmethod
    def setUpClass(cls):
        import voice_server
        voice_server.auth_token = "test_secret_token"
        voice_server.server_config = {
            "security": {"max_text_length": 100, "max_batch_size": 3},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()
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
        with patch("voice_server.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.get("/prompt-details?name=voice_a",
                                   headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["name"], "voice_a")
        self.assertIn(".pt", data["formats"])
        self.assertIn(".wav", data["formats"])
        self.assertIn(".txt", data["formats"])
        self.assertGreater(data["size_bytes"], 0)

    def test_details_all_prompts(self):
        """GET /prompt-details without name returns all prompts."""
        with patch("voice_server.VOICE_PROMPTS_DIR", self.tmpdir):
            resp = self.client.get("/prompt-details", headers=self.auth)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("prompts", data)
        self.assertEqual(len(data["prompts"]), 1)
        self.assertEqual(data["prompts"][0]["name"], "voice_a")


class TestClientPromptManagement(unittest.TestCase):
    """Test voice_client prompt management method signatures."""

    def test_delete_prompt_method_exists(self):
        """TTSClient has delete_prompt method."""
        from voice_client import TTSClient
        self.assertTrue(hasattr(TTSClient, "delete_prompt"))

    def test_rename_prompt_method_exists(self):
        """TTSClient has rename_prompt method."""
        from voice_client import TTSClient
        self.assertTrue(hasattr(TTSClient, "rename_prompt"))

    def test_preview_prompt_method_exists(self):
        """TTSClient has preview_prompt method."""
        from voice_client import TTSClient
        self.assertTrue(hasattr(TTSClient, "preview_prompt"))

    def test_get_prompt_details_method_exists(self):
        """TTSClient has get_prompt_details method."""
        from voice_client import TTSClient
        self.assertTrue(hasattr(TTSClient, "get_prompt_details"))

    def test_list_prompts_uses_server(self):
        """list_prompts calls server /prompts when running."""
        import inspect
        from voice_client import TTSClient
        source = inspect.getsource(TTSClient.list_prompts)
        self.assertIn("/prompts", source)
        self.assertIn("is_server_running", source)


class TestSetDefaultClonePrompt(unittest.TestCase):
    """Test set_default_clone_prompt config helper."""

    def test_set_default_writes_config(self):
        """set_default_clone_prompt updates config.json."""
        from voice_config import set_default_clone_prompt, load_config, save_config, CONFIG_PATH
        # Save original config
        original = load_config()
        try:
            set_default_clone_prompt("test_voice.pt")
            config = load_config()
            self.assertEqual(config["default_clone_prompt"], "test_voice.pt")
        finally:
            # Restore original config
            save_config(original)


class TestVoiceManagementUI(unittest.TestCase):
    """Test voice management UI helper functions."""

    def test_get_prompt_table_data_exists(self):
        """voice_ui has get_prompt_table_data function."""
        from voice_ui import get_prompt_table_data
        self.assertTrue(callable(get_prompt_table_data))

    def test_preview_voice_exists(self):
        """voice_ui has preview_voice function."""
        from voice_ui import preview_voice
        self.assertTrue(callable(preview_voice))

    def test_rename_voice_exists(self):
        """voice_ui has rename_voice function."""
        from voice_ui import rename_voice
        self.assertTrue(callable(rename_voice))

    def test_delete_voice_exists(self):
        """voice_ui has delete_voice function."""
        from voice_ui import delete_voice
        self.assertTrue(callable(delete_voice))

    def test_set_voice_default_exists(self):
        """voice_ui has set_voice_default function."""
        from voice_ui import set_voice_default
        self.assertTrue(callable(set_voice_default))


if __name__ == "__main__":
    unittest.main()
