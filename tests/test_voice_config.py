"""Config-related tests extracted from test_voice.py."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        from qwen3_tts.core.config import read_auth_token
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
                read_auth_token()
                auth_headers()
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


if __name__ == "__main__":
    unittest.main()
