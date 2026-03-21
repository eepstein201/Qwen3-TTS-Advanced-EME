#!/usr/bin/env python3
"""Config loading, validation, and accessor function tests.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_config.py -v

No GPU, models, or running server required.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Dummy decorator for when pytest is not available
    class _DummyMarkerFunc:
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition, **kwargs):
            return lambda f: f
    class _DummyMarker:
        def __call__(self, func):
            return func
        def __getattr__(self, name):
            return _DummyMarkerFunc()
        @property
        def unit(self):
            return self
    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()
    class _DummyPytest:
        mark = _DummyMark()
    pytest = _DummyPytest()


# =========================================================================
# Config Edge Cases
# =========================================================================

@pytest.mark.unit
class TestConfigEdgeCases(unittest.TestCase):
    """Test config constants and accessor functions."""

    def test_get_backend_returns_string(self):
        """get_backend() returns 'torch' or 'mlx'."""
        from qwen3_tts.core.config import get_backend
        result = get_backend()
        self.assertIn(result, ("torch", "mlx"))

    def test_get_model_size_default(self):
        """get_model_size() returns '1.7B' or '0.6B'."""
        from qwen3_tts.core.config import get_model_size
        result = get_model_size()
        self.assertIn(result, ("1.7B", "0.6B"))

    def test_model_info_keys(self):
        """MODEL_INFO has size-based keys."""
        from qwen3_tts.core.config import MODEL_INFO
        self.assertIn("1.7B", MODEL_INFO)
        self.assertIn("0.6B", MODEL_INFO)

    def test_custom_speakers_have_fields(self):
        """Each entry in CUSTOM_VOICE_SPEAKERS has a 'name' key."""
        from qwen3_tts.core.config import CUSTOM_VOICE_SPEAKERS
        self.assertGreater(len(CUSTOM_VOICE_SPEAKERS), 0)
        for key, entry in CUSTOM_VOICE_SPEAKERS.items():
            self.assertIn("name", entry, f"Speaker '{key}' missing 'name' field")

    def test_prosody_presets_all_strings(self):
        """All values in DEFAULT_PROSODY_PRESETS are non-empty strings."""
        from qwen3_tts.core.config import DEFAULT_PROSODY_PRESETS
        self.assertGreater(len(DEFAULT_PROSODY_PRESETS), 0)
        for key, value in DEFAULT_PROSODY_PRESETS.items():
            self.assertIsInstance(value, str, f"Preset '{key}' is not a string")
            self.assertTrue(len(value) > 0, f"Preset '{key}' is empty")

    def test_valid_backends_constant(self):
        """VALID_BACKENDS contains 'torch' and 'mlx'."""
        from qwen3_tts.core.config import VALID_BACKENDS
        self.assertIn("torch", VALID_BACKENDS)
        self.assertIn("mlx", VALID_BACKENDS)


@pytest.mark.unit
class TestConfigValidation(unittest.TestCase):
    """Test validate_config() catches bad values."""

    def test_valid_config_no_issues(self):
        """A well-formed config produces no validation issues."""
        from qwen3_tts.core.config import validate_config
        config = {
            "advanced": {"backend": "mlx", "model_size": "1.7B"},
            "generation": {"temperature": 0.7},
            "security": {"max_text_length": 10000},
        }
        _, issues = validate_config(config)
        self.assertEqual(issues, [])

    def test_invalid_backend(self):
        from qwen3_tts.core.config import validate_config
        _, issues = validate_config({"advanced": {"backend": "invalid"}})
        self.assertTrue(any("backend" in i for i in issues))

    def test_invalid_model_size(self):
        from qwen3_tts.core.config import validate_config
        _, issues = validate_config({"advanced": {"model_size": "99B"}})
        self.assertTrue(any("model_size" in i for i in issues))

    def test_temperature_out_of_range(self):
        from qwen3_tts.core.config import validate_config
        _, issues = validate_config({"generation": {"temperature": 5.0}})
        self.assertTrue(any("temperature" in i for i in issues))

    def test_negative_max_text_length(self):
        from qwen3_tts.core.config import validate_config
        _, issues = validate_config({"security": {"max_text_length": -1}})
        self.assertTrue(any("max_text_length" in i for i in issues))

    def test_empty_config_no_issues(self):
        from qwen3_tts.core.config import validate_config
        _, issues = validate_config({})
        self.assertEqual(issues, [])


# =========================================================================
# Config Function Tests
# =========================================================================

@pytest.mark.unit
class TestConfigFunctions(unittest.TestCase):
    """Tests for config.py utility functions."""

    def test_save_config_roundtrip(self):
        """save_config writes JSON that load_config can read back."""
        from qwen3_tts.core import config as cfg

        original_path = cfg.CONFIG_PATH
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({"test_key": "test_value"}, f)
                tmp_path = f.name
            cfg.CONFIG_PATH = tmp_path
            cfg._config_cache["data"] = None
            cfg._config_cache["mtime"] = 0

            test_config = {"generation": {"temperature": 0.5}, "test": True}
            cfg.save_config(test_config)

            with open(tmp_path) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["generation"]["temperature"], 0.5)
            self.assertTrue(loaded["test"])
        finally:
            cfg.CONFIG_PATH = original_path
            cfg._config_cache["data"] = None
            cfg._config_cache["mtime"] = 0
            os.unlink(tmp_path)

    def test_get_device_returns_valid_string(self):
        """get_device returns one of 'cuda', 'mps', or 'cpu'."""
        from qwen3_tts.core.config import get_device
        result = get_device()
        self.assertIn(result, ("cuda", "mps", "cpu"))

    def test_get_server_url_returns_url(self):
        """get_server_url returns a proper URL string."""
        from qwen3_tts.core.config import get_server_url
        config = {"server": {"host": "127.0.0.1", "port": 5123}}
        result = get_server_url(config)
        self.assertEqual(result, "http://127.0.0.1:5123")

    def test_get_server_url_defaults(self):
        """get_server_url uses defaults when config is empty."""
        from qwen3_tts.core.config import get_server_url
        result = get_server_url({})
        self.assertEqual(result, "http://127.0.0.1:5123")

    def test_get_torch_dtype_name_returns_valid(self):
        """get_torch_dtype_name returns a valid dtype string."""
        from qwen3_tts.core.config import get_torch_dtype_name
        result = get_torch_dtype_name()
        self.assertIn(result, ("float32", "float16", "bfloat16"))

    def test_get_mlx_quantization_returns_valid(self):
        """get_mlx_quantization returns a valid quantization string."""
        from qwen3_tts.core.config import get_mlx_quantization
        result = get_mlx_quantization()
        self.assertIn(result, ("4bit", "8bit", "bf16"))

    def test_get_torch_model_name_returns_hf_id(self):
        """get_torch_model_name returns a HuggingFace model ID."""
        from qwen3_tts.core.config import get_torch_model_name
        result = get_torch_model_name("clone")
        self.assertIn("Qwen", result)
        self.assertIn("Base", result)

    def test_get_mlx_model_name_returns_hf_id(self):
        """get_mlx_model_name returns a HuggingFace model ID."""
        from qwen3_tts.core.config import get_mlx_model_name
        result = get_mlx_model_name("clone")
        self.assertIn("mlx-community", result)
        self.assertIn("Base", result)

    def test_get_prosody_presets_returns_dict(self):
        """get_prosody_presets returns a dict with expected keys."""
        from qwen3_tts.core.config import get_prosody_presets
        result = get_prosody_presets()
        self.assertIsInstance(result, dict)
        self.assertIn("excited", result)
        self.assertIn("calm", result)
        self.assertIn("whisper", result)

    def test_get_optimal_attn_config_no_cuda(self):
        """get_optimal_attn_config returns sdpa/float32 when no CUDA."""
        from qwen3_tts.core.config import get_optimal_attn_config
        with patch("qwen3_tts.core.config.get_cuda_capability", return_value=None):
            attn, dtype, load_8bit = get_optimal_attn_config()
            self.assertEqual(attn, "sdpa")
            self.assertEqual(dtype, "float32")
            self.assertFalse(load_8bit)

    def test_get_optimal_attn_config_ampere(self):
        """get_optimal_attn_config returns flash_attention_2 for Ampere+ GPU with flash_attn installed."""
        from qwen3_tts.core.config import get_optimal_attn_config
        with patch("qwen3_tts.core.config.get_cuda_capability", return_value=(8, 0)):
            with patch("qwen3_tts.core.config._has_flash_attn", return_value=True):
                attn, dtype, load_8bit = get_optimal_attn_config()
                self.assertEqual(attn, "flash_attention_2")
                self.assertEqual(dtype, "bfloat16")
                self.assertFalse(load_8bit)

    def test_get_optimal_attn_config_ampere_no_flash_attn(self):
        """get_optimal_attn_config falls back to sdpa when flash_attn not installed."""
        from qwen3_tts.core.config import get_optimal_attn_config
        with patch("qwen3_tts.core.config.get_cuda_capability", return_value=(8, 9)):
            with patch("qwen3_tts.core.config._has_flash_attn", return_value=False):
                attn, dtype, load_8bit = get_optimal_attn_config()
                self.assertEqual(attn, "sdpa")
                self.assertEqual(dtype, "bfloat16")
                self.assertFalse(load_8bit)

    def test_get_optimal_attn_config_turing(self):
        """get_optimal_attn_config returns sdpa/float16/8bit for Turing GPU."""
        from qwen3_tts.core.config import get_optimal_attn_config
        with patch("qwen3_tts.core.config.get_cuda_capability", return_value=(7, 5)):
            attn, dtype, load_8bit = get_optimal_attn_config()
            self.assertEqual(attn, "sdpa")
            self.assertEqual(dtype, "float16")
            self.assertTrue(load_8bit)


@pytest.mark.unit
class TestHFConsolidatedConstant(unittest.TestCase):
    """Test that HF_CACHE is defined once in config.py and imported by all users."""

    def test_hf_cache_single_source(self):
        """HF_CACHE is imported from config in all tool modules."""
        import pathlib
        from qwen3_tts.core.config import HF_CACHE as config_hf_cache

        # Expected value
        expected = pathlib.Path.home() / ".cache" / "huggingface" / "hub"

        # Config should have the correct value
        self.assertEqual(config_hf_cache, expected)

        # All tool modules should import from config
        from qwen3_tts.tools.model_cache import HF_CACHE as model_cache_hf_cache
        from qwen3_tts.tools.uninstall import HF_CACHE as uninstall_hf_cache
        from qwen3_tts.tools.healthcheck import HF_CACHE as healthcheck_hf_cache

        # All should be the same object (single source of truth)
        self.assertIs(model_cache_hf_cache, config_hf_cache,
                      "model_cache.HF_CACHE should be imported from config")
        self.assertIs(uninstall_hf_cache, config_hf_cache,
                      "uninstall.HF_CACHE should be imported from config")
        self.assertIs(healthcheck_hf_cache, config_hf_cache,
                      "healthcheck.HF_CACHE should be imported from config")


if __name__ == "__main__":
    unittest.main()
