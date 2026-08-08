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
            "security": {
                "max_text_length": 10000,
                "rate_limits": {
                    "generate": "20/minute",
                    "model_ops": "3/minute",
                    "transcribe": "15/minute",
                    "prompt_ops": "10/minute",
                    "config_ops": "1/minute",
                },
            },
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

    def test_validate_server_url_accepts_localhost_variants(self):
        """_validate_server_url accepts all localhost variants."""
        from qwen3_tts.core.config import _validate_server_url
        for host in ("127.0.0.1", "localhost"):
            url = f"http://{host}:5123"
            result = _validate_server_url(url)
            self.assertEqual(result, url)
        # IPv6 requires brackets in URL syntax
        ipv6_url = "http://[::1]:5123"
        result = _validate_server_url(ipv6_url)
        self.assertEqual(result, ipv6_url)

    def test_validate_server_url_rejects_bind_all(self):
        """_validate_server_url rejects 0.0.0.0 (bind-all, not a valid client target)."""
        from qwen3_tts.core.config import _validate_server_url
        with self.assertRaises(ValueError):
            _validate_server_url("http://0.0.0.0:5123")

    def test_validate_server_url_rejects_external_hosts(self):
        """_validate_server_url rejects non-localhost hosts."""
        from qwen3_tts.core.config import _validate_server_url
        for host in ("evil.com", "169.254.169.254", "10.0.0.1",
                      "metadata.google.internal", "192.168.1.1"):
            with self.assertRaises(ValueError, msg=f"Should reject {host}"):
                _validate_server_url(f"http://{host}:5123")

    def test_validate_server_url_rejects_bad_scheme(self):
        """_validate_server_url rejects non-http(s) schemes."""
        from qwen3_tts.core.config import _validate_server_url
        with self.assertRaises(ValueError):
            _validate_server_url("ftp://localhost:5123")

    def test_validate_server_url_rejects_invalid_port(self):
        """_validate_server_url rejects ports outside 1-65535."""
        from qwen3_tts.core.config import _validate_server_url
        with self.assertRaises(ValueError):
            _validate_server_url("http://localhost:0")
        with self.assertRaises(ValueError):
            _validate_server_url("http://localhost:99999")

    def test_validate_server_url_accepts_https(self):
        """_validate_server_url accepts https scheme."""
        from qwen3_tts.core.config import _validate_server_url
        result = _validate_server_url("https://localhost:5123")
        self.assertEqual(result, "https://localhost:5123")

    def test_get_server_url_rejects_external_host_config(self):
        """get_server_url raises ValueError for non-localhost host in config."""
        from qwen3_tts.core.config import get_server_url
        config = {"server": {"host": "evil.com", "port": 5123}}
        with self.assertRaises(ValueError):
            get_server_url(config)

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

    def test_get_optimal_attn_config_ampere_defaults_to_sdpa(self):
        """PRF-4: Ampere+ defaults to sdpa even with flash_attn installed (#333 NaN risk)."""
        from qwen3_tts.core.config import get_optimal_attn_config
        with patch("qwen3_tts.core.config.get_cuda_capability", return_value=(8, 0)):
            with patch("qwen3_tts.core.config._has_flash_attn", return_value=True):
                attn, dtype, load_8bit = get_optimal_attn_config()
                self.assertEqual(attn, "sdpa")
                self.assertEqual(dtype, "bfloat16")
                self.assertFalse(load_8bit)

    def test_get_optimal_attn_config_ampere_fa2_optin(self):
        """PRF-4: flash_attention_2 is honoured only when explicitly requested."""
        from qwen3_tts.core.config import get_optimal_attn_config
        with patch("qwen3_tts.core.config.get_cuda_capability", return_value=(8, 0)):
            with patch("qwen3_tts.core.config._has_flash_attn", return_value=True):
                attn, _, _ = get_optimal_attn_config("flash_attention_2")
                self.assertEqual(attn, "flash_attention_2")

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

    def test_load_config_corrupt_json_raises_valueerror(self):
        """load_config must raise ValueError with clear message on corrupt JSON (R-42)."""
        import pathlib

        from qwen3_tts.core import config as cfg
        original_path = cfg.CONFIG_PATH
        original_cache = dict(cfg._config_cache)
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                f.write("{not valid json")
                bad_path = f.name
            cfg.CONFIG_PATH = pathlib.Path(bad_path)
            cfg._config_cache["data"] = None
            cfg._config_cache["mtime"] = 0
            with pytest.raises(ValueError, match="corrupt or invalid JSON"):
                cfg.load_config()
        finally:
            cfg.CONFIG_PATH = original_path
            cfg._config_cache.update(original_cache)
            os.unlink(bad_path)


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
        from qwen3_tts.tools.healthcheck import HF_CACHE as healthcheck_hf_cache
        from qwen3_tts.tools.model_cache import HF_CACHE as model_cache_hf_cache
        from qwen3_tts.tools.uninstall import HF_CACHE as uninstall_hf_cache

        # All should be the same object (single source of truth)
        self.assertIs(model_cache_hf_cache, config_hf_cache,
                      "model_cache.HF_CACHE should be imported from config")
        self.assertIs(uninstall_hf_cache, config_hf_cache,
                      "uninstall.HF_CACHE should be imported from config")
        self.assertIs(healthcheck_hf_cache, config_hf_cache,
                      "healthcheck.HF_CACHE should be imported from config")


@pytest.mark.unit
class TestSanitizeLog(unittest.TestCase):
    """Tests for sanitize_log utility."""

    def test_strips_newlines(self):
        from qwen3_tts.core.config import sanitize_log
        self.assertEqual(sanitize_log("hello\nworld"), "hello\\nworld")

    def test_strips_carriage_return(self):
        from qwen3_tts.core.config import sanitize_log
        self.assertEqual(sanitize_log("hello\rworld"), "hello\\rworld")

    def test_strips_null_bytes(self):
        from qwen3_tts.core.config import sanitize_log
        self.assertEqual(sanitize_log("hello\x00world"), "helloworld")

    def test_handles_non_string(self):
        from qwen3_tts.core.config import sanitize_log
        self.assertEqual(sanitize_log(42), "42")
        self.assertEqual(sanitize_log(None), "None")

    def test_passes_clean_string(self):
        from qwen3_tts.core.config import sanitize_log
        self.assertEqual(sanitize_log("clean_string"), "clean_string")

    def test_combined_control_chars(self):
        from qwen3_tts.core.config import sanitize_log
        self.assertEqual(sanitize_log("a\nb\rc\x00d"), "a\\nb\\rcd")


class TestSafePathJoin(unittest.TestCase):
    """Tests for safe_path_join utility."""

    def test_valid_filename(self):
        from qwen3_tts.core.config import safe_path_join
        result = safe_path_join("/tmp/base", "file.txt")
        self.assertEqual(result, os.path.realpath("/tmp/base/file.txt"))

    def test_valid_nested_path(self):
        from qwen3_tts.core.config import safe_path_join
        result = safe_path_join("/tmp/base", "sub", "file.txt")
        self.assertEqual(result, os.path.realpath("/tmp/base/sub/file.txt"))

    def test_rejects_parent_traversal(self):
        from qwen3_tts.core.config import safe_path_join
        with self.assertRaises(ValueError):
            safe_path_join("/tmp/base", "../etc/passwd")

    def test_rejects_absolute_override(self):
        from qwen3_tts.core.config import safe_path_join
        with self.assertRaises(ValueError):
            safe_path_join("/tmp/base", "/etc/passwd")

    def test_rejects_deep_traversal(self):
        from qwen3_tts.core.config import safe_path_join
        # Double-dot in subdir that resolves outside
        with self.assertRaises(ValueError):
            safe_path_join("/tmp/base", "subdir/../../outside")
        # Multiple levels
        with self.assertRaises(ValueError):
            safe_path_join("/tmp/base", "a/b/../../../etc")

    def test_accepts_base_dir_itself(self):
        from qwen3_tts.core.config import safe_path_join
        result = safe_path_join("/tmp/base", "")
        self.assertEqual(result, os.path.realpath("/tmp/base"))

    def test_accepts_dot_current_dir(self):
        from qwen3_tts.core.config import safe_path_join
        result = safe_path_join("/tmp/base", ".", "file.txt")
        self.assertEqual(result, os.path.realpath("/tmp/base/file.txt"))


# =========================================================================
# Shipped prompt references (regression guard — repo-audit-2026-07-31 P0-1)
# =========================================================================

@pytest.mark.unit
class TestDefaultConfigPromptReferences(unittest.TestCase):
    """Every prompt named by get_default_config() must resolve on a fresh install.

    No voice prompt ships with the package, so any filename seeded into
    get_default_config() is a dangling reference on a new machine. The two
    seeds are NOT equally safe, and that asymmetry is the whole finding:

    - ``default_clone_prompt`` degrades gracefully. get_default_clone_prompt()
      checks the configured file and falls through to a backend-aware scan when
      it is missing, so a dangling value is harmless.
    - ``aliases[*]["prompt"]`` does **not**. interface/generate.py resolves
      ``alias_prompt or get_default_clone_prompt(config)``, and a truthy
      alias_prompt short-circuits the fallback entirely — a dangling alias
      raises FileNotFoundError.

    Historically the shipped ``default`` alias pointed at ``default_clone.pt``,
    which existed nowhere, so the one alias ``tts list aliases`` advertised was
    the one that failed on every fresh install (fixed in b98501a; both seeds are
    now empty). These tests exist so re-seeding a dangling alias fails loudly
    here instead of silently in a new user's first command.
    """

    def _prompt_exists(self, name):
        """True if *name* is usable as a prompt, in either backend's format.

        Accepts all three spellings a prompt name appears in: a bare base, a
        torch ``.pt``, or an MLX ``.wav``. get_default_clone_prompt() strips only
        ``.pt`` in its existence check while its MLX fallback *returns* a
        ``.wav`` filename, so a stricter helper here would reject a name the
        production code just handed back as valid.
        """
        from qwen3_tts.core.config import VOICE_PROMPTS_DIR, safe_path_join

        base = name
        for suffix in (".pt", ".wav"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        prompts_dir = str(VOICE_PROMPTS_DIR)
        pt_exists = os.path.exists(safe_path_join(prompts_dir, f"{base}.pt"))
        mlx_exists = os.path.exists(
            safe_path_join(prompts_dir, f"{base}.wav")
        ) and os.path.exists(safe_path_join(prompts_dir, f"{base}.txt"))
        return pt_exists or mlx_exists

    def test_seeded_aliases_reference_resolvable_prompts(self):
        """A seeded alias prompt must exist — the alias path has no fallback."""
        from qwen3_tts.core.config import get_default_config

        dangling = [
            (alias_name, spec["prompt"])
            for alias_name, spec in get_default_config().get("aliases", {}).items()
            if isinstance(spec, dict) and spec.get("prompt")
            and not self._prompt_exists(spec["prompt"])
        ]
        self.assertEqual(
            dangling,
            [],
            "get_default_config() seeds an alias whose prompt does not exist. "
            "interface/generate.py does `alias_prompt or get_default_clone_prompt"
            "(config)`, so a truthy alias prompt short-circuits the missing-prompt "
            "fallback and `tts --alias <name>` raises FileNotFoundError on a fresh "
            f"install. Ship no alias, or ship the prompt file: {dangling}",
        )

    def test_seeded_default_clone_prompt_resolves_or_falls_back(self):
        """A seeded default_clone_prompt must exist or leave the fallback intact."""
        from qwen3_tts.core.config import get_default_config

        configured = get_default_config().get("default_clone_prompt")
        if not configured:
            self.skipTest("no default_clone_prompt seeded — fallback path applies")
        self.assertTrue(
            self._prompt_exists(configured),
            f"get_default_config() seeds default_clone_prompt={configured!r}, "
            "which does not exist. This one degrades safely today, but seeding a "
            "name that never resolves is misleading — prefer None so the "
            "backend-aware scan is the single source of truth.",
        )

    def test_default_clone_prompt_resolution_never_raises(self):
        """Resolving the shipped config must not raise, whatever is on disk."""
        from qwen3_tts.core.config import get_default_clone_prompt, get_default_config

        resolved = get_default_clone_prompt(get_default_config())
        # None (no prompts installed) is a valid outcome; a name is not required.
        if resolved is not None:
            self.assertIsInstance(resolved, str)
            self.assertTrue(
                self._prompt_exists(resolved),
                f"fallback returned {resolved!r}, which does not exist on disk",
            )

    def test_pt_prompt_is_not_seeded_for_mlx_default(self):
        """A .pt seed is torch-only and wrong for the default Apple-Silicon path."""
        from qwen3_tts.core.config import get_default_config

        config = get_default_config()
        pt_seeds = [
            f"default_clone_prompt={value}"
            for value in [config.get("default_clone_prompt")]
            if isinstance(value, str) and value.endswith(".pt")
        ] + [
            f"aliases.{name}.prompt={spec['prompt']}"
            for name, spec in config.get("aliases", {}).items()
            if isinstance(spec, dict)
            and isinstance(spec.get("prompt"), str)
            and spec["prompt"].endswith(".pt")
        ]
        self.assertEqual(
            pt_seeds,
            [],
            "a .pt prompt is torch-only; MLX needs a .wav + .txt pair, and MLX is "
            f"the default backend on Apple Silicon: {pt_seeds}",
        )


if __name__ == "__main__":
    unittest.main()
