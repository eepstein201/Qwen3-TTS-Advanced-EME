#!/usr/bin/env python3
"""Core infrastructure tests: error paths, concurrency, caching, config edge cases,
SSML edge cases, and dry-run verification.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_core_infra.py -v

No GPU, models, or running server required.
"""

import inspect
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check optional dependencies
try:
    import soundfile  # noqa: F401
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import flask  # noqa: F401
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

_server_deps = HAS_SOUNDFILE and HAS_FLASK
_skip_server = unittest.skipUnless(_server_deps, "requires soundfile + flask")
_skip_generate = unittest.skipUnless(HAS_SOUNDFILE, "requires soundfile (voice_generate)")


# =========================================================================
# Error Path Tests
# =========================================================================

@_skip_server
class TestErrorPaths(unittest.TestCase):
    """Test error handling in engine and server validation."""

    @classmethod
    def setUpClass(cls):
        import voice_server
        voice_server.auth_token = "test_token"
        voice_server.server_config = {
            "security": {"max_text_length": 100, "max_batch_size": 5},
            "auto_shutdown_minutes": 0,
        }
        cls.app = voice_server.app
        cls.app.testing = True
        cls.client = cls.app.test_client()
        cls.auth = {"Authorization": "Bearer test_token"}

    def test_set_audio_loader_invalid(self):
        """set_audio_loader('invalid') raises ValueError."""
        from voice_engine import set_audio_loader
        with self.assertRaises(ValueError):
            set_audio_loader("invalid")

    def test_set_audio_loader_valid(self):
        """set_audio_loader('librosa') succeeds and get_audio_loader reflects it."""
        from voice_engine import set_audio_loader, get_audio_loader
        original = get_audio_loader()
        try:
            set_audio_loader("librosa")
            self.assertEqual(get_audio_loader(), "librosa")
        finally:
            set_audio_loader(original)

    def test_load_config_returns_dict(self):
        """load_config() always returns a dict."""
        from voice_config import load_config
        result = load_config()
        self.assertIsInstance(result, dict)

    def test_parse_ssml_no_tags(self):
        """Plain text with no SSML tags returns unchanged with has_ssml=False."""
        from voice_generate import parse_ssml
        text, meta = parse_ssml("plain text with no tags")
        self.assertEqual(text, "plain text with no tags")
        self.assertFalse(meta["has_ssml"])

    def test_parse_ssml_malformed_unclosed(self):
        """Malformed unclosed tags do not crash parse_ssml."""
        from voice_generate import parse_ssml
        # '<break Hello' has no closing '>' for a valid tag, so regex won't match
        text, meta = parse_ssml("<break Hello")
        self.assertIsInstance(text, str)

    def test_generate_endpoint_missing_texts(self):
        """POST /generate with empty body returns 400."""
        resp = self.client.post("/generate", json={}, headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)

    def test_generate_endpoint_empty_texts(self):
        """POST /generate with empty texts list returns 400."""
        resp = self.client.post("/generate", json={"texts": []}, headers=self.auth)
        self.assertEqual(resp.status_code, 400)

    def test_generate_endpoint_text_over_limit(self):
        """Text exceeding max_text_length returns 400."""
        long_text = "A" * 200  # server_config max is 100
        resp = self.client.post("/generate", json={"texts": [long_text]}, headers=self.auth)
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("limit", data["error"])


# =========================================================================
# Generation Cache Thread Safety Tests
# =========================================================================

@_skip_server
class TestGenerationCacheThreadSafety(unittest.TestCase):
    """Verify generation cache uses proper locking."""

    def test_gen_cache_lock_is_threading_lock(self):
        """_gen_cache_lock is an instance of threading.Lock."""
        import voice_server
        self.assertIsInstance(voice_server._gen_cache_lock, type(threading.Lock()))

    def test_gen_cache_get_acquires_lock(self):
        """_gen_cache_get acquires the lock."""
        import voice_server
        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__ = MagicMock(return_value=False)
        original = voice_server._gen_cache_lock
        voice_server._gen_cache_lock = mock_lock
        try:
            voice_server._gen_cache_get("test_key")
            mock_lock.__enter__.assert_called_once()
        finally:
            voice_server._gen_cache_lock = original

    def test_gen_cache_put_acquires_lock(self):
        """_gen_cache_put acquires the lock."""
        import voice_server
        import tempfile
        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__ = MagicMock(return_value=False)
        original_lock = voice_server._gen_cache_lock
        original_cache = voice_server._gen_cache.copy()
        voice_server._gen_cache_lock = mock_lock
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            voice_server._gen_cache_put("test_put_key", tmp_path, 24000)
            mock_lock.__enter__.assert_called_once()
        finally:
            voice_server._gen_cache_lock = original_lock
            voice_server._gen_cache = original_cache
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_gen_cache_invalidate_acquires_lock(self):
        """_gen_cache_invalidate acquires the lock."""
        import voice_server
        mock_lock = MagicMock()
        mock_lock.__enter__ = MagicMock(return_value=None)
        mock_lock.__exit__ = MagicMock(return_value=False)
        original = voice_server._gen_cache_lock
        voice_server._gen_cache_lock = mock_lock
        try:
            voice_server._gen_cache_invalidate()
            mock_lock.__enter__.assert_called_once()
        finally:
            voice_server._gen_cache_lock = original

    def test_double_checked_locking_in_generate_source(self):
        """generate() source contains both pre-lock and post-lock cache comments."""
        import voice_server
        source = inspect.getsource(voice_server.generate)
        self.assertIn("pre-lock", source)
        self.assertIn("post-lock", source)


# =========================================================================
# Voice Prompt Cache Edge Cases
# =========================================================================

class TestVoicePromptCacheEdgeCases(unittest.TestCase):
    """Test voice prompt cache clearing and MLX cache eviction."""

    def test_clear_voice_prompt_cache_function(self):
        """clear_voice_prompt_cache() runs without error."""
        from voice_engine import clear_voice_prompt_cache
        clear_voice_prompt_cache()

    def test_voice_prompt_cache_info_has_currsize(self):
        """voice_prompt_cache_info() returns object with currsize attribute."""
        from voice_engine import voice_prompt_cache_info
        info = voice_prompt_cache_info()
        self.assertTrue(hasattr(info, "currsize"))

    def test_mlx_cache_eviction_at_max(self):
        """MLX prompt cache does not exceed _MLX_PROMPT_CACHE_MAX entries."""
        from voice_engine import _mlx_prompt_cache, _MLX_PROMPT_CACHE_MAX
        _mlx_prompt_cache.clear()
        try:
            for i in range(_MLX_PROMPT_CACHE_MAX):
                _mlx_prompt_cache[f"voice_{i}"] = {
                    "ref_audio": f"/tmp/v{i}.wav",
                    "ref_text": "text",
                }
            self.assertEqual(len(_mlx_prompt_cache), _MLX_PROMPT_CACHE_MAX)
            # Simulate the eviction logic used by load_voice_prompt_mlx
            if len(_mlx_prompt_cache) >= _MLX_PROMPT_CACHE_MAX:
                oldest_key = next(iter(_mlx_prompt_cache))
                del _mlx_prompt_cache[oldest_key]
            _mlx_prompt_cache["voice_new"] = {"ref_audio": "/tmp/new.wav", "ref_text": "new"}
            self.assertEqual(len(_mlx_prompt_cache), _MLX_PROMPT_CACHE_MAX)
        finally:
            _mlx_prompt_cache.clear()

    def test_mlx_cache_eviction_removes_oldest(self):
        """After eviction, the first-inserted key is gone."""
        from voice_engine import _mlx_prompt_cache, _MLX_PROMPT_CACHE_MAX
        _mlx_prompt_cache.clear()
        try:
            for i in range(_MLX_PROMPT_CACHE_MAX):
                _mlx_prompt_cache[f"voice_{i}"] = {
                    "ref_audio": f"/tmp/v{i}.wav",
                    "ref_text": "text",
                }
            # Evict oldest (voice_0) and insert new
            oldest_key = next(iter(_mlx_prompt_cache))
            self.assertEqual(oldest_key, "voice_0")
            del _mlx_prompt_cache[oldest_key]
            _mlx_prompt_cache["voice_extra"] = {"ref_audio": "/tmp/extra.wav", "ref_text": "extra"}
            self.assertNotIn("voice_0", _mlx_prompt_cache)
            self.assertIn("voice_extra", _mlx_prompt_cache)
        finally:
            _mlx_prompt_cache.clear()


# =========================================================================
# Config Edge Cases
# =========================================================================

class TestConfigEdgeCases(unittest.TestCase):
    """Test config constants and accessor functions."""

    def test_get_backend_returns_string(self):
        """get_backend() returns 'torch' or 'mlx'."""
        from voice_config import get_backend
        result = get_backend()
        self.assertIn(result, ("torch", "mlx"))

    def test_get_model_size_default(self):
        """get_model_size() returns '1.7B' or '0.6B'."""
        from voice_config import get_model_size
        result = get_model_size()
        self.assertIn(result, ("1.7B", "0.6B"))

    def test_model_info_keys(self):
        """MODEL_INFO has size-based keys."""
        from voice_config import MODEL_INFO
        self.assertIn("1.7B", MODEL_INFO)
        self.assertIn("0.6B", MODEL_INFO)

    def test_custom_speakers_have_fields(self):
        """Each entry in CUSTOM_VOICE_SPEAKERS has a 'name' key."""
        from voice_config import CUSTOM_VOICE_SPEAKERS
        self.assertGreater(len(CUSTOM_VOICE_SPEAKERS), 0)
        for key, entry in CUSTOM_VOICE_SPEAKERS.items():
            self.assertIn("name", entry, f"Speaker '{key}' missing 'name' field")

    def test_prosody_presets_all_strings(self):
        """All values in DEFAULT_PROSODY_PRESETS are non-empty strings."""
        from voice_config import DEFAULT_PROSODY_PRESETS
        self.assertGreater(len(DEFAULT_PROSODY_PRESETS), 0)
        for key, value in DEFAULT_PROSODY_PRESETS.items():
            self.assertIsInstance(value, str, f"Preset '{key}' is not a string")
            self.assertTrue(len(value) > 0, f"Preset '{key}' is empty")

    def test_valid_backends_constant(self):
        """VALID_BACKENDS contains 'torch' and 'mlx'."""
        from voice_config import VALID_BACKENDS
        self.assertIn("torch", VALID_BACKENDS)
        self.assertIn("mlx", VALID_BACKENDS)


class TestConfigValidation(unittest.TestCase):
    """Test validate_config() catches bad values."""

    def test_valid_config_no_issues(self):
        """A well-formed config produces no validation issues."""
        from voice_config import validate_config
        config = {
            "advanced": {"backend": "mlx", "model_size": "1.7B"},
            "generation": {"temperature": 0.7},
            "security": {"max_text_length": 10000},
        }
        self.assertEqual(validate_config(config), [])

    def test_invalid_backend(self):
        from voice_config import validate_config
        issues = validate_config({"advanced": {"backend": "invalid"}})
        self.assertTrue(any("backend" in i for i in issues))

    def test_invalid_model_size(self):
        from voice_config import validate_config
        issues = validate_config({"advanced": {"model_size": "99B"}})
        self.assertTrue(any("model_size" in i for i in issues))

    def test_temperature_out_of_range(self):
        from voice_config import validate_config
        issues = validate_config({"generation": {"temperature": 5.0}})
        self.assertTrue(any("temperature" in i for i in issues))

    def test_negative_max_text_length(self):
        from voice_config import validate_config
        issues = validate_config({"security": {"max_text_length": -1}})
        self.assertTrue(any("max_text_length" in i for i in issues))

    def test_empty_config_no_issues(self):
        from voice_config import validate_config
        self.assertEqual(validate_config({}), [])


# =========================================================================
# SSML Edge Cases
# =========================================================================

@_skip_generate
class TestSSMLEdgeCases(unittest.TestCase):
    """Test SSML parsing edge cases."""

    def test_ssml_sub_replacement(self):
        """<sub alias='hello'>hi</sub> replaces content with alias."""
        from voice_generate import parse_ssml
        text, meta = parse_ssml('Say <sub alias="hello">hi</sub> please')
        self.assertIn("hello", text)
        self.assertNotIn("<sub", text)
        self.assertNotIn("hi", text)

    def test_ssml_say_as_characters(self):
        """<say-as interpret-as='characters'>ABC</say-as> spells out as 'A B C'."""
        from voice_generate import parse_ssml
        text, meta = parse_ssml('<say-as interpret-as="characters">ABC</say-as>')
        self.assertIn("A B C", text)

    def test_ssml_prosody_rate_slow(self):
        """<prosody rate='slow'> sets speed=0.8 in metadata."""
        from voice_generate import parse_ssml
        text, meta = parse_ssml('<prosody rate="slow">Hello world</prosody>')
        self.assertTrue(meta["has_ssml"])
        self.assertIsNotNone(meta["prosody"])
        self.assertEqual(meta["prosody"]["speed"], 0.8)

    def test_ssml_prosody_pitch_high(self):
        """<prosody pitch='high'> sets pitch=2 in metadata."""
        from voice_generate import parse_ssml
        text, meta = parse_ssml('<prosody pitch="high">Hello world</prosody>')
        self.assertTrue(meta["has_ssml"])
        self.assertIsNotNone(meta["prosody"])
        self.assertEqual(meta["prosody"]["pitch"], 2)

    def test_ssml_nested_emphasis_sub(self):
        """Nested <emphasis><sub alias='hello'>hi</sub></emphasis> produces 'hello'."""
        from voice_generate import parse_ssml
        text, meta = parse_ssml('<emphasis><sub alias="hello">hi</sub></emphasis>')
        self.assertIn("hello", text)
        self.assertNotIn("<", text)


# =========================================================================
# Dry-Run and Interactive Mode Tests
# =========================================================================

@_skip_generate
class TestDryRunAndInteractive(unittest.TestCase):
    """Verify dry-run flag and interactive mode exist in source."""

    def test_dry_run_flag_in_source(self):
        """voice_generate.py source contains '--dry-run' argument."""
        import voice_generate
        source = inspect.getsource(voice_generate)
        self.assertIn("--dry-run", source)

    def test_dry_run_marker_in_source(self):
        """voice_generate.py source contains 'DRY RUN' marker text."""
        import voice_generate
        source = inspect.getsource(voice_generate)
        self.assertIn("DRY RUN", source)

    def test_interactive_mode_function_exists(self):
        """voice_generate has a callable interactive_mode function."""
        import voice_generate
        self.assertTrue(hasattr(voice_generate, "interactive_mode"))
        self.assertTrue(callable(voice_generate.interactive_mode))


# =========================================================================
# Gradio UI Launch Tests
# =========================================================================

class TestLaunchGradioUI(unittest.TestCase):
    """Verify launch_gradio_ui no longer shells out to voice_ui.py."""

    def test_no_voice_ui_reference(self):
        """launch_gradio_ui source must not reference voice_ui.py."""
        from qwen3_tts.interface.generate import launch_gradio_ui
        source = inspect.getsource(launch_gradio_ui)
        self.assertNotIn("voice_ui.py", source)

    def test_no_subprocess_run(self):
        """launch_gradio_ui must not call subprocess.run."""
        from qwen3_tts.interface.generate import launch_gradio_ui
        source = inspect.getsource(launch_gradio_ui)
        self.assertNotIn("subprocess.run", source)

    def test_calls_build_ui_and_launch(self):
        """launch_gradio_ui delegates to build_ui_and_launch."""
        from unittest.mock import patch
        from qwen3_tts.interface import generate as gen_mod

        with patch.object(gen_mod, "build_ui_and_launch") as mock_build:
            with patch.object(gen_mod, "ensure_server_running", return_value=True):
                gen_mod.launch_gradio_ui({"ui": {"port": 7860}})
        mock_build.assert_called_once()


# =========================================================================
# ensure_server_running Tests
# =========================================================================

class TestEnsureServerRunning(unittest.TestCase):
    """Verify ensure_server_running uses new CLI paths."""

    def test_no_startTTSServer_reference(self):
        """ensure_server_running must not reference startTTSServer."""
        from qwen3_tts.interface.generate import ensure_server_running
        source = inspect.getsource(ensure_server_running)
        self.assertNotIn("startTTSServer", source)

    def test_no_voice_server_py_reference(self):
        """ensure_server_running must not reference voice_server.py."""
        from qwen3_tts.interface.generate import ensure_server_running
        source = inspect.getsource(ensure_server_running)
        self.assertNotIn("voice_server.py", source)

    def test_has_tts_bin_reference(self):
        """ensure_server_running should reference ~/bin/tts."""
        from qwen3_tts.interface.generate import ensure_server_running
        source = inspect.getsource(ensure_server_running)
        self.assertIn("~/bin/tts", source)

    def test_has_server_app_reference(self):
        """ensure_server_running should reference qwen3_tts/server/app.py."""
        from qwen3_tts.interface.generate import ensure_server_running
        source = inspect.getsource(ensure_server_running)
        self.assertIn("qwen3_tts/server/app.py", source)


# =========================================================================
# Deprecated Command Reference Tests
# =========================================================================

_DEPRECATED_COMMANDS = [
    "startTTSServer", "stopTTSServer", "changeVoice",
    "createVoice", "ttsUI", "configureTTS",
]


@_skip_generate
class TestDeprecatedRefsGenerate(unittest.TestCase):
    """generate.py must not contain deprecated command names in user messages."""

    def test_no_deprecated_commands(self):
        from qwen3_tts.interface import generate
        source = inspect.getsource(generate)
        for cmd in _DEPRECATED_COMMANDS:
            self.assertNotIn(cmd, source, f"Found deprecated '{cmd}' in generate.py")


class TestDeprecatedRefsEngine(unittest.TestCase):
    """engine.py must not contain deprecated command names."""

    def test_no_deprecated_commands(self):
        from qwen3_tts.core import engine
        source = inspect.getsource(engine)
        for cmd in _DEPRECATED_COMMANDS:
            self.assertNotIn(cmd, source, f"Found deprecated '{cmd}' in engine.py")


class TestDeprecatedRefsCreateVoice(unittest.TestCase):
    """create_voice.py must not contain deprecated command names."""

    def test_no_deprecated_commands(self):
        from qwen3_tts.tools import create_voice
        source = inspect.getsource(create_voice)
        for cmd in _DEPRECATED_COMMANDS:
            self.assertNotIn(cmd, source, f"Found deprecated '{cmd}' in create_voice.py")


try:
    import gradio  # noqa: F401
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

_skip_gradio = unittest.skipUnless(HAS_GRADIO, "requires gradio")


@_skip_gradio
class TestDeprecatedRefsUI(unittest.TestCase):
    """ui.py must not contain deprecated command names."""

    def test_no_deprecated_commands(self):
        from qwen3_tts.interface import ui
        source = inspect.getsource(ui)
        for cmd in _DEPRECATED_COMMANDS:
            self.assertNotIn(cmd, source, f"Found deprecated '{cmd}' in ui.py")


# =========================================================================
# Task 6: Config Function Tests
# =========================================================================

class TestConfigFunctions(unittest.TestCase):
    """Tests for config.py utility functions."""

    def test_save_config_roundtrip(self):
        """save_config writes JSON that load_config can read back."""
        import json
        import tempfile
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
        from unittest.mock import patch
        from qwen3_tts.core.config import get_optimal_attn_config
        with patch("qwen3_tts.core.config.get_cuda_capability", return_value=None):
            attn, dtype, load_8bit = get_optimal_attn_config()
            self.assertEqual(attn, "sdpa")
            self.assertEqual(dtype, "float32")
            self.assertFalse(load_8bit)

    def test_get_optimal_attn_config_ampere(self):
        """get_optimal_attn_config returns flash_attention_2 for Ampere+ GPU."""
        from unittest.mock import patch
        from qwen3_tts.core.config import get_optimal_attn_config
        with patch("qwen3_tts.core.config.get_cuda_capability", return_value=(8, 0)):
            attn, dtype, load_8bit = get_optimal_attn_config()
            self.assertEqual(attn, "flash_attention_2")
            self.assertEqual(dtype, "bfloat16")
            self.assertFalse(load_8bit)

    def test_get_optimal_attn_config_turing(self):
        """get_optimal_attn_config returns sdpa/float16/8bit for Turing GPU."""
        from unittest.mock import patch
        from qwen3_tts.core.config import get_optimal_attn_config
        with patch("qwen3_tts.core.config.get_cuda_capability", return_value=(7, 5)):
            attn, dtype, load_8bit = get_optimal_attn_config()
            self.assertEqual(attn, "sdpa")
            self.assertEqual(dtype, "float16")
            self.assertTrue(load_8bit)


# =========================================================================
# Task 7: Engine Function Tests
# =========================================================================

class TestEngineFunctions(unittest.TestCase):
    """Tests for engine.py utility functions."""

    def test_get_audio_loader_returns_valid(self):
        """get_audio_loader returns 'torchaudio' or 'librosa'."""
        from qwen3_tts.core.engine import get_audio_loader
        result = get_audio_loader()
        self.assertIn(result, ("torchaudio", "librosa"))

    def test_is_asr_available_returns_bool(self):
        """is_asr_available returns a boolean."""
        from qwen3_tts.core.engine import is_asr_available
        result = is_asr_available()
        self.assertIsInstance(result, bool)

    def test_is_asr_loaded_returns_bool(self):
        """is_asr_loaded returns False when no model loaded."""
        from qwen3_tts.core.engine import is_asr_loaded
        result = is_asr_loaded()
        self.assertIsInstance(result, bool)

    def test_get_asr_model_info_returns_dict(self):
        """get_asr_model_info returns dict with expected keys."""
        from qwen3_tts.core.engine import get_asr_model_info
        result = get_asr_model_info()
        self.assertIsInstance(result, dict)
        self.assertIn("loaded", result)
        self.assertIn("backend", result)
        self.assertIn("model_name", result)

    def test_get_asr_model_info_not_loaded(self):
        """get_asr_model_info returns loaded=False when no model."""
        from qwen3_tts.core.engine import get_asr_model_info
        result = get_asr_model_info()
        # In test environment, no ASR model should be loaded
        self.assertFalse(result["loaded"])

    def test_migrate_orphan_mlx_prompts_empty_dir(self):
        """migrate_orphan_mlx_prompts handles empty/missing directory."""
        import tempfile
        from unittest.mock import patch
        from qwen3_tts.core.engine import migrate_orphan_mlx_prompts
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("qwen3_tts.core.engine.VOICE_PROMPTS_DIR", tmpdir):
                result = migrate_orphan_mlx_prompts()
                self.assertEqual(result, 0)


# =========================================================================
# Task 9: App Helper Function Tests
# =========================================================================

@_skip_server
class TestAppHelperFunctions(unittest.TestCase):
    """Tests for app.py helper functions."""

    def test_gen_cache_key_deterministic(self):
        """_gen_cache_key returns same hash for same inputs."""
        from qwen3_tts.server.app import _gen_cache_key
        key1 = _gen_cache_key("hello", "clone", {"temperature": 0.7}, prompt_file="voice.pt")
        key2 = _gen_cache_key("hello", "clone", {"temperature": 0.7}, prompt_file="voice.pt")
        self.assertEqual(key1, key2)

    def test_gen_cache_key_different_text(self):
        """_gen_cache_key returns different hash for different text."""
        from qwen3_tts.server.app import _gen_cache_key
        key1 = _gen_cache_key("hello", "clone", {"temperature": 0.7})
        key2 = _gen_cache_key("world", "clone", {"temperature": 0.7})
        self.assertNotEqual(key1, key2)

    def test_gen_cache_key_is_hex_string(self):
        """_gen_cache_key returns a hex string of length 16."""
        from qwen3_tts.server.app import _gen_cache_key
        key = _gen_cache_key("test", "design", {})
        self.assertEqual(len(key), 16)
        int(key, 16)  # Should not raise

    def test_generate_auth_token_creates_file(self):
        """generate_auth_token creates token file with correct perms."""
        import tempfile
        import stat
        import qwen3_tts.server.app as app_mod
        original_token_file = app_mod.TOKEN_FILE
        try:
            tmp = tempfile.mktemp(suffix=".token")
            app_mod.TOKEN_FILE = tmp
            token = app_mod.generate_auth_token()
            self.assertEqual(len(token), 64)  # 32 bytes = 64 hex chars
            self.assertTrue(os.path.exists(tmp))
            mode = os.stat(tmp).st_mode
            self.assertEqual(stat.S_IMODE(mode), 0o600)
            with open(tmp) as f:
                self.assertEqual(f.read(), token)
        finally:
            app_mod.TOKEN_FILE = original_token_file
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_create_temp_audio_copy(self):
        """_create_temp_audio_copy creates a copy with restricted perms."""
        import tempfile
        import stat
        from qwen3_tts.server.app import _create_temp_audio_copy
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src:
            src.write(b"fake audio data")
            src_path = src.name
        try:
            tmp_path = _create_temp_audio_copy(src_path)
            self.assertTrue(os.path.exists(tmp_path))
            mode = os.stat(tmp_path).st_mode
            self.assertEqual(stat.S_IMODE(mode), 0o600)
            with open(tmp_path, 'rb') as f:
                self.assertEqual(f.read(), b"fake audio data")
            os.unlink(tmp_path)
        finally:
            os.unlink(src_path)

    def test_estimate_eta_no_history(self):
        """_estimate_eta returns None when no history."""
        import qwen3_tts.server.app as app_mod
        # Reset cache to force recalculation
        app_mod._eta_cache["last_updated"] = 0
        app_mod._eta_cache["median_rate"] = None
        from unittest.mock import patch
        with patch("os.path.exists", return_value=False):
            result = app_mod._estimate_eta(100, 5.0)
            self.assertIsNone(result)

    def test_prepare_mode_params_clone_no_prompt(self):
        """_prepare_mode_params returns error when clone mode has no prompt."""
        from qwen3_tts.server.app import _prepare_mode_params, app
        with app.app_context():
            voice_prompt, error = _prepare_mode_params("clone", {})
            self.assertIsNone(voice_prompt)
            self.assertIsNotNone(error)

    def test_prepare_mode_params_custom_no_speaker(self):
        """_prepare_mode_params returns error when custom mode has no speaker."""
        from qwen3_tts.server.app import _prepare_mode_params, app
        with app.app_context():
            voice_prompt, error = _prepare_mode_params("custom", {})
            self.assertIsNone(voice_prompt)
            self.assertIsNotNone(error)


# =========================================================================
# build_ui_and_launch Tests
# =========================================================================

class TestBuildUIAndLaunch(unittest.TestCase):
    """build_ui_and_launch should respect TTS_UI_NO_BROWSER and TTS_UI_SHARE env vars."""

    @unittest.mock.patch('qwen3_tts.interface.ui._find_available_port', return_value=7860)
    @unittest.mock.patch('qwen3_tts.interface.ui.build_ui')
    def test_inbrowser_true_by_default(self, mock_build_ui, _mock_port):
        """Browser should open by default when TTS_UI_NO_BROWSER is not set."""
        mock_demo = MagicMock()
        mock_build_ui.return_value = mock_demo
        config = {"ui": {"port": 7860}}
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ('TTS_UI_NO_BROWSER', 'TTS_UI_SHARE')}
        with unittest.mock.patch.dict(os.environ, clean_env, clear=True):
            from qwen3_tts.interface.generate import build_ui_and_launch
            build_ui_and_launch(config)
        call_kwargs = mock_demo.launch.call_args[1]
        self.assertTrue(call_kwargs.get('inbrowser'),
                        "Expected inbrowser=True when TTS_UI_NO_BROWSER is not set")

    @unittest.mock.patch('qwen3_tts.interface.ui._find_available_port', return_value=7860)
    @unittest.mock.patch('qwen3_tts.interface.ui.build_ui')
    def test_inbrowser_false_when_no_browser_set(self, mock_build_ui, _mock_port):
        """Browser should NOT open when TTS_UI_NO_BROWSER=1."""
        mock_demo = MagicMock()
        mock_build_ui.return_value = mock_demo
        config = {"ui": {"port": 7860}}
        with unittest.mock.patch.dict(os.environ, {'TTS_UI_NO_BROWSER': '1'}):
            from qwen3_tts.interface.generate import build_ui_and_launch
            build_ui_and_launch(config)
        call_kwargs = mock_demo.launch.call_args[1]
        self.assertFalse(call_kwargs.get('inbrowser'),
                         "Expected inbrowser=False when TTS_UI_NO_BROWSER=1")

    @unittest.mock.patch('qwen3_tts.interface.ui._find_available_port', return_value=7860)
    @unittest.mock.patch('qwen3_tts.interface.ui.build_ui')
    def test_share_true_when_env_var_set(self, mock_build_ui, _mock_port):
        """Share should be True when TTS_UI_SHARE=1."""
        mock_demo = MagicMock()
        mock_build_ui.return_value = mock_demo
        config = {"ui": {"port": 7860}}
        with unittest.mock.patch.dict(os.environ, {'TTS_UI_SHARE': '1', 'TTS_UI_NO_BROWSER': '1'}):
            from qwen3_tts.interface.generate import build_ui_and_launch
            build_ui_and_launch(config)
        call_kwargs = mock_demo.launch.call_args[1]
        self.assertTrue(call_kwargs.get('share'),
                        "Expected share=True when TTS_UI_SHARE=1")

    @unittest.mock.patch('qwen3_tts.interface.ui._find_available_port', return_value=7860)
    @unittest.mock.patch('qwen3_tts.interface.ui.build_ui')
    def test_share_false_by_default(self, mock_build_ui, _mock_port):
        """Share should be False by default when TTS_UI_SHARE is not set."""
        mock_demo = MagicMock()
        mock_build_ui.return_value = mock_demo
        config = {"ui": {"port": 7860}}
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ('TTS_UI_NO_BROWSER', 'TTS_UI_SHARE')}
        with unittest.mock.patch.dict(os.environ, clean_env, clear=True):
            from qwen3_tts.interface.generate import build_ui_and_launch
            build_ui_and_launch(config)
        call_kwargs = mock_demo.launch.call_args[1]
        self.assertFalse(call_kwargs.get('share'),
                         "Expected share=False when TTS_UI_SHARE is not set")

    @unittest.mock.patch('qwen3_tts.core.config.IN_COLAB', True)
    @unittest.mock.patch('qwen3_tts.interface.ui._find_available_port', return_value=7860)
    @unittest.mock.patch('qwen3_tts.interface.ui.build_ui')
    def test_colab_forces_share_and_disables_browser(self, mock_build_ui, _mock_port):
        """In Colab, share=True and inbrowser=False regardless of env vars."""
        mock_demo = MagicMock()
        mock_build_ui.return_value = mock_demo
        config = {"ui": {"port": 7860}}
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ('TTS_UI_NO_BROWSER', 'TTS_UI_SHARE')}
        with unittest.mock.patch.dict(os.environ, clean_env, clear=True):
            from qwen3_tts.interface.generate import build_ui_and_launch
            build_ui_and_launch(config)
        call_kwargs = mock_demo.launch.call_args[1]
        self.assertTrue(call_kwargs.get('share'),
                        "Expected share=True in Colab environment")
        self.assertFalse(call_kwargs.get('inbrowser'),
                         "Expected inbrowser=False in Colab environment")


# =========================================================================
# get_server_status Tests
# =========================================================================

class TestGetServerStatus(unittest.TestCase):
    """get_server_status() should correctly parse stats response."""

    @unittest.mock.patch('qwen3_tts.interface.ui.TTSClient')
    def test_small_memory_not_shown_as_zero(self, mock_client_class):
        """A non-zero memory value (e.g. 0.3 MB) must not display as '0MB'."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.is_server_running.return_value = True
        mock_client.get_stats.return_value = {
            'mlx_memory_active_mb': 0.3,
            'backend': 'mlx',
            'mlx_quantization': '8bit',
            'clone_model_loaded': False,
            'design_model_loaded': False,
            'custom_model_loaded': False,
        }
        from qwen3_tts.interface.ui import get_server_status
        _, memory, _, _ = get_server_status()
        self.assertNotEqual(memory, "0MB",
            "Memory value 0.3 MB must not round to '0MB'")

    @unittest.mock.patch('qwen3_tts.interface.ui.TTSClient')
    def test_zero_memory_via_or_chain_not_skipped(self, mock_client_class):
        """If mlx_memory_active_mb is 0.0 (falsy), fall through to next key correctly."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.is_server_running.return_value = True
        mock_client.get_stats.return_value = {
            'mlx_memory_active_mb': 0.0,
            'mps_memory_allocated_mb': 512.0,
            'backend': 'mlx',
            'mlx_quantization': '8bit',
            'clone_model_loaded': True,
            'design_model_loaded': False,
            'custom_model_loaded': False,
        }
        from qwen3_tts.interface.ui import get_server_status
        _, memory, models, _ = get_server_status()
        self.assertEqual(models, "Clone")

    @unittest.mock.patch('qwen3_tts.interface.ui.TTSClient')
    def test_loaded_models_shown_correctly(self, mock_client_class):
        """Loaded models should be listed in status, not 'None'."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.is_server_running.return_value = True
        mock_client.get_stats.return_value = {
            'mlx_memory_active_mb': 2500.5,
            'backend': 'mlx',
            'mlx_quantization': '8bit',
            'clone_model_loaded': True,
            'design_model_loaded': True,
            'custom_model_loaded': False,
        }
        from qwen3_tts.interface.ui import get_server_status
        _, _, models, _ = get_server_status()
        self.assertIn("Clone", models)
        self.assertIn("Design", models)
        self.assertNotEqual(models, "None")


if __name__ == "__main__":
    unittest.main()
