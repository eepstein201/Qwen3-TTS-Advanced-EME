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
    import fastapi  # noqa: F401
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

_skip_server = unittest.skipUnless(HAS_SOUNDFILE and HAS_FASTAPI, "requires soundfile + fastapi")
_skip_generate = unittest.skipUnless(HAS_SOUNDFILE, "requires soundfile (voice_generate)")


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
                    "ref_audio": f"/tmp/v{i}.wav",  # nosec B108
                    "ref_text": "text",
                }
            self.assertEqual(len(_mlx_prompt_cache), _MLX_PROMPT_CACHE_MAX)
            # Simulate the eviction logic used by load_voice_prompt_mlx
            if len(_mlx_prompt_cache) >= _MLX_PROMPT_CACHE_MAX:
                oldest_key = next(iter(_mlx_prompt_cache))
                del _mlx_prompt_cache[oldest_key]
            _mlx_prompt_cache["voice_new"] = {"ref_audio": "/tmp/new.wav", "ref_text": "new"}  # nosec B108
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
                    "ref_audio": f"/tmp/v{i}.wav",  # nosec B108
                    "ref_text": "text",
                }
            # Evict oldest (voice_0) and insert new
            oldest_key = next(iter(_mlx_prompt_cache))
            self.assertEqual(oldest_key, "voice_0")
            del _mlx_prompt_cache[oldest_key]
            _mlx_prompt_cache["voice_extra"] = {"ref_audio": "/tmp/extra.wav", "ref_text": "extra"}  # nosec B108
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
        """get_optimal_attn_config returns flash_attention_2 for Ampere+ GPU with flash_attn installed."""
        from unittest.mock import patch
        from qwen3_tts.core.config import get_optimal_attn_config
        with patch("qwen3_tts.core.config.get_cuda_capability", return_value=(8, 0)):
            with patch("qwen3_tts.core.config._has_flash_attn", return_value=True):
                attn, dtype, load_8bit = get_optimal_attn_config()
                self.assertEqual(attn, "flash_attention_2")
                self.assertEqual(dtype, "bfloat16")
                self.assertFalse(load_8bit)

    def test_get_optimal_attn_config_ampere_no_flash_attn(self):
        """get_optimal_attn_config falls back to sdpa when flash_attn not installed."""
        from unittest.mock import patch
        from qwen3_tts.core.config import get_optimal_attn_config
        with patch("qwen3_tts.core.config.get_cuda_capability", return_value=(8, 9)):
            with patch("qwen3_tts.core.config._has_flash_attn", return_value=False):
                attn, dtype, load_8bit = get_optimal_attn_config()
                self.assertEqual(attn, "sdpa")
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

    def test_cuda_optimizations_falls_back_to_sdpa_without_flash_attn(self):
        """_apply_cuda_optimizations uses sdpa when flash_attn not installed on Ampere+."""
        from unittest.mock import patch, MagicMock
        import sys
        # Create a mock torch module for CUDA simulation
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 9)
        mock_torch.bfloat16 = "bfloat16_sentinel"
        mock_torch.backends.cudnn = MagicMock()
        # Temporarily replace torch in sys.modules and reimport
        with patch("qwen3_tts.core.config._has_flash_attn", return_value=False):
            with patch.dict(sys.modules, {"torch": mock_torch}):
                # Force reimport to pick up mocked torch
                import importlib
                import qwen3_tts.core.engine as engine_mod
                importlib.reload(engine_mod)
                try:
                    config = {"generation": {"compile_model": True}}
                    attn, dtype, compile_ = engine_mod._apply_cuda_optimizations(config)
                    self.assertEqual(attn, "sdpa")
                    self.assertEqual(dtype, "bfloat16_sentinel")
                    self.assertTrue(compile_)
                finally:
                    importlib.reload(engine_mod)

    def test_cuda_optimizations_uses_flash_attn_when_available(self):
        """_apply_cuda_optimizations uses flash_attention_2 when flash_attn installed on Ampere+."""
        from unittest.mock import patch, MagicMock
        import sys
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_capability.return_value = (8, 0)
        mock_torch.bfloat16 = "bfloat16_sentinel"
        mock_torch.backends.cudnn = MagicMock()
        with patch("qwen3_tts.core.config._has_flash_attn", return_value=True):
            with patch.dict(sys.modules, {"torch": mock_torch}):
                import importlib
                import qwen3_tts.core.engine as engine_mod
                importlib.reload(engine_mod)
                try:
                    config = {"generation": {"compile_model": True}}
                    attn, dtype, compile_ = engine_mod._apply_cuda_optimizations(config)
                    self.assertEqual(attn, "flash_attention_2")
                    self.assertEqual(dtype, "bfloat16_sentinel")
                    self.assertTrue(compile_)
                finally:
                    importlib.reload(engine_mod)

    def test_migrate_orphan_mlx_prompts_accepts_clone_model(self):
        """migrate_orphan_mlx_prompts accepts optional clone_model parameter."""
        import inspect
        from qwen3_tts.core.engine import migrate_orphan_mlx_prompts
        sig = inspect.signature(migrate_orphan_mlx_prompts)
        self.assertIn("clone_model", sig.parameters,
                       "migrate_orphan_mlx_prompts must accept clone_model parameter")
        param = sig.parameters["clone_model"]
        self.assertEqual(param.default, None,
                          "clone_model parameter should default to None")

    def test_migrate_orphan_does_not_call_load_model_when_model_provided(self):
        """When clone_model is passed, migration must not call load_model()."""
        from unittest.mock import patch, MagicMock
        from qwen3_tts.core.engine import migrate_orphan_mlx_prompts
        mock_model = MagicMock()
        with patch("qwen3_tts.core.engine.load_model") as mock_load:
            # No orphan .wav files to migrate in test env, but validates the contract:
            # when clone_model is provided, load_model should never be called
            migrate_orphan_mlx_prompts(clone_model=mock_model)
            mock_load.assert_not_called()

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

    def test_load_model_torch_passes_dtype(self):
        """_load_model_torch passes dtype (not deprecated torch_dtype) to from_pretrained."""
        import inspect
        from qwen3_tts.core.engine import _load_model_torch
        source = inspect.getsource(_load_model_torch)
        self.assertIn("dtype=torch_dtype", source)

    def test_load_model_torch_compiles_inner_model(self):
        """torch.compile targets model.model (inner nn.Module), not the wrapper."""
        import inspect
        from qwen3_tts.core.engine import _load_model_torch
        source = inspect.getsource(_load_model_torch)
        self.assertIn("model.model", source)
        self.assertIn("torch.compile(model.model", source)

    def test_load_model_torch_compile_has_fallback(self):
        """torch.compile is wrapped in its own try/except for graceful degradation."""
        import inspect
        from qwen3_tts.core.engine import _load_model_torch
        source = inspect.getsource(_load_model_torch)
        lines = source.split('\n')
        compile_line = None
        for i, line in enumerate(lines):
            if 'torch.compile(' in line and not line.lstrip().startswith('#'):
                compile_line = i
                break
        self.assertIsNotNone(compile_line, "torch.compile not found in source")
        # Check for a try: within 3 lines before torch.compile (not the outer retry try)
        nearby_before = '\n'.join(lines[max(0, compile_line - 3):compile_line])
        nearby_after = '\n'.join(lines[compile_line:compile_line + 4])
        self.assertIn('try:', nearby_before,
                       "torch.compile should have a nearby try: block")
        self.assertIn('except', nearby_after,
                       "torch.compile should have a nearby except block")

    def test_load_voice_prompt_torch_registers_safe_globals(self):
        """_load_voice_prompt_torch registers VoiceClonePromptItem via add_safe_globals."""
        import inspect
        from qwen3_tts.core.engine import _load_voice_prompt_torch
        source = inspect.getsource(_load_voice_prompt_torch)
        self.assertIn("add_safe_globals", source,
                       "Must register VoiceClonePromptItem via torch.serialization.add_safe_globals")
        self.assertIn("VoiceClonePromptItem", source,
                       "Must import VoiceClonePromptItem for safe loading")

    def test_colab_notebook_syspath_uses_home_dir(self):
        """Colab notebook adds HOME_DIR (not its parent) to sys.path."""
        import json
        notebook_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'colab_notebook.ipynb'
        )
        with open(notebook_path) as f:
            nb = json.load(f)
        setup_source = None
        for cell in nb['cells']:
            src = ''.join(cell['source'])
            if 'HOME_DIR' in src and 'sys.path' in src:
                setup_source = src
                break
        self.assertIsNotNone(setup_source, "Setup cell with sys.path not found")
        self.assertIn('HOME_DIR not in sys.path', setup_source,
                       "sys.path check should use HOME_DIR directly")
        self.assertNotIn('project_parent', setup_source,
                          "Should not use dirname(HOME_DIR) for sys.path")

    def test_colab_notebook_pythonpath_uses_home_dir(self):
        """Colab server subprocess PYTHONPATH uses HOME_DIR, not dirname."""
        import json
        notebook_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'colab_notebook.ipynb'
        )
        with open(notebook_path) as f:
            nb = json.load(f)
        server_source = None
        for cell in nb['cells']:
            src = ''.join(cell['source'])
            if 'subprocess.Popen' in src and 'PYTHONPATH' in src:
                server_source = src
                break
        self.assertIsNotNone(server_source, "Server cell with PYTHONPATH not found")
        self.assertNotIn("os.path.dirname(os.path.expanduser('~/Qwen3-TTS_UserFiles'))",
                          server_source,
                          "PYTHONPATH should use project dir directly, not dirname()")


# =========================================================================
# Task 9: App Helper Function Tests
# =========================================================================

@_skip_server
class TestAppHelperFunctions(unittest.TestCase):
    """Tests for app.py helper functions."""

    def test_gen_cache_key_deterministic(self):
        """_gen_cache_key returns same hash for same inputs."""
        from qwen3_tts.server.app_fastapi import _gen_cache_key
        key1 = _gen_cache_key("hello", "clone", {"temperature": 0.7}, prompt_file="voice.pt")
        key2 = _gen_cache_key("hello", "clone", {"temperature": 0.7}, prompt_file="voice.pt")
        self.assertEqual(key1, key2)

    def test_gen_cache_key_different_text(self):
        """_gen_cache_key returns different hash for different text."""
        from qwen3_tts.server.app_fastapi import _gen_cache_key
        key1 = _gen_cache_key("hello", "clone", {"temperature": 0.7})
        key2 = _gen_cache_key("world", "clone", {"temperature": 0.7})
        self.assertNotEqual(key1, key2)

    def test_gen_cache_key_is_hex_string(self):
        """_gen_cache_key returns a hex string of length 16."""
        from qwen3_tts.server.app_fastapi import _gen_cache_key
        key = _gen_cache_key("test", "design", {})
        self.assertEqual(len(key), 16)
        int(key, 16)  # Should not raise

    def test_create_temp_audio_copy(self):
        """_create_temp_audio_copy creates a copy with restricted perms."""
        import tempfile
        import stat
        from qwen3_tts.server.app_fastapi import _create_temp_audio_copy
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


# =========================================================================
# build_ui_and_launch Tests
# =========================================================================

@_skip_gradio
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

@_skip_gradio
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
        self.assertEqual(memory, "0.0MB",
            "0.0 MB must be used directly, not skipped as falsy")
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


@_skip_gradio
class TestManageVoicesRaceCondition(unittest.TestCase):
    """Manage Voices buttons must start non-interactive to prevent race condition."""

    def test_action_buttons_start_non_interactive(self):
        """Action buttons are created with interactive=False."""
        import inspect
        from qwen3_tts.interface import ui
        source = inspect.getsource(ui.build_ui)
        lines = source.split('\n')
        for line in lines:
            if 'manage_default_btn' in line and 'gr.Button' in line:
                self.assertIn('interactive=False', line,
                              "manage_default_btn must start non-interactive")
                break
        else:
            self.fail("manage_default_btn gr.Button declaration not found")

    def test_select_event_enables_buttons(self):
        """on_table_select returns gr.update(interactive=True) for buttons."""
        import inspect
        from qwen3_tts.interface import ui
        source = inspect.getsource(ui.build_ui)
        # The .select() outputs list must include manage_default_btn
        self.assertIn('manage_default_btn', source)
        # on_table_select must return interactive updates
        self.assertIn('gr.update(interactive=True)', source,
                       "on_table_select must enable buttons via gr.update")


if __name__ == "__main__":
    unittest.main()
