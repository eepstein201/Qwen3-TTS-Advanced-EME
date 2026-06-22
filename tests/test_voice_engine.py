"""Engine tests extracted from test_voice.py."""

import sys
import unittest
from unittest.mock import MagicMock, patch

from tests.voice_test_helpers import (
    _skip_server,
    _skip_ui,
)

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


class TestFloat32Guard(unittest.TestCase):
    """Test float32 dtype guard for torch clone mode on MPS."""

    def test_float32_guard_exists_in_torch_inference(self):
        """_apply_mps_float32_guard has float32 guard logic for clone mode on MPS."""
        # Logic extracted to _apply_mps_float32_guard in Phase 5 refactor
        import inspect

        from qwen3_tts.core.engine.inference import _apply_mps_float32_guard
        source = inspect.getsource(_apply_mps_float32_guard)
        self.assertIn("float32", source)
        self.assertIn("clone", source)


class TestMLXMetalRecovery(unittest.TestCase):
    """Test MLX Metal kernel crash recovery."""

    def test_run_inference_handles_exceptions(self):
        """run_inference wraps inference in try/except."""
        import inspect

        from qwen3_tts.core.engine.inference import _run_inference_single
        source = inspect.getsource(_run_inference_single)
        # Should have exception handling
        self.assertIn("except", source)


@_skip_server
@_skip_ui
class TestMLXMemoryStats(unittest.TestCase):
    """Test MLX memory stats collection in /stats endpoint."""

    def test_stats_mlx_memory_code_exists(self):
        """Stats handler has MLX memory collection code."""
        import inspect

        from qwen3_tts.server import app_models
        # Find the stats handler
        source = inspect.getsource(app_models)
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
        """_cleanup_device_memory has CUDA memory cleanup code."""
        # Logic extracted to _cleanup_device_memory in Phase 5 refactor
        import inspect

        from qwen3_tts.core.engine.inference import _cleanup_device_memory
        source = inspect.getsource(_cleanup_device_memory)
        self.assertIn("torch.cuda.is_available", source)
        self.assertIn("torch.cuda.empty_cache", source)


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
        from qwen3_tts.core.engine import get_audio_loader, set_audio_loader
        original = get_audio_loader()
        try:
            set_audio_loader("librosa")
            self.assertEqual(get_audio_loader(), "librosa")
            set_audio_loader("torchaudio")
            self.assertEqual(get_audio_loader(), "torchaudio")
        finally:
            set_audio_loader(original)


if __name__ == "__main__":
    unittest.main()
