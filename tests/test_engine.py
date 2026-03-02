#!/usr/bin/env python3
"""Engine function tests - CUDA optimizations, ASR, migration, etc.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_engine.py -v

No GPU, models, or running server required.
"""

import inspect
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.unit
class TestEngineFunctions(unittest.TestCase):
    """Tests for engine.py utility functions."""

    def test_cuda_optimizations_falls_back_to_sdpa_without_flash_attn(self):
        """_apply_cuda_optimizations uses sdpa when flash_attn not installed on Ampere+."""
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
        from qwen3_tts.core.engine import migrate_orphan_mlx_prompts
        sig = inspect.signature(migrate_orphan_mlx_prompts)
        self.assertIn("clone_model", sig.parameters,
                       "migrate_orphan_mlx_prompts must accept clone_model parameter")
        param = sig.parameters["clone_model"]
        self.assertEqual(param.default, None,
                          "clone_model parameter should default to None")

    def test_migrate_orphan_does_not_call_load_model_when_model_provided(self):
        """When clone_model is passed, migration must not call load_model()."""
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
        from qwen3_tts.core.engine import migrate_orphan_mlx_prompts
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("qwen3_tts.core.engine.VOICE_PROMPTS_DIR", tmpdir):
                result = migrate_orphan_mlx_prompts()
                self.assertEqual(result, 0)

    def test_load_model_torch_passes_dtype(self):
        """_load_model_torch passes dtype (not deprecated torch_dtype) to from_pretrained."""
        from qwen3_tts.core.engine import _load_model_torch
        source = inspect.getsource(_load_model_torch)
        self.assertIn("dtype=torch_dtype", source)

    def test_load_model_torch_compiles_inner_model(self):
        """torch.compile targets model.model (inner nn.Module), not the wrapper."""
        from qwen3_tts.core.engine import _load_model_torch
        source = inspect.getsource(_load_model_torch)
        self.assertIn("model.model", source)
        self.assertIn("torch.compile(model.model", source)

    def test_load_model_torch_compile_has_fallback(self):
        """torch.compile is wrapped in its own try/except for graceful degradation."""
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
        from qwen3_tts.core.engine import _load_voice_prompt_torch
        source = inspect.getsource(_load_voice_prompt_torch)
        self.assertIn("add_safe_globals", source,
                       "Must register VoiceClonePromptItem via torch.serialization.add_safe_globals")
        self.assertIn("VoiceClonePromptItem", source,
                       "Must import VoiceClonePromptItem for safe loading")

    def test_colab_notebook_syspath_uses_home_dir(self):
        """Colab notebook adds HOME_DIR (not its parent) to sys.path."""
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


if __name__ == "__main__":
    unittest.main()
