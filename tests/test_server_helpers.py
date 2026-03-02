#!/usr/bin/env python3
"""Server helper function tests - app helpers, launch, build UI, status.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_server_helpers.py -v

No GPU, models, or running server required.
"""

import inspect
import os
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pytest

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

try:
    import gradio  # noqa: F401
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

_skip_gradio = unittest.skipUnless(HAS_GRADIO, "requires gradio")


# =========================================================================
# Gradio UI Launch Tests
# =========================================================================

@pytest.mark.unit
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
        from qwen3_tts.interface import generate as gen_mod

        with patch.object(gen_mod, "build_ui_and_launch") as mock_build:
            with patch.object(gen_mod, "ensure_server_running", return_value=True):
                gen_mod.launch_gradio_ui({"ui": {"port": 7860}})
        mock_build.assert_called_once()


# =========================================================================
# ensure_server_running Tests
# =========================================================================

@pytest.mark.unit
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
# App Helper Function Tests
# =========================================================================

@pytest.mark.unit
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

    def test_create_temp_audio_copy(self):
        """_create_temp_audio_copy creates a copy with restricted perms."""
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


# =========================================================================
# build_ui_and_launch Tests
# =========================================================================

@pytest.mark.unit
@_skip_gradio
class TestBuildUIAndLaunch(unittest.TestCase):
    """build_ui_and_launch should respect TTS_UI_NO_BROWSER and TTS_UI_SHARE env vars."""

    @patch('qwen3_tts.interface.ui._find_available_port', return_value=7860)
    @patch('qwen3_tts.interface.ui.build_ui')
    def test_inbrowser_true_by_default(self, mock_build_ui, _mock_port):
        """Browser should open by default when TTS_UI_NO_BROWSER is not set."""
        mock_demo = MagicMock()
        mock_build_ui.return_value = mock_demo
        config = {"ui": {"port": 7860}}
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ('TTS_UI_NO_BROWSER', 'TTS_UI_SHARE')}
        with patch.dict(os.environ, clean_env, clear=True):
            from qwen3_tts.interface.generate import build_ui_and_launch
            build_ui_and_launch(config)
        call_kwargs = mock_demo.launch.call_args[1]
        self.assertTrue(call_kwargs.get('inbrowser'),
                        "Expected inbrowser=True when TTS_UI_NO_BROWSER is not set")

    @patch('qwen3_tts.interface.ui._find_available_port', return_value=7860)
    @patch('qwen3_tts.interface.ui.build_ui')
    def test_inbrowser_false_when_no_browser_set(self, mock_build_ui, _mock_port):
        """Browser should NOT open when TTS_UI_NO_BROWSER=1."""
        mock_demo = MagicMock()
        mock_build_ui.return_value = mock_demo
        config = {"ui": {"port": 7860}}
        with patch.dict(os.environ, {'TTS_UI_NO_BROWSER': '1'}):
            from qwen3_tts.interface.generate import build_ui_and_launch
            build_ui_and_launch(config)
        call_kwargs = mock_demo.launch.call_args[1]
        self.assertFalse(call_kwargs.get('inbrowser'),
                         "Expected inbrowser=False when TTS_UI_NO_BROWSER=1")

    @patch('qwen3_tts.interface.ui._find_available_port', return_value=7860)
    @patch('qwen3_tts.interface.ui.build_ui')
    def test_share_true_when_env_var_set(self, mock_build_ui, _mock_port):
        """Share should be True when TTS_UI_SHARE=1."""
        mock_demo = MagicMock()
        mock_build_ui.return_value = mock_demo
        config = {"ui": {"port": 7860}}
        with patch.dict(os.environ, {'TTS_UI_SHARE': '1', 'TTS_UI_NO_BROWSER': '1'}):
            from qwen3_tts.interface.generate import build_ui_and_launch
            build_ui_and_launch(config)
        call_kwargs = mock_demo.launch.call_args[1]
        self.assertTrue(call_kwargs.get('share'),
                        "Expected share=True when TTS_UI_SHARE=1")

    @patch('qwen3_tts.interface.ui._find_available_port', return_value=7860)
    @patch('qwen3_tts.interface.ui.build_ui')
    def test_share_false_by_default(self, mock_build_ui, _mock_port):
        """Share should be False by default when TTS_UI_SHARE is not set."""
        mock_demo = MagicMock()
        mock_build_ui.return_value = mock_demo
        config = {"ui": {"port": 7860}}
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ('TTS_UI_NO_BROWSER', 'TTS_UI_SHARE')}
        with patch.dict(os.environ, clean_env, clear=True):
            from qwen3_tts.interface.generate import build_ui_and_launch
            build_ui_and_launch(config)
        call_kwargs = mock_demo.launch.call_args[1]
        self.assertFalse(call_kwargs.get('share'),
                         "Expected share=False when TTS_UI_SHARE is not set")

    @patch('qwen3_tts.core.config.IN_COLAB', True)
    @patch('qwen3_tts.interface.ui._find_available_port', return_value=7860)
    @patch('qwen3_tts.interface.ui.build_ui')
    def test_colab_forces_share_and_disables_browser(self, mock_build_ui, _mock_port):
        """In Colab, share=True and inbrowser=False regardless of env vars."""
        mock_demo = MagicMock()
        mock_build_ui.return_value = mock_demo
        config = {"ui": {"port": 7860}}
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ('TTS_UI_NO_BROWSER', 'TTS_UI_SHARE')}
        with patch.dict(os.environ, clean_env, clear=True):
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

@pytest.mark.unit
@_skip_gradio
class TestGetServerStatus(unittest.TestCase):
    """get_server_status() should correctly parse stats response."""

    @patch('qwen3_tts.interface.ui.TTSClient')
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

    @patch('qwen3_tts.interface.ui.TTSClient')
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

    @patch('qwen3_tts.interface.ui.TTSClient')
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


@pytest.mark.unit
@_skip_gradio
class TestManageVoicesRaceCondition(unittest.TestCase):
    """Manage Voices buttons must start non-interactive to prevent race condition."""

    def test_action_buttons_start_non_interactive(self):
        """Action buttons are created with interactive=False."""
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
        from qwen3_tts.interface import ui
        source = inspect.getsource(ui.build_ui)
        # The .select() outputs list must include manage_default_btn
        self.assertIn('manage_default_btn', source)
        # on_table_select must return interactive updates
        self.assertIn('gr.update(interactive=True)', source,
                       "on_table_select must enable buttons via gr.update")


if __name__ == "__main__":
    unittest.main()
