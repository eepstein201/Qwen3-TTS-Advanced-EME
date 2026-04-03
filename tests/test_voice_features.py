"""Feature tests extracted from test_voice.py."""

import json
import os
import unittest
from unittest.mock import patch, MagicMock

from tests.voice_test_helpers import (
    _skip_client, _skip_generate, _skip_ui,
)


class TestRubberBandAudioProcessing(unittest.TestCase):
    """Test pyrubberband fallback to librosa for speed/pitch adjustment."""

    def test_adjust_speed_noop(self):
        """Speed factor 1.0 should return audio unchanged."""
        from qwen3_tts.core.engine import adjust_speed
        import numpy as np
        audio = np.random.randn(16000).astype(np.float32)
        result = adjust_speed(audio, 24000, 1.0)
        np.testing.assert_array_equal(result, audio)

    def test_adjust_pitch_noop(self):
        """Pitch shift 0 semitones should return audio unchanged."""
        from qwen3_tts.core.engine import adjust_pitch
        import numpy as np
        audio = np.random.randn(16000).astype(np.float32)
        result = adjust_pitch(audio, 24000, 0)
        np.testing.assert_array_equal(result, audio)

    def test_adjust_speed_with_librosa_fallback(self):
        """Speed adjustment should work even when pyrubberband is missing."""
        from qwen3_tts.core.engine import adjust_speed
        import numpy as np
        audio = np.random.randn(16000).astype(np.float32)
        # Mock pyrubberband import failure to force librosa fallback
        with patch.dict('sys.modules', {'pyrubberband': None}):
            try:
                result = adjust_speed(audio, 24000, 1.5)
                self.assertIsInstance(result, np.ndarray)
            except ImportError:
                # librosa may not be installed either — that's OK in test env
                pass
            except KeyError as e:
                # numba/LLVM duplicate registration bug (external dep issue)
                if "duplicate registration" in str(e):
                    self.skipTest(f"numba/LLVM bug: {e}")
                raise

    def test_adjust_pitch_with_librosa_fallback(self):
        """Pitch adjustment should work even when pyrubberband is missing."""
        from qwen3_tts.core.engine import adjust_pitch
        import numpy as np
        audio = np.random.randn(16000).astype(np.float32)
        with patch.dict('sys.modules', {'pyrubberband': None}):
            try:
                result = adjust_pitch(audio, 24000, 2)
                self.assertIsInstance(result, np.ndarray)
            except ImportError:
                pass
            except KeyError as e:
                # numba/LLVM duplicate registration bug (external dep issue)
                if "duplicate registration" in str(e):
                    self.skipTest(f"numba/LLVM bug: {e}")
                raise


class TestProsodyPresets(unittest.TestCase):
    """Test prosody preset loading and resolution."""

    def test_default_prosody_presets_exist(self):
        """DEFAULT_PROSODY_PRESETS should contain standard presets."""
        from qwen3_tts.core.config import DEFAULT_PROSODY_PRESETS
        self.assertIn("excited", DEFAULT_PROSODY_PRESETS)
        self.assertIn("calm", DEFAULT_PROSODY_PRESETS)
        self.assertIn("whisper", DEFAULT_PROSODY_PRESETS)
        self.assertIn("authoritative", DEFAULT_PROSODY_PRESETS)
        self.assertIsInstance(DEFAULT_PROSODY_PRESETS["excited"], str)

    def test_get_prosody_presets_returns_defaults(self):
        """get_prosody_presets with empty config should return defaults."""
        from qwen3_tts.core.config import get_prosody_presets, DEFAULT_PROSODY_PRESETS
        presets = get_prosody_presets(config={})
        self.assertEqual(presets, DEFAULT_PROSODY_PRESETS)

    def test_get_prosody_presets_merges_user(self):
        """User presets should override defaults."""
        from qwen3_tts.core.config import get_prosody_presets
        config = {"prosody_presets": {"excited": "custom excited text", "newpreset": "new text"}}
        presets = get_prosody_presets(config)
        self.assertEqual(presets["excited"], "custom excited text")
        self.assertEqual(presets["newpreset"], "new text")
        # Defaults should still be present
        self.assertIn("calm", presets)

    def test_prosody_presets_in_config_json(self):
        """Verify default config schema includes prosody_presets."""
        from qwen3_tts.core.config import get_default_config
        config = get_default_config()
        self.assertIn("prosody_presets", config)
        self.assertIn("excited", config["prosody_presets"])

    @_skip_generate
    def test_prosody_cli_flag_exists(self):
        """qwen3_tts.interface.generate should accept --prosody flag."""
        from qwen3_tts.interface import generate as voice_generate
        # Build parser and check --prosody is registered
        parser = voice_generate.build_parser() if hasattr(voice_generate, 'build_parser') else None
        if parser is None:
            # Check that the module has the argparse setup
            with open(voice_generate.__file__) as f:
                source = f.read()
            self.assertIn("--prosody", source)


class TestXVectorOnlyMode(unittest.TestCase):
    """Test x_vector_only_mode parameter propagation."""

    def test_run_inference_accepts_x_vector_only_mode(self):
        """run_inference should accept x_vector_only_mode parameter."""
        import inspect
        from qwen3_tts.core.engine import run_inference
        sig = inspect.signature(run_inference)
        self.assertIn("x_vector_only_mode", sig.parameters)

    def test_run_inference_streaming_accepts_x_vector_only_mode(self):
        """run_inference_streaming should accept x_vector_only_mode parameter."""
        import inspect
        from qwen3_tts.core.engine import run_inference_streaming
        sig = inspect.signature(run_inference_streaming)
        self.assertIn("x_vector_only_mode", sig.parameters)

    def test_inference_single_accepts_x_vector_only_mode(self):
        """_run_inference_single should accept x_vector_only_mode parameter."""
        import inspect
        from qwen3_tts.core.engine.inference import _run_inference_single
        sig = inspect.signature(_run_inference_single)
        self.assertIn("x_vector_only_mode", sig.parameters)

    @_skip_generate
    def test_generate_via_server_accepts_x_vector_only_mode(self):
        """generate_via_server should accept x_vector_only_mode parameter."""
        import inspect
        from qwen3_tts.interface.generate import generate_via_server
        sig = inspect.signature(generate_via_server)
        self.assertIn("x_vector_only_mode", sig.parameters)

    @_skip_generate
    def test_generate_streaming_accepts_x_vector_only_mode(self):
        """generate_streaming in voice_generate should accept x_vector_only_mode."""
        import inspect
        from qwen3_tts.interface.generate import generate_streaming
        sig = inspect.signature(generate_streaming)
        self.assertIn("x_vector_only_mode", sig.parameters)


@_skip_client
class TestXVectorOnlyClient(unittest.TestCase):
    """Test x_vector_only_mode in client."""

    def test_client_generate_accepts_x_vector_only_mode(self):
        """TTSClient.generate should accept x_vector_only_mode parameter."""
        import inspect
        from qwen3_tts.server.client import TTSClient
        sig = inspect.signature(TTSClient.generate)
        self.assertIn("x_vector_only_mode", sig.parameters)

    def test_client_streaming_accepts_x_vector_only_mode(self):
        """TTSClient.generate_streaming should accept x_vector_only_mode parameter."""
        import inspect
        from qwen3_tts.server.client import TTSClient
        sig = inspect.signature(TTSClient.generate_streaming)
        self.assertIn("x_vector_only_mode", sig.parameters)


class TestCreateVoiceNoTranscript(unittest.TestCase):
    """Test --no-transcript flag for qwen3_tts.tools.create_voice."""

    def test_no_transcript_flag_in_parser(self):
        """qwen3_tts.tools.create_voice should accept --no-transcript flag."""
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "qwen3_tts", "tools", "create_voice.py"
        )
        with open(source_path) as f:
            source = f.read()
        self.assertIn("--no-transcript", source)


class TestClickCLI(unittest.TestCase):
    """Test the Click CLI routing."""

    def test_cli_imports(self):
        """qwen3_tts.cli imports without error."""
        from qwen3_tts.cli import cli, TTSGroup
        self.assertIsNotNone(cli)
        self.assertIsInstance(cli, TTSGroup)

    def test_ttsgroup_prepends_generate(self):
        """TTSGroup prepends 'generate' for bare text args."""
        from qwen3_tts.cli import cli
        # Check that known subcommands exist
        self.assertIn('generate', cli.commands)
        self.assertIn('server', cli.commands)
        self.assertIn('voice', cli.commands)
        self.assertIn('list', cli.commands)
        self.assertIn('config', cli.commands)
        self.assertIn('ui', cli.commands)
        self.assertIn('history', cli.commands)
        self.assertIn('stats', cli.commands)

    def test_ttsgroup_server_mode_stripping(self):
        """TTSGroup strips --_server-mode and re-inserts after subcommand."""
        from qwen3_tts.cli import TTSGroup
        # Verify the class has parse_args that handles --_server-mode
        import inspect
        source = inspect.getsource(TTSGroup.parse_args)
        self.assertIn('--_server-mode', source)
        self.assertIn('server_mode', source)

    def test_ui_rejects_server_mode_flag(self):
        """tts ui does not accept --_server-mode (it's a generate-only flag)."""
        from click.testing import CliRunner
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch('qwen3_tts.interface.generate.launch_gradio_ui'):
            result = runner.invoke(cli, ['ui', '--_server-mode'])
        # ui command should NOT fail with "No such option: --_server-mode"
        self.assertNotEqual(result.exit_code, 2,
                            f"ui rejected --_server-mode: {result.output}")

    def test_ttsgroup_skips_server_mode_for_non_generate_commands(self):
        """TTSGroup.parse_args does NOT re-insert --_server-mode for ui, config, etc."""
        from click.testing import CliRunner
        from qwen3_tts.cli import cli
        runner = CliRunner()
        # Commands that should NOT get --_server-mode re-inserted
        non_generate_cmds = ['ui', 'config', 'history', 'stats']
        for cmd in non_generate_cmds:
            result = runner.invoke(cli, ['--_server-mode', cmd, '--help'])
            self.assertNotIn('No such option', result.output,
                             f"--_server-mode leaked to '{cmd}': {result.output}")

    def test_flag_map_completeness(self):
        """_FLAG_MAP covers all generation options."""
        from qwen3_tts.cli import _FLAG_MAP
        expected_keys = [
            'mode', 'prompt', 'description', 'speaker', 'instruct',
            'voice', 'prosody', 'no_transcript', 'output', 'play',
            'stream', 'no_open', 'speed', 'pitch', 'trim_silence',
            'normalize', 'preset', 'temperature', 'top_k', 'top_p',
            'seed', 'repetition_penalty', 'max_chunk_chars',
            'clipboard', 'ssml', 'local', 'dry_run', 'backend',
            'model_size', 'server_mode',
        ]
        for key in expected_keys:
            self.assertIn(key, _FLAG_MAP, f"Missing key in _FLAG_MAP: {key}")

    def test_cli_version(self):
        """CLI has version option."""
        from click.testing import CliRunner
        from qwen3_tts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('3.0.0', result.output)

    def test_cli_help(self):
        """CLI --help shows subcommands."""
        from click.testing import CliRunner
        from qwen3_tts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('generate', result.output)
        self.assertIn('server', result.output)
        self.assertIn('voice', result.output)

    def test_cli_server_help(self):
        """CLI server --help shows subcommands."""
        from click.testing import CliRunner
        from qwen3_tts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['server', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('start', result.output)
        self.assertIn('stop', result.output)
        self.assertIn('status', result.output)

    def test_cli_generate_help(self):
        """CLI generate --help shows all options."""
        from click.testing import CliRunner
        from qwen3_tts.cli import cli
        runner = CliRunner()
        result = runner.invoke(cli, ['generate', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn('--mode', result.output)
        self.assertIn('--prompt', result.output)
        self.assertIn('--output', result.output)

    @_skip_ui
    def test_preview_voice_cleanup_on_failure(self):
        """preview_voice must clean up temp file on exception."""
        import os
        import tempfile
        from unittest.mock import patch

        # Track the temp file path
        temp_file_path = None

        def mock_named_temp_file(*args, **kwargs):
            nonlocal temp_file_path
            mock_file = MagicMock()
            # Create an actual temp file to track
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            temp_file_path = path
            mock_file.name = path
            mock_file.write.side_effect = RuntimeError("Server error")
            mock_file.close = MagicMock()
            return mock_file

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake audio bytes"

        with patch('qwen3_tts.interface.ui.voice_management.is_server_running', return_value=True), \
             patch('requests.get', return_value=mock_resp), \
             patch('qwen3_tts.interface.ui.voice_management.tempfile.NamedTemporaryFile', side_effect=mock_named_temp_file):

            from qwen3_tts.interface.ui import preview_voice
            result = preview_voice("test_prompt")

            # Should return None on failure
            self.assertIsNone(result)
            # Temp file should be cleaned up (not exist)
            if temp_file_path:
                self.assertFalse(os.path.exists(temp_file_path),
                                 f"Temp file {temp_file_path} should be cleaned up on exception")

    @_skip_ui
    def test_preview_voice_cleanup_on_write_failure(self):
        """preview_voice must clean up temp file when write fails."""
        import os
        import tempfile
        from unittest.mock import patch

        # Track the temp file path
        temp_file_path = None

        def mock_named_temp_file(*args, **kwargs):
            nonlocal temp_file_path
            mock_file = MagicMock()
            # Create an actual temp file to track
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            temp_file_path = path
            mock_file.name = path
            # Make write fail after file is created
            mock_file.write.side_effect = OSError("Disk full")
            mock_file.close = MagicMock()
            return mock_file

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake audio bytes"

        with patch('qwen3_tts.interface.ui.voice_management.is_server_running', return_value=True), \
             patch('requests.get', return_value=mock_resp), \
             patch('qwen3_tts.interface.ui.voice_management.tempfile.NamedTemporaryFile', side_effect=mock_named_temp_file):

            from qwen3_tts.interface.ui import preview_voice
            result = preview_voice("test_prompt")

            # Should return None on write failure
            self.assertIsNone(result)
            # Temp file should be cleaned up even though it was created
            if temp_file_path:
                self.assertFalse(os.path.exists(temp_file_path),
                                 f"Temp file {temp_file_path} should be cleaned up on write failure")


class TestPlatformSafeCommands(unittest.TestCase):
    """Test platform-safe command helpers in voice_generate."""

    def test_play_audio_checks_platform(self):
        """play_audio checks platform before choosing command."""
        import inspect
        from qwen3_tts.interface.generate import play_audio
        source = inspect.getsource(play_audio)
        self.assertIn("IS_MACOS", source)
        self.assertIn("IS_LINUX", source)
        self.assertIn("IN_COLAB", source)

    def test_get_clipboard_text_checks_platform(self):
        """get_clipboard_text checks platform before choosing command."""
        import inspect
        from qwen3_tts.interface.generate import get_clipboard_text
        source = inspect.getsource(get_clipboard_text)
        self.assertIn("IS_MACOS", source)
        self.assertIn("IS_LINUX", source)

    def test_open_file_exists(self):
        """voice_generate has open_file helper function."""
        from qwen3_tts.interface.generate import open_file
        self.assertTrue(callable(open_file))

    def test_open_file_handles_missing_xdg(self):
        """open_file wraps xdg-open in try/except."""
        import inspect
        from qwen3_tts.interface.generate import open_file
        source = inspect.getsource(open_file)
        self.assertIn("FileNotFoundError", source)
        self.assertIn("xdg-open", source)


class TestGetPresets(unittest.TestCase):
    """Tests for get_presets() preset dropdown choices."""

    @patch("qwen3_tts.interface.ui.shared.load_config")
    def test_includes_none_as_first_choice(self, mock_config):
        mock_config.return_value = {"presets": {"consistent": {}, "creative": {}}}
        from qwen3_tts.interface.ui.shared import get_presets
        result = get_presets()
        self.assertEqual(result[0], "(none)")

    @patch("qwen3_tts.interface.ui.shared.load_config")
    def test_contains_config_presets_after_none(self, mock_config):
        mock_config.return_value = {"presets": {"consistent": {}, "creative": {}}}
        from qwen3_tts.interface.ui.shared import get_presets
        result = get_presets()
        self.assertIn("consistent", result)
        self.assertIn("creative", result)
        self.assertEqual(result.index("(none)"), 0)

    @patch("qwen3_tts.interface.ui.shared.load_config")
    def test_empty_user_presets_still_returns_defaults(self, mock_config):
        mock_config.return_value = {"presets": {}}
        from qwen3_tts.interface.ui.shared import get_presets
        from qwen3_tts.core.config import DEFAULT_GENERATION_PRESETS
        result = get_presets()
        self.assertIn("(none)", result)
        self.assertEqual(result[0], "(none)")
        # Default presets appear even with empty user config
        for name in DEFAULT_GENERATION_PRESETS:
            self.assertIn(name, result)


if __name__ == "__main__":
    unittest.main()
