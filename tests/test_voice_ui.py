"""UI tests extracted from test_voice.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.voice_test_helpers import (
    _skip_ui,
)


@_skip_ui
class TestUIHistoryFunctions(unittest.TestCase):
    """Test voice_ui generation history functions."""

    def test_history_functions_exist(self):
        """voice_ui has history-related functions."""
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(hasattr(voice_ui, "add_to_history"))
        self.assertTrue(hasattr(voice_ui, "get_history_data"))
        self.assertTrue(hasattr(voice_ui, "MAX_HISTORY_SIZE"))

    def test_add_to_history(self):
        """add_to_history adds entries to history and returns new list."""
        from qwen3_tts.interface import ui as voice_ui
        history = []

        history = voice_ui.add_to_history(history, "clone", "Test text", "/path/to/audio.wav", 5)
        self.assertEqual(len(history), 1)
        entry = history[0]
        self.assertEqual(entry["mode"], "Clone")
        self.assertEqual(entry["chunks"], 5)
        self.assertEqual(entry["path"], "/path/to/audio.wav")

    def test_history_max_size(self):
        """History doesn't exceed MAX_HISTORY_SIZE."""
        from qwen3_tts.interface import ui as voice_ui
        history = []

        # Add more than max entries
        for i in range(voice_ui.MAX_HISTORY_SIZE + 5):
            history = voice_ui.add_to_history(history, "clone", f"Text {i}", f"/path/{i}.wav", 1)

        self.assertEqual(len(history), voice_ui.MAX_HISTORY_SIZE)

    def test_add_to_history_does_not_mutate_input(self):
        """add_to_history returns a new list, does not mutate the input."""
        from qwen3_tts.interface import ui as voice_ui
        original = []
        result = voice_ui.add_to_history(original, "clone", "Test", "/path.wav", 1)
        self.assertEqual(len(original), 0)
        self.assertEqual(len(result), 1)

    def test_get_history_data_format(self):
        """get_history_data returns list of lists."""
        from qwen3_tts.interface import ui as voice_ui
        history = []
        history = voice_ui.add_to_history(history, "clone", "Test text", "/path/test.wav", 3)

        data = voice_ui.get_history_data(history)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)
        self.assertIsInstance(data[0], list)
        # Should be [time, mode, text, chunks]
        self.assertEqual(len(data[0]), 4)

    def test_history_text_truncation(self):
        """Long text is truncated in history entries."""
        from qwen3_tts.interface import ui as voice_ui
        history = []

        long_text = "A" * 100  # 100 character text
        history = voice_ui.add_to_history(history, "clone", long_text, "/path/test.wav", 1)

        entry = history[0]
        # Text should be truncated to 40 chars + "..."
        self.assertLessEqual(len(entry["text"]), 43)
        self.assertTrue(entry["text"].endswith("..."))


@_skip_ui
class TestUICancelFunction(unittest.TestCase):
    """Test voice_ui cancel streaming function."""

    def test_cancel_streaming_generation_exists(self):
        """voice_ui has cancel_streaming_generation function."""
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(hasattr(voice_ui, "cancel_streaming_generation"))
        self.assertTrue(callable(voice_ui.cancel_streaming_generation))

    def test_cancel_streaming_generation_returns_tuple(self):
        """cancel_streaming_generation returns a 2-tuple (status, status_html)."""
        from qwen3_tts.interface.ui import cancel_streaming_generation

        mock_client = MagicMock()
        mock_client.cancel_generation.return_value = {"status": "no_active_generation"}

        with patch("qwen3_tts.server.client.TTSClient", return_value=mock_client):
            result = cancel_streaming_generation()

        self.assertIsInstance(result, tuple)
        # Returns (status_text, status_html) — no audio element with WaveSurfer
        self.assertEqual(len(result), 2)

    def test_cancel_streaming_generation_status_text(self):
        """cancel_streaming_generation returns status text as first element."""
        from qwen3_tts.interface.ui import cancel_streaming_generation

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("qwen3_tts.interface.ui.generation.is_server_running", return_value=True), \
             patch("requests.post", return_value=mock_response):
            result = cancel_streaming_generation()

        # First element is status text
        self.assertIn("cancelled", result[0].lower())


@_skip_ui
class TestUITextInfo(unittest.TestCase):
    """Test voice_ui text info helper functions."""

    def test_update_text_info_exists(self):
        """voice_ui has update_text_info function."""
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(hasattr(voice_ui, "update_text_info"))

    def test_update_text_info_empty(self):
        """update_text_info returns empty string for empty input."""
        from qwen3_tts.interface.ui import update_text_info
        self.assertEqual(update_text_info(""), "")
        self.assertEqual(update_text_info(None), "")

    def test_update_text_info_short(self):
        """update_text_info shows char count for short text."""
        from qwen3_tts.interface.ui import update_text_info
        result = update_text_info("Hello")
        self.assertIn("5 chars", result)

    def test_update_text_info_long(self):
        """update_text_info shows chunks estimate for long text."""
        from qwen3_tts.interface.ui import update_text_info
        long_text = "A" * 1000  # 1000 chars = ~2 chunks
        result = update_text_info(long_text)
        self.assertIn("1000 chars", result)
        self.assertIn("chunks", result)


@_skip_ui
class TestUIModelSettings(unittest.TestCase):
    """Test voice_ui model settings functions (Phase 19: MLX-First Architecture)."""

    def test_model_settings_functions_exist(self):
        """voice_ui has model settings functions."""
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(hasattr(voice_ui, "get_current_model_settings"))
        self.assertTrue(hasattr(voice_ui, "apply_model_settings"))
        self.assertTrue(callable(voice_ui.get_current_model_settings))
        self.assertTrue(callable(voice_ui.apply_model_settings))

    def test_get_current_model_settings_returns_tuple(self):
        """get_current_model_settings returns a 3-tuple."""
        from qwen3_tts.interface.ui import get_current_model_settings
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
        from qwen3_tts.interface.ui import apply_model_settings
        # Without server running, should return error message
        result = apply_model_settings("1.7B", "8bit")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        msg, html = result
        self.assertIsInstance(msg, str)
        self.assertIsInstance(html, str)

    def test_apply_model_settings_requires_server(self):
        """apply_model_settings returns error when server not running."""
        from qwen3_tts.interface.ui import apply_model_settings
        with unittest.mock.patch("qwen3_tts.interface.ui.shared.is_server_running", return_value=False):
            msg, _ = apply_model_settings("0.6B", "4bit")
        self.assertIn("not running", msg.lower())


@_skip_ui
class TestUIModelSettingsImports(unittest.TestCase):
    """Test voice_ui imports required for model settings."""

    def test_model_settings_imports(self):
        """voice_ui imports required constants for model settings."""
        from qwen3_tts.interface import ui as voice_ui
        # Should have imported these from qwen3_tts.core.config
        self.assertTrue(hasattr(voice_ui, "VALID_MODEL_SIZES"))
        self.assertTrue(hasattr(voice_ui, "VALID_MLX_QUANTIZATIONS"))
        self.assertTrue(hasattr(voice_ui, "get_backend"))
        self.assertTrue(hasattr(voice_ui, "get_model_size"))
        self.assertTrue(hasattr(voice_ui, "get_mlx_quantization"))


@_skip_ui
class TestVoiceManagementUI(unittest.TestCase):
    """Test voice management UI helper functions."""

    def test_get_prompt_table_data_exists(self):
        """voice_ui has get_prompt_table_data function."""
        from qwen3_tts.interface.ui import get_prompt_table_data
        self.assertTrue(callable(get_prompt_table_data))

    def test_preview_voice_exists(self):
        """voice_ui has preview_voice function."""
        from qwen3_tts.interface.ui import preview_voice
        self.assertTrue(callable(preview_voice))

    def test_rename_voice_exists(self):
        """voice_ui has rename_voice function."""
        from qwen3_tts.interface.ui import rename_voice
        self.assertTrue(callable(rename_voice))

    def test_delete_voice_exists(self):
        """voice_ui has delete_voice function."""
        from qwen3_tts.interface.ui import delete_voice
        self.assertTrue(callable(delete_voice))

    def test_set_voice_default_exists(self):
        """voice_ui has set_voice_default function."""
        from qwen3_tts.interface.ui import set_voice_default
        self.assertTrue(callable(set_voice_default))

    def test_delete_voice_prompt_rejects_path_traversal(self):
        """delete_voice_prompt must reject names with .. or /"""
        from qwen3_tts.interface.generate import delete_voice_prompt
        result = delete_voice_prompt("../evil_file")
        self.assertFalse(result, "Expected False for traversal name '../evil_file'")

    def test_rename_voice_prompt_rejects_path_traversal(self):
        """rename_voice_prompt must reject names with .. or /"""
        from qwen3_tts.interface.generate import rename_voice_prompt
        result = rename_voice_prompt("../evil", "safe_name")
        self.assertFalse(result, "Expected False for traversal name '../evil'")


@_skip_ui
class TestManageModelsUI(unittest.TestCase):
    """Test Manage Models UI helper functions."""

    def test_get_model_table_data_exists(self):
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(callable(getattr(voice_ui, "get_model_table_data", None)))

    def test_toggle_model_exists(self):
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(callable(getattr(voice_ui, "toggle_model", None)))

    def test_get_model_status_html_exists(self):
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(callable(getattr(voice_ui, "get_model_status_html", None)))

    def test_update_startup_defaults_exists(self):
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(callable(getattr(voice_ui, "update_startup_defaults", None)))

    def test_get_audio_loader_setting_exists(self):
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(callable(getattr(voice_ui, "get_audio_loader_setting", None)))

    def test_set_audio_loader_setting_exists(self):
        from qwen3_tts.interface import ui as voice_ui
        self.assertTrue(callable(getattr(voice_ui, "set_audio_loader_setting", None)))


@_skip_ui
class TestProsodyUI(unittest.TestCase):
    """Test prosody preset UI helpers."""

    def test_get_prosody_choices_function(self):
        """get_prosody_choices should return list with (none) first."""
        from qwen3_tts.interface import ui as voice_ui
        choices = voice_ui.get_prosody_choices()
        self.assertIsInstance(choices, list)
        self.assertEqual(choices[0], "(none)")
        self.assertTrue(len(choices) > 1)

    def test_apply_prosody_preset_none(self):
        """Selecting (none) should return empty string."""
        from qwen3_tts.interface import ui as voice_ui
        result = voice_ui.apply_prosody_preset("(none)")
        self.assertEqual(result, "")

    def test_apply_prosody_preset_valid(self):
        """Selecting a valid preset should return its instruction text."""
        from qwen3_tts.interface import ui as voice_ui
        from qwen3_tts.core.config import DEFAULT_PROSODY_PRESETS
        result = voice_ui.apply_prosody_preset("excited")
        self.assertEqual(result, DEFAULT_PROSODY_PRESETS["excited"])


if __name__ == "__main__":
    unittest.main()
