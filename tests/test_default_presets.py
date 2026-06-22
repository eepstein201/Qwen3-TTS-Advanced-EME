#!/usr/bin/env python3
"""Tests for default generation presets feature.

Covers:
  - DEFAULT_GENERATION_PRESETS: exists in config with >= 5 entries
  - get_generation_presets(): merges defaults with user config (user overrides win)
  - get_presets(): returns defaults even when config.json has no presets key
  - _prepare_streaming_config: applies default presets when selected

Run: pytest tests/test_default_presets.py -v
"""
import unittest
from unittest.mock import patch


class TestDefaultGenerationPresetsConstant(unittest.TestCase):
    """Tests for the DEFAULT_GENERATION_PRESETS constant in config.py."""

    def test_constant_exists(self):
        from qwen3_tts.core.config import DEFAULT_GENERATION_PRESETS
        self.assertIsInstance(DEFAULT_GENERATION_PRESETS, dict)

    def test_has_at_least_5_presets(self):
        from qwen3_tts.core.config import DEFAULT_GENERATION_PRESETS
        self.assertGreaterEqual(len(DEFAULT_GENERATION_PRESETS), 5)

    def test_each_preset_is_dict_with_valid_keys(self):
        from qwen3_tts.core.config import DEFAULT_GENERATION_PRESETS
        valid_keys = {"temperature", "top_k", "top_p", "repetition_penalty", "seed"}
        for name, params in DEFAULT_GENERATION_PRESETS.items():
            self.assertIsInstance(params, dict, f"Preset '{name}' must be a dict")
            self.assertTrue(
                set(params.keys()).issubset(valid_keys),
                f"Preset '{name}' has unexpected keys: {set(params.keys()) - valid_keys}"
            )
            if "temperature" in params:
                self.assertGreater(params["temperature"], 0)
                self.assertLessEqual(params["temperature"], 2.0)

    def test_known_preset_names_present(self):
        from qwen3_tts.core.config import DEFAULT_GENERATION_PRESETS
        # At minimum these common-use presets should exist
        expected = {"stable", "natural", "expressive"}
        self.assertTrue(
            expected.issubset(set(DEFAULT_GENERATION_PRESETS.keys())),
            f"Missing expected presets: {expected - set(DEFAULT_GENERATION_PRESETS.keys())}"
        )


class TestGetGenerationPresets(unittest.TestCase):
    """Tests for get_generation_presets() merge behavior."""

    def test_returns_defaults_when_no_user_presets(self):
        from qwen3_tts.core.config import (
            DEFAULT_GENERATION_PRESETS,
            get_generation_presets,
        )
        with patch("qwen3_tts.core.config.load_config", return_value={}):
            result = get_generation_presets()
        self.assertEqual(result, DEFAULT_GENERATION_PRESETS)

    def test_user_presets_override_defaults(self):
        from qwen3_tts.core.config import get_generation_presets
        user_cfg = {"presets": {"stable": {"temperature": 0.1}}}
        with patch("qwen3_tts.core.config.load_config", return_value=user_cfg):
            result = get_generation_presets()
        self.assertEqual(result["stable"]["temperature"], 0.1)

    def test_user_presets_added_alongside_defaults(self):
        from qwen3_tts.core.config import (
            DEFAULT_GENERATION_PRESETS,
            get_generation_presets,
        )
        user_cfg = {"presets": {"my_custom": {"temperature": 0.6}}}
        with patch("qwen3_tts.core.config.load_config", return_value=user_cfg):
            result = get_generation_presets()
        self.assertIn("my_custom", result)
        # Defaults still present
        for key in DEFAULT_GENERATION_PRESETS:
            self.assertIn(key, result)

    def test_accepts_config_directly(self):
        from qwen3_tts.core.config import get_generation_presets
        result = get_generation_presets(config={"presets": {"foo": {"temperature": 0.5}}})
        self.assertIn("foo", result)


class TestGetPresetsIncludesDefaults(unittest.TestCase):
    """Tests that get_presets() in shared.py returns default presets."""

    def test_includes_defaults_when_no_user_presets(self):
        from qwen3_tts.core.config import DEFAULT_GENERATION_PRESETS
        from qwen3_tts.interface.ui.shared import get_presets
        with patch("qwen3_tts.interface.ui.shared.load_config", return_value={}):
            result = get_presets()
        self.assertIn("(none)", result)
        for name in DEFAULT_GENERATION_PRESETS:
            self.assertIn(name, result)

    def test_user_presets_appear_alongside_defaults(self):
        from qwen3_tts.interface.ui.shared import get_presets
        with patch("qwen3_tts.interface.ui.shared.load_config",
                   return_value={"presets": {"my_preset": {"temperature": 0.5}}}):
            result = get_presets()
        self.assertIn("my_preset", result)
        self.assertIn("(none)", result)


class TestPresetAppliedInStreamingConfig(unittest.TestCase):
    """Tests that preset params are applied in _prepare_streaming_config."""

    @patch("qwen3_tts.interface.ui.generation.load_config")
    @patch("qwen3_tts.interface.ui.generation.is_server_running", return_value=True)
    def test_default_preset_applied(self, _mock_srv, mock_cfg):
        from qwen3_tts.core.config import DEFAULT_GENERATION_PRESETS
        from qwen3_tts.interface.ui.generation import _prepare_streaming_config
        mock_cfg.return_value = {}  # no user presets — defaults should kick in
        preset_name = next(iter(DEFAULT_GENERATION_PRESETS))
        preset_params = DEFAULT_GENERATION_PRESETS[preset_name]

        result, status = _prepare_streaming_config(
            "clone", "hello", preset_name,
            0.7, 50, 0.95, 1.05, "",
            prompt_file="test.wav",
        )
        self.assertIsNotNone(result, f"Expected config, got error: {status}")
        payload = result.get("payload", {})
        for key, val in preset_params.items():
            if key in ("temperature", "top_k", "top_p", "repetition_penalty"):
                self.assertEqual(payload.get(key), val,
                                 f"Preset key '{key}' not applied: {payload.get(key)} != {val}")


if __name__ == "__main__":
    unittest.main()
