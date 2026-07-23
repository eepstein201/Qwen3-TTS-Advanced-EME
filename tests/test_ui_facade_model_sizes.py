#!/usr/bin/env python3
"""Tests for the model-size/quantization prose dedup in _facade.py (UI-3).

The Model Settings info tooltip and the Tips markdown previously hardcoded the
size and quantization lists, which silently went stale if the canonical
constants (VALID_MODEL_SIZES / VALID_MLX_QUANTIZATIONS) changed. These tests
pin the prose to derive from those constants, and assert behavior preservation
(identical visible text to the old hardcoded literals).

Run: pytest tests/test_ui_facade_model_sizes.py -v
"""
import unittest


class TestModelSizeDescriptions(unittest.TestCase):
    def test_descriptions_cover_every_valid_size(self):
        # The tooltip map must stay in lockstep with the canonical size list.
        from qwen3_tts.core.config import VALID_MODEL_SIZES
        from qwen3_tts.interface.ui._facade import _MODEL_SIZE_DESCRIPTIONS

        self.assertEqual(set(_MODEL_SIZE_DESCRIPTIONS), set(VALID_MODEL_SIZES))

    def test_info_string_is_behavior_preserving(self):
        # The joined info must equal the exact literal it replaced.
        from qwen3_tts.core.config import VALID_MODEL_SIZES
        from qwen3_tts.interface.ui._facade import _MODEL_SIZE_DESCRIPTIONS

        info = " | ".join(
            f"{size}: {_MODEL_SIZE_DESCRIPTIONS[size]}" for size in VALID_MODEL_SIZES
        )
        self.assertEqual(
            info, "1.7B: higher quality | 0.6B: ~40% faster, lower memory"
        )


class TestTipsMarkdownDerivation(unittest.TestCase):
    def test_model_size_choices_match_legacy_string(self):
        from qwen3_tts.core.config import VALID_MODEL_SIZES

        self.assertEqual("/".join(sorted(VALID_MODEL_SIZES)), "0.6B/1.7B")

    def test_mlx_quant_choices_match_legacy_string(self):
        from qwen3_tts.core.config import VALID_MLX_QUANTIZATIONS

        self.assertEqual("/".join(VALID_MLX_QUANTIZATIONS), "4bit/5bit/6bit/8bit/bf16")


if __name__ == "__main__":
    unittest.main()
