#!/usr/bin/env python3
"""Tests for the generation-defaults single source of truth (UI-3).

Covers:
  - DEFAULT_GENERATION_PARAMS content
  - get_default_config()["generation"] stays in lockstep with the constant
    (so the config defaults, the validate_config clamp, and the UI can't drift)
  - get_generation_defaults(): reads config, falls back per missing key, falls
    back when the whole section is missing, and falls back entirely when
    load_config() raises
  - _build_common_controls(): the four gr.Slider value= defaults are driven by
    get_generation_defaults() (guards against a silent revert to hardcoded
    0.7 / 50 / 0.95 / 1.05)

Run: pytest tests/test_generation_defaults.py -v
"""
import unittest
from unittest.mock import patch

try:
    import gradio as gr

    HAS_GRADIO = True
except ImportError:  # pragma: no cover - gradio is an optional extra
    HAS_GRADIO = False


class TestDefaultGenerationParams(unittest.TestCase):
    """The constant is the canonical source of the four sampling defaults."""

    def test_constant_exposes_exactly_the_four_ui_params(self):
        from qwen3_tts.core.config import DEFAULT_GENERATION_PARAMS

        self.assertEqual(
            set(DEFAULT_GENERATION_PARAMS),
            {"temperature", "top_k", "top_p", "repetition_penalty"},
        )

    def test_constant_values_match_legacy_hardcoded_defaults(self):
        # Behavior-preserving: these are the exact values that used to be
        # hardcoded in generation.py (sliders + payload fallbacks).
        from qwen3_tts.core.config import DEFAULT_GENERATION_PARAMS

        self.assertEqual(DEFAULT_GENERATION_PARAMS["temperature"], 0.7)
        self.assertEqual(DEFAULT_GENERATION_PARAMS["top_k"], 50)
        self.assertEqual(DEFAULT_GENERATION_PARAMS["top_p"], 0.95)
        self.assertEqual(DEFAULT_GENERATION_PARAMS["repetition_penalty"], 1.05)

    def test_default_config_generation_block_matches_constant(self):
        # get_default_config()["generation"] must be driven by the constant,
        # so the config defaults can never drift from it.
        from qwen3_tts.core.config import DEFAULT_GENERATION_PARAMS, get_default_config

        generation = get_default_config()["generation"]
        for key, value in DEFAULT_GENERATION_PARAMS.items():
            self.assertEqual(
                generation[key],
                value,
                f"{key} drifted from DEFAULT_GENERATION_PARAMS",
            )


class TestGetGenerationDefaults(unittest.TestCase):
    """The getter reads the user's config and degrades gracefully."""

    def test_reads_present_values_from_config(self):
        with patch(
            "qwen3_tts.core.config.load_config",
            return_value={"generation": {"temperature": 0.42, "top_k": 11}},
        ):
            from qwen3_tts.core.config import get_generation_defaults

            defaults = get_generation_defaults()
        self.assertEqual(defaults["temperature"], 0.42)
        self.assertEqual(defaults["top_k"], 11)

    def test_falls_back_per_missing_key(self):
        # top_k/top_p/repetition_penalty absent from config -> fall back to the
        # constant; the one present key (temperature) is honored.
        with patch(
            "qwen3_tts.core.config.load_config",
            return_value={"generation": {"temperature": 0.42}},
        ):
            from qwen3_tts.core.config import (
                DEFAULT_GENERATION_PARAMS,
                get_generation_defaults,
            )

            defaults = get_generation_defaults()
        self.assertEqual(defaults["temperature"], 0.42)
        self.assertEqual(defaults["top_k"], DEFAULT_GENERATION_PARAMS["top_k"])
        self.assertEqual(defaults["top_p"], DEFAULT_GENERATION_PARAMS["top_p"])
        self.assertEqual(
            defaults["repetition_penalty"],
            DEFAULT_GENERATION_PARAMS["repetition_penalty"],
        )

    def test_falls_back_when_generation_section_missing(self):
        with patch(
            "qwen3_tts.core.config.load_config",
            return_value={"advanced": {"backend": "mlx"}},
        ):
            from qwen3_tts.core.config import (
                DEFAULT_GENERATION_PARAMS,
                get_generation_defaults,
            )

            defaults = get_generation_defaults()
        self.assertEqual(defaults, dict(DEFAULT_GENERATION_PARAMS))

    def test_falls_back_entirely_on_load_error(self):
        # Corrupt/missing config.json must not break UI default resolution.
        with patch("qwen3_tts.core.config.load_config", side_effect=OSError("no file")):
            from qwen3_tts.core.config import (
                DEFAULT_GENERATION_PARAMS,
                get_generation_defaults,
            )

            defaults = get_generation_defaults()
        self.assertEqual(defaults, dict(DEFAULT_GENERATION_PARAMS))

    def test_only_returns_the_four_ui_params(self):
        # The getter must not leak the non-UI generation keys (seed,
        # max_chunk_chars, ...) into the slider/payload layer.
        with patch(
            "qwen3_tts.core.config.load_config",
            return_value={
                "generation": {
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.95,
                    "repetition_penalty": 1.05,
                    "seed": 42,
                    "max_chunk_chars": 999,
                    "max_new_tokens": 1,
                }
            },
        ):
            from qwen3_tts.core.config import get_generation_defaults

            defaults = get_generation_defaults()
        self.assertEqual(set(defaults), {"temperature", "top_k", "top_p", "repetition_penalty"})


@unittest.skipUnless(HAS_GRADIO, "gradio not installed")
class TestCommonControlsWiring(unittest.TestCase):
    """The Gradio sliders must derive value= from get_generation_defaults()."""

    def test_sliders_derive_value_from_get_generation_defaults(self):
        from qwen3_tts.interface.ui import generation as gen_mod

        fake = {
            "temperature": 0.33,
            "top_k": 9,
            "top_p": 0.5,
            "repetition_penalty": 1.21,
        }
        with patch.object(gen_mod, "get_generation_defaults", return_value=fake):
            with gr.Blocks():
                controls = gen_mod._build_common_controls()
        self.assertEqual(controls["temp"].value, 0.33)
        self.assertEqual(controls["top_k"].value, 9)
        self.assertEqual(controls["top_p"].value, 0.5)
        self.assertEqual(controls["rep"].value, 1.21)


if __name__ == "__main__":
    unittest.main()
