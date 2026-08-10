"""Tests for the CONFIG.md drift checker (qwen3_tts.tools.check_config_docs)."""

import unittest

from qwen3_tts.tools.check_config_docs import (
    check_drift,
    flatten_config,
    normalize,
    parse_defaults_from_markdown,
)

# A minimal CONFIG.md table exercising the comparison cases:
# exact bare value, quoted value, platform-dependent value with prose,
# numeric, bool, null, and a multi-word string default.
SAMPLE_MD = """
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `server.port` | integer | `5123` | Server port. |
| `server.host` | string | `"127.0.0.1"` | Bind address. |
| `advanced.backend` | string | `"mlx"` (Apple Silicon), `"torch"` elsewhere | Backend. |
| `generation.silence_gap_seconds` | float | `0.0` | Gap between chunks. |
| `models.clone.load_at_startup` | boolean | `true` | Load at startup. |
| `default_clone_prompt` | string/null | `null` | Auto-scans voice_prompts/. |
| `default_voice_description` | string | `"A calm, friendly male voice."` | Design default. |
"""

# Defaults that AGREE with SAMPLE_MD.
AGREEING = {
    "server.port": 5123,
    "server.host": "127.0.0.1",
    "advanced.backend": "mlx",  # also documented as "torch" alternative
    "generation.silence_gap_seconds": 0.0,
    "models.clone.load_at_startup": True,
    "default_clone_prompt": None,
    "default_voice_description": "A calm, friendly male voice.",
}


class TestNormalize(unittest.TestCase):
    def test_bool_renders_lowercase(self):
        self.assertEqual(normalize(True), "true")
        self.assertEqual(normalize(False), "false")

    def test_none_renders_null(self):
        self.assertEqual(normalize(None), "null")

    def test_string_strips_quotes_and_backticks(self):
        self.assertEqual(normalize("`8bit`"), "8bit")
        self.assertEqual(normalize('"mlx"'), "mlx")
        self.assertEqual(normalize("20/minute"), "20/minute")


class TestFlattenConfig(unittest.TestCase):
    def test_nested_dict_flattens_to_dotted_paths(self):
        cfg = {"server": {"port": 5123, "host": "127.0.0.1"}, "language": "English"}
        flat = flatten_config(cfg)
        self.assertEqual(flat["server.port"], 5123)
        self.assertEqual(flat["server.host"], "127.0.0.1")
        self.assertEqual(flat["language"], "English")

    def test_empty_leaf_dict_produces_no_scalars(self):
        flat = flatten_config({"aliases": {}})
        self.assertNotIn("aliases", flat)


class TestParseDefaults(unittest.TestCase):
    def test_extracts_key_to_default_cell(self):
        parsed = parse_defaults_from_markdown(SAMPLE_MD)
        self.assertIn("server.port", parsed)
        self.assertIn("default_voice_description", parsed)
        # Non-table lines are ignored.
        self.assertEqual(len(parsed), 7)


class TestCheckDrift(unittest.TestCase):
    def test_no_drift_when_defaults_agree(self):
        self.assertEqual(check_drift(SAMPLE_MD, AGREEING), [])

    def test_detects_wrong_scalar_default(self):
        drifted = dict(AGREEING, server_port_unused=0)
        drifted["server.port"] = 8080  # documented 5123, actual 8080
        mismatches = check_drift(SAMPLE_MD, drifted)
        keys = {d.key for d in mismatches}
        self.assertIn("server.port", keys)

    def test_detects_reintroduced_default_clone_prompt(self):
        # The exact regression this tool exists for: someone re-documents the
        # old non-null default.
        bad_md = SAMPLE_MD.replace(
            "| `default_clone_prompt` | string/null | `null` |",
            '| `default_clone_prompt` | string | `"default_clone.pt"` |',
        )
        mismatches = check_drift(bad_md, AGREEING)
        keys = {d.key for d in mismatches}
        self.assertIn("default_clone_prompt", keys)

    def test_numeric_equivalence_zero_variants(self):
        # 0 vs 0.0 must NOT be flagged.
        self.assertEqual(check_drift(SAMPLE_MD, AGREEING), [])

    def test_platform_alternative_matches(self):
        # On Linux the backend default is "torch", which is documented as the
        # second quoted alternative — must not be flagged.
        linux_defaults = dict(AGREEING, **{"advanced.backend": "torch"})
        self.assertEqual(check_drift(SAMPLE_MD, linux_defaults), [])

    def test_keys_only_in_prose_are_skipped(self):
        # A key in the defaults dict but absent from the markdown is not flagged.
        only_code = dict(AGREEING, some_future_key=123)
        self.assertEqual(check_drift(SAMPLE_MD, only_code), [])


class TestRealConfigMd(unittest.TestCase):
    """Regression contract: the shipped CONFIG.md must match get_default_config."""

    def test_no_drift_in_shipped_config_md(self):
        from pathlib import Path

        from qwen3_tts.core.config import get_default_config

        config_doc = (
            Path(__file__).resolve().parents[1] / "docs" / "CONFIG.md"
        )
        text = config_doc.read_text(encoding="utf-8")
        actual = flatten_config(get_default_config())
        self.assertEqual(
            check_drift(text, actual),
            [],
            "docs/CONFIG.md default values drifted from get_default_config(); "
            "run `python -m qwen3_tts.tools.check_config_docs --fix`.",
        )


if __name__ == "__main__":
    unittest.main()
