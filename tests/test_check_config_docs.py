"""Tests for the CONFIG.md drift checker (qwen3_tts.tools.check_config_docs)."""

import unittest
from unittest.mock import patch

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

        config_doc = Path(__file__).resolve().parents[1] / "docs" / "CONFIG.md"
        text = config_doc.read_text(encoding="utf-8")
        actual = flatten_config(get_default_config())
        self.assertEqual(
            check_drift(text, actual),
            [],
            "docs/CONFIG.md default values drifted from get_default_config(); "
            "run `python -m qwen3_tts.tools.check_config_docs --fix`.",
        )


class TestDriftRow(unittest.TestCase):
    """4B.3 item 13: Drift.as_row rendering (previously missed line 54)."""

    def test_as_row_renders_documented_and_actual(self):
        from qwen3_tts.tools.check_config_docs import Drift

        row = Drift(key="default_clone_prompt", documented="none", actual="null")
        self.assertEqual(
            row.as_row(),
            "default_clone_prompt: documented `none` → should be `null`",
        )


class TestMainExitCodes(unittest.TestCase):
    """4B.3 item 13: main() reporting/exit-code arms (missed 169-195).

    check_drift is patched so the drift outcome is controlled without
    needing a hand-written CONFIG.md that matches get_default_config().
    """

    def _existing_doc(self):
        import os
        import tempfile
        from pathlib import Path

        fd, path = tempfile.mkstemp(suffix=".md")
        with os.fdopen(fd, "w") as f:
            f.write("| `a.b` | bool | true | desc |\n")
        self.addCleanup(os.unlink, path)
        return Path(path)

    def test_missing_doc_returns_2(self):
        import contextlib
        import io
        from pathlib import Path

        from qwen3_tts.tools import check_config_docs as ccd

        with (
            patch.object(ccd, "CONFIG_DOC", Path("/nonexistent/CONFIG.md")),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()) as err,
        ):
            self.assertEqual(ccd.main([]), 2)
        self.assertIn("not found", err.getvalue())

    def test_no_drift_returns_0_and_prints_ok(self):
        import contextlib
        import io

        from qwen3_tts.tools import check_config_docs as ccd

        with (
            patch.object(ccd, "CONFIG_DOC", self._existing_doc()),
            patch.object(ccd, "check_drift", return_value=[]),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            self.assertEqual(ccd.main([]), 0)
        self.assertIn("OK", out.getvalue())

    def test_drift_returns_1_and_lists_keys(self):
        import contextlib
        import io

        from qwen3_tts.tools import check_config_docs as ccd
        from qwen3_tts.tools.check_config_docs import Drift

        with (
            patch.object(ccd, "CONFIG_DOC", self._existing_doc()),
            patch.object(
                ccd,
                "check_drift",
                return_value=[Drift(key="a.b", documented="true", actual="false")],
            ),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            self.assertEqual(ccd.main([]), 1)
        output = out.getvalue()
        self.assertIn("DRIFT", output)
        self.assertIn("a.b", output)
        self.assertNotIn("Suggested corrections", output)

    def test_drift_with_fix_flag_prints_corrections(self):
        import contextlib
        import io

        from qwen3_tts.tools import check_config_docs as ccd
        from qwen3_tts.tools.check_config_docs import Drift

        with (
            patch.object(ccd, "CONFIG_DOC", self._existing_doc()),
            patch.object(
                ccd,
                "check_drift",
                return_value=[Drift(key="a.b", documented="true", actual="false")],
            ),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            self.assertEqual(ccd.main(["--fix"]), 1)
        output = out.getvalue()
        self.assertIn("Suggested corrections", output)
        self.assertIn("a.b: documented `true` → should be `false`", output)


if __name__ == "__main__":
    unittest.main()
