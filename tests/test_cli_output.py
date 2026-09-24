"""Task 2.1 (Step 7B): canonical CLI output/formatting module.

``qwen3_tts.cli_output`` is the single output idiom for the whole CLI
(docs/plans/2026-09-07-interface-quality.plan.md, Phase 2 Task 2.1): pure
formatters (``fmt_*`` — spelling-identical to the UI's shared.py helpers),
a ``Column``/``render_table`` pair whose separator widths derive from
``Column.width``, tty-guarded color, and severity wrappers whose glyphs are
separate prefix elements so pinned literal substrings stay contiguous.

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_cli_output.py -v
"""

import io
import os
import re
import unittest
from unittest.mock import patch

from qwen3_tts import cli_output
from qwen3_tts.tools import _shared

ANSI_RE = re.compile(r"\x1b\[")


class _FakeStream(io.StringIO):
    """A stream whose tty-ness is scripted; StringIO defaults to non-tty."""

    def __init__(self, tty: bool):
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class TestSymbols(unittest.TestCase):
    def test_symbol_vocabulary(self):
        self.assertEqual(
            cli_output.SYMBOLS,
            {"success": "✓", "warning": "⚠", "error": "✗", "info": "ℹ"},
        )


class TestFmtDuration(unittest.TestCase):
    def test_spelling_matches_the_ui_helper(self):
        cases = {
            0: "0:00",
            12.4: "0:12",
            59.9: "1:00",  # rounds before splitting, like shared.fmt_duration
            65: "1:05",
            3599: "59:59",
            3712: "1:01:52",
        }
        for seconds, expected in cases.items():
            with self.subTest(seconds=seconds):
                self.assertEqual(cli_output.fmt_duration(seconds), expected)

    def test_unknown_values_render_the_em_dash(self):
        for seconds in (None, -1, float("inf"), float("nan")):
            with self.subTest(seconds=seconds):
                self.assertEqual(cli_output.fmt_duration(seconds), "—")


class TestFmtBytes(unittest.TestCase):
    def test_one_decimal_units(self):
        cases = {
            0: "0.0 B",
            512: "512.0 B",
            1536: "1.5 KB",
            5 * 1024**2: "5.0 MB",
            2 * 1024**3: "2.0 GB",
            3 * 1024**4: "3.0 TB",
        }
        for num, expected in cases.items():
            with self.subTest(num=num):
                self.assertEqual(cli_output.fmt_bytes(num), expected)

    def test_unknown_values_render_the_em_dash(self):
        for num in (None, -1, float("inf"), float("nan")):
            with self.subTest(num=num):
                self.assertEqual(cli_output.fmt_bytes(num), "—")

    def test_drift_guard_against_tools_shared_format_size(self):
        # T2.3 makes _format_size an alias of fmt_bytes; equality must hold
        # from the first commit, not only after the delegation.
        for num in (0, 512, 1024, 1536, 1024**2, 1024**3, 1024**4, 1024**5):
            with self.subTest(num=num):
                self.assertEqual(cli_output.fmt_bytes(num), _shared._format_size(num))


class TestFmtEta(unittest.TestCase):
    def test_single_eta_spelling(self):
        cases = {
            0: "~0s",
            12: "~12s",
            59: "~59s",
            60: "~1m",
            80: "~1m 20s",
            3600: "~1h",
            3900: "~1h 5m",
        }
        for seconds, expected in cases.items():
            with self.subTest(seconds=seconds):
                self.assertEqual(cli_output.fmt_eta(seconds), expected)

    def test_unknown_values_render_the_em_dash(self):
        for seconds in (None, -1, float("inf"), float("nan")):
            with self.subTest(seconds=seconds):
                self.assertEqual(cli_output.fmt_eta(seconds), "—")


class TestColorEnabled(unittest.TestCase):
    def _check(self, tty: bool, expect: bool, no_color: str | None) -> None:
        with patch.dict(os.environ):
            if no_color is None:
                os.environ.pop("NO_COLOR", None)
            else:
                os.environ["NO_COLOR"] = no_color
            self.assertEqual(cli_output.color_enabled(_FakeStream(tty)), expect)

    def test_tty_without_no_color_enables_color(self):
        self._check(tty=True, expect=True, no_color=None)

    def test_any_nonempty_no_color_disables_color(self):
        for value in ("1", "false", "0"):
            with self.subTest(value=value):
                self._check(tty=True, expect=False, no_color=value)

    def test_empty_no_color_keeps_color_enabled(self):
        # os.environ.get("NO_COLOR") is falsy when empty — the pinned rule.
        self._check(tty=True, expect=True, no_color="")

    def test_non_tty_disables_color_regardless(self):
        self._check(tty=False, expect=False, no_color=None)


class TestPaint(unittest.TestCase):
    def test_enabled_tty_returns_styled_text(self):
        with patch.dict(os.environ, {}):
            os.environ.pop("NO_COLOR", None)
            stream = _FakeStream(tty=True)
            styled = cli_output.paint(stream, "hello", "red")
        self.assertIn("hello", styled)
        self.assertRegex(styled, r"\x1b\[")

    def test_disabled_returns_the_text_unchanged(self):
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            plain = cli_output.paint(_FakeStream(tty=True), "hello", "red", "bold")
        self.assertEqual(plain, "hello")

    def test_non_tty_returns_the_text_unchanged(self):
        with patch.dict(os.environ, {}):
            os.environ.pop("NO_COLOR", None)
            plain = cli_output.paint(_FakeStream(tty=False), "hello", "red")
        self.assertEqual(plain, "hello")


class TestSeverityWrappers(unittest.TestCase):
    def _capture(self, fn, *args, **kwargs):
        out, err = io.StringIO(), io.StringIO()
        with (
            patch("sys.stdout", out),
            patch("sys.stderr", err),
        ):
            fn(*args, **kwargs)
        return out.getvalue(), err.getvalue()

    def test_success_warn_info_write_stdout_with_glyph_prefix(self):
        for fn, glyph in (
            (cli_output.success, "✓"),
            (cli_output.warn, "⚠"),
            (cli_output.info, "ℹ"),
        ):
            with self.subTest(fn=fn.__name__):
                out, err = self._capture(fn, "Saved preset")
                self.assertEqual(err, "")
                self.assertTrue(out.startswith(glyph))
                self.assertIn("Saved preset", out)

    def test_error_always_writes_stderr(self):
        out, err = self._capture(cli_output.error, "Error: server not running")
        self.assertEqual(out, "")
        self.assertTrue(err.startswith("✗"))
        self.assertIn("Error: server not running", err)

    def test_message_stays_contiguous_after_the_glyph(self):
        # Pinned literals ("Error: …", "No presets configured") must survive
        # adoption verbatim — glyph is a separate prefix element.
        _, err = self._capture(cli_output.error, "Error: voice not found")
        self.assertIn("Error: voice not found", err)

    def test_explicit_file_wins_over_the_default_stream(self):
        buf = io.StringIO()
        out, err = self._capture(cli_output.error, "boom", file=buf)
        self.assertEqual((out, err), ("", ""))
        self.assertIn("boom", buf.getvalue())

    def test_no_ansi_escapes_on_a_non_tty_stream(self):
        for fn in (cli_output.success, cli_output.warn, cli_output.error):
            with self.subTest(fn=fn.__name__):
                out, err = self._capture(fn, "plain")
                self.assertIsNone(ANSI_RE.search(out + err))


class TestHeader(unittest.TestCase):
    def test_shape_matches_tools_shared_print_header(self):
        # T2.3 turns print_header into a thin delegate of this function —
        # byte-identical output keeps tests/test_model_cache_commands green.
        buf = io.StringIO()
        cli_output.header("Model Cache", file=buf)
        self.assertEqual(
            buf.getvalue(), "\n" + "=" * 60 + "\n  Model Cache\n" + "=" * 60 + "\n"
        )

    def test_rule_and_width_are_honored(self):
        buf = io.StringIO()
        cli_output.header("Voices", rule="-", width=10, file=buf)
        self.assertEqual(
            buf.getvalue(), "\n" + "-" * 10 + "\n  Voices\n" + "-" * 10 + "\n"
        )

    def test_default_stream_is_stdout(self):
        out = io.StringIO()
        with patch("sys.stdout", out):
            cli_output.header("Cache")
        self.assertIn("Cache", out.getvalue())


class TestKvLine(unittest.TestCase):
    def test_pads_key_to_the_default_width(self):
        buf = io.StringIO()
        cli_output.kv_line("Backend", "mlx", file=buf)
        self.assertEqual(buf.getvalue(), "Backend".ljust(22) + "mlx\n")

    def test_key_width_is_configurable(self):
        buf = io.StringIO()
        cli_output.kv_line("Backend", "torch", key_width=10, file=buf)
        self.assertEqual(buf.getvalue(), "Backend".ljust(10) + "torch\n")

    def test_overlong_key_is_not_truncated(self):
        buf = io.StringIO()
        key = "a-very-long-setting-name"
        cli_output.kv_line(key, "on", file=buf)
        self.assertEqual(buf.getvalue(), key + "on\n")


class TestColumn(unittest.TestCase):
    def test_frozen_dataclass_with_align_default(self):
        from dataclasses import FrozenInstanceError

        col = cli_output.Column(key="name", title="Name", width=10)
        self.assertEqual(col.align, "left")
        with self.assertRaises(FrozenInstanceError):
            col.width = 20  # type: ignore[misc]


class TestRenderTable(unittest.TestCase):
    def _columns(self):
        return (
            cli_output.Column("name", "Name", 10),
            cli_output.Column("size", "Size", 8, align="right"),
        )

    def test_pure_function_writes_nothing(self):
        out, err = io.StringIO(), io.StringIO()
        with (
            patch("sys.stdout", out),
            patch("sys.stderr", err),
        ):
            rendered = cli_output.render_table(
                self._columns(), [{"name": "a", "size": 1}]
            )
        self.assertEqual(out.getvalue() + err.getvalue(), "")
        self.assertIsInstance(rendered, str)

    def test_alignment_indent_and_separator_widths(self):
        rendered = cli_output.render_table(
            self._columns(),
            [
                {"name": "voice_a", "size": 1024},
                {"name": "voice_b", "size": 5},
            ],
        )
        expected = "\n".join(
            [
                "  Name            Size",
                "  ----------  --------",
                "  voice_a         1024",
                "  voice_b            5",
            ]
        )
        self.assertEqual(rendered, expected)

    def test_separator_width_comes_from_column_width_not_title_length(self):
        populated = cli_output.render_table(
            self._columns(), [{"name": "x", "size": 1}]
        ).splitlines()
        # "Name"/"Size" are 4 chars wide; the dashes must run the full
        # Column.width (10/8) — the model_cache separator bug this replaces.
        self.assertEqual(populated[1].strip(), "-" * 10 + "  " + "-" * 8)

    def test_empty_rows_with_message_return_the_message_verbatim(self):
        rendered = cli_output.render_table(
            self._columns(), [], empty_message="No presets configured"
        )
        self.assertEqual(rendered, "No presets configured")

    def test_empty_rows_without_message_return_empty_string(self):
        self.assertEqual(cli_output.render_table(self._columns(), []), "")


if __name__ == "__main__":
    unittest.main()
