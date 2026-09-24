"""Task 2.6 (Step 7B): help-text pass — epilogs, metavars, _FLAG_MAP completeness.

docs/plans/2026-09-07-interface-quality.plan.md, Phase 2 Task 2.6: `Examples:`
epilogs on ~10 key commands, Click metavars so `generate`'s usage line shows
`TEXT` instead of `TEXT...`, the `_FLAG_MAP` introspection guard that would
have caught `--list-speakers`/`--list-presets`/`--list-prosody` being
unreachable dead flags, and confirming the cache-prune epilog never
regresses the fixed `--unused 30d` typo.

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_cli_help.py -v
"""

import unittest

from click.testing import CliRunner

from qwen3_tts.cli import _FLAG_MAP, cli


class TestExamplesEpilogs(unittest.TestCase):
    """Each of the ~10 key commands documents at least one runnable example."""

    CASES = [
        (["generate", "--help"], "generate"),
        (["batch", "--help"], "batch"),
        (["srt", "--help"], "srt"),
        (["dialogue", "--help"], "dialogue"),
        (["server", "start", "--help"], "server start"),
        (["server", "stop", "--help"], "server stop"),
        (["voice", "create", "--help"], "voice create"),
        (["voice", "rebuild", "--help"], "voice rebuild"),
        (["list", "speakers", "--help"], "list speakers"),
        (["config", "edit", "--help"], "config edit"),
        (["history", "--help"], "history"),
        (["doctor", "--help"], "doctor"),
    ]

    def test_each_command_help_exits_zero_and_has_examples(self):
        runner = CliRunner()
        for args, label in self.CASES:
            with self.subTest(command=label):
                result = runner.invoke(cli, args)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Examples:", result.output)


class TestGenerateMetavar(unittest.TestCase):
    def test_usage_line_shows_text_not_ellipsis(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "--help"])
        usage_line = result.output.splitlines()[0]
        self.assertIn("TEXT", usage_line)
        self.assertNotIn("TEXT...", usage_line)


class TestCachePruneEpilogRegression(unittest.TestCase):
    def test_prune_help_never_advertises_30d(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["cache", "prune", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("30d", result.output)


class TestFlagMapCompleteness(unittest.TestCase):
    """Every _FLAG_MAP entry must resolve to a real argparse flag.

    This is the guard that would have caught --list-speakers/--list-presets/
    --list-prosody being declared in _FLAG_MAP (or as Click options) but not
    actually wired to a matching argparse flag on generate.py's parser.
    """

    def test_every_flag_map_value_is_a_real_argparse_flag(self):
        from qwen3_tts.interface.generate import _build_parser

        parser = _build_parser()
        known_flags = set(parser._option_string_actions.keys())
        for key, (flag, _typ) in _FLAG_MAP.items():
            with self.subTest(key=key):
                self.assertIn(
                    flag,
                    known_flags,
                    f"_FLAG_MAP[{key!r}] = {flag!r} has no matching argparse flag",
                )

    def test_list_speakers_presets_prosody_are_reachable_via_click(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["generate", "--help"])
        self.assertIn("--list-speakers", result.output)
        self.assertIn("--list-presets", result.output)
        self.assertIn("--list-prosody", result.output)


if __name__ == "__main__":
    unittest.main()
