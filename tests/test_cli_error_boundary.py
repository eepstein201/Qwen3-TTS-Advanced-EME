"""Task 2.2 (Step 7B): CLI exception boundary + exit-code contract.

Exit codes: 0 success · 1 error · 2 completed via the server (reserved,
treat as success — pinned deliberately by tests/test_cli_commands.py).
``TTSGroup.invoke`` is the CLI-wide boundary: TTSError/TTSGenericError
render via ``cli_output.error`` (first production caller of
``TTSError.format_cli()``) and exit 1; Click's own UsageError/Abort pass
through untouched.

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_cli_error_boundary.py -v
"""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from qwen3_tts.cli import cli
from qwen3_tts.core.config import InvalidInputError, TTSError
from qwen3_tts.interface.cli.srt import process_srt_file
from qwen3_tts.interface.generate_server import TTSGenericError


def _make_srt_args(**overrides):
    defaults = dict(
        output=None,
        mode=None,
        prompt=None,
        description=None,
        trim_silence=False,
        normalize=False,
        speed=None,
        pitch=None,
        play=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestCLIErrorBoundary(unittest.TestCase):
    def _invoke_generate(self, main):
        with patch("qwen3_tts.interface.generate.main", main):
            return CliRunner().invoke(cli, ["generate", "hello"])

    def test_tts_error_exits_1_with_format_cli_suggestion_on_stderr(self):
        main = MagicMock(
            side_effect=TTSError("Voice prompt 'x' not found.", recovery="config")
        )
        result = self._invoke_generate(main)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Voice prompt 'x' not found.", result.stderr)
        self.assertIn("Suggestion:", result.stderr)
        self.assertNotIn("Traceback", result.stderr + result.stdout)

    def test_ttsgeneric_error_exits_1_without_traceback(self):
        main = MagicMock(side_effect=TTSGenericError("server connection lost"))
        result = self._invoke_generate(main)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("server connection lost", result.stderr)
        self.assertNotIn("Traceback", result.stderr + result.stdout)

    def test_success_exits_0(self):
        result = self._invoke_generate(MagicMock(return_value=None))
        self.assertEqual(result.exit_code, 0)

    def test_server_completion_still_exits_2(self):
        # Deliberate contract (pinned by tests/test_cli_commands.py): 2 means
        # "completed via the server", not an error. The boundary must not
        # swallow or re-map SystemExit.
        result = self._invoke_generate(MagicMock(return_value=True))
        self.assertEqual(result.exit_code, 2)

    def test_boundary_covers_non_generate_commands(self):
        with patch(
            "qwen3_tts.cli.stats_command",
            MagicMock(side_effect=TTSError("Server is not running.")),
        ):
            result = CliRunner().invoke(cli, ["stats"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Server is not running.", result.stderr)

    def test_click_usage_errors_pass_through_untouched(self):
        result = CliRunner().invoke(cli, ["--nope"])
        self.assertEqual(result.exit_code, 2)  # Click's own convention
        self.assertNotIn("Suggestion:", result.stderr + result.stdout)


class TestGenerateOutputPathValidation(unittest.TestCase):
    def _args(self, output):
        # Every earlier branch in _handle_generation is falsy except the
        # text resolution, so execution reaches the output-path check.
        return SimpleNamespace(
            repl=False,
            watch=None,
            srt=None,
            dialogue=None,
            voice=None,
            prompt=None,
            mode=None,
            description=None,
            clipboard=False,
            dry_run=False,
            batch=None,
            text=("hi",),
            text_override="hi",
            ssml=False,
            output=output,
        )

    def _assert_rejected(self, output):
        from qwen3_tts.interface import generate

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                generate, "LAST_TEXT_FILE", os.path.join(tmp, "last_text.txt")
            ),
        ):
            with self.assertRaises(TTSError) as raised:
                generate._handle_generation(self._args(output), {}, {}, True, 500)
        self.assertEqual(raised.exception.recovery, "config")
        self.assertIn("Invalid output path", raised.exception.user_message)

    def test_traversal_output_path_raises_config_tts_error(self):
        self._assert_rejected("../evil.wav")

    def test_absolute_output_path_raises_config_tts_error(self):
        self._assert_rejected("/tmp/evil.wav")


class TestSrtNoSubtitles(unittest.TestCase):
    def test_empty_srt_raises_invalid_input_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.srt")
            with open(path, "w"):
                pass
            with self.assertRaises(InvalidInputError):
                process_srt_file(path, {}, _make_srt_args(), {}, False)

    def test_error_message_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.srt")
            with open(path, "w"):
                pass
            with self.assertRaises(InvalidInputError) as raised:
                process_srt_file(path, {}, _make_srt_args(), {}, False)
            self.assertIn(path, raised.exception.user_message)


if __name__ == "__main__":
    unittest.main()
