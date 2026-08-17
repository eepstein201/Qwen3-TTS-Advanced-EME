#!/usr/bin/env python3
"""Static checks on install.sh's history output folder handling.

These are text assertions (no bash execution) so the suite stays runnable on
any platform. Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_install_script.py -q
"""
import os
import unittest

INSTALL_SH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "install.sh"
)


class TestInstallScriptHistoryOutputDir(unittest.TestCase):
    def setUp(self):
        with open(INSTALL_SH) as f:
            self.text = f.read()

    def test_config_template_has_history_output_directory(self):
        self.assertIn('"history_output_directory"', self.text)

    def test_default_matches_the_python_default(self):
        from qwen3_tts.core.config import get_default_config

        self.assertIn(get_default_config()["history_output_directory"], self.text)

    def test_creates_both_subfolders(self):
        self.assertIn("Automated Output", self.text)
        self.assertIn("Manual Downloads", self.text)

    def test_prompt_defaults_to_the_default_path_on_empty_input(self):
        # Follows the existing VAR=${VAR:-default} idiom used for every other
        # prompt in this script, so pressing Enter accepts the default.
        self.assertRegex(self.text, r"HISTORY_OUTPUT_DIR=\$\{HISTORY_OUTPUT_DIR:-")


class TestInstallScriptMatchesPythonDefaults(unittest.TestCase):
    """install.sh writes config.json from a hand-maintained heredoc rather
    than calling get_default_config(), so the two drift silently. PR #190
    changed the language default to "auto" and the heredoc kept "English",
    so every fresh install reverted the fix.
    """

    def setUp(self):
        with open(INSTALL_SH) as f:
            self.text = f.read()

    def test_language_default_matches_python(self):
        from qwen3_tts.core.config import get_default_config

        expected = get_default_config()["language"]
        self.assertIn(f'"language": "{expected}"', self.text)

    def test_no_stale_english_language_default(self):
        self.assertNotIn('"language": "English"', self.text)

    def test_default_clone_prompt_is_not_a_dangling_filename(self):
        """get_default_config() ships None: no prompt ships with the package,
        so any seeded filename references a file that does not exist."""
        self.assertNotIn('"default_clone_prompt": "default_clone.pt"', self.text)


if __name__ == "__main__":
    unittest.main()
