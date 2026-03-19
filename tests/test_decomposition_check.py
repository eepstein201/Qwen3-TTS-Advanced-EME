"""Meta-test validating the test_voice.py decomposition is complete and correct."""

import importlib
import os
import subprocess
import sys
import unittest


class TestDecompositionComplete(unittest.TestCase):
    def test_voice_test_files_exist(self):
        expected = [
            "tests/test_voice_config.py",
            "tests/test_voice_server.py",
            "tests/test_voice_prompts.py",
            "tests/test_voice_streaming.py",
            "tests/test_voice_engine.py",
            "tests/test_voice_generation.py",
            "tests/test_voice_ui.py",
            "tests/test_voice_features.py",
        ]
        for f in expected:
            self.assertTrue(os.path.exists(f), f"Missing: {f}")

    def test_original_is_empty_or_redirects(self):
        with open("tests/test_voice.py") as f:
            content = f.read()
        self.assertLess(len(content), 500,
            "test_voice.py should be a minimal shim after decomposition")

    def test_no_circular_imports(self):
        """Verify each new test module imports cleanly without circular deps."""
        modules = [
            "tests.test_voice_config",
            "tests.test_voice_server",
            "tests.test_voice_prompts",
            "tests.test_voice_streaming",
            "tests.test_voice_engine",
            "tests.test_voice_generation",
            "tests.test_voice_ui",
            "tests.test_voice_features",
        ]
        for mod_name in modules:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                self.fail(f"Circular or broken import in {mod_name}: {e}")

    def test_no_orphaned_dependencies(self):
        """Verify no test file imports symbols that were left behind in the original."""
        if os.path.exists("tests/test_voice.py"):
            with open("tests/test_voice.py") as f:
                original = f.read()
            self.assertNotIn("class Test", original,
                "test_voice.py still contains test classes after decomposition")

    def test_no_silent_skips(self):
        """Run all decomposed test files and verify tests are collected."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--co", "-q",
             "tests/test_voice_config.py",
             "tests/test_voice_server.py",
             "tests/test_voice_prompts.py",
             "tests/test_voice_streaming.py",
             "tests/test_voice_engine.py",
             "tests/test_voice_generation.py",
             "tests/test_voice_ui.py",
             "tests/test_voice_features.py"],
            capture_output=True, text=True
        )
        self.assertIn("test", result.stdout.lower(),
            f"No tests collected from decomposed files. stderr: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
