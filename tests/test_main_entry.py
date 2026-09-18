"""Entry-point coverage for ``python -m qwen3_tts`` (qwen3_tts/__main__.py).

4B.3 item 13: __main__.py was the only 0%-covered module (2 statements).
A subprocess is the faithful way to exercise it — importing __main__ in
process would run the CLI in the test interpreter.
"""

import os
import subprocess
import sys
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestMainEntryPoint(unittest.TestCase):
    def test_python_dash_m_invokes_cli(self):
        result = subprocess.run(
            [sys.executable, "-m", "qwen3_tts", "--help"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=_REPO_ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Usage", result.stdout)


if __name__ == "__main__":
    unittest.main()
