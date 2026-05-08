"""Path injection tests for PR 4.

RED → GREEN TDD: tests written first, verified failing before implementation.

Covers:
  - generate_interactive.run_repl: output_dir join inside REPL loop
  - tools.create_voice.create_and_save_voice_prompt: test_output join with
    user-supplied prompt_name (base_name traversal vector)

Sites already covered by safe_path_join (no new tests needed):
  - cli/srt.py lines 81, 94  — already uses safe_path_join
  - cli/dialogue.py lines 154, 176  — already uses safe_path_join
  - generate.py process_batch lines 115, 134  — already uses safe_path_join
  - generate.py _handle_generation line 535  — already uses safe_path_join
  - generate_interactive.py interactive_mode line 389  — already uses safe_path_join
  - generate_interactive.run_watch_mode line 624  — already uses safe_path_join
  - generate_interactive._ProgressPoller  — no path joins
  - generate_helpers.auto_increment_filename  — takes full path, no join
  - tools/create_voice.py VOICE_PROMPTS_DIR joins  — already uses safe_path_join
  - tools/create_voice.py temp wav (USER_FILES_DIR constant)  — low risk constant
"""

from __future__ import annotations

import os
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helper: locate production root
# ---------------------------------------------------------------------------

PROD_ROOT = Path(__file__).parents[2] / "qwen3_tts"


# ---------------------------------------------------------------------------
# Test: generate_interactive.run_repl  output_path join
# ---------------------------------------------------------------------------

class TestReplOutputPathTraversal(unittest.TestCase):
    """run_repl builds output_path via os.path.join(output_dir, ...).

    The filename is hardcoded (repl_N.wav) so there is no filename-level
    traversal risk, but the join should still use safe_path_join so that
    a malformed output_directory in config cannot escape the intended base.

    RED expectation: the current code uses bare os.path.join so this
    structural assertion will FAIL until safe_path_join is applied.
    """

    def test_repl_uses_safe_path_join_not_bare_join(self):
        """Structural: run_repl must not use bare os.path.join for output_path."""
        src = (PROD_ROOT / "interface" / "generate_interactive.py").read_text()
        # After the REPL counter line, there should be no bare os.path.join
        # building the output path.  We look for the specific line that was
        # present before the fix.
        bad_pattern = "os.path.join(output_dir,"
        # The bad pattern must NOT appear inside run_repl after the fix.
        # We locate the run_repl function body and check within it.
        repl_start = src.find("def run_repl(")
        self.assertGreater(repl_start, 0, "run_repl not found in source")
        repl_body = src[repl_start:]
        self.assertNotIn(
            bad_pattern,
            repl_body,
            "run_repl still uses bare os.path.join(output_dir, ...) — "
            "replace with safe_path_join",
        )


class TestReplOutputPathTraversalFunctional(unittest.TestCase):
    """Functional traversal test for run_repl output path construction.

    Note on scope: output_dir comes from config (user-controlled directory
    choice).  safe_path_join(output_dir, filename) guards the filename
    component — ensuring the file stays inside output_dir — but cannot
    restrict the choice of output_dir itself (that is intentional user
    control, similar to `--output /arbitrary/path/`).

    The guard that matters is: a traversal in the *filename* component must
    be rejected even if output_dir is a legitimate path.  This test verifies
    that safe_path_join is wired up so filename-level traversal is blocked.
    """

    def test_repl_filename_component_traversal_is_blocked(self):
        """safe_path_join must block traversal in the filename component.

        We directly invoke safe_path_join with a legitimate output_dir but
        a traversal filename to confirm the guard works as expected.
        """
        import tempfile
        from qwen3_tts.core.config import safe_path_join

        with tempfile.TemporaryDirectory() as tmp_dir:
            # A filename with traversal should raise ValueError
            with self.assertRaises(ValueError):
                safe_path_join(tmp_dir, "../../etc/passwd.wav")

            # A hardcoded filename like repl_1.wav should succeed
            result = safe_path_join(tmp_dir, "repl_1.wav")
            self.assertTrue(result.startswith(tmp_dir) or
                            result.startswith(os.path.realpath(tmp_dir)),
                            f"Expected path inside {tmp_dir}, got {result}")


# ---------------------------------------------------------------------------
# Test: tools.create_voice  test_output path with traversal prompt_name
# ---------------------------------------------------------------------------

class TestCreateVoiceTestOutputTraversal(unittest.TestCase):
    """create_and_save_voice_prompt builds test_output via os.path.join.

    base_name is derived from prompt_name (CLI arg).  A traversal prompt_name
    like '../../../etc/passwd.pt' produces base_name='../../../etc/passwd'
    and test_output='/path/to/UserFiles/test_../../../etc/passwd.wav'.

    RED expectation: bare os.path.join on line 109 allows this traversal.
    After fix, safe_path_join raises ValueError.
    """

    def test_create_voice_uses_safe_path_join_for_test_output(self):
        """Structural: create_and_save_voice_prompt must not use bare os.path.join
        for test_output."""
        src = (PROD_ROOT / "tools" / "create_voice.py").read_text()
        fn_start = src.find("def create_and_save_voice_prompt(")
        self.assertGreater(fn_start, 0, "create_and_save_voice_prompt not found")
        fn_body = src[fn_start:]
        # The specific vulnerable pattern
        bad_pattern = 'os.path.join(USER_FILES_DIR, f"test_'
        self.assertNotIn(
            bad_pattern,
            fn_body,
            "create_and_save_voice_prompt still uses bare os.path.join for "
            "test_output — replace with safe_path_join",
        )

    def test_traversal_prompt_name_raises_value_error(self):
        """Traversal prompt_name must raise ValueError before writing test output."""
        import numpy as np

        traversal_prompt = "../../../etc/passwd.pt"

        fake_wav = np.zeros(100)
        fake_sr = 22050
        fake_voice_prompt = {"x_vector": np.zeros(64)}

        with patch("soundfile.read", return_value=(fake_wav, fake_sr)), \
             patch("soundfile.write"), \
             patch("shutil.copy2"), \
             patch("torch.save"), \
             patch("qwen3_tts.core.engine.load_model", return_value=MagicMock()), \
             patch("qwen3_tts.core.engine.create_voice_prompt",
                   return_value=fake_voice_prompt), \
             patch("qwen3_tts.core.engine.run_inference",
                   return_value=(fake_wav, fake_sr)):
            from qwen3_tts.tools.create_voice import create_and_save_voice_prompt

            with self.assertRaises(ValueError) as ctx:
                create_and_save_voice_prompt(
                    audio_path="/tmp/test_audio.wav",
                    transcript="hello",
                    prompt_name=traversal_prompt,
                    test_generation=True,
                    mlx_only=False,
                )

            self.assertIn(
                "traversal",
                str(ctx.exception).lower(),
                "ValueError raised but message doesn't mention traversal",
            )

    def test_absolute_prompt_name_raises_value_error(self):
        """Absolute-path prompt_name must also raise ValueError."""
        import numpy as np

        absolute_prompt = "/etc/passwd.pt"

        fake_wav = np.zeros(100)
        fake_sr = 22050
        fake_voice_prompt = {"x_vector": np.zeros(64)}

        with patch("soundfile.read", return_value=(fake_wav, fake_sr)), \
             patch("soundfile.write"), \
             patch("shutil.copy2"), \
             patch("torch.save"), \
             patch("qwen3_tts.core.engine.load_model", return_value=MagicMock()), \
             patch("qwen3_tts.core.engine.create_voice_prompt",
                   return_value=fake_voice_prompt), \
             patch("qwen3_tts.core.engine.run_inference",
                   return_value=(fake_wav, fake_sr)):
            from qwen3_tts.tools.create_voice import create_and_save_voice_prompt

            with self.assertRaises(ValueError) as ctx:
                create_and_save_voice_prompt(
                    audio_path="/tmp/test_audio.wav",
                    transcript="hello",
                    prompt_name=absolute_prompt,
                    test_generation=True,
                    mlx_only=False,
                )

            self.assertIn(
                "traversal",
                str(ctx.exception).lower(),
                "ValueError raised but message doesn't mention traversal",
            )


# ---------------------------------------------------------------------------
# Test: structural scan — no new bare os.path.join(user_dir, ...) patterns
# ---------------------------------------------------------------------------

class TestNoNewBareJoinsInTargetModules(unittest.TestCase):
    """Regression guard: target modules must not introduce new bare joins.

    Checks that every os.path.join call in these modules either:
      (a) uses only constant/hardcoded path components, or
      (b) has been replaced by safe_path_join.

    We enumerate known-safe calls explicitly and assert no unvetted
    os.path.join remains in the function bodies that handle user input.
    """

    def _get_source(self, *rel_parts: str) -> str:
        return (PROD_ROOT.joinpath(*rel_parts)).read_text()

    def test_srt_no_bare_user_join(self):
        src = self._get_source("interface", "cli", "srt.py")
        self.assertNotIn("os.path.join(output_dir,", src,
                         "cli/srt.py has bare os.path.join(output_dir, ...)")

    def test_dialogue_no_bare_user_join(self):
        src = self._get_source("interface", "cli", "dialogue.py")
        self.assertNotIn("os.path.join(output_dir,", src,
                         "cli/dialogue.py has bare os.path.join(output_dir, ...)")

    def test_generate_no_bare_output_dir_join(self):
        src = self._get_source("interface", "generate.py")
        self.assertNotIn("os.path.join(output_dir,", src,
                         "generate.py has bare os.path.join(output_dir, ...)")

    def test_generate_interactive_no_bare_output_dir_join(self):
        src = self._get_source("interface", "generate_interactive.py")
        self.assertNotIn("os.path.join(output_dir,", src,
                         "generate_interactive.py has bare os.path.join(output_dir, ...)")


if __name__ == "__main__":
    unittest.main()
