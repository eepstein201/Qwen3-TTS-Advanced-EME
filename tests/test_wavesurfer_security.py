"""The script re-executor must not use eval() on DOM script content.

get_script_reexecutor_fn() re-runs scripts that Gradio's innerHTML injection
leaves inert. The inline-script branch used eval(s.textContent), which executes
arbitrary DOM script text — a DOM-based XSS amplifier. It must use the same
Blob + URL.createObjectURL pattern already used for module scripts.

See docs/reviews/e2e-review-2026-07-01.md (Phase 5 security, H2).
"""

import unittest

from qwen3_tts.interface.wavesurfer_js import get_script_reexecutor_fn


class TestScriptReexecutorNoEval(unittest.TestCase):
    def test_no_eval_call(self):
        js = get_script_reexecutor_fn()
        self.assertNotIn("eval(", js, "re-executor must not eval() DOM script content")

    def test_inline_branch_still_reinjects(self):
        # Behavior preserved: inline scripts that create elements are still
        # re-injected (now via Blob), so WaveSurfer loading still works.
        js = get_script_reexecutor_fn()
        self.assertIn("createElement", js, "inline-script detection retained")
        self.assertIn("createObjectURL", js, "inline scripts re-injected via Blob URL")


if __name__ == "__main__":
    unittest.main()
