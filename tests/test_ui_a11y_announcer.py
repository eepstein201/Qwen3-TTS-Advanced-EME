#!/usr/bin/env python3
"""Tests for the sr-only aria-live status announcer (A11Y-1).

Covers `_announce_status` + the unified stop/cancel vocab constants in
qwen3_tts.interface.ui.generation, and the in-DOM announcer component built by
`_build_generate_buttons_and_output`.

Run: python -m pytest tests/test_ui_a11y_announcer.py -v
No GPU, models, or running server required.
"""
import inspect
import unittest

try:
    import gradio as gr  # noqa: F401
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

_skip = unittest.skipUnless(HAS_GRADIO, "requires gradio")


@_skip
class TestAnnounceStatus(unittest.TestCase):
    """The sr-only aria-live helper `_announce_status`."""

    def test_message_has_live_region_and_clip_styles(self):
        # Arrange / Act
        from qwen3_tts.interface.ui.generation import _announce_status

        html = _announce_status("ok")
        # Assert — semantic live region + visually-hidden clip styles
        self.assertIn('role="status"', html)
        self.assertIn('aria-live="polite"', html)
        self.assertIn(">ok<", html)
        self.assertIn("clip:rect(0,0,0,0)", html)
        self.assertIn("position:absolute", html)

    def test_html_is_escaped(self):
        from qwen3_tts.interface.ui.generation import _announce_status

        html = _announce_status("<script>alert(1)</script>")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_empty_string_is_blank_region(self):
        from qwen3_tts.interface.ui.generation import _announce_status

        html = _announce_status("")
        self.assertIn('role="status"', html)
        # Never render the literal string "None" or stray content
        self.assertNotIn("None", html)

    def test_none_is_blank_region(self):
        from qwen3_tts.interface.ui.generation import _announce_status

        html = _announce_status(None)
        self.assertIn('role="status"', html)
        self.assertNotIn("None", html)


@_skip
class TestStatusVocab(unittest.TestCase):
    """The unified stop/cancel vocabulary constants are sane."""

    _NAMES = (
        "STATUS_STOP_LABEL",
        "STATUS_STOP_CONFIRM_ARM",
        "STATUS_STOP_CONFIRM_HINT",
        "STATUS_GENERATION_STOPPING",
        "STATUS_GENERATION_STOPPED",
        "STATUS_STOP_CANCELED",
    )

    def test_all_constants_non_empty(self):
        import qwen3_tts.interface.ui.generation as gen

        for name in self._NAMES:
            value = getattr(gen, name)
            self.assertIsInstance(value, str)
            self.assertTrue(value.strip(), f"{name} must be non-empty")

    def test_confirm_arm_uses_stop_vocab_not_cancel(self):
        from qwen3_tts.interface.ui.generation import STATUS_STOP_CONFIRM_ARM

        # Unified vocab should not leak the old "Cancel"/"aborted" wording.
        self.assertNotIn("Cancel", STATUS_STOP_CONFIRM_ARM)
        self.assertNotIn("aborted", STATUS_STOP_CONFIRM_ARM)
        self.assertIn("Stop", STATUS_STOP_CONFIRM_ARM)


@_skip
class TestAnnouncerComponent(unittest.TestCase):
    """`_build_generate_buttons_and_output` exposes a status_announcer component."""

    def test_status_announcer_present_and_sr_only(self):
        from qwen3_tts.interface.ui.generation import _build_generate_buttons_and_output

        with gr.Blocks():
            btns = _build_generate_buttons_and_output("clone")
        self.assertIn("status_announcer", btns)
        value = str(btns["status_announcer"].value)
        self.assertTrue(value.startswith("<div role=\"status"))
        self.assertIn('aria-live="polite"', value)

    def test_wire_accepts_status_announcer_kwarg(self):
        from qwen3_tts.interface.ui.generation import _wire_generation_tab

        params = inspect.signature(_wire_generation_tab).parameters
        self.assertIn("status_announcer", params)


if __name__ == "__main__":
    unittest.main()
