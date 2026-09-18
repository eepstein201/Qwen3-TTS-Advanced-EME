"""Closure coverage for interface/ui/tabs_generation.py (4B.3 item 13).

The three mode ``*_config_handler`` closures and ``save_design_as_prompt``
are created inside the ``_build_*_tab`` builders, so the only way to reach
them is to build the tab with gradio and every collaborator mocked, then
capture the fns off the wiring calls:

- config handlers come from the ``config_handler`` kwarg passed to
  ``generation._wire_generation_tab``
- ``save_design_as_prompt`` comes from the ``fn`` kwarg of the
  ``design_save_btn.click(...)`` wiring (all ``gr.Button()`` calls share
  one mock return value, so the wiring is found by fn name)

Collaborators are module-style imports in tabs_generation, so each is
patched at its binding site in that module.
"""

import unittest
from unittest.mock import MagicMock, patch

try:
    import gradio  # noqa: F401

    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

skip_if_no_gradio = unittest.skipUnless(HAS_GRADIO, "requires gradio")

_TABS = "qwen3_tts.interface.ui.tabs_generation"


class _TabHarness:
    """Build tabs with mocked collaborators; expose the wired closures."""

    def __init__(self):
        from qwen3_tts.interface.ui import tabs_generation

        self.module = tabs_generation
        self.mock_generation = MagicMock()
        self.mock_shared = MagicMock()
        self.mock_model_management = MagicMock()
        self.mock_voice_helpers = MagicMock()
        self.mock_core_config = MagicMock()
        self.mock_gr = MagicMock()
        self._patches = [
            patch(_TABS + ".generation", self.mock_generation),
            patch(_TABS + ".shared", self.mock_shared),
            patch(_TABS + ".model_management", self.mock_model_management),
            patch(_TABS + ".voice_helpers", self.mock_voice_helpers),
            patch(_TABS + ".core_config", self.mock_core_config),
            patch(_TABS + ".gr", self.mock_gr),
        ]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False

    def build_all(self):
        self.module._build_clone_tab(None, None)
        self.module._build_design_tab(None, None, None)
        self.module._build_custom_tab(None, None)

    def config_handler(self, mode):
        for call in self.mock_generation._wire_generation_tab.call_args_list:
            if call.args and call.args[0] == mode and "config_handler" in call.kwargs:
                return call.kwargs["config_handler"]
        raise AssertionError(f"no config_handler wired for mode {mode!r}")

    def save_design_handler(self):
        self.module._build_design_tab(None, None, None)
        for call in self.mock_gr.Button.return_value.click.call_args_list:
            fn = call.kwargs.get("fn")
            if (
                fn is not None
                and getattr(fn, "__name__", "") == "save_design_as_prompt"
            ):
                return fn
        raise AssertionError("save_design_as_prompt never wired to a Button click")


class TestConfigHandlerClosures(unittest.TestCase):
    """The clone/design/custom config handlers delegate with the right
    mode-specific kwargs (previously missed lines 105, 271, 452)."""

    @skip_if_no_gradio
    def test_handlers_delegate_with_mode_specific_kwargs(self):
        with _TabHarness() as h:
            h.build_all()
            h.mock_generation._prepare_streaming_config.reset_mock()

            h.config_handler("clone")(
                "t", "p.pt", "PRE", 0.1, 2, 0.3, 1.1, 42, True, False
            )
            h.config_handler("design")("t2", "desc", "PRE2", 0.2, 3, 0.4, 1.2, 7, False)
            h.config_handler("custom")(
                "t3", "ryan", "instr", "PRE3", 0.3, 4, 0.5, 1.3, 9, True
            )

        calls = {
            c.args[0]: c
            for c in h.mock_generation._prepare_streaming_config.call_args_list
        }
        self.assertEqual(
            calls["clone"].args, ("clone", "t", "PRE", 0.1, 2, 0.3, 1.1, 42)
        )
        self.assertEqual(
            calls["clone"].kwargs,
            {"prompt_file": "p.pt", "no_transcript": True, "seed_lock_chunks": False},
        )
        self.assertEqual(
            calls["design"].args, ("design", "t2", "PRE2", 0.2, 3, 0.4, 1.2, 7)
        )
        self.assertEqual(
            calls["design"].kwargs,
            {"description": "desc", "seed_lock_chunks": False},
        )
        self.assertEqual(
            calls["custom"].args, ("custom", "t3", "PRE3", 0.3, 4, 0.5, 1.3, 9)
        )
        self.assertEqual(
            calls["custom"].kwargs,
            {"speaker": "ryan", "instruct": "instr", "seed_lock_chunks": True},
        )


class TestSaveDesignAsPrompt(unittest.TestCase):
    """save_design_as_prompt validation/success arms (missed 331-390)."""

    @skip_if_no_gradio
    def test_blank_name_rejected(self):
        with _TabHarness() as h:
            fn = h.save_design_handler()
            status, update = fn("   ", [])
        self.assertEqual(status, "Please enter a voice name.")
        self.assertIs(update, h.mock_gr.update.return_value)

    @skip_if_no_gradio
    def test_unsanitizable_name_rejected(self):
        with _TabHarness() as h:
            fn = h.save_design_handler()
            status, _ = fn("bad name!!", [])
        self.assertIn("Voice name may only contain", status)

    @skip_if_no_gradio
    def test_config_validation_error_surfaced(self):
        with _TabHarness() as h:
            h.mock_core_config.validate_voice_name.side_effect = ValueError(
                "reserved name"
            )
            fn = h.save_design_handler()
            status, _ = fn("ok_name", [])
        self.assertEqual(status, "reserved name")

    @skip_if_no_gradio
    def test_no_history_entries_reports_missing(self):
        with _TabHarness() as h:
            fn = h.save_design_handler()
            status, _ = fn("ok_name", [])
        self.assertIn("No recent Design mode output found", status)

    @skip_if_no_gradio
    def test_non_design_history_entries_ignored(self):
        with _TabHarness() as h:
            fn = h.save_design_handler()
            status, _ = fn("ok_name", [{"mode": "Clone", "path": "/tmp/x.wav"}])
        self.assertIn("No recent Design mode output found", status)

    @skip_if_no_gradio
    def test_absolute_path_with_dotdot_rejected_as_traversal(self):
        with _TabHarness() as h:
            fn = h.save_design_handler()
            status, _ = fn("ok_name", [{"mode": "Design", "path": "~/../evil.wav"}])
        self.assertIn("Path traversal detected", status)

    @skip_if_no_gradio
    def test_path_outside_home_rejected(self):
        with _TabHarness() as h:
            fn = h.save_design_handler()
            status, _ = fn("ok_name", [{"mode": "Design", "path": "/etc/hosts"}])
        self.assertIn("must be under home directory", status)

    @skip_if_no_gradio
    def test_existing_design_output_saved_as_x_vector_prompt(self):
        with _TabHarness() as h:
            h.mock_core_config.get_backend.return_value = "mlx"
            fn = h.save_design_handler()
            with (
                patch(_TABS + ".os.path.exists", return_value=True),
                patch(
                    "qwen3_tts.tools.create_voice.create_and_save_voice_prompt"
                ) as mock_create,
            ):
                status, update = fn(
                    "ok_name", [{"mode": "Design", "path": "~/design.wav"}]
                )
        self.assertEqual(status, "Saved voice prompt: ok_name")
        self.assertEqual(
            update,
            h.mock_gr.update(choices=h.mock_shared.get_voice_prompts.return_value),
        )
        create_args, create_kwargs = mock_create.call_args
        saved_path = create_args[0]
        self.assertTrue(
            saved_path.endswith("design.wav"),
            saved_path,
        )
        self.assertEqual(create_args[1:], ("", "ok_name"))
        self.assertEqual(
            create_kwargs,
            {
                "test_generation": False,
                "mlx_only": True,
                "x_vector_only_mode": True,
            },
        )

    @skip_if_no_gradio
    def test_create_failure_surfaced_as_error(self):
        with _TabHarness() as h:
            h.mock_core_config.get_backend.return_value = "mlx"
            fn = h.save_design_handler()
            with (
                patch(_TABS + ".os.path.exists", return_value=True),
                patch(
                    "qwen3_tts.tools.create_voice.create_and_save_voice_prompt",
                    side_effect=RuntimeError("engine exploded"),
                ),
            ):
                status, _ = fn("ok_name", [{"mode": "Design", "path": "~/design.wav"}])
        self.assertEqual(status, "Error: engine exploded")


if __name__ == "__main__":
    unittest.main()
