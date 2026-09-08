#!/usr/bin/env python3
"""Step 0E Task 1 — the Gradio share surface requires credentials.

Pins, at the single launch-kwarg helper and at both real launch sites:
- ``share=True`` forces a non-empty ``auth`` (user, password) tuple onto the
  launch kwargs, from both env vars when configured, else generated.
- Generated credentials are printed to the console; ``share=False`` prints
  nothing about credentials and returns no ``auth`` key.
- The env override is all-or-nothing: exactly one of ``TTS_UI_USERNAME`` /
  ``TTS_UI_PASSWORD`` fails closed with RuntimeError.
- Credential generation failure fails closed (RuntimeError, never unauth
  launch kwargs).
- ``allowed_paths`` narrows to {output_dir, tempdir}: ``~/Downloads`` itself
  is granted only when it IS the resolved output dir, never as a blanket
  parent entry (the legacy third set-literal entry is gone).
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


def _printed(mock_print):
    """Join every print() argument a test captured into one string."""
    return " ".join(
        str(arg) for call in mock_print.call_args_list for arg in call.args
    )


class TestShareRequiresAuthHelper(unittest.TestCase):
    """get_gradio_launch_kwargs(config, *, share) credential contract."""

    def test_share_true_generates_and_prints_credentials(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

        # Empty strings behave as unset so the ambient environment can't leak in.
        env = {"TTS_UI_USERNAME": "", "TTS_UI_PASSWORD": ""}
        with patch.dict(os.environ, env), patch("builtins.print") as mock_print:
            kwargs = get_gradio_launch_kwargs({}, share=True)
        auth = kwargs["auth"]
        self.assertIsInstance(auth, tuple)
        user, password = auth
        self.assertIsInstance(user, str)
        self.assertIsInstance(password, str)
        self.assertTrue(user, "generated username must be non-empty")
        self.assertTrue(password, "generated password must be non-empty")
        self.assertTrue(user.startswith("tts-"))
        # Design ruling: the user is "tts-" + 6 generated characters.
        self.assertEqual(len(user), len("tts-") + 6)
        printed = _printed(mock_print)
        self.assertIn(user, printed)
        self.assertIn(password, printed)

    def test_share_true_prefers_both_env_vars_verbatim(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

        env = {"TTS_UI_USERNAME": "alice", "TTS_UI_PASSWORD": "s3cret"}
        with patch.dict(os.environ, env):
            kwargs = get_gradio_launch_kwargs({}, share=True)
        self.assertEqual(kwargs["auth"], ("alice", "s3cret"))

    def test_share_true_single_env_var_fails_closed(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

        for partial in (
            {"TTS_UI_USERNAME": "alice", "TTS_UI_PASSWORD": ""},
            {"TTS_UI_USERNAME": "", "TTS_UI_PASSWORD": "s3cret"},
        ):
            with self.subTest(partial=partial):
                with patch.dict(os.environ, partial):
                    with self.assertRaises(RuntimeError) as ctx:
                        get_gradio_launch_kwargs({}, share=True)
                message = str(ctx.exception)
                self.assertIn("TTS_UI_USERNAME", message)
                self.assertIn("TTS_UI_PASSWORD", message)

    def test_share_true_generation_failure_fails_closed(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

        env = {"TTS_UI_USERNAME": "", "TTS_UI_PASSWORD": ""}
        with patch.dict(os.environ, env):
            with patch(
                "secrets.token_urlsafe", side_effect=OSError("entropy unavailable")
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    get_gradio_launch_kwargs({}, share=True)
        self.assertIn("credentials", str(ctx.exception).lower())

    def test_share_false_has_no_auth_and_prints_no_credentials(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

        with patch("builtins.print") as mock_print:
            kwargs = get_gradio_launch_kwargs({}, share=False)
        self.assertNotIn("auth", kwargs)
        printed = _printed(mock_print)
        self.assertNotIn("Username", printed)
        self.assertNotIn("Password", printed)

    def test_default_call_without_share_kwarg_has_no_auth(self):
        """Existing two-arg callers keep today's exact dict (no auth key)."""
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

        with patch("builtins.print") as mock_print:
            kwargs = get_gradio_launch_kwargs({})
        self.assertNotIn("auth", kwargs)
        self.assertNotIn("Password", _printed(mock_print))

    def test_allowed_paths_narrow_to_output_dir_and_tempdir(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

        # The default web-UI output root lives UNDER ~/Downloads; granting the
        # Downloads parent itself as a blanket entry is what this narrows away.
        config = {"output_directory": "~/Downloads/Qwen3-TTS Output"}
        kwargs = get_gradio_launch_kwargs(config)
        allowed = kwargs["allowed_paths"]
        output_dir = os.path.realpath(
            os.path.expanduser("~/Downloads/Qwen3-TTS Output")
        )
        downloads = os.path.realpath(os.path.expanduser("~/Downloads"))
        self.assertIn(output_dir, allowed)
        self.assertIn(tempfile.gettempdir(), allowed)
        self.assertNotIn(downloads, allowed)
        self.assertEqual(len(allowed), 2)


class TestFacadeMainLaunchAuth(unittest.TestCase):
    """The _facade.main() launch site forwards share and receives auth."""

    def test_main_share_true_launches_with_auth(self):
        import qwen3_tts.interface.ui._facade as facade_mod
        import qwen3_tts.interface.ui.shared as ui_shared

        demo = MagicMock()
        real_helper = ui_shared.get_gradio_launch_kwargs
        captured = {}

        def spy(config, **kwargs):
            captured.update(kwargs)
            return real_helper(config, **kwargs)

        env = {"TTS_UI_USERNAME": "", "TTS_UI_PASSWORD": ""}
        with patch.object(facade_mod, "build_ui", return_value=demo), \
             patch.object(facade_mod, "load_config", return_value={}), \
             patch.object(facade_mod, "_find_available_port", return_value=7860), \
             patch.object(facade_mod, "TTSClient"), \
             patch.object(facade_mod, "IN_COLAB", False), \
             patch.object(ui_shared, "get_gradio_launch_kwargs", spy), \
             patch.dict(os.environ, env), \
             patch.object(sys, "argv", ["qwen3-tts-ui", "--share"]):
            facade_mod.main()

        # The launch site must forward its share decision into the helper.
        self.assertEqual(captured, {"share": True})
        launch_kwargs = demo.launch.call_args.kwargs
        self.assertTrue(launch_kwargs["share"])
        auth = launch_kwargs["auth"]
        self.assertIsInstance(auth, tuple)
        user, password = auth
        self.assertTrue(user)
        self.assertTrue(password)


class TestBuildUiAndLaunchAuth(unittest.TestCase):
    """The generate_server.build_ui_and_launch site forwards share + auth."""

    def test_build_ui_and_launch_share_true_launches_with_auth(self):
        import qwen3_tts.core.config as _cfg
        import qwen3_tts.interface.generate_server as gs
        import qwen3_tts.interface.ui._facade as facade_mod
        import qwen3_tts.interface.ui.shared as ui_shared

        # Restore the config global before mutating it (leak-safe ordering).
        orig = _cfg.IN_COLAB
        self.addCleanup(setattr, _cfg, "IN_COLAB", orig)
        _cfg.IN_COLAB = False

        demo = MagicMock()
        real_helper = ui_shared.get_gradio_launch_kwargs
        captured = {}

        def spy(config, **kwargs):
            captured.update(kwargs)
            return real_helper(config, **kwargs)

        env = {
            "TTS_UI_SHARE": "1",
            "TTS_UI_USERNAME": "",
            "TTS_UI_PASSWORD": "",
        }
        with patch.object(facade_mod, "build_ui", return_value=demo), \
             patch.object(facade_mod, "_find_available_port", return_value=7860), \
             patch.object(ui_shared, "get_gradio_launch_kwargs", spy), \
             patch.dict(os.environ, env):
            gs.build_ui_and_launch({})

        # The launch site must forward its share decision into the helper.
        self.assertEqual(captured, {"share": True})
        launch_kwargs = demo.launch.call_args.kwargs
        self.assertTrue(launch_kwargs["share"])
        auth = launch_kwargs["auth"]
        self.assertIsInstance(auth, tuple)
        user, password = auth
        self.assertTrue(user)
        self.assertTrue(password)


if __name__ == "__main__":
    unittest.main()
