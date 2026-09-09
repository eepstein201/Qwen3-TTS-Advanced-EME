#!/usr/bin/env python3
"""Step 0E Task 1 — the Gradio share surface requires credentials.

Pins, at the single launch-kwarg helper and at both real launch sites:
- ``share=True`` forces a non-empty ``auth`` (user, password) tuple onto the
  launch kwargs, from both env vars when configured, else generated.
- Generated delivery is hybrid: the password is written — as its sole line —
  to a 0600 credentials file, and only the USERNAME + FILE PATH are printed;
  the password string never reaches any print/log argument. ``share=False``
  prints nothing about credentials and returns no ``auth`` key.
- Env-configured credentials (both set) write nothing and print nothing.
- The env override is all-or-nothing: exactly one of ``TTS_UI_USERNAME`` /
  ``TTS_UI_PASSWORD`` fails closed with RuntimeError.
- Credential generation OR file-delivery failure fails closed (RuntimeError,
  never unauth launch kwargs).
- ``allowed_paths`` narrows to the history output root + tempdir:
  ``~/Downloads`` itself is granted only when it IS the resolved root, never
  as a blanket parent entry.
"""

import os
import stat
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

_CREDENTIALS_SEAM = "qwen3_tts.interface.ui.shared._ui_credentials_path"
_WRITER_SEAM = "qwen3_tts.interface.ui.shared._write_ui_credentials"


def _printed(mock_print):
    """Join every print() argument a test captured into one string."""
    return " ".join(
        str(arg) for call in mock_print.call_args_list for arg in call.args
    )


def _patched_credentials_path(testcase):
    """A throwaway credentials-file path with its tempdir cleaned up after."""
    tmp = tempfile.TemporaryDirectory()
    testcase.addCleanup(tmp.cleanup)
    return os.path.join(tmp.name, ".ui_share_credentials")


class TestShareRequiresAuthHelper(unittest.TestCase):
    """get_gradio_launch_kwargs(config, *, share) credential contract."""

    def test_share_true_delivers_password_via_0600_file(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

        # Empty strings behave as unset so the ambient environment can't leak in.
        env = {"TTS_UI_USERNAME": "", "TTS_UI_PASSWORD": ""}
        cred_path = _patched_credentials_path(self)
        with patch.dict(os.environ, env), \
             patch(_CREDENTIALS_SEAM, return_value=cred_path), \
             patch("builtins.print") as mock_print:
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

        # Only the username and the credentials-file path are printed; the
        # password string itself never appears in any print/log argument.
        printed = _printed(mock_print)
        self.assertIn(user, printed)
        self.assertIn(cred_path, printed)
        self.assertNotIn(password, printed)

        # The 0600 file carries the password as its sole single-line content.
        with open(cred_path) as f:
            self.assertEqual(f.read(), password + "\n")
        self.assertEqual(stat.S_IMODE(os.stat(cred_path).st_mode), 0o600)

    def test_share_true_env_credentials_write_and_print_nothing(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

        cred_path = _patched_credentials_path(self)
        env = {"TTS_UI_USERNAME": "alice", "TTS_UI_PASSWORD": "s3cret"}
        with patch.dict(os.environ, env), \
             patch(_CREDENTIALS_SEAM, return_value=cred_path), \
             patch("builtins.print") as mock_print:
            kwargs = get_gradio_launch_kwargs({}, share=True)
        self.assertEqual(kwargs["auth"], ("alice", "s3cret"))
        # The user supplied both: no file delivery, no console banner.
        self.assertFalse(os.path.exists(cred_path))
        mock_print.assert_not_called()

    def test_share_true_write_failure_fails_closed(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

        cred_path = _patched_credentials_path(self)
        env = {"TTS_UI_USERNAME": "", "TTS_UI_PASSWORD": ""}
        with patch.dict(os.environ, env), \
             patch(_CREDENTIALS_SEAM, return_value=cred_path), \
             patch(_WRITER_SEAM, side_effect=OSError("disk full")):
            with self.assertRaises(RuntimeError) as ctx:
                get_gradio_launch_kwargs({}, share=True)
        self.assertIn("credentials", str(ctx.exception).lower())

    def test_autouse_fixture_repoints_the_seam_away_from_the_real_config_dir(self):
        """conftest's autouse fixture repoints the credentials seam for EVERY
        pytest-run test, so no pytest run can create the real ~/.config file
        (fix round 4). The unittest batch runner subprocess bypasses conftest
        entirely — seam patches there must come from the tests themselves."""
        # Runtime guard, NOT a skipUnless decorator: the decorator's condition
        # evaluates at collection, before pytest sets PYTEST_CURRENT_TEST, so
        # the pin would always report skipped under both runners.
        if "PYTEST_CURRENT_TEST" not in os.environ:
            self.skipTest(
                "the isolation guarantee lives in conftest.py's autouse "
                "fixture, which only fires under pytest; the unittest batch "
                "runner has no fixtures"
            )
        from qwen3_tts.interface.ui.shared import _ui_credentials_path

        real_dir = os.path.expanduser("~/.config/qwen3-tts")
        path = _ui_credentials_path()
        self.assertNotEqual(
            os.path.dirname(path),
            real_dir,
            f"credentials seam resolves into the real config dir: {path}",
        )

    def test_seam_honors_tts_ui_credentials_dir_override(self):
        """``TTS_UI_CREDENTIALS_DIR`` moves the credentials file — the knob
        that isolates the unittest batch subprocesses, which never see the
        conftest fixture. Env-based, so this passes under BOTH runners."""
        import qwen3_tts.interface.ui.shared as ui_shared

        override = tempfile.TemporaryDirectory()
        self.addCleanup(override.cleanup)
        # patch.dict restores the key EXACTLY (prior value, or absence) — a
        # plain addCleanup(pop) would strip an AMBIENT knob set by the batch
        # runner and leave every later test in the process unprotected.
        with patch.dict(os.environ, {"TTS_UI_CREDENTIALS_DIR": override.name}):
            path = ui_shared._ui_credentials_path()
        self.assertEqual(os.path.dirname(path), override.name)
        self.assertEqual(os.path.basename(path), ".ui_share_credentials")

    def test_seam_rejects_override_outside_the_system_tempdir(self):
        """The knob is test-infra: an override outside the system tempdir fails
        closed instead of silently pointing the credentials file somewhere
        else (CodeQL py/path-injection: dominating guard on the sink)."""
        import qwen3_tts.interface.ui.shared as ui_shared

        outside = os.path.expanduser("~")
        with patch.dict(os.environ, {"TTS_UI_CREDENTIALS_DIR": outside}):
            with self.assertRaises(RuntimeError) as ctx:
                ui_shared._ui_credentials_path()
        message = str(ctx.exception)
        self.assertIn("TTS_UI_CREDENTIALS_DIR", message)
        self.assertIn("tempdir", message)

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

    def test_allowed_paths_narrow_to_history_root_and_tempdir(self):
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

        # The grant keys on history_output_directory (the web-UI output root),
        # NOT the legacy output_directory key whose default IS ~/Downloads.
        config = {"history_output_directory": "~/TTSOutput"}
        kwargs = get_gradio_launch_kwargs(config)
        allowed = kwargs["allowed_paths"]
        history_root = os.path.realpath(os.path.expanduser("~/TTSOutput"))
        downloads = os.path.realpath(os.path.expanduser("~/Downloads"))
        self.assertIn(history_root, allowed)
        self.assertIn(tempfile.gettempdir(), allowed)
        self.assertNotIn(downloads, allowed)
        self.assertNotIn(
            os.path.realpath(os.path.expanduser("~/Downloads/Qwen3-TTS Output")),
            allowed,
        )
        self.assertEqual(len(allowed), 2)

    def test_default_config_grants_history_root_not_downloads(self):
        """Under ALL-default config the grant is the history root, not ~/Downloads.

        The pre-fix resolver (`_resolve_output_dir`, the legacy `output_directory`
        key whose default is `~/Downloads`) made the narrowing a no-op under
        defaults: {~/Downloads, tempdir} equalled the effective BASE grant.
        """
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

        kwargs = get_gradio_launch_kwargs({})
        allowed = kwargs["allowed_paths"]
        downloads = os.path.realpath(os.path.expanduser("~/Downloads"))
        history_root = os.path.realpath(
            os.path.expanduser("~/Downloads/Qwen3-TTS Output")
        )
        self.assertIn(history_root, allowed)
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
        cred_path = _patched_credentials_path(self)
        with patch.object(facade_mod, "build_ui", return_value=demo), \
             patch.object(facade_mod, "load_config", return_value={}), \
             patch.object(facade_mod, "_find_available_port", return_value=7860), \
             patch.object(facade_mod, "TTSClient"), \
             patch.object(facade_mod, "IN_COLAB", False), \
             patch.object(ui_shared, "get_gradio_launch_kwargs", spy), \
             patch.object(ui_shared, "_ui_credentials_path", return_value=cred_path), \
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
        cred_path = _patched_credentials_path(self)
        with patch.object(facade_mod, "build_ui", return_value=demo), \
             patch.object(facade_mod, "_find_available_port", return_value=7860), \
             patch.object(ui_shared, "get_gradio_launch_kwargs", spy), \
             patch.object(ui_shared, "_ui_credentials_path", return_value=cred_path), \
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
