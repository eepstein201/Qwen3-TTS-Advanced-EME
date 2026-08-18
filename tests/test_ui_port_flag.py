#!/usr/bin/env python3
"""`tts ui --port` must actually reach the Gradio launch.

`ui_command()` passes the flag by setting TTS_UI_PORT, but nothing read it
back, so the UI always bound the config/default port. The failure is silent —
the UI starts fine on the wrong port and the user's browser lands on a
connection error with no indication the flag was dropped. Confirmed live: a
process launched with `--port 7866` was listening on 7860.

Its siblings TTS_UI_SHARE / TTS_UI_NO_BROWSER were already honoured; they are
covered here too so the whole trio stays wired.

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_ui_port_flag.py -q
"""
import os
import unittest
from unittest.mock import MagicMock, patch

_UI_ENV = ("TTS_UI_PORT", "TTS_UI_SHARE", "TTS_UI_NO_BROWSER")


class _LaunchHarness(unittest.TestCase):
    """Drive build_ui_and_launch with the UI env vars under our control."""

    def setUp(self):
        # These leak between tests otherwise: build_ui_and_launch reads them
        # from the real environment, and a stray value flips share/inbrowser.
        self._saved = {k: os.environ.pop(k, None) for k in _UI_ENV}
        self.addCleanup(self._restore)

    def _restore(self):
        for key, value in self._saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def _launch(self, config, **env):
        """Return the port that actually reached ``demo.launch()``.

        Asserting only what `_find_available_port` was *called with* would be a
        hollow proxy — it would still pass if the resolved port never reached
        the launch. `_find_available_port` is stubbed to pass its argument
        through (it really does return the preferred port when free), so the
        value observed on `launch(server_port=...)` is the end-to-end result.
        """
        from qwen3_tts.interface.generate_server import build_ui_and_launch

        demo = MagicMock()
        for key, value in env.items():
            os.environ[key] = value

        with patch("qwen3_tts.interface.ui.build_ui", return_value=demo), patch(
            "qwen3_tts.interface.ui._find_available_port",
            side_effect=lambda preferred, **kw: preferred,
        ), patch("qwen3_tts.core.config.IN_COLAB", False):
            build_ui_and_launch(config)

        self.demo = demo
        demo.launch.assert_called_once()
        return demo.launch.call_args[1]["server_port"]


class TestUiPortFlagReachesLaunch(_LaunchHarness):
    def test_env_port_is_used(self):
        requested = self._launch({"ui": {"port": 7860}}, TTS_UI_PORT="7866")

        self.assertEqual(requested, 7866)

    def test_env_port_overrides_config(self):
        """--port is an explicit per-run instruction; config is the default."""
        requested = self._launch({"ui": {"port": 7900}}, TTS_UI_PORT="7866")

        self.assertEqual(requested, 7866)

    def test_config_port_used_when_flag_absent(self):
        requested = self._launch({"ui": {"port": 7900}})

        self.assertEqual(requested, 7900)

    def test_default_port_when_nothing_configured(self):
        requested = self._launch({})

        self.assertEqual(requested, 7860)

    def test_unparseable_env_port_falls_back_instead_of_crashing(self):
        """A bad value must not take down the UI on startup."""
        requested = self._launch({"ui": {"port": 7900}}, TTS_UI_PORT="not-a-port")

        self.assertEqual(requested, 7900)


class TestUiFlagSiblingsStillHonoured(_LaunchHarness):
    """--share and --no-browser were already wired; keep them that way."""

    def test_share_flag_is_honoured(self):
        self._launch({"ui": {"port": 7860}}, TTS_UI_SHARE="1")

        self.assertIs(self.demo.launch.call_args[1]["share"], True)

    def test_no_browser_flag_is_honoured(self):
        self._launch({"ui": {"port": 7860}}, TTS_UI_NO_BROWSER="1")

        self.assertIs(self.demo.launch.call_args[1]["inbrowser"], False)

    def test_browser_opens_by_default(self):
        self._launch({"ui": {"port": 7860}})

        self.assertIs(self.demo.launch.call_args[1]["inbrowser"], True)


if __name__ == "__main__":
    unittest.main()
