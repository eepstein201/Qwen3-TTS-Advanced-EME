#!/usr/bin/env python3
"""Unit tests for the E2E gradio-version guard.

The E2E harnesses launch the Gradio UI with ``subprocess.Popen([sys.executable,
...])``, so the version under test is whichever interpreter ran pytest rather
than whatever pyproject.toml pins. ``assert_supported_gradio`` exists to make
that mismatch loud instead of silent.
"""

import sys
import types
import unittest
from unittest.mock import patch

from tests.e2e_helpers import BANNED_GRADIO_PREFIXES, assert_supported_gradio


def _fake_gradio(version):
    mod = types.ModuleType("gradio")
    mod.__version__ = version
    return mod


class TestAssertSupportedGradio(unittest.TestCase):
    def test_raises_on_banned_version(self):
        with patch.dict(sys.modules, {"gradio": _fake_gradio("6.14.0")}):
            with self.assertRaises(RuntimeError) as ctx:
                assert_supported_gradio()
        msg = str(ctx.exception)
        self.assertIn("6.14.0", msg)
        self.assertIn("conda run -n qwen3-tts-mlx", msg, "must tell the caller the fix")

    def test_raises_on_any_banned_patch_release(self):
        with patch.dict(sys.modules, {"gradio": _fake_gradio("6.14.7")}):
            with self.assertRaises(RuntimeError):
                assert_supported_gradio()

    def test_allows_supported_versions(self):
        for version in ("6.8.0", "6.20.0", "6.15.2"):
            with self.subTest(version=version):
                with patch.dict(sys.modules, {"gradio": _fake_gradio(version)}):
                    assert_supported_gradio()  # must not raise

    def test_allows_version_that_merely_starts_similarly(self):
        # 6.1.4 / 6.140.x must not be caught by a sloppy prefix match
        for version in ("6.1.4", "6.140.0"):
            with self.subTest(version=version):
                with patch.dict(sys.modules, {"gradio": _fake_gradio(version)}):
                    assert_supported_gradio()  # must not raise

    def test_no_gradio_installed_is_not_an_error(self):
        # Each harness has its own skip path when gradio is absent.
        real_import = __builtins__["__import__"] if isinstance(
            __builtins__, dict
        ) else __builtins__.__import__

        def _raise_for_gradio(name, *args, **kwargs):
            if name == "gradio":
                raise ImportError("no gradio")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_raise_for_gradio):
            assert_supported_gradio()  # must not raise

    def test_missing_version_attribute_is_not_an_error(self):
        mod = types.ModuleType("gradio")  # no __version__
        with patch.dict(sys.modules, {"gradio": mod}):
            assert_supported_gradio()  # must not raise

    def test_banned_prefixes_are_dotted(self):
        # A prefix like "6.14" (no trailing dot) would also match 6.140.x.
        for prefix in BANNED_GRADIO_PREFIXES:
            self.assertTrue(
                prefix.endswith("."),
                f"{prefix!r} must end with '.' to avoid matching 6.140.x",
            )


if __name__ == "__main__":
    unittest.main()
