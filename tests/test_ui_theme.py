"""Task 1.1 (Step 7A): the ``ui/theme.py`` design-token layer.

``TOKENS`` is the one canonical fallback set; ``var()`` and ``UI_CSS`` both
derive from it, so a token can never be styled with a value the stylesheet
does not declare. Gradio 6 moved ``css``/``theme`` from ``gr.Blocks()`` to
``launch()``, so ``get_gradio_launch_kwargs`` stays the single wiring site.

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_ui_theme.py -v
"""

import ast
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import gradio  # noqa: F401

    HAS_GRADIO = True
except ImportError:  # pragma: no cover - environment without the ui extra
    HAS_GRADIO = False

_THEME_PATH = (
    Path(__file__).resolve().parent.parent
    / "qwen3_tts"
    / "interface"
    / "ui"
    / "theme.py"
)
_EXPECTED_TOKENS = {
    "surface",
    "surface_alt",
    "border",
    "border_strong",
    "accent",
    "accent_soft",
    "text",
    "text_subtle",
    "text_inverse",
    "success",
    "warning",
    "error",
    "info",
    "loading",
    "focus",
    "radius_sm",
    "radius_md",
    "radius_lg",
    "dur_fast",
    "dur_slow",
}
_SEVERITIES = {"info", "success", "warning", "error", "loading"}


class TestThemeTokens(unittest.TestCase):
    def setUp(self):
        from qwen3_tts.interface.ui import theme

        self.theme = theme

    def test_token_keys_are_the_documented_set(self):
        self.assertEqual(set(self.theme.TOKENS), _EXPECTED_TOKENS)

    def test_stylesheet_declares_exactly_the_tokens(self):
        declared = re.findall(r"--tts-([a-z_]+):", self.theme.UI_CSS)
        self.assertEqual(sorted(declared), sorted(self.theme.TOKENS))

    def test_stylesheet_values_match_the_fallbacks(self):
        for key, value in self.theme.TOKENS.items():
            self.assertIn(f"--tts-{key}:{value};", self.theme.UI_CSS)

    def test_var_uses_the_canonical_fallback(self):
        fallback = self.theme.TOKENS["accent"]
        self.assertEqual(self.theme.var("accent"), f"var(--tts-accent,{fallback})")

    def test_var_rejects_unknown_tokens(self):
        with self.assertRaises(KeyError):
            self.theme.var("no_such_token")

    def test_loading_is_distinct_from_info(self):
        self.assertNotEqual(self.theme.TOKENS["loading"], self.theme.TOKENS["info"])

    def test_severity_classes_cover_every_severity_and_are_styled(self):
        self.assertEqual(
            dict(self.theme.SEVERITY_CLASS), {s: f"tts-sev-{s}" for s in _SEVERITIES}
        )
        for cls in self.theme.SEVERITY_CLASS.values():
            self.assertIn(f".{cls}", self.theme.UI_CSS)


class TestThemeStylesheet(unittest.TestCase):
    def setUp(self):
        from qwen3_tts.interface.ui.theme import UI_CSS

        self.css = UI_CSS

    def test_required_rules_present(self):
        for needle in (
            ".gr-hidden",
            ".tts-num",
            "tabular-nums",
            ":focus-visible",
            ".tts-history td:nth-child(6)",
            ".tts-history td:nth-child(7)",
            ".tts-empty",
            "prefers-reduced-motion",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.css)

    def test_gr_hidden_rule_is_preserved(self):
        # JS-bridge components rely on it (visible=False is banned for them).
        self.assertRegex(self.css, r"\.gr-hidden\s*\{[^}]*display:\s*none\s*!important")

    def test_never_transitions_all_properties(self):
        self.assertNotRegex(self.css, r"transition(-property)?\s*:\s*all\b")

    def test_reduced_motion_block_zeroes_transitions(self):
        block = self.css.split("prefers-reduced-motion", 1)[1]
        self.assertRegex(block, r"transition(-duration)?\s*:\s*(none|0s?)\b")


class TestThemeIsImportLight(unittest.TestCase):
    def test_only_stdlib_imports(self):
        tree = ast.parse(_THEME_PATH.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        self.assertTrue(roots <= set(sys.stdlib_module_names) | {"__future__"}, roots)


@unittest.skipUnless(HAS_GRADIO, "gradio not installed")
class TestThemeWiring(unittest.TestCase):
    @patch("qwen3_tts.core.config.IN_COLAB", False)
    def test_launch_kwargs_carry_the_theme_stylesheet(self):
        from qwen3_tts.interface.ui import theme
        from qwen3_tts.interface.ui.shared import get_gradio_launch_kwargs

        self.assertEqual(get_gradio_launch_kwargs({})["css"], theme.UI_CSS)

    def test_history_table_carries_the_history_class(self):
        from qwen3_tts.interface.ui._facade import build_ui

        config = build_ui().get_config_file()
        history = [
            c
            for c in config["components"]
            if c.get("type") == "dataframe"
            and "Remove" in (c.get("props", {}).get("headers") or [])
        ]
        self.assertEqual(len(history), 1)
        self.assertIn("tts-history", history[0]["props"].get("elem_classes") or [])


if __name__ == "__main__":
    unittest.main()
