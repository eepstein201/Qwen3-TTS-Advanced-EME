#!/usr/bin/env python3
"""Regression guard: no ``select`` listener may be attached to a ``gr.Tab``.

Background
----------
The per-mode "model loaded" badges used to be refreshed by a ``select``
listener on each ``gr.Tab`` (Clone/Design/Custom). With such a listener
attached, Gradio 6.14+ recurses infinitely inside the Dataframe frontend the
next time a tab containing a ``gr.Dataframe`` is opened::

    RangeError: Maximum call stack size exceeded
        at Object.get [as groupedColumnMode] (assets/Index-*.js)

The uncaught error kills the whole page, so the Manage Voices / Manage Models
tabs render nothing. The crash follows the *listener*, not its outputs (it
still happens when the handler writes to a component outside the tabs), and
``gr.Tabs.select`` never fires, so the badges are refreshed by the shared
status ``gr.Timer`` instead.

These tests fail if anyone re-introduces a ``gr.Tab``-scoped ``select``
listener, and pin the timer wiring that replaced it.

Run: pytest tests/test_ui_tab_select_wiring.py -v
"""

import unittest

try:
    import gradio  # noqa: F401

    HAS_GRADIO = True
except ImportError:  # pragma: no cover - environment without the ui extra
    HAS_GRADIO = False


@unittest.skipUnless(HAS_GRADIO, "gradio not installed")
class TestNoTabScopedSelectListener(unittest.TestCase):
    """The built Blocks config must contain no tabitem-triggered select event."""

    @classmethod
    def setUpClass(cls):
        from qwen3_tts.interface.ui._facade import build_ui

        cls.demo = build_ui()
        cls.config = cls.demo.get_config_file()
        cls.components = {c["id"]: c for c in cls.config["components"]}

    def _targets(self, dep):
        """Yield (component_id, event_name) for every trigger of *dep*."""
        for target in dep.get("targets", []) or []:
            if isinstance(target, (list, tuple)) and len(target) >= 2:
                yield target[0], target[1]

    def test_no_select_listener_on_any_tab(self):
        offenders = []
        for dep in self.config["dependencies"]:
            for comp_id, event in self._targets(dep):
                comp = self.components.get(comp_id, {})
                if comp.get("type") == "tabitem" and event == "select":
                    offenders.append((comp_id, comp.get("props", {}).get("label")))

        self.assertEqual(
            offenders,
            [],
            "A select listener is attached to a gr.Tab: this crashes the Gradio "
            "6.14+ Dataframe frontend with 'Maximum call stack size exceeded'. "
            f"Offending tabs: {offenders}",
        )

    def test_model_indicators_are_refreshed_by_a_timer(self):
        """The badges must still be kept live after dropping the tab listeners."""
        timer_ids = {
            cid for cid, c in self.components.items() if c.get("type") == "timer"
        }
        self.assertTrue(timer_ids, "expected a gr.Timer driving the status polling")

        wired = False
        for dep in self.config["dependencies"]:
            triggers = {cid for cid, _ in self._targets(dep)}
            if not triggers & timer_ids:
                continue
            outputs = dep.get("outputs", []) or []
            if len(outputs) == 3 and all(
                self.components.get(o, {}).get("type") == "html" for o in outputs
            ):
                wired = True
        self.assertTrue(
            wired,
            "no timer tick refreshes the three per-mode model status indicators",
        )


class TestGetAllModelStatusHtml(unittest.TestCase):
    """get_all_model_status_html must mirror get_model_status_html per model."""

    def test_returns_one_badge_per_indicator_type(self):
        from unittest.mock import patch

        from qwen3_tts.interface.ui import model_management as mm

        payload = {
            "models": {
                "clone": {"loaded": True, "memory_mb": 2500},
                "design": {"loaded": False, "loading": True},
                "custom": {"loaded": False},
            }
        }

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return payload

        with (
            patch.object(mm, "load_config", return_value={}),
            patch.object(mm, "is_server_running", return_value=True),
            patch(
                "qwen3_tts.core.http_client.server_request", return_value=_Resp()
            ),
        ):
            badges = mm.get_all_model_status_html()

        self.assertEqual(len(badges), len(mm.MODEL_INDICATOR_TYPES))
        self.assertIn("Loaded (2500MB)", badges[0])
        self.assertIn("Loading design", badges[1])
        self.assertIn("Not loaded", badges[2])

    def test_server_down_returns_warning_for_every_indicator(self):
        from unittest.mock import patch

        from qwen3_tts.interface.ui import model_management as mm

        with (
            patch.object(mm, "load_config", return_value={}),
            patch.object(mm, "is_server_running", return_value=False),
        ):
            badges = mm.get_all_model_status_html()

        self.assertEqual(len(badges), len(mm.MODEL_INDICATOR_TYPES))
        for badge in badges:
            self.assertIn("Server not running", badge)


if __name__ == "__main__":
    unittest.main()
