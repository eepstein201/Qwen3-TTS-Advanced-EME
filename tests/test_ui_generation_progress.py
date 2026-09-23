"""Task 1.6 (Step 7A): live generation progress + the stop-confirm defect fix.

``/generation-status`` only carries ``progress_pct``/``chunk_total``/``eta_sec``
for authed callers while progress is known. The UI must consume those fields
as given and never derive a percent from ``chunk_index`` alone — unknown
progress takes the confirm path (the safe default).

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_ui_generation_progress.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

try:
    import gradio  # noqa: F401

    HAS_GRADIO = True
except ImportError:  # pragma: no cover - environment without the ui extra
    HAS_GRADIO = False


def _status(payload, status_code=200):
    return MagicMock(status_code=status_code, json=lambda: payload)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestCancelConfirmationProgress(unittest.TestCase):
    def _prepare(self, payload):
        from qwen3_tts.interface.ui import generation

        with patch.object(generation, "load_config", return_value={}), \
                patch.object(generation, "is_server_running", return_value=True), \
                patch("qwen3_tts.core.http_client.server_request",
                      return_value=_status(payload)):
            return generation._prepare_cancel_confirmation()

    def test_absent_progress_takes_the_confirm_path(self):
        msg, should_proceed, pct, chunks, eta = self._prepare(
            {"active": True, "chunk_index": 3}
        )
        self.assertFalse(should_proceed)
        self.assertEqual(
            (msg, pct, chunks, eta),
            ("Stop generation?\nProgress: unknown\nChunks: n/a\nETA: n/a", 0, 0, None),
        )
        self.assertNotIn("300%", msg)

    def test_low_progress_is_the_fast_path(self):
        # chunk ratio alone would read 75%; the server's progress_pct is the truth.
        _, should_proceed, pct, _, _ = self._prepare(
            {"active": True, "chunk_index": 3, "chunk_total": 4, "progress_pct": 7.0}
        )
        self.assertTrue(should_proceed)
        self.assertEqual(pct, 7.0)

    def test_known_progress_confirms_with_server_values(self):
        msg, should_proceed, pct, chunks, eta = self._prepare(
            {
                "active": True,
                "chunk_index": 5,
                "chunk_total": 12,
                "progress_pct": 42.0,
                "eta_sec": 30,
            }
        )
        self.assertFalse(should_proceed)
        self.assertEqual(msg, "Stop generation?\nProgress: 42%\nChunks: 5/12\nETA: ~30s")
        self.assertEqual((pct, chunks, eta), (42.0, 5, 30))



@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestPollGenerationProgress(unittest.TestCase):
    def _poll(self, response=None, side_effect=None):
        from qwen3_tts.interface.ui import generation

        with patch("qwen3_tts.core.http_client.server_request",
                   return_value=response, side_effect=side_effect) as req:
            html = generation.poll_generation_progress()
        return html, req

    def test_known_progress_renders_percent_and_eta(self):
        from qwen3_tts.interface.ui.components import ProgressIndicator

        html, req = self._poll(_status(
            {"active": True, "elapsed_sec": 12.0, "progress_pct": 42.0, "eta_sec": 80.0}
        ))
        expected = ProgressIndicator(
            percent=42.0, eta_s=80.0, message="Generating…"
        ).render()
        self.assertEqual(html, expected)
        self.assertIn("~1m 20s", html)
        req.assert_called_once_with("GET", "/generation-status", timeout=2)

    def test_absent_progress_is_elapsed_only(self):
        from qwen3_tts.interface.ui.components import ProgressIndicator

        html, _ = self._poll(_status(
            {"active": True, "elapsed_sec": 65.0, "chunk_index": 3}
        ))
        expected = ProgressIndicator(
            mode="indeterminate", message="Generating… 1:05 elapsed"
        ).render()
        self.assertEqual(html, expected)
        self.assertNotIn("aria-valuenow", html)
        self.assertNotIn("%", html.split(">", 1)[1])

    def test_nothing_to_show_renders_empty(self):
        cases = {
            "inactive": {"response": _status({"active": False})},
            "non_200": {"response": _status({}, status_code=503)},
            "exception": {"side_effect": OSError("refused")},
        }
        for name, kwargs in cases.items():
            with self.subTest(case=name):
                self.assertEqual(self._poll(**kwargs)[0], "")


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestProgressTimerSteps(unittest.TestCase):
    def test_poll_interval_respects_the_request_budget(self):
        from qwen3_tts.interface.ui import generation

        self.assertGreaterEqual(generation.PROGRESS_POLL_SECONDS, 2.0)

    def test_start_and_stop_toggle_active(self):
        import gradio as gr

        from qwen3_tts.interface.ui import generation

        started, stopped = generation.start_progress_timer(), generation.stop_progress_timer()
        self.assertIsInstance(started, gr.Timer)
        self.assertIsInstance(stopped, gr.Timer)
        self.assertTrue(started.active)
        self.assertFalse(stopped.active)

    def test_idle_tab_tick_makes_no_request(self):
        import gradio as gr

        from qwen3_tts.interface.ui import generation

        for guard in (None, {"generating": False}):
            with self.subTest(guard=guard), patch(
                "qwen3_tts.core.http_client.server_request"
            ) as req:
                html, timer = generation._poll_progress_for_tab(guard)
                req.assert_not_called()
                self.assertEqual(html, "")
                self.assertEqual(timer, gr.skip())

    def test_generating_tab_tick_renders_progress(self):
        import gradio as gr

        from qwen3_tts.interface.ui import generation

        with patch("qwen3_tts.core.http_client.server_request",
                   return_value=_status({"active": True, "elapsed_sec": 5.0})):
            html, timer = generation._poll_progress_for_tab({"generating": True})
        self.assertIn("Generating… 0:05 elapsed", html)
        self.assertEqual(timer, gr.skip())

    def test_tick_past_the_hard_cap_disarms_the_timer(self):
        from qwen3_tts.interface.ui import generation

        elapsed = generation.PROGRESS_POLL_MAX_MINUTES * 60 + 1
        with patch("qwen3_tts.core.http_client.server_request",
                   return_value=_status({"active": True, "elapsed_sec": elapsed})):
            html, timer = generation._poll_progress_for_tab({"generating": True})
        self.assertEqual(html, "")
        self.assertFalse(timer.active)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestProgressTimerWiring(unittest.TestCase):
    """One shared inactive Timer; each tab arms it, disarms it, and gates its tick."""

    @classmethod
    def setUpClass(cls):
        import gradio as gr

        from qwen3_tts.interface.ui import generation
        from qwen3_tts.interface.ui._facade import build_ui

        cls.gr = gr
        cls.generation = generation
        cls.demo = build_ui()
        cls.timers = [
            b for b in cls.demo.blocks.values()
            if isinstance(b, gr.Timer) and b.value == generation.PROGRESS_POLL_SECONDS
        ]
        cls.fns = list(cls.demo.fns.values())

    def _timer_fns(self):
        timer = self.timers[0]
        return [f for f in self.fns if timer in f.outputs]

    def test_exactly_one_progress_timer_inactive_at_build(self):
        self.assertEqual(len(self.timers), 1)
        self.assertFalse(self.timers[0].active)

    def test_each_tab_arms_the_timer(self):
        arms = [f for f in self._timer_fns() if f.fn is self.generation.start_progress_timer]
        self.assertEqual(len(arms), 3)

    def test_each_tab_disarms_the_timer_and_clears_its_progress(self):
        disarms = [
            f for f in self._timer_fns()
            if len(f.outputs) == 2 and isinstance(f.outputs[1], self.gr.HTML)
            and f.targets[0][1] == "then"
        ]
        self.assertEqual(len(disarms), 3)
        self.assertEqual(len({id(f.outputs[1]) for f in disarms}), 3)
        timer_update, html = disarms[0].fn()
        self.assertFalse(timer_update.active)
        self.assertEqual(html, "")

    def test_each_tab_ticks_through_its_own_guard(self):
        ticks = [f for f in self.fns if f.fn is self.generation._poll_progress_for_tab]
        self.assertEqual(len(ticks), 3)
        for f in ticks:
            with self.subTest(fn=f):
                self.assertEqual(f.targets, [(self.timers[0]._id, "tick")])
                self.assertIsInstance(f.inputs[0], self.gr.State)
        self.assertEqual(len({id(f.inputs[0]) for f in ticks}), 3)
        self.assertEqual(len({id(f.outputs[0]) for f in ticks}), 3)


if __name__ == "__main__":
    unittest.main()
