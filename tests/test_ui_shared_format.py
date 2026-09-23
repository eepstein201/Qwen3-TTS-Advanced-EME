"""Task 1.3 (Step 7A): the UI's single set of display-format helpers.

Every helper is pure and renders unknown/invalid input as an em dash, so the
UI never shows ``nan``, ``-3s`` or ``None``. ``fmt_size`` must agree with the
CLI's ``tools._shared._format_size`` (the cross-layer drift guard).

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_ui_shared_format.py -v
"""

import datetime
import unittest

try:
    import gradio  # noqa: F401

    HAS_GRADIO = True
except ImportError:  # pragma: no cover - environment without the ui extra
    HAS_GRADIO = False

DASH = "—"
_INVALID = (None, -1, -0.5, float("inf"), float("-inf"), float("nan"))


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestFormatHelpers(unittest.TestCase):
    def setUp(self):
        from qwen3_tts.interface.ui import shared

        self.shared = shared

    def _check(self, fn, cases):
        for value, expected in cases:
            with self.subTest(fn=fn.__name__, value=value):
                self.assertEqual(fn(value), expected)

    def test_fmt_duration(self):
        self._check(
            self.shared.fmt_duration,
            [
                (0, "0:00"),
                (5, "0:05"),
                (65.4, "1:05"),
                (3599, "59:59"),
                (3600, "1:00:00"),
                (3725, "1:02:05"),
            ]
            + [(v, DASH) for v in _INVALID],
        )

    def test_fmt_size(self):
        self._check(
            self.shared.fmt_size,
            [
                (0, "0.0 B"),
                (512, "512.0 B"),
                (1536, "1.5 KB"),
                (1024**2, "1.0 MB"),
                (int(2.5 * 1024**3), "2.5 GB"),
            ]
            + [(v, DASH) for v in _INVALID],
        )

    def test_fmt_size_matches_the_cli_formatter(self):
        from qwen3_tts.tools._shared import _format_size

        for value in (0, 512, 1536, 1024**2, 3 * 1024**3):
            with self.subTest(value=value):
                self.assertEqual(self.shared.fmt_size(value), _format_size(value))

    def test_fmt_eta(self):
        self._check(
            self.shared.fmt_eta,
            [
                (0, "~0s"),
                (12, "~12s"),
                (12.6, "~13s"),
                (60, "~1m"),
                (80, "~1m 20s"),
                (3600, "~1h"),
                (3900, "~1h 5m"),
            ]
            + [(v, DASH) for v in _INVALID],
        )

    def test_fmt_memory_mb(self):
        self._check(
            self.shared.fmt_memory_mb,
            [
                (2500, "2500 MB"),
                (2500.4, "2500 MB"),
                (512.25, "512.2 MB"),
                (999.94, "999.9 MB"),
            ]
            + [(v, DASH) for v in (*_INVALID, 0)],
        )

    def test_format_history_time_today_is_clock_time(self):
        now = datetime.datetime(2026, 9, 6, 18, 0, 0).timestamp()
        ts = datetime.datetime(2026, 9, 6, 14, 32, 7).timestamp()
        self.assertEqual(self.shared.format_history_time(ts, now=now), "14:32:07")

    def test_format_history_time_other_day_has_the_date(self):
        now = datetime.datetime(2026, 9, 7, 9, 0, 0).timestamp()
        ts = datetime.datetime(2026, 9, 6, 14, 32, 7).timestamp()
        self.assertEqual(self.shared.format_history_time(ts, now=now), "Sep 6 14:32")

    def test_format_history_time_falsy_is_dash(self):
        for ts in (0, None):
            with self.subTest(ts=ts):
                self.assertEqual(self.shared.format_history_time(ts), DASH)


if __name__ == "__main__":
    unittest.main()
