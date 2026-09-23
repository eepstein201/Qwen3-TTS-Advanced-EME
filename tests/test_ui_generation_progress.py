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


if __name__ == "__main__":
    unittest.main()
