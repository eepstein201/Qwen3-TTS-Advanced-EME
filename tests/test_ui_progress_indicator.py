#!/usr/bin/env python3
"""Phase 1b — ProgressIndicator component + 5 UI wiring points.

RED tests for:
  * ProgressIndicator class (bounded + indeterminate modes)
  * poll_model_load_progress helper (uses new server `loading: bool` field)
  * 5 wiring points use the new component:
      - model_management.toggle_model     (load/unload)
      - model_management.toggle_asr       (load ASR)
      - voice_management.auto_transcribe_audio
      - shared.enhance_description_with_ai
      - generation._generate_server_side  (chunk counter)

No GPU, models, or running server required.

Run: pytest tests/test_ui_progress_indicator.py -v --tb=short
"""

import inspect
import unittest
from unittest.mock import MagicMock, patch

try:
    import gradio  # noqa: F401
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

_skip = unittest.skipUnless(HAS_GRADIO, "requires gradio")


# ---------------------------------------------------------------------------
# ProgressIndicator component (extends components.py)
# ---------------------------------------------------------------------------

@_skip
class TestProgressIndicatorBounded(unittest.TestCase):
    """ProgressIndicator(mode='bounded') renders WCAG-compliant progressbar."""

    def test_bounded_emits_progressbar_role(self):
        from qwen3_tts.interface.ui.components import ProgressIndicator

        html = ProgressIndicator(percent=42, mode="bounded").render()

        self.assertIn('role="progressbar"', html)

    def test_bounded_emits_aria_valuenow(self):
        from qwen3_tts.interface.ui.components import ProgressIndicator

        html = ProgressIndicator(percent=42, mode="bounded").render()

        self.assertIn('aria-valuenow="42"', html)

    def test_bounded_emits_aria_min_max(self):
        from qwen3_tts.interface.ui.components import ProgressIndicator

        html = ProgressIndicator(percent=42, mode="bounded").render()

        self.assertIn('aria-valuemin="0"', html)
        self.assertIn('aria-valuemax="100"', html)

    def test_bounded_shows_eta_when_provided(self):
        from qwen3_tts.interface.ui.components import ProgressIndicator

        html = ProgressIndicator(percent=42, eta_s=15, mode="bounded").render()

        # ETA must be visible in some form (e.g. "~15s" or "15s remaining")
        self.assertTrue(
            "15" in html and ("s" in html or "sec" in html.lower()),
            f"ETA not surfaced in bounded render: {html!r}",
        )

    def test_bounded_clamps_percent_to_0_100(self):
        from qwen3_tts.interface.ui.components import ProgressIndicator

        # Negative
        html_low = ProgressIndicator(percent=-5, mode="bounded").render()
        self.assertIn('aria-valuenow="0"', html_low)

        # Over 100
        html_high = ProgressIndicator(percent=150, mode="bounded").render()
        self.assertIn('aria-valuenow="100"', html_high)


@_skip
class TestProgressIndicatorIndeterminate(unittest.TestCase):
    """ProgressIndicator(mode='indeterminate') renders busy state."""

    def test_indeterminate_emits_progressbar_role(self):
        from qwen3_tts.interface.ui.components import ProgressIndicator

        html = ProgressIndicator(mode="indeterminate", message="Loading…").render()

        self.assertIn('role="progressbar"', html)

    def test_indeterminate_emits_aria_busy_true(self):
        from qwen3_tts.interface.ui.components import ProgressIndicator

        html = ProgressIndicator(mode="indeterminate", message="Loading…").render()

        self.assertIn('aria-busy="true"', html)

    def test_indeterminate_omits_aria_valuenow(self):
        from qwen3_tts.interface.ui.components import ProgressIndicator

        html = ProgressIndicator(mode="indeterminate", message="Loading…").render()

        self.assertNotIn("aria-valuenow", html)

    def test_indeterminate_includes_message(self):
        from qwen3_tts.interface.ui.components import ProgressIndicator

        html = ProgressIndicator(mode="indeterminate", message="Transcribing…").render()

        self.assertIn("Transcribing", html)


@_skip
class TestProgressIndicatorXSSSafe(unittest.TestCase):
    """ProgressIndicator escapes user content."""

    def test_message_is_html_escaped(self):
        from qwen3_tts.interface.ui.components import ProgressIndicator

        html = ProgressIndicator(
            mode="indeterminate", message="<script>alert(1)</script>"
        ).render()

        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)


# ---------------------------------------------------------------------------
# poll_model_load_progress helper (new in components.py)
# ---------------------------------------------------------------------------

@_skip
class TestPollModelLoadProgress(unittest.TestCase):
    """poll_model_load_progress returns structured progress dict."""

    def _mock_models_response(self, *, loaded, loading, memory_mb=2500, load_time_sec=None):
        return {
            "models": {
                "clone": {
                    "loaded": loaded,
                    "loading": loading,
                    "memory_mb": memory_mb,
                    "load_time_sec": load_time_sec,
                },
                "design": {"loaded": False, "loading": False, "memory_mb": 0},
                "custom": {"loaded": False, "loading": False, "memory_mb": 0},
            },
        }

    def test_returns_dict_with_required_keys(self):
        from qwen3_tts.interface.ui.components import poll_model_load_progress

        with patch("qwen3_tts.interface.ui.components.is_server_running", return_value=True), \
             patch("qwen3_tts.interface.ui.components.load_config", return_value={}), \
             patch("qwen3_tts.core.http_client.server_request") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: self._mock_models_response(loaded=False, loading=True),
            )
            result = poll_model_load_progress("clone")

        self.assertIsInstance(result, dict)
        self.assertIn("state", result)
        self.assertIn("memory_mb", result)
        self.assertIn("eta_s", result)

    def test_state_loading_when_server_says_loading(self):
        from qwen3_tts.interface.ui.components import poll_model_load_progress

        with patch("qwen3_tts.interface.ui.components.is_server_running", return_value=True), \
             patch("qwen3_tts.interface.ui.components.load_config", return_value={}), \
             patch("qwen3_tts.core.http_client.server_request") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: self._mock_models_response(loaded=False, loading=True),
            )
            result = poll_model_load_progress("clone")

        self.assertEqual(result["state"], "loading")

    def test_state_loaded_when_done(self):
        from qwen3_tts.interface.ui.components import poll_model_load_progress

        with patch("qwen3_tts.interface.ui.components.is_server_running", return_value=True), \
             patch("qwen3_tts.interface.ui.components.load_config", return_value={}), \
             patch("qwen3_tts.core.http_client.server_request") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: self._mock_models_response(
                    loaded=True, loading=False, memory_mb=3500, load_time_sec=33.0
                ),
            )
            result = poll_model_load_progress("clone")

        self.assertEqual(result["state"], "loaded")
        self.assertEqual(result["memory_mb"], 3500)


# ---------------------------------------------------------------------------
# 5 wiring points — each handler must use ProgressIndicator
# ---------------------------------------------------------------------------

@_skip
class TestModelLoadHandlerUsesProgressIndicator(unittest.TestCase):
    """toggle_model('clone','load') wires ProgressIndicator for live feedback."""

    def test_toggle_model_load_imports_progress_indicator(self):
        """Source-level: handler module references ProgressIndicator."""
        from qwen3_tts.interface.ui import model_management

        src = inspect.getsource(model_management)
        self.assertIn(
            "ProgressIndicator",
            src,
            "model_management.py does not use ProgressIndicator",
        )

    def test_toggle_model_load_imports_poll_helper(self):
        """Source-level: load ETA is wired via the badge renderer.

        toggle_model no longer calls poll_model_load_progress (the discarded
        pre-load probe): the blocking POST makes an in-handler indicator
        unrenderable, so the live ETA is surfaced by the shared badge Timer
        through _badge_from_models_payload reading load_time_sec from /models.
        """
        from qwen3_tts.interface.ui import model_management

        src = inspect.getsource(model_management._badge_from_models_payload)
        self.assertIn(
            "load_time_sec",
            src,
            "loading badge does not surface the prior-load ETA",
        )


@_skip
class TestASRLoadHandlerUsesProgressIndicator(unittest.TestCase):
    """toggle_asr('load') uses indeterminate ProgressIndicator."""

    def test_toggle_asr_emits_indeterminate_progress(self):
        from qwen3_tts.interface.ui import model_management

        src = inspect.getsource(model_management.toggle_asr)
        self.assertIn(
            "ProgressIndicator",
            src,
            "toggle_asr does not use ProgressIndicator for indeterminate progress",
        )


@_skip
class TestAIEnhancementUsesProgressIndicator(unittest.TestCase):
    """enhance_description_with_ai shows inline 'Enhancing…' progress."""

    def test_enhance_handler_uses_progress_indicator(self):
        from qwen3_tts.interface.ui import shared

        src = inspect.getsource(shared.enhance_description_with_ai)
        self.assertIn(
            "ProgressIndicator",
            src,
            "enhance_description_with_ai does not show inline progress",
        )


@_skip
class TestAutoTranscribeUsesProgressIndicator(unittest.TestCase):
    """auto_transcribe_audio shows inline 'Transcribing…' spinner."""

    def test_auto_transcribe_uses_progress_indicator(self):
        from qwen3_tts.interface.ui import voice_management

        src = inspect.getsource(voice_management.auto_transcribe_audio)
        self.assertIn(
            "ProgressIndicator",
            src,
            "auto_transcribe_audio does not show inline progress",
        )


@_skip
class TestStreamingChunkCounterSurfaced(unittest.TestCase):
    """generation._generate_server_side surfaces chunk counter from R-51 plumbing."""

    def test_generation_surfaces_chunk_count(self):
        from qwen3_tts.interface.ui import generation

        src = inspect.getsource(generation._generate_server_side)
        # Either the new component or a "Chunk N of M" pattern derived from
        # client.last_chunk_count must appear in the handler.
        has_progress = "ProgressIndicator" in src or "last_chunk_count" in src
        self.assertTrue(
            has_progress,
            "_generate_server_side does not surface chunk counter (R-51) for live progress",
        )


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------

@_skip
class TestComponentsExportsProgressIndicator(unittest.TestCase):
    """ProgressIndicator and poll_model_load_progress are public symbols."""

    def test_progress_indicator_is_importable(self):
        # Plain import — failing means the symbol doesn't exist yet.
        from qwen3_tts.interface.ui.components import ProgressIndicator  # noqa: F401

    def test_poll_model_load_progress_is_importable(self):
        from qwen3_tts.interface.ui.components import (
            poll_model_load_progress,  # noqa: F401
        )


if __name__ == "__main__":
    unittest.main()
