"""Tests for audio pipeline — inference configuration behavior."""
import inspect
import pytest


class TestMetalRetrySilenceGap:
    """Tests for R-27: Metal retry path should not use hardcoded silence gap."""

    def test_metal_retry_does_not_hardcode_silence(self):
        """Metal retry sub-chunk concatenation must not use sr * 0.1 literal.

        The silence gap should come from config (generation.silence_gap_seconds),
        not be hardcoded to 0.1 seconds.
        """
        from qwen3_tts.core.engine import inference

        source = inspect.getsource(inference._run_inference_single)
        assert "sr * 0.1" not in source, (
            "Metal retry still uses hardcoded 0.1s silence — should read from config"
        )
