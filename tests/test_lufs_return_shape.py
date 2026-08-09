#!/usr/bin/env python3
"""Regression: _maybe_apply_lufs must always return (audio, sample_rate).

The early-return path returned a tuple but the normalization path returned a
bare array, while both call sites in run_inference unpack two values. Enabling
the documented `generation.lufs_normalize` option therefore crashed every
generation with "too many values to unpack (expected 2)". The option defaults
to false, so nothing exercised it.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_lufs_return_shape.py -v

No GPU, models, or running server required.
"""

import unittest
from unittest.mock import patch

import numpy as np

try:
    import pytest
    HAS_PYTEST = True
except ImportError:  # pragma: no cover
    HAS_PYTEST = False

    class _DummyMarker:
        def __call__(self, func):
            return func

        def __getattr__(self, name):
            return self

    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarker()

    class _DummyPytest:
        mark = _DummyMark()

    pytest = _DummyPytest()


SR = 24000


def _audio(n=2400):
    return np.random.randn(n).astype(np.float32)


@pytest.mark.unit
class TestMaybeApplyLufsReturnShape(unittest.TestCase):
    """Both branches must honour the documented (audio, sample_rate) contract."""

    def test_disabled_returns_pair(self):
        from qwen3_tts.core.engine.inference import _maybe_apply_lufs

        cfg = {"generation": {"lufs_normalize": False}}
        result = _maybe_apply_lufs(_audio(), SR, config=cfg)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_enabled_returns_pair(self):
        """The bug: this path returned the bare array."""
        from qwen3_tts.core.engine.inference import _maybe_apply_lufs

        cfg = {"generation": {"lufs_normalize": True, "lufs_target": -16.0}}
        with patch(
            "qwen3_tts.core.engine.inference.process_audio",
            side_effect=lambda a, sr, **kw: a,
        ):
            result = _maybe_apply_lufs(_audio(), SR, config=cfg)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_enabled_result_unpacks_like_callers_do(self):
        """Mirrors run_inference: `wav, sr = _maybe_apply_lufs(...)`."""
        from qwen3_tts.core.engine.inference import _maybe_apply_lufs

        cfg = {"generation": {"lufs_normalize": True, "lufs_target": -16.0}}
        with patch(
            "qwen3_tts.core.engine.inference.process_audio",
            side_effect=lambda a, sr, **kw: a,
        ):
            wav, sr = _maybe_apply_lufs(_audio(), SR, config=cfg)

        self.assertEqual(sr, SR)
        self.assertEqual(len(wav), 2400)

    def test_enabled_preserves_sample_rate(self):
        from qwen3_tts.core.engine.inference import _maybe_apply_lufs

        cfg = {"generation": {"lufs_normalize": True, "lufs_target": -20.0}}
        with patch(
            "qwen3_tts.core.engine.inference.process_audio",
            side_effect=lambda a, sr, **kw: a,
        ):
            _, sr = _maybe_apply_lufs(_audio(), 16000, config=cfg)

        self.assertEqual(sr, 16000)

    def test_enabled_passes_target_through(self):
        from qwen3_tts.core.engine.inference import _maybe_apply_lufs

        cfg = {"generation": {"lufs_normalize": True, "lufs_target": -20.0}}
        with patch(
            "qwen3_tts.core.engine.inference.process_audio",
            side_effect=lambda a, sr, **kw: a,
        ) as proc:
            _maybe_apply_lufs(_audio(), SR, config=cfg)

        self.assertEqual(proc.call_args.kwargs["lufs_target"], -20.0)

    def test_enabled_returns_processed_audio(self):
        """The normalized audio, not the input, must come back."""
        from qwen3_tts.core.engine.inference import _maybe_apply_lufs

        cfg = {"generation": {"lufs_normalize": True, "lufs_target": -16.0}}
        marker = np.full(2400, 0.25, dtype=np.float32)
        with patch(
            "qwen3_tts.core.engine.inference.process_audio",
            side_effect=lambda a, sr, **kw: marker,
        ):
            wav, _ = _maybe_apply_lufs(_audio(), SR, config=cfg)

        np.testing.assert_array_equal(wav, marker)


if __name__ == "__main__":
    unittest.main()
