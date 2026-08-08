#!/usr/bin/env python3
"""PRF-6: post-hoc rate control for clone mode.

The model's own rate control is broken for voice cloning (upstream #290 —
output lands at 41-48 s no matter what is asked for), so a requested rate is
applied after generation with the existing pyrubberband/librosa time-stretch
helper. Design and custom modes keep native `instruct` rate control and must
not be stretched twice.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_clone_rate_control.py -v

No GPU, models, or running server required.
"""

import importlib.util
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
_HAS_STRETCH = bool(importlib.util.find_spec("librosa")) or bool(
    importlib.util.find_spec("pyrubberband")
)
_skip_stretch = unittest.skipUnless(
    _HAS_STRETCH, "needs librosa or pyrubberband for real time-stretch"
)


def _speech_like(seconds=1.0, sr=SR):
    """A voiced-ish tone; real stretching needs actual signal, not silence."""
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.4 * np.sin(2 * np.pi * 160.0 * t)).astype(np.float32)


def _cfg(**generation):
    return {"generation": generation}


@pytest.mark.unit
class TestResolveCloneSpeed(unittest.TestCase):
    """Where the requested rate comes from and which values are honoured."""

    def test_gen_params_speed_is_used(self):
        from qwen3_tts.core.engine.inference import _resolve_clone_speed

        self.assertEqual(_resolve_clone_speed({"speed": 1.25}, _cfg()), 1.25)

    def test_config_fallback(self):
        from qwen3_tts.core.engine.inference import _resolve_clone_speed

        self.assertEqual(_resolve_clone_speed({}, _cfg(clone_speed=0.8)), 0.8)

    def test_gen_params_overrides_config(self):
        from qwen3_tts.core.engine.inference import _resolve_clone_speed

        self.assertEqual(
            _resolve_clone_speed({"speed": 1.5}, _cfg(clone_speed=0.8)), 1.5
        )

    def test_unset_returns_none(self):
        from qwen3_tts.core.engine.inference import _resolve_clone_speed

        self.assertIsNone(_resolve_clone_speed({}, _cfg()))

    def test_unity_returns_none(self):
        """1.0 means 'leave it alone' — don't pay for a no-op stretch."""
        from qwen3_tts.core.engine.inference import _resolve_clone_speed

        self.assertIsNone(_resolve_clone_speed({"speed": 1.0}, _cfg()))

    def test_non_numeric_is_ignored(self):
        from qwen3_tts.core.engine.inference import _resolve_clone_speed

        self.assertIsNone(_resolve_clone_speed({"speed": "fast"}, _cfg()))

    def test_non_positive_is_ignored(self):
        from qwen3_tts.core.engine.inference import _resolve_clone_speed

        self.assertIsNone(_resolve_clone_speed({"speed": 0.0}, _cfg()))
        self.assertIsNone(_resolve_clone_speed({"speed": -2.0}, _cfg()))

    def test_extremes_are_clamped(self):
        """A 20x request would destroy the audio, not speed it up."""
        from qwen3_tts.core.engine.inference import _resolve_clone_speed

        self.assertEqual(_resolve_clone_speed({"speed": 20.0}, _cfg()), 2.0)
        self.assertEqual(_resolve_clone_speed({"speed": 0.01}, _cfg()), 0.5)


@pytest.mark.unit
class TestMaybeApplySpeedMode(unittest.TestCase):
    """Only clone mode gets post-hoc control."""

    def test_clone_invokes_time_stretch(self):
        from qwen3_tts.core.engine.inference import _maybe_apply_speed

        audio = _speech_like(0.2)
        with patch(
            "qwen3_tts.core.engine.inference.process_audio",
            side_effect=lambda a, sr, **kw: a,
        ) as proc:
            _maybe_apply_speed(audio, SR, {"speed": 1.5}, "clone", config=_cfg())

        self.assertEqual(proc.call_args.kwargs["speed"], 1.5)

    def test_design_is_untouched(self):
        from qwen3_tts.core.engine.inference import _maybe_apply_speed

        audio = _speech_like(0.2)
        with patch("qwen3_tts.core.engine.inference.process_audio") as proc:
            out, sr = _maybe_apply_speed(
                audio, SR, {"speed": 1.5}, "design", config=_cfg()
            )

        proc.assert_not_called()
        np.testing.assert_array_equal(out, audio)
        self.assertEqual(sr, SR)

    def test_custom_is_untouched(self):
        from qwen3_tts.core.engine.inference import _maybe_apply_speed

        audio = _speech_like(0.2)
        with patch("qwen3_tts.core.engine.inference.process_audio") as proc:
            _maybe_apply_speed(audio, SR, {"speed": 1.5}, "custom", config=_cfg())

        proc.assert_not_called()

    def test_always_returns_pair(self):
        from qwen3_tts.core.engine.inference import _maybe_apply_speed

        result = _maybe_apply_speed(_speech_like(0.2), SR, {}, "clone", config=_cfg())
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_stretch_failure_returns_original_audio(self):
        """A broken rubberband install must not fail the whole generation."""
        from qwen3_tts.core.engine.inference import _maybe_apply_speed

        audio = _speech_like(0.2)
        with patch(
            "qwen3_tts.core.engine.inference.process_audio",
            side_effect=RuntimeError("rubberband missing"),
        ):
            out, sr = _maybe_apply_speed(
                audio, SR, {"speed": 1.5}, "clone", config=_cfg()
            )

        np.testing.assert_array_equal(out, audio)
        self.assertEqual(sr, SR)


@_skip_stretch
@pytest.mark.unit
class TestCloneDurationActuallyChanges(unittest.TestCase):
    """The acceptance criterion: duration tracks the rate factor for real."""

    def _duration(self, audio, sr=SR):
        return len(audio) / sr

    def test_faster_rate_shortens_audio(self):
        from qwen3_tts.core.engine.inference import _maybe_apply_speed

        audio = _speech_like(1.0)
        out, sr = _maybe_apply_speed(audio, SR, {"speed": 2.0}, "clone", config=_cfg())

        # 2x speed => about half the duration (allow resampler slack).
        self.assertAlmostEqual(self._duration(out, sr), 0.5, delta=0.08)

    def test_slower_rate_lengthens_audio(self):
        from qwen3_tts.core.engine.inference import _maybe_apply_speed

        audio = _speech_like(1.0)
        out, sr = _maybe_apply_speed(audio, SR, {"speed": 0.5}, "clone", config=_cfg())

        self.assertAlmostEqual(self._duration(out, sr), 2.0, delta=0.15)

    def test_duration_is_monotonic_in_rate(self):
        from qwen3_tts.core.engine.inference import _maybe_apply_speed

        audio = _speech_like(1.0)
        durations = []
        for speed in (0.75, 1.25, 1.75):
            out, sr = _maybe_apply_speed(
                audio, SR, {"speed": speed}, "clone", config=_cfg()
            )
            durations.append(self._duration(out, sr))

        self.assertGreater(durations[0], durations[1])
        self.assertGreater(durations[1], durations[2])

    def test_unity_rate_leaves_length_untouched(self):
        from qwen3_tts.core.engine.inference import _maybe_apply_speed

        audio = _speech_like(1.0)
        out, _ = _maybe_apply_speed(audio, SR, {"speed": 1.0}, "clone", config=_cfg())
        self.assertEqual(len(out), len(audio))

    def test_sample_rate_is_preserved(self):
        """Rate control changes duration, never the sample rate."""
        from qwen3_tts.core.engine.inference import _maybe_apply_speed

        _, sr = _maybe_apply_speed(
            _speech_like(0.5), SR, {"speed": 1.5}, "clone", config=_cfg()
        )
        self.assertEqual(sr, SR)


@pytest.mark.unit
class TestWiredIntoRunInference(unittest.TestCase):
    """Guard against the helper existing but never being called."""

    def _source(self):
        import inspect

        from qwen3_tts.core.engine.inference import run_inference

        return inspect.getsource(run_inference)

    def test_run_inference_applies_speed(self):
        self.assertIn("_maybe_apply_speed", self._source())

    def test_both_single_and_multi_chunk_paths_apply_speed(self):
        """Single-chunk returns early, so each path needs its own call."""
        self.assertGreaterEqual(self._source().count("_maybe_apply_speed"), 2)

    def test_speed_is_applied_before_lufs(self):
        """LUFS must measure the stretched audio, not the pre-stretch audio."""
        source = self._source()
        self.assertLess(
            source.index("_maybe_apply_speed"),
            source.index("_maybe_apply_lufs"),
        )


if __name__ == "__main__":
    unittest.main()
