#!/usr/bin/env python3
"""PRF-2: phase-aligned chunk splices (zero-crossing snap + RMS level match).

The raised-cosine crossfade shape is already correct for correlated speech;
the residual seam artifact comes from splicing at an arbitrary phase and from
level steps between independently generated chunks. These tests cover the two
pre-crossfade corrections and assert the existing crossfade behaviour is
preserved.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_phase_aligned_splices.py -v

No GPU, models, or running server required.
"""

import unittest

import numpy as np

try:
    import pytest
    HAS_PYTEST = True
except ImportError:  # pragma: no cover - pytest always present in test env
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


# ---------------------------------------------------------------------------
# _snap_to_zero_crossing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSnapToZeroCrossing(unittest.TestCase):
    """The splice point should land on a sign change near the chunk head."""

    def test_finds_sign_change(self):
        from qwen3_tts.core.engine.inference import _snap_to_zero_crossing

        # Negative for 5 samples, then positive: crossing is at index 5.
        head = np.array([-3, -2, -2, -1, -1, 1, 2, 3], dtype=np.float32)
        self.assertEqual(_snap_to_zero_crossing(head, max_search=8), 5)

    def test_returns_zero_when_no_crossing(self):
        """A constant (DC) signal has no crossing — splice must not move."""
        from qwen3_tts.core.engine.inference import _snap_to_zero_crossing

        head = np.ones(64, dtype=np.float32) * 0.5
        self.assertEqual(_snap_to_zero_crossing(head, max_search=32), 0)

    def test_respects_search_window(self):
        """A crossing beyond max_search must be ignored."""
        from qwen3_tts.core.engine.inference import _snap_to_zero_crossing

        head = np.concatenate([
            np.ones(50, dtype=np.float32),
            np.ones(50, dtype=np.float32) * -1.0,
        ])
        self.assertEqual(_snap_to_zero_crossing(head, max_search=10), 0)

    def test_prefers_earliest_crossing(self):
        from qwen3_tts.core.engine.inference import _snap_to_zero_crossing

        head = np.array([1, -1, 1, -1, 1], dtype=np.float32)
        self.assertEqual(_snap_to_zero_crossing(head, max_search=5), 1)

    def test_handles_short_and_empty_input(self):
        from qwen3_tts.core.engine.inference import _snap_to_zero_crossing

        self.assertEqual(_snap_to_zero_crossing(np.array([], dtype=np.float32), 8), 0)
        self.assertEqual(_snap_to_zero_crossing(np.array([1.0], np.float32), 8), 0)

    def test_sine_splice_lands_near_zero(self):
        """For a sinusoid, the snapped sample should be near zero amplitude."""
        from qwen3_tts.core.engine.inference import _snap_to_zero_crossing

        t = np.arange(SR // 100, dtype=np.float32) / SR
        head = np.sin(2 * np.pi * 220.0 * t + 1.0).astype(np.float32)
        idx = _snap_to_zero_crossing(head, max_search=int(SR * 0.002))
        self.assertLess(abs(float(head[idx])), 0.15)


# ---------------------------------------------------------------------------
# _align_offset
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAlignOffset(unittest.TestCase):
    """Cross-correlation alignment: the actual phase fix for crossfades."""

    def test_recovers_known_offset(self):
        """A head that starts k samples early must report a lag of k."""
        from qwen3_tts.core.engine.inference import _align_offset

        k = 37
        t = np.arange(SR // 4, dtype=np.float32) / SR
        full = np.sin(2 * np.pi * 200.0 * t).astype(np.float32)
        split = 3000
        tail = full[:split]
        head = full[split - k :]  # starts k samples before the true continuation

        lag = _align_offset(tail, head, max_lag=240)
        self.assertAlmostEqual(lag, k, delta=2)

    def test_aligned_signals_report_zero_lag(self):
        from qwen3_tts.core.engine.inference import _align_offset

        t = np.arange(SR // 4, dtype=np.float32) / SR
        full = np.sin(2 * np.pi * 200.0 * t).astype(np.float32)
        split = 3000
        lag = _align_offset(full[:split], full[split:], max_lag=240)
        self.assertLessEqual(lag, 2)

    def test_zero_max_lag_returns_zero(self):
        from qwen3_tts.core.engine.inference import _align_offset

        sig = _sine(200.0, 2000)
        self.assertEqual(_align_offset(sig, sig, max_lag=0), 0)

    def test_silent_tail_returns_zero(self):
        from qwen3_tts.core.engine.inference import _align_offset

        tail = np.zeros(2000, dtype=np.float32)
        self.assertEqual(_align_offset(tail, _sine(200.0, 2000), max_lag=240), 0)

    def test_short_head_returns_zero(self):
        from qwen3_tts.core.engine.inference import _align_offset

        self.assertEqual(
            _align_offset(_sine(200.0, 2000), np.zeros(4, dtype=np.float32), 240), 0
        )

    def test_short_tail_returns_zero(self):
        from qwen3_tts.core.engine.inference import _align_offset

        self.assertEqual(
            _align_offset(np.zeros(4, dtype=np.float32), _sine(200.0, 2000), 240), 0
        )


# ---------------------------------------------------------------------------
# _seam_gain
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSeamGain(unittest.TestCase):
    """Level matching across the seam, bounded so it can't rescale wildly."""

    def test_equal_levels_gives_unity(self):
        from qwen3_tts.core.engine.inference import _seam_gain

        tail = np.ones(100, dtype=np.float32) * 0.4
        head = np.ones(100, dtype=np.float32) * 0.4
        self.assertAlmostEqual(_seam_gain(tail, head), 1.0, places=5)

    def test_quiet_head_is_boosted(self):
        from qwen3_tts.core.engine.inference import _seam_gain

        tail = np.ones(100, dtype=np.float32) * 0.4
        head = np.ones(100, dtype=np.float32) * 0.2
        self.assertAlmostEqual(_seam_gain(tail, head, max_db=12.0), 2.0, places=4)

    def test_loud_head_is_attenuated(self):
        from qwen3_tts.core.engine.inference import _seam_gain

        tail = np.ones(100, dtype=np.float32) * 0.2
        head = np.ones(100, dtype=np.float32) * 0.4
        self.assertAlmostEqual(_seam_gain(tail, head, max_db=12.0), 0.5, places=4)

    def test_gain_is_clamped(self):
        """A 40 dB mismatch must not produce a 100x gain."""
        from qwen3_tts.core.engine.inference import _seam_gain

        tail = np.ones(100, dtype=np.float32) * 0.5
        head = np.ones(100, dtype=np.float32) * 0.005
        gain = _seam_gain(tail, head, max_db=3.0)
        self.assertLessEqual(gain, 10 ** (3.0 / 20) + 1e-6)
        self.assertGreaterEqual(gain, 10 ** (-3.0 / 20) - 1e-6)

    def test_silent_head_returns_unity(self):
        """Digital silence must not divide by zero or explode the gain."""
        from qwen3_tts.core.engine.inference import _seam_gain

        tail = np.ones(100, dtype=np.float32) * 0.4
        head = np.zeros(100, dtype=np.float32)
        self.assertEqual(_seam_gain(tail, head), 1.0)

    def test_silent_tail_returns_unity(self):
        from qwen3_tts.core.engine.inference import _seam_gain

        tail = np.zeros(100, dtype=np.float32)
        head = np.ones(100, dtype=np.float32) * 0.4
        self.assertEqual(_seam_gain(tail, head), 1.0)

    def test_empty_input_returns_unity(self):
        from qwen3_tts.core.engine.inference import _seam_gain

        empty = np.array([], dtype=np.float32)
        self.assertEqual(_seam_gain(empty, empty), 1.0)


# ---------------------------------------------------------------------------
# Integration with _crossfade_chunks
# ---------------------------------------------------------------------------


def _sine(freq: float, n: int, amp: float = 1.0, phase: float = 0.0):
    t = np.arange(n, dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


@pytest.mark.unit
class TestCrossfadeAlignment(unittest.TestCase):
    """_crossfade_chunks applies the corrections before the raised cosine."""

    def test_level_step_is_reduced_at_seam(self):
        """A 6 dB louder second chunk should not step up 2x across the seam."""
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        n = SR // 2
        quiet = _sine(220.0, n, amp=0.2)
        loud = _sine(220.0, n, amp=0.4)  # +6 dB

        result = _crossfade_chunks([quiet, loud], SR, crossfade_ms=50)

        seam = n
        window = int(SR * 0.05)
        before = float(np.sqrt(np.mean(result[seam - 2 * window : seam - window] ** 2)))
        after = float(np.sqrt(np.mean(result[seam + window : seam + 2 * window] ** 2)))
        ratio = after / before

        # Uncorrected this is ~2.0; the bounded match must pull it toward 1.
        self.assertLess(ratio, 1.7, "level step across the seam was not reduced")
        self.assertGreater(ratio, 0.9, "level match must not invert the step")

    def test_no_energy_dip_at_any_phase_offset(self):
        """The crossfade must not cancel when chunks meet out of phase.

        Crossfading two correlated signals that are out of phase causes
        destructive interference — an audible dip. Unaligned, an anti-phase
        seam drops to ~0.71 of reference RMS (29% energy loss); a
        quarter-period offset to ~0.87. Alignment must hold every offset near
        unity, which is what makes this test discriminating.
        """
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        n = SR // 2
        fade = int(SR * 0.05)
        for phase in (0.0, np.pi / 4, np.pi / 2, np.pi * 0.75, np.pi):
            with self.subTest(phase=phase):
                a = _sine(180.0, n, amp=0.5)
                b = _sine(180.0, n, amp=0.5, phase=phase)
                result = _crossfade_chunks([a, b], SR, crossfade_ms=50)

                ref = float(np.sqrt(np.mean(result[n - 3 * fade : n - 2 * fade] ** 2)))
                seam = float(np.sqrt(np.mean(result[n - fade : n] ** 2)))
                self.assertGreater(
                    seam / ref, 0.93, f"energy dip at the seam for phase {phase}"
                )

    def test_seam_has_no_step_discontinuity(self):
        """Regression guard: the fade itself stays continuous sample-to-sample."""
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        n = SR // 2
        a = _sine(180.0, n, amp=0.5)
        b = _sine(180.0, n, amp=0.5, phase=np.pi * 0.75)
        result = _crossfade_chunks([a, b], SR, crossfade_ms=50)

        transition = result[n - 200 : n + 200]
        max_jump = float(np.max(np.abs(np.diff(transition))))
        # A 180 Hz sine at 24 kHz moves ~0.024/sample at its steepest.
        self.assertLess(max_jump, 0.1, "splice introduced a discontinuity")

    def test_output_is_float32(self):
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        result = _crossfade_chunks([_sine(200.0, 4000), _sine(200.0, 4000)], SR)
        self.assertEqual(result.dtype, np.float32)

    def test_no_nan_or_inf(self):
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        result = _crossfade_chunks(
            [_sine(200.0, 4000, amp=0.3), _sine(200.0, 4000, amp=0.9)], SR
        )
        self.assertTrue(np.all(np.isfinite(result)))


@pytest.mark.unit
class TestExistingBehaviourPreserved(unittest.TestCase):
    """PRF-2 must not alter the disabled/silence paths or single chunks."""

    def test_crossfade_disabled_is_exact_concat(self):
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        c1 = np.ones(100, dtype=np.float32)
        c2 = np.ones(200, dtype=np.float32) * 2.0
        result = _crossfade_chunks([c1, c2], SR, crossfade_ms=0)
        self.assertEqual(len(result), 300)
        np.testing.assert_array_equal(result[:100], c1)
        np.testing.assert_array_equal(result[100:], c2)

    def test_silence_gap_length_unchanged(self):
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        c1 = np.ones(SR, dtype=np.float32)
        c2 = np.ones(SR, dtype=np.float32)
        result = _crossfade_chunks([c1, c2], SR, crossfade_ms=0, silence_gap_s=0.5)
        self.assertEqual(len(result), 2 * SR + int(0.5 * SR))

    def test_single_chunk_passthrough(self):
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        chunk = np.ones(1000, dtype=np.float32)
        np.testing.assert_array_equal(_crossfade_chunks([chunk], SR), chunk)

    def test_empty_returns_empty(self):
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        self.assertEqual(len(_crossfade_chunks([], SR)), 0)

    def test_dc_chunks_still_smooth(self):
        """Regression for test_p3_p4_remediation: DC chunks have no crossing."""
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        c1 = np.ones(SR, dtype=np.float32) * 0.5
        c2 = np.ones(SR, dtype=np.float32) * -0.5
        result = _crossfade_chunks([c1, c2], SR, crossfade_ms=50)
        mid = len(c1)
        transition = result[mid - 100 : mid + 100]
        self.assertLess(float(np.max(np.abs(np.diff(transition)))), 0.1)

    def test_three_chunks_shorter_than_concat(self):
        from qwen3_tts.core.engine.inference import _crossfade_chunks

        chunks = [np.ones(500, dtype=np.float32) * i for i in range(1, 4)]
        result = _crossfade_chunks(chunks, SR, crossfade_ms=10)
        self.assertLess(len(result), 1500)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
