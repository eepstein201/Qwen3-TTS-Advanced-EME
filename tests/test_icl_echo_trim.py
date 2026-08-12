#!/usr/bin/env python3
"""PRF-8: ASR-trim the ICL echo-tail on cloned output.

In-context-learning cloning sometimes re-speaks the tail of the reference
transcript before the requested text (upstream #341). The existing ASR is
reused to transcribe the head of the output, spot that echo, and clip it.

x_vector_only_mode carries no transcript into the prompt, so it cannot
produce this echo and must never be probed.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_icl_echo_trim.py -v

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
REFERENCE = "This is my reference recording for cloning, thanks for listening."


def _tone(seconds, sr=SR, amp=0.4, freq=160.0):
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(seconds, sr=SR):
    return np.zeros(int(seconds * sr), dtype=np.float32)


def _cfg(**generation):
    return {"generation": generation}


# ---------------------------------------------------------------------------
# Echo detection (pure text)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectEchoPrefix(unittest.TestCase):
    """Does the output's head repeat the tail of the reference transcript?"""

    def test_detects_exact_tail_echo(self):
        from qwen3_tts.core.engine.inference import _detect_echo_prefix

        head = "thanks for listening. Today we discuss the weather."
        self.assertIsNotNone(_detect_echo_prefix(head, REFERENCE))

    def test_detects_despite_punctuation_and_case(self):
        from qwen3_tts.core.engine.inference import _detect_echo_prefix

        head = "THANKS, FOR LISTENING!! today we discuss the weather"
        self.assertIsNotNone(_detect_echo_prefix(head, REFERENCE))

    def test_returns_matched_text(self):
        from qwen3_tts.core.engine.inference import _detect_echo_prefix

        head = "thanks for listening. Today we discuss the weather."
        self.assertIn("listening", _detect_echo_prefix(head, REFERENCE))

    def test_no_echo_returns_none(self):
        from qwen3_tts.core.engine.inference import _detect_echo_prefix

        head = "Today we discuss the weather in some detail."
        self.assertIsNone(_detect_echo_prefix(head, REFERENCE))

    def test_short_incidental_overlap_is_ignored(self):
        """One shared word is coincidence, not an echo."""
        from qwen3_tts.core.engine.inference import _detect_echo_prefix

        head = "listening is a skill worth practising every day."
        self.assertIsNone(_detect_echo_prefix(head, REFERENCE, min_words=3))

    def test_empty_inputs_return_none(self):
        from qwen3_tts.core.engine.inference import _detect_echo_prefix

        self.assertIsNone(_detect_echo_prefix("", REFERENCE))
        self.assertIsNone(_detect_echo_prefix("thanks for listening", ""))
        self.assertIsNone(_detect_echo_prefix(None, None))

    def test_prefers_longest_matching_tail(self):
        from qwen3_tts.core.engine.inference import _detect_echo_prefix

        head = "for cloning thanks for listening. Now the real text."
        matched = _detect_echo_prefix(head, REFERENCE)
        self.assertIsNotNone(matched)
        self.assertIn("cloning", matched)


# ---------------------------------------------------------------------------
# Silence boundary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindSilenceBoundary(unittest.TestCase):
    """Clip on the pause after the echo, not mid-word."""

    def test_finds_the_gap(self):
        from qwen3_tts.core.engine.inference import _find_silence_boundary

        audio = np.concatenate([_tone(1.0), _silence(0.3), _tone(1.0)])
        idx = _find_silence_boundary(audio, SR, int(1.1 * SR), window_s=0.6)

        # Gap spans 1.0s..1.3s.
        self.assertGreater(idx, int(0.95 * SR))
        self.assertLess(idx, int(1.35 * SR))

    def test_falls_back_to_estimate_without_a_gap(self):
        from qwen3_tts.core.engine.inference import _find_silence_boundary

        audio = _tone(2.0)
        target = int(1.0 * SR)
        idx = _find_silence_boundary(audio, SR, target, window_s=0.3)
        self.assertAlmostEqual(idx, target, delta=int(0.35 * SR))

    def test_clamped_within_audio(self):
        from qwen3_tts.core.engine.inference import _find_silence_boundary

        audio = _tone(0.5)
        idx = _find_silence_boundary(audio, SR, int(10 * SR), window_s=0.3)
        self.assertLessEqual(idx, len(audio))
        self.assertGreaterEqual(idx, 0)


# ---------------------------------------------------------------------------
# Orchestration + guards
# ---------------------------------------------------------------------------


def _audio_with_echo():
    """Echo (0.8 s), pause, then the real content."""
    return np.concatenate([_tone(0.8), _silence(0.25), _tone(1.5)])


@pytest.mark.unit
class TestTrimIclEchoGuards(unittest.TestCase):
    """Never probe when an echo is impossible or ASR isn't already loaded."""

    def _patched(self, transcript="thanks for listening. Now the real text.",
                 asr_loaded=True):
        return (
            patch(
                "qwen3_tts.core.engine.inference._transcribe_probe",
                return_value=transcript,
            ),
            patch("qwen3_tts.core.engine.asr.is_asr_loaded", return_value=asr_loaded),
        )

    def test_x_vector_only_mode_is_never_probed(self):
        """No transcript in the prompt => no echo possible (#341 sidestep)."""
        from qwen3_tts.core.engine.inference import _trim_icl_echo

        audio = _audio_with_echo()
        probe_p, asr_p = self._patched()
        with probe_p as probe, asr_p:
            out, sr = _trim_icl_echo(
                audio, SR, REFERENCE, "clone", True, config=_cfg()
            )

        probe.assert_not_called()
        np.testing.assert_array_equal(out, audio)
        self.assertEqual(sr, SR)

    def test_non_clone_modes_are_untouched(self):
        from qwen3_tts.core.engine.inference import _trim_icl_echo

        audio = _audio_with_echo()
        for mode in ("design", "custom"):
            with self.subTest(mode=mode):
                probe_p, asr_p = self._patched()
                with probe_p as probe, asr_p:
                    out, _ = _trim_icl_echo(
                        audio, SR, REFERENCE, mode, False, config=_cfg()
                    )
                probe.assert_not_called()
                np.testing.assert_array_equal(out, audio)

    def test_missing_reference_text_skips(self):
        from qwen3_tts.core.engine.inference import _trim_icl_echo

        audio = _audio_with_echo()
        probe_p, asr_p = self._patched()
        with probe_p as probe, asr_p:
            _trim_icl_echo(audio, SR, "", "clone", False, config=_cfg())
        probe.assert_not_called()

    def test_skips_when_asr_not_loaded(self):
        """Don't pull a heavy ASR model into a generation that didn't ask."""
        from qwen3_tts.core.engine.inference import _trim_icl_echo

        audio = _audio_with_echo()
        probe_p, asr_p = self._patched(asr_loaded=False)
        with probe_p as probe, asr_p:
            out, _ = _trim_icl_echo(audio, SR, REFERENCE, "clone", False, config=_cfg())
        probe.assert_not_called()
        np.testing.assert_array_equal(out, audio)

    def test_disabled_by_config(self):
        from qwen3_tts.core.engine.inference import _trim_icl_echo

        audio = _audio_with_echo()
        probe_p, asr_p = self._patched()
        with probe_p as probe, asr_p:
            _trim_icl_echo(
                audio, SR, REFERENCE, "clone", False,
                config=_cfg(trim_icl_echo=False),
            )
        probe.assert_not_called()


@pytest.mark.unit
class TestTrimIclEchoBehaviour(unittest.TestCase):
    """The acceptance criterion: a detected echo actually gets clipped."""

    def _run(self, audio, transcript):
        from qwen3_tts.core.engine.inference import _trim_icl_echo

        with patch(
            "qwen3_tts.core.engine.inference._transcribe_probe",
            return_value=transcript,
        ), patch("qwen3_tts.core.engine.asr.is_asr_loaded", return_value=True):
            return _trim_icl_echo(audio, SR, REFERENCE, "clone", False, config=_cfg())

    def test_echo_is_clipped(self):
        audio = _audio_with_echo()
        out, sr = self._run(audio, "thanks for listening. Now the real text.")

        self.assertLess(len(out), len(audio), "echo was not clipped")
        self.assertEqual(sr, SR)

    def test_clip_lands_near_the_pause(self):
        """~0.8 s of echo + a 0.25 s pause: the cut belongs in that gap."""
        audio = _audio_with_echo()
        out, _ = self._run(audio, "thanks for listening. Now the real text.")

        removed = (len(audio) - len(out)) / SR
        self.assertGreater(removed, 0.5)
        self.assertLess(removed, 1.4)

    def test_clean_output_is_untouched(self):
        audio = _audio_with_echo()
        out, _ = self._run(audio, "Now the real text with nothing echoed.")
        np.testing.assert_array_equal(out, audio)

    def test_transcription_failure_returns_original(self):
        """ASR problems must not cost us the generated audio."""
        from qwen3_tts.core.engine.inference import _trim_icl_echo

        audio = _audio_with_echo()
        with patch(
            "qwen3_tts.core.engine.inference._transcribe_probe",
            side_effect=RuntimeError("asr exploded"),
        ), patch("qwen3_tts.core.engine.asr.is_asr_loaded", return_value=True):
            out, sr = _trim_icl_echo(
                audio, SR, REFERENCE, "clone", False, config=_cfg()
            )

        np.testing.assert_array_equal(out, audio)
        self.assertEqual(sr, SR)

    def test_never_clips_everything(self):
        """A pathological match must not return empty audio."""
        audio = _tone(0.4)
        out, _ = self._run(audio, "thanks for listening")
        self.assertGreater(len(out), 0)

    def test_always_returns_pair(self):
        audio = _audio_with_echo()
        result = self._run(audio, "Now the real text.")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


@pytest.mark.unit
class TestReferenceTextFromPrompt(unittest.TestCase):
    """Best-effort transcript lookup on a model-built prompt object."""

    def test_reads_attribute(self):
        from qwen3_tts.core.engine.inference import _reference_text_from_prompt

        class Prompt:
            transcript = "hello there"

        self.assertEqual(_reference_text_from_prompt(Prompt()), "hello there")

    def test_reads_dict_key(self):
        from qwen3_tts.core.engine.inference import _reference_text_from_prompt

        self.assertEqual(
            _reference_text_from_prompt({"prompt_text": "hello there"}), "hello there"
        )

    def test_none_prompt(self):
        from qwen3_tts.core.engine.inference import _reference_text_from_prompt

        self.assertIsNone(_reference_text_from_prompt(None))

    def test_missing_transcript(self):
        from qwen3_tts.core.engine.inference import _reference_text_from_prompt

        self.assertIsNone(_reference_text_from_prompt(object()))

    def test_blank_transcript_is_none(self):
        from qwen3_tts.core.engine.inference import _reference_text_from_prompt

        self.assertIsNone(_reference_text_from_prompt({"transcript": "   "}))

    def test_non_string_is_ignored(self):
        """A tensor named `text` must not be mistaken for a transcript."""
        from qwen3_tts.core.engine.inference import _reference_text_from_prompt

        self.assertIsNone(_reference_text_from_prompt({"text": np.zeros(4)}))


@pytest.mark.unit
class TestWiredIntoRunInference(unittest.TestCase):
    """Guard against the trim existing but never being called."""

    def _source(self):
        import inspect

        from qwen3_tts.core.engine.inference import run_inference

        return inspect.getsource(run_inference)

    def _helper_source(self):
        """WS2 moved the per-chunk steps into the shared _postprocess_chunk."""
        import inspect

        from qwen3_tts.core.engine.inference import _postprocess_chunk

        return inspect.getsource(_postprocess_chunk)

    def test_run_inference_calls_trim(self):
        self.assertIn("_trim_icl_echo", self._helper_source())

    def test_both_paths_call_trim(self):
        """Single-chunk returns early, so each path needs its own call."""
        self.assertGreaterEqual(self._source().count("_postprocess_chunk"), 2)

    def test_streaming_path_also_trims(self):
        """WS2: the streaming paths run the same helper, so the echo trim is
        no longer batch-only. Both backend branches must call it."""
        import inspect

        from qwen3_tts.core.engine.inference import run_inference_streaming

        self.assertGreaterEqual(
            inspect.getsource(run_inference_streaming).count("_postprocess_chunk"), 2
        )

    def test_accepts_reference_text_argument(self):
        import inspect

        from qwen3_tts.core.engine.inference import run_inference

        params = inspect.signature(run_inference).parameters
        self.assertIn("reference_text", params)
        self.assertIsNone(params["reference_text"].default)

    def test_trim_runs_before_lufs(self):
        """Trim first so loudness measures only the audio that ships.

        The trim moved into _postprocess_chunk (WS2), so the ordering is now
        "helper before LUFS" in run_inference, plus "trim before speed" inside
        the helper.
        """
        self.assertLess(
            self._helper_source().index("_trim_icl_echo"),
            self._helper_source().index("_maybe_apply_speed"),
        )
        source = self._source()
        self.assertLess(
            source.index("_postprocess_chunk"), source.index("_maybe_apply_lufs")
        )


@pytest.mark.unit
class TestPostProcessingOrder(unittest.TestCase):
    """The three post-processing stages must run echo-trim → speed → LUFS.

    Each stage is independently correct and independently tested, so nothing
    else in the suite would notice a reorder — but the order carries meaning:

    * the trim deletes generated audio, so it has to happen before the signal
      is time-stretched (PRF-6), or the stretch is applied to samples that are
      about to be thrown away;
    * LUFS has to run last, because loudness must be measured on the audio
      that actually ships, not on a longer/faster intermediate.

    These assert the runtime call order rather than the order of names in the
    source, so both the single-chunk and multi-chunk paths are covered — a
    source-index check only ever sees the first occurrence and would miss a
    reorder in the second path.
    """

    SR = 24000

    def _record_order(self, chunk_count):
        """Run run_inference with every stage stubbed, return the call order."""
        import qwen3_tts.core.engine.inference as inf

        calls = []
        audio = np.zeros(self.SR, dtype=np.float32)

        def _stage(name):
            def _fn(a, sample_rate, *args, **kwargs):
                calls.append(name)
                return a, sample_rate

            return _fn

        class _ConfigProvider:
            def load(self):
                return {"generation": {}}

        with (
            patch.object(inf, "_prepare_text_chunks", return_value=["x"] * chunk_count),
            patch.object(inf, "_run_inference_single", return_value=(audio, self.SR)),
            patch.object(inf, "_crossfade_chunks", return_value=audio),
            patch.object(inf, "_trim_icl_echo", side_effect=_stage("trim")),
            patch.object(inf, "_maybe_apply_speed", side_effect=_stage("speed")),
            patch.object(inf, "_maybe_apply_lufs", side_effect=_stage("lufs")),
        ):
            inf.run_inference(
                object(),
                "some text",
                "clone",
                {},
                config_provider=_ConfigProvider(),
            )
        return calls

    def test_single_chunk_path_order(self):
        self.assertEqual(self._record_order(1), ["trim", "speed", "lufs"])

    def test_multi_chunk_path_order(self):
        self.assertEqual(self._record_order(3), ["trim", "speed", "lufs"])

    def test_every_stage_runs_exactly_once_per_generation(self):
        """A stage applied twice would double-stretch or double-normalize."""
        for chunk_count in (1, 3):
            with self.subTest(chunks=chunk_count):
                calls = self._record_order(chunk_count)
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(set(calls)), 3)


if __name__ == "__main__":
    unittest.main()
