"""Reference audio below the model's native rate must be upsampled on save.

Measured 2026-08-16: an 8 kHz reference `.wav` makes MLX clone generation
fail to emit EOS — it runs to the token cap on every attempt (3/3, up to
47.8x the expected token count), producing minutes of looped audio for a
12-character input. Resampling the *same* audio with the *same* transcript
to 24 kHz restored normal termination (512/512 capped -> 222/178 uncapped).

Why this must be fixed at prompt-creation time rather than at load time:
the MLX clone path passes `ref_audio=<path>` straight to mlx-audio, which
opens the file itself. `load_audio_for_cloning()` (which does resample)
is never involved, so whatever rate is on disk is what the model sees.
"""

import os
import unittest
from unittest.mock import patch

import numpy as np

from qwen3_tts.core.engine.audio_processing import (
    DEFAULT_SAMPLE_RATE,
    ensure_min_sample_rate,
)


def _tone(seconds, sr, freq=220.0):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestEnsureMinSampleRate(unittest.TestCase):
    def test_upsamples_below_target(self):
        audio = _tone(1.0, 8000)

        out, sr, resampled = ensure_min_sample_rate(audio, 8000)

        self.assertTrue(resampled)
        self.assertEqual(sr, DEFAULT_SAMPLE_RATE)
        self.assertGreater(len(out), len(audio))

    def test_preserves_duration(self):
        audio = _tone(2.0, 8000)

        out, sr, _ = ensure_min_sample_rate(audio, 8000)

        self.assertAlmostEqual(len(out) / sr, len(audio) / 8000, places=2)

    def test_leaves_target_rate_untouched(self):
        audio = _tone(1.0, DEFAULT_SAMPLE_RATE)

        out, sr, resampled = ensure_min_sample_rate(audio, DEFAULT_SAMPLE_RATE)

        self.assertFalse(resampled)
        self.assertEqual(sr, DEFAULT_SAMPLE_RATE)
        self.assertIs(out, audio)

    def test_never_downsamples_higher_rates(self):
        """48 kHz references work fine; downsampling would only lose data."""
        audio = _tone(0.5, 48000)

        out, sr, resampled = ensure_min_sample_rate(audio, 48000)

        self.assertFalse(resampled)
        self.assertEqual(sr, 48000)
        self.assertIs(out, audio)

    def test_stereo_is_reduced_to_mono(self):
        mono = _tone(0.5, 8000)
        stereo = np.stack([mono, mono], axis=-1)

        out, sr, resampled = ensure_min_sample_rate(stereo, 8000)

        self.assertTrue(resampled)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(sr, DEFAULT_SAMPLE_RATE)

    def test_output_is_float32(self):
        audio = _tone(0.5, 8000).astype(np.float64)

        out, _, _ = ensure_min_sample_rate(audio, 8000)

        self.assertEqual(out.dtype, np.float32)

    def test_empty_audio_is_returned_unchanged(self):
        """Never raise on a degenerate input — the caller reports the error."""
        audio = np.zeros(0, dtype=np.float32)

        out, sr, resampled = ensure_min_sample_rate(audio, 8000)

        self.assertFalse(resampled)
        self.assertEqual(sr, 8000)
        self.assertEqual(len(out), 0)


class TestGradioCreateVoiceUpsamples(unittest.TestCase):
    """The Gradio Create Voice tab byte-copied the upload into
    voice_prompts/<name>.wav with no resampling, so the MLX runaway bug stayed
    fully reachable from the primary GUI path even after tools/create_voice.py
    was fixed. MLX reads that .wav by path (voice_prompt.py returns
    {"ref_audio": wav_path}), so the on-disk rate is what the model sees.
    """

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _source_wav(self, sr):
        import soundfile as sf

        path = os.path.join(self.tmp.name, f"src{sr}.wav")
        sf.write(path, _tone(2.0, sr), sr)
        return path

    def _create(self, src, name):
        """Invoke the real UI handler against a temp prompts dir."""
        import soundfile as sf

        from qwen3_tts.interface.ui import voice_management as vm

        with patch.object(vm, "VOICE_PROMPTS_DIR", self.tmp.name), patch.object(
            vm, "load_config", return_value={"advanced": {"backend": "mlx"}}
        ), patch.object(vm, "get_voice_prompts", return_value=[]), patch.object(
            vm, "get_default_clone_prompt", return_value=None
        ):
            vm.create_voice_prompt(src, "hello there friend", name)
        return sf.info(os.path.join(self.tmp.name, f"{name}.wav"))

    def test_low_rate_upload_is_upsampled_on_disk(self):
        info = self._create(self._source_wav(8000), "low")

        self.assertGreaterEqual(info.samplerate, DEFAULT_SAMPLE_RATE)

    def test_duration_is_preserved_when_upsampling(self):
        info = self._create(self._source_wav(8000), "low2")

        self.assertAlmostEqual(info.duration, 2.0, places=1)

    def test_adequate_rate_upload_is_copied_byte_for_byte(self):
        """Re-encoding a good upload would be a gratuitous quality risk."""
        info = self._create(self._source_wav(48000), "high")

        self.assertEqual(info.samplerate, 48000)

    def test_stereo_upload_lands_on_disk_as_mono(self):
        """Assert the ARTIFACT, not the helper's return value.

        ensure_min_sample_rate downmixed in memory but reported
        was_resampled=False, so this handler byte-copied the untouched stereo
        original. The unit test passed because it inspected the returned array;
        the file MLX actually opens was still two channels.
        """
        import soundfile as sf

        src = os.path.join(self.tmp.name, "stereo48k.wav")
        mono = _tone(1.0, 48000)
        sf.write(src, np.stack([mono, mono], axis=-1), 48000)

        info = self._create(src, "stereo_disk")

        self.assertEqual(info.channels, 1)
        self.assertEqual(info.samplerate, 48000)

    def test_unreadable_upload_is_refused_and_writes_nothing(self):
        """If the rate cannot be inspected, the guarantee cannot be delivered.

        Logging a warning and byte-copying anyway ships exactly the
        unverified prompt this code exists to prevent — the caller has no way
        to know, and a runaway generation looks like a hang.
        """
        import gradio as gr

        bad = os.path.join(self.tmp.name, "not_audio.wav")
        with open(bad, "w") as f:
            f.write("this is not a wav file")

        with self.assertRaises(gr.Error):
            self._create(bad, "unreadable")

        self.assertFalse(
            os.path.exists(os.path.join(self.tmp.name, "unreadable.wav")),
            "refused creation must not leave a prompt behind",
        )


class TestLegacyLowRatePromptWarnsOnLoad(unittest.TestCase):
    """Prompts created before the resampling fix are still on disk and still
    broken. `tts voice rebuild` does not repair them — it regenerates the .pt
    and leaves the .wav alone, while MLX reads the .wav. Warning at the load
    choke-point is what makes that visible.
    """

    def setUp(self):
        import tempfile

        from qwen3_tts.core.engine import voice_prompt as vp

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # load_voice_prompt_mlx caches by prompt NAME in a module-level LRU.
        # Without this, a prompt loaded here stays cached for the rest of the
        # session and any later test loading the same name gets a cache hit
        # instead of exercising the real path.
        self.addCleanup(vp.clear_voice_prompt_cache)

    def _prompt(self, name, sr):
        import soundfile as sf

        sf.write(os.path.join(self.tmp.name, f"{name}.wav"), _tone(1.0, sr), sr)
        with open(os.path.join(self.tmp.name, f"{name}.txt"), "w") as f:
            f.write("hello there friend")

    def _load(self, name):
        from qwen3_tts.core.engine import voice_prompt as vp

        vp.clear_voice_prompt_cache()
        with patch.object(vp, "VOICE_PROMPTS_DIR", self.tmp.name):
            return vp.load_voice_prompt_mlx(name)

    def test_warns_for_below_native_rate(self):
        self._prompt("legacy", 8000)

        with self.assertLogs("tts.engine", level="WARNING") as logs:
            self._load("legacy")

        self.assertTrue(
            any("8000" in ln and "24000" in ln for ln in logs.output),
            f"expected a rate warning naming both rates, got: {logs.output}",
        )

    def test_silent_for_adequate_rate(self):
        self._prompt("fine", DEFAULT_SAMPLE_RATE)

        with patch("qwen3_tts.core.engine.voice_prompt.logger") as mock_log:
            self._load("fine")

        rate_warnings = [c for c in mock_log.warning.call_args_list if "Hz" in str(c)]
        self.assertEqual(rate_warnings, [])

    def test_load_still_returns_the_prompt(self):
        """The warning is advisory — it must never break loading."""
        self._prompt("legacy2", 8000)

        result = self._load("legacy2")

        self.assertEqual(result["ref_text"], "hello there friend")
        self.assertTrue(result["ref_audio"].endswith("legacy2.wav"))


class TestEnsureMinSampleRateNeverSilentlyFails(unittest.TestCase):
    """F7: the ImportError branch logged a warning and returned
    was_resampled=False, and create_voice.py then copied the original
    low-rate file to disk — silently shipping the exact poisonous prompt
    the function exists to prevent. Found by both Santa reviewers.
    """

    def test_raises_when_it_cannot_resample_a_low_rate_clip(self):
        audio = _tone(1.0, 8000)

        # Simulate librosa being unavailable.
        with patch.dict("sys.modules", {"librosa": None}):
            with self.assertRaises(RuntimeError) as ctx:
                ensure_min_sample_rate(audio, 8000)

        self.assertIn("8000", str(ctx.exception))

    def test_does_not_raise_when_no_resample_is_needed(self):
        """A missing librosa is irrelevant when the rate is already fine."""
        audio = _tone(1.0, DEFAULT_SAMPLE_RATE)

        with patch.dict("sys.modules", {"librosa": None}):
            out, sr, resampled = ensure_min_sample_rate(audio, DEFAULT_SAMPLE_RATE)

        self.assertFalse(resampled)
        self.assertEqual(sr, DEFAULT_SAMPLE_RATE)

    def test_resample_failure_is_reported_not_swallowed(self):
        audio = _tone(1.0, 8000)

        with patch("librosa.resample", side_effect=ValueError("boom")):
            with self.assertRaises(RuntimeError):
                ensure_min_sample_rate(audio, 8000)


class TestEnsureMinSampleRateAlwaysReturnsMono(unittest.TestCase):
    """F8: mono reduction sat *after* the `sr >= target_sr` early return, so a
    48 kHz stereo reference passed straight through and MLX received a
    multi-channel file. Found by Santa reviewer C.
    """

    def test_stereo_at_adequate_rate_is_still_reduced_to_mono(self):
        mono = _tone(0.5, 48000)
        stereo = np.stack([mono, mono], axis=-1)

        out, sr, was_modified = ensure_min_sample_rate(stereo, 48000)

        self.assertEqual(out.ndim, 1)
        self.assertEqual(sr, 48000)

    def test_stereo_downmix_reports_the_audio_was_modified(self):
        """The third element gates whether callers write the ARRAY or byte-copy
        the ORIGINAL file. A stereo downmix that reports False makes every
        caller copy the untouched stereo original, so the mono guarantee never
        reaches disk — it only ever existed in memory.
        """
        mono = _tone(0.5, 48000)
        stereo = np.stack([mono, mono], axis=-1)

        _, _, was_modified = ensure_min_sample_rate(stereo, 48000)

        self.assertTrue(was_modified)

    def test_untouched_mono_reports_no_modification(self):
        """Byte-copying is the better outcome when nothing changed."""
        _, _, was_modified = ensure_min_sample_rate(_tone(0.5, 48000), 48000)

        self.assertFalse(was_modified)

    def test_stereo_at_low_rate_is_mono_and_upsampled(self):
        mono = _tone(0.5, 8000)
        stereo = np.stack([mono, mono], axis=-1)

        out, sr, resampled = ensure_min_sample_rate(stereo, 8000)

        self.assertEqual(out.ndim, 1)
        self.assertEqual(sr, DEFAULT_SAMPLE_RATE)
        self.assertTrue(resampled)


if __name__ == "__main__":
    unittest.main()
