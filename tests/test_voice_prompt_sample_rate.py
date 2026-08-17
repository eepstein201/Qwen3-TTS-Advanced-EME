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


class TestLegacyLowRatePromptWarnsOnLoad(unittest.TestCase):
    """Prompts created before the resampling fix are still on disk and still
    broken. `tts voice rebuild` does not repair them — it regenerates the .pt
    and leaves the .wav alone, while MLX reads the .wav. Warning at the load
    choke-point is what makes that visible.
    """

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

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


if __name__ == "__main__":
    unittest.main()
