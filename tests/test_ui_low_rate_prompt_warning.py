#!/usr/bin/env python3
"""A below-native-rate voice prompt must be visible to a *browser* user.

`load_voice_prompt_mlx` warns when a prompt's reference .wav is under the
model's native rate, but that warning goes to the `tts.engine` logger — which,
for the web UI, means it lands in .voice_server.log where the user never looks.
In the CLI the same warning is printed to the terminal and is genuinely useful;
in the browser it was invisible, so the user just saw a generation that ran on
and on with no explanation.

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_ui_low_rate_prompt_warning.py -q
"""
import os
import unittest
from unittest.mock import patch

import numpy as np

from qwen3_tts.core.engine.audio_processing import DEFAULT_SAMPLE_RATE


def _tone(seconds, sr, freq=220.0):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class _PromptDir(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, name, sr):
        import soundfile as sf

        sf.write(os.path.join(self.tmp.name, f"{name}.wav"), _tone(0.5, sr), sr)
        with open(os.path.join(self.tmp.name, f"{name}.txt"), "w") as f:
            f.write("hello there friend")


class TestLowRatePromptWarningHelper(_PromptDir):
    def _warn(self, name):
        from qwen3_tts.interface.ui import shared

        with patch.object(shared, "VOICE_PROMPTS_DIR", self.tmp.name):
            return shared.low_rate_prompt_warning(name)

    def test_returns_a_message_naming_both_rates(self):
        self._write("legacy", 8000)

        msg = self._warn("legacy")

        self.assertIsNotNone(msg)
        self.assertIn("8000", msg)
        self.assertIn(str(DEFAULT_SAMPLE_RATE), msg)

    def test_says_rebuild_will_not_fix_it(self):
        """The obvious next move is `tts voice rebuild`, which does NOT help:
        it regenerates the .pt and leaves the .wav that MLX actually reads."""
        self._write("legacy", 8000)

        msg = self._warn("legacy")

        self.assertIn("rebuild", msg.lower())

    def test_silent_for_adequate_rate(self):
        self._write("fine", DEFAULT_SAMPLE_RATE)

        self.assertIsNone(self._warn("fine"))

    def test_silent_for_higher_rate(self):
        self._write("high", 48000)

        self.assertIsNone(self._warn("high"))

    def test_accepts_a_name_with_the_wav_extension(self):
        """The UI passes prompt filenames, not bare base names."""
        self._write("legacy", 8000)

        self.assertIsNotNone(self._warn("legacy.wav"))

    def test_missing_prompt_is_not_an_error(self):
        """Never break a generation over a diagnostic."""
        self.assertIsNone(self._warn("nope"))

    def test_unreadable_wav_is_not_an_error(self):
        with open(os.path.join(self.tmp.name, "broken.wav"), "w") as f:
            f.write("not a wav")

        self.assertIsNone(self._warn("broken"))


class TestManageVoicesTableFlagsLowRatePrompts(_PromptDir):
    def _rows(self):
        from qwen3_tts.interface.ui import voice_management as vm

        with patch.object(vm, "VOICE_PROMPTS_DIR", self.tmp.name), patch.object(
            vm, "get_voice_prompts", return_value=["legacy.wav", "fine.wav"]
        ), patch.object(vm, "load_config", return_value={}):
            return vm.get_prompt_table_data()

    def test_low_rate_prompt_is_flagged_in_the_table(self):
        self._write("legacy", 8000)
        self._write("fine", DEFAULT_SAMPLE_RATE)

        rows = {r[0]: r[1] for r in self._rows()}

        self.assertIn("8000", rows["legacy"])

    def test_adequate_rate_prompt_is_not_flagged(self):
        self._write("legacy", 8000)
        self._write("fine", DEFAULT_SAMPLE_RATE)

        rows = {r[0]: r[1] for r in self._rows()}

        self.assertNotIn("8000", rows["fine"])
        self.assertNotIn("Hz", rows["fine"])

    def test_table_shape_is_unchanged(self):
        """Manage Voices declares headers=[Name, Format, Default]; adding a
        column here would desync the Dataframe."""
        self._write("legacy", 8000)
        self._write("fine", DEFAULT_SAMPLE_RATE)

        for row in self._rows():
            self.assertEqual(len(row), 3)


if __name__ == "__main__":
    unittest.main()
