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
from unittest.mock import MagicMock, patch

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

    def test_a_non_string_name_is_not_an_error(self):
        """This helper promises never to raise. A None from an unset dropdown
        must not AttributeError out of a diagnostic and kill the generation it
        exists to annotate."""
        from qwen3_tts.interface.ui import shared

        self.assertIsNone(shared.low_rate_prompt_rate(None))
        self.assertIsNone(shared.low_rate_prompt_warning(None))

    def test_unreadable_wav_is_not_an_error(self):
        with open(os.path.join(self.tmp.name, "broken.wav"), "w") as f:
            f.write("not a wav")

        self.assertIsNone(self._warn("broken"))


class TestManageVoicesTableFlagsLowRatePrompts(_PromptDir):
    def _rows(self):
        from qwen3_tts.interface.ui import shared
        from qwen3_tts.interface.ui import voice_management as vm

        # The rate lookup lives in `shared`, so patch it at its definition
        # site as well — patching only vm.VOICE_PROMPTS_DIR would leave the
        # helper reading the real voice_prompts/ directory.
        with patch.object(vm, "VOICE_PROMPTS_DIR", self.tmp.name), patch.object(
            shared, "VOICE_PROMPTS_DIR", self.tmp.name
        ), patch.object(
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


class TestGenerationSurfacesTheWarning(unittest.TestCase):
    """The moment of pain is the generation itself — warn before the wait."""

    def _generate(self, payload, warning):
        from qwen3_tts.interface.ui import generation as gen

        stream_config = {"server_side": True, "payload": payload}

        # Patch gradio.Warning at the library, NOT `gen.gr`:
        # _generate_server_side does its own `import gradio as gr` inside the
        # function body, which shadows the module-level name, so patching the
        # module attribute silently has no effect and the real warning fires.
        with patch.object(
            gen.shared, "low_rate_prompt_warning", return_value=warning
        ) as probe, patch(
            "qwen3_tts.server.client.TTSClient", return_value=MagicMock()
        ), patch("gradio.Warning") as mock_warning, patch.object(
            gen, "add_to_history"
        ), patch.object(
            gen, "save_generation_metadata"
        ), patch.object(
            gen, "format_status_display", return_value=""
        ):
            # NOT wrapped in try/except: swallowing here would let a crash
            # before the warning logic masquerade as "correctly stayed silent",
            # which is precisely the hollow negative test this guards against.
            # _generate_server_side handles its own generation errors and
            # returns a status tuple, so it does not raise.
            gen._generate_server_side(
                payload.get("mode", "clone"),
                payload.get("text", ""),
                [],
                stream_config,
            )
        self.probe = probe
        return mock_warning

    def test_warns_when_the_clone_prompt_is_low_rate(self):
        mock_warning = self._generate(
            {"mode": "clone", "prompt_file": "legacy.wav", "text": "hi"},
            "legacy is 8000 Hz",
        )

        mock_warning.assert_called_once_with("legacy is 8000 Hz")

    def test_silent_when_the_prompt_is_fine(self):
        mock_warning = self._generate(
            {"mode": "clone", "prompt_file": "fine.wav", "text": "hi"}, None
        )

        # Assert the check actually RAN and chose to stay quiet. Without this,
        # the test would also pass if the code never reached the check at all.
        self.probe.assert_called_once_with("fine.wav")
        mock_warning.assert_not_called()

    def test_silent_for_non_clone_modes(self):
        """Custom and design use no reference audio at all, so the prompt is
        never inspected — here the check correctly does not run."""
        mock_warning = self._generate(
            {"mode": "custom", "speaker": "ryan", "text": "hi"}, "should not appear"
        )

        self.probe.assert_not_called()
        mock_warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
