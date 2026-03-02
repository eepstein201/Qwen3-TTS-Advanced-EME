#!/usr/bin/env python3
"""Deprecated command reference tests - verify old commands aren't referenced.

Run with:
    cd ~/Qwen3-TTS_UserFiles && python -m pytest tests/test_deprecated_refs.py -v

No GPU, models, or running server required.
"""

import inspect
import os
import sys
import unittest

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Check optional dependencies
try:
    import soundfile  # noqa: F401
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

_skip_generate = unittest.skipUnless(HAS_SOUNDFILE, "requires soundfile (voice_generate)")

try:
    import gradio  # noqa: F401
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

_skip_gradio = unittest.skipUnless(HAS_GRADIO, "requires gradio")

_DEPRECATED_COMMANDS = [
    "startTTSServer", "stopTTSServer", "changeVoice",
    "createVoice", "ttsUI", "configureTTS",
]


@pytest.mark.unit
@_skip_generate
class TestDeprecatedRefsGenerate(unittest.TestCase):
    """generate.py must not contain deprecated command names in user messages."""

    def test_no_deprecated_commands(self):
        from qwen3_tts.interface import generate
        source = inspect.getsource(generate)
        for cmd in _DEPRECATED_COMMANDS:
            self.assertNotIn(cmd, source, f"Found deprecated '{cmd}' in generate.py")


@pytest.mark.unit
class TestDeprecatedRefsEngine(unittest.TestCase):
    """engine.py must not contain deprecated command names."""

    def test_no_deprecated_commands(self):
        from qwen3_tts.core import engine
        source = inspect.getsource(engine)
        for cmd in _DEPRECATED_COMMANDS:
            self.assertNotIn(cmd, source, f"Found deprecated '{cmd}' in engine.py")


@pytest.mark.unit
class TestDeprecatedRefsCreateVoice(unittest.TestCase):
    """create_voice.py must not contain deprecated command names."""

    def test_no_deprecated_commands(self):
        from qwen3_tts.tools import create_voice
        source = inspect.getsource(create_voice)
        for cmd in _DEPRECATED_COMMANDS:
            self.assertNotIn(cmd, source, f"Found deprecated '{cmd}' in create_voice.py")


@pytest.mark.unit
@_skip_gradio
class TestDeprecatedRefsUI(unittest.TestCase):
    """ui.py must not contain deprecated command names."""

    def test_no_deprecated_commands(self):
        from qwen3_tts.interface import ui
        source = inspect.getsource(ui)
        for cmd in _DEPRECATED_COMMANDS:
            self.assertNotIn(cmd, source, f"Found deprecated '{cmd}' in ui.py")


if __name__ == "__main__":
    unittest.main()
