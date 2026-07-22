#!/usr/bin/env python3
"""Tests for qwen3_tts.interface.generate_helpers — part 2.

Split from test_generate_helpers.py (over 800 lines).
Covers: _decode_base64_result, _save_base64_result, get_generation_params.

Run: python -m pytest tests/test_generate_helpers_part2.py -v
"""

import base64
import io
import os
import tempfile
import unittest
from types import SimpleNamespace

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

    class _DummyMarkerFunc:
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition, **kwargs):
            return lambda f: f

    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()

    class _DummyPytest:
        mark = _DummyMark()

    pytest = _DummyPytest()


# =========================================================================
# _decode_base64_result / _save_base64_result
# =========================================================================


class TestDecodeBase64Result(unittest.TestCase):
    """Tests for _decode_base64_result — base64 audio decoding roundtrip."""

    def test_roundtrip_decode(self):
        """Decodes base64-encoded WAV to numpy array + sample rate."""
        try:
            import numpy as np
            import soundfile as sf
        except ImportError:
            self.skipTest("requires numpy and soundfile")

        from qwen3_tts.interface.generate_helpers import _decode_base64_result

        # Create a tiny valid WAV in memory
        audio_data = np.zeros(100, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, audio_data, 24000, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode()

        wav, sr = _decode_base64_result({"audio_base64": b64})
        self.assertEqual(sr, 24000)
        self.assertEqual(len(wav), 100)



class TestSaveBase64Result(unittest.TestCase):
    """Tests for _save_base64_result — write base64 audio to file."""

    def test_saves_to_file(self):
        """Decodes and writes raw bytes to output path."""
        from qwen3_tts.interface.generate_helpers import _save_base64_result
        raw_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt "
        b64 = base64.b64encode(raw_bytes).decode()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            _save_base64_result({"audio_base64": b64}, path)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), raw_bytes)
        finally:
            os.unlink(path)


# =========================================================================
# get_generation_params
# =========================================================================


class TestGetGenerationParams(unittest.TestCase):
    """Tests for get_generation_params — merge config, preset, and CLI args."""

    def test_defaults_from_config(self):
        """Returns config defaults when no preset or arg overrides."""
        from qwen3_tts.interface.generate_helpers import get_generation_params
        args = SimpleNamespace(
            preset=None, temperature=None, top_k=None, top_p=None,
            repetition_penalty=None, seed=None,
        )
        config = {"generation": {"temperature": 0.7, "top_k": 50}}
        params = get_generation_params(args, config)
        self.assertEqual(params["temperature"], 0.7)
        self.assertEqual(params["top_k"], 50)

    def test_preset_override(self):
        """Preset values override config defaults."""
        from qwen3_tts.interface.generate_helpers import get_generation_params
        args = SimpleNamespace(
            preset="consistent", temperature=None, top_k=None, top_p=None,
            repetition_penalty=None, seed=None,
        )
        config = {
            "generation": {"temperature": 0.7, "top_k": 50},
            "presets": {"consistent": {"temperature": 0.5, "seed": 42}},
        }
        params = get_generation_params(args, config)
        self.assertEqual(params["temperature"], 0.5)
        self.assertEqual(params["seed"], 42)

    def test_arg_override(self):
        """CLI args override both config and preset."""
        from qwen3_tts.interface.generate_helpers import get_generation_params
        args = SimpleNamespace(
            preset="consistent", temperature=0.9, top_k=None, top_p=None,
            repetition_penalty=None, seed=None,
        )
        config = {
            "generation": {"temperature": 0.7},
            "presets": {"consistent": {"temperature": 0.5}},
        }
        params = get_generation_params(args, config)
        self.assertEqual(params["temperature"], 0.9)


if __name__ == "__main__":
    unittest.main()
