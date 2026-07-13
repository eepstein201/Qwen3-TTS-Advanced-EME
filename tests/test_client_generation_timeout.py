"""Tests for the client-side generation read timeout scaling.

Long texts are chunked server-side and generated sequentially (~40-70s per
chunk on MLX), so total generation time grows linearly with text length.
A fixed 600s read timeout made the client abandon generations the server
went on to complete (observed: 9108 chars / 12 chunks = 657s server-side).

The client must scale its read timeout with text length: 600s floor,
plus a generous per-character budget for longer texts.

Run with:
    python -m pytest tests/test_client_generation_timeout.py -v --tb=short
"""

import base64
import io
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

try:
    import numpy as np
    import soundfile as sf
    HAS_AUDIO_DEPS = True
except ImportError:
    HAS_AUDIO_DEPS = False


TIMEOUT_FLOOR_SEC = 600
TIMEOUT_PER_CHAR_SEC = 0.25


def _make_config():
    """Create a temp config file and return its path."""
    data = {
        "server": {"host": "127.0.0.1", "port": 5123},
        "presets": {},
        "aliases": {},
        "generation": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        "output_directory": "~/Downloads",
        "default_clone_prompt": "default.pt",
        "default_voice_description": "neutral voice",
        "default_speaker": "ryan",
        "language": "English",
        "prosody_presets": {},
    }
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return f.name


def _wav_base64(duration_samples=240):
    """Return base64-encoded WAV bytes for a short silent clip."""
    buf = io.BytesIO()
    sf.write(buf, np.zeros(duration_samples, dtype="float32"), 24000, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mock_generate_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "results": [
            {
                "audio_base64": _wav_base64(),
                "sample_rate": 24000,
                "chunks": 1,
                "seed": 42,
            }
        ]
    }
    return resp


@unittest.skipUnless(HAS_AUDIO_DEPS, "requires numpy, soundfile")
class TestGenerationTimeoutScaling(unittest.TestCase):
    """_generate_via_server scales its read timeout with text length."""

    def setUp(self):
        self.cfg = _make_config()
        from qwen3_tts.server.client import TTSClient

        self.client = TTSClient(config_path=self.cfg)
        self.session = MagicMock()
        self.session.post.return_value = _mock_generate_response()
        self.client._session = self.session

    def tearDown(self):
        os.unlink(self.cfg)

    def _posted_timeout(self):
        _, kwargs = self.session.post.call_args
        return kwargs["timeout"]

    def _generate(self, text):
        return self.client._generate_via_server(
            text,
            mode="custom",
            prompt=None,
            description=None,
            speaker="ryan",
            instruct=None,
            gen_params={},
        )

    def test_short_text_uses_floor_timeout(self):
        """Short texts keep the 600s floor."""
        self._generate("Hello world")
        self.assertEqual(self._posted_timeout(), TIMEOUT_FLOOR_SEC)

    def test_long_text_scales_timeout(self):
        """A 10000-char text gets a proportionally larger timeout."""
        self._generate("x" * 10000)
        self.assertEqual(
            self._posted_timeout(), int(10000 * TIMEOUT_PER_CHAR_SEC)
        )

    def test_boundary_text_never_below_floor(self):
        """Texts around the floor boundary never get less than 600s."""
        self._generate("x" * 2000)  # 2000 * 0.25 = 500 < 600 floor
        self.assertEqual(self._posted_timeout(), TIMEOUT_FLOOR_SEC)


class TestGenerationTimeoutHelper(unittest.TestCase):
    """The _generation_timeout helper itself."""

    def test_helper_floor_and_scaling(self):
        from qwen3_tts.server.client.generator import _generation_timeout

        self.assertEqual(_generation_timeout(0), TIMEOUT_FLOOR_SEC)
        self.assertEqual(_generation_timeout(100), TIMEOUT_FLOOR_SEC)
        self.assertEqual(_generation_timeout(2400), TIMEOUT_FLOOR_SEC)
        self.assertEqual(_generation_timeout(10000), 2500)
        self.assertEqual(_generation_timeout(20000), 5000)


if __name__ == "__main__":
    unittest.main()
