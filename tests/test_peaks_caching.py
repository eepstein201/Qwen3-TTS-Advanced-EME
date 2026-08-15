#!/usr/bin/env python3
"""MED-1: waveform peaks are cached on the generation-cache entry.

Peaks were only computed on the cache-miss path, so a generation-cache hit
returned a result with no ``peaks`` field at all — forcing the client to
recompute (or skip) waveform rendering for audio the server had already
analyzed. The peaks must be computed once per generated audio asset, stored
on the ``gen_cache`` entry, and echoed verbatim on every cache hit.

Run: python -m pytest tests/test_peaks_caching.py -v
"""
import os
import unittest
from unittest.mock import MagicMock, patch

try:
    import numpy as np
    from fastapi.testclient import TestClient
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi, numpy")

_APP_GENERATION = "qwen3_tts.server.app_generation"

_PAYLOAD = {
    "text": "Hello peaks cache",
    "mode": "clone",
    "prompt_file": "voice.wav",
}


def _fake_peaks_factory(calls):
    """Counting stand-in for calculate_waveform_peaks."""

    def _fake_peaks(wav, num_peaks=500):
        calls.append(num_peaks)
        return [0.1] * num_peaks

    return _fake_peaks


@_skip
class TestPeaksCachedOnGenerationCacheHit(unittest.TestCase):
    """MED-1: /generate cache hits echo stored peaks without recomputing."""

    def setUp(self):
        from qwen3_tts.server.app import app
        from tests.conftest import _init_app_state, _restore_app_state, _save_app_state

        self._restore_app_state = _restore_app_state
        self._original_state = _save_app_state(app)
        _init_app_state(app, auth_token="test_token")
        app.state.models["clone"] = MagicMock()
        app.state.server_config = {
            "auto_shutdown_minutes": 0,
            "security": {"max_text_length": 50000, "max_batch_size": 20},
        }
        self.app = app
        self.client = TestClient(app, raise_server_exceptions=False)
        self.headers = {"Authorization": "Bearer test_token"}

    def tearDown(self):
        # Remove cache temp files the handler created, then restore app.state.
        for entry in list(getattr(self.app.state, "gen_cache", {}).values()):
            main_file = entry.get("main_file")
            if main_file and os.path.exists(main_file):
                try:
                    os.remove(main_file)
                except OSError:
                    pass
        self._restore_app_state(self.app, self._original_state)

    def test_peaks_computed_once_and_echoed_on_cache_hit(self):
        """Second identical /generate must reuse the stored peaks.

        The generation cache makes the second request skip inference; the
        peaks computed for the first response must be stored on the cache
        entry and echoed on the hit, not recomputed or dropped.
        """
        calls = []
        wav = np.zeros(4800, dtype=np.float32)

        with patch(
            f"{_APP_GENERATION}._check_memory_available",
            return_value=(True, 4000),
        ), patch(
            "qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock()
        ), patch(
            "qwen3_tts.core.engine.run_inference", return_value=(wav, 24000)
        ), patch(
            "qwen3_tts.core.engine.audio_processing.calculate_waveform_peaks",
            side_effect=_fake_peaks_factory(calls),
        ), patch(
            "soundfile.write"
        ):
            r1 = self.client.post("/generate", json=_PAYLOAD, headers=self.headers)
            r2 = self.client.post("/generate", json=_PAYLOAD, headers=self.headers)

        self.assertEqual(r1.status_code, 200, r1.text)
        self.assertEqual(r2.status_code, 200, r2.text)

        expected_peaks = [0.1] * 500
        result1 = r1.json()["results"][0]
        result2 = r2.json()["results"][0]

        # The hit response must carry the same peaks as the miss response.
        self.assertEqual(result1.get("peaks"), expected_peaks)
        self.assertEqual(result2.get("peaks"), expected_peaks)

        # Peaks are computed exactly once — never recomputed on the hit.
        self.assertEqual(len(calls), 1, f"peaks computed {len(calls)} times")

    def test_generation_cache_entry_stores_peaks(self):
        """The gen_cache entry written after a miss must carry the peaks.

        The cache entry is the persistence layer for hit responses — without
        peaks on the entry, hits cannot echo them.
        """
        calls = []
        wav = np.zeros(4800, dtype=np.float32)

        with patch(
            f"{_APP_GENERATION}._check_memory_available",
            return_value=(True, 4000),
        ), patch(
            "qwen3_tts.core.engine.load_voice_prompt", return_value=MagicMock()
        ), patch(
            "qwen3_tts.core.engine.run_inference", return_value=(wav, 24000)
        ), patch(
            "qwen3_tts.core.engine.audio_processing.calculate_waveform_peaks",
            side_effect=_fake_peaks_factory(calls),
        ), patch(
            "soundfile.write"
        ):
            resp = self.client.post("/generate", json=_PAYLOAD, headers=self.headers)

        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(len(self.app.state.gen_cache), 1)
        entry = next(iter(self.app.state.gen_cache.values()))
        self.assertEqual(entry.get("peaks"), [0.1] * 500)


if __name__ == "__main__":
    unittest.main()
