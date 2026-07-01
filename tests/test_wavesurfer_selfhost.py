"""WaveSurfer must be self-hosted (no external CDN).

Loading WaveSurfer from unpkg (a floating @7 tag, no SRI) was a supply-chain
risk: a CDN or package compromise executes arbitrary JS in the operator's
browser. The library is now vendored under interface/static and loaded by the
StreamingPlayer module via a Blob URL, so no external origin is contacted.

See docs/reviews/e2e-review-2026-07-01.md (Phase 5 security, H3).
"""

import os
import unittest

from qwen3_tts.interface import wavesurfer_js


class TestWaveSurferSelfHosted(unittest.TestCase):
    def test_vendored_asset_present_and_nontrivial(self):
        path = os.path.join(
            os.path.dirname(wavesurfer_js.__file__), "static", "wavesurfer.esm.js"
        )
        self.assertTrue(os.path.exists(path), "vendored wavesurfer.esm.js must exist")
        self.assertGreater(os.path.getsize(path), 5000, "vendored asset looks truncated")

    def test_no_cdn_in_player_js(self):
        js = wavesurfer_js.get_streaming_player_js()
        self.assertNotIn("unpkg.com", js, "player JS must not load WaveSurfer from a CDN")
        self.assertIn("WAVESURFER_SRC", js, "vendored source must be embedded")

    def test_no_cdn_in_loader_js(self):
        js = wavesurfer_js.get_wavesurfer_loader_js()
        self.assertNotIn("unpkg.com", js, "loader must not reference a CDN")

    def test_player_js_imports_from_blob_url(self):
        js = wavesurfer_js.get_streaming_player_js()
        self.assertIn("_wsModuleUrl()", js, "dynamic import must use the vendored blob URL")


if __name__ == "__main__":
    unittest.main()
