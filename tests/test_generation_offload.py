"""Blocking generation-path I/O must be offloaded off the event loop.

handle_generate / handle_generate_stream ran several blocking calls directly in
their async bodies (voice-prompt file load, WAV encode, cache write, waveform
peaks) — some while holding the inference lock — stalling the event loop and any
in-flight streaming/health probes. They must be dispatched via asyncio.to_thread.

Driving the full generation pipeline in a unit test is impractical (needs a real
model + engine), so this asserts the offload wiring at the source level, matching
the AST/source-inspection convention already used for this module
(tests/test_streaming_and_peaks.py).

See docs/reviews/e2e-review-2026-07-01.md (Phase 5 FastAPI, HIGH).
"""

import inspect
import re
import unittest

from qwen3_tts.server import app_generation


def _offloaded(src: str, callable_name: str) -> bool:
    """True if callable_name is dispatched via asyncio.to_thread (whitespace-robust)."""
    return re.search(
        r"asyncio\.to_thread\(\s*" + re.escape(callable_name), src
    ) is not None


class TestGenerationOffload(unittest.TestCase):
    def setUp(self):
        self.src = inspect.getsource(app_generation)

    def test_voice_prompt_load_is_offloaded(self):
        """The offload now lives one layer down, in server/prompt_loading.py
        (#214 item 1) -- load_voice_prompt_serialized wraps load_voice_prompt
        in asyncio.to_thread internally so it can probe unlocked with
        allow_create=False first, then re-enter under inference_lock only
        when a torch auto-create is actually needed. app_generation.py now
        calls that wrapper directly instead of
        asyncio.to_thread(load_voice_prompt, ...) at its call sites -- the
        offload property is preserved, just relocated.
        """
        self.assertTrue(
            re.search(r"await\s+load_voice_prompt_serialized\(", self.src),
            "app_generation.py must dispatch voice-prompt loading through "
            "load_voice_prompt_serialized (server/prompt_loading.py)",
        )
        from qwen3_tts.server import prompt_loading

        prompt_src = inspect.getsource(prompt_loading)
        self.assertTrue(
            _offloaded(prompt_src, "load_voice_prompt"),
            "load_voice_prompt_serialized must dispatch load_voice_prompt "
            "via asyncio.to_thread",
        )
        # No direct (blocking) call remains.
        self.assertNotIn("= load_voice_prompt(prompt_file)", self.src)

    def test_wav_encode_is_offloaded(self):
        self.assertTrue(_offloaded(self.src, "sf.write"))

    def test_waveform_peaks_is_offloaded(self):
        self.assertTrue(_offloaded(self.src, "calculate_waveform_peaks"))

    def test_result_b64_encode_is_offloaded(self):
        """The b64 ENCODE of generated audio (multi-MB payload) must leave the
        event loop — sf.write being offloaded is not enough while the encode
        of the same buffer runs inline afterwards."""
        self.assertTrue(
            _offloaded(self.src, "_b64_encode"),
            "audio base64 encoding must be dispatched via asyncio.to_thread",
        )
        self.assertNotIn(
            "b64_audio = base64.b64encode(", self.src,
            "inline base64 encode remains in the generation path",
        )

    def test_cache_tempfile_creation_is_offloaded(self):
        """The generation-cache tempfile creation (open + chmod) is blocking
        file IO — same treatment as the writes that follow it."""
        self.assertTrue(
            _offloaded(self.src, "_stage_cache_tempfile"),
            "cache tempfile creation must be dispatched via asyncio.to_thread",
        )
        self.assertNotIn(
            "cache_file = tempfile.NamedTemporaryFile(", self.src,
            "inline tempfile creation remains in the generation path",
        )

    def test_wav_response_decode_is_offloaded(self):
        """The audio/wav content-negotiation path decodes the result's b64
        payload back to bytes — blocking CPU on the same scale as the encode,
        on the loop."""
        self.assertTrue(
            _offloaded(self.src, "base64.b64decode"),
            "the audio/wav response path must decode base64 via "
            "asyncio.to_thread",
        )
        self.assertNotIn(
            "audio_bytes = base64.b64decode(", self.src,
            "inline base64 decode remains in the audio/wav response path",
        )


if __name__ == "__main__":
    unittest.main()
