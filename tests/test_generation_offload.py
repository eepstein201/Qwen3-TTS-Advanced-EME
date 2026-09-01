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


if __name__ == "__main__":
    unittest.main()
