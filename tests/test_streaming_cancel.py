"""Streaming generation must honor the /cancel-generation flag.

The streaming inference thread stopped only on `stop_event` (set when the client
disconnects). The HTTP /cancel-generation endpoint sets
`generation_state["cancelled"]`, which nothing consulted — so cancelling a
stream over HTTP had no effect (it worked only via client disconnect). The stop
decision now honors both, matching the batch path.

Since Step 0C Task 3 the flag is read through ``GenerationStateGuard`` (a
``threading.Lock``) rather than a raw dict ``.get``: the stop-check runs on the
daemon inference thread, which an asyncio lock excludes nothing. The helper
takes the guard, not the dict.

See docs/reviews/e2e-review-2026-07-01.md (Phase 5 FastAPI, HIGH; Item B).
"""

import threading
import unittest
from types import SimpleNamespace

from qwen3_tts.server.app_generation import _should_stop_streaming
from qwen3_tts.server.generation_state_guard import GenerationStateGuard


def _guard_with(cancelled):
    """A real guard over a state whose dict carries the canonical key set.

    Production always has every canonical key (the lifespan idle shape), so
    the fixture mirrors that instead of a bare ``{}``.
    """
    state = SimpleNamespace()
    state.generation_state = {
        "active": False,
        "start_time": 0.0,
        "text_length": 0,
        "mode": "",
        "batch_index": 0,
        "batch_total": 0,
        "chunk_index": 0,
        "chunk_total": 0,
        "generation_id": None,
        "cancelled": cancelled,
    }
    return GenerationStateGuard(state)


class TestShouldStopStreaming(unittest.TestCase):
    def test_stops_when_client_disconnected(self):
        ev = threading.Event()
        ev.set()
        self.assertTrue(_should_stop_streaming(ev, _guard_with(False)))

    def test_stops_when_cancel_flag_set(self):
        ev = threading.Event()  # not set (client still connected)
        self.assertTrue(_should_stop_streaming(ev, _guard_with(True)))

    def test_continues_when_neither(self):
        ev = threading.Event()
        self.assertFalse(_should_stop_streaming(ev, _guard_with(False)))

    def test_reads_the_flag_through_the_guard(self):
        """The cancel read must go through ``is_cancelled`` — the guard's
        ``threading.Lock`` is what makes the daemon thread's read safe."""
        ev = threading.Event()
        guard = _guard_with(False)
        self.assertFalse(_should_stop_streaming(ev, guard))
        guard.set_cancelled()
        self.assertTrue(
            _should_stop_streaming(ev, guard),
            "a cancel written through the guard was not seen by the "
            "streaming stop-check",
        )

    def test_cancelled_key_is_a_required_canonical_key(self):
        """The guard reads the CANONICAL key (no ``.get`` default): a dict
        missing it fails fast instead of silently reading False. Production
        always has it; the old ``{}`` fixture was a test artifact."""
        state = SimpleNamespace()
        state.generation_state = {}
        with self.assertRaises(KeyError):
            _should_stop_streaming(threading.Event(), GenerationStateGuard(state))


if __name__ == "__main__":
    unittest.main()
