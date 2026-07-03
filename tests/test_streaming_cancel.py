"""Streaming generation must honor the /cancel-generation flag.

The streaming inference thread stopped only on `stop_event` (set when the client
disconnects). The HTTP /cancel-generation endpoint sets
`generation_state["cancelled"]`, which nothing consulted — so cancelling a
stream over HTTP had no effect (it worked only via client disconnect). The stop
decision now honors both, matching the batch path.

See docs/reviews/e2e-review-2026-07-01.md (Phase 5 FastAPI, HIGH; Item B).
"""

import threading
import unittest

from qwen3_tts.server.app_generation import _should_stop_streaming


class TestShouldStopStreaming(unittest.TestCase):
    def test_stops_when_client_disconnected(self):
        ev = threading.Event()
        ev.set()
        self.assertTrue(_should_stop_streaming(ev, {"cancelled": False}))

    def test_stops_when_cancel_flag_set(self):
        ev = threading.Event()  # not set (client still connected)
        self.assertTrue(_should_stop_streaming(ev, {"cancelled": True}))

    def test_continues_when_neither(self):
        ev = threading.Event()
        self.assertFalse(_should_stop_streaming(ev, {"cancelled": False}))

    def test_continues_when_flag_missing(self):
        ev = threading.Event()
        self.assertFalse(_should_stop_streaming(ev, {}))


if __name__ == "__main__":
    unittest.main()
