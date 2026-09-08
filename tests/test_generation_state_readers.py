"""Every READER of ``generation_state`` must route through the guard too.

Step 0C Task 4: Tasks 1-3 routed the three generation paths (batch, HTTP
streaming, /ws); this module pins the READ side. ``/generation-status`` and
``/queue-status`` are PUBLIC endpoints, ``/cancel-generation`` flips the
cancel flag, ``handle_unload_model`` gates its 409 on ``active``+``mode``,
and ``detect_degraded_generation`` feeds /health + /stats. Every one of them
read (or wrote) the plain dict with separate unlocked key reads, so a state
reset landing between two reads produced answers computed across two
different generations.

The centerpiece is the plan's named torn-read regression: a deterministic,
event-driven interleave that parks the ``/generation-status`` handler exactly
between two unlocked reads and resets the state underneath it. Zero sleeps —
the handoff is by ``threading.Event``, and the interleave point is the
handler's own ``time.time()`` call, so the ordering is guaranteed by
construction rather than by timing.

The routing pins reuse the recording proxy from
``tests.test_generation_state_routing`` (same module batch, same proxy
semantics: record every guard call, delegate to a real guard so the writes
actually land).

Run: python -m pytest tests/test_generation_state_readers.py -v
"""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from qwen3_tts.server.generation_state_guard import GenerationStateGuard, guard_for
from tests.test_generation_state_routing import (
    _generation_lock_tripwire,
    _RecordingGuard,
)

_APP = "qwen3_tts.server.app"


def _make_reader_state(**overrides):
    """An app.state stand-in holding the FULL ten-key generation_state.

    Mirrors the idle shape ``app_lifespan.lifespan`` provisions, so the
    guard's canonical-key reads (no ``.get`` defaults) resolve exactly as in
    production; ``overrides`` seed a coherent active generation afterwards.
    Only the attributes the reader handlers actually touch are present.
    """
    state = SimpleNamespace()
    state.last_activity = 0
    state.server_config = {"auto_shutdown_minutes": 0}
    state.shutdown_timer = None
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
        "cancelled": False,
    }
    state.generation_state.update(overrides)
    return state


def _recording_for(state):
    """A recording proxy over a REAL guard bound to *state*."""
    return _RecordingGuard(GenerationStateGuard(state))


def _request(state):
    """Build a MagicMock request whose ``app.state`` points to *state*."""
    request = MagicMock()
    request.app.state = state
    return request


class TestGenerationStatusSnapshotIsAtomic(unittest.IsolatedAsyncioTestCase):
    """The plan's named torn-read regression (Step 0C ruling F1).

    ``/generation-status`` is a PUBLIC endpoint whose ``elapsed_sec``
    combines two different keys of ``generation_state``. Read unlocked, a
    reset landing between the ``active`` read and the ``start_time`` read
    reported a bogus elapsed time computed across two different generations;
    routed through the guard, the whole response is built from ONE atomic
    ``snapshot()``.
    """

    ACTIVE_START_TIME = 1000.0
    FAKE_NOW = 5000.0
    OWNER_ID = "torn-probe-owner"

    async def test_generation_status_elapsed_is_never_computed_across_a_reset(self):
        """A full state reset interleaved into /generation-status must not
        leak into the reported elapsed_sec.

        Construction: the state holds a coherent active generation
        (``start_time`` = 1000.0). The patched ``time.time`` — the name THIS
        handler resolves, app.py's module-scope ``time`` binding — parks the
        handler thread on the first invocation, i.e. exactly between the
        ``active`` read and the ``start_time`` read on the unlocked reader.
        A second thread then runs the full reset under the guard's lock and
        releases the handler. The only consistent answer is the elapsed time
        of the generation the reader actually saw; an elapsed computed from
        the RESET (idle) ``start_time`` of 0.0 is the torn read.
        """
        from qwen3_tts.server.app import generation_status

        state = _make_reader_state(
            active=True,
            start_time=self.ACTIVE_START_TIME,
            text_length=42,
            mode="clone",
            batch_index=0,
            batch_total=1,
            chunk_index=1,
            chunk_total=3,
            generation_id=self.OWNER_ID,
        )
        handler_at_interleave = threading.Event()
        reset_done = threading.Event()
        reset_results = []
        guard = guard_for(state)

        def fake_time():
            handler_at_interleave.set()
            reset_done.wait(timeout=10)
            return self.FAKE_NOW

        def reset_under_the_guard():
            if not handler_at_interleave.wait(timeout=10):
                reset_results.append("handler never reached time.time()")
                return
            reset_results.append(guard.reset_if_owner(self.OWNER_ID))
            reset_done.set()

        resetter = threading.Thread(
            target=reset_under_the_guard, name="state-resetter"
        )
        resetter.start()
        try:
            with patch(f"{_APP}.time", SimpleNamespace(time=fake_time)):
                result = await asyncio.wait_for(
                    generation_status(_request(state)), timeout=15
                )
        finally:
            resetter.join(timeout=10)

        # The interleave really happened: the reset ran, under the guard,
        # while the handler was parked between its reads.
        self.assertEqual(reset_results, [True], "the guarded reset never ran")
        # The reader saw the ACTIVE generation — so its elapsed_sec must be
        # measured from THAT generation's start_time, never from the idle 0.0
        # the reset left behind.
        self.assertTrue(result["active"])
        self.assertEqual(
            result.get("elapsed_sec"),
            round(self.FAKE_NOW - self.ACTIVE_START_TIME, 1),
            "/generation-status computed elapsed_sec across a state reset -- "
            f"the multi-key read is not atomic: {result!r}",
        )


class TestReadersRouteThroughGuard(unittest.IsolatedAsyncioTestCase):
    """Each reader must resolve the guard once and touch the dict only via
    it — the recording proxy captures every guard call the handler makes."""

    async def test_cancel_active_generation_routes_through_the_guard(self):
        """An active cancel flips the flag via ``set_cancelled`` and reads the
        id back AFTER the write (today's ordering) — and never enters the
        asyncio ``generation_lock`` (the tripwire fails the test if it does)."""
        from qwen3_tts.server.app import cancel_generation

        state = _make_reader_state(active=True, generation_id="gen-cancel-1")
        state.generation_lock = _generation_lock_tripwire("/cancel-generation")
        recording = _recording_for(state)

        with patch(f"{_APP}.guard_for", return_value=recording, create=True):
            result = await cancel_generation(_request(state), _auth=None)

        self.assertEqual(
            result,
            {"status": "cancellation_requested", "generation_id": "gen-cancel-1"},
        )
        self.assertEqual(
            recording.names(),
            ["snapshot", "set_cancelled", "snapshot"],
            "/cancel-generation touched generation_state outside the guard, "
            f"or in an unexpected order (recorded: {recording.names()})",
        )
        self.assertEqual(recording.first("snapshot")[1], ("active",))
        self.assertEqual(recording.all_of("snapshot")[1][1], ("generation_id",))
        self.assertTrue(state.generation_state["cancelled"])

    async def test_cancel_without_active_generation_reads_the_flag_via_the_guard(self):
        """The idle path answers ``no_active_generation`` from a guard
        snapshot and writes nothing."""
        from qwen3_tts.server.app import cancel_generation

        state = _make_reader_state()
        state.generation_lock = _generation_lock_tripwire("/cancel-generation")
        recording = _recording_for(state)

        with patch(f"{_APP}.guard_for", return_value=recording, create=True):
            result = await cancel_generation(_request(state), _auth=None)

        self.assertEqual(result, {"status": "no_active_generation"})
        self.assertEqual(recording.names(), ["snapshot"])
        self.assertEqual(recording.first("snapshot")[1], ("active",))
        self.assertFalse(state.generation_state["cancelled"])

    async def test_queue_status_reads_the_active_flag_through_the_guard(self):
        """The public queue probe reads ``active`` from a guard snapshot; the
        ``pending_lock`` block around the queue list is a different lock and
        stays untouched."""
        from qwen3_tts.server.app import queue_status

        state = _make_reader_state(active=True)
        state.pending_requests = ["r1", "r2"]
        state.pending_lock = asyncio.Lock()
        recording = _recording_for(state)

        with patch(f"{_APP}.guard_for", return_value=recording, create=True):
            result = await queue_status(_request(state))

        self.assertEqual(result, {"queue_length": 2, "active": True})
        self.assertEqual(recording.names(), ["snapshot"])
        self.assertEqual(recording.first("snapshot")[1], ("active",))

    def test_unload_active_check_reads_an_atomic_snapshot(self):
        """The unload gate reads ``active`` and ``mode`` in ONE snapshot (an
        active generation for the mode still 409s)."""
        from qwen3_tts.server.app_models import handle_unload_model

        state = _make_reader_state(active=True, mode="clone")
        recording = _recording_for(state)

        with patch(
            "qwen3_tts.server.app_models.guard_for",
            return_value=recording,
            create=True,
        ):
            with self.assertRaises(HTTPException) as caught:
                handle_unload_model(state, SimpleNamespace(model_type="clone"))

        self.assertEqual(caught.exception.status_code, 409)
        self.assertEqual(recording.names(), ["snapshot"])
        self.assertEqual(
            recording.first("snapshot")[1], ("active", "mode")
        )

    def test_unload_idle_path_reads_the_same_snapshot(self):
        """An idle state passes the gate through the same single snapshot."""
        from qwen3_tts.server.app_models import handle_unload_model

        state = _make_reader_state()
        state.models = {"clone": None}
        recording = _recording_for(state)

        with patch(
            "qwen3_tts.server.app_models.guard_for",
            return_value=recording,
            create=True,
        ):
            result = handle_unload_model(state, SimpleNamespace(model_type="clone"))

        self.assertEqual(
            result, {"status": "already_unloaded", "model": "clone"}
        )
        self.assertEqual(recording.names(), ["snapshot"])
        self.assertEqual(recording.first("snapshot")[1], ("active", "mode"))

    def test_degraded_detection_reads_an_atomic_snapshot(self):
        """Degraded detection snapshots exactly the keys it uses — which is
        what lets a partial test state (and the real idle shape) work."""
        from qwen3_tts.server.app_lifespan import detect_degraded_generation

        state = _make_reader_state(active=True, start_time=0.0, text_length=22)
        recording = _recording_for(state)

        with patch(
            "qwen3_tts.server.app_lifespan.guard_for",
            return_value=recording,
            create=True,
        ):
            result = detect_degraded_generation(state, now=7314.2)

        self.assertTrue(result["degraded"])
        self.assertEqual(recording.names(), ["snapshot"])
        self.assertEqual(
            recording.first("snapshot")[1],
            ("active", "start_time", "text_length"),
        )

    def test_degraded_detection_idle_state_returns_the_early_shape(self):
        """An idle state still takes exactly one snapshot and short-circuits
        to the healthy shape (boolean semantics preserved)."""
        from qwen3_tts.server.app_lifespan import detect_degraded_generation

        state = _make_reader_state()
        recording = _recording_for(state)

        with patch(
            "qwen3_tts.server.app_lifespan.guard_for",
            return_value=recording,
            create=True,
        ):
            result = detect_degraded_generation(state, now=1e9)

        self.assertFalse(result["degraded"])
        self.assertIsNone(result["elapsed_sec"])
        self.assertEqual(recording.names(), ["snapshot"])


if __name__ == "__main__":
    unittest.main()
