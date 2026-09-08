"""Batch ``/generate`` must route every ``generation_state`` touch through the guard.

Step 0C Task 2: ``handle_generate``'s batch path mutated
``app.state.generation_state`` (a plain dict shared with worker threads) with
no lock at all — the pre-loop ``cancelled`` clear, the loop cancel check, the
per-item begin, the ``_chunk_progress`` callback, the ``chunk_total`` capture
and both resets. Task 1 landed ``GenerationStateGuard`` (a ``threading.Lock``
per app state); Task 2 routes the batch path through it, behavior-preserving.

These tests pin the ROUTING at the real call site: a recording proxy replaces
``app_generation.guard_for`` so every guard method the handler calls is
captured, while the calls still delegate to a real guard (so the writes land
in the dict and the handler behaves exactly as in production). Driving the
full pipeline is impractical without a model, so ``run_inference`` is faked —
but the fake invokes ``progress_callback`` from the executor thread, which is
EXACTLY the thread production fires it on. That is why the guard must be a
``threading.Lock``: the callback's write has to be serialized against the
event loop, and an ``asyncio.Lock`` would exclude nothing there.

Run: python -m pytest tests/test_generation_state_routing.py -v
"""

import asyncio
import contextlib
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

try:
    import numpy as np
    import soundfile  # noqa: F401

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

_APP_GENERATION = "qwen3_tts.server.app_generation"
_ENGINE = "qwen3_tts.core.engine"

# Distinct progress pairs the fake inference feeds the routed callback. The
# pairing total == 2 * index is the torn-read canary: a snapshot must never
# observe the two keys from different writes.
_PROGRESS_CALLS = ((1, 2), (2, 4), (3, 6))

_skip = unittest.skipUnless(_HAS_DEPS, "requires numpy + soundfile")


class _RecordingGuard:
    """Transparent proxy recording every guard call the handler makes.

    Delegation is explicit per method (not ``__getattr__``) so the recorded
    shape shows the exact arguments the production call site passes — a
    renamed or re-signed guard method raises AttributeError here instead of
    silently bypassing the record.
    """

    def __init__(self, inner):
        self._inner = inner
        # (method_name, args...) tuples; list.append is GIL-atomic and the
        # worker thread records alongside the loop thread.
        self.calls = []

    def begin(self, generation_id, **kwargs):
        self.calls.append(("begin", generation_id, tuple(sorted(kwargs.items()))))
        return self._inner.begin(generation_id, **kwargs)

    def update_progress(self, chunk_index, chunk_total):
        self.calls.append(("update_progress", chunk_index, chunk_total))
        return self._inner.update_progress(chunk_index, chunk_total)

    def clear_cancelled(self):
        self.calls.append(("clear_cancelled",))
        return self._inner.clear_cancelled()

    def set_cancelled(self):
        self.calls.append(("set_cancelled",))
        return self._inner.set_cancelled()

    def is_cancelled(self):
        self.calls.append(("is_cancelled",))
        return self._inner.is_cancelled()

    def reset_if_owner(self, generation_id):
        self.calls.append(("reset_if_owner", generation_id))
        return self._inner.reset_if_owner(generation_id)

    def snapshot(self, keys=None):
        self.calls.append(("snapshot", tuple(keys) if keys is not None else None))
        return self._inner.snapshot(keys)

    # Introspection helpers for the assertions below.

    def names(self):
        return [call[0] for call in self.calls]

    def first(self, name):
        return next(call for call in self.calls if call[0] == name)

    def all_of(self, name):
        return [call for call in self.calls if call[0] == name]


def _fail_generation_lock(*_args, **_kwargs):
    raise AssertionError(
        "handle_generate used state.generation_lock on the batch path -- "
        "Task 2 replaced it with the threading.Lock guard, which is the only "
        "primitive that excludes the worker-thread progress callback"
    )


def _make_state():
    """Minimal app.state for exercising ``handle_generate``.

    Mirrors ``tests.test_batch_generation_state_ownership._make_state``: a
    SimpleNamespace so the test is self-contained (no real FastAPI app). The
    guard is deliberately NOT pre-provisioned — the handler must resolve it
    lazily via ``guard_for`` exactly as production would on a bare state.
    ``generation_lock`` is wired to FAIL if touched: the routed batch path
    must no longer need it.
    """
    state = SimpleNamespace()
    state.auth_token = "test_token"  # nosec B105
    state.models = {
        "clone": MagicMock(),
        "design": MagicMock(),
        "custom": MagicMock(),
    }
    state.model_load_times = {}
    state.model_load_errors = {"clone": None, "design": None, "custom": None}
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
    state.request_queue = set()
    state.request_queue_lock = threading.Lock()
    state.generation_lock = AsyncMock(side_effect=_fail_generation_lock)
    state.pending_requests = []
    state.last_activity = 0
    state.models_loaded = threading.Event()
    state.models_loaded.set()
    state.gen_cache = {}
    state.gen_cache_lock = threading.Lock()
    state.inference_lock = asyncio.Lock()
    state.eta_cache = {"median_rate": None, "last_updated": 0}
    state.eta_cache_lock = threading.Lock()
    state.shutdown_timer = None
    state.server_config = {
        "security": {"max_text_length": 50000, "max_batch_size": 20},
        "auto_shutdown_minutes": 0,
        "vllm": {"enabled": False, "fallback_to_torch": True},
    }
    return state


def _make_request(state):
    """Build a MagicMock request whose ``app.state`` points to *state*."""
    request = MagicMock()
    request.app.state = state
    request.headers = {"accept": "application/json"}
    return request


async def _drive_batch_async(
    state,
    recording,
    progress_calls=_PROGRESS_CALLS,
    on_inference_done=None,
    guard_for_mock=None,
):
    """Coroutine body of :func:`_drive_batch` (callable inside a live loop).

    ``run_inference`` is faked: the fake pulls ``progress_callback`` from the
    kwargs and invokes it from THIS executor thread — the same thread
    production's engine fires it on — so every assertion is made about a
    genuinely off-loop routed write. Returns the handler result.

    ``on_inference_done`` runs on the executor thread after the progress
    calls (the concurrency test and the visibility probe use it).
    ``guard_for_mock`` defaults to a mock returning *recording*.
    """
    fake_wav = np.zeros(500, dtype=np.float32)

    def fake_inference(model, text, **kwargs):
        callback = kwargs.get("progress_callback")
        assert callback is not None, (
            "handle_generate passed no progress_callback to run_inference; "
            "the batch chunk progress is unpinnable"
        )
        for chunk_index, chunk_total in progress_calls:
            callback(chunk_index, chunk_total)
        if on_inference_done is not None:
            on_inference_done()
        return fake_wav, 24000

    from qwen3_tts.server.app_generation import handle_generate
    from qwen3_tts.server.validation import GenerateRequest

    req = GenerateRequest(text="routing probe", mode="design",
                          voice_description="friendly")
    # The handler must receive THE recording guard itself. A caller-supplied
    # mock is installed via new= (it already returns the recording); the
    # default patches with return_value= directly — NOT by nesting a mock in
    # return_value, which would hand the handler a bare Mock and silently
    # bypass the record.
    if guard_for_mock is not None:
        guard_patch = patch(f"{_APP_GENERATION}.guard_for", new=guard_for_mock)
    else:
        guard_patch = patch(f"{_APP_GENERATION}.guard_for", return_value=recording)
    with patch(
        f"{_APP_GENERATION}._check_memory_available",
        return_value=(True, 4096),
    ), patch(
        "qwen3_tts.server.validation._validate_generation_request"
    ), guard_patch, patch(
        f"{_ENGINE}.run_inference", side_effect=fake_inference
    ):
        try:
            return await handle_generate(
                request=_make_request(state),
                state=state,
                req=req,
                security={"max_text_length": 50000, "max_batch_size": 20},
                config_provider=None,
            )
        finally:
            # Hygiene: the batch path writes a real cache WAV per item
            # (NamedTemporaryFile(delete=False)) into the test-local
            # ``state.gen_cache`` and never unlinks it, so every drive
            # leaks a file into the tempdir. Unlink what this drive left
            # behind (the handler's own eviction path is what production
            # relies on; a test has no reason to keep them).
            for entry in list(state.gen_cache.values()):
                cache_path = (
                    entry.get("main_file") if isinstance(entry, dict) else None
                )
                if cache_path and os.path.exists(cache_path):
                    with contextlib.suppress(OSError):
                        os.unlink(cache_path)


def _drive_batch(state, recording, **kwargs):
    """Run a single design-mode batch through ``handle_generate``."""
    return asyncio.run(_drive_batch_async(state, recording, **kwargs))


@_skip
class TestBatchProgressRoutesThroughGuard(unittest.TestCase):
    """The batch progress callback's write must be serialized by the guard."""

    def setUp(self):
        self.state = _make_state()
        from qwen3_tts.server.generation_state_guard import GenerationStateGuard

        self.recording = _RecordingGuard(GenerationStateGuard(self.state))

    def test_progress_callback_routes_through_update_progress(self):
        """A worker-thread call of the routed callback reaches the dict as an
        atomic guard ``update_progress`` pair — not a raw unlocked write."""
        observed = []

        def probe_from_worker_thread():
            # Runs on the executor thread, right after the last routed
            # progress write: the pair must already be visible and intact.
            observed.append(
                self.recording.snapshot(["chunk_index", "chunk_total"])
            )

        result = _drive_batch(
            self.state,
            self.recording,
            on_inference_done=probe_from_worker_thread,
        )

        self.assertEqual(
            self.recording.all_of("update_progress"),
            [("update_progress", ci, ct) for ci, ct in _PROGRESS_CALLS],
            "the batch progress callback did not route its writes through "
            "the guard's update_progress",
        )
        self.assertEqual(
            observed,
            [{"chunk_index": _PROGRESS_CALLS[-1][0],
              "chunk_total": _PROGRESS_CALLS[-1][1]}],
            "a snapshot taken from the worker thread right after the routed "
            "callback did not observe the written pair",
        )
        # Behavior preserved: the routed writes feed the response field, and
        # the finally reset still returns the dict to idle afterwards.
        self.assertEqual(result["results"][0]["chunks"], _PROGRESS_CALLS[-1][1])
        self.assertEqual(self.state.generation_state["chunk_total"], 0)

    def test_concurrent_snapshot_never_sees_a_torn_pair(self):
        """While a worker thread hammers the routed callback, a loop-side
        snapshot through the SAME guard never observes chunk_index from one
        write and chunk_total from another.

        Scope note, stated in-file on purpose: this is a serialization-
        PROPERTY pin, NOT a RED driver. Under CPython a raw two-key
        ``dict.update`` is GIL-atomic, so this test cannot detect reverting
        the callback to an unlocked single update — its detector power is
        against SPLITTING the (chunk_index, chunk_total) pair across two
        lock acquisitions, not against an unlocked two-key update. The RED
        drivers for the routing are the call-record pins (see
        ``test_progress_callback_routes_through_update_progress`` and the
        lifecycle sequence test).

        Deterministic: the reader runs as a task on the event loop and yields
        with ``asyncio.sleep(0)`` — a scheduling yield with zero wall-clock
        cost, not a timed sleep — so reader/writer interleave is guaranteed
        by the scheduler rather than by timing.
        """
        torn = []
        stop = threading.Event()
        iterations = []

        async def reader_until_stop():
            count = 0
            while not stop.is_set():
                snap = self.recording.snapshot(["chunk_index", "chunk_total"])
                if snap["chunk_total"] != 2 * snap["chunk_index"]:
                    torn.append(snap)
                count += 1
                # Scheduling yield (zero wall-clock): let the executor
                # thread's routed writes interleave.
                await asyncio.sleep(0)
            iterations.append(count)

        async def run():
            reader = asyncio.ensure_future(reader_until_stop())
            try:
                await _drive_batch_async(self.state, self.recording)
            finally:
                stop.set()
                await reader

        asyncio.run(run())

        self.assertEqual(
            torn,
            [],
            f"snapshot observed torn progress pairs: {torn[:5]!r}",
        )
        # The reader must actually have run: a zero-iteration reader proves
        # nothing about serialization.
        self.assertGreater(iterations[0], 0, "the reader task never ran")


@_skip
class TestBatchLifecycleRoutesThroughGuard(unittest.TestCase):
    """begin / cancel-check / chunk-count capture / resets all route through."""

    def setUp(self):
        self.state = _make_state()
        from qwen3_tts.server.generation_state_guard import GenerationStateGuard

        self.recording = _RecordingGuard(GenerationStateGuard(self.state))

    def test_full_batch_call_sequence_goes_through_the_guard(self):
        """One single-item batch must touch the dict ONLY via the guard, in
        the production order: pre-loop clear -> loop cancel check -> begin ->
        progress -> chunk_total snapshot -> in-lock reset -> finally reset."""
        result = _drive_batch(self.state, self.recording)

        names = self.recording.names()
        self.assertEqual(
            names,
            [
                "clear_cancelled",
                "is_cancelled",
                "begin",
                # one update_progress per fed progress pair
                *["update_progress"] * len(_PROGRESS_CALLS),
                "snapshot",
                "reset_if_owner",
                "reset_if_owner",
            ],
            "the batch path touched generation_state outside the guard, or "
            "in an unexpected order",
        )

        # begin() carries the per-item fields and a generation id...
        begin = self.recording.first("begin")
        begin_gen_id, begin_kwargs = begin[1], dict(begin[2])
        self.assertEqual(begin_kwargs["mode"], "design")
        self.assertEqual(begin_kwargs["text_length"], len("routing probe"))
        self.assertEqual(begin_kwargs["batch_index"], 0)
        self.assertEqual(begin_kwargs["batch_total"], 1)
        self.assertIsInstance(begin_gen_id, str)

        # ...the chunk-total capture asks for exactly the canonical key...
        self.assertEqual(self.recording.first("snapshot")[1], ("chunk_total",))

        # ...and BOTH resets are ownership-bound to begin's id (the in-lock
        # final-item reset and the finally safety net).
        resets = [call[1] for call in self.recording.all_of("reset_if_owner")]
        self.assertEqual(
            resets,
            [begin_gen_id, begin_gen_id],
            "the batch resets are not bound to the generation id begin() "
            "stamped",
        )

        # Behavior preserved: the routed snapshot feeds the response field.
        self.assertEqual(result["results"][0]["chunks"], _PROGRESS_CALLS[-1][1])
        self.assertFalse(result["cancelled"])

    def test_chunk_count_capture_reads_the_routed_snapshot(self):
        """The chunk count stored on the result comes from the guard snapshot
        taken under inference_lock, not a raw dict read."""
        calls = [(4, 9), (7, 11)]  # last pair wins, no 2x relationship needed
        result = _drive_batch(self.state, self.recording, progress_calls=calls)
        self.assertEqual(result["results"][0]["chunks"], 11)
        self.assertEqual(
            self.recording.first("snapshot"),
            ("snapshot", ("chunk_total",)),
        )

    def test_guard_is_resolved_once_per_handler_invocation(self):
        """``guard_for`` must be called once per ``handle_generate`` call —
        not per chunk or per loop iteration (it takes a module-level
        construction lock per call: fine at handler frequency, wrong at
        chunk frequency)."""
        guard_for_mock = MagicMock(return_value=self.recording)
        _drive_batch(self.state, self.recording, guard_for_mock=guard_for_mock)
        self.assertEqual(
            guard_for_mock.call_count,
            1,
            "handle_generate resolved the guard more than once for a "
            "single-item batch",
        )

    def test_cancel_check_reads_through_the_guard(self):
        """The per-item cancel check must read via ``is_cancelled`` — a stale
        unlocked read is exactly what the guard exists to prevent."""
        _drive_batch(self.state, self.recording)
        self.assertEqual(self.recording.names().count("is_cancelled"), 1)
        # The cancel flag is cleared through the guard before the loop, so a
        # stale True from a prior request cannot abort this batch.
        self.assertFalse(self.state.generation_state["cancelled"])


if __name__ == "__main__":
    unittest.main()
