"""Every ``generation_state`` touch must route through the guard.

Step 0C Task 2 routed ``handle_generate``'s batch path; Task 3 routes the
HTTP streaming path (``audio_stream_generator``) and the ``/ws`` path
(``websocket._stream_generation``). All three paths mutate
``app.state.generation_state`` (a plain dict shared with worker threads) —
historically with no lock at all, or under an ``asyncio.Lock`` that excludes
nothing off the loop. Task 1 landed ``GenerationStateGuard`` (a
``threading.Lock`` per app state).

These tests pin the ROUTING at the real call site: a recording proxy replaces
``guard_for`` in the module under test so every guard method the handler calls
is captured, while the calls still delegate to a real guard (so the writes land
in the dict and the handler behaves exactly as in production). Driving the
full pipeline is impractical without a model, so ``run_inference`` /
``run_inference_streaming`` are faked — but the streaming fakes run on a REAL
daemon thread (the handler spawns ``inference_thread`` itself), which is
EXACTLY the thread production fires the progress callback on. That is why the
guard must be a ``threading.Lock``: the callback's write has to be serialized
against the event loop, and an ``asyncio.Lock`` would exclude nothing there.

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

    def set_cancelled(self, target_id=None):
        self.calls.append(("set_cancelled", target_id))
        return self._inner.set_cancelled(target_id)

    def is_cancelled_for(self, generation_id):
        self.calls.append(("is_cancelled_for", generation_id))
        return self._inner.is_cancelled_for(generation_id)

    def register_pending(self, generation_id):
        self.calls.append(("register_pending", generation_id))
        return self._inner.register_pending(generation_id)

    def deregister_pending(self, generation_id):
        self.calls.append(("deregister_pending", generation_id))
        return self._inner.deregister_pending(generation_id)

    def peek_pending_target(self):
        self.calls.append(("peek_pending_target",))
        return self._inner.peek_pending_target()

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


def _make_state():
    """Minimal app.state for exercising ``handle_generate``.

    Mirrors ``tests.test_batch_generation_state_ownership._make_state``: a
    SimpleNamespace so the test is self-contained (no real FastAPI app). The
    guard is deliberately NOT pre-provisioned — the handler must resolve it
    lazily via ``guard_for`` exactly as production would on a bare state.
    ``generation_lock`` is wired to FAIL if ENTERED (see
    ``_generation_lock_tripwire``): the routed batch path must no longer
    need it.
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
    state.generation_lock = _generation_lock_tripwire("batch")
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


def _generation_lock_tripwire(path_name):
    """Return a ``generation_lock`` stand-in that FAILS the test on ENTRY.

    The side effect must live on the ``__aenter__`` CHILD mock, not on the
    parent: an ``AsyncMock``'s dunder children do not inherit the parent's
    ``side_effect``, so ``AsyncMock(side_effect=_fail)`` is INERT on
    ``async with lock:`` — the exact statement these paths must never issue
    again. With the side effect on ``__aenter__``, ANY entry raises.
    ``__aexit__`` resolves falsy (never suppresses), so a tripped entry
    surfaces the failure instead of being swallowed by an exit mocking
    success. Mere attribute access is deliberately NOT detected: the
    contract these paths are held to is "no entry".

    Task 3 removed the last two ``generation_lock`` blocks that guarded
    ``generation_state`` (both in ``websocket.py``); Task 4 removes the
    ``/cancel-generation`` block in ``app.py``. No generation path may take
    the asyncio lock again — a ``threading.Lock`` is the only primitive that
    excludes the worker threads.
    """

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            f"the {path_name} path entered state.generation_lock -- the "
            "threading.Lock generation-state guard is the only primitive "
            "that excludes the worker threads"
        )

    lock = AsyncMock(name=f"generation_lock[{path_name}]")
    lock.__aenter__.side_effect = _fail
    lock.__aexit__.return_value = False
    return lock


def _make_stream_state():
    """``_make_state`` plus what the HTTP streaming path needs.

    ``audio_stream_generator`` takes ``state.pending_lock`` around the pending
    queue, which the batch driver never touches.
    """
    state = _make_state()
    state.pending_lock = asyncio.Lock()
    state.generation_lock = _generation_lock_tripwire("streaming")
    return state


def _make_ws_state():
    """``_make_state`` with a /ws-specific ``generation_lock`` tripwire."""
    state = _make_state()
    state.generation_lock = _generation_lock_tripwire("/ws")
    return state


def _make_stream_request(text, mode="custom"):
    """Build the ``GenerateRequest`` the streaming driver sends."""
    from qwen3_tts.server.validation import GenerateRequest

    return GenerateRequest(text=text, mode=mode)


async def _drive_stream_async(
    state,
    recording,
    text="streaming routing probe",
    progress_calls=_PROGRESS_CALLS,
    on_stream_done=None,
    guard_for_mock=None,
):
    """Drive one ``/generate-stream`` request through ``handle_generate_stream``.

    Returns the raw streamed body. ``run_inference_streaming`` is faked: the
    fake invokes the routed ``progress_callback`` and yields one chunk per
    pair, so the callback fires on the handler's OWN daemon ``inference_thread``
    — the identical thread production fires it on. ``on_stream_done`` runs on
    that thread after the last routed write (the visibility probe).
    """
    fake_chunk = np.zeros(100, dtype=np.float32)

    def fake_streaming(**kwargs):
        callback = kwargs.get("progress_callback")
        assert callback is not None, (
            "handle_generate_stream passed no progress_callback to "
            "run_inference_streaming; the streaming chunk progress is "
            "unpinnable"
        )
        for chunk_index, chunk_total in progress_calls:
            callback(chunk_index, chunk_total)
            yield (fake_chunk, 24000)
        if on_stream_done is not None:
            on_stream_done()

    from qwen3_tts.server.app_generation import handle_generate_stream

    req = _make_stream_request(text)
    if guard_for_mock is not None:
        guard_patch = patch(f"{_APP_GENERATION}.guard_for", new=guard_for_mock)
    else:
        guard_patch = patch(f"{_APP_GENERATION}.guard_for", return_value=recording)
    with patch(
        f"{_APP_GENERATION}._check_memory_available",
        return_value=(True, 4096),
    ), patch(
        # The name the handler RESOLVES: app_generation imports
        # _validate_generation_request at module scope, so patching the
        # definition site in qwen3_tts.server.validation rebinds a symbol
        # the handler never looks up (real validation ran — harmless, but
        # the patch lied about being in control).
        f"{_APP_GENERATION}._validate_generation_request"
    ), guard_patch, patch(
        f"{_ENGINE}.run_inference_streaming", side_effect=fake_streaming
    ):
        response = await handle_generate_stream(
            request=MagicMock(),
            state=state,
            req=req,
            security={"max_text_length": 50000},
            config_provider=None,
        )
        body = b""
        async for part in response.body_iterator:
            body += part
        return body


def _stream_frame_count(body):
    """Count length-prefixed audio frames in a streamed body."""
    import struct

    offset, frames = 0, 0
    while offset + 8 <= len(body):
        _sr, length = struct.unpack("<II", body[offset : offset + 8])
        offset += 8 + length
        frames += 1
    return frames


async def _drive_ws_async(
    state,
    recording,
    text="ws routing probe",
    guard_for_mock=None,
):
    """Drive one /ws generation through ``websocket._stream_generation``.

    ``guard_for`` is patched in ``qwen3_tts.server.websocket`` with
    ``create=True``: before Task 3 the module has no such attribute at all,
    and a created-but-never-called patch makes the RED failure an EMPTY
    RECORDING assertion (the missing routing) rather than a patch-mechanics
    AttributeError.
    """
    from qwen3_tts.server.websocket import _stream_generation

    fake_chunk = np.zeros(100, dtype=np.float32)

    def fake_streaming(**_kwargs):
        yield (fake_chunk, 24000)

    ws = MagicMock()
    ws.send_bytes = AsyncMock()
    ws.send_json = AsyncMock()

    if guard_for_mock is not None:
        guard_patch = patch(
            "qwen3_tts.server.websocket.guard_for", new=guard_for_mock, create=True
        )
    else:
        guard_patch = patch(
            "qwen3_tts.server.websocket.guard_for",
            return_value=recording,
            create=True,
        )
    with patch(
        f"{_ENGINE}.run_inference_streaming", side_effect=fake_streaming
    ), patch(
        # Left on the definition site DELIBERATELY: websocket.py imports
        # _validate_generation_request INSIDE the function, so it resolves
        # the definition site at call time and this patch IS live.
        "qwen3_tts.server.validation._validate_generation_request"
    ), patch(
        "qwen3_tts.server.app_lifespan._check_memory_available",
        return_value=(True, 4096),
    ), guard_patch:
        await asyncio.wait_for(
            _stream_generation(
                websocket=ws,
                app_state=state,
                text=text,
                mode="custom",
                data={"text": text, "mode": "custom"},
                stop_event=threading.Event(),
                disconnect_event=threading.Event(),
            ),
            timeout=10,
        )
    return ws


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
        # The name the handler RESOLVES (module-scope import in
        # app_generation) — see the same patch in _drive_stream_async.
        f"{_APP_GENERATION}._validate_generation_request"
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


class TestGenerationLockTripwireIsArmed(unittest.TestCase):
    """The tripwire must TRIP — an inert tripwire guards nothing.

    Step 0C Task 4 fold: the state factories used to build
    ``AsyncMock(side_effect=_fail)``, which never fires on ``async with``
    (a mock's dunder children do not inherit the parent's ``side_effect``),
    so the tripwire was inert exactly where it mattered. These tests prove
    the fixed construction raises on entry in every factory.
    """

    def test_entering_the_tripwired_lock_raises_in_every_factory(self):
        """``async with state.generation_lock:`` must fail the test on all
        three factory states (batch, streaming, /ws)."""
        cases = (
            (_make_state, "batch"),
            (_make_stream_state, "streaming"),
            (_make_ws_state, "/ws"),
        )
        for factory, path_name in cases:
            with self.subTest(factory=factory.__name__):
                lock = factory().generation_lock
                with self.assertRaises(AssertionError) as caught:
                    asyncio.run(lock.__aenter__())
                self.assertIn(path_name, str(caught.exception))

    def test_tripwire_exit_never_suppresses(self):
        """``__aexit__`` resolves falsy, so a tripped entry surfaces the
        failure instead of an exit mocking success."""
        lock = _make_ws_state().generation_lock
        self.assertFalse(asyncio.run(lock.__aexit__(None, None, None)))

    def test_naive_side_effect_mock_is_inert_on_entry(self):
        """Documents WHY the side effect lives on ``__aenter__``: the naive
        parent-level construction this fold replaced does not trip on the
        entry protocol at all. If this ever fails, mock dunder semantics
        changed and the tripwire can be simplified back."""
        naive = AsyncMock(side_effect=AssertionError("inert by construction"))
        asyncio.run(naive.__aenter__())  # must NOT raise


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
        the production order: mint-time pending registration -> loop
        target-aware cancel check -> begin -> progress -> chunk_total
        snapshot -> in-lock reset -> finally deregister -> finally reset.

        The pre-loop blanket ``clear_cancelled`` is gone (Controller Ruling
        E): it ran AFTER the pending registration and would erase a Window-2
        cancel addressed to this very batch."""
        result = _drive_batch(self.state, self.recording)

        names = self.recording.names()
        self.assertEqual(
            names,
            [
                "register_pending",
                "is_cancelled_for",
                "begin",
                # one update_progress per fed progress pair
                *["update_progress"] * len(_PROGRESS_CALLS),
                "snapshot",
                "reset_if_owner",
                "deregister_pending",
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

        # ...the mint-time pending registration, the loop's target-aware
        # cancel check, and the finally deregistration are all bound to the
        # SAME id begin() stamped — one id, attributed end to end.
        self.assertEqual(self.recording.first("register_pending")[1], begin_gen_id)
        self.assertEqual(self.recording.first("is_cancelled_for")[1], begin_gen_id)
        self.assertEqual(self.recording.first("deregister_pending")[1], begin_gen_id)

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
        """The per-item cancel check must read via ``is_cancelled_for`` — a
        stale unlocked read is exactly what the guard exists to prevent, and
        the check must be addressed to THIS batch's own id (Controller
        Ruling E: there is no more pre-loop blanket clear to fall back on;
        target-matched honoring is the whole stale-flag defense now)."""
        _drive_batch(self.state, self.recording)
        cancel_checks = self.recording.all_of("is_cancelled_for")
        self.assertEqual(len(cancel_checks), 1)
        begin_gen_id = self.recording.first("begin")[1]
        self.assertEqual(cancel_checks[0][1], begin_gen_id)


@_skip
class TestStreamingRoutesThroughGuard(unittest.TestCase):
    """The HTTP streaming path must touch generation_state ONLY via the guard."""

    def setUp(self):
        self.state = _make_stream_state()
        from qwen3_tts.server.generation_state_guard import GenerationStateGuard

        self.recording = _RecordingGuard(GenerationStateGuard(self.state))
        self.text = "streaming routing probe"

    def test_streaming_begin_and_finally_reset_route_through_the_guard(self):
        """The unlocked pre-thread begin and the finally reset must both go
        through the guard, bound to one generation id, in that order — the
        reset only via ``reset_if_owner`` (ownership-checked), never a raw
        idle write."""
        body = asyncio.run(
            _drive_stream_async(self.state, self.recording, text=self.text)
        )

        # The pipeline really ran: one audio frame per fed progress pair.
        self.assertEqual(
            _stream_frame_count(body),
            len(_PROGRESS_CALLS),
            "the fake streaming pipeline did not deliver the expected frames",
        )

        names = self.recording.names()
        self.assertTrue(
            names,
            "the streaming path made NO guard calls at all -- its begin and "
            "finally reset are not routed through the guard",
        )
        self.assertEqual(
            names[0],
            "begin",
            "the streaming path did not route its begin through the guard",
        )
        self.assertEqual(
            names[-1],
            "reset_if_owner",
            "the streaming finally did not route its reset through the guard",
        )
        self.assertEqual(
            set(names),
            {"begin", "update_progress", "is_cancelled_for", "reset_if_owner"},
            "the streaming path touched generation_state outside the guard "
            f"(recorded methods: {sorted(set(names))})",
        )

        # begin() carries the streaming fields — and NOT the batch fields,
        # which the old 6-key streaming begin never wrote.
        begin = self.recording.first("begin")
        begin_gen_id, begin_kwargs = begin[1], dict(begin[2])
        self.assertEqual(begin_kwargs["mode"], "custom")
        self.assertEqual(begin_kwargs["text_length"], len(self.text))
        self.assertNotIn(
            "batch_index",
            begin_kwargs,
            "the streaming begin must not stamp batch progress",
        )
        self.assertNotIn("batch_total", begin_kwargs)
        self.assertIsInstance(begin_gen_id, str)

        # The finally reset is ownership-bound to begin's id.
        self.assertEqual(
            [call[1] for call in self.recording.all_of("reset_if_owner")],
            [begin_gen_id],
            "the streaming reset is not bound to the generation id begin() "
            "stamped",
        )

        # Behavior preserved: the dict returns to the idle shape afterwards.
        self.assertFalse(self.state.generation_state["active"])
        self.assertIsNone(self.state.generation_state["generation_id"])
        self.assertEqual(self.state.generation_state["chunk_total"], 0)
        self.assertEqual(self.state.generation_state["chunk_index"], 0)

    def test_streaming_progress_callback_routes_through_update_progress(self):
        """The progress callback, fired on the handler's own daemon inference
        thread, must reach the dict as atomic guard ``update_progress`` pairs
        — and the stop-check must read via ``is_cancelled_for``, addressed to
        this stream's own generation id (a cancel addressed to some OTHER
        generation must not stop an unrelated stream)."""
        observed = []

        def probe_from_inference_thread():
            # Runs on the daemon inference thread, right after the last
            # routed progress write: the pair must be visible and intact.
            observed.append(
                self.recording.snapshot(["chunk_index", "chunk_total"])
            )

        asyncio.run(
            _drive_stream_async(
                self.state,
                self.recording,
                text=self.text,
                on_stream_done=probe_from_inference_thread,
            )
        )

        self.assertEqual(
            self.recording.all_of("update_progress"),
            [("update_progress", ci, ct) for ci, ct in _PROGRESS_CALLS],
            "the streaming progress callback did not route its writes "
            "through the guard's update_progress",
        )
        # One stop-check per streamed chunk, each reading through the guard,
        # each addressed to THIS stream's own generation id.
        cancel_checks = self.recording.all_of("is_cancelled_for")
        self.assertEqual(
            len(cancel_checks),
            len(_PROGRESS_CALLS),
            "the streaming thread did not do its per-chunk cancel check "
            "through the guard",
        )
        begin_gen_id = self.recording.first("begin")[1]
        self.assertTrue(
            all(call[1] == begin_gen_id for call in cancel_checks),
            "the streaming stop-check is not addressed to this stream's own "
            "generation id",
        )
        self.assertEqual(
            observed,
            [{"chunk_index": _PROGRESS_CALLS[-1][0],
              "chunk_total": _PROGRESS_CALLS[-1][1]}],
            "a snapshot taken from the inference thread right after the "
            "routed callback did not observe the written pair",
        )

    def test_streaming_progress_never_observes_a_torn_pair(self):
        """Serialization-PROPERTY pin, not a RED driver (same scope as the
        batch twin below): while the daemon thread hammers the routed
        callback, a loop-side snapshot through the SAME guard never observes
        chunk_index from one write and chunk_total from another."""
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
                await asyncio.sleep(0)
            iterations.append(count)

        async def run():
            reader = asyncio.ensure_future(reader_until_stop())
            try:
                await _drive_stream_async(self.state, self.recording)
            finally:
                stop.set()
                await reader

        asyncio.run(run())

        self.assertEqual(
            torn,
            [],
            f"snapshot observed torn progress pairs: {torn[:5]!r}",
        )
        self.assertGreater(iterations[0], 0, "the reader task never ran")

    def test_streaming_guard_is_resolved_once_per_request(self):
        """``guard_for`` must be called once per streaming request — not per
        chunk (it takes a module-level construction lock per call)."""
        guard_for_mock = MagicMock(return_value=self.recording)
        asyncio.run(
            _drive_stream_async(
                self.state, self.recording, guard_for_mock=guard_for_mock
            )
        )
        self.assertEqual(
            guard_for_mock.call_count,
            1,
            "handle_generate_stream resolved the guard more than once for a "
            "single request",
        )


@_skip
class TestWsRoutesThroughGuard(unittest.TestCase):
    """The /ws path must touch generation_state ONLY via the guard."""

    def setUp(self):
        self.state = _make_ws_state()
        from qwen3_tts.server.generation_state_guard import GenerationStateGuard

        self.recording = _RecordingGuard(GenerationStateGuard(self.state))
        self.text = "ws routing probe"

    def test_ws_begin_and_finally_reset_route_through_the_guard(self):
        """The begin under the old ``generation_lock`` block and the finally
        reset must both go through the guard — and the asyncio lock must not
        be ENTERED: the tripwire raises on ``async with`` entry (see
        ``_generation_lock_tripwire``; attribute access is not detected)."""
        asyncio.run(_drive_ws_async(self.state, self.recording, text=self.text))

        names = self.recording.names()
        self.assertEqual(
            names,
            ["begin", "reset_if_owner"],
            "the /ws path touched generation_state outside the guard, or in "
            f"an unexpected order (recorded: {names})",
        )

        begin = self.recording.first("begin")
        begin_gen_id, begin_kwargs = begin[1], dict(begin[2])
        self.assertEqual(begin_kwargs["mode"], "custom")
        self.assertEqual(begin_kwargs["text_length"], len(self.text))
        self.assertIsInstance(begin_gen_id, str)

        self.assertEqual(
            [call[1] for call in self.recording.all_of("reset_if_owner")],
            [begin_gen_id],
            "the /ws reset is not bound to the generation id begin() stamped",
        )

        # Behavior preserved: the dict returns to the idle shape afterwards.
        self.assertFalse(self.state.generation_state["active"])
        self.assertIsNone(self.state.generation_state["generation_id"])
        self.assertEqual(self.state.generation_state["cancelled"], False)

    def test_ws_guard_is_resolved_once_per_generation(self):
        """``guard_for`` must be called once per /ws generation scope — the
        route is a persistent multi-request socket, but the state object is
        fixed, so per-``_stream_generation`` resolution is the scope that
        owns begin/reset."""
        guard_for_mock = MagicMock(return_value=self.recording)
        asyncio.run(
            _drive_ws_async(
                self.state, self.recording, guard_for_mock=guard_for_mock
            )
        )
        self.assertEqual(
            guard_for_mock.call_count,
            1,
            "_stream_generation resolved the guard more than once for a "
            "single generation",
        )


if __name__ == "__main__":
    unittest.main()
