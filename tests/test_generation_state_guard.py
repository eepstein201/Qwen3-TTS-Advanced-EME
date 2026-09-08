"""Tests for the generation_state thread-safety guard (Step 0C Task 1).

``generation_state`` is mutated from worker threads (streaming progress
callbacks, inference offload threads) while guarded — where at all — by an
``asyncio.Lock`` owned by the event loop. An asyncio lock excludes neither
thread callbacks nor mixes, so several sites are unlocked outright. Task 1
introduces a ``threading.Lock``-based guard plus its tests; production call
sites are routed through it by later tasks.

The pins here are behavioral (the guard's contract), not implementation:
every method must be atomic with respect to ``threading.Lock`` and must
resolve ``app_state.generation_state`` at EACH call (late binding), so tests
can seed/replace the dict object and the guard must follow.

Run: python -m pytest tests/test_generation_state_guard.py -v
"""

import threading
import time
import unittest
from types import SimpleNamespace

from qwen3_tts.server.generation_state_guard import (
    GenerationStateGuard,
    guard_for,
)

CANONICAL_KEYS = frozenset(
    {
        "active",
        "start_time",
        "text_length",
        "mode",
        "batch_index",
        "batch_total",
        "chunk_index",
        "chunk_total",
        "generation_id",
        "cancelled",
    }
)

IDLE_STATE = {
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


def _make_state():
    """A minimal app.state: the dict present, no guard provisioned yet."""
    state = SimpleNamespace()
    state.generation_state = dict(IDLE_STATE)
    return state


class TestGenerationStateGuardBegin(unittest.TestCase):
    """begin() writes the exact production keyset, in one atomic step."""

    def setUp(self):
        self.state = _make_state()
        self.guard = GenerationStateGuard(self.state)

    def test_begin_sets_exact_canonical_keyset(self):
        self.guard.begin(
            "gen-1",
            mode="clone",
            text_length=42,
            start_time=100.0,
            batch_index=2,
            batch_total=5,
        )
        snap = self.guard.snapshot()
        self.assertEqual(frozenset(snap), CANONICAL_KEYS)
        self.assertEqual(
            snap,
            {
                "active": True,
                "start_time": 100.0,
                "text_length": 42,
                "mode": "clone",
                "batch_index": 2,
                "batch_total": 5,
                # begin() never touches the chunk counters — those belong to
                # the progress callback, exactly like today's begins.
                "chunk_index": 0,
                "chunk_total": 0,
                "generation_id": "gen-1",
                "cancelled": False,
            },
        )

    def test_begin_defaults_match_production(self):
        self.guard.begin("gen-2")
        snap = self.guard.snapshot(["active", "mode", "text_length",
                                    "batch_index", "batch_total",
                                    "generation_id", "cancelled"])
        self.assertEqual(
            snap,
            {
                "active": True,
                "mode": "",
                "text_length": 0,
                "batch_index": 0,
                "batch_total": 0,
                "generation_id": "gen-2",
                "cancelled": False,
            },
        )

    def test_begin_defaults_start_time_to_now(self):
        before = time.time()
        self.guard.begin("gen-3")
        after = time.time()
        snap = self.guard.snapshot(["start_time"])
        self.assertGreaterEqual(snap["start_time"], before)
        self.assertLessEqual(snap["start_time"], after)

    def test_begin_re_clears_cancelled(self):
        # Deliberate semantics preservation: today's begins re-clear a stale
        # cancel flag. The #237 erase race is owned by a later step; this pin
        # exists so that step must consciously move this assertion with it.
        self.guard.begin("gen-4")
        self.guard.set_cancelled()
        self.guard.begin("gen-5")
        self.assertFalse(self.guard.is_cancelled())


class TestGenerationStateGuardOwnerReset(unittest.TestCase):
    """reset_if_owner() resets all ten keys only when the id still owns."""

    def setUp(self):
        self.state = _make_state()
        self.guard = GenerationStateGuard(self.state)

    def test_reset_if_owner_true_path_resets_all_ten_keys(self):
        self.guard.begin("gen-1", mode="clone", text_length=42,
                         start_time=100.0, batch_index=1, batch_total=3)
        self.guard.update_progress(7, 9)
        self.guard.set_cancelled()

        self.assertTrue(self.guard.reset_if_owner("gen-1"))
        snap = self.guard.snapshot()
        self.assertEqual(frozenset(snap), CANONICAL_KEYS)
        self.assertEqual(snap, IDLE_STATE)

    def test_reset_if_owner_false_path_leaves_dict_untouched(self):
        self.guard.begin("gen-1", mode="design", text_length=10,
                         start_time=50.0, batch_index=0, batch_total=2)
        self.guard.update_progress(3, 4)
        before = self.guard.snapshot()

        self.assertFalse(self.guard.reset_if_owner("gen-other"))
        self.assertEqual(self.guard.snapshot(), before)

    def test_reset_if_owner_after_replacement_uses_new_dict(self):
        self.guard.begin("gen-a")
        # The replacement dict carries the ownership id plus dirty progress;
        # only the NEW dict may be consulted (True) and reset (keys to idle).
        new_state = dict(IDLE_STATE)
        new_state["generation_id"] = "gen-a"
        new_state["chunk_index"] = 5
        self.state.generation_state = new_state
        self.assertTrue(self.guard.reset_if_owner("gen-a"))
        self.assertEqual(new_state, IDLE_STATE)


class TestGenerationStateGuardSnapshot(unittest.TestCase):
    """snapshot() returns a fresh dict; callers never hold the live one."""

    def setUp(self):
        self.state = _make_state()
        self.guard = GenerationStateGuard(self.state)

    def test_snapshot_reflects_live_values(self):
        self.guard.begin("gen-1", mode="custom", text_length=7)
        self.guard.update_progress(2, 6)
        snap = self.guard.snapshot()
        self.assertEqual(snap["generation_id"], "gen-1")
        self.assertEqual(snap["mode"], "custom")
        self.assertEqual(snap["chunk_index"], 2)
        self.assertEqual(snap["chunk_total"], 6)

    def test_snapshot_returns_a_new_dict(self):
        self.guard.begin("gen-1")
        snap = self.guard.snapshot()
        self.assertIsNot(snap, self.state.generation_state)
        snap["chunk_index"] = 999
        snap["generation_id"] = "mutated"
        self.assertEqual(self.state.generation_state["chunk_index"], 0)
        self.assertEqual(self.state.generation_state["generation_id"], "gen-1")

    def test_snapshot_subset_returns_only_requested_keys(self):
        self.guard.begin("gen-1")
        self.guard.update_progress(1, 2)
        snap = self.guard.snapshot(["chunk_index", "chunk_total"])
        self.assertEqual(snap, {"chunk_index": 1, "chunk_total": 2})

    def test_snapshot_of_live_dict_is_consistent_across_keys(self):
        # A snapshot taken between two in-lock updates can legitimately differ
        # in TIME, but within one snapshot the pair must be the pair written
        # together by update_progress — never a torn mix.
        self.guard.update_progress(5, 10)
        snap = self.guard.snapshot(["chunk_index", "chunk_total"])
        self.assertEqual(snap["chunk_total"], 2 * snap["chunk_index"])


class TestGenerationStateGuardProgress(unittest.TestCase):
    def setUp(self):
        self.state = _make_state()
        self.guard = GenerationStateGuard(self.state)

    def test_update_progress_sets_both_chunk_keys(self):
        self.guard.update_progress(3, 12)
        snap = self.guard.snapshot(["chunk_index", "chunk_total"])
        self.assertEqual(snap, {"chunk_index": 3, "chunk_total": 12})

    def test_update_progress_overwrites_previous_pair(self):
        self.guard.update_progress(1, 2)
        self.guard.update_progress(2, 4)
        snap = self.guard.snapshot(["chunk_index", "chunk_total"])
        self.assertEqual(snap, {"chunk_index": 2, "chunk_total": 4})


class TestGenerationStateGuardCancellation(unittest.TestCase):
    def setUp(self):
        self.state = _make_state()
        self.guard = GenerationStateGuard(self.state)

    def test_set_cancelled_then_is_cancelled(self):
        self.assertFalse(self.guard.is_cancelled())
        self.guard.set_cancelled()
        self.assertTrue(self.guard.is_cancelled())

    def test_clear_cancelled_then_is_cancelled(self):
        self.guard.set_cancelled()
        self.assertTrue(self.guard.is_cancelled())
        self.guard.clear_cancelled()
        self.assertFalse(self.guard.is_cancelled())

    def test_is_cancelled_reads_through_to_live_dict(self):
        self.state.generation_state["cancelled"] = True
        self.assertTrue(self.guard.is_cancelled())


class TestGenerationStateGuardLateBinding(unittest.TestCase):
    """The dict is resolved at each call, never captured at construction."""

    def setUp(self):
        self.state = _make_state()
        self.guard = GenerationStateGuard(self.state)

    def test_guard_follows_dict_replacement(self):
        self.guard.begin("gen-a")
        self.guard.set_cancelled()
        old = self.state.generation_state
        self.assertTrue(old["cancelled"])

        # Swap in a fresh idle dict — the guard must operate on the NEW one.
        self.state.generation_state = dict(IDLE_STATE)
        self.assertFalse(self.guard.is_cancelled())

        self.guard.set_cancelled()
        self.assertTrue(self.state.generation_state["cancelled"])
        # The OLD dict is untouched by post-swap writes.
        self.assertTrue(old["cancelled"])

        self.guard.begin("gen-b")
        self.assertEqual(self.state.generation_state["generation_id"], "gen-b")
        self.assertEqual(old["generation_id"], "gen-a")

    def test_guard_for_returns_guard_bound_to_swapped_dict(self):
        guard = guard_for(self.state)
        self.state.generation_state = dict(IDLE_STATE)
        guard.begin("gen-x")
        self.assertEqual(self.state.generation_state["generation_id"], "gen-x")


class TestGuardForProvisioning(unittest.TestCase):
    """guard_for() caches one guard per state and self-provisions on fakes."""

    def test_guard_for_is_idempotent(self):
        state = _make_state()
        first = guard_for(state)
        second = guard_for(state)
        self.assertIs(first, second)
        self.assertIs(state.generation_state_guard, first)

    def test_guard_for_self_provisions_on_bare_namespace(self):
        state = SimpleNamespace()
        state.generation_state = dict(IDLE_STATE)
        self.assertFalse(hasattr(state, "generation_state_guard"))
        guard = guard_for(state)
        self.assertIsInstance(guard, GenerationStateGuard)
        self.assertIs(state.generation_state_guard, guard)
        guard.begin("gen-1")
        self.assertTrue(state.generation_state["active"])

    def test_guard_for_does_not_touch_generation_lock(self):
        # The asyncio generation_lock attribute is NOT the guard's business:
        # provisioning must leave it alone (absent stays absent).
        state = SimpleNamespace()
        state.generation_state = dict(IDLE_STATE)
        guard_for(state)
        self.assertFalse(hasattr(state, "generation_lock"))


class TestLifespanWiring(unittest.TestCase):
    """lifespan() provisions the guard eagerly alongside the dict."""

    def test_lifespan_provisions_guard_next_to_the_dict(self):
        import inspect

        import qwen3_tts.server.app_lifespan as app_lifespan_module

        source = inspect.getsource(app_lifespan_module.lifespan)
        self.assertIn(
            "GenerationStateGuard(app.state)",
            source,
            "lifespan() must construct the guard eagerly next to "
            "app.state.generation_state",
        )

    def test_lifespan_keeps_asyncio_generation_lock(self):
        import inspect

        import qwen3_tts.server.app_lifespan as app_lifespan_module

        source = inspect.getsource(app_lifespan_module.lifespan)
        self.assertIn("app.state.generation_lock = asyncio.Lock()", source)


class TestGenerationStateGuardConcurrency(unittest.TestCase):
    """Deterministic thread smoke: in-lock pairs never read torn."""

    WRITER_COUNT = 4
    WRITER_ITERATIONS = 2000

    def test_parallel_writers_never_tear_the_progress_pair(self):
        guard = GenerationStateGuard(_make_state())

        stop = threading.Event()
        failures = []
        errors = []

        def writer(base):
            try:
                for offset in range(self.WRITER_ITERATIONS):
                    guard.update_progress(base + offset, 2 * (base + offset))
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(exc)

        def reader():
            try:
                while not stop.is_set():
                    snap = guard.snapshot(["chunk_index", "chunk_total"])
                    if snap["chunk_total"] != 2 * snap["chunk_index"]:
                        failures.append(snap)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(exc)

        JOIN_TIMEOUT_SEC = 30.0
        readers = [
            threading.Thread(target=reader, name=f"reader-{n}", daemon=True)
            for n in range(2)
        ]
        writers = [
            threading.Thread(
                target=writer, args=(n * self.WRITER_ITERATIONS,), name=f"writer-{n}"
            )
            for n in range(self.WRITER_COUNT)
        ]

        for thread in readers + writers:
            thread.start()
        for thread in writers:
            thread.join(timeout=JOIN_TIMEOUT_SEC)
            self.assertFalse(
                thread.is_alive(),
                f"{thread.name} did not finish within {JOIN_TIMEOUT_SEC}s",
            )

        stop.set()
        for thread in readers:
            thread.join(timeout=JOIN_TIMEOUT_SEC)
            self.assertFalse(
                thread.is_alive(),
                f"{thread.name} did not finish within {JOIN_TIMEOUT_SEC}s",
            )

        self.assertEqual(errors, [])
        self.assertEqual(failures, [])
