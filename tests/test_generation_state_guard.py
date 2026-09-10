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

Task 5 appends the Step 0C exit-criterion structural pin
(``TestGenerationStateRawAccessIsPinnedToTheGuard``): production modules must
not touch the dict raw at all.

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
        "cancel_target_id",
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
    "cancel_target_id": None,
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
                "cancel_target_id": None,
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

    def test_begin_erases_a_stale_non_matching_non_pending_cancel(self):
        # Step 1A moved this pin: begin() no longer blanket re-clears
        # `cancelled` on every call. A cancel addressed to a DIFFERENT,
        # non-pending generation id is genuinely stale by the time this
        # begin() runs, so it is still erased here — preserving the old
        # re-clear's status-surface hygiene for the one case where no live
        # generation can ever honor it.
        #
        # Fix round 1 (Important 1): gen-4 must have actually FINISHED
        # (reset_if_owner ran, returning generation_id to idle) before its
        # cancel counts as stale here — while gen-4 is still the recorded
        # owner, begin()'s fourth case now preserves a cancel targeted at
        # it, exactly as a live-owner cancel must survive a sibling's
        # begin(). The cancel is set AFTER reset_if_owner (modeling a cancel
        # that arrives once gen-4 has already gone idle), so this exercises
        # the genuinely-stale branch rather than the live-owner one.
        self.guard.begin("gen-4")
        self.guard.reset_if_owner("gen-4")
        self.guard.set_cancelled("gen-4")
        self.guard.begin("gen-5")
        self.assertFalse(self.guard.snapshot(["cancelled"])["cancelled"])
        self.assertIsNone(self.guard.snapshot(["cancel_target_id"])["cancel_target_id"])

    def test_begin_preserves_a_cancel_targeted_at_the_same_id(self):
        # Window 1's fix: a cancel addressed to THIS batch must survive its
        # own next begin(), so every later item in the batch still sees it.
        self.guard.begin("gen-6")
        self.guard.set_cancelled("gen-6")
        self.guard.begin("gen-6")
        self.assertTrue(self.guard.is_cancelled_for("gen-6"))

    def test_begin_preserves_a_cancel_targeted_at_a_pending_sibling(self):
        # Window 2's fix (Controller Ruling D): a cancel addressed to a
        # sibling generation that has minted but not yet begun must survive
        # an unrelated begin() — otherwise an ordinary concurrent batch
        # erases a cancel nobody has had the chance to honor yet.
        self.guard.register_pending("gen-pending")
        self.guard.set_cancelled("gen-pending")
        self.guard.begin("gen-other")
        self.assertTrue(self.guard.is_cancelled_for("gen-pending"))

    def test_begin_preserves_a_cancel_targeted_at_the_live_owner_across_a_different_begin(self):
        # Fix round 1 (Important 1): a cancel addressed to the CURRENT owner
        # must survive a concurrent, different generation's begin(), not just
        # its own. Batch A begins, is cancelled, releases the lock between
        # items without reaching its own next begin() — a sibling's begin("B")
        # must not erase A's still-live cancel. At the moment begin("B") runs,
        # state["generation_id"] is still "A" (not yet overwritten), which is
        # exactly the signal that distinguishes this from a genuinely stale
        # target.
        self.guard.begin("gen-A")
        self.guard.set_cancelled("gen-A")
        self.guard.begin("gen-B")
        self.assertTrue(self.guard.is_cancelled_for("gen-A"))

    def test_begin_leaves_cancelled_untouched_when_no_target_is_set(self):
        # cancel_target_id is None: there is nothing to erase, so begin()
        # must leave `cancelled` exactly as it found it — including when it
        # is already True. Starting from cancelled=False (the idle default)
        # would pass identically under a wrong blanket `cancelled = False`
        # re-clear, so this pin deliberately starts from cancelled=True with
        # no target and asserts it SURVIVES begin().
        self.guard.set_cancelled()  # cancelled=True, cancel_target_id=None
        self.guard.begin("gen-7")
        self.assertTrue(self.guard.snapshot(["cancelled"])["cancelled"])
        self.assertIsNone(self.guard.snapshot(["cancel_target_id"])["cancel_target_id"])

    def test_begin_consumes_its_own_pending_registration(self):
        self.guard.register_pending("gen-9")
        self.guard.begin("gen-9")
        self.assertFalse(self.guard._pop_pending_target("gen-9"))


class TestGenerationStateGuardOwnerReset(unittest.TestCase):
    """reset_if_owner() resets all eleven keys only when the id still owns."""

    def setUp(self):
        self.state = _make_state()
        self.guard = GenerationStateGuard(self.state)

    def test_reset_if_owner_true_path_resets_all_eleven_keys(self):
        self.guard.begin("gen-1", mode="clone", text_length=42,
                         start_time=100.0, batch_index=1, batch_total=3)
        self.guard.update_progress(7, 9)
        self.guard.set_cancelled("gen-1")

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

    def test_snapshot_materializes_keys_before_taking_the_lock(self):
        # The keys argument is caller-supplied and may be a generator: its
        # code must run BEFORE the lock is acquired, or a generator that
        # itself touches the guard would deadlock on the non-reentrant lock.
        # The probe reads _lock.locked() on purpose — "the lock is not held
        # while caller code runs" is precisely the contract under test.
        held_while_materializing = []

        def keys_generator():
            held_while_materializing.append(self.guard._lock.locked())
            yield "chunk_index"
            yield "chunk_total"

        snap = self.guard.snapshot(keys_generator())
        self.assertEqual(snap, {"chunk_index": 0, "chunk_total": 0})
        self.assertEqual(held_while_materializing, [False])

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

    def test_set_cancelled_records_the_target_id(self):
        self.guard.set_cancelled("gen-1")
        snap = self.guard.snapshot(["cancelled", "cancel_target_id"])
        self.assertEqual(snap, {"cancelled": True, "cancel_target_id": "gen-1"})

    def test_set_cancelled_defaults_target_to_none(self):
        # The zero-arg call stays valid: no production call site uses it any
        # more (every one now passes an explicit target), but the default
        # exists for direct unit-test convenience, per the guard's docstring.
        self.guard.set_cancelled()
        snap = self.guard.snapshot(["cancelled", "cancel_target_id"])
        self.assertEqual(snap, {"cancelled": True, "cancel_target_id": None})


class TestGenerationStateGuardIsCancelledFor(unittest.TestCase):
    """is_cancelled_for() is the target-matched read the fix relies on."""

    def setUp(self):
        self.state = _make_state()
        self.guard = GenerationStateGuard(self.state)

    def test_true_when_target_matches(self):
        self.guard.set_cancelled("gen-1")
        self.assertTrue(self.guard.is_cancelled_for("gen-1"))

    def test_false_when_target_does_not_match(self):
        self.guard.set_cancelled("gen-1")
        self.assertFalse(self.guard.is_cancelled_for("gen-2"))

    def test_false_when_cancelled_but_target_is_none(self):
        # A cancel_target_id of None is honored by nobody, even though
        # `cancelled` itself is True.
        self.state.generation_state["cancelled"] = True
        self.state.generation_state["cancel_target_id"] = None
        self.assertFalse(self.guard.is_cancelled_for("gen-1"))

    def test_does_not_keyerror_on_the_hand_rolled_ten_key_shape(self):
        # 13 test modules elsewhere hand-roll the old ten-key state dict
        # without cancel_target_id; is_cancelled_for must not KeyError on it.
        ten_key_state = dict(IDLE_STATE)
        del ten_key_state["cancel_target_id"]
        ten_key_state["cancelled"] = True
        self.state.generation_state = ten_key_state
        self.assertFalse(self.guard.is_cancelled_for("gen-1"))


class TestGenerationStateGuardPendingRegistry(unittest.TestCase):
    """The pending registry tracks ids minted but not yet begun."""

    def setUp(self):
        self.state = _make_state()
        self.guard = GenerationStateGuard(self.state)

    def test_register_then_pop_round_trips(self):
        self.guard.register_pending("gen-1")
        self.assertTrue(self.guard._pop_pending_target("gen-1"))

    def test_pop_returns_false_when_never_registered(self):
        self.assertFalse(self.guard._pop_pending_target("gen-never"))

    def test_pop_is_one_shot(self):
        self.guard.register_pending("gen-1")
        self.assertTrue(self.guard._pop_pending_target("gen-1"))
        self.assertFalse(self.guard._pop_pending_target("gen-1"))

    def test_deregister_pending_is_idempotent_on_an_unregistered_id(self):
        # discard-style: deregistering an id that was never registered must
        # not raise, must be safe to repeat, and must not disturb a cancel
        # targeted at some unrelated id (there was nothing for it to match).
        self.guard.set_cancelled("gen-1")
        self.guard.deregister_pending("gen-never-registered")
        self.guard.deregister_pending("gen-never-registered")
        self.assertTrue(self.guard.is_cancelled_for("gen-1"))

    def test_deregister_pending_removes_the_id(self):
        self.guard.register_pending("gen-1")
        self.guard.deregister_pending("gen-1")
        self.assertFalse(self.guard._pop_pending_target("gen-1"))

    def test_deregister_pending_clears_a_cancel_targeted_at_it(self):
        self.guard.register_pending("gen-x")
        self.guard.set_cancelled("gen-x")
        self.guard.deregister_pending("gen-x")
        self.assertFalse(self.guard.snapshot(["cancelled"])["cancelled"])
        self.assertIsNone(self.guard.snapshot(["cancel_target_id"])["cancel_target_id"])

    def test_deregister_pending_does_not_clear_a_cancel_targeted_elsewhere(self):
        self.guard.register_pending("gen-x")
        self.guard.register_pending("gen-y")
        self.guard.set_cancelled("gen-y")
        self.guard.deregister_pending("gen-x")
        self.assertTrue(self.guard.is_cancelled_for("gen-y"))

    def test_peek_pending_target_returns_none_when_nothing_pending(self):
        self.assertIsNone(self.guard.peek_pending_target())

    def test_peek_pending_target_returns_the_only_pending_id(self):
        self.guard.register_pending("gen-solo")
        self.assertEqual(self.guard.peek_pending_target(), "gen-solo")

    def test_peek_pending_target_is_deterministic_across_several_pending(self):
        # Documented tie-break: lexicographically smallest of the pending
        # ids. Registration order is reversed here on purpose to prove the
        # result does not depend on insertion order.
        self.guard.register_pending("gen-z")
        self.guard.register_pending("gen-a")
        self.guard.register_pending("gen-m")
        self.assertEqual(self.guard.peek_pending_target(), "gen-a")

    def test_peek_pending_target_does_not_consume_the_registration(self):
        self.guard.register_pending("gen-1")
        self.guard.peek_pending_target()
        self.assertEqual(self.guard.peek_pending_target(), "gen-1")
        self.assertTrue(self.guard._pop_pending_target("gen-1"))


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
        self.assertFalse(self.guard.snapshot(["cancelled"])["cancelled"])

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

    def test_concurrent_first_touch_returns_one_shared_guard(self):
        # Simultaneous first-touches must all receive the SAME guard object.
        # A last-writer-wins race here hands different threads guards backed
        # by DIFFERENT threading.Locks — zero mutual exclusion. Release every
        # contender from one Event (no sleeps); join with timeouts and assert
        # completion.
        state = _make_state()
        release = threading.Event()
        results = []
        errors = []
        THREADS = 4
        JOIN_TIMEOUT_SEC = 30.0

        def touch():
            try:
                release.wait(timeout=JOIN_TIMEOUT_SEC)
                results.append(guard_for(state))
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(exc)

        threads = [
            threading.Thread(target=touch, name=f"first-touch-{n}")
            for n in range(THREADS)
        ]
        for thread in threads:
            thread.start()
        release.set()
        for thread in threads:
            thread.join(timeout=JOIN_TIMEOUT_SEC)
            self.assertFalse(
                thread.is_alive(),
                f"{thread.name} did not finish within {JOIN_TIMEOUT_SEC}s",
            )

        self.assertEqual(errors, [])
        self.assertEqual(len(results), THREADS)
        for guard in results:
            self.assertIs(guard, results[0])
        self.assertIs(state.generation_state_guard, results[0])


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


class TestGenerationStateRawAccessIsPinnedToTheGuard(unittest.TestCase):
    """Step 0C exit criterion, held structurally: every production
    ``generation_state`` read/write goes through the guard.

    Tasks 1-4 routed every generation path and every reader through
    ``GenerationStateGuard``. This pin keeps that true: the five production
    modules that ever touched the dict are scanned at the TOKEN level, and
    every surviving ``generation_state`` NAME occurrence must sit inside the
    allowlist below.

    Deliberate mechanics:

    - The scan collects NAME tokens spelled exactly ``generation_state``.
      Comments, docstrings and string literals are different token kinds, so
      a comment mentioning the dict is not an access, and the attribute the
      lifespan provisions (``generation_state_guard``) is a DIFFERENT
      identifier that is not flagged.
    - The allowlist is (file basename, line substring) tuples and every
      entry must stay LIVE: an entry that no longer matches fails the pin,
      so reformatting the init block forces a conscious allowlist update
      instead of a silent widening.
    - The scanned file list is asserted too: a rename, move or deletion of
      any target module FAILS the pin rather than quietly unpinning it.

    A new raw access fails this test on purpose: route it through
    ``guard_for(...)`` / the guard's methods, or move the allowlist
    consciously, with a reason in the diff.
    """

    #: (module, expected basename) — basename mismatch means the file was
    #: renamed/moved and the pin's scan silently shrank.
    TARGET_MODULES = (
        ("qwen3_tts.server.app_generation", "app_generation.py"),
        ("qwen3_tts.server.websocket", "websocket.py"),
        ("qwen3_tts.server.app_models", "app_models.py"),
        ("qwen3_tts.server.app", "app.py"),
        ("qwen3_tts.server.app_lifespan", "app_lifespan.py"),
    )

    #: The only permitted raw ``generation_state`` references in production
    #: code: app_lifespan's init block, which CREATES the dict the guard
    #: then owns. Everything else must go through the guard.
    ALLOWLIST = (
        ("app_lifespan.py", "app.state.generation_state = {"),
    )

    def _flagged_occurrences(self):
        """(scanned basenames, [(basename, lineno, stripped line), ...])."""
        import importlib
        import tokenize
        from pathlib import Path

        scanned = []
        occurrences = []
        for module_name, basename in self.TARGET_MODULES:
            module = importlib.import_module(module_name)
            path = Path(module.__file__).resolve()
            self.assertEqual(
                path.name,
                basename,
                f"{module_name} resolved to {path.name}, not {basename}: the "
                "file was renamed or moved — update TARGET_MODULES loudly, "
                "never let the pin silently shrink its scan",
            )
            self.assertTrue(
                path.is_file(),
                f"pin target {basename} does not exist at {path}: the scan "
                "would silently pass over a missing file",
            )
            scanned.append(basename)
            with open(path, "rb") as handle:
                for token in tokenize.tokenize(handle.readline):
                    is_generation_state_name = (
                        token.type == tokenize.NAME
                        and token.string == "generation_state"
                    )
                    if is_generation_state_name:
                        occurrences.append(
                            (basename, token.start[0], token.line.strip())
                        )
        return scanned, occurrences

    def test_scan_covers_every_target_file(self):
        scanned, _occurrences = self._flagged_occurrences()
        self.assertEqual(scanned, [basename for _, basename in self.TARGET_MODULES])

    def test_no_raw_generation_state_access_outside_the_allowlist(self):
        _scanned, occurrences = self._flagged_occurrences()
        unlisted = [
            (basename, lineno, line)
            for basename, lineno, line in occurrences
            if not any(
                basename == allowed_file and needle in line
                for allowed_file, needle in self.ALLOWLIST
            )
        ]
        self.assertEqual(
            unlisted,
            [],
            "raw generation_state access(es) outside the guard: route them "
            "through guard_for()/GenerationStateGuard, or extend "
            "ALLOWLIST consciously with a reason in the diff",
        )

    def test_allowlist_entries_are_all_live(self):
        _scanned, occurrences = self._flagged_occurrences()
        for allowed_file, needle in self.ALLOWLIST:
            matches = [
                (basename, lineno, line)
                for basename, lineno, line in occurrences
                if basename == allowed_file and needle in line
            ]
            self.assertTrue(
                matches,
                f"allowlist entry ({allowed_file!r}, {needle!r}) matched "
                "nothing: the init block moved or was reworded — update the "
                "allowlist consciously, never leave a dead entry widening it",
            )

    def test_raw_occurrence_count_equals_the_allowlisted_count(self):
        """Anti-vacuity: the allowlist must be the WHOLE raw surface, so the
        total occurrence count must equal the allowlisted one. A second raw
        access that happens to repeat an allowlisted line shape still fails
        here."""
        _scanned, occurrences = self._flagged_occurrences()
        allowlisted = [
            (basename, lineno, line)
            for basename, lineno, line in occurrences
            if any(
                basename == allowed_file and needle in line
                for allowed_file, needle in self.ALLOWLIST
            )
        ]
        self.assertEqual(
            len(occurrences),
            len(allowlisted),
            f"unexpected raw generation_state occurrence(s): {occurrences!r} "
            f"vs allowlisted {allowlisted!r}",
        )
        self.assertEqual(
            len(occurrences),
            len(self.ALLOWLIST),
            "the raw generation_state surface changed size: one raw "
            "occurrence per ALLOWLIST entry is the whole permitted set",
        )
