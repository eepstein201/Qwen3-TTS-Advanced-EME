"""Thread-safety guard for the server's ``generation_state`` dict.

Why this module exists
----------------------
``app.state.generation_state`` is a plain dict shared by the event loop AND
worker threads (streaming progress callbacks, inference offload threads,
cancel watchers). It is guarded — where at all — by ``generation_lock``, an
``asyncio.Lock``: an event-loop lock provides zero mutual exclusion against
plain threads, and it does not compose with other locks either, so several
sites mutate the dict with nothing held at all.

This guard owns a ``threading.Lock`` instead. A ``threading.Lock`` is the
only primitive here that actually excludes: it is honored by worker threads
and by event-loop callbacks alike (a loop callback running ``with`` it simply
blocks the loop, which is the correct behavior for a dict update measured in
microseconds), and it nests safely under the existing asyncio orchestration
without any deadlock risk of the ``lock-across-await`` kind, because no
method ever awaits while holding it.

Late binding
------------
The guard never captures the dict at construction: ``_state`` is a property
that resolves ``app_state.generation_state`` at EVERY call. Tests (and any
future re-init) replace the dict object with a fresh one; a captured
reference would silently keep mutating the orphaned dict while everyone else
reads the new one. Resolving per call keeps the guard correct across that
swap, and keeps a single guard instance valid for the process lifetime.

Cancel attribution (issue #237 / Step 1A)
------------------------------------------
A cancel is no longer a bare boolean: ``set_cancelled(target_id)`` records
WHICH generation it addresses in ``cancel_target_id``, and
``is_cancelled_for(generation_id)`` is the target-matched read that only
that generation should act on. ``begin()`` erases a stale cancel (one
targeted at some OTHER, non-pending generation) but preserves one targeted
at itself or at a still-pending sibling (see ``begin()``'s docstring for the
three-case rule) — this is what fixes the two cancel-loss races: a cancel
addressed to the running batch survives that batch's own next ``begin()``,
and a cancel addressed to a batch that has been minted but not yet begun
(tracked in the pending registry) survives an unrelated concurrent
``begin()`` too. Every production call site is now target-aware
(``is_cancelled_for``); the untargeted ``is_cancelled()``/``clear_cancelled()``
pair from Task 1 has been removed (Controller Ruling A) since an untargeted
read/clear is precisely the erase that issue #237 is about.

Stdlib only — no FastAPI, engine, torch, or mlx imports, so the module can
be imported at module scope anywhere (the lazy-heavy-import rule is
respected by construction).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from typing import Any

__all__ = ["GenerationStateGuard", "guard_for"]

# Serializes guard_for()'s lazy check-then-set so simultaneous first-touches
# cannot each construct a guard and last-write the attribute (which would
# hand different threads guards backed by different threading.Locks).
# Production never hits this path — app_lifespan provisions eagerly — so
# this lock only serves fake/test states touched from several threads.
_CONSTRUCTION_LOCK = threading.Lock()

# The eleven canonical keys of ``generation_state`` (see the idle shape in
# ``app_lifespan.lifespan``). ``snapshot()`` defaults to exactly these, so
# callers never receive a dict with surprise keys.
_CANONICAL_KEYS: tuple[str, ...] = (
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
)

# The idle shape every key returns to on ``reset_if_owner()``. ``dict.update``
# copies the values out; the constant itself is never handed to callers.
_IDLE_STATE_VALUES: dict[str, Any] = {
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


def _read_state_key(state: dict[str, Any], name: str) -> Any:
    """Read ``name`` off ``state``, defaulting ``cancel_target_id`` to None.

    ``cancel_target_id`` is the 11th key, added in issue #237 / Step 1A.
    Several test modules (and any other caller holding an older
    hand-rolled ten-key ``generation_state`` shape) predate it; a hard
    subscript on that one key would ``KeyError`` for them. Every other
    canonical key is still read with a hard subscript — those have always
    been part of the contract and a missing one is a real bug, not a
    version-skew shape to tolerate.
    """
    if name == "cancel_target_id":
        return state.get(name)
    return state[name]


class GenerationStateGuard:
    """Serialize every mutation of ``app_state.generation_state``.

    One guard per app state, one ``threading.Lock`` per guard. All methods
    are atomic: they take the lock, resolve the live dict ONCE, apply the
    whole update, and release. No method blocks on I/O or awaits, so the
    lock is never held across a yield point.
    """

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state
        self._lock = threading.Lock()
        # Generation ids minted (registered) but not yet ``begin()``-ed.
        # Guards the second cancel-loss race (#237 / Step 1A): a cancel
        # addressed to one of these ids must survive an unrelated
        # concurrent begin(). Accessed only under ``self._lock`` — no
        # separate lock for this set.
        self._pending_ids: set[str] = set()

    @property
    def _state(self) -> dict[str, Any]:
        """The live dict, resolved at each call — never captured.

        Late binding: ``app_lifespan`` (and tests) may replace
        ``app_state.generation_state`` with a new dict object; a guard that
        had captured the old reference would mutate a dict nobody reads.
        """
        return self._app_state.generation_state

    def begin(
        self,
        generation_id: str,
        *,
        mode: str = "",
        text_length: int = 0,
        start_time: float | None = None,
        batch_index: int = 0,
        batch_total: int = 0,
    ) -> None:
        """Mark a generation active, in one atomic update.

        Sets ``active=True``, ``generation_id``, ``start_time`` (defaulting
        to now), and the four provided fields. Chunk counters are left to
        ``update_progress``.

        Cancel handling (issue #237 / Step 1A) — four cases, in order:

        - ``cancel_target_id == generation_id``: preserved. A cancel
          addressed to THIS batch must survive its own next ``begin()`` so
          every later item in the batch still honors it.
        - ``cancel_target_id`` names a still-PENDING sibling (minted via
          ``register_pending`` but not yet begun): preserved. Otherwise an
          ordinary concurrent batch's ``begin()`` would erase a cancel that
          sibling never had the chance to honor.
        - ``cancel_target_id`` names the state's CURRENT owner (fix round 1,
          Important 1) — i.e. ``state["generation_id"]`` at the moment this
          runs, before it is overwritten with the incoming id: preserved.
          A cancel addressed to a live, already-begun batch must survive a
          concurrent, DIFFERENT generation's ``begin()`` too, not just its
          own; once the true owner's own ``begin()`` or ``reset_if_owner``
          has run, ``generation_id`` no longer names it, so this cannot
          over-preserve a target that has genuinely gone stale.
        - ``cancel_target_id`` is set, non-matching, NOT pending, and NOT
          the current owner (genuinely stale): erased — this is the
          status-surface hygiene the old blanket re-clear gave us, kept for
          the one case where no live generation can ever honor the cancel.
        - ``cancel_target_id is None``: nothing to erase; ``cancelled`` is
          left untouched.

        Also consumes *generation_id*'s own pending registration, if any —
        a batch that registered at mint time is no longer "pending" once it
        begins.
        """
        with self._lock:
            state = self._state
            target = state.get("cancel_target_id")
            if (
                target is not None
                and target != generation_id
                and target != state.get("generation_id")
                and target not in self._pending_ids
            ):
                state["cancelled"] = False
                state["cancel_target_id"] = None
            self._pop_pending_target(generation_id)
            state["active"] = True
            state["generation_id"] = generation_id
            state["start_time"] = time.time() if start_time is None else start_time
            state["mode"] = mode
            state["text_length"] = text_length
            state["batch_index"] = batch_index
            state["batch_total"] = batch_total

    def update_progress(self, chunk_index: int, chunk_total: int) -> None:
        """Record chunk progress; safe to call from any worker thread."""
        with self._lock:
            state = self._state
            state["chunk_index"] = chunk_index
            state["chunk_total"] = chunk_total

    def set_cancelled(self, target_id: str | None = None) -> None:
        """Set ``cancelled=True`` and ``cancel_target_id=target_id``, atomically.

        Every production call site (Task 2) now passes an explicit target —
        the currently-active generation id or a pending one. A cancel with
        ``cancel_target_id is None`` is honored by nobody; the default only
        exists for direct unit-test convenience.
        """
        with self._lock:
            state = self._state
            state["cancelled"] = True
            state["cancel_target_id"] = target_id

    def is_cancelled_for(self, generation_id: str) -> bool:
        """True only when a cancel is set AND addressed to *generation_id*.

        Reads ``cancel_target_id`` via ``.get`` (never a hard subscript):
        several test modules hand-roll the old ten-key state shape and must
        not ``KeyError`` here.
        """
        with self._lock:
            state = self._state
            return (
                bool(state["cancelled"])
                and state.get("cancel_target_id") == generation_id
            )

    def register_pending(self, generation_id: str) -> None:
        """Mark *generation_id* minted-but-not-begun, atomically.

        Part of the pending registry (issue #237 / Step 1A): a cancel
        addressed to a generation that has registered here but not yet
        called ``begin()`` must survive an unrelated concurrent ``begin()``.
        """
        with self._lock:
            self._pending_ids.add(generation_id)

    def deregister_pending(self, generation_id: str) -> None:
        """Remove *generation_id* from the pending registry, atomically.

        Idempotent (``set.discard``, never ``set.remove``): deregistering an
        id that was never registered, or already removed, is a no-op.

        Also clears a cancel addressed to *generation_id*: once it is
        deregistered pending, nobody will ever check ``is_cancelled_for``
        against it again, so its cancel can never be honored and must not
        linger on the public ``/generation-status`` surface.
        """
        with self._lock:
            state = self._state
            self._pending_ids.discard(generation_id)
            if state.get("cancel_target_id") == generation_id:
                state["cancelled"] = False
                state["cancel_target_id"] = None

    def peek_pending_target(self) -> str | None:
        """Return an id to target a cancel at, or ``None`` if none is pending.

        Answers "is anything pending, and what should I target?" for
        ``/cancel-generation``'s inactive branch (issue #237 / Step 1A,
        Window 2): a cancel arriving before any generation is active should
        still latch onto a pending id, deterministically chosen (see the
        tie-break rule below).

        Does not mutate the registry and does not hand out the mutable set
        itself — a snapshot decision, consistent with every other read here.

        When more than one id is pending, the lexicographically smallest is
        returned. Pending ids are opaque ``uuid4`` hex prefixes with no
        meaningful ordering (insertion order is not tracked — ``_pending_ids``
        is a plain set), so this is an arbitrary but fully deterministic
        tie-break: the same call against the same registry always returns
        the same id.
        """
        with self._lock:
            if not self._pending_ids:
                return None
            return min(self._pending_ids)

    def _pop_pending_target(self, generation_id: str) -> bool:
        """Discard *generation_id* from the pending registry; True if present.

        Private and unlocked: the only caller is ``begin()``, which already
        holds ``self._lock`` (the lock is not reentrant); direct test use is
        single-threaded and needs no locking of its own.
        """
        was_pending = generation_id in self._pending_ids
        self._pending_ids.discard(generation_id)
        return was_pending

    def reset_if_owner(self, generation_id: str) -> bool:
        """Reset to the idle shape if ``generation_id`` still owns the state.

        Returns True and resets ALL eleven keys when the state's
        ``generation_id`` matches; returns False and touches nothing when it
        does not, so a generation that lost ownership (superseded by a newer
        one) can never clobber the new owner's progress.
        """
        with self._lock:
            state = self._state
            if state["generation_id"] != generation_id:
                return False
            state.update(_IDLE_STATE_VALUES)
            return True

    def snapshot(self, keys: Iterable[str] | None = None) -> dict[str, Any]:
        """Return a NEW dict of the requested keys (default: all eleven).

        Callers never receive the live dict, so a snapshot can be read
        without the lock and cannot be used to bypass the guard. Within one
        snapshot the keys are mutually consistent (read in one locked step).

        ``keys`` is materialized BEFORE the lock is taken: it is
        caller-supplied and may be a generator, and resuming caller code
        while holding a non-reentrant lock would deadlock any generator that
        touches the guard.

        ``cancel_target_id`` is read via ``.get`` (defaulting to ``None``),
        never a hard subscript: several test modules hand-roll the old
        ten-key state shape and pass it through real handlers that call this
        method with no ``keys`` argument — they must not ``KeyError`` here.
        Every other key is still a hard subscript, unchanged.
        """
        requested = list(keys) if keys is not None else None
        with self._lock:
            state = self._state
            names = _CANONICAL_KEYS if requested is None else requested
            return {name: _read_state_key(state, name) for name in names}


def guard_for(app_state: Any) -> GenerationStateGuard:
    """Return the app's guard, constructing-and-caching it on first touch.

    Production: ``app_lifespan`` provisions it eagerly (see the wiring in
    ``lifespan()``). Fake/test states that never provisioned one get a
    transient guard on first handler touch — late binding of
    ``app_state.generation_state`` keeps both paths correct.

    Idempotent, including under concurrent first-touch: the check-then-set
    runs inside the module-level ``_CONSTRUCTION_LOCK``, so the first guard
    cached on the state wins and every contender receives the SAME object.
    Unserialized, two simultaneous first-touches could each construct a
    guard and last-write the attribute — leaving callers holding guards
    backed by DIFFERENT ``threading.Lock`` s, i.e. zero mutual exclusion.
    Production never hits the lazy path (lifespan provisions eagerly); the
    construction lock covers fake/test states touched from multiple threads.
    """
    with _CONSTRUCTION_LOCK:
        guard = getattr(app_state, "generation_state_guard", None)
        if guard is None:
            guard = GenerationStateGuard(app_state)
            app_state.generation_state_guard = guard
    return guard
