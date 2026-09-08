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

Deliberately preserved semantics
--------------------------------
``begin()`` re-clears ``cancelled``, exactly as every production begin does
today. That re-clear is itself a small race (a stale cancel flag from a
concurrent request is erased rather than honored) and is tracked as issue
#237 / Step 1A. Task 1 is atomicity only: no semantic change, so the
re-clear is preserved and pinned by ``tests/test_generation_state_guard.py``
until that step moves it.

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

# The ten canonical keys of ``generation_state`` (see the idle shape in
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
}


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

        Sets ``active=True``, ``generation_id``, ``cancelled=False`` (the
        re-clear is deliberate — see the module docstring: that erase is
        issue #237 / Step 1A's race, preserved here unchanged), ``start_time``
        (defaulting to now), and the four provided fields. Chunk counters are
        left to ``update_progress``.
        """
        with self._lock:
            state = self._state
            state["active"] = True
            state["generation_id"] = generation_id
            state["cancelled"] = False
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

    def clear_cancelled(self) -> None:
        """Set ``cancelled=False``, atomically."""
        with self._lock:
            self._state["cancelled"] = False

    def set_cancelled(self) -> None:
        """Set ``cancelled=True``, atomically."""
        with self._lock:
            self._state["cancelled"] = True

    def is_cancelled(self) -> bool:
        """Read ``cancelled`` under the lock (no torn/stale reads)."""
        with self._lock:
            return bool(self._state["cancelled"])

    def reset_if_owner(self, generation_id: str) -> bool:
        """Reset to the idle shape if ``generation_id`` still owns the state.

        Returns True and resets ALL ten keys when the state's
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
        """Return a NEW dict of the requested keys (default: all ten).

        Callers never receive the live dict, so a snapshot can be read
        without the lock and cannot be used to bypass the guard. Within one
        snapshot the keys are mutually consistent (read in one locked step).
        """
        with self._lock:
            state = self._state
            if keys is None:
                return {name: state[name] for name in _CANONICAL_KEYS}
            return {name: state[name] for name in keys}


def guard_for(app_state: Any) -> GenerationStateGuard:
    """Return the app's guard, constructing-and-caching it on first touch.

    Production: ``app_lifespan`` provisions it eagerly (see the wiring in
    ``lifespan()``). Fake/test states that never provisioned one get a
    transient guard on first handler touch — late binding of
    ``app_state.generation_state`` keeps both paths correct. Idempotent: the
    first guard cached on the state wins.
    """
    guard = getattr(app_state, "generation_state_guard", None)
    if guard is None:
        guard = GenerationStateGuard(app_state)
        app_state.generation_state_guard = guard
    return guard
