#!/usr/bin/env python3
"""Issue #214 item 1 -- torch auto-create-from-.wav must be serialized.

Torch-backend only. ``load_voice_prompt`` -> ``_load_voice_prompt_torch`` ->
``_auto_create_pt_from_wav`` (``qwen3_tts/core/engine/voice_prompt.py``) calls
``load_model("clone")`` **and** ``create_voice_prompt(...)`` -- real GPU
inference -- as a side effect of "just loading a prompt" when the .pt is
missing or corrupt and a sibling .wav exists. All three server call sites
(``app_generation.py`` x2, ``websocket.py`` x1) invoked this via
``asyncio.to_thread(load_voice_prompt, ...)`` strictly BEFORE acquiring
``inference_lock`` -- the same unsynchronized-concurrent-inference shape
PRs #211/#212/#213/#230 already closed everywhere else
(ml-explore/mlx#3078, Blaizzy/mlx-audio#638/#733).

MLX is unaffected: ``load_voice_prompt_mlx`` never creates -- it only builds
a ``{"ref_audio", "ref_text"}`` dict from disk. The fix must be a provable
no-op there.

The trap: ``load_model()`` (``model_loader.py``) has NO memoization -- every
call is a full multi-minute weight construction. A naive
"pre-load-then-lock" fix discards the pre-built model and reconstructs it a
second time INSIDE the lock, reintroducing the exact starvation the #212
split exists to prevent. The fix must forward the already-built model via
``clone_model=`` (mirrors ``migrate_orphan_mlx_prompts(clone_model=None)``
already in this file).

Contract pinned by these tests:

  * ``load_voice_prompt_serialized`` (``qwen3_tts/server/prompt_loading.py``)
    runs an unlocked probe (``allow_create=False``); only when a create is
    actually needed does it build/reuse the clone model OUTSIDE the lock and
    re-enter ``load_voice_prompt(..., allow_create=True, clone_model=...)``
    UNDER ``inference_lock`` as a leaf acquisition.
  * clone-model construction (when needed) runs UNLOCKED; the create itself
    runs LOCKED.
  * the pre-built model is forwarded via ``clone_model=`` -- never dropped,
    which would silently reconstruct it under the lock.
  * an already-loaded clone model (``state.models["clone"]``) is reused;
    ``load_model`` is never called.
  * MLX is a provable no-op: ``inference_lock`` is never acquired and
    ``load_model`` is never called.
  * ``allow_create=False`` preserves today's contract byte-for-byte except
    for the new ``VoicePromptCreateRequired`` signal: missing .pt + no .wav
    -> ``None``; corrupt .pt + no .wav -> re-raises the original error;
    missing/corrupt .pt + .wav present -> raises
    ``VoicePromptCreateRequired``.
  * ``load_voice_prompt(pf, allow_create=True)`` never raises
    ``VoicePromptCreateRequired``.
  * two callers racing the same missing prompt converge on exactly one
    create (via ``_load_voice_prompt_torch``'s top-of-function cache
    re-check, once the two locked calls are serialized by
    ``inference_lock``).

No GPU, models, or running server required (except the convergence test,
which drives the real engine cache with the deep torch/create seams mocked)
-- locks are real asyncio.Locks, inference is patched at the engine facade
(the handler imports function-locally, so the patched attribute is what it
resolves at call time).

Run: pytest tests/test_issue214_prompt_create_serialization.py -v --tb=short
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

try:
    import fastapi  # noqa: F401

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_skip = unittest.skipUnless(HAS_DEPS, "requires fastapi")
_skip_torch = unittest.skipUnless(HAS_TORCH, "requires torch")


def _make_state(clone_model=None):
    """Build an app.state stand-in with a REAL asyncio lock.

    The lock must be real -- tests assert on ``lock.locked()`` observed
    from inside the patched engine calls, so a MagicMock lock would make
    every assertion hollow.
    """
    state = MagicMock()
    state.models = {"clone": clone_model}
    state.inference_lock = asyncio.Lock()
    return state


@_skip
class TestLoadVoicePromptSerializedOrdering(unittest.TestCase):
    """Lock/ordering + call-arg contract of load_voice_prompt_serialized."""

    def test_load_model_unlocked_then_create_locked(self):
        """Clone-model construction (when needed) must run UNLOCKED; the
        create must run LOCKED. Recorded as one unified events list so
        ordering and lock state are asserted separately."""
        from qwen3_tts.core.engine import VoicePromptCreateRequired
        from qwen3_tts.server.prompt_loading import load_voice_prompt_serialized

        state = _make_state(clone_model=None)
        events = []
        built_model = object()

        def _load_prompt(prompt_file, *, allow_create, clone_model=None):
            if not allow_create:
                raise VoicePromptCreateRequired(prompt_file)
            events.append(
                {"kind": "create", "lock_held": state.inference_lock.locked()}
            )
            return "created-prompt"

        def _load_model(model_type, **kwargs):
            events.append(
                {"kind": "load_model", "lock_held": state.inference_lock.locked()}
            )
            return built_model

        with (
            patch(
                "qwen3_tts.core.engine.load_voice_prompt", side_effect=_load_prompt
            ) as mock_load_prompt,
            patch(
                "qwen3_tts.core.engine.model_loader.load_model",
                side_effect=_load_model,
            ),
        ):
            result = asyncio.run(
                load_voice_prompt_serialized(state, "missing.pt")
            )

        self.assertEqual(result, "created-prompt")
        self.assertEqual(
            [e["kind"] for e in events],
            ["load_model", "create"],
            "clone-model construction must precede the create",
        )
        self.assertEqual(
            [e["lock_held"] for e in events],
            [False, True],
            "clone-model construction must run UNLOCKED (minutes of weight "
            "construction must not starve /generate, per the #212 split); "
            "the create inference must run LOCKED",
        )
        self.assertEqual(
            [c.kwargs["allow_create"] for c in mock_load_prompt.call_args_list],
            [False, True],
            "the probe call must pass allow_create=False, the locked call "
            "allow_create=True",
        )
        self.assertIsNotNone(
            mock_load_prompt.call_args_list[1].kwargs["clone_model"],
            "the pre-loaded model must be forwarded, or the locked call "
            "reconstructs it under the lock",
        )
        self.assertIs(
            mock_load_prompt.call_args_list[1].kwargs["clone_model"],
            built_model,
            "the exact model built outside the lock must be forwarded in, "
            "not a fresh/different one",
        )

    def test_already_loaded_clone_model_skips_load_model(self):
        """When state.models['clone'] is already populated, load_model must
        never be called -- the existing model is reused."""
        from qwen3_tts.core.engine import VoicePromptCreateRequired
        from qwen3_tts.server.prompt_loading import load_voice_prompt_serialized

        existing_model = object()
        state = _make_state(clone_model=existing_model)
        events = []

        def _load_prompt(prompt_file, *, allow_create, clone_model=None):
            if not allow_create:
                raise VoicePromptCreateRequired(prompt_file)
            events.append(
                {
                    "lock_held": state.inference_lock.locked(),
                    "clone_model": clone_model,
                }
            )
            return "created-prompt"

        with (
            patch(
                "qwen3_tts.core.engine.load_voice_prompt", side_effect=_load_prompt
            ),
            patch(
                "qwen3_tts.core.engine.model_loader.load_model"
            ) as mock_load_model,
        ):
            result = asyncio.run(
                load_voice_prompt_serialized(state, "missing.pt")
            )

        self.assertEqual(result, "created-prompt")
        mock_load_model.assert_not_called()
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["lock_held"])
        self.assertIs(events[0]["clone_model"], existing_model)

    def test_fast_path_never_touches_lock_or_load_model(self):
        """When the unlocked probe succeeds (no create needed),
        inference_lock must never be acquired and load_model must never be
        called."""
        from qwen3_tts.server.prompt_loading import load_voice_prompt_serialized

        state = _make_state(clone_model=None)

        with (
            patch(
                "qwen3_tts.core.engine.load_voice_prompt",
                return_value="cached-prompt",
            ) as mock_load_prompt,
            patch(
                "qwen3_tts.core.engine.model_loader.load_model"
            ) as mock_load_model,
        ):
            result = asyncio.run(
                load_voice_prompt_serialized(state, "existing.pt")
            )

        self.assertEqual(result, "cached-prompt")
        mock_load_model.assert_not_called()
        self.assertFalse(state.inference_lock.locked())
        mock_load_prompt.assert_called_once_with(
            "existing.pt", allow_create=False
        )

    def test_mlx_is_a_provable_no_op(self):
        """MLX never creates -- the probe call must succeed without ever
        touching inference_lock or load_model."""
        from qwen3_tts.server.prompt_loading import load_voice_prompt_serialized

        state = _make_state(clone_model=None)
        lock_states_seen = []

        def _mlx_probe(prompt_file, *, allow_create, clone_model=None):
            # load_voice_prompt_mlx never raises VoicePromptCreateRequired.
            lock_states_seen.append(state.inference_lock.locked())
            return {"ref_audio": "/tmp/x.wav", "ref_text": "hello"}

        with (
            patch(
                "qwen3_tts.core.engine.load_voice_prompt", side_effect=_mlx_probe
            ),
            patch(
                "qwen3_tts.core.engine.model_loader.load_model"
            ) as mock_load_model,
        ):
            result = asyncio.run(load_voice_prompt_serialized(state, "voice.wav"))

        self.assertEqual(result, {"ref_audio": "/tmp/x.wav", "ref_text": "hello"})
        mock_load_model.assert_not_called()
        self.assertEqual(lock_states_seen, [False])
        self.assertFalse(state.inference_lock.locked())


class _AcquireSignallingLock:
    """A real ``asyncio.Lock`` that announces each acquire ATTEMPT.

    ``acquire_attempted`` is set BEFORE the underlying acquire is awaited, so
    a test that already holds the lock can observe "the coroutine under test
    is now parked in its capture->acquire window" without polling private
    asyncio internals or sleeping. Scheduling semantics are delegated to the
    wrapped real lock -- this observes, it does not fake mutual exclusion (a
    MagicMock lock would make every ``locked()`` assertion hollow).
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self.acquire_calls = 0
        self.acquire_attempted = asyncio.Event()

    async def acquire(self):
        self.acquire_calls += 1
        self.acquire_attempted.set()
        return await self._lock.acquire()

    def release(self):
        self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc):
        self.release()
        return False

    def locked(self):
        return self._lock.locked()


@_skip
class TestLoadVoicePromptSerializedSlotReRead(unittest.TestCase):
    """The FIFTH capture->acquire window (T5 sibling).

    ``load_voice_prompt_serialized`` captures ``state.models['clone']`` into a
    local and then waits on ``inference_lock`` -- and that wait is a whole
    queued generation, the widest such window in the server. The locked call
    runs REAL torch create inference, so an unload->RELOAD landing in the wait
    must not build the prompt on the orphaned pre-unload object
    (``unload_model_cleanup`` is gc.collect + a cache flush; the orphan keeps
    working, so the defect is silent), and an unload alone must surface the
    same retryable 503 the generation paths raise.

    The window is exercised REALISTICALLY: the test holds ``inference_lock``,
    launches the coroutine so it parks on the acquire, mutates the slot while
    it is parked, then releases. No pre-lock seam is abused to fake it.
    """

    @staticmethod
    def _probe_raises_then_records(state, events):
        """Return a ``load_voice_prompt`` double: probe signals "create
        required", the locked call records the forwarded model."""
        from qwen3_tts.core.engine import VoicePromptCreateRequired

        def _load_prompt(prompt_file, *, allow_create, clone_model=None):
            if not allow_create:
                raise VoicePromptCreateRequired(prompt_file)
            events.append(
                {
                    "clone_model": clone_model,
                    "lock_held": state.inference_lock.locked(),
                }
            )
            return "created-prompt"

        return _load_prompt

    def _run_with_slot_mutated_while_parked(self, state, mutate, load_model=None):
        """Run ``load_voice_prompt_serialized`` while the test holds the lock,
        applying *mutate* once the coroutine is parked on the acquire.

        Returns ``(result_or_exception, events)``; the exception is returned
        rather than raised so callers assert on it explicitly.
        """
        from qwen3_tts.server.prompt_loading import load_voice_prompt_serialized

        events = []
        load_prompt = self._probe_raises_then_records(state, events)

        async def _scenario():
            with (
                patch(
                    "qwen3_tts.core.engine.load_voice_prompt",
                    side_effect=load_prompt,
                ),
                patch(
                    "qwen3_tts.core.engine.model_loader.load_model",
                    side_effect=load_model or (lambda *a, **k: object()),
                ),
            ):
                async with state.inference_lock:
                    # Ours; watch for the coroutine's own attempt below.
                    state.inference_lock.acquire_attempted.clear()
                    task = asyncio.ensure_future(
                        load_voice_prompt_serialized(state, "missing.pt")
                    )
                    await asyncio.wait_for(
                        state.inference_lock.acquire_attempted.wait(), timeout=10
                    )
                    # The coroutine is now parked INSIDE acquire: the capture
                    # already happened, the create has not.
                    self.assertEqual(
                        events, [], "the create ran before the lock was free"
                    )
                    mutate()
                try:
                    return await asyncio.wait_for(task, timeout=10)
                except Exception as exc:  # returned, not swallowed
                    return exc

        return asyncio.run(_scenario()), events

    def test_create_runs_on_the_model_reloaded_while_parked(self):
        """unload->RELOAD in the window: the slot is non-None again, so only
        REBINDING the under-lock re-read keeps the create off the orphan."""
        state = _make_state(clone_model=MagicMock(name="original-clone"))
        old_model = state.models["clone"]
        reloaded = MagicMock(name="reloaded-clone")
        state.inference_lock = _AcquireSignallingLock()

        result, events = self._run_with_slot_mutated_while_parked(
            state, lambda: state.models.__setitem__("clone", reloaded)
        )

        self.assertEqual(result, "created-prompt", result)
        self.assertEqual(len(events), 1, f"expected one create, got {events!r}")
        self.assertTrue(events[0]["lock_held"], "the create must run LOCKED")
        self.assertIs(
            events[0]["clone_model"],
            reloaded,
            "the prompt was created on the pre-unload orphan -- "
            "load_voice_prompt_serialized never re-reads the clone slot under "
            "inference_lock",
        )
        self.assertIsNot(
            events[0]["clone_model"],
            old_model,
            "the orphaned capture reached the create inference",
        )
        self.assertGreaterEqual(
            state.inference_lock.acquire_calls,
            2,
            "the coroutine never acquired inference_lock",
        )

    def test_unload_while_parked_raises_retryable_503(self):
        """The null half: the slot was populated at capture time and is gone
        under the lock -- a real unload. Same classified, retryable 503 the
        generation paths raise, and the create must never run."""
        from fastapi import HTTPException

        state = _make_state(clone_model=MagicMock(name="original-clone"))
        state.inference_lock = _AcquireSignallingLock()

        result, events = self._run_with_slot_mutated_while_parked(
            state, lambda: state.models.__setitem__("clone", None)
        )

        self.assertIsInstance(
            result,
            HTTPException,
            f"expected a retryable 503, got result {result!r}",
        )
        self.assertEqual(result.status_code, 503)
        self.assertIsInstance(result.detail, dict, f"detail: {result.detail!r}")
        self.assertEqual(result.detail.get("error"), "model_unloaded")
        self.assertEqual(result.detail.get("recovery"), "retry")
        self.assertEqual(
            events,
            [],
            "create inference ran against the orphaned model before the 503",
        )

    def test_locally_built_model_survives_an_empty_slot_under_the_lock(self):
        """Behavior-preservation pin for the FALLBACK branch.

        ``load_model`` (core/engine/model_loader.py) returns the model and
        never writes ``state.models`` -- so when the slot was empty at capture
        time it is STILL empty under the lock, by design. An unconditional
        under-lock guard would 503 a path that works today; this pins that it
        does not, and that the locally built model is the one forwarded.
        """
        state = _make_state(clone_model=None)
        built = MagicMock(name="locally-built-clone")
        state.inference_lock = _AcquireSignallingLock()

        result, events = self._run_with_slot_mutated_while_parked(
            state, lambda: None, load_model=lambda *a, **k: built
        )

        self.assertEqual(result, "created-prompt", result)
        self.assertEqual(len(events), 1, f"expected one create, got {events!r}")
        self.assertIs(
            events[0]["clone_model"],
            built,
            "the locally built model must still reach the create when the "
            "slot is legitimately empty -- 503-ing here is a regression",
        )
        self.assertIsNone(
            state.models["clone"],
            "load_model must not be assumed to populate the slot",
        )

    def test_fallback_prefers_a_slot_that_filled_while_parked(self):
        """Fallback branch, other direction: a /load-model that finished
        during the wait published the LIVE model into the slot, so the create
        must use it rather than the private copy this call built."""
        state = _make_state(clone_model=None)
        built = MagicMock(name="locally-built-clone")
        published = MagicMock(name="published-clone")
        state.inference_lock = _AcquireSignallingLock()

        result, events = self._run_with_slot_mutated_while_parked(
            state,
            lambda: state.models.__setitem__("clone", published),
            load_model=lambda *a, **k: built,
        )

        self.assertEqual(result, "created-prompt", result)
        self.assertEqual(len(events), 1, f"expected one create, got {events!r}")
        self.assertIs(
            events[0]["clone_model"],
            published,
            "the slot published while this call waited is the live model -- "
            "the create must re-read it, not use the pre-lock capture",
        )


@_skip
@_skip_torch
class TestEngineAllowCreateContract(unittest.TestCase):
    """Engine-level: allow_create=False behavior contract, no server needed.

    These five tests drive the REAL ``_load_voice_prompt_torch``, whose body
    does a function-level ``import torch`` (voice_prompt.py) for the cache
    type. Torchless environments (CI ``.[test]``, .venv-lock) cannot satisfy
    it, so the whole class is gated on torch like the #192 sibling module.
    """

    def setUp(self):
        from qwen3_tts.core.engine.voice_prompt import (
            _torch_prompt_cache,
            _torch_prompt_cache_lock,
        )

        with _torch_prompt_cache_lock:
            _torch_prompt_cache.clear()

    @patch("qwen3_tts.core.engine.voice_prompt.get_backend", return_value="torch")
    def test_missing_pt_no_wav_returns_none(self, _mock_be):
        from qwen3_tts.core.engine.voice_prompt import _load_voice_prompt_torch

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch(
                    "qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", tmpdir
                ),
                patch(
                    "qwen3_tts.core.engine.voice_prompt.safe_path_join",
                    side_effect=lambda d, f: os.path.join(d, f),
                ),
            ):
                result = _load_voice_prompt_torch(
                    "nowhere.pt", allow_create=False
                )

        self.assertIsNone(result)

    @patch("qwen3_tts.core.engine.voice_prompt.get_backend", return_value="torch")
    def test_missing_pt_with_wav_raises_create_required(self, _mock_be):
        from qwen3_tts.core.engine import VoicePromptCreateRequired
        from qwen3_tts.core.engine.voice_prompt import _load_voice_prompt_torch

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "voice.wav"), "wb") as f:
                f.write(b"fake-wav-bytes")
            with (
                patch(
                    "qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", tmpdir
                ),
                patch(
                    "qwen3_tts.core.engine.voice_prompt.safe_path_join",
                    side_effect=lambda d, f: os.path.join(d, f),
                ),
            ):
                with self.assertRaises(VoicePromptCreateRequired) as ctx:
                    _load_voice_prompt_torch("voice.pt", allow_create=False)

        self.assertEqual(ctx.exception.prompt_file, "voice.pt")

    @patch("qwen3_tts.core.engine.voice_prompt.get_backend", return_value="torch")
    def test_corrupt_pt_no_wav_reraises_original_error(self, _mock_be):
        from qwen3_tts.core.engine.voice_prompt import _load_voice_prompt_torch

        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = os.path.join(tmpdir, "broken.pt")
            with open(pt_path, "wb") as f:
                f.write(b"NOT_A_VALID_PT_FILE")

            with (
                patch.dict(sys.modules, {"torch": MagicMock()}),
                patch(
                    "qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", tmpdir
                ),
                patch(
                    "qwen3_tts.core.engine.voice_prompt.safe_path_join",
                    side_effect=lambda d, f: os.path.join(d, f),
                ),
                patch(
                    "qwen3_tts.core.engine.voice_prompt._load_pt_safe",
                    side_effect=RuntimeError("Cannot load broken.pt safely"),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    _load_voice_prompt_torch("broken.pt", allow_create=False)

    @patch("qwen3_tts.core.engine.voice_prompt.get_backend", return_value="torch")
    def test_corrupt_pt_with_wav_raises_create_required(self, _mock_be):
        from qwen3_tts.core.engine import VoicePromptCreateRequired
        from qwen3_tts.core.engine.voice_prompt import _load_voice_prompt_torch

        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = os.path.join(tmpdir, "broken.pt")
            with open(pt_path, "wb") as f:
                f.write(b"NOT_A_VALID_PT_FILE")
            with open(os.path.join(tmpdir, "broken.wav"), "wb") as f:
                f.write(b"fake-wav-bytes")

            with (
                patch.dict(sys.modules, {"torch": MagicMock()}),
                patch(
                    "qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", tmpdir
                ),
                patch(
                    "qwen3_tts.core.engine.voice_prompt.safe_path_join",
                    side_effect=lambda d, f: os.path.join(d, f),
                ),
                patch(
                    "qwen3_tts.core.engine.voice_prompt._load_pt_safe",
                    side_effect=RuntimeError("Cannot load broken.pt safely"),
                ),
            ):
                with self.assertRaises(VoicePromptCreateRequired) as ctx:
                    _load_voice_prompt_torch("broken.pt", allow_create=False)

        self.assertEqual(ctx.exception.prompt_file, "broken.pt")

    @patch("qwen3_tts.core.engine.voice_prompt.get_backend", return_value="torch")
    def test_allow_create_true_never_raises_create_required(self, _mock_be):
        """load_voice_prompt(pf, allow_create=True) must never raise
        VoicePromptCreateRequired -- it must always do the create inline."""
        from qwen3_tts.core.engine.voice_prompt import load_voice_prompt

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "voice.wav"), "wb") as f:
                f.write(b"fake-wav-bytes")
            with open(os.path.join(tmpdir, "voice.txt"), "w") as f:
                f.write("hello world")

            with (
                patch(
                    "qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR", tmpdir
                ),
                patch(
                    "qwen3_tts.core.engine.voice_prompt.safe_path_join",
                    side_effect=lambda d, f: os.path.join(d, f),
                ),
                patch(
                    "qwen3_tts.core.engine.voice_prompt._auto_create_pt_from_wav",
                    return_value="created",
                ) as mock_create,
            ):
                result = load_voice_prompt("voice.pt", allow_create=True)

        self.assertEqual(result, "created")
        mock_create.assert_called_once()


@_skip
class TestConcurrentCreateConvergence(unittest.TestCase):
    """Two callers racing the same missing prompt must converge on ONE
    create -- proven via the real engine cache, with only the deep
    torch/create seams mocked."""

    def setUp(self):
        from qwen3_tts.core.engine.voice_prompt import (
            _torch_prompt_cache,
            _torch_prompt_cache_lock,
        )

        with _torch_prompt_cache_lock:
            _torch_prompt_cache.clear()
        self.addCleanup(self._clear_cache)

    def _clear_cache(self):
        from qwen3_tts.core.engine.voice_prompt import (
            _torch_prompt_cache,
            _torch_prompt_cache_lock,
        )

        with _torch_prompt_cache_lock:
            _torch_prompt_cache.clear()

    def test_two_racing_callers_create_exactly_once(self):
        from qwen3_tts.server.prompt_loading import load_voice_prompt_serialized

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "race.wav"), "wb") as f:
                f.write(b"fake-wav-bytes")
            with open(os.path.join(tmpdir, "race.txt"), "w") as f:
                f.write("hello world")

            create_calls = []

            def _fake_create(model, ref_audio, ref_sr, transcript, **kwargs):
                create_calls.append(1)
                return f"prompt-#{len(create_calls)}"

            state = _make_state(clone_model=object())

            async def _scenario():
                with (
                    patch.dict(sys.modules, {"torch": MagicMock()}),
                    patch(
                        "qwen3_tts.core.engine.voice_prompt.VOICE_PROMPTS_DIR",
                        tmpdir,
                    ),
                    patch(
                        "qwen3_tts.core.engine.voice_prompt.safe_path_join",
                        side_effect=lambda d, f: os.path.join(d, f),
                    ),
                    patch(
                        "qwen3_tts.core.engine.voice_prompt.get_backend",
                        return_value="torch",
                    ),
                    patch(
                        "qwen3_tts.core.engine.voice_prompt.load_audio_for_cloning",
                        return_value=("fake-audio", 24000),
                    ),
                    patch(
                        "qwen3_tts.core.engine.inference.create_voice_prompt",
                        side_effect=_fake_create,
                    ),
                ):
                    results = await asyncio.gather(
                        load_voice_prompt_serialized(state, "race.pt"),
                        load_voice_prompt_serialized(state, "race.pt"),
                    )
                return results

            results = asyncio.run(_scenario())

        self.assertEqual(
            len(create_calls),
            1,
            "two racing callers for the same missing prompt must converge "
            "on exactly one create -- the top-of-function cache re-check "
            "in _load_voice_prompt_torch is load-bearing for this",
        )
        self.assertEqual(results, ["prompt-#1", "prompt-#1"])


if __name__ == "__main__":
    unittest.main()
