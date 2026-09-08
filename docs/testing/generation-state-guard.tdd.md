# TDD Evidence: Step 0C — `generation_state` threading discipline (`GenerationStateGuard`)

**Source plan:** docs/plans/2026-09-08-step-0c-generation-state-thread-safety.plan.md
(Step 0C of docs/plans/2026-09-06-consolidated-backlog-priority.plan.md)

**Branch:** `fix/generation-state-thread-safety` (cut from main @ `1bc081c`, the 0B merge)

## Problem

`app.state.generation_state` is a plain dict touched by BOTH the event loop and
worker threads. `state.generation_lock` is an `asyncio.Lock`, which provides zero
mutual exclusion against non-event-loop threads — so `_chunk_progress` (called via
`progress_callback` inside `run_inference` / `run_inference_streaming`, i.e. on the
`asyncio.to_thread` worker and the streaming daemon thread) mutated the dict with no
effective lock, and several sites (`/generate-stream`'s begin and finally,
`handle_unload_model`'s read, the batch pre-loop clear and finally, every multi-key
read in `/generation-status`, `/queue-status`, `detect_degraded_generation`) took no
lock at all. Consequences: unserialized thread-vs-loop mutations, and torn multi-key
reads (an `elapsed_sec` computed across a concurrent reset). The fix is a small
`threading.Lock`-owning guard (`GenerationStateGuard` + `guard_for`) that every
read/write routes through, with `snapshot()` giving readers one atomic step.

## Task report

| Task | Site | RED | GREEN |
|---|---|---|---|
| 1 | Guard module (`generation_state_guard.py`, new) + lifespan wiring | `ModuleNotFoundError: No module named 'qwen3_tts.server.generation_state_guard'` (collection error, 1 error) | `24 passed in 0.12s`; fix round `26 passed` both envs |
| 2 | Batch `/generate` routing (7 sites in `app_generation.py`) | 6 failed / 0 passed at BASE — `AttributeError: module 'qwen3_tts.server.app_generation' does not have the attribute 'guard_for'` (the routing did not exist) | 6 passed; unlocked-`_chunk_progress` mutant → 2 failed / 4 passed (the two record-dependent pins) |
| 3 | `/generate-stream` + `/ws` routing (6 sites) | 5 failed / 7 passed at BASE — empty recordings and `0 != 1` resolution counts | 12/12 routing + 5 streaming-cancel, both envs; unlocked-mutant → 2 failed / 10 passed |
| 4 | Readers (`app.py` ×3, `app_models.py`, `app_lifespan.py`) | **8 failed** against the unlocked readers (see the torn-read RED verbatim below) | 8 passed; reverted-reader mutant → exactly the named regression fails, 7 pass |
| 5 | Exit-criterion structural pin (this file's "The pin" section) | GREEN at HEAD by construction (the tree is already compliant); RED proven by mutant — 2 of the 4 pin tests failed, naming the exact file/line/occurrence | 30 passed in the module (26 guard + 4 pin) |

Every RED failure above was a genuine assertion/collection failure (no skip), with
the exception disclosed where a pin is a CONTRACT pin rather than a RED driver
(Task 1's first-touch test, and the two torn-pair property pins).

Source: `.superpowers/sdd/detailed-step-0c-execution-quiet-lantern/task-{1,2,3,4,5}-report.md`.

## The F1 ruling and its verified mechanism

The plan's named regression — "a `/cancel-generation` landing between the streaming
acquire and the begin write is silently clobbered" — was ruled (F1, controller
pre-flight) probably impossible, with the RED effort redirected at the demonstrable
defects. Task 2's implementer verified the claim against source before any edit:

- **Verdict: F1 CONFIRMED, with a premise correction.** The acquire→begin window on
  the streaming path has ZERO explicit `await` keywords, but it DOES contain an
  await-shaped construct the controller's premise missed: `async with
  state.pending_lock` (the pending-queue removal, ~`:815`). That correction does not
  change the verdict.
- **The verified no-clobber mechanism:** every prior generation's reset is in-lock on
  the SAME `inference_lock` (batch `:588`, streaming `:968`, `/ws` `:605`), so
  `active` is necessarily `False` by the time the next generation enters the window.
  `/cancel-generation` refuses to write when `active` is false — it answers
  `no_active_generation` (`app.py:860`) and writes nothing. There is nothing in the
  window to clobber.
- **Residual recorded for Step 1A:** a cancel arriving during another generation's
  begin window is answered `no_active_generation` while that generation proceeds —
  a semantics question owned by #237/Step 1A, not a clobber.
- **The reachable cancel-erase is a DIFFERENT case (owner: #237 / Step 1A):** the
  eraser is not a stale prior generation but a LIVE CONCURRENT BATCH in its
  inter-item tail. The batch's in-lock `reset_if_owner` fires only on the FINAL item,
  so between items the state still reads `active=True` for the batch — and a
  streaming begin's `cancelled: False` re-clear can then erase a cancel that targeted
  that live batch, which runs to completion. This erase semantics is pre-existing and
  deliberately PRESERVED by `begin()`; 1A's fix must address it at the guard's atomic
  begin point.

## The torn-read RED (Task 4, verbatim)

```
E       AssertionError: 5000.0 != 4000.0 : /generation-status computed elapsed_sec across a state reset -- the multi-key read is not atomic: {'active': True, 'batch_index': 0, 'chunk_index': 1, 'cancelled': False, 'elapsed_sec': 5000.0}
```

Construction (zero sleeps, fully event-driven): a coherent active state
(`start_time=1000.0`), `/generation-status`'s `time` binding patched from app.py's
module globals; the fake clock signals an Event on its FIRST invocation (which lands
exactly between the unlocked `active` read and the `start_time` read), blocks on a
second Event while a real `threading.Thread` runs `reset_if_owner`, then returns the
fixed `5000.0`. The garbage `FIXED - 0.0` is the torn read; GREEN reads one
pre-reset snapshot (`5000.0 - 1000.0 = 4000.0`). The full RED run was 8 failed with
three honest signatures: the torn read above, the (newly armed) `generation_lock`
tripwire firing on BASE's real `/cancel-generation` block, and six empty recordings
where the routing did not exist.

### The tripwire is armed on `__aenter__` (Task 4 fold)

`AsyncMock(side_effect=_fail)` never fires on `async with lock:` — a mock's dunder
children do not inherit the parent's `side_effect`, so the Task-2/3 tripwire
construction was inert and the docstrings overclaimed. Fixed in ONE place used by all
three state factories: `lock.__aenter__.side_effect = _fail` (with
`__aexit__.return_value = False` so a tripped entry surfaces instead of an exit
mocking success). `TestGenerationLockTripwireIsArmed` (3 tests) pins that entry
raises for every factory, that exit never suppresses, and WHY (the naive
construction is inert). Mere attribute access is deliberately NOT detected — the
contract is "no entry".

### The moved-pin judgment (Task 3)

`tests/test_voice_server.py::TestGenerateStreamIdCheck::test_generate_stream_checks_generation_id`
was a source-string pin on the exact old ownership-check line; the routing made it
stale and it moved in the same commit as the routing (repo precedent for direct-test
updates). The reviewer verified the moved pin's teeth IMPROVED: it now fails on a
raw-reset revert, on guard deletion, and on guard bypass, where the old string pin
failed only on text. Disclosed as a third moved test beyond the brief's named set.

## Design decisions

| Decision | Rationale |
|---|---|
| `threading.Lock` (NOT asyncio) | Worker threads (`asyncio.to_thread` progress callbacks, the streaming daemon inference thread, the `/ws` cancel watcher) AND the event loop touch the dict; an asyncio lock excludes neither thread callbacks nor mixes with the existing lock orchestration. |
| Late binding | `_state` is a property resolving `app_state.generation_state` at EVERY call — never captured at construction. The ~130 existing test references that seed/replace the dict keep working, and a guard survives a dict re-init. Pinned by `TestGenerationStateGuardLateBinding`. |
| `begin()` re-clears `cancelled` (preserved) | The re-clear exists to stop a stale flag from a different request truncating a batch. Its erase race is #237 / Step 1A's fix, NOT this branch's. `test_begin_re_clears_cancelled` pins it so 1A must move that assertion consciously. |
| Fail-fast canonical keys | `snapshot(keys)` direct-indexes the requested keys and `is_cancelled()` reads the canonical `cancelled` key with no default: a hand-rolled partial dict raises `KeyError` instead of silently answering. Production always carries the full ten-key shape (lifespan init + `begin()`), and `test_health_degraded` stays green because the degraded reader requests exactly its three keys. |
| `generation_lock` survives | Its remaining user (`/update-model-config`'s model-slot nulling, `app_models.py:329`) guards MODEL SLOTS, not `generation_state`. Untouched; ledger-noted as a pre-existing oddity. Its three former `generation_state` blocks (batch begin, batch final reset, `/ws` begin+finally, `/cancel-generation`) are gone. |
| `guard_for()` construction is serialized | A module-level `_CONSTRUCTION_LOCK` wraps the check-then-set so two concurrent first-touchers can never hold guards backed by DIFFERENT locks (zero mutual exclusion). Production provisions eagerly in lifespan, so the lock serves fakes/tests. |
| `snapshot()` materializes keys pre-lock | `list(keys)` runs before acquisition: a caller-supplied generator that itself touches the guard would otherwise deadlock on the non-reentrant lock. Pinned with a `_lock.locked()` probe — the only non-hanging detector. |
| Guard resolved once per handler | `guard_for` takes the module-level construction lock per call — fine at request frequency, wrong at chunk frequency. Every routed site resolves once per invocation (pinned by the resolution-count tests). |

## The pin (Task 5)

`tests/test_generation_state_guard.py::TestGenerationStateRawAccessIsPinnedToTheGuard`
holds the 0C exit criterion structurally: the five production modules that ever
touched the dict (`app_generation.py`, `websocket.py`, `app_models.py`, `app.py`,
`app_lifespan.py`) are scanned at the TOKEN level, and every surviving
`generation_state` NAME occurrence must be inside an explicit allowlist.

- **Mechanism:** the scan collects `tokenize` NAME tokens spelled exactly
  `generation_state`. Comments, docstrings and string literals are different token
  kinds (so mentioning the dict in prose is not an access), and the lifespan's
  provisioning attribute `generation_state_guard` is a different identifier (the
  word boundary of a NAME token excludes it) — so the allowlist reduces to exactly
  ONE entry: `app_lifespan.py`'s init block, the line that CREATES the dict the
  guard then owns.
- **Loud targets:** every scanned module's resolved basename is asserted against the
  expected filename, and each file's existence is asserted — a rename, move or
  deletion FAILS the pin instead of silently shrinking the scan.
- **Live allowlist:** each allowlist entry must match at least one occurrence, so
  reformatting the init block fails the pin with an "update the allowlist
  consciously" message instead of quietly leaving a dead entry.
- **Anti-vacuity:** the total occurrence count must equal the allowlisted count AND
  `len(ALLOWLIST)` — a second raw access that repeats an allowlisted line shape
  still fails.
- **RED proof (mutant, verified landed before trusting the run):** a raw
  `state.generation_state.get("active")` probe added to `/queue-status` in `app.py`
  (grep `MUTANT` → 1) → `test_no_raw_generation_state_access_outside_the_allowlist`
  and `test_raw_occurrence_count_equals_the_allowlisted_count` both FAILED, naming
  `('app.py', 625, '_mutant_raw_probe = state.generation_state.get("active") …')`.
  Restore proven byte-identical: sha256 `abc2244e…` identical before AND after,
  `git diff -- qwen3_tts/` empty, `grep -c MUTANT` → 0, then the module re-ran
  30/30 green.

## Test file summary (counted, not assumed)

`pytest <file> --collect-only` (`qwen3-tts-mlx`, HEAD `489ee7f`):

- `tests/test_generation_state_guard.py` — **30**
  (`TestGenerationStateGuardBegin` 4, `TestGenerationStateGuardOwnerReset` 3,
  `TestGenerationStateGuardSnapshot` 5, `TestGenerationStateGuardProgress` 2,
  `TestGenerationStateGuardCancellation` 3, `TestGenerationStateGuardLateBinding` 2,
  `TestGuardForProvisioning` 4, `TestLifespanWiring` 2,
  `TestGenerationStateGuardConcurrency` 1, and the Task 5 pin
  `TestGenerationStateRawAccessIsPinnedToTheGuard` 4)
- `tests/test_generation_state_routing.py` — **15**
  (`TestBatchProgressRoutesThroughGuard` 2, `TestBatchLifecycleRoutesThroughGuard` 4,
  `TestStreamingRoutesThroughGuard` 4, `TestWsRoutesThroughGuard` 2,
  `TestGenerationLockTripwireIsArmed` 3)
- `tests/test_generation_state_readers.py` — **8**
  (`TestGenerationStatusSnapshotIsAtomic` 1 — the named torn-read regression —
  plus `TestReadersRouteThroughGuard` 7: 2 cancel, 1 queue-status, 2 unload,
  2 degraded)

Torchless `.venv-310` proxy (RUN-not-skip, zero skips) over all three modules:
**53 passed, 1 pre-existing httpx deprecation warning** (30 + 15 + 8).

## Gates (Step 0C, observed at HEAD `489ee7f`)

| Gate | Command | Result |
|---|---|---|
| Full non-e2e, gate convention | `conda run -n qwen3-tts-mlx python -m pytest tests/ -m "not e2e" -q --tb=short --ignore=tests/evaluations` | **3217 passed, 4 skipped, 92 deselected, 0 failed** in 49.96 s |
| Full non-e2e, raw form (Task 4's comparison figure) | `conda run -n qwen3-tts-mlx python -m pytest tests/ -m "not e2e" -q --tb=short` | **3232 passed, 5 skipped, 92 deselected, 0 failed** in 47.41 s — exactly **+4** over Task 4's 3228 (the four pin tests; `tests/evaluations` holds the 16-test delta between the two forms, unchanged since 0B) |
| Batch 3 (Server Infrastructure) | `conda run -n qwen3-tts-mlx python tests/run_batches.py --batch 3` | 1/1 batches passed |
| ruff | `conda run -n qwen3-tts-mlx ruff check qwen3_tts tests` | All checks passed |
| mypy (entry-point form) | `conda run -n qwen3-tts-mlx mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface` | Success: no issues found in 58 source files (pre-existing `annotation-unchecked` notes only) |
| Torchless RUN-not-skip (all Task 1-5 modules) | `./.venv-310/bin/python -m pytest tests/test_generation_state_guard.py tests/test_generation_state_routing.py tests/test_generation_state_readers.py -q -rs` | **53 passed, 0 skipped** |
| Batch registration | `tests/test_batches_coverage.py` | passed (all three modules registered in batch 3) |
| Stale-count sync grep | `git grep -n "3163\|3228\|3161\|3151\|3232\|3217" -- CLAUDE.md docs/COMMANDS.md docs/CONTRIBUTING.md docs/RUNBOOK.md Makefile` | zero hits — no count line to sync |

## Survivor grep proof (the exit criterion, stated)

`grep -rn "generation_state" qwen3_tts/server/*.py` — every hit is an import, a
comment/docstring, the guard module itself, or the single allowlisted init line
`app_lifespan.py:409: app.state.generation_state = {`. `app_generation.py`,
`websocket.py`, `app_models.py` and `app.py` contain zero raw dict accesses; the
provisioning line at `app_lifespan.py:426` is the `generation_state_guard`
attribute, not the dict. `generation_lock`'s only remaining user is
`app_models.py:329` (`/update-model-config` model slots).

## Commit list (in order)

- `36513ff` — `feat(server): add threading.Lock guard for generation_state (Step 0C Task 1)`
- `b1fbc59` — `fix(server): serialize guard_for first-touch, materialize snapshot keys pre-lock`
- `08ee6d3` — `refactor(server): route the batch /generate path through the generation-state guard`
- `6cf5f7e` — `test(server): stop the routing test driver leaking cache WAVs and label the concurrency pin's scope`
- `f833b47` — `test(server): pin streaming + /ws generation_state routing through the guard (RED)`
- `c2144a1` — `refactor(server): route the streaming and /ws paths through the generation-state guard`
- `357c191` — `test(server): arm the generation_lock tripwire on __aenter__ and repoint the inert validation patches`
- `b04a8a9` — `test(server): pin the reader-side generation_state routing (RED against unlocked readers)`
- `63d6254` — `fix(server): route the generation_state readers through the guard atomically`
- `489ee7f` — `test(server): pin the generation_state guard-access exit criterion`

## Accepted residuals (all pre-existing, disclosed, owned elsewhere)

1. **The live-batch cancel-erase** (above) — #237 / Step 1A, to be fixed at the
   guard's atomic begin point. The batch pre-loop `clear_cancelled()`
   (`app_generation.py:311`) is a second, independent erase site in the #237 family: it
   runs before `inference_lock` is acquired, so a cancel aimed at a live concurrent
   generation (stream, `/ws`, or another batch) is erased by a request that never owned
   the flag — stream begins, the user cancels (`cancelled=True`), a batch request's
   pre-loop clear erases the flag, and the stream's stop-check then sees `False` and
   runs to completion; 1A's fix must cover this site, not only `begin()`'s re-clear.
2. **`/cancel-generation` takes the guard's lock twice** (active check, then the
   post-write id read) — matches the pre-guard observable ordering; two
   non-contended microsecond acquisitions per cancel.
3. **`/ws` chunk progress is invisible to `/generation-status`** — the `/ws`
   inference thread passes no `progress_callback`, so `chunk_index`/`chunk_total`
   stay at their begin values. Pre-existing, faithfully preserved.
4. **`load_model` fallback cost + the `:821` lock-attribute cache** — recorded in the
   consolidated plan's Step 0B follow-ups, not this step's scope.
