# TDD Evidence — Phase 2b (#214 item 2): the /unload-asr race

## The defect

Two halves, both in the ASR path.

**Half A — `unload_asr_model()` mutated shared globals unlocked.**
`core/engine/asr.py` defines `_asr_lock` and every *loader* takes it
(`_ensure_asr_torch_loaded`, `load_asr_model`). Unload nulled
`_asr_model_mlx`/`_asr_model_torch` with no lock — the module's only
unsynchronized writer.

**Half B — `handle_transcribe` had a check-then-use window across the lock.**
`is_asr_loaded()` was checked *before* `inference_lock`; `transcribe_audio`
(which lazily loads on first call) ran *inside* it. An `/unload-asr` landing in
that window made a multi-minute model rebuild happen while `inference_lock` was
held, starving `/generate`.

## RED

New module `tests/test_issue214_unload_asr_race.py`, written before any
implementation existed (working tree at that point contained only the test file
plus its `BATCHES` registration):

```
FAILED ...::test_unload_acquires_and_releases_the_recorded_lock
FAILED ...::test_unload_blocks_while_lock_held_by_another_thread
FAILED ...::test_genuine_first_use_load_still_runs_outside_lock_with_recheck
FAILED ...::test_post_lock_recheck_detects_concurrent_unload
4 failed
```

`AssertionError: 0 not greater than or equal to 1 : unload_asr_model() never
acquired _asr_lock -- it mutates the shared ASR globals unlocked`

## The fix is bigger than the plan scoped, and here is why

The plan specified: take `_asr_lock` in unload + a post-lock recheck in
`/transcribe`. That was implemented and went GREEN. **Adversarial review then
showed the specification itself was insufficient**, and three independent
reviewers converged on it:

`/unload-asr` did not take `inference_lock`, so an unload could still land
between the post-lock recheck and `transcribe_audio`'s own lazy-load check
(`asr.py:168`; `_transcribe_torch` calls `_ensure_asr_torch_loaded()`
unconditionally, so on torch the recheck removed no structural step at all).
The window shrank from unbounded to microseconds of scheduler timing — a real
improvement, but the code comment asserted an absolute the code did not provide.

**Resolution: `/unload-asr` acquires `inference_lock`** (leaf acquisition,
preserving the inference_lock-outermost order). This closes the window
structurally instead of narrowing it, and closes the *identical* check-then-use
on the ICL echo-trim probe (`inference.py:1155` → `:1163`) for free — that runs
inside `run_inference` with `inference_lock` already held, so a lock-taking
unload can never interleave with it.

Consequence: the unload now queues behind an in-flight generation, so
`UNLOAD_ASR_TIMEOUT_SEC` (=900) was added and wired into the UI — same defect
class as the `/load-model` 120s (#211) and `/transcribe` 60s (#212).

## Mutation evidence

Passing tests prove little on their own; each guard was verified to **fail**
against a targeted mutant.

| Mutant | Before | After |
|---|---|---|
| Recheck moved BEFORE `inference_lock` | **SURVIVED** (all assertions passed) | KILLED |
| `/unload-asr` route drops `inference_lock` | **SURVIVED** (all 9 passed) | KILLED |

**Both survivals were defects in tests I had already accepted.**

1. `test_post_lock_recheck_detects_concurrent_unload` asserted call counts and
   "didn't reload", but never observed the lock — so it could not distinguish a
   post-lock recheck from a pre-lock one, which is the only property it exists
   for. Fixed by recording lock state at each `is_asr_loaded()` call and
   asserting `recheck_lock_states == [False, True]`.
2. `test_route_source_acquires_inference_lock` used
   `"inference_lock" in inspect.getsource(...)`. The route's own docstring
   explains the lock at length, so the substring stayed true with the
   `async with` deleted — **the documentation defeated the guard**. Rewritten to
   parse the function as an AST and require an actual `AsyncWith` node whose
   context expression names `inference_lock`.

## An existing test was modified — justification and proof

`tests/test_issue192_transcribe_serialization.py::_run_handler` patched
`is_asr_loaded` with a fixed `return_value=asr_preloaded`, i.e. permanently
`False` on the load path — claiming "still not loaded" even after
`load_asr_model()` returned. Replaced with a stateful stub that flips `True`
when the load runs.

Verified two ways, independently reproduced by a reviewer:

1. Reintroducing the original #192 defect (moving the load back inside
   `inference_lock`) still fails the **updated** stub with
   `AssertionError: True is not false : ASR model load ... must run outside
   inference_lock` — detection power intact.
2. Reverting *only* the stub while keeping the corrected production code makes
   the test fail with a **false-positive 503 on the happy path**, because a
   permanently-`False` `is_asr_loaded` trips the new recheck on an ordinary
   first-use load.

So the old stub was not merely less realistic — it was actively broken against
correct code. Every pre-existing assertion is byte-for-byte unchanged.

## Also fixed (review findings)

- **UI rendered the 503 as "Unknown error."** FastAPI serializes
  `HTTPException(detail={...})` as `{"detail": {...}}`, so `_error_response`'s
  structured body is nested. `voice_management.py` read
  `resp.json().get("error")` → always `None`. Now reads the nested payload and
  surfaces `recovery="retry"` as an actionable suggestion.
- **`gc.collect()` before `torch.cuda.empty_cache()`** (was inverted vs
  `unload_model_cleanup`). Nulling a global only drops a reference; calling
  `empty_cache()` before gc collects returns blocks to the caching allocator,
  not the driver.
- Explicit `return` after `_error_response` (typed `-> None`, not `NoReturn`),
  matching the convention at `app_models.py:273`.
- `addCleanup` registered before mutating module globals, restoring originals.

## Gates

- `ruff check qwen3_tts tests` — All checks passed
- `mypy qwen3_tts/{core,server,interface}` — Success, 54 source files
- `bandit -r qwen3_tts -c pyproject.toml` — exit 0
- `pytest tests/ -m "not e2e"` — **3028 passed**, 12 skipped
- CLAUDE.md 298/300 (net-zero edit: struck `unload-asr race` from the
  remaining-#192 list, documented the 503 on the `/transcribe` API row)

## Not covered

No e2e test drives a real `/unload-asr` against a live server during a real
generation — plan Phase 2d owns that tier. The timeout figure (900s) is taken
from the repo's existing precedents, not measured against a real ASR load.

## Deferred (tracked, not fixed here)

- `/load-asr`'s 60s client timeout — pre-existing (audit M6, plan Phase 3c); it
  does not take `inference_lock`, so this change does not worsen it.
- `_error_response` should be typed `NoReturn`; changing it is repo-wide and may
  surface unreachable-code errors elsewhere.
- `handle_unload_asr` lacks the exception handling its `handle_load_asr` sibling
  has.
- `handle_unload_asr` blocks a shared default-executor worker (16 on this box)
  for the acquire; `TTS_DISABLE_RATE_LIMITING=1` removes the 5/min cap.
- `is_asr_loaded()` ORs the mlx and torch slots rather than being
  backend-aware — no reachable trigger found; flagged unverified.
