# Step 3A — `seed_lock_chunks` (TDD evidence)

**Executed:** 2026-09-13 → 2026-09-17 · **PR #304**, squash `e8eb6440` · branch `fix/phase3-seed-lock-chunks` (base `e12bf21d`)
**Protocol:** two-gate adversarial SDD — RED `b33cbf52` → Gate-A strengthenings `bd0a2dfd` → GREEN `9b76c019`
**Plan entry:** `docs/plans/2026-09-06-consolidated-backlog-priority.plan.md` § "Step 3A"

This file preserves the lane's two working artifacts verbatim, moved into the repo before the
worktree was removed: the implementer brief (the requirements the agent was held to) and the
implementer report (Gate A failure output, the fix round, GREEN evidence, and the WS4.5
investigation). They are the primary evidence behind the plan's Step 3A DONE block; the plan
carries the verdicts, this file carries the workings.

---

## Part 1 — Implementer brief (was `.sdd-brief-3a.md`)

# Task brief — Step 3A: make `seed_lock_chunks` real

Branch: `fix/phase3-seed-lock-chunks` (worktree, based on `e12bf21d`).
Plan entry: `docs/plans/2026-09-06-consolidated-backlog-priority.plan.md` § "Step 3A — 3d".

## The defect (verified against current source, not the plan's stale line numbers)

`seed_lock_chunks` appears in `qwen3_tts/core/engine/inference.py` in exactly two
places, both inside `run_inference`:

- signature, `inference.py:1355`
- `seed = gen_params.get("seed") if seed_lock_chunks else None`, `inference.py:1427`

Three concrete problems:

1. **The `False` branch does nothing rather than the prescribed thing.** With
   `seed_lock_chunks=False` the loop-level seed becomes `None`, i.e. no seeding at
   all. The specified behaviour is a per-chunk *derived* seed `base_seed + chunk_index`.

2. **The inner runners override the loop anyway, so the flag is a no-op today.**
   `_run_inference_torch` (`inference.py:234-235`) does
   `torch.manual_seed(gen_params["seed"])` unconditionally on **every** chunk, and
   `_run_inference_mlx` / `_run_inference_mlx_streaming` call
   `_set_seed_for_backend(gen_params["seed"])` inline. So every chunk already gets
   the identical base seed regardless of the flag's value.

3. **`run_inference_streaming` has no `seed_lock_chunks` parameter at all**
   (`inference.py:1614-1627`). Its two call sites pass none:
   - `qwen3_tts/server/app_generation.py:~1100` (`gen_params=seeded_params`)
   - `qwen3_tts/server/websocket.py:~462` (`gen_params=gen_params`)

   Both call sites already *have* the flag available —
   `app_generation.py` has `req.seed_lock_chunks` (used at `:439` and `:709` on the
   batch paths) and `websocket.py` parses it at `:356` into the validated `req`.

## Required behaviour (this is the spec — implement exactly this)

Effective seed for text-chunk index `i`, given `base_seed = gen_params.get("seed")`:

| `base_seed` | `seed_lock_chunks` | effective seed for chunk `i` |
|---|---|---|
| `None` | either | `None` — no seeding at all |
| set | `True`  | `base_seed` (identical every chunk — voice consistency) |
| set | `False` | `base_seed + i` (reproducible run, chunks free to vary) |

Single-chunk generations are `i = 0`, so both flag values give `base_seed`. That is
correct and must stay true.

This behaviour must hold on **both** runners (`run_inference` and
`run_inference_streaming`) and **both** backends (torch and MLX).

## Prescribed shape (rulings already made — do not redesign these)

- Add a small module-level helper in `inference.py` next to `_set_seed_for_backend`:

  ```python
  def _chunk_seed(base_seed: int | None, chunk_index: int, seed_lock_chunks: bool) -> int | None:
  ```

  Returns `None` when `base_seed is None`; `base_seed` when `seed_lock_chunks`;
  otherwise `base_seed + chunk_index`.

- **Delivery mechanism:** the inner runners seed from `gen_params["seed"]`, so the
  effective seed must reach them. Per text chunk, build a **new** dict
  `{**gen_params, "seed": effective}` and pass that to the inner call. Never mutate
  the caller's `gen_params` (project immutability rule).

- **Batch path (`run_inference`)**: keep the existing explicit
  `_set_seed_for_backend(...)` call in the multi-chunk loop as the backend-agnostic
  seeding point, but call it with the *effective* per-chunk seed, and also pass the
  per-chunk `gen_params` copy into `_run_inference_single`. The resulting
  double-seed with an identical value is intentional and harmless — it preserves the
  observable seam the existing tests pin.

- **Streaming path (`run_inference_streaming`)**: add
  `seed_lock_chunks: bool = False` to the signature (keyword, defaulted, documented
  in the docstring's Args block). In **both** the MLX branch and the torch-fallback
  branch, pass a per-chunk `gen_params` copy carrying the effective seed. Do **not**
  add a new explicit `_set_seed_for_backend` call in streaming — the inner paths
  already seed, and adding one would change the existing call-count seam.

- **Call sites**: thread `seed_lock_chunks=req.seed_lock_chunks` through
  `app_generation.py`'s `run_inference_streaming(...)` call and
  `websocket.py`'s `run_inference_streaming(...)` call.

## Existing tests that MUST be updated (they encode the old, wrong behaviour)

`tests/test_seed_lock_chunks.py`:

- `TestSeedLockChunksInRunInference::test_no_reseed_when_disabled` — asserts
  `_set_seed_for_backend` is never called when the flag is `False`. Under the spec it
  is now called once per chunk with `99, 100`. Rewrite the assertion; do not delete
  the test.
- `TestMLXStreamingSeedApplication::test_mlx_streaming_multi_chunk_seeds_each` —
  asserts `42, 42, 42`. With streaming's default `seed_lock_chunks=False` this becomes
  `42, 43, 44`. Update, and **add** a sibling test passing `seed_lock_chunks=True`
  that pins `42, 42, 42`.
- `test_reseeds_before_each_chunk` (flag `True`) and `test_no_reseed_single_chunk`
  should still pass unchanged. If either needs changing, that is a signal your
  implementation drifted from the spec — stop and report rather than editing them.

## New coverage required

- `_chunk_seed` unit tests for all four rows of the table above.
- Batch multi-chunk, flag `False`, base seed set: `_set_seed_for_backend` receives
  `base, base+1, base+2` **and** the `gen_params` handed to `_run_inference_single`
  carries the matching per-chunk seed (assert on `mock_single.call_args_list`).
- Batch: caller's `gen_params` dict is not mutated (immutability pin).
- Batch: `base_seed is None` → `_set_seed_for_backend` not called, for both flag values.
- Streaming torch-fallback branch: per-chunk seeds reach `_run_inference_single`,
  both flag values.
- Both call sites forward the flag: assert `run_inference_streaming` was called with
  `seed_lock_chunks=<the request's value>`. Follow the existing convention in
  `tests/test_engine_streaming.py` / `tests/test_streaming_thread_lifecycle.py` for
  how those call sites are exercised — do not invent a new harness shape.

## WS4.5 add-on — investigate, do NOT force

The plan carries an add-on: replace the global `torch.manual_seed` (`inference.py:235`
and `:1292` inside `_set_seed_for_backend`) with a per-call `torch.Generator`, so
reproducibility survives if concurrency is ever allowed. It is functionally safe today
under the single `inference_lock`.

Do this **only if** the downstream call path genuinely accepts a `generator=`
argument. Trace it and report the evidence either way. If the upstream
`qwen_tts` / `transformers` generate call does not take a generator, report
NOT FEASIBLE with the file:line evidence — do not fabricate a wrapper, and do not
change the seeding mechanism speculatively. This add-on is explicitly optional for
this PR.

## Workflow — TWO-GATE, stop at RED

**Gate A (stop here and report):** write the tests first, run them, confirm they FAIL
for the right reason (assertion about seed values, not an import/TypeError that masks
everything). Commit the tests alone. Then STOP and report — do not implement yet.
Your report must include the exact failure output per test.

You will be sent a review, then told to proceed to GREEN.

## Constraints (project rules — non-negotiable)

- All `torch` / `mlx` imports stay lazy (inside functions). Never at module scope.
- Immutable style: new dicts, never in-place mutation of caller-owned structures.
- **No AI authorship attribution anywhere** — no `Co-Authored-By`, no "Generated by
  Claude", no AI references in commit messages, code comments, or docs. Commits are
  authored solely by the repository owner. This overrides any default your harness has.
- Do not commit to `main`, do not push, do not open a PR. Commit on the current branch only.
- Conventional-commit messages (`fix:` / `test:`).
- Run everything via `conda run -n qwen3-tts-mlx python -m pytest ...`.
- Any command that could exceed ~5 minutes: run it backgrounded to a log file and poll
  the tail, never a blocking foreground wait.

## Verify commands

```
conda run -n qwen3-tts-mlx python -m pytest tests/test_seed_lock_chunks.py tests/test_engine_streaming.py -v
conda run -n qwen3-tts-mlx python -m ruff check qwen3_tts tests
conda run -n qwen3-tts-mlx python -m mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface
```

## Report file

Write your full report to `.sdd-report-3a.md` in the worktree root. Return only:
status, commit SHAs, a one-line test summary, and any concerns.

---

## Part 2 — Implementer report (was `.sdd-report-3a.md`)

# Step 3A Gate A report — seed_lock_chunks tests at RED

Branch: `fix/phase3-seed-lock-chunks` (worktree, base `e12bf21d`)
Gate A commit: **`b33cbf52`** — `test: pin Step 3A seed_lock_chunks spec at RED (Gate A, tests only)`
Scope: tests only. No production file was touched. Stopped at RED per the brief.

## What was built

All new/updated coverage lives in the existing **`tests/test_seed_lock_chunks.py`**
(already registered in `BATCHES` in `tests/run_batches.py` — no new module created,
no registration change needed). 473 insertions, 6 deletions. Every `Test*` class
subclasses `unittest.TestCase` (the two async call-site tests use
`unittest.IsolatedAsyncioTestCase`, a TestCase subclass — async-hygiene compliant).

### Existing tests updated (per brief — not deleted)

1. `TestSeedLockChunksInRunInference::test_no_reseed_when_disabled` → rewritten as
   **`test_derived_seeds_when_disabled`**. The brief ordered "rewrite the
   assertion; do not delete the test"; I additionally renamed it because the old
   name asserts the opposite of the new body (never-reseed vs reseed-with-derived).
   Docstring cites the Step 3A rewrite. **Flag for reviewer:** if you want the
   original name kept, say so and I will revert the rename only.
   New body: 2 chunks, base 99, flag `False` → `_set_seed_for_backend` called
   `[call(99), call(100)]`.
2. `TestMLXStreamingSeedApplication::test_mlx_streaming_multi_chunk_seeds_each` —
   assertion updated from `[42, 42, 42]` to `[42, 43, 44]` (streaming default
   `seed_lock_chunks=False`), docstring updated.
   **Added sibling** `test_mlx_streaming_seed_lock_true_pins_base_seed` passing
   `seed_lock_chunks=True` → `[42, 42, 42]`.

### Protected tests — untouched and passing (brief's drift detector)

- `TestSeedLockChunksInRunInference::test_reseeds_before_each_chunk` — PASSED at RED
- `TestSeedLockChunksInRunInference::test_no_reseed_single_chunk` — PASSED at RED

No other existing test in the module was modified.

### New coverage (brief's list, item by item)

| Brief requirement | Test |
|---|---|
| `_chunk_seed` unit tests, all four table rows | `TestChunkSeedHelper` — 4 row tests + `test_chunk_zero_returns_base_for_both_flag_values` (pins the brief's i=0 property) |
| Batch multi-chunk flag=False: `_set_seed_for_backend` gets `base, base+1, base+2` AND per-chunk `gen_params` reaches `_run_inference_single` | `TestBatchPerChunkSeedDelivery::test_false_flag_seeds_and_delivers_per_chunk_seeds` (asserts `mock_single.call_args_list`; module helper `_gen_params_of` reads the params dict positionally-or-by-keyword so the call convention is not pinned) |
| Batch immutability pin | `TestBatchPerChunkSeedDelivery::test_caller_gen_params_not_mutated` (dict-equality snapshot also catches key injection) |
| Batch `base_seed=None` → no seeding, both flag values | `test_none_base_seed_no_seeding_flag_true` / `..._flag_false` |
| Streaming torch-fallback per-chunk seeds, both flag values | `TestStreamingTorchFallbackPerChunkSeeds::test_default_flag_delivers_derived_seeds` / `test_lock_true_delivers_base_seed_every_chunk` |
| Both call sites forward the flag | `TestStreamingCallSitesForwardFlag::test_http_generate_stream_forwards_seed_lock_chunks` (`handle_generate_stream`, harness mirrors `tests/test_streaming_thread_lifecycle.py`) / `test_ws_stream_generation_forwards_seed_lock_chunks` (`_stream_generation` with mock ws, same convention). Assert `mock.call_args.kwargs["seed_lock_chunks"] is True` with `req.seed_lock_chunks=True`. |

## RED evidence (exact failures)

Run: `conda run -n qwen3-tts-mlx python -m pytest tests/test_seed_lock_chunks.py -v`
Result: **13 failed, 14 passed in 2.22s**. Combined with `tests/test_engine_streaming.py`
(brief's verify command): **13 failed, 26 passed** — all 12 engine-streaming tests green
(no collateral damage).

### RED on seed-value / flag-forwarding assertions (the "right reason")

```
FAILED TestSeedLockChunksInRunInference::test_derived_seeds_when_disabled
  AssertionError: 0 != 2
  (tests/test_seed_lock_chunks.py: mock_set_seed.call_count — today flag=False seeds never)

FAILED TestBatchPerChunkSeedDelivery::test_false_flag_seeds_and_delivers_per_chunk_seeds
  AssertionError: 0 != 3
  (same root: loop never calls _set_seed_for_backend when the flag is False)

FAILED TestMLXStreamingSeedApplication::test_mlx_streaming_multi_chunk_seeds_each
  AssertionError: Calls not found.
  Expected: [call(42), call(43), call(44)]
    Actual: [call(42), call(42), call(42)]

FAILED TestStreamingTorchFallbackPerChunkSeeds::test_default_flag_delivers_derived_seeds
  AssertionError: Lists differ: [42, 42, 42] != [42, 43, 44]
  First differing element 1: 42 != 43

FAILED TestStreamingCallSitesForwardFlag::test_http_generate_stream_forwards_seed_lock_chunks
  AssertionError: None is not true : handle_generate_stream must forward
  req.seed_lock_chunks to run_inference_streaming (got None)

FAILED TestStreamingCallSitesForwardFlag::test_ws_stream_generation_forwards_seed_lock_chunks
  AssertionError: None is not true : websocket._stream_generation must forward
  req.seed_lock_chunks to run_inference_streaming (got None)
```

### RED with structurally unavoidable errors (new API surface cannot exist pre-fix)

```
FAILED TestChunkSeedHelper::test_none_base_seed_returns_none_with_lock
FAILED TestChunkSeedHelper::test_none_base_seed_returns_none_without_lock
FAILED TestChunkSeedHelper::test_lock_true_returns_base_seed_unchanged
FAILED TestChunkSeedHelper::test_lock_false_returns_base_plus_chunk_index
FAILED TestChunkSeedHelper::test_chunk_zero_returns_base_for_both_flag_values
  ImportError: cannot import name '_chunk_seed' from 'qwen3_tts.core.engine.inference'

FAILED TestMLXStreamingSeedApplication::test_mlx_streaming_seed_lock_true_pins_base_seed
FAILED TestStreamingTorchFallbackPerChunkSeeds::test_lock_true_delivers_base_seed_every_chunk
  TypeError: run_inference_streaming() got an unexpected keyword argument 'seed_lock_chunks'
```

These seven cannot fail any other way at Gate A: the helper / parameter they
exercise does not exist yet. Every behavioural test whose subject is observable
under the *current* signature fails on the seed-value assertion instead (above).
No masking import noise anywhere else in the module.

### Pins that PASS at RED (by design — they guard the GREEN implementation)

- `test_caller_gen_params_not_mutated` — current code doesn't mutate; the pin
  exists so a naive in-place `gen_params["seed"] = effective` fix fails.
- `test_none_base_seed_no_seeding_flag_true` / `_flag_false` — current code
  already no-ops on `seed=None`; pins that the fix doesn't seed `None + i`.

## Gates run

- `pytest tests/test_seed_lock_chunks.py tests/test_engine_streaming.py -v`: 13 F / 26 P (expected RED)
- `pytest tests/test_async_test_hygiene.py tests/test_batched_testcase_hygiene.py tests/test_batches_coverage.py -q`: **10 passed** (async tests on IsolatedAsyncioTestCase; module already in `BATCHES`)
- Lint: the mlx env has no `ruff` module (`python -m ruff` → "No module named ruff");
  ran the globally installed `ruff 0.16.7` from the worktree root against the same
  targets: `ruff check qwen3_tts tests` → **All checks passed!**
- `mypy qwen3_tts/{core,server,interface}`: not run at Gate A — zero production
  files changed and mypy's path list excludes `tests/`, so the result is
  unchanged from the base commit. Will run at GREEN.

## WS4.5 add-on investigation — verdict: NOT FEASIBLE (evidence)

The downstream torch call path does **not** accept a `generator=` argument;
any per-call `torch.Generator` would be silently dropped. Evidence
(torch conda env `qwen3-tts`, site-packages):

- `qwen_tts/inference/qwen3_tts_model.py:470/637/732` —
  `generate_voice_clone` / `generate_voice_design` / `generate_custom_voice`
  accept `**kwargs` and forward to `Qwen3TTSForConditionalGeneration.generate(...)`.
- `qwen_tts/core/models/modeling_qwen3_tts.py:2022-2043` — `generate()`'s
  explicit parameter list has **no** `generator`; extras land in `**kwargs` (:2042).
- `modeling_qwen3_tts.py:2044-2066` — the `talker_kwargs` dict built from those
  params forwards **no** generator (the only `kwargs` reads are two
  `getattr(kwargs, "output_hidden_states", ...)` calls, which are dict-vs-dotted
  bugs that always take the default).
- `modeling_qwen3_tts.py:1671-1680` — the codec-predictor `generate(...)` call
  passes a closed kwarg set (`do_sample/top_p/top_k/temperature/...`); nothing
  threads a generator into the sampling loop.

Per the brief: no wrapper fabricated, no speculative change. The global
`torch.manual_seed` stays (functionally safe under the single `inference_lock`).

## Concerns / notes for review

1. **Rename flag** — `test_no_reseed_when_disabled` → `test_derived_seeds_when_disabled`
   (rationale above; revert is a one-liner if the original name is preferred).
2. `TestStreamingTorchFallbackPerChunkSeeds::test_default_flag_delivers_derived_seeds`
   intentionally does NOT pass `seed_lock_chunks` (relying on the default `False`)
   so its Gate-A failure is the seed-value assertion rather than a TypeError; the
   `True` sibling passes the kwarg explicitly and TypeErrors at RED as expected.
3. The global `ruff` (0.16.7) was used because the mlx env lacks the ruff module —
   same config (`.ruff.toml`), same targets.
4. `.sdd-brief-3a.md` and this report are intentionally left untracked; only the
   test file is in commit `b33cbf52`.

---

# GREEN — review round applied and implementation landed

## Review round (Important + Minors) — commit `bd0a2dfd`

`test: tighten Gate A polarity and identity pins (review round)`

- **Important finding closed:** `TestStreamingCallSitesForwardFlag` refactored
  into `_forwarded_via_http/_forwarded_via_ws(seed_lock_chunks)` helpers plus
  FOUR polarity tests (http/ws x True/False), all asserting with `assertIs`.
  Verified RED at the pre-fix code — all four failed with
  `AssertionError: None is not True/False : ... must forward ... verbatim`
  (call sites passed no kwarg → `.get()` → None). A call site hardcoding
  `True` can no longer pass while mis-forwarding False.
- **Minor 1:** `test_false_flag_seeds_and_delivers_per_chunk_seeds` now binds
  the caller's `gen_params` and adds `assertIsNot(params, gen_params)` for
  every delivered copy. (Its standalone RED is subsumed by the earlier
  `call_count 0 != 3` assertion in the same test; at the pre-fix code the
  caller's own object was delivered, so the identity pin also fails there.)
- **Minor 2:** `test_none_base_seed_no_seeding_flag_true` additionally pins the
  delivered copies carry `"seed": None` (guards `{"seed": 0}` = None + index).
  Passes at pre-fix code by design (pin for the GREEN implementation).

Strengthened RED: **15 failed / 14 passed** (was 13/14).

## GREEN implementation — commit `9b76c019`

`fix(engine,server): make seed_lock_chunks control per-chunk seeds (Step 3A)`

Exactly the brief's prescribed shape; 3 files, +59/−7:

1. `qwen3_tts/core/engine/inference.py`
   - `_chunk_seed(base_seed, chunk_index, seed_lock_chunks) -> int | None`
     added in the Seed-helpers section next to `_set_seed_for_backend`.
   - `run_inference` multi-chunk loop: `base_seed = gen_params.get("seed")`;
     per chunk `effective_seed = _chunk_seed(...)`; the existing loop-level
     `_set_seed_for_backend` call now receives the effective seed (guarded by
     `is not None`, unchanged shape); `_run_inference_single` receives
     `chunk_params = {**gen_params, "seed": effective_seed}` (new dict,
     caller never mutated — the intentional double-seed with an identical
     value is documented in-code). Single-chunk path untouched.
   - `run_inference_streaming`: `seed_lock_chunks: bool = False` added to the
     signature (keyword, defaulted, documented in the Args block). BOTH the
     MLX branch (via `_run_inference_mlx_streaming`) and the torch-fallback
     branch (via `_run_inference_single`) build the per-chunk
     `{**gen_params, "seed": _chunk_seed(...)}` copy. No new explicit
     `_set_seed_for_backend` call in streaming (call-count seam preserved).
2. `qwen3_tts/server/app_generation.py` — the `/generate-stream` call now
   passes `seed_lock_chunks=req.seed_lock_chunks`.
3. `qwen3_tts/server/websocket.py` — the `/ws` call now passes
   `seed_lock_chunks=req.seed_lock_chunks`.

WS4.5 untouched per ruling: `torch.manual_seed` stays; upstream drops
`generator=` (evidence above and in the Gate A section).

## GREEN verification

- `conda run -n qwen3-tts-mlx python -m pytest tests/test_seed_lock_chunks.py` → **29 passed** (all RED tests now green, both protected tests still untouched and green)
- Combined brief verify: `pytest tests/test_seed_lock_chunks.py tests/test_engine_streaming.py` → **41 passed**
- Batch runner (touched files' owning batches), backgrounded:
  batch 3 Server Infrastructure → **901 tests, passed**;
  batch 4 Engine & UI → **845 tests, passed**
- `ruff check qwen3_tts tests` (global ruff 0.16.7 — mlx env has no ruff module) → **All checks passed**
- `mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface` (global mypy 2.1.0 — mlx env has no mypy module either) → **Success: no issues found in 58 source files**
- No attribution trailers in any commit (grepped `b33cbf52`, `bd0a2dfd`, `9b76c019`)

## Commits (this branch, on top of `e12bf21d`)

| SHA | Type | Contents |
|---|---|---|
| `b33cbf52` | test | Gate A RED tests |
| `bd0a2dfd` | test | review-round strengthenings (RED 15F/14P) |
| `9b76c019` | fix | Step 3A GREEN implementation |

Not pushed; no PR; brief and this report remain untracked.
