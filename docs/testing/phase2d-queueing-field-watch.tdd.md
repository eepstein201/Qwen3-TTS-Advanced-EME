# Phase 2d + Item 5 + T5 — TDD evidence

Branch `fix/phase2d-queueing-and-field-watch` (closes #214). Plan:
`~/.claude/plans/peppy-moseying-diffie.md` (r4, team-built: 3 explorers + 3
reviewers).

## Deliverable 1 — T5 (`/unload-model` queued-generation window)

### RED (commit `8ff88ab`, + Gate A fixes `9f43dae`, `de84d23`)

`tests/test_issue214_unload_queued_window.py`: 14→15 tests, **all genuinely
RED against unfixed `main` under BOTH pytest and bare unittest (no
conftest)**. Failure reasons verified one-by-one: lying-200 while the lock is
held; `async with` contexts `NONE`; `acquire_calls == 0`; stale
`active=[True]` / `[True, True]` at lock grant; `HTTPException not raised`
(no re-read); `timeout='10'`; UI borrows the load constant.

Gate A (tests-only, implementation does not exist): agy round-1 **FAIL**
(2 CRITICAL — 503 test passable by a pre-lock re-read; UI pin
docstring-defeatable) → fixed; python-reviewer **PASS** (WS leg silently
skipped — `websocket_endpoint` does not exist, real target
`_stream_generation`; ruff I001; dead `inference_calls`; first-match AST
weaknesses; gen_cache temp-file leak) → fixed; fastapi-reviewer **PASS**
(enclosure pins order-insensitive → `inference_calls == []` asserted; UI
pin rejects the toggle_asr house shape → branch-shape accepted; plain-tuple
`client` vs starlette<1.6 → `Address`) → fixed; agy round-2 **FAIL**
(first-match and last-assignment AST weaknesses; streaming/WS
structural-only) → fixed: all-instances AST guards, streaming behavioral
twin (drives the response body_iterator; streaming inference recorder never
called), WS stays structural-only (shared proven helper) — the
double-acquire restructure residual is documented in ARCHITECTURE.md.

### GREEN (commit `ce1b7a6`)

15/15 under pytest AND bare unittest. Five mutants killed (each applied
temporarily, owning tests failed, reverted):

| Mutant | Killing tests |
|---|---|
| M1: revert the route lock | all 3 `TestUnloadModelRouteSerializesOnInferenceLock` |
| M2: disable the final-item in-lock reset | both batch-reset tests (observed `[True]` / `[True, True]`) |
| M3: re-read moved AFTER inference (inside lock) | 503 behavioral test (`inference_calls` non-empty) |
| M4: route-level pre-lock `already_unloaded` short-circuit | `test_already_unloaded_still_acquires_the_lock` (`acquire_calls == 0`) |
| M5: client `timeout=10` literal | `test_tts_client_unload_model_uses_the_constant` |

Regression: batch 3 via the batch runner (bare unittest) green; 107
neighboring tests (state-ownership, response contracts, streaming lifecycle,
load-model dedup, websocket) green.

## Deliverable 2 — e2e queuing tier (commit `1f538b8`)

`tests/test_e2e_queueing.py`: 4 opt-in wire-level tests; module preflight
proves rate limiting disabled (skips otherwise); ≥0.6 s polls; ≥900 s
model-op timeouts; uuid texts defeat the generation cache; prompt created
then deleted in `finally`. Registered in `INTENTIONALLY_UNBATCHED`.

Live run: see PR body (CI does not execute this tier — the manual run in
the PR body is the only execution evidence, per the architect condition).

## Deliverable 3 — docs

- `ARCHITECTURE.md`: `### /unload-model leaf lock (T5)` +
  `### Cap-warning field watch (#214 item 5)`; the stale clause at the end
  of the 2c section replaced (it had mis-numbered T5 as "item 4" and
  mis-diagnosed the fix as "needs a mode-aware claim" — the leaf lock
  supersedes that).
- `CLAUDE.md` 242/300 (net-zero inline appends): Log-level sentence gains
  `/unload-model`+`/update-model-config` in the enumeration + the item-5
  grep clause; `app_models.py` row notes the T5 lock.
- #214 comment: posted at PR-open with the item-5 grep correction
  (`_warn_if_cap_reached` is a function name — greps to nothing; the
  greppable string is `"token cap without emitting EOS"`; MLX-only; include
  `.voice_server.log.1`; torch invisible to this watch; baseline hit
  2026-08-24 predates the closed pairs).
