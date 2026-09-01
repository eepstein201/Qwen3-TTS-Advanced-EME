# Phase 2c — `/load-model` in-flight dedup: TDD evidence

Branch `fix/phase2c-load-model-dedup` off `main` @ `c0314a9`. Plan: `~/.claude/plans/peppy-moseying-diffie.md` (r3, approved 2026-09-01 after a 5-agent review round on r2).

## RED (tests first, before `model_loading.py` existed)

Run against `main`'s code (`.venv-310`, torchless CI proxy): **17 failed, 3 passed, 4 subtests passed.**

Genuine assertion-level REDs (defect demonstrated live, not import noise):

| Test | Failure against main |
|---|---|
| `test_two_concurrent_loads_call_load_model_once` | `['clone', 'clone'] != ['clone']` — the double weight construction, reproduced |
| `test_three_claimants_still_build_once` | same double-build signature |
| `test_warmup_throw_keeps_model_and_reports_warmup_failed` | `HTTPException 500 load_failed 'warmup boom'` — the W1 discard, live |
| `test_warmup_failure_reaches_attached_waiter` | `['design', 'design'] != ['design']` |
| `test_waiter_empty_error_maps_to_retryable_503` | `500 != 503` |
| `test_completed_unload_bumps_epoch` | `0 != 1` |
| both `_error_payload` unwrap tests | `'_error_payload' not found in source` |

The 3 passing-on-main tests are the documented vacuous-until-stubbed shapes (M6 subtests; M4's message flow), made load-bearing by the mutation matrix below.

## GREEN

`.venv-310` over the six touched/adjacent modules (`test_issue214_load_model_dedup`, `test_response_contracts`, `test_models_loading_flag`, `test_model_swap_recovery`, `test_issue192_warmup_serialization`, `test_fastapi_server`): **122 passed**. Batch 3: **Ran 699 tests — 1/1 batches passed.** Full non-e2e: **3029 passed, 17 failed** — all 17 byte-identical to `main`'s set (proven: `git stash` → same 17 failures on main's code → `stash pop`); torch-dependent tests erroring in the torchless proxy, tracked separately.

## Gate A — mutation matrix (per-mutant wrong-stub discipline)

Each mutant was injected into the real implementation, the killing test run against it (must FAIL), and the file restored:

| # | Mutant injected | Killing test | Result |
|---|---|---|---|
| M1 | claim CAS dropped (`if False:`) | two-concurrent-loads | KILLED |
| M2 | `with MODEL_LOAD_LOCK:` → `if True:` | claim-blocks-while-held (OS thread) | KILLED |
| M3 | waiter returns `WaitResult.OK` without waiting | waiter-returns-only-after-load-finished | KILLED |
| M4 | waiter classifies every outcome OK | waiter-gets-owner-failure-message | KILLED |
| M5 | loader fabricates ownership, skips claim | loader-waits-for-http-owned-load | KILLED |
| M6 | release conditioned on `outcome is OK` | slot-cleared-on-each-except-path | KILLED |
| M7 | empty-error guard dropped (`and record.error` removed) | empty-error-maps-to-retryable-503 | KILLED |
| M8 | loader ignores ATTACH (`if False:`) | ready-gate-holds-while-http-load-in-flight | KILLED |
| M9 | `if current.epoch == epoch:` → `if True:` | update-config-bumps-epoch-and-caller-reclaims | KILLED |
| M10 | `warmup_failed = True` → `False` (flag hidden) | warmup-throw-keeps-model-and-reports | KILLED |

**10/10 killed.** Plus the RED-against-main evidence, which is the strongest possible demonstration for the discard-style mutants (M1/M10 kill against the actual historical defect, not a synthetic one).

## Santa Reviewer B round (python-reviewer) — PASS, 0 CRITICAL; 3 MED addressed

B independently re-derived and confirmed the transient 500-rewrap defect (fixed during Reviewer A's round) via a live probe, and confirmed the Timer-race in the disconnect test (fixed). Remaining findings:

| Finding | Verdict | Disposition |
|---|---|---|
| MED Failure paths stomp a superseder's CANCELLED — a waiter parked behind the superseder's `done.set()` but scheduled after the stale owner's `release(FAILED)` would nondeterministically mirror a 500 for an invalidated load | REAL | `release_model_load` is now **first-terminal-writer-wins** under `MODEL_LOAD_LOCK`: a CANCELLED classification is never overwritten by a later non-CANCELLED release (done + slot clear still run). Killing test: `TestReleaseFirstWriterWins` |
| MED `test_response_contracts` coalesced round-trip started its thread before the patch context (worst case: a real multi-GB engine load in a unit test) | REAL (latent — 6/6 probes lost the race) | Thread start moved inside the patch context, matching the documented pattern |
| MED ARCHITECTURE.md Phase 2c entry described the pre-fix design (no lazy-supersede, no owner re-check, no discard-503, no `code`/`recovery`, stale mutant count) | REAL | Entry rewritten; /ready 14.5-min ops residual added |

LOWs accepted/fixed: dead `_LoadRecord.ok` removed; `_safe()` now strips CR/LF (restoring the log-injection defense); `_recover_from_failed_load` under the lock accepted (bounded, failure-only); untyped `state` contract accepted (mypy `--check-untyped-defs` clean). B verified sound: ABA/stale-record, epoch-guard atomicity (the old `/unload-model` resurrection window is closed), lock ordering (never held across an await, never nested), release-in-`finally` completeness, and the test harness quality.

**Final verification after both rounds:** seven modules **141 passed** · full non-e2e **3034 passed / 17 pre-existing (byte-identical to main)** · batch 3 **704 tests passed** · ruff/mypy clean · bandit 0 HIGH · CLAUDE.md 293/300.

## Santa Reviewer A round (fastapi-reviewer) — PASS, 0 CRITICAL; 2 HIGH + 4 MED + 3 LOW, all addressed

| Finding | Verdict | Disposition |
|---|---|---|
| **HIGH** Supersede is LAZY (fires only at the next claim) — an in-flight owner finishing after an epoch bump with no intervening claimant installs old-config weights and answers 200 | REAL | Guard widened to `_is_superseded` (record CANCELLED **or** epoch mismatch) under `MODEL_LOAD_LOCK`; mutators now bump the epoch under the same lock; a discarded load answers **503 `load_in_progress`**, never 200. Killing test: `test_in_flight_load_discarded_when_epoch_bumps_mid_flight` |
| **HIGH** `_background_load` has no already-loaded fast path — with the new H2 wait, a user's HTTP load racing multi-model startup gets built a second time | REAL (shape pre-existing, cheap fix in touched code) | Loader skips (and logs) any model already in `state.models`; release stays correct |
| MED Superseded owner's failure path nulls a NEWER load via unconditional `_recover_from_failed_load` + stale `/health` error | REAL | Recovery + `model_load_errors` write gated on `not _is_superseded` |
| MED The moved body dropped the old `logger.error(..., exc_info=True)` from HTTP load failures | REAL regression | Restored on all three except paths |
| MED Vacuous tests still assert the deleted `models_loading` dict; their `_make_state()` helpers carry no real record table | REAL | Both `_make_state()`s now build real `model_loads`/`model_config_epoch`; the flag assertions re-pointed at the released slot; stale docstring line fixed |
| MED The new disconnect plumbing was untested (an always-True probe would 503 every duplicate) | REAL | `TestWaiterDisconnect`: always-True probe → immediate 503; always-False probe → normal attach + `deduped` |
| LOW waiter's 500 always said `recovery: retry` (config failures invited retry loops) | REAL | `_LoadRecord` carries `code`/`recovery`; owner passes them; waiter mirrors them |
| LOW `_safe()` redacted non-path slashes (`layers 1/2` → `<path>`) | REAL | Pattern now requires a non-word char before `/` |
| LOW `/unload-model`'s 409 used a plain-string `detail` vs the structured family | REAL | 409 routed through `_error_response(409, "load_in_progress", …)` |

One regression caught by the tests during this round, in my own fix: the first discard path raised `_error_response(503)` **inside** the `try`, so the catch-all re-wrapped it as a 500 `unknown_error` — moved after the `finally`, with `outcome` pinned to CANCELLED so release never stomps the superseder's classification. Full suite re-run after the round: **3033 passed / 17 pre-existing**, batch 3 **703 tests**, bandit 0 HIGH.

## Cross-family review round (agy, `gemini-3.1-pro-high`) — 3 findings, all verified real, all fixed

| Finding | Verdict | Fix | Killing test added |
|---|---|---|---|
| Superseded owner's unconditional `state.models[t] = model` clobbers the newer-config model after an epoch bump | REAL (the half of C1 the plan had scoped out) | Owner assigns under `MODEL_LOAD_LOCK` only while `record.outcome is not CANCELLED` (supersede marks the record; check+assign atomic vs supersede) | M9 extended: superseded owner finishes last → `models[t]` must still be the epoch-1 model |
| Claim-table imports sat BEFORE the `try` — a broken-install ImportError leaked the claim (870 s → 503 forever) | REAL (exact startup-wedge class) | Imports moved to the FIRST statements inside the `try` (preserves the #192 pre-lock binding seam) | Existing M6 matrix pins release-on-every-path |
| `_background_load`'s release dropped `error_msg` — waiters on a failed startup load got the generic 503 | REAL | Release passes `error=_sanitize_error(error_msg)` | New: waiter-on-failed-startup-load asserts the cause in `detail` |

Re-mutation of the two behavioral fixes: **M11 assign-guard-dropped KILLED, M12 loader-error-dropped KILLED — 12/12 total.** One regression introduced and caught during the fix: the first guard predicate read the slot back through `state.model_loads` (a write Mock states don't persist) and broke three shipped #192 tests — replaced with the record-outcome predicate, which is state-agnostic and equally atomic.

## Live smoke (PM2/CLI server, MLX, model cache warm)

`two concurrent POST /load-model design` against a restarted server:

* `{"status":"loaded","model":"design"}` and `{"status":"loaded","model":"design","deduped":true}`
* `.voice_server.log`: exactly **one** `Loading design (...)` → `Loaded design model in 1.6s` pair — no second weight construction.

## Implementation notes discovered during GREEN

* The #192 warm-up patch seam requires importing `_warmup_model`/`_warmup_disabled` **before any await** (the old handler's exact shape) — a call-time indirection re-resolves after the #192 tests' patch window closes and their post-lock `wait_for` runs the real warm-up. Caught by `test_warmup_deferred_while_generation_holds_lock`.
* The python-review-fixes guard (`no success dict on ImportError with a mocked no-raise `_error_response``) is a real fall-through hazard: the moved owner body originally dropped the explicit `return None` guards; they are restored, and the test's patch seam moved with the code to `qwen3_tts.server.model_loading._error_response`.
* conftest `_init_app_state` gained `model_loads` + `model_config_epoch` (the record table is part of the minimal test-state contract now).

## Gates

ruff clean · mypy clean (56 files) · bandit 0 HIGH (disconnect probe `# nosec B110` documented) · CLAUDE.md 293/300 · batches-coverage gate green.

## CI round on PR #233 — bare-State AttributeError, caught by CI, not by local pytest

PR #233's first push red'd **every** Tests matrix leg + docker: `AttributeError:
'State' object has no attribute 'model_loads'` in `tests/test_voice_server.py::
TestLoadModelEndpoint` (batch 2). The two tests pass under pytest (the conftest
initializes the new attrs) but the batch runner and the docker step run bare
unittest with **no conftest** — a harness dimension the pytest-only local gates
cannot see. Fix (`fe71397`): `claim_model_load` lazily creates the table (the
gate is the module-scope lock, never consulted on state — defensive-read pattern,
not fail-open), `release_model_load` tolerates its absence, and
`TestBareStateRegression` pins the bare-State path. Reproduced locally first via
`python -m unittest tests.test_voice_server.TestLoadModelEndpoint` (the exact CI
harness), fixed, re-verified: that suite OK, batch 3 at 705 passed, seven-module
set 141+ green. Lesson recorded in memory: after touching server handlers that
read new state attrs, run the owning batch via the batch runner, not just pytest.
