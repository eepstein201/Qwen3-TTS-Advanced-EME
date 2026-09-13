# TDD Evidence: Step 4A — un-hollowed E2E model-load tests + frozen Manage Models Dataframe findings

**Source plan:** Step 4A of docs/plans/2026-09-06-consolidated-backlog-priority.plan.md

**Branch:** `fix/e2e-model-load-assertions` (cut from main @ `fe2999a1`). This is a
TEST-ONLY change (zero product code). The genuine product findings below are the
starting input for Step 4B.

**Date:** 2026-09-12 (live runs) — committed 2026-09-13.

## Scope of the test change (`tests/test_e2e_playwright.py`)

1. **Five server-side asserts (the non-negotiable gate).** Every previously-discarded
   `_wait_for_model_state()` return is now `assertTrue(...)` with a message naming the
   model and expected state, e.g. *"design model did not report loaded=True on /health
   within 60s of the UI Load click"*. Sites: test_08 load, test_09 setup-load +
   unload, test_10 load + unload.
2. **Five narrowed exception wrappers.** The `except Exception: pass` around each
   `wait_for_any_textarea_contains` wait now catches only
   `playwright.sync_api.TimeoutError` (module-top alias `PlaywrightTimeoutError`;
   class is `skipUnless(HAS_PLAYWRIGHT)`-gated). Rationale: the status textarea
   lagging is cosmetic because the `/health` assert is the authoritative gate; any
   other exception (dead page, broken wait machinery) propagates and fails the test.
   The per-poll `try/except` inside `_wait_for_model_state` / `_ensure_model_unloaded`
   (correct polling hygiene) and every unrelated swallow-all in the file are untouched.
3. **Three table checks — hardened, then reverted by RECORDED DEVIATION.** All three
   `wait_for_table_row_refreshed` checks were converted to hard asserts per the plan,
   failed identically in two consecutive live runs (see evidence), and were reverted
   to warn-and-continue per the plan's own deviation clause. The warn text now cites
   `/health` (not `/models`) as the verification source, and
   `wait_for_table_row_refreshed`'s docstring points here. The five server-side
   asserts remain hard regardless.

## Live-run evidence

All runs: `conda run -n qwen3-tts-mlx python -m pytest tests/test_e2e_playwright.py
-m e2e -k "load_model or load_unload_cycle" -v --tb=short` against the PM2-managed
server (`tts-server-5123`). Idle verified via the public `/generation-status`
(`{'active': False, ...}`) before each run. Server restarts via
`conda run -n qwen3-tts-mlx tts server restart`.

Environment note for 4B: the MAIN checkout's `config.json` (uncommitted local
modification; the PM2 server reads it) sets `load_at_startup: true` for ALL THREE
models — every restart comes up with clone+design+custom resident (~7.5 GB of MLX
8-bit weights on the 16 GB M2 Pro). The committed config keeps design/custom
on-demand. Relevant when assessing memory pressure in 4B's load/unload cycling.

| Run | Server state | test_08 | test_09 | test_10 |
|-----|--------------|---------|---------|---------|
| baseline (file unchanged) | fresh restart, idle | PASSED (hollow) | PASSED (hollow) | PASSED (hollow) — 211.62s |
| run 1 (hardened form) | post-baseline, idle | FAILED: table assert | PASSED | FAILED: table assert |
| run 2 (hardened form) | fresh restart, idle | FAILED: same table assert | PASSED | FAILED: Refresh-button `Locator.click` timeout |
| run 3 (deviation form) | fresh restart, idle | PASSED | PASSED | PASSED — 211.30s, exit 0 |

In runs 1–3, every `_wait_for_model_state` assert PASSED: the server-side load and
unload paths execute correctly and promptly (design loads in ~2 s warm; server log
`Loaded design model in 1.6–2.8s`, `Warm-up complete` within ~1 s).

Run 1/2 failure text (identical both runs, both tests):

```
AssertionError: False is not true : Manage Models table did not re-render 'Loaded'
despite server confirmation of the design load
```

i.e. `/health` says `design_model_loaded: true`, `wait_for_table_row_refreshed`
waited 3×30 s (including its two Refresh-button retries, which clicked successfully
in run 1) and the DOM never contained the design row with "Loaded".

Run 2 test_10 additionally died mid-helper (did not recur in runs 1/3):

```
playwright._impl._errors.TimeoutError: Locator.click: Timeout 30000ms exceeded.
Call log:
  - waiting for locator("div[role='tabpanel']:visible").first.locator("button")
        .filter(has_text="Refresh").first
```

No visible tabpanel could be found ~30 s after the load completed — the page/panel
state degraded mid-test. Root cause unknown; recorded for 4B, deliberately not coded
around (a dead page must fail the test loudly, not warn).

## DOM ground truth (the deviation's evidence)

Direct probe with chrome-devtools against a manually launched UI subprocess (same
launch command as the test's `setUpClass`, same worktree code, **gradio 6.20.0**),
server state design=loaded:

- The visible Manage Models tabpanel's table renders exactly **ONE row**:
  `["design", "Not loaded", "2500MB", "Yes"]`. The clone/custom/asr rows that
  `get_model_table_data()` returns are absent from the rendered DOM.
- `GET /models` (authed) at the same moment: `models.design = {"loaded": true,
  "loading": false, "memory_mb": 2500, "load_time_sec": 2.8, ...}`, and `/health`
  agrees — the backend data is fresh and correct; the rendered row is stale.
- The 5 s status-Timer self-heal (`_facade.py:349-356`, `status_timer.tick(fn=
  get_model_table_data, outputs=model_table)`) IS running — the page's network log
  shows continuous `queue/join` → `queue/data` SSE cycling (222 requests observed).
  The frontend drops every delivered table update.
- **Server-side unload → load cycle while watching the DOM** (bounded 20 s + 30 s
  waits, page focused, `visibilityState: "visible"`): the row stayed
  `["design", "Not loaded", "2500MB", "Yes"]` throughout. The memory cell alone
  proves staleness: while design was UNLOADED the row still read "2500MB", where a
  fresh fetch renders "—" (`memory_mb=0`).
- Browser console: no JS crash (no RangeError/Svelte errors — unlike the 6.14
  Dataframe recursion). One `net::ERR_NETWORK_IO_SUSPENDED` resource error appeared
  while the probe page was backgrounded; the focused-page re-check above rules the
  backgrounding out as the cause of the freeze. (Caveat recorded for honesty: the
  probe's Refresh click was a JS-dispatched synthetic event, which gradio may
  ignore — but the pytest runs used real trusted Playwright clicks, including two
  successful Refresh clicks in run 1, and saw the same freeze.)

**Conclusion:** the I4-documented "dropped Dataframe re-render" quirk is not an
occasional dropped render — on gradio 6.20.0 the Manage Models `gr.Dataframe`
renders one stale row and NEVER re-renders, not from the toggle handler's returned
value, not from the Refresh button's output, and not from the 5 s self-heal Timer.
This is a genuine UI defect (frontend), orthogonal to the healthy server-side
load/unload path.

## Failure catalog for Step 4B

1. **Frozen Manage Models Dataframe (CONFIRMED, persistent).** Evidence above.
   Server state correct; UI table frozen at a stale row. Fix directions to
   investigate: gradio 6.20 Dataframe value-update path (possibly interacts with
   `wrap=True`/`interactive=False`/row-count changes — note the rendered table has
   FEWER rows than the data, so the component may be diffing against a phantom
   initial state, cf. `get_table_data()`'s phantom-row dedup note), or replace the
   table with a non-Dataframe component. Reproduces without Playwright (manual UI +
   chrome-devtools), so 4B does not need the E2E harness to iterate.
2. **Intermittent page/panel degradation mid-test (OBSERVED ONCE).** Run 2 test_10:
   no visible tabpanel for 30 s after a successful load. Possibly the same frontend
   malfunction progressing (freeze → dead panel), possibly unrelated. Not reproducible
   on demand (runs 1/3 clean).
3. **Server-side load/unload health at 1-cycle scope (GOOD NEWS, constrains 4B).**
   All five server-side asserts green in three runs; no load/unload request ever
   failed; no server death at this scope. The original "server dies mid-load after
   several cycles" hypothesis remains untested at ≥5 cycles — that is 4B's mandate.
   Note the 3-models-resident startup state (see environment note) when designing
   the memory-trend capture.

## Gates

- `ruff check tests/test_e2e_playwright.py` → "All checks passed!" (exit 0), run
  after the final edit.
- mypy/bandit: not applicable (tests excluded from both).
- Batch 6 (`python tests/run_batches.py --batch 6`): dispatcher-owned per
  coordinator directive — not executed in this lane. The lane's accidentally-started
  instance was killed, its orphaned UI subprocesses (port 7866) cleaned, and the
  server left idle and healthy (`/generation-status active: False`, all three
  models loaded) for the dispatcher's run.
