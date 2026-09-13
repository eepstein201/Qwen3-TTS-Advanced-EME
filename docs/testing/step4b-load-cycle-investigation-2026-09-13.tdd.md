# TDD Evidence: Step 4B — repeated model load/unload cycling vs. the server-death hypothesis

**Source plan:** Step 4B of docs/plans/2026-09-06-consolidated-backlog-priority.plan.md

**Branch:** `fix/e2e-load-cycle-investigation` (cut from main @ `0617c2b8`). Docs-only
lane — no product or test code changed; the deliverables are the verdict, the
evidence, and two tracked-issue drafts.

**Date:** 2026-09-13 (live campaign 11:29–11:36 EDT).

**Hypothesis under test:** the server dies mid-load after several load/unload
cycles, suspected memory pressure on the 16 GB M2 Pro (historical observation:
`.voice_server.log` ends mid-load). Step 4A proved the 1-cycle path healthy;
the ≥5-cycle death hypothesis was untested.

## Protocol

- **Target:** the live PM2 server `tts-server-5123` (`http://127.0.0.1:5123`),
  running main-checkout code @ `0617c2b8` — identical to the worktree base, so
  the campaign exercised exactly the code under change.
- **Residency observed (not assumed):** the MAIN checkout's uncommitted
  `config.json` sets `models.{clone,design,custom}.load_at_startup: true`
  (backend mlx, quant 8bit) — every restart comes up all-three-resident
  (~9.5 GB footprint). Same environment note as the 4A doc.
- **Prep:** `conda run -n qwen3-tts-mlx tts server restart` (PM2
  `restart_time` 26→27, expected); `/ready` 200 and `/generation-status`
  `active: false` confirmed before the campaign; server log byte offset
  188366 recorded for per-cycle segment extraction.
- **Campaign script:** `/tmp/4b_cycle.py` (pure stdlib; tooling, not repo
  code). Per transition: `POST /load-model` / `POST /unload-model`
  (`{"model_type": ...}`, Bearer-authed), poll `/health` until the model
  reports the target state (180 s bound), settle 10 s, then capture load
  latency, process footprint, `/stats` MLX memory fields, and system free %.
  12 s between cycles; HTTP 429/5xx tolerated with backoff (never hit).
  Results written incrementally to `/tmp/4b_results.json`.
- **Sequence:** design 1 initial unload + **5 full load→unload cycles**;
  custom 1 initial unload + **5 full load→unload cycles**; clone **3
  unload→reload cycles**; then an unmeasured restore of full residency.
  The initial unloads exist because design/custom start resident under the
  live config — without them, cycle 1's "load" would be a no-op attach, not
  a full weight construction.
- **Memory metric = `footprint <pid>`, NOT `ps -o rss`.** On this macOS +
  MLX setup `ps rss` reports ~15.7 MB for a process whose true physical
  footprint is 9,549 MB — the model weights live in IOAccelerator (Metal)
  memory that `ps rss` does not count. `footprint` reports it (baseline
  detail: 4648 MB of the 5.1 GB clone-only footprint was IOAccelerator).
  `mlx_memory_active_mb` from `/stats` is MLX's own allocator view.

## Per-transition results

| event | model | cycle | latency_s | footprint_mb | mlx_active_mb | free_pct |
|---|---|---|---|---|---|---|
| baseline | – | – | – | 9524.0 | 9014.64 | 33% |
| unload | design | 0 | 0.8 | 9507.0 | 6017.45 | 31% |
| load | design | 1 | 4.7 | 9545.0 | 9014.59 | 28% |
| unload | design | 1 | 0.2 | 9513.0 | 6017.45 | 31% |
| load | design | 2 | 3.4 | 9900.0 | 9014.59 | 46% |
| unload | design | 2 | 0.5 | 9481.0 | 6017.45 | 72% |
| load | design | 3 | 0.8 | 9521.0 | 9014.59 | 50% |
| unload | design | 3 | 0.1 | 9491.0 | 6017.45 | 73% |
| load | design | 4 | 0.8 | 9497.0 | 9014.59 | 51% |
| unload | design | 4 | 0.1 | 9484.0 | 6017.45 | 73% |
| load | design | 5 | 0.7 | 9882.0 | 9014.59 | 48% |
| unload | design | 5 | 0.1 | 9485.0 | 6017.45 | 73% |
| unload | custom | 0 | 0.1 | 9458.0 | 3020.25 | 73% |
| load | custom | 1 | 1.8 | 9468.0 | 6017.39 | 68% |
| unload | custom | 1 | 0.1 | 9472.0 | 3020.25 | 73% |
| load | custom | 2 | 4.1 | 9501.0 | 6017.39 | 42% |
| unload | custom | 2 | 0.1 | 9472.0 | 3020.25 | 47% |
| load | custom | 3 | 1.4 | 9488.0 | 6017.39 | 45% |
| unload | custom | 3 | 0.1 | 9476.0 | 3020.25 | 48% |
| load | custom | 4 | 1.3 | 9494.0 | 6017.39 | 47% |
| unload | custom | 4 | 0.1 | 9483.0 | 3020.25 | 50% |
| load | custom | 5 | 1.3 | 9501.0 | 6017.39 | 46% |
| unload | custom | 5 | 0.1 | 9487.0 | 3020.25 | 40% |
| unload | clone | 1 | 0.1 | 9461.0 | 0.0 | 40% |
| load | clone | 1 | 1.5 | 9493.0 | 3020.2 | 32% |
| unload | clone | 2 | 0.2 | 9462.0 | 0.0 | 36% |
| load | clone | 2 | 2.6 | 9478.0 | 3020.2 | 33% |
| unload | clone | 3 | 0.1 | 9464.0 | 0.0 | 32% |
| load | clone | 3 | 7.2 | 9476.0 | 3020.2 | 31% |
| load | design | restore | 6.2 | 6513.0 | 6017.39 | 47% |
| load | custom | restore | 2.2 | 9557.0 | 9014.59 | 30% |
| final | – | – | – | 9557.0 | 9014.59 | 30% |

## Trend analysis (the leak signal)

- **Post-unload footprint is flat, not monotonic.** Design: 9507 → 9513 →
  9481 → 9491 → 9484 → 9485 MB (ends BELOW its start). Custom: 9458 → 9472 →
  9472 → 9476 → 9483 → 9487 MB (+29 MB over 5 cycles = 0.3%, inside noise).
  Clone: 9461 → 9462 → 9464 MB. Baseline→final: 9524 → 9557 MB (+33 MB).
- **MLX active-set post-unload is bit-identical every cycle** for a given
  residency: 6017.45 MB (clone+custom), 3020.25 MB (one model), 0.0 MB
  (all unloaded). Deterministic allocator behavior — zero leak in MLX's
  active set across 13 measured loads / 16 unloads.
- **No death.** PM2 `restart_time` 27 before AND after (the only bump was
  the pre-campaign CLI restart); server PID 75766 unchanged throughout;
  `/generation-status` idle; the campaign's `.voice_server.log` segment
  (from offset 188367, 57 lines) contains zero warnings/errors/tracebacks;
  script exit 0.
- **Load latencies IMPROVE across cycles** (design 4.7 → 0.7 s; custom
  4.1 → 1.3 s): MLX's buffer cache is reused on reload, not accumulated.
- **Footprint does not drop on unload** — 9.46 GB with ZERO models loaded
  (clone cycle 1). `unload_model_cleanup()` (`qwen3_tts/core/engine/asr.py`,
  MLX branch) runs only `gc.collect()`; the torch branch's
  `mps.empty_cache()` analogue (`mx.clear_cache()`) is absent, so freed
  weights stay in MLX's in-process Metal buffer cache. This cache is
  **purgeable and reused**: mid-restore the OS reclaimed ~3 GB of it when
  system free dipped (the 6513 MB sample with mlx_active 6017 MB), and warm
  reloads take 0.7 s. **Retained-but-reusable is not a leak** — see issue
  draft 2.

## Historical death forensics (what the "log ends mid-load" events actually were)

A scanner (`/tmp/4b_scan_deaths.py`) over both log rotations found 9
incomplete-load death signatures: 8 in `.voice_server.log.1` (Apr 13, May 9,
May 10, May 13, **May 18 design**, Jul 26, Jul 29, **Aug 1 design**) and 1
in the current log (**Sep 5, clone**). Classifying each by its surrounding
lines yields two mechanisms — neither is "plain load/unload cycling leaks
until death":

1. **Mechanism B (Sep 5 15:52, the most recent and the likely remembered
   observation) — operator restart storm during a PM2 upgrade.** The PM2
   daemon log shows PM2 upgraded 6.0.14 → 7.0.4 at 15:45:04, then ~8
   explicit stop/start cycles 15:36–15:51 (EVERY exit is preceded by
   "Stopping app" — zero unsolicited crash exits), each stop accompanied by
   "failed to kill - retrying in 100ms" ×3–6 and "process tree killed (3
   pids)". At 15:51:57 a fresh python began loading all three models; 5 s
   later (15:52:02) it logged "FastAPI server shutting down..." — a stop
   landed mid-startup-load — and never logged completion. The machine then
   ran SERVER-LESS for 15 minutes (next start 16:07:35, matching the app
   log's 16:07:36). The kernel unified log for 15:49–15:53 contains ZERO
   kill/memorystatus events — **definitively not a memory kill**.
2. **Mechanism A (Aug 1, Jul 26, May 18) — hard kills on the
   `/update-model-config` unload-all→reload-all path.** Each signature is
   "Model config updated: ... Models unloaded ... Loading <model>..." with
   NO subsequent line and no shutdown line (hard kill), then a fresh start
   26 s–weeks later. Aggravators visible in context: Jul 26 raced an active
   cancelled 7-chunk generation; Aug 1 shows a SECOND design load starting
   while the first was still in warm-up — the documented pre-#214
   concurrent-load double-allocation bug class (~2.5 GB extra on MLX), since
   fixed by the per-load records; May 18 sat amid a client hammering
   `POST /unload-model` with a missing token plus config churn. Kernel
   records for those dates are beyond unified-log retention (inconclusive),
   but the broader record shows genuine memory pressure in this regime: 420
   "Low memory: 1.0–2.0 GB available" warnings historically, all clustered
   on `/update-model-config` reload churn — including bf16 quantization
   cycling, where 3×bf16 ≈ 21 GB simply exceeds the 16 GB machine (custom
   bf16 load took 59.7 s = thrashing).
3. **Distractor ruled out:** the PM2 error log's April crashloop (41,692
   "Started server process" entries, 12 lines apart) is
   `[Errno 48] address already in use` — the double-start bug that predates
   the startup lock (`_acquire_startup_lock`); long fixed and unrelated to
   memory or cycling.

## Verdict

**(c) No death and no leak at the ≥5-cycle scope.** 13 measured loads / 16
unloads across design (5 cycles), custom (5 cycles), and clone (3 cycles)
left footprint flat (±30 MB), MLX active-set bit-identical per residency,
PM2 restart_time unchanged, and the log segment clean. The scope that WOULD
kill this server is different and now enumerated with evidence: (i) 3×bf16
residency (arithmetic: ~21 GB > 16 GB RAM — the `/update-model-config`
bf16 churn regime behind the 420 low-memory warnings), (ii) the pre-#214
concurrent-load double-allocation race (fixed), or (iii) an operator
restart landing mid-startup-load (Sep 5 — graceful-shutdown signature,
kernel-clean, 15-min outage from the restart storm itself, not memory).
The historical "server dies mid-load after several load/unload cycles"
observation is thereby root-caused as a **misattribution**: the deaths
co-occurred with cycling sessions but were caused by mechanisms A/B, not by
the load/unload path leaking.

## Disposition — tracked-issue drafts (coordinator files; do not `gh create` here)

### Issue draft 1 — not filed as an issue

The verdict itself (mechanisms A and B above, with this doc as evidence)
satisfies 4B's "root-caused and filed as a tracked issue with evidence"
exit criterion; no separate issue is needed for a disconfirmed hypothesis.
Residual follow-up worth a line somewhere if bf16 cycling is ever a
workflow: `/update-model-config` does not gate quantization choices on
available memory (3×bf16 > 16 GB) — candidate guard, LOW priority.

### Issue draft 2 — MLX unload never returns memory to the OS

**Title:** `unload_model_cleanup()` MLX branch lacks `mx.clear_cache()` —
unloaded models stay resident in the process footprint (9.5 GB held with 0
models loaded)

**Body:** On the MLX backend, `POST /unload-model` nulls the slot and runs
`unload_model_cleanup()` (`qwen3_tts/core/engine/asr.py:328`), whose MLX
branch performs only `gc.collect()` — the torch branch's equivalent
(`torch.mps.empty_cache()`) has no MLX counterpart. Measured (Step 4B
campaign, 2026-09-13, mlx 0.32.2): unloading ALL THREE models drops
`mlx_memory_active_mb` to 0.0 but the process footprint stays ~9.46 GB.
The memory sits in MLX's in-process Metal buffer cache: it IS reused by
the next load (warm reloads 0.7 s vs 4.7 s cold) and IS purgeable by the
OS under pressure (a mid-campaign sample showed the OS reclaiming ~3 GB of
it when system free dipped to ~31%). So this is NOT a leak — but "unload
to free memory" does not promptly return memory to the system, which on a
16 GB machine both misleads memory-pressure diagnosis (see the 4B
hypothesis history) and withholds headroom from co-resident apps until the
OS decides to purge.

**Fix direction:** call `mx.clear_cache()` (optionally with
`mx.reset_peak_memory()`) in the MLX branch of `unload_model_cleanup()`,
available since mlx 0.32.x (verified `hasattr(mx, 'clear_cache') == True`).
Tradeoff to weigh: clearing on every unload sacrifices the warm-reload
latency win; consider clearing only when it would actually help (e.g. on
explicit unload when system free % is low, or expose a `/free-memory`
knob) rather than unconditionally.

**Repro:** restart the server, `POST /unload-model` for all three types,
then `footprint <server_pid>` — footprint stays ~9.5 GB while
`/stats` reports `mlx_memory_active_mb: 0.0`.

**Severity:** LOW–MEDIUM (resource-return behavior; no crash, no leak).

### Issue draft 3 — frozen Manage Models Dataframe (4A findings 1+2, carried)

**Title:** Manage Models `gr.Dataframe` frozen at one stale row on gradio
6.20.0 — never re-renders despite correct server state and a live 5 s
self-heal Timer

**Body:** Full DOM evidence is in
`docs/testing/step4a-e2e-unhollow-2026-09-12.tdd.md` ("DOM ground truth"
section): the table renders exactly one stale row
(`design | Not loaded | 2500MB | Yes`), missing clone/custom/asr rows,
never updating across a server-side unload→load cycle, while `/models` and
`/health` report correct fresh state and the 5 s status-Timer's SSE
(`queue/join` → `queue/data`) flows continuously (222 requests observed) —
the frontend drops every delivered update. No JS console crash. Related
symptom (4A finding 2, observed once, not reproducible on demand): the
tabpanel itself disappearing ~30 s after a load ("no visible tabpanel",
`Locator.click` timeout). Fix directions: gradio 6.20 Dataframe
value-update path (possibly interacting with `wrap=True` /
`interactive=False` / row-count changes — the rendered table has FEWER
rows than the data, cf. `get_table_data()`'s phantom-row dedup note), or
replace the table with a non-Dataframe component. Reproduces without
Playwright (manual UI + devtools).

**Connection to server memory state (asked by the 4B brief): NONE FOUND.**
The 4B campaign shows the server-side model state and memory are healthy
and stable across load/unload cycling (flat footprint, clean logs), and 4A
showed the freeze with the server state verifiably correct — the defect is
entirely frontend.

## Deviations from the 4B brief

1. **Initial unload before counted cycles** (design/custom start resident
   under the live uncommitted config) — otherwise cycle 1's "load" would be
   a no-op attach and the campaign would be hollow at its core.
2. **`ps rss` replaced by `footprint`** for the RSS capture — `ps rss`
   undercounts Metal/IOAccelerator memory ~600× (15.7 MB vs 9,549 MB for
   the same process), which would have made the entire memory-trend
   measurement meaningless.
3. **Kernel `log show` checks:** Sep-5 window conclusive (zero kill
   events); May/Jul/Aug-1 windows are beyond unified-log retention —
   mechanism-A classification rests on the app-log signatures (no shutdown
   lines = hard kill) plus the documented pre-#214 race.
4. **No second server instance on :5125** — no product fix was warranted,
   so there was nothing to verify against worktree code.
5. **Cap-warning watch:** `grep -n "token cap without emitting EOS"
   .voice_server.log*` — all 15 hits pre-date the campaign (latest
   2026-09-10, generation-time clone/custom warnings with the known
   below-24 kHz-reference cause); zero during cycling. Not a 4B lead.

## Reproduction commands

```bash
# prep
pm2 jlist                       # record pid + restart_time
conda run -n qwen3-tts-mlx tts server restart
python3 /tmp/4b_poll_ready.py   # /ready 200 + residency + idle check
wc -c /Users/ericepstein/Qwen3-TTS_UserFiles/.voice_server.log   # offset

# campaign (background + poll; ~6 min at 8bit warm)
nohup conda run --no-capture-output -n qwen3-tts-mlx \
  python /tmp/4b_cycle.py > /tmp/4b_cycle.log 2>&1 &
tail -n 25 /tmp/4b_cycle.log      # progress; results land in /tmp/4b_results.json

# death-signature scan over both rotations
python3 /tmp/4b_scan_deaths.py

# kernel kill check (Sep-5 window; the only retained death window)
/usr/bin/log show --style syslog --start "2026-09-05 15:49:00" \
  --end "2026-09-05 15:53:00" --predicate \
  'process == "kernel" AND (eventMessage CONTAINS[c] "kill" OR eventMessage CONTAINS[c] "memorystatus")'
```

Campaign artifacts (ephemeral, /tmp): `/tmp/4b_cycle.py`, `/tmp/4b_cycle.log`,
`/tmp/4b_results.json`, `/tmp/4b_scan_deaths.py`. The per-transition table
above is the durable copy of `/tmp/4b_results.json`.
