# Parallel Tier 1 & 2 Backlog Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Lanes marked PARALLEL run concurrently in isolated worktrees; lanes marked GATED wait on the named gate.

**Goal:** Clear all eight Tier 1 & 2 backlog items (protocols.py deletion, PRF-7 review, FOLLOWUP-1 decision, audit LOWs, MED-1 peaks cache, PRF-9 measurement, GEN-2 route contracts, false-green test fixes) with maximum parallelism and zero write-surface collisions.

**Architecture:** Eight items decompose into seven implementation lanes plus one runtime measurement lane. Lanes A–D run in parallel in isolated git worktrees (disjoint write surfaces); Lane E is config/doc-only; Lane F is a runtime measurement from the main checkout (no code changes); Lane G (GEN-2) is gated on PR #176 merging because it edits `app.py`. All lanes land as separate feature branches → PRs → user merges.

**Tech Stack:** FastAPI/Pydantic (GEN-2), slowapi, pytest/unittest, MLX runtime (PRF-9 measurement), bandit (nosec verification).

**Spec:** `docs/plans/consolidated-roadmap.md` (Priority 1 + PRF sections) and `docs/reviews/` audit findings; ranked-list conversation 2026-08-15.

## Global Constraints

- **No AI authorship attribution** anywhere — commits, comments, PR bodies (CLAUDE.md rule).
- **NEVER amend commits; NEVER force-push.** Every lane creates new commits on its own branch. Claude never pushes or merges; the user does.
- **Branch prefixes matter:** `feat/*`, `fix/*`, `feature/*` get push CI; `chore/*`/`docs/*` get none (PR still triggers the matrix). Prefer `fix/` or `feat/` prefixes.
- **TDD mandatory:** write the failing test, watch it fail (RED), implement (GREEN), refactor. Verify the *detector* — sabotage the code under test and confirm the test catches it — before trusting a green run.
- **Lazy imports:** torch/mlx/transformers/qwen_tts never at module scope.
- **New test modules must be added to `BATCHES` in `tests/run_batches.py`** (explicit allowlist, enforced by `tests/test_batches_coverage.py`). Removed modules must be removed from it.
- **CLAUDE.md is capped at 300 lines (currently 296).** One lane per wave may edit it; everything else defers CLAUDE.md updates to merge-time reconciliation by the coordinating session.
- **Local gates before handoff:** `ruff check qwen3_tts tests`, `mypy qwen3_tts/{core,server,interface}` (54 files), owning batch via `python tests/run_batches.py --batch N`, plus full `pytest tests/ -m "not e2e" --ignore=tests/evaluations`.
- Test env: `conda run -n qwen3-tts-mlx` (spell it out; bare `tts`/`python` resolve elsewhere).
- Merge commands are issued by Claude **only after `gh pr checks <n>` shows every check green**.

## Parallel Lane Matrix

| Lane | Item | Parallel? | Write surface | Risk | Verification |
|---|---|---|---|---|---|
| A | Delete protocols.py | yes (wave 1) | `core/protocols.py`, `tests/test_protocols.py`, `tests/run_batches.py:217` | low | zero-importer grep + batch 1 |
| B | Audit LOWs bundle | yes (wave 1) | `interface/{generate,generate_interactive,generate_server}.py`, `ui/_facade.py`, `ui/shared.py`, `server/websocket.py`, `server/app_generation.py` + tests | medium | bandit per-site, new unit tests |
| C | False-green test fixes | yes (wave 1) | `tests/test_async_concurrency.py`, `tests/test_vllm_async_nonblocking.py`, `tests/test_websocket_rate_limit.py` | medium | mutation check (test goes red when code sabotaged) |
| D | MED-1 peaks cache | yes (wave 1) | `server/app_generation.py` + new test | medium | cache-hit test (peaks computed once) |
| E | FOLLOWUP-1 decision | yes (wave 1, gated on user answer) | `config.json`(no), `docs/plans/consolidated-roadmap.md`, **CLAUDE.md (sole wave-1 owner)**, maybe one test | low | decision recorded + default-asserting test |
| F | PRF-9 measurement | yes (wave 1, **main checkout only** — runtime lane) | none in code; produces `docs/reviews/prf9-max-new-tokens-measurement-2026-08-15.md` | medium (runtime) | documented measurements + go/no-go |
| G | GEN-2 route contracts | **GATED on PR #176 merged** (edits `app.py`) | `server/validation.py`, `server/app.py`, new `tests/test_response_contracts.py` | medium | per-route model-validation tests |
| — | PRF-7 (PR #164 review) | parallel, read-only + merge gate | none | low | `gh pr checks 164` green → user merges |

**Collision management:**
- `tests/run_batches.py`: lanes A (removes a line) and any lane adding a module touch different lines — auto-merges cleanly; coordinating session verifies after each merge.
- `CLAUDE.md`: lane E owns it in wave 1; lane G owns it in wave 2.
- Lane F runs the live server from the **main working copy** (editable install resolves there — see memory `editable-install-cwd-vs-server-code`); no implementation lane may run the server or `tts server restart` while F is measuring. Lanes A–D/G run unit tests only (no server), so no contention.
- Lane B and Lane D both touch `server/app_generation.py` (B: the missing-prompt 404 in `/generate`+`/generate-stream`; D: the peaks call ~line 514). **Sub-gate: B3 and D edit the same file — either serialize B3 after D merges, or B3 works in its own worktree and the coordinating session reconciles at merge time (different functions, auto-mergeable). Chosen: both in worktrees, accept the auto-merge, coordinator verifies combined tests after both merge.**

**Wave structure:**
```
Wave 0 (gate):  PR #176 + #177 CI green → user merges both. PR #164 review → user merges.
Wave 1 (parallel): Lanes A, B, C, D, E, F  (worktrees for A–D; E small; F runtime on main checkout)
Wave 1 close:  coordinator collects branches → user pushes all → PRs opened → CI green → user merges (A before B/D order-independent)
Wave 2 (gated): Lane G (GEN-2) after #176 + wave-1 app_generation merges are on main
```

---

## Lane A — Delete dead `protocols.py` (Tier 1 #1)

**Branch:** `fix/remove-dead-protocols` · **Worktree:** yes

**Files:**
- Delete: `qwen3_tts/core/protocols.py` (304 lines)
- Delete: `tests/test_protocols.py` (if present)
- Modify: `tests/run_batches.py:217` (remove `"tests.test_protocols",`)

**Interfaces:** Produces nothing; consumes nothing. Pure removal.

- [ ] **A1. Prove zero importers (defensive re-check, not memory trust)**

```bash
grep -rn "core.protocols\|core import protocols\|from .protocols\|from ..protocols" \
  qwen3_tts/ tests/ --include="*.py" | grep -v "qwen3_tts/core/protocols.py" | wc -l
# Expected: 0. If non-zero, STOP and report the callers.
```

- [ ] **A2. Delete the module and its test, deregister from BATCHES**

```bash
git rm qwen3_tts/core/protocols.py
git rm tests/test_protocols.py   # only if it exists; check with /bin/ls tests/test_protocols.py
# Edit tests/run_batches.py: remove the line containing "tests.test_protocols",
```

- [ ] **A3. Verify nothing breaks**

```bash
conda run -n qwen3-tts-mlx python tests/run_batches.py --batch 5   # line 217 sits in batch 5's list
conda run -n qwen3-tts-mlx python -m pytest tests/test_batches_coverage.py -q
conda run -n qwen3-tts-mlx ruff check qwen3_tts tests
```
Expected: batch 5 passes; coverage-guard passes (no unregistered-module complaint).

- [ ] **A4. Commit**

```bash
git commit -m "chore: remove dead protocols.py (zero callers, grep-proven)

Abstract protocols for engine components with no importers anywhere in
qwen3_tts/ or tests/ (verified by grep at deletion time; the module was
flagged in the 2026-08-03 audit and re-verified 2026-08-15). The engine
submodules define their own interfaces; the indirection layer was never
consumed."
```

---

## Lane B — Audit LOWs bundle (Tier 1 #4): dead nosec, WS auth audit log, MLX missing-prompt 500→404

**Branch:** `fix/audit-lows-bundle` · **Worktree:** yes

**Files:**
- Modify: `qwen3_tts/interface/generate.py:539`, `generate_interactive.py:346`, `generate_server.py:397`, `ui/_facade.py:122`, `ui/shared.py:799` (nosec audit — line refs current as of 2026-08-15)
- Modify: `qwen3_tts/server/websocket.py:119`
- Modify: `qwen3_tts/server/app_generation.py:346,670`
- Test: extend `tests/test_websocket_security.py` (or nearest existing WS test module) and the generation endpoint tests

### B1. Dead `# nosec` sweep

- [ ] **B1.1 For each of the five sites, test whether the nosec is dead**

For each file, remove the `# nosec ...` comment (code unchanged), then:

```bash
conda run -n qwen3-tts-mlx bandit qwen3_tts/interface/generate.py -c pyproject.toml 2>&1 | tail -3
```
Decision rule: **bandit reports nothing new → the nosec is dead, delete it.** Bandit fires → the nosec is live; restore it and (if it lacks one) add a justification comment like the one on `shared.py:76`.

- [ ] **B1.2 Verify both gates together** (ruff AND bandit — see memory: isort/ruff can move nosec comments off the name line):

```bash
conda run -n qwen3-tts-mlx ruff check qwen3_tts && conda run -n qwen3-tts-mlx bandit -r qwen3_tts -c pyproject.toml
```

### B2. Audit-log WS auth failures

Current state: `websocket.py:119` logs auth failure at DEBUG with no client IP and no reason — invisible in default INFO runs. The connection-rejection path 9 lines up (line 89) shows the house pattern.

- [ ] **B2.1 Write the failing test** (in the WS security test module):

```python
def test_ws_auth_failure_is_audit_logged(self):
    """Auth failures must be visible at WARNING with client IP (audit trail)."""
    import logging
    from qwen3_tts.server.websocket import websocket_tts_handler  # noqa: F401
    import qwen3_tts.server.websocket as ws_mod

    with self.assertLogs("tts.server.websocket", level="WARNING") as logs:
        # Call the auth-failure branch the way the handler reaches it:
        # a first message with a bad token. Use the existing test-harness
        # pattern in this module for a rejected first message.
        ...  # reuse this module's existing bad-token first-message helper
    joined = "\n".join(logs.output)
    self.assertIn("WebSocket auth failed", joined)
```
(Adapt to the module's existing harness — the point under test is the level and the message, on the bad-token path.)

- [ ] **B2.2 Watch it fail (RED)** — run with `-k auth` and confirm no WARNING record.

- [ ] **B2.3 Fix** — change `websocket.py:119` from:
```python
logger.debug("WebSocket auth failed; releasing slot", exc_info=True)
```
to:
```python
logger.warning(
    "WebSocket auth failed from %s: %s",
    sanitize_log(client_ip),
    sanitize_log(e),
    exc_info=True,
)
```
(import `sanitize_log` from `qwen3_tts.core.config` if not already imported in this module — check the header first).

- [ ] **B2.4 GREEN + commit** both B1+B2 together only after B3, or commit B1 and B2 as separate commits — either is fine; keep messages scoped.

### B3. MLX missing-prompt 500 → 404

Root cause (verified 2026-08-15): `load_voice_prompt_mlx` **raises** `FileNotFoundError` for a missing prompt, while the torch loader returns `None`. `app_generation.py:346-352` and `:670-674` only map the `None` case to 404 — the raise escapes as an unhandled 500 on MLX (the default Apple-Silicon backend).

- [ ] **B3.1 Write the failing test** (generation endpoint test module, TestClient):

```python
def test_generate_missing_prompt_returns_404_on_mlx(self):
    """MLX raises FileNotFoundError for missing prompts — must be 404, not 500."""
    with patch("qwen3_tts.core.config.get_backend", return_value="mlx"), \
         patch("qwen3_tts.server.app_generation.load_voice_prompt",
               side_effect=FileNotFoundError("Voice prompt not found")):
        resp = self.client.post(
            "/generate",
            json={"text": "hi", "mode": "clone", "prompt_file": "ghost"},
            headers=self.auth,
        )
    self.assertEqual(resp.status_code, 404)
    self.assertIn("not found", resp.json()["detail"].lower())
```
(Match the module's existing TestClient/auth-harness idiom; mirror it for `/generate-stream`.)

- [ ] **B3.2 RED** — run; expect failure showing 500.

- [ ] **B3.3 Fix both call sites** (`app_generation.py:346` and `:670`):

```python
try:
    voice_prompt = await asyncio.to_thread(load_voice_prompt, prompt_file)
except FileNotFoundError as e:
    raise HTTPException(status_code=404, detail=str(e)) from e
if voice_prompt is None:
    raise HTTPException(
        status_code=404, detail=f"Voice prompt not found: {prompt_file}"
    )
```
Also check the `/ws` prompt-load site (search `load_voice_prompt` in `websocket.py`) and apply the same mapping via its error frame if it exists there.

- [ ] **B3.4 GREEN + Lane B gates**:

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/test_websocket_security.py -q   # or the actual modules touched
conda run -n qwen3-tts-mlx python tests/run_batches.py --batch 3
```

- [ ] **B3.5 Commit**

```bash
git commit -m "fix(server): audit-log WS auth failures; 404 for MLX missing prompts; prune dead nosec

- websocket.py: auth failure logged at WARNING with sanitized client IP
  (was DEBUG-only, invisible in default INFO runs)
- app_generation.py: FileNotFoundError from load_voice_prompt_mlx mapped
  to 404 like the torch None path — was an unhandled 500 on the default
  Apple-Silicon backend (audit M, voice_prompt.py:302)
- removed # nosec comments proven dead by bandit (per-site verification)"
```

---

## Lane C — False-green test fixes (Tier 2 #8)

**Branch:** `fix/false-green-tests` · **Worktree:** yes

**Files:** `tests/test_async_concurrency.py`, `tests/test_vllm_async_nonblocking.py`, `tests/test_websocket_rate_limit.py` (test-only changes)

**Method for all three (per the verify-the-detector memory):** first prove the test is hollow by *sabotaging* the behavior it claims to guard and watching it still pass; then fix the assertion; then prove the fixed test goes RED under the same sabotage; then revert the sabotage and confirm GREEN.

- [ ] **C1. `test_vllm_async_nonblocking`** — audit: passes vacuously. Sabotage: make the mocked "blocking" call actually `time.sleep(0.2)` synchronously inside what should be an async path; run the test. If it still passes, the test never detected blocking. Fix: assert on measurable asynchrony (e.g. an `asyncio.Event` set by the callee while the caller awaits, or elapsed-time bound that a synchronous sleep would violate). Prove RED under sabotage, GREEN after revert.
- [ ] **C2. `test_async_concurrency`** — same protocol. Typical hollow shape: mocks return instantly so "concurrent" and "serial" are indistinguishable. Fix: make the mocked work sleep ~50 ms and assert total wall-time < N×sleep (true concurrency) rather than merely "all tasks completed".
- [ ] **C3. `test_websocket_rate_limit` global branch** — audit: the global-limiter branch asserts nothing observable. Sabotage: remove/disable the global limiter in a scratch edit; run; expect it still passes. Fix: assert the observable outcome (429/close code 1013/limit header) on the N+1'th connection, or convert the branch to an explicit skip with a written reason if genuinely untestable in-process — an honest skip beats a false green.
- [ ] **C4. Gates + commit**

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/test_async_concurrency.py tests/test_vllm_async_nonblocking.py tests/test_websocket_rate_limit.py -v
# then the owning batches for all three modules (check tests/run_batches.py for each)
git commit -m "test: fix three false-green tests (detector verified by sabotage)

Each test passed while the behavior it claimed to guard was sabotaged.
Assertions now fail under sabotage and pass on real code. See the
2026-08-03 audit (three false-green concurrency/vLLM tests)."
```

---

## Lane D — MED-1: cache wavesurfer peaks (Tier 2 #5)

**Branch:** `fix/peaks-cache-med1` · **Worktree:** yes

**Files:**
- Modify: `qwen3_tts/server/app_generation.py:510-519` (peaks computation)
- Test: new `tests/test_peaks_caching.py` (+ register in `BATCHES`, batch 3 list)

**Verified current state (2026-08-15):** `calculate_waveform_peaks` runs via `asyncio.to_thread` on the in-memory `wav` after **every** generation result, including generation-cache hits that skip inference — recomputed per request/playback with no peaks cache.

- [ ] **D1. Write the failing test**

```python
"""MED-1: waveform peaks are computed at most once per audio asset."""
from unittest.mock import patch


def test_peaks_computed_once_on_generation_cache_hit(...):
    """Second /generate with identical input (generation-cache hit) must not
    re-run calculate_waveform_peaks."""
    import qwen3_tts.server.app_generation as ag

    calls = []
    real = ag.calculate_waveform_peaks  # if imported lazily, patch at source module instead

    with patch("qwen3_tts.core.engine.audio_processing.calculate_waveform_peaks",
               side_effect=lambda a, num_peaks=500: calls.append(1) or [0.1] * num_peaks):
        r1 = client.post("/generate", json=SAME_PAYLOAD, headers=auth)
        r2 = client.post("/generate", json=SAME_PAYLOAD, headers=auth)
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["results"][0]["peaks"] == r2.json()["results"][0]["peaks"]
    assert len(calls) == 1
```
(Use the generation-endpoint test module's TestClient idiom; ensure the generation cache is enabled and the same cache key is produced — identical payload, cache enabled in test config.)

- [ ] **D2. RED** — expect `len(calls) == 2`.

- [ ] **D3. Implement a peaks cache** on `app.state` (follows the `gen_cache` house pattern; LRU-bounded by the existing `cache.generation_max` style — reuse `state.gen_cache` entry: store `peaks` on the cache entry and return them on hits; no second cache needed):

```python
# on the generation-cache hit path: return the stored entry's "peaks"
# on the miss path, after computing:
state.gen_cache[cache_key]["peaks"] = peaks
```
If the hit path currently rebuilds the response dict, extend it to read `entry.get("peaks")` and only compute when absent.

- [ ] **D4. GREEN + register the test module in `tests/run_batches.py` (batch 3) + gates**:

```bash
conda run -n qwen3-tts-mlx python -m pytest tests/test_peaks_caching.py -v
conda run -n qwen3-tts-mlx python tests/run_batches.py --batch 3
conda run -n qwen3-tts-mlx python -m pytest tests/test_batches_coverage.py -q
git commit -m "feat(server): cache waveform peaks on the generation cache entry (MED-1)

Peaks were recomputed via to_thread on every request, including
generation-cache hits that skip inference entirely. Stored on the
gen_cache entry; computed once per audio asset."
```

---

## Lane E — FOLLOWUP-1: record the `load_at_startup` decision (Tier 1 #3)

**Branch:** `docs/record-startup-defaults-decision` (docs-only → PR triggers CI) · **Gated on the user's answer** (asked at plan review; default recommendation: **keep `false`/`false`** — 5 GB of startup memory for models most sessions never open; on-demand load is one click in Manage Models and now shows a live ETA badge thanks to #175).

**Files:**
- Modify: `docs/plans/consolidated-roadmap.md` (FOLLOWUP-1 row → ✅ with rationale)
- Modify: `CLAUDE.md` (sole wave-1 CLAUDE.md owner — one sentence in Key Settings if the decision is keep-false: "design/custom load on demand by design (FOLLOWUP-1, 2026-08-15)")
- Test: only if flipped to true — `tests/test_config.py` assertion that startup loads the expected models. Keep-false needs no new test (existing default tests already pin it).

- [ ] **E1. Record the decision** in the roadmap row with rationale (memory cost vs availability; the #175 ETA badge reduces the pain of on-demand loads).
- [ ] **E2. CLAUDE.md** one-liner (stay ≤300 lines — currently 296, this adds ~1 line; if over, trim elsewhere).
- [ ] **E3. Commit**

```bash
git commit -m "docs: record FOLLOWUP-1 decision — design/custom stay load-on-demand

~5 GB additional startup memory buys availability for models most
sessions never open; on-demand load is one click and now shows a live
ETA badge (PR #175)."
```

---

## Lane F — PRF-9: measure raising the MLX `max_new_tokens` cap (Tier 2 #6) — measurement only, no ship

**Branch:** `docs/prf9-max-new-tokens-measurement` · **Runs from the MAIN checkout** (editable install resolves there; server must run this code) · **Runtime lane — coordinate: no other lane may restart the server while F measures.**

**Files:**
- Create: `docs/reviews/prf9-max-new-tokens-measurement-2026-08-15.md`
- Temporarily modify via **config only** (`tts config edit` / direct `config.json` edits): `generation.max_new_tokens`, `generation.max_chunk_chars` — **restore both afterward**

**Protocol:**
- [ ] **F1. Baseline at 2048** (current default): `TTS_LOG_LEVEL=DEBUG conda run -n qwen3-tts-mlx tts server stop && ... tts server start`; generate a ~2,000-char single-chunk text (`max_chunk_chars: 0` disables chunking) and a ~5,000-char text; record RTF, `/stats` `mlx_memory_peak_mb`, audio validity (duration ≈ expected, non-silent tail).
- [ ] **F2. Repeat at 8192** then **16384** (`generation.max_new_tokens`; restart server each time — all generation config is read at request time via `load_config`, but restart to be safe and to reset peak-memory counters).
- [ ] **F3. Stability checks at each tier:** no mid-generation truncation (compare final duration vs char-count expectation at ~12 Hz ≈ 0.083 s/token), no NaN/corruption (LUFS/peak sanity), memory headroom on the M2 Pro (16 GB? record the machine's RAM), watch `.voice_server.log` timings.
- [ ] **F4. Write the measurement doc** with a table (cap → RTF, peak MB, stable y/n) and an explicit **go/no-go**: ship the raised default only if 8192 is stable with acceptable peak memory; 16384 is informational.
- [ ] **F5. Restore config** to `max_new_tokens: 2048`, `max_chunk_chars: 500`; restart server; commit only the doc.

```bash
git add docs/reviews/prf9-max-new-tokens-measurement-2026-08-15.md
git commit -m "docs: PRF-9 measurement — MLX max_new_tokens 2048/8192/16384 on M2 Pro

Validation-gated precursor to raising the chunking cap. [go/no-go
verdict + numbers in the doc]"
```

---

## Lane G — GEN-2: Pydantic `response_model=` contracts (Tier 2 #7) — **GATED: start only after PR #176 merges and wave-1's `app_generation.py` lanes (B3, D) are on main**

**Branch:** `feat/response-contracts-gen2` · **Worktree:** yes

**Files:**
- Modify: `qwen3_tts/server/validation.py` (new Pydantic response models — keep `HealthResponse` company; do not create a new module unless validation.py would exceed ~800 lines)
- Modify: `qwen3_tts/server/app.py` + handler modules (add `response_model=` to route decorators; **zero handler body changes** — the contract must describe existing output, not change it)
- Test: new `tests/test_response_contracts.py` (+ register in BATCHES, batch 3)

**Grouping (four independently-mergeable commits on one branch):**
- [ ] **G1. Public/status routes**: `/ready`, `/queue-status`, `/generation-status`, `/stats`, `/models`. Model example:

```python
class ReadyResponse(BaseModel):
    status: str
    models_loaded: bool  # match the actual field names by reading each handler's return dict FIRST

class ModelEntry(BaseModel):
    loaded: bool
    loading: bool
    description: str
    memory_mb: int
    repo_id: str
    load_at_startup: bool
    load_time_sec: float | None

class ModelsResponse(BaseModel):
    models: dict[str, ModelEntry]
    asr: dict[str, Any]          # tighten in a follow-up if stable
    backend: str
    model_size: str
```
Test pattern per route (TestClient, auth headers):

```python
def test_models_response_matches_contract(self):
    resp = self.client.get("/models", headers=self.auth)
    self.assertEqual(resp.status_code, 200)
    ModelsResponse.model_validate(resp.json())  # raises on mismatch — the whole point
```

- [ ] **G2. Generation routes**: `/generate` (result rows with `audio_base64`, `sample_rate`, `peaks`, `chunks`, `seed`), `/transcribe`.
- [ ] **G3. Ops routes**: `/load-model`, `/unload-model`, `/update-model-config`, `/update-startup-config`, `/load-asr`, `/unload-asr`, `/prompts`, `/prompt-details`, `/delete-prompt`, `/rename-prompt`, `/preview-prompt`.
- [ ] **G4. Config/shutdown + OpenAPI check**: `/update-startup-config`, `/shutdown` (already `-> Response` from #176 — response_model intentionally omitted there), plus:

```python
def test_openapi_generates_cleanly(self):
    import qwen3_tts.server.app as app_mod
    spec = app_mod.app.openapi()   # must not raise; spot-check a typed route
    self.assertIn("ModelsResponse", spec["components"]["schemas"])
```

- [ ] **G5. Gates + commit** (one commit per group, `feat(server): response contracts — <group>`; final `git commit` for the test module registration). Update **CLAUDE.md** (wave-2 owner): note in the Server API section that routes carry Pydantic contracts.

**Danger notes for G:** (1) adding `response_model=` makes FastAPI *filter* response fields to the model — read each handler's actual return dict before writing the model, or the contract will silently drop fields. (2) `/health` already has `HealthResponse`; don't duplicate. (3) `response_model_exclude_none=False` default keeps `null` fields — fine.

---

## Wave 3 — follow-on lanes (recorded 2026-08-15; surfaced by wave-1 execution)

Both items are recorded in the roadmap's LOW-cleanups list via the wave-1 reconciliation; these lanes make them actionable. They can run in parallel — disjoint write surfaces (`app_generation.py` vs a test module).

### Lane H — Fix the on-demand-load mismatch (`/generate` `model_not_loaded`)

**Branch:** `fix/on-demand-model-load` · **Worktree:** yes · **Gated on:** #185 (GEN-2) merging — it edits `app_generation.py`'s response layer.

**Premise (verified 2026-08-15):** `/generate` with `mode: design` (or `custom`) on a freshly-started server returns `{"detail": {"error": "model_not_loaded", "recovery": "restart", ...}}` — but `docs/RUNBOOK.md` states models load "on demand when a request needs them", and FOLLOWUP-1's keep-false decision (design/custom `load_at_startup: false`) explicitly relies on on-demand being one request away. The RUNBOOK and the FOLLOWUP-1 rationale describe the intended behavior; the handler is what's wrong. (Also fix the misleading `recovery: "restart"` — the actual remedy is loading the model, not restarting the server.)

**Files:**
- Modify: `qwen3_tts/server/app_generation.py` (the `/generate` + `/generate-stream` model-lookup path)
- Test: extend `tests/test_voice_server.py` (or the module owning generation endpoint tests — locate by content)

- [ ] **H1. Write the failing test** (RED):

```python
def test_generate_loads_model_on_demand(self):
    """/generate with an unloaded model loads it instead of erroring (RUNBOOK contract)."""
    with patch("qwen3_tts.core.engine.load_model") as mock_load, \
         patch("qwen3_tts.core.engine.run_inference",
               return_value=(np.zeros(2400, dtype=np.float32), 24000)):
        resp = self.client.post(
            "/generate",
            json={"text": "hello", "mode": "design",
                  "voice_description": "A calm voice"},
            headers=self.auth,
        )
    self.assertEqual(resp.status_code, 200)
    mock_load.assert_called_once()
```
(Adapt to the module's TestClient/auth harness; mirror for `/generate-stream`. Watch it fail with today's `model_not_loaded`.)

- [ ] **H2. Implement on-demand load** in the handler's model-resolution step:
  - if `state.models.get(model_type) is None`: set `loading_map[model_type] = True` (existing `/models` infrastructure, so badges show the load), `await asyncio.to_thread(load_model, model_type)`, store into `state.models` + `model_load_times`, clear the flag — mirroring `handle_load_model`'s guarded sequence (in `finally`).
  - A load failure returns the same sanitized error shape `/load-model` produces (503 + `model_load_errors` entry via `_recover_from_failed_load` — PRF-5), NOT a bare 500.
  - Keep it OUTSIDE `inference_lock` (load is not GPU-inference; matches the voice-prompt-load precedent in the same handler).
  - Concurrency: two simultaneous first-requests must not double-load — re-check `state.models` under the existing loading map/lock pattern before committing to load; a second request arriving mid-load may wait or return 503-with-loading, but must not corrupt state. Test this explicitly.

- [ ] **H3. Fix the error text** for the (now rarer) genuinely-cannot-load path: `recovery: "restart"` → guidance pointing at `POST /load-model` / the Manage Models tab.

- [ ] **H4. RUNBOOK already claims this — no doc change needed**; instead add the one-line confirmation to the roadmap LOW-cleanups entry when marking resolved.

- [ ] **H5. Gates:** batch 3; ruff; mypy (53 files); full non-E2E suite; commit `fix(server): load design/custom models on demand in /generate (RUNBOOK contract)`.

### Lane I — De-hollow `tests/test_server_peaks.py`

**Branch:** `fix/test-server-peaks-unittest` · **Worktree:** yes · **Parallel-safe** (test module only; no overlap with Lane H).

**Premise (verified by lane D, 2026-08-15):** the module's tests consume a pytest fixture for the client/app; the batch runner executes modules via `python -m unittest`, where fixture-dependent tests silently don't run — they pass hollow in every batch gate (the same false-green family #181 fixed).

- [ ] **I1. Prove the hollowness first** (detector discipline): run `conda run -n qwen3-tts-mlx python -m unittest tests.test_server_peaks -v` and record how many tests actually execute vs. the module's test count under pytest. Sabotage the peaks endpoint (e.g. make `/generate` return no `peaks` field) and confirm the unittest run still reports OK.
- [ ] **I2. Rewrite the module as `unittest.TestCase`** with an explicit `setUp` building the app + TestClient via the `tests/conftest._init_app_state` pattern (the exact idiom `tests/test_peaks_caching.py` from lane D uses — read it first and copy the harness). Keep every assertion; only the harness changes.
- [ ] **I3. Verify the fixed module goes RED under the I1 sabotage** and GREEN after revert; confirm it now runs real under both `python -m unittest tests.test_server_peaks` (correct test count, all OK) and pytest.
- [ ] **I4. Gates:** batch 3 (the module's owner — confirm in BATCHES); full non-E2E suite; commit `test: convert test_server_peaks to unittest so batch gates exercise it for real`.

---

## PRF-7 (Tier 1 #2) — review mlx-audio 0.4.7 (PR #164)

Read-only lane, no branch:
- [ ] `gh pr view 164 --json files,commits` + diff review against the 0.4.7 changelog; confirm `pyproject.toml` floor bump only, `requirements.lock` untouched (mlx excluded by policy).
- [ ] `gh pr checks 164` all green → hand the user `gh pr merge 164 --squash --delete-branch`.
- [ ] Post-merge validation in the MLX env (memory: dep knot): `conda run -n qwen3-tts-mlx pip install -e ".[mlx]" && pip check` + one smoke generation; then mark PRF-7 ✅ in the roadmap (fold into Lane E's docs commit or a standalone one).

## Self-Review (done)

- Spec coverage: all 8 ranked items map to lanes A(1) / PRF-7(2) / E(3) / B(4) / D(5) / F(6) / G(7) / C(8). ✓
- Placeholders: B2.1's harness reuse and Lane C's sabotage steps are investigation-shaped by design — each names the exact file, the exact hollow shape from the audit, and the acceptance criterion (RED under sabotage). No TBDs. ✓
- Type consistency: G1's models reference handler fields verified this session (`loading`, `load_time_sec` from `handle_list_models`; `peaks`/`seed` from the generation result). ✓
- Known soft spot: B2.1's test body adapts to the module's existing harness — the executor must read that module first; acceptance is the WARNING-level assertion, not the scaffolding.
