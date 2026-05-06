# Phase 1b Resume Checkpoint

**Written:** 2026-05-06
**Purpose:** Resume Phase 1b (Progress indicators for 5 long ops) after `/clear` without losing in-flight context.
**Plan reference:** `~/.claude/plans/can-you-review-my-polished-boot.md` (Phase 1b section)
**Phase 1a checkpoint (predecessor):** `phase-1a-RESUME.md`

---

## TL;DR

Phase 1b task #13 (server-side `loading: bool` on `/models`) is **DONE and committed locally** on `feature/ui-phase-1b-progress`. RED + GREEN both committed. Branch is 2 commits ahead of main, **not pushed**.

Resume next at **task #2** — RED tests for the UI `ProgressIndicator` component + 5 wiring points.

```bash
# To resume: run these to confirm state
cd /Users/ericepstein/Qwen3-TTS_UserFiles
git branch --show-current      # expect: feature/ui-phase-1b-progress
git log --oneline main..HEAD   # expect: 19e97fa, 269133c
git status --short qwen3_tts/ tests/   # expect: empty (clean)
```

---

## What was discovered (mid-Phase-1b plan correction)

The original plan said "no server changes needed for Phase 1b." That was wrong:

- Phase 1a's `qwen3_tts/interface/ui/components.py:poll_model_loading_state()` reads `info.get("loading")` from the `/models` response.
- The server (`qwen3_tts/server/app_models.py:handle_list_models`) **never emitted that field**.
- So the `"loading"` branch was dead code in real operation — only fired from test mocks.

User chose **Option A** (add the server contract) over **Option B** (downgrade UI to indeterminate spinner). New task #13 inserted before #2.

---

## Branch state

- **Active branch:** `feature/ui-phase-1b-progress`
- **Commits ahead of main:** 2 (both unpushed)

```
19e97fa feat(phase-1b): GREEN — server emits loading:bool on /models
269133c test(phase-1b): RED — server-side loading:bool field on /models
ae7c596 (main) Merge feature/ui-phase-1a-status-banner: Phase 1a StatusBanner + a11y
```

### Uncommitted on this branch
- Pre-existing noise (leave alone): `.voice_server.log.1` modified, `config.json` modified, `.claude/hookify.*.local.md`, `.claude/settings*.json`, `.github/commands/`, `.github/workflows/gemini-*.yml`, `.playwright-mcp/`, `phase-1a-RESUME.md`, `phase-1a-smoke/`
- This file: `phase-1b-RESUME.md` (also untracked — meant to be local reference)

---

## What's done (task #13)

✅ Server-side `loading: bool` field on `/models`. UI's existing `poll_model_loading_state` now returns `"loading"` from real server state, not just mocks.

**Files changed:**
| File | Change |
|------|--------|
| `qwen3_tts/server/app_lifespan.py` | Added `app.state.models_loading = {clone, design, custom: False}` in lifespan setup. Wrapped `_background_load`'s `load_model()` call in try/finally that flips the flag True before / False after. |
| `qwen3_tts/server/app_models.py:handle_list_models` | Added `"loading"` to entry dict, mutex with `"loaded"` (loaded model can't simultaneously be loading). |
| `qwen3_tts/server/app_models.py:handle_load_model` | Wrapped `load_model()` call in try/finally that flips the flag. |
| `tests/test_models_loading_flag.py` | NEW — 225 lines, 8 unit tests. |

**Server contract (post-Phase-1b) for each model entry on `/models`:**
```python
{
  "loaded": bool,           # model object exists
  "loading": bool,          # NEW: load_model() in flight (mutex with loaded)
  "description": str,
  "memory_mb": int,
  "repo_id": str,
  "load_at_startup": bool,
  "load_time_sec": float | None,
}
```

**Tests passing:**
- 8/8 new (`tests/test_models_loading_flag.py`)
- 33/33 regression (`test_python_review_fixes.py` + `test_ui_status_banner.py`)

**Backwards-compat:** all access uses `getattr(state, "models_loading", None)` so test mocks built by hand still work.

---

## What's next (in order)

### Step 1 — Task #2: RED tests for ProgressIndicator + 5 wirings

Create `tests/test_ui_progress_indicator.py` with failing assertions for:

1. **`ProgressIndicator(percent=42, eta_s=15, mode="bounded").render()`** returns:
   - `role="progressbar"`
   - `aria-valuenow="42"`, `aria-valuemin="0"`, `aria-valuemax="100"`
   - Visible "42%" + "~15s" labels
2. **`ProgressIndicator(mode="indeterminate").render()`** returns:
   - `role="progressbar"`
   - `aria-busy="true"`
   - No `aria-valuenow`
3. **`poll_model_load_progress("clone")`** (NEW helper to add) returns dict `{percent: int, eta_s: float|None, memory_mb: int}` polling `/models`. Use `load_time_sec` from a recent successful load as ETA seed; `memory_mb` from entry.
4. Model-load handler in `model_management.py` emits "Loading clone… 42% (1850MB, ~15s)" via the new component
5. ASR-load handler emits indeterminate ProgressIndicator
6. AI-enhancement handler in `shared.py` emits "Enhancing…" inline status
7. Auto-transcribe handler in `voice_management.py` emits "Transcribing…" inline spinner
8. Streaming generation in `generation.py` surfaces "Chunk N of M" from `client.last_chunk_count`

Run:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts-mlx
python -m pytest tests/test_ui_progress_indicator.py -v
# Expect: all RED
```

Commit RED:
```bash
git add tests/test_ui_progress_indicator.py
git commit -m "test(phase-1b): RED — ProgressIndicator + 5 UI wirings"
```

### Steps 2-7 — Tasks #1, #8/#5/#4/#7/#9, #10, #6, #3, #12, #11

Follow the dependency graph in TaskList. After RED:
- Implement `ProgressIndicator` in `components.py` (task #1)
- Wire 5 UI files (tasks #4, #5, #7, #8, #9 — parallelizable, no merge conflicts)
- GREEN run (task #10)
- No-regression batches 1-5 (task #6)
- UI smoke + visual diff vs `phase-1a-smoke/` (task #3)
- E2E batch 6 (task #12)
- Final commit + push to `origin feature/ui-phase-1b-progress` (task #11)

User reviews and merges to main. Claude **never** pushes to main.

---

## Things NOT to touch in Phase 1b

- `apply_model_settings` — flagged fragile
- Server APIs **other than the loading flag we just added**
- Phase 1a code (`StatusBanner`, `status_badge`, `severity_icon`) — leave alone, extend `components.py`

---

## Hooks Disabled (already in `.claude/settings.local.json`)

```json
"env": {
  "ECC_GATEGUARD": "off",
  "ECC_DISABLED_HOOKS": "stop:claude-judge-continuation"
}
```

These take effect on next session restart. Mid-session they keep firing — ignore.

---

## Quick environment refresher

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts-mlx
cd /Users/ericepstein/Qwen3-TTS_UserFiles
git branch --show-current   # should print: feature/ui-phase-1b-progress
```

CLAUDE.md mandates feature-branch workflow — never push to main, never amend commits.

---

## After Phase 1b merges

Phase 1c (`feature/ui-phase-1c-confirms`) — confirm patterns for destructive actions (delete voice, unload model, generate-while-generating). Pure UI. Reuses the `StatusBanner` from 1a + `ConfirmButton` to be added in `components.py`.

---

## If anything looks wrong

If `git log --oneline main..HEAD` doesn't show both `19e97fa` and `269133c`, or `tests/test_models_loading_flag.py` is missing, **stop and ask the user**. The state described above was captured at write time — divergence means something happened between sessions.
