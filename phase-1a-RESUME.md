# Phase 1a Resume Checkpoint

**Written:** 2026-05-01
**Purpose:** Resume Phase 1a (StatusBanner + a11y) after `/clear` without losing in-flight context.
**Plan reference:** `/Users/ericepstein/.claude/plans/can-you-review-my-polished-boot.md`
**Phase 0 findings:** `phase-0-baseline/PHASE-0-BASELINE.md`

---

## TL;DR

You're mid-Phase 1a on `feature/ui-phase-1a-status-banner`. RED is committed. `components.py` is on disk but untracked. Two files still need edits to make 5 tests pass, then run batches 1-5, smoke test the UI, and commit GREEN.

```bash
# To resume: read this file, confirm state below matches reality,
# then jump to "Next steps" section.
git log --oneline -3
git status --short
```

---

## Branch state

- **Active branch:** `feature/ui-phase-1a-status-banner`
- **Branched from:** `main` (commit `7c3231d`)
- **Phase 0 baseline branch (separate, unpushed):** `feature/ui-phase-0-baseline` — committed locally as `c861543`. Push when ready: `git push -u origin feature/ui-phase-0-baseline`. Do NOT merge into Phase 1a; review/merge those PRs separately.

### Commits on this branch
```
65f8a3f test(phase-1a): RED - add failing tests for StatusBanner + a11y fixes
7c3231d fix: complete 5/6-bit MLX surfacing in install wizard, cache parser, client docstring  (= main)
```

### Uncommitted on this branch (verified at checkpoint time)
- **Untracked:** `qwen3_tts/interface/ui/components.py` (8.7 KB) - already written
- **Untracked unrelated to Phase 1a (pre-existing, leave alone):** `.claude/hookify.*.local.md`, `.claude/settings*.json`, `.github/commands/`, `.github/workflows/gemini-*.yml`
- **Modified unrelated (pre-existing, leave alone):** `.voice_server.log.1`, `config.json`

---

## What's done

### 1. Phase 0 - fully complete and committed on `feature/ui-phase-0-baseline`
- Capability matrix verified (Gradio 6.8.0: `gr.Progress`, `gr.BrowserState`, `gr.themes.Soft.set` all present)
- `/load-model` behavior confirmed: blocks HTTP response (`asyncio.to_thread`), so Phase 1b uses polling-around-future, NOT fire-and-forget
- Confirm-pattern decided: **two-step button** (label flips to "Confirm Delete (5)")
- Test baseline: 1,598 unittests green across batches 1-5 (30.3 s)
- Warm vs cold model load: 12 s vs 91 s (~7.5x slower with empty cache)
- 4 fresh-user bugs surfaced (separate-task backlog):
  1. `tts uninstall config` -> `AttributeError: 'str' object has no attribute 'exists'`
  2. `tts server start` -> `FileNotFoundError` when `config.json` missing
  3. `tts doctor` / `tts config show` -> same crash as #2
  4. UI shows "Loaded (2500MB)" *during* cold model download (status is stale; **Phase 1a fixes this** via `poll_model_loading_state` in `components.py`)
- Warm + cold + cold-loading screenshots committed under `phase-0-baseline/`
- `phase-0-baseline/capture_screenshots.py` is a reusable Playwright harness for visual-diff regression checks across phases

### 2. Phase 1a so far - partially done
- **RED checkpoint committed** (`65f8a3f`): `tests/test_ui_status_banner.py` with 18 tests covering StatusBanner severity rendering, aria-live markup, XSS escaping, thread safety, `poll_model_loading_state`, no-emoji invariants, aria-label on color-coded badges, and the "stale Loaded" regression test.
- **`qwen3_tts/interface/ui/components.py` written but not committed.** Public API:
  - `StatusBanner` class - `render(message, severity)` returns `<div role="status" aria-live="polite" ...>` HTML; thread-safe via `threading.Lock` (mirrors R-50 pattern in `shared.py`'s `_history_lock`)
  - `poll_model_loading_state(model_type, timeout=5.0)` -> `"loaded" | "loading" | "not_loaded" | "unknown"` - polls live `/models` endpoint
  - `status_badge(message, severity)` helper for inline table-cell badges
  - `severity_icon(name)` returns inline SVG (info/check/warn/x/spinner)
  - 5 severities: `info`, `success`, `warning`, `error`, `loading`
  - WCAG 4.5:1 colors: `#1a4480` info, `#0c5d00` success, `#7d4f00` warning, `#9b1c1c` error
  - Inline Heroicons-style SVGs (`currentColor` fill, `aria-hidden="true"`)

### 3. Test status after `components.py` was written
After running `python -m pytest tests/test_ui_status_banner.py -v`:
- **15 / 18 tests passing** (StatusBanner rendering, severity, escaping, thread safety, all `poll_model_loading_state` cases, `format_status_display_no_emoji`)
- **5 / 18 still failing** - these are the targets of the next two edits (see "Pending edits" below)

---

## What's next (in order)

### Step 1 - Edit `qwen3_tts/interface/ui/model_management.py:196-231`

Replace the existing `get_model_status_html` function. Current code uses a literal `'<span style="color: green;">checkmark Loaded ({memory:.0f}MB)</span>'` (with the actual unicode checkmark character) which fails the no-emoji + has-aria-label + reflects-/models tests.

**Drop-in replacement:**

```python
def get_model_status_html(model_type):
    """Get HTML status indicator for a specific model.

    Reflects live /models state (not stale config). When the server reports
    `loading: True` or the model is in flight, returns a "Loading" badge so
    the UI doesn't claim a model is "Loaded" mid-download.

    Args:
        model_type: 'clone', 'design', or 'custom'

    Returns:
        HTML string with accessible status badge (SVG + text + aria-label).
    """
    from qwen3_tts.interface.ui.components import status_badge

    config = load_config()

    if not is_server_running(config):
        return status_badge("Server not running", severity="warning")

    try:
        import requests
        url = get_server_url(config)
        resp = requests.get(f"{url}/models", timeout=5, headers=auth_headers())

        if resp.status_code != 200:
            return status_badge("Error", severity="error")

        data = resp.json()
        info = data.get("models", {}).get(model_type, {}) or {}
        loaded = info.get("loaded", False)
        loading = info.get("loading", False)
        memory = info.get("memory_mb", 0)

        if loaded:
            return status_badge(f"Loaded ({memory:.0f}MB)", severity="success")
        if loading:
            return status_badge(f"Loading {model_type}...", severity="loading")
        return status_badge("Not loaded", severity="info")

    except Exception as e:
        logger.error("Failed to get model status: %s", e)
        return status_badge("Error", severity="error")
```

This makes 4 of the 5 failing tests pass:
- `test_get_model_status_html_no_emoji_loaded` - status_badge uses SVG, no checkmark character
- `test_get_model_status_html_shows_loading_state` - "Loading clone..." emitted when server reports `loading: True`
- `test_get_model_status_html_has_aria_label` - `status_badge` always emits `aria-label`
- `test_get_model_status_html_reflects_models_endpoint_not_config` - branches purely on `/models` response, never reads config startup flag

**Also update `model_management.py:63`** (in `get_model_table_data`): the literal emoji+text in `"checkmark Loaded"` should remain plain text since this is a Dataframe cell that gets rendered by Gradio; just change to `"Loaded"`. The screen-reader story for the Dataframe is separate (Gradio's table widget). Simplest fix: drop the emoji, leave plain text.

### Step 2 - Edit `qwen3_tts/interface/ui/shared.py:253-273`

Add `role="status"` and `aria-live="polite"` to the existing `format_status_display` `<div>`. Current markup (lines 266-273):

```python
return f"""
<div style="padding: 10px; background: var(--block-background-fill, #f5f5f5); border-radius: 5px; margin-bottom: 15px; border: 1px solid var(--block-border-color, #e0e0e0);">
    <strong>Status:</strong> {status_html} |
    <strong>Backend:</strong> {html_mod.escape(str(backend))} |
    <strong>Memory:</strong> {html_mod.escape(str(memory))} |
    <strong>Models:</strong> {html_mod.escape(str(models))}
</div>
"""
```

Change opening div to:

```html
<div role="status" aria-live="polite" style="...">
```

This makes the 5th failing test pass:
- `test_format_status_display_has_aria` - checks for `aria-` or `role="`

### Step 3 - Run tests, confirm GREEN

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts-mlx
python -m pytest tests/test_ui_status_banner.py -v
# Expect: 18 passed, 0 failed
```

### Step 4 - No-regression run

```bash
for b in 1 2 3 4 5; do
  python tests/run_batches.py --batch $b 2>&1 | tail -3
done
# Expect: each prints "Total: 1/1 batches passed"
# Baseline counts to match: 258 / 479 / 366 / 452 / 43 = 1,598 total
```

### Step 5 - Smoke-test UI

```bash
tts server start          # warm cache - ~12s to ready
nohup python -m qwen3_tts.interface.ui._facade --port 7861 > /tmp/p1a-smoke.log 2>&1 &
sleep 8
curl -s http://127.0.0.1:7861/ -o /dev/null -w "%{http_code}\n"  # expect 200
python phase-0-baseline/capture_screenshots.py --state warm \
   --out phase-1a-smoke 2>&1
# Eyeball the 6 PNGs - "Loaded" badge should now have an SVG checkmark, not the emoji
kill $(lsof -ti:7861) 2>/dev/null
tts server stop
```

### Step 6 - Commit GREEN

Stage the 3 changed files only (avoid the pre-existing untracked .claude/.github noise):

```bash
git add qwen3_tts/interface/ui/components.py
git add qwen3_tts/interface/ui/model_management.py
git add qwen3_tts/interface/ui/shared.py

git commit -m "feat(ui-phase-1a): GREEN - StatusBanner, accessible badges, live /models state

- Add qwen3_tts/interface/ui/components.py:
  * StatusBanner class with thread-safe render() (R-50 lock pattern)
  * role=\"status\" + aria-live=\"polite\" markup
  * status_badge() helper for inline table-cell indicators
  * severity_icon() with Heroicons-style inline SVGs
  * poll_model_loading_state() polls live /models endpoint
- Update model_management.get_model_status_html: SVG-based badges,
  shows 'Loading' mid-download (fixes Phase 0 bug #4: stale 'Loaded')
- Update model_management.get_model_table_data: drop emoji from cell text
- Update shared.format_status_display: add role=\"status\" + aria-live

Tests: 18/18 status_banner tests green; batches 1-5 unchanged at 1,598 green."
```

### Step 7 - Mark task done, dispatch reviews (or proceed manually)

Per `superpowers:subagent-driven-development`:
- Spec compliance review (does it match plan section Phase 1a verbatim?)
- Code quality review (file size, lazy imports, no scope creep, no `apply_model_settings` touched)

If subagents still hit the Bash-permission wall (per memory `feedback_background_agents_permissions.md`), review the diff manually with `git diff main..HEAD` and the `everything-claude-code:python-review` skill.

---

## The 5 still-failing tests (verbatim from last pytest run)

```
FAILED tests/test_ui_status_banner.py::test_format_status_display_has_aria
  assert ('aria-' in '<div style="padding: 10px; ...">' or 'role="' in ...')
  -> fix: add role="status" aria-live="polite" to shared.py:266 div

FAILED tests/test_ui_status_banner.py::test_get_model_status_html_no_emoji_loaded
  AssertionError: Emoji 'check' found in model status HTML
  assert 'check' not in '<span style="color: green;">checkmark Loaded (3500MB)</span>'
  -> fix: rewrite model_management.py:196-231 to use status_badge

FAILED tests/test_ui_status_banner.py::test_get_model_status_html_shows_loading_state
  assert ('Loading' in '<span style="color: gray;">Not loaded</span>' or ...)
  -> fix: branch on info.get('loading') in rewritten get_model_status_html

FAILED tests/test_ui_status_banner.py::test_get_model_status_html_has_aria_label
  assert 'aria-label' in '<span style="color: green;">checkmark Loaded (2500MB)</span>'
  -> fix: status_badge() always emits aria-label

FAILED tests/test_ui_status_banner.py::test_get_model_status_html_reflects_models_endpoint_not_config
  Mock setup: config has load_at_startup=True, /models response has loaded=False, loading=True
  Old code claimed 'Loaded' from the config flag - must claim 'Loading' from the live /models response
  -> fix: rewritten get_model_status_html ignores config['models'][...]['load_at_startup']
```

All 5 are addressed by Step 1 + Step 2 above.

---

## Things NOT to touch in Phase 1a

Per the approved plan and CLAUDE.md:
- `apply_model_settings` - explicitly off-limits (flagged fragile)
- Server APIs (`/generate`, `/load-model`, etc.) - Phase 1a is pure UI
- `qwen3_tts/interface/ui/_facade.py` - the plan says "Replace inline status textboxes" but the right scope here is editing the 4 files already touched + reusing the new StatusBanner. **`_facade.py` global StatusBanner mount is deferred to Phase 1b/2** unless the spec reviewer flags it as required. The current 4 edits already satisfy items 1-3 of Phase 1a's "Changes" list (component, emoji removal, aria). Item 2 ("Replace inline status textboxes in `_facade.py`") is best done as a small follow-up after the UI smoke test confirms the new badges render correctly.
- Backend / server / `app_*.py` - no server changes needed for Phase 1a

---

## Quick environment refresher

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts-mlx
cd /Users/ericepstein/Qwen3-TTS_UserFiles
git branch --show-current   # should print: feature/ui-phase-1a-status-banner
```

CLAUDE.md mandates feature-branch workflow - never push to main, never amend commits.

---

## After Phase 1a merges

Phase 1b (`feature/ui-phase-1b-progress`) reuse notes I already scouted:
- `qwen3_tts/server/app_lifespan.py:273` - `_background_load()` daemon thread; sets `app.state.models_loaded` Event when all done. UI can poll `/models` to report which submodel finished during cold start.
- `qwen3_tts/server/app_lifespan.py:302` - writes `model_load_times[model_type]` as each completes; per-model "loaded in 33s" signal is free.
- `qwen3_tts/interface/ui/generation.py:198` - already returns `chunks = client.last_chunk_count`. Phase 1b's "live streaming chunk counter" plugs into existing data, no server change needed.

Phase 1c (`feature/ui-phase-1c-confirms`):
- `qwen3_tts/interface/ui/generation.py:45` - `cancel_streaming_generation()` already exists; the generate-while-generating race guard routes through it without server work.

These mean Phase 1b and 1c are pure-UI changes (no server work needed), as the plan hoped.

---

## If anything looks wrong

If `git log --oneline -3` doesn't show `65f8a3f`, or `qwen3_tts/interface/ui/components.py` is missing, **stop and ask the user** before proceeding. The state described above was captured at write time; any divergence means something happened between sessions.
