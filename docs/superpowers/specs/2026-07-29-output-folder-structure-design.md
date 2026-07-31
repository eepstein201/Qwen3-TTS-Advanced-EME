# Dedicated Output Folder Structure — Design

**Date:** 2026-07-29
**Status:** Approved, pending implementation plan

## Problem

Web-UI generations are written flat into `~/Downloads` as `voice_ui_<hex>.wav` plus a
`.json` sidecar. Three consequences:

1. **The user's Downloads folder becomes a dumping ground.** A dev machine accumulated
   106+ `voice_ui_*` files interleaved with the user's real downloads, with no way to
   tell app-managed files from files the user chose to keep.
2. **"Remove" in Recent Generations is list-only.** It hides a row but leaves the
   `.wav`/`.json` on disk, so entries reappear after an app restart and disk usage grows
   without bound.
3. **It blocks a correctness fix.** The known `history_df` render race (a stale history
   row can win a delivery-order race against a fresh generation's refresh) is cleanly
   fixed by re-deriving the table from disk on every update. That fix is only safe once
   Remove genuinely deletes the file — otherwise a disk re-read resurrects removed rows.

## Goals

- Separate app-managed output from user-curated keepers, in a clearly named location.
- Make Remove actually free disk space, without turning a single misclick into
  irreversible data loss.
- Give the user an explicit "keep this one" action.
- Unblock the disk-rederive race fix (Phase 2).

## Non-Goals

- **No migration.** Files already sitting flat in `~/Downloads` are left untouched:
  orphaned and invisible to the history table going forward.
- **CLI generation is untouched.** `tts [TEXT]`, `tts batch`, `tts srt`, `tts dialogue`,
  and interactive/REPL/watch modes keep using `output_directory` with exactly today's
  meaning. This feature is scoped to the web UI's Recent Generations flow.
- **Subfolder names are not independently configurable.** Only their shared parent path
  is. Overriding the parent moves both.

## Configuration

One new top-level config key, added to `get_default_config()` in `core/config.py`
(currently line ~310, beside the existing `output_directory`):

```json
"history_output_directory": "~/Downloads/Qwen3-TTS Output"
```

Two fixed-name subfolders always exist beneath it:

| Path | Contents | Written by |
| --- | --- | --- |
| `<parent>/Automated Output/` | every web-UI generation (`.wav` + `.json` sidecar) | generation flow |
| `<parent>/Manual Downloads/` | user-curated keepers | per-row Download action only |

Name uses the hyphenated `Qwen3-TTS` spelling to match the repo, CLAUDE.md, and docs.

`output_directory` keeps its current default (`~/Downloads`) and its current CLI meaning.
The two keys are independent; nothing reads one as a fallback for the other.

### Folder creation

Two mechanisms, because "on installation" is not a single deterministic hook across this
project's install paths (conda setup, `pip install -e`, Colab, Docker):

1. **`install.sh`** creates the default structure during setup. The existing
   "customize" branch (which already prompts for backend / model size / quantization via
   `read -p`) gains one optional prompt: output folder location, Enter for default. The
   "recommended settings" fast path uses the default without prompting.
2. **`build_ui()`** lazily `os.makedirs(..., exist_ok=True)` both subfolders on startup.
   Idempotent, and covers every install path that skips `install.sh`.

### Path safety

Every new filesystem operation — save, delete, download-copy — reuses the containment
pattern already used for `output_directory` in `interface/ui/generation.py`:
`safe_path_join` plus a "resolves under the home directory" check, rejecting `..`
traversal. Deletes additionally require the target to resolve inside `Automated Output`,
and download destinations inside `Manual Downloads`. No new trust boundary is introduced.

## Recent Generations table

`history_df` gains a 7th column, appended after Remove:

```
["Time", "Mode", "Text Preview", "Seed", "Chunks", "Remove", "Download"]
```

`HISTORY_COL_DOWNLOAD = 6` joins the existing `HISTORY_COL_DELETE = 5` and
`HISTORY_COL_TEXT_PREVIEW = 2` constants in `interface/ui/history_panel.py`. Both new
actions route through the **existing** `on_history_select` column dispatch — one
Dataframe `.select()` binding, no new event wiring.

> Constraint: **never attach a `select` listener to a `gr.Tab`** (see CLAUDE.md and
> `tests/test_ui_tab_select_wiring.py`). This design adds no new listeners at all.

### Remove — now destructive, two-step confirm

Reuses the two-step confirm semantics of `confirm_step()` / `on_clear_history_click`,
adapted because a Dataframe cell has no button label to flip:

1. **First click** on a row's Remove cell arms *that row*: the table re-renders with
   that cell's text changed from `✕` to `Confirm?`, and the status banner reads
   "Click again within 5s to permanently delete this file." A 5s window opens.
2. **Second click on the same row** within 5s deletes the `history_state` entry *and*
   the `.wav` + `.json` files from `Automated Output`.
3. **Click on a different row while one is armed** cancels the first row's arm and arms
   the new row instead. A delete never fires from a click the user didn't repeat on that
   same row.
4. **Timeout expiry** re-arms fresh rather than executing.

**Armed state is keyed by the entry's file path, not its row index.** A generation
completing between the two clicks prepends a row and shifts every index; keying by path
prevents confirming a delete against whatever now occupies the old position.

State: a new `gr.State({"armed_path": None, "ts": 0.0})`, independent of Download's.

### Download — one click normally, confirm only on collision

1. **Click Download** copies the row's `.wav` into `Manual Downloads/` under its
   original filename.
2. **If that name already exists there**, do not overwrite. Arm that row (its Download
   cell shows `Overwrite?`), explain via the status banner, and overwrite only on a
   second click on the same row within 5s.

Same path-keyed armed-state mechanism, but its own independent `gr.State` so an armed
Download and an armed Remove cannot interfere.

### Status messages

All new user-facing copy goes through the existing
`StatusBanner().render(text, level)` helper, matching today's Copy / Clear All /
Remove flashes (`role=status`, `aria-live=polite`).

## Data flow

```
Generate (web UI)
  └─> generation.py: save .wav + .json  ──> <parent>/Automated Output/
                                                    │
_load_initial_history / history refresh ────────────┘  (scans Automated Output)
                                                    │
                                              history_df rows
                                           ┌────────┴────────┐
                                    Remove (col 5)      Download (col 6)
                                    arm → confirm       copy (arm only on collision)
                                          │                    │
                              delete .wav + .json      copy .wav ──> Manual Downloads/
```

## Testing

TDD per repo convention: failing test first, then implementation.

**New coverage**
- `history_output_directory` present in `get_default_config()` with the documented default.
- Folder creation is idempotent across repeated `build_ui()` calls (`exist_ok=True`).
- `install.sh`'s new prompt accepts Enter-for-default.
- Generations save into `Automated Output/`; `load_history_from_disk` scans that subfolder.
- Delete state machine: arm-then-confirm on the same path; different-row click re-arms
  instead of executing; timeout re-arms; files removed only on confirmed second click;
  path outside `Automated Output` is rejected.
- Download state machine: no-collision copies on a single click; collision arms instead
  of overwriting; second same-path click overwrites; destination outside
  `Manual Downloads` is rejected.

**Existing tests to update.** These reference `Downloads` and need auditing against the
new default (exact per-file changes to be determined during planning, not guessed here):
`test_ui_facade.py`, `test_ui_generation_ext.py`, `test_ui_headless.py`,
`test_generate_helpers.py`, `test_e2e_playwright.py`, `test_e2e_history_clear_copy.py`,
`test_create_voice_functions.py`, `test_client_voices.py`, `test_client_models.py`,
`test_client_generator.py`, `test_client_generation_timeout.py`. Client/CLI-side
references are expected to need **no** change (CLI behavior is unchanged) — the audit
confirms that rather than assuming it.

**Gates:** `ruff check qwen3_tts tests`, `mypy qwen3_tts/{core,server,interface}`,
full non-E2E suite, then E2E with a healthy server (`curl 127.0.0.1:5123/health` first —
a dead server cascades into unrelated red tests).

## Phasing

**Phase 1 — this design.** Config key, folder creation (install.sh + lazy), save-path
change, history scan-path change, hard-delete with two-step confirm, Download with
collision confirm.

**Phase 2 — sequenced immediately after.** With Remove now deleting the file, both
`_load_initial_history` and the generation chain's `history_df` refresh can re-derive
from disk on every update, with no soft-delete tracking required. That closes the render
race for real, and `test_13_history_row_populates_seed_in_all_tabs`'s
skip-on-race-detection guard is removed as part of it.

Phase 2 depends on Phase 1: building the race fix first would mean designing around the
current soft-delete semantics and then discarding that work once hard-delete lands.

## Prior investigation (context)

The race Phase 2 closes was diagnosed on 2026-07-29. Seed propagation was verified
**correct at every layer** — UI entry, UI payload, server `req.seed`/`used_seed`, disk
preload, post-`add_to_history`, and the `history_df` refresh input all carried the
requested seed on every probed run. The symptom (`test_13` reporting an unrelated seed)
is a stale *row* winning a delivery-order race, not a seed defect. A partial mitigation
(keep-whichever-is-newer in `_load_initial_history`) is committed on
`fix/history-df-preload-race` (`fdb08e5`); it helps only when the preload handler
executes late, since at its own call time there is not yet anything to compare against.
