# Plan: User-Defined Prosody Preset Builder (save + delete, UI-only) — v3

## Context

The Custom tab's Style Instruction box is free-form instruct text (Qwen3-TTS's native natural-language prosody control). 8 **factory** prosody presets exist; saving a custom instruct as a reusable named preset requires hand-editing `config.json` (`generate.py:450`). This adds the missing save/delete flow to the Gradio UI.

Research-validated (2026-09-01, memory `prosody-preset-feature-research.md`): "user-defined vs factory presets" is the industry convention; instruct text stored **stripped-verbatim** (no parsing layer); no vendor ships saved instruct presets.

**Locked decisions (user, 2026-09-01):** UI only · forbid factory-name collisions · own-preset overwrite requires confirm · save + delete ship · no CLI/server/endpoint changes.

**Review provenance:** two full team-review rounds (a11y-architect, architect, python-reviewer, planner). Round 1: 4× SHIP-WITH-EDITS → v2. Round 2 (against `main` @ `3254cf6`, post-#233): 4× SHIP-WITH-EDITS with every round-1 fold verified landed; round-2 must-fixes folded here: conditional `value=` reset unimplementable without dropdown **inputs** (4-way convergence), rule-1 raw-base naming (2-way), `config=` persistence semantics (planner). Working tree == origin/main; all citations re-verified at `3254cf6`.

## Architecture

- **`core/config/presets.py`** — data layer: raw/factory-excluded reads, name validation, save/delete CRUD (presets.py grows ~119→~260 lines; dependency direction preserved).
- **`interface/voice_helpers.py`** — UI-format only: user-preset choice rendering, choice→name parsing, shared `_format_prosody_choice`. Stays gradio-free.
- **`interface/ui/tabs_generation.py`** — thin module-level handlers + Custom-tab wiring; **`interface/ui/_facade.py`** — plumbs `design_prosody` (4th return; `clone_prompt` precedent).
- Untouched: `generation.py` (726/800), `shared.py` (879), `server/`, CLI, `apply_prosody_preset` behavior.

## Data layer — new public API in `core/config/presets.py`

```python
PROSODY_NAME_MAX_LEN = 40
_FACTORY_PROSODY_NAMES = frozenset(k.lower() for k in DEFAULT_PROSODY_PRESETS)   # module-private

_raw_user_prosody_entries(config) -> dict      # PRIVATE: isinstance-guarded config.get("prosody_presets", {}) — the RAW key, unfiltered
get_user_prosody_presets(config=None) -> dict[str, str]
    # filtered VIEW off the raw read: drop factory-named keys (case-insens), keep only str values
    # non-empty after strip; returns a NEW dict (never aliases caller's)
validate_prosody_preset_name(name) -> str | None          # error message or None
save_user_prosody_preset(name, instruct_text, config=None) -> tuple[bool, str]
delete_user_prosody_preset(name, config=None) -> tuple[bool, str]
```

Validation rejects: empty/whitespace name or text; len > 40 (on stripped); case-insensitive factory collision; `"(none)"` and `" - "` (**defence-in-depth** — the `^[a-zA-Z0-9_\-\.]+$` charset already excludes both; comment says so). `..` deliberately omitted (config key, never a path component — comment). Case asymmetry documented: factory collision case-insensitive; user-vs-user uniqueness case-sensitive (dict keys).

**Writer contract:**
- Writers strip name AND text internally, store stripped forms. Duplicate text under a different name is **valid — name-keyed, no dedup**.
- `config` overrides the BASE READ only; **`save_config` is ALWAYS called** (persistence semantics pinned — tests must patch `save_config` to stay disk-free, see step 1). Handlers pass `config=None` into writers (fresh load).
- Write base = `_raw_user_prosody_entries` — **NEVER `get_user_prosody_presets()`** (a filtered-base save silently drops seeded entries and hand-edited factory-shadowed overrides, a documented meaningful state per `presets.py:108`); never a bare replacement (config.json seeds all 8 defaults, `io.py:370-379`). Idiom: `save_config({**config, "prosody_presets": {**_raw_user_prosody_entries(config), name: text}})`. Filtered view used ONLY for existence checks and dropdown rendering.
- Exception handling: catch **`(ValueError, OSError)`** around the config load (`io.py` converts corrupt JSON AND validation errors to `ValueError`; `JSONDecodeError` ⊂ `ValueError`, mirroring `presets.py:115`; `FileNotFoundError` may be differentiated as "config.json is missing — run tts config") → `(False, corrupt-msg)` and **never save**; catch `OSError` around `save_config` itself (`io.py:250-267` unguarded) → `(False, f"Couldn't save prosody preset '{name}' — the config file couldn't be written ({err.strerror or 'unknown error'}). Check disk space and permissions, then try again.")`. Raw exception text never surfaces (path-leak, CWE-209 convention).
- Delete membership test: `name in _raw_user_prosody_entries(config)` minus factory names (NOT the filtered view — hand-edited junk entries stay deletable while never being listed). Unknown/not-user → `(False, "Preset '<name>' no longer exists — it may have been removed outside the UI.")`; factory-named → `(False, "'<name>' is a built-in preset and can't be deleted.")` (defence-in-depth; factory names are unreachable from the UI dropdown).
- **Legacy hand-created presets sharing a factory name are treated as factory** — invisible in the manage UI; the merged apply view still prefers them. Deliberate; documented in CONFIG.md prose.
- **The writer owns user-facing failure/success strings; handlers display `(ok, message)` verbatim.** The pinned-copy list below governs VALIDATOR messages only.
- Imports: lazy-per-call facade idiom (`presets.py:110` precedent); seam = `qwen3_tts.core.config.load_config/save_config`.
- Facade re-exports (`core/config/__init__.py` import block ~158-163 AND `__all__` ~277-279): `PROSODY_NAME_MAX_LEN`, `get_user_prosody_presets`, `validate_prosody_preset_name`, `save_user_prosody_preset`, `delete_user_prosody_preset`. `_FACTORY_PROSODY_NAMES` stays private.

## UI layer

`voice_helpers.py`: `PROSODY_NONE_CHOICE = "(none)"`, `PROSODY_CHOICE_SEPARATOR = " - "`; `_format_prosody_choice(name, text)` shared by `get_user_prosody_choices(config=None)` (sorted, `(none)`-first, **display-truncates text at 80 chars + "…"** — stored/applied text unaffected; factory texts are 35-55 chars so `get_prosody_choices` output is observably unchanged, name recovery via `split(" - ")[0]` safe) and delegated from `get_prosody_choices` (signature pinned; body may delegate — no test patches its body); `prosody_choice_to_name(choice) -> str`. `tabs_generation.NONE_CHOICE` (line 22) stays untouched.

`tabs_generation.py` Custom tab, **after `custom_preset`** (end of left Column ~441; generation controls stay contiguous; closed accordion = `Description Builder` precedent at line 201):

| Component | Spec |
|---|---|
| `gr.Accordion("My prosody presets", open=False)` | inner Markdown: "Save the current Style Instruction as a named preset, or delete presets you created. Built-in presets can't be changed. The preset saves exactly what's in the Style Instruction box — including any preset text appended via the Style Preset dropdown." |
| `prosody_preset_name` Textbox | label "Preset name", placeholder "e.g., 'storyteller'", info f"Letters, numbers, dashes, underscores, and dots only. Max {PROSODY_NAME_MAX_LEN} characters." |
| `prosody_preset_save_btn` Button | base "Save as preset", arm **"Confirm Overwrite? (click again)"**; `variant="secondary", size="sm"` |
| `prosody_preset_delete_btn` Button | base "Delete preset", arm **"Confirm Delete? (click again)"** (repo convention, `tabs_management.py:186`); `variant="stop", size="sm"` |
| 2× status Textbox | `label="", show_label=False`, non-interactive, `max_lines=2, container=False` (`manage_status` precedent, `tabs_management.py:126-132`) |
| `prosody_preset_delete_dropdown` Dropdown | label "Preset to delete", info "Only presets you created appear here.", choices `get_user_prosody_choices()`, value NONE_CHOICE |
| **1× shared** sr-only announcer `gr.HTML` | via `generation._announce_status` (returns the sr-only string, `generation.py:51-68`); inside the accordion; **never `visible=False`** (Gradio 6 removes it from the DOM, `generation.py:56-57`) |
| `prosody_preset_save_state` / `prosody_preset_delete_state` | `gr.State` **target-keyed**: `{"armed": bool, "ts": float, "armed_name": str \| None}`; fresh per control |

Discoverability: amend `custom_prosody` info (`tabs_generation.py:432`) to "Select a preset to fill the instruction field, or type your own below, then save it under 'My prosody presets'."

## Write-path rules (the traps)

1. Write base = `_raw_user_prosody_entries` — never the merged `get_prosody_presets()`, never the filtered view, never bare replacement. Filtered view = existence checks + rendering only.
2. **Swallow-on-corrupt = pure reads only.** Any write-path config-load failure returns `(False, msg)` — a swallowed `{}` base composed with the merge idiom would **wipe the entire config.json**.
3. OSError never escapes a handler (strands armed `gr.State` + stale button).
4. Validation lives in the save path and **precedes arming** — factory-named hard-errors on the FIRST click, never arms. The handler **strips the name once, up front**, and uses the stripped form for validation, existence, `armed_name`, and confirm comparison — byte-identical to the writer's internal strip (a whitespace-variant must never bypass overwrite-confirm).
5. Handlers pass `config=None` into writers; ONE shared config load feeds the classify calls. `_config_lock` covers atomicity, not the read-modify-write window — accepted single-user-local risk.
6. `tuple[bool, str]` returns, **no `gr.Error`** (a raise aborts output application and strands armed state; every failure here is expected control flow that must still reset the button).

## Wiring

```python
prosody_preset_save_btn.click(fn=_on_save_prosody_preset,
    inputs=[prosody_preset_save_state, prosody_preset_name, custom_instruct,
            custom_prosody, design_prosody],                      # 5 INPUTS — dropdowns must be inputs
    outputs=[prosody_preset_save_state, prosody_preset_save_btn, prosody_preset_save_status,
             <announcer>, custom_prosody, design_prosody, prosody_preset_delete_dropdown])  # 7 OUTPUTS
prosody_preset_delete_btn.click(fn=_on_delete_prosody_preset,
    inputs=[prosody_preset_delete_state, prosody_preset_delete_dropdown,
            custom_prosody, design_prosody],                      # 4 INPUTS
    outputs=[same 7-slot shape])
```
Arity 7 = **outputs only**. A component may appear in both lists; a handler can only READ what it receives in `inputs=` — that is why the conditional-reset rule requires the two dropdowns as inputs. `design_prosody` may arrive `None` (lazy tabpanel never visited) → None-guard to "(none)" → no reset.

**Target-keyed confirm:** `confirm_step` stays un-keyed; the HANDLER owns the name comparison (`history_panel.py:134-183` precedent). Confirm applies only when `armed_name == <stripped current name/selection>` AND within 5 s; mismatch or expiry re-arms for the current target.

**Branch invariants (both flows):** every branch except ARM ends with state `{armed: False, ts: 0.0, armed_name: None}` and the button reset to its base label; only the arm branch ends armed. A mismatched-target click re-arms for the new target (branch ③ below — `armed_name` updates to the stripped current name); it never saves or deletes on the mismatch itself.

**Save branches (in order):** ① validation failure (validator message, pinned copy) → hard error, disarmed, **never arms** · ② stripped name not in user view → save now → ok: "Saved preset 'x'. It now appears in the Style Preset dropdown." (+ " (N characters, saved verbatim)." if N>200) / fail: writer message · ③ in user view, unarmed/mismatched/expired → arm ("A preset named 'x' already exists. Click again within 5s to overwrite it."; target-changed: "Changed to 'x' — click again within 5s to overwrite it."; expired: "Your confirmation timed out. Click again within 5s to overwrite 'x'.") · ④ armed_name==name, fresh → save → ok: "Updated preset 'x'." / fail: writer message + disarm.
**Delete branches:** ① nothing selected → "Select one of your presets to delete." · ② raw-minus-factory membership miss → **render the data-layer classification message** (never a UI-local literal) · ③ miss-armed/mismatched/expired → arm ("Delete preset 'x'? Click again within 5s to confirm."; "Now deleting 'x' — click again within 5s to confirm."; "Your confirmation timed out. Click again within 5s to delete 'x'.") · ④ fresh confirm → delete → "Deleted preset 'x'." / fail: writer message + disarm.

**Pinned copy (validator messages only; writer messages display verbatim):** empty name → "Enter a preset name." · empty instruct → "Type a Style Instruction first — the preset saves exactly what's in that box." · >40 → f"Preset name is too long ({PROSODY_NAME_MAX_LEN} characters max)." · factory → "'<name>' is a built-in preset — pick a different name." · charset → "Preset names can only contain letters, numbers, dashes, underscores, and dots."

**Dropdown refresh cells:**
- Success: `prosody_preset_delete_dropdown` → `gr.update(choices=get_user_prosody_choices(), value=NONE_CHOICE)` **if the deleted/overwritten name equals its current selection**, else `gr.update(choices=…)` only (selection preservation — save-success must not wipe an unrelated pending delete). `custom_prosody`/`design_prosody` → fresh `choices=get_prosody_choices()`, **`value=NONE_CHOICE` ONLY IF** `prosody_choice_to_name(<its own input value>)` == the affected preset — stale `"name - old text"` labels must not survive an overwrite, and never re-set a refreshed `"name - text"` string (re-fired `.change` re-appends → double text, `voice_helpers.py:85-86`). Unrelated selections untouched (reset re-fires `.change`; `(none)` returns existing text verbatim, `voice_helpers.py:74-75`).
- Non-success: `gr.update()` on all three dropdowns (state/button/status/announcer still updated). Every branch updates the announcer.

## Implementation steps (TDD, commit per step)

0. `git checkout main && git pull && git checkout -b feature/prosody-preset-builder` (main now @ `3254cf6`). Abort = delete the branch; no revert path needed.
1. **RED** — new `tests/test_prosody_preset_builder.py` (`unittest.TestCase`; function-level imports; `_read_source` helper copied from `test_ui_confirm_patterns.py:147`; `HAS_GRADIO` try/except + `skipUnless` on gradio-dependent classes), registered in `BATCHES` batch 4 (`run_batches.py`, after `test_mlx_generate_kwargs`) **same commit**. Classes:
   - `TestValidateProsodyPresetName` — empty; >40; "excited"; "EXCITED"; "(none)"; "slow - really"; charset; valid → None.
   - `TestGetUserProsodyPresets` — seeded-8 factory config: none of the 8 appear; str-only; non-empty-after-strip; new dict (no aliasing).
   - `TestSaveUserProsodyPreset` — **disk-free, NOT patch-free**: pass `config=` AND `patch("qwen3_tts.core.config.save_config")` with a recording side_effect; assert the CAPTURED PAYLOAD (raw base + new key only, seeded entries and shadowed overrides preserved — never filtered); stripped-verbatim storage ("  Foo  " → key "Foo"); seeded-8 + "excited" → `(False, …)`; duplicate text allowed; overwrite own → `(True, "Updated…")`.
   - `TestDeleteUserProsodyPreset` — same disk-free pattern; removes only the named raw key; junk entry (non-str value) deletable; seeded-8 "excited" → `(False, "built-in")`; unknown → `(False, "no longer exists…")`.
   - Failure paths — load `side_effect=ValueError` (the `io.py:211` shape) → `(False, corrupt-msg)` **and `save_config` not called**; `FileNotFoundError` → missing-file msg; `save_config` `side_effect=OSError` → `(False, strerror-msg)`; both writers.
   - `TestGetUserProsodyChoices` — **exact literal format** (e.g. `config={"zz": "text"}` → `["(none)", "zz - text"]`); sorted; user-only on seeded-8; 80-char truncation; `PROSODY_NONE_CHOICE == "(none)"` == the `NONE_CHOICE = "(none)"` literal via `_read_source` (no gradio import).
   - `TestProsodyPresetHandlerContracts` (HAS_GRADIO) — direct handler calls; **outputs arity 7 per branch, inputs arity 5/4**; `result[0]` state dict (`.get("armed")`); branch-invariant assertions (disarm + button base label on every non-arm branch; armed label only in arm branches); patch `qwen3_tts.core.config.{validate_prosody_preset_name, get_user_prosody_presets, save_user_prosody_preset, delete_user_prosody_preset}` per branch (pins handler→facade call style) and `qwen3_tts.interface.voice_helpers.{get_prosody_choices, get_user_prosody_choices}` for refresh branches; arm-then-change-target → re-arm, target never applied; seeded-config "excited" first click → hard error, stays disarmed; **expiry case both handlers** (`{"armed": True, "ts": time.time()-6, "armed_name": "x"}` → re-arm with timed-out copy, target not applied); conditional-reset cases (mismatched current selection → no value reset).
   - `TestProsodyPresetUiWiring` — `_read_source`: `.click(` blocks with the full **input** name sets (5/4 incl. `custom_prosody`, `design_prosody`) and 7-slot **output** lists; no `gr.Tab` listener; amended info-string present; `variant="stop"` on delete button.
   Plus **edit `tests/test_ui_facade.py`** — add `@patch("qwen3_tts.interface.voice_helpers.get_user_prosody_choices", return_value=["(none)"])` to both `build_ui` tests (**lines 224 and 244**).
   RED evidence: `python -m unittest tests.test_prosody_preset_builder -v` → per-test failures (no collection error). **Commit `test: prosody preset builder tests (RED)`.** Gate A = santa round on the test diff alone.
2. **GREEN data layer** — `core/config/presets.py` per API + facade re-exports. **Commit `feat: user-defined prosody preset data layer`.** Gate A PASS gates this step.
3. **GREEN UI** — `voice_helpers.py` constants/formatter/choices; `tabs_generation.py` (`components` import; 4 label constants; module-level `_on_save_prosody_preset`/`_on_delete_prosody_preset` — **zero direct `load_config`/`save_config` calls and zero re-implemented validation logic**: they call `validate_prosody_preset_name` + `get_user_prosody_presets` for branch decisions and the CRUD functions for all writes, module-style through the `core_config` alias so `qwen3_tts.core.config.*` patches bind at call time; one shared config load for classify; Accordion + wiring; `_build_design_tab` 4th return at :397; `_build_custom_tab(…, design_prosody)` at :400); `_facade.py:264/:268`. **Commit `feat: Custom-tab prosody preset save/delete`.** Gate B = santa round on the full diff.
4. **Docs** — `CLAUDE.md` line-5 features sentence: append clause (guard ≤300 — verify `wc -l` at edit time; **must add 0 lines**). `docs/CONFIG.md` §`prosody_presets` (~176): prose only — add the save/delete sentence AND the legacy-shadowing note ("a hand-edited factory-name override still applies via the merged view but is not manageable in the UI; restore the factory text by editing config.json"). README bullet (~587). **No COMMANDS.md / Colab / Docker / install.sh changes.** **Commit `docs: prosody preset builder`.**

## Adversarial gates — santa-loop verification (both gates)

Each gate = **ecc:santa-loop**: the implementing agent dispatches TWO parallel reviewer subagents with repo read access (no self-review). Reviewer rules: verify every claimed defect against code before reporting; review the ASSIGNED diff only; an unverifiable fact is stated as such, never guessed. Gate A consumes `git show <red-commit> -- tests/test_prosody_preset_builder.py tests/run_batches.py` (tests alone at RED — catches hollow tests); Gate B consumes `git diff main...HEAD` (full feature diff, tests + data layer + UI + docs). Verdicts: PASS or findings → fix → re-review (findings under the never-amend rule land as additional commits; re-review consumes the cumulative diff) until **2×PASS**; cap 3 rounds, then stop and surface to the user. Gate A gates step 2; Gate B gates step 4.

## Verification

```bash
python -m unittest tests.test_prosody_preset_builder tests.test_batches_coverage -v
python tests/run_batches.py --batch 4
python -m pytest tests/test_voice_helpers.py tests/test_voice_ui.py tests/test_ui_facade.py tests/test_ui_confirm_patterns.py tests/test_config.py tests/test_default_presets.py tests/test_ui_tab_select_wiring.py tests/test_voice_features.py -q
python -m ruff check qwen3_tts tests
python -m mypy qwen3_tts/core qwen3_tts/interface
bandit -r qwen3_tts -c pyproject.toml
python -m qwen3_tts.tools.check_config_docs
python -m unittest tests.test_claude_md -v && wc -l CLAUDE.md
python -m pytest tests/ -m "not e2e" -q     # CI-coverage-equivalent pre-push gate
```
Manual (`tts server start` + custom model loaded, then `tts ui`): save → appears in both tabs' Style Preset dropdowns without restart → `tts list prosody` shows it → generate with it → overwrite-confirm → delete-confirm → factory presets unaffected → seeded-config check: with a default-reset config.json, "My presets" stays empty and no factory name is deletable → **5 s confirm windows: if a scripted/browser click misses, invoke the module-level handler directly with an armed state dict as the definitive check** → keyboard: tab into accordion, toggle Enter/Space; confirm statuses announced (VoiceOver / Accessibility Inspector); arm Save, wait >5 s, click again — status text must change ("timed out" copy).

## Risks honored

Merge/corrupt-write composition (rule 2 + failure-path tests) · factory-seeded & shadowed-override states (raw-base accessor + filtered-view isolation + seeded tests) · unkeyed confirm (target-keyed state + re-arm + expiry tests) · whitespace-variant bypass (handler strips once) · arm-state leak across branches (branch invariants) · unimplementable conditional reset (dropdowns as inputs, 4-way-reviewed) · OSError/ValueError stranding (catch tuples + strerror-only messages) · stale labels + double-append (conditional value= resets, own-input-value comparisons) · patch seams (function-level test imports; `_read_source`; lazy-facade idiom; disk-free payload-capture tests; hermetic `build_ui` patches; handler→facade call style pinned by per-branch patches) · silent confirm-expiry (pinned distinguishing copy + expiry tests) · format drift of `" - "` rendering (exact-literal format test) · batch registration atomic with test file · `generation.py`/`shared.py` untouched · no `gr.Tab` listener · user pushes personally; after push I open the PR.
