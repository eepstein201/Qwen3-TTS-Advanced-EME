# Track T1b — User-Defined Generation-Preset Builder (Clone Tab)

Date: 2026-09-21 · Status: EXECUTED 2026-09-21 (approved v1.1 with D1a; RED → GREEN → Gate B on branch `feature/generation-preset-builder`)
Predecessor: Track T1 (prosody preset builder, PR #322) — this spec mirrors its
shape, mechanics, and gate process wherever the two features rhyme.

## 1. Context and user need

The user wants the T1 save/delete experience on the **Clone tab** — but for the
**"Preset" dropdown** (`tabs_generation.py:254`), not prosody. A live feasibility
probe (2026-09-21, 4 generations against the running server, ASR-verified) proved:

- `instruct` on clone requests is **ignored** (byte-identical output with/without,
  same seed) — the clone path has no style channel (`inference.py` torch
  `generate_voice_clone(**clone_kwargs)` and MLX `model.generate(...)` both omit it).
- Style text prepended to the spoken text is **read aloud verbatim** (ASR transcript).

So style/prosody presets cannot exist on clone. What clone *does* honor is the
generation preset: `_prepare_streaming_config` applies it on **all three modes**
(`generation.py:237-241`, blind `gen_params.update(presets[preset])`). All three
tabs carry a "Preset" dropdown (`:254` clone, `:416` design, `:623` custom), fed
by `shared.get_presets()` = `"(none)" + list(get_generation_presets(config))`,
and `get_generation_presets` (`core/config/presets.py:71-91`) already merges
user entries from `config["presets"]` — the backend is complete; only a builder
UI is missing.

**Feature:** a "My generation presets" accordion on the Clone tab that saves the
four current sampling-slider values (temperature / top-k / top-p / repetition
penalty) under a user-chosen name, with two-step overwrite/delete confirm —
exactly T1's UX, applied to the dropdown the user actually drives.

## 2. Verified anchors (re-verify at session start — T1 pre-flight lesson)

| Anchor | Location | Verified |
|---|---|---|
| Clone "Preset" dropdown | `tabs_generation.py:253-255` | 2026-09-21 |
| Design "Preset" dropdown | `tabs_generation.py:415-417` | 2026-09-21 |
| Custom "Preset" dropdown | `tabs_generation.py:622-624` | 2026-09-21 |
| Preset application (blind update) | `generation.py:237-241` | 2026-09-21 |
| Factory presets (8) | `core/config/presets.py:19-69` — stable, natural, expressive, audiobook, conversational, broadcast, dramatic, whisper | 2026-09-21 |
| `get_generation_presets` merge | `core/config/presets.py:71-91` (user overrides same key, adds new) | 2026-09-21 |
| Sampling sliders | `generation.py:430-433` (temp 0.1–1.5 s0.05, top_k 1–100 s1, top_p 0.1–1.0 s0.01, rep 1.0–2.0 s0.01) | 2026-09-21 |
| Server param ranges (contract) | `server/validation.py:61-64` — temp [0,2], top_k [1,1000], top_p [0,1], rep [0.5,2] | 2026-09-21 |
| `_build_clone_tab` signature/return | `tabs_generation.py:212` `(status_html, history_state)` → 4-tuple | 2026-09-21 |
| Facade unpacks clone 4-tuple | `_facade.py:305-307` | 2026-09-21 |
| Build order | `_facade.py:306,310,313` — clone, then design, then custom | 2026-09-21 |
| Closures test calls clone builder | `tests/test_ui_tabs_generation_closures.py:66` — `_build_clone_tab(None, None)`, no unpack | 2026-09-21 |
| T1 helpers to reuse (generic mechanics) | `tabs_generation.py:34-72` — `_PROSODY_*` button-label constants, `_PROSODY_DISARMED_STATE`, `_prosody_arm_is_fresh`, `_prosody_disarmed_result` | 2026-09-21 |

## 3. Scope

**In:** data layer (`core/config/presets.py` + facade re-exports), one UI helper
(`shared.py`), clone-tab accordion + two handlers, cross-tab dropdown refresh
wiring (`_facade.py`), tests, docs (CLAUDE.md Features line, CONFIG.md `presets`
prose, this plan copied to `docs/plans/`).

**Out:** any CLI/server/endpoint change; any instruct-on-clone behavior (proven
dead); seed / seed_lock / x-vector-only as preset fields; renaming the existing
`_prosody_*` helpers (surgical rule — reuse as-is); changes to factory presets.

## 4. Data layer — `qwen3_tts/core/config/presets.py`

```python
GENERATION_PRESET_NAME_MAX_LEN = 40
GENERATION_PRESET_PARAM_KEYS = ("temperature", "top_k", "top_p", "repetition_penalty")
GENERATION_PRESET_PARAM_RANGES = {  # mirrors server/validation.py:61-64
    "temperature": (0.0, 2.0), "top_k": (1, 1000),
    "top_p": (0.0, 1.0), "repetition_penalty": (0.5, 2.0),
}
_FACTORY_GENERATION_PRESET_NAMES = frozenset(k.lower() for k in DEFAULT_GENERATION_PRESETS)
```

- `validate_generation_preset_name(name) -> str | None` — error copy or None.
  Rejects: empty/whitespace; >40 chars; charset outside `[A-Za-z0-9 _\-\.]`;
  `..`; factory collision **case-insensitive** (a user `"Stable"` must NOT
  silently override factory `stable` — the merge at `:74` would).
- `validate_generation_preset_params(params) -> str | None` — requires **exactly**
  the 4 keys, numeric (`isinstance(x, (int, float))` and `not isinstance(x, bool)`),
  each within `GENERATION_PRESET_PARAM_RANGES`. Defense at the boundary: the
  downstream apply is a blind `gen_params.update`, so junk keys must never reach
  `config["presets"]`.
- `get_user_generation_presets(config=None) -> dict` — `config["presets"]`
  minus factory names (case-insensitive), as **new dicts** (never an alias).
  Corrupt/missing config swallows to `{}` — read paths never explode the UI.
- `save_user_generation_preset(name, params) -> (ok, msg)` — load **raw** config
  (factory entries, unrelated keys, and non-preset keys must survive byte-for-
  byte in structure), validate params, set `config["presets"][name] = params`,
  write via atomic `save_config`. Corrupt/missing config → `(False, msg)`,
  file untouched.
- `delete_user_generation_preset(name) -> (ok, msg)` — membership miss →
  `(False, "no longer exists")`; factory name → `(False, factory-msg)` (writer
  double-guards even though the validator runs first).
- Re-export all of the above from `qwen3_tts/core/config/__init__.py` (+ `__all__`).

## 5. UI helper — `qwen3_tts/interface/ui/shared.py`

- `get_user_generation_preset_choices() -> list[str]` — `["(none)"] + sorted(
  get_user_generation_presets())`. Bare names (no `name - text` formatter — the
  Preset dropdown shows bare factory names today). Feeds the delete dropdown.

## 6. Clone-tab UI + handlers — `qwen3_tts/interface/ui/tabs_generation.py`

**Accordion** `"My generation presets"`, `open=False`, in the clone tab's left
column (`scale=2`) directly under the Preset dropdown. Components: name
`gr.Textbox` (placeholder "Preset name (max 40 characters)"), Save button
`"Save as preset"`, Delete button `"Delete preset"`, save/delete status
`gr.Textbox`s (non-interactive), delete `gr.Dropdown` (user-only choices),
`gr.HTML` announcer (reuse `generation._announce_status("")`), two `gr.State`
armed dicts (reuse `dict(_PROSODY_DISARMED_STATE)`).

**Reuse, do not rename:** `_prosody_arm_is_fresh`, `_prosody_disarmed_result`,
`_PROSODY_DISARMED_STATE`, and the four `_PROSODY_*_BTN_*` label constants —
the mechanics are generic; the names are prosody-flavored but correct.

**`_on_save_generation_preset(state, name, temp, top_k, top_p, rep,
*three_preset_dropdown_values) -> N-tuple`** (module-level, mirrors `_on_save_prosody_preset`):
strip-once name → empty → copy; validator error → copy; params validator
(defense-in-depth; sliders are naturally in range); existing name and no fresh
arm → **arm** (overwrite confirm); fresh arm → save via writer; success →
`gr.update(choices=shared.get_presets())` on **all three** Preset dropdowns +
delete-dropdown refresh + disarm; failure → writer msg + disarm (no paths —
CWE-209 discipline).

**`_on_delete_generation_preset(state, selection, *three_preset_dropdown_values)`**
— branch order: ① nothing selected → copy; ② raw-minus-factory membership miss
(or factory name classification) → message, never arm; ③ arm (5 s window);
④ fresh arm → delete + conditional value reset: `value=NONE_CHOICE` only where a
dropdown's own current value equals the deleted name, else `gr.update(choices=…)`
without `value` (T1's conditional-reset trap).

**Wiring (D1a — full live refresh; DECIDED by user 2026-09-21):** `_build_clone_tab`
grows its return to a 5-tuple whose 5th element is a **dict** of builder refs
(buttons, states, name box, statuses, delete dropdown, announcer — avoids a
13-tuple); `_build_design_tab`/`_build_custom_tab` each return their preset
dropdown as one extra element (`_facade.py:309`/`:313` unpack lines updated);
both `.click()` wires live in `_facade.py` immediately after
`_build_custom_tab`, where all three dropdowns exist. Save/delete handlers
return `gr.update(choices=…)` slots for **all three** Preset dropdowns (8-slot
output lists: state, button, status, announcer, 3× preset dropdown, delete
dropdown). Closures test is unaffected (no unpack, no signature change).

## 7. Pinned user-facing copy

| Key | String |
|---|---|
| Save base / arm | `Save as preset` / `Confirm Overwrite? (click again)` |
| Delete base / arm | `Delete preset` / `Confirm Delete? (click again)` |
| Empty name | `Type a preset name first.` |
| Bad name | `Preset names are 1-40 characters (letters, numbers, space, dash, underscore, dot).` |
| Factory collision | `Factory presets can't be overwritten — pick a different name.` |
| Save arm status | `A preset named '<name>' exists — click Save again to overwrite.` |
| Saved | `Saved preset '<name>'.` |
| Save failed | `Could not save: <writer message>` |
| Delete nothing selected | `Select one of your presets to delete.` |
| Delete arm status | `Click Delete again to delete '<name>'.` |
| Deleted | `Deleted preset '<name>'.` |
| Membership miss | `Preset '<name>' no longer exists.` |

## 8. Write-path traps (tests must pin each)

1. **Raw-base preservation** — save/delete must not drop factory presets,
   unrelated config keys, or junk dict entries in `config["presets"]`.
2. **Factory-name silent override** — merge semantics (`:74`) make a same-key
   user preset override silently; the case-insensitive validator is the only
   barrier. Pin `" Stable "` (whitespace + case variant) rejection.
3. **Corrupt-config swallow on reads**; `(False, msg)` + untouched file on writes.
4. **Immutability** — `get_user_generation_presets` returns new dicts, never the
   loaded config's inner objects; writers never mutate their input base.
5. **Arm branches never call the writer** (T1 Gate-A C1/C2 — patch writers in
   every arm-branch test and assert `assert_not_called()`).
6. **Key whitelist** — junk/extra/missing keys rejected by
   `validate_generation_preset_params`; bools rejected (bool is int subclass).
7. **Expiry re-arms, never executes**; mismatched-target re-arms.
8. **Delete classification before arming** — factory/miss branches can never arm.
9. **Conditional reset** — value reset only for the dropdown(s) holding the
   deleted name; stale-label bug class from T1.

## 9. Tests — `tests/test_generation_preset_builder.py`

Mirrors T1's module shape; all `Test*` subclass `unittest.TestCase`; register in
`BATCHES` batch 4 (`tests/run_batches.py`). Classes: name validation (charset,
length, factory case-insensitive incl. whitespace variants), params validation
(each key missing/extra/out-of-range-both-sides/bool), user-preset reader
(raw-minus-factory, swallow, no-alias), save writer (raw base, corrupt, success
msg), delete writer (miss, factory, keeps others), save handler (arm branches
with writer-not-called, success renders message, output-slot pinning incl. all
three dropdown updates, whitespace re-arm), delete handler (branch order
①→④, expiry, conditional reset positive+negative), wiring (accordion source
window in clone tab, inputs include the four sliders, `outputs=` list pinned,
no `gr.Tab` select listener — covered by `test_ui_tab_select_wiring.py`).

Batch-4 neighbor: `tests/test_ui_tabs_generation_closures.py` keeps passing
unchanged (`:66` calls the builder without unpacking).

## 10. Gates and process

RED (tests only) → **Gate A** (two parallel reviewers, tests alone, 2×PASS,
cap 3 rounds) → GREEN (data layer → UI/handlers → facade wiring) → **Gate B**
(full diff, 2×PASS) → docs commit (CLAUDE.md Features line, CONFIG.md `presets`
prose, this plan copied to `docs/plans/2026-09-21-generation-preset-builder.plan.md`)
→ full gates: non-e2e suite, batch 4, `.venv-310` on touched modules
(env-specific imports), ruff/mypy/bandit, `test_claude_md`, config-docs check
→ feature-branch PR (no AI attribution anywhere).

All work in a fresh worktree (this session is forked; the shared checkout also
has a parallel UA-graph read window that a worktree does not disturb).
Conda: `conda run -n qwen3-tts-mlx`. No server restart needed (UI-only feature;
verify in the UI after relaunch).

## 11. Decisions

- **D1 (DECIDED 2026-09-21, user picked (a)):** full live refresh on all three
  tabs — 3 small return-tuple grows + facade wiring. Clone-only (b) rejected.
- **D2 (decided unless overridden):** factory-name overwrite **rejected** — T1
  precedent; silent-override hazard otherwise.
- **D3 (decided unless overridden):** accordion under the Preset dropdown, left
  column, `open=False`; presets contain exactly the 4 sampling params; no seed /
  seed_lock / no_transcript in presets.
