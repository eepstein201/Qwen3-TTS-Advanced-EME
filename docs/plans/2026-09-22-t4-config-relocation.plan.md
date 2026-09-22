# Track T4 — Relocate the runtime config to `~/.config/qwen3-tts/`

**Status:** SPEC — awaiting go-ahead to implement.
**Raised:** 2026-09-22 (after the T1b merge, #327).
**Scope decision:** config only. Settled 2026-09-22.
**Track, not a wave step** — outside the 49-step Open count; never moves that number.

---

## Problem

`paths._resolve_config_path()` prefers `USER_FILES_DIR/config.json` and falls back to a
repo-root `config.json` for CI and source-tree runs. On the maintainer's install
`USER_FILES_DIR` **is** the repo checkout, so both arms resolve to the same inode:

```
CONFIG_PATH        = /Users/<user>/Qwen3-TTS_UserFiles/config.json
repo-root fallback = /Users/<user>/Qwen3-TTS_UserFiles/config.json
os.path.samefile() = True
```

That single file holds two incompatible roles:

| Role | Wants |
|---|---|
| Committed project default (CI depends on it) | tracked, stock values, stable |
| Live user state (UI writes here) | untracked, personal, mutable |

Because origin is **public**, every user-named value the UI saves lands in a tracked
file. Two features already write there — prosody preset names (T1) and generation preset
names (T1b) — and the preset builders are the pattern going forward, not the exception.

On CI the two paths genuinely differ, so the collision is invisible there. It only
manifests on an install where the checkout sits at `~/Qwen3-TTS_UserFiles`.

### What is *not* affected

Verified 2026-09-22 — these are already outside the repo or already ignored, and T4
must not touch them:

| Artifact | Location | Status |
|---|---|---|
| `voice_prompts/` (188 MB) | in checkout | gitignored (`.gitignore:27`), 0 files ever committed |
| `.voice_server.{log,pid,lock}` | in checkout | gitignored |
| Generation history | `~/.voice_history.jsonl` | already outside |
| Auth token | `~/.config/qwen3-tts/` | already outside |
| Generation output | `~/Downloads/Qwen3-TTS Output` | already outside |

An earlier draft of the T4 entry in
`docs/plans/2026-09-06-consolidated-backlog-priority.plan.md` claimed relocation would
"also cover voice prompts and generation history." **That is wrong** — both were already
protected. That line was corrected in the same commit that added this spec.

### Interim guard in place

`git update-index --skip-worktree config.json` hides the file from `git add -A`,
`git add .`, and `git commit -a`. It is **per-clone** (a fresh clone or second worktree
has none) and makes `git pull` error if an upstream commit ever touches `config.json`.
T4 retires it.

**Nothing has leaked.** Preset names appear in zero commits on any branch, and
`origin/main:config.json` carries stock defaults with no `prosody_presets` key.

---

## Design — mirror the auth token exactly

`core/config/auth.py` already solved this problem for the token: a canonical new
location, a legacy fallback that is read but never written, and a warning that names the
migration. T4 reuses that shape rather than inventing one.

```python
# paths.py
_CONFIG_DIR          = pathlib.Path(os.path.expanduser("~/.config/qwen3-tts"))
CONFIG_PATH          = str(_CONFIG_DIR / "config.json")   # canonical; ALWAYS the write target
_LEGACY_CONFIG_PATH  = os.path.join(USER_FILES_DIR, "config.json")
_REPO_CONFIG_PATH    = <__file__-anchored repo root>/config.json
```

- **`CONFIG_PATH` becomes unconditional** — no filesystem probing at import. It is where
  config *lives*, the same way `TOKEN_FILE` is.
- **Reads** (`load_config`) try `CONFIG_PATH`, then `_LEGACY_CONFIG_PATH`, then
  `_REPO_CONFIG_PATH`. Falling back past the first arm logs a warning naming the target,
  matching `read_auth_token()`'s wording.
- **Writes** (`save_config`) always go to `CONFIG_PATH`, `mkdir(parents=True,
  exist_ok=True)` on `_CONFIG_DIR`, keeping the existing atomic temp-file + `os.replace`.
- **Repo-root `config.json` stays tracked and stock.** It reverts to a committed default
  that CI reads, and nothing writes to it again.

`_resolve_config_path()` is deleted; the read chain replaces it. No environment variable
is introduced — feeding one into `open()` is a CodeQL `py/path-injection` source, and the
existing `__file__` anchoring already covers CI.

### Why migration is a documented copy, not code

Automatic migration cannot distinguish the two cases it must tell apart:

- On the maintainer's install, `_LEGACY_CONFIG_PATH` **is** the repo-root file and holds
  real user state that should move.
- On CI or a fresh clone, that same path is stock defaults that should **not** be copied
  into `~/.config`.

Any `samefile()` guard that suppresses the CI case also suppresses the one install that
needs migrating. Rather than ship a heuristic that is wrong in one direction or the
other, migration is a single documented command run once (step 4). The read-chain
fallback means an un-migrated install keeps working in the meantime — exactly how the
legacy token path behaves.

---

## Implementation steps

Each step verifiable; TDD throughout (RED first), two-gate adversarial review.

1. **Audit direct `CONFIG_PATH` consumers** → verify: enumerate all 36 references across
   11 modules and classify each as *read*, *write*, or *display*. Any site that reads the
   file directly instead of calling `load_config()` must move to the read chain or be
   shown to be display-only. **This is the real work of T4** — the `paths.py` edit itself
   is ~10 lines. Known consumers: `cli_config.py`, `tools/uninstall.py`,
   `tools/healthcheck.py`, `core/engine/inference.py`, `core/config/{errors,io}.py`,
   `server/client/_base.py`, `interface/{generate,generate_server}.py`.

2. **RED: write the tests** → verify: each fails for the stated reason.
   - `CONFIG_PATH` is `~/.config/qwen3-tts/config.json` regardless of what exists on disk
   - `load_config()` prefers the canonical path when present
   - `load_config()` falls back to legacy, and warns, when only legacy exists
   - `load_config()` falls back to repo-root when neither exists (the CI case)
   - `save_config()` writes to the canonical path even when the read came from legacy
   - `save_config()` creates `~/.config/qwen3-tts/` when absent
   - a save never modifies `_LEGACY_CONFIG_PATH` or `_REPO_CONFIG_PATH`
     (assert on file mtime/content, not just on the return value)

3. **GREEN: `paths.py` + `io.py`** → verify: step-2 tests pass; `ruff`, `mypy`, `bandit`
   clean; Batches 1–4 green; `make check-config-docs` passes.

4. **Migrate this install** (one time, documented in RUNBOOK):
   ```bash
   mkdir -p ~/.config/qwen3-tts
   cp ~/Qwen3-TTS_UserFiles/config.json ~/.config/qwen3-tts/config.json
   chmod 600 ~/.config/qwen3-tts/config.json
   cd ~/Qwen3-TTS_UserFiles
   git update-index --no-skip-worktree config.json
   git checkout -- config.json          # restore the tracked stock default
   ```
   → verify: `tts config path` prints the new location; `tts config show` still lists the
   user's presets; `git status` shows `config.json` clean; `git ls-files -v config.json`
   shows `H`, not `S`.

5. **Docs + plan reconciliation** → verify: `docs/CONFIG.md` and `docs/RUNBOOK.md` name
   the new location; CLAUDE.md's settings table updated; T4 marked DONE with its PR
   number. (The overstated "voice prompts and generation history" claim was already
   corrected alongside this spec.)

---

## Risks

| Risk | Mitigation |
|---|---|
| A direct `CONFIG_PATH` reader is missed and silently reads the wrong file | Step 1 is an explicit enumeration, not a grep-and-hope; step 2 asserts no write reaches the legacy or repo paths |
| CI regression — repo-root config no longer found | Dedicated test for the "neither canonical nor legacy exists" case; the `__file__` anchoring is unchanged |
| Tests patch `qwen3_tts.core.config.CONFIG_PATH` (the facade attribute) and break | 29 existing patch sites across 6 files; the facade re-export stays, so the seam is preserved — but each must be re-run, not assumed |
| A second install/worktree reads a stale legacy config | The warning names the canonical path on every fallback read |
| `~/.config/qwen3-tts/` permissions | Mirror the token: `mkdir(parents=True, exist_ok=True)`; config is not a secret but sits beside one |

## Rejected alternatives

- **Gitignore `config.json` in place** — leaves the dual-role file, keeps user data in the
  checkout, and breaks CI's committed default (the file must stay tracked for CI to read).
- **Environment-variable override** — CodeQL `py/path-injection` source; the existing
  resolver docstring already rejects this for the same reason.
- **Automatic migration on import** — cannot distinguish "user state" from "stock CI
  defaults" at the one path where they collide (see above); also makes module import
  perform filesystem writes.
- **Moving `voice_prompts/`, logs, PID/lock** — zero exposure benefit (all already
  ignored or outside), and moving 188 MB of irreplaceable voice data carries real risk.

---

## Out of scope

- Relocating any artifact other than `config.json`.
- Rewriting published history. Two personal voice-prompt names were scrubbed from the
  working tree in #329 but remain in prior commits; removing them there needs a
  force-push and its own explicit decision.
