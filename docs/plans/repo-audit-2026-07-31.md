# Repo Audit — 2026-07-31

Evidence-based audit of `main` @ `e94b806` (post output-folder merge), run against the
systematic-debugging, e2e-testing, test-coverage, python-testing and python-patterns
skill checklists. Every finding below was **measured**, not inferred.

## Status — all five P0/P1 findings closed (2026-08-02)

| Finding | Branch | Commit |
| --- | --- | --- |
| P0-1 broken shipped alias | `fix/p0-1-default-config-prompt-refs` | `bf5a90b` (fix was already on `main` in `b98501a`; this adds the missing guard) |
| P0-2 false-green async test | `fix/p0-2-false-green-async-test` | `04377d3` |
| P1-1 E2E arbitrary sleeps | `fix/p1-1-e2e-condition-polls` | `475617a` |
| P1-2 dead code + DRY | `fix/p1-2-history-refresh-dry` | `3eb7114` |
| P1-3 alias prompt resolution | `fix/p1-3-missing-prompt-error` | `1286dd7` |

Each branch is cut from `main` and was verified **standalone**, not merely as part of a
stack — they merge in any order. Remaining: **P2-1**, **P2-2**, **P3** (untouched).

Two findings surfaced *by* this work and are not yet written up as sections:
1. **`/health` returns `"status":"ok"` for an unusable server.** During P1-1 the server
   logged `Inference complete: 22 chars, 7314.2s` — a two-hour generation — while still
   reporting ok. Every generation-bearing E2E test blew `GEN_TIMEOUT_MS`, which read
   exactly like a code regression. A health signal that cannot distinguish healthy from
   catatonic costs real debugging time.
2. **`test_e2e_playwright._get_auth_token()` reads the legacy `~/.voice_server_token`**,
   not the canonical `~/.config/qwen3-tts/.voice_server_token`; `.voice_server.log` shows
   `Auth failure: missing_token ... on POST /unload-model`. Likely contributes to
   model-management test flakiness. Uninvestigated.

## Headline

The repo is in good health. All static gates pass, coverage is above target, and the
error-handling hygiene is genuinely excellent. There is **one real product defect**
(a broken shipped alias) and **one false-green test**. The rest is maintainability debt,
led by four files that breach the project's own 800-line rule.

### What's already clean (measured, no action needed)

| Gate | Result |
| --- | --- |
| `ruff check qwen3_tts tests` | clean |
| `mypy` core + server + interface | clean, 45 files |
| `bandit -lll` | 0 HIGH |
| Non-E2E suite | 2500 passed, 4 skipped |
| Coverage | **85%** overall (target 80%) |
| Bare `except:` | **0** |
| `TODO`/`FIXME`/`HACK`/`XXX` | **0** |
| Mutable default args | **0** |
| Silent `except Exception: pass` | 1 |

Only **one** module sits below 80% coverage. The 40+ `skipUnless` guards are all
legitimate optional-dependency gates, not disabled tests.

---

## Ranked findings

Ranked by **severity × blast radius**, then tie-broken by **inverse risk-of-fix** —
i.e. cheap, safe, high-value work first.

### P0-1 — Shipped `default` alias is broken on every fresh install — ✅ DONE (2026-07-31)

**Severity: Medium-High · Blast radius: every new user · Risk to fix: Very low · Effort: 15 min**

**Resolved:** the config fix landed in `b98501a` (option 1 — both seeds emptied). That
commit shipped no test, so the regression guard the finding asked for was added separately
as `TestDefaultConfigPromptReferences` in `tests/test_config.py` (batch 1). It encodes the
asymmetry that makes this finding non-obvious: a dangling `default_clone_prompt` is safe
(the backend-aware scan catches it) while a dangling `aliases[*]["prompt"]` is not
(`generate.py` short-circuits the fallback). Proven non-vacuous by running it against the
exact pre-fix seeds — 3 of its 4 tests go red, covering the alias failure, the misleading
seed, and the `.pt`-on-MLX secondary problem.

**Residual (not fixed, tracked below as P1-3):** option 3 was not applied, so the
short-circuit at `generate.py:711` still means a *user-created* alias pointing at a
deleted prompt raises an unhandled `FileNotFoundError`. The shipped config can no longer
trigger it, but the sharp edge remains for hand-written aliases.

`get_default_config()` ships exactly one alias, and it points at a file that does not
exist anywhere in the repo:

```python
# qwen3_tts/core/config.py:373
"aliases": {
    "default": {"prompt": "default_clone.pt", "preset": "consistent"},
},
# also: default_clone_prompt: 'default_clone.pt'
```

`find . -name "default_clone*"` → **no matches**.

The `default_clone_prompt` key degrades gracefully (`get_default_clone_prompt()` falls
back to the first available prompt on disk — verified). **The alias path
does not.** At `qwen3_tts/interface/generate.py:712`:

```python
prompt_file = alias_prompt or get_default_clone_prompt(config)
```

A truthy `alias_prompt` short-circuits the fallback, so `tts --alias default` resolves to
the missing `default_clone.pt` and raises `FileNotFoundError`. The one alias advertised by
`tts list aliases` is the one that fails.

Secondary problem: `.pt` is torch-only. On the MLX backend a prompt needs `.wav` + `.txt`,
so even a present `default_clone.pt` would be wrong for the default Apple-Silicon path.

**Fix options** (pick one):
1. Drop the `aliases` seed from `get_default_config()` — ship `{}` like the live config already has. Simplest, no behavioural surprise.
2. Point both keys at a real, backend-appropriate prompt.
3. Make the alias path fall back too: `prompt_file = alias_prompt or ...` → resolve-and-verify, falling back when the aliased file is missing.

Option 1 is recommended: the shipped alias adds no value and its only effect today is a
hard failure. Add a regression test asserting every prompt referenced by
`get_default_config()` either exists or resolves through the fallback.

### P0-2 — `test_vllm_concurrent_integration.py` is a false green — ✅ DONE (2026-07-31)

**Severity: Medium · Blast radius: test signal integrity · Risk to fix: Very low · Effort: 5 min**

**Resolved:** file deleted; guard added as `tests/test_async_test_hygiene.py` (registered in
`BATCHES` batch 1). Suite went 2500 → 2499 excluding the guard, 2502 including its 3 tests;
`RuntimeWarning: coroutine ... was never awaited` count is now 0 under `-W error::RuntimeWarning`.
The guard covers both false-green shapes — `async def test_*` on a plain `TestCase`, and unmarked
bare coroutine tests (pytest-asyncio `strict` mode) — and was verified to fire on a synthetic probe
for each while ignoring `IsolatedAsyncioTestCase`, `@pytest.mark.asyncio`, and nested `asyncio.run`.

The file is a no-op **twice over**:

```python
class TestVLLMConcurrentIntegration(unittest.TestCase):   # not IsolatedAsyncioTestCase
    async def test_concurrent_generation_requests(self):  # never awaited
        ...
        self.skipTest("Requires full server setup with authentication")  # line 35, unconditional
```

1. `async def` inside a plain `unittest.TestCase` returns a coroutine that is never
   awaited. unittest sees a non-`None` return and reports **passed** without executing a
   single line of the body.
2. Even if it did run, line 35 skips unconditionally.

It inflates the suite count and emits `RuntimeWarning: coroutine ... was never awaited` on
every run. Verified via `pytest -W error::RuntimeWarning`.

For contrast, the two sibling async suites are correct and should be left alone:
`test_python_review_fixes.py:231` uses `unittest.IsolatedAsyncioTestCase`, and
`test_vllm_async_nonblocking.py` wraps inner coroutines in `asyncio.run(...)`.

**Fix:** delete the file (it tests nothing and never can in its current form), or convert
to `IsolatedAsyncioTestCase` and implement it properly behind a real server-availability
guard. Deleting is recommended — an honest absence beats a fake pass.

**Guard the class, not just the instance:** add a meta-test asserting no
`unittest.TestCase` subclass in `tests/` declares an `async def test_*`. This is a
recurring, silent failure mode.

### P1-1 — E2E leans on arbitrary sleeps — ✅ DONE (2026-08-02)

**Resolved** on `fix/p1-1-e2e-condition-polls` (`475617a`). All **16** call sites
converted; `grep -c wait_for_timeout` is now 0 in both E2E suites. Each sleep was replaced
by the condition it stood in for — tab activation, listbox open/option/closed (6 sleeps
factored into 3 shared helpers), cancel, model-table refresh, accordion state, seed-field
commit, sidecar-on-disk, and the seed broadcast. `poll_until()` landed in
`tests/e2e_helpers.py` for conditions the browser cannot see (disk).

Verified on a **freshly restarted** server: full `test_e2e_playwright.py` = 9 passed,
2 failed, 2 skipped, where the 2 are exactly `test_08` / `test_10` — this doc's own
known-red set, re-confirmed red on clean `main` the same day.
`test_e2e_history_clear_copy.py` went 5/5 consecutive green at ~6 s vs 10.5 s before,
because polls return on the condition instead of always sleeping.

**Two regressions were introduced during the conversion and fixed**, both proven mine by
stash + re-running the identical command on clean `main`:
`test_01` read the clipboard with no wait of its own (the clipboard write is a *separate*
async effect from the status update a status-poll would catch); and `offsetParent` is
`null` for `position: fixed` elements, which Gradio 6 uses for the dropdown listbox — so
that check read an open dropdown as hidden and made the mirror "closed" check vacuously
true. Use `getClientRects().length`.

**Caveat on the 5× criterion below:** met for `test_e2e_history_clear_copy.py` (5/5) and
for `test_12`+`test_13` (3/3, the contested pair). The remaining `test_e2e_playwright.py`
tests ran once each in full-suite context — each pass costs ~4.5 min *and* degrades the
server. If a converted wait flakes later, that is the gap it came through.

The known-red `test_08` / `test_10` were reassessed as this finding suggested: they are
**not** part of the sleep cluster. Both fail on a 15 s `wait_for_function` on clean `main`,
independent of this change.

**Severity: Medium · Blast radius: CI trust / wasted sessions · Risk to fix: Low-Medium · Effort: Medium**

16 `wait_for_timeout` vs 10 `wait_for_function` across the E2E suite; **15 of the 16 are in
`tests/test_e2e_playwright.py`**. The e2e-testing skill flags fixed sleeps as the primary
flakiness source, and this is not theoretical here: the `test_13` blind
`wait_for_timeout(3000)` was part of what made that test unreadable for two sessions
(fixed 2026-07-31 by switching to a condition-based `_wait_for_history_row`).

**Fix:** convert the remaining sleeps to condition polls, prioritising any that gate an
assertion (as opposed to cosmetic settle-time after a click). `_wait_for_history_row` and
`_wait_for_visible_status` are the in-repo patterns to copy. Note the known-red
`test_08_load_model` / `test_10_load_unload_cycle` 15s `wait_for_function` timeouts likely
belong to this cluster — reassess them after the conversion.

### P1-2 — Dead production code + a DRY violation — ✅ DONE (2026-08-02)

**Resolved** on `fix/p1-2-history-refresh-dry` (`3eb7114`), taking the preferred option
(wire the facade to the shared function). The two were not quite interchangeable, which is
*why* they had diverged: the `_facade` closure returned `(history_state, history_df)`
because the generation chains write both in one step, while the shared function returned
rows only. `refresh_history_from_disk` now returns both halves, so the duplication has
nowhere left to hide.

`_facade` calls it module-style (`_shared.refresh_history_from_disk`) so `mock.patch` keeps
targeting the definition site, per CLAUDE.md on moved-module patch seams — and no `@patch`
anywhere referenced the old name, checked before moving anything.

The unit test gained an assertion on the **state** half: `_facade` wires it straight into
`history_state`, so a stale list leaking through would resurrect the exact render race
`test_13` exists to prevent, and the rows-only assertion would not have caught it.

Verified: full non-E2E suite **2500 passed, 4 skipped — identical to `main`'s baseline**,
so the refactor is behaviour-preserving.

**Severity: Low · Risk to fix: Very low · Effort: 10 min**

`shared.refresh_history_from_disk()` (`qwen3_tts/interface/ui/shared.py:760`) has **zero**
production callers — only `tests/test_history_disk_rederive.py:31`. Meanwhile
`_facade.py:419-422` carries an inline `_refresh_history` closure duplicating the same
disk-derive logic.

**Fix:** wire `_facade.py` to call the shared function (preserving its `history_state`
return), or delete the shared one and let the test target the closure. Wiring is preferred
— it keeps the unit test meaningful and removes the duplication.

### P1-3 — Alias prompt resolution has no fallback and no friendly error — ✅ DONE (2026-08-02)

**Resolved** on `fix/p1-3-missing-prompt-error` (`1286dd7`), but **not** by the
"fall back" option sketched below — deliberately. Falling back for a prompt the user
*named* would generate in a different voice than they asked for, with nothing in the
output saying so. That is worse than failing. `_resolve_prompt_file()` therefore keeps the
two cases apart:

- **implicit** (nothing named anywhere) → fall back to the backend-aware scan, as before;
- **explicit** (`--prompt`, or an alias's prompt) → verify it exists, and on a miss print
  which source named it and `sys.exit(1)`, matching the unknown-alias handler directly
  above it in the same file.

The existence check moved to `config.prompt_file_exists()` and is now **shared** with
`get_default_clone_prompt()` rather than copied — applying P1-2's own lesson so the two
cannot drift. It also now accepts the `.wav` spelling, which `get_default_clone_prompt()`'s
MLX fallback returns but its check did not recognise (harmless before, since the scan
returned the same name anyway, but wrong).

Verified: 5 new tests, proven non-vacuous by replaying the old one-liner — **3 of the 5 go
red**. The other 2 cover the preserved fallback path and correctly pass either way. Full
suite 2505 passed, 4 skipped; ruff and mypy clean. `test_voice_alias_resolution` needed its
*fixture* fixed rather than the implementation weakened: it asserts alias dispatch using a
fake `narrator.pt`, so the new check correctly exited before its assertion ran.

**Severity: Low-Medium · Blast radius: users with hand-written aliases · Risk to fix: Low · Effort: 15 min**

Split out of P0-1, whose config fix removed the *shipped* trigger without addressing the
underlying sharp edge. At `qwen3_tts/interface/generate.py:711`:

```python
prompt_file = alias_prompt or get_default_clone_prompt(config)
```

A truthy `alias_prompt` short-circuits the missing-prompt fallback, and `prompt_file` is
passed straight downstream with no existence check — so an alias pointing at a prompt the
user later renamed or deleted surfaces as an unhandled `FileNotFoundError`. Note the
sibling paths in the same file *do* handle this politely: `Error: SRT file not found: …`
(line 609), and the same for dialogue (616) and batch (661) inputs. The alias path is the
odd one out.

**Fix:** resolve-and-verify — fall back when the aliased prompt is missing, and print a
`Error: alias '<name>' points at a missing prompt: <file>` in the style of the neighbouring
handlers rather than raising. Guarded seed-side already by
`TestDefaultConfigPromptReferences`; this would close the user-config half.

### P2-1 — Four files breach the 800-line hard limit

**Severity: Medium (maintainability) · Risk to fix: HIGH · Effort: Large**

CLAUDE.md and the global coding-style rule both cap files at 800 lines.

| File | Lines | Over |
| --- | --- | --- |
| `qwen3_tts/core/config.py` | **1443** | +80% |
| `qwen3_tts/core/engine/inference.py` | **1110** | +39% |
| `qwen3_tts/interface/generate.py` | **865** | +8% |
| `qwen3_tts/server/app.py` | **821** | +3% |
| `qwen3_tts/interface/ui/shared.py` | 798 | **2 lines from breach** |

**This is ranked P2 despite real value, because the risk is documented and high.** Two
prior incidents are on record: mock patch seams silently break on file splits (tests go
green while hitting real disk), and CodeQL dismissals do not follow moved code, reddening
an otherwise pure refactor. `app.py` is also excluded from mypy with a written rationale.

**Approach if taken:** one file per PR, `config.py` first (worst offender, and it is a
constants/IO grab-bag that splits along clean seams). Before each split, grep
`@patch("<module>.<name>")` across `tests/` and re-point every seam. Expect to re-dismiss
CodeQL alerts. Do **not** bundle these with behavioural changes.

`shared.py` at 798 deserves a pre-emptive note: the next feature to touch it will breach
the limit, so plan its extraction before adding to it.

### P2-2 — Four functions exceed the 50-line SRP limit

**Severity: Low · Risk to fix: Low-Medium · Effort: Medium**

Via `python -m qwen3_tts.tools.solid_analyzer qwen3_tts` — only 4 violations repo-wide,
which is a good result for a codebase this size:

| Function | Lines |
| --- | --- |
| `edit` | 109 |
| `rebuild` | 106 |
| `stop` | 95 |
| `_generation_options` | 67 |

**Fix:** extract cohesive blocks. Natural companions to P2-1 — `edit` and `rebuild` are
CLI command bodies and split cleanly into validate / apply / report phases.

### P3 — Housekeeping

| Item | Detail | Effort |
| --- | --- | --- |
| CLAUDE.md headroom | 289 / 300 hard limit (`tests/test_claude_md.py`). 11 lines left; prefer in-place rewrites. Consider a compaction pass to buy room. | Small |
| `server/client/__init__.py` at 69% | Only sub-80% module. The 4 uncovered lines are the `generate()` convenience wrapper and the `__main__` demo block. Add one test for the wrapper; consider deleting the `__main__` block. | 10 min |
| `.voice_server.log` pollution | 98 `Voice prompt not found` entries, 90 for a `default` prompt, generated by test runs. Known issue (tests write to the production log). Makes the log untrustworthy for real debugging. Consider redirecting the log during tests. | Small |

---

## Recommended sequence

1. ~~**P0-2** (delete the false-green test + add the `async def` meta-guard)~~ — ✅ done 2026-07-31.
2. ~~**P0-1** (fix the shipped alias + regression test)~~ — ✅ done 2026-07-31 (fix in `b98501a`, guard added after). Residual split out as **P1-3**.
3. ~~**P1-2** (dead code / DRY)~~ — ✅ done 2026-08-02 (`3eb7114`).
4. ~~**P1-1** (E2E sleep conversion)~~ — ✅ done 2026-08-02 (`475617a`). Also closed
   **P1-3** (`1286dd7`), which was split out of P0-1 mid-sweep.
5. **P2-1 / P2-2** — only as deliberate, isolated PRs. Not alongside feature work. **Still open.**
6. **P3** — opportunistic. **Still open.**

Items 1-4 are complete. The estimate held for P1-2 and P1-3; **P1-1 ran well over** — the
conversion itself was quick, but two self-inflicted regressions and a degraded server that
still reported `"status":"ok"` accounted for most of the time. Budget E2E conversions by
verification cost, not diff size.

## Verification for each change

- After every item: `conda run -n qwen3-tts-mlx ruff check qwen3_tts tests` and the
  non-E2E suite (baseline **2500 passed, 4 skipped**).
- P0-1: ✅ met differently and more strongly. `tts --alias default` is no longer
  assertable — the shipped config now seeds **no** aliases, so there is no `default` alias
  to invoke. The guard instead asserts the invariant behind that command (every prompt
  `get_default_config()` names must resolve), which holds for any future seed rather than
  one hardcoded alias name, and is backend-agnostic. Non-vacuity was proven by replaying
  the pre-fix seeds: 3 of 4 tests go red.
- P0-2: suite count must **drop by 1** (2499) — a count that stays at 2500 means the
  no-op test is still being counted.
- P1-1: run each converted E2E test 5× consecutively; flakiness shows up in repetition,
  not a single pass. **Partially met — see the caveat in the P1-1 section.** 5/5 for
  `test_e2e_history_clear_copy.py`, 3/3 for `test_12`+`test_13`, once each in full-suite
  context for the rest.
- P1-2: non-E2E suite must stay at exactly the baseline (**2500 passed, 4 skipped**) — a
  behaviour-preserving refactor that changes the count changed behaviour.
- P1-3: replay the old `alias_prompt or get_default_clone_prompt(config)` one-liner against
  the new tests; **3 of 5 must go red**. A guard written against already-fixed code proves
  nothing — the failure mode P0-2 exists to punish.
- Always spell out `conda run -n qwen3-tts-mlx` for E2E — a bare `python -m pytest`
  launches the UI under the banned gradio 6.14.0 (now guarded by
  `tests/e2e_helpers.py::assert_supported_gradio`).
- **Before trusting ANY E2E failure, check `.voice_server.log` inference times, not just
  `curl /health`.** A server can report `"status":"ok"` while taking 7314 s to synthesise
  22 characters; every generation-bearing test then blows `GEN_TIMEOUT_MS` and looks like
  a code regression. Restart the server between full E2E passes — degradation accumulates
  across load/unload tests, so the failures cluster in the *later* tests and get
  misattributed to whatever changed most recently.
