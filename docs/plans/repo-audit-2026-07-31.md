# Repo Audit — 2026-07-31

Evidence-based audit of `main` @ `e94b806` (post output-folder merge), run against the
systematic-debugging, e2e-testing, test-coverage, python-testing and python-patterns
skill checklists. Every finding below was **measured**, not inferred.

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

### P0-1 — Shipped `default` alias is broken on every fresh install

**Severity: Medium-High · Blast radius: every new user · Risk to fix: Very low · Effort: 15 min**

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

### P1-1 — E2E leans on arbitrary sleeps

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

### P1-2 — Dead production code + a DRY violation

**Severity: Low · Risk to fix: Very low · Effort: 10 min**

`shared.refresh_history_from_disk()` (`qwen3_tts/interface/ui/shared.py:760`) has **zero**
production callers — only `tests/test_history_disk_rederive.py:31`. Meanwhile
`_facade.py:419-422` carries an inline `_refresh_history` closure duplicating the same
disk-derive logic.

**Fix:** wire `_facade.py` to call the shared function (preserving its `history_state`
return), or delete the shared one and let the test target the closure. Wiring is preferred
— it keeps the unit test meaningful and removes the duplication.

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
2. **P0-1** (fix the shipped alias + regression test) — the only real user-facing defect found.
3. **P1-2** (dead code / DRY) — small, self-contained, clears a known wart.
4. **P1-1** (E2E sleep conversion) — highest ongoing payoff; do it before the next feature so future failures are trustworthy.
5. **P2-1 / P2-2** — only as deliberate, isolated PRs. Not alongside feature work.
6. **P3** — opportunistic.

Items 1-3 together are well under an hour and carry near-zero regression risk. Item 4 is
the one that most improves day-to-day signal quality.

## Verification for each change

- After every item: `conda run -n qwen3-tts-mlx ruff check qwen3_tts tests` and the
  non-E2E suite (baseline **2500 passed, 4 skipped**).
- P0-1: assert `tts --alias default` succeeds on a config derived from
  `get_default_config()`, on both backends.
- P0-2: suite count must **drop by 1** (2499) — a count that stays at 2500 means the
  no-op test is still being counted.
- P1-1: run each converted E2E test 5× consecutively; flakiness shows up in repetition,
  not a single pass.
- Always spell out `conda run -n qwen3-tts-mlx` for E2E — a bare `python -m pytest`
  launches the UI under the banned gradio 6.14.0 (now guarded by
  `tests/e2e_helpers.py::assert_supported_gradio`).
