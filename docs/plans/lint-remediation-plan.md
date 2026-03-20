# Phase 8: Pre-existing Lint Cleanup

Audit date: 2026-03-20
Ruff version: system `/opt/homebrew/bin/ruff`
Scope: `qwen3_tts/` (source) + `tests/` (test suite)

## Summary

- **Source code (`qwen3_tts/`): 0 errors** -- clean
- **Test code (`tests/`): 83 errors** across 26 files
- Safe fixes (auto-fixable): 52
- Safe fixes (manual): 10
- Low risk (manual): 12
- Medium risk (requires review): 9

Ruff reports: `63 fixable with --fix (11 hidden fixes enabled with --unsafe-fixes)`

## Error Code Breakdown

| Code | Count | Description | Risk |
|------|-------|-------------|------|
| F401 | 52 | Unused import | Safe (most auto-fixable) |
| F811 | 11 | Redefinition of unused name | Safe (auto-fixable) |
| F841 | 10 | Local variable assigned but never used | Low risk |
| F541 | 1 | f-string without placeholders | Safe (auto-fixable) |
| E401 | 1 | Multiple imports on one line | Safe (auto-fixable) |
| E402 | 1 | Module-level import not at top of file | Medium risk |
| E731 | 1 | Lambda assignment (use def instead) | Medium risk |
| E741 | 2 | Ambiguous variable name (`l`) | Safe (manual rename) |

Total by error family:
- **F-series (pyflakes)**: 74 errors
- **E-series (pycodestyle)**: 5 errors
- **Style/naming**: 4 errors

---

## 8A: Safe Fixes (auto-fixable with `ruff check --fix`)

**52 errors.** These are purely cosmetic removals of unused imports, redundant f-string prefixes, import splitting, and redefinitions. No behavior change possible.

### F401: Unused imports (41 auto-fixable instances)

| File | Line(s) | Import(s) to remove |
|------|---------|---------------------|
| `tests/test_async_concurrency.py` | 88 | `qwen3_tts.core.engine.voice_prompt` |
| `tests/test_cli_daemonization.py` | 7, 10 | `subprocess`, `time` (both also re-imported locally) |
| `tests/test_client.py` | 420 | `io` (imported on same line as `base64` via E401) |
| `tests/test_fastapi_endpoints.py` | 16 | `unittest.mock.MagicMock` |
| `tests/test_fastapi_server.py` | 8, 41 | `pathlib.Path`, `fastapi.testclient.TestClient` |
| `tests/test_healthcheck.py` | 5 | `pathlib` |
| `tests/test_integration.py` | 23, 25, 27, 28 (x2), 393, 396, 397 | `tempfile`, `time`, `pathlib.Path`, `MagicMock`, `patch`, `generate`, `healthcheck`, `model_cache` |
| `tests/test_model_cache.py` | 5 | `datetime.timedelta` |
| `tests/test_ocp_strategy.py` | 2, 3 | `pytest`, `unittest.mock` |
| `tests/test_protocols.py` | 2, 3 | `pytest`, `runtime_checkable` |
| `tests/test_remediation_2026_03_03.py` | 12 | `platform` |
| `tests/test_remediation_2026_03_04.py` | 17 | `unittest.mock.MagicMock` |
| `tests/test_solid_analyzer.py` | 2, 3, 4, 5 | `ast`, `tempfile`, `pathlib.Path`, `unittest.TestCase` |
| `tests/test_validation.py` | 3 | `unittest.mock` |
| `tests/test_voice_config.py` | 7, 46, 310 | `MagicMock`, `TOKEN_FILE`, `platform` |
| `tests/test_voice_features.py` | 7, 102 | `MagicMock`, `argparse` |
| `tests/test_voice_generation.py` | 8 | `MagicMock` |
| `tests/test_voice_helpers.py` | 2, 3 | `pytest`, `unittest.mock` |
| `tests/test_voice_prompts.py` | 7, 427 | `MagicMock`, `CONFIG_PATH` |
| `tests/test_voice_server.py` | 5, 7 | `time`, `MagicMock` |
| `tests/test_voice_streaming.py` | 6 (x2) | `patch`, `MagicMock` |
| `tests/test_voice_ui.py` | 6 (x2), 11 | `patch`, `MagicMock`, `_skip_generate` |
| `tests/test_wavesurfer_js.py` | 471 | `os` |

### F811: Redefinition of unused name (11 auto-fixable instances)

These are local re-imports of names already imported at module scope. The module-scope import is unused (covered above in F401); the local re-import is the one actually used.

| File | Line | What is redefined |
|------|------|-------------------|
| `tests/test_cli_daemonization.py` | 90 | `subprocess` (module-scope import on L7 is unused) |
| `tests/test_voice_features.py` | 301, 341 | `MagicMock` (module-scope import on L7 is unused) |
| `tests/test_voice_generation.py` | 186, 202 | `MagicMock` (module-scope import on L8 is unused) |
| `tests/test_voice_prompts.py` | 542, 593 | `MagicMock` (module-scope import on L7 is unused) |
| `tests/test_voice_ui.py` | 97 (x2), 112 (x2) | `patch`, `MagicMock` (module-scope import on L6 is unused) |

**Fix pattern:** Remove the unused module-scope import (already counted in F401 above). The local re-imports become the sole definition, resolving both F401 and F811.

### F541: f-string without placeholders (1 instance)

| File | Line | Current | Fix |
|------|------|---------|-----|
| `tests/run_batches.py` | 282 | `f"\nSome batches failed..."` | Remove `f` prefix |

### E401: Multiple imports on one line (1 instance)

| File | Line | Current | Fix |
|------|------|---------|-----|
| `tests/test_client.py` | 420 | `import io, base64` | Split into `import base64` (remove `io` per F401) |

### Execution

```bash
# Dry run to preview
ruff check tests/ --fix --diff

# Apply safe auto-fixes
ruff check tests/ --fix

# Apply unsafe fixes (F811 redefinitions -- safe in practice)
ruff check tests/ --unsafe-fixes --fix

# Verify tests still pass
python -m pytest tests/ -v --tb=short
```

**Note:** `--fix` only applies "safe" fixes (52 of the 63 fixable). The remaining 11 are "unsafe fixes" (the F811 redefinitions) which require `--unsafe-fixes` flag. These are still safe in practice since they just remove the unused module-scope import, but ruff categorizes them as unsafe because the name was technically defined.

---

## 8B: Low Risk Manual Fixes

**12 errors.** These require manual edits but have no behavioral impact.

### F841: Unused local variables (10 instances)

| File | Line | Variable | Context | Fix |
|------|------|----------|---------|-----|
| `tests/test_async_concurrency.py` | 35 | `call_count` | Dict `{"count": 0}` assigned but never read | Remove assignment or add assertion using it |
| `tests/test_engine.py` | 286 | `extra` | `actual_public - expected_public` computed but never asserted | Add `self.assertEqual(extra, set())` or remove |
| `tests/test_remediation_2026_03_03.py` | 49 | `issues` | `_validate(config)` return value not used | Replace with `self._validate(config)` (no assignment) or add assertion |
| `tests/test_remediation_2026_03_03.py` | 66 | `issues` | Same pattern | Same fix |
| `tests/test_remediation_2026_03_03.py` | 77 | `issues` | Same pattern | Same fix |
| `tests/test_remediation_2026_03_03.py` | 88 | `issues` | Same pattern | Same fix |
| `tests/test_remediation_2026_03_03.py` | 192 | `original_mlx` | Saved but never restored/asserted | Remove or use in assertion |
| `tests/test_remediation_2026_03_03.py` | 351 | `original_start` | Saved but never restored/asserted | Remove or use in assertion |
| `tests/test_voice_config.py` | 67 | `token` | Return value of function not used | Replace with `_` or add assertion |
| `tests/test_voice_config.py` | 68 | `headers` | Return value of function not used | Replace with `_` or add assertion |

**Fix approach:** Most of these are test methods that call a function for its side effects but discard the return value. The cleanest fix is to either:
1. Call without assignment (e.g., `self._validate(config)` with no left-hand side), OR
2. Add an assertion on the return value (preferred -- strengthens the test)

### E741: Ambiguous variable name (2 instances)

| File | Line | Variable | Context | Fix |
|------|------|----------|---------|-----|
| `tests/test_wavesurfer_js.py` | 168 | `l` | List comprehension: `[l for l in js.split('\n') if '300000' in l]` | Rename to `line` |
| `tests/test_wavesurfer_js.py` | 176 | `l` | List comprehension: `[l for l in js.split('\n') if '60000' in l]` | Rename to `line` |

---

## 8C: Medium Risk (requires review)

**9 errors.** These could affect behavior if handled incorrectly.

### F401 in try/except blocks (6 instances -- NOT auto-fixable)

These imports exist solely to set availability flags (`HAS_DEPS`, `HAS_SIM_DEPS`, etc.). Ruff correctly reports them as "unused" because the imported names are never referenced after the `try` block, but removing them would break the availability detection.

| File | Line(s) | Import(s) | Purpose |
|------|---------|-----------|---------|
| `tests/evaluations/test_speaker_similarity.py` | 85, 86, 87 | `torch`, `torchaudio`, `WavLMForXVector` | Sets `HAS_SIM_DEPS` flag |
| `tests/test_fastapi_endpoints.py` | 52, 54 | `TestClient`, `numpy` | Sets `HAS_DEPS` flag |
| `tests/voice_test_helpers.py` | 31 | `soundfile` | Sets `HAS_FASTAPI` flag |

**Fix:** Add `# noqa: F401` comments to suppress. These imports ARE necessary -- they test whether the package is installed. Some already have noqa comments (e.g., `soundfile` on line 18 of `voice_test_helpers.py`) but others are missing them.

**Alternative (better):** Replace with `importlib.util.find_spec()` pattern, which is what ruff's hint suggests. This avoids actually importing the heavy modules just to check availability. Example:
```python
import importlib.util
HAS_SIM_DEPS = all(
    importlib.util.find_spec(mod) is not None
    for mod in ("torch", "torchaudio", "transformers")
)
```

### F401 in test_protocols.py (4 instances -- lines 27, 55, 79, 106)

These imports are inside test methods where the imported name IS used within the same method (passed to `issubclass()`). Ruff flags them incorrectly because the name is used only as a type check argument.

| File | Line | Import | Used on |
|------|------|--------|---------|
| `tests/test_protocols.py` | 27 | `ConfigProvider` | Same method, `issubclass()` and attribute checks |
| `tests/test_protocols.py` | 55 | `Generator` | Same method, `issubclass()` and attribute checks |
| `tests/test_protocols.py` | 79 | `ServerManager` | Same method, `issubclass()` and attribute checks |
| `tests/test_protocols.py` | 106 | `PromptManager` | Same method, `issubclass()` and attribute checks |

**Fix:** These are auto-fixable by ruff but the auto-fix would BREAK the tests. Add `# noqa: F401` comments to suppress. Do NOT run `--fix` on these.

**IMPORTANT:** When running `ruff check --fix` in Phase 8A, either:
1. Exclude `test_protocols.py`: `ruff check tests/ --fix --exclude tests/test_protocols.py`, OR
2. Fix `test_protocols.py` lines 2-3 first (remove truly unused `pytest` and `runtime_checkable`), then add `# noqa: F401` to lines 27/55/79/106 before running `--fix`

### E402: Module-level import not at top of file (1 instance)

| File | Line | Import | Context |
|------|------|--------|---------|
| `tests/voice_test_helpers.py` | 96 | `from contextlib import asynccontextmanager` | Placed after function definition (lines 52-93) |

**Fix:** Move import to the top imports block (after line 9). `contextlib` is stdlib so no ordering issues. Verify `_setup_fastapi_app_state()` above it does not depend on import order (it does not).

### E731: Lambda assignment (1 instance)

| File | Line | Current | Context |
|------|------|---------|---------|
| `tests/test_fastapi_endpoints.py` | 62 | `_skip = lambda f: f` | Fallback decorator when pytest unavailable |

**Fix:** Convert to def:
```python
else:
    def _skip(f):
        return f
```
Semantically identical. Verify `_skip` is only used as a decorator (it is -- applied via `@_skip` on test functions).

---

## Execution Strategy

### Recommended approach: one commit per risk tier

**Commit 1: 8A Safe auto-fixes**
```bash
# First, manually add noqa to test_protocols.py lines 27/55/79/106 to prevent breakage
# Then apply ruff safe fixes
ruff check tests/ --fix
# Then apply unsafe fixes (F811 redefinitions)
ruff check tests/ --unsafe-fixes --fix
# Verify
python -m pytest tests/ -v --tb=short
# Check remaining errors
ruff check tests/
```

**Commit 2: 8B Low risk manual fixes**
- Rename `l` to `line` in `test_wavesurfer_js.py` (2 occurrences)
- Fix F841 unused variables (10 occurrences across 4 files)
- Verify: `python -m pytest tests/ -v --tb=short && ruff check tests/`

**Commit 3: 8C Medium risk reviewed fixes**
- Add `# noqa: F401` to try/except availability imports (6 imports across 3 files)
- Move `asynccontextmanager` import to top in `voice_test_helpers.py`
- Convert lambda to def in `test_fastapi_endpoints.py`
- Verify: `python -m pytest tests/ -v --tb=short && ruff check tests/`

### Expected final state

After all three commits: `ruff check qwen3_tts/ tests/` should report **0 errors** (some with `# noqa` suppressions for intentional try/except imports).

### Notes

1. The `--unsafe-fixes` flag in Commit 1 is safe here because all 11 F811 cases are module-scope imports shadowed by local re-imports. Removing the module-scope import leaves the local import as the sole definition.
2. For `test_protocols.py`, the 4 F401 errors at lines 27/55/79/106 are false positives -- the imports ARE used within the same method. Adding `# noqa: F401` is the correct fix.
3. The `test_speaker_similarity.py` imports (torch, torchaudio, transformers) could be replaced with `importlib.util.find_spec()` for a cleaner pattern, but `# noqa: F401` is acceptable and lower risk.
4. Source code (`qwen3_tts/`) is already clean -- no lint errors at all.
