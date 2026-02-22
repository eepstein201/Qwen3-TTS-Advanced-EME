# Flash-Attn Wheel Selection Tests — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add TDD test coverage for the GitHub API-based flash-attn wheel selection logic introduced in `colab_notebook.ipynb` Cell 1.

**Architecture:** The existing pattern in `tests/test_flash_attn_install.py` extracts notebook logic into standalone helper functions and tests them in isolation — no GPU, no Colab, no network required. We add two new helpers (`_parse_wheel_candidates` and `_select_best_wheel`) plus edge-case tests for the existing `_normalize_torch_version` and a new `_normalize_cuda_to_numeric`.

**Tech Stack:** Python 3, unittest, re (standard library only)

---

## Context

The notebook's flash-attn block was rewritten to query the GitHub releases API and select the best compatible pre-built wheel. The selection logic has three testable layers:

| Layer | What it does | Existing test coverage? |
|-------|-------------|----------------------|
| Torch version normalization | `"2.10.0+cu124"` → `"2100"` | Partial — no 2-digit minor test |
| CUDA numeric conversion | `"12.8"` → `128` (for `≤` comparison) | None |
| Candidate parsing | Regex-match wheel filenames, extract `(cu_num, th_num)` | None |
| Best wheel selection | Filter to `≤ installed`, sort descending, return ordered | None |

---

### Task 1: Add torch normalization test for 2-digit minor version

**Files:**
- Modify: `tests/test_flash_attn_install.py`

**Step 1: Write the failing test**

Add to `TestFlashAttnWheelUrlConstruction`:

```python
def test_torch_normalization_two_digit_minor(self):
    """torch '2.10.0+cu128' normalizes to '2100' (not '210' or error)."""
    self.assertEqual(_normalize_torch_version("2.10.0+cu128"), "2100")
```

**Step 2: Run test to verify it fails or passes**

Run: `python -m pytest tests/test_flash_attn_install.py::TestFlashAttnWheelUrlConstruction::test_torch_normalization_two_digit_minor -v`

Expected: This SHOULD pass because `_normalize_torch_version` already handles this via `''.join(parts)`. If it passes, that confirms existing code is correct and this test documents the behavior. Move on.

**Step 3: Commit**

```bash
git add tests/test_flash_attn_install.py
git commit -m "test: add torch normalization test for 2-digit minor version (2.10.0)"
```

---

### Task 2: Add `_normalize_cuda_to_numeric` helper + tests

**Files:**
- Modify: `tests/test_flash_attn_install.py`

**Step 1: Write the failing tests**

Add these tests to `TestFlashAttnWheelUrlConstruction`:

```python
def test_cuda_numeric_128(self):
    """CUDA '12.8' → 128 (numeric for ≤ comparison with wheel tags)."""
    self.assertEqual(_normalize_cuda_to_numeric("12.8"), 128)

def test_cuda_numeric_124(self):
    """CUDA '12.4.0' → 124."""
    self.assertEqual(_normalize_cuda_to_numeric("12.4.0"), 124)

def test_cuda_numeric_121(self):
    """CUDA '12.1' → 121."""
    self.assertEqual(_normalize_cuda_to_numeric("12.1"), 121)

def test_cuda_numeric_118(self):
    """CUDA '11.8' → 118."""
    self.assertEqual(_normalize_cuda_to_numeric("11.8"), 118)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_flash_attn_install.py -k "cuda_numeric" -v`

Expected: FAIL with `NameError: name '_normalize_cuda_to_numeric' is not defined`

**Step 3: Write minimal implementation**

Add above the test class:

```python
def _normalize_cuda_to_numeric(cuda_string):
    """'12.8' or '12.4.0' -> 128 or 124 (major*10 + minor, as int)."""
    parts = cuda_string.split(".")
    return int(parts[0]) * 10 + int(parts[1])
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_flash_attn_install.py -k "cuda_numeric" -v`

Expected: All 4 PASS

**Step 5: Commit**

```bash
git add tests/test_flash_attn_install.py
git commit -m "feat: add _normalize_cuda_to_numeric helper with tests"
```

---

### Task 3: Add `_parse_wheel_candidates` helper + tests

**Files:**
- Modify: `tests/test_flash_attn_install.py`

**Step 1: Write the failing tests**

Add a new test class:

```python
# Realistic wheel filenames for test fixtures
_SAMPLE_ASSETS = [
    "flash_attn-2.7.4+cu124torch260cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    "flash_attn-2.7.4+cu124torch251cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    "flash_attn-2.7.4+cu121torch260cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    "flash_attn-2.7.4+cu124torch260cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    "flash_attn-2.7.4+cu121torch251cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
    "flash_attn-2.7.4.tar.gz",          # source tarball — should be ignored
    "checksums.txt",                      # non-wheel — should be ignored
]


class TestParseWheelCandidates(unittest.TestCase):

    def test_filters_to_matching_python_version(self):
        """Only wheels for cp312 are returned when py_str='cp312'."""
        result = _parse_wheel_candidates(_SAMPLE_ASSETS, "cp312")
        self.assertEqual(len(result), 3)

    def test_excludes_wrong_python_version(self):
        """cp311 wheels excluded when asking for cp312."""
        result = _parse_wheel_candidates(_SAMPLE_ASSETS, "cp312")
        names = [r[2] for r in result]
        self.assertTrue(all("cp312" in n for n in names))

    def test_extracts_cuda_and_torch_numbers(self):
        """Parsed tuples contain correct (cu_num, th_num, name)."""
        result = _parse_wheel_candidates(_SAMPLE_ASSETS, "cp312")
        cu_th_pairs = [(r[0], r[1]) for r in result]
        self.assertIn((124, 260), cu_th_pairs)
        self.assertIn((124, 251), cu_th_pairs)
        self.assertIn((121, 260), cu_th_pairs)

    def test_ignores_non_wheel_files(self):
        """Tarballs and text files are not matched."""
        result = _parse_wheel_candidates(_SAMPLE_ASSETS, "cp312")
        names = [r[2] for r in result]
        self.assertTrue(all(n.endswith(".whl") for n in names))

    def test_empty_assets_returns_empty(self):
        """Empty asset list returns empty candidate list."""
        result = _parse_wheel_candidates([], "cp312")
        self.assertEqual(result, [])

    def test_no_matching_python_returns_empty(self):
        """No cp313 wheels in sample → empty list."""
        result = _parse_wheel_candidates(_SAMPLE_ASSETS, "cp313")
        self.assertEqual(result, [])
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_flash_attn_install.py::TestParseWheelCandidates -v`

Expected: FAIL with `NameError: name '_parse_wheel_candidates' is not defined`

**Step 3: Write minimal implementation**

Add above the test classes:

```python
def _parse_wheel_candidates(asset_names, py_str):
    """
    Match flash-attn wheel filenames and extract (cu_num, th_num, name) tuples.

    Args:
        asset_names: list of filename strings from a GitHub release
        py_str:      e.g. "cp312"

    Returns:
        List of (cu_num: int, th_num: int, name: str) for matching wheels.
    """
    pattern = re.compile(
        r"flash_attn-[\d.]+\+cu(\d+)torch(\d+)cxx11abiFALSE"
        r"-(" + py_str + r")-\3-linux_x86_64\.whl"
    )
    candidates = []
    for name in asset_names:
        m = pattern.match(name)
        if m:
            candidates.append((int(m.group(1)), int(m.group(2)), name))
    return candidates
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_flash_attn_install.py::TestParseWheelCandidates -v`

Expected: All 6 PASS

**Step 5: Run full test file to check no regressions**

Run: `python -m pytest tests/test_flash_attn_install.py -v`

Expected: All tests PASS (10 existing + 1 from Task 1 + 4 from Task 2 + 6 here = 21)

**Step 6: Commit**

```bash
git add tests/test_flash_attn_install.py
git commit -m "feat: add _parse_wheel_candidates helper with 6 tests"
```

---

### Task 4: Add `_select_best_wheel` helper + tests

**Files:**
- Modify: `tests/test_flash_attn_install.py`

**Step 1: Write the failing tests**

Add a new test class:

```python
class TestSelectBestWheel(unittest.TestCase):

    def test_prefers_highest_compatible_cuda_and_torch(self):
        """With cu_installed=128, th_installed=2100, picks cu124/torch260 over cu121/torch251."""
        candidates = [
            (121, 251, "flash_attn-2.7.4+cu121torch251-whl"),
            (124, 260, "flash_attn-2.7.4+cu124torch260-whl"),
            (121, 260, "flash_attn-2.7.4+cu121torch260-whl"),
        ]
        result = _select_best_wheel(candidates, cu_installed=128, th_installed=2100)
        self.assertEqual(result[0], (124, 260, "flash_attn-2.7.4+cu124torch260-whl"))

    def test_filters_out_wheels_above_installed_cuda(self):
        """cu126 wheel excluded when installed CUDA is only 124."""
        candidates = [
            (126, 260, "flash_attn-cu126-whl"),
            (124, 260, "flash_attn-cu124-whl"),
        ]
        result = _select_best_wheel(candidates, cu_installed=124, th_installed=260)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 124)

    def test_filters_out_wheels_above_installed_torch(self):
        """torch260 wheel excluded when installed torch is only 251."""
        candidates = [
            (124, 260, "flash_attn-torch260-whl"),
            (124, 251, "flash_attn-torch251-whl"),
        ]
        result = _select_best_wheel(candidates, cu_installed=124, th_installed=251)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], 251)

    def test_falls_back_to_highest_when_none_compatible(self):
        """When ALL candidates are newer than installed, returns highest-first as fallback."""
        candidates = [
            (126, 260, "flash_attn-cu126torch260-whl"),
            (124, 251, "flash_attn-cu124torch251-whl"),
        ]
        result = _select_best_wheel(candidates, cu_installed=118, th_installed=240)
        # Nothing is ≤ installed, so fallback: highest first
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 126)  # highest cuda first

    def test_empty_candidates_returns_empty(self):
        """No candidates → empty result."""
        result = _select_best_wheel([], cu_installed=128, th_installed=2100)
        self.assertEqual(result, [])

    def test_sort_order_cuda_then_torch(self):
        """Among compatible wheels, sort by (cuda DESC, torch DESC)."""
        candidates = [
            (121, 260, "a"),
            (124, 251, "b"),
            (124, 260, "c"),
        ]
        result = _select_best_wheel(candidates, cu_installed=128, th_installed=2100)
        self.assertEqual(
            [(r[0], r[1]) for r in result],
            [(124, 260), (124, 251), (121, 260)]
        )
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_flash_attn_install.py::TestSelectBestWheel -v`

Expected: FAIL with `NameError: name '_select_best_wheel' is not defined`

**Step 3: Write minimal implementation**

Add above the test classes:

```python
def _select_best_wheel(candidates, cu_installed, th_installed):
    """
    Select compatible wheels sorted by preference (highest CUDA, then highest torch).

    Args:
        candidates:    list of (cu_num, th_num, name) tuples
        cu_installed:  numeric CUDA version of the runtime (e.g. 128 for CUDA 12.8)
        th_installed:  numeric torch version of the runtime (e.g. 2100 for torch 2.10.0)

    Returns:
        Ordered list of (cu_num, th_num, name) — best match first.
        If no candidates have versions ≤ installed, returns ALL candidates
        sorted highest-first as a last-resort fallback.
    """
    if not candidates:
        return []
    compat = [c for c in candidates if c[0] <= cu_installed and c[1] <= th_installed]
    if not compat:
        # Fallback: nothing compatible, try highest available
        return sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)
    return sorted(compat, key=lambda x: (x[0], x[1]), reverse=True)
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_flash_attn_install.py::TestSelectBestWheel -v`

Expected: All 6 PASS

**Step 5: Run full test file**

Run: `python -m pytest tests/test_flash_attn_install.py -v`

Expected: All 27 tests PASS

**Step 6: Commit**

```bash
git add tests/test_flash_attn_install.py
git commit -m "feat: add _select_best_wheel helper with 6 tests"
```

---

### Task 5: Run full test suite to verify no regressions

**Files:**
- None (verification only)

**Step 1: Run all project tests**

Run: `python -m unittest discover -v tests/`

Expected: All tests pass (334+ existing + 17 new = 351+). No errors, no warnings.

**Step 2: Commit (no-op if nothing changed)**

If everything is green, no commit needed. If any test needed a fix, commit the fix.

---

### Task 6: Update CLAUDE.md test count

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update the test count in the Testing section**

Change `334+ tests across 3 files` to `351+ tests across 3 files` (or whatever the actual count is after running the full suite).

Also update the flash-attn test file description:

> `tests/test_flash_attn_install.py` — 27 tests: URL construction, version normalization, candidate parsing, wheel selection

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update test counts after flash-attn wheel selection tests"
```
