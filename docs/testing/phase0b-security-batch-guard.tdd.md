# TDD Evidence — Phase 0b: register `tests/security/` with the batch guard

**Source plan:** `~/.claude/plans/review-entire-repo-for-ancient-possum.md`, Phase 0b.
**Branch:** `fix/security-batch-guard` · **Base:** `main` @ `871fc43`

## What was missing

`tests/test_batches_coverage.py::_modules_on_disk()` globbed `tests/*.py` and
`tests/evaluations/*.py` only. `tests/security/` (6 modules) was invisible to
the guard, so none of them could ever be flagged as unregistered — and none
were registered in `tests/run_batches.py`'s `BATCHES`, meaning they never ran
in the local batch gate at all (only in CI's full-suite `coverage` job).

## RED gate (genuine, not inverted)

Unlike Phase 0a, this is a real RED: the code doesn't yet see the directory.
Widening the glob to include `tests/security/` immediately fails the guard,
listing exactly the 6 unregistered modules — no inversion needed.

```
$ pytest tests/test_batches_coverage.py -v
FAILED test_every_test_module_is_batched_or_declared
Test modules missing from BATCHES in tests/run_batches.py — they never run
in the batch gates:
  tests.security.test_path_injection
  tests.security.test_play_audio
  tests.security.test_seed_bounds
  tests.security.test_server_request
  tests.security.test_ssrf_callsites
  tests.security.test_voice_name_validation

1 failed, 2 passed
```

## A second defect found while fixing the first

Registering all 6 into `BATCHES` would have been the naive fix. Before doing
that, each module was run individually through the batch runner's actual
invocation (`python -m unittest tests.security.<module> -v`) to confirm it
collects real tests — the same "prove the branch is reachable" discipline
from Phase 0a, applied here to a different unreachability shape.

Result: **3 of the 6 modules are plain pytest-style classes, not
`unittest.TestCase` subclasses.** `unittest.TestLoader.loadTestsFromModule`
only collects `TestCase` subclasses, so under the batch runner these three
silently report `Ran 0 tests ... OK` — a hollow pass indistinguishable from
success in the batch runner's output.

```
$ python -m unittest tests.security.test_seed_bounds -v
Ran 0 tests in 0.000s
OK

$ python -m unittest tests.security.test_voice_name_validation -v
Ran 0 tests in 0.000s
OK

$ python -m unittest tests.security.test_play_audio -v
Ran 0 tests in 0.000s
OK
```

The other 3 use `unittest.TestCase` and collect real tests:

```
$ python -m unittest tests.security.test_path_injection -v   # Ran 21 tests ... OK
$ python -m unittest tests.security.test_server_request -v   # Ran 11 tests ... OK
$ python -m unittest tests.security.test_ssrf_callsites -v   #  Ran 6 tests ... OK
```

Blindly registering all 6 in `BATCHES` would have made
`test_every_test_module_is_batched_or_declared` pass while the batch gate
still silently skipped 3 modules' worth of assertions — the exact failure
mode this guard exists to catch, just one level deeper. Instead, the fix
mirrors the existing e2e pattern already in `INTENTIONALLY_UNBATCHED`:

| Module | Disposition | Reason |
|---|---|---|
| `tests.security.test_path_injection` | `BATCHES` (batch 1) | `unittest.TestCase`, 21 tests |
| `tests.security.test_server_request` | `BATCHES` (batch 1) | `unittest.TestCase`, 11 tests |
| `tests.security.test_ssrf_callsites` | `BATCHES` (batch 1) | `unittest.TestCase`, 6 tests |
| `tests.security.test_play_audio` | `INTENTIONALLY_UNBATCHED` | pytest-style class, collects 0 under `unittest` |
| `tests.security.test_seed_bounds` | `INTENTIONALLY_UNBATCHED` | pytest-style class, collects 0 under `unittest` |
| `tests.security.test_voice_name_validation` | `INTENTIONALLY_UNBATCHED` | pytest-style class, collects 0 under `unittest` |

Batch 1 ("Core Utilities — low risk, pure unit tests") was chosen for the 3
registered modules: none use `TestClient`/`app.state`/FastAPI lifespan
(confirmed via grep), matching the rest of batch 1's profile.

## GREEN evidence

```
$ pytest tests/test_batches_coverage.py -v
3 passed

$ python tests/run_batches.py --batch 1
Ran 498 tests in 5.242s
OK (skipped=1)
✓ Batch 1 passed

$ ruff check qwen3_tts tests
All checks passed!

$ mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface
Success: no issues found in 53 source files

$ bandit -r qwen3_tts -c pyproject.toml
No issues identified. (0 High, 0 Medium, 0 Low)

$ pytest tests/ -m "not e2e" --ignore=tests/evaluations/test_speaker_similarity.py
2969 passed, 10 skipped, 88 deselected, 72 subtests passed

$ python -m qwen3_tts.tools.check_config_docs
OK: CONFIG.md defaults match get_default_config() (66 keys).

$ wc -l CLAUDE.md
298 CLAUDE.md   # unchanged by this PR; within the 300-line guard
```

`--ignore=tests/evaluations/test_speaker_similarity.py` is the pre-existing
P1 workaround (`libtorchcodec` collection failure), tracked in the plan's
Phase 0 "Found during implementation" section — not introduced or affected
by this change. Skip count (4 → 10) versus the Phase 0a evidence doc is
environment-dependent and unrelated to this diff; not investigated further
here since none of the newly-skipped tests are in the modules this PR
touches.

## Coverage and known gaps

- The 3 `INTENTIONALLY_UNBATCHED` modules already run under plain
  `pytest`/CI `coverage` (proven above: they're part of the 2969-passed
  count when the whole suite runs, since pytest's collector — unlike
  `unittest`'s — picks up plain classes with `test_*` methods). Nothing here
  reduces their real coverage; it only stops the batch guard from claiming
  a false victory over them.
- No production code changed. This PR touches only
  `tests/test_batches_coverage.py` and `tests/run_batches.py`.
