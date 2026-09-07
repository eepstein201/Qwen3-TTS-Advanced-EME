# TDD Evidence: Step 0A — max_chunk_chars bounds

**Source plan:** docs/plans/2026-09-06-step-0a-max-chunk-chars-bounds.plan.md
(elaborates Wave 0 Step 0A of docs/plans/2026-09-06-consolidated-backlog-priority.plan.md)

## Task report

| Task | Command | RED | GREEN |
|---|---|---|---|
| 1 schema bounds | `pytest tests/test_validation.py::TestMaxChunkCharsBounds -v` | 2 failed, 3 passed — both rejection tests fail with `AssertionError: ValidationError not raised` (field unconstrained today); the 0/None/boundary acceptance tests pass as written | 53 passed (`tests/test_validation.py` full module) + 8 passed (`tests/security/test_seed_bounds.py`) |
| 2 join clamp | `pytest tests/test_stream_error_frame.py::TestStreamThreadJoinTimeoutScales::test_negative_max_chunk_chars_cannot_collapse_join_to_floor -v` | 1 failed — `AssertionError: 90.0 != 12500.0` (the live-verified defect: `min(-1, 50000)` → 90 s floor) | 11 passed (`tests/test_stream_error_frame.py` full module, incl. the 5 pre-existing `TestStreamThreadJoinTimeoutScales` cases) |

## What is guaranteed

| # | Guarantee | Test | Type | Result |
|---|---|---|---|---|
| 1 | max_chunk_chars=-1 rejected at request validation | test_validation.py::TestMaxChunkCharsBounds::test_negative_max_chunk_chars_rejected | unit | PASS |
| 2 | max_chunk_chars=10001 rejected | test_validation.py::…::test_max_chunk_chars_above_limit_rejected | unit | PASS |
| 3 | 0 (chunking disabled) and None (read config) stay legal | test_max_chunk_chars_zero_still_accepted / test_max_chunk_chars_none_still_accepted | unit | PASS |
| 4 | Boundaries 1 and 10000 accepted | test_max_chunk_chars_boundaries_accepted | unit | PASS |
| 5 | Join timeout never collapses to the 90 s floor on a negative | test_stream_error_frame.py::…::test_negative_max_chunk_chars_cannot_collapse_join_to_floor | unit | PASS |

## Local gates (observed)

| Gate | Command | Result |
|---|---|---|
| ruff | `ruff check qwen3_tts tests` | All checks passed! |
| mypy | `mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface` | Success: no issues found in 57 source files |
| keyword selection | `pytest tests/ -m "not e2e" --ignore=tests/evaluations -k "max_chunk_chars"` | 16 passed (3226 deselected); collect-only confirms all 6 new tests selected by name (5 `TestMaxChunkCharsBounds` methods + `test_negative_max_chunk_chars_cannot_collapse_join_to_floor`) plus 10 pre-existing matches (`test_raising_max_chunk_chars_raises_the_timeout`, `TestGenCacheKey::test_max_chunk_chars_distinguishes_hash`, 3 × `test_voice_generation`, `test_with_max_chunk_chars`, …) |
| targeted sweep | `pytest tests/test_validation.py tests/test_validation_ext.py tests/test_stream_error_frame.py tests/test_streaming_thread_lifecycle.py tests/test_engine_streaming.py tests/test_streaming_mlx_chunking.py tests/test_seed_lock_chunks.py tests/security/ -q` | 162 passed |
| batch 3 (owns `test_stream_error_frame.py`) | `python tests/run_batches.py --batch 3` | PASS (exit 0, "1/1 batches passed") |
| batch 5 (owns `test_validation.py`) | `python tests/run_batches.py --batch 5` | FAILED — pre-existing environment breakage, not this change: `tests/evaluations/test_speaker_similarity.py:98` imports torchcodec at module scope and the conda env's torchcodec cannot dlopen its native lib (`libavutil.60.dylib`), so the batch's single `python -m unittest` invocation aborts at collection with 0 tests run. The plan pre-authorized the fallback for exactly this; executed: `python -m unittest tests.test_validation -v` (the runner's own mechanism, minus the poisoned module) → **Ran 5 tests, OK** — `TestMaxChunkCharsBounds` genuinely collects and passes under unittest, closing the hollow-batch-5-gate concern. Residual gate value for the file's 48 pytest-style tests is carried by the direct pytest run (53 passed) and the targeted sweep (162 passed). NOTE: this torchcodec breakage is the live defect the unmerged branch `fix/torchcodec-collection-guard` (2026-08-29) targets — a standing candidate follow-up, out of Step 0A scope. |
| baseline suite | `pytest tests/ -m "not e2e" --ignore=tests/evaluations --tb=short -q` | 3146 passed, 4 skipped, 92 deselected in 76.38s — zero failures vs the known-green baseline |

## Coverage and known gaps

- Surfacing: HTTP callers see FastAPI's default 422 (no custom
  RequestValidationError handler in app.py — verified by code inspection,
  not by an HTTP-level test); /ws surfaces the ValidationError as an
  in-band error frame (websocket.py:345).
- Known gap (out of scope): config.json hand-edits are not clamped —
  validate_config() (core/config/io.py) does not range-check
  generation.max_chunk_chars; the documented 0-10000 range there stays
  advisory. Negative config values behave as "chunking disabled" at the
  engine gate (inference.py:1243), which remains untouched in Step 0A.
- Known gap (accepted): the CLI flag --max-chunk-chars is unbounded
  client-side; a user value above 10000 now returns the server's 422 —
  consistent with the documented 0-10000 range.
- Docs: CLAUDE.md and docs/CONFIG.md need no change — the documented
  0-10000 range already matches the now-enforced request bound, so no
  documentation drift is created by this fix.
- WS-path rejection is inherited via the shared model (constructed at
  websocket.py:327, ValidationError caught at :345); no dedicated
  transport-level test added — the model-level test is the contract, and
  the 422/error-frame surfacing is verified by inspection. A transport-level
  422/error-frame test is a possible follow-up (Wave 4B territory), not
  Step 0A scope.

## Merge evidence

Branch history: test RED commit → fix GREEN commit, per task
(`e73e509` → `d418f00`, `287d1f1` → `b871b14`). If squash-merged,
this section plus the PR body carries the RED/GREEN summary.

Commit list (in order): `e73e509` test RED (Task 1) · `d418f00` fix GREEN (Task 1) ·
`287d1f1` test RED (Task 2) · `b871b14` fix GREEN (Task 2) · docs (this commit).
