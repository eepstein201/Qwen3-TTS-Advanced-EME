# TDD Evidence — Phase 1a/1b/1c: HIGH-severity server correctness

**Source plan:** `~/.claude/plans/review-entire-repo-for-ancient-possum.md`, Phase 1.
**Branch:** `fix/phase1-server-correctness` · **Base:** `main` @ `0ddaad8`

Three independent HIGH findings, each with a genuine RED before its fix. All
three were re-confirmed at their cited source lines before any test was
written (no drift from the plan's 2026-08-24 verification pass).

---

## 1a (H1/T1) — `/ws` join used the flat 90 s floor

### The defect

`server/websocket.py:530` called `_await_inference_thread_done(done)` with no
`timeout`, so the wait fell back to `_STREAM_THREAD_JOIN_FLOOR_SEC` = 90.0.
The HTTP path (`app_generation.py:823`) has scaled the same join with
`_stream_thread_join_timeout(len(text), req.max_chunk_chars)` since the
constant was removed there. Both `text` and `req` were already in scope at the
WS call site — nothing but an oversight.

The join must cover ONE chunk's generation. Raise `generation.max_chunk_chars`
above the 500-char default and 90 s expires mid-generation, releasing
`inference_lock` while `model.generate()` is still on the GPU — the exact race
the join exists to prevent. Only the helper and the HTTP call site were
tested; nothing pinned `/ws`.

### RED

```
$ pytest tests/test_streaming_thread_lifecycle.py::TestWsStreamJoinTimeout -v
FAILED test_ws_join_timeout_scales_with_text_and_chunk_size
E   AssertionError: None != 375.0 : /ws join timeout must be derived from the
    text length and max_chunk_chars (expected 375.0s, got None)

FAILED test_ws_logs_when_inference_thread_outlives_the_join
E   AssertionError: no logs of level ERROR or higher triggered on
    tts.server.websocket

2 failed
```

`None` — not `90.0` — because pre-fix no `timeout` argument was passed at all.

### GREEN

Pass the derived timeout, and log when it expires (the HTTP path already does;
a lock released with the thread still running was previously silent on `/ws`).

```
$ pytest tests/test_streaming_thread_lifecycle.py -v
6 passed
```

The scaling test also asserts `expected > _STREAM_THREAD_JOIN_FLOOR_SEC`, so
the fixture can never shrink to a size where the derived value coincides with
the floor and the assertion passes vacuously.

---

## 1b (H2) — startup could wedge the server in 503 forever

### The defect

`_background_load` (`server/app_lifespan.py`) is the **only** producer of
`models_loaded`, and `set()` sat after the loop with no `finally`. The
per-model `except` caught only `(ImportError, RuntimeError, OSError,
ValueError, MemoryError)`. Any other exception escaped, killed the daemon
thread, skipped every remaining model, and left `models_loaded` clear — so
`/ready` answers 503 permanently, every waiter hangs, and `model_load_errors`
is empty, meaning `/health` reports no reason at all.

### Correction to the plan's stated trigger

The plan proposed `HfHubHTTPError` / `RevisionNotFoundError` (via a typo in the
`models.<type>.revision` knob) as the escaping exception. **That is wrong** —
verified locally:

```
$ python -c "import huggingface_hub.errors as e; print([c.__name__ for c in e.RevisionNotFoundError.__mro__])"
['RevisionNotFoundError', 'HfHubHTTPError', 'HTTPError', 'OSError', 'Exception', 'BaseException', 'object']
```

Both subclass `OSError`, so the existing tuple already caught them. The
reachable gap is library **API drift**, which surfaces as `AttributeError`,
`TypeError` or `KeyError` — a shape this repo has hit before (the mlx-audio
`max_new_tokens` → `max_tokens` kwarg rename). The tests use those types.

### RED

```
$ pytest tests/test_fastapi_app_ext2.py::TestBackgroundLoad -v
FAILED test_unexpected_load_error_is_recorded_not_fatal
E   AttributeError: 'NoneType' has no attribute 'generate'
FAILED test_unexpected_load_error_still_signals_readiness
E   KeyError: 'model_id'
FAILED test_unexpected_migration_error_still_signals_readiness
E   TypeError: unexpected keyword argument
FAILED test_load_error_is_sanitized_for_public_health
E   AttributeError: cannot read /Users/someone/models/design/weights.safetensors

4 failed, 5 passed
```

Every one propagates straight out of `_background_load` — exactly the escape
that kills the loader thread in production.

### GREEN

Per-model `except Exception` that logs and records a `_sanitize_error()`'d
entry (mirroring `handle_load_model`'s catch-all, which has had one since
PRF-5), plus `finally: models_loaded.set()` around the whole body — which also
covers the post-loop MLX prompt migration, whose own narrow tuple had the same
hole.

```
$ pytest tests/test_fastapi_app_ext2.py::TestBackgroundLoad tests/test_issue192_warmup_serialization.py tests/test_fastapi_app_ext.py -q
51 passed
```

The fourth test pins sanitization: `/health` is public, so a recorded startup
error must not leak absolute paths (CWE-209).

---

## 1c (H3 + L19) — a batch could come back silently short

### The defect

Three linked problems, all ending at the same client crash:

1. **`app_generation.py` post-lock cache branch** — the `continue` sat OUTSIDE
   the `os.path.exists` guard. Entry present + backing file gone meant the loop
   advanced without appending anything, dropping the item from the response.
   The file genuinely can vanish: `cleanup_resources()` unlinks cache files and
   eviction races a long batch. The pre-lock branch has always fallen through
   correctly — only the post-lock copy was wrong.
2. **Cancelled batch** — the loop just `break`s and returned
   `{"results": [...]}` with no indication of why it was short. A batch
   cancelled before its first item is a 200 with `results: []`, indistinguishable
   from success.
3. **`client/generator.py:265`** — indexed `resp.json()["results"][0]`
   unconditionally, so both cases surfaced as a bare `IndexError: list index out
   of range`.

### RED

Server:

```
$ pytest tests/test_batch_generation_state_ownership.py::TestBatchResultCompleteness -v
FAILED test_vanished_cache_file_regenerates_instead_of_dropping_result
E   AssertionError: 1 != 2 : batch returned fewer results than texts: a cache
    entry with a vanished file dropped its item instead of regenerating
    (client would raise IndexError on a 200)
FAILED test_cancelled_batch_marks_the_response_cancelled
E   AssertionError: None is not true : a truncated batch did not report
    cancelled=True

2 failed, 1 passed
```

Client — failing at precisely the line the review named:

```
$ pytest tests/test_client_generator.py::TestGenerateViaServer -v
FAILED test_empty_results_raises_generation_error_not_indexerror
E   IndexError: list index out of range
    qwen3_tts/server/client/generator.py:265
FAILED test_cancelled_empty_results_says_cancelled
E   IndexError: list index out of range
    qwen3_tts/server/client/generator.py:265

2 failed, 5 passed
```

The third server test (`test_uncancelled_batch_reports_not_cancelled`) passed
at RED and is deliberate: it stops the fix from being "always report
cancelled", which would satisfy the other two vacuously.

### GREEN

Move the `continue` inside the exists-guard and log a warning; add
`cancelled: bool = False` to the `GenerateResponse` contract and set it on the
truncating `break`; make the client raise `GenerationError` — naming
cancellation when the flag says so — instead of indexing blindly.

```
$ pytest tests/test_batch_generation_state_ownership.py tests/test_client_generator.py tests/test_response_contracts.py tests/test_generation_lock_scope.py tests/test_peaks_caching.py tests/test_validation.py -q
109 passed
```

**One test correction during GREEN:** the first assertions checked
`str(exception)`, but `GenerationError.__init__` routes its argument to
`technical_detail` and hardcodes the user-facing string to "Audio generation
failed." The assertions now target `.technical_detail`, which is what
`format_cli()` / `format_gradio()` actually surface. The implementation was
correct; the assertion was aimed at the wrong attribute.

---

## Gates

```
$ pytest tests/ -m "not e2e" -q --ignore=tests/evaluations/test_speaker_similarity.py
2979 passed, 11 skipped, 88 deselected

$ python tests/run_batches.py --batch 2   →  1/1 batches passed
$ python tests/run_batches.py --batch 3   →  1/1 batches passed
$ ruff check qwen3_tts tests              →  All checks passed
$ mypy qwen3_tts/{core,server,interface}  →  Success: no issues found in 53 source files
$ bandit -r qwen3_tts -c pyproject.toml   →  0 findings (7 pre-existing stale-nosec warnings only)
$ python -m qwen3_tts.tools.check_config_docs → OK (66 keys)
$ wc -l CLAUDE.md                         →  298   (unchanged; both edits extend existing lines)
```

`--ignore=tests/evaluations/test_speaker_similarity.py` is the plan's interim
workaround for pre-existing **P1** (libtorchcodec fails at collection, which
`Interrupts` the whole run). It is local-env-only — CI is unaffected — and is
still owned by Phase 0. It is not fixed here and must not become permanent.

## Not covered

No e2e test exercises these paths against a live server. `/ws` join scaling and
the vanished-cache-file fall-through are both unit-pinned only; the plan's
Phase 2d owns the e2e queuing tier.
