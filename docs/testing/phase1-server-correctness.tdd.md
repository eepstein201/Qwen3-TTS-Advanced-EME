# TDD Evidence — Phase 1a/1b/1c: HIGH-severity server correctness

**Source plan:** `~/.claude/plans/review-entire-repo-for-ancient-possum.md`, Phase 1.
**Branch:** `fix/phase1-server-correctness` · **Base:** `main` @ `0ddaad8`

Three independent HIGH findings, each with a genuine RED before its fix. All
three were re-confirmed at their cited source lines before any test was
written (no drift from the plan's 2026-08-24 verification pass).

Code review of the branch then found three incomplete closures of those same
bug classes — recorded as **1R1/1R2/1R3** below, each likewise RED-first. Three
lower-severity review findings were deferred to the plan rather than fixed
here (see "Deferred to the plan" at the end).

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

---

## Review follow-ups (1R1/1R2/1R3)

Code review of this branch found three places where the bug classes above were
only partly closed. Each was reproduced on the branch head before its fix.

### 1R1 — the 1b `try/finally` opened too late

`_background_load`'s own comment claims "everything below runs under a finally
that signals readiness", but the function-local engine import and the config
parsing that builds `models_to_load` sat **above** the `try:`. Two reachable
escapes still wedged the server in permanent 503:

```
RESULT [engine-import-failure]:  readiness NEVER SET -> permanent 503 WEDGE
  (escaped: ImportError: libmlx.dylib: symbol not found)
RESULT [malformed-config]:       readiness NEVER SET -> permanent 503 WEDGE
  (escaped: AttributeError: 'bool' object has no attribute 'get')
```

The second comes from a hand-edited `config.json` with `"models": {"clone":
true}` — `settings.get()` on a bool. The first is the native-install breakage
class this repo has hit before (PR #100).

**RED** (`tests/test_fastapi_app_ext2.py::TestBackgroundLoad`):

```
FAILED test_engine_import_failure_still_signals_readiness
  - ModuleNotFoundError: import of qwen3_tts.core.engine halted; None in sys.modules
FAILED test_malformed_models_config_still_signals_readiness
  - AttributeError: 'bool' object has no attribute 'get'
```

**GREEN** — `try:` now opens on the first statement of the body, with a
top-level `except Exception` that logs the fatal setup error so `/health`
carries a reason instead of the server hanging at 503 with nothing recorded.

### 1R2 — `generate_dialogue` kept the unguarded `results[0]`

The 1c client fix landed in `_generate_via_server` but its sibling call site in
the **same module** (`generator.py`, `generate_dialogue`) still indexed
`resp.json()["results"][0]`. `cancel_generation()` is the next method in that
class, so cancelling mid-dialogue is the natural trigger.

**RED** (`tests/test_client_generator.py::TestGenerateDialogue`):

```
FAILED test_empty_results_raises_generation_error_not_indexerror
  - IndexError: list index out of range
FAILED test_cancelled_empty_results_says_cancelled
  - IndexError: list index out of range
```

**GREEN** — the guard is extracted to a module-level `_first_result(payload)`
helper (DRY) and both call sites route through it. Any future single-text
caller in this module must use it too.

### 1R3 — the CLI path under-delivered silently

`interface/generate_server.py::generate_via_server` guarded only a **missing**
`results` key, never an empty or short list, and ignored the new `cancelled`
flag. Its nine callers then either index `results[0]` (`cli/srt.py`,
`cli/dialogue.py`, `generate_interactive.py` — bare `IndexError`, or a
`FAILED, skipping: list index out of range` under their broad `except`) or
iterate it (`interface/generate.py`), **writing fewer .wav files than the user
asked for with exit code 0 and no warning**.

**RED** (`tests/test_generate_server.py::TestGenerateViaServerShortBatch`):

```
FAILED test_empty_results_raises_instead_of_returning_empty  - TTSGenericError not raised
FAILED test_cancelled_batch_names_cancellation               - TTSGenericError not raised
FAILED test_short_batch_raises_rather_than_dropping_texts    - TTSGenericError not raised
```

**GREEN** — `len(results) != expected` now raises `TTSGenericError` naming both
counts and, when the server says so, cancellation as the cause.
`test_complete_batch_is_returned_unchanged` passes at RED by design: it stops
the fix degrading to an always-raise that would satisfy the other three
vacuously (same guard pattern as 1c's `test_uncancelled_batch_...`).

---

## Gates

Re-run on the branch head after the follow-ups:

```
$ pytest tests/ -m "not e2e" -q --ignore=tests/evaluations/test_speaker_similarity.py
2993 passed, 5 skipped, 88 deselected

$ python tests/run_batches.py --batch 2   →  1/1 batches passed
$ python tests/run_batches.py --batch 3   →  1/1 batches passed
$ python tests/run_batches.py --batch 4   →  1/1 batches passed   (owns test_generate_server)
$ ruff check qwen3_tts tests              →  All checks passed
$ mypy qwen3_tts/{core,server,interface}  →  Success: no issues found in 53 source files
$ bandit -r qwen3_tts -c pyproject.toml   →  0 High / 0 Medium / 0 Low
$ python -m qwen3_tts.tools.check_config_docs → OK (66 keys)
$ wc -l CLAUDE.md                         →  298   (unchanged; all edits extend existing lines)
$ pytest tests/test_batches_coverage.py   →  3 passed (no unregistered modules)
```

`--ignore=tests/evaluations/test_speaker_similarity.py` is the plan's interim
workaround for pre-existing **P1** (libtorchcodec fails at collection, which
`Interrupts` the whole run). It is local-env-only — CI is unaffected — and is
still owned by Phase 0. It is not fixed here and must not become permanent.

## Not covered

No e2e test exercises these paths against a live server. `/ws` join scaling and
the vanished-cache-file fall-through are both unit-pinned only; the plan's
Phase 2d owns the e2e queuing tier. The one existing cancel e2e
(`test_e2e_playwright.py::test_06_cancel_generation`) drives Gradio and by its
own comments tolerates the cancel timing out, so it never asserts on the
`/generate` response shape and could not have caught 1c.

## Review follow-ups (1R4/1R5/1R6)

Three LOW findings deferred when this work was first written; folded in here.
None is a correctness defect, so all three land **GREEN on first run** — the
honest RED gate is to prove each new assertion is *live* by inverting the
**production** side and observing the failure (Phase 0a convention).

### 1R4 — an assertion that also passed when the key was gone

`test_uncancelled_batch_reports_not_cancelled` asserted
`assertFalse(result.get("cancelled"))`. `.get()` returns `None` for a missing
key and `assertFalse(None)` passes, so the test could not distinguish "flag is
False" from "flag was dropped" — the exact regression clients would break on.
Tightened to `assertIn("cancelled", result)` + `assertIs(result["cancelled"], False)`.

**Liveness proof.** Changed the uncancelled return in `app_generation.py` from
`{"results": results, "cancelled": cancelled}` to `{"results": results}`:

```
E   AssertionError: 'cancelled' not found in {'results': [...]}
FAILED tests/test_batch_generation_state_ownership.py::TestBatchResultCompleteness
       ::test_uncancelled_batch_reports_not_cancelled
```

The old `assertFalse` form passes under that same inversion. Reverted; green.

### 1R5 — the flag was never pinned on the wire

Every 1c server test calls `handle_generate` directly, which bypasses FastAPI's
`response_model` filtering entirely. `tests/test_response_contracts.py` exists
precisely to assert fields survive that filter, but had only been extended for
per-result fields. Added a top-level `assertIn("cancelled", payload)` +
`assertIs(payload["cancelled"], False)` to `test_generate_matches_contract`.

**Liveness proof — two inversions**, because the first attempt was a false
negative worth recording: deleting `cancelled: bool` at `validation.py:194` did
**not** fail the test. That line belongs to `GenerationStatusResponse`, not
`GenerateResponse` — the wrong class. The real field is `validation.py:143`
(`cancelled: bool = False`).

```
# A: delete cancelled from GenerateResponse (validation.py:143)
E   AssertionError: 'cancelled' not found in {'results': [...]}   → FAILED

# B: add response_model_exclude_defaults=True to the /generate route
E   AssertionError: 'cancelled' not found in {'results': [...]}   → FAILED
```

Inversion B is the specific regression this finding named, and the field's
`= False` default is exactly what makes `exclude_defaults` able to drop it.
Both reverted; `git diff` on both files empty.

### 1R6 — the docstring conflated `0` with `None`

`_stream_thread_join_timeout`'s docstring claimed "`0`/`None` disables
chunking". Only `0` does. A request-level `None` means "read
`generation.max_chunk_chars` from config" (`inference.py:1354` →
`_get_max_chunk_chars`, default 500), so the real chunk is normally 500 chars
while the bound computed is the whole text.

**Fix is the docstring, not the code.** Resolving `None` through config here
would tighten the bound toward one chunk — i.e. make this timeout *shorter*.
The only dangerous error for this join is one that is too short: it releases
`inference_lock` with the model still generating, the precise race the join
exists to prevent. Over-generous is the fail-safe direction, so the behavior
stays and the docstring now says why.

`tests/test_stream_error_frame.py` carried the same conflation — one subTest
loop over `(0, None)` under the comment "0 disables chunking". Split into two
named tests with their real, different reasons.

**Liveness proof.** Added the rejected `if max_chunk_chars is None: max_chunk_chars = 500`
resolution to the helper:

```
FAILED tests/test_stream_error_frame.py::TestStreamThreadJoinTimeoutScales
       ::test_unspecified_chunk_size_is_deliberately_over_generous
1 failed, 1 passed
```

The `None` case fails and the `0` case passes — the two are now genuinely
independent, which the single merged subTest loop could not show. Reverted; green.
