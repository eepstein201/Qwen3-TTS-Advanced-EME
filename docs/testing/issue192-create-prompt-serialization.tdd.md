# TDD Evidence — /create-voice-prompt inference serialized under `inference_lock` (#192)

Source plan: `/Users/ericepstein/.claude/plans/functional-sniffing-flamingo.md`
Branch: `feature/issue-192-serialize-create-voice-prompt` (base `632553d`)
Task evidence: `.superpowers/sdd/functional-sniffing-flamingo/task-1-report.md`
Test module: `tests/test_issue192_create_prompt_serialization.py`

## User journey

As a server operator, I want /create-voice-prompt's MLX inference serialized
under `inference_lock`, so a concurrent generation never corrupts MLX state
(#192).

Background: unsynchronized concurrent MLX inference is upstream-unsafe
(ml-explore/mlx#3078, Blaizzy/mlx-audio#638, #733); corruption manifests as
EOS-never-emitted runaway generations served behind HTTP 200. `/generate`,
`/generate-stream`, `/ws`, the design warm-up (PR #211) and `/transcribe`
(PR #212) already serialize on `app.state.inference_lock` —
`/create-voice-prompt` (`create_voice_prompt` → `model.create_voice_clone_prompt`)
was the last reachable unsynchronized pair. With this change, all MLX
inference reachable through the API serializes on `inference_lock`.

## Task report (TDD cycle)

**Task 1 — serialize the handler (RED → GREEN).** Wrote
`tests/test_issue192_create_prompt_serialization.py` first, confirmed it
failed 10/10 against the then-sync handler, then rewrote
`handle_create_voice_prompt` async with the leaf-acquisition pattern:
decode/staging/`load_audio_for_cloning` and the `.pt` save run via
`asyncio.to_thread` UNLOCKED; `create_voice_prompt` runs via `to_thread`
inside `async with state.inference_lock`; the route awaits the handler
directly; the UI client moved to `CREATE_PROMPT_TIMEOUT_SEC` (900,
`qwen3_tts/core/http_client.py`).
Commits: `1ccd8d4` (RED reproducer), `314c068` (GREEN fix),
`5ce1875` (BATCHES registration, gate-forced).

**Task 2 — registration/doc sweep + this evidence report.** Verified the
BATCHES registration (already done by `5ce1875`), updated the eight
stale-claim doc sites to the new concurrency map, and wrote this file.

### RED evidence (copied from task-1-report.md)

Command:

```
conda run --no-capture-output -n qwen3-tts-mlx python -m pytest tests/test_issue192_create_prompt_serialization.py -v --tb=short
```

Result: **10 failed, 0 passed** — every failure an intended cause, none a
test-setup defect (the module imported cleanly; every patch target that was
supposed to exist did).

| Test | Intended failure |
|---|---|
| `test_create_runs_with_inference_lock_held` | `AttributeError: module 'qwen3_tts.server.app_prompts' does not have the attribute '_save_pt'` — patch target is the not-yet-implemented helper |
| `test_audio_load_runs_outside_lock_and_before_create` | same `_save_pt` AttributeError |
| `test_save_runs_after_lock_released` | same `_save_pt` AttributeError |
| `test_create_runs_off_event_loop_thread` | same `_save_pt` AttributeError |
| `test_create_uses_captured_clone_model_reference` | same `_save_pt` AttributeError |
| `test_decode_and_staging_run_off_event_loop_thread` | same `_save_pt` AttributeError |
| `test_create_deferred_while_generation_holds_lock` | same `_save_pt` AttributeError (patch entry precedes `ensure_future`, so the still-sync handler never runs) |
| `TestSavePtHelper::test_save_pt_calls_torch_save` | `ImportError: cannot import name '_save_pt'` (torch IS importable in the mlx env, so the guard did not skip) |
| `TestCreatePromptRouteShape::test_endpoint_awaits_handler_directly` | `AssertionError: 'await handle_create_voice_prompt(' not found` — route still reads `await asyncio.to_thread(handle_create_voice_prompt, state, req)` |
| `TestCreatePromptTimeoutDrift::test_ui_client_uses_shared_constant` | `ImportError: cannot import name 'CREATE_PROMPT_TIMEOUT_SEC'` |

Note from task-1-report.md on the dominant `_save_pt` AttributeError: entering
the `patch(...)` context fails before the handler call, so the sync-handler
TypeError never surfaces in the run. This is deliberate — without the patch,
the RED sync handler would run to its real inline `torch.save` and write a
real `.pt` into `VOICE_PROMPTS_DIR`.

### GREEN evidence (copied from task-1-report.md)

1. `conda run --no-capture-output -n qwen3-tts-mlx python -m pytest tests/test_issue192_create_prompt_serialization.py -v --tb=short`
   → **10 passed** (7 serialization + `_save_pt` helper + route shape + timeout drift)
2. `conda run --no-capture-output -n qwen3-tts-mlx python -m pytest tests/test_create_voice_endpoint.py -v --tb=short`
   → **8 passed** (module untouched — no reflexive edits were needed)
3. `conda run --no-capture-output -n qwen3-tts-mlx python -m pytest tests/test_python_review_fixes.py -v --tb=short`
   → **14 passed** (incl. the converted `test_handle_create_voice_prompt_no_success_on_import_error`)

## Per-guarantee table

| # | guaranteed | test id | type | result | evidence |
|---|---|---|---|---|---|
| 1 | `create_voice_prompt` runs with `inference_lock` HELD (leaf acquisition) | `TestCreatePromptSerialization::test_create_runs_with_inference_lock_held` | unit (async, real `asyncio.Lock`, engine seams patched) | PASS | GREEN run 1: 10 passed; RED: `_save_pt` AttributeError |
| 2 | audio staging runs OUTSIDE the lock AND before the create | `TestCreatePromptSerialization::test_audio_load_runs_outside_lock_and_before_create` | unit (event order + `lock.locked()` observed inside fakes) | PASS | GREEN run 1; RED: `_save_pt` AttributeError |
| 3 | the `.pt` save runs AFTER the lock is released | `TestCreatePromptSerialization::test_save_runs_after_lock_released` | unit | PASS | GREEN run 1; RED: `_save_pt` AttributeError |
| 4 | the create runs in a worker thread, never on the event loop | `TestCreatePromptSerialization::test_create_runs_off_event_loop_thread` | unit | PASS | GREEN run 1; RED: `_save_pt` AttributeError |
| 5 | the clone-model reference is captured ONCE before the lock (concurrent `/unload-model` safety) | `TestCreatePromptSerialization::test_create_uses_captured_clone_model_reference` | unit | PASS | GREEN run 1; RED: `_save_pt` AttributeError |
| 6 | b64 decode + tempfile write run off the loop; `finally` still removes the tempfile | `TestCreatePromptSerialization::test_decode_and_staging_run_off_event_loop_thread` | unit | PASS | GREEN run 1; RED: `_save_pt` AttributeError |
| 7 | a held `inference_lock` DEFERS the create; staging proceeds while the lock is taken | `TestCreatePromptSerialization::test_create_deferred_while_generation_holds_lock` | unit (concurrent scenario) | PASS | GREEN run 1; RED: `_save_pt` AttributeError |
| 8 | `_save_pt` saves via lazy-imported `torch.save` | `TestSavePtHelper::test_save_pt_calls_torch_save` | unit (skips without torch) | PASS | GREEN run 1; RED: `ImportError: cannot import name '_save_pt'` |
| 9 | the `/create-voice-prompt` route awaits the async handler directly (not re-wrapped in `to_thread`) | `TestCreatePromptRouteShape::test_endpoint_awaits_handler_directly` | source-shape guard | PASS | GREEN run 1; RED: `AssertionError: 'await handle_create_voice_prompt(' not found` |
| 10 | the UI client uses `CREATE_PROMPT_TIMEOUT_SEC` (>660 s), not a hardcoded 60 s | `TestCreatePromptTimeoutDrift::test_ui_client_uses_shared_constant` | source-shape guard | PASS | GREEN run 1; RED: `ImportError: cannot import name 'CREATE_PROMPT_TIMEOUT_SEC'` |
| 11 | existing `/create-voice-prompt` endpoint behavior unchanged by the async rewrite | `tests/test_create_voice_endpoint.py` (suite, 8 tests) | unit/integration (TestClient) | PASS (8/8) | GREEN run 2; module untouched by the fix |
| 12 | the converted import-error regression test still guards no-success-on-`ImportError` | `tests/test_python_review_fixes.py` (suite, 14 tests) | unit | PASS (14/14) | GREEN run 3; RED for the module conversion documented in task-1-report.md |

## Coverage

Command (exact, run 2026-08-19 on branch
`feature/issue-192-serialize-create-voice-prompt`):

```
conda run --no-capture-output -n qwen3-tts-mlx python -m pytest tests/test_issue192_create_prompt_serialization.py tests/test_create_voice_endpoint.py tests/test_python_review_fixes.py --cov=qwen3_tts.server.app_prompts --cov-report=term -q
```

Result: **32 passed**; `qwen3_tts/server/app_prompts.py` — 211 stmts,
162 missed, **23%** (missing: 47-81, 94-142, 155-239, 258-279, 295-346,
403-404, 447, 452-459).

This is a scoped three-module run, not the module's full-suite number.
Intentional gaps:

- Lines 47-81 / 94-142 / 155-239 / 258-279 / 295-346 are the four sibling
  prompt handlers (`handle_list_prompts`, `handle_delete_prompt`,
  `handle_rename_prompt`, `handle_preview_prompt`, `handle_prompt_details`)
  plus their helpers — out of scope for this fix and covered by other suite
  modules (e.g. `tests/test_voice_prompts.py`, `tests/test_fastapi_app_ext*.py`,
  `tests/test_voice_server.py`, `tests/test_response_contracts.py`).
- Lines 403-404 (invalid-base64 400), 447 (`except HTTPException` re-raise)
  and 452-459 (`creation_failed`/`unknown_error` branches) of the create
  handler are error paths not driven by these three modules; the
  `import_error` branch (448-451) IS hit here by
  `tests/test_python_review_fixes.py::test_handle_create_voice_prompt_no_success_on_import_error`.
