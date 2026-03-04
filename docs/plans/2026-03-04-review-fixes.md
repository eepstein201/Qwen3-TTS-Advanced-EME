# Code Review Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 2 critical and 5 important issues found in post-merge code review of P1+P2 recommendations.

**Architecture:** Targeted fixes to `app.py` and `engine.py`. Add dedicated test file for the new R-1 through R-12 features.

**Tech Stack:** Python, FastAPI, threading, unittest

---

### Task 1: Fix SIGTERM infinite recursion (C-1)

**Files:**
- Modify: `qwen3_tts/server/app.py` — lines 336, 518, 1619, 1654-1661

**Step 1: Fix `_signal_handler` — reset handler before sending SIGTERM**

In `run_server()` around line 1654, change:

```python
def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    # Reset handler to default to prevent re-entry
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    # Set shutdown event
    shutdown_event = getattr(app.state, "shutdown_event", None)
    if shutdown_event is not None:
        shutdown_event.set()
    cleanup_resources(app.state)
    os.kill(os.getpid(), signal.SIGTERM)
```

**Step 2: Fix `auto_shutdown` — use `sys.exit()` instead of SIGTERM**

At line 336, change `os.kill(os.getpid(), signal.SIGTERM)` to `sys.exit(0)`.

Import `sys` is already at top of file.

**Step 3: Fix `cleanup_pid` — use `sys.exit()` instead of SIGTERM**

At line 518, change `os.kill(os.getpid(), signal.SIGTERM)` to `sys.exit(0)`.

**Step 4: Fix `/shutdown` endpoint — use `sys.exit()` instead of SIGTERM**

At line 1619, change `os.kill(os.getpid(), signal.SIGTERM)` to `sys.exit(0)`.

**Step 5: Run tests**

Run: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts && python tests/run_batches.py --batch 3`
Expected: All server tests pass.

**Step 6: Commit**

```bash
git add qwen3_tts/server/app.py
git commit -m "fix: prevent SIGTERM infinite recursion in shutdown paths"
```

---

### Task 2: Replace inline validation in /generate with shared helper (C-2)

**Files:**
- Modify: `qwen3_tts/server/app.py` — lines 1212-1231

**Step 1: Replace inline validation with `_validate_generation_request()` call**

Replace the block at lines 1212-1231 (mode check, path traversal, speaker check) with:

```python
    _validate_generation_request(req, security)
```

Where `security` is `state.server_config.get("security", {})`. This variable already exists earlier in the function as the source of `max_text_length` and `max_batch_size`.

Note: We need to check that the security dict is available. Look at how `/generate` currently gets its security config — it uses `state.server_config.get("security", {})` at the top of the function. Pass that to the helper.

**Step 2: Run tests**

Run: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts && python tests/run_batches.py --batch 3`
Expected: All server tests pass.

**Step 3: Commit**

```bash
git add qwen3_tts/server/app.py
git commit -m "fix: use shared _validate_generation_request() in /generate endpoint"
```

---

### Task 3: Adopt `_error_response()` helper in key endpoints (I-1)

**Files:**
- Modify: `qwen3_tts/server/app.py`

**Step 1: Replace ad-hoc HTTPException raises with `_error_response()` calls**

Target the main endpoints where errors use the `{"error": ..., "detail": ..., "recovery": ...}` format:
- `/generate` model-not-loaded error (~line 1241)
- `/generate` generation failed error (~line 1419)
- `/load-model` errors
- `/shutdown` errors

Only convert errors that ALREADY return structured dicts. Don't convert simple string-detail errors (like validation errors) — those are fine as-is.

**Step 2: Run tests**

Run: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts && python tests/run_batches.py --batch 3`

**Step 3: Commit**

```bash
git add qwen3_tts/server/app.py
git commit -m "refactor: adopt _error_response() helper for structured error responses"
```

---

### Task 4: Add thread-safe lock to `_torch_prompt_cache` (I-5)

**Files:**
- Modify: `qwen3_tts/core/engine.py` — around line 437

**Step 1: Add lock and wrap cache operations**

After `_torch_prompt_cache = OrderedDict()` (line 437), add:

```python
_torch_prompt_cache_lock = threading.Lock()
```

Then wrap all cache accesses in `_load_voice_prompt_torch()` with `with _torch_prompt_cache_lock:`. There are 4 locations:
1. Cache hit check + `move_to_end` (lines 452-454)
2. Auto-create cache store (lines 476-479)
3. Normal load cache store (lines 494-497)
4. Unsafe load cache store (lines 518-521)

Also wrap `clear_voice_prompt_cache()` and `voice_prompt_cache_info()`.

Ensure `import threading` is present at top of engine.py.

**Step 2: Run tests**

Run: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts && python tests/run_batches.py --batch 4`

**Step 3: Commit**

```bash
git add qwen3_tts/core/engine.py
git commit -m "fix: add threading lock to _torch_prompt_cache for thread safety"
```

---

### Task 5: Add cache hit/miss counters (I-2)

**Files:**
- Modify: `qwen3_tts/core/engine.py`

**Step 1: Add counters**

After the lock declaration, add:

```python
_torch_prompt_cache_hits = 0
_torch_prompt_cache_misses = 0
```

Increment `_torch_prompt_cache_hits` on cache hit (inside the lock, line ~452). Increment `_torch_prompt_cache_misses` after loading (inside the lock, at each cache store location).

Use `global` keyword in the function to modify these.

**Step 2: Update `voice_prompt_cache_info()` to report real values**

Replace the hardcoded `hits=0, misses=0` with the actual counter values.

**Step 3: Run tests**

Run: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts && python tests/run_batches.py --batch 4`

**Step 4: Commit**

```bash
git add qwen3_tts/core/engine.py
git commit -m "fix: restore cache hit/miss tracking for voice prompt cache"
```

---

### Task 6: Fix HealthResponse Pydantic model for 503 case (I-3)

**Files:**
- Modify: `qwen3_tts/server/app.py` — lines 149-161, 543-556

**Step 1: Add 503 response model to /health decorator**

Change the `/health` decorator to include a `responses` parameter:

```python
@app.get("/health", response_model=HealthResponse, responses={503: {"description": "Models loading"}})
```

And change the loading response to return a proper `HealthResponse`-shaped dict or use `response_model_exclude_unset=True` on the decorator.

Simplest fix: just add `model_load_errors` as an Optional field to `HealthResponse`.

**Step 2: Run tests**

Run: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts && python tests/run_batches.py --batch 3`

**Step 3: Commit**

```bash
git add qwen3_tts/server/app.py
git commit -m "fix: HealthResponse Pydantic model covers 503 loading case"
```

---

### Task 7: Fix content negotiation response_model conflict (I-4)

**Files:**
- Modify: `qwen3_tts/server/app.py` — /generate endpoint decorator

**Step 1: Update /generate decorator**

The simplest fix: when returning a raw `Response` object, FastAPI already skips Pydantic serialization. But to make the OpenAPI schema accurate, add `responses` to document the alternative:

```python
@app.post("/generate", response_model=GenerateResponse, responses={200: {"content": {"audio/wav": {}}}})
```

This documents that the endpoint can return either JSON or WAV depending on Accept header.

**Step 2: Run tests**

Run: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts && python tests/run_batches.py --batch 3`

**Step 3: Commit**

```bash
git add qwen3_tts/server/app.py
git commit -m "fix: document content negotiation in /generate OpenAPI responses"
```

---

### Task 8: Write tests for R-1 through R-12 features

**Files:**
- Create: `tests/test_remediation_2026_03_04.py`

**Step 1: Write test file**

Test the following pure/unit-testable features:
- `_validate_audio()` — NaN, clipping, silence, normal audio
- `_validate_generation_request()` — valid request, bad mode, path traversal, bad speaker
- `_error_response()` — raises HTTPException with correct structure
- `_torch_prompt_cache` thread safety — concurrent access doesn't crash
- `voice_prompt_cache_info()` — returns correct structure
- HealthResponse / GenerateResponse Pydantic models — valid data serializes correctly

**Step 2: Run new tests**

Run: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts && python -m unittest tests.test_remediation_2026_03_04 -v`
Expected: All pass.

**Step 3: Run full suite**

Run: `source ~/miniforge3/etc/profile.d/conda.sh && conda activate qwen3-tts && python tests/run_batches.py`
Expected: All 5 batches pass.

**Step 4: Commit**

```bash
git add tests/test_remediation_2026_03_04.py
git commit -m "test: add tests for R-1 through R-12 review features"
```

---

## Verification

After all tasks:
1. `python tests/run_batches.py` — all batches pass
2. `git log --oneline -10` — 8 clean commits
3. Review CLAUDE.md "Known issues" section — remove items that were fixed
