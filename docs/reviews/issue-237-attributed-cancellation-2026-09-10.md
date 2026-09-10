# Issue #237 — Attributed Cancellation: Evidence Document

**Date:** 2026-09-10
**Branch:** `fix/issue237-batch-cancel-race`
**Commits in scope:** `2f75b1f2` (Task 1 — guard layer) .. `a1cfed50` (Task 1 fix round 1) ..
`818265d2` (Task 2 — server layer) .. `502882b` (Task 2 fix round 1) .. `4fbf1003` (docstring
correction) .. `d8197f0` (Step 1A paperwork)

This document is Part 3 of the Task 3 brief for Step 1A of the consolidated backlog plan
(`docs/plans/2026-09-06-consolidated-backlog-priority.plan.md`, Step 1A). It proves the fix for
issue #237 (a `/cancel-generation` request can be silently lost to two race windows) with real
command output, then records every counted gate from Part 1 and the live-server E2E result from
Part 2.

---

## The defect this fix closes

Before this branch, `generation_state["cancelled"]` was a bare, untargeted boolean.
`/cancel-generation` set it; every generation's inner loop checked it; and — critically — each
batch item's `begin()` call **blanket-cleared** it back to `False` on every invocation, on the
theory that a cancel not yet observed by the time the next item starts must be stale. That blanket
clear is exactly what erases a cancel that arrives in the narrow window between one item's check
and the next item's `begin()` (Window 1), and it left `/cancel-generation` with nothing to latch
onto at all when it arrives before any generation has gone active (Window 2) — that request instead
read `active: False` and bounced as `no_active_generation`.

The fix (`generation_state_guard.py`'s `GenerationStateGuard`) replaces the bare boolean with an
**attributed** cancel: `set_cancelled(target_id)` records which generation id the cancel addresses,
`is_cancelled_for(generation_id)` is the only read production code performs, and `begin()`'s erase
rule is no longer unconditional — it only erases a cancel that is genuinely stale (see Window 1
below for the exact four-case rule).

---

## Window 1 closed — a cancel landing after item *i*'s check but before item *i+1*'s `begin()`

**Mechanism.** `begin()` (`qwen3_tts/server/generation_state_guard.py:146-204`) erases
`cancel_target_id` only when all four of these hold: a target is set, it does not equal the
incoming `generation_id`, it does not equal the state's *current* owner (`state["generation_id"]`,
read **before** this call overwrites it), and it is not a still-pending sibling id. The docstring
states the rule as four ordered cases:

```
- cancel_target_id == generation_id: preserved (a cancel addressed to THIS batch
  survives its own next begin()).
- cancel_target_id names a still-PENDING sibling: preserved.
- cancel_target_id names the state's CURRENT owner (fix round 1, Important 1) —
  i.e. state["generation_id"] at the moment this runs, before it is overwritten
  with the incoming id: preserved.
- cancel_target_id is set, non-matching, NOT pending, and NOT the current owner
  (genuinely stale): erased.
```

The production code (`generation_state_guard.py:190-198`):

```python
with self._lock:
    state = self._state
    target = state.get("cancel_target_id")
    if (
        target is not None
        and target != generation_id
        and target != state.get("generation_id")
        and target not in self._pending_ids
    ):
        state["cancelled"] = False
        state["cancel_target_id"] = None
    self._pop_pending_target(generation_id)
    state["active"] = True
    state["generation_id"] = generation_id
    ...
```

The third case (`target != state.get("generation_id")`) is Controller Ruling I / fix round 1's
"Important 1" fix — without it, a cancel addressed to a live, already-begun batch A is still
erased the moment a concurrent, DIFFERENT generation B calls `begin()`, because A is no longer
"pending" (its own `begin()` already consumed that registration) yet A ≠ B. That is the exact
Window-1 interleaving: batch A is between item *i*'s inference (where the cancel is set) and item
*i+1*'s check (which reads `is_cancelled_for`); a sibling stream/batch's `begin("B")` runs in
between and must not erase A's still-live cancel.

**The pin that fails without it** — `tests/test_generation_state_guard.py::TestGenerationStateGuardBegin::test_begin_preserves_a_cancel_targeted_at_the_live_owner_across_a_different_begin`:

```python
def test_begin_preserves_a_cancel_targeted_at_the_live_owner_across_a_different_begin(self):
    # Fix round 1 (Important 1): a cancel addressed to the CURRENT owner
    # must survive a concurrent, different generation's begin(), not just
    # its own. Batch A begins, is cancelled, releases the lock between
    # items without reaching its own next begin() — a sibling's begin("B")
    # must not erase A's still-live cancel. At the moment begin("B") runs,
    # state["generation_id"] is still "A" (not yet overwritten), which is
    # exactly the signal that distinguishes this from a genuinely stale
    # target.
    self.guard.begin("gen-A")
    self.guard.set_cancelled("gen-A")
    self.guard.begin("gen-B")
    self.assertTrue(self.guard.is_cancelled_for("gen-A"))
```

With the third case deleted from `begin()`'s erase condition, `begin("gen-B")` sees
`target="gen-A"`, `target != "gen-B"`, `target not in self._pending_ids` (A already consumed its
own pending registration) → erased → `is_cancelled_for("gen-A")` is `False` → pin fails.

A second, server-level pin drives the real batch handler through the same interleaving end to end
(`tests/test_batch_generation_state_ownership.py::TestBatchGenerationStateOwnership::test_cancel_survives_a_concurrent_sibling_begin_mid_batch`,
lines 463-476): item 0's mocked inference sets a cancel on the batch's own id, then calls
`guard.begin("concurrent-sibling-generation")` before the batch's own next per-item check —
"reproducing 'batch A releases the lock between items, a concurrent stream/batch begins as B, A's
next check must still see its own cancel'."

---

## Window 2 closed — a cancel arriving before item 0 makes the state active

**Mechanism.** `/cancel-generation` (`qwen3_tts/server/app.py:891-928`) now has two branches. If a
generation is already active, it targets the cancel at the active id as before. If nothing is
active yet, it falls through to a second branch that did not exist pre-fix:

```python
# Window 2 (issue #237 / Step 1A): nothing is active yet, but a batch may
# have minted its id and registered it pending before ever taking the
# lock. Latch the cancel onto that id so the batch's first item honors it
# instead of the request bouncing as no_active_generation.
pending_id = guard.peek_pending_target()
if pending_id is not None:
    guard.set_cancelled(pending_id)
    logger.info("Generation cancellation requested (pending)")
    return {
        "status": "cancellation_requested",
        "generation_id": pending_id,
    }

return {"status": "no_active_generation"}
```

The latch is the **pending registry** (`register_pending` / `deregister_pending` /
`peek_pending_target`, `generation_state_guard.py:245-284`): `handle_generate` registers its minted
id as pending immediately, before it ever queues for the inference lock or calls `begin()`. A
cancel that lands in that window now finds a target to attach to (`peek_pending_target()` — a
non-mutating, deterministic `min()` over the pending set) instead of reading `active: False` and
bouncing.

**The pin that fails without it** —
`tests/test_batch_generation_state_ownership.py::TestCancelGenerationPendingLatch::test_cancel_with_a_registered_pending_id_stops_the_batch_before_it_begins`
(lines 790-820+): the test registers a pending id exactly as `handle_generate` does at mint time,
calls the real `cancel_generation` handler while nothing is active, and asserts (a) the response is
`cancellation_requested` addressed to the pending id, and (b) once the batch actually runs, its
very first per-item check sees the cancel and stops **before** calling `begin()` or running any
inference at all. Without `peek_pending_target()` / the pending-registry branch, the handler falls
straight through to `return {"status": "no_active_generation"}` and the batch runs to completion
untouched — the exact issue #237 symptom.

---

## The no-client-channel proof

The fix creates the server-side correlation the two independent HTTP requests (`/generate` and
`/cancel-generation`) otherwise have no way to share, with **zero client API change**. Grounded in
the actual shapes:

**Request — `GenerateRequest`** (`qwen3_tts/server/validation.py:35-54`) carries no generation-id
or correlation field at all:

```python
class GenerateRequest(BaseModel):
    """Request model for /generate and /generate-stream endpoints."""

    text: str | None = None
    texts: list[str] | None = None
    mode: str = "clone"
    prompt_file: str | None = None
    voice_description: str = ""
    language: str = "auto"
    speaker: str | None = None
    instruct: str = ""
    temperature: float = ...
    top_k: int = ...
    top_p: float = ...
    repetition_penalty: float = ...
    max_new_tokens: int = ...
    seed: int | None = ...
    max_chunk_chars: int | None = ...
    x_vector_only_mode: bool = False
    seed_lock_chunks: bool = False
```

No field here identifies the generation to a later cancel — the client never chooses or sends an
id.

**Response — `GenerateResponse`** (`validation.py:144-151`) carries no id the client must
remember and echo back:

```python
class GenerateResponse(BaseModel):
    """Response model for /generate endpoint."""

    results: list[GenerateResult]
    cancelled: bool = False
```

**`/cancel-generation`'s own request/response** (`app.py:891-899`, `:908-928`) takes **no request
body at all** (`async def cancel_generation(request: Request, _auth: None = Depends(verify_auth))`
— no Pydantic model parameter) and its response only ever returns
`{"status": ..., "generation_id": ...}` where `generation_id` is **informational** — the server
already knew it before responding; the client never has to supply, store, or resend it.

The correlation is entirely server-side: `handle_generate` mints a `uuid4`-derived id, registers it
pending, and `begin()`s it; `/cancel-generation` reads whichever id the guard's own state (active
owner or pending registry) currently names and targets the cancel at that. Two independent,
unrelated HTTP requests are matched up purely through shared server state — the SDK, the CLI
client, and the Gradio UI needed **zero code changes** to gain this fix; `git diff` confirms no
client-facing schema field was added, removed, or renamed by this branch (`server/client/` is
untouched by `818265d2`/`502882b`/`4fbf1003`).

---

## The stale-flag defense that replaced the blanket clear

**Mechanism.** The old defense was a blanket `clear_cancelled()` run once per batch, before the
loop starts, on the theory that any lingering flag from a previous, unrelated cancel must be
cleared before this batch begins. Step 1A deletes that blanket clear (Controller Ruling E) because
it is exactly what re-introduces Window 2 (it would erase a cancel that just latched onto this
batch's own pending id). The replacement defense is **target-matching**: every check is
`is_cancelled_for(own_id)`, so a cancel addressed to a *different* id is structurally incapable of
stopping this batch — there is no shared, untargeted flag left to read.

**The pin** —
`tests/test_batch_generation_state_ownership.py::TestBatchGenerationStateOwnership::test_batch_not_stopped_by_a_differently_targeted_cancel`
(lines 405-435) and its response-shape sibling
`TestBatchResultCompleteness::test_differently_targeted_cancel_does_not_truncate_the_batch`
(lines 684-...): item 0's mocked inference calls `guard.set_cancelled("some-unrelated-generation-id")`
— a cancel that exists in the state but targets a foreign id — and the pin asserts the batch runs
to completion untouched, reporting `cancelled: False` with both results present, exactly like the
uncancelled case:

```python
def mock_inference(model, text, **kwargs):
    guard = guard_for(state)
    guard.set_cancelled("some-unrelated-generation-id")
    return fake_wav, 24000
```

Without target-matching (i.e. under the old bare-boolean `is_cancelled()` read), this exact
interleaving would truncate the batch after item 0 — a cancel meant for someone else's generation
silently stopping this one, which is the inverse failure mode Controller Ruling E named as strictly
worse than the pre-1A blanket-clear defense it replaces.

---

## Part 1 — counted gate output

Every invocation below used `conda run -n qwen3-tts-mlx` (ambient conda resolves correctly via
`conda run`; the homebrew-base fallback path was not needed). Neither
`prefer-batch-runner-over-raw-pytest` nor `require-server-ready-before-full-test` prompted during
this run — no interactive PreToolUse prompt was surfaced for the raw full-suite invocation or the
batch runner calls in this session.

### 1. Full non-E2E suite, counted

```
conda run -n qwen3-tts-mlx python -m pytest tests/ -m "not e2e" --ignore=tests/evaluations -q
```

```
========= 3325 passed, 5 skipped, 92 deselected, 3 warnings in 54.09s ==========
```
Exit code: 0.

**Delta from the 3300 passed / 4 skipped / 0 failed baseline (`a482440`, pre-1A):**

- **+25 passed.** `git diff --numstat a482440..4fbf1003 -- 'tests/*.py'` shows all added/changed
  test-file lines confined to exactly the branch's own test surface: `tests/test_generation_state_guard.py`,
  `tests/test_generation_state_routing.py`, `tests/test_generation_state_readers.py`,
  `tests/test_batch_generation_state_ownership.py`, `tests/test_streaming_cancel.py`,
  `tests/conftest.py`, `tests/voice_test_helpers.py` — the same six modules (plus the two shared
  fixture files) Task 1 and Task 2 touched, per-range: Task 1 (`a482440..2f75b1f2..a1cfed50`) added
  the guard's new pins (135+12 lines, 8+5 removed); Task 2 (`a1cfed50..818265d2`) added 258/52/57/35
  new lines across the batch-ownership, readers, routing, and streaming-cancel modules; Task 2 fix
  round 1 (`818265d2..502882b`) added 72+27 more lines to the batch-ownership and guard modules for
  the Important-1 fix and its pins. No unrelated test file changed. This accounts for the pass-count
  increase; no unexplained delta.
- **+1 skipped (5 vs 4).** The new skip is
  `tests/test_ui_headless.py:311: live TTS server on :5123 is down — start it (tts server start) to
  run the create-from-audio E2E case`. This is a pre-existing, server-liveness-conditional skip
  (unrelated to this branch's code) that fires because the server was still stopped at the moment
  this suite ran, ahead of Part 2's live-server leg. It is expected to clear once the server is up
  and is not attributable to any change in this diff.
- **0 failed**, matching the baseline.

### 2. Torchless leg — RAN, not skipped

```
./.venv-310/bin/python -m pytest tests/test_generation_state_guard.py \
  tests/test_generation_state_routing.py tests/test_generation_state_readers.py \
  tests/test_batch_generation_state_ownership.py tests/test_streaming_cancel.py -v
```

```
=============== 91 passed, 1 warning, 3 subtests passed in 1.84s ===============
```
Exit code: 0. All 91 tests across the five modules PASSED (0 skipped) — this proves the torchless
CI-proxy leg actually ran the suite rather than skipping it (Python 3.10.21, no torch installed in
`.venv-310`). Task 2's review had already added no sixth module beyond these five plus the guard
module itself (verified: `git diff --stat a1cfed50..818265d2 -- tests/` and
`818265d2..502882b -- tests/` touch only `test_batch_generation_state_ownership.py` and
`test_generation_state_guard.py` beyond the original five named in the brief).

### 3. ruff

```
conda run -n qwen3-tts-mlx ruff check qwen3_tts tests
```

```
All checks passed!
```
Exit code: 0.

### 4. mypy (entry-point form)

```
conda run -n qwen3-tts-mlx mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface
```

```
Success: no issues found in 58 source files
```
Exit code: 0. (Two informational `annotation-unchecked` notes on untyped-function bodies in
`app_lifespan.py` and `app_generation.py` — pre-existing, not errors, do not affect the 0-issue
result.)

### 5. bandit

```
conda run -n qwen3-tts-mlx bandit -r qwen3_tts -c pyproject.toml
```

```
Total issues (by severity):
    Undefined: 0
    Low: 0
    Medium: 0
    High: 0
```
0 HIGH confirmed. Exit code: 0.

### 6. Batch runner — batches 2 and 3

```
conda run -n qwen3-tts-mlx python tests/run_batches.py --batch 2
```
```
Ran 529 tests in 5.002s

OK
✓ Batch 2 passed
```

```
conda run -n qwen3-tts-mlx python tests/run_batches.py --batch 3
```
```
Ran 861 tests in 25.952s

OK (skipped=3)
✓ Batch 3 passed
```

---

## Part 2 — live-server E2E cancel result: NOT RUN

The server was confirmed stopped at dispatch time; the controller asked the user to run
`tts server stop && tts server start` in parallel with this dispatch. This implementer did not
start or stop the server itself, per the binding constraint.

**Evidence of what was observed** — a bounded poll of 30 attempts at 10s intervals (5 minutes
total), run after Parts 1, 3 (mechanism sections), and 4 completed:

```
for i in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5123/ready)
  echo "poll $i: $code"
  if [ "$code" = "200" ]; then break; fi
  sleep 10
done
```

Result: all 30 polls returned `000` (connection refused — nothing listening on port 5123 at all,
not a 503-while-loading). Corroborating checks:

- `lsof -i :5123` — no process bound to the port.
- `ps aux | grep -i "tts_server\|uvicorn\|qwen3"` — no server process found (only unrelated
  Claude Desktop filesystem-extension processes).
- `tail -20 .voice_server.log` — the log's last entry is
  `2026-09-09 16:10:27 [tts] INFO: FastAPI server shutting down...`; no startup attempt appears
  for 2026-09-10 at all.

**Disposition: the E2E leg is reported NOT RUN**, not skipped and not passing. The evidence above
shows the server was never brought up during this dispatch's window — this is not a hollow
sub-second pass being reported as green; the E2E command
(`conda run -n qwen3-tts-mlx python -m pytest tests/ -m e2e -v -k "cancel or queue"`) was not even
attempted, because attempting it against a down server would produce loud skips indistinguishable
from a real pass only if misread, and the brief is explicit that this must not be reported as a
pass. This leg needs to be handed back for a follow-up run once the server is confirmed live
(`curl -s http://127.0.0.1:5123/ready` returning `200`).
