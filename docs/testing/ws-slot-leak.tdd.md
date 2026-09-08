# TDD Evidence: Step 0D — WebSocket connection-slot leak on the pre-auth path (one handler-level `finally`)

**Source plan:** docs/plans/2026-09-08-step-0d-ws-slot-leak.plan.md
(Step 0D of docs/plans/2026-09-06-consolidated-backlog-priority.plan.md)

**Branch:** `fix/ws-slot-leak-preauth` (cut from main @ `b9c61f1`, the 0C merge)

## Problem

`_ws_try_acquire` (`websocket.py:90`) reserves a per-IP slot BEFORE `websocket.accept()`,
so every exit from `websocket_tts_handler` owes exactly one release. That release used to
live in FOUR scattered places, each reached only if the socket call immediately before it
succeeded: the non-dict-JSON branch (base `:112`), the invalid-token branch (base `:122`),
the auth `except` (base `:136`), and the post-auth message-loop `finally` (base `:280`).
Any socket call that raises on a vanished peer — `accept()`, the error `send_json()`, the
4001 `close()` — skips the release that follows it. `accept()` (base `:95`) sat outside
every `try`, so a peer vanishing mid-handshake leaked a slot with no release anywhere on
its path. Repeated leaks exhaust `_WS_MAX_TOTAL` (50): an unauthenticated client can hold
slots it is not using — a slot-exhaustion DoS of the same family the `:101-105` comment
records as fixed once before, via a different escape.

The most reachable of the four is the **auth-except double fault**. When the original auth
failure is a disconnect (the common case — the client hung up during the 10 s auth
window), the `except` block's own `await websocket.close(code=4001, ...)` raises a FRESH
`RuntimeError` on the dead socket, which escapes the handler before the release line at
base `:136` runs. Every disconnect-flavored auth failure leaks.

The fix: ONE `try:` opened immediately after the acquisition block, with a single
handler-level `finally: _ws_release(app_state, client_ip)`; the four scattered releases
DELETED; the over-limit 1013 rejection left OUTSIDE the try (it never acquires).

## The leak paths (base `b9c61f1` line numbers)

| Base site | Path | Escape |
|---|---|---|
| `:95` | `accept()` outside any `try` | peer vanishes mid-handshake → the exception propagates and no release exists anywhere on the path (the plan's named escape) |
| `:112` | non-dict-JSON branch | `close(4001)` raises → the release after it never runs |
| `:122` | invalid-token branch | the error `send_json` or the 4001 `close()` raises → same |
| `:136` | auth `except` | the except's own `close(4001)` raises fresh on the dead socket and escapes BEFORE the release — the most reachable (any disconnect-flavored auth failure) |
| `:280` | message-loop `finally` | correct in isolation; redundant under the single-finally design and deleted (its `stop_event.set()` stays) |

## Task report

| Task | Scope | RED | GREEN |
|---|---|---|---|
| 1 | the single try/finally wrap + 7 leak scenarios (`tests/test_websocket_slot_release.py`, new; batch 3) | **5 failed / 2 passed** at BASE+tests — every leak scenario stuck at `{'1.2.3.4': 1}` | **7 passed**, both interpreters (mlx py3.11.14; torchless `.venv-310` py3.10.21 RUN-not-skip) |
| 2 (fold) | scenario-5 no-masking regex pin + scenario-3 modeling docstrings | n/a (test-doc fold; teeth verified by mutant, below) | **7 passed** |

## The RED run (Task 1, verbatim)

`conda run -n qwen3-tts-mlx python -m pytest tests/test_websocket_slot_release.py -v --tb=short`
against BASE+tests (pre-fix tree) → **5 failed, 2 passed in 0.16s**. Full capture:
`.superpowers/sdd/detailed-step-0d-execution-misty-harbor/task-1-red-output.log`.

Every leak scenario failed the SAME way — the counter stuck at the leaked reservation:

```
tests/test_websocket_slot_release.py:114: in test_accept_failure_releases_slot
    self._assert_slot_freed(state, "accept() raised")
E   AssertionError: {'1.2.3.4': 1} != {}
E   - {'1.2.3.4': 1}
E   + {} : connection slot leaked: accept() raised
```

Per-path evidence, one line each (all identical in shape, `{'1.2.3.4': 1}` != {}):

- **accept raises** — `connection slot leaked: accept() raised`
- **non-dict first message + close raises** — the captured log shows the double-fault
  trace: the inner close's failure caught by the auth except, whose own close raises
  again and escapes.
  ```
  WARNING  tts.server.websocket:websocket.py:107 ... first message not a JSON object
  WARNING  tts.server.websocket:websocket.py:129 ... close failed: peer already gone
    File ".../websocket.py", line 111, in websocket_tts_handler
      await websocket.close(code=4001, reason="Authentication failed")
  RuntimeError: close failed: peer already gone
  ```
- **invalid token + send_json raises** — `RuntimeError: send failed: peer already gone`
  from `websocket.py:120`, then the stuck counter.
- **invalid token + send OK, close raises** — `RuntimeError: close failed: peer already
  gone` from `websocket.py:121`, then the stuck counter.
- **auth disconnect + close raises (scenario 5, the double fault)** —
  ```
  WARNING  tts.server.websocket:websocket.py:129 WebSocket auth failed from 1.2.3.4:
  Traceback (most recent call last):
    File ".../websocket.py", line 99, in websocket_tts_handler
      auth_msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
  starlette.websockets.WebSocketDisconnect
  E   AssertionError: {'1.2.3.4': 1} != {} : connection slot leaked: auth disconnect + close() raised
  ```
  The traceback the except logs is the disconnect; the exception that ESCAPES (and that
  the pre-fix `assertRaises(RuntimeError)` accepted) is the handler's own close failure.
  At RED the exception-surface assertion passed while the count assertion failed — the
  leak, not the exception surface, was the defect.

The two GREEN-at-RED scenarios are structural pins by design: the over-limit rejection
(scenario 6 — counter `{'1.2.3.4': 5}` untouched, close 1013, never accepted) and the
shared-IP sibling guard (scenario 7 — seed 2 siblings, one failing auth, counter back to
exactly 2). Only scenarios 1-5 drive RED.

## GREEN

`conda run -n qwen3-tts-mlx python -m pytest tests/test_websocket_slot_release.py -v` →
**7 passed in 0.10s**, all seven named:

```
test_accept_failure_releases_slot PASSED
test_auth_disconnect_with_failing_handler_close_releases_slot PASSED
test_failing_auth_preserves_sibling_slot_for_same_ip PASSED
test_invalid_token_close_failure_releases_slot PASSED
test_invalid_token_send_failure_on_dead_socket_releases_slot PASSED
test_non_dict_auth_with_failing_close_releases_slot PASSED
test_over_limit_rejection_leaves_counter_untouched PASSED
```

Torchless `.venv-310` proxy (RUN-not-skip): **7 passed, 0 skipped** (1 pre-existing
starlette/httpx TestClient deprecation warning, the same one 0C recorded).

## Design decisions

| Decision | Rationale |
|---|---|
| ONE handler-level `finally` | `accept()`, the auth step, and the message loop all sit inside one `try:` opened at `:95` with zero statements between it and the acquisition return; the `finally: _ws_release(...)` at `:286` is the only release call site. Every exit — return or raise, auth or message loop — crosses exactly one release. |
| DELETION, not idempotence | `_ws_release` decrements the per-IP counter unconditionally, so a second release for the same IP steals a LIVE sibling's reservation: `conns[ip] = 2` + double release → `0` → key deleted → the surviving connection uncounted (the cap then under-counts and the leak becomes a slow capacity loss). Making the scattered releases idempotent would also have kept four sites that each need future audit; deleting them leaves one site to reason about. Pinned by scenario 7. |
| Over-limit rejection outside the `try` | The `:90-93` rejection (log + close 1013 + return) runs after `_ws_try_acquire` returned False — this connection never incremented anything, so the `finally` must not release. Releasing there would decrement a stranger's count — the same sibling-steal, from the other side. Pinned by scenario 6 (counter `{'1.2.3.4': 5}` untouched). |
| `finally`, not `except` | A `finally` runs on `BaseException` — including `asyncio.CancelledError` (a `BaseException` since 3.8) and `GeneratorExit` — where an `except Exception` chain would let the slot leak. The review traced every BaseException exit through the handler to exactly one release. |
| Exception surface unchanged | The `finally` only frees the slot: `accept()`/`close()` raising still propagates exactly as before (FastAPI logs it), close codes 1013/4001/1011 and every log line are untouched. Scenario 5's message pin (below) holds the no-masking half of this. |
| Direct-drive fakes, not `TestClient` | TestClient cannot make `close()` raise — the 0B/0C precedent in `test_websocket_rate_limit.py`. `_RaisingCloseWebSocket(raises_on=...)` raises `RuntimeError` per operation on EVERY call, so the except block's own close retry fails the way a real dead socket does. |
| Zero sleeps, deterministic | No interleaving is simulated: the pre-fix defect is deterministic (a raising stub is sufficient), so each scenario is a single awaited call with the counter asserted afterward. |

## Scenario 3's modeling (why the send_json-only sketch was abandoned)

The plan sketched a single-op stub (`raises_on="send_json"`). With ONLY `send_json`
raising and `close()` succeeding, the auth except releases cleanly and returns — that
variant is NOT a leak, so it could never RED on the pre-fix code and would stay red after
any fix (the swallowed exception is by-design behavior). A failed send implies the peer is
gone, so a subsequent `close()` fails too; scenario 3 therefore raises on BOTH
(`raises_on={"send_json", "close"}`), modeling a real vanished peer. Recorded in the
module and method docstrings (Task 2 fold).

## The Task 2 fold (the no-masking pin)

Scenario 5's bare `assertRaises(RuntimeError)` became
`assertRaisesRegex(RuntimeError, "close failed")` — a substituted `RuntimeError` (or the
disconnect that started the fault) can no longer pass, so the test pins the no-masking
property it claims: what surfaces after the fix is the handler's own close failure,
exactly as it did before it. Teeth verified by mutant: with `_RAISES["close"]` substituted
to `"substituted failure unrelated to close"`, the scenario fails with
`"close failed" does not match "substituted failure unrelated to close"` (1 ran, 1
failed); restored, the module is 7/7 green.

## Final state (counted, not assumed)

`pytest --collect-only` (`qwen3-tts-mlx`, HEAD `b5532d9`):

- `tests/test_websocket_slot_release.py` — **7** (`TestWebSocketSlotRelease`: scenarios
  1-5 the leak paths, scenario 6 the rejection-path pin, scenario 7 the shared-IP sibling
  pin)

Full non-e2e at the same HEAD: **3224 passed, 4 skipped, 92 deselected, 0 failed** —
exactly **+7** over 0C's 3217 (the new module; nothing else moved).

Release-site census: `_ws_release` appears at the `:57` definition and ONE call site
(`:286`, the handler-level finally); the four base sites (`:112`, `:122`, `:136`, `:280`)
are deleted. The `try:` opens at `:95` with zero statements between it and the
acquisition return; the rejection block (`:90-93`) is outside it. `git diff -w`
base..fix is **17 insertions + 11 deletions (28 changed lines) across 4 hunks**, versus
**348 raw** (177 + 171 — the one-level re-indent of the region that moved inside the
try); the hunks are the re-indent, the try/finally, and the two comment blocks.

## Gates (Step 0D, observed at HEAD `b5532d9`)

| Gate | Command | Result |
|---|---|---|
| Full non-e2e, gate convention | `conda run -n qwen3-tts-mlx python -m pytest tests/ -m "not e2e" -q --tb=short --ignore=tests/evaluations` | **3224 passed, 4 skipped, 92 deselected, 0 failed** in 53.07 s |
| Batch 3 (Server Infrastructure, incl. the new module) | `conda run -n qwen3-tts-mlx python tests/run_batches.py --batch 3` | 1/1 batches passed |
| ruff | `conda run -n qwen3-tts-mlx ruff check qwen3_tts tests` | All checks passed |
| mypy (entry-point form) | `conda run -n qwen3-tts-mlx mypy qwen3_tts/core qwen3_tts/server qwen3_tts/interface` | Success: no issues found in 58 source files (pre-existing `annotation-unchecked` note only) |
| Torchless RUN-not-skip (new module) | `./.venv-310/bin/python -m pytest tests/test_websocket_slot_release.py -q -rs` | **7 passed, 0 skipped** |
| Batch registration | `tests/test_batches_coverage.py` | passed (registered in batch 3, Task 1) |
| Stale-count sync grep | `git grep -n "3217\|3228\|3232" -- CLAUDE.md docs/COMMANDS.md docs/CONTRIBUTING.md docs/RUNBOOK.md Makefile` | zero hits (grep exit 1) — no count line to sync |

bandit was not re-run in Task 2 (not on this task's gate list): Task 1 ran it clean at
`21de710` (exit 0, 0 HIGH), and Task 2 touches only test and docs files.

## Commit list (in order)

- `75b20cd` — `test(server): add RED leak tests for the websocket slot release`
- `21de710` — `fix(server): release the websocket slot in one handler-level finally`
- `b5532d9` — `test(server): tighten the slot-release pin and document the scenario-3 modeling`

## Accepted residuals (disclosed)

1. `ruff format --check` flags `websocket.py` — format-dirty at BASE too (verified by
   stash → check → pop in Task 1), across lines this branch never touches. Formatting is
   not a project gate (`ruff check` is); reformatting would have violated the
   touch-nothing-else constraint.
2. Coverage % not measured (no `--cov` run requested). The new module is behavioral —
   exception surface plus counter state — not implementation-poking.
