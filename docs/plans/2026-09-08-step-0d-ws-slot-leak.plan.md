# Execution Plan — Step 0D: WebSocket connection-slot leak on the pre-auth path

Spec: `docs/plans/2026-09-06-consolidated-backlog-priority.plan.md` Wave 0, Step 0D (~:154-169 pre-0C numbering).
Branch: `fix/ws-slot-leak-preauth` (spec-named), cut from main @ `b9c61f1` (the 0C merge).
Execution: superpowers:subagent-driven-development. Controller never implements.

## Global Constraints

1. **Exactly one release per acquired slot.** The fix is ONE `try: ... finally: _ws_release(app_state, client_ip)` wrapping everything AFTER a successful `_ws_try_acquire`; the four scattered `_ws_release` calls are DELETED (not kept — `_ws_release` decrements the per-IP counter, so a double release would steal a sibling connection's slot on a shared IP). The over-limit rejection return (`websocket.py:90-93`) stays OUTSIDE the try/finally: no slot was acquired there, so the finally must NOT run.
2. Behavior-preserving otherwise: response codes/reasons (1013, 4001), log lines, message-loop semantics, the `/ws` persistent-socket + frame conventions from 0B/0C — untouched. Exception PROPAGATION semantics: `accept()` raising still propagates (FastAPI logs it) — the only change is the slot is now freed.
3. Comment honesty: the `:101-105` historical comment (the past narrow-exception leak fix) is updated to describe the new invariant (every acquired slot is released by the handler-level finally); no stale claims survive.
4. Tests: direct-drive fake websockets per the file's 0B/0C precedent (`_RecordingCloseWebSocket`, `_FakeDisconnectWebSocket`, raising stubs); IsolatedAsyncioTestCase; event-driven; ZERO sleeps; every new test RUN-not-skip in the torchless `.venv-310` proxy; register any new module in `tests/run_batches.py` BATCHES.
5. Env: ALL python/pytest/ruff/mypy via `conda run -n qwen3-tts-mlx` (mypy ENTRY-POINT form — `python -m mypy` falsely reports missing). Never stage `config.json`; explicit-pathspec commits; never `--amend`; NO AI attribution; shell-quote backticks in commit messages.
6. Line numbers are descriptive (post-0C tree, surveyed 2026-09-08); locate by content.

## Controller survey (2026-09-08, post-0C tree @ b9c61f1)

`_WS_MAX_TOTAL = 50`, `_WS_MAX_PER_IP` per-IP cap; `_ws_try_acquire`/`_ws_release` under `_ws_conn_lock` (`websocket.py:42-65`). Acquisition at `:90`. LEAK PATHS after acquisition:
- `:95` `await websocket.accept()` — outside any try; peer-vanished mid-handshake → exception propagates → slot leaked (the plan's named escape).
- `:106-113` non-dict-JSON branch: `close()` then release at `:112` — close raising skips the release.
- `:115-123` invalid-token branch: `send_json` + `close()`, release at `:122` — either raising skips it.
- `:125-137` auth `except`: `close()` at `:135` INSIDE the handler, release at `:136` — when the original exception was a disconnect, this close raises a FRESH exception that escapes before the release (the most reachable leak).
- `:280` post-auth message-loop finally release — correct today, becomes redundant under the outer finally (delete).
The over-limit rejection (`:90-93`, close 1013, return) acquires nothing — correct today, must stay outside the new try.
`_ws_release` callers: ONLY websocket.py (grep-verified). Over-release hazard: on a shared IP, `conns[ip]=2` + double release → 0 → deleted → the live sibling connection uncounted — why the scattered releases must be deleted, not made idempotent.

## Task 1 — the try/finally wrap + leak tests (the whole fix)

- Wrap everything after the `:90-93` acquisition block in `try: ... finally: _ws_release(app_state, client_ip)`; delete the releases at `:112`, `:122`, `:136`, `:280`; update the `:101-105` comment to the new invariant (constraint 3).
- RED-first tests (new class in `tests/test_websocket.py` or the routing module — your call; register if new): for EACH of the four leak paths, drive `websocket_tts_handler` directly with a fake websocket whose `accept`/`close`/`send_json` raises (a configurable raising stub per scenario), assert (a) the exception surfaces exactly as before and (b) `app_state._ws_connections` is EMPTY afterward (slot freed). RED on current code = the count stays 1 (leaked). Also one positive test: over-limit rejection (acquire returns False) does NOT touch `_ws_connections` and still closes 1013.
- Double-release guard test: on a shared IP with two live connections, whatever the handler does must leave the sibling's count intact (the single-finally design makes this structural — pin it).
- Gates: file suites green (existing ws tests unchanged), ruff, mypy entry-point, torchless RUN-not-skip.

## Task 2 — gates + evidence + paperwork

- Full gates: full non-e2e (`--ignore=tests/evaluations` convention; 0C measured 3217/4/0), batch 3, ruff, mypy, torchless RUN-not-skip on all new classes, stale-count grep + sync only in clean files.
- Evidence doc `docs/testing/ws-slot-leak.tdd.md` (0B/0C style): per-path RED/GREEN with the four leak scenarios, design decisions (single-finally, deletion-not-idempotence rationale, rejection-path exclusion), Final-state block with counted totals.
- Tracked plan copy `docs/plans/2026-09-08-step-0d-ws-slot-leak.plan.md`.
- SEPARATE docs commit on `docs/plans/2026-09-06-consolidated-backlog-priority.plan.md`: the **0C status-mark** (executed via PR #270, merged `b9c61f1`, 2026-09-08) per the established convention. Nothing else pending rides (the 12 follow-ups landed on 0C's branch; the pin follow-ups wait for their own touch).
- CLAUDE.md: check whether the `/ws` prose documents the slot behavior; if the invariant sentence ("every acquired slot is released by a handler-level finally") earns a line, add it; keep ≤300 lines.

## Verify / Exit criteria (from the spec)

- Verify: `pytest tests/test_websocket*.py -v`; ruff; mypy.
- Exit: slot count returns to baseline after a simulated mid-handshake disconnect AND after each auth-failure branch, even when the close/send call itself raises.

## Task ordering

T1 (fix + tests) → T2 (gates + docs). Serial; T2 needs T1's counted totals.
