# TDD Evidence: Atomic auth-token write

**Source plan:** None — derived during this run from a live-debugging finding
(server log `Auth failure: missing_token` during `tts server restart`).

## User journey

As a user running `tts ui`, I want the model-status poll to authenticate on
every attempt even while the server is restarting, so that the UI never shows
spurious auth failures (and the token file is never observed empty).

## Root cause

`_write_auth_token` (`qwen3_tts/server/app_lifespan.py`) used
`open(TOKEN_FILE, "w")` — a truncate-then-write. Between truncation and the
write completing the file is **empty on disk**. `auth_headers()` reads the
token fresh on every call, so a Gradio UI `/models` poll landing in that window
got `""` → `{}` headers → request with no `Authorization` → server logs
`Auth failure: missing_token`.

## Task report

| # | What is guaranteed | Test | Type | Result | Evidence |
|---|---|---|---|---|---|
| 1 | Token file is replaced atomically (new inode), not truncated in place | `TestAuthTokenWriteIsAtomic::test_write_replaces_file_atomically_not_truncates_in_place` | unit | PASS | RED: same inode `218394094==218394094`; GREEN: inode changes |
| 2 | A concurrent reader never observes an empty/partial token during writes | `TestAuthTokenWriteIsAtomic::test_concurrent_reader_never_observes_empty_or_partial_token` | unit (concurrency) | PASS | RED: **511** empty reads in 800 writes; GREEN: **0** |
| 3 | Write failure still aborts startup (RuntimeError) | `TestWriteAuthToken::test_raises_when_write_fails` | unit | PASS | failure injected at `tempfile.mkstemp` (new write path) |
| 4 | Restricted perms preserved (0o600 file, 0o700 parent) | `TestWriteAuthToken::test_writes_token_with_restricted_perms` | unit | PASS | unchanged by fix |

**Validation command:** `python -m pytest tests/test_auth_token_write.py -v` → `7 passed`.
**Static gates:** `ruff check` → All checks passed; `mypy` → no issues.

## Fix

Switched `_write_auth_token` to the same atomic pattern already used for config
(`core/config/io.py`) and the PID file (`core/config/pid.py`): `tempfile.mkstemp`
in the same directory → `fsync` → `os.replace` (POSIX-atomic). The previous token
stays fully readable until the new one is in place.

## Commits (RED → GREEN)

- `1aea8f3` `test(server): reproduce non-atomic auth-token write race` (RED)
- `b041156` `fix(server): write auth token atomically to close missing_token race` (GREEN)

Branch `fix/auth-token-atomic-write`. Safe to squash — RED/GREEN evidence preserved here.
