# TDD Evidence: Step 0F — auth-failure handling + client-facing error sanitization

**Source plan:** Step 0F of docs/plans/2026-09-06-consolidated-backlog-priority.plan.md
(two related-but-distinct error-handling gaps, bundled; plus the 0B-execution fold-in
on the classified streaming code)

**Branch:** `fix/auth-and-stream-error-sanitization` (cut from main @ `2ffd7b3`, the
#276 merge). Tasks 1–2 landed four commits: RED `727391e` → GREEN `b2b96bc` (Task 1,
auth 401/audit + scheme strip + four sanitize sinks), RED `28b18e5` → GREEN `0250c09`
(Task 2, terminal-frame sanitize + classification preservation + body normalization).
This doc covers BOTH tasks; gates at the bottom are counted at `0250c09` + the docs.

## Problem and threat model

Three gaps, one theme — the server's failure paths leaked more than they should, in
both directions (too much to the client, too little to the operator):

1. **Non-ASCII bearer token → 500-with-no-audit.** `secrets.compare_digest` raises
   `TypeError` on non-ASCII `str` input (`app.py` `verify_auth`), so a probe carrying
   one got a 500 + traceback instead of a cheap 401 — and because the exception
   escaped BEFORE the audit block, the R-26 `Auth failure: invalid_token` WARNING
   never fired: probing produced no audit-log entries at all (an audit-log bypass on
   exactly the traffic you most want logged). The `/ws` sibling `_verify_token` had
   the same latent TypeError, absorbed by its broad auth except into a bare
   `WebSocketDisconnect` instead of the structured invalid-token frame.
2. **Unsanitized exception text to clients (CWE-209 class).** `str(e)` from
   `FileNotFoundError`/`ValueError` carries absolute filesystem paths. The repo
   already closed this class on `/health` (`<path>` redaction) and at the `/ws`
   terminal-frame site (`websocket.py:636-641`), but four client-facing sinks still
   shipped raw `str(e)`: both clone-prompt 404s on `/generate` + `/generate-stream`
   (`app_generation.py:450`, `:809`), the `/ws` Invalid-request cascade frame
   (`websocket.py:255`), and the `/ws` MLX prompt-loader FileNotFoundError frame
   (`websocket.py:409`). The `/generate-stream` TERMINAL frame (below) was the fifth
   and worst — it is the one path that reaches the client on every mid-stream failure.
3. **The classify-collapse on the in-lock 503.** When the model unloads during the
   capture→iterate window, the in-lock guard raises an HTTPException whose dict detail
   carries `{"error": "...was unloaded while the generation queued; retry", "code":
   "model_unloaded"}`. `/generate-stream`'s flatten site caught it, took
   `str(e.detail)` as the message, and emitted the terminal frame with the DEFAULT
   code `inference_failed` — the classified `code` the `/ws` sibling preserves was
   dropped on the floor, and the raw dict text went to the client unsanitized. The
   RED run also exposed a pre-existing body-shape drift: the pre-stream
   `model_not_loaded` 503 used key `message` and omitted `recovery`, and because
   `state.model_load_errors` holds `None` (not a missing key) when there is no load
   error, `.get(mode, "Model not loaded")` never fired its default — the body shipped
   `"message": null` to clients most of the time.

## Fix shapes

**Task 1 (commits `727391e` → `b2b96bc`):**

- **`_tokens_equal(candidate, stored)` — bytes compare, not try/except** (`app.py:226`):
  `secrets.compare_digest(candidate.encode("utf-8", "replace"),
  stored.encode("utf-8", "replace"))`. Why bytes over `try/except TypeError`: one
  expression, no second code path to audit; keeps the constant-time property for
  non-ASCII input instead of failing fast OUTSIDE the comparison; and `"replace"`
  also absorbs lone surrogates, which strict UTF-8 encoding raises on. `verify_auth`
  (`:287`) and the `/ws` `_verify_token` closure (`:997`) both call it, so a
  non-ASCII token now takes the normal invalid-token path on BOTH surfaces (401 +
  audit on HTTP; error frame + 4001 on WS) instead of a 500/bare-disconnect.
- **`_strip_bearer_scheme(header_value)` — one shared helper, three call sites**
  (`app.py:210`; used at `:246` `verify_auth`, `:268` `_get_rate_limit_key`, `:286`
  `_get_token_key`). Strips ONE leading scheme token case-insensitively (RFC 6750
  2.1) and passes everything else through, replacing three `.replace("Bearer ", "")`
  calls. The rate-limit bucket is the reason it must be shared and exact: `_get_token_key`
  hashes the whole credential into the bucket key, and the old global `.replace()`
  ate the FIRST occurrence anywhere in the token — a token literally containing
  `"Bearer "` mid-string was hashed as its remainder (`sha256("abcdef")` for the
  credential `"abc Bearer def"`), silently merging two clients into one bucket (and
  `verify_auth` 401-ing the same credential the bucket had already seen). One helper
  at the definition site means the three consumers cannot drift.
- **Four sanitize sinks** (late-import `_sanitize_error` at each, mirroring the
  `websocket.py:636` terminal-frame precedent — no new module-scope imports):
  `app_generation.py:450` + `:809` (`detail=_sanitize_error(str(e))` on the clone
  404s), `websocket.py:255` (`Invalid request: {str(e)}` cascade frame),
  `websocket.py:409` (MLX loader FileNotFoundError frame).

**Task 2 (commits `28b18e5` → `0250c09`, all in `app_generation.py`):**

- **Terminal frame sanitized** (`:1027-1029`): the mid-stream failure emitter yields
  `encode_stream_error_frame(_sanitize_error(thread_error[0]))` — the last
  unsanitized `str(e)` on the streaming surface, closing the same CWE-209 class as
  the four Task-1 sinks.
- **Classification preserved through the flatten site** (`:875-895`): the
  `except HTTPException` branch now derives the frame code from a dict detail —
  `_detail.get("error") or _detail.get("code") or STREAM_ERROR_CODE_INFERENCE_FAILED`
  (default kept for str/other details) — and passes `code=_code` through
  `encode_stream_error_frame`, so the in-lock guard's `model_unloaded` survives the
  flatten instead of degrading to `inference_failed`. The message is sanitized here
  too (a non-dict detail can carry a path; same terminal-frame contract). The
  classification was REFRAMED, not added: the 0B fold-in originally described it as
  the HTTP body omitting `code`; the actual defect was the terminal FRAME flattening
  a classified 503 it already received.
- **Pre-stream `model_not_loaded` body normalized** (`:746-760`): `message` →
  `detail`, `"recovery": "restart"` added, matching the batch path's identical body
  (`:256-264` — the established server contract for this exact code, `restart` in the
  documented recovery vocabulary of `core/config/errors.py`). The `message: null` bug
  the RED exposed is fixed at the source:
  `state.model_load_errors.get(mode, "Model not loaded")` →
  `.get(mode) or "Model not loaded"` — the key normally EXISTS with value None, so
  the old default never fired. `recovery` is deliberately NOT carried on the terminal
  frame: the wire payload contract is `{"error", "code"}` (ONE format in
  `core/stream_protocol.py`, shared with the CLI and TTSClient); changing the payload
  shape would fork the contract, and for this error the message already ends in
  "; retry".

## The RED run (verbatim)

**Task 1** — 9 new RED tests + 2 rate-limit neighbours, run against `727391e^`
(`conda run -n qwen3-tts-mlx python -m pytest <nodes> -v --tb=short`): **11 failed,
1 passed** (the pass is `test_ascii_mismatch_still_401_and_audits`, the deliberate
refactor guard, green before and after by design). Verbatim failure lines:

```
FAILED tests/test_fastapi_app_ext2.py::TestVerifyAuthNonAscii::test_lowercase_bearer_scheme_authenticates - fastapi.exceptions.HTTPException: 401: Unauthorized
FAILED tests/test_fastapi_app_ext2.py::TestVerifyAuthNonAscii::test_non_ascii_bearer_token_returns_401_not_500 - TypeError: comparing strings with non-ASCII characters is not supported
FAILED tests/test_fastapi_app_ext2.py::TestVerifyAuthNonAscii::test_non_ascii_bearer_token_still_audits_invalid_token - TypeError: comparing strings with non-ASCII characters is not supported
FAILED tests/test_fastapi_app_ext2.py::TestBatchClonePromptErrorSanitized::test_batch_prompt_not_found_sanitizes_path - AssertionError: '<path>' not found in "[Errno 2] No such file or directory: '/Users/victim/voices/leaky.pt'" : unsanitized 404 detail: "[Errno 2] No such file or directory: '/Users/victim/voices/leaky.pt'"
FAILED tests/test_fastapi_app_ext2.py::TestGenerateStream::test_stream_prompt_not_found_sanitizes_path - AssertionError: '<path>' not found in "[Errno 2] No such file or directory: '/Users/victim/voices/leaky_clone.pt'" : unsanitized 404 detail: "[Errno 2] No such file or directory: '/Users/victim/voices/leaky_clone.pt'"
FAILED tests/test_websocket.py::TestWebSocketAuth::test_non_ascii_token_gets_invalid_token_treatment - starlette.websockets.WebSocketDisconnect
FAILED tests/test_websocket.py::TestWebSocketErrorReporting::test_prompt_load_file_error_sanitizes_path - AssertionError: '<path>' not found in "[Errno 2] No such file or directory: '/Users/victim/voices/leaky_clone.pt'" : unsanitized ws error: "[Errno 2] No such file or directory: '/Users/victim/voices/leaky_clone.pt'"
FAILED tests/test_websocket.py::TestWebSocketStreamErrorCascade::test_value_error_payload_is_sanitized - AssertionError: '<path>' not found in 'Invalid request: clone prompt missing: /Users/victim/voices/leaky_clone.pt' : unsanitized ws error: 'Invalid request: clone prompt missing: /Users/victim/voices/leaky_clone.pt'
FAILED tests/test_rate_limiting.py::TestRateLimitKeyFunctions::test_get_token_key_preserves_midstring_bearer - AssertionError: credential was mangled: got hash of a stripped value ('010971ea0013a09d'), expected the whole credential hashed ('cad2ec2eaea549b7')
FAILED tests/test_rate_limiting.py::TestRateLimitKeyFunctions::test_get_rate_limit_key_preserves_midstring_bearer - AssertionError: assert '192.168.1.100:010971ea0013a09d' == '192.168.1.100:cad2ec2eaea549b7'
FAILED tests/test_rate_limiting.py::TestRateLimitKeyFunctions::test_get_token_key_strips_lowercase_bearer_scheme - AssertionError: lowercase scheme not stripped: got '36b67c7f2560ee06', expected '4f6e1f6505526386'
```

Every failure is one of the right reasons: the `TypeError` (500-not-401, and the
audit test dying on the same exception before the R-26 record could fire), the raw
paths in the four sinks, the mangled credential hash (`010971ea0013a09d` =
sha256 of `"abcdef"`, the mid-string scheme eaten), and the WS asymmetry (bare
`WebSocketDisconnect` where the invalid-token frame was expected).

**Task 2** — three pins, run against BASE `b2b96bc`. Verbatim:

```
FAILED tests/test_issue214_unload_queued_window.py::TestStreamingTerminalFrameContract::test_streaming_terminal_frame_message_is_sanitized - AssertionError: '<path>' not found in 'voice prompt unreadable: /Users/alice/secret-voices/bob/reference.wav (corrupt header)' : terminal frame message was not sanitized: 'voice prompt unreadable: /Users/alice/secret-voices/bob/reference.wav (corrupt header)'
```

```
FAILED tests/test_issue214_unload_queued_window.py::TestPostLockSlotReRead::test_streaming_terminal_frame_keeps_the_classified_code - AssertionError: 'inference_failed' != 'model_unloaded'
- inference_failed
+ model_unloaded
 : the in-lock guard's classified 503 was flattened to the default code: {'error': 'design model was unloaded while the generation queued; retry', 'code': 'inference_failed'}
```

```
FAILED tests/test_issue214_unload_queued_window.py::TestStreamingTerminalFrameContract::test_pre_stream_model_not_loaded_body_carries_detail_and_recovery - AssertionError: 'detail' not found in {'error': 'model_not_loaded', 'message': None, 'model_type': 'design'} : pre-stream model_not_loaded body keys: ['error', 'message', 'model_type']
```

Pin 3's RED is the shape drift AND the `message: null` bug in one line — the body
carried `message: None` because the `.get(mode, default)` default is unreachable
(key exists, value None).

## Mutation proofs

**Task 1** — removed `_sanitize_error` from the streaming clone 404 sink
(`app_generation.py` `detail=_sanitize_error(str(e))` → `detail=str(e)`), on disk:

```
FAILED tests/test_fastapi_app_ext2.py::TestGenerateStream::test_stream_prompt_not_found_sanitizes_path
E   AssertionError: '<path>' not found in "[Errno 2] No such file or directory: '/Users/victim/voices/leaky_clone.pt'" : unsanitized 404 detail: ...
============================== 1 failed in 1.43s ===============================
```

Restored (exact inverse edit; all sinks grep-verified), green again.

**Task 2** — dropped the `code=` pass-through at the flatten site (frame reverts to
the default code):

```
FAILED tests/test_issue214_unload_queued_window.py::TestPostLockSlotReRead::test_streaming_terminal_frame_keeps_the_classified_code
AssertionError: 'inference_failed' != 'model_unloaded'
: the in-lock guard's classified 503 was flattened to the default code: {'error': 'design model was unloaded while the generation queued; retry', 'code': 'inference_failed'}
============================== 1 failed in 1.37s ===============================
```

Exactly pin 2 went red (pins 1/3 untouched by the mutant); restored, green again.

## Deliberate behavior change (pinned): lowercase scheme accepted

`_strip_bearer_scheme` strips the scheme case-insensitively, so `"bearer <token>"`
now authenticates and hashes exactly like `"Bearer <token>"` — RFC 6750 2.1 schemes
are case-insensitive, and the old code 401-ed lowercase schemes while the bucket
logic hashed them inconsistently. Pinned by
`TestVerifyAuthNonAscii::test_lowercase_bearer_scheme_authenticates` and
`TestRateLimitKeyFunctions::test_get_token_key_strips_lowercase_bearer_scheme`.
Anything that relied on 401-ing lowercase-scheme requests changes behavior here, on
purpose.

## Final state (counted, not assumed — at HEAD with the docs, this task)

- Full non-e2e: **3295 passed, 4 skipped, 92 deselected, 0 failed** in 62.98 s
  (exit 0) — the same total the Task-2 GREEN recorded, so the docs commits moved
  nothing.
- Batch 3 (0F's home batch; all five touched test modules are batch-3 members or
  the deliberately-unbatched `test_rate_limiting.py`): **1/1 batches passed**.
- ruff `qwen3_tts tests`: clean. mypy (entry-point form, core+server+interface):
  **Success: no issues found in 58 source files** (3 pre-existing
  `annotation-unchecked` notes).
- Torchless RUN-not-skip on the five touched modules
  (`test_fastapi_app_ext2`, `test_fastapi_server`, `test_issue214_unload_queued_window`,
  `test_rate_limiting`, `test_websocket`): **207 passed, 0 skipped** (zero `SKIPPED`
  lines, `-rs` census) — runs, does not skip, without torch.
- Stale-count grep (`3295|3238|3292` across CLAUDE.md, COMMANDS, CONTRIBUTING,
  RUNBOOK, Makefile): **zero hits** — no count line to sync.

## Accepted residuals (disclosed)

1. **Pin 3 asserts `detail`/`recovery` presence but not `message` removal.**
   `test_pre_stream_model_not_loaded_body_carries_detail_and_recovery` would still
   pass if a regression re-added the old `message` key alongside the correct ones;
   the shape drift is pinned only half-way (from-review nit).
2. **The pin-3 class docstring overstates which reader saw the `null`.** It says
   "clients reading ``detail`` saw None", but the JSON `null` rode the old
   `message` key — `message`-reading clients saw literal `null`, while
   `detail`-reading clients saw an ABSENT key. Same defect, mislabeled reader.
3. **The `/ws` in-lock guard (`websocket.py:544-547`) spreads the classified dict
   without sanitizing its human message** (`dict(e.detail)` → `send_json`). Safe
   today — the dict detail is server-authored text ("…model was unloaded while the
   generation queued; retry"), no exception interpolation — but it is now the ONE
   terminal sink of its class without a `_sanitize_error` wrap, so a future edit
   that routes client-influenced text through that detail would ship it verbatim.
   Pre-existing, deferred as a future candidate.
