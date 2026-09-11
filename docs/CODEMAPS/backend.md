<!-- Generated: 2026-09-11 | Files scanned: server/ (8.0k LOC), config/pm2.py (128 LOC) | Token estimate: ~850 -->

# Backend — FastAPI Server (:5123)

Base `http://127.0.0.1:5123`. Bearer-token auth on all endpoints except the public set.

## Routes (grouped by handler module)
- **Generation** — `app_generation.py`, `websocket.py`: POST `/generate`, POST `/generate-stream`, WS `/ws`, POST `/cancel-generation`
- **Models** — `app_models.py`: GET `/models`; POST `/load-model`, `/unload-model`, `/update-model-config`, `/update-startup-config`; POST `/load-asr`, `/unload-asr`, `/transcribe`
- **Prompts** — `app_prompts.py`: GET `/prompts`, `/preview-prompt`, `/prompt-details`; POST `/create-voice-prompt` (backend-dispatched — MLX is inference-free, #236), `/delete-prompt`, `/rename-prompt`
- **System** — `app.py`, `app_lifespan.py`: GET `/health`, `/ready`, `/generation-status`, `/queue-status`, `/stats`; POST `/shutdown`

**Public (no auth):** `/health` `/ready` `/generation-status` `/queue-status` — all registered as plain `@app.get(...)` decorators alongside authed routes; auth is enforced per-route via `Depends(verify_auth)`, not by route-registration style.

All JSON routes carry Pydantic `response_model=` contracts (`server/validation.py`, 18 response models covering health/ready/stats/models/generate/transcribe/prompts/ops) — binary routes (`/generate-stream`, `/ws`, `/preview-prompt`, `/shutdown`) deliberately untyped. Guarded by `tests/test_response_contracts.py`.

## Middleware & Security
- **Bearer auth** — token `~/.config/qwen3-tts/.voice_server_token`; write is atomic (temp file + fsync + `os.replace`)
- **Token parsing (#281)** — `_strip_bearer_scheme` removes ONE leading scheme token, case-insensitively (RFC 6750); the old `.replace("Bearer ", "")` mangled credentials containing "Bearer " mid-string (rate-limit bucket mis-hash + auth failures). `_tokens_equal` compares UTF-8-encoded bytes via `secrets.compare_digest` — a non-ASCII token is now a 401, not a TypeError 500; WS `_verify_token` does the same
- **CORS** — configurable allowlist
- **Rate limiting** — slowapi, env-tunable per limit (`TTS_RATE_LIMIT_{GENERATE,MODEL_OPS,TRANSCRIBE,PROMPT_OPS,CONFIG_OPS,GLOBAL}`); `TTS_DISABLE_RATE_LIMITING=1` disables all (tests only). Global pre-auth ceiling (`TTS_RATE_LIMIT_GLOBAL`, default 120/min) is decoupled from the 10/min `/generate` limit so UI polling doesn't 429
- **IP resolution** — `X-Forwarded-For` honored ONLY when direct peer is in `TTS_TRUSTED_PROXIES` (comma-separated IPs; loopback by default)
- **Body-size DoS** — `RequestBodySizeLimitMiddleware` (~100 MB); rejects oversized bodies without buffering, counters off ASGI stream; `Content-Length` fast path
- **WS validation** — `/ws` validates `Origin` header against CORS allowlist (CSWSH defense); absent Origin allowed (auth is per-message token)
- **Error sanitization** — `/health` redacts filesystem paths to `<path>` (CWE-209)
- **Startup lock** — `_acquire_startup_lock()` (`app_lifespan.py`) takes an exclusive non-blocking `flock` on `.voice_server.lock` before anything else in `lifespan()`, aborting a losing `tts server start`/`tts ui` process before it can clobber the winner's auth token
- **starlette** pinned `>=1.6.0,<2` — custom body-size middleware retained over native `max_body_size` (pre-auth no-buffer ordering + Content-Length fast path)
- **PM2 process-supervision (#248)** — new `core/config/pm2.py` detects PM2-managed servers via `pm2_owner_of_port(port)` (walks ancestor chain to find online PM2 app) and `pm2_registered_app(port)` (matches by naming convention for stopped apps). `cli_server.py` `start`/`stop`/`restart` commands now auto-delegate to `pm2 start`/`pm2 stop`/`pm2 restart` when detected, instead of spawning/killing directly (which PM2's autorestart would undo). `tts ui` stop button uses the same detection in `interface/ui/_facade.py`

## Client
`server/client/` — `TTSClient` (generator / models / voices / config_fetcher / _base). CLI & UI call the server through this. Surfaces `last_seed`, `last_chunk_count`. Per-route timeouts scale for long-running ops: `_generation_timeout(len(text))` for `/generate`, `LOAD_MODEL_TIMEOUT_SEC`/`TRANSCRIBE_TIMEOUT_SEC`/`CREATE_PROMPT_TIMEOUT_SEC`/`UNLOAD_ASR_TIMEOUT_SEC` = 900s for the leaf-locked inference paths (#192/#214 — see architecture.md).

## Generation path
`/generate → cache check → run_inference (chunk + backend.generate + _postprocess_chunk) → {chunks, seed}`.
Stream path returns length-prefixed float32 chunks (`[sr:4][len:4][payload]`); `/ws` is bidirectional with cancel + disconnect detection, and marks `generation_state` active so `/generation-status` sees WS work too.

## Generation-state guard & attributed cancellation (0C #270 / 1A #283)
`server/generation_state_guard.py` — `GenerationStateGuard` wraps `app.state.generation_state` in a `threading.Lock` (worker threads AND the event loop touch the dict; the asyncio `generation_lock` cannot serialize them). `guard_for(state)` resolves the guard late-bound; `snapshot(keys?)` is one atomic read — `/generation-status`, `/queue-status`, and `detect_degraded_generation` all snapshot instead of unlocked `.get` chains. EVERY read/write routes through the guard — raw dict access is pinned by `tests/test_generation_state_guard.py::TestGenerationStateRawAccessIsPinnedToTheGuard` (allowlist: only the lifespan init block). Idle shape now carries 11 keys including `cancel_target_id` (1A).

- `/cancel-generation` — ONE `snapshot(["active","generation_id"])`, then targeted `set_cancelled(target_id)`; when nothing is active, latches onto `peek_pending_target()` (a batch that minted its id but hasn't begun); else `no_active_generation`. Never a bare untargeted flag.
- Batch path — `register_pending(batch_gen_id)` at id-mint → per-item `begin(...)` → `deregister_pending` + `reset_if_owner(batch_gen_id)` at batch end. `begin()` erases a stale cancel target in exactly four cases (own id / still-pending sibling / the state's CURRENT owner are all preserved; a non-matching, non-pending, non-owner target is erased; `None` = no-op).
- Streaming — `_should_stop_streaming(stop_event, guard, gen_id)` = client disconnect OR `guard.is_cancelled_for(gen_id)` (target-aware: a cancel for another generation must not stop this stream).

## Streaming failure semantics
- Headers commit before the body iterates → no mid-stream status code. Server emits a terminal frame with `sample_rate == 0` (`STREAM_ERROR_SENTINEL_SR`, now defined once in `core/stream_protocol.py`) carrying JSON `{"error","code"}`. `TTSClient.generate_streaming` and the CLI's `iter_stream_chunks` both parse it through the same module — previously duplicated and drifted (#229), fixed by consolidating to one parser + `tests/test_stream_protocol.py` anti-re-fork assertions.
- `/ws`'s GENERIC failure handler closes with RFC 6455 `1011` after its error message; CLASSIFIED errors (validation, the in-lock model guard, the prompt-loader 503, mid-stream `thread_error`) instead spread `dict(e.detail)` as top-level `error`/`detail`/`recovery` fields on a structured frame and RETURN WITHOUT CLOSING — the socket is a persistent multi-request channel the client retries on. Never reintroduce a `close()` on a classified path.
- **WS connection slot (0D #271):** acquired pre-auth (`_ws_try_acquire`), released by ONE handler-level `finally` (`_ws_release`) — never re-add per-branch releases (a second release steals a live sibling's slot on a shared IP); the over-limit `1013` rejection never acquires and sits outside the try.
- `_stream_thread_join_timeout(text_len, max_chunk_chars)` (`app_generation.py`) scales the inference-thread join with configured chunk size for BOTH `/generate-stream` and `/ws` — never a constant, or a raised `max_chunk_chars`/slow join releases `inference_lock` mid-generation. Pinned by `tests/test_streaming_thread_lifecycle.py::TestWsStreamJoinTimeout` (#263: join timeout is clamped to a ceiling so an unbounded chunk-size × length product can't make it effectively infinite).
- **Request-boundary bounds (#263):** `max_chunk_chars` is validated to 0–10000 in `server/validation.py` (the request schema), not trusted from config — a config value outside the range can't reach `_prepare_text_chunks`.

## Inference serialization (#192 / #214) — see architecture.md for the full picture
`server/prompt_loading.py` — `load_voice_prompt_serialized(state, prompt_file)`: fast path is an unlocked disk load; only when torch must BUILD the prompt does it re-enter under `inference_lock` as a leaf, with the clone model built OUTSIDE the lock and forwarded via `clone_model=` so the locked section is create-inference only. `/unload-asr` (`app.py`) now also acquires `inference_lock` for the unload itself, closing a race where an unload could interleave with in-flight ASR generate. `/unload-model` (`app_models.py`) now holds `inference_lock` for the unload itself too, closing the queued-generation window (#214 item 4, closes #214).

`server/model_loading.py` (new, #214 item 3) — per-load-type CAS records under `MODEL_LOAD_LOCK`: `claim_model_load` gives the first caller `OWNER` and any concurrent duplicate caller `ATTACH` (awaits the owner's `done` Event, `MODEL_LOAD_WAIT_TIMEOUT_SEC=870`, retryable 503 on timeout) instead of reissuing the load. `load_model_deduped` is the owner body; `release_model_load` runs in `finally`. `state.model_config_epoch` bumps on `/update-model-config`/`/unload-model` so a stale-epoch waiter never attaches to a now-irrelevant load.

**Under-lock model rebind (0B #269):** `_require_model_under_lock(state, mode)` (`app_generation.py`) re-reads `state.models[mode]` UNDER `inference_lock`; a `None` slot raises 503 `model_unloaded`/`retry`. Every generation path MUST rebind its local (`model = _require_model_under_lock(...)`) — five guarded capture paths: batch `/generate`, `/generate-stream`, `/ws`, the torch `/create-voice-prompt` branch, and `prompt_loading.load_voice_prompt_serialized` (provenance split: slot-captured → 503 on unload, locally-built fallback → kept). An `/unload-model` landing in the capture→acquire window otherwise answers 200 against an orphaned local and re-opens the #233 double-allocation through a new route.

**Echo-trim ASR preload (#193 / 1C #284):** `_ensure_asr_for_echo_trim(...)` (`app_generation.py`) ensure-loads ASR UNLOCKED (`asyncio.to_thread`) BEFORE a clone generation queues for `inference_lock`, on BOTH `/generate` and `/generate-stream`; gated on the probe's own three conditions (`trim_icl_echo` default-true, `mode=clone`, transcript resolvable via `_reference_text_from_prompt`); no failure latch — one warning, generation proceeds. The in-lock `is_asr_loaded()` check in `_trim_icl_echo` is now the documented DEGRADATION path (an `/unload-asr` landed in the preload window → ship untrimmed, never rebuild in-lock — `_transcribe_mlx` would lazily reload). Engine side: `_trim_icl_echo`/`_postprocess_chunk` gained `trim_cap_samples` — the 50% cut cap scales to the FIRST chunk on the multi-chunk batch path (WS9.4: `run_inference` passes `len(all_audio[0])`), never half the combined signal.

## Key files (LOC)
app_generation.py 1163 · app.py 1133 · app_lifespan.py 821 · websocket.py 653 · client/generator.py 620 · app_models.py 589 · app_prompts.py 561 · validation.py 505 (33 Pydantic models) · model_loading.py 541 · generation_state_guard.py 371 (new, 0C #270) · vllm_client.py 332 · prompt_loading.py 73
