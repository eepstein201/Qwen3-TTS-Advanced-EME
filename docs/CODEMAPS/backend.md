<!-- Generated: 2026-09-02 | Files scanned: server/ (7.3k LOC) | Token estimate: ~620 -->

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
- **CORS** — configurable allowlist
- **Rate limiting** — slowapi, env-tunable per limit (`TTS_RATE_LIMIT_{GENERATE,MODEL_OPS,TRANSCRIBE,PROMPT_OPS,CONFIG_OPS,GLOBAL}`); `TTS_DISABLE_RATE_LIMITING=1` disables all (tests only). Global pre-auth ceiling (`TTS_RATE_LIMIT_GLOBAL`, default 120/min) is decoupled from the 10/min `/generate` limit so UI polling doesn't 429
- **IP resolution** — `X-Forwarded-For` honored ONLY when direct peer is in `TTS_TRUSTED_PROXIES` (comma-separated IPs; loopback by default)
- **Body-size DoS** — `RequestBodySizeLimitMiddleware` (~100 MB); rejects oversized bodies without buffering, counters off ASGI stream; `Content-Length` fast path
- **WS validation** — `/ws` validates `Origin` header against CORS allowlist (CSWSH defense); absent Origin allowed (auth is per-message token)
- **Error sanitization** — `/health` redacts filesystem paths to `<path>` (CWE-209)
- **Startup lock** — `_acquire_startup_lock()` (`app_lifespan.py`) takes an exclusive non-blocking `flock` on `.voice_server.lock` before anything else in `lifespan()`, aborting a losing `tts server start`/`tts ui` process before it can clobber the winner's auth token
- **starlette** pinned `>=1.6.0,<2` — custom body-size middleware retained over native `max_body_size` (pre-auth no-buffer ordering + Content-Length fast path)

## Client
`server/client/` — `TTSClient` (generator / models / voices / config_fetcher / _base). CLI & UI call the server through this. Surfaces `last_seed`, `last_chunk_count`. Per-route timeouts scale for long-running ops: `_generation_timeout(len(text))` for `/generate`, `LOAD_MODEL_TIMEOUT_SEC`/`TRANSCRIBE_TIMEOUT_SEC`/`CREATE_PROMPT_TIMEOUT_SEC`/`UNLOAD_ASR_TIMEOUT_SEC` = 900s for the leaf-locked inference paths (#192/#214 — see architecture.md).

## Generation path
`/generate → cache check → run_inference (chunk + backend.generate + _postprocess_chunk) → {chunks, seed}`.
Stream path returns length-prefixed float32 chunks (`[sr:4][len:4][payload]`); `/ws` is bidirectional with cancel + disconnect detection, and marks `generation_state` active so `/generation-status` sees WS work too.

## Streaming failure semantics
- Headers commit before the body iterates → no mid-stream status code. Server emits a terminal frame with `sample_rate == 0` (`STREAM_ERROR_SENTINEL_SR`, now defined once in `core/stream_protocol.py`) carrying JSON `{"error","code"}`. `TTSClient.generate_streaming` and the CLI's `iter_stream_chunks` both parse it through the same module — previously duplicated and drifted (#229), fixed by consolidating to one parser + `tests/test_stream_protocol.py` anti-re-fork assertions.
- `/ws` closes with RFC 6455 `1011` after its error message.
- `_stream_thread_join_timeout(text_len, max_chunk_chars)` (`app_generation.py`) scales the inference-thread join with configured chunk size for BOTH `/generate-stream` and `/ws` — never a constant, or a raised `max_chunk_chars`/slow join releases `inference_lock` mid-generation. Pinned by `tests/test_streaming_thread_lifecycle.py::TestWsStreamJoinTimeout`.

## Inference serialization (#192 / #214) — see architecture.md for the full picture
`server/prompt_loading.py` — `load_voice_prompt_serialized(state, prompt_file)`: fast path is an unlocked disk load; only when torch must BUILD the prompt does it re-enter under `inference_lock` as a leaf, with the clone model built OUTSIDE the lock and forwarded via `clone_model=` so the locked section is create-inference only. `/unload-asr` (`app.py`) now also acquires `inference_lock` for the unload itself, closing a race where an unload could interleave with in-flight ASR generate. `/unload-model` (`app_models.py`) now holds `inference_lock` for the unload itself too, closing the queued-generation window (#214 item 4, closes #214).

`server/model_loading.py` (new, #214 item 3) — per-load-type CAS records under `MODEL_LOAD_LOCK`: `claim_model_load` gives the first caller `OWNER` and any concurrent duplicate caller `ATTACH` (awaits the owner's `done` Event, `MODEL_LOAD_WAIT_TIMEOUT_SEC=870`, retryable 503 on timeout) instead of reissuing the load. `load_model_deduped` is the owner body; `release_model_load` runs in `finally`. `state.model_config_epoch` bumps on `/update-model-config`/`/unload-model` so a stale-epoch waiter never attaches to a now-irrelevant load.

## Key files (LOC)
app.py 1064 · app_generation.py 997 · app_lifespan.py 805 · client/generator.py 620 · app_models.py 588 · websocket.py 620 · app_prompts.py 549 · validation.py 504 (32 Pydantic models) · model_loading.py 541 (new, #214 item 3) · vllm_client.py 332 · prompt_loading.py 46
