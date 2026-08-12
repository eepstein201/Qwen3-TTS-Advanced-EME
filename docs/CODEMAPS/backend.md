<!-- Generated: 2026-08-12 | Files scanned: server/ (5.8k LOC) | Token estimate: ~520 -->

# Backend — FastAPI Server (:5123)

Base `http://127.0.0.1:5123`. Bearer-token auth on all endpoints except the public set.

## Routes (grouped by handler module)
- **Generation** — `app_generation.py`, `websocket.py`: POST `/generate`, POST `/generate-stream`, WS `/ws`, POST `/cancel-generation`
- **Models** — `app_models.py`: GET `/models`; POST `/load-model`, `/unload-model`, `/update-model-config`, `/update-startup-config`; POST `/load-asr`, `/unload-asr`, `/transcribe`
- **Prompts** — `app_prompts.py`: GET `/prompts`, `/preview-prompt`, `/prompt-details`; POST `/create-voice-prompt`, `/delete-prompt`, `/rename-prompt`
- **System** — `app.py`, `app_lifespan.py`: GET `/health`, `/ready`, `/generation-status`, `/queue-status`, `/stats`; POST `/shutdown`

**Public (no auth):** `/health` `/ready` `/generation-status` `/queue-status` (`/generate` + `/health` register via `app.add_api_route` — not decorators, easy to miss in greps.)

## Middleware & Security (PR #153)
- **Bearer auth** — token `~/.config/qwen3-tts/.voice_server_token`
- **CORS** — configurable allowlist
- **Rate limiting** — slowapi, env-tunable per limit; `TTS_DISABLE_RATE_LIMITING=1` disables all (tests only)
- **IP resolution** — `X-Forwarded-For` honored ONLY when direct peer is in `TTS_TRUSTED_PROXIES` (comma-separated IPs; loopback by default)
- **Body-size DoS** — `RequestBodySizeLimitMiddleware` (~100 MB); rejects oversized bodies without buffering, counters off ASGI stream
- **WS validation** — `/ws` validates `Origin` header against CORS allowlist (CSWSH defense)
- **Error sanitization** — `/health` redacts filesystem paths to `<path>` (CWE-209)
- **starlette** pinned `>=1.6.0,<2` (#167) — was an unpinned fastapi transitive the lock drifted to 1.3.1. Custom body-size middleware is retained over native `max_body_size` (pre-auth no-buffer ordering + Content-Length fast path).

## Client
`server/client/` — `TTSClient` (generator / models / voices / config_fetcher / _base). CLI & UI call the server through this. Surfaces `last_seed`, `last_chunk_count`.

## Generation path
`/generate → cache check → run_inference (chunk + backend.generate + _postprocess_chunk) → {chunks, seed}`.
Stream path returns length-prefixed float32 chunks (`[sr:4][len:4][payload]`); `/ws` is bidirectional with cancel + disconnect detection.

## Streaming failure semantics (WS2, #160)
- Headers commit before the body iterates → no mid-stream status code. Server emits a terminal frame with `sample_rate == 0` (`STREAM_ERROR_SENTINEL_SR`, `app_generation.py:99`) carrying JSON `{"error","code"}`; the CLI (`interface/generate_server.py:33`, constant deliberately duplicated so the CLI never imports FastAPI) raises `TTSGenericError` instead of decoding it as float32. Lockstep guarded by `tests/test_stream_error_frame.py`.
- `/ws` closes with RFC 6455 `1011` after its error message.
- `_stream_thread_join_timeout(text_len, max_chunk_chars)` (`app_generation.py:69`) scales the inference-thread join with configured chunk size — never a constant, or a raised `max_chunk_chars` releases `inference_lock` mid-generation.

## Key files (LOC)
app.py 963 · app_generation.py 858 · client/generator.py 624 · websocket.py 556 · app_models.py 551 · app_lifespan.py 545 · app_prompts.py 432 · vllm_client.py 332 · validation.py 298 (`_validate_generation_request`, `_VALID_SPEAKER_NAMES`)
