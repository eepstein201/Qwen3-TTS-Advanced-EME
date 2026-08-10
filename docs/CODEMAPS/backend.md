<!-- Generated: 2026-08-10 | Files scanned: server/ (4.3k LOC) | Token estimate: ~450 -->

# Backend — FastAPI Server (:5123)

Base `http://127.0.0.1:5123`. Bearer-token auth on all endpoints except the public set.

## Routes (grouped by handler module)
- **Generation** — `app_generation.py`, `websocket.py`: POST `/generate`, POST `/generate-stream`, WS `/ws`, POST `/cancel-generation`
- **Models** — `app_models.py`: GET `/models`; POST `/load-model`, `/unload-model`, `/update-model-config`, `/update-startup-config`; POST `/load-asr`, `/unload-asr`, `/transcribe`
- **Prompts** — `app_prompts.py`: GET `/prompts`, `/preview-prompt`, `/prompt-details`; POST `/create-voice-prompt`, `/delete-prompt`, `/rename-prompt`
- **System** — `app.py`, `app_lifespan.py`: GET `/health`, `/ready`, `/generation-status`, `/queue-status`, `/stats`; POST `/shutdown`

**Public (no auth):** `/health` `/ready` `/generation-status` `/queue-status`
(`/generate` + `/health` register via `app.add_api_route` — app.py:389,679 — not decorators, easy to miss in greps.)

## Middleware
Bearer auth (token `~/.config/qwen3-tts/.voice_server_token`) · CORS · slowapi rate-limit (env-tunable; `TTS_DISABLE_RATE_LIMITING=1` for tests) · body-size · `X-Forwarded-For` honored only behind `TTS_TRUSTED_PROXIES`

## Client
`server/client/` — `TTSClient` (generator / models / voices / config_fetcher / _base). CLI & UI call the server through this. Surfaces `last_seed`, `last_chunk_count`.

## Generation path
`/generate → cache check → run_inference (chunk + backend.generate + post-proc) → {chunks, seed}`.
Stream path returns length-prefixed float32 chunks; `/ws` is bidirectional with cancel + disconnect detection.

## Key files (LOC)
app.py 864 · app_generation.py 788 · app_models.py 551 · app_lifespan.py 545 · websocket.py 502 · app_prompts.py 432 · validation.py 298 (`_validate_generation_request`, `_VALID_SPEAKER_NAMES`)
