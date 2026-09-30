<!-- Generated: 2026-09-24 | Files scanned: 19 .py (8.4k LOC) | Token estimate: ~1500 -->

# Backend — FastAPI Server (:5123)

Base `http://127.0.0.1:5123`. Bearer-token auth on all endpoints except the public set.

## Routes (24, all registered in `app.py`; grouped by handler module)
Rate-limit column: `_rate_limit(...)` decorator (all hybrid IP+token); every HTTP route also sits under the 120/min global pre-auth ceiling — `/ws` does not (`SlowAPIMiddleware` subclasses Starlette's `BaseHTTPMiddleware`, which passes non-`http` ASGI scopes straight through untouched); `/ws` is bounded instead by its own per-IP `_ws_try_acquire` connection cap.
- **Inline in `app.py`** — GET `/health`, `/ready`, `/generation-status`, `/queue-status` (public); POST `/cancel-generation`, `/shutdown`
- **`app_generation.py`** — POST `/generate` → `handle_generate(request, state, req, security, config_provider)` [generate]; POST `/generate-stream` → `handle_generate_stream(...)` [generate]
- **`websocket.py`** — WS `/ws` → `websocket_tts_handler(websocket, app_state, verify_token_fn, config_provider=None)` (no decorator; auth = first message)
- **`app_models.py`** — GET `/stats` → `handle_stats`, GET `/models` → `handle_list_models`; POST `/load-model`, `/unload-model`, `/update-model-config`, `/load-asr`, `/unload-asr` [model_ops]; POST `/update-startup-config` [config_ops]; POST `/transcribe` [transcribe]
- **`app_prompts.py`** — GET `/prompts`, `/preview-prompt`, `/prompt-details`; POST `/create-voice-prompt` (backend-dispatched — MLX is inference-free, #236), `/delete-prompt`, `/rename-prompt` [prompt_ops on the three POSTs]

**Public (no auth):** `/health` `/ready` `/generation-status` `/queue-status` — no `Depends(verify_auth)`; every other HTTP route declares it. `/generation-status` adds `batch_total`/`chunk_total` (+ `progress_pct`/`eta_sec` while active and known) only when `request_is_authed(request)` — helpers `chunk_progress_pct` / `estimate_eta_sec` in `app_lifespan.py`; this is the web UI's live-progress source.

19 routes carry a Pydantic `response_model=` (18 distinct model classes in `server/validation.py`, 33 models total — `ModelOpResponse` covers both `/load-model` and `/unload-model`). Untyped: binary `/generate-stream`, `/ws`, `/preview-prompt`, `/shutdown`, plus `/cancel-generation` (plain dict: `status`, `generation_id` only when a generation was actually targeted). Guarded by `tests/test_response_contracts.py`.

## Middleware & Security (`app.py`)
- **Order (outer → inner):** `SlowAPIMiddleware` (global IP-keyed ceiling) → `security_headers` (nosniff, `X-Frame-Options: DENY`, Referrer-Policy) → `RequestBodySizeLimitMiddleware` (~100 MB, no buffering, `Content-Length` fast path) → `CORSMiddleware` → ExceptionMiddleware → routing → `Depends(verify_auth)` → per-route `_rate_limit`
- **Auth** — `verify_auth(request)` → 401 + audit log (`missing_token`/`invalid_token`); `_strip_bearer_scheme` (one case-insensitive scheme token, #281); `_tokens_equal` = `secrets.compare_digest` on UTF-8 bytes. Token file `~/.config/qwen3-tts/.voice_server_token`, written atomically by `_write_auth_token` (`app_lifespan.py`)
- **CORS** — `allow_origin_regex` localhost/127.0.0.1 any port (+ `*.gradio.live` in Colab); GET/POST; headers Authorization, Content-Type. `/ws` checks `Origin` against the same regex before `accept()` (absent Origin allowed)
- **Rate limits** — 4 slowapi `Limiter`s (`limiter_global`, `_hybrid`, `_ip`, `_token`); env overrides `TTS_RATE_LIMIT_{GENERATE,MODEL_OPS,TRANSCRIBE,PROMPT_OPS,CONFIG_OPS,GLOBAL}`, kill-switch `TTS_DISABLE_RATE_LIMITING=1`; `X-Forwarded-For` honored only from `TTS_TRUSTED_PROXIES` (loopback default). Details: `docs/rate-limiting.md`
- **Startup lock** — `_acquire_startup_lock()` (`app_lifespan.py`): non-blocking `flock` on `.voice_server.lock` before anything else in `lifespan()`
- **Error sanitization** — `_sanitize_error(msg)` (`app_lifespan.py`) redacts absolute paths to `<path>` (CWE-209) in `model_load_errors` (surfaced by `/health`) and error details
- **PM2 (#248)** — `core/config/pm2.py`: `pm2_owner_of_port(port)`, `pm2_registered_app(port)`; used by `cli_server.py` start/stop/restart and the UI stop button (`_facade.py`)

## Client
`server/client/` — `TTSClient` (`_base`, `generator`, `models`, `voices`, `config_fetcher`); surfaces `last_seed`, `last_chunk_count`. `_generation_timeout(text_len)` (`client/generator.py`) scales `/generate`; `LOAD_MODEL_`/`TRANSCRIBE_`/`CREATE_PROMPT_`/`UNLOAD_ASR_`/`UNLOAD_MODEL_TIMEOUT_SEC` = 900 live in `core/http_client.py`.

## Generation path
```
/generate → cache check → empty slot? load_model_deduped (pre-lock, #300)
          → inference_lock → _require_model_under_lock (rebind) → run_inference → {chunks, seed}
/generate-stream → same, frames [sr:4][len:4][payload]; sr==0 = terminal error frame (core/stream_protocol.py)
/ws → websocket_tts_handler; no auto-load ("Model '…' not loaded"); classified errors never close()
```

## Concurrency helpers (key symbols)
- `generation_state_guard.py` — `GenerationStateGuard` (threading.Lock over the 11-key `generation_state`), `guard_for(app_state)`; cancel = `set_cancelled(target_id)` / `peek_pending_target()` (1A #283)
- `model_loading.py` — `claim_model_load(state, model_type)` → OWNER/ATTACH, `load_model_deduped(state, model_type, request=None)`, `release_model_load(...)`; `MODEL_LOAD_WAIT_TIMEOUT_SEC=870`
- `app_generation.py` — `_require_model_under_lock(state, mode)`, `_ensure_asr_for_echo_trim(...)`, `_stream_thread_join_timeout(...)`, `_should_stop_streaming(stop_event, guard, gen_id)`
- `websocket.py` — `_ws_try_acquire(app_state, client_ip)` / `_ws_release(...)` (one handler-level `finally`)
- `prompt_loading.py` — `load_voice_prompt_serialized(state, prompt_file)` (torch auto-create under `inference_lock`)
Full rationale: CLAUDE.md + `docs/00-Foundations/ARCHITECTURE.md` ("Inference Serialization").

## Key files (LOC)
app_generation.py 1243 · app.py 1223 · app_lifespan.py 870 · websocket.py 656 · client/generator.py 620 · app_models.py 600 · app_prompts.py 558 · validation.py 558 (33 Pydantic models) · model_loading.py 536 · generation_state_guard.py 371 · vllm_client.py 335 · prompt_loading.py 73
