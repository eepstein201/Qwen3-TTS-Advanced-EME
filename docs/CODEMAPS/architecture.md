<!-- Generated: 2026-09-11 | Files scanned: 75 .py (28.7k LOC) | Token estimate: ~620 -->

# Architecture — Qwen3-TTS

Multilingual TTS with voice cloning. Three modes: **clone** (from audio), **design** (from description), **custom** (9 speakers). Runs on Mac (MLX/torch), Linux, Colab (CUDA).

## Dispatch
```
config.json → core.config → core.engine (dispatch on advanced.backend)
                              ├── "torch" → qwen_tts        (lazy import)
                              ├── "mlx"   → mlx_audio        (lazy import)
                              └── "vllm"  → engine_vllm + server/vllm_client
```

## Layers
- **core/** — `config/` (io, models, runtime, pid, presets, paths, auth, errors, pm2) + `engine/` (text_processing, audio_processing, voice_prompt, model_loader, inference, asr) + `http_client` (single server chokepoint) + `stream_protocol` (wire-format: sentinel, cap, encode/decode/parse — shared by server AND CLI, no FastAPI/torch/mlx)
- **server/** — FastAPI :5123. `app.py` (routes + middleware) → `app_generation` / `app_models` / `app_prompts` (handlers) + `app_lifespan` + `websocket` + `validation` + `generation_state_guard` (threading.Lock guard + attributed-cancel state, 0C/1A) + `prompt_loading` (torch auto-create-from-.wav serialization) + `model_loading` (per-load CAS records, dedups concurrent `/load-model`) + `client/` (TTSClient)
- **interface/** — `cli.py` (Click groups) + `generate*.py` (CLI gen) + `cli/` (batch, srt, dialogue) + `ui/` (Gradio)

`core/protocols.py` removed (#179) — zero-caller dead module, grep-proven.

## Generation flow
`text → _prepare_text_chunks (≤max_chunk_chars, bounded 0–10000 at the request boundary in validation.py, #263) → backend.generate → _postprocess_chunk → combine (phase-align crossfade) → LUFS norm → output file + history`

**Unified pipeline (WS2, #160):** `engine/inference.py::_postprocess_chunk` (echo-trim → clone speed → audio validation) is called by BOTH `run_inference` and `run_inference_streaming`, both backends — streaming output matches batch. LUFS is deliberately outside it (EBU R128 gates over the whole signal), so batch-only. Since #193/1C the echo trim's 50% cap is anchored to the FIRST chunk (`trim_cap_samples=len(all_audio[0])`, WS9.4) — never half the combined signal.

**Attributed cancellation (1A #283):** `/cancel-generation` targets a generation id (`cancel_target_id`), never a bare flag — a cancel arriving before a batch's first `begin()` is latched onto the pending id and honored by item 1 instead of bouncing. `generation_state` is read/written only through `GenerationStateGuard` (threading.Lock; 0C #270) — see backend.md.

**Streaming wire format (WS2/#229):** ONE parser lives in `core/stream_protocol.py` — sentinel `sample_rate==0`, error-frame encode/decode, `iter_stream_chunks`. Previously implemented twice and drifted (only the CLI checked the sentinel; `TTSClient` decoded the JSON error payload as float32). Guarded by `tests/test_stream_protocol.py` + `tests/test_stream_error_frame.py`.

## Inference serialization (#192 / #214)
Every GPU-inference-reachable path now serializes on `state.inference_lock`, acquired as a **leaf** (never held while waiting on something else), with `inference_lock`-outermost order preserved everywhere:
- `/generate`, `/ws` — outermost holders
- Model warm-up (design), `/transcribe` ASR generate, torch `/create-voice-prompt`, torch auto-create-from-`.wav` (`server/prompt_loading.py`), `/unload-asr` — all leaf-acquire
- **`/create-voice-prompt` is backend-dispatched (#236):** torch keeps the clone-gated, leaf-locked flow above; **MLX is inference-free** — `save_voice_prompt_mlx` writes the `.wav`+`.txt` pair directly with no clone gate and no lock, since there is no GPU work to serialize
- `/load-model` dedups concurrent callers for the same model type via `model_loading.py`'s per-load CAS records (`claim_model_load`/`release_model_load`) instead of a lock — a duplicate caller attaches to the owner's `done` Event rather than reissuing the load (#214 item 3)
- `/unload-model` closes the queued-generation window: it now holds `inference_lock` for the unload itself, so it can no longer interleave with an in-flight generation queued behind it (#214 item 4, closes #214)
- **Under-lock model rebind (0B #269):** every generation capture path re-reads the model slot under the lock via `_require_model_under_lock(state, mode)` and MUST rebind its local — an unload in the capture→acquire window yields a retryable 503, not inference on an orphaned object (5 guarded paths; see backend.md)
- **Echo-trim ASR preload (#193 / 1C #284):** the server ensure-loads ASR UNLOCKED (`_ensure_asr_for_echo_trim`, `asyncio.to_thread`) before a clone generation queues for the lock, gated on the trim probe's own three conditions, and KEEPS it loaded — a cold HF download must never run in-lock; an in-lock `is_asr_loaded()` miss means the unload won the race → ship untrimmed (cosmetic), never fail the generation
- Each long op has its own extended HTTP client timeout (`LOAD_MODEL_TIMEOUT_SEC`/`TRANSCRIBE_TIMEOUT_SEC`/`CREATE_PROMPT_TIMEOUT_SEC`/`UNLOAD_ASR_TIMEOUT_SEC` = 900s) since a request can now queue behind another's inference
- `TTS_SKIP_WARMUP=1` skips warm-up entirely (ablation control)

## Principles
- Lazy imports everywhere (no torch/mlx at module scope)
- 2 conda envs (qwen3-tts torch / qwen3-tts-mlx) — transformers version conflict
- 3 distinct HF models (Clone / Design / Custom)

## Heaviest modules (LOC)
inference.py 1785 · app_generation.py 1163 · app.py 1133 · generate.py 902 · ui/shared.py 982 · app_lifespan.py 821 · generate_interactive.py 780
_(inference.py, app_generation.py, app.py, app_lifespan.py, ui/shared.py exceed the 800-line guideline — known structural debt, see project memory `project_open_structural_debt.md`)_

## Layer size
core/ 7.5k · server/ 8.0k · interface/ 9.2k (ui/ 5.2k) · tools/ 2.2k · tests/ 188 modules, ~3.4k test functions
