<!-- Generated: 2026-09-24 | Files scanned: 79 .py (31.3k LOC) | Token estimate: ~1400 -->

# Architecture — Qwen3-TTS

Multilingual TTS with voice cloning. Three modes: **clone** (from audio), **design** (from description), **custom** (9 speakers). Runs on Mac (MLX/torch), Linux, Colab (CUDA).

## Dispatch
```
config.json → core.config → core.engine (dispatch on advanced.backend)
                              ├── "torch" → qwen_tts        (lazy import)
                              ├── "mlx"   → mlx_audio        (lazy import)
                              └── "vllm"  → no engine strategy (inference.py raises ValueError);
                                           /generate uses server VLLMAdapter (engine_vllm + server/vllm_client)
                                           only when vllm.enabled=true (default false → 400 at validation.py)
```

## Entry points
`pyproject [project.scripts]`: `tts` → `qwen3_tts.cli:cli` (Click `TTSGroup`) · `test`/`test-unit`/`test-integration`/`test-quick`/`test-parallel`/`test-cov` → `qwen3_tts.test_cli` · server app `qwen3_tts/server/app.py` (`app = FastAPI(...)`) · UI `qwen3_tts/interface/ui/_facade.py` (`build_ui()`, `main()`)

## Layers
- **core/** — `config/` (io, models, runtime, pid, presets, paths, auth, errors, pm2) + `engine/` (text_processing, audio_processing, voice_prompt, model_loader, inference, asr) + `http_client` (single server chokepoint) + `stream_protocol` (wire-format: sentinel, cap, encode/decode/parse — shared by server AND CLI, no FastAPI/torch/mlx)
- **server/** — FastAPI :5123. `app.py` (routes + middleware) → `app_generation` / `app_models` / `app_prompts` (handlers) + `app_lifespan` + `websocket` + `validation` + `generation_state_guard` (threading.Lock guard + attributed-cancel state, 0C/1A) + `prompt_loading` (torch auto-create-from-.wav serialization) + `model_loading` (per-load CAS records, dedups concurrent `/load-model`) + `client/` (TTSClient)
- **package root** — `cli.py` (Click root + `_FLAG_MAP` argv delegation) + `cli_server.py` / `cli_voice.py` / `cli_config.py` (groups) + `cli_output.py` / `cli_tables.py` (shared CLI formatting + list renderers, #347)
- **interface/** — `generate*.py` (CLI gen) + `cli/` (batch, srt, dialogue) + `ui/` (Gradio; `theme.py` design tokens → `UI_CSS`)

`core/protocols.py` removed (#179) — zero-caller dead module, grep-proven.

## Generation flow
`text → _prepare_text_chunks (≤max_chunk_chars, bounded 0–10000 at the request boundary in validation.py, #263) → backend.generate → _postprocess_chunk → combine (phase-align crossfade) → LUFS norm → output file + history`

**Per-chunk seeds (3A #304):** `seed_lock_chunks=true` locks ONE seed across all chunks of a generation; default false derives a fresh seed per chunk (`_resolve_generation_seed` per segment; 5-layer propagation CLI→request→engine).

**Unified pipeline (WS2, #160):** `engine/inference.py::_postprocess_chunk` (echo-trim → clone speed → audio validation) is called by BOTH `run_inference` and `run_inference_streaming`, both backends — streaming output matches batch. LUFS is deliberately outside it (EBU R128 gates over the whole signal), so batch-only. Since #193/1C the echo trim's 50% cap is anchored to the FIRST chunk (`trim_cap_samples=len(all_audio[0])`, WS9.4) — never half the combined signal.

**Attributed cancellation (1A #283):** `/cancel-generation` targets a generation id (`cancel_target_id`), never a bare flag — a cancel arriving before a batch's first `begin()` is latched onto the pending id and honored by item 1 instead of bouncing. `generation_state` is read/written only through `GenerationStateGuard` (threading.Lock; 0C #270) — see backend.md.

**Streaming wire format (WS2/#229):** ONE parser lives in `core/stream_protocol.py` — sentinel `sample_rate==0`, error-frame encode/decode, `iter_stream_chunks`. Previously implemented twice and drifted (only the CLI checked the sentinel; `TTSClient` decoded the JSON error payload as float32). Guarded by `tests/test_stream_protocol.py` + `tests/test_stream_error_frame.py`.

## Inference serialization (#192 / #214)
Every GPU-inference-reachable path serializes on `state.inference_lock`. `/generate`, `/generate-stream`, `/ws` hold it outermost; these acquire it as a **leaf** (never while waiting on anything else): design warm-up, `/transcribe`, torch `/create-voice-prompt`, torch auto-create-from-`.wav` (`server/prompt_loading.py`), `/unload-asr`, `/unload-model` (since #214), `/update-model-config` (T5). MLX `/create-voice-prompt` is inference-free and unlocked (#236). Pre-lock work: on-demand model load (`load_model_deduped`, #300) and echo-trim ASR preload (`_ensure_asr_for_echo_trim`, 1C #284). Symbols: backend.md "Concurrency helpers". `TTS_SKIP_WARMUP=1` skips warm-up.

## Principles
- Lazy imports everywhere (no torch/mlx at module scope)
- 2 conda envs (qwen3-tts torch / qwen3-tts-mlx) — transformers version conflict
- 3 distinct HF models (Clone / Design / Custom)

## Heaviest modules (LOC)
inference.py 1849 · app_generation.py 1243 · app.py 1223 · ui/shared.py 1102 · ui/tabs_generation.py 1017 · generate.py 891 · generate_interactive.py 885 · app_lifespan.py 870 · ui/generation.py 869
_(all nine exceed the 800-line guideline — known structural debt)_

## Layer size (tracked .py, `wc -l`)
core/ 7.9k · server/ 8.3k · interface/ 10.3k (ui/ 6.2k) · tools/ 2.5k · package root 2.2k · tests/ 215 `test_*.py` modules, ~4.1k `def test_` functions
