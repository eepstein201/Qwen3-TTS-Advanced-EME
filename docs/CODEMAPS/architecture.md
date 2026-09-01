<!-- Generated: 2026-09-01 | Files scanned: 72 .py (26.6k LOC) | Token estimate: ~520 -->

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
- **core/** — `config/` (io, models, runtime, pid, presets, paths, auth, errors) + `engine/` (text_processing, audio_processing, voice_prompt, model_loader, inference, asr) + `http_client` (single server chokepoint) + `stream_protocol` (wire-format: sentinel, cap, encode/decode/parse — shared by server AND CLI, no FastAPI/torch/mlx)
- **server/** — FastAPI :5123. `app.py` (routes + middleware) → `app_generation` / `app_models` / `app_prompts` (handlers) + `app_lifespan` + `websocket` + `validation` + `prompt_loading` (torch auto-create-from-.wav serialization) + `client/` (TTSClient)
- **interface/** — `cli.py` (Click groups) + `generate*.py` (CLI gen) + `cli/` (batch, srt, dialogue) + `ui/` (Gradio)

`core/protocols.py` removed (#179) — zero-caller dead module, grep-proven.

## Generation flow
`text → _prepare_text_chunks (≤max_chunk_chars) → backend.generate → _postprocess_chunk → combine (phase-align crossfade) → LUFS norm → output file + history`

**Unified pipeline (WS2, #160):** `engine/inference.py::_postprocess_chunk` (echo-trim → clone speed → audio validation) is called by BOTH `run_inference` and `run_inference_streaming`, both backends — streaming output matches batch. LUFS is deliberately outside it (EBU R128 gates over the whole signal), so batch-only.

**Streaming wire format (WS2/#229):** ONE parser lives in `core/stream_protocol.py` — sentinel `sample_rate==0`, error-frame encode/decode, `iter_stream_chunks`. Previously implemented twice and drifted (only the CLI checked the sentinel; `TTSClient` decoded the JSON error payload as float32). Guarded by `tests/test_stream_protocol.py` + `tests/test_stream_error_frame.py`.

## Inference serialization (#192 / #214)
Every GPU-inference-reachable path now serializes on `state.inference_lock`, acquired as a **leaf** (never held while waiting on something else), with `inference_lock`-outermost order preserved everywhere:
- `/generate`, `/ws` — outermost holders
- Model warm-up (design), `/transcribe` ASR generate, `/create-voice-prompt`, torch auto-create-from-`.wav` (`server/prompt_loading.py`), `/unload-asr` — all leaf-acquire
- Each has its own long HTTP client timeout (`LOAD_MODEL_TIMEOUT_SEC`/`TRANSCRIBE_TIMEOUT_SEC`/`CREATE_PROMPT_TIMEOUT_SEC`/`UNLOAD_ASR_TIMEOUT_SEC` = 900s) since a request can now queue behind another's inference
- `TTS_SKIP_WARMUP=1` skips warm-up entirely (ablation control)

## Principles
- Lazy imports everywhere (no torch/mlx at module scope)
- 2 conda envs (qwen3-tts torch / qwen3-tts-mlx) — transformers version conflict
- 3 distinct HF models (Clone / Design / Custom)

## Heaviest modules (LOC)
inference.py 1769 · app.py 1026 · app_generation.py 903 · generate.py 902 · ui/shared.py 879 · generate_interactive.py 781 · app_lifespan.py 732
_(inference.py, app.py, app_generation.py, app_lifespan.py exceed the 800-line guideline — known structural debt, see project memory `project_open_structural_debt.md`)_

## Layer size
core/ 7.2k · server/ 6.5k · interface/ui/ 5.1k · tools/ 2.2k · tests/ 179 modules, 3127 test functions
