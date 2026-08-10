<!-- Generated: 2026-08-10 | Files scanned: 71 .py (25k LOC) | Token estimate: ~450 -->

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
- **core/** — `config/` (io, models, runtime, pid, presets, paths, auth, errors) + `engine/` (text_processing, audio_processing, voice_prompt, model_loader, inference, asr) + `http_client` (single server chokepoint) + `protocols`
- **server/** — FastAPI :5123. `app.py` (routes + middleware) → `app_generation` / `app_models` / `app_prompts` (handlers) + `app_lifespan` + `websocket` + `validation` + `client/` (TTSClient)
- **interface/** — `cli.py` (Click groups) + `generate*.py` (CLI gen) + `cli/` (batch, srt, dialogue) + `ui/` (Gradio)

## Generation flow
`text → _prepare_text_chunks (≤max_chunk_chars) → backend.generate → audio → post-proc (ASR echo-trim, phase-align splice, rate stretch, LUFS norm) → output file + history`

## Principles
- Lazy imports everywhere (no torch/mlx at module scope)
- 2 conda envs (qwen3-tts torch / qwen3-tts-mlx) — transformers version conflict
- 3 distinct HF models (Clone / Design / Custom)

## Heaviest modules (LOC)
inference.py 1532 · generate.py 902 · app.py 864 · ui/shared.py 803 · app_generation.py 788
_(inference.py >800-line guideline — known structural debt)_
