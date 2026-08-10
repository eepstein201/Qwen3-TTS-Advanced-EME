<!-- Generated: 2026-08-10 | Token estimate: ~400 -->

# Dependencies — Qwen3-TTS

## Models (HuggingFace)
3 distinct models — **Clone**, **Design**, **Custom** (~3.5 GB torch / ~2.5 GB MLX 8-bit each). Revision-pinned via `models.<type>.revision` (default `"main"`). ASR model loaded on-demand (`/load-asr`).

## Backends
- **torch** → `qwen_tts` (env `qwen3-tts`, transformers 4.57)
- **mlx** → `mlx_audio` (env `qwen3-tts-mlx`, Apple Silicon, transformers 5.0rc)
- **vllm** → `core/engine_vllm.py` + `server/vllm_client.py` (CUDA)

Note: FA2 NaN risk (upstream #333) → default SDPA. A `transformers<5` cap re-blocks the gradio floor (same knot).

## Audio
`pyrubberband` (primary, needs `rubberband` binary) + `librosa` (fallback) · `soundfile` · EBU R128 LUFS normalization (optional).

## Server / UI
FastAPI + uvicorn · slowapi (rate-limit) · Gradio (pin `!=6.14.*`) · Click (CLI)

## Process management
PM2 — `ecosystem.config.cjs`, service `tts-server-5123` (conda: `qwen3-tts-mlx`), port 5123.

## Python
3.10+. Editable install (`pyproject.toml`). `requirements.lock` pins test+ui+dev (standalone envs only — never install into the platform conda envs).
