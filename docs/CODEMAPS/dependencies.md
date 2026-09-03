<!-- Generated: 2026-09-03 | Token estimate: ~450 -->

# Dependencies — Qwen3-TTS

## Models (HuggingFace)
3 distinct models — **Clone**, **Design**, **Custom** (~3.5 GB torch / ~2.5 GB MLX 8-bit each). Revision-pinned via `models.<type>.revision` (default `"main"`). ASR model loaded on-demand (`/load-asr`); its unload now serializes on `inference_lock` too (#214 item 2).

## Backends
- **torch** → `qwen_tts` (env `qwen3-tts`, transformers 4.57)
- **mlx** → `mlx_audio>=0.5.0` (env `qwen3-tts-mlx`, Apple Silicon, transformers 5.0rc) — bumped from >=0.4.7 (#222; evaluated 0.4.8→0.5.1, GO — `docs/reviews/mlx-audio-0.5.1-evaluation-2026-09-01.md`)
- **vllm** → `core/engine_vllm.py` + `server/vllm_client.py` (CUDA)

Note: FA2 NaN risk (upstream #333) → default SDPA. A `transformers<5` cap re-blocks the gradio floor (same knot).

## Audio
`pyrubberband` (primary, needs `rubberband` binary) + `librosa>=0.11.0` (fallback; now also in the `test` extra — `ensure_min_sample_rate()` raises rather than silently writing a below-native-rate reference, so tests need it installed to exercise the guarantee rather than skip) · `soundfile` · EBU R128 LUFS normalization (optional).

## Server / UI
`fastapi>=0.141.1` · `starlette>=1.6.0,<2` (explicit floor, #167 — not a bare transitive) · `uvicorn[standard]>=0.52.1` · `slowapi>=0.1.10` · `gradio>=6.0.0,!=6.14.*,<7` · Click (CLI)

Gradio floor is capped in practice: `>=6.15` needs `huggingface-hub>=1.2`, which `transformers<4.58` (torch env) forbids. Do not raise it.

## Process management
PM2 — `ecosystem.config.cjs`, service `tts-server-5123` (conda: `qwen3-tts-mlx`), port 5123. CLI/UI auto-detect PM2 supervision and delegate `tts server start/stop/restart` + UI stop-button to `pm2 start/stop/restart` (#248; prevents autorestart from undoing intentional stops).

## Python
3.10+. Editable install (`pyproject.toml`). `requirements.lock` pins test+ui+dev (standalone envs only — never install into the platform conda envs). `mypy` type-checking exclusion list shrank: `server/app.py` is back in scope (its mypy annotation debt was cleared, #176) — only `vllm_client.py`/`engine_vllm.py` remain excluded (optional vLLM-Omni backend, looser typing).
