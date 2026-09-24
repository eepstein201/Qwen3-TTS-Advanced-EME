<!-- Generated: 2026-09-24 | Files scanned: 2 .py (0.8k LOC) | Token estimate: ~850 -->

# Dependencies — Qwen3-TTS

## Models (HuggingFace)
3 distinct models — **Clone**, **Design**, **Custom** (~3.5 GB torch / ~2.5 GB MLX 8-bit each). IDs from `MODEL_INFO` (`core/config/models.py`):
- torch: `Qwen/Qwen3-TTS-12Hz-{1.7B,0.6B}-{Base,VoiceDesign,CustomVoice}` (Base = clone)
- mlx: `mlx-community/Qwen3-TTS-12Hz-{1.7B,0.6B}-{Base,VoiceDesign,CustomVoice}-{quant}`
- ASR (`core/engine/asr.py`): torch `openai/whisper-base`; MLX `mlx-community/whisper-large-v3-turbo` (processor files from `openai/whisper-large-v3-turbo`)
- vLLM processor default: `Qwen/Qwen2-Audio-7B-Instruct`
Revision-pinned via `models.<type>.revision` (default `"main"`). ASR model loaded on-demand (`/load-asr`); its unload now serializes on `inference_lock` too (#214 item 2).

## Backends
- **torch** extra → `torch>=2.13.0`, `torchaudio>=2.11.0`, `qwen-tts>=0.1.1` (imports as `qwen_tts`), `transformers>=4.57.3` (env `qwen3-tts`)
- **mlx** extra → `mlx>=0.32.2`, `mlx-audio>=0.5.4`, `mlx-lm>=0.31.3`, `huggingface_hub>=0.36.2` (env `qwen3-tts-mlx`, Apple Silicon)
- **vllm** extra → `vllm>=0.8`, `vllm-omni>=0.14.0`; code `core/engine_vllm.py` + `server/vllm_client.py` (CUDA)
- **cuda** extra → `accelerate>=1.12.0`, `bitsandbytes>=0.43.1`

Note: FA2 NaN risk (upstream #333) → default SDPA. A `transformers<5` cap re-blocks the gradio floor (same knot).

## Audio
`audio` extra: `pyrubberband>=0.4.0` (primary, needs `rubberband` binary) + `librosa>=0.11.0` (fallback; now also in the `test` extra — `ensure_min_sample_rate()` raises rather than silently writing a below-native-rate reference, so tests need it installed to exercise the guarantee rather than skip) · `soundfile>=0.14.0` · `pyloudnorm>=0.2.0` (EBU R128 LUFS) · `pydub>=0.25.1` (shells out to ffmpeg).

## Server / UI
`server` extra: `fastapi>=0.141.1` · `starlette>=1.6.0,<2` · `uvicorn[standard]>=0.53.0` · `slowapi>=0.1.10` · `soundfile>=0.14.0` · `psutil>=7.2.2`. `ui` extra: `gradio>=6.0.0,!=6.14.*,<7`. Base deps: `click>=8.5.0`, `pySBD>=0.3.4`, `num2words>=0.5.14`, `requests>=2.34.2`. Other extras: `rich>=15.0.0`, `prompt-enhancer` (`anthropic>=1.7.0`), `dev` (pytest, ruff>=0.16.8, mypy>=2.3.1, bandit>=1.9.4, …), `test`.

Gradio floor is capped in practice: `>=6.15` needs `huggingface-hub>=1.2`, which `transformers<4.58` (torch env) forbids. Do not raise it.

## External services
HuggingFace Hub (model + ASR downloads) · Anthropic API (optional prompt enhancer, key from env named by `prompt_enhancer.api_key_env`) · Gradio share tunnel (`tts ui --share`, auth enforced)

## Process management
PM2 — `ecosystem.config.cjs`, service `tts-server-5123` (conda: `qwen3-tts-mlx`), port 5123. CLI/UI auto-detect PM2 supervision and delegate `tts server start/stop/restart` + UI stop-button to `pm2 start/stop/restart` (#248; prevents autorestart from undoing intentional stops).

## Python
3.10+. Editable install (`pyproject.toml`). `requirements.lock` pins test+ui+dev+audio (standalone envs only — never install into the platform conda envs). `mypy` type-checking exclusion list shrank: `server/app.py` is back in scope (its mypy annotation debt was cleared, #176) — only `vllm_client.py`/`engine_vllm.py` remain excluded (optional vLLM-Omni backend, looser typing).
