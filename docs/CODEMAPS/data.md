<!-- Generated: 2026-09-02 | Token estimate: ~460 -->

# Data & Storage — Qwen3-TTS

No database. Persistence = config JSON + filesystem.

**Load/save contract** (`core/config/io.py`): `save_config` is atomic (temp file + `os.replace`). `load_config` never silently falls back to defaults — a non-dict or unusable `config.json` raises `ValueError` naming the path and the `tts config` reset (WS3, #157), so real user settings can't be quietly ignored.

## config.json (canonical schema)
- **advanced**: `backend` (mlx/torch/vllm), `model_size` (1.7B/0.6B), `mlx_quantization` (4–8bit/bf16), `torch_quantization` (none/8bit/4bit), `audio_loader` (torchaudio/librosa), `attn_implementation` (auto = SDPA)
- **generation**: `max_chunk_chars` (500), `lufs_normalize` (false), `lufs_target` (−16), `silence_gap_seconds` (0.0 = crossfade), `clone_speed` (0.5–2.0, PRF-6), `trim_icl_echo` (true, PRF-8), `language` (default `"auto"`)
- **models.{clone,design,custom}.revision** — HF pin (default `"main"`); `load_at_startup` (clone: true; design/custom: false, on-demand by design)
- **security.rate_limits** — `generate` (10/min), `model_ops` (5/min), `transcribe` (10/min), `config_ops` (2/min), `global` (120/min, decoupled from `generate`)
- **history_output_directory** — `~/Downloads/Qwen3-TTS Output`

## File storage
- **voice_prompts/** — `.pt` (torch) + `.wav`/`.txt` (mlx) dual format. MLX pair creation is now inference-free (`save_voice_prompt_mlx`, #236) — a direct write, no clone gate, no lock; torch still builds via clone inference
- **Output** — `~/Downloads/Qwen3-TTS Output/{Automated Output` (generations; Remove = hard-delete), `Manual Downloads` (kept files)`}`
- **Runtime** — `.voice_server.pid`, `.voice_server.log`, `.voice_server.lock` (startup-race exclusive flock)
- **Auth token** — `~/.config/qwen3-tts/.voice_server_token` (legacy `~/.voice_server_token`); written atomically (temp file + fsync + `os.replace`)

## Caches
- Generation cache (server-side; keyed minus seed — seed stored on the entry and echoed on hits). Also caches waveform peaks (`calculate_waveform_peaks`, 500 points) computed before the entry is stored, so history playback doesn't recompute.
- HuggingFace cache (managed via `tts cache {list,size,prune,clear}`)
