<!-- Generated: 2026-09-24 | Files scanned: 5 .py (1.8k LOC) | Token estimate: ~1250 -->

# Data & Storage — Qwen3-TTS

No database. Persistence = config JSON + filesystem.

**Load/save contract** (`core/config/io.py`): `save_config` is atomic (temp file + `os.replace`). `load_config` never silently falls back to defaults — a non-dict or unusable `config.json` raises `ValueError` naming the path and the `tts config` reset (WS3, #157), so real user settings can't be quietly ignored.

## config.json (canonical schema)
- **top-level**: `default_voice_description`, `default_clone_prompt` (null), `default_speaker` (`"ryan"`), `output_directory` (`~/Downloads`, CLI), `history_output_directory`, `language` (`"auto"` — top-level, not under `generation`)
- **server**: `host` 127.0.0.1, `port` 5123, `auto_shutdown_minutes` 0 · **ui**: `port` 7860 · **cache**: `voice_prompt_max` 10, `generation_max` 5, `eta_ttl_seconds` 30 · **vllm**: 10 keys (`enabled` false, …) · **aliases**: `{}` · **prompt_enhancer**: `enabled` false, `provider`, `api_key_env`, `model`
- **advanced**: `backend` (mlx/torch/vllm), `model_size` (1.7B/0.6B), `mlx_quantization` (4–8bit/bf16), `torch_quantization` (none/8bit/4bit), `audio_loader` (torchaudio/librosa), `attn_implementation` (auto = SDPA), `dtype` (`"bfloat16"`), `vllm_enabled` (false), `vllm_fallback_to_torch` (true)
- **generation**: `temperature` 0.7, `top_k` 50, `top_p` 0.95, `repetition_penalty` 1.05, `seed` null, `max_new_tokens` 2048, `max_chunk_tokens` 200, `compile_model` true, `max_chunk_chars` (500; bounded 0–10000 at the request boundary, #263), `lufs_normalize` (false), `lufs_target` (−16), `silence_gap_seconds` (0.0 = crossfade), `clone_speed` (0.5–2.0, PRF-6) and `trim_icl_echo` (true, PRF-8) — these two are engine `.get()` fallbacks, NOT keys of `get_default_config()` (`trim_icl_echo`: since #193/1C the server force-loads ASR unlocked before the lock and keeps it loaded — an in-lock miss ships untrimmed)
- **models.{clone,design,custom}.revision** — HF pin (default `"main"`); `load_at_startup` (clone: true; design/custom: false, on-demand by design)
- **security**: `max_text_length` 50000, `max_batch_size` 20. **security.rate_limits** is not in `get_default_config()`: `validate_config()` adds `generate` (10/min), `model_ops` (5/min), `transcribe` (10/min), `prompt_ops` (10/min), `config_ops` (2/min) when missing; `global` (120/min, decoupled from `generate`) is supplied only by `server/app.py` at import
- **presets** — merged over 8 built-in generation presets (`DEFAULT_GENERATION_PRESETS`); defaults ship user presets `consistent` + `creative`; UI save/delete of user entries (T1b #327)
- **prosody_presets** — 8 factory presets + user-defined entries (UI save/delete, T1 #322): writers compose off the RAW dict (factory keys + junk entries survive every save); factory names reserved case-insensitively; a hand-edited factory-keyed override still applies via the merged view but is not manageable in the UI
- **history_output_directory** — `~/Downloads/Qwen3-TTS Output`

## File storage
User files root `USER_FILES_DIR` = `~/Qwen3-TTS_UserFiles` (`core/config/paths.py`); `config.json` resolves there first, else repo-root `config.json`.
- **voice_prompts/** — `.pt` (torch) + `.wav`/`.txt` (mlx) dual format. MLX pair creation is now inference-free (`save_voice_prompt_mlx`, #236) — a direct write, no clone gate, no lock; torch still builds via clone inference
- **Output** — `~/Downloads/Qwen3-TTS Output/{Automated Output` (generations; Remove = hard-delete), `Manual Downloads` (kept files)`}` — this root (+ system tempdir) is also the UI's narrowed `allowed_paths` surface (0E #276)
- **Runtime** (under `USER_FILES_DIR`) — `.voice_server.pid`, `.voice_server.log`, `.voice_server.lock` (startup-race exclusive flock); `voice_prompts/` also lives there
- **Generation history** — `~/.voice_history.jsonl` (`HISTORY_FILE`; appended by `interface/generate_helpers.py`, read by `tts history` and `server/app_lifespan.py`)
- **Auth token** — `~/.config/qwen3-tts/.voice_server_token` (legacy `~/.voice_server_token`); written atomically (temp file + fsync + `os.replace`)
- **UI share credentials** — `~/.config/qwen3-tts/.ui_share_credentials`, 0600, sole line = the generated share password (truncated per launch; username + path printed once, password never logged; 0E #276)

## Caches
- Generation cache (server-side; keyed minus seed — seed stored on the entry and echoed on hits). Also caches waveform peaks (`calculate_waveform_peaks`, 500 points) computed before the entry is stored, so history playback doesn't recompute.
- HuggingFace cache (managed via `tts cache {list,size,prune,clear}`)
- **In-memory `generation_state`** — 11 idle keys since 1A (#283, adds `cancel_target_id`); every read/write goes through `GenerationStateGuard`'s `threading.Lock` (0C #270), never raw dict access
