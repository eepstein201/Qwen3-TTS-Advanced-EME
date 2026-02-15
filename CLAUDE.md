# Qwen3-TTS Project

This directory contains Eric's custom Qwen3-TTS setup for voice cloning and text-to-speech generation.

## Quick Reference

### Commands
- `changeVoice` - Main TTS command (prompts to start server if not running, post-generation menu)
- `startTTSServer` - Manually start the persistent model server
- `stopTTSServer` - Stop the server (uses auth token for graceful shutdown)
- `createVoice` - Create a new voice clone from audio (auto-MLX-only when backend is MLX)
- `ttsUI` - Launch Gradio web interface (default http://localhost:7860, auto-fallback if busy)
- `configureTTS` - Reconfigure settings with hardware detection (backend, model size, quantization)

### Usage Examples
```bash
# Basic usage (no args = interactive prompt: CLI or Web UI)
changeVoice

# Launch web interface directly (auto-starts server)
changeVoice --ui

# Basic generation (will prompt about server)
changeVoice "Hello world" -o greeting

# With generation parameters
changeVoice "Text" --temperature 0.5 --seed 42 -o output

# Use preset
changeVoice "Text" --preset consistent -o output

# Batch processing
changeVoice "Text one" "Text two" -o ~/Downloads/

# Voice design mode (not clone)
changeVoice "Text" -m design -o output

# Premium speaker mode (CustomVoice)
changeVoice "Text" -m custom -s ryan -o output
changeVoice "Text" -m custom -s ryan -i "speak with enthusiasm" -o output

# List available voice prompts
changeVoice --list-prompts

# List presets
changeVoice --list-presets

# List premium speakers
changeVoice --list-speakers

# Multi-speaker dialogue
changeVoice --dialogue conversation.json -o output
```

## Architecture

### Module Split

The codebase uses a layered architecture to avoid loading heavy dependencies (torch, model weights) unless needed:

| Module | Purpose | Imports torch? |
|--------|---------|----------------|
| `voice_config.py` | Constants, config helpers, error classes, `CUSTOM_VOICE_SPEAKERS`, `MODEL_INFO`, `TOKEN_FILE`, auth helpers | No |
| `voice_engine.py` | `load_model()`, `run_inference()`, `create_voice_prompt()`, LRU cache, audio processing, backend dispatch, text chunking | No (lazy imports per backend) |
| `voice_server.py` | Flask server with auth, validation, progress tracking, structured errors, backend-aware `/prompts` | No (lazy via voice_engine) |
| `voice_client.py` | HTTP client library for server API (`get_health()`, `load_model()`, `generate()`) | No (lazy voice_engine for audio only) |
| `voice_generate.py` | CLI generation with progress display, post-gen menu support | No (lazy voice_engine for local mode) |
| `voice_ui.py` | Gradio web interface with auto-load, progress bars, stop server button | No (HTTP only) |
| `create_custom_voice.py` | Voice clone prompt creation from audio files, saves dual-format (.pt + .wav/.txt) | Yes (via voice_engine) |
| `requirements-mlx.txt` | MLX backend dependencies (separate conda env) | N/A |

### Files in this directory
- `install.sh` - Automated installation script (torch + optional MLX envs, wrapper scripts)
- `voice_generate.py` - CLI generation with `--backend` override, batch support, SSML, SRT, dialogue
- `voice_server.py` - Flask server with auth, validation, logging, progress tracking, backend-aware `/prompts`
- `voice_client.py` - Python API client library (`get_health()`, `load_model()`, `generate()`)
- `voice_ui.py` - Gradio web interface (Clone/Design/Custom tabs, auto-load models, backend status indicator)
- `voice_config.py` - Shared constants, config helpers, error classes, backend helpers (no torch/mlx)
- `voice_engine.py` - Backend dispatch engine: lazy-loads torch or mlx per config
- `config.json` - Settings: server, generation params, presets, security, backend selection
- `create_custom_voice.py` - Voice clone prompt creation (saves .pt + .wav/.txt dual format)
- `requirements-mlx.txt` - MLX backend pip dependencies (for `qwen3-tts-mlx` env)
- `voice_prompts/` - Voice clone files (.pt for torch, .wav/.txt for MLX)
- `bin/` - Wrapper scripts (canonical source, copied to ~/bin/ by install.sh)
- `tests/` - Test suite: 161 tests (`python -m unittest discover -v tests/`)

### Wrapper scripts in ~/bin/ (installed from bin/)
- `changeVoice` - Server detection, generation, post-generation menu; auto-selects conda env by backend
- `startTTSServer` - Starts server; activates correct conda env, sets `PYTORCH_ENABLE_MPS_FALLBACK=1` for torch
- `stopTTSServer` - Graceful shutdown with auth token support (no conda env needed)
- `createVoice` - Wrapper for voice creation; auto-MLX-only when backend is MLX (--force-torch for .pt)
- `ttsUI` - Launch Gradio web interface; auto-selects conda env by backend
- `configureTTS` - Reconfigure settings with hardware detection; calls `install.sh --reconfigure`

**IMPORTANT:** The wrapper scripts in `~/bin/` are copies of the canonical scripts in `bin/`. After updating the repo (e.g., adding backend support, new flags), you must re-copy them:
```bash
cp bin/* ~/bin/ && chmod +x ~/bin/*
```
Or re-run `./install.sh`. Stale wrapper scripts are a common source of bugs (e.g., old scripts always activating the `qwen3-tts` torch env even when config says `"backend": "mlx"`).

## Security

### API Token Authentication
- Server generates a 32-byte hex token on startup, written to `~/.voice_server_token` (0o600 perms)
- All endpoints except `/health` and `/generation-status` require `Authorization: Bearer <token>`
- `voice_client.py` and `voice_generate.py` read the token automatically via `voice_config.auth_headers()`
- `stopTTSServer` reads the token for authenticated graceful shutdown
- Token is cleaned up on server shutdown

### Input Validation
- `security.max_text_length` (default: 10,000 chars) - per-text limit
- `security.max_batch_size` (default: 20 texts) - batch limit
- `prompt_file` path traversal prevention (rejects `..` and `/`)
- `mode` validated against `["clone", "design", "custom"]`
- `speaker` validated against `CUSTOM_VOICE_SPEAKERS`

### Network Binding
- Server binds to `127.0.0.1` by default (localhost only)
- `--public` flag to bind to `0.0.0.0` (with warning)
- Gradio UI uses `server_name="127.0.0.1"` by default
- Gradio port: configurable via `ui.port` (default 7860), auto-fallback to next available port in range +0..+9 if busy

### Temp File Security
- Temp files created with `0o600` permissions

## Logging

Structured logging replaces `print()` throughout:
- `tts` - server logger (RotatingFileHandler: 5MB, 1 backup + stderr)
- `tts.engine` - model/inference logger
- `tts.cli` - CLI generation logger
- `tts.ui` - Gradio UI logger

Server log file: `.voice_server.log`

## Structured Error Responses

Server returns JSON errors with recovery hints:
```json
{
  "error": "Human-readable message",
  "detail": "Technical details",
  "recovery": "restart|config|retry|bug"
}
```

CLI and UI parse `recovery` to show actionable guidance.

## Progress & ETA

- Server tracks `generation_state` (active, start_time, text_length, mode, chunk_index, chunk_total)
- `/generation-status` endpoint (public, no auth) for polling — includes chunk progress for long texts
- ETA estimated from `~/.voice_history.jsonl` median chars/sec
- CLI: background thread with spinner (`Generating... 12s elapsed [chunk 2/5]`)
- Gradio: `gr.Progress()` with threaded polling, capped at 95% (shows "Generating chunk 2/5... 12s" for chunked texts)
- Gradio auto-load: `_ensure_model_loaded()` checks `/health` before generation, calls `client.load_model()` if needed (progress shows "Loading {mode} model (first use)...")

## Post-Generation Menu (CLI)

When using server mode (exit code 2), `changeVoice` shows:
1. Same settings (re-run with auto-incremented filename)
2. Edit text (opens `$EDITOR`, re-runs with `--text-override`)
3. New settings (fresh interactive mode)
4. Exit (prompt to stop server)

Output filenames auto-increment: `output.wav` -> `output_2.wav` -> `output_3.wav`

## Technical Details

### Conda Environments

| Environment | Backend | Key packages | transformers version |
|-------------|---------|-------------|---------------------|
| `qwen3-tts` | torch (default) | qwen-tts, torch, flask, soundfile, gradio | 4.57.3 |
| `qwen3-tts-mlx` | mlx | mlx-audio, mlx, mlx-lm, flask, gradio | 5.0.0rc3 |

The two environments cannot be merged due to a hard `transformers` version conflict between `qwen-tts` and `mlx-audio`. Wrapper scripts automatically activate the correct environment based on `config.json`.

### Models (cached in ~/.cache/huggingface/hub/)

**PyTorch models (1.7B default, ~3.5GB each):**
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` - Voice cloning from audio samples
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` - Voice description mode
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` - 9 premium pre-trained speakers

**PyTorch models (0.6B lightweight, ~2GB each):**
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base` - Voice cloning (faster, lower memory)
- `Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign` - Voice description
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` - Premium speakers

**MLX models (1.7B quantized, ~2.5GB each):**
- `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-{quant}` - Voice cloning
- `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-{quant}` - Voice description
- `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-{quant}` - Premium speakers

**MLX models (0.6B quantized, ~1.5GB each):**
- `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-{quant}` - Voice cloning
- `mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-{quant}` - Voice description
- `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-{quant}` - Premium speakers

Where `{quant}` is `4bit`, `8bit` (default), or `bf16`.

### Server
- Runs on `localhost:5123`
- PID file: `.voice_server.pid`
- Log file: `.voice_server.log`
- Auth token: `~/.voice_server_token`

### Optimizations Applied
- **Torch backend:** SDPA attention (`attn_implementation="sdpa"`), `torch.inference_mode()`, MPS-safe multinomial patch
- **MLX backend:** 8-bit/4-bit quantized models, native Apple Silicon Neural Engine
- **Both:** Voice prompt caching (LRU for torch), lazy imports (no unnecessary library loading)
- **Text chunking:** Long texts (>500 chars) auto-split at sentence boundaries, generated per-chunk, concatenated with 100ms silence gaps — prevents timeouts and improves progress visibility
- Generation parameters exposed (temperature, top_k, top_p, seed, repetition_penalty)

## Testing

Run the test suite (no GPU, models, or running server required):
```bash
python -m unittest discover -v tests/
```

161 tests across 39 test classes (2 skipped when MLX not installed):
- `TestTTSConfig` - error hierarchy, format helpers, auth token, model info, speakers
- `TestServerValidation` - text length, batch size, mode, speaker, path traversal
- `TestServerAuth` - public endpoints, auth enforcement, token validation
- `TestSSMLParsing` - SSML tag parsing
- `TestSRTParsing` - SRT subtitle parsing
- `TestAutoIncrementFilename` - filename auto-increment logic
- `TestBackendConfig` - `get_backend()`, `get_mlx_quantization()`, `get_mlx_model_name()`, env var override, MLX model info
- `TestMLXVoicePrompt` - MLX voice prompt loading (.wav/.txt), dispatch, .pt-only error
- `TestBackendDispatch` - `load_model()` and `run_inference()` dispatch to correct backend
- `TestMLXInferenceCloneValidation` - MLX clone mode input validation (no model needed)
- `TestMLXImport` - MLX library import checks (`unittest.skipIf` when mlx not installed)
- `TestLazyImports` - Verify `voice_engine` does not import torch/mlx at module scope
- `TestModelSize` - 0.6B model configuration, `get_model_size()`, model name resolution
- `TestStreaming` - streaming API exists, function signatures, torch fallback
- `TestStreamingServerEndpoint` - `/generate-stream` auth, validation
- `TestASR` - ASR functions, lazy loading, torch restriction, result types
- `TestStability` - retry delays, max_chunk_chars config
- `TestFloat32Guard` - float32 dtype guard for torch clone mode
- `TestMLXMetalRecovery` - exception handling for Metal crashes
- `TestTextChunking` - sentence splitting, boundary cases, content preservation
- `TestHealthEndpointInfo` - `/health` returns backend, model_size, model_loaded fields
- `TestGenerationStatus` - `/generation-status` endpoint behavior
- `TestLoadModelEndpoint` - `/load-model` auth, validation, valid types
- `TestMLXMemoryStats` - MLX memory stats in `/stats` endpoint
- `TestGenerationStateFields` - generation_state initial values and fields
- `TestCancelGenerationEndpoint` - cancel endpoint auth, behavior
- `TestUIHistoryFunctions` - history add/get, truncation, max size
- `TestUICancelFunction` - cancel returns tuple, clears audio
- `TestUITextInfo` - text info helper functions
- `TestStreamingEndpointStructure` - `/generate-stream` endpoint validation
- `TestGenerationFunctionsReturnHistory` - generation functions return history data
- `TestGenerateStreamIdCheck` - generation_id race condition fix
- `TestCheckGenerationCancelled` - _check_generation_cancelled helper
- `TestCreateVoiceBackendOverride` - createVoice forces TTS_BACKEND=torch
- `TestStreamingClientMethod` - streaming client method exists and signatures
- `TestUIModelSettings` - model settings UI apply/get functions
- `TestUIModelSettingsImports` - model settings imports (VALID_MODEL_SIZES, etc.)
- `TestUpdateModelConfigEndpoint` - `/update-model-config` auth, validation, model clearing
- `TestClientUpdateModelConfig` - client update_model_config method

## Config Structure (config.json)
```json
{
  "default_voice_description": "...",
  "default_clone_prompt": "my_voice.pt",
  "output_directory": "~/Downloads",
  "language": "English",
  "server": { "host": "127.0.0.1", "port": 5123, "auto_shutdown_minutes": 0 },
  "models": {
    "clone": { "load_at_startup": true },
    "design": { "load_at_startup": false },
    "custom": { "load_at_startup": false }
  },
  "security": {
    "max_text_length": 10000,
    "max_batch_size": 20
  },
  "generation": { "temperature": 0.7, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.05, "seed": null, "max_chunk_chars": 500 },
  "presets": {
    "consistent": { "temperature": 0.5, "top_k": 30, "seed": 42 },
    "creative": { "temperature": 0.9, "top_p": 0.98 }
  },
  "ui": { "port": 7860 },
  "advanced": {
    "dtype": "bfloat16",
    "backend": "torch",
    "mlx_quantization": "8bit",
    "model_size": "1.7B"
  }
}
```

### Generation Configuration (`generation` section)

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `temperature` | `0.0`–`2.0` | `0.7` | Sampling temperature. |
| `top_k` | `1`–`200` | `50` | Top-k sampling. |
| `top_p` | `0.0`–`1.0` | `0.95` | Top-p (nucleus) sampling. |
| `repetition_penalty` | `1.0`–`2.0` | `1.05` | Repetition penalty. |
| `seed` | integer or `null` | `null` | Random seed for reproducibility. |
| `max_chunk_chars` | `0`–`10000` | `500` | Max chars per chunk for long texts. `0` disables chunking. |

**CLI override:** `changeVoice --max-chunk-chars 800 "long text..." -o output` overrides for that run.

### UI Configuration (`ui` section)

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `port` | `1024`–`65535` | `7860` | Preferred Gradio UI port. If busy, auto-scans +1..+9. |

### Backend Configuration (`advanced` section)

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `backend` | `"torch"`, `"mlx"` | `"torch"` | Inference backend. Wrapper scripts activate the correct conda env. |
| `dtype` | `"float32"`, `"float16"`, `"bfloat16"` | `"float32"` | PyTorch dtype (torch backend only). |
| `mlx_quantization` | `"4bit"`, `"8bit"`, `"bf16"` | `"8bit"` | MLX model quantization level (mlx backend only). |
| `model_size` | `"1.7B"`, `"0.6B"` | `"1.7B"` | Model size. 0.6B is ~40% faster with lower memory. |

**CLI overrides:**
- `changeVoice --backend mlx "text" -o output` — sets `TTS_BACKEND` env var for that run
- `changeVoice --model-size 0.6B "text" -o output` — sets `TTS_MODEL_SIZE` env var for that run

## MLX Backend

### Overview

MLX is an alternative inference backend for Apple Silicon Macs that runs natively on the Neural Engine and GPU. Benefits over PyTorch/MPS:
- Lower thermal output (~40-50°C vs ~80-90°C)
- Less battery drain
- Uses 8-bit quantized models (~2-3GB vs ~5-7GB per model)

### Architecture

```
config.json  →  voice_config.py  →  voice_engine.py (dispatch)
                                    ├── backend="torch" → lazy import torch/qwen_tts
                                    │   Models: Qwen/Qwen3-TTS-12Hz-1.7B-*
                                    │   Env: qwen3-tts
                                    └── backend="mlx"   → lazy import mlx_audio
                                        Models: mlx-community/Qwen3-TTS-12Hz-1.7B-*-{quant}
                                        Env: qwen3-tts-mlx
```

All backend-specific imports are lazy (local to `_*_torch()` or `_*_mlx()` functions). Neither `torch` nor `mlx` is imported at module scope in `voice_engine.py` or `voice_server.py`.

### Switching Backends

1. Edit `config.json`: set `"advanced": {"backend": "mlx"}`
2. Restart the server: `stopTTSServer && startTTSServer`
3. Wrapper scripts automatically activate the correct conda environment

Or use the CLI override for a single run:
```bash
changeVoice --backend mlx "Hello world" -o test
```

### Voice Prompt Format

| | PyTorch | MLX |
|---|---|---|
| Voice clone input | `.pt` tensor file | `.wav` + `.txt` file pair |
| Created by | `model.create_voice_clone_prompt()` | Raw reference audio + transcript |

`createVoice` saves both formats automatically (.pt + .wav + .txt). For legacy `.pt`-only prompts, re-create them with `createVoice` using the original audio.

### Installing the MLX Backend

Run `install.sh` and select the MLX option, or manually:
```bash
conda create -n qwen3-tts-mlx python=3.11 -y
conda activate qwen3-tts-mlx
pip install -r requirements-mlx.txt
```

### MLX Troubleshooting

**"mlx-audio is not installed"**
- You're using the torch conda env with `backend: "mlx"`. Switch to the MLX env or run `install.sh` to create it.

**"only has a .pt file (torch format)"**
- The voice prompt was created before the MLX backend was added. Re-create it with `createVoice` — this saves both `.pt` (torch) and `.wav`/`.txt` (MLX) files.

**Missing `.wav`/`.txt` files for Clone mode**
- MLX clone mode needs a `.wav` reference audio and `.txt` transcript alongside the `.pt` file in `voice_prompts/`. Run `createVoice` with the original audio to regenerate all formats.

**MLX model download fails**
- Ensure you can access `mlx-community` repos on HuggingFace. Check quantization setting: `advanced.mlx_quantization` must be `"4bit"`, `"8bit"`, or `"bf16"`.

**Wrong conda env activated**
- Wrapper scripts read `advanced.backend` from `config.json` and activate the correct env. If you run Python directly, ensure you've activated the right env manually.

## Implementation Roadmap

### Phase 1-9: ✅ COMPLETE
See README.md for full phase history.

### Phase 10: Security, Reliability & UX ✅ COMPLETE
- [x] Architecture split - `voice_config.py` (no torch) + `voice_engine.py` (torch)
- [x] API token authentication - Bearer token on all endpoints
- [x] Input validation - text length, batch size, mode, speaker, path traversal
- [x] Network binding - localhost default, `--public` flag
- [x] Structured logging - RotatingFileHandler, logger hierarchy
- [x] Structured error responses - JSON with recovery hints
- [x] Progress & ETA display - server tracking, CLI spinner, Gradio progress bar
- [x] Stop Server button in Gradio UI
- [x] Post-generation menu in CLI (re-run, edit, new settings)
- [x] Auto-increment output filenames
- [x] Test suite (36 tests, no GPU required)
- [x] Wrapper scripts in repo (`bin/`) for reproducible installation

### Phase 11: MLX Backend Integration ✅ COMPLETE
- [x] Backend config helpers - `get_backend()`, `get_mlx_quantization()`, `MLX_MODEL_INFO`
- [x] Lazy import refactor - neither torch nor mlx imported at module scope
- [x] Backend dispatch - `load_model()`, `run_inference()`, `load_voice_prompt()` dispatch by backend
- [x] MLX inference - `_load_model_mlx()`, `_run_inference_mlx()` for all 3 modes
- [x] Separate conda environments - `qwen3-tts` (torch) + `qwen3-tts-mlx` (mlx)
- [x] Wrapper script env switching - auto-activate correct conda env
- [x] Dual-format voice prompts - `.pt` (torch) + `.wav`/`.txt` (MLX) saved together
- [x] UI/CLI integration - backend indicator in Gradio, `--backend` CLI flag, `/health` info
- [x] `install.sh` MLX option - optional MLX env creation + model download
- [x] Backend-aware test suite (110 tests, MLX tests skipped when mlx not installed)
- [x] Documentation - CLAUDE.md, README.md updated

### Phase 12: UI Auto-Load & Backend-Aware Prompts ✅ COMPLETE
- [x] Backend-aware `/prompts` endpoint - lists `.pt` for torch, `.wav`+`.txt` pairs for MLX
- [x] `TTSClient.get_health()` - check loaded models and backend via `/health`
- [x] `TTSClient.load_model(mode)` - on-demand model loading via `/load-model` (120s timeout)
- [x] Gradio auto-load - `_ensure_model_loaded()` checks `/health`, loads model before generation
- [x] Status bar refresh - updates after each generation to reflect newly loaded models
- [x] Voice prompt dropdown default - uses `get_default_clone_prompt()` from config
- [x] Updated footer tips - MLX format, auto-load behavior, backend switching
- [x] Dynamic port fallback - `_find_available_port()` scans +0..+9, configurable via `ui.port`
- [x] Generation timeout increased from 300s to 600s for long MLX texts

### Phase 13: Text Chunking & Long-Form Reliability ✅ COMPLETE
- [x] `_split_text()` — sentence-boundary text splitter with clause/word fallback
- [x] Chunked inference in `run_inference()` — auto-splits long texts, generates per-chunk, concatenates with 100ms silence gaps
- [x] `progress_callback` support — server tracks `chunk_index`/`chunk_total` in `generation_state`
- [x] CLI chunk progress — spinner shows `[chunk 2/5]` during multi-chunk generation
- [x] Gradio chunk progress — progress bar shows "Generating chunk 2/5..." during long texts
- [x] `generation.max_chunk_chars` config option (default: 500, 0 to disable)
- [x] `--max-chunk-chars` CLI flag for per-run override

### Phase 14: 0.6B Lightweight Model Support ✅ COMPLETE
- [x] `MODEL_INFO` and `MLX_MODEL_INFO` now nested by size (`1.7B`, `0.6B`)
- [x] `get_model_size()` config helper with `TTS_MODEL_SIZE` env var override
- [x] `get_torch_model_name()`, `get_mlx_model_name()` use configured size
- [x] `--model-size` CLI flag for per-run override
- [x] `/health` and `/models` endpoints return `model_size` field
- [x] Gradio UI shows model size in backend indicator: "MLX (8bit, 1.7B)"
- [x] `advanced.model_size` config option (default: `"1.7B"`, alternative: `"0.6B"`)

### Phase 15: Streaming Audio Playback ✅ COMPLETE
- [x] `_run_inference_mlx_streaming()` — yields audio chunks as MLX model generates
- [x] `run_inference_streaming()` — public API, MLX native streaming or torch chunked fallback
- [x] `/generate-stream` endpoint — streams raw float32 audio chunks
- [x] `--stream` CLI flag — plays audio chunks as they arrive via streaming endpoint
- [x] `generate_streaming()` client function — streams from server, plays, saves combined audio

### Phase 16: Auto-Transcribe Reference Audio (ASR) ✅ COMPLETE
- [x] `transcribe_audio()` — lazy-loads ASR model on first use (NOT at server startup)
- [x] `is_asr_available()` — checks MLX backend + mlx_audio.stt importable (no model load)
- [x] `--auto-transcribe` flag for `createVoice` — transcribes reference audio automatically
- [x] Interactive prompt when no transcript provided (MLX only): "Auto-transcribe with MLX ASR?"
- [x] Transcript confirmation before saving voice prompt

### Phase 17: Stability Hardening ✅ COMPLETE
- [x] Float32 guard for torch clone mode on MPS — auto-overrides dtype to float32 with warning
- [x] Model download retry with exponential backoff — 3 attempts (5s, 15s, 45s delays)
- [x] MLX Metal kernel crash recovery — catches Metal errors, retries with smaller sub-chunks

### Phase 18: UI History Integration & Improvements ✅ COMPLETE
- [x] History panel auto-update — generation functions return history data, UI wired to update panel
- [x] Cancel button clears audio — `cancel_streaming_generation()` returns `None` for audio to clear player
- [x] `_check_generation_cancelled()` helper — checks server for cancellation state
- [x] MLX memory stats in `/stats` — `mlx_memory_active_mb`, `mlx_memory_peak_mb` fields
- [x] `get_server_status()` prefers MLX memory — checks `mlx_memory_active_mb` before `mps_memory_allocated_mb`
- [x] Generation state race condition fix — `generate_stream` only resets state if `generation_id` matches
- [x] `createVoice` backend override — forces `TTS_BACKEND=torch` when running in torch env
- [x] Test suite expanded (base: 161 tests) — added tests for history, cancel, memory stats, generation state

### Phase 19: MLX-First Architecture ✅ COMPLETE
- [x] `install.sh` rewrite — MLX as primary backend, torch as optional fallback
- [x] Hardware detection — detects Apple Silicon vs Intel, RAM size
- [x] Configuration wizard — interactive prompts for backend, model size, quantization
- [x] Intel Mac guardrails — hard-locks to torch, no MLX options shown
- [x] `configureTTS` command — wrapper for `install.sh --reconfigure`
- [x] `voice_config.py` defaults changed — `get_backend()` returns "mlx" by default
- [x] `createVoice` auto-MLX-only — when backend is MLX, skips .pt creation (--force-torch to override)
- [x] `/update-model-config` server endpoint — switch model variants without restart
- [x] `TTSClient.update_model_config()` — client method for model settings
- [x] Gradio UI model selection — "Model Settings" accordion with dropdowns for model size and MLX quantization
- [x] Test suite expanded to 161 tests — added tests for model settings UI and endpoint
- [x] Documentation updates — README.md positions MLX as default, adds configureTTS, updates test count
- [x] Dynamic model download — handled by HuggingFace hub (auto-downloads on first use with retry)

### Phase 20: Cleanup & Stabilization ✅ COMPLETE
- [x] Fix critical bugs in server and UI (Phase 20a) — resolved startup and rendering issues
- [x] Rename `tts_*` files to `voice_*` prefix (Phase 20b) — consistent naming across all modules
- [x] Consolidate duplicate generation functions (Phase 20c) — removed redundant code paths
- [x] Improve model settings feedback message (Phase 20d) — clearer UI responses on config changes
- [x] Fix stale references and `MODEL_INFO` lookup bug (Phase 20e) — updated imports after rename
- [x] Handle `ImportError` in `/load-model` endpoint gracefully — returns structured error instead of 500
