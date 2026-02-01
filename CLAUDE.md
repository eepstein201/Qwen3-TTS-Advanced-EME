# Qwen3-TTS Project

This directory contains Eric's custom Qwen3-TTS setup for voice cloning and text-to-speech generation.

## Quick Reference

### Commands
- `changeVoice` - Main TTS command (prompts to start server if not running, post-generation menu)
- `startTTSServer` - Manually start the persistent model server
- `stopTTSServer` - Stop the server (uses auth token for graceful shutdown)
- `createVoice` - Create a new voice clone from audio
- `ttsUI` - Launch Gradio web interface (http://localhost:7860)

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
| `tts_config.py` | Constants, config helpers, error classes, `CUSTOM_VOICE_SPEAKERS`, `MODEL_INFO`, `TOKEN_FILE`, auth helpers | No |
| `tts_engine.py` | `load_model()`, `run_inference()`, `create_voice_prompt()`, LRU cache, audio processing, backend dispatch | No (lazy imports per backend) |
| `tts_server.py` | Flask server with auth, validation, progress tracking, structured errors, backend-aware `/prompts` | No (lazy via tts_engine) |
| `tts_client.py` | HTTP client library for server API (`get_health()`, `load_model()`, `generate()`) | No (lazy tts_engine for audio only) |
| `tts_generate.py` | CLI generation with progress display, post-gen menu support | No (lazy tts_engine for local mode) |
| `tts_ui.py` | Gradio web interface with auto-load, progress bars, stop server button | No (HTTP only) |
| `create_custom_voice.py` | Voice clone prompt creation from audio files, saves dual-format (.pt + .wav/.txt) | Yes (via tts_engine) |
| `requirements-mlx.txt` | MLX backend dependencies (separate conda env) | N/A |

### Files in this directory
- `install.sh` - Automated installation script (torch + optional MLX envs, wrapper scripts)
- `tts_generate.py` - CLI generation with `--backend` override, batch support, SSML, SRT, dialogue
- `tts_server.py` - Flask server with auth, validation, logging, progress tracking, backend-aware `/prompts`
- `tts_client.py` - Python API client library (`get_health()`, `load_model()`, `generate()`)
- `tts_ui.py` - Gradio web interface (Clone/Design/Custom tabs, auto-load models, backend status indicator)
- `tts_config.py` - Shared constants, config helpers, error classes, backend helpers (no torch/mlx)
- `tts_engine.py` - Backend dispatch engine: lazy-loads torch or mlx per config
- `config.json` - Settings: server, generation params, presets, security, backend selection
- `create_custom_voice.py` - Voice clone prompt creation (saves .pt + .wav/.txt dual format)
- `requirements-mlx.txt` - MLX backend pip dependencies (for `qwen3-tts-mlx` env)
- `voice_prompts/` - Voice clone files (.pt for torch, .wav/.txt for MLX)
- `bin/` - Wrapper scripts (canonical source, copied to ~/bin/ by install.sh)
- `tests/` - Test suite: 66 tests (`python -m unittest discover -v tests/`)

### Wrapper scripts in ~/bin/ (installed from bin/)
- `changeVoice` - Server detection, generation, post-generation menu; auto-selects conda env by backend
- `startTTSServer` - Starts server; activates correct conda env, sets `PYTORCH_ENABLE_MPS_FALLBACK=1` for torch
- `stopTTSServer` - Graceful shutdown with auth token support (no conda env needed)
- `createVoice` - Wrapper for voice creation (always uses `qwen3-tts` torch env)
- `ttsUI` - Launch Gradio web interface; auto-selects conda env by backend

**IMPORTANT:** The wrapper scripts in `~/bin/` are copies of the canonical scripts in `bin/`. After updating the repo (e.g., adding backend support, new flags), you must re-copy them:
```bash
cp bin/* ~/bin/ && chmod +x ~/bin/*
```
Or re-run `./install.sh`. Stale wrapper scripts are a common source of bugs (e.g., old scripts always activating the `qwen3-tts` torch env even when config says `"backend": "mlx"`).

## Security

### API Token Authentication
- Server generates a 32-byte hex token on startup, written to `~/.tts_server_token` (0o600 perms)
- All endpoints except `/health` and `/generation-status` require `Authorization: Bearer <token>`
- `tts_client.py` and `tts_generate.py` read the token automatically via `tts_config.auth_headers()`
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

### Temp File Security
- Temp files created with `0o600` permissions

## Logging

Structured logging replaces `print()` throughout:
- `tts` - server logger (RotatingFileHandler: 5MB, 1 backup + stderr)
- `tts.engine` - model/inference logger
- `tts.cli` - CLI generation logger
- `tts.ui` - Gradio UI logger

Server log file: `.tts_server.log`

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

- Server tracks `generation_state` (active, start_time, text_length, mode)
- `/generation-status` endpoint (public, no auth) for polling
- ETA estimated from `~/.tts_history.jsonl` median chars/sec
- CLI: background thread with spinner (`Generating... 12s elapsed`)
- Gradio: `gr.Progress()` with threaded polling, capped at 95%
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

**PyTorch models:**
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` - Voice cloning from audio samples (~3.5GB)
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` - Voice description mode (~3.5GB)
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` - 9 premium pre-trained speakers (~3.5GB)

**MLX models (quantized, smaller):**
- `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-{quant}` - Voice cloning (~2.5GB)
- `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-{quant}` - Voice description (~2.5GB)
- `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-{quant}` - Premium speakers (~2.5GB)

Where `{quant}` is `4bit`, `8bit` (default), or `bf16`.

### Server
- Runs on `localhost:5123`
- PID file: `.tts_server.pid`
- Log file: `.tts_server.log`
- Auth token: `~/.tts_server_token`

### Optimizations Applied
- **Torch backend:** SDPA attention (`attn_implementation="sdpa"`), `torch.inference_mode()`, MPS-safe multinomial patch
- **MLX backend:** 8-bit/4-bit quantized models, native Apple Silicon Neural Engine
- **Both:** Voice prompt caching (LRU for torch), lazy imports (no unnecessary library loading)
- Generation parameters exposed (temperature, top_k, top_p, seed, repetition_penalty)

## Testing

Run the test suite (no GPU, models, or running server required):
```bash
python -m unittest discover -v tests/
```

66 tests across 12 test classes (2 skipped when MLX not installed):
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
- `TestLazyImports` - Verify `tts_engine` does not import torch/mlx at module scope

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
  "generation": { "temperature": 0.7, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.05, "seed": null },
  "presets": {
    "consistent": { "temperature": 0.5, "top_k": 30, "seed": 42 },
    "creative": { "temperature": 0.9, "top_p": 0.98 }
  },
  "advanced": {
    "dtype": "bfloat16",
    "backend": "torch",
    "mlx_quantization": "8bit"
  }
}
```

### Backend Configuration (`advanced` section)

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `backend` | `"torch"`, `"mlx"` | `"torch"` | Inference backend. Wrapper scripts activate the correct conda env. |
| `dtype` | `"float32"`, `"float16"`, `"bfloat16"` | `"float32"` | PyTorch dtype (torch backend only). |
| `mlx_quantization` | `"4bit"`, `"8bit"`, `"bf16"` | `"8bit"` | MLX model quantization level (mlx backend only). |

**CLI override:** `changeVoice --backend mlx "text" -o output` sets `TTS_BACKEND` env var for that run without modifying config.json.

## MLX Backend

### Overview

MLX is an alternative inference backend for Apple Silicon Macs that runs natively on the Neural Engine and GPU. Benefits over PyTorch/MPS:
- Lower thermal output (~40-50°C vs ~80-90°C)
- Less battery drain
- Uses 8-bit quantized models (~2-3GB vs ~5-7GB per model)

### Architecture

```
config.json  →  tts_config.py  →  tts_engine.py (dispatch)
                                    ├── backend="torch" → lazy import torch/qwen_tts
                                    │   Models: Qwen/Qwen3-TTS-12Hz-1.7B-*
                                    │   Env: qwen3-tts
                                    └── backend="mlx"   → lazy import mlx_audio
                                        Models: mlx-community/Qwen3-TTS-12Hz-1.7B-*-{quant}
                                        Env: qwen3-tts-mlx
```

All backend-specific imports are lazy (local to `_*_torch()` or `_*_mlx()` functions). Neither `torch` nor `mlx` is imported at module scope in `tts_engine.py` or `tts_server.py`.

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
- [x] Architecture split - `tts_config.py` (no torch) + `tts_engine.py` (torch)
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
- [x] Backend-aware test suite (66 tests, MLX tests skipped when mlx not installed)
- [x] Documentation - CLAUDE.md, README.md updated

### Phase 12: UI Auto-Load & Backend-Aware Prompts ✅ COMPLETE
- [x] Backend-aware `/prompts` endpoint - lists `.pt` for torch, `.wav`+`.txt` pairs for MLX
- [x] `TTSClient.get_health()` - check loaded models and backend via `/health`
- [x] `TTSClient.load_model(mode)` - on-demand model loading via `/load-model` (120s timeout)
- [x] Gradio auto-load - `_ensure_model_loaded()` checks `/health`, loads model before generation
- [x] Status bar refresh - updates after each generation to reflect newly loaded models
- [x] Voice prompt dropdown default - uses `get_default_clone_prompt()` from config
- [x] Updated footer tips - MLX format, auto-load behavior, backend switching
