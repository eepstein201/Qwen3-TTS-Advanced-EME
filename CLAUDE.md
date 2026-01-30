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
| `tts_engine.py` | `load_model()`, `run_inference()`, `create_voice_prompt()`, LRU cache, audio processing, MPS management | Yes (on import) |
| `tts_server.py` | Flask server with auth, validation, progress tracking, structured errors | Yes (via tts_engine) |
| `tts_client.py` | HTTP client library for server API | No (lazy tts_engine for audio only) |
| `tts_generate.py` | CLI generation with progress display, post-gen menu support | No (lazy tts_engine for local mode) |
| `tts_ui.py` | Gradio web interface with progress bars, stop server button | No (HTTP only) |
| `create_custom_voice.py` | Voice clone prompt creation from audio files | Yes (via tts_engine) |

### Files in this directory
- `install.sh` - Automated installation script (copies wrapper scripts from `bin/`)
- `tts_generate.py` - Main generation script with SDPA optimization, inference_mode, batch support
- `tts_server.py` - Flask server with auth, validation, logging, progress tracking
- `tts_client.py` - Python API client library
- `tts_ui.py` - Gradio web interface (Clone/Design/Custom tabs, stop server button)
- `tts_config.py` - Shared constants, config helpers, error classes (no torch)
- `tts_engine.py` - Model loading, inference, audio processing (torch required)
- `config.json` - Settings: server config, generation params, presets, security limits
- `create_custom_voice.py` - Script to create voice clone prompts from audio
- `voice_prompts/` - Directory containing .pt voice clone files
- `bin/` - Wrapper scripts (canonical source, copied to ~/bin/ by install.sh)
- `tests/` - Test suite (run with `python -m unittest discover -v tests/`)

### Wrapper scripts in ~/bin/ (installed from bin/)
- `changeVoice` - Server detection, generation, post-generation menu (re-run, edit, new settings)
- `startTTSServer` - Starts server with `PYTORCH_ENABLE_MPS_FALLBACK=1`, waits for ready
- `stopTTSServer` - Graceful shutdown with auth token support
- `createVoice` - Wrapper for voice creation
- `ttsUI` - Launch Gradio web interface

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

## Post-Generation Menu (CLI)

When using server mode (exit code 2), `changeVoice` shows:
1. Same settings (re-run with auto-incremented filename)
2. Edit text (opens `$EDITOR`, re-runs with `--text-override`)
3. New settings (fresh interactive mode)
4. Exit (prompt to stop server)

Output filenames auto-increment: `output.wav` -> `output_2.wav` -> `output_3.wav`

## Technical Details

### Conda Environment
- Name: `qwen3-tts`
- Location: `~/miniforge3/envs/qwen3-tts`
- Key packages: qwen-tts, torch, flask, soundfile, gradio

### Models (cached in ~/.cache/huggingface/hub/)
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` - For voice cloning from audio samples
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` - For voice description mode
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` - For 9 premium pre-trained speakers

### Server
- Runs on `localhost:5123`
- PID file: `.tts_server.pid`
- Log file: `.tts_server.log`
- Auth token: `~/.tts_server_token`

### Optimizations Applied
- SDPA attention (`attn_implementation="sdpa"`)
- `torch.inference_mode()` for faster inference
- Voice prompt caching (LRU cache in server)
- Generation parameters exposed (temperature, top_k, top_p, seed, repetition_penalty)

## Testing

Run the test suite (no GPU, models, or running server required):
```bash
python -m unittest discover -v tests/
```

36 tests across 6 test classes:
- `TestTTSConfig` - error hierarchy, format helpers, auth token, model info, speakers
- `TestServerValidation` - text length, batch size, mode, speaker, path traversal
- `TestServerAuth` - public endpoints, auth enforcement, token validation
- `TestSSMLParsing` - SSML tag parsing
- `TestSRTParsing` - SRT subtitle parsing
- `TestAutoIncrementFilename` - filename auto-increment logic

## Config Structure (config.json)
```json
{
  "default_voice_description": "...",
  "default_clone_prompt": "default_clone.pt",
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
  }
}
```

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
