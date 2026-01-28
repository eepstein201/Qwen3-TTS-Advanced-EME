# Qwen3-TTS Project

This directory contains Eric's custom Qwen3-TTS setup for voice cloning and text-to-speech generation.

## Quick Reference

### Commands
- `changeVoice` - Main TTS command (prompts to start server if not running)
- `startTTSServer` - Manually start the persistent model server
- `stopTTSServer` - Stop the server
- `createVoice` - Create a new voice clone from audio
- `ttsUI` - Launch Gradio web interface (http://localhost:7860)

### Usage Examples
```bash
# Basic usage (will prompt about server)
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

### Files in this directory
- `install.sh` - Automated installation script for fresh setups
- `tts_generate.py` - Main generation script with SDPA optimization, inference_mode, batch support
- `tts_server.py` - Flask server that keeps models in memory (~95% faster)
- `tts_client.py` - Python API client library
- `tts_ui.py` - Gradio web interface (Clone/Design/Custom tabs)
- `config.json` - Settings: server config, generation params, presets
- `create_custom_voice.py` - Script to create voice clone prompts from audio
- `voice_prompts/` - Directory containing .pt voice clone files

### Wrapper scripts in ~/bin/
- `changeVoice` - Wrapper that handles server detection and user prompts
- `startTTSServer` - Starts server, waits for ready
- `stopTTSServer` - Graceful shutdown
- `createVoice` - Wrapper for voice creation
- `ttsUI` - Launch Gradio web interface

## Technical Details

### Conda Environment
- Name: `qwen3-tts`
- Location: `~/miniforge3/envs/qwen3-tts`
- Key packages: qwen-tts, torch, flask, soundfile

### Models (cached in ~/.cache/huggingface/hub/)
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` - For voice cloning from audio samples
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` - For voice description mode
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` - For 9 premium pre-trained speakers

### Server
- Runs on `localhost:5123`
- PID file: `.tts_server.pid`
- Log file: `.tts_server.log`

### Optimizations Applied
- SDPA attention (`attn_implementation="sdpa"`)
- `torch.inference_mode()` for faster inference
- Voice prompt caching (LRU cache in server)
- Generation parameters exposed (temperature, top_k, top_p, seed, repetition_penalty)

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
  "generation": { "temperature": 0.7, "top_k": 50, "top_p": 0.95, "repetition_penalty": 1.05, "seed": null },
  "presets": {
    "consistent": { "temperature": 0.5, "top_k": 30, "seed": 42 },
    "creative": { "temperature": 0.9, "top_p": 0.98 }
  }
}
```

## Implementation Roadmap

### Phase 1: Quick Wins ✅ COMPLETE
- [x] Health endpoint - Already exists at `/health`
- [x] `--play` flag - Immediate audio playback using `afplay`
- [x] `--clipboard` input - Read text from clipboard via `pbpaste`
- [x] `--trim-silence` - Auto-trim leading/trailing silence
- [x] `--dry-run` mode - Show what would be generated without inference

### Phase 2: Workflow Improvements ✅ COMPLETE
- [x] Voice prompt management - `--delete-prompt`, `--rename-prompt`, `--preview-prompt`
- [x] Favorites/aliases - `--voice` flag with config aliases, `--list-aliases`
- [x] History log - `--history [N]` shows recent generations from `~/.tts_history.jsonl`
- [x] GPU memory stats - `/stats` endpoint + `--stats` CLI flag

### Phase 3: Server Enhancements ✅ COMPLETE
- [x] Auto-shutdown - `auto_shutdown_minutes` in config (0 = disabled)
- [x] Queue system - `threaded=True` with generation lock for safe concurrency

### Phase 4: Audio Processing ✅ COMPLETE
- [x] Audio normalization - `--normalize` flag for -3dB peak normalization
- [x] Speed adjustment - `--speed FACTOR` (1.2 = 20% faster, 0.8 = 20% slower)
- [x] Pitch adjustment - `--pitch SEMITONES` (+2 = higher, -2 = lower)
- [x] Multi-speaker concatenation - `--dialogue FILE` for multi-speaker dialogues

### Phase 5: Integration Features ✅ COMPLETE
- [x] Interactive REPL mode - `--repl` for rapid iteration with commands
- [x] Watch mode - `--watch DIR` monitors folder for `.txt` files
- [x] Subtitle/SRT support - `--srt FILE` generates audio for each subtitle
- [x] API client library - `tts_client.py` as importable Python module

### Phase 6: Advanced ✅ COMPLETE
- [x] SSML support - `--ssml` flag parses `<break>`, `<emphasis>`, `<sub>`, `<say-as>`, `<prosody>`

### Phase 7: CustomVoice Integration ✅ COMPLETE
- [x] Premium speakers - 9 pre-trained voices (Ryan, Aiden, Vivian, Serena, etc.)
- [x] `-m custom -s SPEAKER` - Select premium speaker by name
- [x] `-i INSTRUCT` - Style instructions (e.g., "speak with enthusiasm")
- [x] `--list-speakers` - Show available premium speakers
- [x] Server support - All three models loaded in tts_server.py
- [x] Python API - tts_client.py updated with speaker/instruct params

### Phase 8: Configurable Model Loading ✅ COMPLETE
- [x] Config-driven model loading - `models` section in config.json
- [x] `--list-models` - Show models, load status, and memory usage
- [x] On-demand loading - Prompt user to load required model if not loaded
- [x] `/models` endpoint - Server API to check model status
- [x] `/load-model` endpoint - Server API to load models dynamically
- [x] Memory optimization - Load only needed models (~3.5GB each)

### Phase 9: Installation & Web UI ✅ COMPLETE
- [x] `install.sh` - Automated installation script for fresh setups
  - Checks prerequisites (macOS, Apple Silicon, conda, disk space)
  - Creates conda environment with all dependencies
  - Creates directories and config.json
  - Creates wrapper scripts in ~/bin/
  - Optional model pre-download
  - Supports `--dry-run` for preview
- [x] `tts_ui.py` - Gradio web interface
  - Three tabs: Clone Mode, Design Mode, Custom Mode
  - Server status display (connection, memory, loaded models)
  - All generation parameters exposed
  - Audio processing options (trim, normalize, speed, pitch)
  - Built-in audio player
- [x] `ttsUI` command - Launch web UI from terminal
