# Qwen3-TTS Voice Generation System

Clone any voice from an audio sample, design voices from text descriptions, or choose from 9 premium speakers. Powered by Qwen3-TTS models with a persistent server for fast generation.

**Platforms:** Mac (Apple Silicon with MLX, Intel with PyTorch), Linux with NVIDIA GPU, Google Colab

## Install

```bash
cd ~/Qwen3-TTS_UserFiles
./install.sh
```

The installer detects your hardware, walks you through backend/model/quantization choices, creates the right conda environment, and installs commands to `~/bin/`.

```bash
configureTTS             # Re-run the setup wizard anytime
configureTTS --show      # Compare current settings to recommendations
```

## Quick Start

```bash
startTTSServer                          # Load model (~30-60s first time)
changeVoice "Hello, world!" -o hello    # Generate speech → hello.wav
stopTTSServer                           # Free memory when done
```

Or skip straight to the web UI:

```bash
ttsUI                    # Opens http://localhost:7860
changeVoice --ui         # Same thing, auto-starts server
```

## Three Voice Modes

### Clone (default) — sound like anyone

Record or upload 10-30 seconds of clean speech, create a voice prompt, then generate in that voice.

```bash
# Create a voice clone
createVoice recording.wav my_voice -t "transcript of what they said"
createVoice recording.wav my_voice --auto-transcribe    # Let Whisper handle it
createVoice recording.wav my_voice --no-transcript      # Speaker embedding only (no transcript needed)

# Generate with it
changeVoice "Hello" -o output                   # Uses default voice
changeVoice "Hello" -p my_voice.pt -o output    # Specific voice
changeVoice "Hello" --no-transcript -o output    # Clone without transcript
```

### Design — describe the voice you want

```bash
changeVoice "Hello" -m design -d "A warm, friendly female voice with a slight British accent" -o output
```

### Custom — 9 premium pre-trained speakers

Speakers: `ryan`, `aiden`, `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ono_anna`, `sohee`

```bash
changeVoice "Hello" -m custom -s ryan -o output
changeVoice "Hello" -m custom -s vivian -i "speak with excitement" -o output
changeVoice "Hello" -m custom -s ryan --prosody excited -o output   # Use a prosody preset
changeVoice --list-speakers
changeVoice --list-prosody
```

## Web Interface

Six tabs for everything you need:

| Tab | What it does |
|-----|-------------|
| **Clone Mode** | Generate with a cloned voice, pick from your voice prompts |
| **Design Mode** | Type a voice description and generate |
| **Custom Mode** | Pick a premium speaker, optionally add style instructions |
| **Create Voice** | Upload audio + transcript (or auto-transcribe) to create a new voice |
| **Manage Voices** | Preview, rename, delete voices; set your default |
| **Manage Models** | Load/unload models, set startup defaults, switch audio loader |

Models auto-load on first use. Status indicators show what's loaded. Cancel button stops generation mid-stream.

```bash
ttsUI --port 8080        # Custom port
ttsUI --share            # Public URL (Colab does this automatically)
ttsUI --no-browser       # Don't open browser
```

## CLI Reference

### Generation

```bash
changeVoice "Text" -o output                       # Basic
changeVoice "Text" -o output --play                 # Auto-play after
changeVoice "Text" --stream -o output               # Stream as it generates
changeVoice --clipboard -o from_clip                 # From clipboard
changeVoice "One" "Two" "Three" -o ~/Downloads/      # Batch from args
changeVoice --batch texts.json -o ~/Downloads/       # Batch from JSON array
```

### Tuning Output

```bash
changeVoice "Text" --preset consistent -o output     # Reproducible output
changeVoice "Text" --preset creative -o output        # More variation
changeVoice "Text" --temperature 0.5 --seed 42 -o output
changeVoice "Text" --speed 1.2 -o fast               # 20% faster (pyrubberband)
changeVoice "Text" --pitch -2 -o deep                 # Lower pitch (pyrubberband)
changeVoice "Text" --normalize --trim-silence -o clean
```

### Advanced

```bash
changeVoice --repl                                   # Interactive REPL
changeVoice --watch ~/Desktop/tts_input -o output    # Watch folder for .txt files
changeVoice --srt subtitles.srt -o subs              # Generate from SRT subtitles
changeVoice --dialogue convo.json -o dialogue         # Multi-speaker dialogue
changeVoice 'Hello <break time="500ms"/> world.' --ssml -o output
```

### Backend & Model Overrides

```bash
changeVoice --backend mlx "Text" -o output           # Force MLX for this run
changeVoice --model-size 0.6B "Text" -o output       # Use lighter model
changeVoice --list-backends                           # Show current config
```

### Voice Management (CLI)

```bash
changeVoice --list-prompts                           # All voice prompts
changeVoice --preview-prompt my_voice                # Play a voice preview
changeVoice --rename-prompt old_name new_name
changeVoice --delete-prompt unwanted
```

### Info & Stats

```bash
changeVoice --stats                                  # Server memory, cache, uptime
changeVoice --history 10                             # Last 10 generations
changeVoice --list-presets
changeVoice --list-aliases
changeVoice --list-models
```

## Voice Aliases

Save voice + preset combinations in `config.json`:

```json
"aliases": {
  "narrator": { "prompt": "narrator.pt", "preset": "consistent" },
  "character": { "prompt": "character.pt", "preset": "creative" }
}
```

```bash
changeVoice "Text" -v narrator -o output
```

## Python API

```python
from voice_client import TTSClient

client = TTSClient()

# Generate speech
audio_path = client.generate(
    "Hello world",
    output="output.wav",
    mode="clone",              # "clone", "design", or "custom"
    voice="narrator",          # Voice alias
    speed=1.1,
    normalize=True,
)

# Model management
client.load_model("design")
client.unload_model("design")
client.get_models()                                    # Status of all models
client.update_model_config(model_size="0.6B")          # Switch model variant
client.update_startup_config(clone=True, design=False)  # Startup defaults

# Voice management
client.list_prompts()
client.get_prompt_details("my_voice")
client.preview_prompt("my_voice")
client.rename_prompt("old", "new")
client.delete_prompt("unwanted")

# Multi-speaker dialogue
lines = [
    {"mode": "custom", "speaker": "ryan", "text": "Hello!"},
    {"mode": "custom", "speaker": "aiden", "text": "Hi there!"},
]
client.generate_dialogue(lines, output="dialogue.wav")
```

## Configuration

All settings live in `config.json`. Edit directly or use `configureTTS`.

| Setting | Values | Default | Description |
|---------|--------|---------|-------------|
| `advanced.backend` | `"mlx"`, `"torch"` | Platform-aware | MLX on Apple Silicon, torch elsewhere |
| `advanced.model_size` | `"1.7B"`, `"0.6B"` | `"1.7B"` | 0.6B is ~40% faster, uses less memory |
| `advanced.mlx_quantization` | `"4bit"`, `"8bit"`, `"bf16"` | `"8bit"` | MLX quantization level |
| `advanced.audio_loader` | `"torchaudio"`, `"librosa"` | `"torchaudio"` | Audio loading backend |
| `generation.temperature` | `0.0`-`2.0` | `0.7` | Higher = more variation |
| `generation.max_chunk_chars` | `0`-`10000` | `500` | Auto-splits long text (0 = no splitting) |
| `models.*.load_at_startup` | `true`/`false` | clone=true | Which models to preload |
| `server.auto_shutdown_minutes` | `0`+ | `0` | Auto-stop after idle (0 = never) |

### Presets

- **consistent** — temperature 0.5, seed 42, top_k 30. Same input = same output.
- **creative** — temperature 0.9, top_p 0.98. More expressive, varied output.

### Prosody Presets

Quick style selection for custom/design modes — `--prosody excited`, `--prosody calm`, etc. Built-in presets: excited, calm, whisper, authoritative, slow, fast, dramatic, conversational. Add your own in `config.json` under `prosody_presets`.

## MLX Backend (Apple Silicon)

MLX runs natively on Apple Silicon — lower thermals (~40-50C vs ~80-90C), less battery drain, quantized models use less memory.

```bash
configureTTS                                    # Switch backend in the wizard
# or edit config.json: "advanced": {"backend": "mlx"}
stopTTSServer && startTTSServer                 # Restart to apply
```

MLX voice cloning uses `.wav` + `.txt` file pairs instead of `.pt` tensors. `createVoice` saves all formats automatically.

**Quantization:** `4bit` (smallest, fastest) | `8bit` (default, balanced) | `bf16` (highest quality)

## Google Colab

A ready-to-run notebook is included (`colab_notebook.ipynb`).

1. Upload `Qwen3-TTS_UserFiles/` to Google Drive at `My Drive/Qwen3-TTS_UserFiles/`
2. Open `colab_notebook.ipynb` in Colab, select a T4+ GPU runtime
3. Run all cells — mounts Drive, installs deps, starts server, opens Gradio with a public URL

The system auto-detects Colab: binds `0.0.0.0`, enables Gradio sharing, uses CUDA.

## Troubleshooting

**Server won't start:** `cat .voice_server.log` for details. Kill stuck processes: `pkill -f voice_server.py && rm .voice_server.pid`

**Wrong conda env:** Wrapper scripts auto-switch, but if you updated them: `cp bin/* ~/bin/ && chmod +x ~/bin/*`

**Slow generation:** Make sure the server is running (`startTTSServer`). Without it, models reload every time.

**Bad audio quality:** Use `--preset consistent` or lower temperature (`--temperature 0.5`). Set a seed (`--seed 42`).

**Voice clone doesn't match:** Use cleaner source audio. 10-30 seconds of a single speaker, no background noise or music.

**Out of memory:** `stopTTSServer` to free everything. Use `--model-size 0.6B` or unload unused models in the Manage Models tab.

**MLX errors:** Make sure `advanced.backend` in config.json matches your conda env. Run `install.sh` to fix.

## Testing

```bash
python -m unittest discover -v tests/
```

266 tests, no GPU or running server required. Run inside a conda env (`qwen3-tts` or `qwen3-tts-mlx`) for full coverage — tests gracefully skip when optional dependencies are missing.

## Project Structure

```
~/Qwen3-TTS_UserFiles/
├── voice_config.py         # Config, constants, errors, platform detection
├── voice_engine.py         # Inference engine, audio processing, ASR, caching
├── voice_server.py         # Flask API server (port 5123)
├── voice_client.py         # Python client library
├── voice_generate.py       # CLI tool
├── voice_ui.py             # Gradio web interface (port 7860)
├── create_custom_voice.py  # Voice clone creation
├── config.json             # All settings
├── install.sh              # Installer with hardware detection
├── colab_notebook.ipynb    # Google Colab notebook
├── requirements-mlx.txt    # MLX environment dependencies
├── requirements-cuda.txt   # CUDA/Colab dependencies
├── voice_prompts/          # Voice files (.pt, .wav, .txt)
├── bin/                    # Wrapper scripts → installed to ~/bin/
└── tests/test_voice.py     # Test suite
```
