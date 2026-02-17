# Qwen3-TTS Voice Generation System

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Colab-lightgrey)

Clone any voice from an audio sample, design voices from text descriptions, or choose from 9 premium speakers. Powered by Qwen3-TTS models with a persistent server for fast generation.

**Platforms:** Mac (Apple Silicon with MLX, Intel with PyTorch), Linux with NVIDIA GPU, Google Colab

```python
from qwen3_tts.server.client import generate
generate("Hello world!", output="hello.wav")  # That's it.
```

## System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| Disk | ~3 GB per model | ~10 GB (all 3 models) |
| macOS | Apple Silicon M1+ | M2+ |
| Linux / Colab | NVIDIA T4+ (CUDA) | A10G / L4 |

Note: 0.6B models use half the memory and run ~40% faster than the default 1.7B.

## Install

```bash
cd ~/Qwen3-TTS_UserFiles
./install.sh
```

The installer detects your hardware, walks you through backend/model/quantization choices, creates the right conda environment, and installs commands to `~/bin/`.

```bash
tts config               # Re-run the setup wizard anytime
tts config --show        # Compare current settings to recommendations
```

## Quick Start

```bash
tts server start                        # Load model (~30-60s first time)
tts "Hello, world!" -o hello            # Generate speech -> hello.wav
echo "Hello world" | tts -o hello       # Pipe text directly
tts server stop                         # Free memory when done
```

Or skip straight to the web UI:

```bash
tts ui                   # Opens http://localhost:7860
tts --ui                 # Same thing, auto-starts server
```

## Three Voice Modes

| Feature | Clone | Design | Custom |
|---------|-------|--------|--------|
| Sound like a specific person | Yes | No | No |
| Describe voice in text | No | Yes | No |
| Pre-trained speakers | No | No | 9 speakers |
| Reference audio needed | Yes (5-15s) | No | No |
| Style instructions | No | Via description | Yes (--instruct) |
| Prosody presets | Yes (post-processing) | Yes | Yes |
| Transcript needed | Optional (--no-transcript) | N/A | N/A |

### Clone (default) -- sound like anyone

Record or upload 5-15 seconds of clean speech, create a voice prompt, then generate in that voice.

```bash
# Create a voice clone
tts voice create recording.wav --name my_voice -t "transcript of what they said"
tts voice create recording.wav --name my_voice --auto-transcribe    # Let Whisper handle it
tts voice create recording.wav --name my_voice --no-transcript      # Speaker embedding only (no transcript needed)

# Generate with it
tts "Hello" -o output                   # Uses default voice
tts "Hello" -p my_voice.pt -o output    # Specific voice
tts "Hello" --no-transcript -o output    # Clone without transcript
```

### Design -- describe the voice you want

```bash
tts "Hello" -m design -d "A warm, friendly female voice with a slight British accent" -o output
```

### Custom -- 9 premium pre-trained speakers

Speakers: `ryan`, `aiden`, `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ono_anna`, `sohee`

```bash
tts "Hello" -m custom -s ryan -o output
tts "Hello" -m custom -s vivian -i "speak with excitement" -o output
tts "Hello" -m custom -s ryan --prosody excited -o output   # Use a prosody preset
tts list speakers
```

## Web Interface

Six tabs for everything you need:

| Tab | What it does |
|-----|-------------|
| **Clone Mode** | Generate with a cloned voice, pick from your voice prompts, style adjustments via post-processing |
| **Design Mode** | Type a voice description (or use the Description Builder), generate, and optionally save as a reusable voice prompt |
| **Custom Mode** | Pick a premium speaker, optionally add style instructions |
| **Create Voice** | Upload audio + transcript (or auto-transcribe) to create a new voice |
| **Manage Voices** | Preview, rename, delete voices; set your default |
| **Manage Models** | Load/unload models, set startup defaults, switch audio loader |

Models auto-load on first use. Status indicators show what's loaded. Cancel button stops generation mid-stream. The UI also exposes a programmatic API via `gradio_client` for automation and testing.

```bash
tts ui --port 8080       # Custom port
tts ui --share           # Public URL (Colab does this automatically)
tts ui --no-browser      # Don't open browser
```

## CLI Reference

### Generation

```bash
tts "Text" -o output                                # Basic
tts "Text" -o output --play                          # Auto-play after
tts "Text" --stream -o output                        # Stream as it generates
tts --clipboard -o from_clip                         # From clipboard
tts "One" "Two" "Three" -o ~/Downloads/              # Batch from args
tts batch texts.json -o ~/Downloads/                 # Batch from JSON array
```

### Tuning Output

```bash
tts "Text" --preset consistent -o output              # Reproducible output
tts "Text" --preset creative -o output                 # More variation
tts "Text" --temperature 0.5 --seed 42 -o output
tts "Text" --speed 1.2 -o fast                        # 20% faster (pyrubberband)
tts "Text" --pitch -2 -o deep                          # Lower pitch (pyrubberband)
tts "Text" --normalize --trim-silence -o clean
```

### Advanced

```bash
tts repl                                             # Interactive REPL
tts watch ~/Desktop/tts_input -o output              # Watch folder for .txt files
tts srt subtitles.srt -o subs                        # Generate from SRT subtitles
tts dialogue convo.json -o dialogue                  # Multi-speaker dialogue
tts --dry-run "Text" -o output                       # Preview settings without generating
tts                                                  # Interactive mode (no args)
```

#### SSML Markup

Pass `--ssml` to enable SSML tag processing in your text:

```bash
tts 'Hello <break time="500ms"/> world.' --ssml -o output
```

| Tag | Example | Effect |
|-----|---------|--------|
| `<break>` | `<break time="500ms"/>` | Insert pause |
| `<sub>` | `<sub alias="doctor">Dr.</sub>` | Text substitution |
| `<say-as>` | `<say-as interpret-as="characters">ABC</say-as>` | Spell out as "A B C" |
| `<emphasis>` | `<emphasis>important</emphasis>` | Emphasis |
| `<prosody>` | `<prosody rate="slow" pitch="low">text</prosody>` | Speed/pitch hints |

### Backend & Model Overrides

```bash
tts --backend mlx "Text" -o output                   # Force MLX for this run
tts --model-size 0.6B "Text" -o output               # Use lighter model
tts list backends                                    # Show current config
```

### Voice Management (CLI)

```bash
tts voice list                                       # All voice prompts
tts voice preview my_voice                           # Play a voice preview
tts voice rename old_name new_name
tts voice delete unwanted
```

### Info & Stats

```bash
tts stats                                            # Server memory, cache, uptime
tts history 10                                       # Last 10 generations
tts list presets
tts list aliases
tts list models
tts list prosody                                     # List prosody presets
```

## Voice Aliases

Save voice + preset combinations in `config.json`:

```json
"aliases": {
  "narrator": { "prompt": "narrator.pt", "preset": "consistent" },
  "designer": { "mode": "design", "description": "A warm British female voice", "preset": "creative" },
  "ryan_excited": { "mode": "custom", "speaker": "ryan", "instruct": "speak with excitement" }
}
```

Supported fields: `mode`, `prompt`, `preset`, `description`, `speaker`, `instruct`.

```bash
tts "Text" -v narrator -o output
```

## Python API

```python
# Quick one-liner (creates client internally)
from qwen3_tts.server.client import generate
generate("Hello!", output="hello.wav")

# Full client usage
from qwen3_tts.server.client import TTSClient

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

# Streaming generation
for wav_chunk, sr in client.generate_streaming("Long text...", output="stream.wav"):
    pass  # Audio plays as it generates

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

All settings live in `config.json`. Edit directly or use `tts config`.

| Setting | Values | Default | Description |
|---------|--------|---------|-------------|
| `language` | `"English"`, etc. | `"English"` | Default language for generation |
| `advanced.backend` | `"mlx"`, `"torch"` | Platform-aware | MLX on Apple Silicon, torch elsewhere |
| `advanced.model_size` | `"1.7B"`, `"0.6B"` | `"1.7B"` | 0.6B is ~40% faster, uses less memory |
| `advanced.mlx_quantization` | `"4bit"`, `"8bit"`, `"bf16"` | `"8bit"` | MLX quantization level |
| `advanced.audio_loader` | `"torchaudio"`, `"librosa"` | `"torchaudio"` | Audio loading backend |
| `advanced.dtype` | `"float32"`, `"float16"`, `"bfloat16"` | `"bfloat16"` | Tensor dtype (torch backend) |
| `generation.temperature` | `0.0`-`2.0` | `0.7` | Higher = more variation |
| `generation.max_chunk_chars` | `0`-`10000` | `500` | Auto-splits long text (0 = no splitting) |
| `generation.max_new_tokens` | `1`-`32768` | `2048` | Max tokens per generation (safety limit) |
| `generation.compile_model` | `true`/`false` | `true` | Enable torch.compile (Ampere+ GPUs) |
| `models.*.load_at_startup` | `true`/`false` | clone=true | Which models to preload |
| `server.auto_shutdown_minutes` | `0`+ | `0` | Auto-stop after idle (0 = never) |

### Presets

- **consistent** -- temperature 0.5, seed 42, top_k 30. Same input = same output.
- **creative** -- temperature 0.9, top_p 0.98. More expressive, varied output.

### Prosody Presets

Quick style selection for custom/design modes -- `--prosody excited`, `--prosody calm`, etc. Built-in presets: excited, calm, whisper, authoritative, slow, fast, dramatic, conversational. Add your own in `config.json` under `prosody_presets`.

### Environment Overrides

Override config.json for a single session:

```bash
TTS_BACKEND=torch tts "Text" -o output
TTS_MODEL_SIZE=0.6B tts "Text" -o output
```

## MLX Backend (Apple Silicon)

MLX runs natively on Apple Silicon -- lower thermals (~40-50C vs ~80-90C), less battery drain, quantized models use less memory.

```bash
tts config                                     # Switch backend in the wizard
# or edit config.json: "advanced": {"backend": "mlx"}
tts server stop && tts server start            # Restart to apply
```

MLX voice cloning uses `.wav` + `.txt` file pairs instead of `.pt` tensors. `tts voice create` saves all formats automatically.

**Quantization:** `4bit` (smallest, fastest) | `8bit` (default, balanced) | `bf16` (highest quality)

## Model Size Guide

| | 1.7B (default) | 0.6B (lite) |
|---|---|---|
| Quality | Higher fidelity | Good for most uses |
| Speed | Baseline | ~40% faster |
| RAM (torch) | ~3.5 GB/model | ~2 GB/model |
| RAM (MLX 8-bit) | ~2.5 GB/model | ~1.5 GB/model |
| Best for | Production, voice cloning | Quick iteration, low memory |

```bash
tts --model-size 0.6B "Text" -o output             # One-off override
tts config                                         # Change default permanently
```

## Google Colab

A ready-to-run notebook is included (`colab_notebook.ipynb`).

1. Upload `Qwen3-TTS_UserFiles/` to Google Drive at `My Drive/Qwen3-TTS_UserFiles/`
2. Open `colab_notebook.ipynb` in Colab, select a T4+ GPU runtime (L4 recommended for best performance)
3. Edit the **Settings** form cell at the top to configure your session
4. Run all cells -- detects GPU tier, mounts Drive, installs deps (uses `uv` for faster pip installs when available), starts server, opens Gradio with a public shareable URL

The notebook auto-detects your GPU and applies optimal settings: Flash Attention 2 + bfloat16 on Ampere+ GPUs (L4/A100), SDPA + float16 + 8-bit quantization on Turing GPUs (T4).

### Settings Form

The first code cell is a Colab form with these fields:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_SIZE` | `1.7B` | `1.7B` or `0.6B` |
| `PRELOAD_CLONE` | `True` | Load clone model at startup |
| `PRELOAD_DESIGN` | `False` | Load design model at startup |
| `PRELOAD_CUSTOM` | `False` | Load custom model at startup |
| `DEFAULT_MODE` | `design` | Default generation mode |
| `MAX_NEW_TOKENS` | `2048` | Max tokens per generation (safety limit) |
| `COMPILE_MODEL` | `True` | Enable torch.compile (best on Ampere+ GPUs) |
| `AUTO_LAUNCH_UI` | `True` | Launch Gradio UI automatically |

### Voice Cloning on Colab

The notebook includes a voice cloning cell:

1. Run the **Voice Cloning** cell -- it prompts you to upload a `.wav` or `.mp3` file (5-15 seconds of clear speech)
2. The clone model loads automatically if needed
3. A voice prompt is saved to `voice_prompts/` and can be used immediately in the Gradio UI or via the Python client

The system auto-detects Colab: binds `0.0.0.0`, enables Gradio sharing, uses CUDA.

## FAQ

**Do I need a GPU?**
No on Mac -- MLX runs on the Neural Engine / GPU built into Apple Silicon. On Linux, yes -- you need an NVIDIA GPU with CUDA support.

**How much disk space?**
About 3 GB per model. With all 3 models (clone, design, custom) that is ~10 GB. MLX 4-bit models are smaller.

**Can I use multiple voices in one file?**
Yes -- use dialogue mode: `tts dialogue convo.json -o output`. The JSON file contains an array of objects with `mode`, `speaker`/`prompt`, and `text` fields.

**How do I improve clone quality?**
Use clean audio: 5-15 seconds of a single speaker, no background noise or music. Generate with `--preset consistent` for reliable results.

**Is there a web API?**
Yes -- the server runs on port 5123 with REST endpoints. You can also use the Python API (see above) or the Gradio web interface.

## Troubleshooting

**Server won't start:** `tts server log` for details. Kill stuck processes: `pkill -f voice_server.py && rm .voice_server.pid`

**Wrong conda env:** Wrapper scripts auto-switch, but if you updated them: `cp bin/* ~/bin/ && chmod +x ~/bin/*`

**Slow generation:** Make sure the server is running (`tts server start`). Without it, models reload every time.

**Bad audio quality:** Use `--preset consistent` or lower temperature (`--temperature 0.5`). Set a seed (`--seed 42`).

**Voice clone doesn't match:** Use cleaner source audio. 5-15 seconds of a single speaker, no background noise or music.

**Out of memory:** `tts server stop` to free everything. Use `--model-size 0.6B` or unload unused models in the Manage Models tab.

**MLX errors:** Make sure `advanced.backend` in config.json matches your conda env. Run `install.sh` to fix.

## Testing

```bash
python -m unittest discover -v tests/
```

334+ tests, no GPU or running server required. Run inside a conda env (`qwen3-tts` or `qwen3-tts-mlx`) for full coverage -- tests gracefully skip when optional dependencies are missing.

## Project Structure

```
~/Qwen3-TTS_UserFiles/
├── qwen3_tts/                  # Python package
│   ├── cli.py                  # Click CLI entry point
│   ├── core/
│   │   ├── config.py           # Config, constants, platform detection
│   │   └── engine.py           # Inference, audio processing, ASR
│   ├── server/
│   │   ├── app.py              # Flask API server (port 5123)
│   │   └── client.py           # Python client library
│   ├── interface/
│   │   ├── generate.py         # CLI generation logic
│   │   └── ui.py               # Gradio web interface (port 7860)
│   └── tools/
│       └── create_voice.py     # Voice clone creation
├── bin/                        # Bash wrappers → ~/bin/
├── tests/                      # 334+ tests
├── config.json                 # All settings
├── install.sh                  # Installer with hardware detection
├── colab_notebook.ipynb        # Google Colab notebook
├── pyproject.toml              # Package metadata and dependencies
└── voice_prompts/              # Voice files (.pt, .wav, .txt)
```

## Migration from v1

The v2 unified CLI consolidates all commands under a single `tts` entry point. Old commands still work as deprecation shims but will be removed in a future release.

| Old Command | New Command |
|-------------|-------------|
| `startTTSServer` | `tts server start` |
| `stopTTSServer` | `tts server stop` |
| `changeVoice "Hello" -o hello` | `tts "Hello" -o hello` |
| `changeVoice --list-prompts` | `tts voice list` |
| `changeVoice --list-speakers` | `tts list speakers` |
| `changeVoice --preview-prompt NAME` | `tts voice preview NAME` |
| `changeVoice --rename-prompt OLD NEW` | `tts voice rename OLD NEW` |
| `changeVoice --delete-prompt NAME` | `tts voice delete NAME` |
| `changeVoice --stats` | `tts stats` |
| `changeVoice --history` | `tts history` |
| `changeVoice --repl` | `tts repl` |
| `changeVoice --batch FILE` | `tts batch FILE` |
| `changeVoice --srt FILE` | `tts srt FILE` |
| `changeVoice --dialogue FILE` | `tts dialogue FILE` |
| `changeVoice --watch DIR` | `tts watch DIR` |
| `createVoice audio.wav name` | `tts voice create audio.wav --name name` |
| `ttsUI` | `tts ui` |
| `configureTTS` | `tts config` |
| `configureTTS --show` | `tts config show` |
| `cat .voice_server.log` | `tts server log` |

Python imports have also moved to package paths:

| Old Import | New Import |
|------------|-----------|
| `from voice_client import TTSClient` | `from qwen3_tts.server.client import TTSClient` |
| `from voice_client import generate` | `from qwen3_tts.server.client import generate` |
| `from voice_config import ...` | `from qwen3_tts.core.config import ...` |

## License

The source code in this repository is licensed under Apache 2.0. See [LICENSE](LICENSE).

The Qwen3-TTS model weights are subject to their own license provided by Qwen Research/Alibaba Cloud.
