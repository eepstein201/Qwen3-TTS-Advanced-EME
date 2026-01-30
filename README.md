# Qwen3-TTS Voice Generation System

A powerful text-to-speech system with voice cloning, built on Qwen3-TTS models. Features include voice cloning from audio samples, a persistent server for fast generation, audio processing, SSML markup, and more.

## Table of Contents

- [Getting Started](#getting-started)
- [Quick Start](#quick-start)
- [Web Interface](#web-interface)
- [Commands Reference](#commands-reference)
- [Voice Modes](#voice-modes)
- [Audio Processing](#audio-processing)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [Python API](#python-api)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Tips & Best Practices](#tips--best-practices)

---

## Getting Started

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Conda (miniforge recommended)
- ~8GB VRAM for model loading

### Installation

Run the installation script for automated setup:

```bash
cd ~/Qwen3-TTS_UserFiles
./install.sh
```

This will:
- Check prerequisites (macOS, Apple Silicon, conda)
- Create the `qwen3-tts` conda environment
- Install all dependencies
- Create wrapper scripts in `~/bin/`
- Optionally pre-download models

**Preview what will be done:**
```bash
./install.sh --dry-run
```

**Manual installation:** The system uses the `qwen3-tts` conda environment with:
- `qwen-tts` - Qwen3 TTS models
- `torch` - PyTorch with MPS support
- `flask` - Server framework
- `librosa` - Audio processing
- `soundfile`, `gradio`, `watchdog` - Additional dependencies

### First Run

1. **Start the TTS server** (recommended for fast generation):
   ```bash
   startTTSServer
   ```
   This loads models into memory (~30-60 seconds) for subsequent fast generation (~2-5 seconds).

2. **Generate your first audio**:
   ```bash
   changeVoice "Hello, world!" -o hello
   ```

3. **When done, stop the server**:
   ```bash
   stopTTSServer
   ```

---

## Web Interface

A Gradio-based web interface provides an easy-to-use alternative to CLI commands.

### Launching the UI

```bash
# Option 1: Direct launch (auto-starts server if needed)
changeVoice --ui

# Option 2: Interactive prompt
changeVoice
# Then select "2. Web Interface (Gradio UI in browser)"

# Option 3: Dedicated command
ttsUI
```

The server starts automatically when launching the UI if it's not already running.

This opens your browser to `http://localhost:7860` with a graphical interface.

### UI Features

- **Three tabs** for Clone, Design, and Custom modes
- **Status bar** showing server connection, memory usage, and loaded models
- **Voice prompt dropdown** (Clone mode)
- **Speaker selection** with descriptions (Custom mode)
- **Advanced settings** (collapsible): temperature, top-k, top-p, repetition penalty, seed
- **Audio processing** (collapsible): trim silence, normalize, speed, pitch
- **Built-in audio player** for immediate playback

### UI Options

```bash
ttsUI --port 8080      # Use a different port
ttsUI --share          # Create a public URL (for sharing)
ttsUI --no-browser     # Don't auto-open browser
```

---

## Quick Start

### Basic Usage

```bash
# Simple generation (opens file automatically)
changeVoice "Your text here" -o output_name

# Generate and play immediately
changeVoice "Quick test" -o test --play

# Generate from clipboard
changeVoice --clipboard -o from_clipboard

# Use a preset for consistent output
changeVoice "Reproducible output" --preset consistent -o output
```

### With Audio Processing

```bash
# Speed up by 20%
changeVoice "Faster speech" --speed 1.2 -o fast

# Lower pitch by 2 semitones
changeVoice "Deeper voice" --pitch -2 -o deep

# Normalize volume and trim silence
changeVoice "Clean audio" --normalize --trim-silence -o clean

# Combine multiple options
changeVoice "Full processing" --speed 1.1 --pitch -1 --normalize --trim-silence -o processed
```

### Using Voice Aliases

```bash
# Use a configured voice alias (combines prompt + preset)
changeVoice "Hello" -v default -o greeting

# List available aliases
changeVoice --list-aliases
```

---

## Commands Reference

### Main Commands

| Command | Description |
|---------|-------------|
| `changeVoice` | Main TTS generation command |
| `startTTSServer` | Start the persistent TTS server |
| `stopTTSServer` | Stop the TTS server |
| `createVoice` | Create a new voice clone from audio |
| `ttsUI` | Launch the Gradio web interface |

### changeVoice Options

#### Input/Output
| Option | Description |
|--------|-------------|
| `"text"` | Text to synthesize (or path to .txt file) |
| `-o, --output NAME` | Output filename (saved to ~/Downloads/) |
| `--clipboard` | Read text from system clipboard |
| `--batch FILE` | Process JSON array of texts |
| `--no-open` | Don't open the output file |
| `--play` | Play audio immediately after generation |

#### Voice Selection
| Option | Description |
|--------|-------------|
| `-v, --voice NAME` | Use a voice alias from config |
| `-p, --prompt FILE` | Use specific voice prompt (.pt file) |
| `-m, --mode clone\|design\|custom` | Voice mode (clone, design, or custom) |
| `-d, --description TEXT` | Voice description (for design mode) |
| `-s, --speaker NAME` | Premium speaker for custom mode (ryan, aiden, etc.) |
| `-i, --instruct TEXT` | Style instruction for custom mode |
| `--preset NAME` | Use named preset (consistent, creative) |

#### Generation Parameters
| Option | Description |
|--------|-------------|
| `--temperature FLOAT` | Sampling temperature (default: 0.7) |
| `--top-k INT` | Top-k sampling (default: 50) |
| `--top-p FLOAT` | Top-p nucleus sampling (default: 0.95) |
| `--seed INT` | Random seed for reproducibility |
| `--repetition-penalty FLOAT` | Repetition penalty (default: 1.05) |

#### Audio Processing
| Option | Description |
|--------|-------------|
| `--trim-silence` | Remove leading/trailing silence |
| `--normalize` | Normalize to -3dB peak |
| `--speed FACTOR` | Speed adjustment (1.2 = 20% faster) |
| `--pitch SEMITONES` | Pitch shift (+2 = higher, -2 = lower) |

#### Advanced Features
| Option | Description |
|--------|-------------|
| `--ssml` | Enable SSML markup parsing |
| `--repl` | Start interactive REPL mode |
| `--watch DIR` | Watch directory for .txt files |
| `--srt FILE` | Process SRT subtitle file |
| `--dialogue FILE` | Process dialogue JSON with multiple speakers |
| `--save-individual` | Save individual files for each dialogue line |
| `--dry-run` | Show what would be generated |

#### Utility
| Option | Description |
|--------|-------------|
| `--list-prompts` | List available voice prompts |
| `--list-presets` | List available presets |
| `--list-aliases` | List voice aliases |
| `--list-speakers` | List premium CustomVoice speakers |
| `--stats` | Show server statistics |
| `--history [N]` | Show last N generations |
| `--local` | Force local generation (skip server) |
| `--ui, --gui` | Launch the Gradio web interface |

---

## Voice Modes

### Clone Mode (Default)

Uses a voice prompt file (.pt) created from an audio sample to clone that voice.

```bash
# Use default clone prompt
changeVoice "Hello" -o output

# Use specific voice prompt
changeVoice "Hello" -p narrator.pt -o output
```

### Design Mode

Generates a voice from a text description (no audio sample needed).

```bash
# Use design mode with description
changeVoice "Hello" -m design -d "A warm, friendly female voice" -o output
```

### Custom Mode (Premium Speakers)

Uses pre-trained premium speakers from the CustomVoice model. No audio sample or description needed.

**Available Speakers:**

| Speaker | Language | Description |
|---------|----------|-------------|
| `ryan` | English | Dynamic male, strong rhythm |
| `aiden` | English | Sunny American male, clear midrange |
| `vivian` | Chinese | Bright young female |
| `serena` | Chinese | Warm, gentle female |
| `uncle_fu` | Chinese | Seasoned male, mellow timbre |
| `dylan` | Chinese | Youthful Beijing male |
| `eric` | Chinese | Lively Chengdu male |
| `ono_anna` | Japanese | Playful female |
| `sohee` | Korean | Warm female, rich emotion |

```bash
# Use premium speaker
changeVoice "Hello" -m custom -s ryan -o output

# With style instruction
changeVoice "Hello" -m custom -s ryan -i "speak with enthusiasm" -o output

# List all speakers
changeVoice --list-speakers
```

### Creating Voice Clones

```bash
# Create a new voice clone from audio
createVoice path/to/audio.wav my_voice_name

# The new prompt will be saved to voice_prompts/my_voice_name.pt
```

---

## Audio Processing

### Speed Adjustment

```bash
--speed 0.8    # 20% slower
--speed 1.0    # Normal (default)
--speed 1.2    # 20% faster
--speed 1.5    # 50% faster
```

### Pitch Adjustment

```bash
--pitch -4     # Much lower (4 semitones down)
--pitch -2     # Lower
--pitch 0      # Normal (default)
--pitch 2      # Higher
--pitch 4      # Much higher (4 semitones up)
```

### Normalization

The `--normalize` flag adjusts the audio to a -3dB peak level, ensuring consistent volume across generations.

### Silence Trimming

The `--trim-silence` flag removes silence from the beginning and end of the audio, using a -40dB threshold.

---

## Advanced Features

### SSML Markup

Enable SSML parsing with `--ssml` for fine-grained control:

```bash
changeVoice 'Hello <break time="500ms"/> world.' --ssml -o output
```

#### Supported SSML Tags

| Tag | Example | Effect |
|-----|---------|--------|
| `<break>` | `<break time="500ms"/>` | Insert pause |
| `<emphasis>` | `<emphasis>important</emphasis>` | Emphasis (natural) |
| `<sub>` | `<sub alias="NASA">N.A.S.A.</sub>` | Substitution |
| `<say-as>` | `<say-as interpret-as="characters">ABC</say-as>` | Spell out |
| `<prosody>` | `<prosody rate="fast" pitch="high">text</prosody>` | Speed/pitch hints |

### Interactive REPL Mode

Start an interactive session for rapid iteration:

```bash
changeVoice --repl
```

REPL Commands:
```
/voice NAME    - Switch voice alias
/preset NAME   - Switch preset
/prompt NAME   - Switch voice prompt
/play on|off   - Toggle auto-play
/speed FACTOR  - Set speed
/pitch SEMI    - Set pitch shift
/status        - Show current settings
/quit          - Exit
```

### Watch Mode

Monitor a directory and automatically generate TTS for new .txt files:

```bash
changeVoice --watch ~/Desktop/tts_input -o ~/Desktop/tts_output
```

### SRT Subtitle Processing

Generate audio for each subtitle in an SRT file:

```bash
changeVoice --srt subtitles.srt -o ~/Downloads/subtitles
```

This creates:
- Individual audio files for each subtitle (`subtitles_001.wav`, etc.)
- A combined audio file (`subtitles_combined.wav`)

### Multi-Speaker Dialogue

Generate dialogue with multiple speakers using a JSON file:

```bash
changeVoice --dialogue conversation.json -o ~/Downloads/
```

**Simple format** (inline speaker config):
```json
[
  {"mode": "custom", "speaker": "ryan", "text": "Hello, how are you?"},
  {"mode": "custom", "speaker": "aiden", "text": "I'm doing great, thanks!"},
  {"mode": "clone", "prompt": "narrator.pt", "text": "They shook hands warmly."}
]
```

**Named speakers format** (reusable speaker definitions):
```json
{
  "speakers": {
    "Alice": {"mode": "custom", "speaker": "vivian"},
    "Bob": {"mode": "clone", "prompt": "bob_voice.pt"},
    "Narrator": {"mode": "design", "description": "A deep, calm male narrator"}
  },
  "lines": [
    {"speaker": "Alice", "text": "Hello Bob!"},
    {"speaker": "Bob", "text": "Hi Alice! How are you?"},
    {"speaker": "Narrator", "text": "Alice smiled warmly."},
    {"speaker": "Alice", "text": "I'm wonderful, thank you!"}
  ],
  "pause_ms": 500
}
```

Options:
- `--save-individual`: Also save each line as a separate file
- `pause_ms` in JSON: Silence between lines (default: 500ms)

---

## Configuration

### Config File Location

`~/Qwen3-TTS_UserFiles/config.json`

### Config Structure

```json
{
  "default_voice_description": "A calm, friendly voice...",
  "default_clone_prompt": "default_clone.pt",
  "output_directory": "~/Downloads",
  "language": "English",
  "server": {
    "host": "127.0.0.1",
    "port": 5123,
    "auto_shutdown_minutes": 0
  },
  "models": {
    "clone": { "load_at_startup": true },
    "design": { "load_at_startup": false },
    "custom": { "load_at_startup": false }
  },
  "security": {
    "max_text_length": 10000,
    "max_batch_size": 20
  },
  "generation": {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "seed": null
  },
  "presets": {
    "consistent": {
      "temperature": 0.5,
      "top_k": 30,
      "seed": 42
    },
    "creative": {
      "temperature": 0.9,
      "top_p": 0.98
    }
  },
  "aliases": {
    "default": {
      "prompt": "default_clone.pt",
      "preset": "consistent"
    }
  }
}
```

### Model Configuration

Control which models load at server startup to manage memory usage (~3.5GB per model):

```bash
# See available models and their status
changeVoice --list-models
```

| Model | Purpose | Memory |
|-------|---------|--------|
| `clone` | Voice cloning from audio samples | ~3.5GB |
| `design` | Generate voice from text description | ~3.5GB |
| `custom` | 9 premium pre-trained speakers | ~3.5GB |

**On-demand loading:** If you use a feature requiring an unloaded model, you'll be prompted to load it dynamically. This lets you start with minimal memory and add models as needed.

```json
"models": {
  "clone": { "load_at_startup": true },
  "design": { "load_at_startup": false },
  "custom": { "load_at_startup": true }
}
```

### Auto-Shutdown

Set `auto_shutdown_minutes` to automatically stop the server after inactivity:

```json
"server": {
  "auto_shutdown_minutes": 30
}
```

### Creating Voice Aliases

Add aliases to quickly switch between voice configurations:

```json
"aliases": {
  "narrator": {
    "prompt": "narrator.pt",
    "preset": "consistent"
  },
  "character": {
    "prompt": "character.pt",
    "preset": "creative"
  }
}
```

---

## Python API

### Installation

The API client is included at `~/Qwen3-TTS_UserFiles/tts_client.py`.

### Basic Usage

```python
from tts_client import TTSClient, generate

# Quick one-liner
audio_path = generate("Hello world", output="greeting.wav")

# Full control with client
client = TTSClient()

# Check server status
if client.is_server_running():
    stats = client.get_stats()
    print(f"Memory usage: {stats['mps_memory_allocated_mb']}MB")

# Generate with options
audio_path = client.generate(
    "Hello world",
    output="~/Downloads/output.wav",
    voice="narrator",           # Use voice alias
    speed=1.1,                  # 10% faster
    pitch=-2,                   # Lower pitch
    normalize=True,             # Normalize volume
    trim_silence=True           # Trim silence
)

# Generate with premium speaker (custom mode)
audio_path = client.generate(
    "Hello world",
    mode="custom",
    speaker="Ryan",             # Premium speaker
    instruct="speak cheerfully" # Optional style instruction
)

# List available resources
print(client.list_prompts())    # Voice prompts
print(client.list_presets())    # Generation presets
print(client.list_aliases())    # Voice aliases

# Generate multi-speaker dialogue
lines = [
    {"mode": "custom", "speaker": "ryan", "text": "Hello there!"},
    {"mode": "custom", "speaker": "aiden", "text": "Hi! Nice to meet you."},
]
audio_path = client.generate_dialogue(lines, output="dialogue.wav", pause_ms=500)
```

### API Reference

```python
client = TTSClient(config_path=None)  # Uses default config

# Properties
client.config           # Configuration dictionary
client.server_url       # Server URL string

# Methods
client.is_server_running()      # Check server status
client.get_stats()              # Get server statistics
client.list_prompts()           # List voice prompts
client.list_presets()           # List presets
client.list_aliases()           # List voice aliases
client.resolve_alias(name)      # Get alias configuration
client.reload_config()          # Reload configuration

# Generation
client.generate(
    text,                       # Text to synthesize
    output=None,                # Output path
    mode="clone",               # clone, design, or custom
    prompt=None,                # Voice prompt file (clone mode)
    description=None,           # Voice description (design mode)
    speaker=None,               # Speaker name (custom mode)
    instruct=None,              # Style instruction (custom mode)
    voice=None,                 # Voice alias
    preset=None,                # Preset name
    temperature=None,           # Sampling temperature
    top_k=None,                 # Top-k sampling
    top_p=None,                 # Top-p sampling
    seed=None,                  # Random seed
    repetition_penalty=None,    # Repetition penalty
    speed=None,                 # Speed factor
    pitch=None,                 # Pitch semitones
    normalize=False,            # Normalize audio
    trim_silence=False,         # Trim silence
    use_server=True             # Use server if running
)
```

---

## FAQ

### Q: Why should I use the server?

**A:** The server keeps models loaded in memory, reducing generation time from ~30-60 seconds (cold start) to ~2-5 seconds. If you're generating multiple clips, the server is much faster.

### Q: How do I create a voice clone?

**A:** Use the `createVoice` command with a clean audio sample:
```bash
createVoice path/to/audio.wav voice_name
```
For best results, use 10-30 seconds of clear speech without background noise.

### Q: What's the difference between clone and design mode?

**A:**
- **Clone mode** uses a voice prompt file created from an audio sample to replicate that specific voice.
- **Design mode** generates a voice from a text description, useful when you don't have an audio sample.

### Q: How do I get consistent output?

**A:** Use the `consistent` preset or set a fixed seed:
```bash
changeVoice "Text" --preset consistent -o output
# or
changeVoice "Text" --seed 42 --temperature 0.5 -o output
```

### Q: Can I process multiple texts at once?

**A:** Yes, use batch mode:
```bash
# Multiple arguments
changeVoice "Text one" "Text two" "Text three" -o ~/Downloads/batch/

# From JSON file
echo '["Text one", "Text two"]' > texts.json
changeVoice --batch texts.json -o ~/Downloads/batch/
```

### Q: How much memory does this use?

**A:** The models use approximately 8GB of MPS (GPU) memory when loaded. Check with:
```bash
changeVoice --stats
```

---

## Troubleshooting

### Server Won't Start

1. **Check if already running:**
   ```bash
   curl http://127.0.0.1:5123/health
   ```

2. **Check the log file:**
   ```bash
   cat ~/Qwen3-TTS_UserFiles/.tts_server.log
   ```

3. **Kill any stuck processes:**
   ```bash
   pkill -f tts_server.py
   rm ~/Qwen3-TTS_UserFiles/.tts_server.pid
   ```

### Generation is Slow

- Make sure the server is running (`startTTSServer`)
- Check that you're not using `--local` flag
- Verify server is responsive: `changeVoice --stats`

### Audio Quality Issues

1. **Robotic/distorted output:**
   - Lower the temperature: `--temperature 0.5`
   - Use the `consistent` preset

2. **Inconsistent voice:**
   - Set a fixed seed: `--seed 42`
   - Use lower temperature

3. **Too much silence:**
   - Use `--trim-silence`

4. **Volume too low/high:**
   - Use `--normalize`

### Voice Clone Doesn't Sound Right

- Ensure source audio is clean (no background noise, music)
- Use 10-30 seconds of clear speech
- Try different parts of the source audio
- The voice may work better with certain types of text

### "Model not found" Error

Ensure models are downloaded. They should be cached in `~/.cache/huggingface/hub/`. The first run will download them automatically.

### Memory Issues

If you run out of memory:
1. Stop the server: `stopTTSServer`
2. Close other GPU-intensive applications
3. Restart the server

---

## Tips & Best Practices

### For Best Quality

1. **Use the consistent preset** for reproducible, stable output
2. **Normalize your output** for consistent volume levels
3. **Trim silence** for cleaner audio files
4. **Keep text segments reasonable** - very long texts may have quality degradation

### For Fastest Workflow

1. **Always use the server** - it's 10-20x faster
2. **Use voice aliases** to quickly switch configurations
3. **Use REPL mode** for rapid iteration and testing
4. **Set up watch mode** for batch processing workflows

### For Voice Cloning

1. **Source audio quality matters** - use clean, clear recordings
2. **10-30 seconds is ideal** - too short lacks variety, too long may confuse the model
3. **Single speaker only** - don't mix multiple voices
4. **Consistent recording conditions** - same microphone, room, distance

### For Production Use

1. **Set auto-shutdown** if running on shared resources
2. **Use the Python API** for integration with other tools
3. **Log generations** with `--history` to track what was created
4. **Use presets** to maintain consistency across sessions

---

## Security

### API Token Authentication

The server generates a random auth token on startup. All API requests (except `/health` and `/generation-status`) require the token:

```
Authorization: Bearer <token>
```

The CLI and Python API handle this automatically. The token is stored at `~/.tts_server_token` and cleaned up on shutdown.

### Input Validation

The server validates all inputs:
- **Text length:** max 10,000 characters per text (configurable in `config.json` under `security.max_text_length`)
- **Batch size:** max 20 texts per request (configurable under `security.max_batch_size`)
- **Path traversal:** prompt file names cannot contain `..` or `/`
- **Mode/speaker:** validated against known values

### Network Binding

The server binds to `127.0.0.1` (localhost only) by default. Use `--public` to bind to `0.0.0.0`:
```bash
python tts_server.py --public
```

---

## Progress Display

### CLI Progress
When generating via the server, a live progress spinner shows elapsed time and ETA:
```
⠋ Generating... 5s elapsed / ~12s ETA
```

ETA is estimated from your generation history (median characters/second).

### Gradio Progress
The web interface shows a progress bar during generation, capped at 95% until completion.

---

## Post-Generation Menu

After generating with the server, the CLI shows a menu:
```
What would you like to do?
  1. Same settings (re-generate with same arguments)
  2. Edit text (open in editor, then re-generate)
  3. New settings (start fresh with interactive mode)
  4. Exit
```

Output filenames auto-increment (`output.wav` -> `output_2.wav` -> `output_3.wav`) so previous files are never overwritten.

---

## Testing

Run the test suite (no GPU, models, or running server required):

```bash
python -m unittest discover -v tests/
```

36 tests covering config, server validation, authentication, SSML/SRT parsing, and filename logic.

---

## Files & Directories

```
~/Qwen3-TTS_UserFiles/
├── install.sh              # Installation script
├── tts_generate.py         # Main generation script
├── tts_server.py           # Persistent server (auth, validation, logging)
├── tts_client.py           # Python API client
├── tts_ui.py               # Gradio web interface
├── tts_config.py           # Shared config, constants, error classes
├── tts_engine.py           # Model loading & inference engine
├── config.json             # Configuration
├── create_custom_voice.py  # Voice cloning script
├── voice_prompts/          # Voice prompt files (.pt)
│   └── default_clone.pt
├── bin/                    # Wrapper scripts (canonical source)
│   ├── changeVoice
│   ├── startTTSServer
│   ├── stopTTSServer
│   ├── createVoice
│   └── ttsUI
├── tests/                  # Test suite
│   └── test_tts.py
├── .tts_server.pid         # Server PID file (runtime)
└── .tts_server.log         # Server log (runtime)

~/bin/                      # Installed wrapper scripts (copied from bin/)
├── changeVoice
├── startTTSServer
├── stopTTSServer
├── createVoice
└── ttsUI

~/.tts_server_token         # Auth token (runtime, 0600 perms)
~/.tts_history.jsonl        # Generation history
~/.tts_last_text            # Last generated text (for edit-and-rerun)
```

---

## Version History

All features implemented across 10 phases:

- **Phase 1:** Core usability (`--play`, `--clipboard`, `--trim-silence`, `--dry-run`)
- **Phase 2:** Workflow (`--voice` aliases, `--history`, `--stats`, prompt management)
- **Phase 3:** Server enhancements (auto-shutdown, queue system, threading)
- **Phase 4:** Audio processing (`--normalize`, `--speed`, `--pitch`, `--dialogue`)
- **Phase 5:** Integration (`--repl`, `--watch`, `--srt`, Python API)
- **Phase 6:** Advanced (`--ssml` markup support)
- **Phase 7:** CustomVoice (9 premium speakers, `-m custom -s SPEAKER`)
- **Phase 8:** Configurable model loading (on-demand, `/load-model` API)
- **Phase 9:** Installation & Web UI (`install.sh`, Gradio interface, `changeVoice --ui`)
- **Phase 10:** Security, reliability & UX (auth tokens, input validation, logging, structured errors, progress/ETA, post-generation menu, test suite)
