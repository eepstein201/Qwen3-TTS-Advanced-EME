# Qwen3-TTS Command Reference

> Install, testing, and code-quality tables are derived from `pyproject.toml` and the `Makefile`; the `tts` command tables are curated against the Click CLI (`qwen3_tts/cli*.py`). Reconcile against `tts --help` when the CLI changes.

## Installation & Setup Commands

| Command | Description |
|---------|-------------|
| `pip install -e .` | Install Qwen3-TTS in editable mode with all core dependencies |
| `pip install -e ".[torch]"` | Install with PyTorch backend support |
| `pip install -e ".[mlx]"` | Install with MLX backend support (Apple Silicon) |
| `pip install -e ".[server]"` | Install with FastAPI server dependencies |
| `pip install -e ".[ui]"` | Install with Gradio web UI dependencies |
| `pip install -e ".[dev]"` | Install with development tools (pytest, ruff, mypy, bandit) |
| `pip install -e ".[test]"` | Install test dependencies (works in any Python env) |
| `make install` | Quick install with all dependencies |
| `tts doctor` | Check installation health and dependencies |

## Core CLI Commands

| Command | Description |
|---------|-------------|
| `tts "TEXT"` | Generate audio from text (default command) |
| `tts generate "TEXT"` | Explicit generate command |
| `tts ui` | Launch Gradio web interface |
| `tts stats` | Show server statistics (memory, cache, history) |

## Voice Management Commands

| Command | Description |
|---------|-------------|
| `tts voice list` | List all voice clone prompts |
| `tts voice create AUDIO` | Create voice clone from audio file |
| `tts voice delete NAME` | Delete voice prompt |
| `tts voice rename OLD NEW` | Rename voice prompt |
| `tts voice preview NAME` | Play voice prompt |
| `tts voice rebuild NAME` | Regenerate a prompt's `.pt` (torch-only; forces `TTS_BACKEND=torch`) |
| `tts voice info NAME` | Show prompt metadata (formats, size, duration) via server |
| `tts list speakers` | List premium speakers (Custom mode) |
| `tts list presets` | List generation presets |
| `tts list prosody` | List prosody presets (Custom/Design mode) |
| `tts list aliases` | List voice aliases from config |

## Configuration Commands

| Command | Description |
|---------|-------------|
| `tts config` | Run interactive configuration wizard |
| `tts config show` | Show current settings |
| `tts config edit` | Set individual settings via flags (`--backend`, `--model-size`, `--mlx-quantization`, `--torch-quantization`, `--language`, `--output-dir`, `--voice-description`); with no flags, interactively prompts for the voice description only. Does not open `config.json` in an editor — use `tts config path` to locate it. |
| `tts config path` | Print config.json file path |
| `tts list models` | Show models and load status |
| `tts list backends` | Show available backends (torch, mlx, vllm) |

## Advanced Commands

| Command | Description |
|---------|-------------|
| `tts history [N]` | Show last N generations (default: 10) |
| `tts batch FILE` | Process batch JSON file with array of texts |
| `tts srt FILE` | Process SRT subtitle file |
| `tts dialogue FILE` | Process multi-speaker dialogue JSON |
| `tts repl` | Start interactive REPL mode |
| `tts watch DIR` | Watch directory for .txt files and generate audio |

## Cache Management Commands

| Command | Description |
|---------|-------------|
| `tts cache list` | List all cached HuggingFace models |
| `tts cache size` | Show total cache size on disk |
| `tts cache prune` | Remove models unused for N days |
| `tts cache clear` | Remove all cached models |

## Uninstall Commands

| Command | Description |
|---------|-------------|
| `tts uninstall models` | Remove all cached HuggingFace models |
| `tts uninstall voices` | Remove all voice prompts |
| `tts uninstall config` | Reset config.json to defaults |
| `tts uninstall environment` | Print conda removal commands |
| `tts uninstall all` | Run all uninstall steps |

## Testing Commands

| Command | Description |
|---------|-------------|
| `make test` | Run full test suite (pytest) |
| `make test-batch` | Run all test batches (1-6) |
| `make test-quick` | Run quick subset of tests |
| `make test-core` | Run Batch 1: Core utilities |
| `make test-voice` | Run Batch 2: Voice & CLI |
| `make test-server` | Run Batch 3: Server infrastructure |
| `make test-engine` | Run Batch 4: Engine & UI |
| `make test-optional` | Run Batch 5: Optional (pytest-dependent) |
| `make test-e2e` | Run Batch 6: E2E Playwright (requires server) |
| `python tests/run_batches.py` | Run all test batches with failure isolation |
| `python tests/run_batches.py --batch N` | Run specific batch (1-6) |
| `python tests/run_full_suite.py --full` | Full suite with multi-environment testing |

> The package also installs console-script wrappers around pytest: `test`,
> `test-unit`, `test-integration`, `test-quick`, `test-parallel`, `test-cov`.
> These are convenience entry points, not `tts` subcommands.

## Code Quality Commands

| Command | Description |
|---------|-------------|
| `make lint` | Run ruff linter (`ruff check qwen3_tts/ tests/`) |
| `make format` | Format code (`ruff format qwen3_tts/ tests/`) |
| `make coverage` | Run test coverage analysis |
| `make solid-score` | SOLID-compliance analyzer report |
| `ruff check qwen3_tts/ tests/` | Fast linting with ruff |
| `ruff format qwen3_tts/ tests/` | Format code with ruff |
| `mypy qwen3_tts/{core,server,interface}` | Type check with mypy |
| `bandit -r qwen3_tts -c pyproject.toml` | Security scan (target: 0 HIGH) |

## Generation Options

All generation commands support these options (use `tts generate --help` for full list):

| Option | Description |
|--------|-------------|
| `-m, --mode` | Voice mode: clone, design, custom |
| `-p, --prompt` | Voice clone prompt filename |
| `-d, --description` | Voice description (design mode) |
| `-s, --speaker` | Premium speaker name (custom mode) |
| `-i, --instruct` | Style instruction (custom mode) |
| `-v, --voice` | Voice alias from config |
| `--prosody` | Prosody preset (custom/design mode) |
| `--no-transcript` | Clone using speaker embedding only (transcript-free) |
| `-o, --output` | Output filename or directory |
| `--play` | Play audio after generation |
| `--stream` | Stream audio playback as it generates |
| `--speed` | Speed factor (1.2=faster, 0.8=slower) |
| `--pitch` | Pitch shift in semitones |
| `--preset` | Named preset from config |
| `--seed` | Random seed for reproducibility |
| `--temperature` | Sampling temperature (0.7-1.0) |
| `--top-k` | Top-k sampling (1-50) |
| `--top-p` | Top-p (nucleus) sampling (0.8-1.0) |
| `--backend` | Override backend (torch, mlx) |
| `--model-size` | Override model size (1.7B, 0.6B) |
| `--local` | Force local generation (skip server) |
| `--normalize` | Normalize audio to -3dB peak |
| `--trim-silence` | Trim leading/trailing silence |
| `--repetition-penalty` | Repetition penalty |
| `--max-chunk-chars` | Max chars per chunk (0 disables chunking) |
| `--max-new-tokens` | Max new tokens per chunk |
| `--ssml` | Enable SSML markup parsing |
| `--clipboard` | Read text from clipboard |
| `--no-open` | Don't open the output file |
| `--dry-run` | Show what would be generated without running |

## Server Commands

| Command | Description |
|---------|-------------|
| `tts server start` | Start model server (default port 5123) |
| `tts server stop` | Graceful server shutdown |
| `tts server status` | Show server status and loaded models |
| `tts server log` | Tail server log file |
| `tts server restart` | Restart server (stop + start) |
