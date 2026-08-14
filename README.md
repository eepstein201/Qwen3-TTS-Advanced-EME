# Qwen3-TTS Voice Generation System

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Version](https://img.shields.io/badge/version-3.0.0-green)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Colab-lightgrey)

Clone any voice from an audio sample, design voices from text descriptions, or choose from 9 premium speakers. Powered by Qwen3-TTS models with a high-concurrency, asynchronous FastAPI server.

## Table of Contents
1. [Features & Capabilities](#features--capabilities)
2. [System Requirements & Optimal Configurations](#system-requirements--optimal-configurations)
3. [Run on Google Colab (Zero Install)](#run-on-google-colab-zero-install)
4. [Local Installation Paths (Docker vs Native)](#local-installation-paths)
5. [Quick Start (Local)](#quick-start-local)
6. [Voice Modes](#three-voice-modes)
7. [Hardware Backends](#hardware-backends)
8. [Interfaces (CLI, UI, Python)](#interfaces)
9. [Configuration](#configuration)
10. [Troubleshooting & FAQ](#troubleshooting--faq)
11. [Documentation](#documentation)
12. [Developer & Architecture](#developer--architecture)

---

## Features & Capabilities

Qwen3-TTS isn't just a research script; it is engineered to be a production-ready synthesis engine. Here is what sets it apart:

### 🔒 Local-First by Design
* **Fully Local & Private:** All generation runs on your hardware — no cloud API calls, no audio or text leaving your machine. The only network traffic is between you and your own localhost server.
* **Three Modes, One Server:** Clone, Design, and Custom share a single persistent FastAPI server with on-demand model loading/unloading, so you can swap workflows without reloading anything.
* **Reproducible via Seeds:** When no seed is supplied the server generates one and echoes it back on every result (see [Seeds](#interfaces)); reuse a seed to reproduce a voice exactly.
* **Dual-Format Voice Prompts:** Voice prompts save as `.pt` (torch) and `.wav`+`.txt` (MLX) pairs, so a cloned voice works on both backends without re-creating it.

### ☁️ Zero-Setup Cloud Execution
* **1-Click Google Colab Deployment:** Test the entire system without installing a single package locally. The included Colab notebook automatically detects cloud GPUs, configures optimal performance settings, and generates a shareable public URL for the Web UI.

### 🎙️ The Audio Engine
* **Zero-Shot Voice Cloning:** Clone any human voice using just 5 to 15 seconds of reference audio. 
    * *Why it matters:* You don't need hours of clean studio data. Whisper auto-transcription handles the text extraction, or use `x_vector_only_mode` to clone entirely without a transcript — it works for both generation *and* voice-prompt creation, so a single audio sample with no transcript at all can become a reusable saved voice.
* **Prompt-Based Voice Design:** Generate entirely new voices by simply describing them (e.g., *"A warm, friendly British female voice speaking quickly"*). 
    * *Why it matters:* Gives you infinite creative control for video game NPCs, audiobooks, or brand personas without hiring a voice actor.
* **Premium Pre-Trained Speakers:** Includes 9 highly optimized, built-in voices for immediate plug-and-play generation.

### ⚙️ Production Architecture
* **Chunked Streaming:** Long texts are split into sentence-aligned chunks and streamed back as each chunk finishes, so playback can begin well before the full generation completes. (Chunks generate sequentially — expect ~40-70 s per chunk on MLX/M2 Pro; this is "start hearing audio sooner", not real-time synthesis.)
* **Serialized GPU Access:** Strict `asyncio` locking queues concurrent requests so model state is never corrupted under concurrent load — one generation at a time, by design.
* **Rate Limiting (on by default):** Per-endpoint limits via `slowapi` (a hard dependency) plus a global pre-auth ceiling on all routes, with reverse-proxy-aware IP resolution (`X-Forwarded-For` honored only for trusted proxies).
* **Smooth Multi-Chunk Audio:** Raised-cosine crossfade between audio chunks eliminates audible clicks at boundaries. Chunks are phase-aligned and level-matched before the crossfade; configurable silence gaps are also supported.
* **LUFS Normalization:** Optional EBU R128 loudness normalization via `pyloudnorm` for broadcast-ready audio output.

### 💻 Hardware Flexibility
* **Apple Silicon Native (MLX):** Deeply optimized for macOS. Utilizes unified memory to run fast, quiet, and cool on M1/M2/M3 chips without draining your battery.
* **High-Speed vLLM Integration:** Official support for vLLM-Omni on Linux/NVIDIA setups, yielding 3-4x faster generation speeds using PagedAttention memory management.

---

## System Requirements & Optimal Configurations

Qwen3-TTS performance relies heavily on your hardware and backend combination. Below are the exact recommended settings for each deployment type to avoid Out-of-Memory (OOM) crashes and maximize speed.

### General Base Requirements (All Local Systems)
* **Python:** 3.10+ (3.12 recommended for native installs)
* **Disk Space:** each of the three models costs ~2.5 GB (MLX 8-bit) to ~3.5 GB (torch) — ~7.5-10.5 GB total to cache Clone, Design, and Custom models

> ### ⚠️ Honest Caveats (Upstream Issues That Shape the Defaults)
> A few defaults here exist *because of* known upstream Qwen3-TTS issues, not by preference:
> * **Clone speed is post-hoc** (`generation.clone_speed`): the model's native rate control is broken for cloning (upstream #290), so speed changes time-stretch the generated audio instead.
> * **SDPA by default** (`advanced.attn_implementation: "auto"`): Flash Attention 2 can produce NaNs on Qwen3-TTS (upstream #333); it is opt-in only.
> * **ICL echo trim is effectively dormant** (`generation.trim_icl_echo`, default `true`): cloning sometimes re-speaks the reference transcript's tail before the requested text (upstream #341). The trim only fires when ASR is already loaded *and* a reference transcript is resolvable — no server/UI caller passes one yet — so it is safe to leave enabled but currently inert.

### 1. Apple Silicon (macOS)
| Hardware | Recommendation | Optimal `config.json` Settings |
| :--- | :--- | :--- |
| **Minimum** | Base M1 (8GB Unified Memory) | `backend: "mlx"`, `model_size: "0.6B"`, `mlx_quantization: "4bit"` |
| **Recommended** | M2/M3/M4 (16GB+ Unified) | `backend: "mlx"`, `model_size: "1.7B"`, `mlx_quantization: "8bit"` |

### 2. Standard PyTorch (Linux / Intel Mac)
| Hardware | Recommendation | Optimal `config.json` Settings |
| :--- | :--- | :--- |
| **Minimum** | NVIDIA T4 16GB | `backend: "torch"`, `model_size: "1.7B"`, `dtype: "float16"`, `compile_model: false` |
| **Recommended**| NVIDIA L4 / A10G (Ampere+) | `backend: "torch"`, `model_size: "1.7B"`, `dtype: "bfloat16"`, `compile_model: true` |

### 3. Production vLLM-Omni (Linux NVIDIA GPUs)
*Yields 3-4x faster generation. Requires Linux and CUDA.*
| Hardware | Recommendation | Optimal `config.json` Settings |
| :--- | :--- | :--- |
| **Minimum** | NVIDIA T4 16GB (Compute 7.0+) | `backend: "vllm"`, `model_size: "1.7B"`, `vllm_gpu_memory_utilization: 0.7` |
| **Recommended**| NVIDIA L4 / A100 (24GB+ VRAM)| `backend: "vllm"`, `model_size: "1.7B"`, `vllm_gpu_memory_utilization: 0.9` |

---

## Run on Google Colab (Zero Install)

The fastest way to use Qwen3-TTS is via the cloud. This requires **zero local installation**, keeps your hardware clean, and automatically leverages NVIDIA cloud GPUs.

A highly optimized notebook (`colab_notebook.ipynb`) is included in the repository. 

**Steps to Run:**
1. Upload the entire `Qwen3-TTS_UserFiles/` directory to your Google Drive (e.g., to `My Drive/Qwen3-TTS_UserFiles/`).
2. Open `colab_notebook.ipynb` in Google Colab.
3. Select a **GPU Runtime** (T4 is free, L4 is recommended for best performance).
4. Edit the **Settings Form** at the top of the notebook to configure your defaults (e.g., `1.7B` vs `0.6B` model).
5. **Run All Cells**.

**What the Notebook Automates:**
* Connects to your Google Drive to save your cloned voices persistently.
* Auto-detects the GPU tier and applies optimal settings (Flash Attention 2 for Ampere+, SDPA for Turing).
* Starts the FastAPI server in the background so Colab doesn't kill it.
* Generates a **Public Gradio URL** so you can access the Web UI from any browser.

---

## Local Installation Paths

If you prefer to run the system locally, there are two ways to install Qwen3-TTS. **Choose your path based on your hardware to avoid massive performance penalties.**

> ### ⚠️ Crucial Hardware Caveats
> * **Linux / NVIDIA Users:** You are highly encouraged to use **Path 1: Docker**. It keeps your host OS perfectly clean, prevents Python dependency conflicts, and ensures CUDA parity.
> * **Apple Silicon (M1/M2/M3) Users:** You **MUST** use **Path 2: Native Python**. Docker on macOS cannot access the Apple Neural Engine/GPU. Running this in Docker on a Mac will force CPU-only inference, destroying performance.

---

### Path 1: Docker (Recommended for Linux/NVIDIA)
*Zero host pollution. Requires Docker and `nvidia-container-toolkit`.*

1. Clone the repository:
```bash
git clone https://github.com/eepstein201/Qwen3-TTS-Advanced-EME.git
cd Qwen3-TTS-Advanced-EME
```

2. Build and run the production-ready container:
```bash
# For standard PyTorch backend
docker build -t qwen3-tts .
docker compose up -d

# OR, for high-speed vLLM-Omni backend (Recommended for heavy loads)
docker build -f Dockerfile.vllm -t qwen3-tts-vllm .
docker compose up -d
```
Your server is now running natively. The web UI is available at `http://localhost:7860` and the API at `http://localhost:5123`. Voice prompts and downloaded models are safely stored in Docker volumes.

---

### Path 2: Native Python (Required for macOS / Apple Silicon)
*Direct host installation using `pip` and the `hatchling` build system.*

1. Clone and navigate to the directory:
```bash
git clone https://github.com/eepstein201/Qwen3-TTS-Advanced-EME.git
cd Qwen3-TTS-Advanced-EME
```

2. Install based on your specific backend:
* **macOS (Apple Silicon / MLX):**
    ```bash
    pip install -e ".[mlx,server,audio,rich]"
    ```
* **Linux / Intel Mac (Standard PyTorch):**
    ```bash
    pip install -e ".[torch,server,audio,cuda,rich]"
    ```
* **Linux (vLLM-Omni for Maximum Speed):**
    ```bash
    pip install -e ".[torch,vllm,server,audio,cuda,rich]"
    ```

> ### ⚠️ The `torch` and `mlx` Extras Conflict — Never Install Both in One Environment
> The two extras pin incompatible `transformers` versions (MLX needs the newer line; torch/Qwen3-TTS weights need the older one). Installing both into the same environment breaks one of them. On a Mac where you want both backends, use **two separate conda environments** — `qwen3-tts` (torch) and `qwen3-tts-mlx` (MLX) — and install only that environment's extra into each:
> ```bash
> conda create -n qwen3-tts python=3.12 && conda run -n qwen3-tts pip install -e ".[torch,server,audio,rich]"
> conda create -n qwen3-tts-mlx python=3.12 && conda run -n qwen3-tts-mlx pip install -e ".[mlx,server,audio,rich]"
> ```
> Pick one backend per machine unless you specifically need both.

3. **Pro-Tip (Native Install Only): Add a Shell Alias**
To run the `tts` command from anywhere without polluting your global shell or manually activating environments, add this to your `~/.zshrc` or `~/.bashrc`:
```bash
# For standard Conda users
alias tts="conda run --no-capture-output -n qwen3-tts tts"

# For Apple Silicon Conda users
alias tts="conda run --no-capture-output -n qwen3-tts-mlx tts"
```

---

## Quick Start (Local)

Start the server in the background (models take 30-60s to load on first boot):
```bash
tts server start
```

Generate your first audio:
```bash
tts "Hello, world!" -o hello
```

Open the Web UI:
```bash
tts ui
```

Stop the server and free memory:
```bash
tts server stop
```

---

## Three Voice Modes

| Feature | Clone | Design | Custom |
|---------|-------|--------|--------|
| **Purpose** | Sound like a specific person | Describe a voice via text | 9 premium pre-trained voices |
| **Requires Audio?** | Yes (5-15s) | No | No |
| **Requires Text?** | Optional (`--no-transcript`) | Yes | No |
| **Style Control** | Post-processing | Description prompt | `--instruct` flag |

### 1. Clone Mode
```bash
# Create the voice
tts voice create recording.wav --name my_voice --auto-transcribe

# Use the voice
tts "Hello" -p my_voice.pt -o output
```

### 2. Design Mode
```bash
tts "Hello" -m design -d "A warm, friendly female voice with a slight British accent" -o output
```

### 3. Custom Mode
Built-in speakers: `ryan`, `aiden`, `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ono_anna`, `sohee`.
```bash
tts "Hello" -m custom -s ryan --prosody excited -o output
```

---

## Hardware Backends

You can configure your backend permanently via `tts config` or `config.json`. 

1.  **MLX (Apple Silicon):** Native macOS backend. Uses `.wav` + `.txt` file pairs for cloning. Quantization options: `4bit`, `5bit`, `6bit`, `8bit` (default), `bf16`.
2.  **PyTorch:** Standard backend. Auto-detects Flash Attention 2 on Ampere+ GPUs.
3.  **vLLM-Omni (Linux/NVIDIA):** 3-4x faster throughput. Safely managed via POSIX process groups to prevent zombie VRAM leaks.
    * *Usage:* set the backend first — `tts config edit --backend vllm` (or `TTS_BACKEND=vllm`) — then `tts server start` (`tts server start` accepts only `--public` / `--foreground`).

---

## Interfaces

### Command Line Interface (CLI)
The `tts` command is fully featured with Rich progress bars.
```bash
tts "Text" --stream -o output                        # Stream chunks as they generate
tts "Text" --preset creative -o output               # Use generation presets
tts "Text" --speed 1.2 --normalize -o processed      # Audio post-processing
tts "Text" --seed 12345 -o output                    # Reproducible generation (see below)
tts batch texts.json -o ~/Downloads/                 # Batch processing
tts srt subtitles.srt                                # SRT subtitles
tts dialogue script.json                             # Multi-speaker dialogue
tts watch ./inbox                                   # Watch a directory for .txt files
tts repl                                            # Interactive REPL
tts history 10                                      # Last 10 generations
tts server status                                    # Check memory/health
tts voice info my_voice                              # Prompt metadata (via server)
tts voice rebuild my_voice                           # Regenerate .pt prompts (torch-only)
tts config path                                      # Print config.json path
tts list speakers|presets|aliases|prosody|models|backends  # List reference data
```

**Seeds — reproducible voices:** when no `--seed` is supplied the server generates one and echoes it back on every result — the `"seed"` field on `/generate`, the `X-Seed` header on `/generate-stream`, the `"complete"` message on `/ws`, and `client.last_seed` in the Python API. The seed is recorded in the web UI's history; reuse it to reproduce a voice exactly.

### Gradio Web UI
Run `tts ui` to launch a local browser interface with tabs for Cloning, Designing, Custom Voices, Model Management, and Voice Prompt Management.
```bash
tts ui --port 8080 --share  # Run on custom port and generate public URL
```

**Where web-UI output goes:** generations are saved under `history_output_directory` (default `~/Downloads/Qwen3-TTS Output`) in an `Automated Output/` subfolder, each with a `.json` sidecar. In Recent Generations, **Remove** permanently deletes the file from disk (two-step, path-keyed confirm), and **Download** copies the `.wav` into a `Manual Downloads/` subfolder (confirm only on a name collision). CLI generation output (`output_directory`, default `~/Downloads`) is unaffected.

### Python API
You can integrate Qwen3-TTS directly into your Python apps.
```python
from qwen3_tts.server.client import TTSClient

client = TTSClient()
audio_path = client.generate(
    "Hello world",
    mode="clone",
    voice="narrator",
    output="output.wav"
)

# Real-time streaming
for wav_chunk, sr in client.generate_streaming("Long text...", output="stream.wav"):
    pass 
```

---

## Configuration

Settings are stored in `config.json`. Use `tts config edit` to modify settings:

```bash
tts config edit --backend mlx           # Set backend
tts config edit --model-size 0.6B       # Set model size
tts config edit --mlx-quantization 4bit # Set MLX quantization
tts config edit --language Spanish      # Set default language
tts config edit                          # Interactive voice description editor
```

| Key | Default | Description |
|-----|---------|-------------|
| `advanced.backend` | Auto | `mlx`, `torch`, or `vllm` |
| `advanced.model_size` | `1.7B` | `1.7B` (High fidelity) or `0.6B` (Fast/Light) |
| `advanced.torch_quantization` | `none` | PyTorch backend quantization: `none`, `8bit`, `4bit` |
| `advanced.mlx_quantization` | `8bit` | MLX backend quantization: `4bit`, `5bit`, `6bit`, `8bit`, `bf16` |
| `advanced.attn_implementation` | `auto` | `auto` (SDPA), `sdpa`, `flash_attention_2`, `eager`. FA2 is opt-in only — it can produce NaNs on Qwen3-TTS (upstream #333) |
| `advanced.vllm_gpu_memory_utilization`| `0.7` | VRAM reservation for vLLM (0.1 - 1.0) |
| `cache.voice_prompt_max` | `10` | Max voice prompts cached in memory (LRU) |
| `cache.generation_max` | `5` | Max generation results cached (SHA256 key) |
| `cache.eta_ttl_seconds` | `30` | ETA cache TTL in seconds |
| `generation.max_chunk_chars` | `500` | Auto-splits long texts (0 disables). Prevents *silent truncation*: a single MLX generate call is capped at `max_new_tokens=2048` (~170 s of audio), so unchunked long text cuts off mid-sentence |
| `generation.max_chunk_tokens` | `200` | Max tokens per chunk (torch backend) |
| `generation.temperature` | `0.7` | Higher = more varied output |
| `generation.silence_gap_seconds` | `0.0` | Silence between chunks (0 uses a phase-aligned, level-matched crossfade) |
| `generation.lufs_normalize` | `false` | Apply EBU R128 loudness normalization after combining chunks |
| `generation.lufs_target` | `-16.0` | Target loudness in LUFS (used only when `lufs_normalize` is `true`) |
| `generation.clone_speed` | unset | Clone-mode-only post-hoc rate (0.5-2.0). The model's native rate control is broken for cloning (upstream #290), so speed time-stretches the generated audio. Design/custom keep native `instruct` rate control |
| `generation.trim_icl_echo` | `true` | Clips ICL echo of the reference transcript (upstream #341). Only fires when ASR is already loaded *and* a reference transcript is resolvable — currently dormant |
| `history_output_directory` | `~/Downloads/Qwen3-TTS Output` | Root for web-UI output (`Automated Output/` for generations, `Manual Downloads/` for kept files) |
| `models.<type>.revision` | `main` | HuggingFace branch/tag/SHA for `clone`/`design`/`custom` model downloads |
| `security.rate_limits.generate` | `10/minute` | Rate limit for generation endpoints |
| `security.rate_limits.model_ops` | `5/minute` | Rate limit for model management endpoints |

*You can use environment variables to override settings per session (e.g., `TTS_BACKEND=vllm tts "text"`).*

---

## Rate Limiting

Rate limiting is **on by default** (no opt-in needed) to prevent abuse and ensure fair resource allocation. It uses **slowapi** — a hard dependency of the `server` extra, so there is nothing extra to install — with support for multiple strategies: per-IP, per-token, and hybrid (both). A global, per-IP, pre-auth ceiling on **all** routes (default `120/minute` via `SlowAPIMiddleware`) also applies, so unauthenticated floods cannot bypass the per-route limits entirely.

### Configuration

Rate limits are configured in `config.json` under `security.rate_limits` (defaults shown):

```json
{
  "security": {
    "rate_limits": {
      "generate": "10/minute",
      "model_ops": "5/minute",
      "transcribe": "10/minute",
      "prompt_ops": "10/minute",
      "config_ops": "2/minute",
      "global": "120/minute"
    }
  }
}
```

**Environment overrides** (read once at server import — restart to apply):
* `TTS_DISABLE_RATE_LIMITING=1` disables every limiter (test/CI only)
* `TTS_RATE_LIMIT_GENERATE`, `TTS_RATE_LIMIT_MODEL_OPS`, `TTS_RATE_LIMIT_TRANSCRIBE`, `TTS_RATE_LIMIT_PROMPT_OPS`, `TTS_RATE_LIMIT_CONFIG_OPS`, `TTS_RATE_LIMIT_GLOBAL` override individual limits (e.g. `TTS_RATE_LIMIT_GENERATE=120/minute`)
* `TTS_TRUSTED_PROXIES` (comma-separated IPs, loopback by default): `X-Forwarded-For` is honored for client-IP keying only when the direct TCP peer is in this allowlist — set it when running behind a reverse proxy

The `global` ceiling is deliberately decoupled from the `generate` limit: it must stay above the Gradio UI's ~24/minute `/health`+`/models` polling, or `/health` starts returning 429 and the UI reports "Disconnected / Server not running".

### Rate Limit Strategies

- **Hybrid** (default): Enforces both per-IP and per-token limits simultaneously
- **Per-IP**: Rate limits based on client IP address only
- **Per-Token**: Rate limits based on authentication token only

### Protected Endpoints

Effective limits as actually applied (a few endpoints currently use a different limit than their named category — noted in the table):

| Endpoint | Effective limit | Strategy |
|----------|-------|----------|
| `/generate`, `/generate-stream` | `generate` (default 10/minute) | Hybrid |
| `/load-model`, `/unload-model` | `model_ops` (default 5/minute) | Hybrid |
| `/update-model-config` | `model_ops` | Hybrid |
| `/load-asr`, `/unload-asr` | `model_ops` | Hybrid |
| `/transcribe` | uses the `generate` limit (its own `transcribe` category is defined but not yet applied) | Hybrid |
| `/create-voice-prompt` | uses the `model_ops` limit (its own `prompt_ops` category is defined but not yet applied) | Hybrid |
| `/delete-prompt`, `/rename-prompt` | hardcoded `10/minute` | Hybrid |
| `/update-startup-config` | hardcoded `2/minute` (its own `config_ops` category is defined but not yet applied) | Hybrid |
| All routes (pre-auth ceiling) | `global` (default 120/minute) | Per-IP |

### Error Responses

When rate limits are exceeded, the server returns HTTP 429:

```json
{
  "detail": "Rate limit exceeded"
}
```

### Testing Rate Limits

Test rate limiting with curl:

```bash
# Set auth token
export TOKEN="your-auth-token"

# Send repeated requests (should hit 429 after limit)
for i in {1..25}; do
  curl -X POST http://localhost:5123/generate \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"text": "test"}'
done
```

---

## Utility Commands

### Health Check
```bash
tts doctor          # Run diagnostics
```
Checks installation, dependencies, model cache, and server status.

### Cache Management
```bash
tts cache list      # Show cached models
tts cache size      # Show disk usage
tts cache prune     # Remove old/unused entries
tts cache clear     # Empty entire cache
```
Voice prompts and generation results are cached for performance. Use these commands to manage disk usage.

### Uninstall / Cleanup
```bash
tts uninstall models    # Remove downloaded models (~10GB)
tts uninstall voices    # Remove voice prompts
tts uninstall config    # Remove config.json
tts uninstall all       # Full cleanup (everything above)
tts uninstall environment  # Print conda-env removal commands (does not remove them)
```

---

## API Endpoints

The FastAPI server (port 5123) provides the following endpoints:

| Endpoint | Auth Required | Description |
|----------|---------------|-------------|
| `GET /health` | No | Health check (always available) |
| `GET /ready` | No | Kubernetes readiness probe (503 while loading) |
| `GET /generation-status` | No | Poll generation progress |
| `GET /queue-status` | No | Request queue length and active status |
| `POST /generate` | Yes | Generate audio (JSON response, see below) |
| `POST /generate-stream` | Yes | Stream audio chunks (float32 PCM, see below) |
| `WebSocket /ws` | Yes | Bidirectional real-time TTS streaming (auth via first message) |
| `POST /load-model` | Yes | Load a model on-demand |
| `POST /unload-model` | Yes | Unload model to free memory |
| `POST /update-model-config` | Yes | Change model size, quantization, audio loader |
| `POST /update-startup-config` | Yes | Set which models load at startup |
| `GET /models` | Yes | List model status and memory |
| `POST /load-asr` | Yes | Load ASR model for transcription |
| `POST /unload-asr` | Yes | Unload ASR model to free memory |
| `POST /transcribe` | Yes | Transcribe audio to text using ASR |
| `GET /prompts` | Yes | List voice prompts (supports `offset`/`limit` pagination) |
| `POST /create-voice-prompt` | Yes | Create voice clone prompt from uploaded audio |
| `POST /delete-prompt` | Yes | Delete a voice prompt |
| `POST /rename-prompt` | Yes | Rename a voice prompt |
| `GET /preview-prompt` | Yes | Return .wav audio bytes for a prompt |
| `GET /prompt-details` | Yes | Prompt metadata (formats, size, duration) |
| `GET /stats` | Yes | Memory and cache statistics |
| `POST /cancel-generation` | Yes | Cancel active generation |
| `POST /shutdown` | Yes | Graceful server shutdown |

Authentication uses Bearer tokens from `~/.config/qwen3-tts/.voice_server_token` (legacy fallback: `~/.voice_server_token`).

**WebSocket `/ws` — bidirectional real-time streaming.** Beyond the HTTP endpoints, `/ws` carries full-duplex generation: the client authenticates with its first message, then sends text and receives binary audio chunks as they are produced. It supports live cancellation — a concurrent watcher reads frames during generation and distinguishes an explicit cancel from a client disconnect — and validates the browser `Origin` header against the CORS allowlist before the socket is accepted. The generation seed is returned on the `"complete"` message.

### `POST /generate` Response

Returns JSON (not a binary audio stream):

```json
{
  "results": [
    {
      "index": 0,
      "audio_base64": "...",
      "sample_rate": 24000,
      "chunks": 1,
      "seed": 1234567890
    }
  ]
}
```

Each result contains base64-encoded WAV audio, the number of chunks it was built from, and the generation seed (server-generated when not supplied — reuse it to reproduce the output). Decode with standard base64 libraries. Long texts are chunked automatically, producing multiple results.

### `POST /generate-stream` Wire Format

The response body is a sequence of **length-prefixed binary frames**, one per audio chunk — not a raw float32 stream. Each frame is:

```
[sample_rate: 4 bytes, uint32 LE][length: 4 bytes, uint32 LE][payload: `length` bytes]
```

The payload is float32 little-endian PCM samples at `sample_rate` Hz (typically 24000). Because Starlette commits the 200 response headers before the body streams, a mid-stream failure cannot change the status code — instead the server emits a **terminal error frame** with `sample_rate == 0` (never valid for real audio) whose payload is JSON `{"error": "...", "code": "..."}`. Treat that frame as an error, not audio. On success the `X-Seed` response header carries the generation seed.

```python
# Minimal Python consumer
import json, struct, httpx

def read_exact(r, n):
    buf = b""
    while len(buf) < n:
        part = r.read(n - len(buf))
        if not part:
            raise EOFError("stream ended mid-frame")
        buf += part
    return buf

with httpx.stream("POST", url, json=payload, headers=headers) as r:
    r.raise_for_status()
    raw = r.raw  # sync byte stream
    while True:
        try:
            header = read_exact(raw, 8)
        except EOFError:
            break  # clean end of stream
        sample_rate, length = struct.unpack("<II", header)
        payload = read_exact(raw, length)
        if sample_rate == 0:
            err = json.loads(payload)   # terminal error frame
            raise RuntimeError(f"server error: {err['error']} ({err['code']})")
        samples = struct.unpack(f"<{length // 4}f", payload)
        play(samples, sample_rate)
```

---

## Troubleshooting & FAQ

* **Server won't start:** Run `tts server log` to view the error trace.
* **CUDA Out of Memory:** The server locks concurrent requests to prevent this. If it happens on boot, run `tts server stop` to clear VRAM, then switch to the lighter model: `tts config edit --model-size 0.6B`. If using vLLM, lower the `vllm_gpu_memory_utilization` to `0.5`.
* **Subprocess/vLLM won't die:** The FastAPI server traps `SIGTERM` and kills the process group. If you force-killed the terminal (`kill -9`), find the PID with `nvidia-smi` and kill it manually.
* **Bad Audio Quality:** Lower the temperature (`--temperature 0.5`) or use `--preset consistent`. Ensure your reference audio has absolutely zero background noise.

---

## Documentation

Start at the **[documentation index](docs/README.md)**, which links every reference guide, the live roadmaps, and historical plans. Key guides:

### [Command Reference](docs/COMMANDS.md)
Complete CLI command reference for all `tts` commands, including:
- Installation and setup commands
- Core CLI commands (generate, server, UI)
- Voice management commands
- Configuration commands
- Advanced commands (batch, SRT, dialogue, REPL)
- Cache management commands
- Testing commands
- Code quality commands

### [Configuration Reference](docs/CONFIG.md)
Complete environment variable and `config.json` reference, including:
- Environment variables (HF_HOME, ANTHROPIC_API_KEY, etc.)
- Models section (clone, design, custom configuration)
- Advanced section (backend, model size, quantization, audio loader)
- Generation settings (silence gaps, chunking, LUFS normalization)
- Rate limiting configuration
- Server settings (host, port, auto-shutdown)
- Prompt enhancer (AI voice description)
- Generation presets and prosody presets
- Cache and UI settings

### [Contributing Guide](docs/CONTRIBUTING.md)
Development environment setup and testing procedures, including:
- Prerequisites (Python, Conda, Git)
- Installation steps (MLX and Torch backends)
- Available scripts and development tools
- Project structure overview
- Testing procedures (2000+ tests across 6 batches)
- Code style enforcement (black, ruff, mypy)
- Development workflow (feature branches, commits, PRs)
- Troubleshooting common issues

### [Operations Runbook](docs/RUNBOOK.md)
Deployment and operational procedures, including:
- Local development setup
- Server deployment (PM2, health checks)
- Model management (load/unload, status checks)
- Backup and recovery procedures
- Monitoring and alerting
- Troubleshooting common issues
- Maintenance tasks (weekly, monthly, quarterly)
- Upgrade procedures
- Security considerations
- Performance tuning

### [Rate Limiting Guide](docs/rate-limiting.md)
Deep dive on the slowapi-based rate-limiting architecture: per-IP / per-token / hybrid strategies, the `security.rate_limits` config format, and testing rate limits.

---

## Developer & Architecture

### Engine Architecture

The core engine (`qwen3_tts/core/engine/`) is organized as 6 submodules following a strict dependency DAG:

| Module | Purpose |
|--------|---------|
| `text_processing.py` | Text normalization, sentence splitting, currency expansion |
| `audio_processing.py` | Audio I/O, effects, silence trimming, LUFS normalization |
| `voice_prompt.py` | Voice prompt loading and LRU caching |
| `model_loader.py` | Torch/MLX model loading with warm-up |
| `inference.py` | Stateless generation dispatch with crossfade |
| `asr.py` | Whisper-based speech recognition |

All public functions are re-exported through `engine/__init__.py` for backward compatibility.

### Testing

## Running Tests

### Universal Test Installation

Tests now work in **any** Python environment with a single installation command:

```bash
pip install -e ".[test]"
```

This installs all required dependencies including gradio, pytest, and playwright. No conda environment required!

### Test Execution

**Run all tests using the batch runner (2000+ tests across 100+ modules, organized in 6 batches):
```bash
python tests/run_batches.py        # Run all batches
python tests/run_batches.py --batch 1  # Run a specific batch
make test-batch                    # Or use the Makefile
```

**Run specific test modules directly:**
```bash
python -m unittest tests.test_voice_generation -v
python -m unittest tests.test_voice_prompts -v
```

**Platform Support:**
- ✅ **Linux CPU** - Full test suite support
- ✅ **Linux GPU/CUDA** - Full test suite with CUDA detection  
- ✅ **macOS MLX** - Full test suite with MLX backend support
- ✅ **Google Colab** - Full test suite support (free/pro tiers)
- ✅ **Docker** - Containerized testing support

**Full suite runner** (multi-environment testing with server lifecycle management):
```bash
python tests/run_full_suite.py --full --env mlx    # MLX environment only
python tests/run_full_suite.py --full --env torch   # Torch environment only
python tests/run_full_suite.py --full --env all     # Both environments
python tests/run_full_suite.py --full --test-type unit  # Unit tests only
python tests/run_full_suite.py --full --dry-run     # Preview without running
```

**Migration from conda-only testing:**
```bash
# Old way (still works):
conda activate qwen3-tts-mlx
python -m unittest tests.test_module -v

# New way (works anywhere):
pip install -e ".[test]"  # One-time setup
python -m unittest tests.test_module -v  # Works in any environment!
```

### Troubleshooting

**Tests fail with import errors:**
```bash
# Ensure test dependencies are installed
pip install -e ".[test]"
```

**Specific test module not found:**
```bash
# Run from project root directory
cd /path/to/Qwen3-TTS_UserFiles
python -m unittest tests.test_module_name -v
```

## License
The source code in this repository is licensed under Apache 2.0. The Qwen3-TTS model weights are subject to their own license provided by Qwen Research/Alibaba Cloud.