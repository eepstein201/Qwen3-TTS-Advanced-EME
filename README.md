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
11. [Developer & Architecture](#developer--architecture)

---

## Features & Capabilities

Qwen3-TTS isn't just a research script; it is engineered to be a production-ready synthesis engine. Here is what sets it apart:

### ☁️ Zero-Setup Cloud Execution
* **1-Click Google Colab Deployment:** Test the entire system without installing a single package locally. The included Colab notebook automatically detects cloud GPUs, configures optimal performance settings, and generates a shareable public URL for the Web UI.

### 🎙️ The Audio Engine
* **Zero-Shot Voice Cloning:** Clone any human voice using just 5 to 15 seconds of reference audio. 
    * *Why it matters:* You don't need hours of clean studio data. Whisper auto-transcription handles the text extraction, or use "embedding-only" mode to clone a voice without a transcript.
* **Prompt-Based Voice Design:** Generate entirely new voices by simply describing them (e.g., *"A warm, friendly British female voice speaking quickly"*). 
    * *Why it matters:* Gives you infinite creative control for video game NPCs, audiobooks, or brand personas without hiring a voice actor.
* **Premium Pre-Trained Speakers:** Includes 9 highly optimized, built-in voices for immediate plug-and-play generation.

### ⚙️ Production Architecture
* **True Zero-Latency Streaming:** Built on FastAPI with asynchronous queues. Streams raw audio bytes back to the client the millisecond they are inferred.
* **Bulletproof GPU Serialization:** Strict `asyncio` locking queues concurrent requests, ensuring your GPU never crashes under high web traffic.
* **Rate Limiting:** Optional per-endpoint rate limiting via `slowapi` with reverse-proxy-aware IP resolution (supports Colab tunnels, Gradio share links).
* **Smooth Multi-Chunk Audio:** Raised-cosine crossfade between audio chunks eliminates audible clicks at boundaries. Configurable silence gaps also supported.
* **LUFS Normalization:** Optional EBU R128 loudness normalization via `pyloudnorm` for broadcast-ready audio output.

### 💻 Hardware Flexibility
* **Apple Silicon Native (MLX):** Deeply optimized for macOS. Utilizes unified memory to run fast, quiet, and cool on M1/M2/M3 chips without draining your battery.
* **High-Speed vLLM Integration:** Official support for vLLM-Omni on Linux/NVIDIA setups, yielding 3-4x faster generation speeds using PagedAttention memory management.

---

## System Requirements & Optimal Configurations

Qwen3-TTS performance relies heavily on your hardware and backend combination. Below are the exact recommended settings for each deployment type to avoid Out-of-Memory (OOM) crashes and maximize speed.

### General Base Requirements (All Local Systems)
* **Python:** 3.10+ (3.12 recommended for native installs)
* **Disk Space:** ~3 GB per model (Total ~10 GB to cache Clone, Design, and Custom models)

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
git clone [https://github.com/your-repo/Qwen3-TTS.git](https://github.com/your-repo/Qwen3-TTS.git)
cd Qwen3-TTS
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
git clone [https://github.com/your-repo/Qwen3-TTS.git](https://github.com/your-repo/Qwen3-TTS.git)
cd Qwen3-TTS
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

1.  **MLX (Apple Silicon):** Native macOS backend. Uses `.wav` + `.txt` file pairs for cloning. Quantization options: `4bit`, `8bit` (default), `bf16`.
2.  **PyTorch:** Standard backend. Auto-detects Flash Attention 2 on Ampere+ GPUs.
3.  **vLLM-Omni (Linux/NVIDIA):** 3-4x faster throughput. Safely managed via POSIX process groups to prevent zombie VRAM leaks.
    * *Usage:* `tts server start --backend vllm`

---

## Interfaces

### Command Line Interface (CLI)
The `tts` command is fully featured with Rich progress bars.
```bash
tts "Text" --stream -o output                        # Stream real-time
tts "Text" --preset creative -o output               # Use generation presets
tts "Text" --speed 1.2 --normalize -o processed      # Audio post-processing
tts batch texts.json -o ~/Downloads/                 # Batch processing
tts server status                                    # Check memory/health
```

### Gradio Web UI
Run `tts ui` to launch a local browser interface with tabs for Cloning, Designing, Custom Voices, Model Management, and Voice Prompt Management.
```bash
tts ui --port 8080 --share  # Run on custom port and generate public URL
```

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

Settings are stored in `config.json`.

| Key | Default | Description |
|-----|---------|-------------|
| `advanced.backend` | Auto | `mlx`, `torch`, or `vllm` |
| `advanced.model_size` | `1.7B` | `1.7B` (High fidelity) or `0.6B` (Fast/Light) |
| `advanced.torch_quantization` | `none` | PyTorch backend quantization: `none`, `8bit`, `4bit` |
| `advanced.mlx_quantization` | `8bit` | MLX backend quantization: `4bit`, `8bit`, `bf16` |
| `advanced.vllm_gpu_memory_utilization`| `0.7` | VRAM reservation for vLLM (0.1 - 1.0) |
| `cache.voice_prompt_max` | `10` | Max voice prompts cached in memory (LRU) |
| `cache.generation_max` | `5` | Max generation results cached (SHA256 key) |
| `cache.eta_ttl_seconds` | `30` | ETA cache TTL in seconds |
| `generation.max_chunk_chars` | `500` | Auto-splits long texts to prevent timeouts |
| `generation.max_chunk_tokens` | `200` | Max tokens per chunk (torch backend) |
| `generation.temperature` | `0.7` | Higher = more varied output |
| `generation.silence_gap_seconds` | `0.0` | Silence between chunks (0 uses crossfade) |
| `security.rate_limits.generate` | `10/minute` | Rate limit for generation endpoints |
| `security.rate_limits.model_ops` | `5/minute` | Rate limit for model management endpoints |

*You can use environment variables to override settings per session (e.g., `TTS_BACKEND=vllm tts "text"`).*

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
tts uninstall environment  # Remove conda envs
```

---

## API Endpoints

The FastAPI server (port 5123) provides the following endpoints:

| Endpoint | Auth Required | Description |
|----------|---------------|-------------|
| `GET /health` | No | Health check (always available) |
| `GET /ready` | No | Kubernetes readiness probe (503 while loading) |
| `GET /generation-status` | No | Poll generation progress |
| `POST /generate` | Yes | Generate audio (JSON response, see below) |
| `POST /generate-stream` | Yes | Stream audio chunks (float32 PCM, see below) |
| `POST /load-model` | Yes | Load a model on-demand |
| `POST /unload-model` | Yes | Unload model to free memory |
| `POST /update-model-config` | Yes | Change model size, quantization, audio loader |
| `POST /update-startup-config` | Yes | Set which models load at startup |
| `GET /models` | Yes | List model status and memory |
| `GET /prompts` | Yes | List voice prompts (supports `offset`/`limit` pagination) |
| `POST /delete-prompt` | Yes | Delete a voice prompt |
| `POST /rename-prompt` | Yes | Rename a voice prompt |
| `GET /preview-prompt` | Yes | Return .wav audio bytes for a prompt |
| `GET /prompt-details` | Yes | Prompt metadata (formats, size, duration) |
| `GET /stats` | Yes | Memory and cache statistics |
| `POST /cancel-generation` | Yes | Cancel active generation |
| `POST /shutdown` | Yes | Graceful server shutdown |

Authentication uses Bearer tokens from `~/.voice_server_token`.

### `POST /generate` Response

Returns JSON (not a binary audio stream):

```json
{
  "results": [
    {
      "index": 0,
      "audio_base64": "...",
      "sample_rate": 24000
    }
  ]
}
```

Each result contains base64-encoded WAV audio. Decode with standard base64 libraries. Long texts are chunked automatically, producing multiple results.

### `POST /generate-stream` Wire Format

Streams raw float32 little-endian samples as binary chunks at 24000 Hz:

```python
# Python consumer
import struct, httpx
with httpx.stream("POST", url, json=payload, headers=headers) as r:
    for chunk in r.iter_bytes():
        samples = struct.unpack(f'<{len(chunk)//4}f', chunk)
```

---

## Troubleshooting & FAQ

* **Server won't start:** Run `tts server log` to view the error trace.
* **CUDA Out of Memory:** The server locks concurrent requests to prevent this. If it happens on boot, run `tts server stop` to clear VRAM, then switch to the lighter model: `tts config` -> `0.6B`. If using vLLM, lower the `vllm_gpu_memory_utilization` to `0.5`.
* **Subprocess/vLLM won't die:** The FastAPI server traps `SIGTERM` and kills the process group. If you force-killed the terminal (`kill -9`), find the PID with `nvidia-smi` and kill it manually.
* **Bad Audio Quality:** Lower the temperature (`--temperature 0.5`) or use `--preset consistent`. Ensure your reference audio has absolutely zero background noise.

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

Run the test suite using the batch runner (560+ tests across 17 test files, organized in 5 batches):
```bash
python tests/run_batches.py        # Run all batches
python tests/run_batches.py --batch 1  # Run a specific batch
make test-batch                    # Or use the Makefile
```

## License
The source code in this repository is licensed under Apache 2.0. The Qwen3-TTS model weights are subject to their own license provided by Qwen Research/Alibaba Cloud.