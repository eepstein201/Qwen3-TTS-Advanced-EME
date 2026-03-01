# Qwen3-TTS Voice Generation System

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Version](https://img.shields.io/badge/version-3.0.0-green)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Colab-lightgrey)

Clone any voice from an audio sample, design voices from text descriptions, or choose from 9 premium speakers. Powered by Qwen3-TTS models with a high-concurrency, asynchronous FastAPI server.

## Table of Contents
1. [Features](#features)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Voice Modes](#three-voice-modes)
6. [Hardware Backends](#hardware-backends)
7. [Interfaces (CLI, UI, Python)](#interfaces)
8. [Deployment (Docker & Colab)](#deployment)
9. [Configuration](#configuration)
10. [Troubleshooting & FAQ](#troubleshooting--faq)
11. [Developer & v3.0 Architecture](#developer--v30-architecture)

---

## Features

* **Zero-Shot Voice Cloning:** Record 5-15 seconds of audio to clone a voice. Whisper automatically handles transcription, or use embedding-only mode to skip transcripts entirely.
* **Voice Design:** Generate unique voices purely from text descriptions (e.g., *"A warm British female voice"*).
* **Apple Silicon Native:** Optimized MLX backend for macOS. Runs fast on unified memory with lower thermals.
* **Production-Grade Server:** Built on FastAPI/Uvicorn. Features real streaming, GPU concurrency locks to prevent Out-Of-Memory (OOM) crashes, and worker-safe state.
* **High-Speed vLLM Integration:** 3-4x faster inference on NVIDIA GPUs using the official vLLM-Omni backend.

---

## System Requirements

| Resource | Minimum | Recommended (1.7B Model) |
|----------|---------|--------------------------|
| **Python** | 3.10 | 3.12 |
| **RAM** | 8 GB | 16 GB |
| **Disk** | ~3 GB per model | ~10 GB (all 3 models) |
| **macOS** | Apple Silicon M1 | M2+ |
| **Linux/Colab** | NVIDIA T4+ (CUDA 7.0+) | A10G / L4 / A100 |

> **Note:** Use the `0.6B` model variants if you are resource-constrained. They use half the memory and run ~40% faster than the default `1.7B` models.

---

## Installation

This is a pure Python package using the `hatchling` build system.

**1. Clone the repository and navigate to the directory:**
```bash
cd ~/Qwen3-TTS_UserFiles
```

**2. Install based on your hardware:**

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

**3. Pro-Tip: Add a Shell Alias**
To run the `tts` command from anywhere without polluting your global shell or manually activating environments, add this to your `~/.zshrc` or `~/.bashrc`:

```bash
# For Conda users
alias tts="conda run --no-capture-output -n qwen3-tts tts"
# (Use 'qwen3-tts-mlx' for Apple Silicon envs)
```

---

## Quick Start

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

## Deployment

### Docker
Production-ready containers are available. We do not use editable installs (`-e`) in the Dockerfile to ensure clean, lean images.
```bash
docker build -t qwen3-tts .
docker compose up -d
```

### Google Colab
Use `colab_notebook.ipynb` to run this in the cloud. It features automatic hardware detection, optimal PyTorch compilation for Ampere GPUs, and runs FastAPI in the foreground to prevent Colab from killing the process.

---

## Configuration

Settings are stored in `config.json`.

| Key | Default | Description |
|-----|---------|-------------|
| `advanced.backend` | Auto | `mlx`, `torch`, or `vllm` |
| `advanced.model_size` | `1.7B` | `1.7B` (High fidelity) or `0.6B` (Fast/Light) |
| `advanced.vllm_gpu_memory_utilization`| `0.7` | VRAM reservation for vLLM (0.1 - 1.0) |
| `generation.max_chunk_chars` | `500` | Auto-splits long texts to prevent timeouts |
| `generation.temperature` | `0.7` | Higher = more varied output |

*You can use environment variables to override settings per session (e.g., `TTS_BACKEND=vllm tts "text"`).*

---

## Troubleshooting & FAQ

* **Server won't start:** Run `tts server log` to view the error trace.
* **CUDA Out of Memory:** The server locks concurrent requests to prevent this. If it happens on boot, run `tts server stop` to clear VRAM, then switch to the lighter model: `tts config` -> `0.6B`. If using vLLM, lower the `vllm_gpu_memory_utilization` to `0.5`.
* **Subprocess/vLLM won't die:** The FastAPI server traps `SIGTERM` and kills the process group. If you force-killed the terminal (`kill -9`), find the PID with `nvidia-smi` and kill it manually.
* **Bad Audio Quality:** Lower the temperature (`--temperature 0.5`) or use `--preset consistent`. Ensure your reference audio has absolutely zero background noise.

---

## Developer & v3.0 Architecture

Run the test suite using `pytest` (23+ tests covering endpoints, metadata, and core infra):
```bash
pytest tests/ -v
```

### v3.0 Release Notes
* **Flask to FastAPI:** Replaced synchronous Flask with ASGI FastAPI.
* **True Streaming:** Replaced fake chunk collection with `asyncio.Queue` and thread-safe event loops for zero-latency audio streaming.
* **Thread Safety:** Eliminated global variables. State and models are now isolated in `app.state` to support Uvicorn multi-worker deployments.
* **GPU Serialization:** Introduced strict `asyncio.Lock()` to prevent concurrent API requests from crashing the GPU.
* **Package Modernization:** Migrated from `setuptools` to `hatchling`. Eliminated 11 legacy bash shims in favor of standard Python entry points (`[project.scripts]`).

## License
The source code in this repository is licensed under Apache 2.0. The Qwen3-TTS model weights are subject to their own license provided by Qwen Research/Alibaba Cloud.