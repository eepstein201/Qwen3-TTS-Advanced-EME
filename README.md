# Qwen3-TTS Voice Generation System

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Version](https://img.shields.io/badge/version-3.0.0-green)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Colab-lightgrey)

Clone any voice from an audio sample, design voices from text descriptions, or choose from 9 premium speakers. Powered by Qwen3-TTS models with a high-concurrency, asynchronous FastAPI server.

## Table of Contents
1. [Features & Capabilities](#features--capabilities)
2. [System Requirements & Optimal Configurations](#system-requirements--optimal-configurations)
3. [Installation Paths (Docker vs Native)](#installation-paths)
4. [Quick Start](#quick-start)
5. [Voice Modes](#three-voice-modes)
6. [Hardware Backends](#hardware-backends)
7. [Interfaces (CLI, UI, Python)](#interfaces)
8. [Google Colab](#google-colab)
9. [Configuration](#configuration)
10. [Troubleshooting & FAQ](#troubleshooting--faq)
11. [Developer & v3.0 Architecture](#developer--v30-architecture)

---

## Features & Capabilities

Qwen3-TTS isn't just a research script; it is engineered to be a production-ready synthesis engine. Here is what sets it apart:

### 🎙️ The Audio Engine
* **Zero-Shot Voice Cloning:** Clone any human voice using just 5 to 15 seconds of reference audio. 
    * *Why it matters:* You don't need hours of clean studio data or expensive model fine-tuning. Whisper auto-transcription handles the text extraction, or you can use "embedding-only" mode to clone a voice without knowing the transcript at all.
* **Prompt-Based Voice Design:** Generate entirely new, unique voices by simply describing them in text (e.g., *"A warm, friendly British female voice speaking quickly"*). 
    * *Why it matters:* Gives you infinite creative control for video game NPCs, audiobooks, or brand personas without ever needing to hire a voice actor.
* **Premium Pre-Trained Speakers:** Includes 9 highly optimized, built-in voices.
    * *Why it matters:* Plug-and-play readiness. If you just need a high-quality voice immediately, you can bypass cloning/designing entirely.

### ⚙️ Production Architecture (v3.0)
* **True Zero-Latency Streaming:** Built on FastAPI with asynchronous queues.
    * *Why it matters:* Unlike systems that wait for the entire audio file to generate before playing, this streams raw audio bytes back to the client the millisecond they are inferred. Essential for real-time conversational AI or interactive agents.
* **Bulletproof GPU Serialization:** Strict `asyncio` locking mechanisms for hardware access.
    * *Why it matters:* AI models are notorious for Out-Of-Memory (OOM) crashes when multiple users hit an endpoint simultaneously. This server queues concurrent requests, ensuring your GPU never crashes under high traffic.

### 💻 Hardware Flexibility
* **Apple Silicon Native (MLX):** Deeply optimized for macOS using the MLX framework.
    * *Why it matters:* Most open-source TTS runs terribly on Macs. This utilizes unified memory to run fast, quiet, and cool on M1/M2/M3 chips without draining your battery or requiring an expensive cloud GPU.
* **High-Speed vLLM Integration:** Official support for vLLM-Omni on Linux/NVIDIA setups.
    * *Why it matters:* Yields 3-4x faster generation speeds using PagedAttention memory management. If you are deploying this to a production server, this dramatically reduces your compute costs and latency.

### 🛠️ Developer Experience
* **Tri-Interface Access:** A Gradio Web UI for visual control, a robust CLI for bash scripting, and a Python client for direct app integration.
    * *Why it matters:* Adapts perfectly to your workflow, whether you are a non-technical creator, a power-user running cron jobs, or an engineer building a SaaS product.

---

## System Requirements & Optimal Configurations

Qwen3-TTS performance relies heavily on your hardware and backend combination. Below are the minimum requirements and the exact recommended settings for each deployment type to avoid Out-of-Memory (OOM) crashes and maximize speed.

### General Base Requirements (All Systems)
* **Python:** 3.10+ (3.12 recommended for native installs)
* **Disk Space:** ~3 GB per model (Total ~10 GB to cache Clone, Design, and Custom models)

---

### 1. Apple Silicon (macOS)
Runs natively using the MLX framework, leveraging unified memory for high efficiency and lower thermals.

| Hardware | Recommendation | Optimal `config.json` Settings |
| :--- | :--- | :--- |
| **Minimum** | Base M1 (8GB Unified Memory) | `backend: "mlx"`, `model_size: "0.6B"`, `mlx_quantization: "4bit"` |
| **Recommended** | M2/M3/M4 (16GB+ Unified) | `backend: "mlx"`, `model_size: "1.7B"`, `mlx_quantization: "8bit"` |

* **Pro-Tip:** If using an 8GB Mac, loading multiple models simultaneously will crash it. Unload unused models via the UI or CLI before switching tasks.

---

### 2. Standard PyTorch (Linux / Intel Mac)
The standard engine utilizing native PyTorch and HuggingFace Transformers.

| Hardware | Recommendation | Optimal `config.json` Settings |
| :--- | :--- | :--- |
| **Minimum** | NVIDIA T4 16GB | `backend: "torch"`, `model_size: "1.7B"`, `dtype: "float16"`, `compile_model: false` |
| **Recommended**| NVIDIA L4 / A10G (Ampere+) | `backend: "torch"`, `model_size: "1.7B"`, `dtype: "bfloat16"`, `compile_model: true` |

* **Pro-Tip:** Setting `compile_model: true` on Ampere-architecture GPUs (L4, A100) enables `torch.compile` and Flash Attention 2, significantly speeding up inference, but adds a ~1 minute delay on the very first generation.

---

### 3. Production vLLM-Omni (Linux NVIDIA GPUs)
The high-throughput engine. Yields 3-4x faster generation (RTF ~0.399 on H100) using PagedAttention. **Requires Linux and CUDA.**

| Hardware | Recommendation | Optimal `config.json` Settings |
| :--- | :--- | :--- |
| **Minimum** | NVIDIA T4 16GB (Compute 7.0+) | `backend: "vllm"`, `model_size: "1.7B"`, `vllm_gpu_memory_utilization: 0.7` |
| **Recommended**| NVIDIA L4 / A100 (24GB+ VRAM)| `backend: "vllm"`, `model_size: "1.7B"`, `vllm_gpu_memory_utilization: 0.9` |

* **Pro-Tip:** vLLM is incredibly greedy. By default, it attempts to pre-allocate 90% of your GPU memory. If you are running on a smaller 16GB card (like a T4), you *must* drop `vllm_gpu_memory_utilization` to `0.7` or the FastAPI server overhead will trigger an immediate OOM crash.

---

## Installation Paths

There are two ways to install Qwen3-TTS. **Choose your path based on your hardware to avoid massive performance penalties.**

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

## Google Colab
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