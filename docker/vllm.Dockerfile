# vLLM-Omni Docker configuration for TTS with audio processing
# Base image: NVIDIA CUDA runtime with Python 3.10
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create cache directory for HuggingFace models
RUN mkdir -p /root/.cache/huggingface

# Copy vLLM requirements file
COPY requirements-vllm.txt /tmp/

# Install vLLM with audio dependencies
# Note: vllm[audio] includes dependencies for multimodal audio processing
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements-vllm.txt && \
    rm /tmp/requirements-vllm.txt

# Set working directory
WORKDIR /app

# Expose vLLM server port
EXPOSE 5123

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python3 -c "import httpx; httpx.get('http://127.0.0.1:5123/health').raise_for_status()" || exit 1

# Default command: start vLLM server
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", \
     "--port", "5123", \
     "--disable-log-requests"]
