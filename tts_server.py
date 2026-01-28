#!/usr/bin/env python3
"""Persistent TTS server that keeps models loaded in memory for fast generation."""

import json
import os
import signal
import sys
import tempfile
import threading
import time
from functools import lru_cache
from flask import Flask, request, jsonify
import torch
import soundfile as sf

app = Flask(__name__)

# Global model holders
clone_model = None
design_model = None
custom_model = None

# Thread safety for generation
generation_lock = threading.Lock()
request_queue = []
max_concurrent = 1  # TTS generation is memory-intensive, serialize by default

# Auto-shutdown timer
shutdown_timer = None
last_activity = time.time()

CONFIG_PATH = os.path.expanduser("~/Qwen3-TTS_UserFiles/config.json")
VOICE_PROMPTS_DIR = os.path.expanduser("~/Qwen3-TTS_UserFiles/voice_prompts")
PID_FILE = os.path.expanduser("~/Qwen3-TTS_UserFiles/.tts_server.pid")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


# Store config globally for auto-shutdown settings
server_config = {}


def reset_activity_timer():
    """Reset the auto-shutdown timer on activity."""
    global shutdown_timer, last_activity
    last_activity = time.time()

    auto_shutdown_minutes = server_config.get("auto_shutdown_minutes", 0)
    if auto_shutdown_minutes <= 0:
        return  # Auto-shutdown disabled

    # Cancel existing timer
    if shutdown_timer is not None:
        shutdown_timer.cancel()

    # Start new timer
    shutdown_timer = threading.Timer(
        auto_shutdown_minutes * 60,
        auto_shutdown
    )
    shutdown_timer.daemon = True
    shutdown_timer.start()


def auto_shutdown():
    """Auto-shutdown due to inactivity."""
    print(f"\nAuto-shutdown: No activity for {server_config.get('auto_shutdown_minutes', 0)} minutes.")
    print("Shutting down to free VRAM...")
    cleanup_pid()


@lru_cache(maxsize=10)
def load_voice_prompt(prompt_file):
    """Cache voice prompts in memory."""
    prompt_path = os.path.join(VOICE_PROMPTS_DIR, prompt_file)
    if not os.path.exists(prompt_path):
        return None
    return torch.load(prompt_path, weights_only=False)


MODEL_INFO = {
    "clone": {
        "name": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "description": "Voice cloning from audio samples (clone mode)",
        "memory_mb": 3500,
    },
    "design": {
        "name": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "description": "Generate voice from text description (design mode)",
        "memory_mb": 3500,
    },
    "custom": {
        "name": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "description": "9 premium pre-trained speakers (custom mode)",
        "memory_mb": 3500,
    },
}


def load_single_model(model_type):
    """Load a single model by type."""
    global clone_model, design_model, custom_model
    from qwen_tts import Qwen3TTSModel

    info = MODEL_INFO.get(model_type)
    if not info:
        return False

    print(f"Loading {info['name']}...")

    model = Qwen3TTSModel.from_pretrained(
        info["name"],
        attn_implementation="sdpa",
        device_map="mps",
        dtype=torch.float16,
    )

    if model_type == "clone":
        clone_model = model
    elif model_type == "design":
        design_model = model
    elif model_type == "custom":
        custom_model = model

    print(f"Loaded {model_type} model successfully!")
    return True


def load_models():
    """Load TTS models based on config."""
    global clone_model, design_model, custom_model

    models_config = server_config.get("models", {})

    # Default: load clone model if no config
    if not models_config:
        models_config = {"clone": {"load_at_startup": True}}

    models_to_load = []
    for model_type, settings in models_config.items():
        if settings.get("load_at_startup", False):
            models_to_load.append(model_type)

    if not models_to_load:
        print("Warning: No models configured to load at startup.")
        print("Use 'changeVoice --list-models' to see available models.")
        return

    print(f"\nLoading {len(models_to_load)} model(s): {', '.join(models_to_load)}")
    print("(Configure in config.json under 'models' section)\n")

    for model_type in models_to_load:
        load_single_model(model_type)

    print("\nModel loading complete!")


@app.route("/health", methods=["GET"])
def health():
    reset_activity_timer()
    return jsonify({
        "status": "ok",
        "clone_model_loaded": clone_model is not None,
        "design_model_loaded": design_model is not None,
        "custom_model_loaded": custom_model is not None,
    })


@app.route("/stats", methods=["GET"])
def stats():
    """Return server statistics including memory usage."""
    reset_activity_timer()

    idle_seconds = int(time.time() - last_activity)
    auto_shutdown_minutes = server_config.get("auto_shutdown_minutes", 0)

    stats_data = {
        "status": "ok",
        "clone_model_loaded": clone_model is not None,
        "design_model_loaded": design_model is not None,
        "custom_model_loaded": custom_model is not None,
        "voice_prompts_cached": load_voice_prompt.cache_info().currsize,
        "voice_prompts_cache_hits": load_voice_prompt.cache_info().hits,
        "idle_seconds": idle_seconds,
        "auto_shutdown_minutes": auto_shutdown_minutes if auto_shutdown_minutes > 0 else "disabled",
        "generation_queue_size": len(request_queue),
    }

    # MPS (Apple Silicon) memory stats
    if torch.backends.mps.is_available():
        try:
            allocated = torch.mps.current_allocated_memory()
            stats_data["mps_memory_allocated_mb"] = round(allocated / (1024 * 1024), 2)
        except Exception:
            stats_data["mps_memory_allocated_mb"] = "unavailable"

    # CUDA memory stats (for NVIDIA GPUs)
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            stats_data["cuda_memory_allocated_mb"] = round(allocated / (1024 * 1024), 2)
            stats_data["cuda_memory_reserved_mb"] = round(reserved / (1024 * 1024), 2)
        except Exception:
            pass

    return jsonify(stats_data)


@app.route("/models", methods=["GET"])
def list_models():
    """Return information about available models and their load status."""
    reset_activity_timer()

    models_data = {}
    for model_type, info in MODEL_INFO.items():
        if model_type == "clone":
            loaded = clone_model is not None
        elif model_type == "design":
            loaded = design_model is not None
        elif model_type == "custom":
            loaded = custom_model is not None
        else:
            loaded = False

        models_data[model_type] = {
            "loaded": loaded,
            "description": info["description"],
            "memory_mb": info["memory_mb"],
        }

    return jsonify({"models": models_data})


@app.route("/load-model", methods=["POST"])
def load_model_endpoint():
    """Load a model on demand."""
    reset_activity_timer()

    data = request.json
    model_type = data.get("model_type")

    if not model_type:
        return jsonify({"error": "model_type required"}), 400

    if model_type not in MODEL_INFO:
        return jsonify({"error": f"Unknown model type: {model_type}. Valid: clone, design, custom"}), 400

    # Check if already loaded
    if model_type == "clone" and clone_model is not None:
        return jsonify({"status": "already_loaded", "model": model_type})
    if model_type == "design" and design_model is not None:
        return jsonify({"status": "already_loaded", "model": model_type})
    if model_type == "custom" and custom_model is not None:
        return jsonify({"status": "already_loaded", "model": model_type})

    # Load the model (this may take a while)
    with generation_lock:
        success = load_single_model(model_type)

    if success:
        return jsonify({"status": "loaded", "model": model_type})
    else:
        return jsonify({"error": f"Failed to load model: {model_type}"}), 500


@app.route("/prompts", methods=["GET"])
def list_prompts():
    reset_activity_timer()
    prompts = [f for f in os.listdir(VOICE_PROMPTS_DIR) if f.endswith('.pt')]
    return jsonify({"prompts": sorted(prompts)})


@app.route("/generate", methods=["POST"])
def generate():
    reset_activity_timer()

    data = request.json
    texts = data.get("texts", [])
    if isinstance(texts, str):
        texts = [texts]

    mode = data.get("mode", "clone")
    prompt_file = data.get("prompt_file")
    voice_description = data.get("voice_description", "")
    language = data.get("language", "English")

    # Custom mode parameters
    speaker = data.get("speaker")
    instruct = data.get("instruct", "")

    # Check if required model is loaded
    if mode == "clone" and clone_model is None:
        return jsonify({
            "error": "model_not_loaded",
            "model_type": "clone",
            "description": MODEL_INFO["clone"]["description"],
        }), 503
    elif mode == "design" and design_model is None:
        return jsonify({
            "error": "model_not_loaded",
            "model_type": "design",
            "description": MODEL_INFO["design"]["description"],
        }), 503
    elif mode == "custom" and custom_model is None:
        return jsonify({
            "error": "model_not_loaded",
            "model_type": "custom",
            "description": MODEL_INFO["custom"]["description"],
        }), 503

    # Generation parameters
    gen_params = {
        "temperature": data.get("temperature", 0.7),
        "top_k": data.get("top_k", 50),
        "top_p": data.get("top_p", 0.95),
        "repetition_penalty": data.get("repetition_penalty", 1.05),
    }

    seed = data.get("seed")

    # Track this request in queue
    request_id = id(request)
    request_queue.append(request_id)

    try:
        # Acquire lock for thread-safe generation
        with generation_lock:
            if seed is not None:
                torch.manual_seed(seed)

            results = []

            with torch.inference_mode():
                for i, text in enumerate(texts):
                    if mode == "clone":
                        if not prompt_file:
                            return jsonify({"error": "prompt_file required for clone mode"}), 400

                        voice_prompt = load_voice_prompt(prompt_file)
                        if voice_prompt is None:
                            return jsonify({"error": f"Voice prompt not found: {prompt_file}"}), 404

                        wavs, sr = clone_model.generate_voice_clone(
                            text=text,
                            language=language,
                            voice_clone_prompt=voice_prompt,
                            **gen_params,
                        )
                    elif mode == "custom":
                        if not speaker:
                            return jsonify({"error": "speaker required for custom mode"}), 400

                        wavs, sr = custom_model.generate_custom_voice(
                            text=text,
                            speaker=speaker,
                            instruct=instruct,
                            language=language,
                            **gen_params,
                        )
                    else:  # design mode
                        wavs, sr = design_model.generate_voice_design(
                            text=text,
                            instruct=voice_description,
                            language=language,
                            **gen_params,
                        )

                    # Save to temp file
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(temp_file.name, wavs[0], sr)
                    results.append({"index": i, "file": temp_file.name, "sample_rate": sr})

            return jsonify({"results": results})
    finally:
        # Remove from queue
        if request_id in request_queue:
            request_queue.remove(request_id)


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Graceful shutdown endpoint."""
    global shutdown_timer
    if shutdown_timer is not None:
        shutdown_timer.cancel()
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    func = request.environ.get("werkzeug.server.shutdown")
    if func:
        func()
    else:
        os.kill(os.getpid(), signal.SIGTERM)
    return jsonify({"status": "shutting down"})


def write_pid():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def cleanup_pid(signum=None, frame=None):
    global shutdown_timer
    if shutdown_timer is not None:
        shutdown_timer.cancel()
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    sys.exit(0)


if __name__ == "__main__":
    config = load_config()
    server_config = config.get("server", {})
    # Store models config in server_config for access by load_models()
    server_config["models"] = config.get("models", {})
    host = server_config.get("host", "127.0.0.1")
    port = server_config.get("port", 5123)

    # Handle shutdown signals
    signal.signal(signal.SIGTERM, cleanup_pid)
    signal.signal(signal.SIGINT, cleanup_pid)

    # Load models before starting server
    load_models()

    # Write PID file
    write_pid()

    # Start auto-shutdown timer if configured
    auto_shutdown_minutes = server_config.get("auto_shutdown_minutes", 0)
    if auto_shutdown_minutes > 0:
        print(f"Auto-shutdown enabled: {auto_shutdown_minutes} minutes of inactivity")
        reset_activity_timer()

    print(f"\nTTS Server running on http://{host}:{port}")
    print("Use stopTTSServer to shut down.\n")

    # Enable threading for concurrent request handling
    # Generation is serialized via lock, but multiple requests can queue
    app.run(host=host, port=port, threaded=True)
