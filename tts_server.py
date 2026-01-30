#!/usr/bin/env python3
"""Persistent TTS server that keeps models loaded in memory for fast generation."""

import os
import secrets
import signal
import sys
import tempfile
import threading
import time

from flask import Flask, request, jsonify
import torch
import soundfile as sf

from tts_config import (
    CONFIG_PATH,
    VOICE_PROMPTS_DIR,
    PID_FILE,
    TOKEN_FILE,
    MODEL_INFO,
    CUSTOM_VOICE_SPEAKERS,
    load_config,
)
from tts_engine import (
    load_model,
    load_voice_prompt,
    run_inference,
    voice_prompt_cache_info,
)

# Auth token for this server session
auth_token = None

app = Flask(__name__)


def generate_auth_token():
    """Generate a new auth token and write it to TOKEN_FILE."""
    global auth_token
    auth_token = secrets.token_hex(32)
    with open(TOKEN_FILE, "w") as f:
        f.write(auth_token)
    os.chmod(TOKEN_FILE, 0o600)
    return auth_token


# Endpoints that don't require authentication
PUBLIC_ENDPOINTS = {"health", "static"}


@app.before_request
def check_auth():
    """Verify Bearer token on all endpoints except public ones."""
    if auth_token is None:
        return  # Auth not configured (shouldn't happen in normal operation)

    endpoint = request.endpoint
    if endpoint in PUBLIC_ENDPOINTS:
        return

    auth_header = request.headers.get("Authorization", "")
    if auth_header == f"Bearer {auth_token}":
        return

    return jsonify({
        "error": "Authentication required",
        "detail": "Include 'Authorization: Bearer <token>' header. Token is in ~/.tts_server_token",
        "recovery": "restart",
    }), 401


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


def _get_model(model_type):
    """Return the loaded model for a given type, or None."""
    if model_type == "clone":
        return clone_model
    elif model_type == "design":
        return design_model
    elif model_type == "custom":
        return custom_model
    return None


def load_single_model(model_type):
    """Load a single model by type using tts_engine."""
    global clone_model, design_model, custom_model

    if model_type not in MODEL_INFO:
        return False

    print(f"Loading {MODEL_INFO[model_type]['name']}...")
    model = load_model(model_type)

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

    cache_info = voice_prompt_cache_info()

    stats_data = {
        "status": "ok",
        "clone_model_loaded": clone_model is not None,
        "design_model_loaded": design_model is not None,
        "custom_model_loaded": custom_model is not None,
        "voice_prompts_cached": cache_info.currsize,
        "voice_prompts_cache_hits": cache_info.hits,
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
        loaded = _get_model(model_type) is not None
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
    if _get_model(model_type) is not None:
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
    model = _get_model(mode)
    if model is None:
        return jsonify({
            "error": "model_not_loaded",
            "model_type": mode,
            "description": MODEL_INFO.get(mode, {}).get("description", ""),
        }), 503

    # Generation parameters
    gen_params = {
        "temperature": data.get("temperature", 0.7),
        "top_k": data.get("top_k", 50),
        "top_p": data.get("top_p", 0.95),
        "repetition_penalty": data.get("repetition_penalty", 1.05),
    }

    seed = data.get("seed")
    if seed is not None:
        gen_params["seed"] = seed

    # Track this request in queue
    request_id = id(request)
    request_queue.append(request_id)

    try:
        # Acquire lock for thread-safe generation
        with generation_lock:
            results = []

            for i, text in enumerate(texts):
                # Resolve voice prompt for clone mode
                voice_prompt = None
                if mode == "clone":
                    if not prompt_file:
                        return jsonify({"error": "prompt_file required for clone mode"}), 400
                    voice_prompt = load_voice_prompt(prompt_file)
                    if voice_prompt is None:
                        return jsonify({"error": f"Voice prompt not found: {prompt_file}"}), 404
                elif mode == "custom":
                    if not speaker:
                        return jsonify({"error": "speaker required for custom mode"}), 400

                wav, sr = run_inference(
                    model=model,
                    text=text,
                    mode=mode,
                    gen_params=gen_params,
                    language=language,
                    voice_prompt=voice_prompt,
                    voice_description=voice_description,
                    speaker=speaker,
                    instruct=instruct,
                )

                # Save to temp file
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(temp_file.name, wav, sr)
                results.append({"index": i, "file": temp_file.name, "sample_rate": sr})

            return jsonify({"results": results})
    except Exception as e:
        app.logger.error(f"Generation failed: {e}")
        return jsonify({"error": str(e)}), 500
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
    if os.path.exists(TOKEN_FILE):
        os.remove(TOKEN_FILE)
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

    # Generate auth token
    generate_auth_token()
    print(f"Auth token written to {TOKEN_FILE}")

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
