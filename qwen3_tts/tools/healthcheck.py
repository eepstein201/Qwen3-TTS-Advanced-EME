#!/usr/bin/env python3
"""TTS health check and diagnostics utility.

Verifies installation status, backend availability, model cache,
and common configuration issues.
"""

import os
import pathlib
import platform
import sys

from qwen3_tts.core.config import (
    IN_COLAB,
    IS_MACOS,
    IS_LINUX,
    CONFIG_PATH,
    VOICE_PROMPTS_DIR,
    USER_FILES_DIR,
    HISTORY_FILE,
    TOKEN_FILE,
    PID_FILE,
    LOG_FILE,
    HF_CACHE,
    detect_server_state,
)
from qwen3_tts.tools.model_cache import _MLX_MODEL_PREFIXES, _TORCH_MODEL_PREFIXES

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{BOLD}{BLUE}  {text}{RESET}")
    print(f"  {'=' * (len(text) + 2)}")


def _print_check(label: str, status: str, details: str = "") -> None:
    """Print a check result with status indicator."""
    if status == "pass":
        indicator = f"{GREEN}✓{RESET}"
    elif status == "warn":
        indicator = f"{YELLOW}⚠{RESET}"
    else:
        indicator = f"{RED}✗{RESET}"

    print(f"  {indicator} {label}")
    if details:
        print(f"    {details}")


def _print_info(label: str, details: str) -> None:
    """Print an info item."""
    print(f"  {BLUE}ℹ{RESET} {label}")
    if details:
        print(f"    {details}")


def check_python_version() -> tuple:
    """Check Python version compatibility."""
    version = sys.version_info
    major, minor = version.major, version.minor

    if major >= 3 and minor >= 10:
        status = "pass"
        details = f"Python {major}.{minor}.{version.micro}"
    else:
        status = "fail"
        details = f"Python {major}.{minor}.{version.micro} (requires 3.10+)"

    return status, details


def check_backend_availability() -> tuple:
    """Check which backends are available."""
    backends = []
    issues = []

    # Check MLX (Apple Silicon only)
    if IS_MACOS and platform.machine() == "arm64":
        try:
            import mlx
            backends.append("mlx")
        except ImportError:
            issues.append("MLX not installed (run: pip install mlx-audio)")
    else:
        issues.append("MLX requires Apple Silicon (ARM64 macOS)")

    # Check PyTorch
    try:
        import torch
        backends.append("torch")

        # Check CUDA availability
        if torch.cuda.is_available():
            backends.append("torch-cuda")
        else:
            issues.append("CUDA not available (torch backend CPU only)")

        # Check MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            backends.append("torch-mps")
    except ImportError:
        issues.append("PyTorch not installed (run: pip install torch)")

    # Check vLLM (Linux only)
    if IS_LINUX:
        try:
            import vllm
            backends.append("vllm")
        except ImportError:
            issues.append("vLLM not installed (optional, for high-throughput)")

    if backends:
        status = "pass"
        details = f"Available: {', '.join(backends)}"
    else:
        status = "fail"
        details = "No backends available!"

    if issues:
        details += f"\n    Issues:\n    - " + "\n    - ".join(issues)

    return status, details


def check_config() -> tuple:
    """Check configuration file status."""
    config_path = pathlib.Path(CONFIG_PATH)
    if config_path.exists():
        try:
            from qwen3_tts.core.config import load_config, validate_config
            config = load_config()
            issues = validate_config(config)

            if issues:
                return "warn", f"Config has {len(issues)} validation issue(s)"

            backend = config.get("advanced", {}).get("backend")
            model_size = config.get("advanced", {}).get("model_size")
            details = f"Backend: {backend}, Model size: {model_size}"
            return "pass", details
        except Exception as e:
            return "fail", f"Config error: {e}"
    else:
        return "warn", f"Config not found (will be created with defaults)"


def check_model_cache() -> tuple:
    """Check model cache status."""
    if not HF_CACHE.exists():
        return "warn", "HuggingFace cache not found"

    # Count TTS models
    model_count = 0
    total_size = 0

    for model_dir in HF_CACHE.iterdir():
        if model_dir.is_dir() and any(
            model_dir.name.startswith(prefix) for prefix in _TORCH_MODEL_PREFIXES + _MLX_MODEL_PREFIXES
        ):
            model_count += 1
            # Simple size estimate
            try:
                for item in model_dir.rglob("*"):
                    if item.is_file():
                        total_size += item.stat().st_size
            except OSError:
                pass

    if model_count == 0:
        return "info", "No models cached (will download on first use)"

    # Format size
    for unit in ("B", "KB", "MB", "GB"):
        if total_size < 1024.0:
            size_str = f"{total_size:.1f} {unit}"
            break
        total_size /= 1024.0

    return "pass", f"{model_count} model(s) cached ({size_str})"


def check_voice_prompts() -> tuple:
    """Check voice prompts directory."""
    prompts_dir = pathlib.Path(VOICE_PROMPTS_DIR)
    if not prompts_dir.exists():
        return "warn", "Voice prompts directory not found"

    # Count prompts
    pt_files = list(prompts_dir.glob("*.pt"))
    wav_files = list(prompts_dir.glob("*.wav"))

    if not pt_files and not wav_files:
        return "info", "No voice prompts found"

    return "pass", f"{len(pt_files)} .pt, {len(wav_files)} .wav files"


def check_server_status() -> tuple:
    """Check if TTS server is running using unified detection."""
    state = detect_server_state()

    if state["running"]:
        parts = []
        if state["health_ok"]:
            parts.append("health OK")
        if state["pid"]:
            parts.append(f"PID {state['pid']}")
        return "pass", f"Server running ({', '.join(parts)})"

    if state["stale_pid"]:
        return "warn", f"Server not running (stale PID file, PID {state['pid']})"

    return "info", "Server not running"


def check_audio_dependencies() -> tuple:
    """Check audio processing dependencies."""
    issues = []

    # Check for ffmpeg
    try:
        import subprocess
        result = subprocess.run(["ffmpeg", "-version"],
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            has_ffmpeg = True
        else:
            has_ffmpeg = False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        has_ffmpeg = False

    if not has_ffmpeg:
        issues.append("ffmpeg not found")

    # Check for rubberband (optional but recommended)
    try:
        result = subprocess.run(["rubberband", "-h"],
                              capture_output=True, timeout=5)
        has_rubberband = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        has_rubberband = False

    if issues:
        status = "warn"
        details = f"Missing: {', '.join(issues)}"
    else:
        status = "pass"
        details = f"ffmpeg: {'✓' if has_ffmpeg else '✗'}, rubberband: {'✓' if has_rubberband else '✗ (optional)'}"

    return status, details


def check_disk_space() -> tuple:
    """Check available disk space."""
    try:
        import shutil
        stat = shutil.disk_usage(USER_FILES_DIR)
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)

        if free_gb < 5:
            return "warn", f"Low disk space: {free_gb:.1f}GB free (5GB minimum recommended)"
        elif free_gb < 15:
            return "warn", f"Moderate disk space: {free_gb:.1f}GB free (15GB+ recommended for all models)"
        else:
            return "pass", f"{free_gb:.1f}GB free / {total_gb:.1f}GB total"
    except Exception as e:
        return "warn", f"Could not check disk space: {e}"


def run_healthcheck() -> int:
    """Run all health checks and return exit code."""
    print(f"\n{BOLD}Qwen3-TTS Health Check{RESET}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Python: {sys.version.split()[0]}")
    if IN_COLAB:
        print("  Environment: Google Colab")
    print()

    # Run all checks
    checks = [
        ("Python Version", check_python_version()),
        ("Backend Availability", check_backend_availability()),
        ("Configuration", check_config()),
        ("Model Cache", check_model_cache()),
        ("Voice Prompts", check_voice_prompts()),
        ("Server Status", check_server_status()),
        ("Audio Dependencies", check_audio_dependencies()),
        ("Disk Space", check_disk_space()),
    ]

    all_pass = True
    for label, (status, details) in checks:
        if status == "fail":
            all_pass = False
        _print_check(label, status, details)

    print()

    # Summary
    if all_pass:
        print(f"  {GREEN}All checks passed!{RESET}")
        print(f"  Your TTS installation is healthy.")
        return 0
    else:
        print(f"  {YELLOW}Some checks failed or warnings.{RESET}")
        print(f"  See above for details.")
        return 1


def main():
    """CLI entry point for health check command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check TTS installation health",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Show suggested fixes for issues",
    )

    args = parser.parse_args()

    return run_healthcheck()


if __name__ == "__main__":
    sys.exit(main())
