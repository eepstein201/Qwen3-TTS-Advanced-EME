#!/usr/bin/env python3
"""
Full Suite Test Runner for Qwen3-TTS

Comprehensive test runner that:
- Tests on both MLX and Torch environments
- Handles Mac Silicon, Mac Intel, and Linux platforms
- Installs missing dependencies as needed
- Manages server lifecycle (stop, start, load models)
- Runs all tests including E2E and evaluation tests

Usage:
    python tests/run_full_suite.py                    # Standard run (current env)
    python tests/run_full_suite.py --full             # Full run (all envs, all deps)
    python tests/run_full_suite.py --full --env mlx   # Full run in MLX env only
    python tests/run_full_suite.py --full --env torch # Full run in Torch env only
    python tests/run_full_suite.py --dry-run          # Show what would be done
"""

import argparse
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _read_auth_token_for_suite() -> str:
    """Return the server auth token, or "test" if none is installed.

    Delegates to the production reader so the canonical-then-legacy resolution
    lives in exactly one place. Falls back to the previous placeholder rather
    than raising: this runner is also used on machines with no server set up,
    and load_models() only warns on failure.
    """
    try:
        from qwen3_tts.core.config import read_auth_token

        return read_auth_token() or "test"
    except Exception:
        return "test"

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
CONDA_BASE = Path.home() / "miniforge3"
MLX_ENV = "qwen3-tts-mlx"
TORCH_ENV = "qwen3-tts"
SERVER_PORT = 5123
SERVER_STARTUP_TIMEOUT = 120  # seconds

# Environment definitions
ENVIRONMENTS = {
    "mlx": {
        "name": MLX_ENV,
        "platforms": ["macos-arm64"],
        "backend": "mlx",
        "description": "Apple Silicon MLX backend",
    },
    "torch": {
        "name": TORCH_ENV,
        "platforms": ["macos-arm64", "macos-x86_64", "linux"],
        "backend": "torch",
        "description": "Cross-platform PyTorch backend",
    },
}

# Optional dependencies for full testing (per environment)
OPTIONAL_DEPS = {
    "mlx": {
        "evaluation": ["openai-whisper", "jiwer"],
        "speaker_similarity": [],  # Not supported in MLX env (requires torch)
        "e2e": ["playwright"],
    },
    "torch": {
        "evaluation": ["openai-whisper", "jiwer"],
        "speaker_similarity": ["torchaudio", "transformers", "torchcodec"],  # WavLM requires these
        "e2e": ["playwright"],
    },
}

# Models required for full testing
REQUIRED_MODELS = ["clone", "design", "custom"]


def get_platform() -> str:
    """Detect current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        if machine == "arm64":
            return "macos-arm64"
        else:
            return "macos-x86_64"
    elif system == "linux":
        return "linux"
    else:
        return f"{system}-{machine}"


def run_cmd(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command with optional capture."""
    result = subprocess.run(  # CodeQL: cmd is a hardcoded list, not user input [py/command-line-injection]
        cmd,
        capture_output=capture,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        if capture:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"stderr: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def conda_run(env: str, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command in a conda environment."""
    full_cmd = [
        "bash", "-lc",
        f"source {CONDA_BASE}/etc/profile.d/conda.sh && conda activate {env} && {' '.join(cmd)}"
    ]
    return run_cmd(full_cmd, check=check, capture=True)


def check_conda_env_exists(env: str) -> bool:
    """Check if a conda environment exists."""
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True,
    )
    return env in result.stdout


def install_optional_deps(env: str, dry_run: bool = False) -> None:
    """Install optional dependencies for full testing."""
    all_deps = []
    for deps in OPTIONAL_DEPS.values():
        all_deps.extend(deps)

    if not all_deps:
        return

    print(f"\n📦 Installing optional dependencies in {env}...")
    for dep in all_deps:
        print(f"   - {dep}")

    if dry_run:
        print("   [DRY RUN] Skipping installation")
        return

    conda_run(env, ["pip", "install", *all_deps], check=False)


def server_is_running() -> bool:
    """Check if TTS server is running on the expected port."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex(('127.0.0.1', SERVER_PORT))
        return result == 0
    finally:
        sock.close()


def stop_server(env: str, dry_run: bool = False) -> None:
    """Stop the TTS server."""
    print("\n🛑 Stopping TTS server...")
    if dry_run:
        print("   [DRY RUN] Would run: tts server stop")
        return

    conda_run(env, ["tts", "server", "stop"], check=False)
    time.sleep(2)  # Wait for cleanup


def start_server(env: str, dry_run: bool = False) -> bool:
    """Start the TTS server and wait for readiness.

    Sets QWEN3_TTS_BACKEND env var based on environment to ensure correct backend.
    """
    print("\n🚀 Starting TTS server...")

    # Determine correct backend for this environment
    backend = "mlx" if "mlx" in env else "torch"
    print(f"   Using backend: {backend}")

    if dry_run:
        print("   [DRY RUN] Would run: tts server start")
        print("   [DRY RUN] Would wait for server ready on port 5123")
        return True

    # Start server in background with correct backend env var
    subprocess.Popen(
        ["bash", "-lc",
         f"source {CONDA_BASE}/etc/profile.d/conda.sh && conda activate {env} && "
         f"export QWEN3_TTS_BACKEND={backend} && tts server start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to be ready
    print(f"   Waiting for server (timeout: {SERVER_STARTUP_TIMEOUT}s)...")
    start_time = time.time()
    while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
        if server_is_running():
            # Additional check: is server ready (not just loading)?
            result = conda_run(env, ["curl", "-s", f"http://127.0.0.1:{SERVER_PORT}/ready"], check=False)
            if result.returncode == 0:
                print("   ✅ Server ready!")
                return True
        time.sleep(2)

    print("   ❌ Server failed to start within timeout")
    return False


def load_models(env: str, dry_run: bool = False) -> bool:
    """Load all required models."""
    print("\n📥 Loading models...")

    if dry_run:
        for model in REQUIRED_MODELS:
            print(f"   [DRY RUN] Would load: {model}")
        return True

    # Two bugs lived in the previous curl invocation, both silent because this
    # loop only warns on failure:
    #   1. It read the LEGACY token path (~/.voice_server_token) rather than the
    #      canonical ~/.config/qwen3-tts/.voice_server_token, so on a normal
    #      install every request was unauthenticated.
    #   2. conda_run() does `' '.join(cmd)` into `bash -lc` with no quoting, so
    #      "Content-Type: application/json" split into two shell words and curl
    #      never received a valid header at all.
    # Reading the token in-process via the production reader fixes (1) and
    # cannot drift; shlex.quote on every argument fixes (2).
    token = _read_auth_token_for_suite()

    for model in REQUIRED_MODELS:
        print(f"   Loading {model}...")
        result = conda_run(env, [
            "curl", "-s", "-X", "POST",
            shlex.quote(f"http://127.0.0.1:{SERVER_PORT}/load-model"),
            "-H", shlex.quote("Content-Type: application/json"),
            "-H", shlex.quote(f"Authorization: Bearer {token}"),
            "-d", shlex.quote(f'{{"mode": "{model}"}}'),
        ], check=False)
        if result.returncode != 0:
            print(f"   ⚠️  Failed to load {model}")

    # Verify models loaded
    time.sleep(5)
    result = conda_run(env, ["curl", "-s", f"http://127.0.0.1:{SERVER_PORT}/models"], check=False)
    if "loaded" in result.stdout.lower():
        print("   ✅ Models loaded!")
        return True
    return False


def run_tests(env: str, test_type: str, dry_run: bool = False) -> int:
    """Run tests of a specific type."""
    # NOTE: A command-line `-m` OVERRIDES the `-m "not e2e"` default in pytest.ini
    # addopts. So any test-type that passes its own `-m` must re-add `not e2e`
    # explicitly, or it will re-collect the heavy E2E tests and hang against a live
    # server. The `e2e` type intentionally opts back in with `-m e2e`.
    test_commands = {
        "unit": ["python", "-m", "pytest", "tests/", "-q", "--tb=short", "-m", "not e2e and not requires_server"],
        "integration": ["python", "-m", "pytest", "tests/", "-q", "--tb=short", "-m", "integration and not e2e"],
        "e2e": ["python", "-m", "pytest", "tests/", "-m", "e2e", "-v", "--tb=short"],
        "evaluation": ["python", "-m", "pytest", "tests/evaluations/", "-v", "--tb=short"],
        # `all` inherits the pytest.ini `-m "not e2e"` default (no explicit -m), so it
        # excludes E2E. Run the `e2e` type separately for the live-server E2E suite.
        "all": ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
    }

    cmd = test_commands.get(test_type, test_commands["all"])
    print(f"\n🧪 Running {test_type} tests in {env}...")
    print(f"   Command: {' '.join(cmd)}")

    if dry_run:
        print("   [DRY RUN] Skipping test execution")
        return 0

    result = conda_run(env, cmd, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode


def run_full_suite(args: argparse.Namespace) -> int:
    """Run the full test suite."""
    current_platform = get_platform()
    print(f"\n{'='*60}")
    print("Qwen3-TTS Full Test Suite")
    print(f"{'='*60}")
    print(f"Platform: {current_platform}")
    print(f"Dry run: {args.dry_run}")
    print(f"Environments: {args.env}")
    print(f"Test type: {args.test_type}")
    print(f"{'='*60}\n")

    # Determine which environments to test
    envs_to_test = []
    if args.env == "all":
        for env_key, env_config in ENVIRONMENTS.items():
            if current_platform in env_config["platforms"]:
                if check_conda_env_exists(env_config["name"]):
                    envs_to_test.append((env_key, env_config))
                else:
                    print(f"⚠️  Environment {env_config['name']} not found, skipping")
    elif args.env in ENVIRONMENTS:
        env_config = ENVIRONMENTS[args.env]
        if check_conda_env_exists(env_config["name"]):
            envs_to_test.append((args.env, env_config))
        else:
            print(f"❌ Environment {env_config['name']} not found")
            return 1
    else:
        print(f"❌ Unknown environment: {args.env}")
        return 1

    if not envs_to_test:
        print("❌ No valid environments to test")
        return 1

    results = {}

    for env_key, env_config in envs_to_test:
        env_name = env_config["name"]
        print(f"\n{'='*60}")
        print(f"Testing in {env_name} ({env_config['description']})")
        print(f"{'='*60}")

        try:
            # Install optional dependencies if full run
            if args.full:
                install_optional_deps(env_name, args.dry_run)

            # Server lifecycle management
            if args.test_type in ["e2e", "all"]:
                stop_server(env_name, args.dry_run)
                if not start_server(env_name, args.dry_run):
                    print(f"❌ Failed to start server for {env_name}")
                    results[env_name] = 1
                    continue
                if args.full:
                    load_models(env_name, args.dry_run)

            # Run tests
            exit_code = run_tests(env_name, args.test_type, args.dry_run)
            results[env_name] = exit_code

            # Cleanup
            if args.test_type in ["e2e", "all"]:
                stop_server(env_name, args.dry_run)

        except Exception as e:
            print(f"❌ Error testing {env_name}: {e}")
            results[env_name] = 1

    # Summary
    print(f"\n{'='*60}")
    print("Test Suite Summary")
    print(f"{'='*60}")
    for env_name, exit_code in results.items():
        status = "✅ PASSED" if exit_code == 0 else "❌ FAILED"
        print(f"  {env_name}: {status}")

    # Return non-zero if any environment failed
    return 0 if all(code == 0 for code in results.values()) else 1


def main():
    parser = argparse.ArgumentParser(
        description="Full Suite Test Runner for Qwen3-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_full_suite.py                    # Standard run
  python tests/run_full_suite.py --full             # Full run with all deps
  python tests/run_full_suite.py --full --env mlx   # Full run in MLX only
  python tests/run_full_suite.py --full --env all   # Full run in all envs
  python tests/run_full_suite.py --test-type e2e    # Run E2E tests only
  python tests/run_full_suite.py --dry-run          # Show what would be done
        """,
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Full run: install all deps, load all models, run all tests",
    )
    parser.add_argument(
        "--env",
        choices=["mlx", "torch", "all"],
        default="mlx",
        help="Environment(s) to test (default: mlx)",
    )
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "e2e", "evaluation", "all"],
        default="unit",
        help="Type of tests to run (default: unit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )

    args = parser.parse_args()

    # If --full is set, default to all tests
    if args.full and args.test_type == "unit":
        args.test_type = "all"

    sys.exit(run_full_suite(args))


if __name__ == "__main__":
    main()
