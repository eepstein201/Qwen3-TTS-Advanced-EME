#!/usr/bin/env python3
"""
Batch test runner for Qwen3-TTS

Runs tests in isolated groups to prevent hangs from cascading.
Each batch runs in a subprocess with proper cleanup and timeout.

Usage:
    python tests/run_batches.py              # Run all batches
    python tests/run_batches.py --batch 2    # Run only batch 2
    python tests/run_batches.py --continue   # Continue on failure
    python tests/run_batches.py --timeout 60 # Set per-batch timeout
"""

import argparse

# E2E helper for automatic Playwright toggle
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("e2e_helpers", Path(__file__).parent / "e2e_helpers.py")
e2e_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2e_helpers)
playwright_enabled = e2e_helpers.playwright_enabled


# Test batches organized by risk level
# Each batch: (name, [test_modules])
BATCHES = {
    1: {
        "name": "Core Utilities",
        "description": "Low risk - pure unit tests",
        "modules": [
            "tests.test_audio_utils",
            "tests.test_text_processing",
            "tests.test_package_metadata",
            "tests.test_deprecated_refs",
            "tests.test_async_test_hygiene",
            "tests.test_e2e_harness_hygiene",
            "tests.test_config",
            "tests.test_p3_p4_remediation",
            "tests.test_lufs_return_shape",
            "tests.test_clone_rate_control",
            "tests.test_healthcheck",
            "tests.test_healthcheck_ext",
            "tests.test_model_cache",
            "tests.test_model_cache_commands",
        ],
        "timeout": 90,  # Quick tests
    },
    2: {
        "name": "Voice & CLI",
        "description": "Medium risk - minimal external deps",
        "modules": [
            "tests.test_voice_config",
            "tests.test_voice_engine",
            "tests.test_voice_generation",
            "tests.test_voice_features",
            "tests.test_voice_prompts",
            "tests.test_voice_streaming",
            "tests.test_voice_server",
            "tests.test_voice_ui",
            "tests.test_cli_daemonization",
            "tests.test_cli_commands",
            "tests.test_cli_batch",
            "tests.test_cli_dialogue",
            "tests.test_cli_srt",
            "tests.test_cli_ext",
            "tests.test_create_voice_functions",
            "tests.test_uninstall_functions",
            "tests.test_caching",
            "tests.test_server_helpers",
            "tests.test_ui_audio_reset",
        ],
        "timeout": 180,
    },
    3: {
        "name": "Server Infrastructure",
        "description": "High risk - FastAPI TestClient with lifespan",
        "modules": [
            "tests.test_fastapi_server",
            "tests.test_fastapi_endpoints",
            "tests.test_client",
            "tests.test_client_generator",
            "tests.test_client_models",
            "tests.test_client_voices",
            "tests.test_websocket",
            "tests.test_websocket_rate_limit",
            "tests.test_health_degraded",
            "tests.test_auth_token_write",
            "tests.test_streaming_cancel",
            "tests.test_generation_offload",
            "tests.test_async_concurrency",
            "tests.test_remediation_2026_03_03",
            "tests.test_remediation_2026_03_04",
            "tests.test_integration",
            "tests.test_engine_vllm_characterization",
            "tests.test_engine_vllm_ext",
            "tests.test_engine_vllm_ext_part2",
            "tests.test_model_name_validation",
            "tests.test_decoupled_inference",
            "tests.test_docker_config",
            "tests.test_streaming_and_peaks",
            "tests.test_streaming_mlx_chunking",
            "tests.test_history_seed_chunks",
            "tests.test_fastapi_app_ext2",
            "tests.test_fastapi_app_ext2_part2",
            "tests.test_fastapi_app_ext3",
            "tests.test_silent_failure_logging",
            "tests.test_silent_failure_logging_part2",
            "tests.test_streaming_thread_lifecycle",
            "tests.test_batch_generation_state_ownership",
        ],
        "timeout": 180,  # Higher timeout for async operations
    },
    4: {
        "name": "Engine & UI",
        "description": "Highest risk - model loading, Gradio Timer",
        "modules": [
            "tests.test_engine",
            "tests.test_seed_lock_chunks",
            "tests.test_generate_helpers",
            "tests.test_generate_helpers_part2",
            "tests.test_generate_server",
            "tests.test_generate_interactive",
            "tests.test_generate_interactive_ext",
            "tests.test_generate_interactive_ext_part2",
            "tests.test_generate_main",
            "tests.test_generate_server_fallback",
            "tests.test_ui_headless",
            "tests.test_ui_model_management",
            "tests.test_ui_facade",
            "tests.test_ui_shared_ext",
            "tests.test_ui_voice_mgmt",
            "tests.test_ui_generation_ext",
            "tests.test_fastapi_app_ext",
            "tests.test_wavesurfer_js",
            "tests.test_wavesurfer_security",
            "tests.test_wavesurfer_selfhost",
            "tests.test_model_loader_extended",
        ],
        "timeout": 480,  # Engine & UI — macos CI runners run ~3-4x slower than linux
    },
    # Optional batch (requires additional dependencies)
    5: {
        "name": "Optional Tests",
        "description": "Tests requiring optional dependencies (pytest, etc.)",
        "modules": [
            "tests.test_decomposition_check",
            "tests.test_flash_attn_install",
            "tests.test_solid_analyzer",
            "tests.test_protocols",
            "tests.test_voice_helpers",
            "tests.test_validation",
            "tests.test_error_handling",
            "tests.test_ocp_strategy",
            "tests.test_backend_torch",
            "tests.test_colab_paths",
            "tests.evaluations.test_wer",
            "tests.evaluations.test_speaker_similarity",
            "tests.evaluations.test_llm_judge",
        ],
        "timeout": 600,  # speaker_similarity loads WavLM (~300MB) + runs inference under memory pressure
    },
    # E2E browser tests (requires playwright + running TTS server)
    6: {
        "name": "E2E Playwright",
        "description": "Browser-based E2E tests (requires playwright, running server)",
        "modules": [
            "tests.test_e2e_playwright",
        ],
        "timeout": 1800,  # 10 tests × up to ~120s generation + model load/unload cycles
        "setup": "_ensure_models_loaded",
    },
}


def _ensure_models_loaded():
    """Load all models on the TTS server before E2E tests.

    Reads the auth token from the standard config location and issues
    load-model requests for clone, design, and custom.  Silently skips
    if the server is unreachable (the E2E tests will skip/fail on their own).
    """
    import urllib.error
    import urllib.request

    token_path = Path.home() / ".config" / "qwen3-tts" / ".voice_server_token"
    if not token_path.exists():
        # Fallback to legacy path
        token_path = Path.home() / ".voice_server_token"
    if not token_path.exists():
        print("  [setup] No auth token found — skipping model preload")
        return

    token = token_path.read_text().strip()
    url = "http://127.0.0.1:5123"

    for model in ("clone", "design", "custom"):
        try:
            data = json.dumps({"model_type": model}).encode()
            req = urllib.request.Request(
                f"{url}/load-model",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {token}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
                status = body.get("status", "unknown")
                print(f"  [setup] {model}: {status}")
        except (urllib.error.URLError, OSError) as exc:
            print(f"  [setup] {model}: failed ({exc})")


# Registry of setup functions referenced by name in batch definitions
_SETUP_FUNCTIONS = {
    "_ensure_models_loaded": _ensure_models_loaded,
}


class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    BOLD = "\033[1m"


def colorize(text: str, color: str) -> str:
    """Apply color to text if terminal supports it."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.RESET}"
    return text


def run_batch(
    batch_num: int,
    batch_info: dict,
    default_timeout: int,
    verbose: bool = True,
    auto_playwright: bool = True,
) -> tuple[bool, str]:
    """Run a single test batch in a subprocess.

    Args:
        batch_num: Batch number to run
        batch_info: Batch configuration dict
        default_timeout: Default timeout in seconds
        verbose: Enable verbose output
        auto_playwright: Auto-enable Playwright for batch 6 (default: True)

    Returns:
        (success, output) tuple
    """
    name = batch_info["name"]
    modules = batch_info["modules"]
    timeout = batch_info.get("timeout", default_timeout)

    if verbose:
        print(colorize("=" * 60, Colors.BLUE))
        print(colorize(f"Batch {batch_num}: {name}", Colors.BLUE))
        print(colorize(f"  {batch_info['description']}", Colors.BLUE))
        print(colorize(f"  Timeout: {timeout}s", Colors.BLUE))
        print(colorize("=" * 60, Colors.BLUE))

    # Auto-enable Playwright for E2E batch (batch 6)
    playwright_context = None
    if batch_num == 6 and auto_playwright:
        playwright_context = playwright_enabled(auto_enable=True)
        playwright_context.__enter__()
        if verbose:
            print("🎭 Playwright auto-enabled for E2E tests")
            print()
    elif batch_num == 6 and not auto_playwright:
        if verbose:
            print("⚠️  Running E2E tests WITHOUT auto-Playwright (manual control)")
            print("   Manually enable Playwright in .claude/.mcp.json if needed")
            print()

    # Run setup function if defined for this batch
    setup_name = batch_info.get("setup")
    if setup_name and setup_name in _SETUP_FUNCTIONS:
        if verbose:
            print(f"Running setup: {setup_name}")
        _SETUP_FUNCTIONS[setup_name]()
        if verbose:
            print()

    # Build unittest command
    cmd = [
        sys.executable,
        "-m",
        "unittest",
        *modules,
        "-v",
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)}")
        print()

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )

        output = result.stdout + result.stderr

        # Cleanup Playwright context
        if playwright_context:
            playwright_context.__exit__(None, None, None)

        return result.returncode == 0, output

    except subprocess.TimeoutExpired:
        msg = f"ERROR: Batch {batch_num} timed out after {timeout}s"
        if verbose:
            print(colorize(msg, Colors.RED))

        # Cleanup Playwright context
        if playwright_context:
            playwright_context.__exit__(None, None, None)

        return False, msg

    except Exception as e:
        msg = f"ERROR: Batch {batch_num} failed with exception: {e}"
        if verbose:
            print(colorize(msg, Colors.RED))

        # Cleanup Playwright context
        if playwright_context:
            playwright_context.__exit__(None, None, None)

        return False, msg


def run_all_batches(
    batches: dict,
    default_timeout: int,
    specific_batch: int | None = None,
    continue_on_failure: bool = False,
    verbose: bool = True,
    auto_playwright: bool = True,
) -> dict:
    """Run all or a specific test batch.

    Returns:
        Dict with 'passed', 'failed', 'skipped' lists of batch numbers
    """
    results = {"passed": [], "failed": [], "skipped": []}

    for batch_num in sorted(batches.keys()):
        if specific_batch is not None and batch_num != specific_batch:
            results["skipped"].append(batch_num)
            continue

        batch_info = batches[batch_num]
        success, output = run_batch(batch_num, batch_info, default_timeout, verbose, auto_playwright)

        if verbose and output:
            # Print test output
            print(output)

        if success:
            results["passed"].append(batch_num)
            if verbose:
                print(colorize(f"✓ Batch {batch_num} passed\n", Colors.GREEN))
        else:
            results["failed"].append(batch_num)
            if verbose:
                print(colorize(f"✗ Batch {batch_num} failed\n", Colors.RED))

            if not continue_on_failure:
                if verbose:
                    print(
                        colorize(
                            "Stopping due to failure. Use --continue to run all batches.",
                            Colors.YELLOW,
                        )
                    )
                break

    return results


def print_summary(results: dict, batches: dict):
    """Print summary of test results."""
    print(colorize("=" * 60, Colors.BLUE))
    print(colorize("Summary", Colors.BOLD))
    print(colorize("=" * 60, Colors.BLUE))

    if results["passed"]:
        print(colorize("Passed batches:", Colors.GREEN))
        for num in results["passed"]:
            info = batches[num]
            print(f"  {colorize('✓', Colors.GREEN)} Batch {num}: {info['name']}")

    if results["failed"]:
        print(colorize("Failed batches:", Colors.RED))
        for num in results["failed"]:
            info = batches[num]
            print(f"  {colorize('✗', Colors.RED)} Batch {num}: {info['name']}")

    if results["skipped"]:
        print(colorize("Skipped batches:", Colors.YELLOW))
        for num in results["skipped"]:
            info = batches[num]
            print(f"  {colorize('○', Colors.YELLOW)} Batch {num}: {info['name']}")

    total_passed = len(results["passed"])
    total_failed = len(results["failed"])
    total_ran = total_passed + total_failed

    print(colorize("=" * 60, Colors.BLUE))
    print(f"Total: {total_passed}/{total_ran} batches passed")

    if total_failed > 0:
        print(colorize("\nSome batches failed. Run specific batch to debug:", Colors.YELLOW))
        for num in results["failed"]:
            print(f"  python tests/run_batches.py --batch {num}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Qwen3-TTS tests in isolated batches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all batches
  %(prog)s --batch 2          # Run only batch 2
  %(prog)s --continue         # Continue on failure
  %(prog)s --timeout 60       # Set per-batch timeout
  %(prog)s --quiet            # Minimal output (CI mode)
        """,
    )

    parser.add_argument(
        "--batch",
        "-b",
        type=int,
        choices=range(1, len(BATCHES) + 1),
        metavar="N",
        help=f"Run only batch N (1-{len(BATCHES)})",
    )

    parser.add_argument(
        "--continue",
        "-c",
        action="store_true",
        dest="continue_on_failure",
        help="Continue running batches even if one fails",
    )

    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=300,
        metavar="SECONDS",
        help="Default timeout per batch (default: 300s, overrides batch-specific timeout)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output (useful for CI)",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all batches and exit",
    )

    parser.add_argument(
        "--no-auto-playwright",
        action="store_true",
        help="Don't auto-enable Playwright for batch 6 (manual control required)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # List batches and exit
    if args.list:
        print("Test batches:")
        for num, info in BATCHES.items():
            print(f"  Batch {num}: {info['name']}")
            print(f"    {info['description']}")
            print(f"    Modules: {len(info['modules'])}")
            print(f"    Timeout: {info['timeout']}s")
            print()
        return 0

    # Change to project root if needed
    project_root = Path(__file__).parent.parent
    if os.getcwd() != project_root:
        os.chdir(project_root)

    verbose = not args.quiet

    if verbose:
        print("Qwen3-TTS Batch Test Runner")
        print(f"Working directory: {os.getcwd()}")
        print()

    # Run batches
    results = run_all_batches(
        batches=BATCHES,
        default_timeout=args.timeout,
        specific_batch=args.batch,
        continue_on_failure=args.continue_on_failure,
        verbose=verbose,
        auto_playwright=not args.no_auto_playwright,
    )

    # Print summary
    if verbose:
        print_summary(results, BATCHES)

    # Exit code: 0 if all passed, 1 if any failed
    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
