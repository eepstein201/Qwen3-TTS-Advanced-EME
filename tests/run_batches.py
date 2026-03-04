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
import os
import subprocess
import sys
from pathlib import Path


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
            "tests.test_config",
        ],
        "timeout": 60,  # Quick tests
    },
    2: {
        "name": "Voice & CLI",
        "description": "Medium risk - minimal external deps",
        "modules": [
            "tests.test_voice",
            "tests.test_cli_daemonization",
            "tests.test_caching",
            "tests.test_server_helpers",
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
            "tests.test_async_concurrency",
            "tests.test_remediation_2026_03_03",
            "tests.test_remediation_2026_03_04",
        ],
        "timeout": 180,  # Higher timeout for async operations
    },
    4: {
        "name": "Engine & UI",
        "description": "Highest risk - model loading, Gradio Timer",
        "modules": [
            "tests.test_engine",
            "tests.test_generate_server_fallback",
            "tests.test_ui_headless",
        ],
        "timeout": 240,  # Longest timeout
    },
    # Optional batch (requires additional dependencies)
    5: {
        "name": "Optional Tests",
        "description": "Tests requiring optional dependencies (pytest, etc.)",
        "modules": [
            "tests.test_flash_attn_install",
        ],
        "timeout": 30,
    },
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
) -> tuple[bool, str]:
    """Run a single test batch in a subprocess.

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
        return result.returncode == 0, output

    except subprocess.TimeoutExpired:
        msg = f"ERROR: Batch {batch_num} timed out after {timeout}s"
        if verbose:
            print(colorize(msg, Colors.RED))
        return False, msg

    except Exception as e:
        msg = f"ERROR: Batch {batch_num} failed with exception: {e}"
        if verbose:
            print(colorize(msg, Colors.RED))
        return False, msg


def run_all_batches(
    batches: dict,
    default_timeout: int,
    specific_batch: int | None = None,
    continue_on_failure: bool = False,
    verbose: bool = True,
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
        success, output = run_batch(batch_num, batch_info, default_timeout, verbose)

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
        print(colorize(f"\nSome batches failed. Run specific batch to debug:", Colors.YELLOW))
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
    )

    # Print summary
    if verbose:
        print_summary(results, BATCHES)

    # Exit code: 0 if all passed, 1 if any failed
    return 0 if not results["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
