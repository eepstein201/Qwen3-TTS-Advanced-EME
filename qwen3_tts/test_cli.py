"""
Test command wrappers for CLI entry points.
Provides convenient pytest shortcuts via 'test-*' commands.
"""

import sys
from subprocess import run


def _run_pytest(args: list[str]) -> int:
    """Run pytest as a module using the current Python interpreter."""
    cmd = [sys.executable, "-m", "pytest"] + args
    result = run(cmd)
    return result.returncode


def test() -> None:
    """Run all tests."""
    sys.exit(_run_pytest(["tests/"]))


def test_unit() -> None:
    """Run unit tests only."""
    sys.exit(_run_pytest(["-m", "unit", "tests/"]))


def test_integration() -> None:
    """Run integration tests only."""
    sys.exit(_run_pytest(["-m", "integration", "tests/"]))


def test_quick() -> None:
    """Run tests excluding slow ones."""
    sys.exit(_run_pytest(["-m", "not slow", "tests/"]))


def test_parallel() -> None:
    """Run tests in parallel."""
    sys.exit(_run_pytest(["-n", "auto", "tests/"]))


def test_cov() -> None:
    """Run tests with coverage."""
    sys.exit(_run_pytest(["--cov=qwen3_tts", "tests/"]))
