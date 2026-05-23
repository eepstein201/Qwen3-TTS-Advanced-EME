"""
E2E test helpers with automatic Playwright MCP toggle.

When E2E tests run, this automatically:
1. Enables Playwright in .mcp.json before tests
2. Runs the tests
3. Disables Playwright after tests complete

Usage:
    @requires_playwright
    def test_ui_something():
        # This test will auto-enable Playwright

    # Or skip the toggle:
    @requires_playwright(auto_enable=False)
    def test_with_manual_playwright():
        # You must manually enable Playwright first
"""

import json
import contextlib
from pathlib import Path
from functools import wraps
from typing import Callable, Optional

MCP_CONFIG_PATH = Path(".claude/.mcp.json")


def _set_playwright(enabled: bool) -> None:
    """Enable or disable Playwright in .mcp.json."""
    if not MCP_CONFIG_PATH.exists():
        # Create default config if it doesn't exist
        MCP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        default_config = {"mcpServers": {}}
        MCP_CONFIG_PATH.write_text(json.dumps(default_config, indent=2))

    with open(MCP_CONFIG_PATH, "r+") as f:
        config = json.load(f)
        config["mcpServers"] = config.get("mcpServers", {})

        if enabled:
            config["mcpServers"]["playwright"] = True
        else:
            config["mcpServers"]["playwright"] = False

        f.seek(0)
        f.write(json.dumps(config, indent=2))
        f.truncate()


@contextlib.contextmanager
def playwright_enabled(auto_enable: bool = True):
    """
    Context manager to temporarily enable Playwright for E2E tests.

    Args:
        auto_enable: If False, skip the toggle (manual control required)

    Example:
        with playwright_enabled():
            run_e2e_tests()
    """
    original_state = None

    try:
        if auto_enable:
            if MCP_CONFIG_PATH.exists():
                with open(MCP_CONFIG_PATH, "r") as f:
                    config = json.load(f)
                    original_state = config.get("mcpServers", {}).get("playwright", False)
                _set_playwright(enabled=True)
                print("🎭 Playwright enabled for E2E tests")
            else:
                # No .claude/.mcp.json (Docker, CI, fresh checkout). Skip the toggle —
                # Batch 6 will still detect-and-skip when no server is running, and
                # individual E2E tests can manage their own playwright state.
                print(f"🎭 {MCP_CONFIG_PATH} not found; skipping playwright toggle")

        yield

    finally:
        if auto_enable and original_state is not None:
            # Restore original state
            _set_playwright(enabled=original_state)
            print(f"🎭 Playwright restored to: {original_state}")


def requires_playwright(auto_enable: bool = True):
    """
    Decorator to auto-enable Playwright for E2E test functions.

    Args:
        auto_enable: If False, skip auto-toggle (you must manually enable)

    Example:
        @requires_playwright()
        def test_ui_generation():
            # Playwright auto-enabled before this runs
            assert ui_works()

        @requires_playwright(auto_enable=False)
        def test_manual_playwright():
            # You must manually enable Playwright first
            assert ui_works()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with playwright_enabled(auto_enable=auto_enable):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def is_playwright_enabled() -> bool:
    """Check if Playwright is currently enabled in .mcp.json."""
    if not MCP_CONFIG_PATH.exists():
        return False

    with open(MCP_CONFIG_PATH, "r") as f:
        config = json.load(f)
        return config.get("mcpServers", {}).get("playwright", False)
