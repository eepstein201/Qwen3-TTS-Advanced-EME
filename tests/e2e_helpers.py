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

import contextlib
import json
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path

MCP_CONFIG_PATH = Path(".claude/.mcp.json")

# Gradio versions whose Dataframe frontend is known-broken. pyproject.toml
# already excludes 6.14.* from the dependency resolution, but that pin only
# governs managed installs — see assert_supported_gradio().
BANNED_GRADIO_PREFIXES = ("6.14.",)


def assert_supported_gradio() -> None:
    """Fail fast if the interpreter about to launch the UI has a banned gradio.

    E2E harnesses start the Gradio UI as ``subprocess.Popen([sys.executable, ...])``,
    so the version actually under test is whichever interpreter ran pytest — NOT
    whatever the project pins. A bare ``python -m pytest`` can resolve to a pyenv
    shim carrying gradio 6.14.x, whose Dataframe frontend recurses infinitely
    (see CLAUDE.md); any Dataframe result gathered there is untrustworthy. This
    cost two sessions once, chasing a phantom "stale seed" bug.

    Raises:
        RuntimeError: if the resolved gradio version is known-broken.
    """
    try:
        import gradio
    except ImportError:
        # Nothing to launch with; each harness has its own skip path for that.
        return

    version = getattr(gradio, "__version__", "")
    if version.startswith(BANNED_GRADIO_PREFIXES):
        raise RuntimeError(
            f"E2E would launch the Gradio UI under gradio {version} via "
            f"{sys.executable}, which pyproject.toml excludes (!=6.14.*) because "
            f"its Dataframe frontend recurses infinitely. The UI subprocess "
            f"inherits this interpreter, so results here are not trustworthy. "
            f"Re-run naming the environment explicitly, e.g.:\n"
            f"    conda run -n qwen3-tts-mlx python -m pytest <target> -m e2e"
        )


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
                with open(MCP_CONFIG_PATH) as f:
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

    with open(MCP_CONFIG_PATH) as f:
        config = json.load(f)
        return config.get("mcpServers", {}).get("playwright", False)
