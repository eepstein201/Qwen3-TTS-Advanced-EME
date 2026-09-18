"""E2E test helpers: poll/retry, gradio-version guard, rate-limit utilities.

Historical note: this module used to toggle the Playwright MCP registration in
the tracked ``.claude/.mcp.json`` around E2E runs (``@requires_playwright``,
``is_playwright_enabled``). That machinery is retired — see
``playwright_enabled`` for why — and the E2E modules import the playwright
library directly.
"""

import contextlib
import json
import sys
import time


def poll_until(predicate, timeout=15.0, interval=0.25):
    """Poll *predicate* until it returns truthy. Returns the value, or None.

    The Python-side counterpart to ``page.wait_for_function`` for conditions
    the browser cannot observe — files appearing on disk, or Gradio component
    contents simpler to read through a page object than to re-express as JS.
    Prefer ``wait_for_function`` when the condition IS in the DOM; a fixed
    ``page.wait_for_timeout`` is neither, and is the flakiness source this
    exists to replace (see docs/plans/repo-audit-2026-07-31.md, P1-1).

    Swallows predicate exceptions so a not-yet-rendered element polls rather
    than aborting; callers assert on the outcome.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            result = predicate()
            if result:
                return result
        except Exception:
            pass
        time.sleep(interval)
    return None

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


@contextlib.contextmanager
def playwright_enabled(auto_enable: bool = True):
    """Retired no-op (kept for its two call sites).

    This used to flip ``mcpServers.playwright`` in the TRACKED
    ``.claude/.mcp.json`` at setUpModule time and restore it at teardown.
    No test ever read that state — the E2E modules import the playwright
    library directly — while the mid-run mutation dirtied the tree (breaking
    ``git stash pop`` mid-E2E and tripping dirty-tree refusals) and a crashed
    run left the file modified. The toggle is gone; the context manager stays
    so ``run_batches.py`` (batch 6) and ``test_e2e_playwright.setUpModule``
    keep working unchanged.

    Args:
        auto_enable: Ignored (retained for signature compatibility).
    """
    yield


# ---------------------------------------------------------------------------
# Rate-limit helpers for live-server E2E tests
#
# The live server's /generate rate limit (default 10-20/min, in-process
# MemoryStorage, fixed-window) is shared across every e2e module that talks
# to :5123. Run the server with TTS_DISABLE_RATE_LIMITING=1 for the e2e
# profile so suites that fire many requests aren't starved (the helpers below
# then return immediately through their non-429 probes). Against an un-flagged
# server they are best-effort: assert_rejected skips on 429 rather than
# failing, since a 429 is a test-environment issue, not a product regression.
# ---------------------------------------------------------------------------


def wait_for_rate_limit_reset(server_url, token, timeout=70):
    """Block until /generate is no longer rate-limited, up to *timeout* seconds.

    Probes /generate once; sleeps ~one window only if a 429 is observed. With
    TTS_DISABLE_RATE_LIMITING on the server the probe returns a non-429 and this
    returns immediately. The probe itself consumes one /generate request.
    """
    import urllib.error
    import urllib.request

    url = f"{server_url}/generate"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    body = json.dumps({"text": "rate-limit-probe", "mode": "custom"}).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            retry_after = int(e.headers.get("Retry-After", 65))
            time.sleep(min(retry_after + 1, timeout))
    except Exception:
        pass  # Network error / non-429 — assume not rate-limited and proceed.


def assert_rejected(status, expected_codes, context):
    """Assert a request was rejected with an expected code; skip if rate-limited.

    A 429 means the shared live-server limit was hit before the request could
    be evaluated — that is a test-environment issue, not a product regression,
    so it becomes a ``pytest.skip`` rather than a failure.
    """
    import pytest

    if status == 429:
        pytest.skip(f"Rate limit exceeded before '{context}' could be verified")
    assert status in expected_codes, f"{context}, got {status}"


def first_available_voice_prompt(server_url, token):
    """Return the first available voice-prompt name (for clone mode), or None.

    Clone-mode ``/generate`` requires a ``prompt_file``; performance tests that
    exercise real clone generation must supply one, so they don't 400 on the
    missing field (which the old hollow assertions silently accepted).
    """
    import urllib.request

    req = urllib.request.Request(
        f"{server_url}/prompts", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            prompts = json.loads(resp.read().decode()).get("prompts", [])
            return prompts[0] if prompts else None
    except Exception:
        return None
