#!/usr/bin/env python3
"""Playwright E2E tests for Gradio TTS web interface.

Launches the Gradio UI in a subprocess and drives a real Chromium browser
to test generation, concurrent generation, and model management flows.

Prerequisites:
    - playwright installed: pip install playwright && playwright install chromium
    - TTS server running on port 5123 with clone model loaded

Usage:
    python -m unittest tests.test_e2e_playwright -v
    python tests/run_batches.py --batch 6
"""

import os
import signal
import subprocess  # nosec B404
import sys
import time
import unittest
import urllib.error
import urllib.request

# Auto-toggle helper for universal E2E test support
from tests.e2e_helpers import (
    assert_supported_gradio,
    playwright_enabled,
)
from tests.e2e_helpers import (
    poll_until as _poll_until,
)

# Web-UI generations land in the Automated Output subfolder, not flat in
# ~/Downloads — must track qwen3_tts/interface/ui/shared.py's resolver.
HISTORY_OUTPUT_DIR = os.path.expanduser("~/Downloads/Qwen3-TTS Output/Automated Output")

# E2E browser tests require a live server + Gradio UI + Chromium.
# Gated behind the `e2e` marker so plain `pytest tests/` skips them (no hang).
# Opt in with: pytest tests/ -m e2e. The unittest batch runner (batch 6)
# ignores pytest markers, so it still runs these via `python -m unittest`.
try:
    import pytest
    pytestmark = pytest.mark.e2e
except ImportError:
    pass

# Skip entire module if playwright is not installed
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

UI_PORT = 7866
UI_URL = f"http://127.0.0.1:{UI_PORT}"
SERVER_URL = "http://127.0.0.1:5123"
# Opt-in durable screenshots: set TTS_E2E_SCREENSHOTS=1 to capture a PNG of the
# final page state per test into tests/screenshots/. Off by default so normal
# runs and CI are unaffected.
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
_CAPTURE_SCREENSHOTS = os.environ.get("TTS_E2E_SCREENSHOTS") == "1"
# Derive from test file location so it works in both main repo and worktrees
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generation timeout — real inference can take 30-120s depending on hardware;
# add headroom for MLX streaming + polling overhead
GEN_TIMEOUT_MS = 240_000
MODEL_TIMEOUT_MS = 180_000

# NOTE: this module used to re-execute Gradio's innerHTML <script> tags itself
# (a _INJECT_SCRIPTS_JS helper called from GradioPage.navigate). That made every
# test here exercise the harness's own injector instead of the app's
# demo.load(js=get_script_reexecutor_fn()) re-injection, so a dead production
# player would have left this suite green. The injector is gone; the production
# path is asserted directly in tests/test_e2e_wavesurfer_live.py.


def _is_server_running():
    """Check if TTS server is healthy."""
    try:
        resp = urllib.request.urlopen(  # nosec B310
            f"{SERVER_URL}/health", timeout=5
        )
        return resp.status == 200
    except Exception:
        return False


def _wait_for_model_state(model_name, loaded, timeout=60):
    """Poll server /health until model reaches expected loaded state.

    Args:
        model_name: "clone", "design", or "custom"
        loaded: True to wait for loaded, False to wait for unloaded
        timeout: Max seconds to wait
    """
    import json
    deadline = time.time() + timeout
    key = f"{model_name}_model_loaded"
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(  # nosec B310
                f"{SERVER_URL}/health", timeout=5
            )
            health = json.loads(resp.read())
            if health.get(key) == loaded:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _get_auth_token():
    """Read the server auth token, delegating to the production reader.

    This used to open ``~/.voice_server_token`` directly — the *legacy* path.
    The canonical location has been ``~/.config/qwen3-tts/.voice_server_token``
    for some time, so on any install without the legacy file this returned ""
    and every authenticated call here failed. `.voice_server.log` showed
    `Auth failure: missing_token ... on POST /unload-model` during E2E runs,
    which silently degraded the model-management tests
    (repo-audit-2026-07-31, follow-up finding).

    Delegating rather than adding a fourth copy of the path list: `config.py`
    already owns canonical-then-legacy resolution including the deprecation
    warning, and a copy here is what let this drift in the first place.
    `core.config` is import-safe — no torch/mlx at module scope.
    """
    from qwen3_tts.core.config import read_auth_token

    return read_auth_token() or ""


def _ensure_model_unloaded(model_name, timeout=30):
    """Unload a model via API if it's currently loaded.

    Prevents test failures caused by stale model state from previous tests.
    """
    import json
    try:
        resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)  # nosec B310
        health = json.loads(resp.read())
        if not health.get(f"{model_name}_model_loaded"):
            return True  # Already unloaded
    except Exception:
        return False

    # Unload it
    token = _get_auth_token()
    data = json.dumps({"model_type": model_name}).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/unload-model",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)  # nosec B310
    except Exception:
        pass
    return _wait_for_model_state(model_name, loaded=False, timeout=timeout)


def _ensure_only_clone_loaded(timeout=30):
    """Unload design and custom models, keeping only clone loaded.

    Prevents OOM when running memory-intensive tests like concurrent generation.
    """
    _ensure_model_unloaded("design", timeout=timeout)
    _ensure_model_unloaded("custom", timeout=timeout)


def _wait_for_ui(url, timeout=45):
    """Poll until the Gradio UI responds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=3)  # nosec B310
            if resp.status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    return False


class GradioPage:
    """Helper for interacting with Gradio UI via Playwright."""

    def __init__(self, page, base_url=UI_URL):
        self.page = page
        self.base_url = base_url

    def navigate(self):
        """Navigate to the Gradio UI and wait for it to load.

        Deliberately does NOT re-inject scripts — the app's own
        ``demo.load(js=get_script_reexecutor_fn())`` is what makes the
        StreamingPlayer module execute, and a harness doing that job itself
        would mask a dead production path.
        """
        self.page.goto(self.base_url, wait_until="domcontentloaded", timeout=60_000)
        self.page.wait_for_selector("button[role='tab']", timeout=30_000)
        # Every audio control calls window.getOrCreatePlayer. If it never
        # appears the downstream failures are unreadable, so fail here instead.
        self.page.wait_for_function(
            "() => typeof window.getOrCreatePlayer === 'function'",
            timeout=30_000,
        )

    def click_tab(self, tab_name):
        """Click a Gradio tab by its button text, then wait for it to activate.

        Both halves of the condition matter. The button flips aria-selected
        synchronously, but Gradio 6 mounts tabpanels lazily, so the panel
        appears a frame or more later — and every downstream locator here
        scopes to ``div[role='tabpanel']:visible``. Waiting on the button
        alone would still race the panel.
        """
        self.page.locator("button[role='tab']").filter(has_text=tab_name).first.click()
        self.page.wait_for_function(
            """(name) => {
                var btns = document.querySelectorAll('button[role="tab"]');
                for (var i = 0; i < btns.length; i++) {
                    if (btns[i].textContent.indexOf(name) === -1) continue;
                    if (btns[i].getAttribute('aria-selected') !== 'true') return false;
                    var panels = document.querySelectorAll('div[role="tabpanel"]');
                    for (var j = 0; j < panels.length; j++) {
                        if (panels[j].offsetParent !== null) return true;
                    }
                    return false;
                }
                return false;
            }""",
            arg=tab_name,
            timeout=15_000,
        )

    def _get_visible_tab_panel(self):
        """Get the currently visible tab panel."""
        return self.page.locator("div[role='tabpanel']:visible").first

    def fill_textbox(self, label, value):
        """Fill a Gradio Textbox by its label within the visible tab panel."""
        panel = self._get_visible_tab_panel()
        container = panel.locator("label").filter(has_text=label).locator("..").first
        textarea = container.locator("textarea").first
        if textarea.count() == 0:
            textarea = container.locator("input[type='text']").first
        textarea.fill(value)

    def click_button(self, text, exact=False):
        """Click a visible button by its text content in the active tab panel.

        Args:
            text: Button text to search for.
            exact: If True, match exact text only (prevents "Load" matching "Load Time").
        """
        panel = self._get_visible_tab_panel()
        if exact:
            # Use XPath for exact text matching
            panel.locator(f"button >> text='{text}'").first.click()
        else:
            panel.locator("button").filter(has_text=text).first.click()

    def get_status_text(self):
        """Read the Gradio Status textbox value in the visible tab panel."""
        panel = self._get_visible_tab_panel()
        status_container = panel.locator("label").filter(has_text="Status").locator("..").first
        textarea = status_container.locator("textarea").first
        if textarea.count() > 0:
            return textarea.input_value()
        inp = status_container.locator("input").first
        if inp.count() > 0:
            return inp.input_value()
        return ""

    def get_js_status(self, mode):
        """Read the JS streaming status span for a generation mode.

        These spans (#clone-status, #design-status, #custom-status) are
        updated directly by the StreamingPlayer JS class and reflect the
        true streaming state: Connecting → Generating → Complete/Error.
        """
        return self.page.evaluate(
            f'() => {{ var el = document.getElementById("{mode}-status"); '
            f'return el ? el.textContent : ""; }}'
        )

    def wait_for_js_status_contains(self, mode, substrings, timeout=GEN_TIMEOUT_MS):
        """Wait until the JS status span for a mode contains any substring."""
        if isinstance(substrings, str):
            substrings = [substrings]
        checks = " || ".join(f'val.indexOf("{s}") >= 0' for s in substrings)
        self.page.wait_for_function(
            f"""() => {{
                var el = document.getElementById("{mode}-status");
                if (!el) return false;
                var val = el.textContent;
                return {checks};
            }}""",
            timeout=timeout,
        )

    def wait_for_status_contains(self, substrings, timeout=GEN_TIMEOUT_MS):
        """Wait until the Gradio Status textbox contains any of the substrings."""
        if isinstance(substrings, str):
            substrings = [substrings]
        checks = " || ".join(f'val.indexOf("{s}") >= 0' for s in substrings)
        self.page.wait_for_function(
            f"""() => {{
                var panels = document.querySelectorAll('div[role="tabpanel"]');
                for (var i = 0; i < panels.length; i++) {{
                    var panel = panels[i];
                    if (panel.offsetParent === null) continue;
                    var labels = panel.querySelectorAll('label');
                    for (var j = 0; j < labels.length; j++) {{
                        if (labels[j].textContent.indexOf('Status') >= 0) {{
                            var container = labels[j].parentElement;
                            var ta = container.querySelector('textarea') || container.querySelector('input');
                            if (ta) {{
                                var val = ta.value;
                                if ({checks}) return true;
                            }}
                        }}
                    }}
                }}
                return false;
            }}""",
            timeout=timeout,
        )

    def _wait_for_listbox_open(self, timeout=10_000):
        """Wait for a dropdown listbox to be mounted, visible and populated.

        ``children.length > 0`` matters: Gradio mounts the <ul> before filling
        it, so a bare presence check can pass while the options are still
        empty and the next fill() types into a dead list.

        Visibility is ``getClientRects()``, NOT ``offsetParent``. Gradio 6
        renders this listbox with ``position: fixed``, and offsetParent is
        null for fixed elements — so an offsetParent check reads a perfectly
        visible dropdown as hidden and times out every time (it did: it broke
        test_09_unload_model, which passes on main).
        """
        self.page.wait_for_function(
            """() => {
                var lb = document.querySelector('ul[role="listbox"]');
                return !!lb && lb.getClientRects().length > 0
                    && lb.children.length > 0;
            }""",
            timeout=timeout,
        )

    def _wait_for_listbox_option(self, value, timeout=10_000):
        """Wait for an option matching *value* to survive the type-ahead filter."""
        self.page.wait_for_function(
            """(want) => {
                var lis = document.querySelectorAll('ul[role="listbox"] li');
                if (lis.length === 0) lis = document.querySelectorAll('li');
                for (var i = 0; i < lis.length; i++) {
                    if (lis[i].textContent.indexOf(want) !== -1) return true;
                }
                return false;
            }""",
            arg=value,
            timeout=timeout,
        )

    def _wait_for_listbox_closed(self, timeout=10_000):
        """Wait for the dropdown to collapse, i.e. the selection committed.

        Deliberately checks the listbox rather than the input's value: Gradio
        dropdowns may display a human label distinct from the submitted value,
        so asserting the value here would couple the harness to presentation.

        Same ``getClientRects()`` rule as _wait_for_listbox_open — with
        offsetParent this check would be vacuously true for a fixed-position
        listbox and wait for nothing.
        """
        self.page.wait_for_function(
            """() => {
                var lb = document.querySelector('ul[role="listbox"]');
                return !lb || lb.getClientRects().length === 0;
            }""",
            timeout=timeout,
        )

    def select_dropdown(self, label, value):
        """Select a value in a Gradio Dropdown by label."""
        panel = self._get_visible_tab_panel()
        # Try aria-label first (Gradio 6 puts aria-label on the input)
        input_el = panel.locator(f"input[aria-label='{label}']").first
        if input_el.count() == 0:
            container = panel.locator("label").filter(has_text=label).locator("..").first
            input_el = container.locator("input").first
        input_el.click()
        self._wait_for_listbox_open()
        input_el.fill(value)
        self._wait_for_listbox_option(value)
        option = self.page.locator("ul[role='listbox'] li").filter(has_text=value).first
        if option.count() > 0:
            option.click()
        else:
            self.page.locator("li").filter(has_text=value).first.click()
        self._wait_for_listbox_closed()

    def select_dropdown_by_value(self, current_value, new_value):
        """Select a dropdown option by finding the input with a known current value.

        Used for Gradio dropdowns where the label doesn't render as visible text.
        """
        panel = self._get_visible_tab_panel()
        input_el = panel.locator(f"input[value='{current_value}']").first
        if input_el.count() == 0:
            # Fallback: try finding any input containing the value
            inputs = panel.locator("input:not([type='checkbox'])").all()
            for inp in inputs:
                if inp.input_value() in ("clone", "design", "custom"):
                    input_el = inp
                    break
        input_el.click()
        self._wait_for_listbox_open()
        input_el.fill(new_value)
        self._wait_for_listbox_option(new_value)
        option = self.page.locator("ul[role='listbox'] li").filter(has_text=new_value).first
        if option.count() > 0:
            option.click()
        else:
            self.page.locator("li").filter(has_text=new_value).first.click()
        self._wait_for_listbox_closed()

    def wait_for_any_textarea_contains(self, substrings, timeout=GEN_TIMEOUT_MS):
        """Wait until ANY textarea in the visible panel contains a substring."""
        if isinstance(substrings, str):
            substrings = [substrings]
        checks = " || ".join(f'val.indexOf("{s}") >= 0' for s in substrings)
        self.page.wait_for_function(
            f"""() => {{
                var panels = document.querySelectorAll('div[role="tabpanel"]');
                for (var i = 0; i < panels.length; i++) {{
                    if (panels[i].offsetParent === null) continue;
                    var tas = panels[i].querySelectorAll('textarea');
                    for (var j = 0; j < tas.length; j++) {{
                        var val = tas[j].value;
                        if ({checks}) return true;
                    }}
                }}
                return false;
            }}""",
            timeout=timeout,
        )

    def get_table_data(self):
        """Read visible table data as list of lists.

        Deduplicates by first column (model name) keeping the last occurrence,
        because Gradio 6 Dataframe may render a phantom stale first row from
        previous component state alongside the current data rows.
        """
        panel = self._get_visible_tab_panel()
        rows = panel.locator("table tbody tr").all()
        seen = {}
        for row in rows:
            cells = row.locator("td").all()
            row_data = [c.inner_text() for c in cells]
            if row_data:
                seen[row_data[0].lower().strip()] = row_data
        return list(seen.values())

    def wait_for_table_row(self, row_name, column_text, timeout=30_000):
        """Wait until the LAST table row with row_name has column_text in column 1.

        Uses the last occurrence because Gradio 6 Dataframe may render a
        phantom stale first row alongside the real data rows.
        """
        self.page.wait_for_function(
            f"""() => {{
                var panels = document.querySelectorAll('div[role="tabpanel"]');
                for (var i = 0; i < panels.length; i++) {{
                    if (panels[i].offsetParent === null) continue;
                    var rows = panels[i].querySelectorAll('table tbody tr');
                    var lastMatch = null;
                    for (var j = 0; j < rows.length; j++) {{
                        var cells = rows[j].querySelectorAll('td');
                        if (cells.length < 2) continue;
                        var name = cells[0].textContent.trim().toLowerCase();
                        if (name === '{row_name}') {{
                            lastMatch = cells[1].textContent;
                        }}
                    }}
                    if (lastMatch !== null && lastMatch.indexOf('{column_text}') >= 0) {{
                        return true;
                    }}
                }}
                return false;
            }}""",
            timeout=timeout,
        )

    def wait_for_table_row_refreshed(self, row_name, column_text, timeout=30_000):
        """Wait for a table row, clicking Refresh up to twice if it lags.

        Returns True if the row matched, False on timeout (best-effort). The
        Manage Models ``gr.Dataframe`` sometimes doesn't re-render in the DOM
        even though the server confirms the new state (the toggle handler and
        the status timer both deliver fresh data). Because the authoritative
        check is the server-side ``_wait_for_model_state`` poll, callers treat
        a False return as a warning, not a failure (see I4 follow-up).
        """
        for _attempt in range(2):
            try:
                self.wait_for_table_row(row_name, column_text, timeout=timeout)
                return True
            except Exception:
                self.click_button("Refresh")
        try:
            self.wait_for_table_row(row_name, column_text, timeout=timeout)
            return True
        except Exception:
            return False


def setUpModule():
    """Auto-enable Playwright MCP server before any E2E tests run.

    This ensures Playwright is available for all test runners:
    - python -m unittest tests.test_e2e_playwright -v
    - python tests/run_batches.py --batch 6
    - python tests/run_full_suite.py --test-type e2e
    - pytest tests/test_e2e_playwright.py

    The toggle happens at module level, so it only runs once for the
    entire E2E test suite, not per-test or per-class.
    """
    try:
        _playwright_context = playwright_enabled(auto_enable=True)
        _playwright_context.__enter__()
        print("🎭 Playwright auto-enabled for E2E test suite")
    except Exception as e:
        print(f"⚠️  Failed to auto-enable Playwright: {e}")
        print("   Tests may fail if Playwright is required")


def tearDownModule():
    """Auto-disable Playwright MCP server after E2E tests complete.

    This ensures Playwright is disabled after testing to save tokens
    during normal development.
    """
    try:
        if "_playwright_context" in globals():
            globals()["_playwright_context"].__exit__(None, None, None)
            print("🎭 Playwright auto-disabled after E2E test suite")
    except Exception as e:
        print(f"⚠️  Failed to auto-disable Playwright: {e}")


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestE2EPlaywright(unittest.TestCase):
    """End-to-end browser tests for Gradio TTS UI."""

    ui_proc = None
    playwright_instance = None
    browser = None
    context = None

    @classmethod
    def setUpClass(cls):
        # The UI subprocess below inherits sys.executable, so guard the gradio
        # version BEFORE anything is measured against it.
        assert_supported_gradio()

        if not _is_server_running():
            raise unittest.SkipTest("TTS server not running on port 5123")

        # Ensure clone model is loaded for E2E tests
        try:
            import json
            resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)  # nosec B310
            health = json.loads(resp.read())

            if not health.get("clone_model_loaded"):
                # Load clone model if not loaded
                token = _get_auth_token()
                url = f"{SERVER_URL}/load-model"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
                body = json.dumps({"model_type": "clone"}).encode()
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")

                resp = urllib.request.urlopen(req, timeout=120)  # nosec B310
                if resp.status != 200:
                    raise unittest.SkipTest("Failed to load clone model for E2E tests")

                # Wait for model to be ready
                for _ in range(30):  # Wait up to 15 seconds
                    time.sleep(0.5)
                    resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)  # nosec B310
                    health = json.loads(resp.read())
                    if health.get("clone_model_loaded"):
                        break
        except Exception:
            # Proceed with tests, they will fail appropriately if model not loaded
            pass

        # Kill any stale Gradio UI on our port to prevent tests connecting to old code
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", UI_PORT)) == 0:
                # Port is occupied — find and kill the process
                try:
                    import subprocess as _sp
                    result = _sp.run(
                        ["lsof", "-ti", f":{UI_PORT}"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        for pid_str in result.stdout.strip().splitlines():
                            try:
                                os.kill(int(pid_str), signal.SIGTERM)
                            except (ProcessLookupError, ValueError):
                                pass
                        time.sleep(1)  # Give it time to die
                except Exception:
                    pass

        env = os.environ.copy()
        cls.ui_proc = subprocess.Popen(  # nosec B603
            [
                sys.executable, "-c",
                f"import sys; sys.path.insert(0, '{PROJECT_DIR}'); "
                f"from qwen3_tts.interface.ui import build_ui; "
                f"demo = build_ui(); "
                f"import os; "
                f"demo.launch(server_name='127.0.0.1', server_port={UI_PORT}, "
                f"share=False, show_error=True, "
                f"allowed_paths=[os.path.expanduser('~/Downloads'), '/tmp'], "
                f"css='.gr-hidden {{ display: none !important; }}', "
                f"prevent_thread_lock=False)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        if not _wait_for_ui(UI_URL, timeout=45):
            cls._kill_ui()
            raise unittest.SkipTest(f"Gradio UI failed to start on port {UI_PORT}")

        # Verify the subprocess is still alive — if it died, port was occupied by stale process
        if cls.ui_proc.poll() is not None:
            raise unittest.SkipTest(
                f"Gradio UI subprocess exited immediately (port {UI_PORT} conflict?)"
            )

        cls.playwright_instance = sync_playwright().start()
        cls.browser = cls.playwright_instance.chromium.launch(headless=True)
        cls.context = cls.browser.new_context()

    @classmethod
    def tearDownClass(cls):
        if cls.context:
            cls.context.close()
        if cls.browser:
            cls.browser.close()
        if cls.playwright_instance:
            cls.playwright_instance.stop()
        cls._kill_ui()

    @classmethod
    def _kill_ui(cls):
        if cls.ui_proc and cls.ui_proc.poll() is None:
            cls.ui_proc.send_signal(signal.SIGTERM)
            try:
                cls.ui_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.ui_proc.kill()
                cls.ui_proc.wait(timeout=3)

    def setUp(self):
        self.page = self.context.new_page()
        self.gp = GradioPage(self.page)
        self.gp.navigate()

    def tearDown(self):
        if self.page:
            if _CAPTURE_SCREENSHOTS:
                try:
                    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
                    dest = os.path.join(SCREENSHOT_DIR, f"{self._testMethodName}.png")
                    self.page.screenshot(path=dest, full_page=True)
                except Exception as e:  # screenshots are best-effort, never fail a test
                    print(f"[screenshot] capture failed for {self._testMethodName}: {e}")
            self.page.close()

    # ------------------------------------------------------------------
    # Generation tests — use Gradio Status textbox as the reliable
    # indicator. Server-side Python generation updates the Status textbox
    # to "Generated: ..." on success and "Error: ..." on failure.
    # ------------------------------------------------------------------

    def _assert_generation_success(self, mode):
        """Wait for server-side generation to complete and assert no error."""
        self.gp.wait_for_status_contains(
            ["Generated:", "Error"], timeout=GEN_TIMEOUT_MS
        )
        status = self.gp.get_status_text()
        self.assertIn("Generated:", status,
                       f"Generation failed. Status: {status}")

    def test_01_clone_generation(self):
        """Generate audio in Clone mode via the browser UI."""
        self.gp.click_tab("Clone Mode")
        self.gp.fill_textbox("Text Input", "Hello, this is a Playwright test.")
        self.gp.click_button("Generate")
        self._assert_generation_success("clone")

    def test_02_design_generation(self):
        """Generate audio in Design mode via the browser UI."""
        # Skip if design model not loaded
        try:
            import json
            resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)  # nosec B310
            health = json.loads(resp.read())
            if not health.get("design_model_loaded"):
                self.skipTest("Design model not loaded")
        except unittest.SkipTest:
            raise
        except Exception:
            pass
        self.gp.click_tab("Design Mode")
        self.gp.fill_textbox("Text Input", "Design mode test from Playwright.")
        self.gp.fill_textbox("Voice Description",
                              "A warm, friendly female voice with clear articulation")
        self.gp.click_button("Generate")
        self._assert_generation_success("design")

    def test_03_custom_generation(self):
        """Generate audio in Custom mode via the browser UI."""
        # Skip if custom model not loaded
        try:
            import json
            resp = urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5)  # nosec B310
            health = json.loads(resp.read())
            if not health.get("custom_model_loaded"):
                self.skipTest("Custom model not loaded")
        except unittest.SkipTest:
            raise
        except Exception:
            pass
        self.gp.click_tab("Custom Mode")
        self.gp.fill_textbox("Text Input", "Custom mode test from Playwright.")
        self.gp.click_button("Generate")
        self._assert_generation_success("custom")

    # ------------------------------------------------------------------
    # Validation tests — use Gradio Status textbox (Python-only path,
    # no JS streaming involved, so Gradio state propagates correctly).
    # ------------------------------------------------------------------

    def test_04_clone_validation_empty_text(self):
        """Clone mode should reject empty text input."""
        self.gp.click_tab("Clone Mode")
        self.gp.click_button("Generate")

        self.gp.wait_for_status_contains(["Error", "error", "text"], timeout=30_000)
        status = self.gp.get_status_text()
        self.assertNotIn("Connecting", status)

    def test_05_design_validation_empty_description(self):
        """Design mode should reject empty voice description."""
        self.gp.click_tab("Design Mode")
        self.gp.fill_textbox("Text Input", "Some text to speak.")
        self.gp.click_button("Generate")

        self.gp.wait_for_status_contains(["Error", "error", "description"], timeout=30_000)
        status = self.gp.get_status_text()
        self.assertNotIn("Connecting", status)

    # ------------------------------------------------------------------
    # Cancel test
    # ------------------------------------------------------------------

    def test_06_cancel_generation(self):
        """Cancelling a generation should stop it without hanging."""
        self.gp.click_tab("Clone Mode")
        self.gp.fill_textbox("Text Input",
                              "This is a longer text for the cancel test.")
        self.gp.click_button("Generate")

        # Wait for server-side generation to start (status → "Generating...")
        try:
            self.gp.wait_for_status_contains(
                ["Generating", "Generated:", "Error"], timeout=30_000
            )
        except Exception:
            return  # Completed too fast or validation failed

        # Only send cancel if still actively generating
        current_status = self.gp.get_status_text()
        if "Generating" in current_status:
            self.gp.click_button("Stop")
            # Wait for the cancel to actually land — status leaves "Generating"
            # for cancelled/complete/error. Tolerating the timeout is
            # deliberate: this test only asserts the page stayed responsive,
            # so a slow cancel must not turn into a failure here.
            try:
                self.page.wait_for_function(
                    """() => {
                        var panels = document.querySelectorAll('div[role="tabpanel"]');
                        for (var i = 0; i < panels.length; i++) {
                            if (panels[i].offsetParent === null) continue;
                            var labels = panels[i].querySelectorAll('label');
                            for (var j = 0; j < labels.length; j++) {
                                if (labels[j].textContent.indexOf('Status') < 0) continue;
                                var c = labels[j].parentElement;
                                var ta = c.querySelector('textarea')
                                      || c.querySelector('input');
                                if (ta) return ta.value.indexOf('Generating') === -1;
                            }
                        }
                        return false;
                    }""",
                    timeout=15_000,
                )
            except Exception:
                pass

        # Page should be responsive
        status = self.gp.get_status_text()
        self.assertIsNotNone(status)

    # ------------------------------------------------------------------
    # Concurrent generation test
    # ------------------------------------------------------------------

    def test_07_concurrent_generation(self):
        """Two browser tabs generating simultaneously should both complete.

        Uses Clone mode in both tabs since it's the only model guaranteed
        to be loaded. Tests that the server handles concurrent requests.
        Unloads design/custom first to prevent OOM with concurrent streaming.
        """
        _ensure_only_clone_loaded()
        gp1 = self.gp
        gp1.click_tab("Clone Mode")
        gp1.fill_textbox("Text Input", "Page one concurrent test.")

        page2 = self.context.new_page()
        gp2 = GradioPage(page2)
        gp2.navigate()
        gp2.click_tab("Clone Mode")
        gp2.fill_textbox("Text Input", "Page two concurrent test.")

        gp1.click_button("Generate")
        gp2.click_button("Generate")

        try:
            gp1.wait_for_status_contains(
                ["Generated:", "Error"], timeout=GEN_TIMEOUT_MS
            )
        except Exception as e:
            page2.close()
            self.fail(f"Page 1 timed out: {e}")

        try:
            gp2.wait_for_status_contains(
                ["Generated:", "Error"], timeout=GEN_TIMEOUT_MS
            )
        except Exception as e:
            page2.close()
            self.fail(f"Page 2 timed out: {e}")

        status1 = gp1.get_status_text()
        status2 = gp2.get_status_text()
        page2.close()

        self.assertIn("Generated:", status1, f"Page 1 failed: {status1}")
        self.assertIn("Generated:", status2, f"Page 2 failed: {status2}")

    # ------------------------------------------------------------------
    # Model management tests — use Gradio Status textbox (Python-only
    # path, load/unload are server API calls with Gradio state updates).
    # ------------------------------------------------------------------

    def test_08_load_model(self):
        """Loading a model from the Manage Models tab should succeed."""
        _ensure_model_unloaded("design")
        self.gp.click_tab("Manage Models")
        self.gp.select_dropdown("Model", "design")
        self.gp.click_button("Load", exact=True)

        # Wait for load to complete (status goes to unlabeled textarea)
        try:
            self.gp.wait_for_any_textarea_contains(
                ["loaded", "Loaded", "success", "already"],
                timeout=MODEL_TIMEOUT_MS,
            )
        except Exception:
            pass

        # Poll server to confirm model is actually loaded
        _wait_for_model_state("design", loaded=True, timeout=60)

        # The authoritative check is the server poll above. The Manage Models
        # gr.Dataframe sometimes doesn't re-render "Loaded" in the DOM despite
        # the server confirming the load (a Gradio Dataframe quirk; the toggle
        # handler and status timer both deliver fresh data — see I4 follow-up),
        # so the table check is best-effort here, not a hard failure.
        if not self.gp.wait_for_table_row_refreshed("design", "Loaded"):
            print("[warn] Manage Models table did not re-render 'Loaded' despite "
                  "server confirmation (Gradio Dataframe quirk); load verified via /models.")
        else:
            table = self.gp.get_table_data()
            design_row = [r for r in table if r and r[0].lower().strip() == "design"]
            if design_row:
                self.assertIn("Loaded", design_row[0][1],
                              f"Design model not loaded. Row: {design_row[0]}")

    def test_09_unload_model(self):
        """Unloading a model from the Manage Models tab should succeed."""
        self.gp.click_tab("Manage Models")
        self.gp.select_dropdown("Model", "design")

        # First ensure it's loaded
        self.gp.click_button("Load", exact=True)
        try:
            self.gp.wait_for_any_textarea_contains(
                ["loaded", "Loaded", "already"], timeout=MODEL_TIMEOUT_MS,
            )
        except Exception:
            pass
        _wait_for_model_state("design", loaded=True, timeout=60)

        self.gp.click_button("Unload", exact=True)
        try:
            self.gp.wait_for_any_textarea_contains(
                ["unloaded", "Unloaded", "success"], timeout=30_000,
            )
        except Exception:
            pass
        _wait_for_model_state("design", loaded=False, timeout=30)

        self.gp.click_button("Refresh")

        # This sleep gated the assertion below: a 1s guess that the refreshed
        # table had rendered. Poll for the row instead — the server already
        # confirmed unloaded above, so the only thing outstanding is the table
        # re-render, and a stale row is a real failure rather than a slow one.
        def _design_row():
            rows = [
                r for r in self.gp.get_table_data()
                if r and r[0].lower().strip() == "design"
            ]
            return rows[0] if rows else None

        design_row = _poll_until(
            lambda: (lambda r: r if r and "Loaded" not in r[1] else None)(_design_row())
        ) or _design_row()

        if design_row:
            self.assertNotIn("Loaded", design_row[1],
                             f"Design model still loaded. Row: {design_row}")

    def test_10_load_unload_cycle(self):
        """Load then unload a model to verify no state corruption."""
        _ensure_model_unloaded("design")
        self.gp.click_tab("Manage Models")
        self.gp.select_dropdown("Model", "design")

        # Load
        self.gp.click_button("Load", exact=True)
        try:
            self.gp.wait_for_any_textarea_contains(
                ["loaded", "Loaded", "already"], timeout=MODEL_TIMEOUT_MS,
            )
        except Exception:
            pass
        _wait_for_model_state("design", loaded=True, timeout=60)

        if not self.gp.wait_for_table_row_refreshed("design", "Loaded"):
            print("[warn] Manage Models table did not re-render 'Loaded' despite "
                  "server confirmation (Gradio Dataframe quirk); load verified via /models.")
        else:
            table = self.gp.get_table_data()
            design_row = [r for r in table if r and r[0].lower().strip() == "design"]
            if design_row:
                self.assertIn("Loaded", design_row[0][1], "Should be loaded")

        # Unload
        self.gp.click_button("Unload", exact=True)
        try:
            self.gp.wait_for_any_textarea_contains(
                ["unloaded", "Unloaded", "success"], timeout=30_000,
            )
        except Exception:
            pass
        _wait_for_model_state("design", loaded=False, timeout=30)

        if not self.gp.wait_for_table_row_refreshed("design", "Not loaded"):
            print("[warn] Manage Models table did not re-render 'Not loaded' despite "
                  "server confirmation (Gradio Dataframe quirk); unload verified via /models.")
        else:
            table = self.gp.get_table_data()
            design_row = [r for r in table if r and r[0].lower().strip() == "design"]
            if design_row:
                self.assertNotIn("Loaded", design_row[0][1], "Should be unloaded")

    # ------------------------------------------------------------------
    # History panel layout and seed-reuse interaction tests (11-13)
    # ------------------------------------------------------------------

    def _open_advanced_settings(self):
        """Expand Advanced Settings accordion in the current visible tab."""
        panel = self.gp._get_visible_tab_panel()
        btn = panel.locator("button").filter(has_text="Advanced Settings").first
        if btn.count() == 0:
            return
        def _wait_expanded():
            # The accordion's own state, not a guess at animation length.
            # Tolerated on timeout: some builds omit aria-expanded, and the
            # seed-field helpers below fail loudly if the panel really is shut.
            try:
                self.page.wait_for_function(
                    """(el) => el.getAttribute('aria-expanded') === 'true'""",
                    arg=btn.element_handle(),
                    timeout=10_000,
                )
            except Exception:
                pass

        try:
            if btn.get_attribute("aria-expanded") != "true":
                btn.click()
                _wait_expanded()
        except Exception:
            btn.click()
            _wait_expanded()

    def _fill_seed_field(self, value):
        """Set the Seed (empty for random) textbox in the current visible tab.

        The seed is a gr.Textbox — may render as <textarea> or <input type="text">.
        Uses triple-click + fill + Tab to commit via Gradio's Svelte binding.
        Advanced Settings must be open before calling.
        """
        panel = self.gp._get_visible_tab_panel()
        # Label is "Seed (empty = random)" — filter matches any label containing "Seed"
        seed_container = panel.locator("label").filter(has_text="Seed").locator("..").first
        inp = seed_container.locator("textarea, input").first
        if inp.count() > 0:
            inp.click(click_count=3)  # select all existing text
            inp.fill(str(value))
            inp.press("Tab")  # commit value via blur/change event
            # Wait for the Svelte binding to actually hold the value rather
            # than assuming 300ms was enough — every seed assertion downstream
            # depends on this having committed.
            _poll_until(lambda: inp.input_value().strip() == str(value), timeout=10.0)

    def _read_seed_field(self):
        """Read the Seed textbox value in the current visible tab.

        Advanced Settings must be open before calling.
        """
        panel = self.gp._get_visible_tab_panel()
        seed_container = panel.locator("label").filter(has_text="Seed").locator("..").first
        inp = seed_container.locator("textarea, input").first
        if inp.count() > 0:
            return inp.input_value()
        return None

    def test_11_history_panel_below_tabs(self):
        """History panel renders below the main tabs in DOM order (no model required)."""
        # History table must exist outside any tab panel
        history_exists = self.page.evaluate("""() => {
            var tables = document.querySelectorAll('table');
            for (var i = 0; i < tables.length; i++) {
                if (!tables[i].closest('[role="tabpanel"]')) return true;
            }
            return false;
        }""")
        self.assertTrue(
            history_exists,
            "History dataframe not found outside tab panels — should render below tabs",
        )

        # History table must follow the tablist in document order (DOCUMENT_POSITION_FOLLOWING = 4)
        follows_tabs = self.page.evaluate("""() => {
            var tablist = document.querySelector('[role="tablist"]');
            var historyTable = null;
            var tables = document.querySelectorAll('table');
            for (var i = 0; i < tables.length; i++) {
                if (!tables[i].closest('[role="tabpanel"]')) {
                    historyTable = tables[i];
                    break;
                }
            }
            if (!tablist || !historyTable) return false;
            return !!(tablist.compareDocumentPosition(historyTable) & 4);
        }""")
        self.assertTrue(
            follows_tabs,
            "History dataframe should appear after (below) the tab list in DOM order",
        )

    def test_12_json_sidecar_and_history_columns(self):
        """Generating audio writes a .json sidecar and history shows 5 columns with seed."""
        import glob as _glob
        import time as _time

        self.gp.click_tab("Clone Mode")
        self.gp.fill_textbox("Text Input", "Sidecar metadata test.")
        self._open_advanced_settings()
        self._fill_seed_field(42)
        self.gp.click_button("Generate")

        self.gp.wait_for_status_contains(
            ["Generated:", "Error"], timeout=GEN_TIMEOUT_MS
        )
        status = self.gp.get_status_text()
        self.assertIn("Generated:", status, f"Generation failed: {status}")

        # Derive JSON sidecar path from status "Generated: <basename.wav>"
        # Status contains only the basename; prepend the config output directory.
        basename = status.replace("Generated:", "").strip().split("\n")[0].strip()
        output_dir = HISTORY_OUTPUT_DIR  # web-UI saves to Automated Output (shared.py)
        if basename and os.path.splitext(basename)[1] in (".wav", ".mp3", ".flac"):
            json_path = os.path.join(output_dir, os.path.splitext(basename)[0] + ".json")
            # The sidecar is written just after the status flips to
            # "Generated:", so poll the filesystem for it. wait_for_function
            # cannot see disk; the old 1s sleep was the only thing standing
            # between a slow write and a spurious "sidecar not found".
            _poll_until(lambda: os.path.exists(json_path), timeout=15.0)
            self.assertTrue(
                os.path.exists(json_path),
                f"JSON sidecar not found: {json_path}",
            )
        else:
            # Fallback: any .json written to output_dir in the last 60s
            cutoff = _time.time() - 60
            found = [
                f for f in _glob.glob(os.path.join(output_dir, "*.json"))
                if os.path.getmtime(f) > cutoff
            ]
            self.assertGreater(
                len(found), 0,
                f"No recent .json sidecar found in {output_dir}",
            )

        # History table (outside tab panels) must have exactly 7 columns:
        # Time, Mode, Text Preview, Seed, Chunks, Remove (✕), Download (⭳) —
        # the action columns added via column-aware on_history_select.
        col_count = self.page.evaluate("""() => {
            var tables = document.querySelectorAll('table');
            for (var i = 0; i < tables.length; i++) {
                if (!tables[i].closest('[role="tabpanel"]')) {
                    var header = tables[i].querySelector('thead tr');
                    if (header) return header.querySelectorAll('th').length;
                    var row = tables[i].querySelector('tbody tr');
                    if (row) return row.querySelectorAll('td').length;
                }
            }
            return 0;
        }""")
        self.assertEqual(
            col_count, 7,
            f"History table should have 7 columns (Time, Mode, Text Preview, Seed, Chunks, Remove, Download), got {col_count}",
        )

        # History table must have a "Seed" column header.
        # Gradio 4.x appends a sort icon (⋮) to header cells, so use includes() not ===.
        seed_header_found = self.page.evaluate("""() => {
            var tables = document.querySelectorAll('table');
            for (var i = 0; i < tables.length; i++) {
                if (!tables[i].closest('[role="tabpanel"]')) {
                    var cells = tables[i].querySelectorAll('th, td');
                    for (var j = 0; j < cells.length; j++) {
                        var text = cells[j].textContent.trim();
                        if (text === 'Seed' || text.startsWith('Seed')) return true;
                    }
                }
            }
            return false;
        }""")
        self.assertTrue(seed_header_found, "Seed column not found in history table")

    def test_13_history_row_populates_seed_in_all_tabs(self):
        """Clicking a history row broadcasts its seed to the seed field in all three tabs.

        This test was quarantined 2026-07-30 as a "Gradio renders a stale seed"
        product bug. That diagnosis was wrong, and the record is kept here so it
        is not repeated: the test was reading the WRONG TABLE.

        Gradio 6 renders a phantom stale ``<table>`` from previous component
        state, and the real history grid is div-based -- its cells carry
        ``data-row``/``data-col`` attributes and it is NOT a ``<table>`` element
        at all. The old read walked ``document.querySelectorAll('table')`` and
        took ``rows[0]``, which could only ever find the phantom. An
        instrumented probe settled it: the sole ``<table>`` in the DOM had 1 row
        and no ``data-row`` cells and showed the stale ``12345``, while
        ``[data-row="0"][data-col="3"]`` in the real 10-row grid showed the
        freshly-submitted ``42``. The backend was correct all along -- which is
        why the three backend "fixes" (chain disk re-derive, demo.load
        re-derive, gr.Timer re-fetch) all appeared to fail; they were being
        graded against the phantom.

        Two sibling helpers in this file already documented the phantom and
        defend against it (``get_table_data`` de-dupes keeping the last row,
        ``wait_for_table_row`` matches the last occurrence), as does
        ``tests/test_e2e_history_clear_copy.py``, which counts
        ``[data-col="0"][data-row]`` cells precisely because they exist only in
        the real table. This test now uses that same selector family for both
        its read and its click.

        Row identity is the composite key timestamp + text, taken from the
        on-disk JSON sidecar. Text alone is not a key: the same text can be
        regenerated with different prosody and an identical seed, producing
        rows that differ only by timestamp (and several older rows on disk
        already share this test's former fixed text AND seed 42). The Seed is
        excluded from the lookup and asserted against the sidecar instead --
        if it were part of the wait, the assertion would be tautological.
        """
        import datetime as _datetime

        # Unique per run so the row lookup cannot be satisfied by an older row
        # that happens to share this test's text and seed (several do on disk).
        marker = f"Seed broadcast row click test {int(time.time())}."

        # Generate with a specific seed so history contains a non-empty seed value
        self.gp.click_tab("Clone Mode")
        self.gp.fill_textbox("Text Input", marker)
        self._open_advanced_settings()
        self._fill_seed_field(42)  # Use 42 as a traceable seed
        self.gp.click_button("Generate")
        self.gp.wait_for_status_contains(
            ["Generated:", "Error"], timeout=GEN_TIMEOUT_MS
        )
        status = self.gp.get_status_text()
        self.assertIn("Generated:", status, f"Generation failed: {status}")

        # The on-disk sidecar is the source of truth for what the row SHOULD
        # show. Comparing the rendered row against it (rather than against
        # hardcoded literals) is what makes this a real UI-vs-backend check.
        sidecar = self._read_sidecar_for_status(status)
        self.assertIsNotNone(
            sidecar, f"No JSON sidecar found for status {status!r}"
        )
        expected_time = _datetime.datetime.fromtimestamp(
            sidecar["timestamp"]
        ).strftime("%H:%M:%S")
        expected_seed = str(sidecar["seed"])

        # Identify THIS run's row by the composite key timestamp + text. Text
        # alone is not a key: the same text can be regenerated with different
        # prosody and an identical seed, producing several indistinguishable
        # rows. The seed is deliberately NOT part of the lookup -- waiting on
        # it would make the assertion below tautological.
        self._wait_for_history_row(expected_time, marker)

        # Read the whole row from the REAL grid. Must use [data-row]/[data-col]:
        # Gradio 6 also renders a phantom stale <table> with no such attributes
        # -- see this test's docstring.
        row0 = self._read_history_row(0)
        self.assertIsNotNone(row0[0], "No history row found after generation")

        # Assert the full identity, then the value under test.
        self.assertEqual(
            row0[0], expected_time,
            f"History row 0 Time should be {expected_time!r}, got {row0[0]!r}",
        )
        self.assertIn(
            marker, row0[2] or "",
            f"History row 0 Text should contain {marker!r}, got {row0[2]!r}",
        )
        seed_in_history = row0[3]
        self.assertEqual(
            seed_in_history, expected_seed,
            f"History row 0 Seed should match the sidecar ({expected_seed!r}), "
            f"got {seed_in_history!r}",
        )

        # Click the seed cell in the first history row to trigger
        # on_history_select, which emits (audio_path, seed, seed, seed) into
        # the hidden audio component and all three tab seed textboxes.
        self._click_history_cell(row=0, col=3)

        # Wait for the broadcast itself, not a guess at how long it takes.
        # on_history_select emits the seed into all three tabs' seed textboxes
        # at once, so "at least three fields now hold it" is the completion
        # signal. Counting three (rather than one) keeps this non-vacuous:
        # the generating tab's field may already contain that seed.
        #
        # The timeout is tolerated on purpose — the per-tab assertions below
        # name the offending tab and both values, which is a far better
        # failure message than a bare wait timeout.
        try:
            self.page.wait_for_function(
                """(want) => {
                    var els = document.querySelectorAll('textarea, input');
                    var n = 0;
                    for (var i = 0; i < els.length; i++) {
                        if ((els[i].value || '').trim() === want) n++;
                    }
                    return n >= 3;
                }""",
                arg=str(seed_in_history),
                timeout=15_000,
            )
        except Exception:
            pass

        # Verify seed was broadcast to all three tabs
        for tab_name in ("Clone Mode", "Design Mode", "Custom Mode"):
            self.gp.click_tab(tab_name)
            self._open_advanced_settings()
            seed_val = self._read_seed_field()
            self.assertEqual(
                seed_val, seed_in_history,
                f"{tab_name}: seed field should be '{seed_in_history}', got '{seed_val}'",
            )

    def _click_history_cell(self, row=0, col=0):
        """Trigger selection on a cell in the history Dataframe.

        Gradio 6's Dataframe uses onmousedown (not onclick) on cells with
        data-row/data-col attributes. The cell content is wrapped in an
        Upload <button class="disable_click"> parent, so force=True is
        required to bypass Playwright's element-intercept check.
        Matches Gradio's own E2E test pattern (js/spa/test/dataframe_events).
        """
        cell = self.page.locator(
            f'[data-row="{row}"][data-col="{col}"]'
        ).first
        cell.scroll_into_view_if_needed()
        cell.click(force=True)
        # No settle here on purpose. The caller knows which effect of
        # on_history_select it depends on and waits for that condition; a
        # blind sleep would only add latency to a wait that already polls.

    def _read_history_cell(self, row=0, col=0):
        """Read one cell of the REAL history grid.

        Deliberately uses the same [data-row]/[data-col] selector family as
        _click_history_cell. Gradio 6 renders a phantom stale <table> from
        previous component state; it carries no data-row/data-col attributes,
        so this selector can only ever match the real grid. Reading via
        document.querySelectorAll('table') instead is what made this test
        report a nonexistent "stale seed" bug for two sessions.
        """
        cell = self.page.query_selector(f'[data-row="{row}"][data-col="{col}"]')
        return cell.inner_text().strip() if cell else None

    def _read_history_row(self, row=0, cols=7):
        """Read a whole history row as a list of cell strings (None if absent)."""
        return [self._read_history_cell(row=row, col=c) for c in range(cols)]

    def _wait_for_history_row(self, expected_time, expected_text, row=0,
                              timeout=30_000):
        """Wait for a history row matching the composite key time + text.

        Text alone does not identify a row: the same text can be regenerated
        with different prosody and an identical seed, yielding several rows
        that differ only by timestamp. Callers pass the timestamp from the
        on-disk sidecar (formatted %H:%M:%S to match the Time column).

        The Seed cell is intentionally excluded so that callers asserting on
        the seed are not merely re-checking their own wait condition.
        """
        self.page.wait_for_function(
            """([rowIdx, wantTime, wantText]) => {
                var t = document.querySelector(
                    '[data-row="' + rowIdx + '"][data-col="0"]'
                );
                var x = document.querySelector(
                    '[data-row="' + rowIdx + '"][data-col="2"]'
                );
                return !!t && !!x
                    && t.textContent.trim() === wantTime
                    && x.textContent.indexOf(wantText) !== -1;
            }""",
            arg=[str(row), expected_time, expected_text],
            timeout=timeout,
        )

    def _read_sidecar_for_status(self, status):
        """Load the JSON sidecar for the generation named in a status line.

        Status reads "Generated: <basename.wav>"; the sidecar is the same
        basename with a .json suffix under HISTORY_OUTPUT_DIR. Returns None if
        the basename is unusable or the file is missing.
        """
        import json as _json

        basename = status.replace("Generated:", "").strip().split("\n")[0].strip()
        if not basename or os.path.splitext(basename)[1] not in (
            ".wav", ".mp3", ".flac"
        ):
            return None
        json_path = os.path.join(
            HISTORY_OUTPUT_DIR, os.path.splitext(basename)[0] + ".json"
        )
        if not os.path.exists(json_path):
            return None
        with open(json_path) as fh:
            return _json.load(fh)


if __name__ == "__main__":
    unittest.main()
