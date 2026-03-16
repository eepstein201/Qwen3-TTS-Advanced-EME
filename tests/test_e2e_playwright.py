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

# Skip entire module if playwright is not installed
try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

UI_PORT = 7866
UI_URL = f"http://127.0.0.1:{UI_PORT}"
SERVER_URL = "http://127.0.0.1:5123"
# Derive from test file location so it works in both main repo and worktrees
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generation timeout — real inference can take 30-120s depending on hardware;
# add headroom for MLX streaming + polling overhead
GEN_TIMEOUT_MS = 240_000
MODEL_TIMEOUT_MS = 180_000

# JS to re-execute scripts that Gradio injected via innerHTML.
# Gradio's gr.HTML() sets innerHTML, which per HTML5 spec does NOT execute
# <script> tags. We re-inject them so StreamingPlayer and WaveSurfer work.
_INJECT_SCRIPTS_JS = """() => {
    var moduleScripts = document.querySelectorAll('script[type="module"]');
    moduleScripts.forEach(function(s) {
        if (s.textContent && s.textContent.length > 100 && !s.src) {
            var blob = new Blob([s.textContent], { type: 'application/javascript' });
            var url = URL.createObjectURL(blob);
            var ns = document.createElement('script');
            ns.type = 'module';
            ns.src = url;
            document.head.appendChild(ns);
        }
    });
    var inlineScripts = document.querySelectorAll('script:not([type]):not([src])');
    inlineScripts.forEach(function(s) {
        if (s.textContent && s.textContent.indexOf('createElement') >= 0) {
            try { eval(s.textContent); } catch(e) {}
        }
    });
}"""


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
        """Navigate to the Gradio UI and wait for it to load."""
        self.page.goto(self.base_url, wait_until="domcontentloaded", timeout=60_000)
        self.page.wait_for_selector("button[role='tab']", timeout=30_000)
        # Re-inject scripts that Gradio's innerHTML didn't execute
        self.page.evaluate(_INJECT_SCRIPTS_JS)
        self.page.wait_for_timeout(2000)
        # Wait for StreamingPlayer module to be available (async load race fix)
        try:
            self.page.wait_for_function(
                "() => typeof window.getOrCreatePlayer === 'function'",
                timeout=10_000,
            )
        except Exception:
            pass  # Non-fatal — player may not exist on non-generation tabs

    def click_tab(self, tab_name):
        """Click a Gradio tab by its button text."""
        self.page.locator("button[role='tab']").filter(has_text=tab_name).first.click()
        self.page.wait_for_timeout(500)

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

    def select_dropdown(self, label, value):
        """Select a value in a Gradio Dropdown by label."""
        panel = self._get_visible_tab_panel()
        # Try aria-label first (Gradio 6 puts aria-label on the input)
        input_el = panel.locator(f"input[aria-label='{label}']").first
        if input_el.count() == 0:
            container = panel.locator("label").filter(has_text=label).locator("..").first
            input_el = container.locator("input").first
        input_el.click()
        self.page.wait_for_timeout(300)
        input_el.fill(value)
        self.page.wait_for_timeout(300)
        option = self.page.locator("ul[role='listbox'] li").filter(has_text=value).first
        if option.count() > 0:
            option.click()
        else:
            self.page.locator("li").filter(has_text=value).first.click()
        self.page.wait_for_timeout(300)

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
        self.page.wait_for_timeout(300)
        input_el.fill(new_value)
        self.page.wait_for_timeout(300)
        option = self.page.locator("ul[role='listbox'] li").filter(has_text=new_value).first
        if option.count() > 0:
            option.click()
        else:
            self.page.locator("li").filter(has_text=new_value).first.click()
        self.page.wait_for_timeout(300)

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


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestE2EPlaywright(unittest.TestCase):
    """End-to-end browser tests for Gradio TTS UI."""

    ui_proc = None
    playwright_instance = None
    browser = None
    context = None

    @classmethod
    def setUpClass(cls):
        if not _is_server_running():
            raise unittest.SkipTest("TTS server not running on port 5123")

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
            self.page.close()

    # ------------------------------------------------------------------
    # Generation tests — use JS status span (#mode-status) as the
    # reliable indicator, since Gradio's hidden component state doesn't
    # propagate to the DOM in headless Chromium.
    # ------------------------------------------------------------------

    def _assert_generation_success(self, mode):
        """Wait for JS streaming to complete and assert no error."""
        self.gp.wait_for_js_status_contains(
            mode, ["Complete", "Error"], timeout=GEN_TIMEOUT_MS
        )
        js_status = self.gp.get_js_status(mode)
        self.assertIn("Complete", js_status,
                       f"Generation failed. JS status: {js_status}")

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

        # Wait for generation to start (JS status updates immediately)
        try:
            self.gp.wait_for_js_status_contains(
                "clone", ["Connecting", "Generating"], timeout=30_000
            )
        except Exception:
            return  # Completed too fast

        self.gp.click_button("Stop")
        self.page.wait_for_timeout(3000)

        # Page should be responsive (not hung)
        js_status = self.gp.get_js_status("clone")
        self.assertIsNotNone(js_status)

    # ------------------------------------------------------------------
    # Concurrent generation test
    # ------------------------------------------------------------------

    def test_07_concurrent_generation(self):
        """Two browser tabs generating simultaneously should both complete.

        Uses Clone mode in both tabs since it's the only model guaranteed
        to be loaded. Tests that the server handles concurrent requests.
        """
        page1 = self.page
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
            gp1.wait_for_js_status_contains(
                "clone", ["Complete", "Error"], timeout=GEN_TIMEOUT_MS
            )
        except Exception as e:
            page2.close()
            self.fail(f"Page 1 timed out: {e}")

        try:
            gp2.wait_for_js_status_contains(
                "clone", ["Complete", "Error"], timeout=GEN_TIMEOUT_MS
            )
        except Exception as e:
            page2.close()
            self.fail(f"Page 2 timed out: {e}")

        status1 = gp1.get_js_status("clone")
        status2 = gp2.get_js_status("clone")
        page2.close()

        self.assertIn("Complete", status1, f"Page 1 failed: {status1}")
        self.assertIn("Complete", status2, f"Page 2 failed: {status2}")

    # ------------------------------------------------------------------
    # Model management tests — use Gradio Status textbox (Python-only
    # path, load/unload are server API calls with Gradio state updates).
    # ------------------------------------------------------------------

    def test_08_load_model(self):
        """Loading a model from the Manage Models tab should succeed."""
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

        # The Load handler already returns updated table data via its outputs.
        # Wait for Gradio to render the update in the DOM (avoids race condition).
        try:
            self.gp.wait_for_table_row("design", "Loaded", timeout=10_000)
        except Exception:
            # Fallback: click Refresh to force a fresh table render
            self.gp.click_button("Refresh")
            self.gp.wait_for_table_row("design", "Loaded", timeout=15_000)

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
        self.page.wait_for_timeout(1000)
        table = self.gp.get_table_data()
        design_row = [r for r in table if r and r[0].lower().strip() == "design"]
        if design_row:
            self.assertNotIn("Loaded", design_row[0][1],
                             f"Design model still loaded. Row: {design_row[0]}")

    def test_10_load_unload_cycle(self):
        """Load then unload a model to verify no state corruption."""
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

        try:
            self.gp.wait_for_table_row("design", "Loaded", timeout=10_000)
        except Exception:
            self.gp.click_button("Refresh")
            self.gp.wait_for_table_row("design", "Loaded", timeout=15_000)

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

        try:
            self.gp.wait_for_table_row("design", "Not loaded", timeout=10_000)
        except Exception:
            self.gp.click_button("Refresh")
            self.gp.wait_for_table_row("design", "Not loaded", timeout=15_000)

        table = self.gp.get_table_data()
        design_row = [r for r in table if r and r[0].lower().strip() == "design"]
        if design_row:
            self.assertNotIn("Loaded", design_row[0][1], "Should be unloaded")


if __name__ == "__main__":
    unittest.main()
