"""Shared Gradio E2E test harness: launch helpers + the Page Object Model.

Extracted (Step 4C, consolidated backlog plan) from four near-duplicate
copies previously living in tests/test_e2e_playwright.py,
tests/test_e2e_history_clear_copy.py, tests/test_e2e_tab_navigation.py, and
tests/test_e2e_wavesurfer_live.py. This is a pure extraction, not a rewrite:
``launch_ui`` is parameterized so each call site continues to launch the UI
exactly as it did before. Two call sites pass ``allowed_paths`` + ``css``
(serving ``~/Downloads`` and hiding ``.gr-hidden`` elements); two pass
neither. Do not default either argument to a non-None value — that would
change behavior for the minimal call sites.

``GradioPage`` is the project's real Page Object Model, moved verbatim from
tests/test_e2e_playwright.py (only ``base_url`` was made a required argument
instead of defaulting to a module-level UI_URL that no longer exists here).
"""

import os
import signal
import socket
import subprocess  # nosec B404
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Derive from this file's location so it works in both the main repo and worktrees.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Generation timeout — real inference can take 30-120s depending on hardware;
# add headroom for MLX streaming + polling overhead
GEN_TIMEOUT_MS = 240_000


def wait_for_ui(url, timeout=45):
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


def kill_stale_ui_on_port(port):
    """SIGTERM anything listening on `port` so tests never run against stale code."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            try:
                result = subprocess.run(  # nosec B603
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True, text=True, timeout=5,
                )
                for pid_str in (result.stdout or "").strip().splitlines():
                    try:
                        os.kill(int(pid_str), signal.SIGTERM)
                    except (ProcessLookupError, ValueError):
                        pass
                time.sleep(1)  # Give it time to die
            except Exception:
                pass


def launch_ui(port, *, allowed_paths=None, css=None, project_dir=PROJECT_DIR):
    """Launch the Gradio UI as a subprocess for E2E testing.

    ``allowed_paths`` and ``css`` are emitted into the ``demo.launch(...)``
    call only when not None, so the two minimal call sites (tab_navigation,
    wavesurfer_live) produce the same launch they produced before this
    extraction, while the two richer call sites (playwright,
    history_clear_copy) keep serving ``~/Downloads``/``/tmp`` and hiding
    ``.gr-hidden`` elements. ``prevent_thread_lock=False`` was only ever
    present alongside the richer form, so it is tied to that same condition.
    """
    code = (
        f"import sys; sys.path.insert(0, {project_dir!r}); "
        f"from qwen3_tts.interface.ui import build_ui; "
        f"demo = build_ui(); "
        f"demo.launch(server_name='127.0.0.1', server_port={port}, "
        f"share=False, show_error=True"
    )
    richer = allowed_paths is not None or css is not None
    if allowed_paths is not None:
        code += f", allowed_paths={allowed_paths!r}"
    if css is not None:
        code += f", css={css!r}"
    if richer:
        code += ", prevent_thread_lock=False"
    code += ")"

    return subprocess.Popen(  # nosec B603
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def stop_ui(proc):
    """Terminate a UI subprocess launched by launch_ui(), escalating to kill."""
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


# Failure traces: Playwright tracing that only persists when a test fails.
# Traces land in .reports/e2e-traces/<TestClass>.<test>.zip (gitignored with
# the rest of .reports/) — until these existed, a red E2E left no artifact
# beyond the playwright module's own failure screenshots, which is what made
# Step 4B's transient crash undiagnosable after the fact.
TRACE_DIR = Path(".reports/e2e-traces")


def start_tracing(context) -> None:
    """Begin Playwright tracing on a browser context (never fatal)."""
    try:
        context.tracing.start(screenshots=True, snapshots=True)
    except Exception:
        pass  # tracing must never break the test it is observing


def _test_failed(tc) -> bool:
    """Read the unittest outcome mid-tearDown (body failures are recorded
    before tearDown runs; tearDown's own are not, which is what we want).

    py<=3.10 exposed ``_outcome.errors`` as (test, str) pairs; 3.11 moved to
    ``_outcome.result`` (the TestResult) — verified live on 3.11.14.
    """
    outcome = getattr(tc, "_outcome", None)
    if outcome is None:
        return False
    if hasattr(outcome, "errors"):
        pairs = [pair for pair in outcome.errors if pair]
        return any(pair[0] is tc for pair in pairs)
    result = getattr(outcome, "result", None)
    if result is None:
        return False
    pairs = list(getattr(result, "failures", [])) + list(getattr(result, "errors", []))
    return any(t is tc for t, _ in pairs)


def save_trace_if_failed(tc, context, name: str) -> None:
    """Stop tracing; persist the trace only when the test failed.

    Call at the TOP of tearDown, before the context closes. Swallows every
    tracing error — an artifact helper must never mask the test's own result.
    """
    try:
        if _test_failed(tc):
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            context.tracing.stop(path=str(TRACE_DIR / f"{name}.zip"))
        else:
            context.tracing.stop()
    except Exception:
        pass


class GradioPage:
    """Helper for interacting with Gradio UI via Playwright."""

    def __init__(self, page, base_url):
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

    def click_confirm_button(self, text, timeout_ms=4_000):
        """Click a two-step ConfirmButton through BOTH of its steps.

        Destructive actions (Unload, Delete) are wired to ``ConfirmButton``:
        the first click only ARMS the button and relabels it
        "Confirm <text>? (click again)"; the action runs on a second click
        within CONFIRM_TIMEOUT_S (5 s). Tests that clicked once and then waited
        for the state to change could never pass — the second click was never
        sent. The armed label is awaited (not slept on) so the two clicks stay
        inside the 5 s window.
        """
        self.click_button(text, exact=True)
        panel = self._get_visible_tab_panel()
        armed = panel.locator(f"button >> text='Confirm {text}? (click again)'").first
        armed.wait_for(state="visible", timeout=timeout_ms)
        armed.click()

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

        Returns True if the row matched, False on timeout. The Manage Models
        ``gr.Dataframe`` sometimes doesn't re-render in the DOM even though
        the server confirms the new state (the toggle handler and the status
        timer both deliver fresh data). Step 4A hardened callers to assert
        on this return; the hard assertion failed identically in two
        consecutive live runs with server-confirmed loads, and a live DOM
        probe showed the table rendering a single stale row that never
        updates despite fresh ``/models`` data (evidence:
        docs/testing/step4a-e2e-unhollow-2026-09-12.tdd.md). By recorded
        deviation (plan Step 4A status block), callers treat a False return
        as a warning; the authoritative gate is the server-side
        ``_wait_for_model_state`` assert.
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
