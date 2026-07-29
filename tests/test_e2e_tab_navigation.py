#!/usr/bin/env python3
"""E2E guard: clicking through the UI's tabs must never kill the page.

PR #95 found that a ``gr.Tab.select`` listener attached near Manage Models made
Gradio's Dataframe frontend recurse infinitely on gradio 6.14.x
(``RangeError: Maximum call stack size exceeded`` at
``Object.get [as groupedColumnMode]``), which killed the page — the *next* tab
opened after the crash rendered blank. The listener was removed and replaced
with a 5-second ``gr.Timer`` for status polling (see ``_facade.py``).

``tests/test_ui_tab_select_wiring.py`` guards the cause statically: it asserts
no ``select`` listener is attached to any ``gr.Tab`` in the built UI. It does
NOT guard the symptom — nothing else in the suite opens a real browser, clicks
through the tabs, and checks the page survived. This module fills that gap by
porting a throwaway repro harness (written during the original debugging
session) into a permanent regression test.

Three non-obvious things this module preserves from that harness, each because
a naiver version produced a wrong diagnosis in the past:

1. Page errors are collected PER CLICK, not once at the end of a sweep. The
   original misdiagnosis of the crash came from a harness that collected
   ``pageerror`` events only after a full 6-tab sweep, making it impossible to
   tell which click actually caused it.
2. Native Playwright ``.click()`` on a Gradio 6 tab button HANGS. Clicks are
   dispatched via ``page.evaluate()`` JS instead.
3. Tab labels appear on multiple DOM elements, not just the actual tab button.
   Only elements with ``[role="tab"]`` actually switch panels when clicked;
   filtering by visible text alone is not enough.

Caveat on this repo's dev gradio pin (6.20.0): the underlying recursion bug is
fixed upstream there, so re-attaching a ``select`` listener to a ``gr.Tab`` on
this gradio version may well leave every test in this module green. A pass
here is the correct, expected baseline on 6.20.0 — it is NOT evidence that
attaching ``select`` to a ``gr.Tab`` is safe. The ban on doing so
(``tests/test_ui_tab_select_wiring.py``) stays as defence-in-depth for
gradio 6.14.x and any future downgrade.

Prerequisites:
    - playwright installed: pip install playwright && playwright install chromium
    - TTS server running on port 5123

Usage:
    python -m unittest tests.test_e2e_tab_navigation -v
    pytest tests/test_e2e_tab_navigation.py -m e2e
"""

import os
import signal
import subprocess  # nosec B404
import sys
import time
import unittest
import urllib.error
import urllib.request

# E2E browser tests require a live server + Gradio UI + Chromium.
# Gated behind the `e2e` marker so plain `pytest tests/` skips them (no hang).
try:
    import pytest
    pytestmark = pytest.mark.e2e
except ImportError:
    pass

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

# Distinct from test_e2e_playwright.py's 7866 and test_e2e_wavesurfer_live.py's
# 7867 so all three E2E modules can run back to back without racing for a port.
UI_PORT = 7868
UI_URL = f"http://127.0.0.1:{UI_PORT}"
SERVER_URL = "http://127.0.0.1:5123"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Full sweep; "Design Mode" -> "Manage Voices" was the reported minimal repro.
SEQUENCE = [
    "Design Mode", "Manage Voices", "Manage Models", "Custom Mode",
    "Create Voice", "Clone Mode", "Manage Models", "Design Mode",
    "Manage Voices",
]

# The two tabs whose blank-render-after-crash was the actual user-visible
# symptom (both hold a gr.Dataframe).
DATAFRAME_TABS = ("Manage Voices", "Manage Models")

# How long a tab switch (plus any Timer-driven refresh) needs to settle before
# we trust the DOM/error state we just observed.
TAB_SWITCH_SETTLE_SECONDS = 3

# A crashed/blank tab renders next to nothing; a real one renders a full page
# of controls, tables, and copy. 500 chars comfortably separates the two.
MIN_RENDERED_BODY_CHARS = 500

CLICK_TAB_JS = """(label) => {
    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
    const b = tabs.find(x => (x.textContent||'').trim() === label);
    if (!b) return false;
    b.click();
    return true;
}"""

TAB_STATE_JS = """(label) => {
    const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
    const b = tabs.find(x => (x.textContent||'').trim() === label);
    return {
        selected: b ? b.getAttribute('aria-selected') : null,
        bodyLen: (document.body.innerText||'').length,
    };
}"""


def _is_server_running():
    """Return True if the TTS server answers /health."""
    try:
        resp = urllib.request.urlopen(  # nosec B310
            f"{SERVER_URL}/health", timeout=5
        )
        return resp.status == 200
    except Exception:
        return False


def _wait_for_ui(url, timeout=45):
    """Poll the Gradio UI until it serves a page or the timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(url, timeout=3).status == 200:  # nosec B310
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    return False


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestTabNavigationNeverKillsThePage(unittest.TestCase):
    """Clicking through every tab must never throw a page error or blank a tab."""

    ui_proc = None
    playwright_instance = None
    browser = None

    @classmethod
    def setUpClass(cls):
        if not _is_server_running():
            raise unittest.SkipTest("TTS server not running on port 5123")

        cls._kill_port(UI_PORT)
        cls.ui_proc = subprocess.Popen(  # nosec B603
            [
                sys.executable, "-c",
                f"import sys; sys.path.insert(0, {PROJECT_DIR!r}); "
                f"from qwen3_tts.interface.ui import build_ui; "
                f"demo = build_ui(); "
                f"demo.launch(server_name='127.0.0.1', server_port={UI_PORT}, "
                f"share=False, show_error=True)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if not _wait_for_ui(UI_URL):
            cls._kill_ui()
            raise unittest.SkipTest(f"Gradio UI failed to start on port {UI_PORT}")
        if cls.ui_proc.poll() is not None:
            raise unittest.SkipTest(
                f"Gradio UI subprocess exited immediately (port {UI_PORT} conflict?)"
            )

        cls.playwright_instance = sync_playwright().start()
        cls.browser = cls.playwright_instance.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        if cls.browser:
            cls.browser.close()
        if cls.playwright_instance:
            cls.playwright_instance.stop()
        cls._kill_ui()

    @classmethod
    def _kill_port(cls, port):
        """SIGTERM anything listening on `port` so we never test stale code."""
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
            time.sleep(1)
        except Exception:
            pass

    @classmethod
    def _kill_ui(cls):
        if cls.ui_proc is not None:
            cls.ui_proc.terminate()
            try:
                cls.ui_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.ui_proc.kill()
            cls.ui_proc = None

    def setUp(self):
        self.page = self.browser.new_context().new_page()
        self.page_errors = []
        self.page.on("pageerror", lambda e: self.page_errors.append(str(e)))
        # networkidle never settles — the UI polls on a gr.Timer forever.
        self.page.goto(UI_URL, wait_until="load", timeout=60_000)
        self.page.wait_for_selector("button", timeout=30_000)
        # Let the initial render (and any first Timer tick) settle before the
        # baseline error count is trusted.
        time.sleep(TAB_SWITCH_SETTLE_SECONDS)

    def tearDown(self):
        self.page.context.close()

    def _click_tab(self, label):
        """Dispatch a click on the [role="tab"] element matching `label`.

        Native Playwright .click() on a Gradio 6 tab button hangs, so the
        click is dispatched from page context instead. Tab labels appear on
        multiple DOM elements; only [role="tab"] elements actually switch
        panels, so filtering by that role (not just visible text) is required.
        """
        clicked = self.page.evaluate(CLICK_TAB_JS, label)
        time.sleep(TAB_SWITCH_SETTLE_SECONDS)
        return clicked

    def _tab_state(self, label):
        return self.page.evaluate(TAB_STATE_JS, label)

    def test_baseline_load_produces_no_page_errors(self):
        """No pageerror should fire merely from loading the UI."""
        self.assertEqual(
            self.page_errors, [],
            f"page errors fired during initial load, before any tab was "
            f"clicked: {self.page_errors}",
        )

    def test_full_tab_sweep_produces_no_page_errors(self):
        """Every click in the sweep is checked individually.

        Aggregating errors once at the end of the sweep is exactly what
        produced the original misdiagnosis of this bug — a crash triggered by
        an early click could only be blamed on whichever click happened to run
        right before the aggregate check.
        """
        for label in SEQUENCE:
            before = len(self.page_errors)
            clicked = self._click_tab(label)
            new_errors = self.page_errors[before:]
            with self.subTest(tab=label):
                self.assertTrue(
                    clicked, f"tab button {label!r} not found via [role='tab']"
                )
                self.assertEqual(
                    new_errors, [],
                    f"clicking {label!r} produced page errors: {new_errors}",
                )

    def test_dataframe_tabs_still_render_after_the_sweep(self):
        """The actual symptom of the crash: a later tab renders blank.

        The original crash didn't always surface as a Playwright-visible JS
        error on the click that caused it — the visible damage was the next
        Dataframe-bearing tab rendering nothing. This asserts substantial
        content is still present after running the full sweep, independent of
        whether any pageerror fired.
        """
        for label in SEQUENCE:
            self._click_tab(label)

        for label in DATAFRAME_TABS:
            self._click_tab(label)
            state = self._tab_state(label)
            with self.subTest(tab=label):
                self.assertGreater(
                    state["bodyLen"], MIN_RENDERED_BODY_CHARS,
                    f"{label!r} rendered only {state['bodyLen']} chars after "
                    "the sweep — looks like the dead-page symptom even though "
                    "no page error fired",
                )


if __name__ == "__main__":
    unittest.main()
