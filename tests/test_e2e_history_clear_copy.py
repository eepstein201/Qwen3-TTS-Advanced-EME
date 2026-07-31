#!/usr/bin/env python3
"""Playwright E2E tests for PR #90 history features: copy transcript, remove
row, clear all — with VISIBLE status feedback and waveform reset.

Standalone: launches ``build_ui()`` in a subprocess (no TTS server / models
required — the history panel interactions are pure Python + JS). History is
seeded from the existing ``voice_ui_*.json`` sidecars in ~/Downloads via
``demo.load`` → ``load_history_from_disk``.

Prerequisites:
    pip install playwright && playwright install chromium

Usage:
    python -m pytest tests/test_e2e_history_clear_copy.py -v
    python -m unittest tests.test_e2e_history_clear_copy -v
"""

import os
import signal
import subprocess  # nosec B404
import sys
import time
import unittest
import urllib.error
import urllib.request

from tests.e2e_helpers import assert_supported_gradio

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

UI_PORT = 7870
UI_URL = f"http://127.0.0.1:{UI_PORT}"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# JS stub injected before every navigation: capture clipboard writes on
# window.__clipboard so we can assert copy-to-clipboard without depending on
# OS clipboard permissions (which are flaky under automation).
_CLIPBOARD_STUB = """
Object.defineProperty(navigator, 'clipboard', {
  value: {
    writeText: (t) => { window.__clipboard = window.__clipboard || []; window.__clipboard.push(String(t)); return Promise.resolve(); },
    readText: () => Promise.resolve('')
  },
  configurable: true
});
"""

# Spy on the WaveSurfer player's reset() so we can assert the waveform cleared
# on delete / clear-all without inspecting canvas pixels.
_SPY_RESET = """
() => {
  window.__resetCalls = 0;
  if (typeof window.getOrCreatePlayer !== 'function') return false;
  const real = window.getOrCreatePlayer;
  window.getOrCreatePlayer = function(id){
    const p = real.call(this, id);
    if (p && typeof p.reset === 'function' && !p.__spied) {
      const orig = p.reset.bind(p);
      p.reset = () => { window.__resetCalls = (window.__resetCalls||0)+1; return orig(); };
      p.__spied = true;
    }
    return p;
  };
  return true;
}
"""

# Gradio's gr.HTML() sets innerHTML, which does not execute <script> tags.
# Re-inject module scripts so the WaveSurfer StreamingPlayer loads.
_INJECT_SCRIPTS = """() => {
  document.querySelectorAll('script[type="module"]').forEach(function(s) {
    if (s.textContent && s.textContent.length > 100 && !s.src) {
      var url = URL.createObjectURL(new Blob([s.textContent], {type:'application/javascript'}));
      var ns = document.createElement('script'); ns.type = 'module'; ns.src = url;
      document.head.appendChild(ns);
    }
  });
}"""


def _wait_for_ui(url, timeout=45):
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


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestE2EHistoryClearCopy(unittest.TestCase):
    """Browser tests for Recent Generations: copy, remove, clear, visible status."""

    ui_proc = None
    playwright_instance = None
    browser = None

    @classmethod
    def setUpClass(cls):
        # The UI subprocess below inherits sys.executable, so guard the gradio
        # version BEFORE anything is measured against it.
        assert_supported_gradio()

        # Kill any stale UI on our port
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", UI_PORT)) == 0:
                try:
                    r = subprocess.run(  # nosec B603
                        ["lsof", "-ti", f":{UI_PORT}"],
                        capture_output=True, text=True, timeout=5,
                    )
                    for pid_str in (r.stdout or "").strip().splitlines():
                        try:
                            os.kill(int(pid_str), signal.SIGTERM)
                        except (ProcessLookupError, ValueError):
                            pass
                    time.sleep(1)
                except Exception:
                    pass

        env = os.environ.copy()
        cls.ui_proc = subprocess.Popen(  # nosec B603
            [
                sys.executable, "-c",
                f"import sys; sys.path.insert(0, '{PROJECT_DIR}'); "
                f"from qwen3_tts.interface.ui import build_ui; "
                f"demo = build_ui(); "
                f"demo.launch(server_name='127.0.0.1', server_port={UI_PORT}, "
                f"share=False, show_error=True, "
                f"allowed_paths=[__import__('os').path.expanduser('~/Downloads'), '/tmp'], "
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

    @classmethod
    def tearDownClass(cls):
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
        self.context = self.browser.new_context()
        # Stub clipboard before any page script runs.
        self.context.add_init_script(_CLIPBOARD_STUB)
        self.page = self.context.new_page()
        self.page.goto(UI_URL, wait_until="domcontentloaded", timeout=60_000)
        self.page.wait_for_selector("button[role='tab']", timeout=30_000)
        self.page.evaluate(_INJECT_SCRIPTS)
        # Wait for history to seed from disk + player module to load.
        self._wait_for_history_rows(min_rows=1, timeout=30_000)
        try:
            self.page.wait_for_function(
                "() => typeof window.getOrCreatePlayer === 'function'", timeout=10_000
            )
        except Exception:
            pass
        self.page.evaluate(_SPY_RESET)

    def tearDown(self):
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()

    # --- helpers --------------------------------------------------------

    def _wait_for_history_rows(self, min_rows=1, timeout=30_000):
        """Wait until the history table has >= min_rows rendered rows.

        Counts ``[data-col="0"][data-row]`` cells — exactly one per row, present
        only in the real (selectable) table, never in Gradio 6's phantom table.
        """
        self.page.wait_for_function(
            f"""() => document.querySelectorAll('[data-col="0"][data-row]').length >= {min_rows}""",
            timeout=timeout,
        )

    def _history_row_count(self):
        return self.page.evaluate(
            """() => document.querySelectorAll('[data-col="0"][data-row]').length"""
        )

    def _click_history_cell(self, row, col):
        """Force-click a history cell, matching the existing harness (test_13).

        Gradio 6 wraps cell content in a ``disable_click`` upload-button overlay
        and uses ``onmousedown`` on the ``[data-row][data-col]`` cell. A raw
        ``mouse.click`` at coordinates is swallowed by the overlay; Playwright's
        element ``.click(force=True)`` dispatches against the cell node itself
        and reliably fires the ``select`` event. ``[data-row][data-col]`` exists
        in exactly one table (the real one; Gradio's phantom table has none).
        """
        cell = self.page.locator(f'[data-row="{row}"][data-col="{col}"]').first
        cell.scroll_into_view_if_needed(timeout=10_000)
        cell.click(force=True)
        self.page.wait_for_timeout(1500)

    def _visible_status_text(self):
        """Return concatenated text of all VISIBLE role=status regions.

        A region is visible if it is NOT the sr-only announcer (no clip:rect),
        has a non-null offsetParent, and a width > 10px. This excludes the
        invisible _announce_status divs (1px clipped).
        """
        return self.page.evaluate(
            """() => {
                var out = [];
                document.querySelectorAll('[role="status"]').forEach(el => {
                    var st = el.getAttribute('style') || '';
                    if (st.indexOf('clip:rect') >= 0) return;       // sr-only announcer
                    if (el.offsetParent === null) return;            // display:none / hidden
                    if (el.getBoundingClientRect().width < 10) return;
                    var t = el.textContent.trim();
                    if (t) out.push(t);
                });
                return out.join(' || ');
            }"""
        )

    def _wait_for_visible_status(self, substring, timeout=10_000):
        self.page.wait_for_function(
            f"""() => {{
                var found = false;
                document.querySelectorAll('[role="status"]').forEach(el => {{
                    var st = el.getAttribute('style') || '';
                    if (st.indexOf('clip:rect') >= 0) return;
                    if (el.offsetParent === null) return;
                    if (el.getBoundingClientRect().width < 10) return;
                    if (el.textContent.indexOf({substring!r}) >= 0) found = true;
                }});
                return found;
            }}""",
            timeout=timeout,
        )

    def _clipboard_values(self):
        return self.page.evaluate("() => window.__clipboard || []")

    def _reset_calls(self):
        return self.page.evaluate("() => window.__resetCalls || 0")

    # --- tests ----------------------------------------------------------

    def test_01_copy_transcript_to_clipboard_with_visible_status(self):
        """Clicking the Text Preview cell copies the full transcript and shows
        a VISIBLE 'Copied' status (not an invisible sr-only flash)."""
        before = self._history_row_count()
        self._click_history_cell(row=0, col=2)  # Text Preview

        # Clipboard must contain the transcript (non-empty). The full-vs-
        # truncated distinction is asserted in test_ui_facade.py
        # (test_text_preview_column_copies_full_transcript) — here we only
        # confirm the JS side-effect actually wrote to the clipboard.
        clips = self._clipboard_values()
        self.assertGreater(len(clips), 0, "clipboard.writeText was not called")
        self.assertTrue(clips[-1], f"copied text is empty: {clips[-1]!r}")

        # A VISIBLE 'Copied' status must appear (sr-only _announce_status does not count).
        self._wait_for_visible_status("Copied", timeout=10_000)

        # Copy is copy-only: no audio replay → row count unchanged, no waveform reset.
        self.assertEqual(self._history_row_count(), before)
        self.assertEqual(self._reset_calls(), 0)

    def test_02_remove_row_shows_status_and_clears_waveform(self):
        """Removing a row is a path-keyed TWO-STEP confirm.

        The first ✕ click only arms it (cell relabels to "Confirm?", warning
        status, row still present); a second click on the SAME row within
        DELETE_CONFIRM_TIMEOUT_S hard-deletes the file and drops the row.

        This test previously clicked once and asserted the row was gone, which
        was correct before the output-folder feature introduced the confirm
        step — test_03 below was updated for Clear All's two-step, this one was
        missed. Status assertions are read non-blockingly between the clicks so
        the 5s confirm window is not spent waiting on a poll.
        """
        before = self._history_row_count()
        self.assertGreaterEqual(before, 1)

        # --- step 1: arm ---
        self._click_history_cell(row=0, col=5)  # Remove (✕)

        self.assertIn(
            "within 5s", self._visible_status_text(),
            "first ✕ click should show the arming warning, not delete",
        )
        self.assertEqual(
            self._history_row_count(), before,
            "arming must not remove the row",
        )
        self.assertEqual(
            self._remove_cell_label(row=0), "Confirm?",
            "armed row's Remove cell should relabel to Confirm?",
        )

        # --- step 2: confirm (inside the 5s window) ---
        self._click_history_cell(row=0, col=5)

        self.assertEqual(self._history_row_count(), before - 1, "row was not removed")
        self._wait_for_visible_status("Deleted", timeout=10_000)
        self.assertGreater(self._reset_calls(), 0, "waveform was not reset on delete")

    def _remove_cell_label(self, row=0):
        """Text of a row's Remove cell — the glyph, or 'Confirm?' when armed."""
        return self.page.evaluate(
            """(rowIdx) => {
                var c = document.querySelector(
                    '[data-row="' + rowIdx + '"][data-col="5"]'
                );
                return c ? c.textContent.trim() : null;
            }""",
            str(row),
        )

    def test_03_clear_all_two_step_with_visible_status(self):
        """Clear All is a two-step confirm with VISIBLE hints, then empties the
        list and resets the waveform."""
        self.assertGreaterEqual(self._history_row_count(), 1)

        # First click arms the confirm; button relabels + visible hint.
        self.page.locator("button").filter(has_text="Clear All").first.click()
        self._wait_for_visible_status("Click again", timeout=10_000)

        # Second click confirms.
        self.page.locator("button").filter(has_text="Clear All").first.click()
        # The arm hint may have relabeled the button; fall back to matching by
        # the confirm text if the label changed.
        self._wait_for_visible_status("cleared", timeout=10_000)

        self.assertEqual(self._history_row_count(), 0, "list was not cleared")
        self.assertGreater(self._reset_calls(), 0, "waveform was not reset on clear")


if __name__ == "__main__":
    unittest.main()
