#!/usr/bin/env python3
"""E2E guard: the WaveSurfer StreamingPlayer must load via the PRODUCTION path.

Gradio's ``gr.HTML`` sets ``innerHTML``, which per the HTML5 spec does not execute
``<script>`` tags — Gradio even warns about it. The app compensates in
``_facade.py`` with ``demo.load(fn=_load_initial_history, js=get_script_reexecutor_fn())``,
which finds the inert ``script[type="module"]`` in the DOM and re-injects it as a
same-origin Blob URL into ``<head>``, where it does execute.

That compensation is load-bearing: if it breaks, ``window.getOrCreatePlayer`` never
exists and every audio control in the UI (play, download, volume, speed, history
replay, streaming generation) silently does nothing.

This module deliberately does **not** re-inject scripts itself. ``test_e2e_playwright.py``
does (``_INJECT_SCRIPTS_JS``), which means it exercises the *test harness's* injector
rather than the app's — a dead production path would leave it green. Everything here is
asserted, never swallowed.

Prerequisites:
    - playwright installed: pip install playwright && playwright install chromium
    - TTS server running on port 5123

Usage:
    python -m unittest tests.test_e2e_wavesurfer_live -v
    pytest tests/test_e2e_wavesurfer_live.py -m e2e
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

# Distinct from test_e2e_playwright.py's 7866 so the two modules can run
# back to back without racing for the port.
UI_PORT = 7867
UI_URL = f"http://127.0.0.1:{UI_PORT}"
SERVER_URL = "http://127.0.0.1:5123"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Marker the StreamingPlayer module logs once it has evaluated and registered
# the global factory. Its presence proves the module body actually ran.
PLAYER_READY_MARKER = "[StreamingPlayer] Module loaded"
# Marker the production re-executor logs when it re-injects the module script.
REEXECUTOR_MARKER = "[ScriptReexecutor] Re-injected module script"

# Probes the boundaries between Gradio's innerHTML insertion and a working player.
PROBE_JS = """() => {
    const domScripts = Array.from(document.querySelectorAll('script[type="module"]'));
    return {
        module_scripts_in_dom: domScripts.length,
        largest_module_script: Math.max(
            0, ...domScripts.map(s => (s.textContent || '').length)
        ),
        blob_scripts_in_head: document.querySelectorAll('script[src^="blob:"]').length,
        get_or_create_player: typeof window.getOrCreatePlayer,
        streaming_players: typeof window._streamingPlayers,
        clone_waveform: !!document.querySelector('#clone-waveform'),
        clone_play_btn: !!document.querySelector('#clone-play-btn'),
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
class TestWaveSurferLoadsViaProductionPath(unittest.TestCase):
    """The player must come up with no help from the test harness."""

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
        self.console = []
        self.page_errors = []
        self.page.on("console", lambda m: self.console.append(m.text))
        self.page.on("pageerror", lambda e: self.page_errors.append(str(e)))
        # networkidle never settles — the UI polls on a gr.Timer forever.
        self.page.goto(UI_URL, wait_until="load", timeout=60_000)
        self.page.wait_for_selector("button", timeout=30_000)

    def tearDown(self):
        self.page.context.close()

    def _wait_for_player(self, timeout=20_000):
        """Wait for the global player factory, failing loudly if it never appears."""
        try:
            self.page.wait_for_function(
                "() => typeof window.getOrCreatePlayer === 'function'",
                timeout=timeout,
            )
        except Exception as exc:
            probe = self.page.evaluate(PROBE_JS)
            self.fail(
                "window.getOrCreatePlayer never became a function — every audio "
                "control in the UI is dead. The demo.load(js=get_script_reexecutor_fn()) "
                f"re-injection likely broke.\nBoundary probe: {probe}\n"
                f"Console: {self.console}\nPage errors: {self.page_errors}\n"
                f"Underlying: {exc}"
            )

    def test_player_factory_is_defined_without_test_side_injection(self):
        """window.getOrCreatePlayer must exist from the app's own re-injection."""
        self._wait_for_player()
        self.assertEqual(
            self.page.evaluate("() => typeof window.getOrCreatePlayer"),
            "function",
        )

    def test_streaming_player_module_body_executed(self):
        """The module must actually evaluate, not merely sit inert in the DOM."""
        self._wait_for_player()
        self.assertTrue(
            any(PLAYER_READY_MARKER in line for line in self.console),
            f"{PLAYER_READY_MARKER!r} absent from console: {self.console}",
        )

    def test_script_reexecutor_ran_and_injected_blob_module(self):
        """The compensating re-injection must run and land a blob script in <head>."""
        self._wait_for_player()
        self.assertTrue(
            any(REEXECUTOR_MARKER in line for line in self.console),
            f"{REEXECUTOR_MARKER!r} absent from console: {self.console}",
        )
        probe = self.page.evaluate(PROBE_JS)
        self.assertGreater(
            probe["blob_scripts_in_head"], 0,
            f"no blob-URL script reached <head>: {probe}",
        )

    def test_gradio_still_emits_the_module_script_into_the_dom(self):
        """If Gradio ever strips <script> instead of inerting it, re-injection dies."""
        probe = self.page.evaluate(PROBE_JS)
        self.assertGreater(
            probe["module_scripts_in_dom"], 0,
            f"Gradio emitted no module script at all: {probe}",
        )
        self.assertGreater(
            probe["largest_module_script"], 1000,
            f"module script present but suspiciously small: {probe}",
        )

    def test_player_controls_render(self):
        """The player HTML must render, or the factory has nothing to bind to."""
        probe = self.page.evaluate(PROBE_JS)
        self.assertTrue(probe["clone_waveform"], f"#clone-waveform missing: {probe}")
        self.assertTrue(probe["clone_play_btn"], f"#clone-play-btn missing: {probe}")

    def test_page_loads_without_javascript_errors(self):
        """A page error here usually means the module threw while evaluating."""
        self._wait_for_player()
        self.assertEqual(
            self.page_errors, [], f"unexpected page errors: {self.page_errors}"
        )


if __name__ == "__main__":
    unittest.main()
