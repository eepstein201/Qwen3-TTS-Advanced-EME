#!/usr/bin/env python3
"""E2E: Voice Management tab — create / rename / preview / delete via the UI.

Closes 4D gap 1: the Voice Management tab had zero E2E coverage at any level
(the plan credited Step 0G with a create-from-audio case, but no such test
exists on main — verified 2026-09-17; this module covers create too).

The tab is backed by the server endpoints /create-voice-prompt, /rename-prompt,
/preview-prompt, /delete-prompt (rename has rollback-on-failure logic worth
exercising; create/delete touch the filesystem). MLX create is inference-free
(.wav + .txt store), so the create flow is fast and needs no ASR when a
transcript is supplied.

Two gradio-6.20 realities shape the mechanics (both evidence-backed):

* The voices table is a ``gr.Dataframe`` whose value is computed when the UI
  PROCESS builds the app — a page reload re-mounts the frontend against the
  same stale backend value, and the Dataframe update path is the same frozen
  component family as the Manage Models table (issue #298). Fixtures are
  therefore created in ``setUpClass`` BEFORE ``launch_ui`` so they are in the
  true initial render, and mutations are verified through the authed
  ``/prompts`` API (server truth), never through a table re-render.
* Inactive tabpanels are not in the DOM, and the default Clone panel carries
  its own file input — the create upload MUST be scoped to the visible panel
  or it silently lands on the wrong component.

Status predicates are the PRODUCTION strings from ``voice_management.py`` /
``tabs_management.py`` (do not paraphrase — see 4E's test_03 for what a
guessed predicate costs) — recorded here for reference:
  create  -> "Created MLX voice prompt: {name}"     (torch branch differs)
  rename  -> "Renamed '{old}' to '{new}'"
  delete  -> arm: "Click again within 5s to confirm deletion." /
              button relabel "Confirm Delete? (click again)";
              confirm: "Deleted '{name}'"
NOTE: gradio 6.20 does not render Textbox VALUE updates into the DOM (the
status boxes stay blank), so the tests assert OUTCOMES instead — grid cell
text, ``audio[src]`` attached, the armed button's relabel, and the authed
/prompts API.

Prerequisites: live server on :5123, playwright + chromium.

Usage:  pytest tests/test_e2e_voice_management.py -m e2e
"""

import json
import math
import os
import struct
import tempfile
import time
import unittest
import urllib.request

from tests.e2e_ui import (
    GradioPage,
    kill_stale_ui_on_port,
    launch_ui,
    stop_ui,
    wait_for_ui,
)

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

# Distinct from 7866 (playwright), 7867 (wavesurfer), 7868 (tab nav),
# 7870 (history) so modules can run back to back without port races.
UI_PORT = 7869
UI_URL = f"http://127.0.0.1:{UI_PORT}"
SERVER_URL = "http://127.0.0.1:5123"
TOKEN_FILE = os.path.expanduser("~/.config/qwen3-tts/.voice_server_token")


def _make_wav(path, seconds=1.0, rate=24_000):
    """Write a minimal >=24 kHz mono 16-bit sine wav (stdlib only).

    24 kHz matters: the engine writer's ``ensure_min_sample_rate`` raises on
    anything below it, which would fail the create paths under test.
    """
    import wave

    n = int(seconds * rate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            sample = int(0.4 * 32_767 * math.sin(2 * math.pi * 440 * i / rate))
            frames += struct.pack("<h", sample)
        w.writeframes(bytes(frames))


def _create_fixture_via_engine(name):
    """Create a disposable prompt through the engine writer (the same
    on-disk MLX .wav+.txt store the server's create path uses)."""
    from qwen3_tts.core.engine.voice_prompt import save_voice_prompt_mlx

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "ref.wav")
        _make_wav(wav)
        save_voice_prompt_mlx(name, wav, "e2e fixture transcript")


def _authed_get(endpoint):
    """GET an authed JSON endpoint (raises on non-200)."""
    with open(TOKEN_FILE) as f:
        token = f.read().strip()
    req = urllib.request.Request(
        f"{SERVER_URL}{endpoint}", headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _api_prompt_names():
    """BARE prompt names per the server (auth truth).

    ``/prompts`` LISTS suffixed names ("x.wav") while create/delete address
    prompts by bare name — normalize to bare so membership checks match the
    names this module creates.
    """
    listed = _authed_get("/prompts")["prompts"]
    return {n.removesuffix(".wav").removesuffix(".pt") for n in listed}


def _api_delete_prompt(name):
    """Best-effort cleanup via the server."""
    try:
        with open(TOKEN_FILE) as f:
            token = f.read().strip()
        req = urllib.request.Request(
            f"{SERVER_URL}/delete-prompt",
            data=json.dumps({"name": name}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # cleanup only; the test's own assertions are the gate


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestE2EVoiceManagement(unittest.TestCase):
    """Create/rename/preview/delete a disposable voice prompt through the UI."""

    @classmethod
    def setUpClass(cls):
        try:
            urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3)
        except Exception:
            raise unittest.SkipTest("TTS server not running on port 5123")

        cls.suffix = str(int(time.time() * 1000) % 100_000_000)
        cls.fixtures = {
            "rename": f"e2e_vm_ren_{cls.suffix}",
            "preview": f"e2e_vm_prev_{cls.suffix}",
            "delete": f"e2e_vm_del_{cls.suffix}",
        }
        # Fixtures BEFORE launch: the table value is computed at UI build
        # time, and its update path is the frozen Dataframe (#298) — the
        # initial render is the only render these rows get.
        for name in cls.fixtures.values():
            _create_fixture_via_engine(name)
        # Fail fast if the engine writer and the server disagree on the
        # prompts dir (the manage tests would all time out otherwise).
        listed = _api_prompt_names()
        missing = [n for n in cls.fixtures.values() if n not in listed]
        if missing:
            for n in cls.fixtures.values():
                _api_delete_prompt(n)
            raise unittest.SkipTest(
                f"engine-written fixtures not visible via /prompts: {missing}"
            )

        kill_stale_ui_on_port(UI_PORT)
        cls.ui_proc = launch_ui(UI_PORT)

        if not wait_for_ui(UI_URL):
            cls._kill_ui()
            for n in cls.fixtures.values():
                _api_delete_prompt(n)
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
        for n in cls.fixtures.values():
            _api_delete_prompt(n)

    @classmethod
    def _kill_ui(cls):
        stop_ui(cls.ui_proc)
        cls.ui_proc = None

    def setUp(self):
        self.page = type(self).browser.new_context().new_page()
        self.gp = GradioPage(self.page, UI_URL)
        self.gp.navigate()
        self._created = []  # names created by THIS test (cleanup in tearDown)

    def tearDown(self):
        for name in self._created:
            _api_delete_prompt(name)
        self.page.context.close()

    # --- helpers ---------------------------------------------------------

    def _goto_manage_voices(self):
        self.gp.click_tab("Manage Voices")
        self._wait_manage_buttons(interactive=False)

    def _wait_manage_buttons(self, interactive):
        """Wait for Rename/Delete to reach the wanted disabled state."""
        want = "false" if interactive else "true"
        self.page.wait_for_function(
            """(want) => {
                const btns = Array.from(document.querySelectorAll('button'));
                const r = btns.find(b => b.textContent.trim() === 'Rename');
                const d = btns.find(b => b.textContent.trim() === 'Delete');
                return r && d
                    && String(r.disabled) === want
                    && String(d.disabled) === want;
            }""",
            arg=want,
            timeout=10_000,
        )
        rename = self.page.locator("button").filter(has_text="Rename").first
        delete = self.page.locator("button").filter(has_text="Delete").first
        return rename, delete

    def _select_voice_row(self, name):
        """One trusted click on the fixture's cell in the REAL grid.

        Gradio 6 Dataframe renders the visible grid as div-based
        ``[data-row="i"][data-col="j"]`` cells (the ``<table>`` in the DOM
        is a one-row phantom for a11y/copy — never click through it; the
        history module learned this same lesson as test_13).
        """
        idx = self.page.wait_for_function(
            """(t) => {
                const panels = document.querySelectorAll('div[role="tabpanel"]');
                for (const p of panels) {
                    if (p.offsetParent === null) continue;
                    const cells = p.querySelectorAll('[data-row][data-col="0"]');
                    for (const c of cells) {
                        if (c.textContent.includes(t)) {
                            return c.getAttribute('data-row');
                        }
                    }
                }
                return false;
            }""",
            arg=name,
            timeout=15_000,
        )
        cell = (
            self.page.locator("div[role='tabpanel']:visible")
            .first.locator(f'[data-row="{idx}"][data-col="0"]')
        )
        cell.click()

    # --- tests -----------------------------------------------------------

    def test_01_create_from_audio_via_ui(self):
        """Create Voice tab: upload audio + transcript + name -> created."""
        self.gp.click_tab("Create Voice")
        name = f"e2e_vm_create_{self.suffix}_{int(time.time()) % 10_000}"
        self._created.append(name)

        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "ref.wav")
            _make_wav(wav)
            # Scope to the VISIBLE panel: the default Clone tab has its own
            # file input earlier in the DOM, and a page-wide `.first` would
            # silently upload there instead.
            panel = self.page.locator("div[role='tabpanel']:visible").first
            panel.locator("input[type='file']").first.set_input_files(wav)
            # The upload is asynchronous (gradio POSTs the file after the
            # input resolves); clicking Create before it lands passes None
            # to the handler. Completion signal: an <audio> with a src in
            # the panel — gradio keeps its players CSS-hidden even with
            # content, so attached-with-src is the contract, not visible.
            self.page.wait_for_selector(
                "div[role='tabpanel']:visible audio[src]",
                state="attached",
                timeout=30_000,
            )

        self.gp.fill_textbox("Transcript", "e2e fixture transcript")
        self.gp.fill_textbox("Voice Name", name)
        self.gp.click_button("Create Voice Prompt")

        # Outcome asserted via server truth: gradio 6.20 does not render
        # Textbox VALUE updates (the status boxes stay blank in the DOM),
        # and create does not refresh the table (its outputs are the status
        # box + dropdown only) — the API is the real contract.
        self._wait_api_names_contain(name, timeout=30_000)

    def _wait_api_names_contain(self, name, timeout=30_000):
        """Poll the authed /prompts API until name appears (hard-fails)."""
        deadline = time.monotonic() + timeout / 1000
        last = set()
        while time.monotonic() < deadline:
            last = _api_prompt_names()
            if name in last:
                return
            time.sleep(1.0)
        self.fail(f"{name!r} not in /prompts after {timeout}ms; got {sorted(last)[:8]}")

    def _wait_api_names_lack(self, name, timeout=15_000):
        """Poll the authed /prompts API until name disappears (hard-fails)."""
        deadline = time.monotonic() + timeout / 1000
        while time.monotonic() < deadline:
            if name not in _api_prompt_names():
                return
            time.sleep(1.0)
        self.fail(f"{name!r} still in /prompts after {timeout}ms")

    def _wait_grid_cell_contains(self, name, timeout=15_000):
        """Wait until a visible panel's first-column grid cell shows name."""
        self.page.wait_for_function(
            """(t) => {
                const panels = document.querySelectorAll('div[role="tabpanel"]');
                for (const p of panels) {
                    if (p.offsetParent === null) continue;
                    const cells = p.querySelectorAll('[data-row][data-col="0"]');
                    for (const c of cells) {
                        if (c.textContent.includes(t)) return true;
                    }
                }
                return false;
            }""",
            arg=name,
            timeout=timeout,
        )

    def test_02_rename_selected_voice(self):
        """Manage Voices: select a row, rename -> status + API updated."""
        name = self.fixtures["rename"]
        new_name = f"{name}_r"
        self._created.append(new_name)

        self._goto_manage_voices()
        self._select_voice_row(name)
        rename_btn, _ = self._wait_manage_buttons(interactive=True)
        self.gp.fill_textbox("New Name (for rename)", new_name)
        rename_btn.click()

        # The rename handler's outputs include the refreshed table, and the
        # Dataframe update DOES render here — assert the grid directly plus
        # server truth (the status textbox never renders its value in this
        # gradio build).
        self._wait_grid_cell_contains(new_name)
        self._wait_api_names_contain(new_name)
        self.assertNotIn(name, _api_prompt_names())

    def test_03_preview_selected_voice(self):
        """Manage Voices: select a row, Preview -> an audio element renders."""
        name = self.fixtures["preview"]

        self._goto_manage_voices()
        self._select_voice_row(name)
        self._wait_manage_buttons(interactive=True)

        self.gp.click_button("Preview")
        # Success contract: the Preview gr.Audio receives the temp .wav —
        # an <audio> with a src appears (attached; gradio keeps players
        # CSS-hidden even with content). Bounded, hard-failing.
        self.page.wait_for_selector("audio[src]", state="attached", timeout=30_000)

    def test_04_delete_two_step_confirm(self):
        """Manage Voices: delete is a two-step confirm inside 5 s."""
        name = self.fixtures["delete"]

        self._goto_manage_voices()
        self._select_voice_row(name)
        self._wait_manage_buttons(interactive=True)

        # Arm+confirm pair, retried as a unit. Under full-suite load,
        # gradio's event queue can process the confirm click past the 5 s
        # server-side arm window (~1-in-3 observed; isolation is always
        # green — queue latency, not a handler defect). Retrying the
        # USER-VISIBLE pair is the honest remedy; the outcome assertion
        # stays hard and nothing is widened.
        deleted = False
        for _attempt in range(3):
            self.gp.click_button("Delete")
            # Arm signal: the button RELABELS (the status textbox never
            # renders its value in this gradio build). The 5 s arm window
            # starts at HANDLER time — under load the relabel's render can
            # eat most of it — so fire the confirm the instant the label
            # appears via an element-evaluate click (no actionability wait;
            # a single programmatic click on a button is one event, unlike
            # the double-firing Dataframe select case).
            confirm = (
                self.page.locator("button")
                .filter(has_text="Confirm Delete? (click again)")
                .first
            )
            confirm.wait_for(state="attached", timeout=5_000)
            confirm.evaluate("el => el.click()")
            try:
                self._wait_api_names_lack(name, timeout=6_000)
                deleted = True
                break
            except AssertionError:
                continue
        self.assertTrue(deleted, f"{name!r} not deleted after 3 arm+confirm attempts")

        # No grid-absence assertion: a REMOVED row's stale render is the
        # #298 frozen-Dataframe shape — gradio's display update, not the
        # delete contract. The UI flow is proven by the arm relabel +
        # in-window confirm clicks above.


if __name__ == "__main__":
    unittest.main()
