---
name: ui-browser-e2e
description: Drive the Gradio web UI end-to-end through the chrome-devtools MCP — launch and aim the browser, switch lazy-loaded tabs, land two-step confirms inside the 5-second window, and tell by-design symptoms (WaveSurfer warning, Disconnected badge) from real bugs. Use when manually verifying UI behavior, reproducing a UI bug, or checking a UI change the pytest E2E suite does not cover.
---

# Gradio UI E2E via chrome-devtools

An operator runbook for pointing the live `chrome-devtools` MCP tools
(`mcp__plugin_ecc_chrome-devtools__*`) at this repo's Gradio UI. It exists
because every trap below has bitten a real session; the pytest E2E suite
(`tests/e2e_*`, opt-in) does not cover ad-hoc verification flows.

**Values below are quoted with their source file. If a quoted value
disagrees with the file, the file wins — update this skill.** Nothing here
is validated automatically, so a stale value is silent.

## Step 1 — Preconditions

```bash
tts server stop && tts server start   # restart rule: always stop first, or old code serves
tts server status                     # ready + models
```

Launch the UI headless (`tts ui --no-browser`; the flag is defined in
`qwen3_tts/cli_config.py` and consumed via `TTS_UI_NO_BROWSER` in
`qwen3_tts/interface/generate_server.py` — `_facade.py` carries a parallel
argparse spelling for the direct-module path). The UI port comes from
`config["ui"]["port"]`, default **7860**, and auto-bumps 7860→7869 while
busy — read the launcher's "Port … in use, using …" line and aim the
browser at the printed port, not the default.

The restart rule covers the API server only. Verifying a *UI-code* change
also requires relaunching `tts ui` — the Gradio process caches the UI
module.

## Step 2 — Open, snapshot, and the lazy-panel rule

`new_page` → the UI URL, then `take_snapshot` (the a11y tree with element
`uid`s — prefer it over screenshots). Then the rule that shapes everything
else:

**Inactive tab panels are not in the DOM.** Gradio builds tab panels
lazily, so elements belonging to a tab you have never opened are absent
from the snapshot — you cannot click what is not rendered. The repo's own
Playwright helper (`tests/test_e2e_playwright.py::click_tab`) uses a
native click on `button[role="tab"]` and waits for the visible panel —
try that first. When a native click stalls (recorded hang, 2026-07),
fall back to dispatching the click from JS (`evaluate_script`; verify
the selector against the current DOM — Gradio versions drift):

```js
() => {
  const tab = [...document.querySelectorAll('button[role="tab"]')]
    .find(b => b.textContent.trim() === "<TAB LABEL>");
  if (!tab) return "tab not found";
  tab.click();
  return tab.textContent;
}
```

Then `take_snapshot` again — the newly mounted panel is only now in the
DOM.

## Step 3 — Two-step confirms (the 5-second window)

Destructive actions use two-step confirms with a `timeout_s`/`DELETE_CONFIRM_TIMEOUT_S`
of **5.0 s**: `ConfirmButton` (`qwen3_tts/interface/ui/components.py`) for
voice delete, model unload, and cancel; per-row Remove and Clear All use
their own arm/confirm in `qwen3_tts/interface/ui/history_panel.py`. The
**first click arms**, the **second click within 5 s executes**, and after
execution the state resets to disarmed. Two consequences:

- A click landing AFTER the window does not cancel anything: the handler
  takes the arm branch — the button **re-arms with a fresh 5 s window and
  re-shows the confirm prompt**. The dangerous next move is the natural
  one: "it did nothing, click again" — that next click EXECUTES. After
  any late or timed-out click, stop and re-snapshot before clicking
  again.
- **Definitive check:** invoke the backing handler directly (its Python
  function, with the same arguments the UI would pass) and assert on its
  return. The click-through proves wiring; the direct call proves
  behavior. If they disagree, the wiring is the finding.

## Step 4 — Dataframes: one trusted click

`select` events on a `gr.Dataframe` fire on `mousedown`. Synthetic
`mousedown`+`click` pairs double-fire handlers (row actions running
twice). Drive row selection with a **single trusted click** from the
snapshot uid, never a synthesized event pair.

## Step 5 — Reading symptoms: by design ≠ bug

| Symptom | Verdict |
|---|---|
| Console warning about a WaveSurfer `<script>` tag | **By design** — the audio player loads it deliberately; do not "fix" it |
| `/health` is 429ing while the badge shows "Connected" with N/A memory/models, or an orange "Error: …" | Rate limiting, not downtime — `is_server_running()` has counted 429 as up since #171 (`runtime.py`), so a 429 no longer produces the red "Disconnected" badge; the rate-limit config is the bug |
| `/health` reports `"status": "ok"` | Not proof of usability — a wedged inference still reports ok. Cross-check `.voice_server.log` timings |
| Console `RangeError` + page dies when opening a tab holding a `gr.Dataframe` | The banned `select`-on-`gr.Tab` recursion (gradio 6.14.x). Fix is never to attach that listener, not to patch the symptom |

## Step 6 — Cleanup

`close_page` the tab you opened. If you launched the UI for this session,
stop it (`tts server stop` only manages the API server on 5123; the UI is
a separate process — kill what you started). Leave the user's own running
processes alone.

## Guardrails

- Console messages (`list_console_messages`) are evidence, not noise to
  silence: capture them before and after the reproduction.
- When a UI check fails, cite which step above fired before proposing a
  code change — most "UI bugs" in this repo's history were one of the
  five symptoms in Step 5.
