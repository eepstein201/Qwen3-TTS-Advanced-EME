"""WaveSurfer.js integration for Gradio UI.

Provides JavaScript for streaming audio playback with WaveSurfer.js waveform
visualization, replacing Gradio's gr.Audio component. Audio data flows directly
from the TTS server to the browser via fetch() + ReadableStream, bypassing
Gradio's streaming infrastructure entirely.

Wire format (from /generate-stream):
    [sample_rate:4 bytes LE uint32][length:4 bytes LE uint32][audio:length bytes float32 LE]
"""

import json
import os
from functools import lru_cache

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@lru_cache(maxsize=1)
def _get_vendored_wavesurfer() -> str:
    """Return the self-hosted WaveSurfer ESM source (vendored under static/).

    Loaded from disk once and embedded into the StreamingPlayer module so the
    library is served same-origin via a Blob URL instead of an external CDN.
    """
    with open(os.path.join(_STATIC_DIR, "wavesurfer.esm.js"), encoding="utf-8") as f:
        return f.read()


@lru_cache(maxsize=1)
def _get_player_body() -> str:
    """Return the StreamingPlayer class JS (vendored under static/).

    The ~500-line StreamingPlayer implementation lives in a static asset
    (streaming_player.js) rather than inline in this module. Loaded from disk
    once and appended to the dynamic prelude in get_streaming_player_js().
    """
    with open(os.path.join(_STATIC_DIR, "streaming_player.js"), encoding="utf-8") as f:
        return f.read()


def get_script_reexecutor_fn():
    """Return JS function body for demo.load(js=...) to re-execute innerHTML scripts.

    This is the same as get_script_reexecutor_js() but without the <script> wrapper,
    suitable for use with Gradio's demo.load(js=...) parameter which executes
    JavaScript directly.
    """
    return """() => {
        console.log('[ScriptReexecutor] Re-executing scripts from innerHTML...');

        // Re-execute module scripts (like StreamingPlayer)
        var moduleScripts = document.querySelectorAll('script[type="module"]');
        moduleScripts.forEach(function(s) {
            // Only process inline module scripts with substantial content
            if (s.textContent && s.textContent.length > 100 && !s.src) {
                try {
                    var blob = new Blob([s.textContent], { type: 'application/javascript' });
                    var url = URL.createObjectURL(blob);
                    var ns = document.createElement('script');
                    ns.type = 'module';
                    ns.src = url;
                    document.head.appendChild(ns);
                    console.log('[ScriptReexecutor] Re-injected module script');
                } catch(e) {
                    console.error('[ScriptReexecutor] Failed to re-inject module:', e);
                }
            }
        });

        // Re-execute inline scripts that create elements (like WaveSurfer loader).
        // Use Blob + createObjectURL (never eval) so DOM script text is never
        // executed as arbitrary code in the page scope.
        var inlineScripts = document.querySelectorAll('script:not([type]):not([src])');
        inlineScripts.forEach(function(s) {
            if (s.textContent && s.textContent.indexOf('createElement') >= 0) {
                try {
                    var blob = new Blob([s.textContent], { type: 'application/javascript' });
                    var url = URL.createObjectURL(blob);
                    var ns = document.createElement('script');
                    ns.src = url;
                    ns.onload = function() { URL.revokeObjectURL(url); };
                    document.head.appendChild(ns);
                    console.log('[ScriptReexecutor] Re-injected inline script');
                } catch(e) {
                    console.error('[ScriptReexecutor] Failed to re-inject inline:', e);
                }
            }
        });
    }"""


def get_wavesurfer_loader_js():
    """Return a marker comment; WaveSurfer is self-hosted.

    The library is vendored under interface/static and loaded by the
    StreamingPlayer module via a Blob URL (see get_streaming_player_js), so no
    external CDN <script> is injected. Kept as a function for caller
    compatibility.
    """
    return "<!-- WaveSurfer is self-hosted; loaded by the StreamingPlayer module -->"


def get_streaming_player_js():
    """Return the StreamingPlayer class as a JS string.

    StreamingPlayer handles:
    - fetch() to /generate-stream with ReadableStream parsing
    - Web Audio API playback of float32 chunks
    - WaveSurfer waveform visualization (progressive + final)
    - WAV blob creation for saving
    - History playback via loadFile()
    """
    # Embed the vendored WaveSurfer source and expose it as a same-origin Blob
    # URL, so the dynamic import() below never contacts an external CDN.
    prelude = (
        "const WAVESURFER_SRC = " + json.dumps(_get_vendored_wavesurfer()) + ";\n"
        "let _wsModuleUrlCache = null;\n"
        "function _wsModuleUrl() {\n"
        "    if (!_wsModuleUrlCache) {\n"
        "        const blob = new Blob([WAVESURFER_SRC], { type: 'application/javascript' });\n"
        "        _wsModuleUrlCache = URL.createObjectURL(blob);\n"
        "    }\n"
        "    return _wsModuleUrlCache;\n"
        "}\n"
    )
    return prelude + _get_player_body()  # nosec B608


def get_player_html(tab_id):
    """Return HTML template for WaveSurfer container + controls.

    Args:
        tab_id: Unique identifier for the tab (e.g., 'clone', 'design', 'custom').
    """
    return f"""
    <style>
        .ws-btn {{
            padding: 10px 20px;
            border-radius: 4px;
            border: 1px solid var(--border-color-primary, #d1d5db);
            background: var(--button-secondary-background-fill, #e5e7eb);
            color: var(--button-secondary-text-color, #374151);
            cursor: pointer;
            font-size: 0.9em;
        }}
        .ws-btn:focus {{
            outline: 2px solid #4a9eff;
            outline-offset: 2px;
        }}
        .ws-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        .ws-waveform {{
            width: 100%;
            min-height: 80px;
            background: var(--block-background-fill, #f7f7f8);
            border-radius: 8px;
            margin-bottom: 8px;
            overflow: hidden;
        }}
        .ws-controls {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
    </style>
    <div id="{tab_id}-player" style="margin-top: 8px;">
        <div id="{tab_id}-waveform" class="ws-waveform"></div>
        <div class="ws-controls">
            <button id="{tab_id}-play-btn" class="ws-btn"
                    onclick="if (window.getOrCreatePlayer) window.getOrCreatePlayer('{tab_id}').play()"
                    disabled aria-label="Play or pause audio">
                Play / Pause
            </button>
            <button id="{tab_id}-download-btn" class="ws-btn"
                    onclick="if (window.getOrCreatePlayer) window.getOrCreatePlayer('{tab_id}').download()"
                    disabled aria-label="Download audio">
                Download
            </button>
            <input type="range" id="{tab_id}-volume" min="0" max="1" step="0.1" value="1"
                   aria-label="Volume"
                   oninput="if (window.getOrCreatePlayer) window.getOrCreatePlayer('{tab_id}').setVolume(this.value)"
                   style="width: 80px;">
            <select id="{tab_id}-speed" aria-label="Playback speed"
                    onchange="if (window.getOrCreatePlayer) window.getOrCreatePlayer('{tab_id}').setSpeed(this.value)"
                    style="background: var(--button-secondary-background-fill, #e5e7eb); color: var(--button-secondary-text-color, #374151); border: 1px solid var(--border-color-primary, #d1d5db); border-radius: 4px; padding: 4px;">
                <option value="0.5">0.5x</option>
                <option value="0.75">0.75x</option>
                <option value="1" selected>1x</option>
                <option value="1.25">1.25x</option>
                <option value="1.5">1.5x</option>
                <option value="2">2x</option>
            </select>
            <span id="{tab_id}-time" style="color: var(--body-text-color-subdued, #6b7280); font-size: 0.9em;">0:00 / 0:00</span>
            <span id="{tab_id}-status" role="status" aria-live="polite" aria-atomic="true"
                  style="color: var(--body-text-color-subdued, #6b7280); font-size: 0.9em; margin-left: 8px;"></span>
        </div>
    </div>
    """


def get_load_into_player_js(tab_id):
    """JS that reads a gr.Audio element's src and loads it into a WaveSurfer player.

    Used for history replay and Colab fallback: Python sets a hidden gr.Audio
    to a file path, Gradio converts it to an HTTP URL, then this JS extracts
    the URL and feeds it to the tab's StreamingPlayer.loadFile().

    Args:
        tab_id: The tab identifier matching the player instance.
    """
    return f"""
    (audioData) => {{
        if (!audioData) return null;
        try {{
            const url = (typeof audioData === 'object' && audioData.url) ? audioData.url
                      : (typeof audioData === 'string') ? audioData : null;
            if (url) {{
                if (typeof window.getOrCreatePlayer !== 'function') {{
                    console.error('[LoadPlayer] getOrCreatePlayer not available');
                    return audioData;
                }}
                const player = window.getOrCreatePlayer('{tab_id}');
                player.loadFile(url);
            }}
        }} catch(e) {{ console.warn('[LoadPlayer] Error:', e); }}
        return audioData;
    }}
    """


def get_copy_transcript_js():
    """JS that copies a transcript to the clipboard when action === 'copy'.

    Wired as the ``js`` of a ``.then()`` chained after ``on_history_select``.
    Receives the hidden select-payload JSON and the current history-status HTML.
    When the payload action is ``'copy'``, writes ``payload.text`` to the
    clipboard (the UI runs on 127.0.0.1, a secure context, so
    ``navigator.clipboard`` is available) and returns ``payload.ok`` — or
    ``payload.fail`` on rejection / missing clipboard API. For any other action
    (replay/delete) returns the current status HTML unchanged (passthrough), so
    this chain is a no-op for non-copy clicks. No eval / no inline handlers.
    """
    return """
    async (payload, currentStatusHtml) => {
        try {
            if (!payload || payload.action !== 'copy') {
                return currentStatusHtml;
            }
            if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
                return payload.fail || '';
            }
            await navigator.clipboard.writeText(payload.text || '');
            return payload.ok || '';
        } catch (e) {
            console.warn('[CopyTranscript] Error:', e);
            return payload.fail || '';
        }
    }
    """


def get_clear_player_js(tab_id):
    """JS that clears the WaveSurfer waveform + player state for ``tab_id``.

    Wired as a js-only ``.then()`` after history actions. Clears the waveform
    only when ``payload.action`` is ``'delete'`` (row removed) or ``'clear'``
    (Clear All), so replay and copy clicks leave the current waveform touched.
    Passthrough return (the payload) so the hidden component is unchanged.
    """
    return f"""
    (payload) => {{
        try {{
            if (payload && (payload.action === 'delete' || payload.action === 'clear')) {{
                if (typeof window.getOrCreatePlayer === 'function') {{
                    const player = window.getOrCreatePlayer('{tab_id}');
                    if (player && typeof player.reset === 'function') {{
                        player.reset();
                    }}
                }}
            }}
        }} catch (e) {{
            console.warn('[ClearPlayer] Error:', e);
        }}
        return payload;
    }}
    """


def get_streaming_trigger_js(tab_id):
    """Return JS function that reads the config JSON and starts streaming.

    This is used as the `js` parameter in a Gradio `.then()` call. It receives
    the streaming config from Python, starts the StreamingPlayer, and writes
    the result (base64 WAV or error) back to a hidden textbox.

    Args:
        tab_id: The tab identifier matching the player instance.
    """
    return f"""
    async (config) => {{
        if (!config || !config.server_url) {{
            return '';
        }}

        // Guard: check if player module is loaded
        if (typeof window.getOrCreatePlayer !== 'function') {{
            console.error('[StreamingTrigger] getOrCreatePlayer type:', typeof window.getOrCreatePlayer);
            console.error('[StreamingTrigger] window._streamingPlayers:', window._streamingPlayers);
            return 'ERROR:Audio player not loaded. Refresh the page and try again.';
        }}

        try {{
            const player = window.getOrCreatePlayer('{tab_id}');
            const blob = await player.startStreaming(
                config.server_url, config.auth_token, config.payload
            );
            if (blob) {{
                // Convert WAV blob to base64 for Python to save (batched for performance)
                const arrayBuffer = await blob.arrayBuffer();
                const bytes = new Uint8Array(arrayBuffer);
                const chunkSize = 8192;
                let binary = '';
                for (let i = 0; i < bytes.length; i += chunkSize) {{
                    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
                }}
                return btoa(binary);
            }}
            // Distinguish timeout from user cancel for Python error handling
            if (player._timedOut) {{
                return 'TIMEOUT';
            }}
            return '';
        }} catch (e) {{
            console.error('[StreamingTrigger] Error:', e);
            return 'ERROR:' + e.message;
        }}
    }}
    """


def get_cancel_js(tab_id):
    """Return JS that stops local player playback.

    Server-side cancellation is handled by the Python cancel handler
    (cancel_streaming_generation) — no auth token is sent to the browser.

    Args:
        tab_id: The tab identifier matching the player instance.
    """
    return f"""
    async (config) => {{
        try {{
            if (typeof window.getOrCreatePlayer !== 'function') {{
                console.error('[Cancel] getOrCreatePlayer not available');
                return 'Cancelled';
            }}
            const player = window.getOrCreatePlayer('{tab_id}');
            player.stop();
        }} catch (e) {{
            console.error('[Cancel] Error:', e);
        }}
        return 'Cancelled';
    }}
    """
