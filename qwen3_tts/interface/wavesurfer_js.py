"""WaveSurfer.js integration for Gradio UI.

Provides JavaScript for streaming audio playback with WaveSurfer.js waveform
visualization, replacing Gradio's gr.Audio component. Audio data flows directly
from the TTS server to the browser via fetch() + ReadableStream, bypassing
Gradio's streaming infrastructure entirely.

Wire format (from /generate-stream):
    [sample_rate:4 bytes LE uint32][length:4 bytes LE uint32][audio:length bytes float32 LE]
"""


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
    """Return a <script> tag that loads WaveSurfer 7.x from CDN.

    Injects the script once into the page and exposes window.WaveSurfer.
    Includes a fallback check so callers know if loading failed.
    """
    return """
    <script>
    (function() {
        if (window.WaveSurfer) return;
        var s = document.createElement('script');
        s.src = 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js';
        s.type = 'module';
        s.onload = function() { console.log('[WaveSurfer] Loaded from CDN'); };
        s.onerror = function() {
            console.warn('[WaveSurfer] CDN load failed, falling back to <audio>');
            window._wavesurferFailed = true;
        };
        document.head.appendChild(s);
    })();
    </script>
    """


def get_streaming_player_js():
    """Return the StreamingPlayer class as a JS string.

    StreamingPlayer handles:
    - fetch() to /generate-stream with ReadableStream parsing
    - Web Audio API playback of float32 chunks
    - WaveSurfer waveform visualization (progressive + final)
    - WAV blob creation for saving
    - History playback via loadFile()
    """
    return """
    const HARD_TIMEOUT_MS = 300000;
    const IDLE_TIMEOUT_MS = 60000;
    const WAVEFORM_UPDATE_MS = 200;
    const WAVEFORM_MAX_BINS = 500;
    const DEFAULT_SAMPLE_RATE = 24000;
    const MAX_CHUNK_BYTES = 50 * 1024 * 1024;

    class StreamingPlayer {
        constructor(containerId) {
            this.containerId = containerId;
            this.audioContext = null;
            this.allChunks = [];
            this.sampleRate = DEFAULT_SAMPLE_RATE;
            this.nextPlayTime = 0;
            this.isPlaying = false;
            this.abortController = null;
            this.wavesurfer = null;
            this.waveformUpdateTimer = null;
            this.totalSamples = 0;
            this._initWaveSurfer();
        }

        _initWaveSurfer() {
            try {
                const container = document.getElementById(this.containerId + '-waveform');
                if (!container) return;

                if (window._wavesurferFailed) {
                    this._initFallbackAudio();
                    return;
                }

                // Dynamic import for ES module
                import('https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js').then(module => {
                    const WaveSurfer = module.default;
                    this.wavesurfer = WaveSurfer.create({
                        container: container,
                        waveColor: '#4a9eff',
                        progressColor: '#1a6fdd',
                        cursorColor: '#333',
                        barWidth: 2,
                        barGap: 1,
                        barRadius: 2,
                        height: 80,
                        normalize: true,
                        interact: true,
                        backend: 'WebAudio',
                    });
                    this.wavesurfer.on('finish', () => {
                        this._updateControls('stopped');
                    });
                    this.wavesurfer.on('audioprocess', (time) => {
                        const duration = this.wavesurfer.getDuration();
                        this._updateTime(time, duration);
                    });
                    this.wavesurfer.on('ready', () => {
                        const duration = this.wavesurfer.getDuration();
                        this._updateTime(0, duration);
                    });
                    console.log('[StreamingPlayer] WaveSurfer initialized for', this.containerId);
                }).catch(err => {
                    console.warn('[StreamingPlayer] WaveSurfer import failed, using fallback:', err);
                    this._initFallbackAudio();
                });
            } catch (e) {
                console.warn('[StreamingPlayer] Init error:', e);
                this._initFallbackAudio();
            }
        }

        _initFallbackAudio() {
            const container = document.getElementById(this.containerId + '-waveform');
            if (!container) return;
            const audio = document.createElement('audio');
            audio.controls = true;
            audio.style.width = '100%';
            audio.id = this.containerId + '-fallback-audio';
            container.innerHTML = '';
            container.appendChild(audio);
            this._fallbackAudio = audio;
            this._fallbackAudioUrl = null;
        }

        async startStreaming(serverUrl, authToken, payload) {
            this.reset();
            this._timedOut = false;
            this.isPlaying = true;
            this._updateControls('streaming');
            this._updateStatus('Connecting...');

            let timeoutId = null;
            let idleTimeoutId = null;
            const clearAllTimeouts = () => {
                if (timeoutId) { clearTimeout(timeoutId); timeoutId = null; }
                if (idleTimeoutId) { clearTimeout(idleTimeoutId); idleTimeoutId = null; }
            };

            try {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: DEFAULT_SAMPLE_RATE
                });
                // Fire-and-forget resume — do NOT await; AudioContext.resume() returns a
                // Promise that may never resolve if user-activation has expired after
                // Gradio's Python round-trip. Awaiting it would hang startStreaming()
                // before fetch() is ever called, causing "Connecting..." forever.
                this.audioContext.resume().catch(() => {});
                this.nextPlayTime = this.audioContext.currentTime + 0.1;

                this.abortController = new AbortController();

                // Hard timeout for the entire request
                timeoutId = setTimeout(() => {
                    this._timedOut = true;
                    if (this.abortController) this.abortController.abort();
                }, HARD_TIMEOUT_MS);

                const response = await fetch(serverUrl + '/generate-stream', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer ' + authToken,
                    },
                    body: JSON.stringify(payload),
                    signal: this.abortController.signal,
                });

                if (!response.ok) {
                    clearAllTimeouts();
                    const errText = await response.text();
                    throw new Error('Server error ' + response.status + ': ' + errText);
                }

                // Server accepted the request — show generating status
                this._updateStatus('Generating...');

                const reader = response.body.getReader();
                let buffer = new Uint8Array(0);
                let chunkCount = 0;

                // Idle timeout — resets on each received chunk
                const resetIdleTimeout = () => {
                    if (idleTimeoutId) clearTimeout(idleTimeoutId);
                    idleTimeoutId = setTimeout(() => {
                        this._timedOut = true;
                        if (this.abortController) this.abortController.abort();
                    }, IDLE_TIMEOUT_MS);
                };
                resetIdleTimeout();

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    resetIdleTimeout();

                    // Append to buffer
                    const newBuffer = new Uint8Array(buffer.length + value.length);
                    newBuffer.set(buffer);
                    newBuffer.set(value, buffer.length);
                    buffer = newBuffer;

                    // Parse complete frames from buffer
                    while (buffer.length >= 8) {
                        const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                        const sampleRate = view.getUint32(0, true);
                        const audioLen = view.getUint32(4, true);

                        if (buffer.length < 8 + audioLen) break;

                        // Validate frame header values before processing
                        if (sampleRate === 0 || audioLen === 0) {
                            buffer = buffer.slice(8 + audioLen);
                            continue;
                        }
                        if (audioLen > MAX_CHUNK_BYTES) {
                            throw new Error('Chunk too large: ' + audioLen + ' bytes');
                        }

                        // Extract float32 audio data
                        const audioBytes = buffer.slice(8, 8 + audioLen);
                        const float32Data = new Float32Array(audioBytes.buffer, audioBytes.byteOffset, audioLen / 4);

                        this.sampleRate = sampleRate;
                        this.allChunks.push(new Float32Array(float32Data));
                        this.totalSamples += float32Data.length;
                        chunkCount++;

                        this._playChunk(float32Data, sampleRate);
                        this._updateStatus('Streaming... ' + chunkCount + ' chunks (' +
                            (this.totalSamples / sampleRate).toFixed(1) + 's)');

                        // Advance buffer past this frame
                        buffer = buffer.slice(8 + audioLen);
                    }

                    // Throttled waveform update
                    this._scheduleWaveformUpdate();
                }

                clearAllTimeouts();
                this._finalizeWaveform();
                this._updateStatus('Complete: ' + chunkCount + ' chunks (' +
                    (this.totalSamples / this.sampleRate).toFixed(1) + 's)');
                this._updateControls('complete');
                return this._createWavBlob();

            } catch (e) {
                clearAllTimeouts();
                if (e.name === 'AbortError') {
                    if (this._timedOut) {
                        this._updateStatus('Error: Timed out waiting for audio');
                        this._updateControls('error');
                    } else {
                        this._updateStatus('Cancelled');
                        this._updateControls('stopped');
                    }
                    return null;
                }
                console.error('[StreamingPlayer] Streaming error:', e);
                this._updateStatus('Error: ' + e.message);
                this._updateControls('error');
                throw e;
            }
        }

        _playChunk(float32Data, sampleRate) {
            if (!this.audioContext || !this.isPlaying) return;

            // Generation took seconds — user-activation may have refreshed.
            // Attempt resume here so audio plays as soon as context allows.
            if (this.audioContext.state === 'suspended') {
                this.audioContext.resume().catch(e => console.warn('[StreamingPlayer] Resume error:', e));
            }

            try {
                const audioBuffer = this.audioContext.createBuffer(1, float32Data.length, sampleRate);
                audioBuffer.getChannelData(0).set(float32Data);
                const source = this.audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(this.audioContext.destination);

                const now = this.audioContext.currentTime;
                if (this.nextPlayTime < now) {
                    this.nextPlayTime = now + 0.05;
                }
                source.start(this.nextPlayTime);
                this.nextPlayTime += audioBuffer.duration;
            } catch (e) {
                console.warn('[StreamingPlayer] Chunk playback error:', e);
            }
        }

        _scheduleWaveformUpdate() {
            if (this.waveformUpdateTimer) return;
            this.waveformUpdateTimer = setTimeout(() => {
                this.waveformUpdateTimer = null;
                this._updateWaveformPeaks();
            }, WAVEFORM_UPDATE_MS);
        }

        _updateWaveformPeaks() {
            if (!this.wavesurfer || this.allChunks.length === 0) return;

            try {
                // Build peaks array from all chunks
                const totalLen = this.allChunks.reduce((sum, c) => sum + c.length, 0);
                const numBins = Math.min(totalLen, WAVEFORM_MAX_BINS);
                const samplesPerBin = Math.max(1, Math.floor(totalLen / numBins));
                const peaks = new Float32Array(numBins);

                let sampleIdx = 0;
                let chunkIdx = 0;
                let chunkOffset = 0;

                for (let bin = 0; bin < numBins; bin++) {
                    let maxVal = 0;
                    for (let s = 0; s < samplesPerBin && chunkIdx < this.allChunks.length; s++) {
                        const val = Math.abs(this.allChunks[chunkIdx][chunkOffset]);
                        if (val > maxVal) maxVal = val;
                        chunkOffset++;
                        if (chunkOffset >= this.allChunks[chunkIdx].length) {
                            chunkIdx++;
                            chunkOffset = 0;
                        }
                        sampleIdx++;
                    }
                    peaks[bin] = maxVal;
                }

                // Load peaks into WaveSurfer for visualization
                const duration = totalLen / this.sampleRate;
                this.wavesurfer.load('', [peaks], duration);
            } catch (e) {
                console.warn('[StreamingPlayer] Waveform update error:', e);
            }
        }

        _finalizeWaveform() {
            if (this.waveformUpdateTimer) {
                clearTimeout(this.waveformUpdateTimer);
                this.waveformUpdateTimer = null;
            }

            // Create full WAV blob and load into WaveSurfer for scrub/replay
            const wavBlob = this._createWavBlob();
            // Free chunk memory — WAV blob is the authoritative copy from here on
            this.allChunks = [];
            this.totalSamples = 0;
            if (wavBlob && this.wavesurfer) {
                const url = URL.createObjectURL(wavBlob);
                this.wavesurfer.load(url);
                // Primary: revoke after WaveSurfer decodes; fallback: revoke after 30s if ready never fires
                this.wavesurfer.once('ready', () => URL.revokeObjectURL(url));
                setTimeout(() => URL.revokeObjectURL(url), 30000);
            } else if (wavBlob && this._fallbackAudio) {
                if (this._fallbackAudioUrl) {
                    URL.revokeObjectURL(this._fallbackAudioUrl);
                }
                this._fallbackAudioUrl = URL.createObjectURL(wavBlob);
                this._fallbackAudio.src = this._fallbackAudioUrl;
            }
        }

        _createWavBlob() {
            if (this.allChunks.length === 0) return null;

            const totalLen = this.allChunks.reduce((sum, c) => sum + c.length, 0);
            const combined = new Float32Array(totalLen);
            let offset = 0;
            for (const chunk of this.allChunks) {
                combined.set(chunk, offset);
                offset += chunk.length;
            }

            // Build WAV file
            const numChannels = 1;
            const bitsPerSample = 16;
            const bytesPerSample = bitsPerSample / 8;
            const blockAlign = numChannels * bytesPerSample;
            const dataSize = totalLen * blockAlign;
            const headerSize = 44;
            const arrayBuffer = new ArrayBuffer(headerSize + dataSize);
            const view = new DataView(arrayBuffer);

            // RIFF header
            this._writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + dataSize, true);
            this._writeString(view, 8, 'WAVE');

            // fmt chunk
            this._writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);  // PCM
            view.setUint16(22, numChannels, true);
            view.setUint32(24, this.sampleRate, true);
            view.setUint32(28, this.sampleRate * blockAlign, true);
            view.setUint16(32, blockAlign, true);
            view.setUint16(34, bitsPerSample, true);

            // data chunk
            this._writeString(view, 36, 'data');
            view.setUint32(40, dataSize, true);

            // Convert float32 to int16
            let pos = 44;
            for (let i = 0; i < totalLen; i++) {
                const s = Math.max(-1, Math.min(1, combined[i]));
                view.setInt16(pos, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                pos += 2;
            }

            return new Blob([arrayBuffer], { type: 'audio/wav' });
        }

        _writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }

        loadFile(url) {
            if (this.wavesurfer) {
                this.wavesurfer.load(url);
                this._updateControls('complete');
            } else if (this._fallbackAudio) {
                if (this._fallbackAudioUrl) {
                    URL.revokeObjectURL(this._fallbackAudioUrl);
                    this._fallbackAudioUrl = null;
                }
                this._fallbackAudio.src = url;
            }
        }

        play() {
            if (this.wavesurfer) {
                this.wavesurfer.playPause();
            } else if (this._fallbackAudio) {
                if (this._fallbackAudio.paused) {
                    this._fallbackAudio.play();
                } else {
                    this._fallbackAudio.pause();
                }
            }
        }

        setVolume(value) {
            const vol = parseFloat(value);
            if (this.wavesurfer) {
                this.wavesurfer.setVolume(vol);
            }
            if (this._fallbackAudio) {
                this._fallbackAudio.volume = vol;
            }
        }

        setSpeed(value) {
            const rate = parseFloat(value);
            if (this.wavesurfer) {
                this.wavesurfer.setPlaybackRate(rate);
            }
            if (this._fallbackAudio) {
                this._fallbackAudio.playbackRate = rate;
            }
        }

        _formatTime(seconds) {
            if (!isFinite(seconds) || seconds < 0) return '0:00';
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            return m + ':' + (s < 10 ? '0' : '') + s;
        }

        _updateTime(current, duration) {
            const el = document.getElementById(this.containerId + '-time');
            if (el) {
                el.textContent = this._formatTime(current) + ' / ' + this._formatTime(duration);
            }
        }

        stop() {
            this.isPlaying = false;
            if (this.abortController) {
                this.abortController.abort();
                this.abortController = null;
            }
            if (this.audioContext) {
                try { this.audioContext.close(); } catch(e) { console.warn('[StreamingPlayer] AudioContext close error:', e); }
                this.audioContext = null;
            }
            if (this.wavesurfer) {
                this.wavesurfer.stop();
            }
            this._updateControls('stopped');
        }

        reset() {
            this.stop();
            this._timedOut = false;
            this.allChunks = [];
            this.totalSamples = 0;
            this.nextPlayTime = 0;
            if (this.waveformUpdateTimer) {
                clearTimeout(this.waveformUpdateTimer);
                this.waveformUpdateTimer = null;
            }
            if (this.wavesurfer) {
                this.wavesurfer.empty();
            }
            this._updateStatus('');
            this._updateControls('idle');
        }

        download() {
            const blob = this._createWavBlob();
            if (!blob) return;
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = this.containerId + '_' + new Date().toISOString().slice(0,19).replace(/:/g,'-') + '.wav';
            a.click();
            URL.revokeObjectURL(url);
        }

        _updateStatus(text) {
            const el = document.getElementById(this.containerId + '-status');
            if (el) el.textContent = text;
        }

        _updateControls(state) {
            const playBtn = document.getElementById(this.containerId + '-play-btn');
            const downloadBtn = document.getElementById(this.containerId + '-download-btn');
            if (playBtn) {
                playBtn.disabled = (state === 'idle' || state === 'streaming' || state === 'error');
                playBtn.textContent = (state === 'streaming') ? 'Streaming...' : 'Play / Pause';
            }
            if (downloadBtn) {
                downloadBtn.disabled = (state !== 'complete');
            }
        }
    }

    // Global registry of players by tab ID
    window._streamingPlayers = window._streamingPlayers || {};

    function getOrCreatePlayer(tabId) {
        if (!window._streamingPlayers[tabId]) {
            window._streamingPlayers[tabId] = new StreamingPlayer(tabId);
        }
        return window._streamingPlayers[tabId];
    }

    window.getOrCreatePlayer = getOrCreatePlayer;

    // Diagnostic: confirm module loaded and function is available
    console.log('[StreamingPlayer] Module loaded, getOrCreatePlayer =', typeof window.getOrCreatePlayer);
    """


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
