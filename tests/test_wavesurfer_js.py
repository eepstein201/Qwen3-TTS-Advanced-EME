"""Tests for WaveSurfer.js integration module."""

import unittest

# Check for gradio availability
try:
    import gradio as gr  # noqa: F401
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

skip_if_no_gradio = unittest.skipUnless(HAS_GRADIO, "requires gradio")


class TestWaveSurferLoaderJS(unittest.TestCase):
    """Test the WaveSurfer CDN loader script."""

    def test_returns_script_tag(self):
        from qwen3_tts.interface.wavesurfer_js import get_wavesurfer_loader_js
        js = get_wavesurfer_loader_js()
        self.assertIn("<script>", js)
        self.assertIn("</script>", js)

    def test_loads_wavesurfer_from_cdn(self):
        from qwen3_tts.interface.wavesurfer_js import get_wavesurfer_loader_js
        js = get_wavesurfer_loader_js()
        self.assertIn("wavesurfer", js.lower())
        self.assertIn("unpkg.com", js)

    def test_sets_fallback_flag(self):
        from qwen3_tts.interface.wavesurfer_js import get_wavesurfer_loader_js
        js = get_wavesurfer_loader_js()
        self.assertIn("_wavesurferFailed", js)


class TestStreamingPlayerJS(unittest.TestCase):
    """Test the StreamingPlayer class JavaScript."""

    def _get_js(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_player_js
        return get_streaming_player_js()

    def test_contains_class_definition(self):
        js = self._get_js()
        self.assertIn("class StreamingPlayer", js)

    def test_contains_wavesurfer_create(self):
        js = self._get_js()
        self.assertIn("WaveSurfer.create", js)

    def test_contains_readable_stream(self):
        js = self._get_js()
        self.assertIn("getReader()", js)

    def test_contains_binary_parsing(self):
        js = self._get_js()
        self.assertIn("DataView", js)
        self.assertIn("getUint32", js)

    def test_contains_web_audio(self):
        js = self._get_js()
        self.assertIn("AudioContext", js)
        self.assertIn("createBufferSource", js)

    def test_contains_error_handling(self):
        js = self._get_js()
        self.assertIn("try", js)
        self.assertIn("catch", js)

    def test_contains_wav_creation(self):
        js = self._get_js()
        self.assertIn("RIFF", js)
        self.assertIn("WAVE", js)
        self.assertIn("audio/wav", js)

    def test_contains_abort_support(self):
        js = self._get_js()
        self.assertIn("AbortController", js)
        self.assertIn("AbortError", js)

    def test_exposes_global_registry(self):
        js = self._get_js()
        self.assertIn("_streamingPlayers", js)
        self.assertIn("getOrCreatePlayer", js)


class TestPlayerHTML(unittest.TestCase):
    """Test the HTML template for the WaveSurfer container."""

    def test_contains_waveform_container(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn("clone-waveform", html)
        self.assertIn("clone-player", html)

    def test_contains_play_button(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn("clone-play-btn", html)
        self.assertIn("Play", html)

    def test_contains_download_button(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn("clone-download-btn", html)
        self.assertIn("Download", html)

    def test_contains_status_span(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("design")
        self.assertIn("design-status", html)

    def test_different_tabs_have_unique_ids(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        clone_html = get_player_html("clone")
        design_html = get_player_html("design")
        self.assertIn("clone-waveform", clone_html)
        self.assertIn("design-waveform", design_html)
        self.assertNotIn("design-waveform", clone_html)


class TestLoadIntoPlayerJS(unittest.TestCase):
    """Test the JS function for loading audio into a WaveSurfer player."""

    def test_contains_load_file(self):
        from qwen3_tts.interface.wavesurfer_js import get_load_into_player_js
        js = get_load_into_player_js("clone")
        self.assertIn("loadFile", js)

    def test_contains_get_or_create_player(self):
        from qwen3_tts.interface.wavesurfer_js import get_load_into_player_js
        js = get_load_into_player_js("clone")
        self.assertIn("getOrCreatePlayer", js)

    def test_uses_correct_tab_id(self):
        from qwen3_tts.interface.wavesurfer_js import get_load_into_player_js
        js = get_load_into_player_js("history")
        self.assertIn("'history'", js)
        self.assertNotIn("'clone'", js)

    def test_handles_object_url(self):
        from qwen3_tts.interface.wavesurfer_js import get_load_into_player_js
        js = get_load_into_player_js("clone")
        self.assertIn("audioData.url", js)

    def test_handles_string_url(self):
        from qwen3_tts.interface.wavesurfer_js import get_load_into_player_js
        js = get_load_into_player_js("clone")
        self.assertIn("typeof audioData === 'string'", js)

    def test_handles_null_input(self):
        from qwen3_tts.interface.wavesurfer_js import get_load_into_player_js
        js = get_load_into_player_js("clone")
        self.assertIn("if (!audioData)", js)


class TestPlayerConstants(unittest.TestCase):
    """Test that magic numbers are replaced with named constants."""

    def _get_js(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_player_js
        return get_streaming_player_js()

    def test_hard_timeout_constant_used(self):
        js = self._get_js()
        self.assertIn("HARD_TIMEOUT_MS", js)
        # 300000 should only appear in the constant definition, not in setTimeout
        lines_with_300000 = [line for line in js.split('\n') if '300000' in line]
        self.assertEqual(len(lines_with_300000), 1)
        self.assertIn("const HARD_TIMEOUT_MS", lines_with_300000[0])

    def test_idle_timeout_constant_used(self):
        js = self._get_js()
        self.assertIn("IDLE_TIMEOUT_MS", js)
        # 60000 should only appear in the constant definition
        lines_with_60000 = [line for line in js.split('\n') if '60000' in line]
        self.assertEqual(len(lines_with_60000), 1)
        self.assertIn("const IDLE_TIMEOUT_MS", lines_with_60000[0])

    def test_waveform_update_constant_used(self):
        js = self._get_js()
        self.assertIn("WAVEFORM_UPDATE_MS", js)

    def test_waveform_max_bins_constant_used(self):
        js = self._get_js()
        self.assertIn("WAVEFORM_MAX_BINS", js)

    def test_sample_rate_constant(self):
        js = self._get_js()
        self.assertIn("DEFAULT_SAMPLE_RATE", js)

    def test_max_chunk_bytes_constant(self):
        js = self._get_js()
        self.assertIn("MAX_CHUNK_BYTES", js)

    def test_waveform_height_constant(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn("min-height:", html)


class TestDeadCodeRemoval(unittest.TestCase):
    """Test that duplicate function bodies are removed."""

    def test_no_duplicate_reexecutor_body(self):
        from qwen3_tts.interface.wavesurfer_js import get_script_reexecutor_fn
        js = get_script_reexecutor_fn()
        self.assertEqual(js.count("Re-executing scripts"), 1)

    def test_no_duplicate_loader_body(self):
        from qwen3_tts.interface.wavesurfer_js import get_wavesurfer_loader_js
        js = get_wavesurfer_loader_js()
        self.assertEqual(js.count("_wavesurferFailed"), 1)


class TestPlayerAccessibility(unittest.TestCase):
    """Test WCAG 2.1 AA compliance for player controls."""

    def test_play_button_has_aria_label(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn('aria-label="Play or pause audio"', html)

    def test_download_button_has_aria_label(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn('aria-label="Download audio"', html)

    def test_status_has_live_region(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn('role="status"', html)
        self.assertIn('aria-live="polite"', html)

    def test_buttons_have_focus_styles(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn(":focus", html)

    def test_disabled_button_styling(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn(":disabled", html)
        self.assertIn("cursor: not-allowed", html)

    def test_css_class_on_buttons(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn('class="ws-btn"', html)

    def test_css_block_present(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn("<style>", html)
        self.assertIn(".ws-btn", html)


class TestPlayerControls(unittest.TestCase):
    """Test volume, speed, time display, and download filename."""

    def test_volume_slider_in_html(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn('type="range"', html)
        self.assertIn("clone-volume", html)
        self.assertIn('aria-label="Volume"', html)

    def test_volume_handler_in_js(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_player_js
        js = get_streaming_player_js()
        self.assertIn("setVolume", js)

    def test_speed_selector_in_html(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn("<select", html)
        self.assertIn("clone-speed", html)
        self.assertIn("1.5x", html)

    def test_speed_handler_in_js(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_player_js
        js = get_streaming_player_js()
        self.assertIn("playbackRate", js)

    def test_time_display_in_html(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn("clone-time", html)

    def test_time_update_in_js(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_player_js
        js = get_streaming_player_js()
        self.assertIn("_formatTime", js)
        self.assertIn("audioprocess", js)

    def test_download_uses_mode_timestamp(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_player_js
        js = get_streaming_player_js()
        self.assertNotIn("'tts_output.wav'", js)
        self.assertIn("toISOString", js)

    def test_format_time_has_nan_guard(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_player_js
        js = get_streaming_player_js()
        self.assertIn("isFinite", js)

    def test_volume_uses_oninput_not_onchange(self):
        from qwen3_tts.interface.wavesurfer_js import get_player_html
        html = get_player_html("clone")
        self.assertIn('oninput=', html)
        # volume slider should use oninput for live updates
        volume_section = html[html.index('clone-volume'):]
        self.assertNotIn('onchange=', volume_section.split('>')[0])


class TestStreamFormatValidation(unittest.TestCase):
    """Test stream frame validation in the parser loop."""

    def _get_js(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_player_js
        return get_streaming_player_js()

    def test_validates_sample_rate(self):
        js = self._get_js()
        self.assertIn("sampleRate === 0", js)

    def test_validates_audio_length(self):
        js = self._get_js()
        # MAX_CHUNK_BYTES is used in chunk size check
        self.assertIn("MAX_CHUNK_BYTES", js)
        # It should appear more than once (definition + usage)
        self.assertGreater(js.count("MAX_CHUNK_BYTES"), 1)

    def test_rejects_zero_length_frames(self):
        js = self._get_js()
        self.assertIn("audioLen === 0", js)

    def test_chunk_too_large_throws(self):
        js = self._get_js()
        self.assertIn("Chunk too large", js)


class TestMemoryCleanup(unittest.TestCase):
    """Test memory cleanup after finalization."""

    def _get_js(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_player_js
        return get_streaming_player_js()

    def test_chunks_cleared_after_finalize(self):
        js = self._get_js()
        # _finalizeWaveform method body (index 2: after method definition) should clear allChunks
        # split()[0]: before call site; split()[1]: between call and definition; split()[2]: body
        parts = js.split("_finalizeWaveform")
        method_body = "".join(parts[2:])
        self.assertIn("this.allChunks = []", method_body)

    def test_url_revoke_has_timeout_fallback(self):
        js = self._get_js()
        # Should have a setTimeout fallback for revokeObjectURL
        # Count: should appear more than once (once in once('ready'), once in timeout)
        revoke_count = js.count("revokeObjectURL")
        self.assertGreater(revoke_count, 2)


class TestErrorLogging(unittest.TestCase):
    """Test that catch blocks log warnings instead of swallowing silently."""

    def _get_js(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_player_js
        return get_streaming_player_js()

    def test_no_silent_empty_catch(self):
        js = self._get_js()
        self.assertNotIn("catch(e) {}", js)

    def test_audiocontext_close_catch_logs(self):
        js = self._get_js()
        # The audioContext.close() catch should log, not be empty
        # Find the close() section and verify it has a console.warn
        self.assertIn("audioContext.close", js)
        # The pattern "close();" followed eventually by console.warn should exist
        close_idx = js.index("audioContext.close")
        catch_region = js[close_idx:close_idx + 200]
        self.assertIn("console.warn", catch_region)


class TestStreamingTriggerJS(unittest.TestCase):
    """Test the JS trigger function for starting streaming."""

    def test_contains_fetch_call(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_trigger_js
        js = get_streaming_trigger_js("clone")
        self.assertIn("startStreaming", js)
        self.assertIn("config.server_url", js)

    def test_returns_base64(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_trigger_js
        js = get_streaming_trigger_js("clone")
        self.assertIn("btoa", js)

    def test_handles_error(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_trigger_js
        js = get_streaming_trigger_js("clone")
        self.assertIn("ERROR:", js)
        self.assertIn("catch", js)

    def test_uses_tab_id(self):
        from qwen3_tts.interface.wavesurfer_js import get_streaming_trigger_js
        js = get_streaming_trigger_js("design")
        self.assertIn("design", js)


class TestCancelJS(unittest.TestCase):
    """Test the cancel JS function."""

    def test_calls_cancel_endpoint(self):
        from qwen3_tts.interface.wavesurfer_js import get_cancel_js
        js = get_cancel_js("clone")
        self.assertIn("cancel-generation", js)

    def test_stops_player(self):
        from qwen3_tts.interface.wavesurfer_js import get_cancel_js
        js = get_cancel_js("clone")
        self.assertIn("player.stop()", js)


@skip_if_no_gradio
class TestPrepareStreamingConfig(unittest.TestCase):
    """Test the Python config preparation function."""

    def test_returns_error_on_empty_text(self):
        from qwen3_tts.interface.ui import _prepare_streaming_config
        config, status = _prepare_streaming_config("clone", "", "(none)", 0.7, 50, 0.95, 1.05, "")
        self.assertIsNone(config)
        self.assertIn("Error", status)

    def test_returns_error_on_empty_description_for_design(self):
        from qwen3_tts.interface.ui import _prepare_streaming_config
        config, status = _prepare_streaming_config("design", "hello", "(none)", 0.7, 50, 0.95, 1.05, "",
                                                    description="")
        self.assertIsNone(config)
        self.assertIn("Error", status)

    def test_returns_error_when_server_not_running(self):
        from unittest.mock import patch
        from qwen3_tts.interface.ui import _prepare_streaming_config

        with patch('qwen3_tts.interface.ui.generation.is_server_running', return_value=False):
            config, status = _prepare_streaming_config("clone", "hello", "(none)", 0.7, 50, 0.95, 1.05, "")
            self.assertIsNone(config)
            self.assertIn("not running", status)


@skip_if_no_gradio
class TestSaveCompletedAudio(unittest.TestCase):
    """Test the audio save function."""

    def test_returns_cancelled_on_empty(self):
        from qwen3_tts.interface.ui import _save_completed_audio
        status, _, _, _ = _save_completed_audio("", "clone", "hello", [])
        self.assertEqual(status, "Cancelled")

    def test_returns_error_on_error_prefix(self):
        from qwen3_tts.interface.ui import _save_completed_audio
        status, _, _, _ = _save_completed_audio("ERROR:connection refused", "clone", "hello", [])
        self.assertIn("connection refused", status)

    def test_saves_valid_base64_wav(self):
        import base64
        import tempfile
        from unittest.mock import patch
        from qwen3_tts.interface.ui import _save_completed_audio

        # Create a minimal WAV file
        import struct
        wav_data = b'RIFF' + struct.pack('<I', 36) + b'WAVE'
        wav_data += b'fmt ' + struct.pack('<IHHIIHH', 16, 1, 1, 24000, 48000, 2, 16)
        wav_data += b'data' + struct.pack('<I', 0)
        b64 = base64.b64encode(wav_data).decode()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('qwen3_tts.interface.ui.generation.os.path.expanduser',
                       side_effect=lambda p: p.replace("~/Downloads", tmpdir)):
                status, _, history, _ = _save_completed_audio(b64, "clone", "hello", [])
                self.assertIn("Generated:", status)
                self.assertEqual(len(history), 1)


if __name__ == "__main__":
    unittest.main()
