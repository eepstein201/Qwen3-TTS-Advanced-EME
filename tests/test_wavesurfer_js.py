"""Tests for WaveSurfer.js integration module."""

import unittest


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
        from unittest.mock import patch, MagicMock
        from qwen3_tts.interface.ui import _prepare_streaming_config

        with patch('qwen3_tts.interface.ui.TTSClient') as mock_cls:
            mock_client = MagicMock()
            mock_client.is_server_running.return_value = False
            mock_cls.return_value = mock_client

            config, status = _prepare_streaming_config("clone", "hello", "(none)", 0.7, 50, 0.95, 1.05, "")
            self.assertIsNone(config)
            self.assertIn("not running", status)


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
        import os
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
            with patch('qwen3_tts.interface.ui.os.path.expanduser',
                       side_effect=lambda p: p.replace("~/Downloads", tmpdir)):
                status, _, history, _ = _save_completed_audio(b64, "clone", "hello", [])
                self.assertIn("Generated:", status)
                self.assertEqual(len(history), 1)


if __name__ == "__main__":
    unittest.main()
