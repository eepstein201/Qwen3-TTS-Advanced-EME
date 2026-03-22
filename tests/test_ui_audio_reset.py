"""Tests for WaveSurfer-based UI generation wiring.

Replaces the old audio reset JS tests now that WaveSurfer handles playback.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock

# Check for gradio availability
try:
    import gradio as gr
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    gr = None  # type: ignore

skip_if_no_gradio = unittest.skipUnless(HAS_GRADIO, "requires gradio")


class TestWireGenerationTabIntegration(unittest.TestCase):
    """Test the 3-step generation wiring (Python -> JS -> Python)."""

    def setUp(self):
        if not HAS_GRADIO:
            return
        self.mock_btn = Mock(spec=gr.Button)
        # Chain .click().then().then().then().then()
        mock_chain = Mock()
        mock_chain.then = Mock(return_value=mock_chain)
        self.mock_btn.click = Mock(return_value=mock_chain)

        self.mock_cancel_btn = Mock(spec=gr.Button)
        self.mock_cancel_btn.click = Mock(return_value=Mock())
        self.mock_status = Mock(spec=gr.Textbox)
        self.mock_stream_config = Mock(spec=gr.JSON)
        self.mock_result_data = Mock(spec=gr.Textbox)
        self.mock_mode_hidden = Mock(spec=gr.Textbox)
        self.mock_text_hidden = Mock(spec=gr.Textbox)
        self.mock_indicator = Mock(spec=gr.HTML)
        self.mock_text = Mock(spec=gr.Textbox)
        self.mock_text.change = Mock()
        self.mock_info = Mock(spec=gr.Textbox)
        self.mock_html = Mock(spec=gr.HTML)
        self.mock_df = Mock(spec=gr.Dataframe)
        self.mock_history_state = Mock()
        self.mock_audio_url_converter = Mock(spec=gr.Audio)

    @skip_if_no_gradio
    def test_wire_generation_tab_calls_click(self):
        """The click handler should be wired."""
        from qwen3_tts.interface.ui import _wire_generation_tab

        mock_handler = Mock(return_value=(None, "status"))

        _wire_generation_tab(
            mode="clone",
            btn=self.mock_btn,
            cancel_btn=self.mock_cancel_btn,
            status=self.mock_status,
            stream_config=self.mock_stream_config,
            result_data=self.mock_result_data,
            mode_hidden=self.mock_mode_hidden,
            text_hidden=self.mock_text_hidden,
            model_indicator=self.mock_indicator,
            text=self.mock_text,
            text_info=self.mock_info,
            inputs_list=[self.mock_text],
            status_html=self.mock_html,
            history_df=self.mock_df,
            config_handler=mock_handler,
            history_state=self.mock_history_state,
            audio_url_converter=self.mock_audio_url_converter,
        )

        self.mock_btn.click.assert_called_once()

    @skip_if_no_gradio
    def test_wire_generation_tab_chains_then_calls(self):
        """Should chain .then() calls for JS streaming and saving."""
        from qwen3_tts.interface.ui import _wire_generation_tab

        mock_handler = Mock(return_value=(None, "status"))

        _wire_generation_tab(
            mode="clone",
            btn=self.mock_btn,
            cancel_btn=self.mock_cancel_btn,
            status=self.mock_status,
            stream_config=self.mock_stream_config,
            result_data=self.mock_result_data,
            mode_hidden=self.mock_mode_hidden,
            text_hidden=self.mock_text_hidden,
            model_indicator=self.mock_indicator,
            text=self.mock_text,
            text_info=self.mock_info,
            inputs_list=[self.mock_text],
            status_html=self.mock_html,
            history_df=self.mock_df,
            config_handler=mock_handler,
            history_state=self.mock_history_state,
            audio_url_converter=self.mock_audio_url_converter,
        )

        # The chain should have multiple .then() calls
        chain = self.mock_btn.click.return_value
        self.assertTrue(chain.then.called)
        # At least 4 .then() calls: text capture, JS streaming, save/fallback, load into player, model update
        self.assertGreaterEqual(chain.then.call_count, 4)

    @skip_if_no_gradio
    def test_cancel_btn_wired(self):
        """Cancel button should be wired to cancel_streaming_generation."""
        from qwen3_tts.interface.ui import _wire_generation_tab

        mock_handler = Mock()

        _wire_generation_tab(
            mode="clone",
            btn=self.mock_btn,
            cancel_btn=self.mock_cancel_btn,
            status=self.mock_status,
            stream_config=self.mock_stream_config,
            result_data=self.mock_result_data,
            mode_hidden=self.mock_mode_hidden,
            text_hidden=self.mock_text_hidden,
            model_indicator=self.mock_indicator,
            text=self.mock_text,
            text_info=self.mock_info,
            inputs_list=[self.mock_text],
            status_html=self.mock_html,
            history_df=self.mock_df,
            config_handler=mock_handler,
            history_state=self.mock_history_state,
            audio_url_converter=self.mock_audio_url_converter,
        )

        self.mock_cancel_btn.click.assert_called_once()


class TestOnHistorySelect(unittest.TestCase):
    """Test history row selection handler."""

    @skip_if_no_gradio
    def test_valid_index_returns_path(self):
        import tempfile
        import os
        from qwen3_tts.interface.ui import on_history_select
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            evt = Mock()
            evt.index = [0]
            history = [{"path": tmp_path}]
            result = on_history_select(evt, history)
            self.assertEqual(result, tmp_path)
        finally:
            os.unlink(tmp_path)

    @skip_if_no_gradio
    def test_invalid_index_returns_none(self):
        from qwen3_tts.interface.ui import on_history_select
        evt = Mock()
        evt.index = [5]
        result = on_history_select(evt, [{"path": "/tmp/test.wav"}])
        self.assertIsNone(result)

    @skip_if_no_gradio
    def test_missing_file_returns_none(self):
        from qwen3_tts.interface.ui import on_history_select
        evt = Mock()
        evt.index = [0]
        result = on_history_select(evt, [{"path": "/nonexistent/file.wav"}])
        self.assertIsNone(result)

    @skip_if_no_gradio
    def test_empty_history_returns_none(self):
        from qwen3_tts.interface.ui import on_history_select
        evt = Mock()
        evt.index = [0]
        result = on_history_select(evt, [])
        self.assertIsNone(result)


class TestGenerateColabFallback(unittest.TestCase):
    """Test the Colab fallback generation handler."""

    @skip_if_no_gradio
    def test_js_success_returns_saved_path(self):
        import base64
        import tempfile
        from qwen3_tts.interface.ui import _generate_server_side

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('qwen3_tts.server.client.TTSClient') as mock_cls, \
                 patch('qwen3_tts.interface.ui.generation.load_config',
                       return_value={"output_directory": tmpdir}), \
                 patch('qwen3_tts.interface.ui.generation.os.path.expanduser',
                       side_effect=lambda p: p.replace("~/Downloads", tmpdir)):
                mock_client = MagicMock()
                mock_cls.return_value = mock_client

                def fake_generate(**kwargs):
                    with open(kwargs['output'], 'wb') as f:
                        f.write(b'fake wav')
                mock_client.generate.side_effect = fake_generate

                config = {"server_side": True, "payload": {"text": "hello", "mode": "clone"}}
                audio_path, status, _, hist, _ = _generate_server_side(
                    "clone", "hello", [], config)
                self.assertIn("Generated:", status)
                self.assertIsNotNone(audio_path)
                self.assertEqual(len(hist), 1)
                mock_client.generate.assert_called_once()

    @skip_if_no_gradio
    def test_none_config_preserves_error(self):
        from qwen3_tts.interface.ui import _generate_server_side
        audio_path, status, _, _, _ = _generate_server_side(
            "clone", "hello", [], None)
        self.assertIsNone(audio_path)

    @skip_if_no_gradio
    def test_cancel_returns_none(self):
        from qwen3_tts.interface.ui import _generate_server_side
        # Non-server_side config means user cancelled
        config = {"payload": {}}
        audio_path, status, _, _, _ = _generate_server_side(
            "clone", "hello", [], config)
        self.assertIsNone(audio_path)
        self.assertEqual(status, "Cancelled")

    @skip_if_no_gradio
    def test_client_error_returns_error_status(self):
        from qwen3_tts.interface.ui import _generate_server_side

        with patch('qwen3_tts.server.client.TTSClient') as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.generate.side_effect = RuntimeError("connection refused")

            config = {"server_side": True, "payload": {"text": "hello", "mode": "clone"}}
            audio_path, status, _, _, _ = _generate_server_side(
                "clone", "hello", [], config)
            self.assertIsNone(audio_path)
            self.assertIn("connection refused", status)


if __name__ == "__main__":
    unittest.main()
