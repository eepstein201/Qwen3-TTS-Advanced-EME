"""Tests for WaveSurfer-based UI generation wiring.

Replaces the old audio reset JS tests now that WaveSurfer handles playback.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import gradio as gr


class TestWireGenerationTabIntegration(unittest.TestCase):
    """Test the 3-step generation wiring (Python -> JS -> Python)."""

    def setUp(self):
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
        self.mock_audio_output = Mock(spec=gr.Audio)

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
            audio_output=self.mock_audio_output,
        )

        self.mock_btn.click.assert_called_once()

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
            audio_output=self.mock_audio_output,
        )

        # The chain should have multiple .then() calls
        chain = self.mock_btn.click.return_value
        self.assertTrue(chain.then.called)
        # At least 3 .then() calls: text capture, JS streaming, save, model update
        self.assertGreaterEqual(chain.then.call_count, 3)

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
            audio_output=self.mock_audio_output,
        )

        self.mock_cancel_btn.click.assert_called_once()


if __name__ == "__main__":
    unittest.main()
