"""Tests for Gradio Audio component JavaScript reset functionality."""

import unittest
from unittest.mock import Mock, patch
import gradio as gr
from qwen3_tts.interface.ui import _create_audio_reset_js


class TestAudioResetJS(unittest.TestCase):
    """Test the JavaScript reset function generator."""

    def test_returns_non_empty_string(self):
        """JavaScript function should return non-empty string."""
        js = _create_audio_reset_js()
        self.assertIsInstance(js, str)
        self.assertGreater(len(js), 100)

    def test_contains_audio_element_reset(self):
        """Should query and reset audio elements."""
        js = _create_audio_reset_js()
        self.assertIn("querySelectorAll('audio')", js)
        self.assertIn("pause()", js)
        self.assertIn("currentTime", js)

    def test_contains_src_clearing(self):
        """Should remove src attribute to unload buffer."""
        js = _create_audio_reset_js()
        self.assertIn("removeAttribute('src')", js)

    def test_returns_true_on_success(self):
        """Should return true to allow generation to proceed."""
        js = _create_audio_reset_js()
        self.assertIn("return true", js)

    def test_has_error_handling(self):
        """Should wrap in try/catch for non-blocking failure."""
        js = _create_audio_reset_js()
        self.assertIn("try", js)
        self.assertIn("catch", js)
        self.assertIn("console.warn", js)

    def test_no_syntax_errors(self):
        """JavaScript should be valid syntax (basic checks)."""
        js = _create_audio_reset_js()
        # Basic JS syntax validation
        self.assertIn("=>", js)  # Arrow function
        self.assertIn("{", js)   # Opening brace
        self.assertIn("}", js)   # Closing brace


class TestWireGenerationTabIntegration(unittest.TestCase):
    """Test integration of JavaScript reset into button handlers."""

    def setUp(self):
        # Create mock Gradio components
        self.mock_btn = Mock(spec=gr.Button)
        self.mock_btn.click = Mock(return_value=Mock(then=Mock()))

        self.mock_cancel_btn = Mock(spec=gr.Button)
        self.mock_output = Mock(spec=gr.Audio)
        self.mock_status = Mock(spec=gr.Textbox)
        self.mock_indicator = Mock(spec=gr.HTML)
        self.mock_text = Mock(spec=gr.Textbox)
        self.mock_info = Mock(spec=gr.Textbox)
        self.mock_html = Mock(spec=gr.HTML)
        self.mock_df = Mock(spec=gr.Dataframe)

    @patch('qwen3_tts.interface.ui._create_audio_reset_js')
    def test_wire_generation_tab_includes_js_reset(self, mock_reset_js):
        """The click handler should include JavaScript reset."""
        from qwen3_tts.interface.ui import _wire_generation_tab

        mock_reset_js.return_value = "(el) => { return true; }"
        mock_handler = Mock(return_value=(None, "status", "html", [], []))

        _wire_generation_tab(
            mode="clone",
            btn=self.mock_btn,
            cancel_btn=self.mock_cancel_btn,
            output=self.mock_output,
            status=self.mock_status,
            model_indicator=self.mock_indicator,
            text=self.mock_text,
            text_info=self.mock_info,
            inputs_list=[self.mock_text],
            status_html=self.mock_html,
            history_df=self.mock_df,
            handler=mock_handler,
            api_name="generate_clone_streaming"
        )

        # Verify click was called with js parameter
        self.mock_btn.click.assert_called_once()
        call_kwargs = self.mock_btn.click.call_args[1]
        self.assertIn('js', call_kwargs)
        self.assertEqual(call_kwargs['js'], "(el) => { return true; }")

    @patch('qwen3_tts.interface.ui._create_audio_reset_js')
    def test_wire_generation_tab_without_api_name(self, mock_reset_js):
        """Should work without api_name (design/custom tabs)."""
        from qwen3_tts.interface.ui import _wire_generation_tab

        mock_reset_js.return_value = "(el) => { return true; }"
        mock_handler = Mock(return_value=(None, "status", "html", [], []))

        _wire_generation_tab(
            mode="design",
            btn=self.mock_btn,
            cancel_btn=self.mock_cancel_btn,
            output=self.mock_output,
            status=self.mock_status,
            model_indicator=self.mock_indicator,
            text=self.mock_text,
            text_info=self.mock_info,
            inputs_list=[self.mock_text],
            status_html=self.mock_html,
            history_df=self.mock_df,
            handler=mock_handler
            # No api_name
        )

        # Should still include js
        call_kwargs = self.mock_btn.click.call_args[1]
        self.assertIn('js', call_kwargs)


if __name__ == "__main__":
    unittest.main()
