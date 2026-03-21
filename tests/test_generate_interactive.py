"""Tests for generate_interactive module.

Covers _ProgressPoller, delete_voice_prompt, rename_voice_prompt,
and preview_voice_prompt with all external dependencies mocked.
"""

import unittest
from unittest.mock import patch, MagicMock


class TestProgressPollerInit(unittest.TestCase):
    """Tests for _ProgressPoller constructor and lifecycle."""

    def test_constructor_sets_attributes(self):
        """Constructor stores server_url, batch_total, and initializes state."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller

        poller = _ProgressPoller("http://localhost:5123", batch_total=3)

        self.assertEqual(poller.server_url, "http://localhost:5123")
        self.assertEqual(poller.batch_total, 3)
        self.assertIsNone(poller._thread)
        self.assertFalse(poller._stop.is_set())

    def test_has_rich_attribute_exists(self):
        """HAS_RICH class attribute is a boolean."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller

        self.assertIsInstance(_ProgressPoller.HAS_RICH, bool)

    @patch("threading.Thread")
    def test_start_creates_daemon_thread(self, mock_thread_cls):
        """start() creates a daemon thread targeting _run and starts it."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller

        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        poller = _ProgressPoller("http://localhost:5123")
        poller.start()

        mock_thread_cls.assert_called_once_with(target=poller._run, daemon=True)
        mock_thread.start.assert_called_once()

    def test_stop_sets_event_and_joins_thread(self):
        """stop() sets the stop event and joins the thread."""
        from qwen3_tts.interface.generate_interactive import _ProgressPoller

        poller = _ProgressPoller("http://localhost:5123")
        mock_thread = MagicMock()
        poller._thread = mock_thread
        poller._rich_progress = None

        poller.stop()

        self.assertTrue(poller._stop.is_set())
        mock_thread.join.assert_called_once_with(timeout=2)


class TestDeleteVoicePrompt(unittest.TestCase):
    """Tests for delete_voice_prompt path traversal and not-found cases."""

    def test_path_traversal_rejected_dotdot(self):
        """Names containing '..' are rejected."""
        from qwen3_tts.interface.generate_interactive import delete_voice_prompt

        with patch("builtins.print") as mock_print:
            result = delete_voice_prompt("../evil")

        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn("Invalid prompt name", mock_print.call_args[0][0])

    def test_path_traversal_rejected_slash(self):
        """Names containing '/' are rejected."""
        from qwen3_tts.interface.generate_interactive import delete_voice_prompt

        with patch("builtins.print") as mock_print:
            result = delete_voice_prompt("sub/dir")

        self.assertFalse(result)
        self.assertIn("Invalid prompt name", mock_print.call_args[0][0])

    def test_prompt_not_found(self):
        """Returns False and prints error when prompt files do not exist."""
        from qwen3_tts.interface.generate_interactive import delete_voice_prompt

        with patch("os.path.exists", return_value=False), \
             patch("builtins.print") as mock_print:
            result = delete_voice_prompt("nonexistent")

        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn("not found", mock_print.call_args[0][0])


class TestRenameVoicePrompt(unittest.TestCase):
    """Tests for rename_voice_prompt path traversal and not-found cases."""

    def test_path_traversal_rejected(self):
        """Names containing '..' are rejected for both old and new names."""
        from qwen3_tts.interface.generate_interactive import rename_voice_prompt

        with patch("builtins.print") as mock_print:
            result = rename_voice_prompt("good_name", "../../evil")

        self.assertFalse(result)
        self.assertIn("Invalid prompt name", mock_print.call_args[0][0])

    def test_old_prompt_not_found(self):
        """Returns False and prints error when old prompt does not exist."""
        from qwen3_tts.interface.generate_interactive import rename_voice_prompt

        with patch("os.path.exists", return_value=False), \
             patch("builtins.print") as mock_print:
            result = rename_voice_prompt("missing_voice", "new_voice")

        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn("not found", mock_print.call_args[0][0])


class TestPreviewVoicePrompt(unittest.TestCase):
    """Tests for preview_voice_prompt not-found and server-not-running cases."""

    @patch("qwen3_tts.interface.generate_interactive.voice_prompt_exists",
           return_value=False)
    def test_prompt_not_found(self, _mock_exists):
        """Returns False and prints error when prompt does not exist."""
        from qwen3_tts.interface.generate_interactive import preview_voice_prompt

        with patch("builtins.print") as mock_print:
            result = preview_voice_prompt("nonexistent", {})

        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn("not found", mock_print.call_args[0][0])

    @patch("qwen3_tts.interface.generate_interactive.voice_prompt_exists",
           return_value=True)
    @patch("qwen3_tts.interface.generate_interactive.is_server_running",
           return_value=False)
    def test_server_not_running(self, _mock_server, _mock_exists):
        """Returns False and prints error when server is not running."""
        from qwen3_tts.interface.generate_interactive import preview_voice_prompt

        with patch("builtins.print") as mock_print:
            result = preview_voice_prompt("my_voice", {})

        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn("server must be running", mock_print.call_args[0][0].lower())


if __name__ == "__main__":
    unittest.main()
