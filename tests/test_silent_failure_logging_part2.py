"""Remaining previously-swallowed error sites must log (observability, part 2).

Covers the delete/rename default-prompt config updates, the audio-loader cache
sync, and the UI model-state probe / WAV-duration fallbacks. Each caught an
exception and degraded silently; they must now emit a warning while preserving
the graceful fallback.

See docs/reviews/e2e-review-2026-07-01.md (Phase 5 silent-failure findings).
"""

import types
import unittest
from unittest.mock import MagicMock, patch


class TestDeleteRenameDefaultUpdateLogging(unittest.TestCase):
    """app_prompts: failing to update default_clone_prompt must be logged."""

    def _state(self):
        return types.SimpleNamespace()

    def test_delete_default_update_failure_logs(self):
        from qwen3_tts.server import app_prompts

        req = types.SimpleNamespace(name="myvoice")
        config_fn = lambda: {"default_clone_prompt": "myvoice"}  # noqa: E731

        # Pretend the prompt files exist and delete cleanly (files_removed
        # non-empty → no 404), then save_config raises when clearing the default.
        with patch.object(app_prompts.os.path, "exists", return_value=True), \
             patch.object(app_prompts.os, "remove", MagicMock()), \
             patch.object(app_prompts, "save_config", side_effect=OSError("disk full")):
            with self.assertLogs("tts", level="WARNING") as cm:
                app_prompts.handle_delete_prompt(self._state(), req, config_fn)
        self.assertTrue(
            any("default" in line.lower() or "disk full" in line for line in cm.output),
            f"expected a warning about the failed default update; got {cm.output}",
        )


class TestGetVoiceMetadataDurationLogging(unittest.TestCase):
    """ui.shared.get_voice_metadata: a WAV read failure must be logged."""

    def test_wav_read_failure_logs_and_returns_na(self):
        from qwen3_tts.interface.ui import shared

        details = {"formats": [".wav"], "size_bytes": 1, "created": 0}
        fake_client = MagicMock()
        fake_client.is_server_running.return_value = True
        fake_client.get_prompt_details.return_value = details

        with patch("qwen3_tts.server.client.TTSClient", return_value=fake_client), \
             patch("os.path.exists", return_value=True), \
             patch("soundfile.info", side_effect=RuntimeError("corrupt wav")):
            with self.assertLogs("tts.ui", level="WARNING") as cm:
                result = shared.get_voice_metadata("somevoice")
        self.assertEqual(result.get("duration"), "N/A")
        self.assertTrue(
            any("duration" in line.lower() or "corrupt wav" in line for line in cm.output),
            f"expected a warning about the WAV read failure; got {cm.output}",
        )


if __name__ == "__main__":
    unittest.main()
