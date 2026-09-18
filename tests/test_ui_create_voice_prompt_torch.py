#!/usr/bin/env python3
"""The Gradio UI's torch branch of create_voice_prompt (Step 4B.2 item 10).

The MLX branch routes through the engine writer and is covered by the 0G work;
the torch branch is the UI's own code: it requires a running server, base64-
uploads the audio to /create-voice-prompt, and maps a non-200 onto gr.Error.

Every collaborator is patched at its definition site — `server_request` is
imported inside the function body, so it is patched on
``qwen3_tts.core.http_client``, not on the UI module.

Run: python -m pytest tests/test_ui_create_voice_prompt_torch.py -v
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tests.voice_test_helpers import _skip_ui

_TORCH_CONFIG = {"advanced": {"backend": "torch"}}


@_skip_ui
class TestCreateVoicePromptTorchBranch(unittest.TestCase):
    """create_voice_prompt() with advanced.backend = "torch"."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio_path = os.path.join(self.tmp.name, "sample.wav")
        with open(self.audio_path, "wb") as f:
            f.write(b"RIFFfake-audio-bytes")

    def tearDown(self):
        self.tmp.cleanup()

    def _create(
        self,
        *,
        server_running=True,
        status_code=200,
        error_body=None,
        transcript="  hello there  ",
        no_transcript=False,
        voice_name="newvoice",
    ):
        """Drive the torch branch; return (result, server_request mock)."""
        from qwen3_tts.interface.ui import voice_management as vm

        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = error_body or {}

        with (
            patch.object(vm, "VOICE_PROMPTS_DIR", self.tmp.name),
            patch.object(vm, "load_config", return_value=_TORCH_CONFIG),
            patch.object(vm, "is_server_running", return_value=server_running),
            patch.object(vm, "get_voice_prompts", return_value=["newvoice.pt"]),
            patch.object(vm, "get_default_clone_prompt", return_value="newvoice.pt"),
            patch(
                "qwen3_tts.core.http_client.server_request", return_value=resp
            ) as mock_request,
        ):
            result = vm.create_voice_prompt(
                self.audio_path,
                transcript,
                voice_name,
                no_transcript=no_transcript,
            )
        return result, mock_request

    def test_server_must_be_running(self):
        """Without a server the UI refuses instead of failing obscurely."""
        import gradio as gr

        with self.assertRaises(gr.Error) as ctx:
            self._create(server_running=False)

        self.assertIn("Server must be running", str(ctx.exception))

    def test_server_not_running_never_sends_a_request(self):
        """The guard short-circuits before any upload is attempted."""
        import gradio as gr

        from qwen3_tts.interface.ui import voice_management as vm

        with (
            patch.object(vm, "VOICE_PROMPTS_DIR", self.tmp.name),
            patch.object(vm, "load_config", return_value=_TORCH_CONFIG),
            patch.object(vm, "is_server_running", return_value=False),
            patch("qwen3_tts.core.http_client.server_request") as mock_request,
        ):
            with self.assertRaises(gr.Error):
                vm.create_voice_prompt(self.audio_path, "hi", "newvoice")

        mock_request.assert_not_called()

    def test_successful_create_reports_the_torch_path(self):
        """A 200 returns the torch status line and the refreshed prompt list."""
        (status, choices_update, default_update), _ = self._create()

        self.assertIn("torch", status)
        self.assertIn("newvoice", status)
        self.assertEqual(choices_update["choices"], ["newvoice.pt"])
        self.assertEqual(default_update["value"], "newvoice.pt")

    def test_request_targets_the_create_endpoint_with_the_shared_timeout(self):
        """The upload uses CREATE_PROMPT_TIMEOUT_SEC, not a short constant.

        The server serializes prompt creation on inference_lock (#192), so the
        request can queue for minutes behind an in-flight generation; a 60 s
        timeout would abandon a create the server goes on to complete.
        """
        from qwen3_tts.core.http_client import CREATE_PROMPT_TIMEOUT_SEC

        _, mock_request = self._create()

        args, kwargs = mock_request.call_args
        self.assertEqual(args[0], "POST")
        self.assertEqual(args[1], "/create-voice-prompt")
        self.assertEqual(kwargs["timeout"], CREATE_PROMPT_TIMEOUT_SEC)

    def test_payload_carries_base64_audio_and_stripped_transcript(self):
        """The audio file is base64-encoded verbatim; the transcript trimmed."""
        import base64

        _, mock_request = self._create(transcript="  hello there  ")

        payload = mock_request.call_args[1]["json"]
        with open(self.audio_path, "rb") as f:
            expected = base64.b64encode(f.read()).decode()
        self.assertEqual(payload["audio_base64"], expected)
        self.assertEqual(payload["transcript"], "hello there")
        self.assertEqual(payload["name"], "newvoice")
        self.assertFalse(payload["no_transcript"])

    def test_no_transcript_mode_sends_the_flag_and_an_empty_transcript(self):
        """Speaker-embedding-only uploads carry no_transcript=True."""
        _, mock_request = self._create(transcript=None, no_transcript=True)

        payload = mock_request.call_args[1]["json"]
        self.assertTrue(payload["no_transcript"])
        self.assertEqual(payload["transcript"], "")

    def test_server_error_surfaces_the_servers_message(self):
        """A non-200 becomes a gr.Error carrying the server's own reason."""
        import gradio as gr

        with self.assertRaises(gr.Error) as ctx:
            self._create(status_code=500, error_body={"error": "decode failed"})

        self.assertIn("decode failed", str(ctx.exception))

    def test_existing_pt_file_is_refused_before_any_upload(self):
        """An existing .pt short-circuits with 'already exists'."""
        import gradio as gr

        from qwen3_tts.interface.ui import voice_management as vm

        with open(os.path.join(self.tmp.name, "taken.pt"), "wb") as f:
            f.write(b"x")

        with (
            patch.object(vm, "VOICE_PROMPTS_DIR", self.tmp.name),
            patch.object(vm, "load_config", return_value=_TORCH_CONFIG),
            patch.object(vm, "is_server_running", return_value=True),
            patch("qwen3_tts.core.http_client.server_request") as mock_request,
        ):
            with self.assertRaises(gr.Error) as ctx:
                vm.create_voice_prompt(self.audio_path, "hi", "taken")

        self.assertIn("already exists", str(ctx.exception))
        mock_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
