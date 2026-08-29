"""WS2 Task 2.5 — in-band terminal error frame for streamed generation.

Starlette commits the HTTP 200 status headers before the response body is
iterated, so once streaming starts the server cannot signal failure with a
status code. Raising inside the generator merely truncates the connection, which
a client cannot distinguish from a network drop and which carries no error
detail. Worse, before this change a failure *after* the first chunk was dropped
entirely: the stream ended cleanly and the client saved truncated audio as a
successful generation.

The wire format is [sample_rate:4][length:4][payload:length]. A real audio chunk
always carries a non-zero sample rate, so sample_rate 0 is a free sentinel
marking a terminal frame whose payload is JSON {"error", "code"}.
"""

import json
import struct
import unittest
from unittest.mock import MagicMock, patch

from qwen3_tts.interface import generate_server
from qwen3_tts.server import app_generation


class TestErrorFrameEncoding(unittest.TestCase):
    def test_frame_is_parseable_and_carries_the_message(self):
        frame = app_generation.encode_stream_error_frame("boom")
        sr, length = struct.unpack("<II", frame[:8])
        self.assertEqual(sr, app_generation.STREAM_ERROR_SENTINEL_SR)
        self.assertEqual(length, len(frame) - 8)
        payload = json.loads(frame[8:].decode("utf-8"))
        self.assertEqual(payload["error"], "boom")
        self.assertEqual(payload["code"], "inference_failed")

    def test_sentinel_cannot_collide_with_real_audio(self):
        """A real chunk's sample rate is never 0, which is what makes the
        sentinel unambiguous. If someone ever makes 0 a legal sample rate, this
        framing breaks silently — hence the explicit assertion."""
        self.assertEqual(app_generation.STREAM_ERROR_SENTINEL_SR, 0)

    def test_client_and_server_sentinels_match(self):
        """The constant is duplicated across the wire boundary (the CLI must not
        import FastAPI), so it needs a drift guard."""
        self.assertEqual(
            generate_server.STREAM_ERROR_SENTINEL_SR,
            app_generation.STREAM_ERROR_SENTINEL_SR,
        )


class TestClientSurfacesErrorFrame(unittest.TestCase):
    """The client must raise on a terminal frame, not decode JSON as float32."""

    def _run_stream(self, frames):
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_content = MagicMock(return_value=iter(frames))

        # server_request is imported lazily inside generate_streaming, so the
        # patch has to target its definition site, not this module's namespace.
        with (
            patch("qwen3_tts.core.http_client.server_request", return_value=resp),
            patch.object(generate_server, "play_audio"),
        ):
            return generate_server.generate_streaming(
                text="hello",
                mode="custom",
                config={},
                gen_params={},
                output_path="/tmp/out.wav",  # nosec B108
            )

    def test_client_raises_on_terminal_error_frame(self):
        audio = struct.pack("<II", 24000, 8) + b"\x00" * 8
        frames = [audio + app_generation.encode_stream_error_frame("model exploded")]

        with self.assertRaises(generate_server.TTSGenericError) as ctx:
            self._run_stream(frames)

        self.assertIn("model exploded", str(ctx.exception))

    def test_malformed_error_payload_still_raises(self):
        """A corrupt payload must not fall through to "success"."""
        bad = struct.pack("<II", 0, 3) + b"\xff\xfe\xfd"

        with self.assertRaises(generate_server.TTSGenericError):
            self._run_stream([bad])


class TestStreamThreadJoinTimeoutScales(unittest.TestCase):
    """The join must cover ONE chunk, so it has to track max_chunk_chars.

    A constant sized for the 500-char default expires mid-generation once the
    limit is raised, and the caller then releases inference_lock while the model
    is still on the GPU — the race the join exists to prevent.
    """

    def test_default_chunk_size_covers_one_chunk(self):
        """500 chars * 0.25 s = 125 s, comfortably above the old fixed 90 s."""
        self.assertEqual(
            app_generation._stream_thread_join_timeout(10_000, 500),
            500 * app_generation._STREAM_SECONDS_PER_CHAR,
        )

    def test_raising_max_chunk_chars_raises_the_timeout(self):
        small = app_generation._stream_thread_join_timeout(50_000, 500)
        large = app_generation._stream_thread_join_timeout(50_000, 5_000)
        self.assertGreater(large, small)
        self.assertGreaterEqual(large, 5_000 * app_generation._STREAM_SECONDS_PER_CHAR)

    def test_chunking_disabled_scales_with_whole_text(self):
        """0 disables chunking, so one call generates the entire text."""
        self.assertGreaterEqual(
            app_generation._stream_thread_join_timeout(20_000, 0),
            20_000 * app_generation._STREAM_SECONDS_PER_CHAR,
        )

    def test_unspecified_chunk_size_is_deliberately_over_generous(self):
        """None means "read config" (default 500), NOT "chunking disabled".

        Both land on the whole-text bound, but for different reasons, and the
        docstring used to conflate them. Keep them as separate cases: this one
        is over-generous on purpose, because the only dangerous error for this
        join is a bound that is too SHORT — that releases inference_lock while
        the model is still generating.
        """
        self.assertGreaterEqual(
            app_generation._stream_thread_join_timeout(20_000, None),
            20_000 * app_generation._STREAM_SECONDS_PER_CHAR,
        )

    def test_never_below_the_floor_for_short_text(self):
        self.assertEqual(
            app_generation._stream_thread_join_timeout(5, 500),
            app_generation._STREAM_THREAD_JOIN_FLOOR_SEC,
        )


if __name__ == "__main__":
    unittest.main()
