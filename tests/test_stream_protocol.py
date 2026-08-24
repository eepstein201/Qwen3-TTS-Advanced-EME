"""Tests for the shared streaming wire-format parser (core/stream_protocol.py).

The parser used to exist TWICE — in ``interface/generate_server.py`` (CLI) and
``server/client/generator.py`` (``TTSClient``) — and the copies diverged. Only
the CLI checked the terminal error sentinel, so the client decoded a JSON error
payload as float32 and yielded garbage samples with ``sr=0``; and the two
carried different size caps (200 MB vs 100 MB).

These tests pin the single implementation AND the fact that it is single: a
future re-fork would have to defeat the no-second-parser assertions below.

Run: python -m unittest tests.test_stream_protocol -v
"""

import struct
import unittest

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

_skip = unittest.skipUnless(HAS_NUMPY, "requires numpy")

from qwen3_tts.core.stream_protocol import (  # noqa: E402
    MAX_STREAM_CHUNK_BYTES,
    STREAM_ERROR_SENTINEL_SR,
    StreamProtocolError,
    decode_stream_error_payload,
    encode_stream_error_frame,
    iter_stream_chunks,
)


def _audio_frame(samples, sr=24000):
    payload = np.asarray(samples, dtype=np.float32).tobytes()
    return struct.pack("<II", sr, len(payload)) + payload


@_skip
class TestIterStreamChunks(unittest.TestCase):
    """Frame parsing, including boundaries the old copies handled differently."""

    def test_single_frame_round_trips(self):
        chunks = list(iter_stream_chunks([_audio_frame([0.1, 0.2, 0.3])]))
        self.assertEqual(len(chunks), 1)
        samples, sr = chunks[0]
        self.assertEqual(sr, 24000)
        np.testing.assert_array_almost_equal(samples, [0.1, 0.2, 0.3])

    def test_multiple_frames_in_one_block(self):
        body = _audio_frame([1.0, 2.0]) + _audio_frame([3.0])
        chunks = list(iter_stream_chunks([body]))
        self.assertEqual([len(c[0]) for c in chunks], [2, 1])

    def test_frame_split_across_block_boundaries(self):
        """Network blocks do not align with frames — the header itself can split.

        This is the case a naive per-block parser gets wrong: the buffer must
        carry a partial header across iterations.
        """
        body = _audio_frame([1.0, 2.0, 3.0])
        blocks = [body[:3], body[3:9], body[9:]]  # splits mid-header and mid-payload
        chunks = list(iter_stream_chunks(blocks))
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_almost_equal(chunks[0][0], [1.0, 2.0, 3.0])

    def test_empty_stream_yields_nothing(self):
        self.assertEqual(list(iter_stream_chunks([])), [])

    def test_trailing_partial_frame_is_dropped_not_yielded(self):
        """A truncated final frame must not be emitted as short audio."""
        body = _audio_frame([1.0, 2.0])
        chunks = list(iter_stream_chunks([body[:-4]]))
        self.assertEqual(chunks, [])


@_skip
class TestTerminalErrorFrame(unittest.TestCase):
    """sample_rate 0 marks an in-band failure, not audio."""

    def test_sentinel_frame_raises_with_the_server_message(self):
        with self.assertRaises(StreamProtocolError) as ctx:
            list(iter_stream_chunks([encode_stream_error_frame("model exploded")]))
        self.assertIn("model exploded", str(ctx.exception))

    def test_audio_before_the_sentinel_is_yielded_then_the_error_raises(self):
        """Partial audio must reach the consumer, but must not end cleanly."""
        body = _audio_frame([1.0, 2.0]) + encode_stream_error_frame("died")

        received = []
        with self.assertRaises(StreamProtocolError):
            for chunk in iter_stream_chunks([body]):
                received.append(chunk)

        self.assertEqual(len(received), 1)
        self.assertNotEqual(received[0][1], 0)

    def test_corrupt_error_payload_still_raises(self):
        """A payload that isn't valid JSON must not fall through to success."""
        bad = struct.pack("<II", STREAM_ERROR_SENTINEL_SR, 3) + b"\xff\xfe\xfd"
        with self.assertRaises(StreamProtocolError):
            list(iter_stream_chunks([bad]))

    def test_decode_helper_never_raises(self):
        self.assertIsInstance(decode_stream_error_payload(b"\xff\xfe"), str)
        self.assertIsInstance(decode_stream_error_payload(b"[]"), str)
        self.assertEqual(decode_stream_error_payload(b'{"error": "x"}'), "x")

    def test_sentinel_is_zero(self):
        """What makes the sentinel unambiguous is that no real chunk uses 0.

        If 0 ever becomes a legal sample rate this framing breaks silently.
        """
        self.assertEqual(STREAM_ERROR_SENTINEL_SR, 0)


@_skip
class TestMalformedFrames(unittest.TestCase):
    """Caps and decode failures — the length prefix is attacker-influenceable."""

    def test_oversized_declared_length_is_rejected_without_buffering(self):
        header = struct.pack("<II", 24000, MAX_STREAM_CHUNK_BYTES + 1)
        with self.assertRaises(RuntimeError) as ctx:
            list(iter_stream_chunks([header]))
        message = str(ctx.exception).lower()
        self.assertIn("exceed", message)
        # Both historical consumers' tests key on these words; keeping them
        # means neither had to change when the parsers merged.
        self.assertIn("buffer", message)

    def test_buffer_growth_is_capped(self):
        """A stream that never completes a frame must not grow without bound."""
        # Header declares a legal size, but the payload never arrives; the
        # accumulated buffer is what has to be capped.
        header = struct.pack("<II", 24000, MAX_STREAM_CHUNK_BYTES)
        filler = b"\x00" * (1024 * 1024)
        blocks = [header] + [filler] * (MAX_STREAM_CHUNK_BYTES // len(filler) + 2)
        with self.assertRaises(RuntimeError) as ctx:
            list(iter_stream_chunks(blocks))
        self.assertIn("exceed", str(ctx.exception).lower())

    def test_payload_not_a_whole_number_of_float32s(self):
        frame = struct.pack("<II", 24000, 6) + b"\x00" * 6
        with self.assertRaises(RuntimeError) as ctx:
            list(iter_stream_chunks([frame]))
        self.assertIn("parse", str(ctx.exception).lower())
        self.assertIsInstance(ctx.exception.__cause__, ValueError)


class TestSingleImplementation(unittest.TestCase):
    """Guard against the parser being re-forked.

    Every consumer must reach the shared module; a second local copy is what
    let the sentinel check exist in one place and not the other.
    """

    def _read(self, rel_path):
        import os

        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, rel_path)) as f:
            return f.read()

    def test_cli_and_client_share_one_cap(self):
        from qwen3_tts.interface import generate_server
        from qwen3_tts.server.client import _base

        self.assertEqual(generate_server.MAX_STREAM_CHUNK_BYTES, MAX_STREAM_CHUNK_BYTES)
        self.assertEqual(_base.MAX_BUFFER_SIZE, MAX_STREAM_CHUNK_BYTES)

    def test_server_cli_and_client_share_one_sentinel(self):
        from qwen3_tts.interface import generate_server
        from qwen3_tts.server import app_generation

        self.assertEqual(
            generate_server.STREAM_ERROR_SENTINEL_SR, STREAM_ERROR_SENTINEL_SR
        )
        self.assertEqual(
            app_generation.STREAM_ERROR_SENTINEL_SR, STREAM_ERROR_SENTINEL_SR
        )

    def test_consumers_call_the_shared_parser(self):
        for path in (
            "qwen3_tts/interface/generate_server.py",
            "qwen3_tts/server/client/generator.py",
        ):
            with self.subTest(module=path):
                src = self._read(path)
                self.assertIn(
                    "iter_stream_chunks",
                    src,
                    f"{path} does not use the shared frame parser",
                )
                self.assertNotIn(
                    'np.frombuffer(audio_bytes, dtype="<f4")',
                    src,
                    f"{path} appears to have re-forked the frame parser",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
