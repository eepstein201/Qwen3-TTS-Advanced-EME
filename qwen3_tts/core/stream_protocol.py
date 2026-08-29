"""Wire format for length-prefixed streamed audio (`/generate-stream`, `/ws`).

ONE parser, shared by every consumer. Frames are::

    [sample_rate: uint32 LE][length: uint32 LE][payload: length bytes]

A real audio chunk always carries a non-zero sample rate, so ``sample_rate == 0``
is a free sentinel marking a **terminal error frame** whose payload is JSON
``{"error": str, "code": str}``. Starlette commits the HTTP 200 headers before
the body is iterated, so once streaming begins the server cannot signal failure
with a status code; the sentinel frame is how a mid-stream failure reaches the
client at all.

This module exists because that parser was implemented **twice** — in
``interface/generate_server.py`` (CLI) and ``server/client/generator.py``
(``TTSClient``) — and the copies diverged: only the CLI checked the sentinel, so
``TTSClient.generate_streaming()`` decoded the JSON error payload as float32 and
yielded garbage samples with ``sr=0`` instead of raising. The two also carried
different size caps (200 MB vs 100 MB).

No FastAPI, torch, or mlx imports — the CLI must never pull the server stack in,
which is why the sentinel constant used to be duplicated rather than imported.
numpy is imported lazily inside the parser for the same reason.
"""

import json
import struct

# 4-byte sample rate + 4-byte payload length.
STREAM_HEADER_SIZE = 8

# A real chunk's sample rate is never 0, which is what makes the sentinel
# unambiguous. If 0 ever becomes a legal sample rate this framing breaks
# silently — tests/test_stream_error_frame.py asserts the value explicitly.
STREAM_ERROR_SENTINEL_SR = 0

STREAM_ERROR_CODE_INFERENCE_FAILED = "inference_failed"

# Upper bound on a single frame's declared byte length AND on the accumulated
# read buffer. The length prefix is an attacker-influenceable uint32 (up to
# ~4 GB); without a cap a corrupt or hostile stream makes the reader wait for
# and buffer an unbounded amount of data. 100 MB of float32 at 24 kHz is ~17
# minutes of audio in a SINGLE chunk, against a real chunk of ~2 s — so this is
# far above any legitimate frame. The CLI copy previously allowed 200 MB; the
# stricter of the two divergent values wins.
MAX_STREAM_CHUNK_BYTES = 100 * 1024 * 1024


class StreamProtocolError(RuntimeError):
    """The server reported a failure in band via a terminal error frame.

    Distinct from a malformed/oversized frame (plain ``RuntimeError``): this
    means the stream was well-formed and the server is telling us the
    generation failed. Callers re-raise it in their own idiom — the CLI as
    ``TTSGenericError``, ``TTSClient`` as ``GenerationError``.
    """


def encode_stream_error_frame(
    message: str, code: str = STREAM_ERROR_CODE_INFERENCE_FAILED
) -> bytes:
    """Build a terminal error frame for the streaming wire format."""
    payload = json.dumps({"error": message, "code": code}).encode("utf-8")
    return struct.pack("<II", STREAM_ERROR_SENTINEL_SR, len(payload)) + payload


def decode_stream_error_payload(payload: bytes) -> str:
    """Extract the human-readable message from a terminal frame's payload.

    Never raises: a corrupt payload must still surface as an error rather than
    falling through to "success".
    """
    try:
        detail = json.loads(payload.decode("utf-8"))
        message = detail.get("error")
    except (ValueError, UnicodeDecodeError, AttributeError):
        return "server reported a streaming error"
    return message or "unknown streaming error"


def iter_stream_chunks(byte_iter):
    """Parse a byte iterator into ``(samples, sample_rate)`` audio chunks.

    Args:
        byte_iter: iterable of raw byte blocks, e.g. ``resp.iter_content(...)``.
            Block boundaries need not align with frame boundaries.

    Yields:
        ``(numpy.ndarray of float32, sample_rate)`` per complete audio frame.

    Raises:
        StreamProtocolError: a terminal error frame was received. Any audio
            already yielded must NOT be treated as a complete generation.
        RuntimeError: the stream is malformed — an oversized declared length,
            a buffer grown past the cap, or payload bytes that are not a whole
            number of float32 samples.
    """
    import numpy as np

    buffer = b""

    for block in byte_iter:
        buffer += block

        # Guard the accumulated buffer, not just the declared length: a stream
        # that never completes a frame would otherwise grow without bound.
        if len(buffer) > MAX_STREAM_CHUNK_BYTES:
            raise RuntimeError(
                f"Streaming buffer exceeded the maximum size "
                f"({MAX_STREAM_CHUNK_BYTES} bytes). "
                "Possible malformed response from server."
            )

        while len(buffer) >= STREAM_HEADER_SIZE:
            try:
                sr, payload_len = struct.unpack(
                    "<II", buffer[:STREAM_HEADER_SIZE]
                )
            except struct.error as e:  # pragma: no cover — length checked above
                raise RuntimeError(
                    f"Failed to parse streamed audio chunk: {e}"
                ) from e

            if payload_len > MAX_STREAM_CHUNK_BYTES:
                raise RuntimeError(
                    f"Streamed audio chunk length ({payload_len} bytes) "
                    f"exceeds the {MAX_STREAM_CHUNK_BYTES}-byte streaming "
                    "buffer limit"
                )

            total = STREAM_HEADER_SIZE + payload_len
            if len(buffer) < total:
                break  # need more data

            payload = buffer[STREAM_HEADER_SIZE:total]
            buffer = buffer[total:]

            # Terminal error frame — surface it instead of decoding JSON as
            # float32 samples.
            if sr == STREAM_ERROR_SENTINEL_SR:
                raise StreamProtocolError(decode_stream_error_payload(payload))

            try:
                samples = np.frombuffer(payload, dtype="<f4")
            except ValueError as e:
                raise RuntimeError(
                    f"Failed to parse streamed audio chunk: {e}"
                ) from e

            yield samples, sr
