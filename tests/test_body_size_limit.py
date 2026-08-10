"""Tests for the ASGI request-body-size cap (roadmap R-30).

Oversized bodies must be rejected by middleware BEFORE the body is read into
memory / parsed by Pydantic. The middleware enforces the cap two ways:
  1. Fast-path: Content-Length header present and over limit → immediate 413.
  2. Streaming path: actual bytes counted regardless of Content-Length presence
     (catches chunked or header-free transfers).
"""

import unittest
from unittest.mock import patch

from tests.voice_test_helpers import _make_test_client, _skip_server


@_skip_server
class TestBodySizeLimit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qwen3_tts.server.app import app

        cls.app = app
        cls.client = _make_test_client(
            app, server_config={"security": {}, "auto_shutdown_minutes": 0}
        )

    def test_oversized_content_length_returns_413(self):
        """Fast-path: declared Content-Length over cap is rejected with 413."""
        with patch("qwen3_tts.server.app.MAX_REQUEST_BODY_BYTES", 16):
            resp = self.client.post(
                "/generate",
                content=b"x" * 64,  # 64 bytes > 16-byte cap; httpx adds Content-Length
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(resp.status_code, 413)

    def test_missing_content_length_oversized_returns_413(self):
        """Streaming path: body over cap without Content-Length returns 413."""

        def _large_chunks():
            # Yield chunks that sum to more than the patched cap (16 bytes)
            for _ in range(5):
                yield b"xxxxx"  # 5 * 5 = 25 bytes > 16-byte cap

        with patch("qwen3_tts.server.app.MAX_REQUEST_BODY_BYTES", 16):
            resp = self.client.post(
                "/generate",
                content=_large_chunks(),
                headers={"Content-Type": "application/json"},
                # httpx omits Content-Length for generator streams
            )
        self.assertEqual(resp.status_code, 413)

    def test_missing_content_length_small_body_passes(self):
        """Streaming path: small body without Content-Length is not blocked."""

        def _small_chunks():
            yield b'{"texts": ["hi"], "mode": "clone"}'

        resp = self.client.post(
            "/generate",
            content=_small_chunks(),
            headers={"Content-Type": "application/json"},
        )
        # May be rejected by auth/validation, but NOT by the size cap.
        self.assertNotEqual(resp.status_code, 413)

    def test_normal_request_passes_through(self):
        """A small JSON body is not blocked by the size cap."""
        resp = self.client.post(
            "/generate",
            json={"texts": ["hi"], "mode": "clone"},
        )
        self.assertNotEqual(resp.status_code, 413)

    def test_cap_is_named_constant_above_audio_limit(self):
        """The cap is a named constant sized above the largest legit payload."""
        from qwen3_tts.server.app import MAX_REQUEST_BODY_BYTES
        from qwen3_tts.server.validation import MAX_AUDIO_BASE64_BYTES

        self.assertIsInstance(MAX_REQUEST_BODY_BYTES, int)
        self.assertGreater(MAX_REQUEST_BODY_BYTES, MAX_AUDIO_BASE64_BYTES)


@_skip_server
class TestBodySizeLimitNoBuffer(unittest.IsolatedAsyncioTestCase):
    """P6: the body-size cap must reject oversized bodies WITHOUT buffering.

    The prior middleware did ``request._body = b"".join(chunks)``, buffering up
    to MAX_REQUEST_BODY_BYTES of an UNAUTHENTICATED request before the 401 — a
    pre-auth memory-DoS (many large chunked bodies, no token). The ASGI
    middleware counts bytes off the receive callable and aborts as soon as the
    cap is exceeded, never buffering the full body.
    """

    async def test_oversized_chunked_body_rejected_without_buffering(self):
        from qwen3_tts.server.app import RequestBodySizeLimitMiddleware

        cap = 16
        chunk = b"x" * 10
        total_chunks = 10  # 100 bytes >> 16-byte cap

        pulled = 0

        async def fake_receive():
            nonlocal pulled
            if pulled >= total_chunks:
                return {"type": "http.disconnect"}
            pulled += 1
            return {
                "type": "http.request",
                "body": chunk,
                "more_body": pulled < total_chunks,
            }

        sent = []

        async def fake_send(message):
            sent.append(message)

        # An app that tries to buffer the ENTIRE body (the old failure mode).
        async def buffering_app(scope, receive, send):
            body = b""
            while True:
                m = await receive()
                if m["type"] == "http.request":
                    body += m.get("body", b"")
                    if not m.get("more_body"):
                        break
                else:
                    break
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        scope = {"type": "http", "headers": [], "method": "POST", "path": "/"}

        from starlette.exceptions import HTTPException as StarletteHTTPException

        with patch("qwen3_tts.server.app.MAX_REQUEST_BODY_BYTES", cap):
            mw = RequestBodySizeLimitMiddleware(buffering_app)
            # The bare test app has no ExceptionMiddleware, so the 413 surfaces
            # as a raised StarletteHTTPException (in the real app the inner
            # ExceptionMiddleware converts it to the 413 response, exercised by
            # TestBodySizeLimit.test_missing_content_length_oversized above).
            with self.assertRaises(StarletteHTTPException):
                await mw(scope, fake_receive, fake_send)

        # Cap (16) is exceeded during the 2nd 10-byte chunk (20 bytes). The
        # middleware must abort there — NOT drain all 10 chunks (which would
        # buffer the full oversized body, the bug this replaces).
        self.assertLess(
            pulled,
            total_chunks,
            f"middleware buffered the whole oversized body (pulled {pulled}/{total_chunks} chunks)",
        )


if __name__ == "__main__":
    unittest.main()
