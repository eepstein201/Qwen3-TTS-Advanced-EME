#!/usr/bin/env python3
"""E2E: /ws bidirectional streaming against the LIVE server.

Closes 4D gap 2: ``/ws`` had zero E2E coverage — the unit modules drive the
handler through starlette's TestClient, whose unbounded ``receive()`` has hung
CI before. Every receive here is bounded with an explicit timeout; a hang
fails the test instead of stalling the suite.

Wire contract under test (``qwen3_tts/server/websocket.py``):

  connect ws://127.0.0.1:5123/ws            (no Origin header → allowed)
    client sends {"token": …} (first frame, 10 s server-side window)
      → {"status": "authenticated"}          | bad token → {"error": …} + close 4001
    client sends {"text", "mode", "prompt_file", "seed", …}
      → {"status": "generating", "text_length": N}
      → N × binary frames [sample_rate:4 LE u32][length:4 LE u32][float32 LE]
      → terminal {"status": "complete"|"cancelled"|"error", "chunks": N, "seed": S}

Clone mode REQUIRES ``prompt_file`` (error frame before any generating frame)
and — unlike ``/generate`` (Step 2 on-demand load) — does NOT auto-load an
empty model slot: it replies ``{"error": "Model 'clone' not loaded"}``. The
default config loads clone at startup, so a freshly started server qualifies.

Mid-generation ``{"action": "cancel"}`` is read by the concurrent
cancel-watcher (the main loop is blocked inside ``_stream_generation``), so
there is NO ack frame: the cancel surfaces only as the terminal
``{"status": "cancelled", …}`` frame, and the socket must STAY OPEN
(classified outcomes never close — pinned in websocket.py). The inference
thread observes the stop flag at its next chunk yield, so the terminal frame
arrives no earlier than the first chunk boundary; expect ``chunks == 0`` when
the cancel lands inside the first chunk.

Prompt choice: first non-"e2e" entry from ``/prompts`` (suffixed name listed,
bare name addressed — same normalization as test_e2e_voice_management).

Prerequisites: live server on :5123 with clone loaded; ``websockets`` (ships
with uvicorn[standard]; sync client used — no asyncio harness needed).

Usage:  pytest tests/test_e2e_websocket.py -m e2e
"""

import contextlib
import json
import os
import struct
import unittest
import urllib.request

try:
    import pytest

    pytestmark = pytest.mark.e2e
except ImportError:
    pass

try:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

SERVER_URL = "http://127.0.0.1:5123"
WS_URL = "ws://127.0.0.1:5123/ws"
TOKEN_FILE = os.path.expanduser("~/.config/qwen3-tts/.voice_server_token")

# Time-to-first-chunk for clone on M2 Pro is tens of seconds (a ~50-char text
# is one chunk); ASR is NOT force-loaded on /ws (opportunistic trim only), so
# these bounds cover model-warm inference without hanging on a dead server.
AUTH_TIMEOUT = 10
STATUS_TIMEOUT = 30
FIRST_CHUNK_TIMEOUT = 180
CHUNK_TIMEOUT = 60
TERMINAL_TIMEOUT = 60

_TEXT = "Hello from the /ws end-to-end test."


def _read_token():
    with open(TOKEN_FILE) as f:
        return f.read().strip()


def _pick_prompt():
    """First non-'e2e' prompt, bare-named (server lists suffixed names)."""
    req = urllib.request.Request(
        f"{SERVER_URL}/prompts",
        headers={"Authorization": f"Bearer {_read_token()}"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        listed = json.loads(resp.read().decode())["prompts"]
    for name in listed:
        if not name.startswith("e2e"):
            return name.removesuffix(".wav").removesuffix(".pt")
    raise unittest.SkipTest("no non-e2e voice prompt available for clone mode")


def _authed_connect(token):
    """Connect + authenticate; returns the connection after the ack frame."""
    conn = connect(WS_URL, open_timeout=AUTH_TIMEOUT, close_timeout=10)
    conn.send(json.dumps({"token": token}))
    ack = json.loads(conn.recv(timeout=STATUS_TIMEOUT))
    if ack.get("status") != "authenticated":
        conn.close()
        raise AssertionError(f"auth ack was not 'authenticated': {ack}")
    return conn


def _recv_until_terminal(
    conn, first_timeout=FIRST_CHUNK_TIMEOUT, frame_timeout=CHUNK_TIMEOUT
):
    """Bounded receive loop → (control_frames, binary_frames, terminal_frame).

    Every recv has an explicit timeout (TestClient's unbounded receive is the
    documented hang). Raises AssertionError if no terminal frame arrives.
    """
    control, binary, terminal = [], [], None
    timeout = first_timeout
    while terminal is None:
        msg = conn.recv(timeout=timeout)
        timeout = frame_timeout
        if isinstance(msg, bytes):
            binary.append(msg)
        else:
            frame = json.loads(msg)
            if frame.get("status") in ("complete", "cancelled", "error"):
                terminal = frame
            else:
                control.append(frame)
    return control, binary, terminal


@unittest.skipUnless(HAS_WEBSOCKETS, "websockets not installed")
class TestE2EWebsocket(unittest.TestCase):
    """Live /ws: auth handshake, validation, full stream, mid-generation cancel."""

    @classmethod
    def setUpClass(cls):
        try:
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3) as r:
                health = json.loads(r.read().decode())
        except Exception as e:
            raise unittest.SkipTest("TTS server not running on port 5123") from e
        # /ws never auto-loads (no Step-2 path here) — skip loudly, not hollow.
        if not health.get("clone_model_loaded"):
            raise unittest.SkipTest(
                "clone model not loaded — /ws does not auto-load; "
                "restart the server (clone loads at startup by default)"
            )
        cls.token = _read_token()
        cls.prompt = _pick_prompt()

    # --- helpers ---------------------------------------------------------

    def _assert_binary_frame(self, frame):
        """One wire frame: [sr:4][len:4][float32 payload] — full contract."""
        self.assertGreaterEqual(
            len(frame), 8, f"binary frame shorter than header: {len(frame)}"
        )
        sr, length = struct.unpack("<II", frame[:8])
        payload = frame[8:]
        self.assertGreater(sr, 0, "sample_rate must be positive audio, not a sentinel")
        self.assertLessEqual(sr, 192_000, f"implausible sample rate: {sr}")
        self.assertEqual(length, len(payload), "length field must match payload size")
        self.assertEqual(length % 4, 0, f"payload not float32-aligned: {length}")

    # --- tests -----------------------------------------------------------

    def test_01_bad_token_is_rejected_with_4001(self):
        """Auth gate: wrong token → error frame, then close code 4001."""
        conn = connect(WS_URL, open_timeout=AUTH_TIMEOUT, close_timeout=10)
        try:
            conn.send(json.dumps({"token": "not-the-token"}))
            frame = json.loads(conn.recv(timeout=STATUS_TIMEOUT))
            self.assertEqual(frame.get("error"), "Authentication failed")
            with self.assertRaises(ConnectionClosed) as ctx:
                conn.recv(timeout=STATUS_TIMEOUT)
            self.assertEqual(ctx.exception.rcvd.code, 4001)
        finally:
            with contextlib.suppress(ConnectionClosed):
                conn.close()

    def test_02_clone_without_prompt_file_errors_before_generating(self):
        """Validation: clone mode missing prompt_file → error, socket stays open."""
        conn = _authed_connect(self.token)
        try:
            conn.send(json.dumps({"text": _TEXT, "mode": "clone"}))
            frame = json.loads(conn.recv(timeout=STATUS_TIMEOUT))
            self.assertEqual(frame.get("error"), "prompt_file required for clone mode")
            # The socket survives a validation error (persistent channel):
            # prove it by driving one more request through the same socket.
            conn.send(json.dumps({"text": ""}))
            frame = json.loads(conn.recv(timeout=STATUS_TIMEOUT))
            self.assertEqual(frame.get("error"), "No text provided")
        finally:
            conn.close()

    def test_03_clone_generation_streams_frames_and_seed(self):
        """Happy path: generating frame → binary frames → complete + chunk count + seed."""
        conn = _authed_connect(self.token)
        try:
            supplied_seed = 1234
            conn.send(
                json.dumps(
                    {
                        "text": _TEXT,
                        "mode": "clone",
                        "prompt_file": self.prompt,
                        "seed": supplied_seed,
                    }
                )
            )
            control, binary, terminal = _recv_until_terminal(conn)
        finally:
            conn.close()

        self.assertEqual(
            [c.get("status") for c in control],
            ["generating"],
            f"expected exactly the generating preamble, got {control}",
        )
        self.assertEqual(control[0].get("text_length"), len(_TEXT))
        self.assertGreaterEqual(len(binary), 1, "no audio frames received")
        for frame in binary:
            self._assert_binary_frame(frame)
        self.assertEqual(terminal.get("status"), "complete", f"terminal: {terminal}")
        self.assertEqual(
            terminal.get("chunks"),
            len(binary),
            "terminal chunk count must equal binary frames received",
        )
        self.assertEqual(
            terminal.get("seed"),
            supplied_seed,
            "server must echo the client-supplied seed",
        )

    def test_04_cancel_mid_generation_keeps_socket_open(self):
        """Cancel inside the first chunk → terminal 'cancelled', chunks 0, socket open."""
        conn = _authed_connect(self.token)
        try:
            conn.send(
                json.dumps(
                    {
                        "text": _TEXT,
                        "mode": "clone",
                        "prompt_file": self.prompt,
                    }
                )
            )
            frame = json.loads(conn.recv(timeout=STATUS_TIMEOUT))
            self.assertEqual(frame.get("status"), "generating")
            # Cancel inside the first chunk: the cancel-watcher reads it (the
            # main loop is blocked generating), so there is NO ack frame —
            # the terminal frame is the observable contract.
            conn.send(json.dumps({"action": "cancel"}))
            control, binary, terminal = _recv_until_terminal(conn)

            self.assertEqual(
                terminal.get("status"), "cancelled", f"terminal: {terminal}"
            )
            self.assertEqual(
                terminal.get("chunks"),
                0,
                f"cancel inside the first chunk must yield 0 chunks: {terminal}",
            )
            # Socket still open (classified outcomes never close): a fresh
            # request on the same socket gets a normal validation reply.
            conn.send(json.dumps({"text": ""}))
            frame = json.loads(conn.recv(timeout=STATUS_TIMEOUT))
            self.assertEqual(frame.get("error"), "No text provided")
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
