"""The /ws WebSocket handler must cap concurrent connections per IP.

websocket.accept() happens before authentication, and each accepted connection
pins a task/FD for up to the 10s auth window. Without a connection cap, an
unauthenticated client (especially in --public/Colab mode) can flood the server
and exhaust file descriptors. The handler must reject connections over a per-IP
(and global) limit before accepting, without invoking auth.

See docs/reviews/e2e-review-2026-07-01.md (Phase 5 security, H1).
"""

import types
import unittest

from fastapi import WebSocketDisconnect

from qwen3_tts.server.websocket import (
    _WS_MAX_PER_IP,
    websocket_tts_handler,
)


class _FakeWebSocket:
    def __init__(self, client_host="1.2.3.4", messages=None):
        self.client = types.SimpleNamespace(host=client_host)
        self.accepted = False
        self.closed_code = None
        self._messages = list(messages or [])
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def close(self, code=1000, reason=""):
        self.closed_code = code

    async def receive_text(self):
        if self._messages:
            return self._messages.pop(0)
        raise WebSocketDisconnect()

    async def send_json(self, data):
        self.sent.append(data)

    async def send_bytes(self, data):
        self.sent.append(data)


def _app_state(connections=None):
    state = types.SimpleNamespace()
    if connections is not None:
        state._ws_connections = connections
    return state


class TestWebSocketConnectionLimit(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_when_ip_at_limit_without_authenticating(self):
        auth_calls = []

        def verify(token):
            auth_calls.append(token)
            return True

        # This IP already holds the max number of connections.
        state = _app_state({"1.2.3.4": _WS_MAX_PER_IP})
        ws = _FakeWebSocket(client_host="1.2.3.4", messages=['{"token": "x"}'])

        await websocket_tts_handler(ws, state, verify)

        self.assertFalse(ws.accepted, "over-limit connection must not be accepted")
        self.assertIsNotNone(ws.closed_code, "over-limit connection must be closed")
        self.assertEqual(auth_calls, [], "auth must not run for a rejected connection")

    async def test_releases_slot_after_handler_returns(self):
        # Under the limit: connection is accepted, auth fails, slot is released.
        state = _app_state({})
        ws = _FakeWebSocket(client_host="5.6.7.8", messages=['{"token": "bad"}'])

        await websocket_tts_handler(ws, state, lambda t: False)

        self.assertTrue(ws.accepted)
        # No lingering reservation for this IP after the handler returns.
        self.assertEqual(state._ws_connections.get("5.6.7.8", 0), 0)


if __name__ == "__main__":
    unittest.main()
