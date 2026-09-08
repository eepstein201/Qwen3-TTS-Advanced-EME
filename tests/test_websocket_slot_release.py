#!/usr/bin/env python3
"""The /ws connection slot must be released by ONE handler-level finally.

``_ws_try_acquire`` reserves a per-IP slot BEFORE ``websocket.accept()``, so
every exit from ``websocket_tts_handler`` — return or raise, auth or message
loop — must release exactly that one slot. The release used to be scattered
across four branches, and any of them could be skipped when ``accept()`` /
``close()`` / ``send_json()`` raised first. The most reachable leak: the auth
``except`` block, whose own ``close()`` raises again when the original failure
was a disconnect, escaping before its ``_ws_release`` line.

These tests drive the handler directly with fake websockets (TestClient cannot
make ``close()`` raise — the 0B/0C precedent in ``test_websocket_rate_limit.py``).
Every leak scenario asserts BOTH: the exception surfaces exactly as before AND
``app_state._ws_connections`` is empty afterward. RED on the pre-fix code —
the count stays 1 (leaked).

A failed send implies the peer is gone, so a subsequent ``close()`` fails too;
the dead-socket scenarios therefore raise on both operations, matching a real
vanished peer rather than a selective failure no real socket produces.
"""

import types
import unittest

from fastapi import WebSocketDisconnect

from qwen3_tts.server.websocket import (
    _WS_MAX_PER_IP,
    websocket_tts_handler,
)

_RAISES = {
    "accept": "accept failed: peer vanished mid-handshake",
    "close": "close failed: peer already gone",
    "send_json": "send failed: peer already gone",
}


class _RaisingCloseWebSocket:
    """Fake /ws peer whose accept/close/send_json can be made to raise.

    ``raises_on`` is an operation name or a collection of them. Every listed
    operation raises ``RuntimeError`` on EVERY call — the except block's own
    retry of ``close()`` must fail the same way a real dead socket does.
    """

    def __init__(self, client_host="1.2.3.4", messages=None, raises_on=()):
        listed = {raises_on} if isinstance(raises_on, str) else set(raises_on)
        unknown = listed - set(_RAISES)
        if unknown:
            raise ValueError(f"unknown operations: {sorted(unknown)}")
        self.raises_on = frozenset(listed)
        self.client = types.SimpleNamespace(host=client_host)
        self.accepted = False
        self.closed_code = None
        self._messages = list(messages or [])
        self.sent = []

    async def accept(self):
        if "accept" in self.raises_on:
            raise RuntimeError(_RAISES["accept"])
        self.accepted = True

    async def close(self, code=1000, reason=""):
        if "close" in self.raises_on:
            raise RuntimeError(_RAISES["close"])
        self.closed_code = code

    async def receive_text(self):
        if self._messages:
            return self._messages.pop(0)
        raise WebSocketDisconnect()

    async def send_json(self, data):
        if "send_json" in self.raises_on:
            raise RuntimeError(_RAISES["send_json"])
        self.sent.append(data)

    async def send_bytes(self, data):
        self.sent.append(data)


def _app_state(connections=None):
    state = types.SimpleNamespace()
    if connections is not None:
        state._ws_connections = connections
    return state


class TestWebSocketSlotRelease(unittest.IsolatedAsyncioTestCase):
    """Every acquired slot is freed no matter where the handler exits."""

    def _assert_slot_freed(self, state, context):
        self.assertEqual(
            dict(getattr(state, "_ws_connections", {})),
            {},
            f"connection slot leaked: {context}",
        )

    async def test_accept_failure_releases_slot(self):
        """Scenario 1: the peer vanishes before the handshake completes.

        accept() raises before the auth try exists, so no branch release ever
        runs — only a handler-level finally can free this slot.
        """
        state = _app_state({})
        ws = _RaisingCloseWebSocket(client_host="1.2.3.4", raises_on="accept")

        with self.assertRaises(RuntimeError):
            await websocket_tts_handler(ws, state, lambda t: True)

        self.assertFalse(ws.accepted)
        self._assert_slot_freed(state, "accept() raised")

    async def test_non_dict_auth_with_failing_close_releases_slot(self):
        """Scenario 2: non-dict first message and the 4001 close() raises.

        The inner close() failure is caught by the auth except, whose own
        close() raises again and escapes — skipping the branch release.
        """
        state = _app_state({})
        ws = _RaisingCloseWebSocket(
            client_host="1.2.3.4", messages=["42"], raises_on="close"
        )

        with self.assertRaises(RuntimeError):
            await websocket_tts_handler(ws, state, lambda t: True)

        self._assert_slot_freed(state, "non-dict auth close() raised")

    async def test_invalid_token_send_failure_on_dead_socket_releases_slot(self):
        """Scenario 3: invalid token, send_json fails, the socket is dead.

        A failed send means the peer is gone, so the except block's close()
        fails too and escapes before its release line.
        """
        state = _app_state({})
        ws = _RaisingCloseWebSocket(
            client_host="1.2.3.4",
            messages=['{"token": "bad"}'],
            raises_on={"send_json", "close"},
        )

        with self.assertRaises(RuntimeError):
            await websocket_tts_handler(ws, state, lambda t: False)

        self._assert_slot_freed(state, "invalid-token send/close raised")

    async def test_invalid_token_close_failure_releases_slot(self):
        """Scenario 4: invalid token, the error send succeeds, close() raises.

        Isolates the close() failure from the send failure: the branch release
        after close() is still skipped because close() raised first.
        """
        state = _app_state({})
        ws = _RaisingCloseWebSocket(
            client_host="1.2.3.4", messages=['{"token": "bad"}'], raises_on="close"
        )

        with self.assertRaises(RuntimeError):
            await websocket_tts_handler(ws, state, lambda t: False)

        self.assertEqual(
            ws.sent, [{"error": "Authentication failed"}], "error send must succeed"
        )
        self._assert_slot_freed(state, "invalid-token close() raised")

    async def test_auth_disconnect_with_failing_handler_close_releases_slot(self):
        """Scenario 5: auth receive_text disconnects AND close() raises.

        The reachable double fault: the auth except block's own close() raises
        fresh on the dead socket and escapes past its release line.
        """
        state = _app_state({})
        ws = _RaisingCloseWebSocket(client_host="1.2.3.4", raises_on="close")

        with self.assertRaises(RuntimeError):
            await websocket_tts_handler(ws, state, lambda t: True)

        self._assert_slot_freed(state, "auth disconnect + close() raised")

    async def test_over_limit_rejection_leaves_counter_untouched(self):
        """Scenario 6: the rejection path must NOT release a slot it never got.

        The over-limit return stays outside the release-finally: releasing
        there would decrement a counter this connection never incremented.
        """
        state = _app_state({"1.2.3.4": _WS_MAX_PER_IP})
        ws = _RaisingCloseWebSocket(client_host="1.2.3.4", messages=['{"token": "x"}'])

        await websocket_tts_handler(ws, state, lambda t: True)

        self.assertFalse(ws.accepted, "over-limit connection must not be accepted")
        self.assertEqual(ws.closed_code, 1013, "rejection must close with 1013")
        self.assertEqual(
            state._ws_connections,
            {"1.2.3.4": _WS_MAX_PER_IP},
            "a rejected connection must not touch the counter",
        )

    async def test_failing_auth_preserves_sibling_slot_for_same_ip(self):
        """Scenario 7: one connection's exit must not steal a sibling's slot.

        The counter is per-IP, so a second release for the same IP would eat a
        live sibling's reservation. Exactly one release must run per handler.
        """
        state = _app_state({"1.2.3.4": 2})  # two live siblings on one IP
        ws = _RaisingCloseWebSocket(
            client_host="1.2.3.4", messages=['{"token": "bad"}']
        )

        await websocket_tts_handler(ws, state, lambda t: False)

        self.assertEqual(ws.closed_code, 4001, "failed auth must close with 4001")
        self.assertEqual(
            state._ws_connections.get("1.2.3.4"),
            2,
            "the sibling slots must survive exactly intact",
        )


if __name__ == "__main__":
    unittest.main()
