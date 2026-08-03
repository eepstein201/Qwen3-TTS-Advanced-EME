"""SSRF call-site tests for PR 2.

Step 1 (RED): These tests verify that all raw ``requests.get/post(f"{url}/...")``
call sites have been replaced with :func:`server_request`.

Two kinds of checks:
  1. Structural AST scan — zero f-string ``requests.(get|post)(`` patterns in
     production code.
  2. Per-module routing — each refactored function calls ``server_request``,
     not raw ``requests``.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROD_ROOT = Path(__file__).parents[2] / "qwen3_tts"
_PATTERN = re.compile(r'requests\.(get|post)\(f["\']')


def _scan_file(path: Path) -> list[tuple[int, str]]:
    """Return (line_number, line_content) for every SSRF pattern match."""
    hits = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if _PATTERN.search(line):
                hits.append((lineno, line.rstrip()))
    return hits


_SCAN_EXCEPTIONS = frozenset({
    # http_client.py — owns the choke-point itself (no f-string usage there anyway)
    "http_client.py",
    # core/config/runtime.py (formerly core/config.py, split into a package
    # per repo-audit-2026-07-31 P2-1) — Rule B exception: cannot import
    # http_client (circular import). is_server_running() applies
    # _validate_server_url() inline before the requests.get call, which
    # satisfies the trust-boundary requirement.
    "runtime.py",
    # server/client/ — uses requests.Session (not bare requests.get/post) routed
    # through TTSClient which validates the URL via the server_url property
    # (calls get_server_url() → _validate_server_url() internally). This
    # architectural pattern is intentionally out of scope for this PR's
    # pattern scan. Refactoring server/client/ to use server_request would
    # require session-level changes tracked separately.
})


def _collect_all_hits() -> dict[str, list[tuple[int, str]]]:
    """Scan all production .py files; return path→hits mapping (non-empty only)."""
    results: dict[str, list[tuple[int, str]]] = {}
    for py_file in sorted(PROD_ROOT.rglob("*.py")):
        if py_file.name in _SCAN_EXCEPTIONS:
            continue
        hits = _scan_file(py_file)
        if hits:
            results[str(py_file.relative_to(PROD_ROOT.parent))] = hits
    return results


# ---------------------------------------------------------------------------
# Test 1: structural scan
# ---------------------------------------------------------------------------


class TestStructuralScan(unittest.TestCase):
    """Fail if any production file still uses requests.get/post(f"...")."""

    def test_no_raw_requests_fstring_in_production(self):
        hits = _collect_all_hits()
        if hits:
            lines = []
            for path, matches in hits.items():
                for lineno, content in matches:
                    lines.append(f"  {path}:{lineno}: {content}")
            self.fail(
                "Found raw requests.(get|post)(f\"...) patterns — route through server_request:\n"
                + "\n".join(lines)
            )


# ---------------------------------------------------------------------------
# Test 2: per-module routing tests
# ---------------------------------------------------------------------------


def _make_mock_response(status_code=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.content = b""
    return resp


class TestComponentsRouting(unittest.TestCase):
    """poll_model_loading_state routes through server_request."""

    def test_poll_model_loading_state_uses_server_request(self):
        from qwen3_tts.interface.ui import components

        mock_resp = _make_mock_response(
            200, {"models": {"clone": {"loaded": True, "loading": False}}}
        )
        with (
            patch("qwen3_tts.interface.ui.components.is_server_running", return_value=True),
            patch("qwen3_tts.interface.ui.components.load_config", return_value={}),
            patch(
                "qwen3_tts.core.http_client.server_request", return_value=mock_resp
            ) as mock_sr,
        ):
            result = components.poll_model_loading_state("clone")

        mock_sr.assert_called_once()
        call_args = mock_sr.call_args
        self.assertEqual(call_args[0][0].upper(), "GET")
        self.assertIn("/models", call_args[0][1])
        self.assertEqual(result, "loaded")


class TestSharedRouting(unittest.TestCase):
    """get_current_model_settings routes through server_request."""

    def test_get_current_model_settings_uses_server_request(self):
        from qwen3_tts.interface.ui import shared

        mock_resp = _make_mock_response(
            200,
            {"settings": {"model_size": "1.7B", "mlx_quantization": "8bit", "backend": "mlx"}},
        )
        with (
            patch("qwen3_tts.interface.ui.shared.is_server_running", return_value=True),
            patch("qwen3_tts.interface.ui.shared.load_config", return_value={}),
            patch("qwen3_tts.interface.ui.shared.get_backend", return_value="mlx"),
            patch("qwen3_tts.interface.ui.shared.get_model_size", return_value="1.7B"),
            patch("qwen3_tts.interface.ui.shared.get_mlx_quantization", return_value="8bit"),
            patch(
                "qwen3_tts.core.http_client.server_request", return_value=mock_resp
            ) as mock_sr,
        ):
            result = shared.get_current_model_settings()

        mock_sr.assert_called_once()
        call_args = mock_sr.call_args
        self.assertEqual(call_args[0][0].upper(), "GET")
        self.assertIn("/models", call_args[0][1])
        self.assertEqual(result, ("1.7B", "8bit", "mlx"))


class TestModelManagementRouting(unittest.TestCase):
    """get_model_status_html routes through server_request."""

    def test_get_model_status_html_uses_server_request(self):
        from qwen3_tts.interface.ui import model_management

        mock_resp = _make_mock_response(
            200,
            {"models": {"clone": {"loaded": True, "loading": False, "memory_mb": 2500}}},
        )
        with (
            patch(
                "qwen3_tts.interface.ui.model_management.is_server_running",
                return_value=True,
            ),
            patch(
                "qwen3_tts.interface.ui.model_management.load_config", return_value={}
            ),
            patch(
                "qwen3_tts.core.http_client.server_request", return_value=mock_resp
            ) as mock_sr,
        ):
            html = model_management.get_model_status_html("clone")

        mock_sr.assert_called_once()
        call_args = mock_sr.call_args
        self.assertEqual(call_args[0][0].upper(), "GET")
        self.assertIn("/models", call_args[0][1])
        # Should contain some HTML
        self.assertIsInstance(html, str)


class TestGenerationCancelRouting(unittest.TestCase):
    """cancel_streaming_generation routes through server_request."""

    def test_cancel_streaming_generation_uses_server_request(self):
        from qwen3_tts.interface.ui import generation

        mock_resp = _make_mock_response(200, {})
        with (
            patch(
                "qwen3_tts.interface.ui.generation.is_server_running", return_value=True
            ),
            patch("qwen3_tts.interface.ui.generation.load_config", return_value={}),
            patch(
                "qwen3_tts.core.http_client.server_request", return_value=mock_resp
            ) as mock_sr,
        ):
            generation.cancel_streaming_generation()

        mock_sr.assert_called_once()
        call_args = mock_sr.call_args
        self.assertEqual(call_args[0][0].upper(), "POST")
        self.assertIn("/cancel-generation", call_args[0][1])


class TestGenerationServerSideRouting(unittest.TestCase):
    """_generate_server_side (via generate_via_server) routes through server_request."""

    def test_generate_server_side_uses_server_request(self):
        """Verify generate_via_server calls server_request instead of raw requests."""
        from qwen3_tts.interface import generate_server

        mock_resp = _make_mock_response(
            200,
            {
                "results": [
                    {
                        "text": "hi",
                        "audio_base64": "",
                        "duration_sec": 0.5,
                        "sample_rate": 22050,
                        "chunks": 1,
                    }
                ]
            },
        )

        mock_tts_client = MagicMock()
        mock_tts_client.return_value.generate.return_value = mock_resp.json()

        with (
            patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp) as mock_sr,
            patch("qwen3_tts.interface.generate_server.get_server_url", return_value="http://127.0.0.1:5123"),
            patch("qwen3_tts.interface.generate_interactive._ProgressPoller") as mock_poller,
        ):
            mock_poller.return_value.__enter__ = lambda s: s
            mock_poller.return_value.__exit__ = MagicMock(return_value=False)
            mock_poller.return_value.start = MagicMock()
            mock_poller.return_value.stop = MagicMock()

            # Call the function — it should use server_request, not raw requests.post
            try:
                generate_server.generate_via_server(
                    ["hi"],
                    "clone",
                    {},
                    {"server_side": True, "payload": {"text": "hi", "mode": "clone"}},
                )
            except Exception:
                # We only care that server_request was invoked, not about full
                # result processing
                pass

        mock_sr.assert_called()
        # At least one call should be a POST to /generate
        post_calls = [
            c for c in mock_sr.call_args_list
            if c[0][0].upper() == "POST" and "/generate" in c[0][1]
        ]
        self.assertTrue(
            len(post_calls) >= 1,
            f"Expected server_request('POST', '/generate', ...) but got calls: {mock_sr.call_args_list}",
        )


if __name__ == "__main__":
    unittest.main()
