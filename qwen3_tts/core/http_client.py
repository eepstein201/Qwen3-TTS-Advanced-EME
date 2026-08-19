"""Single choke-point for HTTP requests to the local TTS server.

All call sites that previously did ``requests.get(f"{url}/...")`` should route
through :func:`server_request` instead. The URL is re-validated inline in this
same function so CodeQL's data-flow analysis can see the trust boundary at the
``requests.request`` sink.
"""

from __future__ import annotations

from typing import Any

import requests

from qwen3_tts.core.config import (
    _validate_server_url,
    auth_headers,
    get_server_url,
    load_config,
)

__all__ = ["LOAD_MODEL_TIMEOUT_SEC", "server_request"]

# A /load-model issued while a generation is running queues its warm-up
# behind inference_lock (#192 serialization), so the total is load time +
# the queued generation's runtime + warm-up. The old 120s timed out
# client-side while the server kept working, and the visible failure
# invited a retry that double-loads. Bound covers the documented
# whole-text worst case (~660s) plus a cold download; a longer queued
# generation can still exceed it — the residual spurious-timeout window
# is accepted rather than scaling, since the client cannot know the size
# of someone else's queued generation. EVERY /load-model caller must use
# this constant (guarded by tests/test_issue192_warmup_serialization.py).
LOAD_MODEL_TIMEOUT_SEC = 900

_ALLOWED_METHODS = frozenset(
    {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
)


def server_request(
    method: str,
    path: str,
    *,
    timeout: int | float = 10,
    json: Any = None,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    stream: bool = False,
) -> requests.Response:
    """Issue an HTTP request to the local TTS server, re-validating inline.

    Args:
        method: HTTP method (GET/POST/etc.); must be in the allowed-methods allowlist.
        path: Server-relative path; MUST start with ``/`` and not contain ``://``,
            ``?``, or ``#``.
        timeout: Request timeout in seconds.
        json: Optional JSON payload.
        params: Optional query-string parameters (dict).
        headers: Optional headers — merged on top of :func:`auth_headers`.
        stream: If True, stream the response body (required for chunk-by-chunk reads).

    Raises:
        ValueError: If ``method`` is not in the allowed allowlist, if ``path`` is
            not a server-relative path, or if the configured server host is not
            in the allowlist.
    """
    if not isinstance(method, str) or method.upper() not in _ALLOWED_METHODS:
        raise ValueError(f"Invalid HTTP method: {method!r}")
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or "://" in path
        or "?" in path
        or "#" in path
    ):
        raise ValueError(f"Invalid server path: {path!r}")
    base = _validate_server_url(get_server_url(load_config()))
    final_headers = {**auth_headers(), **(headers or {})}
    return requests.request(
        method,
        f"{base}{path}",
        json=json,
        params=params,
        headers=final_headers,
        timeout=timeout,
        stream=stream,
    )
