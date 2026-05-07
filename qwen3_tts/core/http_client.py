"""Single choke-point for HTTP requests to the local TTS server.

All call sites that previously did ``requests.get(f"{url}/...")`` should route
through :func:`server_request` instead. The URL is re-validated inline in this
same function so CodeQL's data-flow analysis can see the trust boundary at the
``requests.request`` sink.
"""
from __future__ import annotations

import requests

from qwen3_tts.core.config import (
    _validate_server_url,
    auth_headers,
    get_server_url,
    load_config,
)

__all__ = ["server_request"]


def server_request(method, path, *, timeout=10, json=None, headers=None):
    """Issue an HTTP request to the local TTS server, re-validating inline.

    Args:
        method: HTTP method (GET/POST/etc.). Passed straight to ``requests``.
        path: Server-relative path; MUST start with ``/`` and not contain ``://``.
        timeout: Request timeout in seconds.
        json: Optional JSON payload.
        headers: Optional headers — merged on top of :func:`auth_headers`.

    Raises:
        ValueError: If ``path`` is not a server-relative path, or if the
            configured server host is not in the allowlist.
    """
    if not isinstance(path, str) or not path.startswith("/") or "://" in path:
        raise ValueError(f"Invalid server path: {path!r}")
    base = _validate_server_url(get_server_url(load_config()))
    final_headers = {**auth_headers(), **(headers or {})}
    return requests.request(
        method,
        f"{base}{path}",
        json=json,
        headers=final_headers,
        timeout=timeout,
    )
