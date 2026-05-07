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

_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"})


def server_request(
    method: str,
    path: str,
    *,
    timeout: int | float = 10,
    json: object = None,
    headers: dict[str, str] | None = None,
) -> "requests.Response":
    """Issue an HTTP request to the local TTS server, re-validating inline.

    Args:
        method: HTTP method (GET/POST/etc.); must be in the allowed-methods allowlist.
        path: Server-relative path; MUST start with ``/`` and not contain ``://``,
            ``?``, or ``#``.
        timeout: Request timeout in seconds.
        json: Optional JSON payload.
        headers: Optional headers — merged on top of :func:`auth_headers`.

    Raises:
        ValueError: If ``method`` is not in the allowed allowlist, if ``path`` is
            not a server-relative path, or if the configured server host is not
            in the allowlist.
    """
    if not isinstance(method, str) or method.upper() not in _ALLOWED_METHODS:
        raise ValueError(f"Invalid HTTP method: {method!r}")
    if not isinstance(path, str) or not path.startswith("/") or "://" in path or "?" in path or "#" in path:
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
