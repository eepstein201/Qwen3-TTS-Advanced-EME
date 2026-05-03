#!/usr/bin/env python3
"""Shared UI components for the Gradio interface.

Phase 1a: StatusBanner + a11y helpers.

Public API:
    StatusBanner         - global accessible status surface
    poll_model_loading_state(model_type) - live /models state, not stale config
    status_badge(message, severity) - inline badge for table cells
    severity_icon(name) - inline SVG by name
"""

from __future__ import annotations

import html as html_mod
import logging
import threading
from typing import Literal

from qwen3_tts.core.config import (
    auth_headers,
    get_server_url,
    is_server_running,
    load_config,
)

logger = logging.getLogger("tts.ui")


# ---------------------------------------------------------------------------
# Severity styling
# ---------------------------------------------------------------------------

Severity = Literal["info", "success", "warning", "error", "loading"]

# Colour tokens chosen to meet WCAG 4.5:1 against light + dark Gradio surfaces.
_SEVERITY_STYLE = {
    "info":    {"color": "#1a4480", "label": "Info",     "icon": "info"},
    "success": {"color": "#0c5d00", "label": "Success",  "icon": "check"},
    "warning": {"color": "#7d4f00", "label": "Warning",  "icon": "warn"},
    "error":   {"color": "#9b1c1c", "label": "Error",    "icon": "x"},
    "loading": {"color": "#1a4480", "label": "Loading",  "icon": "spinner"},
}

# Inline SVGs (Heroicons-style minimal). 16x16 viewBox, currentColor fill so
# they inherit the surrounding span's color. aria-hidden because the text
# label carries the same meaning to screen readers.
_SVG_CHECK = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="14" '
    'height="14" fill="currentColor" aria-hidden="true" focusable="false" '
    'style="vertical-align:-2px;margin-right:4px;">'
    '<path d="M13.485 3.515a1 1 0 010 1.414l-7 7a1 1 0 01-1.414 0l-3-3a1 1 0 '
    '011.414-1.414L5.778 9.808l6.293-6.293a1 1 0 011.414 0z"/></svg>'
)
_SVG_X = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="14" '
    'height="14" fill="currentColor" aria-hidden="true" focusable="false" '
    'style="vertical-align:-2px;margin-right:4px;">'
    '<path d="M3.293 3.293a1 1 0 011.414 0L8 6.586l3.293-3.293a1 1 0 '
    '111.414 1.414L9.414 8l3.293 3.293a1 1 0 01-1.414 1.414L8 9.414l-3.293 '
    '3.293a1 1 0 01-1.414-1.414L6.586 8 3.293 4.707a1 1 0 010-1.414z"/></svg>'
)
_SVG_WARN = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="14" '
    'height="14" fill="currentColor" aria-hidden="true" focusable="false" '
    'style="vertical-align:-2px;margin-right:4px;">'
    '<path d="M8 1.5a1 1 0 01.866.5l6.5 11.25A1 1 0 0114.5 14.75h-13a1 1 0 '
    '01-.866-1.5L7.134 2A1 1 0 018 1.5zM8 6a1 1 0 00-1 1v3a1 1 0 002 0V7a1 '
    '1 0 00-1-1zm0 7.25a.875.875 0 100-1.75.875.875 0 000 1.75z"/></svg>'
)
_SVG_INFO = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="14" '
    'height="14" fill="currentColor" aria-hidden="true" focusable="false" '
    'style="vertical-align:-2px;margin-right:4px;">'
    '<path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 3a1 1 0 110 2 1 1 0 010-2zm1 '
    '8a1 1 0 11-2 0V8a1 1 0 112 0v4z"/></svg>'
)
_SVG_SPINNER = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="14" '
    'height="14" aria-hidden="true" focusable="false" '
    'style="vertical-align:-2px;margin-right:4px;">'
    '<circle cx="8" cy="8" r="6" stroke="currentColor" stroke-width="2" '
    'fill="none" stroke-dasharray="9 6" stroke-linecap="round">'
    '<animateTransform attributeName="transform" type="rotate" '
    'from="0 8 8" to="360 8 8" dur="1s" repeatCount="indefinite"/>'
    '</circle></svg>'
)

_ICON = {
    "info": _SVG_INFO,
    "check": _SVG_CHECK,
    "warn": _SVG_WARN,
    "x": _SVG_X,
    "spinner": _SVG_SPINNER,
}


def severity_icon(name: str) -> str:
    """Return inline SVG for a named status icon (info/check/warn/x/spinner)."""
    return _ICON.get(name, _SVG_INFO)


def status_badge(message: str, severity: Severity = "info") -> str:
    """Render a small inline status badge: SVG icon + visible text + aria-label.

    Use for in-table row indicators (model loaded/not loaded). For top-level
    surfaces use StatusBanner instead.
    """
    style = _SEVERITY_STYLE.get(severity, _SEVERITY_STYLE["info"])
    safe = html_mod.escape(message)
    aria = html_mod.escape(f"{style['label']}: {message}")
    icon = severity_icon(style["icon"])
    return (
        f'<span aria-label="{aria}" style="color:{style["color"]};'
        f'font-weight:500;display:inline-flex;align-items:center;">'
        f'{icon}{safe}</span>'
    )


# ---------------------------------------------------------------------------
# StatusBanner
# ---------------------------------------------------------------------------

class StatusBanner:
    """Global accessible status surface for the Gradio UI.

    Backed by gr.HTML - render() returns the HTML string. The Gradio Block
    updates by passing this string into a gr.HTML's value.

    Thread-safe: concurrent calls from multiple Gradio request threads do
    not corrupt the cached message (mirrors R-50 history_state lock pattern).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_message: str = ""
        self._last_severity: Severity = "info"

    def render(self, message: str, severity: Severity = "info") -> str:
        """Return HTML for a status update.

        - role="status" + aria-live="polite" so screen readers announce updates
          without interrupting the user.
        - Inline SVG + visible text + aria-label so colour is never the only
          carrier of meaning.
        - HTML-escapes user content (XSS-safe).
        """
        if severity not in _SEVERITY_STYLE:
            severity = "info"
        with self._lock:
            self._last_message = message
            self._last_severity = severity

        style = _SEVERITY_STYLE[severity]
        if not message:
            # Empty placeholder still emits the live region container so
            # Gradio's Component diff can replace it without re-mounting.
            return (
                '<div role="status" aria-live="polite" '
                'style="min-height:1.5rem;"></div>'
            )

        safe = html_mod.escape(message)
        aria = html_mod.escape(f"{style['label']}: {message}")
        icon = severity_icon(style["icon"])
        return (
            '<div role="status" aria-live="polite" '
            f'aria-label="{aria}" '
            f'style="padding:8px 12px;border-radius:6px;'
            f'background:var(--block-background-fill,#f8f8f8);'
            f'border:1px solid var(--block-border-color,#e0e0e0);'
            f'color:{style["color"]};font-weight:500;'
            f'display:flex;align-items:center;">'
            f'{icon}<span>{safe}</span></div>'
        )

    @property
    def last_message(self) -> str:
        with self._lock:
            return self._last_message


# ---------------------------------------------------------------------------
# poll_model_loading_state - fixes Phase 0 bug #4 (stale "Loaded" indicator)
# ---------------------------------------------------------------------------

def poll_model_loading_state(model_type: str, timeout: float = 5.0) -> str:
    """Return live loading state for a single model_type.

    Returns:
        "loaded"     - server confirms loaded
        "loading"    - server reports model.loading=True
        "not_loaded" - server confirms not loaded and not loading
        "unknown"    - server unreachable / error / network failure

    This polls the live /models endpoint so the UI reflects actual server
    state, not the static config (load_at_startup=True), which previously
    caused a stale "Loaded (2500MB)" badge during cold downloads.
    """
    try:
        config = load_config()
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.debug("poll_model_loading_state: load_config failed: %s", exc)
        return "unknown"

    if not is_server_running(config):
        return "unknown"

    try:
        import requests
        url = get_server_url(config)
        resp = requests.get(f"{url}/models", timeout=timeout, headers=auth_headers())
    except Exception as exc:
        logger.debug("poll_model_loading_state: request failed: %s", exc)
        return "unknown"

    if resp.status_code != 200:
        return "unknown"

    try:
        info = resp.json().get("models", {}).get(model_type, {}) or {}
    except ValueError:
        return "unknown"

    if info.get("loaded"):
        return "loaded"
    if info.get("loading"):
        return "loading"
    return "not_loaded"
