#!/usr/bin/env python3
"""Shared UI components for the Gradio interface.

Phase 1a: StatusBanner + a11y helpers.
Phase 1b: ProgressIndicator + poll_model_load_progress.

Public API:
    StatusBanner         - global accessible status surface
    ProgressIndicator    - bounded/indeterminate progressbar with WCAG aria
    confirm_step(state, arm_label, original_label) - two-step confirm helper
    poll_model_loading_state(model_type) - live /models state, not stale config
    poll_model_load_progress(model_type) - structured progress dict for UI
    status_badge(message, severity) - inline badge for table cells
    severity_icon(name) - inline SVG by name
"""

from __future__ import annotations

import html as html_mod
import logging
import threading
from typing import Literal

from qwen3_tts.core.config import (
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
        from qwen3_tts.core.http_client import server_request
        resp = server_request("GET", "/models", timeout=timeout)
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


# ---------------------------------------------------------------------------
# ProgressIndicator (Phase 1b) - WCAG progressbar, bounded or indeterminate
# ---------------------------------------------------------------------------

ProgressMode = Literal["bounded", "indeterminate"]


class ProgressIndicator:
    """Accessible progressbar for long operations (model load, ASR, AI enhance).

    Two modes:
      - "bounded": known percent (0-100). Renders aria-valuenow + visible "X%".
      - "indeterminate": unknown duration. Renders aria-busy=true + spinner.

    Output is HTML — feed into a gr.HTML component or splice into status text.
    Thread-safe in the same sense as StatusBanner: instances are cheap and
    typically constructed per render rather than shared.
    """

    def __init__(
        self,
        percent: int | float | None = None,
        eta_s: float | None = None,
        message: str | None = None,
        mode: ProgressMode = "bounded",
    ) -> None:
        self.mode = mode if mode in ("bounded", "indeterminate") else "bounded"
        self.message = message or ""
        self.eta_s = eta_s
        if percent is None:
            self.percent = 0
        else:
            try:
                p = int(round(float(percent)))
            except (TypeError, ValueError):
                p = 0
            # Clamp to valid aria-valuenow range
            self.percent = max(0, min(100, p))

    def render(self) -> str:
        """Return HTML for the progressbar."""
        safe_msg = html_mod.escape(self.message)
        spinner = severity_icon("spinner")

        if self.mode == "indeterminate":
            return (
                '<div role="progressbar" aria-busy="true" '
                f'aria-label="{safe_msg or "Working"}" '
                'style="display:flex;align-items:center;padding:6px 10px;'
                'border-radius:6px;background:var(--block-background-fill,#f8f8f8);'
                'border:1px solid var(--block-border-color,#e0e0e0);'
                'color:#1a4480;font-weight:500;">'
                f'{spinner}<span>{safe_msg}</span></div>'
            )

        # Bounded mode
        eta_part = ""
        if self.eta_s is not None:
            try:
                secs = max(0, int(round(float(self.eta_s))))
                eta_part = f" · ~{secs}s"
            except (TypeError, ValueError):
                eta_part = ""
        label_text = f"{self.percent}%{eta_part}"
        if self.message:
            label_text = f"{safe_msg} — {label_text}"

        bar_width = self.percent  # already clamped
        # Visual bar (purely decorative; aria-valuenow is the source of truth).
        return (
            '<div role="progressbar" '
            f'aria-valuenow="{self.percent}" aria-valuemin="0" aria-valuemax="100" '
            f'aria-label="{html_mod.escape(label_text)}" '
            'style="padding:6px 10px;border-radius:6px;'
            'background:var(--block-background-fill,#f8f8f8);'
            'border:1px solid var(--block-border-color,#e0e0e0);'
            'color:#1a4480;font-weight:500;">'
            f'<div style="display:flex;justify-content:space-between;'
            f'margin-bottom:4px;font-size:0.875rem;"><span>{label_text}</span></div>'
            '<div style="height:6px;border-radius:3px;background:#e0e0e0;'
            'overflow:hidden;">'
            f'<div style="height:100%;width:{bar_width}%;background:#1a4480;'
            'transition:width 200ms ease-out;"></div></div></div>'
        )


# ---------------------------------------------------------------------------
# poll_model_load_progress (Phase 1b) - structured progress dict for UI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# confirm_step (Phase 1c) - two-step confirm for destructive actions
# ---------------------------------------------------------------------------

def confirm_step(
    confirm_state: dict | None,
    arm_label: str,
    original_label: str,
    timeout_s: float = 5.0,
) -> tuple:
    """Two-step confirm helper for destructive actions.

    First call (unarmed or expired): arms the button — returns is_confirmed=False.
    Second call within timeout_s: executes — returns is_confirmed=True.

    Returns (new_state, btn_update, is_confirmed).

    Wire as:
        btn.click(fn=handler, inputs=[confirm_state, ...], outputs=[confirm_state, btn, ...])

    where handler calls confirm_step and acts on is_confirmed.
    """
    import time

    import gradio as gr

    if not isinstance(confirm_state, dict):
        confirm_state = {}

    now = time.time()
    armed = confirm_state.get("armed", False)
    ts = confirm_state.get("ts", 0.0)

    if not armed or (now - ts) > timeout_s:
        new_state = {"armed": True, "ts": now}
        return new_state, gr.update(value=arm_label), False
    else:
        new_state = {"armed": False, "ts": 0.0}
        return new_state, gr.update(value=original_label), True


class ConfirmButton:
    """Two-step confirmation button for destructive actions.

    Wraps confirm_step() logic with a clean API for Gradio wiring.

    Usage:
        confirm_btn = ConfirmButton(
            arm_label="Confirm Delete? (click again)",
            original_label="Delete Voice",
            timeout_s=5.0,
            status_message="Please confirm within 5 seconds"
        )

        # In Gradio click handler:
        new_state, btn_update, status_update, confirmed = confirm_btn.click(state)
        if not confirmed:
            return new_state, btn_update, status_update, gr.update()
        return execute_destructive_action(...)

    Args:
        arm_label: Button text when armed (first click)
        original_label: Original button text (unarmed state)
        timeout_s: Seconds before auto-reset (default 5.0)
        status_message: Status text to show on first click

    Returns:
        Tuple of (state_dict, btn_update, status_update, is_confirmed)
    """

    def __init__(
        self,
        arm_label: str,
        original_label: str,
        timeout_s: float = 5.0,
        status_message: str = "Please confirm within 5 seconds",
    ):
        self.arm_label = arm_label
        self.original_label = original_label
        self.timeout_s = timeout_s
        self.status_message = status_message

    def click(self, confirm_state: dict | None) -> tuple:
        """Handle button click in Gradio event chain.

        Args:
            confirm_state: Current confirmation state dict (or None)

        Returns:
            (new_state, btn_update, status_update, is_confirmed)
        """
        import gradio as gr

        new_state, btn_update, confirmed = confirm_step(
            confirm_state,
            arm_label=self.arm_label,
            original_label=self.original_label,
            timeout_s=self.timeout_s,
        )

        status_update = gr.update(value=self.status_message) if not confirmed else gr.update()

        return new_state, btn_update, status_update, confirmed


def poll_model_load_progress(model_type: str, timeout: float = 5.0) -> dict:
    """Return structured progress for a model load.

    Returns dict with:
        state:     "loading" | "loaded" | "not_loaded" | "unknown"
        memory_mb: int (from /models entry; 0 if unknown)
        eta_s:     float | None (from prior load_time_sec; None if no history)

    Caller uses this to drive a ProgressIndicator. The eta_s is a coarse
    heuristic — the prior measured load duration. It does not reflect remaining
    time in the current load (the server has no way to know).
    """
    default = {"state": "unknown", "memory_mb": 0, "eta_s": None}

    try:
        config = load_config()
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.debug("poll_model_load_progress: load_config failed: %s", exc)
        return default

    if not is_server_running(config):
        return default

    try:
        from qwen3_tts.core.http_client import server_request
        resp = server_request("GET", "/models", timeout=timeout)
    except Exception as exc:
        logger.debug("poll_model_load_progress: request failed: %s", exc)
        return default

    if resp.status_code != 200:
        return default

    try:
        info = resp.json().get("models", {}).get(model_type, {}) or {}
    except ValueError:
        return default

    if info.get("loaded"):
        state = "loaded"
    elif info.get("loading"):
        state = "loading"
    else:
        state = "not_loaded"

    memory_mb = int(info.get("memory_mb") or 0)
    load_time_sec = info.get("load_time_sec")
    eta_s = float(load_time_sec) if load_time_sec is not None else None

    return {"state": state, "memory_mb": memory_mb, "eta_s": eta_s}
