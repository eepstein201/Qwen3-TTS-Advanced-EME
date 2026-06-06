#!/usr/bin/env python3
"""Tests for StatusBanner — Phase 1a RED-then-GREEN suite.

Covers:
  - StatusBanner.render(): info/success/warning/error severity HTML
  - StatusBanner.render(): escapes user-controlled content (XSS)
  - StatusBanner.render(): produces role="status" and aria-live="polite"
  - StatusBanner.render(): clears (empty message) produces placeholder HTML
  - StatusBanner thread safety: concurrent writes don't corrupt state
  - poll_model_loading_state(): returns loading/loaded/not-loaded status
  - format_status_display(): no emoji in output
  - get_model_status_html(): no emoji in output; uses SVG + aria-label
  - get_model_status_html(): reflects /models state, not config

Run: pytest tests/test_ui_status_banner.py -v
"""

import threading

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

    class _DummyMarkerFunc:
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition=None, **kwargs):
            return lambda f: f

    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()

    class _DummyPytest:
        mark = _DummyMark()

    pytest = _DummyPytest()

from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# StatusBanner rendering
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_status_banner_renders_info():
    """StatusBanner.render() with severity='info' produces HTML with aria-live."""
    from qwen3_tts.interface.ui.components import StatusBanner
    banner = StatusBanner()
    html = banner.render("All good", severity="info")
    assert 'role="status"' in html
    assert 'aria-live="polite"' in html
    assert "All good" in html


@pytest.mark.unit
def test_status_banner_renders_success():
    """StatusBanner.render() with severity='success' includes a visual indicator."""
    from qwen3_tts.interface.ui.components import StatusBanner
    banner = StatusBanner()
    html = banner.render("Saved!", severity="success")
    assert "Saved!" in html
    # Must include colour or icon indicating success (not just plain text)
    assert "success" in html.lower() or "green" in html.lower() or "<svg" in html.lower()


@pytest.mark.unit
def test_status_banner_renders_warning():
    """StatusBanner.render() with severity='warning' includes a visual indicator."""
    from qwen3_tts.interface.ui.components import StatusBanner
    banner = StatusBanner()
    html = banner.render("Server slow", severity="warning")
    assert "Server slow" in html
    assert "warning" in html.lower() or "orange" in html.lower() or "<svg" in html.lower()


@pytest.mark.unit
def test_status_banner_renders_error():
    """StatusBanner.render() with severity='error' includes a visual indicator."""
    from qwen3_tts.interface.ui.components import StatusBanner
    banner = StatusBanner()
    html = banner.render("Connection refused", severity="error")
    assert "Connection refused" in html
    assert "error" in html.lower() or "red" in html.lower() or "<svg" in html.lower()


@pytest.mark.unit
def test_status_banner_escapes_message():
    """StatusBanner.render() HTML-escapes user-supplied message content."""
    from qwen3_tts.interface.ui.components import StatusBanner
    banner = StatusBanner()
    xss = '<script>alert("xss")</script>'
    html = banner.render(xss, severity="info")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html or "alert" not in html


@pytest.mark.unit
def test_status_banner_empty_message_renders_placeholder():
    """StatusBanner.render() with empty string produces a valid HTML container."""
    from qwen3_tts.interface.ui.components import StatusBanner
    banner = StatusBanner()
    html = banner.render("", severity="info")
    # Must still produce a container so the Gradio component doesn't break
    assert "<" in html


@pytest.mark.unit
def test_status_banner_no_emoji():
    """StatusBanner.render() does not emit emoji characters for status indicators."""
    from qwen3_tts.interface.ui.components import StatusBanner
    banner = StatusBanner()
    for sev in ("info", "success", "warning", "error"):
        html = banner.render("test message", severity=sev)
        for emoji in ("✅", "❌", "⚠️", "ℹ️", "✓", "✗"):
            assert emoji not in html, f"Emoji {emoji!r} found in {sev} banner output"


# ---------------------------------------------------------------------------
# StatusBanner thread safety
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_status_banner_thread_safety():
    """Concurrent render() calls don't corrupt message state."""
    from qwen3_tts.interface.ui.components import StatusBanner
    banner = StatusBanner()
    results = []
    errors = []

    def worker(msg):
        try:
            html = banner.render(msg, severity="info")
            results.append(html)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(f"msg-{i}",)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"Thread errors: {errors}"
    assert len(results) == 20
    for r in results:
        assert isinstance(r, str) and len(r) > 0


# ---------------------------------------------------------------------------
# poll_model_loading_state
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_poll_model_loading_state_returns_loading():
    """poll_model_loading_state returns 'loading' when server signals loading."""
    from qwen3_tts.interface.ui.components import poll_model_loading_state

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": {
            "clone": {"loaded": False, "loading": True, "memory_mb": 0},
        }
    }

    with patch("qwen3_tts.interface.ui.components.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.components.is_server_running", return_value=True), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
        state = poll_model_loading_state("clone")

    assert state == "loading"


@pytest.mark.unit
def test_poll_model_loading_state_returns_loaded():
    """poll_model_loading_state returns 'loaded' when model is fully loaded."""
    from qwen3_tts.interface.ui.components import poll_model_loading_state

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": {
            "clone": {"loaded": True, "loading": False, "memory_mb": 3500},
        }
    }

    with patch("qwen3_tts.interface.ui.components.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.components.is_server_running", return_value=True), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
        state = poll_model_loading_state("clone")

    assert state == "loaded"


@pytest.mark.unit
def test_poll_model_loading_state_returns_not_loaded():
    """poll_model_loading_state returns 'not_loaded' when model is idle."""
    from qwen3_tts.interface.ui.components import poll_model_loading_state

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": {
            "clone": {"loaded": False, "loading": False, "memory_mb": 0},
        }
    }

    with patch("qwen3_tts.interface.ui.components.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.components.is_server_running", return_value=True), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
        state = poll_model_loading_state("clone")

    assert state == "not_loaded"


@pytest.mark.unit
def test_poll_model_loading_state_server_down():
    """poll_model_loading_state returns 'unknown' when server is not running."""
    from qwen3_tts.interface.ui.components import poll_model_loading_state

    with patch("qwen3_tts.interface.ui.components.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.components.is_server_running", return_value=False):
        state = poll_model_loading_state("clone")

    assert state == "unknown"


@pytest.mark.unit
def test_poll_model_loading_state_connection_error():
    """poll_model_loading_state returns 'unknown' on connection error."""
    from qwen3_tts.interface.ui.components import poll_model_loading_state

    with patch("qwen3_tts.interface.ui.components.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.components.is_server_running", return_value=True), \
         patch("qwen3_tts.core.http_client.server_request", side_effect=ConnectionError("refused")):
        state = poll_model_loading_state("clone")

    assert state == "unknown"


# ---------------------------------------------------------------------------
# format_status_display — no emoji
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_format_status_display_no_emoji():
    """format_status_display HTML output does not contain emoji characters."""
    from qwen3_tts.interface.ui.shared import format_status_display

    with patch("qwen3_tts.interface.ui.shared.get_server_status",
               return_value=("Connected", "1200MB", "Clone, Design", "MLX (8bit, 1.7B)")):
        html = format_status_display()

    for emoji in ("✅", "❌", "⚠️", "✓", "✗", "\U0001f7e2", "\U0001f534"):
        assert emoji not in html, f"Emoji {emoji!r} found in format_status_display output"


@pytest.mark.unit
def test_format_status_display_has_aria():
    """format_status_display HTML includes role or aria attributes for screen readers."""
    from qwen3_tts.interface.ui.shared import format_status_display

    with patch("qwen3_tts.interface.ui.shared.get_server_status",
               return_value=("Connected", "1200MB", "Clone", "MLX (8bit, 1.7B)")):
        html = format_status_display()

    assert "aria-" in html or 'role="' in html


# ---------------------------------------------------------------------------
# get_model_status_html — no emoji, uses SVG, reflects /models not config
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_get_model_status_html_no_emoji_loaded():
    """get_model_status_html for loaded model uses SVG not emoji."""
    from qwen3_tts.interface.ui.model_management import get_model_status_html

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": {"clone": {"loaded": True, "loading": False, "memory_mb": 3500}}
    }

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
        html = get_model_status_html("clone")

    for emoji in ("✅", "❌", "✓", "✗", "⚠️"):
        assert emoji not in html, f"Emoji {emoji!r} found in model status HTML"
    assert "<svg" in html or "aria-label" in html
    assert "Loaded" in html or "loaded" in html


@pytest.mark.unit
def test_get_model_status_html_shows_loading_state():
    """get_model_status_html shows 'Loading' when model is mid-download."""
    from qwen3_tts.interface.ui.model_management import get_model_status_html

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": {"clone": {"loaded": False, "loading": True, "memory_mb": 0}}
    }

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
        html = get_model_status_html("clone")

    assert "Loading" in html or "loading" in html.lower()


@pytest.mark.unit
def test_get_model_status_html_no_emoji_not_loaded():
    """get_model_status_html for not-loaded model uses SVG or text, not emoji."""
    from qwen3_tts.interface.ui.model_management import get_model_status_html

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": {"design": {"loaded": False, "loading": False, "memory_mb": 0}}
    }

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
        html = get_model_status_html("design")

    for emoji in ("✅", "❌", "✓", "✗"):
        assert emoji not in html, f"Emoji {emoji!r} found in model status HTML"
    assert "Not loaded" in html or "not loaded" in html.lower()


@pytest.mark.unit
def test_get_model_status_html_has_aria_label():
    """get_model_status_html output includes aria-label for colour-coded badges."""
    from qwen3_tts.interface.ui.model_management import get_model_status_html

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": {"clone": {"loaded": True, "loading": False, "memory_mb": 2500}}
    }

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
        html = get_model_status_html("clone")

    assert "aria-label" in html


@pytest.mark.unit
def test_get_model_status_html_reflects_models_endpoint_not_config():
    """get_model_status_html reflects live /models response, not config startup flag.

    Regression: previously the indicator showed 'Loaded' because the config
    had load_at_startup=True, even while the model was still downloading.
    """
    from qwen3_tts.interface.ui.model_management import get_model_status_html

    # Server says NOT loaded — model is still downloading
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": {"clone": {"loaded": False, "loading": True, "memory_mb": 0}}
    }
    # Config says startup=True — the old bug would show "Loaded" from this
    config = {"models": {"clone": {"load_at_startup": True}}}

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value=config), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
        html = get_model_status_html("clone")

    # Must NOT claim loaded when model is still downloading
    # "Loaded (" is the old stale pattern (e.g. "Loaded (2500MB)")
    assert "Loaded (" not in html
    # Must show loading/in-progress state
    assert "Loading" in html or "loading" in html.lower()
