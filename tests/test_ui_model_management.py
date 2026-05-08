#!/usr/bin/env python3
"""Tests for UI model management module.

Covers:
  - get_model_table_data(): server running, not running, error
  - toggle_model(): load/unload success, error, server down
  - toggle_asr(): load/unload success, error
  - update_startup_defaults(): saves config correctly
  - get_model_status_html(): loaded, not loaded, error
  - get_audio_loader_setting() / set_audio_loader_setting()

Run: pytest tests/test_ui_model_management.py -v
"""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

    class _DummyMarkerFunc:
        def __init__(self, name=None):
            self._name = name
        def __call__(self, condition, **kwargs):
            return lambda f: f

    class _DummyMark:
        def __getattr__(self, name):
            return _DummyMarkerFunc()

    class _DummyPytest:
        mark = _DummyMark()

    pytest = _DummyPytest()

from unittest.mock import patch, MagicMock


# ---- get_model_table_data ----

@pytest.mark.unit
def test_get_model_table_data_server_not_running():
    """get_model_table_data returns 'server not running' when server is down."""
    from qwen3_tts.interface.ui.model_management import get_model_table_data

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=False):
        rows = get_model_table_data()

    assert len(rows) == 4  # clone, design, custom, asr
    for row in rows:
        assert "server not running" in row[1]


@pytest.mark.unit
def test_get_model_table_data_server_running():
    """get_model_table_data returns model statuses when server responds."""
    from qwen3_tts.interface.ui.model_management import get_model_table_data

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": {
            "clone": {"loaded": True, "memory_mb": 3500},
            "design": {"loaded": False, "memory_mb": 0},
            "custom": {"loaded": True, "memory_mb": 2500},
        }
    }
    config = {"models": {
        "clone": {"load_at_startup": True},
        "design": {"load_at_startup": False},
        "custom": {"load_at_startup": False},
    }}

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value=config), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.interface.ui.model_management.get_server_url", return_value="http://127.0.0.1:5123"), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
        rows = get_model_table_data()

    assert len(rows) == 4  # clone, design, custom, asr
    assert rows[0][0] == "clone"
    assert "Loaded" in rows[0][1]
    assert rows[1][0] == "design"
    assert "Not loaded" in rows[1][1]
    assert rows[3][0] == "asr"


@pytest.mark.unit
def test_get_model_table_data_server_error():
    """get_model_table_data returns error rows on non-200 response."""
    from qwen3_tts.interface.ui.model_management import get_model_table_data

    mock_resp = MagicMock()
    mock_resp.status_code = 500

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.interface.ui.model_management.get_server_url", return_value="http://127.0.0.1:5123"), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
        rows = get_model_table_data()

    assert len(rows) == 4  # clone, design, custom, asr
    assert "error" in rows[0][1]


@pytest.mark.unit
def test_get_model_table_data_connection_error():
    """get_model_table_data handles connection errors gracefully."""
    from qwen3_tts.interface.ui.model_management import get_model_table_data

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.interface.ui.model_management.get_server_url", return_value="http://127.0.0.1:5123"), \
         patch("qwen3_tts.core.http_client.server_request", side_effect=ConnectionError("refused")):
        rows = get_model_table_data()

    assert len(rows) == 4  # clone, design, custom, asr
    assert "error" in rows[0][1].lower()


# ---- toggle_model ----

@pytest.mark.unit
def test_toggle_model_load_success():
    """toggle_model load returns success message."""
    from qwen3_tts.interface.ui.model_management import toggle_model

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "loaded"}

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.interface.ui.model_management.get_server_url", return_value="http://127.0.0.1:5123"), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
         patch("qwen3_tts.interface.ui.model_management.get_model_table_data", return_value=[]), \
         patch("qwen3_tts.interface.ui.model_management.format_status_display", return_value="ok"):
        msg, table, status = toggle_model("clone", "load")

    assert "loaded" in msg
    assert "clone" in msg


@pytest.mark.unit
def test_toggle_model_server_down():
    """toggle_model returns 'Server not running' when down."""
    from qwen3_tts.interface.ui.model_management import toggle_model

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=False), \
         patch("qwen3_tts.interface.ui.model_management.get_model_table_data", return_value=[]), \
         patch("qwen3_tts.interface.ui.model_management.format_status_display", return_value=""):
        msg, table, status = toggle_model("clone", "load")

    assert "not running" in msg.lower()


@pytest.mark.unit
def test_toggle_model_unload_success():
    """toggle_model unload returns success message."""
    from qwen3_tts.interface.ui.model_management import toggle_model

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "unloaded"}

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.interface.ui.model_management.get_server_url", return_value="http://127.0.0.1:5123"), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
         patch("qwen3_tts.interface.ui.model_management.get_model_table_data", return_value=[]), \
         patch("qwen3_tts.interface.ui.model_management.format_status_display", return_value="ok"):
        msg, table, status = toggle_model("design", "unload")

    assert "unloaded" in msg


@pytest.mark.unit
def test_toggle_model_error_response():
    """toggle_model returns error message on non-200."""
    from qwen3_tts.interface.ui.model_management import toggle_model

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {"error": "internal"}

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.interface.ui.model_management.get_server_url", return_value="http://127.0.0.1:5123"), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
         patch("qwen3_tts.interface.ui.model_management.get_model_table_data", return_value=[]), \
         patch("qwen3_tts.interface.ui.model_management.format_status_display", return_value=""):
        msg, table, status = toggle_model("clone", "load")

    assert "Failed" in msg


# ---- toggle_asr ----

@pytest.mark.unit
def test_toggle_asr_load_success():
    """toggle_asr load returns success."""
    from qwen3_tts.interface.ui.model_management import toggle_asr

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "loaded"}

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.interface.ui.model_management.get_server_url", return_value="http://127.0.0.1:5123"), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp), \
         patch("qwen3_tts.interface.ui.model_management.format_status_display", return_value="ok"):
        msg, status = toggle_asr("load")

    assert "loaded" in msg


@pytest.mark.unit
def test_toggle_asr_server_down():
    """toggle_asr returns error when server is down."""
    from qwen3_tts.interface.ui.model_management import toggle_asr

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=False), \
         patch("qwen3_tts.interface.ui.model_management.format_status_display", return_value=""):
        msg, status = toggle_asr("load")

    assert "not running" in msg.lower()


# ---- update_startup_defaults ----

@pytest.mark.unit
def test_update_startup_defaults(tmp_path):
    """update_startup_defaults saves config correctly."""
    from qwen3_tts.interface.ui.model_management import update_startup_defaults

    saved = {}

    def mock_save(cfg):
        saved.update(cfg)

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.save_config", side_effect=mock_save), \
         patch("qwen3_tts.interface.ui.model_management.get_model_table_data", return_value=[]):
        msg, table = update_startup_defaults(True, False, True)

    assert "updated" in msg.lower()
    assert saved["models"]["clone"]["load_at_startup"] is True
    assert saved["models"]["design"]["load_at_startup"] is False
    assert saved["models"]["custom"]["load_at_startup"] is True


# ---- get_model_status_html ----

@pytest.mark.unit
def test_get_model_status_html_loaded():
    """get_model_status_html shows green loaded status."""
    from qwen3_tts.interface.ui.model_management import get_model_status_html

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": {"clone": {"loaded": True, "memory_mb": 3500}}
    }

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.interface.ui.model_management.get_server_url", return_value="http://127.0.0.1:5123"), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
        html = get_model_status_html("clone")

    assert "Loaded" in html
    assert "3500" in html


@pytest.mark.unit
def test_get_model_status_html_not_loaded():
    """get_model_status_html shows gray not-loaded status."""
    from qwen3_tts.interface.ui.model_management import get_model_status_html

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": {"design": {"loaded": False, "memory_mb": 0}}
    }

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=True), \
         patch("qwen3_tts.interface.ui.model_management.get_server_url", return_value="http://127.0.0.1:5123"), \
         patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
        html = get_model_status_html("design")

    assert "Not loaded" in html


@pytest.mark.unit
def test_get_model_status_html_server_down():
    """get_model_status_html shows gray when server is down."""
    from qwen3_tts.interface.ui.model_management import get_model_status_html

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}), \
         patch("qwen3_tts.interface.ui.model_management.is_server_running", return_value=False):
        html = get_model_status_html("clone")

    assert "not running" in html.lower()


# ---- get_audio_loader_setting / set_audio_loader_setting ----

@pytest.mark.unit
def test_get_audio_loader_setting_default():
    """get_audio_loader_setting returns 'torchaudio' by default."""
    from qwen3_tts.interface.ui.model_management import get_audio_loader_setting

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={}):
        assert get_audio_loader_setting() == "torchaudio"


@pytest.mark.unit
def test_get_audio_loader_setting_librosa():
    """get_audio_loader_setting returns configured value."""
    from qwen3_tts.interface.ui.model_management import get_audio_loader_setting

    config = {"advanced": {"audio_loader": "librosa"}}
    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value=config):
        assert get_audio_loader_setting() == "librosa"


@pytest.mark.unit
def test_set_audio_loader_setting():
    """set_audio_loader_setting saves config and returns message."""
    from qwen3_tts.interface.ui.model_management import set_audio_loader_setting

    saved = {}

    def mock_save(cfg):
        saved.update(cfg)

    with patch("qwen3_tts.interface.ui.model_management.load_config", return_value={"advanced": {}}), \
         patch("qwen3_tts.interface.ui.model_management.save_config", side_effect=mock_save):
        msg = set_audio_loader_setting("librosa")

    assert "librosa" in msg
    assert saved["advanced"]["audio_loader"] == "librosa"
