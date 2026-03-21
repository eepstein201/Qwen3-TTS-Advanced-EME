#!/usr/bin/env python3
"""Tests for Colab-specific code paths.

Covers:
  - get_clipboard_text() in Colab → error + sys.exit
  - play_audio() in Colab → skips playback
  - open_file() in Colab → prints path only
  - build_ui_and_launch() in Colab → share=True, inbrowser=False
  - _find_available_port() in Colab → binds 0.0.0.0
  - Server binds 0.0.0.0 in Colab
  - CORS regex allows *.gradio.live in Colab

Note: IN_COLAB, IS_MACOS, IS_LINUX are imported inside functions from
qwen3_tts.core.config, so they must be patched at the source module level.

Run: pytest tests/test_colab_paths.py -v
"""
import os

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
import qwen3_tts.core.config as _cfg


def _set_platform(colab=False, macos=False, linux=False):
    """Context manager to set platform flags on the config module.

    Since IN_COLAB/IS_MACOS/IS_LINUX are imported inside functions from
    qwen3_tts.core.config, we must set them on the source module directly.
    """
    class _PlatformCtx:
        def __enter__(self):
            self._orig = (_cfg.IN_COLAB, _cfg.IS_MACOS, _cfg.IS_LINUX)
            _cfg.IN_COLAB = colab
            _cfg.IS_MACOS = macos
            _cfg.IS_LINUX = linux
            return self
        def __exit__(self, *args):
            _cfg.IN_COLAB, _cfg.IS_MACOS, _cfg.IS_LINUX = self._orig
    return _PlatformCtx()


# ---- get_clipboard_text in Colab ----

@pytest.mark.unit
def test_clipboard_colab_exits(capsys):
    """get_clipboard_text exits with error in Colab."""
    from qwen3_tts.interface.generate_helpers import get_clipboard_text

    with _set_platform(colab=True), pytest.raises(SystemExit):
        get_clipboard_text()

    captured = capsys.readouterr()
    assert "Clipboard not available" in captured.out


# ---- play_audio in Colab ----

@pytest.mark.unit
def test_play_audio_colab_skips():
    """play_audio skips playback silently in Colab."""
    from qwen3_tts.interface.generate_helpers import play_audio

    with _set_platform(colab=True, linux=True), \
         patch("subprocess.run") as mock_run:
        play_audio("/tmp/test.wav")
    mock_run.assert_not_called()


@pytest.mark.unit
def test_play_audio_macos():
    """play_audio calls afplay on macOS."""
    from qwen3_tts.interface.generate_helpers import play_audio

    with _set_platform(macos=True), \
         patch("subprocess.run") as mock_run:
        play_audio("/tmp/test.wav")
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == ["afplay", "/tmp/test.wav"]


@pytest.mark.unit
def test_play_audio_linux():
    """play_audio calls ffplay on Linux."""
    from qwen3_tts.interface.generate_helpers import play_audio

    with _set_platform(linux=True), \
         patch("subprocess.run") as mock_run:
        play_audio("/tmp/test.wav")
    mock_run.assert_called_once()
    assert "ffplay" in mock_run.call_args[0][0]


# ---- open_file in Colab ----

@pytest.mark.unit
def test_open_file_colab_prints_path(capsys):
    """open_file just prints the path in Colab."""
    from qwen3_tts.interface.generate_helpers import open_file

    with _set_platform(colab=True), \
         patch("subprocess.run") as mock_run:
        open_file("/tmp/output.wav")
    mock_run.assert_not_called()
    captured = capsys.readouterr()
    assert "/tmp/output.wav" in captured.out


@pytest.mark.unit
def test_open_file_macos():
    """open_file calls 'open' on macOS."""
    from qwen3_tts.interface.generate_helpers import open_file

    with _set_platform(macos=True), \
         patch("subprocess.run") as mock_run:
        open_file("/tmp/output.wav")
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == ["open", "/tmp/output.wav"]


# ---- build_ui_and_launch Colab paths ----

@pytest.mark.unit
def test_build_ui_and_launch_colab_share():
    """build_ui_and_launch uses share=True, inbrowser=False in Colab."""
    from qwen3_tts.interface.generate_server import build_ui_and_launch

    mock_demo = MagicMock()

    with _set_platform(colab=True), \
         patch("qwen3_tts.interface.ui.build_ui", return_value=mock_demo), \
         patch("qwen3_tts.interface.ui._find_available_port", return_value=7860), \
         patch.dict(os.environ, {}, clear=False):
        env = os.environ.pop("TTS_UI_SHARE", None)
        env2 = os.environ.pop("TTS_UI_NO_BROWSER", None)
        try:
            build_ui_and_launch({"ui": {"port": 7860}})
        finally:
            if env is not None:
                os.environ["TTS_UI_SHARE"] = env
            if env2 is not None:
                os.environ["TTS_UI_NO_BROWSER"] = env2

    mock_demo.launch.assert_called_once()
    call_kwargs = mock_demo.launch.call_args[1]
    assert call_kwargs["share"] is True
    assert call_kwargs["inbrowser"] is False


@pytest.mark.unit
def test_build_ui_and_launch_local():
    """build_ui_and_launch uses share=False, inbrowser=True locally."""
    from qwen3_tts.interface.generate_server import build_ui_and_launch

    mock_demo = MagicMock()

    with _set_platform(colab=False), \
         patch("qwen3_tts.interface.ui.build_ui", return_value=mock_demo), \
         patch("qwen3_tts.interface.ui._find_available_port", return_value=7860), \
         patch.dict(os.environ, {}, clear=False):
        env = os.environ.pop("TTS_UI_SHARE", None)
        env2 = os.environ.pop("TTS_UI_NO_BROWSER", None)
        try:
            build_ui_and_launch({"ui": {"port": 7860}})
        finally:
            if env is not None:
                os.environ["TTS_UI_SHARE"] = env
            if env2 is not None:
                os.environ["TTS_UI_NO_BROWSER"] = env2

    mock_demo.launch.assert_called_once()
    call_kwargs = mock_demo.launch.call_args[1]
    assert call_kwargs["share"] is False
    assert call_kwargs["inbrowser"] is True


@pytest.mark.unit
def test_build_ui_and_launch_no_port(capsys):
    """build_ui_and_launch prints error when no port available."""
    from qwen3_tts.interface.generate_server import build_ui_and_launch

    with _set_platform(colab=False), \
         patch("qwen3_tts.interface.ui.build_ui"), \
         patch("qwen3_tts.interface.ui._find_available_port", return_value=None):
        build_ui_and_launch({"ui": {"port": 7860}})

    captured = capsys.readouterr()
    assert "No available port" in captured.out


# ---- _find_available_port Colab bind address ----

@pytest.mark.unit
def test_find_available_port_colab_binds_all():
    """_find_available_port binds 0.0.0.0 in Colab."""
    with patch("qwen3_tts.interface.ui._facade.IN_COLAB", True):
        from qwen3_tts.interface.ui._facade import _find_available_port
        port = _find_available_port(18900)
        assert port is not None
        assert isinstance(port, int)


@pytest.mark.unit
def test_find_available_port_local_binds_localhost():
    """_find_available_port binds 127.0.0.1 when not in Colab."""
    with patch("qwen3_tts.interface.ui._facade.IN_COLAB", False):
        from qwen3_tts.interface.ui._facade import _find_available_port
        port = _find_available_port(18910)
        assert port is not None
        assert isinstance(port, int)


# ---- get_device Colab ----

@pytest.mark.unit
def test_get_device_colab_cuda():
    """get_device returns 'cuda' in Colab with CUDA_VISIBLE_DEVICES set."""
    with _set_platform(colab=True, linux=True):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            assert _cfg.get_device() == "cuda"


@pytest.mark.unit
def test_get_device_colab_nvidia_device():
    """get_device returns 'cuda' in Colab with /dev/nvidia0."""
    with _set_platform(colab=True, linux=True):
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        with patch.dict(os.environ, env, clear=True), \
             patch("os.path.exists", return_value=True):
            assert _cfg.get_device() == "cuda"


@pytest.mark.unit
def test_get_device_colab_no_gpu():
    """get_device returns 'cpu' in Colab without GPU indicators."""
    with _set_platform(colab=True, linux=True):
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        with patch.dict(os.environ, env, clear=True), \
             patch("os.path.exists", return_value=False):
            assert _cfg.get_device() == "cpu"


# ---- CORS in Colab ----

@pytest.mark.unit
def test_cors_regex_colab_allows_gradio_live():
    """CORS regex in Colab mode allows *.gradio.live origins."""
    import re
    colab_regex = (
        r"(^https?://(localhost|127\.0\.0\.1)(:\d+)?$)"
        r"|(^https://[a-z0-9-]+\.gradio\.live$)"
    )
    assert re.match(colab_regex, "https://abc123-def.gradio.live")
    assert re.match(colab_regex, "http://127.0.0.1:5123")
    assert not re.match(colab_regex, "https://evil.example.com")


@pytest.mark.unit
def test_cors_regex_local_rejects_gradio_live():
    """CORS regex in local mode does NOT allow *.gradio.live."""
    import re
    local_regex = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    assert not re.match(local_regex, "https://abc123.gradio.live")
    assert re.match(local_regex, "http://localhost:5123")
    assert re.match(local_regex, "http://127.0.0.1:7860")
