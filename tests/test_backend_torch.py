#!/usr/bin/env python3
"""Tests for torch backend code paths.

Covers:
  - voice_prompt_exists() with torch backend (.pt files)
  - list_voice_prompts() returns .pt files
  - Backend detection on non-Apple Silicon
  - Model name resolution (torch vs MLX prefixes)
  - get_device() for various platforms
  - get_backend() env var override and config fallback

Run: pytest tests/test_backend_torch.py -v
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

from unittest.mock import patch

# ---- voice_prompt_exists (torch backend) ----

@pytest.mark.unit
def test_voice_prompt_exists_torch_pt_found(tmp_path):
    """voice_prompt_exists returns True when .pt file exists (torch backend)."""
    from qwen3_tts.interface.generate_helpers import voice_prompt_exists

    pt_file = tmp_path / "my_voice.pt"
    pt_file.write_bytes(b"fake")

    with patch("qwen3_tts.interface.generate_helpers.get_backend", return_value="torch"), \
         patch("qwen3_tts.interface.generate_helpers.VOICE_PROMPTS_DIR", str(tmp_path)):
        assert voice_prompt_exists("my_voice.pt") is True


@pytest.mark.unit
def test_voice_prompt_exists_torch_pt_missing(tmp_path):
    """voice_prompt_exists returns False when .pt file missing (torch backend)."""
    from qwen3_tts.interface.generate_helpers import voice_prompt_exists

    with patch("qwen3_tts.interface.generate_helpers.get_backend", return_value="torch"), \
         patch("qwen3_tts.interface.generate_helpers.VOICE_PROMPTS_DIR", str(tmp_path)):
        assert voice_prompt_exists("missing.pt") is False


@pytest.mark.unit
def test_voice_prompt_exists_mlx_pair_found(tmp_path):
    """voice_prompt_exists returns True when .wav+.txt pair exists (mlx backend)."""
    from qwen3_tts.interface.generate_helpers import voice_prompt_exists

    (tmp_path / "my_voice.wav").write_bytes(b"audio")
    (tmp_path / "my_voice.txt").write_text("transcript")

    with patch("qwen3_tts.interface.generate_helpers.get_backend", return_value="mlx"), \
         patch("qwen3_tts.interface.generate_helpers.VOICE_PROMPTS_DIR", str(tmp_path)):
        assert voice_prompt_exists("my_voice.pt") is True


@pytest.mark.unit
def test_voice_prompt_exists_mlx_missing_txt(tmp_path):
    """voice_prompt_exists returns False when .txt missing (mlx backend)."""
    from qwen3_tts.interface.generate_helpers import voice_prompt_exists

    (tmp_path / "my_voice.wav").write_bytes(b"audio")

    with patch("qwen3_tts.interface.generate_helpers.get_backend", return_value="mlx"), \
         patch("qwen3_tts.interface.generate_helpers.VOICE_PROMPTS_DIR", str(tmp_path)):
        assert voice_prompt_exists("my_voice.pt") is False


# ---- list_voice_prompts ----

@pytest.mark.unit
def test_list_voice_prompts_torch_only(tmp_path):
    """list_voice_prompts returns .pt files."""
    from qwen3_tts.interface.generate_helpers import list_voice_prompts

    (tmp_path / "voice_a.pt").write_bytes(b"data")
    (tmp_path / "voice_b.pt").write_bytes(b"data")
    (tmp_path / "notes.txt").write_text("ignore")

    with patch("qwen3_tts.interface.generate_helpers.VOICE_PROMPTS_DIR", str(tmp_path)):
        prompts = list_voice_prompts()
    assert "voice_a.pt" in prompts
    assert "voice_b.pt" in prompts
    assert "notes.txt" not in prompts


@pytest.mark.unit
def test_list_voice_prompts_mlx_pairs(tmp_path):
    """list_voice_prompts returns .wav files with matching .txt."""
    from qwen3_tts.interface.generate_helpers import list_voice_prompts

    (tmp_path / "voice_a.wav").write_bytes(b"audio")
    (tmp_path / "voice_a.txt").write_text("transcript")
    (tmp_path / "orphan.wav").write_bytes(b"audio")  # no matching .txt

    with patch("qwen3_tts.interface.generate_helpers.VOICE_PROMPTS_DIR", str(tmp_path)):
        prompts = list_voice_prompts()
    assert "voice_a.wav" in prompts
    assert "orphan.wav" not in prompts


@pytest.mark.unit
def test_list_voice_prompts_mixed(tmp_path):
    """list_voice_prompts returns both .pt and valid .wav+.txt pairs."""
    from qwen3_tts.interface.generate_helpers import list_voice_prompts

    (tmp_path / "torch_voice.pt").write_bytes(b"data")
    (tmp_path / "mlx_voice.wav").write_bytes(b"audio")
    (tmp_path / "mlx_voice.txt").write_text("transcript")

    with patch("qwen3_tts.interface.generate_helpers.VOICE_PROMPTS_DIR", str(tmp_path)):
        prompts = list_voice_prompts()
    assert "torch_voice.pt" in prompts
    assert "mlx_voice.wav" in prompts


@pytest.mark.unit
def test_list_voice_prompts_empty_dir(tmp_path):
    """list_voice_prompts returns empty list when dir is empty."""
    from qwen3_tts.interface.generate_helpers import list_voice_prompts

    with patch("qwen3_tts.interface.generate_helpers.VOICE_PROMPTS_DIR", str(tmp_path)):
        assert list_voice_prompts() == []


@pytest.mark.unit
def test_list_voice_prompts_missing_dir():
    """list_voice_prompts returns empty list when dir doesn't exist."""
    from qwen3_tts.interface.generate_helpers import list_voice_prompts

    with patch("qwen3_tts.interface.generate_helpers.VOICE_PROMPTS_DIR", "/nonexistent"):
        assert list_voice_prompts() == []


# ---- get_backend ----

@pytest.mark.unit
def test_get_backend_env_override():
    """TTS_BACKEND env var overrides config."""
    from qwen3_tts.core.config import get_backend

    with patch.dict(os.environ, {"TTS_BACKEND": "torch"}):
        assert get_backend() == "torch"


@pytest.mark.unit
def test_get_backend_env_override_mlx():
    """TTS_BACKEND=mlx is respected."""
    from qwen3_tts.core.config import get_backend

    with patch.dict(os.environ, {"TTS_BACKEND": "mlx"}):
        assert get_backend() == "mlx"


@pytest.mark.unit
def test_get_backend_invalid_env_ignored():
    """Invalid TTS_BACKEND env var falls through to config."""
    from qwen3_tts.core.config import get_backend

    with patch.dict(os.environ, {"TTS_BACKEND": "invalid_backend"}), \
         patch("qwen3_tts.core.config.load_config", return_value={"advanced": {"backend": "torch"}}):
        assert get_backend() == "torch"


@pytest.mark.unit
def test_get_backend_config_fallback():
    """get_backend reads from config when no env var set."""
    from qwen3_tts.core.config import get_backend

    with patch.dict(os.environ, {}, clear=False), \
         patch("qwen3_tts.core.config.load_config", return_value={"advanced": {"backend": "torch"}}):
        env = os.environ.pop("TTS_BACKEND", None)
        try:
            assert get_backend() == "torch"
        finally:
            if env is not None:
                os.environ["TTS_BACKEND"] = env


# ---- get_device ----

@pytest.mark.unit
def test_get_device_colab_with_gpu():
    """get_device returns 'cuda' in Colab with GPU."""
    import qwen3_tts.core.config as cfg
    orig_colab = cfg.IN_COLAB
    orig_linux = cfg.IS_LINUX
    try:
        cfg.IN_COLAB = True
        cfg.IS_LINUX = True
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            assert cfg.get_device() == "cuda"
    finally:
        cfg.IN_COLAB = orig_colab
        cfg.IS_LINUX = orig_linux


@pytest.mark.unit
def test_get_device_linux_no_gpu():
    """get_device returns 'cpu' on Linux without GPU."""
    import qwen3_tts.core.config as cfg
    orig_colab = cfg.IN_COLAB
    orig_linux = cfg.IS_LINUX
    orig_macos = cfg.IS_MACOS
    try:
        cfg.IN_COLAB = False
        cfg.IS_LINUX = True
        cfg.IS_MACOS = False
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        with patch.dict(os.environ, env, clear=True), \
             patch("os.path.exists", return_value=False):
            assert cfg.get_device() == "cpu"
    finally:
        cfg.IN_COLAB = orig_colab
        cfg.IS_LINUX = orig_linux
        cfg.IS_MACOS = orig_macos


@pytest.mark.unit
def test_get_device_macos_arm64():
    """get_device returns 'mps' on macOS ARM64."""
    import qwen3_tts.core.config as cfg
    orig_colab = cfg.IN_COLAB
    orig_linux = cfg.IS_LINUX
    orig_macos = cfg.IS_MACOS
    try:
        cfg.IN_COLAB = False
        cfg.IS_LINUX = False
        cfg.IS_MACOS = True
        with patch("platform.machine", return_value="arm64"):
            assert cfg.get_device() == "mps"
    finally:
        cfg.IN_COLAB = orig_colab
        cfg.IS_LINUX = orig_linux
        cfg.IS_MACOS = orig_macos


# ---- Model name resolution ----

@pytest.mark.unit
def test_torch_model_name_clone_17b():
    """get_torch_model_name returns correct repo ID for clone 1.7B."""
    from qwen3_tts.core.config import get_torch_model_name

    with patch("qwen3_tts.core.config.get_model_size", return_value="1.7B"):
        name = get_torch_model_name("clone")
    assert name == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


@pytest.mark.unit
def test_torch_model_name_design_06b():
    """get_torch_model_name returns correct repo ID for design 0.6B."""
    from qwen3_tts.core.config import get_torch_model_name

    with patch("qwen3_tts.core.config.get_model_size", return_value="0.6B"):
        name = get_torch_model_name("design")
    assert name == "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign"


@pytest.mark.unit
def test_torch_model_name_custom_17b():
    """get_torch_model_name returns correct repo ID for custom 1.7B."""
    from qwen3_tts.core.config import get_torch_model_name

    with patch("qwen3_tts.core.config.get_model_size", return_value="1.7B"):
        name = get_torch_model_name("custom")
    assert name == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


@pytest.mark.unit
def test_mlx_model_name_clone_8bit():
    """get_mlx_model_name returns correct repo ID with quantization."""
    from qwen3_tts.core.config import get_mlx_model_name

    with patch("qwen3_tts.core.config.get_model_size", return_value="1.7B"), \
         patch("qwen3_tts.core.config.get_mlx_quantization", return_value="8bit"):
        name = get_mlx_model_name("clone")
    assert name == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"


@pytest.mark.unit
def test_mlx_model_name_design_4bit():
    """get_mlx_model_name returns correct repo ID with 4bit quantization."""
    from qwen3_tts.core.config import get_mlx_model_name

    with patch("qwen3_tts.core.config.get_model_size", return_value="0.6B"), \
         patch("qwen3_tts.core.config.get_mlx_quantization", return_value="4bit"):
        name = get_mlx_model_name("design")
    assert name == "mlx-community/Qwen3-TTS-12Hz-0.6B-VoiceDesign-4bit"


@pytest.mark.unit
def test_torch_model_name_invalid_type():
    """get_torch_model_name raises ValueError for unknown model type."""
    from qwen3_tts.core.config import get_torch_model_name

    with patch("qwen3_tts.core.config.get_model_size", return_value="1.7B"):
        with pytest.raises(ValueError, match="Unknown model type"):
            get_torch_model_name("invalid")


@pytest.mark.unit
def test_torch_model_name_invalid_size():
    """get_torch_model_name raises ValueError for unknown model size."""
    from qwen3_tts.core.config import get_torch_model_name

    with patch("qwen3_tts.core.config.get_model_size", return_value="99B"):
        with pytest.raises(ValueError, match="Unknown model size"):
            get_torch_model_name("clone")


# ---- get_model_size ----

@pytest.mark.unit
def test_get_model_size_env_override():
    """TTS_MODEL_SIZE env var overrides config."""
    from qwen3_tts.core.config import get_model_size

    with patch.dict(os.environ, {"TTS_MODEL_SIZE": "0.6B"}):
        assert get_model_size() == "0.6B"


@pytest.mark.unit
def test_get_model_size_invalid_env():
    """Invalid TTS_MODEL_SIZE env var falls through to config."""
    from qwen3_tts.core.config import get_model_size

    with patch.dict(os.environ, {"TTS_MODEL_SIZE": "99B"}), \
         patch("qwen3_tts.core.config.load_config", return_value={"advanced": {"model_size": "0.6B"}}):
        assert get_model_size() == "0.6B"


# ---- get_model_info ----

@pytest.mark.unit
def test_get_model_info_torch():
    """get_model_info returns torch MODEL_INFO dict when backend is torch."""
    from qwen3_tts.core.config import get_model_info

    with patch("qwen3_tts.core.config.get_model_size", return_value="1.7B"), \
         patch("qwen3_tts.core.config.get_backend", return_value="torch"):
        info = get_model_info("clone")
    assert info["name"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert "memory_mb" in info


@pytest.mark.unit
def test_get_model_info_mlx():
    """get_model_info returns MLX_MODEL_INFO dict when backend is mlx."""
    from qwen3_tts.core.config import get_model_info

    with patch("qwen3_tts.core.config.get_model_size", return_value="1.7B"), \
         patch("qwen3_tts.core.config.get_backend", return_value="mlx"):
        info = get_model_info("clone")
    assert "name_template" in info
    assert "memory_mb" in info


@pytest.mark.unit
def test_get_model_info_unknown_type():
    """get_model_info returns empty dict for unknown model type."""
    from qwen3_tts.core.config import get_model_info

    with patch("qwen3_tts.core.config.get_model_size", return_value="1.7B"), \
         patch("qwen3_tts.core.config.get_backend", return_value="torch"):
        info = get_model_info("nonexistent")
    assert info == {}
