#!/usr/bin/env python3
"""Tests for model_cache.py command functions and uncovered paths.

Covers:
  - _get_model_access_time(): file-based lookup, fallback to mtime, missing dir
  - _get_model_info() MLX parsing: 4bit, 8bit, bf16, unknown quant
  - list_models_cmd(): with models, no models
  - get_size_cmd(): with models, no models
  - prune_models_cmd(): dry run, no old models, confirmed, cancelled
  - clear_cache_cmd(): force, cancelled, no models, OSError
  - main(): argument routing

Run: pytest tests/test_model_cache_commands.py -v
"""
import os
import sys

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pathlib


# ---- _get_model_access_time ----

@pytest.mark.unit
def test_get_model_access_time_pytorch_model_bin(tmp_path):
    """Returns atime of pytorch_model.bin when present."""
    from qwen3_tts.tools.model_cache import _get_model_access_time

    model_file = tmp_path / "pytorch_model.bin"
    model_file.write_bytes(b"fake")
    result = _get_model_access_time(tmp_path)
    assert isinstance(result, datetime)
    assert result != datetime.min


@pytest.mark.unit
def test_get_model_access_time_config_json(tmp_path):
    """Returns atime of config.json when pytorch_model.bin missing."""
    from qwen3_tts.tools.model_cache import _get_model_access_time

    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    result = _get_model_access_time(tmp_path)
    assert isinstance(result, datetime)
    assert result != datetime.min


@pytest.mark.unit
def test_get_model_access_time_model_safetensors(tmp_path):
    """Returns atime of model.safetensors when it's the only known file."""
    from qwen3_tts.tools.model_cache import _get_model_access_time

    st_file = tmp_path / "model.safetensors"
    st_file.write_bytes(b"fake")
    result = _get_model_access_time(tmp_path)
    assert result != datetime.min


@pytest.mark.unit
def test_get_model_access_time_fallback_to_dir_mtime(tmp_path):
    """Falls back to directory mtime when no known files exist."""
    from qwen3_tts.tools.model_cache import _get_model_access_time

    result = _get_model_access_time(tmp_path)
    assert isinstance(result, datetime)
    assert result != datetime.min


@pytest.mark.unit
def test_get_model_access_time_missing_dir():
    """Returns datetime.min for non-existent directory."""
    from qwen3_tts.tools.model_cache import _get_model_access_time

    result = _get_model_access_time(pathlib.Path("/nonexistent/path/abc123"))
    assert result == datetime.min


# ---- _get_model_info MLX parsing ----

@pytest.mark.unit
def test_get_model_info_mlx_4bit():
    """Parses MLX 4bit quantization from model name."""
    from qwen3_tts.tools.model_cache import _get_model_info

    mock_path = MagicMock(spec=pathlib.Path)
    mock_path.name = "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-4bit-something"

    with patch('qwen3_tts.tools.model_cache._get_model_dir_size', return_value=2500000000), \
         patch('qwen3_tts.tools.model_cache._get_model_access_time', return_value=datetime.now()):
        result = _get_model_info(mock_path)

    assert result["backend"] == "mlx"
    assert result["quantization"] == "4bit"
    assert result["model_type"] == "clone"
    assert result["model_size"] == "1.7B"


@pytest.mark.unit
def test_get_model_info_mlx_8bit():
    """Parses MLX 8bit quantization."""
    from qwen3_tts.tools.model_cache import _get_model_info

    mock_path = MagicMock(spec=pathlib.Path)
    mock_path.name = "models--mlx-community--Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit-foo"

    with patch('qwen3_tts.tools.model_cache._get_model_dir_size', return_value=1000000000), \
         patch('qwen3_tts.tools.model_cache._get_model_access_time', return_value=datetime.now()):
        result = _get_model_info(mock_path)

    assert result["backend"] == "mlx"
    assert result["quantization"] == "8bit"
    assert result["model_type"] == "design"
    assert result["model_size"] == "0.6B"


@pytest.mark.unit
def test_get_model_info_mlx_bf16():
    """Parses MLX bf16 quantization."""
    from qwen3_tts.tools.model_cache import _get_model_info

    mock_path = MagicMock(spec=pathlib.Path)
    mock_path.name = "models--mlx-community--Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16-bar"

    with patch('qwen3_tts.tools.model_cache._get_model_dir_size', return_value=3000000000), \
         patch('qwen3_tts.tools.model_cache._get_model_access_time', return_value=datetime.now()):
        result = _get_model_info(mock_path)

    assert result["backend"] == "mlx"
    assert result["quantization"] == "bf16"
    assert result["model_type"] == "custom"


@pytest.mark.unit
def test_get_model_info_mlx_unknown_quant():
    """Parses MLX with unknown quantization."""
    from qwen3_tts.tools.model_cache import _get_model_info

    mock_path = MagicMock(spec=pathlib.Path)
    mock_path.name = "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-experimental"

    with patch('qwen3_tts.tools.model_cache._get_model_dir_size', return_value=2000000000), \
         patch('qwen3_tts.tools.model_cache._get_model_access_time', return_value=datetime.now()):
        result = _get_model_info(mock_path)

    assert result["backend"] == "mlx"
    assert result["quantization"] == "unknown"


@pytest.mark.unit
def test_get_model_info_unrecognized_model():
    """Returns None backend for unrecognized model names."""
    from qwen3_tts.tools.model_cache import _get_model_info

    mock_path = MagicMock(spec=pathlib.Path)
    mock_path.name = "models--other--SomeModel"

    with patch('qwen3_tts.tools.model_cache._get_model_dir_size', return_value=100), \
         patch('qwen3_tts.tools.model_cache._get_model_access_time', return_value=datetime.now()):
        result = _get_model_info(mock_path)

    assert result["backend"] is None
    assert result["model_type"] is None


# ---- list_models_cmd ----

@pytest.mark.unit
def test_list_models_cmd_no_models(capsys):
    """list_models_cmd prints 'no models' when cache empty."""
    from qwen3_tts.tools.model_cache import list_models_cmd

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=[]):
        list_models_cmd()

    captured = capsys.readouterr()
    assert "No TTS models found" in captured.out


@pytest.mark.unit
def test_list_models_cmd_with_models(capsys):
    """list_models_cmd prints formatted table."""
    from qwen3_tts.tools.model_cache import list_models_cmd

    models = [{
        "name": "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base",
        "model_type": "clone",
        "model_size": "1.7B",
        "backend": "torch",
        "last_access": datetime(2026, 3, 15, 10, 30),
        "size_formatted": "3.5 GB",
        "size_bytes": 3500000000,
    }]

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=models), \
         patch('qwen3_tts.tools.model_cache.get_total_size', return_value=3500000000):
        list_models_cmd()

    captured = capsys.readouterr()
    assert "Found 1 cached TTS model" in captured.out
    assert "clone" in captured.out
    assert "torch" in captured.out


@pytest.mark.unit
def test_list_models_cmd_datetime_min(capsys):
    """list_models_cmd shows 'unknown' for datetime.min access time."""
    from qwen3_tts.tools.model_cache import list_models_cmd

    models = [{
        "name": "test",
        "model_type": "clone",
        "model_size": "1.7B",
        "backend": "torch",
        "last_access": datetime.min,
        "size_formatted": "1.0 GB",
        "size_bytes": 1000000000,
    }]

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=models), \
         patch('qwen3_tts.tools.model_cache.get_total_size', return_value=1000000000):
        list_models_cmd()

    captured = capsys.readouterr()
    assert "unknown" in captured.out


# ---- get_size_cmd ----

@pytest.mark.unit
def test_get_size_cmd_no_models(capsys):
    """get_size_cmd prints 'no models' when cache empty."""
    from qwen3_tts.tools.model_cache import get_size_cmd

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=[]):
        get_size_cmd()

    captured = capsys.readouterr()
    assert "No TTS models found" in captured.out


@pytest.mark.unit
def test_get_size_cmd_with_models(capsys):
    """get_size_cmd shows size breakdown by backend."""
    from qwen3_tts.tools.model_cache import get_size_cmd

    models = [
        {"backend": "torch", "size_bytes": 3500000000},
        {"backend": "mlx", "size_bytes": 2500000000},
    ]

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=models), \
         patch('qwen3_tts.tools.model_cache.get_total_size', return_value=6000000000):
        get_size_cmd()

    captured = capsys.readouterr()
    assert "PyTorch models" in captured.out
    assert "MLX models" in captured.out
    assert "1 PyTorch" in captured.out
    assert "1 MLX" in captured.out


# ---- prune_models_cmd ----

@pytest.mark.unit
def test_prune_models_cmd_no_old_models(capsys):
    """prune_models_cmd prints 'no models' when none are old enough."""
    from qwen3_tts.tools.model_cache import prune_models_cmd

    models = [{"last_access": datetime.now(), "name": "recent", "size_bytes": 1000}]

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=models):
        prune_models_cmd(days=30, dry_run=False)

    captured = capsys.readouterr()
    assert "No models found unused" in captured.out


@pytest.mark.unit
def test_prune_models_cmd_dry_run(capsys):
    """prune_models_cmd dry run shows what would be removed."""
    from qwen3_tts.tools.model_cache import prune_models_cmd

    old_time = datetime.now() - timedelta(days=60)
    models = [{"last_access": old_time, "name": "old-model", "size_bytes": 3000000000, "path": "/fake/path"}]

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=models):
        prune_models_cmd(days=30, dry_run=True)

    captured = capsys.readouterr()
    assert "old-model" in captured.out
    assert "Dry run" in captured.out


@pytest.mark.unit
def test_prune_models_cmd_cancelled(capsys):
    """prune_models_cmd cancelled by user."""
    from qwen3_tts.tools.model_cache import prune_models_cmd

    old_time = datetime.now() - timedelta(days=60)
    models = [{"last_access": old_time, "name": "old-model", "size_bytes": 1000, "path": "/fake/path"}]

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=models), \
         patch('builtins.input', return_value="n"):
        prune_models_cmd(days=30, dry_run=False)

    captured = capsys.readouterr()
    assert "Cancelled" in captured.out


@pytest.mark.unit
def test_prune_models_cmd_confirmed(capsys, tmp_path):
    """prune_models_cmd deletes when confirmed."""
    from qwen3_tts.tools.model_cache import prune_models_cmd

    model_dir = tmp_path / "old-model"
    model_dir.mkdir()
    (model_dir / "file.bin").write_bytes(b"data")

    old_time = datetime.now() - timedelta(days=60)
    models = [{"last_access": old_time, "name": "old-model", "size_bytes": 4, "path": str(model_dir)}]

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=models), \
         patch('builtins.input', return_value="y"):
        prune_models_cmd(days=30, dry_run=False)

    captured = capsys.readouterr()
    assert "Removed" in captured.out
    assert "Deleted 1" in captured.out
    assert not model_dir.exists()


@pytest.mark.unit
def test_prune_models_cmd_oserror(capsys):
    """prune_models_cmd handles OSError during deletion."""
    from qwen3_tts.tools.model_cache import prune_models_cmd

    old_time = datetime.now() - timedelta(days=60)
    models = [{"last_access": old_time, "name": "locked-model", "size_bytes": 1000, "path": "/nonexistent/locked"}]

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=models), \
         patch('builtins.input', return_value="y"), \
         patch('shutil.rmtree', side_effect=OSError("Permission denied")):
        prune_models_cmd(days=30, dry_run=False)

    captured = capsys.readouterr()
    assert "Failed to remove" in captured.out


# ---- clear_cache_cmd ----

@pytest.mark.unit
def test_clear_cache_cmd_no_models(capsys):
    """clear_cache_cmd prints 'no models' when cache empty."""
    from qwen3_tts.tools.model_cache import clear_cache_cmd

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=[]), \
         patch('qwen3_tts.tools.model_cache.get_total_size', return_value=0):
        clear_cache_cmd(force=False)

    captured = capsys.readouterr()
    assert "No TTS models found" in captured.out


@pytest.mark.unit
def test_clear_cache_cmd_force(capsys, tmp_path):
    """clear_cache_cmd with force=True skips confirmation."""
    from qwen3_tts.tools.model_cache import clear_cache_cmd

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "file.bin").write_bytes(b"data")

    models = [{"name": "model", "size_bytes": 4, "path": str(model_dir)}]

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=models), \
         patch('qwen3_tts.tools.model_cache.get_total_size', return_value=4):
        clear_cache_cmd(force=True)

    captured = capsys.readouterr()
    assert "Deleted 1" in captured.out
    assert not model_dir.exists()


@pytest.mark.unit
def test_clear_cache_cmd_cancelled(capsys):
    """clear_cache_cmd cancelled by user."""
    from qwen3_tts.tools.model_cache import clear_cache_cmd

    models = [{"name": "model", "size_bytes": 1000, "path": "/fake"}]

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=models), \
         patch('qwen3_tts.tools.model_cache.get_total_size', return_value=1000), \
         patch('builtins.input', return_value="n"):
        clear_cache_cmd(force=False)

    captured = capsys.readouterr()
    assert "Cancelled" in captured.out


@pytest.mark.unit
def test_clear_cache_cmd_oserror(capsys):
    """clear_cache_cmd handles OSError during deletion."""
    from qwen3_tts.tools.model_cache import clear_cache_cmd

    models = [{"name": "locked", "size_bytes": 1000, "path": "/nonexistent/locked"}]

    with patch('qwen3_tts.tools.model_cache.list_models', return_value=models), \
         patch('qwen3_tts.tools.model_cache.get_total_size', return_value=1000), \
         patch('shutil.rmtree', side_effect=OSError("Permission denied")):
        clear_cache_cmd(force=True)

    captured = capsys.readouterr()
    assert "Failed to remove" in captured.out


# ---- _get_model_dir_size OSError path ----

@pytest.mark.unit
def test_get_model_dir_size_oserror_on_stat():
    """_get_model_dir_size skips files with OSError on stat."""
    from qwen3_tts.tools.model_cache import _get_model_dir_size

    mock_path = MagicMock(spec=pathlib.Path)
    mock_path.is_dir.return_value = True

    good_file = MagicMock()
    good_file.is_file.return_value = True
    good_file.stat.return_value.st_size = 1000

    bad_file = MagicMock()
    bad_file.is_file.return_value = True
    bad_file.stat.side_effect = OSError("disk error")

    mock_path.rglob.return_value = [good_file, bad_file]
    result = _get_model_dir_size(mock_path)
    assert result == 1000
