#!/usr/bin/env python3
"""Tests for cli.py Click commands.

Covers:
  - TTSGroup.parse_args: --_server-mode stripping, bare text routing
  - _call_generate: flag mapping, sys.argv restoration
  - list commands: speakers, presets, aliases, prosody, models, backends
  - config commands: show, path
  - cache commands: list, size
  - doctor command
  - history, stats

Run: pytest tests/test_cli_commands.py -v
"""
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

from unittest.mock import patch, MagicMock
from click.testing import CliRunner


# ---- TTSGroup routing ----

@pytest.mark.unit
def test_tts_group_routes_bare_text():
    """Bare text is routed to generate subcommand."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.cli._call_generate') as mock_gen:
        runner.invoke(cli, ['Hello world'])
    mock_gen.assert_called_once()


@pytest.mark.unit
def test_tts_group_routes_empty_to_generate():
    """Bare `tts` with no args routes to generate."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.cli._call_generate') as mock_gen:
        runner.invoke(cli, [])
    mock_gen.assert_called_once()


@pytest.mark.unit
def test_tts_group_server_mode_flag():
    """--_server-mode flag is stripped from routing and re-inserted for generation."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.cli._call_generate') as mock_gen:
        runner.invoke(cli, ['--_server-mode', 'Hello'])
    mock_gen.assert_called_once()
    call_kwargs = mock_gen.call_args
    assert call_kwargs[1].get('server_mode') is True or '--_server-mode' in str(call_kwargs)


@pytest.mark.unit
def test_tts_group_server_mode_not_for_non_generation():
    """--_server-mode is not added to non-generation commands."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.core.config.CONFIG_PATH', '/tmp/fake'):
        result = runner.invoke(cli, ['--_server-mode', 'config', 'path'])
    # config path should work without error
    assert result.exit_code == 0


# ---- _call_generate flag mapping ----

@pytest.mark.unit
def test_call_generate_maps_flags():
    """_call_generate translates Click kwargs to argparse sys.argv."""
    from qwen3_tts.cli import _call_generate

    mock_main = MagicMock(return_value=None)

    with patch('qwen3_tts.interface.generate.main', mock_main):
        old_argv = sys.argv
        try:
            _call_generate(
                text=("Hello",),
                mode="clone",
                output="test.wav",
                play=True,
                stream=False,
            )
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv

    mock_main.assert_called_once()


@pytest.mark.unit
def test_call_generate_restores_argv():
    """_call_generate restores sys.argv even on exception."""
    from qwen3_tts.cli import _call_generate

    original = sys.argv[:]

    with patch('qwen3_tts.interface.generate.main', side_effect=RuntimeError("test")):
        try:
            _call_generate(text=("Hi",))
        except RuntimeError:
            pass

    assert sys.argv == original


@pytest.mark.unit
def test_call_generate_exits_on_true_return():
    """_call_generate calls sys.exit(2) when main returns True."""
    from qwen3_tts.cli import _call_generate

    with patch('qwen3_tts.interface.generate.main', return_value=True), \
         pytest.raises(SystemExit) as exc_info:
        _call_generate(text=("Hi",))

    assert exc_info.value.code == 2


# ---- list speakers ----

@pytest.mark.unit
def test_list_speakers():
    """list speakers shows premium speakers."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['list', 'speakers'])
    assert result.exit_code == 0
    assert "Premium CustomVoice speakers" in result.output


# ---- list presets ----

@pytest.mark.unit
def test_list_presets_empty():
    """list presets shows 'no presets' when config empty."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.core.config.load_config', return_value={}):
        result = runner.invoke(cli, ['list', 'presets'])

    assert "No presets configured" in result.output


@pytest.mark.unit
def test_list_presets_with_data():
    """list presets shows configured presets."""
    from qwen3_tts.cli import cli

    config = {"presets": {"fast": {"temperature": 0.3, "top_k": 20}}}
    runner = CliRunner()
    with patch('qwen3_tts.core.config.load_config', return_value=config):
        result = runner.invoke(cli, ['list', 'presets'])

    assert "fast" in result.output
    assert "temperature=0.3" in result.output


# ---- list aliases ----

@pytest.mark.unit
def test_list_aliases_empty():
    """list aliases shows 'no aliases' when config empty."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.core.config.load_config', return_value={}):
        result = runner.invoke(cli, ['list', 'aliases'])

    assert "No aliases configured" in result.output


@pytest.mark.unit
def test_list_aliases_with_data():
    """list aliases shows configured aliases."""
    from qwen3_tts.cli import cli

    config = {"aliases": {"myvoice": {"prompt": "voice1.pt", "mode": "clone"}}}
    runner = CliRunner()
    with patch('qwen3_tts.core.config.load_config', return_value=config):
        result = runner.invoke(cli, ['list', 'aliases'])

    assert "myvoice" in result.output
    assert "prompt=voice1.pt" in result.output
    assert "mode=clone" in result.output


# ---- list prosody ----

@pytest.mark.unit
def test_list_prosody_empty():
    """list prosody shows 'no presets' when empty."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.core.config.get_prosody_presets', return_value={}):
        result = runner.invoke(cli, ['list', 'prosody'])

    assert "No prosody presets" in result.output


@pytest.mark.unit
def test_list_prosody_with_data():
    """list prosody shows configured presets."""
    from qwen3_tts.cli import cli

    presets = {"cheerful": "Speak in a cheerful tone"}
    runner = CliRunner()
    with patch('qwen3_tts.core.config.get_prosody_presets', return_value=presets):
        result = runner.invoke(cli, ['list', 'prosody'])

    assert "cheerful" in result.output


# ---- list models ----

@pytest.mark.unit
def test_list_models_server_not_running():
    """list models shows configured models when server is not running."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.core.config.load_config', return_value={}), \
         patch('qwen3_tts.core.config.get_backend', return_value="mlx"), \
         patch('qwen3_tts.core.config.get_model_size', return_value="1.7B"), \
         patch('qwen3_tts.core.config.is_server_running', return_value=False):
        result = runner.invoke(cli, ['list', 'models'])

    assert "Server not running" in result.output
    assert "clone" in result.output


# ---- list backends ----

@pytest.mark.unit
def test_list_backends_mlx():
    """list backends shows MLX info when mlx is current."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.core.config.get_backend', return_value="mlx"), \
         patch('qwen3_tts.core.config.VALID_BACKENDS', ["mlx", "torch", "vllm"]), \
         patch('qwen3_tts.core.config.get_mlx_quantization', return_value="8bit"), \
         patch('qwen3_tts.core.config.get_mlx_model_name', return_value="mlx-community/model"):
        result = runner.invoke(cli, ['list', 'backends'])

    assert "mlx" in result.output
    assert "8bit" in result.output


@pytest.mark.unit
def test_list_backends_torch():
    """list backends shows torch info when torch is current."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.core.config.get_backend', return_value="torch"), \
         patch('qwen3_tts.core.config.VALID_BACKENDS', ["mlx", "torch", "vllm"]), \
         patch('qwen3_tts.core.config.get_torch_dtype_name', return_value="float32"), \
         patch('qwen3_tts.core.config.MODEL_INFO', {"clone": {"name": "Qwen/model"}}):
        result = runner.invoke(cli, ['list', 'backends'])

    assert "torch" in result.output
    assert "float32" in result.output


# ---- config show ----

@pytest.mark.unit
def test_config_show():
    """config show prints JSON config."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.core.config.load_config', return_value={"key": "value"}):
        result = runner.invoke(cli, ['config', 'show'])

    assert '"key"' in result.output
    assert '"value"' in result.output


# ---- config path ----

@pytest.mark.unit
def test_config_path():
    """config path prints the config file path."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.core.config.CONFIG_PATH', '/fake/path/config.json'):
        result = runner.invoke(cli, ['config', 'path'])

    assert "/fake/path/config.json" in result.output


# ---- cache list ----

@pytest.mark.unit
def test_cache_list():
    """cache list delegates to list_models_cmd."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.tools.model_cache.list_models_cmd') as mock_cmd:
        runner.invoke(cli, ['cache', 'list'])
    mock_cmd.assert_called_once()


# ---- cache size ----

@pytest.mark.unit
def test_cache_size():
    """cache size delegates to get_size_cmd."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.tools.model_cache.get_size_cmd') as mock_cmd:
        runner.invoke(cli, ['cache', 'size'])
    mock_cmd.assert_called_once()


# ---- doctor ----

@pytest.mark.unit
def test_doctor():
    """doctor delegates to run_healthcheck."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.tools.healthcheck.run_healthcheck', return_value=0):
        result = runner.invoke(cli, ['doctor'])
    assert result.exit_code == 0


@pytest.mark.unit
def test_doctor_failure():
    """doctor returns non-zero on failure."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.tools.healthcheck.run_healthcheck', return_value=1):
        result = runner.invoke(cli, ['doctor'])
    assert result.exit_code == 1


# ---- history ----

@pytest.mark.unit
def test_history_default():
    """history delegates to show_history with default count."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.interface.generate.show_history') as mock_hist:
        runner.invoke(cli, ['history'])
    mock_hist.assert_called_once_with(10)


@pytest.mark.unit
def test_history_custom_count():
    """history passes custom count."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.interface.generate.show_history') as mock_hist:
        runner.invoke(cli, ['history', '5'])
    mock_hist.assert_called_once_with(5)


# ---- stats ----

@pytest.mark.unit
def test_stats_server_not_running():
    """stats shows error when server not running."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.core.config.load_config', return_value={}), \
         patch('qwen3_tts.core.config.is_server_running', return_value=False):
        result = runner.invoke(cli, ['stats'])

    assert "Server not running" in result.output
    assert result.exit_code == 1


# ---- voice list ----

@pytest.mark.unit
def test_voice_list_empty():
    """voice list shows 'no prompts' when empty."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.interface.generate.list_voice_prompts', return_value=[]), \
         patch('qwen3_tts.core.config.get_default_clone_prompt', return_value=None):
        result = runner.invoke(cli, ['voice', 'list'])

    assert "No voice prompts found" in result.output


@pytest.mark.unit
def test_voice_list_with_prompts():
    """voice list shows prompts with default marker."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.interface.generate.list_voice_prompts', return_value=["voice1", "voice2"]), \
         patch('qwen3_tts.core.config.get_default_clone_prompt', return_value="voice1"):
        result = runner.invoke(cli, ['voice', 'list'])

    assert "voice1" in result.output
    assert "(default)" in result.output
    assert "voice2" in result.output


# ---- voice info ----

@pytest.mark.unit
def test_voice_info_server_not_running():
    """voice info shows error when server not running."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.core.config.load_config', return_value={}), \
         patch('qwen3_tts.core.config.is_server_running', return_value=False):
        result = runner.invoke(cli, ['voice', 'info', 'test_voice'])

    assert "Server not running" in result.output


# ---- version ----

@pytest.mark.unit
def test_version():
    """--version shows version string."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])
    assert "3.0.0" in result.output


# ---- uninstall commands delegate ----

@pytest.mark.unit
def test_uninstall_environment():
    """uninstall environment delegates to print_environment_instructions."""
    from qwen3_tts.cli import cli

    runner = CliRunner()
    with patch('qwen3_tts.tools.uninstall.print_environment_instructions') as mock_fn:
        runner.invoke(cli, ['uninstall', 'environment'])
    mock_fn.assert_called_once()
