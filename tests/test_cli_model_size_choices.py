"""Pinning tests: model-size CLI choices must track VALID_MODEL_SIZES.

UI-4 (repo review 2026-07-23, Phase 2). The model-size option is exposed on
three CLI surfaces — the argparse default command, the Click ``generate``
command, and the Click ``config edit`` command. All three must derive their
choices from the single source of truth
``qwen3_tts.core.config.VALID_MODEL_SIZES`` instead of hardcoded literals, so
the choices cannot drift from the constant.

These are characterization/pinning tests for a behavior-preserving refactor:
they pass both before and after the change, and guard against future drift.
"""

from qwen3_tts.core.config import VALID_MODEL_SIZES

_EXPECTED = list(VALID_MODEL_SIZES)


def _find_param(params, name):
    """Return the Click param whose destination name matches."""
    return next(p for p in params if p.name == name)


def test_argparse_default_command_model_size_choices():
    """The argparse `tts` default command derives --model-size from the constant."""
    from qwen3_tts.interface.generate import _build_parser

    action = next(
        a for a in _build_parser()._actions if "--model-size" in (a.option_strings or [])
    )
    assert action.choices == _EXPECTED


def test_click_generate_command_model_size_choices():
    """The Click `generate` command derives --model-size from the constant."""
    from qwen3_tts.cli import generate

    opt = _find_param(generate.params, "model_size")
    assert list(opt.type.choices) == _EXPECTED


def test_click_config_edit_model_size_choices():
    """The Click `config edit` command derives --model-size from the constant."""
    from qwen3_tts.cli_config import config

    edit = config.commands["edit"]
    opt = _find_param(edit.params, "model_size")
    assert list(opt.type.choices) == _EXPECTED
