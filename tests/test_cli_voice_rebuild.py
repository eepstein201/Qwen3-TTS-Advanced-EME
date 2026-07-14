"""Tests for 'tts voice rebuild' backend handling.

The rebuild command creates .pt voice prompts, which requires the torch
backend (qwen_tts.Qwen3TTSModel.create_voice_clone_prompt). On MLX it must
force TTS_BACKEND=torch, or fail with an actionable error if the qwen_tts
package is not installed in the current environment.
"""

import os
import sys
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from qwen3_tts.cli_voice import voice


def _make_prompt_dir(tmp_path):
    """Create a prompts dir with one .wav needing rebuild (no .pt)."""
    (tmp_path / "foo.wav").write_bytes(b"RIFF")
    (tmp_path / "foo.txt").write_text("hello")
    return str(tmp_path)


def _fake_torch():
    """A torch stand-in whose save() writes a real file."""
    fake = MagicMock()
    fake.save.side_effect = lambda obj, path: open(path, "wb").write(b"pt")
    return fake


def test_rebuild_mlx_without_qwen_tts_exits_with_error(tmp_path):
    """On MLX backend without qwen_tts installed, rebuild exits 1 with a hint."""
    prompts_dir = _make_prompt_dir(tmp_path)
    runner = CliRunner()
    with patch("qwen3_tts.core.config.VOICE_PROMPTS_DIR", prompts_dir), \
         patch("qwen3_tts.core.config.get_backend", return_value="mlx"), \
         patch("qwen3_tts.cli_voice._torch_available", return_value=False):
        result = runner.invoke(voice, ["rebuild"])

    assert result.exit_code == 1
    assert "torch" in result.output
    assert "conda activate qwen3-tts" in result.output


def test_rebuild_mlx_with_qwen_tts_forces_torch_backend(tmp_path):
    """On MLX backend with qwen_tts available, load_model must see TTS_BACKEND=torch."""
    prompts_dir = _make_prompt_dir(tmp_path)
    seen_backend = {}

    def fake_load_model(model_type):
        seen_backend["value"] = os.environ.get("TTS_BACKEND")
        return MagicMock()

    runner = CliRunner()
    env_before = os.environ.get("TTS_BACKEND")
    with patch("qwen3_tts.core.config.VOICE_PROMPTS_DIR", prompts_dir), \
         patch("qwen3_tts.core.config.get_backend", return_value="mlx"), \
         patch("qwen3_tts.cli_voice._torch_available", return_value=True), \
         patch("qwen3_tts.core.engine.model_loader.load_model",
               side_effect=fake_load_model), \
         patch("qwen3_tts.core.engine.inference.create_voice_prompt",
               return_value=MagicMock()), \
         patch("qwen3_tts.core.engine.audio_processing.load_audio_for_cloning",
               return_value=(MagicMock(), 16000)), \
         patch.dict(sys.modules, {"torch": _fake_torch()}):
        result = runner.invoke(voice, ["rebuild"])

    assert result.exit_code == 0, result.output
    assert seen_backend["value"] == "torch"
    # Env var must be restored after the command finishes.
    assert os.environ.get("TTS_BACKEND") == env_before


def test_is_pt_valid_registers_safe_globals(tmp_path):
    """_is_pt_valid must add VoiceClonePromptItem to torch safe globals.

    Without registration, torch.load(weights_only=True) rejects every valid
    .pt prompt, so rebuild re-generates all prompts on every run.
    """
    fake_torch = MagicMock()
    fake_item = object()
    fake_qwen_model_mod = MagicMock()
    fake_qwen_model_mod.VoiceClonePromptItem = fake_item

    with patch.dict(sys.modules, {
        "torch": fake_torch,
        "torch.serialization": fake_torch.serialization,
        "qwen_tts": MagicMock(),
        "qwen_tts.inference": MagicMock(),
        "qwen_tts.inference.qwen3_tts_model": fake_qwen_model_mod,
    }):
        from qwen3_tts.cli_voice import _is_pt_valid

        assert _is_pt_valid(str(tmp_path / "x.pt")) is True
        fake_torch.serialization.add_safe_globals.assert_called_once_with(
            [fake_item]
        )


def test_rebuild_mlx_without_qwen_tts_fails_before_scanning(tmp_path):
    """Backend check must run before the .pt validity scan.

    Validation itself needs qwen_tts (safe-globals registration), so in an
    env without it the command must error out immediately instead of
    misclassifying valid prompts as corrupt.
    """
    prompts_dir = _make_prompt_dir(tmp_path)
    (tmp_path / "foo.pt").write_bytes(b"pt")
    runner = CliRunner()
    with patch("qwen3_tts.core.config.VOICE_PROMPTS_DIR", prompts_dir), \
         patch("qwen3_tts.core.config.get_backend", return_value="mlx"), \
         patch("qwen3_tts.cli_voice._torch_available", return_value=False), \
         patch("qwen3_tts.cli_voice._is_pt_valid") as validator:
        result = runner.invoke(voice, ["rebuild"])

    assert result.exit_code == 1
    validator.assert_not_called()


def test_rebuild_torch_backend_skips_probe(tmp_path):
    """On torch backend, no availability probe or env override is needed."""
    prompts_dir = _make_prompt_dir(tmp_path)
    runner = CliRunner()
    with patch("qwen3_tts.core.config.VOICE_PROMPTS_DIR", prompts_dir), \
         patch("qwen3_tts.core.config.get_backend", return_value="torch"), \
         patch("qwen3_tts.cli_voice._torch_available") as probe, \
         patch("qwen3_tts.core.engine.model_loader.load_model",
               return_value=MagicMock()), \
         patch("qwen3_tts.core.engine.inference.create_voice_prompt",
               return_value=MagicMock()), \
         patch("qwen3_tts.core.engine.audio_processing.load_audio_for_cloning",
               return_value=(MagicMock(), 16000)), \
         patch.dict(sys.modules, {"torch": _fake_torch()}):
        result = runner.invoke(voice, ["rebuild"])

    assert result.exit_code == 0, result.output
    probe.assert_not_called()
