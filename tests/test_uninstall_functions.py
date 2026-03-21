"""Tests for qwen3_tts/tools/uninstall.py functions.

Covers:
  - _get_models_size: empty cache, cache with TTS models, OSError handling
  - _list_cached_models: empty, with matching/non-matching dirs
  - uninstall_models: no models, dry run, confirmed, declined, OSError
  - uninstall_voices: no dir, empty dir, dry run, confirmed, OSError
  - uninstall_config: no config, dry run, backup + reset
  - uninstall_all: orchestrator with dry run
  - print_environment_instructions: env detection paths
  - main: CLI entry point with various flags
"""
import json
import pathlib
import tempfile
import unittest
from unittest import mock


# ---------------------------------------------------------------------------
# _get_models_size / _list_cached_models
# ---------------------------------------------------------------------------

class TestGetModelsSize(unittest.TestCase):
    """Tests for _get_models_size."""

    def test_empty_cache_returns_zero(self):
        """Returns 0 when HF_CACHE doesn't exist."""
        from qwen3_tts.tools.uninstall import _get_models_size

        with mock.patch("qwen3_tts.tools.uninstall.HF_CACHE") as mock_cache:
            mock_cache.exists.return_value = False
            self.assertEqual(_get_models_size(), 0)

    def test_counts_tts_model_files(self):
        """Sums file sizes in matching TTS model directories."""
        from qwen3_tts.tools.uninstall import _get_models_size

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir)

            # Create a matching model dir with files
            model_dir = cache_path / "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
            model_dir.mkdir()
            (model_dir / "file1.bin").write_bytes(b"x" * 100)
            (model_dir / "file2.bin").write_bytes(b"y" * 200)

            # Create a non-matching dir (should be ignored)
            other_dir = cache_path / "models--other--something"
            other_dir.mkdir()
            (other_dir / "big.bin").write_bytes(b"z" * 9999)

            with mock.patch("qwen3_tts.tools.uninstall.HF_CACHE", cache_path):
                size = _get_models_size()

            self.assertEqual(size, 300)

    def test_handles_oserror_on_stat(self):
        """Gracefully handles OSError when stat-ing a file."""
        from qwen3_tts.tools.uninstall import _get_models_size

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir)
            model_dir = cache_path / "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
            model_dir.mkdir()
            file_path = model_dir / "file1.bin"
            file_path.write_bytes(b"x" * 100)

            with mock.patch("qwen3_tts.tools.uninstall.HF_CACHE", cache_path):
                # Make stat raise OSError only for the specific file
                original_stat = pathlib.Path.stat

                call_count = {"n": 0}

                def failing_stat(self_path, *args, **kwargs):
                    result = original_stat(self_path, *args, **kwargs)
                    if self_path.name == "file1.bin":
                        call_count["n"] += 1
                        # First call is from is_file(), let it pass
                        # Second call is from st_size, make it fail
                        if call_count["n"] > 1:
                            raise OSError("permission denied")
                    return result

                with mock.patch.object(pathlib.Path, "stat", failing_stat):
                    size = _get_models_size()

            self.assertEqual(size, 0)

    def test_counts_mlx_model_files(self):
        """Counts files in MLX model directories too."""
        from qwen3_tts.tools.uninstall import _get_models_size

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir)
            model_dir = (
                cache_path
                / "models--mlx-community--Qwen3-TTS-12Hz-1.7B-Base-8bit"
            )
            model_dir.mkdir()
            (model_dir / "weights.bin").write_bytes(b"w" * 500)

            with mock.patch("qwen3_tts.tools.uninstall.HF_CACHE", cache_path):
                size = _get_models_size()

            self.assertEqual(size, 500)


class TestListCachedModels(unittest.TestCase):
    """Tests for _list_cached_models."""

    def test_empty_cache(self):
        """Returns empty list when cache doesn't exist."""
        from qwen3_tts.tools.uninstall import _list_cached_models

        with mock.patch("qwen3_tts.tools.uninstall.HF_CACHE") as mock_cache:
            mock_cache.exists.return_value = False
            self.assertEqual(_list_cached_models(), [])

    def test_returns_sorted_matching_dirs(self):
        """Returns sorted list of matching TTS model directory names."""
        from qwen3_tts.tools.uninstall import _list_cached_models

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir)

            # Create matching dirs
            names = [
                "models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base",
                "models--mlx-community--Qwen3-TTS-12Hz-0.6B-Base-4bit",
            ]
            for name in names:
                (cache_path / name).mkdir()

            # Non-matching dir
            (cache_path / "models--unrelated--model").mkdir()

            with mock.patch("qwen3_tts.tools.uninstall.HF_CACHE", cache_path):
                result = _list_cached_models()

            self.assertEqual(len(result), 3)
            # Should be sorted
            self.assertEqual(result, sorted(names))

    def test_skips_files_only_dirs(self):
        """Only includes directories, not files."""
        from qwen3_tts.tools.uninstall import _list_cached_models

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir)
            # Create a file with a matching prefix
            (cache_path / "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base").write_text("not a dir")

            with mock.patch("qwen3_tts.tools.uninstall.HF_CACHE", cache_path):
                result = _list_cached_models()

            self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# uninstall_models
# ---------------------------------------------------------------------------

class TestUninstallModels(unittest.TestCase):
    """Tests for uninstall_models."""

    @mock.patch("qwen3_tts.tools.uninstall._list_cached_models",
                return_value=[])
    def test_no_models_prints_info(self, _mock_list):
        """Prints info when no models found."""
        from qwen3_tts.tools.uninstall import uninstall_models

        with mock.patch("builtins.print") as mock_print:
            uninstall_models()

        # Should print an info message about no models
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("No TTS models", printed)

    @mock.patch("qwen3_tts.tools.uninstall._get_models_size", return_value=1024)
    @mock.patch("qwen3_tts.tools.uninstall._list_cached_models",
                return_value=["models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"])
    def test_dry_run_no_deletion(self, _mock_list, _mock_size):
        """Dry run lists models but doesn't delete."""
        from qwen3_tts.tools.uninstall import uninstall_models

        with mock.patch("builtins.print") as mock_print:
            uninstall_models(dry_run=True)

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("Dry run", printed)

    @mock.patch("qwen3_tts.tools.uninstall._get_models_size", return_value=1024)
    @mock.patch("qwen3_tts.tools.uninstall._list_cached_models",
                return_value=["models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"])
    @mock.patch("builtins.input", return_value="y")
    def test_confirmed_deletion(self, _mock_input, _mock_list, _mock_size):
        """Deletes model dirs when user confirms."""
        from qwen3_tts.tools.uninstall import uninstall_models

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir)
            model_dir = cache_path / "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
            model_dir.mkdir()
            (model_dir / "weight.bin").write_bytes(b"data")

            with mock.patch("qwen3_tts.tools.uninstall.HF_CACHE", cache_path):
                uninstall_models()

            self.assertFalse(model_dir.exists())

    @mock.patch("qwen3_tts.tools.uninstall._get_models_size", return_value=1024)
    @mock.patch("qwen3_tts.tools.uninstall._list_cached_models",
                return_value=["models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"])
    @mock.patch("builtins.input", return_value="n")
    def test_declined_deletion(self, _mock_input, _mock_list, _mock_size):
        """Does not delete when user declines."""
        from qwen3_tts.tools.uninstall import uninstall_models

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir)
            model_dir = cache_path / "models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"
            model_dir.mkdir()

            with mock.patch("qwen3_tts.tools.uninstall.HF_CACHE", cache_path):
                uninstall_models()

            self.assertTrue(model_dir.exists())

    @mock.patch("qwen3_tts.tools.uninstall._get_models_size", return_value=1024)
    @mock.patch("qwen3_tts.tools.uninstall._list_cached_models",
                return_value=["models--Qwen--Qwen3-TTS-12Hz-1.7B-Base"])
    @mock.patch("builtins.input", return_value="y")
    @mock.patch("shutil.rmtree", side_effect=OSError("permission denied"))
    def test_oserror_on_deletion(self, _mock_rmtree, _mock_input, _mock_list,
                                  _mock_size):
        """Prints warning when deletion raises OSError."""
        from qwen3_tts.tools.uninstall import uninstall_models

        with mock.patch("qwen3_tts.tools.uninstall.HF_CACHE",
                        pathlib.Path("/fake")):
            with mock.patch("builtins.print") as mock_print:
                uninstall_models()

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("Failed to remove", printed)


# ---------------------------------------------------------------------------
# uninstall_voices
# ---------------------------------------------------------------------------

class TestUninstallVoices(unittest.TestCase):
    """Tests for uninstall_voices."""

    def test_no_dir_prints_info(self):
        """Prints info when voice prompts directory doesn't exist."""
        from qwen3_tts.tools.uninstall import uninstall_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_dir = pathlib.Path(tmpdir) / "nonexistent"
            with mock.patch("qwen3_tts.tools.uninstall.VOICE_PROMPTS_DIR",
                            fake_dir):
                with mock.patch("builtins.print") as mock_print:
                    uninstall_voices()

            printed = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("No voice prompts directory", printed)

    def test_empty_dir_prints_info(self):
        """Prints info when directory exists but is empty."""
        from qwen3_tts.tools.uninstall import uninstall_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = pathlib.Path(tmpdir) / "voice_prompts"
            prompts_dir.mkdir()

            with mock.patch("qwen3_tts.tools.uninstall.VOICE_PROMPTS_DIR",
                            prompts_dir):
                with mock.patch("builtins.print") as mock_print:
                    uninstall_voices()

            printed = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("No voice prompts found", printed)

    def test_dry_run_lists_files(self):
        """Dry run shows file counts but doesn't delete."""
        from qwen3_tts.tools.uninstall import uninstall_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = pathlib.Path(tmpdir) / "voice_prompts"
            prompts_dir.mkdir()
            (prompts_dir / "voice1.pt").write_bytes(b"data")
            (prompts_dir / "voice1.wav").write_bytes(b"data")
            (prompts_dir / "voice1.txt").write_text("transcript")

            with mock.patch("qwen3_tts.tools.uninstall.VOICE_PROMPTS_DIR",
                            prompts_dir):
                with mock.patch("builtins.print") as mock_print:
                    uninstall_voices(dry_run=True)

            printed = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("Dry run", printed)
            # Files should still exist
            self.assertEqual(len(list(prompts_dir.iterdir())), 3)

    @mock.patch("builtins.input", return_value="y")
    def test_confirmed_deletion(self, _mock_input):
        """Deletes all voice files when confirmed."""
        from qwen3_tts.tools.uninstall import uninstall_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = pathlib.Path(tmpdir) / "voice_prompts"
            prompts_dir.mkdir()
            (prompts_dir / "voice1.pt").write_bytes(b"data")
            (prompts_dir / "voice1.wav").write_bytes(b"data")
            (prompts_dir / "voice1.txt").write_text("transcript")

            with mock.patch("qwen3_tts.tools.uninstall.VOICE_PROMPTS_DIR",
                            prompts_dir):
                uninstall_voices()

            # Directory should be recreated empty
            self.assertTrue(prompts_dir.exists())
            self.assertEqual(len(list(prompts_dir.iterdir())), 0)

    @mock.patch("builtins.input", return_value="n")
    def test_declined_deletion(self, _mock_input):
        """Does not delete when user declines."""
        from qwen3_tts.tools.uninstall import uninstall_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = pathlib.Path(tmpdir) / "voice_prompts"
            prompts_dir.mkdir()
            (prompts_dir / "voice1.pt").write_bytes(b"data")

            with mock.patch("qwen3_tts.tools.uninstall.VOICE_PROMPTS_DIR",
                            prompts_dir):
                uninstall_voices()

            self.assertTrue((prompts_dir / "voice1.pt").exists())

    @mock.patch("builtins.input", return_value="y")
    def test_oserror_on_deletion(self, _mock_input):
        """Prints warning on OSError during deletion."""
        from qwen3_tts.tools.uninstall import uninstall_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = pathlib.Path(tmpdir) / "voice_prompts"
            prompts_dir.mkdir()
            (prompts_dir / "voice1.pt").write_bytes(b"data")

            with mock.patch("qwen3_tts.tools.uninstall.VOICE_PROMPTS_DIR",
                            prompts_dir):
                # Scope the rmtree mock to only affect the uninstall call,
                # not TemporaryDirectory cleanup
                with mock.patch("qwen3_tts.tools.uninstall.shutil.rmtree",
                                side_effect=OSError("nope")):
                    with mock.patch("builtins.print") as mock_print:
                        uninstall_voices()

            printed = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("Failed to remove", printed)


# ---------------------------------------------------------------------------
# uninstall_config
# ---------------------------------------------------------------------------

class TestUninstallConfig(unittest.TestCase):
    """Tests for uninstall_config."""

    def test_no_config_prints_info(self):
        """Prints info when config.json doesn't exist."""
        from qwen3_tts.tools.uninstall import uninstall_config

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_config = pathlib.Path(tmpdir) / "config.json"
            with mock.patch("qwen3_tts.tools.uninstall.CONFIG_PATH",
                            fake_config):
                with mock.patch("builtins.print") as mock_print:
                    uninstall_config()

            printed = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("No config.json", printed)

    def test_dry_run_no_modification(self):
        """Dry run doesn't modify config."""
        from qwen3_tts.tools.uninstall import uninstall_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = pathlib.Path(tmpdir) / "config.json"
            original_data = {"key": "original_value"}
            config_path.write_text(json.dumps(original_data))

            with mock.patch("qwen3_tts.tools.uninstall.CONFIG_PATH",
                            config_path):
                with mock.patch("builtins.print") as mock_print:
                    uninstall_config(dry_run=True)

            printed = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("Dry run", printed)

            # Config should be unchanged
            with open(config_path) as f:
                self.assertEqual(json.load(f), original_data)

    @mock.patch("qwen3_tts.core.config.get_default_config",
                return_value={"reset": True})
    @mock.patch("qwen3_tts.core.config.load_config",
                return_value={"key": "old"})
    def test_backup_and_reset(self, _mock_load, _mock_default):
        """Creates backup and resets config to defaults."""
        from qwen3_tts.tools.uninstall import uninstall_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = pathlib.Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"key": "old"}))

            with mock.patch("qwen3_tts.tools.uninstall.CONFIG_PATH",
                            config_path):
                uninstall_config()

            # Backup should exist
            backup_path = config_path.with_suffix(".backup")
            self.assertTrue(backup_path.exists())

            # Config should have the default values
            with open(config_path) as f:
                data = json.load(f)
            self.assertEqual(data, {"reset": True})

    @mock.patch("qwen3_tts.core.config.get_default_config",
                return_value={"reset": True})
    @mock.patch("qwen3_tts.core.config.load_config",
                side_effect=Exception("corrupt config"))
    def test_corrupt_config_still_resets(self, _mock_load, _mock_default):
        """Resets config even when current config is corrupt."""
        from qwen3_tts.tools.uninstall import uninstall_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = pathlib.Path(tmpdir) / "config.json"
            config_path.write_text("not json")

            with mock.patch("qwen3_tts.tools.uninstall.CONFIG_PATH",
                            config_path):
                uninstall_config()

            with open(config_path) as f:
                data = json.load(f)
            self.assertEqual(data, {"reset": True})

    @mock.patch("qwen3_tts.core.config.get_default_config",
                return_value={"reset": True})
    @mock.patch("qwen3_tts.core.config.load_config",
                return_value={"key": "old"})
    @mock.patch("shutil.copy2", side_effect=OSError("backup failed"))
    def test_backup_failure_prints_warning(self, _mock_copy, _mock_load,
                                            _mock_default):
        """Prints warning but continues if backup fails."""
        from qwen3_tts.tools.uninstall import uninstall_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = pathlib.Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"key": "old"}))

            with mock.patch("qwen3_tts.tools.uninstall.CONFIG_PATH",
                            config_path):
                with mock.patch("builtins.print") as mock_print:
                    uninstall_config()

            printed = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("Could not create backup", printed)

            # Config should still be reset
            with open(config_path) as f:
                data = json.load(f)
            self.assertEqual(data, {"reset": True})


# ---------------------------------------------------------------------------
# print_environment_instructions
# ---------------------------------------------------------------------------

class TestPrintEnvironmentInstructions(unittest.TestCase):
    """Tests for print_environment_instructions."""

    def test_prints_conda_env_commands(self):
        """Prints removal commands when conda envs exist."""
        from qwen3_tts.tools.uninstall import print_environment_instructions

        with tempfile.TemporaryDirectory() as tmpdir:
            # Build a fake home dir with miniforge3/envs
            fake_home = pathlib.Path(tmpdir) / "fakehome"
            envs_path = fake_home / "miniforge3" / "envs"
            envs_path.mkdir(parents=True)
            (envs_path / "qwen3-tts").mkdir()
            (envs_path / "qwen3-tts-mlx").mkdir()

            # Make the first two os.path.exists checks fail so we fall into
            # the pathlib.Path else branch, which correctly uses / operator
            with mock.patch("os.path.exists", return_value=False):
                with mock.patch("pathlib.Path.home",
                                return_value=fake_home):
                    with mock.patch("builtins.print") as mock_print:
                        print_environment_instructions()

            printed = " ".join(str(c) for c in mock_print.call_args_list)
            self.assertIn("qwen3-tts", printed)

    def test_no_envs_found(self):
        """Prints 'no environments found' when none exist."""
        from qwen3_tts.tools.uninstall import print_environment_instructions

        with mock.patch("os.path.exists", return_value=False):
            with mock.patch("pathlib.Path.home",
                            return_value=pathlib.Path("/nonexistent")):
                with mock.patch("builtins.print") as mock_print:
                    print_environment_instructions()

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("No conda environments", printed)


# ---------------------------------------------------------------------------
# uninstall_all
# ---------------------------------------------------------------------------

class TestUninstallAll(unittest.TestCase):
    """Tests for uninstall_all orchestrator."""

    @mock.patch("qwen3_tts.tools.uninstall.print_environment_instructions")
    @mock.patch("qwen3_tts.tools.uninstall.uninstall_config")
    @mock.patch("qwen3_tts.tools.uninstall.uninstall_voices")
    @mock.patch("qwen3_tts.tools.uninstall.uninstall_models")
    def test_calls_all_steps(self, mock_models, mock_voices, mock_config,
                              mock_env):
        """Calls all uninstall steps."""
        from qwen3_tts.tools.uninstall import uninstall_all

        uninstall_all()

        mock_models.assert_called_once_with(False)
        mock_voices.assert_called_once_with(False)
        mock_config.assert_called_once_with(False)
        mock_env.assert_called_once()

    @mock.patch("qwen3_tts.tools.uninstall.print_environment_instructions")
    @mock.patch("qwen3_tts.tools.uninstall.uninstall_config")
    @mock.patch("qwen3_tts.tools.uninstall.uninstall_voices")
    @mock.patch("qwen3_tts.tools.uninstall.uninstall_models")
    def test_passes_dry_run(self, mock_models, mock_voices, mock_config,
                             mock_env):
        """Passes dry_run flag to all steps."""
        from qwen3_tts.tools.uninstall import uninstall_all

        uninstall_all(dry_run=True)

        mock_models.assert_called_once_with(True)
        mock_voices.assert_called_once_with(True)
        mock_config.assert_called_once_with(True)


# ---------------------------------------------------------------------------
# main() CLI entry point
# ---------------------------------------------------------------------------

class TestMainEntryPoint(unittest.TestCase):
    """Tests for the main() CLI entry point."""

    @mock.patch("qwen3_tts.tools.uninstall.uninstall_models")
    def test_models_flag(self, mock_models):
        """--models flag calls uninstall_models."""
        from qwen3_tts.tools.uninstall import main

        with mock.patch("sys.argv", ["uninstall", "--models"]):
            result = main()

        mock_models.assert_called_once_with(dry_run=False)
        self.assertEqual(result, 0)

    @mock.patch("qwen3_tts.tools.uninstall.uninstall_voices")
    def test_voices_flag(self, mock_voices):
        """--voices flag calls uninstall_voices."""
        from qwen3_tts.tools.uninstall import main

        with mock.patch("sys.argv", ["uninstall", "--voices"]):
            result = main()

        mock_voices.assert_called_once_with(dry_run=False)
        self.assertEqual(result, 0)

    @mock.patch("qwen3_tts.tools.uninstall.uninstall_config")
    def test_config_flag(self, mock_config):
        """--config flag calls uninstall_config."""
        from qwen3_tts.tools.uninstall import main

        with mock.patch("sys.argv", ["uninstall", "--config"]):
            result = main()

        mock_config.assert_called_once_with(dry_run=False)
        self.assertEqual(result, 0)

    @mock.patch("qwen3_tts.tools.uninstall.uninstall_all")
    def test_all_flag(self, mock_all):
        """--all flag calls uninstall_all."""
        from qwen3_tts.tools.uninstall import main

        with mock.patch("sys.argv", ["uninstall", "--all"]):
            result = main()

        mock_all.assert_called_once_with(dry_run=False)
        self.assertEqual(result, 0)

    @mock.patch("qwen3_tts.tools.uninstall.uninstall_all")
    def test_all_with_dry_run(self, mock_all):
        """--all --dry-run passes dry_run=True."""
        from qwen3_tts.tools.uninstall import main

        with mock.patch("sys.argv", ["uninstall", "--all", "--dry-run"]):
            result = main()

        mock_all.assert_called_once_with(dry_run=True)
        self.assertEqual(result, 0)

    @mock.patch("qwen3_tts.tools.uninstall.print_environment_instructions")
    def test_environment_flag(self, mock_env):
        """--environment flag calls print_environment_instructions."""
        from qwen3_tts.tools.uninstall import main

        with mock.patch("sys.argv", ["uninstall", "--environment"]):
            result = main()

        mock_env.assert_called_once()
        self.assertEqual(result, 0)

    def test_no_flags_shows_help(self):
        """No flags prints help and returns 0."""
        from qwen3_tts.tools.uninstall import main

        with mock.patch("sys.argv", ["uninstall"]):
            with mock.patch("builtins.print"):
                result = main()

        self.assertEqual(result, 0)

    @mock.patch("qwen3_tts.tools.uninstall.uninstall_models")
    @mock.patch("qwen3_tts.tools.uninstall.uninstall_voices")
    def test_multiple_flags(self, mock_voices, mock_models):
        """Multiple flags call multiple functions."""
        from qwen3_tts.tools.uninstall import main

        with mock.patch("sys.argv",
                        ["uninstall", "--models", "--voices", "--dry-run"]):
            result = main()

        mock_models.assert_called_once_with(dry_run=True)
        mock_voices.assert_called_once_with(dry_run=True)
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
