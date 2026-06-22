#!/usr/bin/env python3
"""Extended CLI tests using Click's CliRunner.

Covers uncovered Click commands:
  - voice list/delete/rename/preview/info
  - list speakers/presets/aliases/prosody/backends
  - config show/path/edit
  - uninstall models/voices/config/all
  - cache list/size/prune/clear
  - doctor, history, stats

Run: pytest tests/test_cli_ext.py -v
"""
import unittest
from unittest.mock import MagicMock, patch

try:
    from click.testing import CliRunner
    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

# Source modules where lazy imports originate
_CFG = "qwen3_tts.core.config"
_GEN = "qwen3_tts.interface.generate"
_HC = "qwen3_tts.tools.healthcheck"
_UNINST = "qwen3_tts.tools.uninstall"
_MCACHE = "qwen3_tts.tools.model_cache"


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestVoiceList(unittest.TestCase):

    def test_no_prompts(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_GEN}.list_voice_prompts", return_value=[]):
            result = runner.invoke(cli, ["voice", "list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No voice prompts", result.output)

    def test_with_prompts(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_GEN}.list_voice_prompts", return_value=["voice1.wav", "voice2.wav"]), \
             patch(f"{_CFG}.get_default_clone_prompt", return_value="voice1.wav"):
            result = runner.invoke(cli, ["voice", "list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("voice1.wav", result.output)
        self.assertIn("(default)", result.output)
        self.assertIn("voice2.wav", result.output)


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestVoiceDelete(unittest.TestCase):

    def test_delete(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_GEN}.delete_voice_prompt") as mock_del:
            result = runner.invoke(cli, ["voice", "delete", "my_voice"])
        self.assertEqual(result.exit_code, 0)
        mock_del.assert_called_once_with("my_voice")


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestVoiceRename(unittest.TestCase):

    def test_rename(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_GEN}.rename_voice_prompt") as mock_ren:
            result = runner.invoke(cli, ["voice", "rename", "old", "new"])
        self.assertEqual(result.exit_code, 0)
        mock_ren.assert_called_once_with("old", "new")


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestVoicePreview(unittest.TestCase):

    def test_preview(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_GEN}.preview_voice_prompt") as mock_preview, \
             patch(f"{_CFG}.load_config", return_value={}):
            result = runner.invoke(cli, ["voice", "preview", "my_voice"])
        self.assertEqual(result.exit_code, 0)
        mock_preview.assert_called_once()


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestVoiceInfo(unittest.TestCase):

    def test_server_not_running(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_CFG}.load_config", return_value={}), \
             patch(f"{_CFG}.is_server_running", return_value=False):
            result = runner.invoke(cli, ["voice", "info", "my_voice"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Server not running", result.output)

    def test_success(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"name": "my_voice", "formats": [".wav", ".txt"]}
        with patch(f"{_CFG}.load_config", return_value={}), \
             patch(f"{_CFG}.is_server_running", return_value=True), \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
            result = runner.invoke(cli, ["voice", "info", "my_voice"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("my_voice", result.output)


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestListSpeakers(unittest.TestCase):

    def test_speakers(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        speakers = {"Anna": {"lang": "en", "desc": "A warm voice"}}
        with patch(f"{_CFG}.CUSTOM_VOICE_SPEAKERS", speakers):
            result = runner.invoke(cli, ["list", "speakers"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Anna", result.output)


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestListPresets(unittest.TestCase):

    def test_no_presets(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_CFG}.load_config", return_value={"presets": {}}):
            result = runner.invoke(cli, ["list", "presets"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No presets", result.output)

    def test_with_presets(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        cfg = {"presets": {"warm": {"temperature": 0.8, "top_k": 50}}}
        with patch(f"{_CFG}.load_config", return_value=cfg):
            result = runner.invoke(cli, ["list", "presets"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("warm", result.output)
        self.assertIn("temperature=0.8", result.output)


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestListAliases(unittest.TestCase):

    def test_no_aliases(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_CFG}.load_config", return_value={"aliases": {}}):
            result = runner.invoke(cli, ["list", "aliases"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No aliases", result.output)

    def test_with_aliases(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        cfg = {"aliases": {"friend": {"prompt": "voice1.wav", "preset": "warm", "mode": "clone"}}}
        with patch(f"{_CFG}.load_config", return_value=cfg):
            result = runner.invoke(cli, ["list", "aliases"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("friend", result.output)
        self.assertIn("prompt=voice1.wav", result.output)


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestListProsody(unittest.TestCase):

    def test_no_prosody(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_CFG}.get_prosody_presets", return_value={}):
            result = runner.invoke(cli, ["list", "prosody"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No prosody", result.output)

    def test_with_prosody(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_CFG}.get_prosody_presets", return_value={"happy": "[happy]"}):
            result = runner.invoke(cli, ["list", "prosody"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("happy", result.output)


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestListBackends(unittest.TestCase):

    def test_mlx_backend(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_CFG}.get_backend", return_value="mlx"), \
             patch(f"{_CFG}.VALID_BACKENDS", ["mlx", "torch"]), \
             patch(f"{_CFG}.get_mlx_quantization", return_value="8bit"), \
             patch(f"{_CFG}.get_mlx_model_name", return_value="mlx-community/model"), \
             patch(f"{_CFG}.get_torch_dtype_name", return_value="float16"), \
             patch(f"{_CFG}.MODEL_INFO", {}):
            result = runner.invoke(cli, ["list", "backends"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("mlx", result.output)
        self.assertIn("8bit", result.output)


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestConfigShow(unittest.TestCase):

    def test_show(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_CFG}.load_config", return_value={"advanced": {"backend": "mlx"}}):
            result = runner.invoke(cli, ["config", "show"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("mlx", result.output)


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestConfigPath(unittest.TestCase):

    def test_path(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_CFG}.CONFIG_PATH", "/fake/config.json"):
            result = runner.invoke(cli, ["config", "path"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("/fake/config.json", result.output)


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestConfigEdit(unittest.TestCase):

    def test_edit_no_change(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        cfg = {"default_voice_description": "warm voice"}
        with patch(f"{_CFG}.load_config", return_value=cfg), \
             patch(f"{_CFG}.save_config"):
            result = runner.invoke(cli, ["config", "edit"], input="warm voice\n")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No changes", result.output)

    def test_edit_changed(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        cfg = {"default_voice_description": "warm voice"}
        with patch(f"{_CFG}.load_config", return_value=cfg), \
             patch(f"{_CFG}.save_config") as mock_save:
            result = runner.invoke(cli, ["config", "edit"], input="cool voice\n")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("updated", result.output)
        mock_save.assert_called_once()


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestDoctor(unittest.TestCase):

    def test_doctor(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_HC}.run_healthcheck", return_value=0):
            result = runner.invoke(cli, ["doctor"])
        self.assertEqual(result.exit_code, 0)


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestHistory(unittest.TestCase):

    def test_history(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_GEN}.show_history") as mock_hist:
            result = runner.invoke(cli, ["history", "5"])
        self.assertEqual(result.exit_code, 0)
        mock_hist.assert_called_once_with(5)


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestStats(unittest.TestCase):

    def test_server_not_running(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_CFG}.load_config", return_value={}), \
             patch(f"{_CFG}.is_server_running", return_value=False):
            result = runner.invoke(cli, ["stats"])
        self.assertNotEqual(result.exit_code, 0)

    def test_success(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"uptime": 100}
        with patch(f"{_CFG}.load_config", return_value={}), \
             patch(f"{_CFG}.is_server_running", return_value=True), \
             patch("qwen3_tts.core.http_client.server_request", return_value=mock_resp):
            result = runner.invoke(cli, ["stats"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("uptime", result.output)


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestUninstallCommands(unittest.TestCase):

    def test_uninstall_models(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_UNINST}.uninstall_models") as mock_fn:
            result = runner.invoke(cli, ["uninstall", "models", "--dry-run"])
        self.assertEqual(result.exit_code, 0)
        mock_fn.assert_called_once_with(dry_run=True)

    def test_uninstall_voices(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_UNINST}.uninstall_voices") as mock_fn:
            result = runner.invoke(cli, ["uninstall", "voices", "--dry-run"])
        self.assertEqual(result.exit_code, 0)
        mock_fn.assert_called_once_with(dry_run=True)

    def test_uninstall_config(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_UNINST}.uninstall_config") as mock_fn:
            result = runner.invoke(cli, ["uninstall", "config", "--dry-run"])
        self.assertEqual(result.exit_code, 0)
        mock_fn.assert_called_once_with(dry_run=True)

    def test_uninstall_all(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_UNINST}.uninstall_all") as mock_fn:
            result = runner.invoke(cli, ["uninstall", "all", "--dry-run"])
        self.assertEqual(result.exit_code, 0)
        mock_fn.assert_called_once_with(dry_run=True)


@unittest.skipUnless(HAS_CLICK, "requires click")
class TestCacheCommands(unittest.TestCase):

    def test_cache_list(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_MCACHE}.list_models_cmd") as mock_fn:
            result = runner.invoke(cli, ["cache", "list"])
        self.assertEqual(result.exit_code, 0)
        mock_fn.assert_called_once()

    def test_cache_size(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_MCACHE}.get_size_cmd") as mock_fn:
            result = runner.invoke(cli, ["cache", "size"])
        self.assertEqual(result.exit_code, 0)
        mock_fn.assert_called_once()

    def test_cache_prune(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_MCACHE}.prune_models_cmd") as mock_fn:
            result = runner.invoke(cli, ["cache", "prune", "--unused", "30"])
        self.assertEqual(result.exit_code, 0)
        mock_fn.assert_called_once_with(days=30, dry_run=False)

    def test_cache_clear(self):
        from qwen3_tts.cli import cli
        runner = CliRunner()
        with patch(f"{_MCACHE}.clear_cache_cmd") as mock_fn:
            result = runner.invoke(cli, ["cache", "clear", "--force"])
        self.assertEqual(result.exit_code, 0)
        mock_fn.assert_called_once_with(force=True)


if __name__ == "__main__":
    unittest.main()
