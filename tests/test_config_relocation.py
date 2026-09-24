"""Track T4 — runtime config lives at ~/.config/qwen3-tts/config.json.

The repo-root config.json stays a tracked stock default (CI reads it); the
user's live config — including user-named presets — must never be written
into the version-controlled checkout. Reads walk canonical -> legacy
(~/Qwen3-TTS_UserFiles/config.json) -> repo root; writes always go to the
canonical path.

Spec: docs/plans/2026-09-22-t4-config-relocation.plan.md
"""

import importlib
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from qwen3_tts.core import config as cfg


class _ThreeLocationsTestCase(unittest.TestCase):
    """Point the canonical, legacy, and repo config paths into a temp dir."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = pathlib.Path(tmp.name)
        self.canonical = root / "home" / ".config" / "qwen3-tts" / "config.json"
        self.legacy = root / "home" / "Qwen3-TTS_UserFiles" / "config.json"
        self.repo = root / "repo" / "config.json"

        for name, value in (
            ("CONFIG_PATH", str(self.canonical)),
            ("_LEGACY_CONFIG_PATH", str(self.legacy)),
            ("_REPO_CONFIG_PATH", str(self.repo)),
        ):
            patcher = mock.patch.object(cfg, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

        self._reset_cache()
        self.addCleanup(self._reset_cache)

    @staticmethod
    def _reset_cache():
        cfg._config_cache["data"] = None
        cfg._config_cache["mtime"] = 0

    @staticmethod
    def _write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))


class TestCanonicalConfigPath(unittest.TestCase):
    def test_config_path_is_under_dot_config(self):
        from qwen3_tts.core.config import paths

        self.assertEqual(
            paths.CONFIG_PATH,
            os.path.expanduser("~/.config/qwen3-tts/config.json"),
        )

    def test_config_path_does_not_probe_the_filesystem(self):
        from qwen3_tts.core.config import paths

        legacy = os.path.join(paths.USER_FILES_DIR, "config.json")
        with mock.patch("os.path.exists", return_value=True):
            everything = importlib.reload(paths).CONFIG_PATH
        with mock.patch("os.path.exists", side_effect=lambda p: p != legacy):
            no_legacy = importlib.reload(paths).CONFIG_PATH
        with mock.patch("os.path.exists", return_value=False):
            nothing = importlib.reload(paths).CONFIG_PATH
        importlib.reload(paths)
        self.assertEqual(everything, no_legacy)
        self.assertEqual(everything, nothing)

    def test_legacy_and_repo_paths_are_distinct_from_canonical(self):
        from qwen3_tts.core.config import paths

        self.assertEqual(
            paths._LEGACY_CONFIG_PATH,
            os.path.join(paths.USER_FILES_DIR, "config.json"),
        )
        self.assertTrue(paths._REPO_CONFIG_PATH.endswith("config.json"))
        self.assertNotEqual(paths._REPO_CONFIG_PATH, paths.CONFIG_PATH)
        self.assertNotEqual(paths._LEGACY_CONFIG_PATH, paths.CONFIG_PATH)


class TestLoadConfigReadChain(_ThreeLocationsTestCase):
    def test_prefers_canonical_when_present(self):
        self._write(self.canonical, {"marker": "canonical"})
        self._write(self.legacy, {"marker": "legacy"})
        self._write(self.repo, {"marker": "repo"})

        self.assertEqual(cfg.load_config()["marker"], "canonical")

    def test_falls_back_to_legacy_and_warns(self):
        self._write(self.legacy, {"marker": "legacy"})
        self._write(self.repo, {"marker": "repo"})

        with self.assertLogs("tts", level="WARNING") as logs:
            loaded = cfg.load_config()

        self.assertEqual(loaded["marker"], "legacy")
        joined = "\n".join(logs.output)
        self.assertIn(str(self.legacy), joined)
        self.assertIn(str(self.canonical), joined)

    def test_falls_back_to_repo_when_neither_exists(self):
        self._write(self.repo, {"marker": "repo"})

        self.assertEqual(cfg.load_config()["marker"], "repo")

    def test_missing_everywhere_raises_on_canonical(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            cfg.load_config()

        self.assertEqual(ctx.exception.filename, str(self.canonical))

    def test_resolver_returns_first_existing_location(self):
        self._write(self.repo, {})
        self.assertEqual(cfg.resolve_config_read_path(), str(self.repo))
        self._write(self.legacy, {})
        self.assertEqual(cfg.resolve_config_read_path(), str(self.legacy))
        self._write(self.canonical, {})
        self.assertEqual(cfg.resolve_config_read_path(), str(self.canonical))


class TestSaveConfigTarget(_ThreeLocationsTestCase):
    def test_save_writes_canonical_even_when_read_came_from_legacy(self):
        self._write(self.legacy, {"marker": "legacy"})
        loaded = cfg.load_config()

        cfg.save_config({**loaded, "marker": "saved"})

        self.assertEqual(json.loads(self.canonical.read_text())["marker"], "saved")
        self.assertEqual(cfg.load_config()["marker"], "saved")

    def test_save_creates_the_config_dir(self):
        self.assertFalse(self.canonical.parent.exists())

        cfg.save_config({"marker": "saved"})

        self.assertTrue(self.canonical.is_file())

    def test_save_never_modifies_legacy_or_repo(self):
        self._write(self.legacy, {"marker": "legacy"})
        self._write(self.repo, {"marker": "repo"})
        before = {
            p: (p.read_bytes(), os.stat(p).st_mtime_ns)
            for p in (self.legacy, self.repo)
        }

        cfg.save_config({"marker": "saved"})

        for path, (content, mtime) in before.items():
            self.assertEqual(path.read_bytes(), content, path)
            self.assertEqual(os.stat(path).st_mtime_ns, mtime, path)


class TestUninstallConfigRelocated(_ThreeLocationsTestCase):
    def test_reset_with_string_config_path_backs_up_outside_checkout(self):
        # CONFIG_PATH is a str; uninstall_config used to call .exists() on it.
        from qwen3_tts.tools.uninstall import uninstall_config

        self._write(self.legacy, {"marker": "legacy"})

        with mock.patch("builtins.print"):
            uninstall_config()

        backup = self.canonical.with_suffix(".backup")
        self.assertEqual(json.loads(backup.read_text())["marker"], "legacy")
        self.assertTrue(self.canonical.is_file())
        self.assertEqual(
            sorted(p.name for p in self.legacy.parent.iterdir()), ["config.json"]
        )

    def test_reports_missing_when_no_config_anywhere(self):
        from qwen3_tts.tools.uninstall import uninstall_config

        with mock.patch("builtins.print") as mock_print:
            uninstall_config()

        printed = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("No config.json", printed)
        self.assertFalse(self.canonical.exists())


class TestHealthcheckConfigRelocated(_ThreeLocationsTestCase):
    def test_legacy_only_warns_with_migration_hint(self):
        from qwen3_tts.tools.healthcheck import check_config

        self._write(self.legacy, {})

        status, details = check_config()

        self.assertEqual(status, "warn")
        self.assertIn(str(self.canonical), details)

    def test_canonical_present_passes(self):
        from qwen3_tts.tools.healthcheck import check_config

        self._write(self.canonical, {})

        status, _ = check_config()

        self.assertEqual(status, "pass")


class TestClientDefaultConfigPath(_ThreeLocationsTestCase):
    def test_default_client_reads_through_the_chain(self):
        from qwen3_tts.server.client._base import _ClientBase

        self._write(self.legacy, {"marker": "legacy"})

        client = _ClientBase()
        self.addCleanup(client.close)

        self.assertEqual(client.config_path, str(self.legacy))


if __name__ == "__main__":
    unittest.main()
