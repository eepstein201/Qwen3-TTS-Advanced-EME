#!/usr/bin/env python3
"""Tests for the per-row Download action (copy into Manual Downloads).

Covers:
  - copy_to_manual_downloads: copies on no collision, reports "exists" without
    overwriting, overwrite=True replaces, refuses sources outside Automated Output.
  - on_history_select download state machine: a name collision arms (no overwrite);
    a second click on the SAME armed path within the timeout overwrites.

on_history_select's signature is (evt, history_list, delete_confirm_state=None,
download_confirm_state=None) — both states are trailing optionals so existing
callers keep working.

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_history_download.py -q
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class TestCopyToManualDownloads(unittest.TestCase):
    def _dirs(self, tmp):
        automated = os.path.join(tmp, "Automated Output")
        manual = os.path.join(tmp, "Manual Downloads")
        os.makedirs(automated)
        os.makedirs(manual)
        return automated, manual

    def _src(self, automated, name="voice_ui_dl.wav", body=b"RIFF"):
        path = os.path.join(automated, name)
        with open(path, "wb") as f:
            f.write(body + b"\x00" * 40)
        return path

    def test_copies_when_no_collision(self):
        from qwen3_tts.interface.ui import shared

        with tempfile.TemporaryDirectory() as tmp:
            automated, manual = self._dirs(tmp)
            src = self._src(automated)
            with patch.object(shared, "resolve_automated_output_dir", return_value=automated), \
                 patch.object(shared, "resolve_manual_downloads_dir", return_value=manual):
                result = shared.copy_to_manual_downloads(src, {})
            self.assertEqual(result, "copied")
            self.assertTrue(os.path.exists(os.path.join(manual, "voice_ui_dl.wav")))

    def test_reports_exists_without_overwriting(self):
        from qwen3_tts.interface.ui import shared

        with tempfile.TemporaryDirectory() as tmp:
            automated, manual = self._dirs(tmp)
            src = self._src(automated, body=b"NEW1")
            with open(os.path.join(manual, "voice_ui_dl.wav"), "wb") as f:
                f.write(b"OLD0")
            with patch.object(shared, "resolve_automated_output_dir", return_value=automated), \
                 patch.object(shared, "resolve_manual_downloads_dir", return_value=manual):
                result = shared.copy_to_manual_downloads(src, {})
            self.assertEqual(result, "exists")
            with open(os.path.join(manual, "voice_ui_dl.wav"), "rb") as f:
                self.assertTrue(f.read().startswith(b"OLD0"), "must not overwrite")

    def test_overwrite_true_replaces_the_file(self):
        from qwen3_tts.interface.ui import shared

        with tempfile.TemporaryDirectory() as tmp:
            automated, manual = self._dirs(tmp)
            src = self._src(automated, body=b"NEW1")
            with open(os.path.join(manual, "voice_ui_dl.wav"), "wb") as f:
                f.write(b"OLD0")
            with patch.object(shared, "resolve_automated_output_dir", return_value=automated), \
                 patch.object(shared, "resolve_manual_downloads_dir", return_value=manual):
                result = shared.copy_to_manual_downloads(src, {}, overwrite=True)
            self.assertEqual(result, "copied")
            with open(os.path.join(manual, "voice_ui_dl.wav"), "rb") as f:
                self.assertTrue(f.read().startswith(b"NEW1"))

    def test_refuses_a_source_outside_automated_output(self):
        from qwen3_tts.interface.ui import shared

        with tempfile.TemporaryDirectory() as tmp:
            automated, manual = self._dirs(tmp)
            outsider = os.path.join(tmp, "outside.wav")
            with open(outsider, "wb") as f:
                f.write(b"RIFF")
            with patch.object(shared, "resolve_automated_output_dir", return_value=automated), \
                 patch.object(shared, "resolve_manual_downloads_dir", return_value=manual):
                result = shared.copy_to_manual_downloads(outsider, {})
            self.assertEqual(result, "refused")
            self.assertFalse(os.path.exists(os.path.join(manual, "outside.wav")))


class TestDownloadConfirmStateMachine(unittest.TestCase):
    """On a name collision, first click arms; second click on the same path overwrites."""

    def _click(self, dl_state, history, row, automated, manual, col=6):
        from qwen3_tts.interface.ui import history_panel, shared

        evt = MagicMock()
        evt.index = [row, col]
        with patch.object(shared, "resolve_automated_output_dir", return_value=automated), \
             patch.object(shared, "resolve_manual_downloads_dir", return_value=manual), \
             patch("qwen3_tts.core.config.load_config", return_value={}):
            return history_panel.on_history_select(evt, history, {}, dl_state)

    def test_collision_first_click_arms_without_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            automated = os.path.join(tmp, "Automated Output")
            manual = os.path.join(tmp, "Manual Downloads")
            os.makedirs(automated)
            os.makedirs(manual)
            src = os.path.join(automated, "voice_ui_dl.wav")
            with open(src, "wb") as f:
                f.write(b"NEW1" + b"\x00" * 4)
            manual_copy = os.path.join(manual, "voice_ui_dl.wav")
            with open(manual_copy, "wb") as f:
                f.write(b"OLD0")
            entry = {"path": src, "seed": 1, "timestamp": 1.0, "text": "hi", "mode": "Clone"}

            result = self._click({}, [entry], 0, automated, manual)
            # Manual copy untouched (armed, not overwritten).
            with open(manual_copy, "rb") as f:
                self.assertTrue(f.read().startswith(b"OLD0"))
            self.assertEqual(result[-1].get("armed_path"), src)

    def test_collision_second_click_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            automated = os.path.join(tmp, "Automated Output")
            manual = os.path.join(tmp, "Manual Downloads")
            os.makedirs(automated)
            os.makedirs(manual)
            src = os.path.join(automated, "voice_ui_dl.wav")
            with open(src, "wb") as f:
                f.write(b"NEW1" + b"\x00" * 4)
            manual_copy = os.path.join(manual, "voice_ui_dl.wav")
            with open(manual_copy, "wb") as f:
                f.write(b"OLD0")
            entry = {"path": src, "seed": 1, "timestamp": 1.0, "text": "hi", "mode": "Clone"}

            armed = self._click({}, [entry], 0, automated, manual)[-1]
            self._click(armed, [entry], 0, automated, manual)
            with open(manual_copy, "rb") as f:
                self.assertTrue(f.read().startswith(b"NEW1"), "second click must overwrite")


if __name__ == "__main__":
    unittest.main()
