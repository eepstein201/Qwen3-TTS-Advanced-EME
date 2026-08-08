#!/usr/bin/env python3
"""Tests for hard-delete Remove: path-keyed two-step confirm + real file deletion.

Covers:
  - remove_history_row_by_path: path-keyed (not index) immutable removal
  - delete_generation_files: deletes .wav + .json, refuses paths outside
    Automated Output, tolerates missing files
  - on_history_select delete state machine: first click arms (no delete),
    second click on the SAME path within 5s deletes, a different row re-arms,
    an expired arm re-arms instead of deleting

The two-step confirm is keyed by path (not row index): a generation landing
between the two clicks prepends a row and shifts every index, so an
index-keyed confirm could delete the wrong entry. on_history_select's
signature is (evt, history_list, delete_confirm_state=None,
download_confirm_state=None). The 10-tuple return is
[audio, clone, design, custom, df, state, payload, status,
 delete_confirm_state(8), download_confirm_state(9)] — so the delete state
these tests inspect is result[8].

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_history_hard_delete.py -q
"""
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch


def _make_entry(directory, name, seed=42):
    """Create a .wav + .json pair under *directory* and return a history entry."""
    wav = os.path.join(directory, f"{name}.wav")
    js = os.path.join(directory, f"{name}.json")
    with open(wav, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 40)
    with open(js, "w") as f:
        f.write('{"timestamp": 1.0}')
    return {"path": wav, "seed": seed, "timestamp": 1.0, "text": name, "mode": "Clone"}


class TestRemoveHistoryRowByPath(unittest.TestCase):
    def test_removes_the_matching_entry_only(self):
        from qwen3_tts.interface.ui.shared import remove_history_row_by_path

        history = [{"path": "/a.wav"}, {"path": "/b.wav"}, {"path": "/c.wav"}]
        result = remove_history_row_by_path(history, "/b.wav")
        self.assertEqual([e["path"] for e in result], ["/a.wav", "/c.wav"])

    def test_unknown_path_returns_unchanged_copy(self):
        from qwen3_tts.interface.ui.shared import remove_history_row_by_path

        history = [{"path": "/a.wav"}]
        result = remove_history_row_by_path(history, "/nope.wav")
        self.assertEqual(result, history)
        self.assertIsNot(result, history)

    def test_does_not_mutate_input(self):
        from qwen3_tts.interface.ui.shared import remove_history_row_by_path

        history = [{"path": "/a.wav"}, {"path": "/b.wav"}]
        remove_history_row_by_path(history, "/a.wav")
        self.assertEqual(len(history), 2)

    def test_non_list_returns_empty(self):
        from qwen3_tts.interface.ui.shared import remove_history_row_by_path

        self.assertEqual(remove_history_row_by_path(None, "/a.wav"), [])


class TestDeleteGenerationFiles(unittest.TestCase):
    def test_deletes_wav_and_json_sidecar(self):
        from qwen3_tts.interface.ui import shared

        with tempfile.TemporaryDirectory() as tmp:
            automated = os.path.join(tmp, "Automated Output")
            os.makedirs(automated)
            entry = _make_entry(automated, "voice_ui_del")
            with patch.object(
                shared, "resolve_automated_output_dir", return_value=automated
            ):
                ok = shared.delete_generation_files(entry["path"], {})
            self.assertTrue(ok)
            self.assertFalse(os.path.exists(entry["path"]))
            self.assertFalse(os.path.exists(entry["path"].replace(".wav", ".json")))

    def test_refuses_a_path_outside_automated_output(self):
        from qwen3_tts.interface.ui import shared

        with tempfile.TemporaryDirectory() as tmp:
            automated = os.path.join(tmp, "Automated Output")
            os.makedirs(automated)
            outsider = os.path.join(tmp, "elsewhere.wav")
            with open(outsider, "wb") as f:
                f.write(b"RIFF")
            with patch.object(
                shared, "resolve_automated_output_dir", return_value=automated
            ):
                ok = shared.delete_generation_files(outsider, {})
            self.assertFalse(ok)
            self.assertTrue(os.path.exists(outsider), "must not delete outside files")

    def test_missing_file_is_not_an_error(self):
        from qwen3_tts.interface.ui import shared

        with tempfile.TemporaryDirectory() as tmp:
            automated = os.path.join(tmp, "Automated Output")
            os.makedirs(automated)
            with patch.object(
                shared, "resolve_automated_output_dir", return_value=automated
            ):
                ok = shared.delete_generation_files(
                    os.path.join(automated, "ghost.wav"), {}
                )
            self.assertTrue(ok)


class TestDeleteConfirmStateMachine(unittest.TestCase):
    """First click arms the clicked row; second click on the SAME path deletes."""

    def _click(self, state, history, row, col=5, config_dir=None):
        from qwen3_tts.interface.ui import history_panel, shared

        evt = MagicMock()
        evt.index = [row, col]
        with patch.object(
            shared, "resolve_automated_output_dir", return_value=config_dir
        ), patch("qwen3_tts.core.config.load_config", return_value={}):
            return history_panel.on_history_select(evt, history, state)

    def test_first_click_arms_without_deleting(self):
        with tempfile.TemporaryDirectory() as tmp:
            automated = os.path.join(tmp, "Automated Output")
            os.makedirs(automated)
            entry = _make_entry(automated, "voice_ui_arm")
            result = self._click({}, [entry], 0, config_dir=automated)
            self.assertTrue(
                os.path.exists(entry["path"]), "first click must not delete"
            )
            new_state = result[8]
            self.assertEqual(new_state.get("armed_path"), entry["path"])

    def test_second_click_same_path_deletes(self):
        with tempfile.TemporaryDirectory() as tmp:
            automated = os.path.join(tmp, "Automated Output")
            os.makedirs(automated)
            entry = _make_entry(automated, "voice_ui_confirm")
            armed = self._click({}, [entry], 0, config_dir=automated)[8]
            self._click(armed, [entry], 0, config_dir=automated)
            self.assertFalse(os.path.exists(entry["path"]))

    def test_click_on_a_different_row_rearms_instead_of_deleting(self):
        with tempfile.TemporaryDirectory() as tmp:
            automated = os.path.join(tmp, "Automated Output")
            os.makedirs(automated)
            a = _make_entry(automated, "voice_ui_a")
            b = _make_entry(automated, "voice_ui_b")
            armed_a = self._click({}, [a, b], 0, config_dir=automated)[8]
            result = self._click(armed_a, [a, b], 1, config_dir=automated)
            self.assertTrue(os.path.exists(a["path"]), "row A must survive")
            self.assertTrue(os.path.exists(b["path"]), "row B must only be armed")
            self.assertEqual(result[8].get("armed_path"), b["path"])

    def test_expired_arm_rearms_instead_of_deleting(self):
        with tempfile.TemporaryDirectory() as tmp:
            automated = os.path.join(tmp, "Automated Output")
            os.makedirs(automated)
            entry = _make_entry(automated, "voice_ui_expired")
            stale = {"armed_path": entry["path"], "ts": 0.0}  # epoch → long expired
            result = self._click(stale, [entry], 0, config_dir=automated)
            self.assertTrue(os.path.exists(entry["path"]))
            self.assertEqual(result[8].get("armed_path"), entry["path"])


class TestClearAllHardDelete(unittest.TestCase):
    """Clear All must delete the underlying files, not just empty the list.

    The generation chain re-derives the table from disk
    (``refresh_history_from_disk`` ignores the in-memory list), so a list-only
    clear resurrects every row on the next Generate. This regressed when commit
    e42bce0 wired the disk re-derive into the generation chain; Clear All's
    list-only behavior predates it. These tests drive the real resolvers (HOME
    under a temp dir, real config) — NOT patched — so the production path is
    exercised end to end.
    """

    def _seed(self, automated, count=3):
        import json as json_mod

        for i in range(1, count + 1):
            wav = os.path.join(automated, f"voice_ui_{i}.wav")
            with open(wav, "wb") as f:
                f.write(b"RIFF" + b"\x00" * 40)
            with open(wav.replace(".wav", ".json"), "w") as f:
                json_mod.dump(
                    {
                        "timestamp": float(i),
                        "mode": "clone",
                        "text": f"t{i}",
                        "seed": i * 100,
                    },
                    f,
                )

    def test_clear_all_deletes_files_so_rows_do_not_resurrect(self):
        from qwen3_tts.interface.ui import history_panel, shared

        with tempfile.TemporaryDirectory() as tmp:
            # history_output_directory == HOME so resolve_*'s home-containment
            # guard passes without patching the resolvers (real production path).
            automated = os.path.join(tmp, "Automated Output")
            os.makedirs(automated)
            self._seed(automated, count=3)
            config = {"history_output_directory": tmp}

            with patch.dict(os.environ, {"HOME": tmp}), patch(
                "qwen3_tts.core.config.load_config", return_value=config
            ):
                history = shared.load_history_from_disk_for_config(config)
                self.assertEqual(len(history), 3)

                # First click arms; second click confirms + deletes. The arm
                # branch returns gr.update() for history_state (output [3]),
                # i.e. "leave unchanged", so the confirming call receives the
                # same list the browser still holds — not armed[3].
                armed = history_panel.on_clear_history_click(
                    {"armed": False, "ts": 0.0}, history
                )
                self.assertTrue(armed[0].get("armed"))  # armed, no delete yet
                self.assertEqual(
                    len(os.listdir(automated)), 6, "arming must not delete"
                )
                result = history_panel.on_clear_history_click(armed[0], history)

                # Files actually removed from disk (.wav + .json sidecars).
                self.assertEqual(sorted(os.listdir(automated)), [])

                # In-memory list cleared.
                self.assertEqual(result[3], [])  # history_state

                # CRITICAL: the next generation's disk re-derive must NOT
                # resurrect the cleared rows.
                entries, rows = shared.refresh_history_from_disk([], config)
                self.assertEqual(entries, [])
                self.assertEqual(rows, [])

    def test_clear_all_on_empty_list_is_a_safe_noop(self):
        from qwen3_tts.interface.ui import history_panel

        with tempfile.TemporaryDirectory() as tmp:
            automated = os.path.join(tmp, "Automated Output")
            os.makedirs(automated)
            config = {"history_output_directory": tmp}

            with patch.dict(os.environ, {"HOME": tmp}), patch(
                "qwen3_tts.core.config.load_config", return_value=config
            ):
                armed = history_panel.on_clear_history_click(
                    {"armed": False, "ts": 0.0}, []
                )
                # See note above: confirming call receives the unchanged list.
                result = history_panel.on_clear_history_click(armed[0], [])

                self.assertEqual(result[3], [])
                self.assertEqual(sorted(os.listdir(automated)), [])


if __name__ == "__main__":
    unittest.main()
