#!/usr/bin/env python3
"""Tests for history output path resolution (Automated Output / Manual Downloads)."""
import os
import unittest


class TestHistoryOutputPathResolvers(unittest.TestCase):
    def test_default_parent_is_qwen3_tts_output_under_downloads(self):
        from qwen3_tts.interface.ui.shared import resolve_history_output_dir

        result = resolve_history_output_dir({})
        expected = os.path.realpath(
            os.path.expanduser("~/Downloads/Qwen3-TTS Output")
        )
        self.assertEqual(result, expected)

    def test_subdirs_are_fixed_names_under_the_parent(self):
        from qwen3_tts.interface.ui.shared import (
            resolve_automated_output_dir,
            resolve_manual_downloads_dir,
        )

        config = {"history_output_directory": "~/Downloads/Custom Location"}
        base = os.path.realpath(os.path.expanduser("~/Downloads/Custom Location"))
        self.assertEqual(
            resolve_automated_output_dir(config),
            os.path.join(base, "Automated Output"),
        )
        self.assertEqual(
            resolve_manual_downloads_dir(config),
            os.path.join(base, "Manual Downloads"),
        )

    def test_path_outside_home_falls_back_to_default(self):
        from qwen3_tts.interface.ui.shared import resolve_history_output_dir

        result = resolve_history_output_dir({"history_output_directory": "/etc"})
        expected = os.path.realpath(
            os.path.expanduser("~/Downloads/Qwen3-TTS Output")
        )
        self.assertEqual(result, expected)

    def test_traversal_falls_back_to_default(self):
        from qwen3_tts.interface.ui.shared import resolve_history_output_dir

        result = resolve_history_output_dir(
            {"history_output_directory": "~/Downloads/../../../etc"}
        )
        expected = os.path.realpath(
            os.path.expanduser("~/Downloads/Qwen3-TTS Output")
        )
        self.assertEqual(result, expected)

    def test_resolvers_do_not_create_directories(self):
        from qwen3_tts.interface.ui.shared import resolve_automated_output_dir

        config = {"history_output_directory": "~/Downloads/NotCreatedByResolver"}
        path = resolve_automated_output_dir(config)
        self.assertFalse(os.path.exists(path), f"resolver created {path}")

    def test_default_config_contains_the_new_key(self):
        from qwen3_tts.core.config import get_default_config

        self.assertEqual(
            get_default_config().get("history_output_directory"),
            "~/Downloads/Qwen3-TTS Output",
        )


if __name__ == "__main__":
    unittest.main()
