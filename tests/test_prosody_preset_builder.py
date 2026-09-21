#!/usr/bin/env python3
"""Track T1: tests for the user-defined prosody preset builder (save/delete).

TDD RED phase — the data layer (core/config/presets.py additions), UI helpers
(interface/voice_helpers.py additions), and Custom-tab wiring
(interface/ui/tabs_generation.py) do not exist yet; every test here fails until
GREEN. Imports are function-level so a missing symbol surfaces as a per-test
failure, never a collection error.

Coverage:
  1. validate_prosody_preset_name — pure name validator
  2. get_user_prosody_presets — filtered view off the raw config key
  3. save_user_prosody_preset / delete_user_prosody_preset — disk-free CRUD
  4. Corrupt-config-load / save-time OSError failure paths (both writers)
  5. get_user_prosody_choices — delete-dropdown rendering
  6. _on_save_prosody_preset / _on_delete_prosody_preset handler contracts
  7. Custom-tab wiring structure
"""

import os
import time
import unittest
from unittest.mock import patch

try:
    import gradio as gr  # noqa: F401

    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False


# A config carrying all 8 factory prosody-preset keys (values are arbitrary —
# the factory-name check is against presets.py's own DEFAULT_PROSODY_PRESETS
# keys, not against whatever text a caller's config happens to hold).
SEEDED_FACTORY_CONFIG = {
    "prosody_presets": {
        "excited": "seeded excited text",
        "calm": "seeded calm text",
        "whisper": "seeded whisper text",
        "authoritative": "seeded authoritative text",
        "slow": "seeded slow text",
        "fast": "seeded fast text",
        "dramatic": "seeded dramatic text",
        "conversational": "seeded conversational text",
    }
}


class TestValidateProsodyPresetName(unittest.TestCase):
    """validate_prosody_preset_name(name) -> str | None."""

    def _validate(self, name):
        from qwen3_tts.core.config import validate_prosody_preset_name

        return validate_prosody_preset_name(name)

    def test_empty_name_rejected(self):
        self.assertIsNotNone(self._validate(""))

    def test_whitespace_only_name_rejected(self):
        self.assertIsNotNone(self._validate("   "))

    def test_name_over_max_len_rejected(self):
        from qwen3_tts.core.config import PROSODY_NAME_MAX_LEN

        self.assertIsNotNone(self._validate("a" * (PROSODY_NAME_MAX_LEN + 1)))

    def test_name_at_max_len_accepted(self):
        from qwen3_tts.core.config import PROSODY_NAME_MAX_LEN

        self.assertIsNone(self._validate("a" * PROSODY_NAME_MAX_LEN))

    def test_valid_name_accepted(self):
        self.assertIsNone(self._validate("storyteller"))

    def test_factory_name_lowercase_rejected(self):
        self.assertIsNotNone(self._validate("excited"))

    def test_factory_name_case_insensitive_rejected(self):
        self.assertIsNotNone(self._validate("EXCITED"))

    def test_none_choice_literal_rejected(self):
        self.assertIsNotNone(self._validate("(none)"))

    def test_choice_separator_in_name_rejected(self):
        self.assertIsNotNone(self._validate("slow - really"))

    def test_special_characters_rejected(self):
        self.assertIsNotNone(self._validate("my@preset"))

    def test_dash_underscore_dot_accepted(self):
        self.assertIsNone(self._validate("story-teller_v1.0"))


class TestGetUserProsodyPresets(unittest.TestCase):
    """get_user_prosody_presets(config=None) -> dict[str, str]."""

    def _get(self, config):
        from qwen3_tts.core.config import get_user_prosody_presets

        return get_user_prosody_presets(config=config)

    def test_factory_names_excluded(self):
        self.assertEqual(self._get({**SEEDED_FACTORY_CONFIG}), {})

    def test_user_only_entry_included(self):
        config = {
            "prosody_presets": {
                **SEEDED_FACTORY_CONFIG["prosody_presets"],
                "storyteller": "Warm and inviting",
            }
        }
        self.assertEqual(self._get(config), {"storyteller": "Warm and inviting"})

    def test_non_string_value_excluded(self):
        config = {"prosody_presets": {"junk": 123}}
        self.assertEqual(self._get(config), {})

    def test_empty_after_strip_excluded(self):
        config = {"prosody_presets": {"blank": "   "}}
        self.assertEqual(self._get(config), {})

    def test_returns_new_dict_not_alias(self):
        config = {"prosody_presets": {"storyteller": "text"}}
        result = self._get(config)
        result["mutated"] = "should not leak"
        self.assertNotIn("mutated", config["prosody_presets"])

    def test_missing_key_returns_empty_dict(self):
        self.assertEqual(self._get({}), {})


class TestSaveUserProsodyPreset(unittest.TestCase):
    """save_user_prosody_preset(name, instruct_text, config=None) -> tuple[bool, str]."""

    def _save(self, name, text, config):
        from qwen3_tts.core.config import save_user_prosody_preset

        return save_user_prosody_preset(name, text, config=config)

    @patch("qwen3_tts.core.config.save_config")
    def test_new_preset_saved_with_raw_base_preserved(self, mock_save):
        config = {
            "prosody_presets": {**SEEDED_FACTORY_CONFIG["prosody_presets"], "junk": 123}
        }
        ok, _msg = self._save("storyteller", "Warm and inviting", config)
        self.assertTrue(ok)
        saved_presets = mock_save.call_args[0][0]["prosody_presets"]
        # Raw base preserved verbatim — seeded factory entries AND the
        # non-str junk entry survive; the write base is never the filtered view.
        self.assertEqual(saved_presets["excited"], "seeded excited text")
        self.assertEqual(saved_presets["junk"], 123)
        self.assertEqual(saved_presets["storyteller"], "Warm and inviting")

    @patch("qwen3_tts.core.config.save_config")
    def test_name_and_text_stored_stripped(self, mock_save):
        self._save("  Foo  ", "  Speak warmly  ", {})
        saved_presets = mock_save.call_args[0][0]["prosody_presets"]
        self.assertIn("Foo", saved_presets)
        self.assertEqual(saved_presets["Foo"], "Speak warmly")

    @patch("qwen3_tts.core.config.save_config")
    def test_factory_name_collision_rejected(self, mock_save):
        ok, _msg = self._save("excited", "My version", {**SEEDED_FACTORY_CONFIG})
        self.assertFalse(ok)
        mock_save.assert_not_called()

    @patch("qwen3_tts.core.config.save_config")
    def test_duplicate_text_under_different_name_allowed(self, _mock_save):
        config = {"prosody_presets": {"a": "Speak warmly"}}
        ok, _msg = self._save("b", "Speak warmly", config)
        self.assertTrue(ok)

    @patch("qwen3_tts.core.config.save_config")
    def test_overwrite_own_preset_succeeds(self, mock_save):
        config = {"prosody_presets": {"storyteller": "old text"}}
        ok, msg = self._save("storyteller", "new text", config)
        self.assertTrue(ok)
        self.assertIn("Updated", msg)
        saved_presets = mock_save.call_args[0][0]["prosody_presets"]
        self.assertEqual(saved_presets["storyteller"], "new text")


class TestDeleteUserProsodyPreset(unittest.TestCase):
    """delete_user_prosody_preset(name, config=None) -> tuple[bool, str]."""

    def _delete(self, name, config):
        from qwen3_tts.core.config import delete_user_prosody_preset

        return delete_user_prosody_preset(name, config=config)

    @patch("qwen3_tts.core.config.save_config")
    def test_removes_only_named_raw_key(self, mock_save):
        config = {
            "prosody_presets": {
                **SEEDED_FACTORY_CONFIG["prosody_presets"],
                "storyteller": "text",
                "other": "text2",
            }
        }
        ok, _msg = self._delete("storyteller", config)
        self.assertTrue(ok)
        saved_presets = mock_save.call_args[0][0]["prosody_presets"]
        self.assertNotIn("storyteller", saved_presets)
        self.assertIn("other", saved_presets)
        self.assertIn("excited", saved_presets)  # factory entries untouched

    @patch("qwen3_tts.core.config.save_config")
    def test_non_string_junk_entry_deletable(self, _mock_save):
        config = {"prosody_presets": {"junk": 123}}
        ok, _msg = self._delete("junk", config)
        self.assertTrue(ok)

    @patch("qwen3_tts.core.config.save_config")
    def test_factory_name_delete_rejected(self, mock_save):
        ok, msg = self._delete("excited", {**SEEDED_FACTORY_CONFIG})
        self.assertFalse(ok)
        self.assertIn("built-in", msg)
        mock_save.assert_not_called()

    @patch("qwen3_tts.core.config.save_config")
    def test_unknown_name_delete_rejected(self, mock_save):
        ok, msg = self._delete("nonexistent", {})
        self.assertFalse(ok)
        self.assertIn("no longer exists", msg)
        mock_save.assert_not_called()


class TestProsodyPresetWriteFailurePaths(unittest.TestCase):
    """Corrupt-config-load and save-time OSError paths for both writers."""

    @patch(
        "qwen3_tts.core.config.load_config",
        side_effect=ValueError("config.json is corrupt"),
    )
    @patch("qwen3_tts.core.config.save_config")
    def test_save_corrupt_config_load_never_saves(self, mock_save, _mock_load):
        from qwen3_tts.core.config import save_user_prosody_preset

        ok, _msg = save_user_prosody_preset(
            "storyteller", "text"
        )  # config=None -> loads
        self.assertFalse(ok)
        mock_save.assert_not_called()

    @patch("qwen3_tts.core.config.load_config", side_effect=FileNotFoundError())
    @patch("qwen3_tts.core.config.save_config")
    def test_save_missing_config_file_never_saves(self, mock_save, _mock_load):
        from qwen3_tts.core.config import save_user_prosody_preset

        ok, _msg = save_user_prosody_preset("storyteller", "text")
        self.assertFalse(ok)
        mock_save.assert_not_called()

    @patch(
        "qwen3_tts.core.config.save_config",
        side_effect=OSError(28, "No space left on device"),
    )
    def test_save_oserror_on_write_reports_strerror(self, _mock_save):
        from qwen3_tts.core.config import save_user_prosody_preset

        ok, msg = save_user_prosody_preset("storyteller", "text", config={})
        self.assertFalse(ok)
        self.assertIn("No space left on device", msg)

    @patch(
        "qwen3_tts.core.config.load_config",
        side_effect=ValueError("config.json is corrupt"),
    )
    @patch("qwen3_tts.core.config.save_config")
    def test_delete_corrupt_config_load_never_saves(self, mock_save, _mock_load):
        from qwen3_tts.core.config import delete_user_prosody_preset

        ok, _msg = delete_user_prosody_preset("storyteller")
        self.assertFalse(ok)
        mock_save.assert_not_called()

    @patch(
        "qwen3_tts.core.config.save_config",
        side_effect=OSError(28, "No space left on device"),
    )
    def test_delete_oserror_on_write_reports_strerror(self, _mock_save):
        from qwen3_tts.core.config import delete_user_prosody_preset

        ok, msg = delete_user_prosody_preset(
            "storyteller", config={"prosody_presets": {"storyteller": "text"}}
        )
        self.assertFalse(ok)
        self.assertIn("No space left on device", msg)


class TestGetUserProsodyChoices(unittest.TestCase):
    """get_user_prosody_choices(config=None) -> list[str] for the delete dropdown."""

    def _choices(self, config):
        from qwen3_tts.interface.voice_helpers import get_user_prosody_choices

        return get_user_prosody_choices(config=config)

    def test_exact_literal_format(self):
        self.assertEqual(
            self._choices({"prosody_presets": {"zz": "text"}}), ["(none)", "zz - text"]
        )

    def test_none_first_when_empty(self):
        self.assertEqual(self._choices({}), ["(none)"])

    def test_sorted_alphabetically(self):
        config = {"prosody_presets": {"zebra": "a", "apple": "b"}}
        self.assertEqual(self._choices(config), ["(none)", "apple - b", "zebra - a"])

    def test_factory_names_excluded_from_seeded_config(self):
        self.assertEqual(self._choices({**SEEDED_FACTORY_CONFIG}), ["(none)"])

    def test_long_text_truncated_at_80_chars(self):
        long_text = "x" * 200
        choices = self._choices({"prosody_presets": {"long": long_text}})
        self.assertNotIn(long_text, choices[1])
        self.assertTrue(choices[1].endswith("…"))

    def test_none_choice_constant_matches_ui_literal(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, "qwen3_tts/interface/voice_helpers.py")) as f:
            helpers_src = f.read()
        with open(os.path.join(base, "qwen3_tts/interface/ui/tabs_generation.py")) as f:
            tabs_src = f.read()
        self.assertIn('PROSODY_NONE_CHOICE = "(none)"', helpers_src)
        self.assertIn('NONE_CHOICE = "(none)"', tabs_src)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestProsodyPresetHandlerContracts(unittest.TestCase):
    """Direct calls to the module-level save/delete handlers in tabs_generation.py."""

    def _save_handler(self):
        from qwen3_tts.interface.ui.tabs_generation import _on_save_prosody_preset

        return _on_save_prosody_preset

    def _delete_handler(self):
        from qwen3_tts.interface.ui.tabs_generation import _on_delete_prosody_preset

        return _on_delete_prosody_preset

    # --- Save handler ---

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices", return_value=["(none)"]
    )
    @patch(
        "qwen3_tts.core.config.save_user_prosody_preset",
        return_value=(True, "Saved preset 'storyteller'."),
    )
    @patch("qwen3_tts.core.config.get_user_prosody_presets", return_value={})
    @patch("qwen3_tts.core.config.validate_prosody_preset_name", return_value=None)
    def test_save_new_preset_outputs_arity_seven(self, *_mocks):
        fn = self._save_handler()
        state = {"armed": False, "ts": 0.0, "armed_name": None}
        result = fn(state, "storyteller", "Speak warmly", "(none)", "(none)")
        self.assertEqual(len(result), 7)
        self.assertFalse(result[0].get("armed"))
        self.assertIsNone(result[0].get("armed_name"))

    @patch(
        "qwen3_tts.core.config.validate_prosody_preset_name",
        return_value="'excited' is a built-in preset — pick a different name.",
    )
    def test_save_factory_name_hard_error_never_arms(self, _mock_validate):
        fn = self._save_handler()
        state = {"armed": False, "ts": 0.0, "armed_name": None}
        result = fn(state, "excited", "Speak warmly", "(none)", "(none)")
        self.assertFalse(result[0].get("armed"))

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    @patch("qwen3_tts.core.config.validate_prosody_preset_name", return_value=None)
    def test_save_existing_name_arms_instead_of_saving(self, *_mocks):
        fn = self._save_handler()
        state = {"armed": False, "ts": 0.0, "armed_name": None}
        result = fn(state, "storyteller", "new text", "(none)", "(none)")
        self.assertTrue(result[0].get("armed"))
        self.assertEqual(result[0].get("armed_name"), "storyteller")

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices", return_value=["(none)"]
    )
    @patch(
        "qwen3_tts.core.config.save_user_prosody_preset",
        return_value=(True, "Updated preset 'storyteller'."),
    )
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    @patch("qwen3_tts.core.config.validate_prosody_preset_name", return_value=None)
    def test_save_armed_fresh_confirm_saves_and_disarms(self, *_mocks):
        fn = self._save_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "storyteller", "new text", "(none)", "(none)")
        self.assertFalse(result[0].get("armed"))

    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old", "other": "x"},
    )
    @patch("qwen3_tts.core.config.validate_prosody_preset_name", return_value=None)
    def test_save_arm_then_change_target_rearms_not_applies(self, *_mocks):
        fn = self._save_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "other", "new text", "(none)", "(none)")
        self.assertTrue(result[0].get("armed"))
        self.assertEqual(result[0].get("armed_name"), "other")

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    @patch("qwen3_tts.core.config.validate_prosody_preset_name", return_value=None)
    def test_save_expired_arm_rearms_not_saves(self, *_mocks):
        fn = self._save_handler()
        state = {"armed": True, "ts": time.time() - 6, "armed_name": "storyteller"}
        result = fn(state, "storyteller", "new text", "(none)", "(none)")
        self.assertTrue(result[0].get("armed"))

    # --- Delete handler ---

    def test_delete_nothing_selected_disarmed(self):
        fn = self._delete_handler()
        state = {"armed": False, "ts": 0.0, "armed_name": None}
        result = fn(state, "(none)", "(none)", "(none)")
        self.assertEqual(len(result), 7)
        self.assertFalse(result[0].get("armed"))

    @patch(
        "qwen3_tts.core.config.delete_user_prosody_preset",
        return_value=(
            False,
            "Preset 'ghost' no longer exists — it may have been removed outside the UI.",
        ),
    )
    def test_delete_unknown_preset_renders_writer_message(self, _mock_delete):
        fn = self._delete_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "ghost"}
        result = fn(state, "ghost", "(none)", "(none)")
        self.assertIn("no longer exists", result[2])

    def test_delete_arms_on_first_click(self):
        fn = self._delete_handler()
        state = {"armed": False, "ts": 0.0, "armed_name": None}
        result = fn(state, "storyteller", "(none)", "(none)")
        self.assertTrue(result[0].get("armed"))
        self.assertEqual(result[0].get("armed_name"), "storyteller")

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices", return_value=["(none)"]
    )
    @patch(
        "qwen3_tts.core.config.delete_user_prosody_preset",
        return_value=(True, "Deleted preset 'storyteller'."),
    )
    def test_delete_armed_fresh_confirm_deletes_and_disarms(self, *_mocks):
        fn = self._delete_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "storyteller", "(none)", "(none)")
        self.assertFalse(result[0].get("armed"))

    def test_delete_expired_arm_rearms_not_deletes(self):
        fn = self._delete_handler()
        state = {"armed": True, "ts": time.time() - 6, "armed_name": "storyteller"}
        result = fn(state, "storyteller", "(none)", "(none)")
        self.assertTrue(result[0].get("armed"))

    @patch(
        "qwen3_tts.core.config.delete_user_prosody_preset",
        return_value=(True, "Deleted preset 'storyteller'."),
    )
    def test_delete_mismatched_current_selection_no_value_reset(self, _mock_delete):
        """Deleting 'storyteller' while an unrelated dropdown shows 'other - x'
        must not reset that unrelated dropdown's value (conditional-reset rule)."""
        fn = self._delete_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "storyteller", "other - x", "(none)")
        self.assertEqual(result[4], gr.update())  # custom_prosody untouched


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestProsodyPresetUiWiring(unittest.TestCase):
    """Structural assertions on tabs_generation.py's / _facade.py's wiring."""

    def _read_source(self, rel_path):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, rel_path)) as f:
            return f.read()

    def test_save_button_click_wires_expected_handler(self):
        src = self._read_source("qwen3_tts/interface/ui/tabs_generation.py")
        self.assertIn("prosody_preset_save_btn.click(", src)
        self.assertIn("fn=_on_save_prosody_preset", src)

    def test_delete_button_click_wires_expected_handler(self):
        src = self._read_source("qwen3_tts/interface/ui/tabs_generation.py")
        self.assertIn("prosody_preset_delete_btn.click(", src)
        self.assertIn("fn=_on_delete_prosody_preset", src)

    def test_design_prosody_and_custom_prosody_are_save_click_inputs(self):
        src = self._read_source("qwen3_tts/interface/ui/tabs_generation.py")
        # Both dropdowns must be readable by the handler (conditional-reset rule).
        save_block_start = src.index("prosody_preset_save_btn.click(")
        save_block = src[save_block_start : save_block_start + 400]
        self.assertIn("design_prosody", save_block)
        self.assertIn("custom_prosody", save_block)

    def test_amended_custom_prosody_info_string_present(self):
        src = self._read_source("qwen3_tts/interface/ui/tabs_generation.py")
        self.assertIn("then save it under 'My prosody presets'", src)

    def test_delete_button_uses_stop_variant(self):
        src = self._read_source("qwen3_tts/interface/ui/tabs_generation.py")
        idx = src.index("prosody_preset_delete_btn")
        snippet = src[idx : idx + 200]
        self.assertIn('variant="stop"', snippet)

    def test_design_tab_return_includes_design_prosody(self):
        src = self._read_source("qwen3_tts/interface/ui/tabs_generation.py")
        self.assertIn(
            'return design_model_indicator, design_chain, design_ctrls["seed"], design_prosody',
            src,
        )

    def test_facade_unpacks_design_prosody_from_design_tab(self):
        src = self._read_source("qwen3_tts/interface/ui/_facade.py")
        call_idx = src.index("_build_design_tab(")
        line_start = src.rfind("\n", 0, call_idx) + 1
        assignment_line = src[line_start:call_idx]
        self.assertIn("design_prosody", assignment_line)

    def test_facade_passes_design_prosody_into_custom_tab_call(self):
        src = self._read_source("qwen3_tts/interface/ui/_facade.py")
        call_idx = src.index("_build_custom_tab(")
        call_block = src[call_idx : call_idx + 150]
        self.assertIn("design_prosody", call_block)


if __name__ == "__main__":
    unittest.main()
