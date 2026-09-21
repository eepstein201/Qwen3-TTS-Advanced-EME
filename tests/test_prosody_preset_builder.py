#!/usr/bin/env python3
"""Track T1: tests for the user-defined prosody preset builder (save/delete).

TDD RED phase, round 2 (strengthened after Gate A round 1 = 2xFAIL) — the data
layer (core/config/presets.py additions), UI helpers (interface/voice_helpers.py
additions), and Custom-tab wiring (interface/ui/tabs_generation.py) do not exist
yet; every test here fails until GREEN. Imports are function-level so a missing
symbol surfaces as a per-test failure, never a collection error.

Coverage:
  1. validate_prosody_preset_name — pure name validator
  2. get_user_prosody_presets — filtered view off the raw config key
  3. save_user_prosody_preset / delete_user_prosody_preset — disk-free CRUD
  4. Corrupt-config-load / save-time OSError failure paths (both writers)
  5. get_user_prosody_choices — delete-dropdown rendering
  6. _format_prosody_choice / prosody_choice_to_name — round-trip contracts
  7. _on_save_prosody_preset / _on_delete_prosody_preset handler contracts
  8. Custom-tab wiring structure

Conventions enforced here (Gate A round 1 findings):
  - Every handler test patches the writer AND the view seam, so no test can
    ever reach the real config.json, and every arm branch asserts
    assert_not_called() (arming must never write).
  - Every non-arm branch pins the FULL disarmed state dict, the button's base
    label, and (on non-success) bare gr.update() on all three dropdowns.
  - Source-text assertions use character windows, never physical lines
    (ruff format wraps long lines; a line-anchored test fails on correct code).
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


# The 8 factory prosody-preset names (values are arbitrary — the factory check
# is against presets.py's own DEFAULT_PROSODY_PRESETS keys, not against
# whatever text a caller's config happens to hold). A FACTORY, not a shared
# module-level dict: a shallow `{**CONFIG}` copy shares the inner
# prosody_presets dict, so an in-place-mutating implementation would poison
# every later test instead of failing cleanly at the offending one.
_FACTORY_PROSODY_SEED = {
    "excited": "seeded excited text",
    "calm": "seeded calm text",
    "whisper": "seeded whisper text",
    "authoritative": "seeded authoritative text",
    "slow": "seeded slow text",
    "fast": "seeded fast text",
    "dramatic": "seeded dramatic text",
    "conversational": "seeded conversational text",
}


def _seeded_factory_config():
    """A fresh config carrying all 8 factory prosody-preset keys."""
    return {"prosody_presets": dict(_FACTORY_PROSODY_SEED)}


SAVE_BTN_BASE = "Save as preset"
DELETE_BTN_BASE = "Delete preset"
EMPTY_INSTRUCT_COPY = (
    "Type a Style Instruction first — the preset saves exactly what's in that box."
)
DISARMED_STATE = {"armed": False, "ts": 0.0, "armed_name": None}


class TestValidateProsodyPresetName(unittest.TestCase):
    """validate_prosody_preset_name(name) -> str | None."""

    def _validate(self, name):
        from qwen3_tts.core.config import validate_prosody_preset_name

        return validate_prosody_preset_name(name)

    def test_empty_name_rejected(self):
        self.assertEqual(self._validate(""), "Enter a preset name.")

    def test_whitespace_only_name_rejected(self):
        self.assertIsNotNone(self._validate("   "))

    def test_name_over_max_len_rejected(self):
        from qwen3_tts.core.config import PROSODY_NAME_MAX_LEN

        self.assertEqual(
            self._validate("a" * (PROSODY_NAME_MAX_LEN + 1)),
            f"Preset name is too long ({PROSODY_NAME_MAX_LEN} characters max).",
        )

    def test_name_at_max_len_accepted(self):
        from qwen3_tts.core.config import PROSODY_NAME_MAX_LEN

        self.assertIsNone(self._validate("a" * PROSODY_NAME_MAX_LEN))

    def test_max_len_is_forty(self):
        # The spec fixes the limit at 40 (interpolated into user-facing copy),
        # so the constant itself is contract, not an implementation detail.
        from qwen3_tts.core.config import PROSODY_NAME_MAX_LEN

        self.assertEqual(PROSODY_NAME_MAX_LEN, 40)

    def test_valid_name_accepted(self):
        self.assertIsNone(self._validate("storyteller"))

    def test_factory_name_lowercase_rejected(self):
        self.assertEqual(
            self._validate("excited"),
            "'excited' is a built-in preset — pick a different name.",
        )

    def test_factory_name_case_insensitive_rejected(self):
        self.assertIsNotNone(self._validate("EXCITED"))

    def test_factory_name_whitespace_variant_rejected(self):
        # The validator strips before the factory check — a whitespace variant
        # must never slip past the collision rule (write-path trap 4).
        self.assertIsNotNone(self._validate("  excited  "))

    def test_none_choice_literal_rejected(self):
        self.assertIsNotNone(self._validate("(none)"))

    def test_choice_separator_in_name_rejected(self):
        # Defence-in-depth smoke: the charset rule already rejects spaces, so
        # any name containing " - " trips charset first — the separator check
        # cannot be exercised independently, by construction.
        self.assertIsNotNone(self._validate("slow - really"))

    def test_special_characters_rejected(self):
        self.assertEqual(
            self._validate("my@preset"),
            "Preset names can only contain letters, numbers, dashes, "
            "underscores, and dots.",
        )

    def test_dash_underscore_dot_accepted(self):
        self.assertIsNone(self._validate("story-teller_v1.0"))


class TestGetUserProsodyPresets(unittest.TestCase):
    """get_user_prosody_presets(config=None) -> dict[str, str]."""

    def _get(self, config):
        from qwen3_tts.core.config import get_user_prosody_presets

        return get_user_prosody_presets(config=config)

    def test_factory_names_excluded(self):
        self.assertEqual(self._get(_seeded_factory_config()), {})

    def test_factory_names_excluded_case_insensitively(self):
        # _FACTORY_PROSODY_NAMES lowercases its keys; the view must match
        # case-insensitively too, or a hand-edited "EXCITED" override would
        # masquerade as a user preset.
        self.assertEqual(self._get({"prosody_presets": {"EXCITED": "x"}}), {})

    def test_user_only_entry_included(self):
        config = {
            "prosody_presets": {
                **_seeded_factory_config()["prosody_presets"],
                "storyteller": "Warm and inviting",
            }
        }
        self.assertEqual(self._get(config), {"storyteller": "Warm and inviting"})

    def test_non_string_value_excluded(self):
        self.assertEqual(self._get({"prosody_presets": {"junk": 123}}), {})

    def test_empty_after_strip_excluded(self):
        self.assertEqual(self._get({"prosody_presets": {"blank": "   "}}), {})

    def test_returns_new_dict_not_alias(self):
        config = {"prosody_presets": {"storyteller": "text"}}
        result = self._get(config)
        result["mutated"] = "should not leak"
        self.assertNotIn("mutated", config["prosody_presets"])

    def test_missing_key_returns_empty_dict(self):
        self.assertEqual(self._get({}), {})

    @patch(
        "qwen3_tts.core.config.load_config",
        side_effect=ValueError("config.json is corrupt"),
    )
    def test_corrupt_config_swallowed_on_read_path(self, _mock_load):
        # Swallow-on-corrupt is for PURE READS ONLY: the view must return {}
        # (a dropdown render must never explode), while the write path (its
        # own tests below) must return (False, msg) and never save.
        self.assertEqual(self._get(None), {})


class TestSaveUserProsodyPreset(unittest.TestCase):
    """save_user_prosody_preset(name, instruct_text, config=None) -> tuple."""

    def _save(self, name, text, config):
        from qwen3_tts.core.config import save_user_prosody_preset

        return save_user_prosody_preset(name, text, config=config)

    @patch("qwen3_tts.core.config.save_config")
    def test_new_preset_saved_with_raw_base_preserved(self, mock_save):
        config = {
            "prosody_presets": {
                **_seeded_factory_config()["prosody_presets"],
                "junk": 123,
            }
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
    def test_new_preset_success_message_pinned(self, _mock_save):
        _ok, msg = self._save("storyteller", "Warm and inviting", {})
        self.assertEqual(
            msg,
            "Saved preset 'storyteller'. It now appears in the Style Preset dropdown.",
        )

    @patch("qwen3_tts.core.config.save_config")
    def test_long_text_success_message_notes_verbatim_save(self, _mock_save):
        _ok, msg = self._save("long", "x" * 250, {})
        self.assertIn("(250 characters, saved verbatim)", msg)

    @patch("qwen3_tts.core.config.save_config")
    def test_name_and_text_stored_stripped(self, mock_save):
        ok, _msg = self._save("  Foo  ", "  Speak warmly  ", {})
        self.assertTrue(ok)
        saved_presets = mock_save.call_args[0][0]["prosody_presets"]
        self.assertIn("Foo", saved_presets)
        self.assertEqual(saved_presets["Foo"], "Speak warmly")

    @patch("qwen3_tts.core.config.save_config")
    def test_whitespace_variant_name_targets_stripped_key(self, mock_save):
        # Trap 4 at the writer level: " storyteller " must land on the
        # existing "storyteller" key (one entry, no whitespace-keyed duplicate).
        config = {"prosody_presets": {"storyteller": "old text"}}
        ok, _msg = self._save(" storyteller ", "new text", config)
        self.assertTrue(ok)
        saved_presets = mock_save.call_args[0][0]["prosody_presets"]
        self.assertEqual(len(saved_presets), 1)
        self.assertIn("storyteller", saved_presets)
        self.assertEqual(saved_presets["storyteller"], "new text")

    @patch("qwen3_tts.core.config.save_config")
    def test_factory_name_collision_rejected(self, mock_save):
        ok, _msg = self._save("excited", "My version", _seeded_factory_config())
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
        self.assertEqual(msg, "Updated preset 'storyteller'.")
        saved_presets = mock_save.call_args[0][0]["prosody_presets"]
        self.assertEqual(saved_presets["storyteller"], "new text")

    @patch("qwen3_tts.core.config.save_config")
    def test_invalid_names_and_empty_text_rejected(self, mock_save):
        from qwen3_tts.core.config import PROSODY_NAME_MAX_LEN

        cases = [
            ("", "text"),
            ("   ", "text"),
            ("a" * (PROSODY_NAME_MAX_LEN + 1), "text"),
            ("my@preset", "text"),
            ("storyteller", ""),
            ("storyteller", "   "),
        ]
        for name, text in cases:
            with self.subTest(name=name, text=text):
                ok, _msg = self._save(name, text, {})
                self.assertFalse(ok)
        mock_save.assert_not_called()

    @patch("qwen3_tts.core.config.save_config")
    def test_empty_instruct_text_rejected_with_pinned_copy(self, mock_save):
        _ok, msg = self._save("storyteller", "", {})
        self.assertEqual(msg, EMPTY_INSTRUCT_COPY)
        mock_save.assert_not_called()

    @patch("qwen3_tts.core.config.save_config")
    def test_save_does_not_mutate_input_config(self, _mock_save):
        config = _seeded_factory_config()
        before = {k: dict(v) for k, v in config.items()}
        ok, _msg = self._save("storyteller", "text", config)
        self.assertTrue(ok)
        self.assertEqual(config, before)


class TestDeleteUserProsodyPreset(unittest.TestCase):
    """delete_user_prosody_preset(name, config=None) -> tuple[bool, str]."""

    def _delete(self, name, config):
        from qwen3_tts.core.config import delete_user_prosody_preset

        return delete_user_prosody_preset(name, config=config)

    @patch("qwen3_tts.core.config.save_config")
    def test_removes_only_named_raw_key(self, mock_save):
        config = {
            "prosody_presets": {
                **_seeded_factory_config()["prosody_presets"],
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
    def test_delete_success_message_pinned(self, _mock_save):
        _ok, msg = self._delete(
            "storyteller", {"prosody_presets": {"storyteller": "x"}}
        )
        self.assertEqual(msg, "Deleted preset 'storyteller'.")

    @patch("qwen3_tts.core.config.save_config")
    def test_non_string_junk_entry_deletable(self, _mock_save):
        config = {"prosody_presets": {"junk": 123}}
        ok, _msg = self._delete("junk", config)
        self.assertTrue(ok)

    @patch("qwen3_tts.core.config.save_config")
    def test_factory_name_delete_rejected(self, mock_save):
        ok, msg = self._delete("excited", _seeded_factory_config())
        self.assertFalse(ok)
        self.assertIn("built-in", msg)
        mock_save.assert_not_called()

    @patch("qwen3_tts.core.config.save_config")
    def test_factory_name_case_insensitive_delete_rejected(self, mock_save):
        ok, msg = self._delete("EXCITED", _seeded_factory_config())
        self.assertFalse(ok)
        self.assertIn("built-in", msg)
        mock_save.assert_not_called()

    @patch("qwen3_tts.core.config.save_config")
    def test_unknown_name_delete_rejected(self, mock_save):
        ok, msg = self._delete("nonexistent", {})
        self.assertFalse(ok)
        self.assertIn("no longer exists", msg)
        mock_save.assert_not_called()

    @patch("qwen3_tts.core.config.save_config")
    def test_delete_does_not_mutate_input_config(self, _mock_save):
        config = _seeded_factory_config()
        config["prosody_presets"]["storyteller"] = "text"
        before = {k: dict(v) for k, v in config.items()}
        ok, _msg = self._delete("storyteller", config)
        self.assertTrue(ok)
        self.assertEqual(config, before)


class TestProsodyPresetWriteFailurePaths(unittest.TestCase):
    """Corrupt-config-load and save-time OSError paths for both writers."""

    @patch(
        "qwen3_tts.core.config.load_config",
        side_effect=ValueError("config.json is corrupt"),
    )
    @patch("qwen3_tts.core.config.save_config")
    def test_save_corrupt_config_load_never_saves(self, mock_save, _mock_load):
        from qwen3_tts.core.config import save_user_prosody_preset

        ok, msg = save_user_prosody_preset(
            "storyteller", "text"
        )  # config=None -> loads
        self.assertFalse(ok)
        self.assertIn("corrupt", msg)
        mock_save.assert_not_called()

    @patch("qwen3_tts.core.config.load_config", side_effect=FileNotFoundError())
    @patch("qwen3_tts.core.config.save_config")
    def test_save_missing_config_file_never_saves(self, mock_save, _mock_load):
        from qwen3_tts.core.config import save_user_prosody_preset

        ok, msg = save_user_prosody_preset("storyteller", "text")
        self.assertFalse(ok)
        self.assertIn("tts config", msg)  # differentiated missing-file copy
        mock_save.assert_not_called()

    @patch(
        "qwen3_tts.core.config.save_config",
        side_effect=OSError(28, "No space left on device", "/Users/secret/config.json"),
    )
    def test_save_oserror_on_write_reports_strerror_only(self, _mock_save):
        from qwen3_tts.core.config import save_user_prosody_preset

        ok, msg = save_user_prosody_preset("storyteller", "text", config={})
        self.assertFalse(ok)
        # CWE-209: strerror surfaces; the path and the errno wrapper never do.
        self.assertIn("No space left on device", msg)
        self.assertNotIn("/Users/secret", msg)
        self.assertNotIn("Errno", msg)

    @patch(
        "qwen3_tts.core.config.load_config",
        side_effect=ValueError("config.json is corrupt"),
    )
    @patch("qwen3_tts.core.config.save_config")
    def test_delete_corrupt_config_load_never_saves(self, mock_save, _mock_load):
        from qwen3_tts.core.config import delete_user_prosody_preset

        ok, msg = delete_user_prosody_preset("storyteller")
        self.assertFalse(ok)
        self.assertIn("corrupt", msg)
        mock_save.assert_not_called()

    @patch("qwen3_tts.core.config.load_config", side_effect=FileNotFoundError())
    @patch("qwen3_tts.core.config.save_config")
    def test_delete_missing_config_file_never_saves(self, mock_save, _mock_load):
        from qwen3_tts.core.config import delete_user_prosody_preset

        ok, msg = delete_user_prosody_preset("storyteller")
        self.assertFalse(ok)
        self.assertIn("tts config", msg)  # differentiated missing-file copy
        mock_save.assert_not_called()

    @patch(
        "qwen3_tts.core.config.save_config",
        side_effect=OSError(28, "No space left on device", "/Users/secret/config.json"),
    )
    def test_delete_oserror_on_write_reports_strerror_only(self, _mock_save):
        from qwen3_tts.core.config import delete_user_prosody_preset

        ok, msg = delete_user_prosody_preset(
            "storyteller", config={"prosody_presets": {"storyteller": "text"}}
        )
        self.assertFalse(ok)
        self.assertIn("No space left on device", msg)
        self.assertNotIn("/Users/secret", msg)
        self.assertNotIn("Errno", msg)


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

    def test_sorted_alphabetically_ascii(self):
        # ASCII sort: uppercase before lowercase ("A" < "b").
        config = {"prosody_presets": {"b": "1", "A": "2"}}
        self.assertEqual(self._choices(config), ["(none)", "A - 2", "b - 1"])

    def test_factory_names_excluded_from_seeded_config(self):
        self.assertEqual(self._choices(_seeded_factory_config()), ["(none)"])

    def test_long_text_truncated_at_exactly_80_chars(self):
        long_text = "x" * 200
        choices = self._choices({"prosody_presets": {"long": long_text}})
        self.assertEqual(choices[1], "long - " + "x" * 80 + "…")

    def test_text_at_80_chars_not_truncated(self):
        text = "y" * 80
        choices = self._choices({"prosody_presets": {"edge": text}})
        self.assertEqual(choices[1], "edge - " + text)

    @patch(
        "qwen3_tts.core.config.load_config",
        side_effect=OSError(5, "Input/output error"),
    )
    def test_unreadable_config_swallowed_to_none_only(self, _mock_load):
        self.assertEqual(self._choices(None), ["(none)"])

    def test_none_choice_constant_value(self):
        from qwen3_tts.interface.voice_helpers import PROSODY_NONE_CHOICE

        self.assertEqual(PROSODY_NONE_CHOICE, "(none)")


class TestProsodyChoiceFormatRoundTrip(unittest.TestCase):
    """_format_prosody_choice / prosody_choice_to_name contracts."""

    def _fmt(self, name, text):
        from qwen3_tts.interface.voice_helpers import _format_prosody_choice

        return _format_prosody_choice(name, text)

    def _to_name(self, choice):
        from qwen3_tts.interface.voice_helpers import prosody_choice_to_name

        return prosody_choice_to_name(choice)

    def test_exact_format(self):
        self.assertEqual(self._fmt("name", "text"), "name - text")

    def test_round_trip(self):
        self.assertEqual(self._to_name(self._fmt("name", "text")), "name")

    def test_round_trip_with_truncated_text(self):
        # The "…" case is exactly where split(" - ")[0] must still recover the
        # name — every conditional-reset comparison depends on this.
        formatted = self._fmt("name", "x" * 200)
        self.assertTrue(formatted.endswith("…"))
        self.assertEqual(self._to_name(formatted), "name")

    def test_round_trip_with_separator_inside_text(self):
        formatted = self._fmt("name", "a - b")
        self.assertEqual(formatted, "name - a - b")
        self.assertEqual(self._to_name(formatted), "name")

    def test_none_choice_sentinel_is_identity(self):
        from qwen3_tts.interface.voice_helpers import PROSODY_NONE_CHOICE

        self.assertEqual(self._to_name(PROSODY_NONE_CHOICE), PROSODY_NONE_CHOICE)

    @patch("qwen3_tts.core.config.load_config")
    def test_factory_choices_render_observably_unchanged(self, mock_load):
        # get_prosody_choices now delegates to _format_prosody_choice; its
        # output for the 8 factory presets must not change (35-55 char texts
        # render whole — no ellipsis anywhere). Empty config: the merged view
        # renders DEFAULT_PROSODY_PRESETS with no overrides. A factory-keyed
        # seeded config would OVERRIDE the defaults (spec line 43 — the merged
        # apply view prefers config entries) and render the seeded texts.
        mock_load.return_value = {}
        from qwen3_tts.core.config.presets import DEFAULT_PROSODY_PRESETS
        from qwen3_tts.interface.voice_helpers import get_prosody_choices

        choices = get_prosody_choices()
        self.assertEqual(choices[0], "(none)")
        self.assertEqual(len(choices), 1 + len(DEFAULT_PROSODY_PRESETS))
        for name, text in DEFAULT_PROSODY_PRESETS.items():
            self.assertIn(f"{name} - {text}", choices)
        self.assertNotIn("…", "".join(choices))


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestProsodyPresetHandlerContracts(unittest.TestCase):
    """Direct calls to the module-level save/delete handlers in tabs_generation.py.

    Every test patches the writer AND the view seam (nothing may touch the
    real config.json), and every arm branch asserts the writer was NOT called.
    Output slots: [0] state, [1] button, [2] status, [3] announcer,
    [4] custom_prosody, [5] design_prosody, [6] delete dropdown.
    """

    def _save_handler(self):
        from qwen3_tts.interface.ui.tabs_generation import _on_save_prosody_preset

        return _on_save_prosody_preset

    def _delete_handler(self):
        from qwen3_tts.interface.ui.tabs_generation import _on_delete_prosody_preset

        return _on_delete_prosody_preset

    def _assert_disarmed(self, state):
        self.assertEqual(state, DISARMED_STATE)

    def _assert_bare_dropdowns(self, result):
        # Non-success rule: all three dropdowns get a bare no-change update.
        self.assertEqual(result[4], gr.update())
        self.assertEqual(result[5], gr.update())
        self.assertEqual(result[6], gr.update())

    def _assert_announced(self, result):
        # The announcer slot must carry generation._announce_status's sr-only
        # aria-live wrapper — not "", not the raw status text (a plain
        # gr.update(value=msg) bypasses the screen-reader contract).
        self.assertTrue(result[3])
        self.assertIn("aria-live", result[3])

    # --- Save handler ---

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)", "storyteller - text"],
    )
    @patch(
        "qwen3_tts.core.config.save_user_prosody_preset",
        return_value=(True, "Saved preset 'storyteller'."),
    )
    @patch("qwen3_tts.core.config.get_user_prosody_presets", return_value={})
    @patch("qwen3_tts.core.config.validate_prosody_preset_name", return_value=None)
    def test_save_new_preset_outputs_arity_seven(self, *_mocks):
        fn = self._save_handler()
        result = fn(
            dict(DISARMED_STATE), "storyteller", "Speak warmly", "(none)", "(none)"
        )
        self.assertEqual(len(result), 7)
        self._assert_disarmed(result[0])
        self.assertEqual(result[1].get("value"), SAVE_BTN_BASE)
        self.assertEqual(result[2], "Saved preset 'storyteller'.")
        self._assert_announced(result)  # announcer updated every branch
        # Success refresh: fresh merged choices, no value key ("(none)" inputs
        # never name the affected preset).
        self.assertEqual(result[4], gr.update(choices=["(none)", "storyteller - text"]))
        self.assertEqual(result[5], gr.update(choices=["(none)", "storyteller - text"]))
        # Save success refreshes the delete dropdown's choices only — it can
        # never reset that dropdown's value (its selection is not an input).
        self.assertEqual(result[6], gr.update(choices=["(none)"]))
        self.assertNotIn("value", result[6])

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)"],
    )
    @patch("qwen3_tts.core.config.save_user_prosody_preset")
    @patch("qwen3_tts.core.config.get_user_prosody_presets", return_value={})
    @patch(
        "qwen3_tts.core.config.validate_prosody_preset_name",
        return_value="'excited' is a built-in preset — pick a different name.",
    )
    def test_save_factory_name_hard_error_never_arms(
        self, mock_validate, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        fn = self._save_handler()
        result = fn(dict(DISARMED_STATE), "excited", "Speak warmly", "(none)", "(none)")
        self._assert_disarmed(result[0])
        self.assertEqual(result[1].get("value"), SAVE_BTN_BASE)
        self.assertEqual(result[2], mock_validate.return_value)
        self._assert_bare_dropdowns(result)
        self._assert_announced(result)
        mock_writer.assert_not_called()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch("qwen3_tts.core.config.save_user_prosody_preset")
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    @patch("qwen3_tts.core.config.validate_prosody_preset_name", return_value=None)
    def test_save_existing_name_arms_instead_of_saving(
        self, _mock_validate, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        fn = self._save_handler()
        result = fn(dict(DISARMED_STATE), "storyteller", "new text", "(none)", "(none)")
        self.assertTrue(result[0].get("armed"))
        self.assertEqual(result[0].get("armed_name"), "storyteller")
        self.assertGreater(result[0]["ts"], time.time() - 2)  # fresh, not stale
        self.assertEqual(result[1].get("value"), "Confirm Overwrite? (click again)")
        self.assertEqual(
            result[2],
            "A preset named 'storyteller' already exists. "
            "Click again within 5s to overwrite it.",
        )
        self._assert_bare_dropdowns(result)
        self._assert_announced(result)
        mock_writer.assert_not_called()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)"],
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
    def test_save_armed_fresh_confirm_saves_and_disarms(
        self, _mock_validate, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        fn = self._save_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "storyteller", "new text", "(none)", "(none)")
        self._assert_disarmed(result[0])
        self.assertEqual(result[1].get("value"), SAVE_BTN_BASE)
        self.assertEqual(result[2], "Updated preset 'storyteller'.")
        self.assertEqual(result[4], gr.update(choices=["(none)"]))
        self.assertEqual(result[5], gr.update(choices=["(none)"]))
        self.assertEqual(result[6], gr.update(choices=["(none)"]))
        mock_writer.assert_called_once()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.core.config.save_user_prosody_preset",
        return_value=(
            False,
            "Couldn't save prosody preset 'storyteller' — the config file "
            "couldn't be written (No space left on device). Check disk space "
            "and permissions, then try again.",
        ),
    )
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    @patch("qwen3_tts.core.config.validate_prosody_preset_name", return_value=None)
    def test_save_failure_renders_writer_message_and_disarms(
        self, _mock_validate, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        fn = self._save_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "storyteller", "new text", "(none)", "(none)")
        self._assert_disarmed(result[0])
        self.assertEqual(result[1].get("value"), SAVE_BTN_BASE)
        self.assertIn("No space left on device", result[2])
        self._assert_bare_dropdowns(result)  # non-success: no refresh
        self._assert_announced(result)
        mock_writer.assert_called_once()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)"],
    )
    @patch("qwen3_tts.core.config.save_user_prosody_preset")
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old", "other": "x"},
    )
    @patch("qwen3_tts.core.config.validate_prosody_preset_name", return_value=None)
    def test_save_arm_then_change_target_rearms_not_applies(
        self, _mock_validate, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        fn = self._save_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "other", "new text", "(none)", "(none)")
        self.assertTrue(result[0].get("armed"))
        self.assertEqual(result[0].get("armed_name"), "other")
        self.assertGreater(result[0]["ts"], time.time() - 2)
        self.assertIn("Confirm", result[1].get("value", ""))
        self.assertEqual(
            result[2], "Changed to 'other' — click again within 5s to overwrite it."
        )
        self._assert_bare_dropdowns(result)
        mock_writer.assert_not_called()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch("qwen3_tts.core.config.save_user_prosody_preset")
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    @patch("qwen3_tts.core.config.validate_prosody_preset_name", return_value=None)
    def test_save_expired_arm_rearms_not_saves(
        self, _mock_validate, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        fn = self._save_handler()
        state = {"armed": True, "ts": time.time() - 6, "armed_name": "storyteller"}
        result = fn(state, "storyteller", "new text", "(none)", "(none)")
        self.assertTrue(result[0].get("armed"))
        self.assertEqual(result[0].get("armed_name"), "storyteller")
        self.assertGreater(result[0]["ts"], time.time() - 2)
        self.assertEqual(
            result[2],
            "Your confirmation timed out. Click again within 5s to "
            "overwrite 'storyteller'.",
        )
        self.assertIn("Confirm", result[1].get("value", ""))
        self._assert_bare_dropdowns(result)
        mock_writer.assert_not_called()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch("qwen3_tts.core.config.save_user_prosody_preset")
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    @patch("qwen3_tts.core.config.validate_prosody_preset_name", return_value=None)
    def test_save_whitespace_variant_name_arms_for_stripped_target(
        self, _mock_validate, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        # Trap 4 at the handler: the comparison uses the stripped form, so a
        # whitespace variant matches the existing key and arms — it can never
        # bypass the overwrite-confirm via the save-now branch.
        fn = self._save_handler()
        result = fn(
            dict(DISARMED_STATE), "  storyteller  ", "new text", "(none)", "(none)"
        )
        self.assertTrue(result[0].get("armed"))
        self.assertEqual(result[0].get("armed_name"), "storyteller")
        mock_writer.assert_not_called()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.core.config.save_user_prosody_preset",
        return_value=(False, EMPTY_INSTRUCT_COPY),
    )
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    @patch("qwen3_tts.core.config.validate_prosody_preset_name", return_value=None)
    def test_save_empty_instruct_text_hard_error_never_arms(
        self, _mock_validate, _mock_view, _mock_writer, _mock_merged, _mock_user
    ):
        # Empty text must hard-error even when the name already exists (the
        # tempting wrong path is arming for an overwrite with nothing to save).
        fn = self._save_handler()
        result = fn(dict(DISARMED_STATE), "storyteller", "   ", "(none)", "(none)")
        self._assert_disarmed(result[0])
        self.assertEqual(result[1].get("value"), SAVE_BTN_BASE)
        self.assertEqual(result[2], EMPTY_INSTRUCT_COPY)
        self._assert_bare_dropdowns(result)

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)", "storyteller - text"],
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
    def test_save_overwrite_resets_dropdowns_showing_saved_preset(
        self, _mock_validate, _mock_view, _mock_writer, _mock_merged, _mock_user
    ):
        # Conditional-reset positive case (save side): a dropdown still
        # showing the overwritten preset's stale label must be reset to
        # (none) — a stale "name - old text" label re-fires .change and
        # double-appends the instruct text.
        fn = self._save_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(
            state, "storyteller", "new text", "storyteller - old", "storyteller - old"
        )
        self._assert_disarmed(result[0])
        self.assertEqual(
            result[4],
            gr.update(choices=["(none)", "storyteller - text"], value="(none)"),
        )
        self.assertEqual(
            result[5],
            gr.update(choices=["(none)", "storyteller - text"], value="(none)"),
        )
        # The delete dropdown is never value-reset by SAVE (not an input).
        self.assertEqual(result[6], gr.update(choices=["(none)"]))

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)", "storyteller - text"],
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
    def test_save_asymmetric_inputs_reset_the_matching_slot_only(
        self, _mock_validate, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        """custom_prosody/design_prosody swap detector (save side): only the
        dropdown whose OWN input names the overwritten preset gets
        value=NONE — a simultaneous signature or wiring swap between the two
        slots fails here (stale-label/double-append trap, spec line 101)."""
        fn = self._save_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "storyteller", "new text", "(none)", "storyteller - old")
        self.assertEqual(result[4], gr.update(choices=["(none)", "storyteller - text"]))
        self.assertNotIn("value", result[4])
        self.assertEqual(
            result[5],
            gr.update(choices=["(none)", "storyteller - text"], value="(none)"),
        )
        # The delete dropdown is never value-reset by SAVE (not an input).
        self.assertEqual(result[6], gr.update(choices=["(none)"]))
        mock_writer.assert_called_once()

    # --- Delete handler ---

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch("qwen3_tts.core.config.delete_user_prosody_preset")
    @patch("qwen3_tts.core.config.get_user_prosody_presets", return_value={})
    def test_delete_nothing_selected_disarmed(
        self, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        fn = self._delete_handler()
        result = fn(dict(DISARMED_STATE), "(none)", "(none)", "(none)")
        self.assertEqual(len(result), 7)
        self._assert_disarmed(result[0])
        self.assertEqual(result[1].get("value"), DELETE_BTN_BASE)
        self.assertEqual(result[2], "Select one of your presets to delete.")
        self._assert_bare_dropdowns(result)
        self._assert_announced(result)
        mock_writer.assert_not_called()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.core.config.delete_user_prosody_preset",
        return_value=(
            False,
            "Preset 'ghost' no longer exists — it may have been removed outside the UI.",
        ),
    )
    @patch("qwen3_tts.core.config.get_user_prosody_presets", return_value={})
    def test_delete_membership_miss_first_click_renders_data_layer_message(
        self, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        # Branch ② BEFORE branch ③: a membership miss on an UNARMED first
        # click must render the data-layer classification message (equality
        # pins "never a UI-local literal") and must NOT arm.
        fn = self._delete_handler()
        result = fn(dict(DISARMED_STATE), "ghost", "(none)", "(none)")
        self._assert_disarmed(result[0])
        self.assertEqual(
            result[2],
            "Preset 'ghost' no longer exists — it may have been removed outside the UI.",
        )
        self._assert_bare_dropdowns(result)
        mock_writer.assert_called_once()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.core.config.delete_user_prosody_preset",
        return_value=(
            False,
            "Preset 'ghost' no longer exists — it may have been removed outside the UI.",
        ),
    )
    @patch("qwen3_tts.core.config.get_user_prosody_presets", return_value={})
    def test_delete_unknown_preset_renders_writer_message(
        self, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        fn = self._delete_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "ghost"}
        result = fn(state, "ghost", "(none)", "(none)")
        self.assertIn("no longer exists", result[2])
        self._assert_disarmed(result[0])
        self.assertEqual(result[1].get("value"), DELETE_BTN_BASE)
        self._assert_bare_dropdowns(result)
        mock_writer.assert_called_once()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch("qwen3_tts.core.config.delete_user_prosody_preset")
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    def test_delete_arms_on_first_click(
        self, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        fn = self._delete_handler()
        result = fn(dict(DISARMED_STATE), "storyteller", "(none)", "(none)")
        self.assertTrue(result[0].get("armed"))
        self.assertEqual(result[0].get("armed_name"), "storyteller")
        self.assertGreater(result[0]["ts"], time.time() - 2)
        self.assertEqual(result[1].get("value"), "Confirm Delete? (click again)")
        self.assertEqual(
            result[2], "Delete preset 'storyteller'? Click again within 5s to confirm."
        )
        self._assert_bare_dropdowns(result)
        self._assert_announced(result)
        mock_writer.assert_not_called()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)", "other - x"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)", "other - x"],
    )
    @patch(
        "qwen3_tts.core.config.delete_user_prosody_preset",
        return_value=(True, "Deleted preset 'storyteller'."),
    )
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    def test_delete_armed_fresh_confirm_deletes_and_disarms(
        self, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        fn = self._delete_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "storyteller", "(none)", "(none)")
        self._assert_disarmed(result[0])
        self.assertEqual(result[1].get("value"), DELETE_BTN_BASE)
        self.assertEqual(result[2], "Deleted preset 'storyteller'.")
        # Delete dropdown: its own selection IS the affected preset -> reset.
        self.assertEqual(
            result[6], gr.update(choices=["(none)", "other - x"], value="(none)")
        )
        # The two apply dropdowns showed "(none)" -> choices only, no value.
        self.assertEqual(result[4], gr.update(choices=["(none)", "other - x"]))
        self.assertEqual(result[5], gr.update(choices=["(none)", "other - x"]))
        self._assert_announced(result)
        mock_writer.assert_called_once()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)", "storyteller - old"],
    )
    @patch("qwen3_tts.core.config.delete_user_prosody_preset")
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    def test_delete_expired_arm_rearms_not_deletes(
        self, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        fn = self._delete_handler()
        state = {"armed": True, "ts": time.time() - 6, "armed_name": "storyteller"}
        result = fn(state, "storyteller", "(none)", "(none)")
        self.assertTrue(result[0].get("armed"))
        self.assertEqual(result[0].get("armed_name"), "storyteller")
        self.assertGreater(result[0]["ts"], time.time() - 2)
        self.assertEqual(
            result[2],
            "Your confirmation timed out. Click again within 5s to delete 'storyteller'.",
        )
        self.assertIn("Confirm", result[1].get("value", ""))
        self._assert_bare_dropdowns(result)
        mock_writer.assert_not_called()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)", "other - x"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)", "other - x"],
    )
    @patch("qwen3_tts.core.config.delete_user_prosody_preset")
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old", "other": "x"},
    )
    def test_delete_arm_then_change_target_rearms(
        self, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        fn = self._delete_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "other", "(none)", "(none)")
        self.assertTrue(result[0].get("armed"))
        self.assertEqual(result[0].get("armed_name"), "other")
        self.assertEqual(
            result[2], "Now deleting 'other' — click again within 5s to confirm."
        )
        self._assert_bare_dropdowns(result)
        mock_writer.assert_not_called()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)", "other - x"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)", "other - x"],
    )
    @patch(
        "qwen3_tts.core.config.delete_user_prosody_preset",
        return_value=(True, "Deleted preset 'storyteller'."),
    )
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    def test_delete_mismatched_current_selection_no_value_reset(
        self, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        """Deleting 'storyteller' while an unrelated dropdown shows 'other - x'
        refreshes that dropdown's choices but must not reset its value
        (conditional-reset rule — reset re-fires .change on the survivor)."""
        fn = self._delete_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "storyteller", "other - x", "(none)")
        self.assertEqual(
            result[4], gr.update(choices=["(none)", "other - x"])
        )  # custom_prosody: choices only
        self.assertNotIn("value", result[4])
        self.assertEqual(result[5], gr.update(choices=["(none)", "other - x"]))
        self.assertNotIn("value", result[5])
        # The delete dropdown's own selection IS the affected preset -> reset.
        self.assertEqual(
            result[6], gr.update(choices=["(none)", "other - x"], value="(none)")
        )
        mock_writer.assert_called_once()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.core.config.delete_user_prosody_preset",
        return_value=(True, "Deleted preset 'storyteller'."),
    )
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    def test_delete_matching_current_selection_resets_both_dropdowns(
        self, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        """Conditional-reset positive case (delete side): dropdowns still
        showing the deleted preset's stale label are reset to (none)."""
        fn = self._delete_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "storyteller", "storyteller - old", "storyteller - old")
        self.assertEqual(result[4], gr.update(choices=["(none)"], value="(none)"))
        self.assertEqual(result[5], gr.update(choices=["(none)"], value="(none)"))
        self.assertEqual(result[6], gr.update(choices=["(none)"], value="(none)"))
        mock_writer.assert_called_once()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.core.config.delete_user_prosody_preset",
        return_value=(True, "Deleted preset 'storyteller'."),
    )
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    def test_delete_asymmetric_inputs_reset_the_matching_slot_only(
        self, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        """custom_prosody/design_prosody swap detector: only the dropdown whose
        OWN input names the deleted preset gets value=NONE — a simultaneous
        signature or wiring swap between the two slots fails here."""
        fn = self._delete_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "storyteller", "(none)", "storyteller - old")
        self.assertEqual(result[4], gr.update(choices=["(none)"]))
        self.assertNotIn("value", result[4])
        self.assertEqual(result[5], gr.update(choices=["(none)"], value="(none)"))
        self.assertEqual(result[6], gr.update(choices=["(none)"], value="(none)"))
        mock_writer.assert_called_once()

    @patch(
        "qwen3_tts.interface.voice_helpers.get_user_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.interface.voice_helpers.get_prosody_choices",
        return_value=["(none)"],
    )
    @patch(
        "qwen3_tts.core.config.delete_user_prosody_preset",
        return_value=(True, "Deleted preset 'storyteller'."),
    )
    @patch(
        "qwen3_tts.core.config.get_user_prosody_presets",
        return_value={"storyteller": "old"},
    )
    def test_delete_none_design_prosody_treated_as_none(
        self, _mock_view, mock_writer, _mock_merged, _mock_user
    ):
        # design_prosody may arrive None (lazy tabpanel never visited): the
        # None-guard maps it to (none) — no crash, no value reset.
        fn = self._delete_handler()
        state = {"armed": True, "ts": time.time(), "armed_name": "storyteller"}
        result = fn(state, "storyteller", "(none)", None)
        self._assert_disarmed(result[0])
        self.assertEqual(result[5], gr.update(choices=["(none)"]))
        self.assertNotIn("value", result[5])

    def test_none_choice_constants_agree_across_modules(self):
        from qwen3_tts.interface import voice_helpers
        from qwen3_tts.interface.ui import tabs_generation

        self.assertEqual(voice_helpers.PROSODY_NONE_CHOICE, tabs_generation.NONE_CHOICE)


@unittest.skipUnless(HAS_GRADIO, "requires gradio")
class TestProsodyPresetUiWiring(unittest.TestCase):
    """Structural assertions on tabs_generation.py's / _facade.py's wiring.

    All source-text assertions use character windows — never physical lines —
    because ruff format wraps long lines and a line-anchored test would fail
    against correctly formatted code. The "no gr.Tab select listener" rule is
    NOT re-checked here: tests/test_ui_tab_select_wiring.py builds the real
    Blocks config and guards it structurally (stronger than source text).
    """

    def _read_source(self, rel_path):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, rel_path)) as f:
            return f.read()

    def _click_block(self, src, marker):
        start = src.index(marker)
        return src[start : start + 1600]

    def _list_window(self, block, list_name):
        start = block.index(f"{list_name}=[")
        return block[start : block.index("]", start)]

    def test_save_click_block_pins_fn_inputs_and_outputs(self):
        src = self._read_source("qwen3_tts/interface/ui/tabs_generation.py")
        block = self._click_block(src, "prosody_preset_save_btn.click(")
        # fn= must be inside the click block, not merely present in the file.
        self.assertIn("fn=_on_save_prosody_preset", block)
        inputs = self._list_window(block, "inputs")
        for name in (
            "prosody_preset_save_state",
            "prosody_preset_name",
            "custom_instruct",
            "custom_prosody",
            "design_prosody",
        ):
            self.assertIn(name, inputs)
        outputs = self._list_window(block, "outputs")
        # The 6 named outputs (the announcer slot sits between status and
        # custom_prosody) must appear in outputs order.
        ordered = [
            "prosody_preset_save_state",
            "prosody_preset_save_btn",
            "prosody_preset_save_status",
            "custom_prosody",
            "design_prosody",
            "prosody_preset_delete_dropdown",
        ]
        positions = [outputs.index(name) for name in ordered]
        self.assertEqual(
            positions, sorted(positions), f"outputs order wrong: {outputs!r}"
        )

    def test_delete_click_block_pins_fn_inputs_and_outputs(self):
        src = self._read_source("qwen3_tts/interface/ui/tabs_generation.py")
        block = self._click_block(src, "prosody_preset_delete_btn.click(")
        self.assertIn("fn=_on_delete_prosody_preset", block)
        inputs = self._list_window(block, "inputs")
        for name in (
            "prosody_preset_delete_state",
            "prosody_preset_delete_dropdown",
            "custom_prosody",
            "design_prosody",
        ):
            self.assertIn(name, inputs)
        outputs = self._list_window(block, "outputs")
        ordered = [
            "prosody_preset_delete_state",
            "prosody_preset_delete_btn",
            "prosody_preset_delete_status",
            "custom_prosody",
            "design_prosody",
            "prosody_preset_delete_dropdown",
        ]
        positions = [outputs.index(name) for name in ordered]
        self.assertEqual(
            positions, sorted(positions), f"outputs order wrong: {outputs!r}"
        )

    def test_amended_custom_prosody_info_string_present(self):
        src = self._read_source("qwen3_tts/interface/ui/tabs_generation.py")
        self.assertIn("then save it under 'My prosody presets'", src)

    def test_delete_button_uses_stop_variant(self):
        src = self._read_source("qwen3_tts/interface/ui/tabs_generation.py")
        idx = src.index("prosody_preset_delete_btn")  # the component definition
        snippet = src[idx : idx + 400]  # window, not line — format-agnostic
        self.assertIn('variant="stop"', snippet)

    def test_design_tab_body_references_design_prosody(self):
        src = self._read_source("qwen3_tts/interface/ui/tabs_generation.py")
        start = src.index("def _build_design_tab(")
        end = src.index("\ndef ", start + 1)  # function span, not one line
        self.assertIn("design_prosody", src[start:end])

    def test_facade_unpacks_design_prosody_from_design_tab(self):
        src = self._read_source("qwen3_tts/interface/ui/_facade.py")
        call_idx = src.index("_build_design_tab(")
        # Backwards character window: ruff format wraps the assignment across
        # lines once it exceeds 88 chars, so the name may sit on any of them.
        window = src[max(0, call_idx - 300) : call_idx + 200]
        self.assertIn("design_prosody", window)

    def test_facade_passes_design_prosody_into_custom_tab_call(self):
        src = self._read_source("qwen3_tts/interface/ui/_facade.py")
        call_idx = src.index("_build_custom_tab(")
        call_block = src[call_idx : call_idx + 150]
        self.assertIn("design_prosody", call_block)


if __name__ == "__main__":
    unittest.main()
