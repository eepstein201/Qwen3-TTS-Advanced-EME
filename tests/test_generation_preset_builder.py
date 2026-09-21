"""Tests for the user-defined generation-preset builder (Track T1b).

Spec: 2026-09-21-clone-generation-preset-builder (v1.1, D1a — full live refresh
on all three tabs). Mirrors the structure and trap coverage of
tests/test_prosody_preset_builder.py (Track T1).

RED round 2 (strengthened after Gate A round 1 = 2xFAIL): the post-delete
choices mocks now match what the handlers observe AFTER the write (and include
a factory name so a user-choices-only implementation cannot false-pass), the
facade wiring is pinned click-block style (fn=, ordered inputs incl. the four
sliders, ordered 8-slot outputs), the shared helper has behavioral coverage,
arm/expiry/mismatch copy is pinned exactly (spec §7), and both re-arm variants
are covered on both handlers.

Round 3 (round 2 = 2xPASS carrying MEDIUM findings): full input-order pins on
both click blocks, delete-writer corrupt-load and confirm-time writer-failure
branches, confirm/miss/factory slot-1 button pins, strerror presence in the
OSError tests, and the reader import hoisted out of the subTest loop.

Traps pinned here (spec §8): raw-base preservation, factory silent override
(case-insensitive + whitespace variants), corrupt-config swallow on reads and
untouched-file on writes, immutability, arm-branches-never-write, param key
whitelist (bool rejection included), expiry re-arm, delete classification
before arming, conditional value reset.
"""

import copy
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    import gradio as gr  # noqa: F401

    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

# Factory entries as get_generation_presets ships them; "junk-entry" stands in
# for a hand-edited non-factory key that must survive raw writes untouched (and
# per spec §4 — reader = raw minus factory, no value filter — it legitimately
# APPEARS in the user view, where the delete dropdown lets the user remove it).
SEEDED_FACTORY_CONFIG = {
    "language": "auto",
    "advanced": {"backend": "mlx"},
    "presets": {
        "stable": {
            "temperature": 0.5,
            "top_k": 30,
            "top_p": 0.90,
            "repetition_penalty": 1.10,
        },
        "natural": {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
        },
        "junk-entry": {"temperature": "not-a-number"},
    },
}

VALID_PARAMS = {
    "temperature": 0.6,
    "top_k": 40,
    "top_p": 0.92,
    "repetition_penalty": 1.08,
}

DISARMED = {"armed": False, "ts": 0.0, "armed_name": None}

# Spec §7 pinned copy (exact strings — handlers render these verbatim).
COPY_BAD_NAME = (
    "Preset names are 1-40 characters (letters, numbers, space, dash, underscore, dot)."
)
COPY_FACTORY = "Factory presets can't be overwritten — pick a different name."
COPY_SAVE_ARM = "A preset named '{name}' exists — click Save again to overwrite."
COPY_DELETE_ARM = "Click Delete again to delete '{name}'."
COPY_MISS = "Preset '{name}' no longer exists."


def _deep_seed():
    return copy.deepcopy(SEEDED_FACTORY_CONFIG)


# ---------------------------------------------------------------------------
# Name validation
# ---------------------------------------------------------------------------


class TestValidateGenerationPresetName(unittest.TestCase):
    def _validator(self):
        from qwen3_tts.core.config import validate_generation_preset_name

        return validate_generation_preset_name

    def test_empty_name_rejected(self):
        self.assertTrue(self._validator()(""))

    def test_whitespace_name_rejected(self):
        self.assertTrue(self._validator()("   "))

    def test_name_over_40_chars_rejected(self):
        self.assertTrue(self._validator()("a" * 41))

    def test_exactly_40_chars_accepted(self):
        self.assertIsNone(self._validator()("a" * 40))

    def test_invalid_charset_rejected(self):
        msg = self._validator()("my preset!")
        self.assertTrue(msg)
        self.assertIn("1-40 characters", msg)

    def test_double_dot_rejected(self):
        self.assertTrue(self._validator()("a..b"))

    def test_factory_name_rejected_case_insensitive(self):
        msg = self._validator()("STABLE")
        self.assertTrue(msg)
        self.assertIn("Factory presets", msg)

    def test_factory_name_variant_with_whitespace_rejected(self):
        # A whitespace + case variant must not slip past the factory check —
        # the merge at get_generation_presets would silently override factory.
        msg = self._validator()("  Stable ")
        self.assertTrue(msg)
        self.assertIn("Factory presets", msg)

    def test_all_factory_names_rejected(self):
        for name in (
            "stable",
            "natural",
            "expressive",
            "audiobook",
            "conversational",
            "broadcast",
            "dramatic",
            "whisper",
        ):
            self.assertTrue(self._validator()(name), name)

    def test_plain_names_accepted(self):
        self.assertIsNone(self._validator()("My Preset 1"))
        self.assertIsNone(self._validator()("a-b_c.d 2"))


# ---------------------------------------------------------------------------
# Params validation (key whitelist — downstream apply is a blind dict update)
# ---------------------------------------------------------------------------


class TestValidateGenerationPresetParams(unittest.TestCase):
    def _validator(self):
        from qwen3_tts.core.config import validate_generation_preset_params

        return validate_generation_preset_params

    def test_valid_params_accepted(self):
        self.assertIsNone(self._validator()(dict(VALID_PARAMS)))

    def test_missing_key_rejected(self):
        for key in VALID_PARAMS:
            params = {k: v for k, v in VALID_PARAMS.items() if k != key}
            self.assertTrue(self._validator()(params), key)

    def test_extra_key_rejected(self):
        params = dict(VALID_PARAMS)
        params["speed"] = 1.5
        self.assertTrue(self._validator()(params))

    def test_junk_key_rejected(self):
        self.assertTrue(self._validator()({"bogus": 1.0}))

    def test_temperature_bounds(self):
        self.assertTrue(self._validator()({**VALID_PARAMS, "temperature": -0.01}))
        self.assertTrue(self._validator()({**VALID_PARAMS, "temperature": 2.01}))
        self.assertIsNone(self._validator()({**VALID_PARAMS, "temperature": 0.0}))
        self.assertIsNone(self._validator()({**VALID_PARAMS, "temperature": 2.0}))

    def test_top_k_bounds(self):
        self.assertTrue(self._validator()({**VALID_PARAMS, "top_k": 0}))
        self.assertTrue(self._validator()({**VALID_PARAMS, "top_k": 1001}))
        self.assertIsNone(self._validator()({**VALID_PARAMS, "top_k": 1}))
        self.assertIsNone(self._validator()({**VALID_PARAMS, "top_k": 1000}))

    def test_top_p_bounds(self):
        self.assertTrue(self._validator()({**VALID_PARAMS, "top_p": -0.01}))
        self.assertTrue(self._validator()({**VALID_PARAMS, "top_p": 1.01}))
        self.assertIsNone(self._validator()({**VALID_PARAMS, "top_p": 0.0}))
        self.assertIsNone(self._validator()({**VALID_PARAMS, "top_p": 1.0}))

    def test_repetition_penalty_bounds(self):
        self.assertTrue(self._validator()({**VALID_PARAMS, "repetition_penalty": 0.49}))
        self.assertTrue(self._validator()({**VALID_PARAMS, "repetition_penalty": 2.01}))
        self.assertIsNone(
            self._validator()({**VALID_PARAMS, "repetition_penalty": 0.5})
        )
        self.assertIsNone(
            self._validator()({**VALID_PARAMS, "repetition_penalty": 2.0})
        )

    def test_bool_value_rejected(self):
        self.assertTrue(self._validator()({**VALID_PARAMS, "temperature": True}))

    def test_string_value_rejected(self):
        self.assertTrue(self._validator()({**VALID_PARAMS, "temperature": "0.7"}))

    def test_non_dict_rejected(self):
        self.assertTrue(self._validator()(None))
        self.assertTrue(self._validator()([VALID_PARAMS]))


# ---------------------------------------------------------------------------
# Reader: raw-minus-factory, swallow, no aliasing
# ---------------------------------------------------------------------------


class TestGetUserGenerationPresets(unittest.TestCase):
    def _reader(self):
        from qwen3_tts.core.config import get_user_generation_presets

        return get_user_generation_presets

    def test_returns_non_factory_entries_only(self):
        seed = _deep_seed()
        seed["presets"]["mine"] = dict(VALID_PARAMS)
        result = self._reader()(config=seed)
        self.assertEqual(
            result,
            {"junk-entry": seed["presets"]["junk-entry"], "mine": dict(VALID_PARAMS)},
        )

    def test_factory_exclusion_is_case_insensitive(self):
        seed = _deep_seed()
        seed["presets"]["STABLE"] = dict(VALID_PARAMS)
        result = self._reader()(config=seed)
        self.assertNotIn("STABLE", result)
        self.assertNotIn("stable", result)

    def test_missing_presets_key_returns_empty(self):
        self.assertEqual(self._reader()(config={}), {})

    def test_corrupt_config_swallows_to_empty(self):
        # Reader resolved OUTSIDE the subTest loop: at RED a missing symbol
        # must fail the parent (pytest renders subTest children SUBFAILED
        # under a PASSED parent, masking the per-test RED signal).
        reader = self._reader()
        for exc in (ValueError("corrupt"), OSError("unreadable")):
            with self.subTest(exc=type(exc).__name__):
                with mock.patch("qwen3_tts.core.config.load_config", side_effect=exc):
                    self.assertEqual(reader(config=None), {})

    def test_returns_new_dict_not_alias(self):
        seed = _deep_seed()
        seed["presets"]["mine"] = dict(VALID_PARAMS)
        result = self._reader()(config=seed)
        self.assertIsNot(result, seed["presets"])
        result["extra"] = {}
        self.assertNotIn("extra", seed["presets"])
        result["mine"]["temperature"] = 99.0
        self.assertEqual(seed["presets"]["mine"]["temperature"], 0.6)


# ---------------------------------------------------------------------------
# Delete-dropdown helper: behavioral coverage (spec §5)
# ---------------------------------------------------------------------------


class TestGetUserGenerationPresetChoices(unittest.TestCase):
    """shared.get_user_generation_preset_choices() -> list[str].

    Spec §5: ``["(none)"] + sorted(get_user_generation_presets())`` — the
    helper calls the reader with no config so the reader's own corrupt-swallow
    applies; bare names (no formatter); "(none)" first.
    """

    def _choices(self):
        from qwen3_tts.interface.ui.shared import get_user_generation_preset_choices

        return get_user_generation_preset_choices()

    def test_empty_config_renders_none_only(self):
        with mock.patch("qwen3_tts.core.config.load_config", return_value={}):
            self.assertEqual(self._choices(), ["(none)"])

    def test_reader_view_sorted_with_factory_excluded(self):
        # Runs the REAL reader over a seeded config: factory "stable" drops out
        # (case-insensitive), user keys sort, junk-entry legitimately appears
        # (spec §4 has no value filter — it stays deletable via this dropdown).
        seed = _deep_seed()
        seed["presets"]["zeta"] = dict(VALID_PARAMS)
        seed["presets"]["alpha"] = dict(VALID_PARAMS)
        with mock.patch("qwen3_tts.core.config.load_config", return_value=seed):
            self.assertEqual(self._choices(), ["(none)", "alpha", "junk-entry", "zeta"])

    def test_reader_corrupt_config_swallows_to_none_only(self):
        with mock.patch(
            "qwen3_tts.core.config.load_config", side_effect=ValueError("corrupt")
        ):
            self.assertEqual(self._choices(), ["(none)"])

    def test_ascii_sort_order(self):
        # sorted() is ASCII: uppercase before lowercase.
        with mock.patch(
            "qwen3_tts.interface.ui.shared.get_user_generation_presets",
            return_value={"b": dict(VALID_PARAMS), "A": dict(VALID_PARAMS)},
        ):
            self.assertEqual(self._choices(), ["(none)", "A", "b"])


# ---------------------------------------------------------------------------
# Writers: raw base preserved, corrupt never saves, sanitized errors
# ---------------------------------------------------------------------------


class TestSaveUserGenerationPreset(unittest.TestCase):
    def _writer(self):
        from qwen3_tts.core.config import save_user_generation_preset

        return save_user_generation_preset

    def test_save_new_preserves_raw_base(self):
        seed = _deep_seed()
        with (
            mock.patch("qwen3_tts.core.config.load_config", return_value=seed),
            mock.patch("qwen3_tts.core.config.save_config") as save_cfg,
        ):
            ok, msg = self._writer()("mine", dict(VALID_PARAMS))
        self.assertTrue(ok, msg)
        self.assertIn("Saved preset 'mine'", msg)
        save_cfg.assert_called_once()
        written = save_cfg.call_args[0][0]
        # Factory + junk + unrelated top-level keys all survive (raw base) —
        # junk survives WITH its value, not just its key.
        self.assertEqual(
            written["presets"]["stable"], SEEDED_FACTORY_CONFIG["presets"]["stable"]
        )
        self.assertEqual(
            written["presets"]["junk-entry"],
            {"temperature": "not-a-number"},
        )
        self.assertEqual(written["presets"]["mine"], VALID_PARAMS)
        self.assertEqual(written["language"], "auto")
        self.assertEqual(written["advanced"], {"backend": "mlx"})

    def test_save_does_not_mutate_loaded_config(self):
        seed = _deep_seed()
        snapshot = _deep_seed()
        with (
            mock.patch("qwen3_tts.core.config.load_config", return_value=seed),
            mock.patch("qwen3_tts.core.config.save_config"),
        ):
            self._writer()("mine", dict(VALID_PARAMS))
        self.assertEqual(seed, snapshot)

    def test_invalid_params_never_write(self):
        seed = _deep_seed()
        with (
            mock.patch("qwen3_tts.core.config.load_config", return_value=seed),
            mock.patch("qwen3_tts.core.config.save_config") as save_cfg,
        ):
            ok, msg = self._writer()("mine", {"temperature": 9.9})
        self.assertFalse(ok)
        self.assertTrue(msg)
        save_cfg.assert_not_called()

    def test_corrupt_config_returns_false_and_never_saves(self):
        with (
            mock.patch(
                "qwen3_tts.core.config.load_config",
                side_effect=ValueError("corrupt"),
            ),
            mock.patch("qwen3_tts.core.config.save_config") as save_cfg,
        ):
            ok, msg = self._writer()("mine", dict(VALID_PARAMS))
        self.assertFalse(ok)
        self.assertTrue(msg)
        save_cfg.assert_not_called()

    def test_write_error_message_has_no_paths(self):
        err = OSError(28, "No space left on device", "/Users/secret/config.json")
        with (
            mock.patch("qwen3_tts.core.config.load_config", return_value=_deep_seed()),
            mock.patch("qwen3_tts.core.config.save_config", side_effect=err),
        ):
            ok, msg = self._writer()("mine", dict(VALID_PARAMS))
        self.assertFalse(ok)
        self.assertIn("No space left on device", msg)
        self.assertNotIn("/Users/secret", msg)
        self.assertNotIn("Errno", msg)


class TestDeleteUserGenerationPreset(unittest.TestCase):
    def _writer(self):
        from qwen3_tts.core.config import delete_user_generation_preset

        return delete_user_generation_preset

    def _seeded(self):
        seed = _deep_seed()
        seed["presets"]["mine"] = dict(VALID_PARAMS)
        seed["presets"]["other"] = dict(VALID_PARAMS)
        return seed

    def test_delete_removes_only_target(self):
        seed = self._seeded()
        with (
            mock.patch("qwen3_tts.core.config.load_config", return_value=seed),
            mock.patch("qwen3_tts.core.config.save_config") as save_cfg,
        ):
            ok, msg = self._writer()("mine")
        self.assertTrue(ok, msg)
        self.assertIn("Deleted preset 'mine'", msg)
        written = save_cfg.call_args[0][0]
        self.assertNotIn("mine", written["presets"])
        self.assertIn("other", written["presets"])
        self.assertIn("stable", written["presets"])

    def test_delete_does_not_mutate_loaded_config(self):
        seed = self._seeded()
        snapshot = copy.deepcopy(seed)
        with (
            mock.patch("qwen3_tts.core.config.load_config", return_value=seed),
            mock.patch("qwen3_tts.core.config.save_config"),
        ):
            self._writer()("mine")
        self.assertEqual(seed, snapshot)

    def test_corrupt_config_returns_false_and_never_saves(self):
        with (
            mock.patch(
                "qwen3_tts.core.config.load_config",
                side_effect=ValueError("corrupt"),
            ),
            mock.patch("qwen3_tts.core.config.save_config") as save_cfg,
        ):
            ok, msg = self._writer()("mine")
        self.assertFalse(ok)
        self.assertTrue(msg)
        save_cfg.assert_not_called()

    def test_membership_miss_renders_no_longer_exists(self):
        with (
            mock.patch(
                "qwen3_tts.core.config.load_config", return_value=self._seeded()
            ),
            mock.patch("qwen3_tts.core.config.save_config") as save_cfg,
        ):
            ok, msg = self._writer()("gone")
        self.assertFalse(ok)
        self.assertIn("no longer exists", msg)
        save_cfg.assert_not_called()

    def test_factory_name_rejected_by_writer(self):
        with (
            mock.patch(
                "qwen3_tts.core.config.load_config", return_value=self._seeded()
            ),
            mock.patch("qwen3_tts.core.config.save_config") as save_cfg,
        ):
            ok, msg = self._writer()("stable")
        self.assertFalse(ok)
        self.assertIn("Factory presets", msg)
        save_cfg.assert_not_called()

    def test_delete_write_error_message_has_no_paths(self):
        err = OSError(28, "No space left on device", "/Users/secret/config.json")
        with (
            mock.patch(
                "qwen3_tts.core.config.load_config", return_value=self._seeded()
            ),
            mock.patch("qwen3_tts.core.config.save_config", side_effect=err),
        ):
            ok, msg = self._writer()("mine")
        self.assertFalse(ok)
        self.assertIn("No space left on device", msg)
        self.assertNotIn("/Users/secret", msg)
        self.assertNotIn("Errno", msg)


# ---------------------------------------------------------------------------
# Save handler
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_GRADIO, "gradio not installed")
class TestSaveHandler(unittest.TestCase):
    def _handler(self):
        from qwen3_tts.interface.ui.tabs_generation import (
            _on_save_generation_preset,
        )

        return _on_save_generation_preset

    def _assert_announced(self, result):
        # Slot 3 carries generation._announce_status's sr-only aria-live
        # wrapper — not "", not the raw status text.
        self.assertTrue(result[3])
        self.assertIn("aria-live", result[3])

    def _assert_bare_dropdowns(self, result):
        # Non-success rule: all four dropdown slots get a bare no-change update.
        for slot in (4, 5, 6, 7):
            self.assertEqual(result[slot], gr.update(), slot)

    def _call(
        self, state, name, existing=None, writer=None, choices=None, user_choices=None
    ):
        if user_choices is None:
            user_choices = ["(none)"] + sorted(existing or {})
        with (
            mock.patch(
                "qwen3_tts.core.config.get_user_generation_presets",
                return_value=existing or {},
            ),
            mock.patch(
                "qwen3_tts.core.config.save_user_generation_preset",
                side_effect=writer or (lambda n, p: (True, f"Saved preset '{n}'.")),
            ) as save_mock,
            mock.patch(
                "qwen3_tts.interface.ui.shared.get_presets",
                return_value=choices or ["(none)"],
            ),
            mock.patch(
                "qwen3_tts.interface.ui.shared.get_user_generation_preset_choices",
                return_value=user_choices,
            ),
        ):
            result = self._handler()(
                dict(state),
                name,
                0.6,
                40,
                0.92,
                1.08,
                "whatever",
                "(none)",
                "(none)",
            )
        return result, save_mock

    def test_empty_name_renders_copy_and_never_saves(self):
        result, save_mock = self._call(dict(DISARMED), "")
        self.assertEqual(len(result), 8)
        self.assertEqual(result[0], DISARMED)
        self.assertEqual(result[1], gr.update(value="Save as preset"))
        self.assertEqual(result[2], "Type a preset name first.")
        self._assert_bare_dropdowns(result)
        self._assert_announced(result)
        save_mock.assert_not_called()

    def test_whitespace_name_renders_copy(self):
        result, save_mock = self._call(dict(DISARMED), "   ")
        self.assertEqual(result[2], "Type a preset name first.")
        save_mock.assert_not_called()

    def test_bad_charset_renders_copy(self):
        result, save_mock = self._call(dict(DISARMED), "mine!")
        self.assertEqual(result[2], COPY_BAD_NAME)
        save_mock.assert_not_called()

    def test_factory_name_rejected_never_arms(self):
        result, save_mock = self._call(dict(DISARMED), "stable")
        self.assertEqual(result[2], COPY_FACTORY)
        self.assertEqual(result[0], DISARMED)
        save_mock.assert_not_called()

    def test_new_preset_saves_and_refreshes_all_three_dropdowns(self):
        result, save_mock = self._call(
            dict(DISARMED),
            "mine",
            existing={},
            choices=["(none)", "natural", "mine"],
            user_choices=["(none)", "mine"],
        )
        save_mock.assert_called_once_with(
            "mine",
            {
                "temperature": 0.6,
                "top_k": 40,
                "top_p": 0.92,
                "repetition_penalty": 1.08,
            },
        )
        self.assertEqual(result[2], "Saved preset 'mine'.")
        self.assertEqual(result[0], DISARMED)
        # The three Preset dropdowns refresh with the MERGED choices (factory
        # sample "natural" present — a user-choices-only feed fails here).
        for slot in (4, 5, 6):
            self.assertEqual(
                result[slot], gr.update(choices=["(none)", "natural", "mine"]), slot
            )
            self.assertNotIn("value", result[slot])
        self.assertEqual(result[7], gr.update(choices=["(none)", "mine"]))
        self._assert_announced(result)

    def test_existing_name_arms_instead_of_saving(self):
        result, save_mock = self._call(
            dict(DISARMED), "mine", existing={"mine": dict(VALID_PARAMS)}
        )
        save_mock.assert_not_called()
        self.assertTrue(result[0]["armed"])
        self.assertEqual(result[0]["armed_name"], "mine")
        self.assertGreater(result[0]["ts"], time.time() - 2)
        self.assertEqual(result[1], gr.update(value="Confirm Overwrite? (click again)"))
        self.assertEqual(result[2], COPY_SAVE_ARM.format(name="mine"))
        self._assert_bare_dropdowns(result)
        self._assert_announced(result)

    def test_fresh_arm_confirms_and_saves(self):
        armed = {
            "armed": True,
            "ts": time.time(),
            "armed_name": "mine",
        }
        result, save_mock = self._call(
            armed,
            "mine",
            existing={"mine": dict(VALID_PARAMS)},
            choices=["(none)", "mine"],
        )
        save_mock.assert_called_once()
        self.assertEqual(result[2], "Saved preset 'mine'.")
        self.assertEqual(result[0], DISARMED)
        self.assertEqual(result[1], gr.update(value="Save as preset"))
        # Overwrite success refreshes exactly like new-save (spec §6).
        for slot in (4, 5, 6):
            self.assertEqual(result[slot], gr.update(choices=["(none)", "mine"]), slot)
        self.assertEqual(result[7], gr.update(choices=["(none)", "mine"]))
        self._assert_announced(result)

    def test_arm_then_mismatched_target_rearms_not_saves(self):
        armed = {"armed": True, "ts": time.time(), "armed_name": "a"}
        result, save_mock = self._call(
            armed,
            "b",
            existing={"a": dict(VALID_PARAMS), "b": dict(VALID_PARAMS)},
        )
        save_mock.assert_not_called()
        self.assertTrue(result[0]["armed"])
        self.assertEqual(result[0]["armed_name"], "b")
        self.assertEqual(result[2], COPY_SAVE_ARM.format(name="b"))

    def test_expired_arm_rearms_not_saves(self):
        # Trap 7: a stale arm must re-arm (fresh ts, arm copy), never execute.
        armed = {"armed": True, "ts": time.time() - 10.0, "armed_name": "mine"}
        result, save_mock = self._call(
            armed, "mine", existing={"mine": dict(VALID_PARAMS)}
        )
        save_mock.assert_not_called()
        self.assertTrue(result[0]["armed"])
        self.assertEqual(result[0]["armed_name"], "mine")
        self.assertGreater(result[0]["ts"], time.time() - 2)
        self.assertEqual(result[2], COPY_SAVE_ARM.format(name="mine"))

    def test_whitespace_variant_cannot_bypass_overwrite_confirm(self):
        result, save_mock = self._call(
            dict(DISARMED), "  mine  ", existing={"mine": dict(VALID_PARAMS)}
        )
        save_mock.assert_not_called()
        self.assertTrue(result[0]["armed"])
        self.assertEqual(result[0]["armed_name"], "mine")

    def test_writer_failure_renders_message_and_disarms(self):
        armed = {"armed": True, "ts": time.time(), "armed_name": "mine"}
        result, _ = self._call(
            armed,
            "mine",
            existing={"mine": dict(VALID_PARAMS)},
            writer=lambda n, p: (False, "disk full"),
        )
        self.assertEqual(len(result), 8)
        self.assertIn("Could not save", result[2])
        self.assertIn("disk full", result[2])
        self.assertEqual(result[0], DISARMED)
        self.assertEqual(result[1], gr.update(value="Save as preset"))
        self._assert_bare_dropdowns(result)
        self._assert_announced(result)

    def test_output_arity_is_eight(self):
        result, _ = self._call(dict(DISARMED), "mine", existing={})
        self.assertEqual(len(result), 8)


# ---------------------------------------------------------------------------
# Delete handler
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_GRADIO, "gradio not installed")
class TestDeleteHandler(unittest.TestCase):
    def _handler(self):
        from qwen3_tts.interface.ui.tabs_generation import (
            _on_delete_generation_preset,
        )

        return _on_delete_generation_preset

    def _assert_announced(self, result):
        self.assertTrue(result[3])
        self.assertIn("aria-live", result[3])

    def _assert_bare_dropdowns(self, result):
        for slot in (4, 5, 6, 7):
            self.assertEqual(result[slot], gr.update(), slot)

    def _call(
        self,
        state,
        selection,
        existing=None,
        writer=None,
        dd_values=("whatever", "(none)", "(none)"),
    ):
        # Post-delete mocks: the merged view the handler reads AFTER the
        # writer — the deleted name gone, a factory name present (so a
        # user-choices-only feed of slots 4-6 cannot false-pass).
        remaining = sorted(k for k in (existing or {}) if k != selection)
        fresh = ["(none)", "natural"] + remaining
        user_fresh = ["(none)"] + remaining
        with (
            mock.patch(
                "qwen3_tts.core.config.get_user_generation_presets",
                return_value=existing or {},
            ),
            mock.patch(
                "qwen3_tts.core.config.delete_user_generation_preset",
                side_effect=writer or (lambda n: (True, f"Deleted preset '{n}'.")),
            ) as delete_mock,
            mock.patch(
                "qwen3_tts.interface.ui.shared.get_presets",
                return_value=fresh,
            ),
            mock.patch(
                "qwen3_tts.interface.ui.shared.get_user_generation_preset_choices",
                return_value=user_fresh,
            ),
        ):
            result = self._handler()(
                dict(state), selection, dd_values[0], dd_values[1], dd_values[2]
            )
        return result, delete_mock

    def test_nothing_selected_renders_copy(self):
        result, delete_mock = self._call(dict(DISARMED), "(none)")
        self.assertEqual(len(result), 8)
        self.assertEqual(result[2], "Select one of your presets to delete.")
        self.assertEqual(result[0], DISARMED)
        self.assertEqual(result[1], gr.update(value="Delete preset"))
        self._assert_bare_dropdowns(result)
        self._assert_announced(result)
        delete_mock.assert_not_called()

    def test_factory_name_classified_before_membership(self):
        # A factory name must classify as factory even if absent from the
        # user view — branch order ② before ③, and it can never arm.
        result, delete_mock = self._call(
            dict(DISARMED), "stable", existing={"mine": dict(VALID_PARAMS)}
        )
        self.assertEqual(result[2], COPY_FACTORY)
        self.assertEqual(result[0], DISARMED)
        self.assertEqual(result[1], gr.update(value="Delete preset"))
        delete_mock.assert_not_called()

    def test_case_variant_factory_classified(self):
        result, delete_mock = self._call(
            dict(DISARMED), "STABLE", existing={"mine": dict(VALID_PARAMS)}
        )
        self.assertEqual(result[2], COPY_FACTORY)
        self.assertEqual(result[0], DISARMED)
        self.assertEqual(result[1], gr.update(value="Delete preset"))
        delete_mock.assert_not_called()

    def test_membership_miss_renders_no_longer_exists(self):
        result, delete_mock = self._call(
            dict(DISARMED), "gone", existing={"mine": dict(VALID_PARAMS)}
        )
        self.assertEqual(result[2], COPY_MISS.format(name="gone"))
        self.assertEqual(result[0], DISARMED)
        self.assertEqual(result[1], gr.update(value="Delete preset"))
        self._assert_bare_dropdowns(result)
        delete_mock.assert_not_called()

    def test_delete_arms_on_first_click(self):
        result, delete_mock = self._call(
            dict(DISARMED), "mine", existing={"mine": dict(VALID_PARAMS)}
        )
        delete_mock.assert_not_called()
        self.assertTrue(result[0]["armed"])
        self.assertEqual(result[0]["armed_name"], "mine")
        self.assertGreater(result[0]["ts"], time.time() - 2)
        self.assertEqual(result[1], gr.update(value="Confirm Delete? (click again)"))
        self.assertEqual(result[2], COPY_DELETE_ARM.format(name="mine"))
        self._assert_bare_dropdowns(result)
        self._assert_announced(result)

    def test_fresh_arm_confirms_and_conditionally_resets(self):
        armed = {"armed": True, "ts": time.time(), "armed_name": "mine"}
        result, delete_mock = self._call(
            armed,
            "mine",
            existing={"mine": dict(VALID_PARAMS), "other": dict(VALID_PARAMS)},
            dd_values=("mine", "other", "(none)"),
        )
        delete_mock.assert_called_once_with("mine")
        self.assertEqual(result[2], "Deleted preset 'mine'.")
        self.assertEqual(result[0], DISARMED)
        self.assertEqual(result[1], gr.update(value="Delete preset"))
        fresh = ["(none)", "natural", "other"]
        # Reset only where the dropdown held the deleted name.
        self.assertEqual(result[4], gr.update(choices=fresh, value="(none)"))
        self.assertEqual(result[5], gr.update(choices=fresh))
        self.assertNotIn("value", result[5])
        self.assertEqual(result[6], gr.update(choices=fresh))
        self.assertNotIn("value", result[6])
        self.assertEqual(
            result[7], gr.update(choices=["(none)", "other"], value="(none)")
        )
        self._assert_announced(result)

    def test_writer_failure_renders_message_and_disarms(self):
        # The writer's ok/msg must be honored at confirm time: the default
        # mock's success message is string-identical to the section-7
        # literal, so without this branch a handler that hardcodes success
        # passes while a corrupt config or OSError fabricates "Deleted.".
        armed = {"armed": True, "ts": time.time(), "armed_name": "mine"}
        result, delete_mock = self._call(
            armed,
            "mine",
            existing={"mine": dict(VALID_PARAMS)},
            writer=lambda n: (False, "disk full"),
        )
        delete_mock.assert_called_once_with("mine")
        self.assertIn("disk full", result[2])
        self.assertEqual(result[0], DISARMED)
        self.assertEqual(result[1], gr.update(value="Delete preset"))
        self._assert_bare_dropdowns(result)

    def test_expired_arm_rearms_not_deletes(self):
        armed = {"armed": True, "ts": time.time() - 10.0, "armed_name": "mine"}
        result, delete_mock = self._call(
            armed, "mine", existing={"mine": dict(VALID_PARAMS)}
        )
        delete_mock.assert_not_called()
        self.assertTrue(result[0]["armed"])
        self.assertGreater(result[0]["ts"], time.time() - 2)
        self.assertEqual(result[2], COPY_DELETE_ARM.format(name="mine"))

    def test_arm_then_mismatched_target_rearms(self):
        # Trap 7: a fresh arm for "a" must NOT execute against selection "b".
        armed = {"armed": True, "ts": time.time(), "armed_name": "a"}
        result, delete_mock = self._call(
            armed,
            "b",
            existing={"a": dict(VALID_PARAMS), "b": dict(VALID_PARAMS)},
        )
        delete_mock.assert_not_called()
        self.assertTrue(result[0]["armed"])
        self.assertEqual(result[0]["armed_name"], "b")
        self.assertGreater(result[0]["ts"], time.time() - 2)
        self.assertEqual(result[2], COPY_DELETE_ARM.format(name="b"))

    def test_output_arity_is_eight(self):
        result, _ = self._call(
            dict(DISARMED), "mine", existing={"mine": dict(VALID_PARAMS)}
        )
        self.assertEqual(len(result), 8)


# ---------------------------------------------------------------------------
# Wiring + constants (character windows, never physical lines)
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_GRADIO, "gradio not installed")
class TestGenerationPresetUiWiring(unittest.TestCase):
    def _tabs_source(self):
        import qwen3_tts.interface.ui.tabs_generation as tg

        return Path(tg.__file__).read_text()

    def _facade_source(self):
        import qwen3_tts.interface.ui._facade as facade

        return Path(facade.__file__).read_text()

    def _shared_source(self):
        import qwen3_tts.interface.ui.shared as shared

        return Path(shared.__file__).read_text()

    def _kwargs_window(self, block, start_marker, end_marker=None):
        # Character window from start_marker to end_marker (default: block
        # end). Used instead of T1's "]" terminator because preset_builder
        # refs contain "]" themselves.
        start = block.index(start_marker)
        end = block.index(end_marker, start) if end_marker else len(block)
        return block[start:end]

    def test_clone_tab_carries_builder_accordion(self):
        src = self._tabs_source()
        start = src.index("def _build_clone_tab")
        end = src.index("def _build_design_tab", start)
        window = src[start:end]
        self.assertIn("My generation presets", window)
        self.assertIn('label="Preset"', window)

    def test_handlers_defined_at_module_level(self):
        src = self._tabs_source()
        self.assertIn("def _on_save_generation_preset(", src)
        self.assertIn("def _on_delete_generation_preset(", src)

    def test_handlers_reuse_generic_prosody_mechanics(self):
        # Scoped to the NEW handler bodies: the needles already exist in the
        # file from T1, so a file-global check would pass hollowly.
        src = self._tabs_source()
        for handler in (
            "_on_save_generation_preset",
            "_on_delete_generation_preset",
        ):
            start = src.index(f"def {handler}(")
            end = src.index("\ndef ", start + 1)
            window = src[start:end]
            self.assertIn("_prosody_arm_is_fresh(", window, handler)
            self.assertIn("_PROSODY_DISARMED_STATE", window, handler)

    def test_none_choice_constant(self):
        from qwen3_tts.interface.ui.tabs_generation import NONE_CHOICE

        self.assertEqual(NONE_CHOICE, "(none)")

    def test_button_labels_match_pinned_copy(self):
        import qwen3_tts.interface.ui.tabs_generation as tg

        self.assertEqual(tg._PROSODY_SAVE_BTN_BASE, "Save as preset")
        self.assertEqual(tg._PROSODY_SAVE_BTN_ARM, "Confirm Overwrite? (click again)")
        self.assertEqual(tg._PROSODY_DELETE_BTN_BASE, "Delete preset")
        self.assertEqual(tg._PROSODY_DELETE_BTN_ARM, "Confirm Delete? (click again)")

    def test_facade_save_click_pins_fn_inputs_outputs(self):
        src = self._facade_source()
        # fn= is unique (the top-of-file import block never carries "fn="),
        # so this anchors on the WIRING, not the import — robust to wherever
        # the handler name first appears.
        idx = src.index("fn=_on_save_generation_preset")
        before = src[max(0, idx - 300) : idx]
        self.assertIn(".click(", before)
        self.assertIn("preset_builder", before)
        block = src[idx : idx + 1600]
        inputs = self._kwargs_window(block, "inputs=[", "outputs=[")
        # The four sliders map POSITIONALLY onto (temp, top_k, top_p, rep) —
        # an order swap here saves presets with swapped params.
        ordered_inputs = [
            'preset_builder["save_state"]',
            'preset_builder["name"]',
            'preset_builder["temp"]',
            'preset_builder["top_k"]',
            'preset_builder["top_p"]',
            'preset_builder["rep"]',
            "clone_preset",
            "design_preset",
            "custom_preset",
        ]
        positions = [inputs.index(ref) for ref in ordered_inputs]
        self.assertEqual(
            positions, sorted(positions), f"inputs order wrong: {inputs!r}"
        )
        outputs = self._kwargs_window(block, "outputs=[")
        ordered_outputs = [
            'preset_builder["save_state"]',
            'preset_builder["save_btn"]',
            'preset_builder["save_status"]',
            'preset_builder["announcer"]',
            "clone_preset",
            "design_preset",
            "custom_preset",
            'preset_builder["delete_dropdown"]',
        ]
        positions = [outputs.index(ref) for ref in ordered_outputs]
        self.assertEqual(
            positions, sorted(positions), f"outputs order wrong: {outputs!r}"
        )

    def test_facade_delete_click_pins_fn_inputs_outputs(self):
        src = self._facade_source()
        idx = src.index("fn=_on_delete_generation_preset")
        before = src[max(0, idx - 300) : idx]
        self.assertIn(".click(", before)
        self.assertIn("preset_builder", before)
        block = src[idx : idx + 1600]
        inputs = self._kwargs_window(block, "inputs=[", "outputs=[")
        # Order matters positionally: a state/selection swap corrupts the
        # arm logic; a design/custom swap misdirects the conditional reset.
        ordered_inputs = [
            'preset_builder["delete_state"]',
            'preset_builder["delete_dropdown"]',
            "clone_preset",
            "design_preset",
            "custom_preset",
        ]
        positions = [inputs.index(ref) for ref in ordered_inputs]
        self.assertEqual(
            positions, sorted(positions), f"inputs order wrong: {inputs!r}"
        )
        outputs = self._kwargs_window(block, "outputs=[")
        ordered_outputs = [
            'preset_builder["delete_state"]',
            'preset_builder["delete_btn"]',
            'preset_builder["delete_status"]',
            'preset_builder["announcer"]',
            "clone_preset",
            "design_preset",
            "custom_preset",
            'preset_builder["delete_dropdown"]',
        ]
        positions = [outputs.index(ref) for ref in ordered_outputs]
        self.assertEqual(
            positions, sorted(positions), f"outputs order wrong: {outputs!r}"
        )

    def test_shared_exposes_user_generation_choices(self):
        src = self._shared_source()
        self.assertIn("def get_user_generation_preset_choices(", src)


if __name__ == "__main__":
    unittest.main()
