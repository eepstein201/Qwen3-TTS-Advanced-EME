#!/usr/bin/env python3
"""Phase 1c: Tests for confirm patterns on destructive actions.

TDD RED phase — all tests start failing; GREEN phase wires in the implementation.

Coverage:
  1. confirm_step() pure logic (8 tests)
  2. generate_guard_check() logic (5 tests)
  3. Structural wiring assertions for _facade.py + generation.py (5 tests)
"""

import time
import unittest


class TestConfirmStep(unittest.TestCase):
    """confirm_step(confirm_state, arm_label, original_label, timeout_s) tests."""

    def _import(self):
        from qwen3_tts.interface.ui.components import confirm_step
        return confirm_step

    def test_01_unarmed_state_arms_on_first_click(self):
        """First click on unarmed state returns is_confirmed=False and armed=True."""
        confirm_step = self._import()
        state = {"armed": False, "ts": 0.0}
        new_state, _, confirmed = confirm_step(state, "Confirm?", "Delete")
        self.assertFalse(confirmed)
        self.assertTrue(new_state.get("armed"))

    def test_02_unarmed_ts_updated_to_now(self):
        """New state timestamp is close to now when arming."""
        confirm_step = self._import()
        state = {"armed": False, "ts": 0.0}
        new_state, _, _ = confirm_step(state, "Confirm?", "Delete")
        self.assertAlmostEqual(new_state["ts"], time.time(), delta=1.0)

    def test_03_armed_within_timeout_confirms(self):
        """Second click within timeout returns is_confirmed=True and armed=False."""
        confirm_step = self._import()
        state = {"armed": True, "ts": time.time()}
        new_state, _, confirmed = confirm_step(state, "Confirm?", "Delete")
        self.assertTrue(confirmed)
        self.assertFalse(new_state.get("armed"))

    def test_04_armed_state_resets_ts_after_confirm(self):
        """State ts resets to 0.0 after confirm."""
        confirm_step = self._import()
        state = {"armed": True, "ts": time.time()}
        new_state, _, _ = confirm_step(state, "Confirm?", "Delete")
        self.assertEqual(new_state["ts"], 0.0)

    def test_05_expired_armed_state_rearms_not_confirms(self):
        """Click after timeout re-arms (is_confirmed=False) instead of confirming."""
        confirm_step = self._import()
        state = {"armed": True, "ts": 0.0}  # epoch = very old
        new_state, _, confirmed = confirm_step(state, "Confirm?", "Delete")
        self.assertFalse(confirmed)
        self.assertTrue(new_state.get("armed"))

    def test_06_none_state_treated_as_unarmed(self):
        """None state is treated as unarmed — arms on first click."""
        confirm_step = self._import()
        new_state, _, confirmed = confirm_step(None, "Confirm?", "Delete")
        self.assertFalse(confirmed)
        self.assertTrue(new_state.get("armed"))

    def test_07_arm_label_in_btn_update_when_arming(self):
        """Button update value is arm_label when arming."""
        confirm_step = self._import()
        state = {"armed": False, "ts": 0.0}
        _, btn_update, _ = confirm_step(state, "Confirm Delete? (click again)", "Delete")
        self.assertEqual(btn_update.get("value"), "Confirm Delete? (click again)")

    def test_08_original_label_in_btn_update_when_confirming(self):
        """Button update value is original_label when confirming."""
        confirm_step = self._import()
        state = {"armed": True, "ts": time.time()}
        _, btn_update, _ = confirm_step(state, "Confirm?", "Delete")
        self.assertEqual(btn_update.get("value"), "Delete")

    def test_09_custom_timeout_respected(self):
        """Custom timeout_s=0 means armed state always expires immediately."""
        confirm_step = self._import()
        state = {"armed": True, "ts": time.time()}
        # timeout_s=0 → even a just-armed click is expired
        _, _, confirmed = confirm_step(state, "Confirm?", "Delete", timeout_s=0)
        self.assertFalse(confirmed)


class TestGenerateGuardCheck(unittest.TestCase):
    """generate_guard_check(gen_guard_state) → (new_state, status_text, is_blocked)."""

    def _import(self):
        from qwen3_tts.interface.ui.generation import generate_guard_check
        return generate_guard_check

    def test_10_not_generating_passes_through(self):
        """When not generating, guard returns is_blocked=False."""
        fn = self._import()
        state = {"generating": False, "armed": False, "ts": 0.0}
        _, _, blocked = fn(state)
        self.assertFalse(blocked)

    def test_11_not_generating_marks_generating_true(self):
        """When guard passes through, new state has generating=True."""
        fn = self._import()
        state = {"generating": False, "armed": False, "ts": 0.0}
        new_state, _, _ = fn(state)
        self.assertTrue(new_state.get("generating"))

    def test_12_generating_unarmed_blocks_and_arms(self):
        """When generating and unarmed, returns is_blocked=True and arms state."""
        fn = self._import()
        state = {"generating": True, "armed": False, "ts": 0.0}
        new_state, status_text, blocked = fn(state)
        self.assertTrue(blocked)
        self.assertTrue(new_state.get("armed"))
        self.assertIn("again", status_text.lower())

    def test_13_generating_armed_within_timeout_unblocks(self):
        """When generating and armed within 5s, guard unblocks (confirmed)."""
        fn = self._import()
        state = {"generating": True, "armed": True, "ts": time.time()}
        _, _, blocked = fn(state)
        self.assertFalse(blocked)

    def test_14_invalid_state_treated_as_not_generating(self):
        """None/invalid state is treated as not generating — guard passes through."""
        fn = self._import()
        _, _, blocked = fn(None)
        self.assertFalse(blocked)


class TestConfirmWiringStructural(unittest.TestCase):
    """Structural tests: confirm wiring present in _facade.py and generation.py."""

    def _read_source(self, rel_path):
        import os
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(base, rel_path)) as f:
            return f.read()

    def test_15_confirm_step_referenced_in_facade(self):
        """`confirm_step` is imported or used in _facade.py."""
        src = self._read_source("qwen3_tts/interface/ui/_facade.py")
        self.assertIn("confirm_step", src)

    def test_16_delete_confirm_state_in_manage_voices_tab(self):
        """`delete_confirm_state` gr.State is present in _facade.py."""
        src = self._read_source("qwen3_tts/interface/ui/_facade.py")
        self.assertIn("delete_confirm_state", src)

    def test_17_unload_confirm_state_in_manage_models_tab(self):
        """`unload_confirm_state` gr.State is present in _facade.py."""
        src = self._read_source("qwen3_tts/interface/ui/_facade.py")
        self.assertIn("unload_confirm_state", src)

    def test_18_gen_guard_state_in_facade(self):
        """`gen_guard_state` is present in _facade.py for generate guard."""
        src = self._read_source("qwen3_tts/interface/ui/_facade.py")
        self.assertIn("gen_guard_state", src)

    def test_19_wire_generation_tab_accepts_gen_guard_state(self):
        """`_wire_generation_tab` signature includes `gen_guard_state` parameter."""
        src = self._read_source("qwen3_tts/interface/ui/generation.py")
        self.assertIn("gen_guard_state", src)


if __name__ == "__main__":
    unittest.main()
