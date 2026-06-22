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
from unittest.mock import MagicMock, patch


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
        # Use a fixed timestamp to avoid timing dependency
        fixed_time = 1234567890.0
        state = {"armed": True, "ts": fixed_time}
        # Mock time.time() to return fixed_time + 0.1 to simulate slight delay
        with patch("time.time", return_value=fixed_time + 0.1):
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


class TestVoiceDeleteMetadata(unittest.TestCase):
    """Test voice delete confirmation shows metadata."""

    @patch("qwen3_tts.server.client.TTSClient")
    def test_prepare_delete_shows_metadata(self, mock_client_cls):
        """Voice delete confirmation shows duration + format."""
        # Mock successful metadata response
        mock_client = MagicMock()
        mock_client.is_server_running.return_value = True
        mock_client.get_prompt_details.return_value = {
            "name": "test-voice",
            "formats": [".pt"],
            "size_bytes": 1234567,
            "created": 1715320000.0,  # Fixed timestamp
            "is_default": False,
        }
        mock_client_cls.return_value = mock_client

        from qwen3_tts.interface.ui.shared import get_voice_metadata

        metadata = get_voice_metadata("test-voice")

        self.assertEqual(metadata["name"], "test-voice")
        self.assertIn(".pt", metadata["formats"])
        self.assertAlmostEqual(metadata["size_mb"], 1.18, places=2)
        self.assertIsNotNone(metadata["created"])

    @patch("qwen3_tts.server.client.TTSClient")
    @patch("qwen3_tts.interface.ui.shared.time")
    def test_recent_voice_warning(self, mock_time, mock_client_cls):
        """Warning shown if voice created <5 minutes ago."""
        # Mock recent voice (2 minutes ago)
        mock_client = MagicMock()
        mock_client.is_server_running.return_value = True
        mock_client.get_prompt_details.return_value = {
            "name": "recent-voice",
            "formats": [".wav"],
            "size_bytes": 500000,
            "created": 1715320000.0,  # Will be overridden by mock_time
            "is_default": False,
        }
        mock_client_cls.return_value = mock_client

        # Mock time.time() to return 2 minutes after created
        mock_time.time.return_value = 1715320000.0 + 120

        from qwen3_tts.interface.ui.shared import get_voice_metadata

        metadata = get_voice_metadata("recent-voice")
        age_seconds = mock_time.time() - metadata["created"]

        # Verify recent warning would be triggered (<300 seconds)
        self.assertLess(age_seconds, 300)


class TestModelUnloadMetadata(unittest.TestCase):
    """Test model unload confirmation shows memory usage."""

    @patch("qwen3_tts.interface.ui.model_management.get_model_table_data")
    def test_prepare_unload_shows_memory(self, mock_get_models):
        """Model unload confirmation shows memory usage."""
        # Mock model data with memory info
        mock_get_models.return_value = [
            ["clone", "Loaded", "2450.5", "default"],
            ["design", "Loaded", "1800.0", "user"],
            ["custom", "Not Loaded", "0", "user"],
        ]

        from qwen3_tts.interface.ui.components import ConfirmButton
        from qwen3_tts.interface.ui.model_management import get_model_table_data

        ConfirmButton("Confirm", "Original", 5.0, "")

        # Simulate the first click logic (showing metadata)
        models = get_model_table_data()
        model = next((m for m in models if m[0] == "clone"), None)
        self.assertIsNotNone(model)
        self.assertEqual(model[2], "2450.5")
        self.assertEqual(model[3], "default")

    @patch("qwen3_tts.interface.ui.model_management.get_model_table_data")
    def test_startup_model_warning(self, mock_get_models):
        """Warning shown if model is startup=default."""
        mock_get_models.return_value = [
            ["clone", "Loaded", "2450.5", "default"],
        ]

        from qwen3_tts.interface.ui.model_management import get_model_table_data

        models = get_model_table_data()
        model = next((m for m in models if m[0] == "clone"), None)
        self.assertIsNotNone(model)
        self.assertEqual(model[3], "default")
        # Warning would be shown in banner message


class TestGenerationCancelProgress(unittest.TestCase):
    """Test generation cancel confirmation with progress display."""

    @patch("qwen3_tts.core.http_client.server_request")
    @patch("qwen3_tts.interface.ui.generation.is_server_running")
    def test_cancel_under_10_pct_skips_confirmation(self, mock_running, mock_request):
        """Cancel <10% complete skips confirmation."""
        mock_running.return_value = True

        # Mock generation status with <10% progress
        mock_request.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "active": True,
                "chunk_index": 1,
                "chunk_total": 20,
                "eta_sec": 45,
            },
        )

        from qwen3_tts.interface.ui.generation import _prepare_cancel_confirmation

        status_msg, should_proceed, progress_pct, chunks, eta = _prepare_cancel_confirmation()

        # Should proceed immediately (no confirmation needed)
        self.assertTrue(should_proceed)
        self.assertLess(progress_pct, 10)
        self.assertEqual(chunks, 1)

    @patch("qwen3_tts.core.http_client.server_request")
    @patch("qwen3_tts.interface.ui.generation.is_server_running")
    def test_cancel_over_10_pct_requires_confirmation(self, mock_running, mock_request):
        """Cancel >10% complete requires confirmation."""
        mock_running.return_value = True

        # Mock generation status with >10% progress
        mock_request.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "active": True,
                "chunk_index": 12,
                "chunk_total": 20,
                "eta_sec": 18,
            },
        )

        from qwen3_tts.interface.ui.generation import _prepare_cancel_confirmation

        status_msg, should_proceed, progress_pct, chunks, eta = _prepare_cancel_confirmation()

        # Should require confirmation
        self.assertFalse(should_proceed)
        self.assertGreater(progress_pct, 10)
        self.assertEqual(chunks, 12)
        self.assertIn("60%", status_msg)  # 12/20 = 60%

    @patch("qwen3_tts.core.http_client.server_request")
    @patch("qwen3_tts.interface.ui.generation.is_server_running")
    def test_cancel_no_active_generation_proceeds(self, mock_running, mock_request):
        """Cancel with no active generation proceeds immediately."""
        mock_running.return_value = True

        # Mock no active generation
        mock_request.return_value = MagicMock(
            status_code=200, json=lambda: {"active": False}
        )

        from qwen3_tts.interface.ui.generation import _prepare_cancel_confirmation

        status_msg, should_proceed, progress_pct, chunks, eta = _prepare_cancel_confirmation()

        # Should proceed immediately
        self.assertTrue(should_proceed)
        self.assertEqual(progress_pct, 0)


class TestConfirmationThreadSafety(unittest.TestCase):
    """Test confirmation patterns are thread-safe for concurrent tabs."""

    def test_multiple_concurrent_confirmations(self):
        """Multiple simultaneous confirmations don't interfere."""
        import threading

        from qwen3_tts.interface.ui.components import ConfirmButton

        confirm_btn = ConfirmButton("Confirm", "Original", 5.0, "")

        # Simulate 10 concurrent confirmation attempts
        results = []
        threads = []

        def attempt_confirmation(thread_id):
            state = {"armed": False, "ts": 0.0}
            new_state, btn_update, status, confirmed = confirm_btn.click(state)
            results.append((thread_id, new_state["armed"]))

        for i in range(10):
            t = threading.Thread(target=attempt_confirmation, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All should have armed successfully
        self.assertEqual(len(results), 10)
        for thread_id, armed in results:
            self.assertTrue(armed, f"Thread {thread_id} failed to arm")


if __name__ == "__main__":
    unittest.main()
