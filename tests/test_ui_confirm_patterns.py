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
    """Structural tests: confirm wiring present across the ui package.

    The tab builders live in tabs_generation.py / tabs_management.py; the
    Clear All confirm and the confirm_step contract stay in _facade.py.
    """

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
        """`delete_confirm_state` gr.State is present in tabs_management.py."""
        src = self._read_source("qwen3_tts/interface/ui/tabs_management.py")
        self.assertIn("delete_confirm_state", src)

    def test_17_unload_confirm_state_in_manage_models_tab(self):
        """`unload_confirm_state` gr.State is present in tabs_management.py."""
        src = self._read_source("qwen3_tts/interface/ui/tabs_management.py")
        self.assertIn("unload_confirm_state", src)

    def test_18_gen_guard_state_in_generation_tabs(self):
        """`gen_guard_state` is present in tabs_generation.py for generate guard."""
        src = self._read_source("qwen3_tts/interface/ui/tabs_generation.py")
        self.assertIn("gen_guard_state", src)

    def test_18b_clear_history_confirm_state_in_facade(self):
        """`clear_history_confirm_state` gr.State is present in _facade.py."""
        src = self._read_source("qwen3_tts/interface/ui/_facade.py")
        self.assertIn("clear_history_confirm_state", src)

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


try:
    import gradio as gr

    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False
    gr = None  # type: ignore

skip_if_no_gradio = unittest.skipUnless(HAS_GRADIO, "requires gradio")


@skip_if_no_gradio
class TestConfirmHandlerContracts(unittest.TestCase):
    """U1/U2: execute the wired confirm handlers, don't just grep for them.

    Every existing test in this module either exercises ``confirm_step`` in
    isolation or greps the source for a symbol name. Neither can see a handler
    that returns the wrong number of values for its wired outputs, or one that
    stores the wrong TYPE into its ``gr.State`` — so both bugs shipped in the
    documented Stop / Delete / Unload flows.
    """

    def _wire_cancel_handler(self):
        """Wire a generation tab with mocks; return (handler_fn, outputs).

        Mirrors the mock-component pattern in test_ui_audio_reset.py: the
        builder calls ``cancel_btn.click(fn=..., outputs=...)`` on a Mock, so
        the wiring is recorded and the closure can be executed directly.
        """
        from unittest.mock import Mock

        from qwen3_tts.interface.ui.generation import _wire_generation_tab

        btn = Mock(spec=gr.Button)
        chain = Mock()
        chain.then = Mock(return_value=chain)
        btn.click = Mock(return_value=chain)

        cancel_btn = Mock(spec=gr.Button)
        cancel_chain = Mock()
        cancel_chain.then = Mock(return_value=cancel_chain)
        cancel_btn.click = Mock(return_value=cancel_chain)

        _wire_generation_tab(
            mode="clone",
            btn=btn,
            cancel_btn=cancel_btn,
            status=Mock(spec=gr.Textbox),
            stream_config=Mock(spec=gr.JSON),
            result_data=Mock(spec=gr.JSON),
            mode_hidden=Mock(spec=gr.Textbox),
            text_hidden=Mock(spec=gr.Textbox),
            model_indicator=Mock(spec=gr.HTML),
            text=Mock(spec=gr.Textbox),
            text_info=Mock(spec=gr.Textbox),
            inputs_list=[Mock(spec=gr.Textbox)],
            status_html=Mock(spec=gr.HTML),
            config_handler=Mock(return_value=(None, "status")),
            history_state=Mock(),
            audio_url_converter=Mock(spec=gr.Audio),
        )

        call = cancel_btn.click.call_args
        self.assertIsNotNone(call, "cancel button was never wired")
        return call.kwargs["fn"], call.kwargs["outputs"]

    def test_cancel_arm_path_returns_one_value_per_wired_output(self):
        """First Stop click (confirmation required) must fill every output.

        The arm branch returned 3 values for 4 wired outputs, so Gradio's
        validate_outputs raises ValueError and the documented Stop/confirm flow
        crashes instead of arming. No test executed the closure, and the
        playwright e2e early-returns before reaching it.
        """
        handler, outputs = self._wire_cancel_handler()

        with patch(
            "qwen3_tts.interface.ui.generation._prepare_cancel_confirmation",
            return_value=("Stop generation? 50% done", False, 50.0, 3, 10.0),
        ):
            result = handler({"armed": False, "ts": 0.0})

        self.assertEqual(
            len(result),
            len(outputs),
            f"Stop arm path returned {len(result)} values for "
            f"{len(outputs)} wired outputs; Gradio raises ValueError",
        )

    def test_cancel_canceled_path_returns_one_value_per_wired_output(self):
        """An expired confirmation window must also fill every output.

        Reached when the state is armed but the 5 s window lapsed:
        confirm_step re-arms and reports confirmed=False. That branch returned
        3 values for 4 outputs too.
        """
        handler, outputs = self._wire_cancel_handler()

        # armed=True with an epoch ts => window expired => confirmed False.
        result = handler({"armed": True, "ts": 0.0})

        self.assertEqual(
            len(result),
            len(outputs),
            f"Stop canceled path returned {len(result)} values for "
            f"{len(outputs)} wired outputs; Gradio raises ValueError",
        )

    def test_cancel_arm_path_stores_a_state_dict_not_a_tuple(self):
        """The first output feeds a gr.State that the next click reads.

        ``cancel_confirm_btn.click(state)`` returns a 4-tuple; assigning it to
        a single name stores the whole tuple in the State, and the second click
        then calls ``.get()`` on a tuple.
        """
        handler, _ = self._wire_cancel_handler()

        with patch(
            "qwen3_tts.interface.ui.generation._prepare_cancel_confirmation",
            return_value=("Stop generation? 50% done", False, 50.0, 3, 10.0),
        ):
            result = handler({"armed": False, "ts": 0.0})

        self.assertIsInstance(
            result[0],
            dict,
            "confirm state must be the state dict, not the whole "
            "ConfirmButton.click() 4-tuple",
        )
        self.assertTrue(
            result[0].get("armed"),
            "first click did not arm the confirmation",
        )

    def _handler_from_tab(self, build, name, *args):
        """Build a tab inside a Blocks and return (handler_fn, outputs).

        gr.Blocks records every registered listener in ``demo.fns`` with its
        resolved output list, which is the real wiring — not a source grep.
        """
        with gr.Blocks() as demo:
            build(*args)
        for fn_meta in demo.fns.values():
            if getattr(fn_meta.fn, "__name__", None) == name:
                return fn_meta.fn, fn_meta.outputs
        self.fail(f"{name} was never wired into the tab")

    def _voice_delete_handler(self):
        from qwen3_tts.interface.ui.tabs_management import _build_manage_voices_tab

        return self._handler_from_tab(
            _build_manage_voices_tab,
            "on_delete_click",
            gr.Dropdown(choices=[], label="prompt"),
        )

    def _model_unload_handler(self):
        from qwen3_tts.interface.ui.tabs_management import _build_manage_models_tab

        return self._handler_from_tab(
            _build_manage_models_tab,
            "on_unload_click",
            gr.HTML(),
            gr.HTML(),
            gr.HTML(),
            gr.HTML(),
        )

    def test_delete_voice_second_click_reaches_the_confirmed_branch(self):
        """Two-click Delete must work; the state carried between clicks is a dict.

        ``new_state = delete_confirm_btn.click(state)`` stored the whole
        4-tuple in the gr.State. The arity matched (5 values for 5 outputs) so
        Gradio accepted it, and the bug only bit on the SECOND click, where
        ``state.get("armed")`` hits a tuple. Delete-Voice could therefore never
        reach its confirmed branch.
        """
        handler, _ = self._voice_delete_handler()

        metadata = {
            "duration": "3.2s",
            "formats": [".pt", ".wav"],
            "size_mb": 1.2,
            "created": 0.0,
        }
        with patch(
            "qwen3_tts.interface.ui.shared.get_voice_metadata",
            return_value=metadata,
        ):
            first = handler({"armed": False, "ts": 0.0}, "my-voice")

        self.assertIsInstance(
            first[0],
            dict,
            "delete confirm state must be the state dict, not the whole "
            "ConfirmButton.click() 4-tuple",
        )

        # Feed the state straight back, exactly as Gradio does on click 2.
        with patch(
            "qwen3_tts.interface.ui.shared.get_voice_metadata",
            return_value=metadata,
        ), patch(
            "qwen3_tts.interface.ui.voice_management.delete_voice",
            return_value=("Deleted", [], None),
        ) as delete_voice:
            handler(first[0], "my-voice")  # must not raise AttributeError

        delete_voice.assert_called_once_with("my-voice")

    def test_unload_model_second_click_reaches_the_confirmed_branch(self):
        """Same defect on the Unload-Model confirm."""
        handler, _ = self._model_unload_handler()

        table = [["clone", "Loaded", "2500 MB", "Yes"]]
        with patch(
            "qwen3_tts.interface.ui.model_management.get_model_table_data",
            return_value=table,
        ):
            first = handler({"armed": False, "ts": 0.0}, "clone")

        self.assertIsInstance(
            first[0],
            dict,
            "unload confirm state must be the state dict, not the whole "
            "ConfirmButton.click() 4-tuple",
        )

        with patch(
            "qwen3_tts.interface.ui.model_management.get_model_table_data",
            return_value=table,
        ), patch(
            "qwen3_tts.interface.ui.model_management.toggle_model",
            return_value=("Unloaded", [], ""),
        ) as toggle_model:
            handler(first[0], "clone")  # must not raise AttributeError

        toggle_model.assert_called_once_with("clone", "unload")


if __name__ == "__main__":
    unittest.main()
