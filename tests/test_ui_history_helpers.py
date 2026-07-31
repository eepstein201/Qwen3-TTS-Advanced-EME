"""Unit tests for Recent Generations history helpers (clear/delete/copy).

Covers the pure, Gradio-independent logic added to shared.py:
- add_to_history preserves a full_text field (for copy-to-clipboard) alongside
  the truncated display text.
- remove_history_row / clear_history are immutable list operations.
- get_history_data appends a 6th "Remove" (✕) column per row.
- history_lock is a usable cross-module threading lock.
"""

import unittest


class TestAddToHistoryFullText(unittest.TestCase):
    """add_to_history must retain the full transcript for copy-to-clipboard."""

    def test_short_text_full_text_equals_display(self):
        from qwen3_tts.interface.ui.shared import add_to_history

        history = add_to_history([], "clone", "Hello", "/tmp/a.wav", 1, seed=7)
        self.assertEqual(history[0]["text"], "Hello")
        self.assertEqual(history[0]["full_text"], "Hello")

    def test_long_text_truncated_display_but_full_text_preserved(self):
        from qwen3_tts.interface.ui.shared import add_to_history

        long_text = "A" * 100
        history = add_to_history([], "clone", long_text, "/tmp/a.wav", 1)
        # Display text is truncated for the table cell...
        self.assertEqual(history[0]["text"], "A" * 40 + "...")
        # ...but the full transcript is retained for copy-to-clipboard.
        self.assertEqual(history[0]["full_text"], long_text)

    def test_does_not_mutate_input_list(self):
        from qwen3_tts.interface.ui.shared import add_to_history

        original = []
        add_to_history(original, "clone", "Hi", "/tmp/a.wav", 1)
        self.assertEqual(original, [])


class TestRemoveHistoryRow(unittest.TestCase):
    """remove_history_row returns a new list; never mutates the input."""

    def test_remove_middle(self):
        from qwen3_tts.interface.ui.shared import remove_history_row

        rows = [{"t": 1}, {"t": 2}, {"t": 3}]
        result = remove_history_row(rows, 1)
        self.assertEqual(result, [{"t": 1}, {"t": 3}])

    def test_remove_first_and_last(self):
        from qwen3_tts.interface.ui.shared import remove_history_row

        rows = [{"t": 1}, {"t": 2}, {"t": 3}]
        self.assertEqual(remove_history_row(rows, 0), [{"t": 2}, {"t": 3}])
        self.assertEqual(remove_history_row(rows, 2), [{"t": 1}, {"t": 2}])

    def test_out_of_range_high_returns_copy_unchanged(self):
        from qwen3_tts.interface.ui.shared import remove_history_row

        rows = [{"t": 1}, {"t": 2}]
        result = remove_history_row(rows, 99)
        self.assertEqual(result, [{"t": 1}, {"t": 2}])

    def test_negative_index_returns_copy_unchanged(self):
        from qwen3_tts.interface.ui.shared import remove_history_row

        rows = [{"t": 1}, {"t": 2}]
        result = remove_history_row(rows, -1)
        self.assertEqual(result, [{"t": 1}, {"t": 2}])

    def test_input_list_not_mutated(self):
        from qwen3_tts.interface.ui.shared import remove_history_row

        rows = [{"t": 1}, {"t": 2}, {"t": 3}]
        remove_history_row(rows, 1)
        self.assertEqual(rows, [{"t": 1}, {"t": 2}, {"t": 3}])

    def test_non_list_returns_empty(self):
        from qwen3_tts.interface.ui.shared import remove_history_row

        self.assertEqual(remove_history_row(None, 0), [])


class TestClearHistory(unittest.TestCase):
    """clear_history returns a fresh empty list regardless of input."""

    def test_clears_populated_list(self):
        from qwen3_tts.interface.ui.shared import clear_history

        self.assertEqual(clear_history([{"t": 1}, {"t": 2}]), [])

    def test_clears_empty_list(self):
        from qwen3_tts.interface.ui.shared import clear_history

        self.assertEqual(clear_history([]), [])

    def test_clears_none(self):
        from qwen3_tts.interface.ui.shared import clear_history

        self.assertEqual(clear_history(None), [])


class TestGetHistoryDataRemoveColumn(unittest.TestCase):
    """get_history_data rows carry a 6th 'Remove' (✕) and 7th 'Download' (⭳) column."""

    def test_row_has_seven_columns_ending_in_remove_and_download_glyphs(self):
        from qwen3_tts.interface.ui.shared import get_history_data

        history = [
            {
                "timestamp": 1710000000,
                "mode": "Clone",
                "text": "Hello",
                "path": "/tmp/a.wav",
                "chunks": 2,
                "seed": 42,
            }
        ]
        rows = get_history_data(history)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), 7)
        self.assertEqual(rows[0][5], "✕")
        self.assertEqual(rows[0][6], "⭳")
        # Earlier columns are unchanged.
        self.assertEqual(rows[0][2], "Hello")
        self.assertEqual(rows[0][3], "42")
        self.assertEqual(rows[0][4], 2)


class TestHistoryLock(unittest.TestCase):
    """history_lock is a usable threading lock shared across modules."""

    def test_is_usable_context_manager(self):
        from qwen3_tts.interface.ui.shared import history_lock

        self.assertTrue(hasattr(history_lock, "acquire"))
        self.assertTrue(hasattr(history_lock, "release"))
        with history_lock:
            pass

    def test_is_a_real_lock_instance(self):
        from qwen3_tts.interface.ui.shared import history_lock

        # threading.Lock() returns a _thread.lock object; comparing against a
        # fresh one isn't meaningful, so assert the type name instead.
        self.assertEqual(type(history_lock).__name__, "lock")


if __name__ == "__main__":
    unittest.main()
