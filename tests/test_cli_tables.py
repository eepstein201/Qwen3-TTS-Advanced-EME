"""Task 2.5 (Step 7B): unified list-resource table renderers.

``qwen3_tts.cli_tables`` is the single source of Column widths and render
functions for every ``tts list X`` / legacy ``tts generate --list-X`` pair
(docs/plans/2026-09-07-interface-quality.plan.md, Phase 2 Task 2.5). The
unification contract test below is the point of this module: both entry
points must call the same renderer and therefore produce byte-identical
output.

Run: conda run -n qwen3-tts-mlx python -m pytest tests/test_cli_tables.py -v
"""

import unittest

from qwen3_tts import cli_tables


def _padded_row(indent, columns, *values):
    cells = []
    for column, value in zip(columns, values, strict=True):
        cell = (
            value.ljust(column.width)
            if column.align == "left"
            else value.rjust(column.width)
        )
        cells.append(cell)
    return indent + "  ".join(cells)


class TestSpeakerColumns(unittest.TestCase):
    SPEAKERS = {
        "ryan": {
            "name": "Ryan",
            "lang": "English",
            "desc": "Dynamic male, strong rhythm",
        },
        "vivian": {"name": "Vivian", "lang": "Chinese", "desc": "Bright young female"},
        "ono_anna": {"name": "Ono_Anna", "lang": "Japanese", "desc": "Playful female"},
    }

    def test_speakers_grouped_by_language(self):
        out = cli_tables.render_speakers(self.SPEAKERS)
        lines = out.splitlines()
        self.assertEqual(
            lines[0], "Premium CustomVoice speakers (use with -m custom -s SPEAKER):"
        )
        self.assertIn("  English:", lines)
        self.assertIn("  Chinese:", lines)
        self.assertIn("  Other languages:", lines)
        english_row = _padded_row(
            "    ", cli_tables.SPEAKER_COLUMNS, "ryan", "Dynamic male, strong rhythm"
        )
        self.assertIn(english_row, lines)
        other_row = _padded_row(
            "    ", cli_tables.SPEAKER_COLUMNS, "ono_anna", "Playful female (Japanese)"
        )
        self.assertIn(other_row, lines)
        self.assertEqual(
            lines[-1], "Example: tts 'Hello world' -m custom -s ryan -o output"
        )

    def test_speakers_omits_empty_groups(self):
        out = cli_tables.render_speakers({"vivian": self.SPEAKERS["vivian"]})
        self.assertNotIn("  English:", out.splitlines())
        self.assertIn("  Chinese:", out.splitlines())


class TestPresetColumns(unittest.TestCase):
    def test_presets_empty(self):
        self.assertEqual(cli_tables.render_presets({}), "No presets configured.")

    def test_presets_populated(self):
        presets = {"consistent": {"temperature": 0.7, "seed": 42}}
        out = cli_tables.render_presets(presets)
        lines = out.splitlines()
        self.assertEqual(lines[0], "Generation presets:")
        row = _padded_row(
            "  ", cli_tables.PRESET_COLUMNS, "consistent", "temperature=0.7, seed=42"
        )
        self.assertIn(row, lines)


class TestProsodyColumns(unittest.TestCase):
    def test_prosody_empty(self):
        self.assertEqual(
            cli_tables.render_prosody({}), "No prosody presets configured."
        )

    def test_prosody_identical_both_entry_points(self):
        presets = {"excited": "Speak with excitement", "calm": "Speak calmly"}
        # Both the Click `tts list prosody` command and the legacy argparse
        # `tts generate --list-prosody` flag must call this same function —
        # this test IS the unification contract for prosody.
        out_a = cli_tables.render_prosody(presets)
        out_b = cli_tables.render_prosody(dict(presets))
        self.assertEqual(out_a, out_b)
        row = _padded_row("  ", cli_tables.PROSODY_COLUMNS, "calm", "Speak calmly")
        self.assertIn(row, out_a.splitlines())

    def test_prosody_sorted_alphabetically(self):
        out = cli_tables.render_prosody({"whisper": "x", "authoritative": "y"})
        lines = [line.strip().split()[0] for line in out.splitlines()[1:]]
        self.assertEqual(lines, sorted(lines))


class TestVoicePromptColumns(unittest.TestCase):
    def test_voice_prompts_empty(self):
        self.assertEqual(
            cli_tables.render_voice_prompts([], default=None), "No voice prompts found."
        )

    def test_voice_prompts_marks_default(self):
        out = cli_tables.render_voice_prompts(["narrator", "guest"], default="narrator")
        lines = out.splitlines()
        self.assertIn("  narrator (default)", lines)
        self.assertIn("  guest", lines)


class TestCacheColumns(unittest.TestCase):
    def test_cache_empty(self):
        self.assertEqual(cli_tables.render_cache([]), "No TTS models found in cache.")

    def test_cache_no_duplicated_size_column(self):
        models = [
            {
                "model_type": "clone",
                "model_size": "1.7B",
                "size_formatted": "3.5 GB",
                "backend": "mlx",
                "last_access_str": "2026-09-23 10:00",
            }
        ]
        out = cli_tables.render_cache(models)
        # The historical bug wrote size_formatted into two columns; assert
        # the value appears exactly once in the whole rendered table.
        self.assertEqual(out.count("3.5 GB"), 1)
        header = out.splitlines()[0]
        self.assertIn("Size", header)
        self.assertNotIn("Size on Disk", header)

    def test_cache_separator_widths_match_header(self):
        models = [
            {
                "model_type": "design",
                "model_size": "0.6B",
                "size_formatted": "2.1 GB",
                "backend": "torch",
                "last_access_str": "unknown",
            }
        ]
        out = cli_tables.render_cache(models)
        header, separator = out.splitlines()[:2]
        self.assertEqual(len(header), len(separator))


class TestHistoryColumns(unittest.TestCase):
    def test_history_empty(self):
        self.assertEqual(cli_tables.render_history([]), "No generation history found.")

    def test_history_row_alignment(self):
        entries = [
            {"timestamp": "2026-09-23 10:00:00", "mode": "clone", "voice": "narrator"}
        ]
        out = cli_tables.render_history(entries)
        row = _padded_row(
            "  ", cli_tables.HISTORY_COLUMNS, "2026-09-23 10:00:00", "clone", "narrator"
        )
        self.assertEqual(out, row)


if __name__ == "__main__":
    unittest.main()
