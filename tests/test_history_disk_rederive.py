#!/usr/bin/env python3
"""The history table re-derives from disk, so event delivery order can't show stale rows.

refresh_history_from_disk must derive history_df rows from the on-disk sidecars,
ignoring the in-memory history_list it is handed — that is what closes the
demo.load() vs generation-refresh delivery race (test_13).
"""


def test_refresh_reflects_disk_not_a_stale_in_memory_list(tmp_path, monkeypatch):
    from qwen3_tts.interface.ui import shared

    # load_history_from_disk enforces home containment; make tmp_path the home
    # dir so the Automated Output subdir passes that guard.
    monkeypatch.setenv("HOME", str(tmp_path))

    automated = tmp_path / "Automated Output"
    automated.mkdir()
    wav = automated / "voice_ui_fresh.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 40)
    (automated / "voice_ui_fresh.json").write_text(
        '{"timestamp": 999.0, "mode": "clone", "text": "fresh", "seed": 42}'
    )

    # An entry that exists only in memory (its file is gone) must NOT survive a
    # disk re-derive — this is exactly the stale-row render race.
    stale = [{"path": "/gone.wav", "seed": 111, "timestamp": 1.0, "text": "stale"}]
    monkeypatch.setattr(
        shared, "resolve_automated_output_dir", lambda config: str(automated)
    )
    rows = shared.refresh_history_from_disk(stale, {})

    seeds = [r[3] for r in rows]
    assert "42" in seeds, "fresh on-disk entry must appear"
    assert "111" not in seeds, "stale in-memory-only entry must not"
