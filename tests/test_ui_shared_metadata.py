import json
import os
import time

import pytest

from qwen3_tts.interface.ui.shared import (
    MAX_HISTORY_SIZE,
    add_to_history,
    get_history_data,
    load_history_from_disk,
    save_generation_metadata,
)


@pytest.fixture()
def home_tmp(tmp_path, monkeypatch):
    """Make tmp_path appear as the home directory for the home-dir guard."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def test_save_generation_metadata_creates_json(home_tmp):
    wav_path = str(home_tmp / "voice_ui_abc12345.wav")
    open(wav_path, "w").close()
    metadata = {
        "timestamp": time.time(),
        "mode": "clone",
        "text": "Hello world",
        "seed": 42,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
    }
    save_generation_metadata(wav_path, metadata)
    json_path = wav_path.replace(".wav", ".json")
    assert os.path.exists(json_path)
    with open(json_path) as f:
        saved = json.load(f)
    assert saved["seed"] == 42
    assert saved["mode"] == "clone"


def test_save_generation_metadata_immutable(home_tmp):
    wav_path = str(home_tmp / "voice_ui_def67890.wav")
    open(wav_path, "w").close()
    metadata = {"mode": "design", "seed": 99}
    original = dict(metadata)
    save_generation_metadata(wav_path, metadata)
    assert metadata == original


def test_load_history_from_disk_reads_json(home_tmp):
    for i in range(3):
        wav = home_tmp / f"voice_ui_{i:08d}.wav"
        wav.write_text("")
        meta = {
            "timestamp": 1000.0 + i,
            "mode": "clone",
            "text": f"Text {i}",
            "seed": i * 10,
        }
        (home_tmp / f"voice_ui_{i:08d}.json").write_text(json.dumps(meta))
    history = load_history_from_disk(str(home_tmp))
    assert len(history) == 3
    assert history[0]["timestamp"] == 1002.0
    assert history[0]["seed"] == 20


def test_load_history_from_disk_caps_at_max(home_tmp):
    for i in range(MAX_HISTORY_SIZE + 5):
        wav = home_tmp / f"voice_ui_{i:08d}.wav"
        wav.write_text("")
        meta = {"timestamp": float(i), "mode": "clone", "text": f"T{i}"}
        (home_tmp / f"voice_ui_{i:08d}.json").write_text(json.dumps(meta))
    history = load_history_from_disk(str(home_tmp))
    assert len(history) == MAX_HISTORY_SIZE


def test_load_history_from_disk_skips_orphan_json(home_tmp):
    meta = {"timestamp": 1.0, "mode": "clone", "text": "orphan"}
    (home_tmp / "voice_ui_orphan.json").write_text(json.dumps(meta))
    history = load_history_from_disk(str(home_tmp))
    assert len(history) == 0


def test_add_to_history_includes_seed():
    history = add_to_history([], "clone", "Hello", "/tmp/test.wav", 1, seed=42)
    assert history[0]["seed"] == 42


def test_add_to_history_seed_none_when_omitted():
    history = add_to_history([], "clone", "Hello", "/tmp/test.wav", 1)
    assert history[0]["seed"] is None


def test_get_history_data_includes_seed_column():
    import time
    history = [{"timestamp": time.time(), "mode": "Clone", "text": "Hi",
                "path": "/tmp/x.wav", "chunks": 1, "seed": 42}]
    rows = get_history_data(history)
    assert len(rows[0]) == 6  # Time, Mode, Text, Seed, Chunks, Remove
    assert rows[0][3] == "42"


def test_get_history_data_seed_none_displays_dash():
    import time
    history = [{"timestamp": time.time(), "mode": "Clone", "text": "Hi",
                "path": "/tmp/x.wav", "chunks": 1, "seed": None}]
    rows = get_history_data(history)
    assert rows[0][3] == "-"
