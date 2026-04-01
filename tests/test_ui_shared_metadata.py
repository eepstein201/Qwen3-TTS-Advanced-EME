import json
import os
import time
import pytest

from qwen3_tts.interface.ui.shared import (
    save_generation_metadata,
    load_history_from_disk,
    MAX_HISTORY_SIZE,
)


def test_save_generation_metadata_creates_json(tmp_path):
    wav_path = str(tmp_path / "voice_ui_abc12345.wav")
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


def test_save_generation_metadata_immutable(tmp_path):
    wav_path = str(tmp_path / "voice_ui_def67890.wav")
    open(wav_path, "w").close()
    metadata = {"mode": "design", "seed": 99}
    original = dict(metadata)
    save_generation_metadata(wav_path, metadata)
    assert metadata == original


def test_load_history_from_disk_reads_json(tmp_path):
    for i in range(3):
        wav = tmp_path / f"voice_ui_{i:08d}.wav"
        wav.write_text("")
        meta = {
            "timestamp": 1000.0 + i,
            "mode": "clone",
            "text": f"Text {i}",
            "seed": i * 10,
        }
        (tmp_path / f"voice_ui_{i:08d}.json").write_text(json.dumps(meta))
    history = load_history_from_disk(str(tmp_path))
    assert len(history) == 3
    assert history[0]["timestamp"] == 1002.0
    assert history[0]["seed"] == 20


def test_load_history_from_disk_caps_at_max(tmp_path):
    for i in range(MAX_HISTORY_SIZE + 5):
        wav = tmp_path / f"voice_ui_{i:08d}.wav"
        wav.write_text("")
        meta = {"timestamp": float(i), "mode": "clone", "text": f"T{i}"}
        (tmp_path / f"voice_ui_{i:08d}.json").write_text(json.dumps(meta))
    history = load_history_from_disk(str(tmp_path))
    assert len(history) == MAX_HISTORY_SIZE


def test_load_history_from_disk_skips_orphan_json(tmp_path):
    meta = {"timestamp": 1.0, "mode": "clone", "text": "orphan"}
    (tmp_path / "voice_ui_orphan.json").write_text(json.dumps(meta))
    history = load_history_from_disk(str(tmp_path))
    assert len(history) == 0
