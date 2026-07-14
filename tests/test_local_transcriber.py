import json
from pathlib import Path

import shorts_generator.local.transcriber as tr


def test_json_cache_roundtrip_preserves_words(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "LOCAL_OUTPUT_DIR", str(tmp_path))
    transcript = {
        "duration": 4.0,
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "hello world",
             "words": [
                 {"start": 0.0, "end": 1.0, "word": "hello"},
                 {"start": 1.0, "end": 2.0, "word": "world"},
             ]},
        ],
    }
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    cache_path = tr._write_json_cache(str(media), transcript)
    assert cache_path.suffix == ".json"

    loaded = tr._load_json_cache(cache_path)
    assert loaded["segments"][0]["words"][1]["word"] == "world"
    assert loaded["duration"] == 4.0


def test_cache_path_is_json(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "LOCAL_OUTPUT_DIR", str(tmp_path))
    p = tr._transcript_cache_path("/some/video.mp4")
    assert p.name == "video.json"
