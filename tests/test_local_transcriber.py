import json
from pathlib import Path
from types import SimpleNamespace

import shorts_generator.local.transcriber as tr


class _FakeWord:
    def __init__(self, start, end, word):
        self.start = start
        self.end = end
        self.word = word


class _FakeSegment:
    def __init__(self, start, end, text, words):
        self.start = start
        self.end = end
        self.text = text
        self.words = words


class _FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel; records the kwargs it was
    called with so the test can assert word_timestamps=True was requested."""

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, **kwargs):
        _FakeWhisperModel.last_transcribe_kwargs = kwargs
        segments = [
            _FakeSegment(
                0.0, 2.0, "the quick fox",
                [
                    _FakeWord(0.0, 0.5, "the"),
                    _FakeWord(0.5, 1.2, "quick"),
                    _FakeWord(1.2, 2.0, "fox"),
                ],
            )
        ]
        info = SimpleNamespace(duration=2.0)
        return iter(segments), info


def test_json_cache_roundtrip_preserves_words(tmp_path):
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


def test_cache_path_is_json(tmp_path):
    p = tr._transcript_cache_path(str(tmp_path / "video.mp4"))
    assert p.name == "video.json"


def test_transcribe_local_requests_word_timestamps_and_collects_words(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")

    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWhisperModel)

    result = tr.transcribe_local(str(media))

    assert _FakeWhisperModel.last_transcribe_kwargs["word_timestamps"] is True
    assert result["segments"][0]["words"] == [
        {"start": 0.0, "end": 0.5, "word": "the"},
        {"start": 0.5, "end": 1.2, "word": "quick"},
        {"start": 1.2, "end": 2.0, "word": "fox"},
    ]

    # and it's actually cached to disk as json, words included
    cache_path = tr._transcript_cache_path(str(media))
    cached = json.loads(cache_path.read_text())
    assert cached["segments"][0]["words"][2]["word"] == "fox"


def test_cache_path_follows_media_directory_not_global_default(tmp_path):
    nested = tmp_path / "run_folder"
    nested.mkdir()
    media = nested / "full_source.mp4"
    media.write_bytes(b"x")

    cache_path = tr._transcript_cache_path(str(media))

    assert cache_path.parent == nested.resolve()
    assert cache_path.name == "full_source.json"
