import json
import os
import subprocess

import pytest

from shorts_generator.local import thread_source as thread_source_module
from shorts_generator.local.thread_source import SourceMismatchError, acquire_clip


def test_find_word_start_returns_first_word_at_or_after_min_time():
    segments = [
        {"start": 0.0, "end": 4.0, "words": [
            {"start": 0.0, "end": 0.5, "word": "hello"},
            {"start": 1.5, "end": 2.0, "word": "world"},
        ]},
        {"start": 4.0, "end": 6.0, "words": [
            {"start": 4.2, "end": 4.5, "word": "next"},
        ]},
    ]
    assert thread_source_module._find_word_start(segments, 1.0) == 1.5
    assert thread_source_module._find_word_start(segments, 3.0) == 4.2


def test_find_word_start_falls_back_to_min_time_when_no_word_found():
    segments = [{"start": 0.0, "end": 1.0, "words": [{"start": 0.0, "end": 0.5, "word": "hi"}]}]
    assert thread_source_module._find_word_start(segments, 10.0) == 10.0


def test_acquire_clip_cuts_directly_when_full_source_present(tmp_path, monkeypatch):
    run_dir = tmp_path / "episode"
    run_dir.mkdir()
    (run_dir / "full_source.mp4").write_bytes(b"fake video bytes")
    (run_dir / "full_source.json").write_text(json.dumps({
        "duration": 100.0,
        "segments": [{"start": 0.0, "end": 10.0, "text": "hello world", "words": []}],
    }))

    calls = {}
    monkeypatch.setattr(
        thread_source_module, "crop_clip_local",
        lambda *a, **k: calls.setdefault("crop_args", (a, k)),
    )
    monkeypatch.setattr(
        thread_source_module, "burn_captions",
        lambda src, segs, start, end, out, **k: open(out, "wb").write(b"captioned"),
    )

    def _fail(*a, **k):
        pytest.fail("network re-acquisition should not run when full_source.mp4 is present")
    monkeypatch.setattr(thread_source_module, "_probe_source_duration", _fail)
    monkeypatch.setattr(thread_source_module, "_download_padded_section", _fail)

    out_path = str(tmp_path / "clip.mp4")
    result = acquire_clip(
        str(run_dir), "https://example.com/video", cached_duration=100.0,
        start_time=1.0, end_time=8.0, out_path=out_path,
    )

    assert result == {"clip_path": out_path}
    assert calls["crop_args"][0][:3] == (str(run_dir / "full_source.mp4"), 1.0, 8.0)


def test_acquire_clip_raises_on_duration_mismatch_before_downloading(tmp_path, monkeypatch):
    run_dir = tmp_path / "episode"
    run_dir.mkdir()
    (run_dir / "full_source.json").write_text(json.dumps({"duration": 5778.0, "segments": []}))
    # full_source.mp4 deliberately absent -- forces the re-acquire fallback path

    monkeypatch.setattr(thread_source_module, "_probe_source_duration", lambda url: 6530.0)

    def _fail_if_called(*a, **k):
        pytest.fail("_download_padded_section should not be called on a duration mismatch")
    monkeypatch.setattr(thread_source_module, "_download_padded_section", _fail_if_called)

    with pytest.raises(SourceMismatchError):
        acquire_clip(
            str(run_dir), "https://example.com/video", cached_duration=5778.0,
            start_time=100.0, end_time=120.0, out_path=str(tmp_path / "out.mp4"),
        )


def test_acquire_clip_proceeds_when_duration_matches_within_tolerance(tmp_path, monkeypatch):
    run_dir = tmp_path / "episode"
    run_dir.mkdir()
    (run_dir / "full_source.json").write_text(json.dumps({"duration": 5778.0, "segments": []}))

    monkeypatch.setattr(thread_source_module, "_probe_source_duration", lambda url: 5778.9)

    padded_path_holder = {}
    def _fake_download(source_url, start_time, end_time, out_path):
        padded_path_holder["out_path"] = out_path
        with open(out_path, "wb") as f:
            f.write(b"fake padded video")
    monkeypatch.setattr(thread_source_module, "_download_padded_section", _fake_download)
    monkeypatch.setattr(
        thread_source_module, "transcribe_local",
        lambda path, model_size=None: {"duration": 30.0, "segments": [
            {"start": 0.0, "end": 5.0, "text": "hi", "words": [{"start": 1.5, "end": 2.0, "word": "hi"}]}
        ]},
    )
    monkeypatch.setattr(thread_source_module, "crop_clip_local", lambda *a, **k: None)
    monkeypatch.setattr(
        thread_source_module, "burn_captions",
        lambda src, segs, start, end, out, **k: open(out, "wb").write(b"captioned"),
    )

    result = acquire_clip(
        str(run_dir), "https://example.com/video", cached_duration=5778.0,
        start_time=956.2, end_time=977.3, out_path=str(tmp_path / "out.mp4"),
    )

    assert result == {"clip_path": str(tmp_path / "out.mp4")}
    assert padded_path_holder["out_path"]
