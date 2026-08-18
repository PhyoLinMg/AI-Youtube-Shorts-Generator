import json
import os
import subprocess

import pytest

from shorts_generator.captions import CaptionError
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
    monkeypatch.setattr(thread_source_module, "_probe_local_duration", lambda path: 100.0)
    monkeypatch.setattr(
        thread_source_module, "transcribe_local",
        lambda path, model_size=None: {"duration": 7.0, "segments": [
            {"start": 0.0, "end": 7.0, "text": "hello world", "words": [{"start": 0.0, "end": 1.0, "word": "hello"}]}
        ]},
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


def test_acquire_clip_raises_on_duration_mismatch_when_full_source_present(tmp_path, monkeypatch):
    """The fast path (full_source.mp4 already on disk, whether pre-existing
    or downloaded ahead of time by a caller) must apply the same
    duration-mismatch guard as the slow re-download path -- see the module
    docstring's incident: a mismatched source must never be captioned."""
    run_dir = tmp_path / "episode"
    run_dir.mkdir()
    (run_dir / "full_source.mp4").write_bytes(b"fake video bytes")
    (run_dir / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))

    monkeypatch.setattr(thread_source_module, "_probe_local_duration", lambda path: 250.0)

    def _fail_if_called(*a, **k):
        pytest.fail("cropping should not run on a duration mismatch")
    monkeypatch.setattr(thread_source_module, "crop_clip_local", _fail_if_called)

    with pytest.raises(SourceMismatchError):
        acquire_clip(
            str(run_dir), "https://example.com/video", cached_duration=100.0,
            start_time=1.0, end_time=8.0, out_path=str(tmp_path / "clip.mp4"),
        )


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


def test_acquire_clip_keeps_uncaptioned_clip_when_burn_captions_fails(tmp_path, monkeypatch):
    """A CaptionError must not propagate uncaught and must not leave a
    stray .captioned.mp4 -- the crop_clip_local output at out_path (already
    written before burn_captions runs) is left in place as a valid,
    playable, if uncaptioned, clip."""
    run_dir = tmp_path / "episode"
    run_dir.mkdir()
    (run_dir / "full_source.mp4").write_bytes(b"fake video bytes")
    (run_dir / "full_source.json").write_text(json.dumps({
        "duration": 100.0,
        "segments": [{"start": 0.0, "end": 10.0, "text": "hello world", "words": []}],
    }))

    out_path = str(tmp_path / "clip.mp4")

    def _fake_crop(source_path, start, end, aspect_ratio, out_path, **kwargs):
        with open(out_path, "wb") as f:
            f.write(b"cropped but uncaptioned")

    def _fail_captions(src, segs, start, end, out, **kwargs):
        # Simulate the real burn path: it may write a partial file before
        # raising.
        with open(out, "wb") as f:
            f.write(b"partial")
        raise CaptionError("no transcript overlaps clip window")

    monkeypatch.setattr(thread_source_module, "crop_clip_local", _fake_crop)
    monkeypatch.setattr(thread_source_module, "burn_captions", _fail_captions)
    monkeypatch.setattr(thread_source_module, "_probe_local_duration", lambda path: 100.0)
    monkeypatch.setattr(
        thread_source_module, "transcribe_local",
        lambda path, model_size=None: {"duration": 7.0, "segments": [
            {"start": 0.0, "end": 7.0, "text": "hello world", "words": [{"start": 0.0, "end": 1.0, "word": "hello"}]}
        ]},
    )

    result = acquire_clip(
        str(run_dir), "https://example.com/video", cached_duration=100.0,
        start_time=1.0, end_time=8.0, out_path=out_path,
    )

    assert result["clip_path"] == out_path
    assert "captions_error" in result
    assert os.path.exists(out_path)
    with open(out_path, "rb") as f:
        assert f.read() == b"cropped but uncaptioned"
    # the partial captioned file must be cleaned up, not left as debris
    assert not os.path.exists(out_path + ".captioned.mp4")


def test_probe_source_duration_raises_clear_error_on_unparseable_output(monkeypatch):
    class _FakeResult:
        stdout = "NA\n"

    monkeypatch.setattr(
        thread_source_module.subprocess, "run",
        lambda *a, **k: _FakeResult(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        thread_source_module._probe_source_duration("https://example.com/video")
    # must not be a bare ValueError/IndexError -- should carry context
    assert "https://example.com/video" in str(exc_info.value)


def test_probe_source_duration_raises_clear_error_on_empty_output(monkeypatch):
    class _FakeResult:
        stdout = ""

    monkeypatch.setattr(
        thread_source_module.subprocess, "run",
        lambda *a, **k: _FakeResult(),
    )

    with pytest.raises(RuntimeError):
        thread_source_module._probe_source_duration("https://example.com/video")


def test_download_padded_section_cleans_up_webm_when_ffmpeg_fails(tmp_path, monkeypatch):
    out_path = str(tmp_path / "padded.mp4")
    # yt-dlp's own re-encode (--force-keyframes-at-cuts) can land the
    # download under any of a few extensions -- see thread_source.py's
    # download_stem + {.mkv,.webm,.mp4} discovery loop.
    downloaded_path = out_path + ".download.webm"
    calls = {"n": 0}

    def _fake_run(cmd, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # simulate yt-dlp actually producing the intermediate download
            with open(downloaded_path, "wb") as f:
                f.write(b"fake webm bytes")
            return subprocess.CompletedProcess(cmd, 0)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(thread_source_module.subprocess, "run", _fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        thread_source_module._download_padded_section(
            "https://example.com/video", 10.0, 20.0, out_path,
        )

    assert not os.path.exists(downloaded_path)
