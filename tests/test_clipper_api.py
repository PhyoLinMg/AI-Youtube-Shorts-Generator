import os
import shutil
import subprocess

import pytest
import requests

from shorts_generator import clipper


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory):
    """Stands in for the mp4 MuAPI would host at the returned clip URL."""
    tmp_dir = tmp_path_factory.mktemp("hosted")
    path = str(tmp_dir / "hosted.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=608x1080:rate=24:duration=4",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            path,
        ],
        check=True,
    )
    return path


def _highlight():
    return {"title": "Test Clip", "start_time": 0.0, "end_time": 3.0, "score": 90}


def _segments():
    return [{"start": 0.0, "end": 3.0, "text": "hello there this is a caption test line"}]


def test_captions_burned_in_by_default(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    out_dir = str(tmp_path / "out")
    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=out_dir,
    )

    assert results[0]["hosted_clip_url"] == "https://hosted.example/short_1.mp4"
    assert os.path.exists(results[0]["clip_url"])
    assert results[0]["clip_url"] != results[0]["hosted_clip_url"]


def test_captions_disabled_keeps_hosted_url(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")

    def _fail_if_called(*a, **k):
        raise AssertionError("_download_to should not be called when captions=False")

    monkeypatch.setattr(clipper, "_download_to", _fail_if_called)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        captions=False,
        out_dir=str(tmp_path / "out"),
    )

    assert results[0]["clip_url"] == "https://hosted.example/short_1.mp4"
    assert "hosted_clip_url" not in results[0]


def test_download_failure_falls_back_to_hosted_url(tmp_path, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")

    def _raise(*a, **k):
        raise requests.RequestException("network down")

    monkeypatch.setattr(clipper, "_download_to", _raise)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
    )

    assert results[0]["clip_url"] == "https://hosted.example/short_1.mp4"
    assert results[0]["captions_error"] == "network down"


def test_word_highlight_flag_forwarded_to_burn(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )
    captured = {}

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        shutil.copyfile(args[0], args[4])
        return args[4]

    monkeypatch.setattr(clipper, "burn_captions", _spy)

    clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
        word_highlight=False,
    )

    assert captured["word_highlight"] is False


def test_output_filename_uses_short_dash_prefix(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
    )

    assert os.path.basename(results[0]["clip_url"]) == "Short-01.mp4"
