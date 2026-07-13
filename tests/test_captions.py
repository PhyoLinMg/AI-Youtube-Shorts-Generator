import os
import shutil
import subprocess

import pytest

from shorts_generator.captions import CaptionError, _chunk_segments, _probe_resolution, _write_ass, burn_captions


def test_chunk_segments_splits_by_word_count_and_time_share():
    segments = [
        {
            "start": 10.0,
            "end": 12.0,
            "text": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen",
        }
    ]

    chunks = _chunk_segments(segments, clip_start=0.0, clip_end=100.0, max_words=7)

    assert chunks == [
        {"start": 10.0, "end": 11.0, "text": "one two three four five six seven"},
        {"start": 11.0, "end": 12.0, "text": "eight nine ten eleven twelve thirteen fourteen"},
    ]


def test_chunk_segments_drops_segments_outside_window():
    segments = [{"start": 5.0, "end": 6.0, "text": "not in the clip"}]

    chunks = _chunk_segments(segments, clip_start=10.0, clip_end=20.0)

    assert chunks == []


def test_chunk_segments_clips_and_shifts_straddling_segment():
    segments = [{"start": 8.0, "end": 12.0, "text": "alpha beta gamma delta"}]

    chunks = _chunk_segments(segments, clip_start=10.0, clip_end=20.0, max_words=7)

    assert chunks == [{"start": 0.0, "end": 2.0, "text": "alpha beta gamma delta"}]


def test_write_ass_contains_resolution_and_fade_tag(tmp_path):
    chunks = [
        {"start": 0.0, "end": 1.0, "text": "hello world"},
        {"start": 1.0, "end": 2.5, "text": "second line here"},
    ]
    ass_path = str(tmp_path / "captions.ass")

    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3)

    content = open(ass_path, encoding="utf-8").read()
    assert "PlayResX: 608" in content
    assert "PlayResY: 1080" in content
    assert content.count("Dialogue:") == 2
    assert "\\fad(300,0)" in content
    assert "hello world" in content
    assert "second line here" in content


def test_write_ass_strips_braces_from_text(tmp_path):
    chunks = [{"start": 0.0, "end": 1.0, "text": "watch this {glitch} moment"}]
    ass_path = str(tmp_path / "captions.ass")

    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3)

    content = open(ass_path, encoding="utf-8").read()
    assert "watch this glitch moment" in content


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory):
    """A tiny 3s 9:16-ish clip generated once for this test module."""
    tmp_dir = tmp_path_factory.mktemp("captions_src")
    path = str(tmp_dir / "clip.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x568:rate=24:duration=3",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            path,
        ],
        check=True,
    )
    return path


def test_probe_resolution_reads_dimensions(synthetic_clip):
    assert _probe_resolution(synthetic_clip) == (320, 568)


def test_burn_captions_produces_output_file(tmp_path, synthetic_clip):
    out_path = str(tmp_path / "burned.mp4")
    segments = [{"start": 0.0, "end": 3.0, "text": "hello there this is a caption test"}]

    result = burn_captions(
        synthetic_clip, segments, clip_start=0.0, clip_end=3.0, out_path=out_path, fade_seconds=0.3
    )

    assert result == out_path
    assert os.path.exists(out_path)
    assert _probe_resolution(out_path) == (320, 568)
    assert not os.path.exists(out_path + ".ass")


def test_burn_captions_raises_when_no_transcript_overlaps(tmp_path, synthetic_clip):
    out_path = str(tmp_path / "burned.mp4")
    segments = [{"start": 100.0, "end": 103.0, "text": "way outside the clip"}]

    with pytest.raises(CaptionError):
        burn_captions(synthetic_clip, segments, clip_start=0.0, clip_end=3.0, out_path=out_path)


def test_burn_captions_raises_caption_error_when_ffmpeg_missing(tmp_path, synthetic_clip, monkeypatch):
    """ffprobe succeeds (so _probe_resolution passes) but ffmpeg itself can't be
    found on PATH — burn_captions must still raise CaptionError, not let the
    raw FileNotFoundError escape past its documented contract."""
    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    real_ffprobe = shutil.which("ffprobe")
    (fake_bin / "ffprobe").symlink_to(real_ffprobe)

    monkeypatch.setenv("PATH", str(fake_bin))

    out_path = str(tmp_path / "burned.mp4")
    segments = [{"start": 0.0, "end": 3.0, "text": "hello there this is a caption test"}]

    with pytest.raises(CaptionError):
        burn_captions(synthetic_clip, segments, clip_start=0.0, clip_end=3.0, out_path=out_path, fade_seconds=0.3)
