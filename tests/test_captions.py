import os
import shutil
import subprocess

import pytest

from shorts_generator.captions import (
    CaptionError,
    _HIGHLIGHT_OPEN,
    _chunk_segments,
    _format_ass_timestamp,
    _probe_resolution,
    _write_ass,
    burn_captions,
)


def test_chunk_segments_splits_by_word_count_and_time_share():
    segments = [
        {
            "start": 10.0,
            "end": 12.0,
            "text": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen",
        }
    ]

    chunks = _chunk_segments(segments, clip_start=0.0, clip_end=100.0, max_words=7)

    assert len(chunks) == 2
    # First chunk
    assert chunks[0]["start"] == 10.0
    assert chunks[0]["end"] == 11.0
    assert chunks[0]["text"] == "one two three four five six seven"
    assert "words" in chunks[0]
    assert [w["text"] for w in chunks[0]["words"]] == ["one", "two", "three", "four", "five", "six", "seven"]
    assert chunks[0]["words"][0]["start"] == 10.0
    assert chunks[0]["words"][-1]["end"] == 11.0

    # Second chunk
    assert chunks[1]["start"] == 11.0
    assert chunks[1]["end"] == 12.0
    assert chunks[1]["text"] == "eight nine ten eleven twelve thirteen fourteen"
    assert "words" in chunks[1]
    assert [w["text"] for w in chunks[1]["words"]] == ["eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen"]
    assert chunks[1]["words"][0]["start"] == 11.0
    assert chunks[1]["words"][-1]["end"] == 12.0


def test_chunk_segments_drops_segments_outside_window():
    segments = [{"start": 5.0, "end": 6.0, "text": "not in the clip"}]

    chunks = _chunk_segments(segments, clip_start=10.0, clip_end=20.0)

    assert chunks == []


def test_chunk_segments_clips_and_shifts_straddling_segment():
    segments = [{"start": 8.0, "end": 12.0, "text": "alpha beta gamma delta"}]

    chunks = _chunk_segments(segments, clip_start=10.0, clip_end=20.0, max_words=7)

    assert len(chunks) == 1
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 2.0
    assert chunks[0]["text"] == "alpha beta gamma delta"
    assert "words" in chunks[0]
    assert [w["text"] for w in chunks[0]["words"]] == ["alpha", "beta", "gamma", "delta"]
    assert chunks[0]["words"][0]["start"] == 0.0
    assert chunks[0]["words"][-1]["end"] == 2.0


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


def test_write_ass_emits_one_dialogue_per_word_with_highlight(tmp_path):
    chunks = [{
        "start": 0.0, "end": 2.0, "text": "alpha beta",
        "words": [
            {"start": 0.0, "end": 1.0, "text": "alpha"},
            {"start": 1.0, "end": 2.0, "text": "beta"},
        ],
    }]
    ass_path = str(tmp_path / "c.ass")
    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3)
    content = open(ass_path, encoding="utf-8").read()
    assert content.count("Dialogue:") == 2          # one per word
    assert "\\c&H00FFFF&" in content                # yellow highlight
    assert "\\b1" in content                        # bold
    assert "\\fscx" not in content                  # no scale bounce
    assert "\\fscy" not in content                  # no scale bounce
    assert content.count("\\fad(300,0)") == 1       # fade on first word only


def test_highlight_open_has_no_scale_bounce():
    """Scale animation on the active word changes its rendered width, which
    re-centers the whole (centered) caption line as focus moves word to
    word. The highlight must pop via color/bold only, never scale."""
    assert "\\fscx" not in _HIGHLIGHT_OPEN
    assert "\\fscy" not in _HIGHLIGHT_OPEN
    assert "\\c&H00FFFF&" in _HIGHLIGHT_OPEN
    assert "\\b1" in _HIGHLIGHT_OPEN


def test_write_ass_word_lines_fill_gaps_between_words(tmp_path):
    """Real whisper word timestamps can have small gaps between consecutive
    words (silences, plosives). Each word's Dialogue line should extend to
    the next word's start (chunk's own end for the last word), not stop at
    its own end — otherwise the whole caption blinks off during the gap."""
    chunks = [{
        "start": 0.0, "end": 2.0, "text": "alpha beta gamma",
        "words": [
            {"start": 0.0, "end": 0.4, "text": "alpha"},   # gap: 0.4 -> 0.6
            {"start": 0.6, "end": 1.3, "text": "beta"},    # gap: 1.3 -> 1.5
            {"start": 1.5, "end": 1.8, "text": "gamma"},   # trailing gap to chunk end 2.0
        ],
    }]
    ass_path = str(tmp_path / "c.ass")
    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3)
    dialogue_lines = [
        l for l in open(ass_path, encoding="utf-8").read().splitlines()
        if l.startswith("Dialogue:")
    ]
    assert len(dialogue_lines) == 3

    def _end_ts(line: str) -> str:
        return line.split(",")[2]

    assert _end_ts(dialogue_lines[0]) == _format_ass_timestamp(0.6)   # -> next word's start
    assert _end_ts(dialogue_lines[1]) == _format_ass_timestamp(1.5)   # -> next word's start
    assert _end_ts(dialogue_lines[2]) == _format_ass_timestamp(2.0)   # -> chunk's own end


def test_write_ass_word_highlight_false_is_one_line_per_chunk(tmp_path):
    chunks = [{
        "start": 0.0, "end": 2.0, "text": "alpha beta",
        "words": [
            {"start": 0.0, "end": 1.0, "text": "alpha"},
            {"start": 1.0, "end": 2.0, "text": "beta"},
        ],
    }]
    ass_path = str(tmp_path / "c.ass")
    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3, word_highlight=False)
    content = open(ass_path, encoding="utf-8").read()
    assert content.count("Dialogue:") == 1
    assert "\\c&H00FFFF&" not in content
    assert "\\fad(300,0)" in content


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


def test_chunk_segments_uses_real_word_timestamps():
    segments = [{
        "start": 10.0, "end": 12.0, "text": "alpha beta gamma",
        "words": [
            {"start": 10.0, "end": 10.5, "word": "alpha"},
            {"start": 10.5, "end": 11.2, "word": "beta"},
            {"start": 11.2, "end": 12.0, "word": "gamma"},
        ],
    }]
    chunks = _chunk_segments(segments, clip_start=10.0, clip_end=20.0, max_words=7)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["text"] == "alpha beta gamma"
    assert [w["text"] for w in c["words"]] == ["alpha", "beta", "gamma"]
    assert c["words"][0]["start"] == 0.0          # 10.0 - clip_start
    assert c["words"][2]["end"] == 2.0            # 12.0 - clip_start
    assert c["start"] == 0.0 and c["end"] == 2.0


def test_chunk_segments_estimates_words_when_absent():
    segments = [{"start": 0.0, "end": 4.0, "text": "hi supercalifragilistic"}]
    chunks = _chunk_segments(segments, clip_start=0.0, clip_end=100.0, max_words=7)
    words = chunks[0]["words"]
    assert [w["text"] for w in words] == ["hi", "supercalifragilistic"]
    # char-length weighted: 2 vs 20 chars over 4.0s -> shorter word gets less time
    assert words[1]["end"] - words[1]["start"] > words[0]["end"] - words[0]["start"]
    assert words[0]["start"] == 0.0
    assert words[-1]["end"] == pytest.approx(4.0)
