import os
import shutil
import subprocess

import pytest

from shorts_generator.captions import (
    CaptionError,
    FONT_DIR,
    _HIGHLIGHT_OPEN,
    _chunk_cut_segments,
    _chunk_segments,
    _escape_ffmpeg_path,
    _format_ass_timestamp,
    _probe_resolution,
    _write_ass,
    burn_captions,
    burn_captions_segments,
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
    # The word group's estimated (character-weighted) per-word windows place
    # "alpha" and "beta" entirely before clip_start=10.0, so they must be
    # dropped rather than kept with mismatched clipped-vs-full text (that
    # mismatch was the bug: the group's aggregate time window got clipped
    # but its full text was kept regardless, producing text that didn't
    # match the displayed time window — and, once chunking happens per
    # kept-span instead of per-clip, duplicated text across chunks).
    segments = [{"start": 8.0, "end": 12.0, "text": "alpha beta gamma delta"}]

    chunks = _chunk_segments(segments, clip_start=10.0, clip_end=20.0, max_words=7)

    assert len(chunks) == 1
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 2.0
    assert chunks[0]["text"] == "gamma delta"
    assert "words" in chunks[0]
    assert [w["text"] for w in chunks[0]["words"]] == ["gamma", "delta"]
    assert chunks[0]["words"][0]["start"] == 0.0
    assert chunks[0]["words"][-1]["end"] == 2.0


def test_chunk_segments_trims_overlap_between_source_segments():
    # Real-world trigger: YouTube auto-captions (caption-only thread
    # ingest, see caption_ingest.py) aren't always strictly sequential --
    # two consecutive segments overlapping by a couple seconds otherwise
    # produces two caption chunks both covering the same wall-clock
    # window, rendering stacked/duplicated text on screen at once.
    segments = [
        {"start": 0.0, "end": 5.0, "text": "alpha beta gamma"},
        {"start": 3.0, "end": 8.0, "text": "delta epsilon zeta"},
    ]

    chunks = _chunk_segments(segments, clip_start=0.0, clip_end=100.0, max_words=7)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "alpha beta gamma"
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 3.0  # trimmed from 5.0 down to the next chunk's start
    assert chunks[1]["text"] == "delta epsilon zeta"
    assert chunks[1]["start"] == 3.0  # never shifted
    assert chunks[1]["end"] == 8.0


def test_chunk_segments_drops_chunk_fully_swallowed_by_overlap():
    # If the overlap is severe enough that trimming would collapse the
    # earlier chunk to zero/negative duration, drop it entirely rather than
    # emit a Dialogue line with no visible window.
    segments = [
        {"start": 0.0, "end": 5.0, "text": "alpha beta gamma"},
        {"start": 0.0, "end": 8.0, "text": "delta epsilon zeta"},
    ]

    chunks = _chunk_segments(segments, clip_start=0.0, clip_end=100.0, max_words=7)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "delta epsilon zeta"


def test_chunk_cut_segments_offsets_second_span_after_first():
    transcript_segments = [
        {"start": 0.0, "end": 2.0, "text": "alpha beta"},
        {"start": 10.0, "end": 12.0, "text": "gamma delta"},
    ]
    cut_segments = [
        {"start_time": 0.0, "end_time": 2.0},
        {"start_time": 10.0, "end_time": 12.0},
    ]

    chunks = _chunk_cut_segments(transcript_segments, cut_segments, max_words=7)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "alpha beta"
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 2.0
    # second span starts at output-timeline offset 2.0 (end of first kept span),
    # not at its own absolute transcript time of 10.0
    assert chunks[1]["text"] == "gamma delta"
    assert chunks[1]["start"] == 2.0
    assert chunks[1]["end"] == 4.0


def test_chunk_cut_segments_no_chunk_straddles_the_gap():
    transcript_segments = [
        {"start": 0.0, "end": 4.0, "text": "one two three four five six seven eight"},
    ]
    # Keep only [0,2] and [2.5,4] of this single 4s transcript segment --
    # a naive whole-envelope chunk pass could produce a chunk spanning the
    # dropped [2,2.5] gap; per-segment chunking must not.
    cut_segments = [
        {"start_time": 0.0, "end_time": 2.0},
        {"start_time": 2.5, "end_time": 4.0},
    ]

    chunks = _chunk_cut_segments(transcript_segments, cut_segments, max_words=7)

    max_span = max(s["end_time"] - s["start_time"] for s in cut_segments)
    for c in chunks:
        # every chunk's word set must come entirely from one kept span, so no
        # chunk can be longer than the longest kept span (here: 2.0s, from
        # [0.0, 2.0]) -- this still rejects a naive whole-envelope chunk pass
        # over [0.0, 4.0], which would yield a 3.5s chunk spanning the gap
        assert (c["end"] - c["start"]) <= max_span + 1e-6


def test_chunk_cut_segments_does_not_duplicate_word_group_straddling_cut_boundary():
    # Regression test for a Critical bug: in the estimate path (no real
    # per-word timestamps -- the only path api mode ever takes), a single
    # max_words-sized word group whose *aggregate* time window straddled an
    # internal cut-segment boundary used to be emitted twice: once
    # (time-clipped, but with its full un-clipped text) in each of the two
    # neighboring kept spans. Concretely, before the fix this produced
    # "one two three four five six seven" as the text of BOTH the first
    # chunk ([0.0, 2.0]) and the second chunk ([2.0, 3.0]), so the caption
    # visibly repeated itself right after the jump cut.
    transcript_segments = [
        {"start": 0.0, "end": 4.0, "text": "one two three four five six seven eight"},
    ]
    cut_segments = [
        {"start_time": 0.0, "end_time": 2.0},
        {"start_time": 2.5, "end_time": 4.0},
    ]

    chunks = _chunk_cut_segments(transcript_segments, cut_segments, max_words=7)

    # Every source word must appear in the combined output text at most once
    # -- no word should be split across two chunks' text.
    all_words = []
    for c in chunks:
        all_words.extend(c["text"].split())
    assert len(all_words) == len(set(all_words)), (
        f"a word appears in more than one chunk's text: {all_words}"
    )

    # In particular, the straddling group's leading words must not show up
    # verbatim in two different chunks.
    texts = [c["text"] for c in chunks]
    assert texts.count("one two three four five six seven") <= 1


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
    assert "\\fscx" not in content                  # no scale bounce — would re-center line
    assert "\\fscy" not in content                  # no scale bounce — would re-center line
    assert content.count("\\fad(300,0)") == 1       # fade on first word only


def test_highlight_open_has_no_scale_bounce():
    """Active word is color+bold only, no size scaling — scaling would
    change the (centered) line's rendered width and re-center it every
    word, reading as the caption jumping/switching too fast."""
    assert "\\fscx" not in _HIGHLIGHT_OPEN
    assert "\\fscy" not in _HIGHLIGHT_OPEN
    assert "\\c&H00FFFF&" in _HIGHLIGHT_OPEN
    assert "\\b1" in _HIGHLIGHT_OPEN


def test_write_ass_uses_montserrat_black_font(tmp_path):
    chunks = [{"start": 0.0, "end": 1.0, "text": "hello world"}]
    ass_path = str(tmp_path / "c.ass")

    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3)

    content = open(ass_path, encoding="utf-8").read()
    assert "Style: Caption,Montserrat Black," in content


def test_write_ass_default_margin_v_is_30_percent_of_height(tmp_path):
    chunks = [{"start": 0.0, "end": 1.0, "text": "hello world"}]
    ass_path = str(tmp_path / "c.ass")

    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3)

    content = open(ass_path, encoding="utf-8").read()
    style_line = next(l for l in content.splitlines() if l.startswith("Style:"))
    margin_v = int(style_line.split(",")[-2])
    assert margin_v == round(1080 * 0.30)


def test_write_ass_custom_bottom_margin_frac_changes_margin_v(tmp_path):
    chunks = [{"start": 0.0, "end": 1.0, "text": "hello world"}]
    ass_path = str(tmp_path / "c.ass")

    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3, bottom_margin_frac=0.06)

    content = open(ass_path, encoding="utf-8").read()
    style_line = next(l for l in content.splitlines() if l.startswith("Style:"))
    margin_v = int(style_line.split(",")[-2])
    assert margin_v == round(1080 * 0.06)


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


def test_burn_captions_forwards_bottom_margin_frac(tmp_path, synthetic_clip, monkeypatch):
    captured = {}
    real_write_ass = _write_ass

    def _spy_write_ass(*args, **kwargs):
        captured.update(kwargs)
        return real_write_ass(*args, **kwargs)

    monkeypatch.setattr("shorts_generator.captions._write_ass", _spy_write_ass)

    out_path = str(tmp_path / "burned.mp4")
    segments = [{"start": 0.0, "end": 3.0, "text": "hello there this is a caption test"}]

    burn_captions(
        synthetic_clip, segments, clip_start=0.0, clip_end=3.0, out_path=out_path,
        fade_seconds=0.3, bottom_margin_frac=0.06,
    )

    assert captured["bottom_margin_frac"] == 0.06


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


def test_burn_captions_segments_produces_output_file(tmp_path, synthetic_clip):
    out_path = str(tmp_path / "burned.mp4")
    transcript_segments = [{"start": 0.0, "end": 3.0, "text": "hello there this is a caption test"}]
    cut_segments = [{"start_time": 0.0, "end_time": 3.0}]

    result = burn_captions_segments(
        synthetic_clip, transcript_segments, cut_segments, out_path, fade_seconds=0.3,
    )

    assert result == out_path
    assert os.path.exists(out_path)


def test_burn_captions_segments_raises_when_no_transcript_overlaps(tmp_path, synthetic_clip):
    out_path = str(tmp_path / "burned.mp4")
    transcript_segments = [{"start": 100.0, "end": 103.0, "text": "way outside"}]
    cut_segments = [{"start_time": 0.0, "end_time": 3.0}]

    with pytest.raises(CaptionError):
        burn_captions_segments(synthetic_clip, transcript_segments, cut_segments, out_path)


def test_escape_ffmpeg_path_escapes_backslashes_and_colons():
    assert _escape_ffmpeg_path("C:\\videos\\out.ass") == "C:/videos/out.ass".replace(":", "\\:")


def test_burn_captions_vf_includes_fontsdir(tmp_path, synthetic_clip, monkeypatch):
    """The subtitles filter must point at the bundled fonts directory so
    Montserrat Black renders even on a host with no matching system font."""
    captured = {}
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr("shorts_generator.captions.subprocess.run", fake_run)

    out_path = str(tmp_path / "burned.mp4")
    segments = [{"start": 0.0, "end": 3.0, "text": "hello there caption test"}]

    burn_captions(synthetic_clip, segments, clip_start=0.0, clip_end=3.0, out_path=out_path, fade_seconds=0.3)

    vf_arg = captured["cmd"][captured["cmd"].index("-vf") + 1]
    assert vf_arg.startswith("subtitles=")
    assert ":fontsdir=" in vf_arg
    assert vf_arg.endswith(_escape_ffmpeg_path(FONT_DIR))


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
