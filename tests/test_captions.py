from shorts_generator.captions import _chunk_segments, _write_ass


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
