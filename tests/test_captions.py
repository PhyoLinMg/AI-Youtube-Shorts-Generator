from shorts_generator.captions import _chunk_segments


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
