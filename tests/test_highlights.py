import json
import os

from shorts_generator.highlights import (
    CHAPTER_SCHEMA_VERSION,
    CLAIM_SPECIFICITY_THRESHOLD,
    HIGHLIGHT_SCHEMA_VERSION,
    MAX_CHAPTER_DURATION_SECONDS,
    MIN_CHAPTER_DURATION_SECONDS,
    _sanitize_chapters,
    _sanitize_highlights,
    _transcript_fingerprint,
    call_highlight_api,
    dedupe_highlights,
    get_highlights,
    get_highlights_cached,
    select_final_highlights,
)


def _raw_highlight(**overrides):
    base = {
        "title": "Big News",
        "start_time": 1.0,
        "end_time": 5.0,
        "score": 90,
        "hook_sentence": "This is the full hook sentence spoken in the clip.",
        "on_screen_hook": "WAIT FOR IT",
        "virality_reason": "because",
        "description": "desc",
    }
    base.update(overrides)
    return base


def test_sanitize_highlights_includes_on_screen_hook():
    cleaned = _sanitize_highlights([_raw_highlight()], duration=100.0)
    assert cleaned[0]["on_screen_hook"] == "WAIT FOR IT"


def test_sanitize_highlights_caps_on_screen_hook_length():
    cleaned = _sanitize_highlights([_raw_highlight(on_screen_hook="x" * 200)], duration=100.0)
    assert len(cleaned[0]["on_screen_hook"]) == 60


def test_sanitize_highlights_defaults_on_screen_hook_to_empty_string():
    raw = {"start_time": 1.0, "end_time": 5.0}
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["on_screen_hook"] == ""


def test_sanitize_highlights_clamps_hook_strength_above_range():
    cleaned = _sanitize_highlights([_raw_highlight(hook_strength=150)], duration=100.0)
    assert cleaned[0]["hook_strength"] == 100


def test_sanitize_highlights_clamps_hook_strength_below_range():
    cleaned = _sanitize_highlights([_raw_highlight(hook_strength=-20)], duration=100.0)
    assert cleaned[0]["hook_strength"] == 0


def test_sanitize_highlights_defaults_hook_fields_when_missing():
    raw = {"start_time": 1.0, "end_time": 5.0}
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["hook_strength"] == 0
    assert cleaned[0]["hook_self_contained"] is False
    assert cleaned[0]["hook_reason"] == ""


def test_sanitize_highlights_coerces_string_hook_self_contained():
    cleaned = _sanitize_highlights([_raw_highlight(hook_self_contained="true")], duration=100.0)
    assert cleaned[0]["hook_self_contained"] is True


def test_sanitize_highlights_keeps_valid_cut_segments():
    raw = _raw_highlight(
        start_time=1.0, end_time=10.0,
        cut_segments=[
            {"start_time": 1.0, "end_time": 3.0},
            {"start_time": 6.0, "end_time": 10.0},
        ],
    )
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["cut_segments"] == [
        {"start_time": 1.0, "end_time": 3.0},
        {"start_time": 6.0, "end_time": 10.0},
    ]


def test_sanitize_highlights_falls_back_to_envelope_when_cut_segments_missing():
    raw = _raw_highlight(start_time=1.0, end_time=5.0)
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["cut_segments"] == [{"start_time": 1.0, "end_time": 5.0}]


def test_sanitize_highlights_falls_back_to_envelope_when_cut_segments_overlap():
    raw = _raw_highlight(
        start_time=1.0, end_time=10.0,
        cut_segments=[
            {"start_time": 1.0, "end_time": 5.0},
            {"start_time": 4.0, "end_time": 10.0},
        ],
    )
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["cut_segments"] == [{"start_time": 1.0, "end_time": 10.0}]


def test_sanitize_highlights_clamps_cut_segments_to_envelope():
    raw = _raw_highlight(
        start_time=2.0, end_time=8.0,
        cut_segments=[{"start_time": 0.0, "end_time": 20.0}],
    )
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["cut_segments"] == [{"start_time": 2.0, "end_time": 8.0}]


def test_sanitize_highlights_caps_cut_segments_at_six():
    raw = _raw_highlight(
        start_time=0.0, end_time=100.0,
        cut_segments=[{"start_time": float(i * 10), "end_time": float(i * 10 + 5)} for i in range(8)],
    )
    cleaned = _sanitize_highlights([raw], duration=200.0)
    assert len(cleaned[0]["cut_segments"]) == 6


def test_sanitize_highlights_keeps_valid_reaction_type():
    raw = _raw_highlight(reaction_type="LOL")
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["reaction_type"] == "LOL"


def test_sanitize_highlights_reaction_type_case_insensitive():
    raw = _raw_highlight(reaction_type="lol")
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["reaction_type"] == "LOL"


def test_sanitize_highlights_defaults_reaction_type_when_invalid():
    raw = _raw_highlight(reaction_type="HYPE")
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["reaction_type"] == "WOW"


def test_sanitize_highlights_defaults_reaction_type_when_missing():
    raw = {"start_time": 1.0, "end_time": 5.0}
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["reaction_type"] == "WOW"


def test_sanitize_highlights_defaults_tightness_reason_to_empty_string():
    raw = {"start_time": 1.0, "end_time": 5.0}
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["tightness_reason"] == ""


def test_sanitize_highlights_includes_tightness_reason():
    raw = _raw_highlight(tightness_reason="cut the walk-back-in, kept the punchline")
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["tightness_reason"] == "cut the walk-back-in, kept the punchline"


def test_sanitize_highlights_clamps_format_clarity_score_above_range():
    cleaned = _sanitize_highlights([_raw_highlight(format_clarity_score=150)], duration=100.0)
    assert cleaned[0]["format_clarity_score"] == 100


def test_sanitize_highlights_clamps_format_clarity_score_below_range():
    cleaned = _sanitize_highlights([_raw_highlight(format_clarity_score=-20)], duration=100.0)
    assert cleaned[0]["format_clarity_score"] == 0


def test_sanitize_highlights_defaults_format_fields_when_missing():
    raw = {"start_time": 1.0, "end_time": 5.0}
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["format_clarity_score"] == 0
    assert cleaned[0]["format_reason"] == ""


def test_sanitize_highlights_includes_format_reason():
    raw = _raw_highlight(format_clarity_score=85, format_reason="single clean before/after beat")
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["format_clarity_score"] == 85
    assert cleaned[0]["format_reason"] == "single clean before/after beat"


def test_sanitize_highlights_clamps_claim_specificity_above_range():
    cleaned = _sanitize_highlights([_raw_highlight(claim_specificity=150)], duration=100.0)
    assert cleaned[0]["claim_specificity"] == 100


def test_sanitize_highlights_clamps_claim_specificity_below_range():
    cleaned = _sanitize_highlights([_raw_highlight(claim_specificity=-20)], duration=100.0)
    assert cleaned[0]["claim_specificity"] == 0


def test_sanitize_highlights_defaults_claim_specificity_fields_when_missing():
    raw = {"start_time": 1.0, "end_time": 5.0}
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["claim_specificity"] == 0
    assert cleaned[0]["claim_specificity_reason"] == ""


def test_sanitize_highlights_includes_claim_specificity_reason():
    raw = _raw_highlight(claim_specificity=88, claim_specificity_reason="names a specific dollar figure")
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["claim_specificity"] == 88
    assert cleaned[0]["claim_specificity_reason"] == "names a specific dollar figure"


def test_transcript_fingerprint_stable_for_identical_transcripts():
    t1 = {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi"}]}
    t2 = {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi"}]}
    assert _transcript_fingerprint(t1) == _transcript_fingerprint(t2)


def test_transcript_fingerprint_changes_when_segments_change():
    t1 = {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi"}]}
    t2 = {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "bye"}]}
    assert _transcript_fingerprint(t1) != _transcript_fingerprint(t2)


def test_transcript_fingerprint_changes_when_duration_changes():
    t1 = {"duration": 10.0, "segments": []}
    t2 = {"duration": 20.0, "segments": []}
    assert _transcript_fingerprint(t1) != _transcript_fingerprint(t2)


def _fake_short_transcript():
    return {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi there"}]}


def _fake_llm_responses(highlight_title):
    def fake_llm_fn(prompt):
        if "Analyze this video transcript" in prompt:
            return '{"content_type": "podcast", "density": "medium"}'
        return (
            '{"highlights": [{"title": "%s", "start_time": 0.0, "end_time": 3.0, "score": 90}]}'
            % highlight_title
        )
    return fake_llm_fn


def test_get_highlights_cached_calls_llm_and_writes_cache_on_miss(tmp_path):
    cache_path = str(tmp_path / "highlights.json")
    transcript = _fake_short_transcript()

    result = get_highlights_cached(
        transcript, num_clips=1, cache_path=cache_path, llm_fn=_fake_llm_responses("Clip")
    )

    assert result["highlights"][0]["title"] == "Clip"
    assert os.path.exists(cache_path)
    with open(cache_path) as f:
        cached = json.load(f)
    assert cached["num_clips"] == 1
    assert cached["transcript_fingerprint"] == _transcript_fingerprint(transcript)
    assert cached["schema_version"] == HIGHLIGHT_SCHEMA_VERSION
    assert cached["highlights"][0]["title"] == "Clip"


def test_get_highlights_cached_skips_llm_on_matching_cache(tmp_path):
    cache_path = str(tmp_path / "highlights.json")
    transcript = _fake_short_transcript()
    with open(cache_path, "w") as f:
        json.dump({
            "transcript_fingerprint": _transcript_fingerprint(transcript),
            "num_clips": 1,
            "schema_version": HIGHLIGHT_SCHEMA_VERSION,
            "highlights": [{"title": "Cached Clip", "start_time": 0.0, "end_time": 3.0, "score": 80}],
        }, f)

    def fail_if_called(prompt):
        raise AssertionError("llm_fn should not be called on a cache hit")

    result = get_highlights_cached(transcript, num_clips=1, cache_path=cache_path, llm_fn=fail_if_called)

    assert result["highlights"][0]["title"] == "Cached Clip"


def test_get_highlights_cached_recomputes_on_num_clips_mismatch(tmp_path):
    cache_path = str(tmp_path / "highlights.json")
    transcript = _fake_short_transcript()
    with open(cache_path, "w") as f:
        json.dump({
            "transcript_fingerprint": _transcript_fingerprint(transcript),
            "num_clips": 1,
            "schema_version": HIGHLIGHT_SCHEMA_VERSION,
            "highlights": [{"title": "Cached Clip", "start_time": 0.0, "end_time": 3.0, "score": 80}],
        }, f)

    result = get_highlights_cached(
        transcript, num_clips=2, cache_path=cache_path, llm_fn=_fake_llm_responses("Fresh Clip")
    )

    assert result["highlights"][0]["title"] == "Fresh Clip"


def test_get_highlights_cached_recomputes_on_fingerprint_mismatch(tmp_path):
    cache_path = str(tmp_path / "highlights.json")
    transcript = _fake_short_transcript()
    with open(cache_path, "w") as f:
        json.dump({
            "transcript_fingerprint": "stale-fingerprint",
            "num_clips": 1,
            "schema_version": HIGHLIGHT_SCHEMA_VERSION,
            "highlights": [{"title": "Cached Clip", "start_time": 0.0, "end_time": 3.0, "score": 80}],
        }, f)

    result = get_highlights_cached(
        transcript, num_clips=1, cache_path=cache_path, llm_fn=_fake_llm_responses("Fresh Clip")
    )

    assert result["highlights"][0]["title"] == "Fresh Clip"


def test_get_highlights_cached_recomputes_on_schema_version_mismatch(tmp_path):
    cache_path = str(tmp_path / "highlights.json")
    transcript = _fake_short_transcript()
    with open(cache_path, "w") as f:
        json.dump({
            "transcript_fingerprint": _transcript_fingerprint(transcript),
            "num_clips": 1,
            "schema_version": HIGHLIGHT_SCHEMA_VERSION - 1,
            "highlights": [{"title": "Cached Clip", "start_time": 0.0, "end_time": 3.0, "score": 80}],
        }, f)

    result = get_highlights_cached(
        transcript, num_clips=1, cache_path=cache_path, llm_fn=_fake_llm_responses("Fresh Clip")
    )

    assert result["highlights"][0]["title"] == "Fresh Clip"


def test_get_highlights_cached_recomputes_on_missing_schema_version(tmp_path):
    """A cache file written before schema_version existed (e.g. by an older
    binary) must be treated as a miss, not silently reused."""
    cache_path = str(tmp_path / "highlights.json")
    transcript = _fake_short_transcript()
    with open(cache_path, "w") as f:
        json.dump({
            "transcript_fingerprint": _transcript_fingerprint(transcript),
            "num_clips": 1,
            "highlights": [{"title": "Cached Clip", "start_time": 0.0, "end_time": 3.0, "score": 80}],
        }, f)

    result = get_highlights_cached(
        transcript, num_clips=1, cache_path=cache_path, llm_fn=_fake_llm_responses("Fresh Clip")
    )

    assert result["highlights"][0]["title"] == "Fresh Clip"


def test_get_highlights_cached_recomputes_on_corrupted_cache_file(tmp_path):
    cache_path = str(tmp_path / "highlights.json")
    with open(cache_path, "w") as f:
        f.write("{not valid json")

    transcript = _fake_short_transcript()

    result = get_highlights_cached(
        transcript, num_clips=1, cache_path=cache_path, llm_fn=_fake_llm_responses("Fresh Clip")
    )

    assert result["highlights"][0]["title"] == "Fresh Clip"
    with open(cache_path) as f:
        cached = json.load(f)
    assert cached["highlights"][0]["title"] == "Fresh Clip"


def test_dedupe_highlights_order_unaffected_by_hook_strength():
    """hook_strength is a human-review-only signal (backtested to invert on
    some content types) and must never become a sort key. Two same-score,
    non-overlapping highlights should keep dedupe_highlights' natural
    (score-based, stable) order regardless of which one has the higher
    hook_strength."""
    low_then_high = [
        {"title": "A", "start_time": 0.0, "end_time": 3.0, "score": 90, "hook_strength": 5},
        {"title": "B", "start_time": 10.0, "end_time": 13.0, "score": 90, "hook_strength": 95},
    ]
    assert [h["title"] for h in dedupe_highlights(low_then_high)] == ["A", "B"]

    high_then_low = [
        {"title": "B", "start_time": 10.0, "end_time": 13.0, "score": 90, "hook_strength": 95},
        {"title": "A", "start_time": 0.0, "end_time": 3.0, "score": 90, "hook_strength": 5},
    ]
    assert [h["title"] for h in dedupe_highlights(high_then_low)] == ["B", "A"]


def test_get_highlights_order_unaffected_by_hook_strength():
    """Same guard as above but through the full get_highlights entry point:
    the highlight with the lower hook_strength but same score comes first
    in the LLM response and must stay first in the output."""
    transcript = {"duration": 20.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi there"}]}

    def fake_llm_fn(prompt):
        if "Analyze this video transcript" in prompt:
            return '{"content_type": "podcast", "density": "medium"}'
        return json.dumps({
            "highlights": [
                {
                    "title": "Low Hook Strength First",
                    "start_time": 0.0,
                    "end_time": 3.0,
                    "score": 90,
                    "hook_strength": 5,
                },
                {
                    "title": "High Hook Strength Second",
                    "start_time": 10.0,
                    "end_time": 13.0,
                    "score": 90,
                    "hook_strength": 95,
                },
            ]
        })

    result = get_highlights(transcript, num_clips=2, llm_fn=fake_llm_fn)

    titles = [h["title"] for h in result["highlights"]]
    assert titles == ["Low Hook Strength First", "High Hook Strength Second"]


def test_call_highlight_api_retry_log_surfaces_real_error(capsys):
    """A stalled/errored llm_fn should be logged with its own message, not
    mislabeled as 'invalid model output' — that label previously hid timeouts
    and network errors behind a JSON-parsing-sounding message."""

    def flaky_llm_fn(prompt):
        raise TimeoutError("request timed out after 180s")

    try:
        call_highlight_api("transcript", {}, duration=100.0, num_clips=3, llm_fn=flaky_llm_fn)
    except RuntimeError as e:
        assert "request timed out after 180s" in str(e)

    out = capsys.readouterr().out
    assert "request timed out after 180s" in out
    assert "invalid model output on attempt" not in out


def test_select_final_highlights_keeps_top_passers_by_score():
    highlights = [
        {"title": "A", "score": 90, "claim_specificity": 85},
        {"title": "B", "score": 95, "claim_specificity": 82},
        {"title": "C", "score": 99, "claim_specificity": 50},  # highest score, fails the gate
    ]
    result = select_final_highlights(highlights, num_clips=2)
    assert [h["title"] for h in result] == ["B", "A"]


def test_select_final_highlights_backfills_when_too_few_passers():
    highlights = [
        {"title": "A", "score": 90, "claim_specificity": 85},  # passes
        {"title": "B", "score": 80, "claim_specificity": 40},  # fails
        {"title": "C", "score": 70, "claim_specificity": 30},  # fails
    ]
    result = select_final_highlights(highlights, num_clips=2)
    assert [h["title"] for h in result] == ["A", "B"]


def test_select_final_highlights_zero_passers_matches_score_only_ranking():
    highlights = [
        {"title": "A", "score": 90, "claim_specificity": 10},
        {"title": "B", "score": 95, "claim_specificity": 20},
        {"title": "C", "score": 70, "claim_specificity": 5},
    ]
    result = select_final_highlights(highlights, num_clips=2)
    assert [h["title"] for h in result] == ["B", "A"]


def test_select_final_highlights_returns_all_when_fewer_than_num_clips():
    highlights = [{"title": "A", "score": 90, "claim_specificity": 85}]
    result = select_final_highlights(highlights, num_clips=3)
    assert [h["title"] for h in result] == ["A"]


def test_select_final_highlights_missing_claim_specificity_defaults_to_non_passer():
    highlights = [
        {"title": "A", "score": 90},  # no claim_specificity key at all
        {"title": "B", "score": 80, "claim_specificity": 85},
    ]
    result = select_final_highlights(highlights, num_clips=2)
    assert [h["title"] for h in result] == ["B", "A"]


def test_claim_specificity_threshold_is_80():
    assert CLAIM_SPECIFICITY_THRESHOLD == 80


def _raw_chapter(**overrides):
    base = {
        "title": "The Origin Story",
        "start_time": 10.0,
        "end_time": 400.0,
        "summary": "They trace the idea back to a late-night argument and explain how it evolved.",
        "interest_reason": "a complete, self-contained origin story with a clear arc",
    }
    base.update(overrides)
    return base


def test_sanitize_chapters_keeps_valid_chapter():
    cleaned = _sanitize_chapters([_raw_chapter()], duration=1000.0)
    assert len(cleaned) == 1
    assert cleaned[0]["title"] == "The Origin Story"
    assert cleaned[0]["start_time"] == 10.0
    assert cleaned[0]["end_time"] == 400.0
    assert cleaned[0]["summary"] == _raw_chapter()["summary"]
    assert cleaned[0]["interest_reason"] == _raw_chapter()["interest_reason"]


def test_sanitize_chapters_drops_shorter_than_min_duration():
    raw = _raw_chapter(start_time=0.0, end_time=30.0)  # 30s < MIN_CHAPTER_DURATION_SECONDS (60)
    cleaned = _sanitize_chapters([raw], duration=1000.0)
    assert cleaned == []


def test_sanitize_chapters_clamps_end_time_to_max_duration():
    raw = _raw_chapter(start_time=0.0, end_time=2000.0)  # way over MAX_CHAPTER_DURATION_SECONDS (900)
    cleaned = _sanitize_chapters([raw], duration=5000.0)
    assert cleaned[0]["end_time"] == 900.0


def test_sanitize_chapters_clamps_to_video_duration():
    raw = _raw_chapter(start_time=90.0, end_time=200.0)
    cleaned = _sanitize_chapters([raw], duration=150.0)
    assert cleaned[0]["end_time"] == 150.0


def test_sanitize_chapters_drops_invalid_start_end():
    raw = _raw_chapter(start_time=100.0, end_time=50.0)  # end before start
    cleaned = _sanitize_chapters([raw], duration=1000.0)
    assert cleaned == []


def test_sanitize_chapters_defaults_missing_fields():
    raw = {"start_time": 0.0, "end_time": 200.0}
    cleaned = _sanitize_chapters([raw], duration=1000.0)
    assert cleaned[0]["title"] == "Untitled Chapter"
    assert cleaned[0]["summary"] == ""
    assert cleaned[0]["interest_reason"] == ""


def test_sanitize_chapters_ignores_non_list_input():
    assert _sanitize_chapters(None, duration=1000.0) == []
    assert _sanitize_chapters("not a list", duration=1000.0) == []


def test_sanitize_chapters_skips_non_dict_entries():
    cleaned = _sanitize_chapters(["not a dict", _raw_chapter()], duration=1000.0)
    assert len(cleaned) == 1


def test_chapter_duration_bounds_are_60_and_900():
    assert MIN_CHAPTER_DURATION_SECONDS == 60
    assert MAX_CHAPTER_DURATION_SECONDS == 900


def test_chapter_schema_version_is_1():
    assert CHAPTER_SCHEMA_VERSION == 1
