import json
import os

from shorts_generator.highlights import (
    HIGHLIGHT_SCHEMA_VERSION,
    _sanitize_highlights,
    _transcript_fingerprint,
    call_highlight_api,
    dedupe_highlights,
    get_highlights,
    get_highlights_cached,
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
