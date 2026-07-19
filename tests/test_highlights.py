import json
import os

from shorts_generator.highlights import (
    _sanitize_highlights,
    _transcript_fingerprint,
    call_highlight_api,
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
    assert cached["highlights"][0]["title"] == "Clip"


def test_get_highlights_cached_skips_llm_on_matching_cache(tmp_path):
    cache_path = str(tmp_path / "highlights.json")
    transcript = _fake_short_transcript()
    with open(cache_path, "w") as f:
        json.dump({
            "transcript_fingerprint": _transcript_fingerprint(transcript),
            "num_clips": 1,
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
