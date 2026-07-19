from shorts_generator.highlights import _sanitize_highlights, _transcript_fingerprint, call_highlight_api


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
