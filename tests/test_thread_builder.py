import json

import pytest

from shorts_generator import thread_builder


def _corpus_entry(idx, title, abstract, run_dir):
    return {"run_dir": run_dir, "title": title, "source_url": f"https://example.com/{idx}", "abstract": abstract}


def _episode(duration, texts_with_times):
    segments = [{"start": s, "end": e, "text": t} for s, e, t in texts_with_times]
    return {"transcript": {"duration": duration, "segments": segments}}


def test_pick_thread_clips_returns_none_when_not_grounded():
    episode_a = _episode(100.0, [(0.0, 10.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 10.0, "world")])
    llm_fn = lambda prompt: json.dumps({"grounded": False, "thesis": "", "bridge": "", "clip_a": {}, "clip_b": {}})

    assert thread_builder.pick_thread_clips(episode_a, episode_b, "shared question?", llm_fn) is None


def test_pick_thread_clips_rejects_non_string_thesis():
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    llm_fn = lambda prompt: json.dumps({
        "grounded": True,
        "thesis": ["not", "a", "string"],
        "bridge": "Here is the other side.",
        "clip_a": {"start_time": 5.0, "end_time": 25.0},
        "clip_b": {"start_time": 2.0, "end_time": 20.0},
    })

    assert thread_builder.pick_thread_clips(episode_a, episode_b, "shared question?", llm_fn) is None


def test_pick_thread_clips_returns_clips_and_narration_on_valid_response():
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    llm_fn = lambda prompt: json.dumps({
        "grounded": True,
        "thesis": "Two guests, one question.",
        "bridge": "Here is the other side.",
        "clip_a": {"start_time": 5.0, "end_time": 25.0},
        "clip_b": {"start_time": 2.0, "end_time": 20.0},
    })

    result = thread_builder.pick_thread_clips(episode_a, episode_b, "shared question?", llm_fn)

    assert result == {
        "thesis": "Two guests, one question.",
        "bridge": "Here is the other side.",
        "clip_a": {"start_time": 5.0, "end_time": 25.0},
        "clip_b": {"start_time": 2.0, "end_time": 20.0},
    }


def test_pick_thread_clips_rejects_span_shorter_than_8_seconds():
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    llm_fn = lambda prompt: json.dumps({
        "grounded": True, "thesis": "t", "bridge": "b",
        "clip_a": {"start_time": 5.0, "end_time": 7.0},
        "clip_b": {"start_time": 2.0, "end_time": 20.0},
    })

    assert thread_builder.pick_thread_clips(episode_a, episode_b, "q?", llm_fn) is None


def test_pick_thread_clips_clamps_end_time_to_episode_duration():
    episode_a = _episode(20.0, [(0.0, 20.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    llm_fn = lambda prompt: json.dumps({
        "grounded": True, "thesis": "t", "bridge": "b",
        "clip_a": {"start_time": 5.0, "end_time": 50.0},
        "clip_b": {"start_time": 2.0, "end_time": 20.0},
    })

    result = thread_builder.pick_thread_clips(episode_a, episode_b, "q?", llm_fn)

    assert result["clip_a"]["end_time"] == 20.0


def test_pick_thread_clips_includes_avoid_ranges_in_prompt_when_given():
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    seen_prompts = []

    def llm_fn(prompt):
        seen_prompts.append(prompt)
        return json.dumps({
            "grounded": True, "thesis": "t", "bridge": "b",
            "clip_a": {"start_time": 5.0, "end_time": 25.0},
            "clip_b": {"start_time": 2.0, "end_time": 20.0},
        })

    thread_builder.pick_thread_clips(
        episode_a, episode_b, "q?", llm_fn,
        avoid_ranges_a=[(0.0, 10.0)], avoid_ranges_b=[(50.0, 60.0)],
    )

    assert "0.0s-10.0s" in seen_prompts[0]
    assert "50.0s-60.0s" in seen_prompts[0]


def test_pick_thread_clips_prompt_has_no_avoid_block_when_none_given():
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    seen_prompts = []

    def llm_fn(prompt):
        seen_prompts.append(prompt)
        return json.dumps({
            "grounded": True, "thesis": "t", "bridge": "b",
            "clip_a": {"start_time": 5.0, "end_time": 25.0},
            "clip_b": {"start_time": 2.0, "end_time": 20.0},
        })

    thread_builder.pick_thread_clips(episode_a, episode_b, "q?", llm_fn)

    assert "already-used" not in seen_prompts[0]


def test_find_same_topic_pairs_returns_up_to_num_pairs_questions():
    entry_a = _corpus_entry(0, "Ep A", "discusses remote work and also housing policy", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "discusses remote work and also housing policy", "/tmp/b")
    llm_fn = lambda prompt: json.dumps({"shared_questions": [
        "Does remote work increase productivity?",
        "Does zoning reform lower housing costs?",
        "A third question that should be dropped",
    ]})

    result = thread_builder.find_same_topic_pairs(entry_a, entry_b, num_pairs=2, llm_fn=llm_fn)

    assert result == [
        "Does remote work increase productivity?",
        "Does zoning reform lower housing costs?",
    ]


def test_find_same_topic_pairs_drops_non_string_items():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    llm_fn = lambda prompt: json.dumps({"shared_questions": ["A real question?", ["not", "a", "string"], "", "   "]})

    result = thread_builder.find_same_topic_pairs(entry_a, entry_b, num_pairs=5, llm_fn=llm_fn)

    assert result == ["A real question?"]


def test_find_same_topic_pairs_dedupes_case_insensitive_duplicates():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    llm_fn = lambda prompt: json.dumps({"shared_questions": [
        "Does X cause Y?", "does x cause y?", "Does X Cause Y?",
    ]})

    result = thread_builder.find_same_topic_pairs(entry_a, entry_b, num_pairs=5, llm_fn=llm_fn)

    assert result == ["Does X cause Y?"]


def test_find_same_topic_pairs_returns_empty_list_on_malformed_llm_output():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    llm_fn = lambda prompt: "not json at all"

    assert thread_builder.find_same_topic_pairs(entry_a, entry_b, num_pairs=3, llm_fn=llm_fn) == []


def test_find_same_topic_pairs_returns_empty_list_when_num_pairs_less_than_one():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    llm_fn = lambda prompt: pytest.fail("llm_fn should not be called when num_pairs < 1")

    assert thread_builder.find_same_topic_pairs(entry_a, entry_b, num_pairs=0, llm_fn=llm_fn) == []


def _transcript(duration, texts_with_times):
    segments = [{"start": s, "end": e, "text": t} for s, e, t in texts_with_times]
    return {"duration": duration, "segments": segments}


def test_select_thread_pairs_returns_empty_list_when_no_shared_questions():
    entry_a = _corpus_entry(0, "Ep A", "unrelated topic one", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "unrelated topic two", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "world")])
    llm_fn = lambda prompt: json.dumps({"shared_questions": []})

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=2, llm_fn=llm_fn)

    assert result == []


def test_select_thread_pairs_returns_one_grounded_pair():
    entry_a = _corpus_entry(0, "Ep A", "argues remote work increases productivity", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "argues remote work decreases productivity", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    responses = [
        json.dumps({"shared_questions": ["Does remote work increase or decrease productivity?"]}),
        json.dumps({
            "grounded": True, "thesis": "Two guests, one question.", "bridge": "Here is the other side.",
            "clip_a": {"start_time": 5.0, "end_time": 25.0},
            "clip_b": {"start_time": 2.0, "end_time": 20.0},
        }),
    ]

    def llm_fn(prompt):
        return responses.pop(0)

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=1, llm_fn=llm_fn)

    assert result == [{
        "shared_question": "Does remote work increase or decrease productivity?",
        "thesis": "Two guests, one question.",
        "bridge": "Here is the other side.",
        "episode_a": {"run_dir": "/tmp/a", "title": "Ep A", "source_url": "https://example.com/0", "start_time": 5.0, "end_time": 25.0},
        "episode_b": {"run_dir": "/tmp/b", "title": "Ep B", "source_url": "https://example.com/1", "start_time": 2.0, "end_time": 20.0},
    }]


def test_select_thread_pairs_returns_multiple_grounded_pairs_for_multiple_questions():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    responses = [
        json.dumps({"shared_questions": ["Question one?", "Question two?"]}),
        json.dumps({
            "grounded": True, "thesis": "t1", "bridge": "b1",
            "clip_a": {"start_time": 0.0, "end_time": 20.0},
            "clip_b": {"start_time": 0.0, "end_time": 20.0},
        }),
        json.dumps({
            "grounded": True, "thesis": "t2", "bridge": "b2",
            "clip_a": {"start_time": 40.0, "end_time": 60.0},
            "clip_b": {"start_time": 40.0, "end_time": 60.0},
        }),
    ]

    def llm_fn(prompt):
        return responses.pop(0)

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=2, llm_fn=llm_fn)

    assert [r["shared_question"] for r in result] == ["Question one?", "Question two?"]


def test_select_thread_pairs_discards_pair_whose_span_overlaps_an_earlier_pick():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    responses = [
        json.dumps({"shared_questions": ["Question one?", "Question two?"]}),
        json.dumps({
            "grounded": True, "thesis": "t1", "bridge": "b1",
            "clip_a": {"start_time": 0.0, "end_time": 20.0},
            "clip_b": {"start_time": 0.0, "end_time": 20.0},
        }),
        # Question two's clip_a overlaps question one's accepted clip_a (0-20 vs 10-30) -- must be discarded.
        json.dumps({
            "grounded": True, "thesis": "t2", "bridge": "b2",
            "clip_a": {"start_time": 10.0, "end_time": 30.0},
            "clip_b": {"start_time": 40.0, "end_time": 60.0},
        }),
    ]

    def llm_fn(prompt):
        return responses.pop(0)

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=2, llm_fn=llm_fn)

    assert len(result) == 1
    assert result[0]["shared_question"] == "Question one?"


def test_select_thread_pairs_accepts_pair_whose_span_only_touches_an_earlier_endpoint():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    responses = [
        json.dumps({"shared_questions": ["Question one?", "Question two?"]}),
        json.dumps({
            "grounded": True, "thesis": "t1", "bridge": "b1",
            "clip_a": {"start_time": 0.0, "end_time": 20.0},
            "clip_b": {"start_time": 0.0, "end_time": 20.0},
        }),
        # clip_a starts exactly where question one's clip_a ended (20.0) -- touching, not overlapping.
        json.dumps({
            "grounded": True, "thesis": "t2", "bridge": "b2",
            "clip_a": {"start_time": 20.0, "end_time": 40.0},
            "clip_b": {"start_time": 40.0, "end_time": 60.0},
        }),
    ]

    def llm_fn(prompt):
        return responses.pop(0)

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=2, llm_fn=llm_fn)

    assert len(result) == 2


def test_select_thread_pairs_discards_pair_when_only_episode_b_span_overlaps():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    responses = [
        json.dumps({"shared_questions": ["Question one?", "Question two?"]}),
        json.dumps({
            "grounded": True, "thesis": "t1", "bridge": "b1",
            "clip_a": {"start_time": 0.0, "end_time": 20.0},
            "clip_b": {"start_time": 50.0, "end_time": 70.0},
        }),
        # clip_a is clearly non-overlapping vs used_ranges_a=[(0,20)] (30-50 vs 0-20).
        # clip_b overlaps vs used_ranges_b=[(50,70)] (60-80 vs 50-70).
        json.dumps({
            "grounded": True, "thesis": "t2", "bridge": "b2",
            "clip_a": {"start_time": 30.0, "end_time": 50.0},
            "clip_b": {"start_time": 60.0, "end_time": 80.0},
        }),
    ]

    def llm_fn(prompt):
        return responses.pop(0)

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=2, llm_fn=llm_fn)

    assert len(result) == 1
    assert result[0]["shared_question"] == "Question one?"


def test_select_thread_pairs_skips_ungroundable_question_and_continues():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    responses = [
        json.dumps({"shared_questions": ["Question one?", "Question two?"]}),
        json.dumps({"grounded": False, "thesis": "", "bridge": "", "clip_a": {}, "clip_b": {}}),
        json.dumps({
            "grounded": True, "thesis": "t2", "bridge": "b2",
            "clip_a": {"start_time": 0.0, "end_time": 20.0},
            "clip_b": {"start_time": 0.0, "end_time": 20.0},
        }),
    ]

    def llm_fn(prompt):
        return responses.pop(0)

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=2, llm_fn=llm_fn)

    assert len(result) == 1
    assert result[0]["shared_question"] == "Question two?"


def test_select_thread_pairs_stops_once_num_clips_reached(monkeypatch):
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])

    monkeypatch.setattr(
        thread_builder, "find_same_topic_pairs",
        lambda entry_a, entry_b, num_pairs, llm_fn: ["Question one?", "Question two?", "Question three?"],
    )

    responses = [
        json.dumps({
            "grounded": True, "thesis": "t1", "bridge": "b1",
            "clip_a": {"start_time": 0.0, "end_time": 20.0}, "clip_b": {"start_time": 0.0, "end_time": 20.0},
        }),
    ]

    def llm_fn(prompt):
        if not responses:
            pytest.fail("select_thread_pairs must stop calling pick_thread_clips once num_clips is reached")
        return responses.pop(0)

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=1, llm_fn=llm_fn)

    assert len(result) == 1


def test_select_thread_pairs_tolerates_malformed_transcript_segment_shape():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    # Segment missing "text"/"start" -- build_transcript_text (called inside
    # pick_thread_clips) will raise; select_thread_pairs must catch that per
    # question and return whatever it has ([] here), not propagate.
    transcript_a = {"duration": 100.0, "segments": [{"end": 10.0}]}
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    llm_fn = lambda prompt: json.dumps({"shared_questions": ["Question one?"]})

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=1, llm_fn=llm_fn)

    assert result == []
