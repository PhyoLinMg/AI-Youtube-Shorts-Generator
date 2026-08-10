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


def test_build_thread_returns_none_when_no_same_topic_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(
        thread_builder, "build_corpus",
        lambda base_dir=None, llm_fn=None: [
            _corpus_entry(0, "Ep A", "unrelated topic one", "/tmp/a"),
            _corpus_entry(1, "Ep B", "unrelated topic two", "/tmp/b"),
        ],
    )
    llm_fn = lambda prompt: json.dumps({"no_match": True, "episode_a_index": None, "episode_b_index": None, "shared_question": ""})

    assert thread_builder.build_thread(base_dir=str(tmp_path), llm_fn=llm_fn) is None


def test_build_thread_returns_none_when_corpus_has_fewer_than_two_episodes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        thread_builder, "build_corpus",
        lambda base_dir=None, llm_fn=None: [_corpus_entry(0, "Only Ep", "abstract", "/tmp/a")],
    )
    llm_fn = lambda prompt: pytest.fail("llm_fn should not be called for topic gate with < 2 episodes")

    assert thread_builder.build_thread(base_dir=str(tmp_path), llm_fn=llm_fn) is None


def test_build_thread_returns_full_shape_on_qualifying_pair(tmp_path, monkeypatch):
    run_dir_a = tmp_path / "a"
    run_dir_b = tmp_path / "b"
    run_dir_a.mkdir()
    run_dir_b.mkdir()
    transcript_a = {"duration": 100.0, "segments": [{"start": 0.0, "end": 30.0, "text": "hello from a"}]}
    transcript_b = {"duration": 100.0, "segments": [{"start": 0.0, "end": 30.0, "text": "hello from b"}]}
    (run_dir_a / "full_source.json").write_text(json.dumps(transcript_a))
    (run_dir_b / "full_source.json").write_text(json.dumps(transcript_b))

    monkeypatch.setattr(
        thread_builder, "build_corpus",
        lambda base_dir=None, llm_fn=None: [
            _corpus_entry(0, "Ep A", "argues remote work increases productivity", str(run_dir_a)),
            _corpus_entry(1, "Ep B", "argues remote work decreases productivity", str(run_dir_b)),
        ],
    )

    responses = [
        json.dumps({
            "no_match": False, "episode_a_index": 0, "episode_b_index": 1,
            "shared_question": "Does remote work increase or decrease productivity?",
        }),
        json.dumps({
            "grounded": True,
            "thesis": "Two guests, one question.",
            "bridge": "Here is the other side.",
            "clip_a": {"start_time": 5.0, "end_time": 25.0},
            "clip_b": {"start_time": 2.0, "end_time": 20.0},
        }),
    ]

    def llm_fn(prompt):
        return responses.pop(0)

    result = thread_builder.build_thread(base_dir=str(tmp_path), llm_fn=llm_fn)

    assert result == {
        "shared_question": "Does remote work increase or decrease productivity?",
        "thesis": "Two guests, one question.",
        "bridge": "Here is the other side.",
        "episode_a": {
            "run_dir": str(run_dir_a), "title": "Ep A", "source_url": "https://example.com/0",
            "start_time": 5.0, "end_time": 25.0,
        },
        "episode_b": {
            "run_dir": str(run_dir_b), "title": "Ep B", "source_url": "https://example.com/1",
            "start_time": 2.0, "end_time": 20.0,
        },
    }


def _topic_gate_llm_response():
    return json.dumps({
        "no_match": False, "episode_a_index": 0, "episode_b_index": 1,
        "shared_question": "Does remote work increase or decrease productivity?",
    })


def test_build_thread_returns_none_when_picked_run_dir_missing_full_source_json(tmp_path, monkeypatch):
    run_dir_a = tmp_path / "a"
    run_dir_b = tmp_path / "b"
    run_dir_a.mkdir()
    run_dir_b.mkdir()
    # run_dir_a has no full_source.json at all -- the open() should fail
    # and build_thread should refuse rather than raise.
    (run_dir_b / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))

    monkeypatch.setattr(
        thread_builder, "build_corpus",
        lambda base_dir=None, llm_fn=None: [
            _corpus_entry(0, "Ep A", "argues remote work increases productivity", str(run_dir_a)),
            _corpus_entry(1, "Ep B", "argues remote work decreases productivity", str(run_dir_b)),
        ],
    )
    llm_fn = lambda prompt: _topic_gate_llm_response()

    assert thread_builder.build_thread(base_dir=str(tmp_path), llm_fn=llm_fn) is None


def test_build_thread_returns_none_when_full_source_json_is_unparseable(tmp_path, monkeypatch):
    run_dir_a = tmp_path / "a"
    run_dir_b = tmp_path / "b"
    run_dir_a.mkdir()
    run_dir_b.mkdir()
    (run_dir_a / "full_source.json").write_text("{not valid json")
    (run_dir_b / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))

    monkeypatch.setattr(
        thread_builder, "build_corpus",
        lambda base_dir=None, llm_fn=None: [
            _corpus_entry(0, "Ep A", "argues remote work increases productivity", str(run_dir_a)),
            _corpus_entry(1, "Ep B", "argues remote work decreases productivity", str(run_dir_b)),
        ],
    )
    llm_fn = lambda prompt: _topic_gate_llm_response()

    assert thread_builder.build_thread(base_dir=str(tmp_path), llm_fn=llm_fn) is None


def test_build_thread_returns_none_when_transcript_shape_is_malformed(tmp_path, monkeypatch):
    run_dir_a = tmp_path / "a"
    run_dir_b = tmp_path / "b"
    run_dir_a.mkdir()
    run_dir_b.mkdir()
    # Valid JSON, but a segment missing "start"/"text" -- build_corpus never
    # validates transcript shape, only that the file parses as JSON, so this
    # sails through build_corpus and must be caught by build_thread itself.
    (run_dir_a / "full_source.json").write_text(json.dumps({
        "duration": 100.0,
        "segments": [{"end": 10.0}],
    }))
    (run_dir_b / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))

    monkeypatch.setattr(
        thread_builder, "build_corpus",
        lambda base_dir=None, llm_fn=None: [
            _corpus_entry(0, "Ep A", "argues remote work increases productivity", str(run_dir_a)),
            _corpus_entry(1, "Ep B", "argues remote work decreases productivity", str(run_dir_b)),
        ],
    )
    responses = [
        _topic_gate_llm_response(),
        json.dumps({
            "grounded": True, "thesis": "t", "bridge": "b",
            "clip_a": {"start_time": 5.0, "end_time": 25.0},
            "clip_b": {"start_time": 2.0, "end_time": 20.0},
        }),
    ]

    def llm_fn(prompt):
        return responses.pop(0)

    assert thread_builder.build_thread(base_dir=str(tmp_path), llm_fn=llm_fn) is None


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


def test_build_thread_returns_none_when_transcript_duration_is_null(tmp_path, monkeypatch):
    run_dir_a = tmp_path / "a"
    run_dir_b = tmp_path / "b"
    run_dir_a.mkdir()
    run_dir_b.mkdir()
    # "duration" key present but literally null -- transcript.get("duration",
    # 0.0) returns None (not the default), which used to blow up the
    # `if duration > 0:` comparison inside _sanitize_clip_span.
    (run_dir_a / "full_source.json").write_text(json.dumps({
        "duration": None,
        "segments": [{"start": 0.0, "end": 30.0, "text": "hello"}],
    }))
    (run_dir_b / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))

    monkeypatch.setattr(
        thread_builder, "build_corpus",
        lambda base_dir=None, llm_fn=None: [
            _corpus_entry(0, "Ep A", "argues remote work increases productivity", str(run_dir_a)),
            _corpus_entry(1, "Ep B", "argues remote work decreases productivity", str(run_dir_b)),
        ],
    )
    responses = [
        _topic_gate_llm_response(),
        json.dumps({
            "grounded": True, "thesis": "t", "bridge": "b",
            "clip_a": {"start_time": 5.0, "end_time": 25.0},
            "clip_b": {"start_time": 2.0, "end_time": 20.0},
        }),
    ]

    def llm_fn(prompt):
        return responses.pop(0)

    assert thread_builder.build_thread(base_dir=str(tmp_path), llm_fn=llm_fn) is None
