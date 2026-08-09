import json
import os

from shorts_generator import corpus


def _write_episode(base_dir, name, duration=100.0, source_url="https://example.com/v1"):
    run_dir = os.path.join(base_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "full_source.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": duration, "segments": [{"start": 0.0, "end": 5.0, "text": "hello world"}]}, f)
    with open(os.path.join(run_dir, "source_url.txt"), "w", encoding="utf-8") as f:
        f.write(source_url)
    return run_dir


def test_list_corpus_run_dirs_requires_both_transcript_and_source_url(tmp_path):
    complete = _write_episode(str(tmp_path), "Complete_Episode")
    incomplete_dir = os.path.join(str(tmp_path), "Incomplete_Episode")
    os.makedirs(incomplete_dir, exist_ok=True)
    with open(os.path.join(incomplete_dir, "full_source.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": 50.0, "segments": []}, f)
    # no source_url.txt written for Incomplete_Episode

    run_dirs = corpus.list_corpus_run_dirs(base_dir=str(tmp_path))

    assert run_dirs == [complete]


def test_list_corpus_run_dirs_empty_base_dir_returns_empty_list(tmp_path):
    assert corpus.list_corpus_run_dirs(base_dir=str(tmp_path / "does_not_exist")) == []


def test_get_abstract_cached_calls_llm_once_then_reuses_cache(tmp_path):
    run_dir = _write_episode(str(tmp_path), "Episode_One")
    transcript = json.load(open(os.path.join(run_dir, "full_source.json")))

    calls = []
    def fake_llm(prompt):
        calls.append(prompt)
        return "an abstract about hello world"

    first = corpus.get_abstract_cached(run_dir, transcript, llm_fn=fake_llm)
    second = corpus.get_abstract_cached(run_dir, transcript, llm_fn=fake_llm)

    assert first == "an abstract about hello world"
    assert second == "an abstract about hello world"
    assert len(calls) == 1


def test_get_abstract_cached_invalidates_on_transcript_change(tmp_path):
    run_dir = _write_episode(str(tmp_path), "Episode_Two")
    transcript_v1 = json.load(open(os.path.join(run_dir, "full_source.json")))

    calls = []
    def fake_llm(prompt):
        calls.append(prompt)
        return f"abstract {len(calls)}"

    corpus.get_abstract_cached(run_dir, transcript_v1, llm_fn=fake_llm)

    transcript_v2 = {**transcript_v1, "segments": [{"start": 0.0, "end": 5.0, "text": "a different episode entirely"}]}
    result = corpus.get_abstract_cached(run_dir, transcript_v2, llm_fn=fake_llm)

    assert result == "abstract 2"
    assert len(calls) == 2


def test_build_corpus_returns_title_source_url_and_abstract(tmp_path):
    _write_episode(str(tmp_path), "My_Episode", source_url="https://example.com/my-episode")

    entries = corpus.build_corpus(base_dir=str(tmp_path), llm_fn=lambda prompt: "summary text")

    assert len(entries) == 1
    assert entries[0]["title"] == "My_Episode"
    assert entries[0]["source_url"] == "https://example.com/my-episode"
    assert entries[0]["abstract"] == "summary text"
    assert entries[0]["run_dir"].endswith("My_Episode")
