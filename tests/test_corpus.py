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


def test_sample_transcript_text_samples_beyond_the_opening(tmp_path):
    # Build a transcript with exactly one (oversized) segment per third, and
    # a distinctive marker as the entire closing-third segment. Head
    # truncation of the whole transcript would never reach it; even
    # sampling across thirds should return it intact.
    marker = "UNIQUE_CLOSING_MARKER_XYZ"
    segments = [
        {"start": 0.0, "end": 1.0, "text": "filler " * 2000},
        {"start": 1.0, "end": 2.0, "text": "filler " * 2000},
        {"start": 2.0, "end": 3.0, "text": marker},
    ]
    transcript = {"duration": 3.0, "segments": segments}

    sample = corpus._sample_transcript_text(transcript, max_chars=6000)

    assert marker in sample


def test_build_corpus_skips_corrupted_transcript_and_keeps_valid_episode(tmp_path):
    valid = _write_episode(str(tmp_path), "Valid_Episode", source_url="https://example.com/valid")
    corrupted_dir = os.path.join(str(tmp_path), "Corrupted_Episode")
    os.makedirs(corrupted_dir, exist_ok=True)
    with open(os.path.join(corrupted_dir, "full_source.json"), "w", encoding="utf-8") as f:
        f.write("{not valid json")
    with open(os.path.join(corrupted_dir, "source_url.txt"), "w", encoding="utf-8") as f:
        f.write("https://example.com/corrupted")

    entries = corpus.build_corpus(base_dir=str(tmp_path), llm_fn=lambda prompt: "summary text")

    assert len(entries) == 1
    assert entries[0]["run_dir"] == valid


def test_list_corpus_run_dirs_excludes_empty_source_url(tmp_path):
    _write_episode(str(tmp_path), "Has_Url")
    empty_dir = os.path.join(str(tmp_path), "Empty_Url")
    os.makedirs(empty_dir, exist_ok=True)
    with open(os.path.join(empty_dir, "full_source.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": 10.0, "segments": []}, f)
    with open(os.path.join(empty_dir, "source_url.txt"), "w", encoding="utf-8") as f:
        f.write("   \n")

    run_dirs = corpus.list_corpus_run_dirs(base_dir=str(tmp_path))

    assert len(run_dirs) == 1
    assert run_dirs[0].endswith("Has_Url")


def test_build_corpus_skips_invalid_utf8_transcript_and_keeps_valid_episode(tmp_path):
    valid = _write_episode(str(tmp_path), "Valid_Episode", source_url="https://example.com/valid")
    bad_dir = os.path.join(str(tmp_path), "Bad_Bytes_Episode")
    os.makedirs(bad_dir, exist_ok=True)
    # Invalid UTF-8 byte sequence -- not valid JSON either, but the point is
    # it must fail at the decode step (UnicodeDecodeError), not JSONDecodeError.
    with open(os.path.join(bad_dir, "full_source.json"), "wb") as f:
        f.write(b'{"duration": 1.0, "segments": [{"text": "\xff\xfe bad bytes"}]}')
    with open(os.path.join(bad_dir, "source_url.txt"), "w", encoding="utf-8") as f:
        f.write("https://example.com/bad-bytes")

    entries = corpus.build_corpus(base_dir=str(tmp_path), llm_fn=lambda prompt: "summary text")

    assert len(entries) == 1
    assert entries[0]["run_dir"] == valid


def test_list_corpus_run_dirs_skips_invalid_utf8_source_url(tmp_path):
    valid = _write_episode(str(tmp_path), "Valid_Episode")
    bad_dir = os.path.join(str(tmp_path), "Bad_Bytes_Url")
    os.makedirs(bad_dir, exist_ok=True)
    with open(os.path.join(bad_dir, "full_source.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": 10.0, "segments": []}, f)
    with open(os.path.join(bad_dir, "source_url.txt"), "wb") as f:
        f.write(b"\xff\xfe not valid utf-8")

    run_dirs = corpus.list_corpus_run_dirs(base_dir=str(tmp_path))

    assert run_dirs == [valid]


def test_build_corpus_skips_episode_when_llm_fn_raises(tmp_path):
    # Simulates a real live-LLM failure mode (network error, rate limit)
    # during abstract generation for one episode among many -- this must
    # not abort the whole corpus build, only skip that one episode.
    valid = _write_episode(str(tmp_path), "A_Valid_Episode", source_url="https://example.com/valid")
    _write_episode(str(tmp_path), "B_Failing_Episode", source_url="https://example.com/failing")

    # list_corpus_run_dirs visits run dirs in sorted (alphabetical) order,
    # so A_Valid_Episode is processed first and B_Failing_Episode second.
    calls = []
    def flaky_llm(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("simulated network error")
        return "summary text"

    entries = corpus.build_corpus(base_dir=str(tmp_path), llm_fn=flaky_llm)

    assert len(entries) == 1
    assert entries[0]["run_dir"] == valid
