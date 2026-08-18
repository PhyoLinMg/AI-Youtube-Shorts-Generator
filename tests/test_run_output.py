import os
import shutil
from datetime import datetime

from shorts_generator import run_output


def test_sanitize_title_replaces_spaces_with_underscores():
    assert run_output.sanitize_title("How to Build a Startup") == "How_to_Build_a_Startup"


def test_sanitize_title_strips_unsafe_characters():
    assert run_output.sanitize_title("A/B: Test?!") == "A_B_Test"


def test_sanitize_title_empty_input_falls_back_to_untitled():
    assert run_output.sanitize_title("") == "untitled"
    assert run_output.sanitize_title("???") == "untitled"


def test_sanitize_title_truncates_long_titles():
    result = run_output.sanitize_title("x" * 150)
    assert len(result) == 100


def test_short_slug_lowercases_hyphenates_and_truncates():
    result = run_output.short_slug("Godfather of AI: We Have 2 Years Before Everything Changes!")
    assert result == "godfather-of-ai-we-have-2"


def test_short_slug_strips_unsafe_characters():
    assert run_output.short_slug("A/B: Test?!") == "a-b-test"


def test_short_slug_empty_input_falls_back_to_untitled():
    assert run_output.short_slug("") == "untitled"
    assert run_output.short_slug("???") == "untitled"


def test_unique_short_filename_slugifies_title():
    used = set()
    assert run_output.unique_short_filename("My Great Clip", used) == "My_Great_Clip.mp4"


def test_unique_short_filename_dedupes_repeated_titles():
    used = set()
    first = run_output.unique_short_filename("Same Title", used)
    second = run_output.unique_short_filename("Same Title", used)
    third = run_output.unique_short_filename("Same Title", used)
    assert [first, second, third] == ["Same_Title.mp4", "Same_Title_2.mp4", "Same_Title_3.mp4"]


def test_unique_short_filename_generic_style_ignores_title():
    used = set()
    first = run_output.unique_short_filename("Anything", used, index=1, style="generic")
    second = run_output.unique_short_filename("Anything Else", used, index=2, style="generic")
    assert [first, second] == ["video1.mp4", "video2.mp4"]


def test_unique_short_filename_generic_style_requires_index():
    used = set()
    try:
        run_output.unique_short_filename("Anything", used, style="generic")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_unique_chapter_filename_numbers_and_slugifies():
    used = set()
    assert run_output.unique_chapter_filename("The Big Reveal", 1, used) == "01_The_Big_Reveal.mp4"
    assert run_output.unique_chapter_filename("A Second Topic", 2, used) == "02_A_Second_Topic.mp4"


def test_unique_chapter_filename_pads_double_digit_index():
    used = set()
    assert run_output.unique_chapter_filename("Topic Ten", 10, used) == "10_Topic_Ten.mp4"


def test_unique_chapter_filename_dedupes_collisions():
    used = set()
    first = run_output.unique_chapter_filename("Same Title", 1, used)
    second = run_output.unique_chapter_filename("Same Title", 1, used)
    assert [first, second] == ["01_Same_Title.mp4", "01_Same_Title_2.mp4"]


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


def test_resolve_title_uses_oembed_title(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        assert "oembed" in url
        assert params["url"] == "https://www.youtube.com/watch?v=abc123"
        return _FakeResponse(200, {"title": "My Cool Video!"})

    monkeypatch.setattr(run_output.requests, "get", fake_get)
    assert run_output.resolve_title("https://www.youtube.com/watch?v=abc123") == "My Cool Video!"


def test_resolve_title_falls_back_on_oembed_network_error(monkeypatch):
    def fake_get(*args, **kwargs):
        raise run_output.requests.RequestException("network down")

    monkeypatch.setattr(run_output.requests, "get", fake_get)
    # Must fall back to the video id, not a constant like "watch" shared by
    # every watch-URL video — otherwise unrelated videos collide on rerun.
    title = run_output.resolve_title("https://www.youtube.com/watch?v=abc123")
    assert title == "abc123"


def test_resolve_title_falls_back_on_non_200(monkeypatch):
    monkeypatch.setattr(run_output.requests, "get", lambda *a, **k: _FakeResponse(404, {}))
    title = run_output.resolve_title("https://www.youtube.com/watch?v=abc123")
    assert title == "abc123"


def test_resolve_title_falls_back_to_video_id_for_youtu_be_link(monkeypatch):
    monkeypatch.setattr(run_output.requests, "get", lambda *a, **k: _FakeResponse(404, {}))
    title = run_output.resolve_title("https://youtu.be/xyz789")
    assert title == "xyz789"


def test_resolve_title_falls_back_to_path_stem_for_non_youtube_url(monkeypatch):
    monkeypatch.setattr(run_output.requests, "get", lambda *a, **k: _FakeResponse(404, {}))
    title = run_output.resolve_title("https://example.com/videos/my-clip.mp4")
    assert title == "my-clip"


def test_resolve_title_for_local_path_uses_filename_stem(tmp_path):
    media = tmp_path / "my_video_file.mp4"
    media.write_bytes(b"x")
    assert run_output.resolve_title(str(media)) == "my_video_file"


def test_resolve_output_dir_builds_expected_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_output.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"title": "How To Build A Startup"}),
    )

    paths = run_output.resolve_output_dir(
        "https://www.youtube.com/watch?v=abc123", base_dir=str(tmp_path)
    )

    assert paths.root == str(tmp_path / "How_To_Build_A_Startup")
    assert paths.shorts_dir == os.path.join(paths.root, "Shorts")
    assert paths.source_video == os.path.join(paths.root, "full_source.mp4")
    assert paths.source_json == os.path.join(paths.root, "full_source.json")
    assert paths.result_json == os.path.join(paths.root, "result.json")
    assert paths.progress_log == os.path.join(paths.root, "progress.log")
    assert paths.highlights_json == os.path.join(paths.root, "highlights.json")
    assert os.path.isdir(paths.shorts_dir)


def test_resolve_output_dir_builds_chapters_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_output.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"title": "How To Build A Startup"}),
    )

    paths = run_output.resolve_output_dir(
        "https://www.youtube.com/watch?v=abc123", base_dir=str(tmp_path)
    )

    assert paths.chapters_dir == os.path.join(paths.root, "Chapters")
    assert paths.chapters_json == os.path.join(paths.root, "chapters.json")
    assert os.path.isdir(paths.chapters_dir)


def test_resolve_output_dir_includes_source_url_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_output.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"title": "How To Build A Startup"}),
    )
    paths = run_output.resolve_output_dir(
        "https://www.youtube.com/watch?v=abc123", base_dir=str(tmp_path)
    )
    assert paths.source_url_txt == os.path.join(paths.root, "source_url.txt")


def test_write_source_url_then_read_source_url_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_output.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"title": "How To Build A Startup"}),
    )
    paths = run_output.resolve_output_dir(
        "https://www.youtube.com/watch?v=abc123", base_dir=str(tmp_path)
    )
    run_output.write_source_url(paths, "https://www.youtube.com/watch?v=abc123")
    assert run_output.read_source_url(paths) == "https://www.youtube.com/watch?v=abc123"


def test_read_source_url_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_output.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"title": "How To Build A Startup"}),
    )
    paths = run_output.resolve_output_dir(
        "https://www.youtube.com/watch?v=abc123", base_dir=str(tmp_path)
    )
    assert run_output.read_source_url(paths) is None


def test_resolve_output_dir_gives_chapters_its_own_result_json_path(tmp_path, monkeypatch):
    # Deliberately separate from result_json (used by generate_shorts) --
    # generate_shorts and generate_chapters can both run against the same
    # video, and sharing one result path means whichever runs second
    # silently clobbers the other's result.
    monkeypatch.setattr(
        run_output.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"title": "How To Build A Startup"}),
    )

    paths = run_output.resolve_output_dir(
        "https://www.youtube.com/watch?v=abc123", base_dir=str(tmp_path)
    )

    assert paths.chapters_result_json == os.path.join(paths.root, "chapters_result.json")
    assert paths.chapters_result_json != paths.result_json


def test_resolve_thread_run_dir_uses_date_and_short_slugs(tmp_path):
    fixed_now = datetime(2026, 8, 18, 14, 30, 0)
    result = run_output.resolve_thread_run_dir(
        "Episode A Title", "Episode B Title", base_dir=str(tmp_path), now=fixed_now,
    )
    assert result == str(tmp_path / "_Threads" / "2026-08-18_episode-a-title_x_episode-b-title")
    assert os.path.isdir(result)


def test_resolve_thread_run_dir_defaults_now_to_current_time(tmp_path):
    result = run_output.resolve_thread_run_dir("Episode A Title", "Episode B Title", base_dir=str(tmp_path))
    today = datetime.now().strftime("%Y-%m-%d")
    assert os.path.basename(result).startswith(today + "_")


def test_archive_stale_thread_run_returns_none_when_no_prior_run(tmp_path):
    out_dir = tmp_path / "thread"
    out_dir.mkdir()
    assert run_output.archive_stale_thread_run(str(out_dir)) is None


def test_archive_stale_thread_run_moves_prior_run_into_raw_stale(tmp_path):
    out_dir = tmp_path / "thread"
    (out_dir / "raw" / "thesis_1").mkdir(parents=True)
    (out_dir / "thread_results.json").write_text("[]")
    (out_dir / "descriptions.txt").write_text("d")
    (out_dir / "thesis_1_Old_Title.mp4").write_bytes(b"old final")
    (out_dir / "raw" / "thesis_1" / "clip_1_a.mp4").write_bytes(b"old raw")

    fixed_now = datetime(2026, 8, 18, 14, 30, 22)
    stale_dir = run_output.archive_stale_thread_run(str(out_dir), now=fixed_now)

    assert stale_dir == str(out_dir / "raw" / "stale" / "143022")
    assert not (out_dir / "thread_results.json").exists()
    assert not (out_dir / "thesis_1_Old_Title.mp4").exists()
    assert (Path(stale_dir) / "thread_results.json").exists()
    assert (Path(stale_dir) / "thesis_1_Old_Title.mp4").exists()
    assert (Path(stale_dir) / "thesis_1" / "clip_1_a.mp4").exists()
    assert set(os.listdir(out_dir)) == {"raw"}
    assert set(os.listdir(out_dir / "raw")) == {"stale"}


def test_archive_stale_thread_run_keeps_earlier_stale_archives(tmp_path):
    """A second same-day re-run must not clobber the first re-run's
    archive -- both timestamps should coexist under raw/stale/."""
    out_dir = tmp_path / "thread"
    (out_dir / "raw" / "stale" / "090000").mkdir(parents=True)
    (out_dir / "raw" / "stale" / "090000" / "old.mp4").write_bytes(b"first archive")
    (out_dir / "thread_results.json").write_text("[]")

    fixed_now = datetime(2026, 8, 18, 14, 30, 22)
    run_output.archive_stale_thread_run(str(out_dir), now=fixed_now)

    assert (out_dir / "raw" / "stale" / "090000" / "old.mp4").exists()
    assert (out_dir / "raw" / "stale" / "143022" / "thread_results.json").exists()


from pathlib import Path

import pytest


def test_capture_progress_log_duplicates_stdout_to_file(tmp_path, capsys):
    log_path = str(tmp_path / "progress.log")
    with run_output.capture_progress_log(log_path):
        print("hello from pipeline")

    captured = capsys.readouterr()
    assert "hello from pipeline" in captured.out

    content = Path(log_path).read_text()
    assert "hello from pipeline" in content
    assert "run start" in content


def test_capture_progress_log_records_failure_and_reraises(tmp_path):
    log_path = str(tmp_path / "progress.log")
    with pytest.raises(RuntimeError):
        with run_output.capture_progress_log(log_path):
            raise RuntimeError("boom")

    content = Path(log_path).read_text()
    assert "FAILED: boom" in content


def test_capture_progress_log_restores_stdout_after(tmp_path):
    import sys
    log_path = str(tmp_path / "progress.log")
    original_stdout = sys.stdout
    with run_output.capture_progress_log(log_path):
        pass
    assert sys.stdout is original_stdout


def test_capture_progress_log_appends_across_calls(tmp_path):
    log_path = str(tmp_path / "progress.log")
    with run_output.capture_progress_log(log_path):
        print("first run")
    with run_output.capture_progress_log(log_path):
        print("second run")

    content = Path(log_path).read_text()
    assert "first run" in content
    assert "second run" in content


def test_write_descriptions_formats_one_line_per_short(tmp_path):
    shorts_dir = str(tmp_path)
    shorts = [
        {"clip_url": "Short-01.mp4", "title": "Title One", "description": "Come watch clip one."},
        {"clip_url": "Short-02.mp4", "title": "Title Two", "description": "Come watch clip two."},
    ]
    path = run_output.write_descriptions(shorts_dir, shorts)
    content = Path(path).read_text()
    assert content == (
        "short 01 - Title One\nhook: 0  self-contained: no\nCome watch clip one.\n\n"
        "short 02 - Title Two\nhook: 0  self-contained: no\nCome watch clip two.\n"
    )


def test_write_descriptions_skips_failed_clips_without_renumbering(tmp_path):
    shorts = [
        {"clip_url": None, "title": "Failed", "error": "boom"},
        {"clip_url": "Short-02.mp4", "title": "Survivor", "description": "Come watch it."},
    ]
    path = run_output.write_descriptions(str(tmp_path), shorts)
    content = Path(path).read_text()
    assert content == "short 02 - Survivor\nhook: 0  self-contained: no\nCome watch it.\n"


def test_write_descriptions_empty_shorts_writes_empty_file(tmp_path):
    path = run_output.write_descriptions(str(tmp_path), [])
    assert Path(path).read_text() == ""


def test_write_descriptions_falls_back_on_missing_fields(tmp_path):
    shorts = [{"clip_url": "Short-01.mp4"}]
    path = run_output.write_descriptions(str(tmp_path), shorts)
    content = Path(path).read_text()
    assert content == "short 01 - Untitled\nhook: 0  self-contained: no\n\n"


def test_write_thread_descriptions_formats_one_block_per_clip(tmp_path):
    threads = [
        {"clip_url": "clip_1.mp4", "title": "Title One #Shorts", "description": "Watch clip one."},
        {"clip_url": "clip_2.mp4", "title": "Title Two #Shorts", "description": "Watch clip two."},
    ]
    path = run_output.write_thread_descriptions(str(tmp_path), threads)
    content = Path(path).read_text()
    assert content == (
        "clip 1 (clip_1.mp4)\nTitle: Title One #Shorts\nDescription: Watch clip one.\n\n"
        "clip 2 (clip_2.mp4)\nTitle: Title Two #Shorts\nDescription: Watch clip two.\n"
    )


def test_write_thread_descriptions_skips_ungrounded_threads_without_renumbering(tmp_path):
    threads = [
        {"clip_url": None, "shared_question": "no clip made"},
        {"clip_url": "clip_2.mp4", "title": "Title Two #Shorts", "description": "Watch clip two."},
    ]
    path = run_output.write_thread_descriptions(str(tmp_path), threads)
    content = Path(path).read_text()
    assert content == "clip 2 (clip_2.mp4)\nTitle: Title Two #Shorts\nDescription: Watch clip two.\n"


def test_write_thread_descriptions_falls_back_to_shared_question(tmp_path):
    threads = [{"clip_url": "clip_1.mp4", "shared_question": "Does X cause Y?", "description": "d"}]
    path = run_output.write_thread_descriptions(str(tmp_path), threads)
    content = Path(path).read_text()
    assert content == "clip 1 (clip_1.mp4)\nTitle: Does X cause Y?\nDescription: d\n"


def test_write_thread_descriptions_uses_actual_clip_url_basename(tmp_path):
    threads = [{
        "clip_url": "/some/output/_Threads/2026-08-18_a_x_b/thesis_1_Is_AI_a_threat.mp4",
        "title": "Is AI a threat? #Shorts", "description": "Watch both takes.",
    }]
    path = run_output.write_thread_descriptions(str(tmp_path), threads)
    content = Path(path).read_text()
    assert content == (
        "clip 1 (thesis_1_Is_AI_a_threat.mp4)\nTitle: Is AI a threat? #Shorts\nDescription: Watch both takes.\n"
    )


def test_write_descriptions_includes_hook_grade_line(tmp_path):
    shorts = [{
        "clip_url": "Short-01.mp4",
        "title": "Title One",
        "description": "Come watch clip one.",
        "hook_strength": 82,
        "hook_self_contained": True,
    }]
    path = run_output.write_descriptions(str(tmp_path), shorts)
    content = Path(path).read_text()
    lines = content.splitlines()
    assert lines[1] == "hook: 82  self-contained: yes"


def test_write_descriptions_appends_hashtags_list(tmp_path):
    shorts = [{
        "clip_url": "Short-01.mp4",
        "title": "Title One",
        "description": "Come watch clip one.",
        "yt_hashtags": ["#Shorts", "#topic"],
    }]
    path = run_output.write_descriptions(str(tmp_path), shorts)
    content = Path(path).read_text()
    assert "#Shorts #topic" in content


def test_write_descriptions_does_not_duplicate_hashtags_already_in_description(tmp_path):
    shorts = [{
        "clip_url": "Short-01.mp4",
        "title": "Title One",
        "description": "Come watch clip one. #Shorts #topic",
        "yt_hashtags": ["#Shorts", "#topic"],
    }]
    path = run_output.write_descriptions(str(tmp_path), shorts)
    content = Path(path).read_text()
    assert content.count("#Shorts") == 1


def test_write_chapter_descriptions_formats_one_block_per_chapter(tmp_path):
    chapters = [
        {
            "clip_url": "01_Topic_One.mp4", "title": "Topic One",
            "start_time": 12.5, "end_time": 340.0,
            "summary": "They discuss the origin of the idea and where it went wrong.",
        },
        {
            "clip_url": "02_Topic_Two.mp4", "title": "Topic Two",
            "start_time": 340.0, "end_time": 610.25,
            "summary": "A concrete example of the technique in practice.",
        },
    ]
    path = run_output.write_chapter_descriptions(str(tmp_path), chapters)
    assert Path(path).name == "chapters_description.txt"
    content = Path(path).read_text()
    assert content == (
        "chapter 01 - Topic One (12.5s - 340.0s)\n"
        "They discuss the origin of the idea and where it went wrong.\n\n"
        "chapter 02 - Topic Two (340.0s - 610.2s)\n"
        "A concrete example of the technique in practice.\n"
    )


def test_write_chapter_descriptions_skips_failed_clips_without_renumbering(tmp_path):
    chapters = [
        {"clip_url": None, "title": "Failed", "error": "boom"},
        {
            "clip_url": "02_Survivor.mp4", "title": "Survivor",
            "start_time": 0.0, "end_time": 60.0, "summary": "It made it through.",
        },
    ]
    path = run_output.write_chapter_descriptions(str(tmp_path), chapters)
    content = Path(path).read_text()
    assert content == "chapter 02 - Survivor (0.0s - 60.0s)\nIt made it through.\n"


def test_write_chapter_descriptions_empty_list_writes_empty_file(tmp_path):
    path = run_output.write_chapter_descriptions(str(tmp_path), [])
    assert Path(path).read_text() == ""


def test_write_chapter_descriptions_falls_back_on_missing_fields(tmp_path):
    chapters = [{"clip_url": "01_X.mp4"}]
    path = run_output.write_chapter_descriptions(str(tmp_path), chapters)
    content = Path(path).read_text()
    assert content == "chapter 01 - Untitled Chapter (0.0s - 0.0s)\n\n"


def _touch(path, mtime):
    with open(path, "w") as f:
        f.write("x")
    os.utime(path, (mtime, mtime))


def test_list_runs_on_missing_base_dir_returns_empty_list(tmp_path):
    missing = str(tmp_path / "does-not-exist")
    assert run_output.list_runs(missing) == []


def test_list_runs_ignores_non_directory_entries(tmp_path):
    (tmp_path / ".DS_Store").write_bytes(b"x")
    assert run_output.list_runs(str(tmp_path)) == []


def test_list_runs_reports_source_only(tmp_path):
    root = tmp_path / "Video_A"
    root.mkdir()
    _touch(str(root / "full_source.mp4"), 1000.0)

    runs = run_output.list_runs(str(tmp_path))
    assert len(runs) == 1
    run = runs[0]
    assert run.name == "Video_A"
    assert run.source_exists is True
    assert run.source_size == 1
    assert run.shorts_count == 0
    assert run.shorts_size == 0


def test_list_runs_reports_shorts_only_and_ignores_descriptions_file(tmp_path):
    root = tmp_path / "Video_B"
    shorts_dir = root / "Shorts"
    shorts_dir.mkdir(parents=True)
    _touch(str(shorts_dir / "Short-01.mp4"), 1000.0)
    _touch(str(shorts_dir / "Short-02.mp4"), 1000.0)
    (shorts_dir / "descriptions.txt").write_text("not a clip")

    runs = run_output.list_runs(str(tmp_path))
    assert len(runs) == 1
    run = runs[0]
    assert run.source_exists is False
    assert run.source_size == 0
    assert run.shorts_count == 2
    assert run.shorts_size == 2


def test_list_runs_reports_both_source_and_shorts(tmp_path):
    root = tmp_path / "Video_C"
    shorts_dir = root / "Shorts"
    shorts_dir.mkdir(parents=True)
    _touch(str(root / "full_source.mp4"), 1000.0)
    _touch(str(shorts_dir / "Short-01.mp4"), 1000.0)

    runs = run_output.list_runs(str(tmp_path))
    run = runs[0]
    assert run.source_exists is True
    assert run.shorts_count == 1


def test_list_runs_sorts_newest_first_by_file_mtime(tmp_path):
    older = tmp_path / "Older_Video"
    older.mkdir()
    _touch(str(older / "full_source.mp4"), 1000.0)

    newer = tmp_path / "Newer_Video"
    newer.mkdir()
    _touch(str(newer / "full_source.mp4"), 2000.0)

    runs = run_output.list_runs(str(tmp_path))
    assert [r.name for r in runs] == ["Newer_Video", "Older_Video"]


def test_summarize_run_returns_expected_shape_for_source_only_folder(tmp_path):
    root = tmp_path / "Video_D"
    root.mkdir()
    _touch(str(root / "full_source.mp4"), 1000.0)

    run = run_output.summarize_run("Video_D", str(root))

    assert isinstance(run, run_output.RunSummary)
    assert run.name == "Video_D"
    assert run.mtime == 1000.0
    assert run.source_exists is True
    assert run.source_size == 1
    assert run.shorts_count == 0
    assert run.shorts_size == 0


def test_list_runs_excludes_threads_directory(tmp_path):
    """resolve_thread_run_dir() creates output/_Threads/<slug>/ as a
    sibling of per-episode run folders -- list_runs() must not surface it
    as a bogus run row on the dashboard's History tab."""
    real_run = tmp_path / "Video_A"
    real_run.mkdir()
    _touch(str(real_run / "full_source.mp4"), 1000.0)

    threads_dir = tmp_path / "_Threads" / "episode_a_x_episode_b"
    threads_dir.mkdir(parents=True)
    _touch(str(threads_dir / "thread.mp4"), 2000.0)

    runs = run_output.list_runs(str(tmp_path))
    assert [r.name for r in runs] == ["Video_A"]


def test_list_runs_skips_folder_that_vanishes_mid_scan(tmp_path, monkeypatch):
    """A run folder deleted between the listdir() scan and its stat calls
    (e.g. a concurrent History-tab delete) must not crash the whole listing —
    it's just omitted, and every other run is still returned."""
    survivor = tmp_path / "Video_Survivor"
    survivor.mkdir()
    _touch(str(survivor / "full_source.mp4"), 1000.0)

    ghost = tmp_path / "Video_Ghost"
    ghost.mkdir()
    _touch(str(ghost / "full_source.mp4"), 2000.0)

    real_summarize_run = run_output.summarize_run

    def flaky_summarize_run(name, root):
        if name == "Video_Ghost":
            shutil.rmtree(root)  # simulate a concurrent delete mid-scan
        return real_summarize_run(name, root)

    monkeypatch.setattr(run_output, "summarize_run", flaky_summarize_run)

    runs = run_output.list_runs(str(tmp_path))
    assert [r.name for r in runs] == ["Video_Survivor"]
