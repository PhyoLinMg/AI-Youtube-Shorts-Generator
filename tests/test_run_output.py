import os

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
    assert os.path.isdir(paths.shorts_dir)


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
    assert content == "short 01 - Title One -- Come watch clip one.\nshort 02 - Title Two -- Come watch clip two.\n"


def test_write_descriptions_skips_failed_clips_without_renumbering(tmp_path):
    shorts = [
        {"clip_url": None, "title": "Failed", "error": "boom"},
        {"clip_url": "Short-02.mp4", "title": "Survivor", "description": "Come watch it."},
    ]
    path = run_output.write_descriptions(str(tmp_path), shorts)
    content = Path(path).read_text()
    assert content == "short 02 - Survivor -- Come watch it.\n"


def test_write_descriptions_empty_shorts_writes_empty_file(tmp_path):
    path = run_output.write_descriptions(str(tmp_path), [])
    assert Path(path).read_text() == ""


def test_write_descriptions_falls_back_on_missing_fields(tmp_path):
    shorts = [{"clip_url": "Short-01.mp4"}]
    path = run_output.write_descriptions(str(tmp_path), shorts)
    content = Path(path).read_text()
    assert content == "short 01 - Untitled -- \n"
