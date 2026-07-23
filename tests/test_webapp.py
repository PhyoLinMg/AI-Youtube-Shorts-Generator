import os
import shutil

import pytest

import shorts_generator.webapp as webapp
from shorts_generator.run_output import RunPaths, RunSummary


@pytest.fixture(autouse=True)
def reset_job():
    webapp.job.status = "idle"
    webapp.job.url = ""
    webapp.job.progress_log = None
    webapp.job.shorts_dir = None
    webapp.job.result = None
    webapp.job.error = None
    yield


@pytest.fixture
def client():
    webapp.app.testing = True
    return webapp.app.test_client()


def _fake_run_paths(tmp_path):
    root = str(tmp_path / "Video_Title")
    shorts_dir = os.path.join(root, "Shorts")
    os.makedirs(shorts_dir, exist_ok=True)
    return RunPaths(
        root=root,
        shorts_dir=shorts_dir,
        source_video=os.path.join(root, "full_source.mp4"),
        source_json=os.path.join(root, "full_source.json"),
        highlights_json=os.path.join(root, "highlights.json"),
        result_json=os.path.join(root, "result.json"),
        progress_log=os.path.join(root, "progress.log"),
    )


def test_index_returns_the_dashboard_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b'id="run-form"' in resp.data
    assert b'id="url"' in resp.data


class _SyncThread:
    """Runs the target synchronously in start(), for deterministic tests."""

    def __init__(self, target, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


def test_run_rejects_blank_url(client):
    resp = client.post("/run", data={"url": "  "})
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "url is required"


def test_run_rejects_concurrent_run(client):
    webapp.job.status = "running"
    resp = client.post("/run", data={"url": "https://youtube.example/x"})
    assert resp.status_code == 409


def test_run_starts_a_job_and_reaches_done(client, monkeypatch, tmp_path):
    fake_paths = _fake_run_paths(tmp_path)
    monkeypatch.setattr(webapp, "resolve_output_dir", lambda url: fake_paths)

    fake_result = {"mode": "api", "output_dir": fake_paths.root, "shorts": []}

    def _fake_generate_shorts(url, **kwargs):
        assert kwargs["paths"] is fake_paths
        return fake_result

    monkeypatch.setattr(webapp, "generate_shorts", _fake_generate_shorts)
    monkeypatch.setattr(webapp.threading, "Thread", _SyncThread)

    resp = client.post("/run", data={"url": "https://youtube.example/x"})
    assert resp.status_code == 202
    assert webapp.job.status == "done"
    assert webapp.job.result == fake_result
    assert webapp.job.progress_log == fake_paths.progress_log


def test_run_rejects_malformed_input_without_wedging_job_state(client, monkeypatch, tmp_path):
    resp = client.post("/run", data={"url": "https://youtube.example/x", "num_clips": "not-a-number"})
    assert resp.status_code == 400
    assert webapp.job.status == "idle"

    fake_paths = _fake_run_paths(tmp_path)
    monkeypatch.setattr(webapp, "resolve_output_dir", lambda url: fake_paths)
    monkeypatch.setattr(webapp, "generate_shorts", lambda url, **kwargs: {"mode": "api", "output_dir": fake_paths.root, "shorts": []})
    monkeypatch.setattr(webapp.threading, "Thread", _SyncThread)

    resp2 = client.post("/run", data={"url": "https://youtube.example/x"})
    assert resp2.status_code == 202
    assert webapp.job.status == "done"


def test_status_tails_the_progress_log_from_an_offset(client, tmp_path):
    log_path = tmp_path / "progress.log"
    log_path.write_text("line one\n")
    webapp.job.progress_log = str(log_path)
    webapp.job.status = "running"

    resp = client.get("/status?offset=0")
    data = resp.get_json()
    assert data["status"] == "running"
    assert data["log"] == "line one\n"
    first_offset = data["offset"]
    assert first_offset == len("line one\n".encode("utf-8"))

    with open(log_path, "a") as f:
        f.write("line two\n")

    resp = client.get(f"/status?offset={first_offset}")
    data = resp.get_json()
    assert data["log"] == "line two\n"


def test_status_serializes_local_clip_as_download_link(client, tmp_path):
    webapp.job.status = "done"
    webapp.job.shorts_dir = str(tmp_path)
    webapp.job.result = {
        "shorts": [
            {"title": "A", "score": 90, "hook_sentence": "hi", "clip_url": str(tmp_path / "Short-01.mp4")},
            {"title": "B", "score": 10, "clip_url": None, "error": "boom"},
        ]
    }

    resp = client.get("/status?offset=0")
    shorts = resp.get_json()["result"]["shorts"]
    assert shorts[0]["download_url"] == "/download/Short-01.mp4"
    assert shorts[1]["download_url"] is None
    assert shorts[1]["error"] == "boom"


def test_status_serializes_hosted_clip_url_unchanged(client, tmp_path):
    webapp.job.status = "done"
    webapp.job.shorts_dir = str(tmp_path)
    webapp.job.result = {
        "shorts": [
            {"title": "A", "score": 90, "clip_url": "https://hosted.example/Short-1.mp4"},
        ]
    }

    resp = client.get("/status?offset=0")
    shorts = resp.get_json()["result"]["shorts"]
    assert shorts[0]["download_url"] == "https://hosted.example/Short-1.mp4"


def test_safe_join_allows_files_inside_shorts_dir(tmp_path):
    shorts_dir = tmp_path / "Shorts"
    shorts_dir.mkdir()
    (shorts_dir / "Short-01.mp4").write_bytes(b"x")

    result = webapp._safe_join(str(shorts_dir), "Short-01.mp4")
    assert result == str(shorts_dir / "Short-01.mp4")


def test_safe_join_blocks_traversal_outside_shorts_dir(tmp_path):
    shorts_dir = tmp_path / "Shorts"
    shorts_dir.mkdir()
    (tmp_path / "secret.txt").write_bytes(b"secret")

    assert webapp._safe_join(str(shorts_dir), "../secret.txt") is None


def test_safe_join_rejects_a_null_byte_in_name(tmp_path):
    shorts_dir = tmp_path / "Shorts"
    shorts_dir.mkdir()

    assert webapp._safe_join(str(shorts_dir), "\x00.mp4") is None


def test_safe_join_blocks_sibling_directory_with_shared_prefix(tmp_path):
    shorts_dir = tmp_path / "Shorts"
    shorts_dir.mkdir()
    sibling = tmp_path / "Shorts_evil"
    sibling.mkdir()
    (sibling / "x.mp4").write_bytes(b"evil")

    assert webapp._safe_join(str(shorts_dir), "../Shorts_evil/x.mp4") is None


def test_download_serves_a_file_inside_shorts_dir(client, tmp_path):
    (tmp_path / "Short-01.mp4").write_bytes(b"video-bytes")
    webapp.job.shorts_dir = str(tmp_path)

    resp = client.get("/download/Short-01.mp4")
    assert resp.status_code == 200
    assert resp.data == b"video-bytes"


def test_download_404s_for_a_missing_file(client, tmp_path):
    webapp.job.shorts_dir = str(tmp_path)
    resp = client.get("/download/does-not-exist.mp4")
    assert resp.status_code == 404


def test_download_404s_when_no_job_has_run_yet(client):
    resp = client.get("/download/Short-01.mp4")
    assert resp.status_code == 404


def test_dashboard_entrypoint_exposes_the_flask_app():
    import dashboard
    assert dashboard.app is webapp.app


def test_history_returns_serialized_run_list(client, monkeypatch):
    fake_runs = [
        RunSummary(
            name="Video_A", mtime=100.0, source_exists=True,
            source_size=123, shorts_count=2, shorts_size=456,
        ),
    ]
    monkeypatch.setattr(webapp, "list_runs", lambda: fake_runs)

    resp = client.get("/history")
    assert resp.status_code == 200
    assert resp.get_json() == {
        "runs": [{
            "name": "Video_A", "mtime": 100.0, "source_exists": True,
            "source_size": 123, "shorts_count": 2, "shorts_size": 456,
        }]
    }


def test_delete_source_removes_the_file_and_returns_updated_summary(client, monkeypatch, tmp_path):
    root = tmp_path / "Video_A"
    root.mkdir()
    (root / "full_source.mp4").write_bytes(b"video-bytes")
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))

    resp = client.post("/history/Video_A/delete-source")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["source_exists"] is False
    assert data["source_size"] == 0
    assert not (root / "full_source.mp4").exists()


def test_delete_source_is_idempotent_when_already_gone(client, monkeypatch, tmp_path):
    root = tmp_path / "Video_A"
    root.mkdir()
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))

    resp = client.post("/history/Video_A/delete-source")
    assert resp.status_code == 200
    assert resp.get_json()["source_exists"] is False


def test_delete_source_404s_for_unknown_run(client, monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))
    resp = client.post("/history/does-not-exist/delete-source")
    assert resp.status_code == 404


def test_delete_source_rejects_while_a_run_is_in_progress(client, monkeypatch, tmp_path):
    root = tmp_path / "Video_A"
    root.mkdir()
    (root / "full_source.mp4").write_bytes(b"video-bytes")
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))
    webapp.job.status = "running"

    resp = client.post("/history/Video_A/delete-source")
    assert resp.status_code == 409
    assert (root / "full_source.mp4").exists()


def test_delete_source_survives_file_vanishing_between_check_and_remove(client, monkeypatch, tmp_path):
    """TOCTOU: if full_source.mp4 disappears between the isfile() check and
    os.remove() (e.g. a concurrent delete), the route must still respond
    cleanly instead of letting FileNotFoundError bubble up as a 500."""
    root = tmp_path / "Video_A"
    root.mkdir()
    (root / "full_source.mp4").write_bytes(b"video-bytes")
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))

    real_remove = os.remove

    def flaky_remove(path):
        # Simulate a concurrent actor deleting the file a moment before our
        # os.remove() call reaches the filesystem: the file is genuinely
        # gone by the time we get there, so the real os.remove() would also
        # raise FileNotFoundError.
        if os.path.exists(path):
            real_remove(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr(webapp.os, "remove", flaky_remove)

    resp = client.post("/history/Video_A/delete-source")
    assert resp.status_code == 200
    assert resp.get_json()["source_exists"] is False


def test_delete_source_404s_when_run_folder_vanishes_before_summarize(client, monkeypatch, tmp_path):
    """If `root` disappears between _resolve_history_run's isdir() check and
    the trailing summarize_run() call (e.g. a concurrent delete-shorts
    request removes the whole folder), summarize_run raises OSError — the
    route must turn that into a clean 404, not a 500."""
    root = tmp_path / "Video_A"
    root.mkdir()
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))

    def flaky_summarize_run(name, root_path):
        raise FileNotFoundError(root_path)

    monkeypatch.setattr(webapp, "summarize_run", flaky_summarize_run)

    resp = client.post("/history/Video_A/delete-source")
    assert resp.status_code == 404


def test_delete_shorts_removes_clips_but_keeps_descriptions(client, monkeypatch, tmp_path):
    root = tmp_path / "Video_A"
    shorts_dir = root / "Shorts"
    shorts_dir.mkdir(parents=True)
    (shorts_dir / "Short-01.mp4").write_bytes(b"clip-one")
    (shorts_dir / "Short-02.mp4").write_bytes(b"clip-two")
    (shorts_dir / "descriptions.txt").write_text("keep me")
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))

    resp = client.post("/history/Video_A/delete-shorts")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["shorts_count"] == 0
    assert data["shorts_size"] == 0
    assert not (shorts_dir / "Short-01.mp4").exists()
    assert not (shorts_dir / "Short-02.mp4").exists()
    assert (shorts_dir / "descriptions.txt").exists()


def test_delete_shorts_is_idempotent_when_already_gone(client, monkeypatch, tmp_path):
    root = tmp_path / "Video_A"
    root.mkdir()
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))

    resp = client.post("/history/Video_A/delete-shorts")
    assert resp.status_code == 200
    assert resp.get_json()["shorts_count"] == 0


def test_delete_shorts_rejects_while_a_run_is_in_progress(client, monkeypatch, tmp_path):
    root = tmp_path / "Video_A"
    shorts_dir = root / "Shorts"
    shorts_dir.mkdir(parents=True)
    (shorts_dir / "Short-01.mp4").write_bytes(b"clip-one")
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))
    webapp.job.status = "running"

    resp = client.post("/history/Video_A/delete-shorts")
    assert resp.status_code == 409
    assert (shorts_dir / "Short-01.mp4").exists()


def test_delete_shorts_survives_a_file_vanishing_mid_delete(client, monkeypatch, tmp_path):
    root = tmp_path / "Video_A"
    shorts_dir = root / "Shorts"
    shorts_dir.mkdir(parents=True)
    (shorts_dir / "Short-01.mp4").write_bytes(b"clip-one")
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))

    # We want our route's own os.remove call to raise even though the file
    # is genuinely gone by the time it runs — simulate by removing the file
    # out from under the route via a monkeypatched os.listdir race is fiddly,
    # so instead just monkeypatch os.remove to always raise FileNotFoundError
    # regardless of real state, proving the route tolerates it either way.
    monkeypatch.setattr(webapp.os, "remove", lambda path: (_ for _ in ()).throw(FileNotFoundError(path)))

    resp = client.post("/history/Video_A/delete-shorts")
    assert resp.status_code == 200


def test_delete_shorts_survives_the_shorts_dir_vanishing_before_listdir(client, monkeypatch, tmp_path):
    """TOCTOU: if the Shorts/ dir disappears between the isdir() check and
    the os.listdir() call (e.g. a concurrent whole-run delete), os.listdir()
    raises FileNotFoundError — the route must tolerate that instead of
    letting it bubble up as an uncaught 500."""
    root = tmp_path / "Video_A"
    shorts_dir = root / "Shorts"
    shorts_dir.mkdir(parents=True)
    (shorts_dir / "Short-01.mp4").write_bytes(b"clip-one")
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))

    real_listdir = os.listdir

    def rmtree_then_listdir(path):
        # Simulate a concurrent actor deleting the whole Shorts/ dir a
        # moment after our isdir() check passed but before our listdir()
        # call reaches the filesystem: by the time listdir() actually runs,
        # the directory is genuinely gone, so the real os.listdir() would
        # also raise FileNotFoundError.
        shutil.rmtree(str(shorts_dir), ignore_errors=True)
        return real_listdir(path)

    monkeypatch.setattr(webapp.os, "listdir", rmtree_then_listdir)

    resp = client.post("/history/Video_A/delete-shorts")
    assert resp.status_code == 200
    assert resp.get_json()["shorts_count"] == 0


def test_index_includes_history_tab_markup(client):
    resp = client.get("/")
    assert b'data-tab="history"' in resp.data
    assert b'id="tab-generate"' in resp.data
    assert b'id="tab-history"' in resp.data
    assert b'id="history-list"' in resp.data
    assert b'id="history-refresh"' in resp.data


def test_history_shorts_reads_result_json_when_present(client, monkeypatch, tmp_path):
    root = tmp_path / "Video_A"
    shorts_dir = root / "Shorts"
    shorts_dir.mkdir(parents=True)
    (shorts_dir / "My_Clip.mp4").write_bytes(b"clip-bytes")
    (root / "result.json").write_text(
        '{"shorts": [{"clip_url": "My_Clip.mp4", "title": "My Clip", "description": "watch this"}]}'
    )
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))

    resp = client.get("/history/Video_A/shorts")
    assert resp.status_code == 200
    shorts = resp.get_json()["shorts"]
    assert len(shorts) == 1
    assert shorts[0]["title"] == "My Clip"
    assert shorts[0]["description"] == "watch this"
    assert shorts[0]["download_url"] == "/history/Video_A/download/My_Clip.mp4"


def test_history_shorts_falls_back_to_clip_files_when_result_json_missing(client, monkeypatch, tmp_path):
    """A run that crashed after cropping but before result.json was written
    (e.g. the write_descriptions hashtags-list bug) still has real clips on
    disk — the endpoint must recover a title from the filename instead of
    reporting zero shorts."""
    root = tmp_path / "Video_A"
    shorts_dir = root / "Shorts"
    shorts_dir.mkdir(parents=True)
    (shorts_dir / "Some_Great_Clip.mp4").write_bytes(b"clip-bytes")
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))

    resp = client.get("/history/Video_A/shorts")
    assert resp.status_code == 200
    shorts = resp.get_json()["shorts"]
    assert len(shorts) == 1
    assert shorts[0]["title"] == "Some Great Clip"
    assert shorts[0]["download_url"] == "/history/Video_A/download/Some_Great_Clip.mp4"


def test_history_shorts_empty_when_result_json_and_shorts_dir_both_missing(client, monkeypatch, tmp_path):
    root = tmp_path / "Video_A"
    root.mkdir()
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))

    resp = client.get("/history/Video_A/shorts")
    assert resp.status_code == 200
    assert resp.get_json()["shorts"] == []


def test_history_shorts_404s_for_unknown_run(client, monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "LOCAL_OUTPUT_DIR", str(tmp_path))
    resp = client.get("/history/does-not-exist/shorts")
    assert resp.status_code == 404
