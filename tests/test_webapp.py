import os

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
