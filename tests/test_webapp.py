import os

import pytest

import shorts_generator.webapp as webapp
from shorts_generator.run_output import RunPaths


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
