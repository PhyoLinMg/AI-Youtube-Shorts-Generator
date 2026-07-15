import pytest

import shorts_generator.webapp as webapp


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


def test_index_returns_the_dashboard_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b'id="run-form"' in resp.data
    assert b'id="url"' in resp.data
