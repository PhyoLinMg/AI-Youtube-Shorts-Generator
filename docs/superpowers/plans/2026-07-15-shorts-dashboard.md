# Shorts Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local Flask dashboard where a user pastes a YouTube link (plus the full CLI option set), submits, watches a live progress log, then sees the resulting shorts — successful ones with an inline preview and download link, failed ones marked as failed.

**Architecture:** `shorts_generator/webapp.py` holds one module-level `Job` (guarded by a `threading.Lock`, since this is a single-user, one-run-at-a-time tool) plus three routes: `POST /run` starts a background thread, `GET /status?offset=N` tails `progress.log` from a byte offset and reports job state, `GET /download/<name>` serves a finished clip. `generate_shorts()` gains an optional `paths` param so the background thread can resolve `RunPaths` itself (learning `progress.log`'s location) before the pipeline starts writing to it.

**Tech Stack:** Flask (new dep, `requirements-web.txt`), vanilla JS (no build step, single template file), Python `threading`.

---

## Task 1: `generate_shorts()` accepts an optional `paths` param

**Files:**
- Modify: `shorts_generator/pipeline.py:162-212` (the `generate_shorts` function signature and its first two lines)
- Test: `tests/test_pipeline.py`

**Why:** `generate_shorts()` currently always resolves `RunPaths` itself (`pipeline.py:211`), and that resolution does a network oEmbed call. The dashboard's background thread needs to resolve paths *itself*, publish `progress_log`'s path into job state, and only then hand those same paths to `generate_shorts()` — otherwise the web layer never learns where `progress.log` lives, and the pipeline would redundantly resolve paths (and hit oEmbed) twice.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py` (it already imports `pipeline_module`, `RunPaths`, `local_downloader_module`, `local_transcriber_module`, `local_clipper_module`, and has a `_paths(tmp_path)` helper and `_fake_transcript()` / `_fake_highlights_result()` — reuse them):

```python
def test_generate_shorts_uses_provided_paths_without_resolving(tmp_path, monkeypatch):
    paths = _paths(tmp_path)

    def _fail_resolve(*a, **k):
        raise AssertionError("resolve_output_dir should not be called when paths is provided")
    monkeypatch.setattr(pipeline_module, "resolve_output_dir", _fail_resolve)

    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", Mock(return_value=[]))

    result = pipeline_module.generate_shorts(
        "https://youtube.example/x",
        mode="local",
        paths=paths,
    )

    assert result["output_dir"] == paths.root
    assert os.path.exists(paths.progress_log)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_pipeline.py::test_generate_shorts_uses_provided_paths_without_resolving -v`
Expected: FAIL with `TypeError: generate_shorts() got an unexpected keyword argument 'paths'`

- [ ] **Step 3: Implement the `paths` param**

In `shorts_generator/pipeline.py`, change the `generate_shorts` signature (currently ending `framing: str = "locked",\n) -> Dict:`) to add the new param, and change its body's first two lines (currently `paths = resolve_output_dir(youtube_url)`):

```python
def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "720",
    language: Optional[str] = None,
    mode: str = "api",
    captions: bool = True,
    caption_fade_duration: float = 0.3,
    word_highlight: bool = True,
    framing: str = "locked",
    paths: Optional[RunPaths] = None,
) -> Dict:
```

and in the body:

```python
    paths = paths or resolve_output_dir(youtube_url)
    with capture_progress_log(paths.progress_log):
```

Also add one line to the docstring's Args section, right after the `framing` entry:

```python
        paths: pre-resolved RunPaths to use instead of resolving them from
            youtube_url. Callers that need to know progress_log's path before
            the pipeline starts (e.g. a background job) should resolve it
            themselves and pass it here.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: PASS (all tests in the file, including the new one)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/pipeline.py tests/test_pipeline.py
git commit -m "feat: let generate_shorts() accept pre-resolved RunPaths"
```

---

## Task 2: Add the Flask dependency

**Files:**
- Create: `requirements-web.txt`

- [ ] **Step 1: Write the file**

```
-r requirements.txt

# Web dashboard (dashboard.py) — optional.
flask>=3.0
```

- [ ] **Step 2: Install it and verify it's importable**

Run: `.venv/bin/pip install -r requirements-web.txt && .venv/bin/python -c "import flask; print(flask.__version__)"`
Expected: prints a version like `3.x.x`, no errors

- [ ] **Step 3: Commit**

```bash
git add requirements-web.txt
git commit -m "feat: add requirements-web.txt for the Flask dashboard"
```

---

## Task 3: App skeleton, `Job` state, `GET /`

**Files:**
- Create: `shorts_generator/webapp.py`
- Create: `shorts_generator/templates/index.html`
- Test: `tests/test_webapp.py`

**Why:** Establishes the module-level `Job` dataclass and lock that every later route reads/writes, plus the one page the browser loads.

- [ ] **Step 1: Write the failing test**

Create `tests/test_webapp.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_webapp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shorts_generator.webapp'`

- [ ] **Step 3: Implement the app skeleton**

Create `shorts_generator/webapp.py`:

```python
"""Flask dashboard: submit a YouTube URL, watch progress, grab the shorts.

Single-user, local tool — at most one pipeline run at a time. State lives in
the module-level `job` object, guarded by `_job_lock` since a background
thread and Flask's request threads touch it concurrently. This one-run-at-a-
time constraint is load-bearing: `capture_progress_log` (run_output.py) swaps
sys.stdout/sys.stderr process-globally, not per-thread, so two concurrent
runs would interleave each other's progress logs.
"""
import os
import threading
from dataclasses import dataclass
from typing import Optional

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from .pipeline import generate_shorts
from .run_output import resolve_output_dir

app = Flask(__name__)


@dataclass
class Job:
    status: str = "idle"  # "idle" | "starting" | "running" | "done" | "failed"
    url: str = ""
    progress_log: Optional[str] = None
    shorts_dir: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


job = Job()
_job_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")
```

Create `shorts_generator/templates/index.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AI Shorts Dashboard</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
    h1 { font-size: 1.4rem; }
    fieldset { border: 1px solid #ddd; border-radius: 8px; margin-bottom: 1.5rem; }
    label { display: block; margin: 0.5rem 0 0.2rem; font-size: 0.85rem; color: #444; }
    input[type=text], input[type=number], select { width: 100%; padding: 0.4rem; box-sizing: border-box; }
    .row { display: flex; gap: 1rem; }
    .row > div { flex: 1; }
    button { margin-top: 1rem; padding: 0.6rem 1.2rem; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    #log { background: #111; color: #0f0; font-family: monospace; font-size: 0.8rem; padding: 0.75rem; height: 260px; overflow-y: auto; white-space: pre-wrap; border-radius: 6px; }
    #error-banner { background: #fee; border: 1px solid #f99; color: #900; padding: 0.6rem; border-radius: 6px; margin-bottom: 1rem; display: none; }
    .short-card { border: 1px solid #ddd; border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem; }
    .short-card video { width: 100%; max-width: 260px; display: block; margin: 0.5rem 0; }
    .short-card.failed { border-color: #f99; background: #fff5f5; }
    .status-line { font-size: 0.85rem; color: #666; margin-bottom: 0.5rem; }
  </style>
</head>
<body>
  <h1>AI Shorts Dashboard</h1>

  <div id="error-banner"></div>

  <form id="run-form">
    <fieldset>
      <label for="url">YouTube URL</label>
      <input type="text" id="url" name="url" placeholder="https://www.youtube.com/watch?v=..." required>

      <div class="row">
        <div>
          <label for="mode">Mode</label>
          <select id="mode" name="mode">
            <option value="api">api</option>
            <option value="local">local</option>
          </select>
        </div>
        <div>
          <label for="num_clips">Num clips</label>
          <input type="number" id="num_clips" name="num_clips" value="3" min="1">
        </div>
      </div>

      <div class="row">
        <div>
          <label for="aspect_ratio">Aspect ratio</label>
          <input type="text" id="aspect_ratio" name="aspect_ratio" value="9:16">
        </div>
        <div>
          <label for="format">Download resolution</label>
          <select id="format" name="format">
            <option value="360">360</option>
            <option value="480">480</option>
            <option value="720" selected>720</option>
            <option value="1080">1080</option>
          </select>
        </div>
      </div>

      <div class="row">
        <div>
          <label for="language">Language (blank = auto-detect)</label>
          <input type="text" id="language" name="language" placeholder="en">
        </div>
        <div>
          <label for="framing">Framing (local mode only)</label>
          <select id="framing" name="framing">
            <option value="locked">locked</option>
            <option value="adaptive">adaptive</option>
          </select>
        </div>
      </div>

      <div class="row">
        <div>
          <label><input type="checkbox" id="captions" name="captions" checked> Burn captions</label>
        </div>
        <div>
          <label><input type="checkbox" id="word_highlight" name="word_highlight" checked> Word highlight</label>
        </div>
        <div>
          <label for="caption_fade_duration">Caption fade (s)</label>
          <input type="number" id="caption_fade_duration" name="caption_fade_duration" value="0.3" step="0.1" min="0">
        </div>
      </div>

      <button type="submit" id="submit-btn">Generate shorts</button>
    </fieldset>
  </form>

  <div class="status-line" id="status-line">idle</div>
  <pre id="log"></pre>

  <div id="results"></div>

  <script>
    const form = document.getElementById("run-form");
    const submitBtn = document.getElementById("submit-btn");
    const statusLine = document.getElementById("status-line");
    const logEl = document.getElementById("log");
    const resultsEl = document.getElementById("results");
    const errorBanner = document.getElementById("error-banner");

    let offset = 0;

    function setBusy(busy) {
      submitBtn.disabled = busy;
    }

    function showError(message) {
      errorBanner.textContent = message;
      errorBanner.style.display = message ? "block" : "none";
    }

    function renderResults(result) {
      resultsEl.innerHTML = "";
      if (!result || !result.shorts) return;
      for (const s of result.shorts) {
        const card = document.createElement("div");
        card.className = "short-card" + (s.download_url ? "" : " failed");
        if (s.download_url) {
          card.innerHTML =
            "<strong>" + (s.title || "Untitled") + "</strong> (score " + s.score + ")" +
            "<div>" + (s.hook_sentence || "") + "</div>" +
            "<video controls src=\"" + s.download_url + "\"></video>" +
            "<a href=\"" + s.download_url + "\" download>Download</a>";
        } else {
          card.innerHTML =
            "<strong>" + (s.title || "Untitled") + "</strong> — failed" +
            "<div>" + (s.error || "unknown error") + "</div>";
        }
        resultsEl.appendChild(card);
      }
    }

    async function poll() {
      try {
        const resp = await fetch("/status?offset=" + offset);
        if (!resp.ok) {
          showError("lost connection to server");
          return;
        }
        showError("");
        const data = await resp.json();
        if (data.log) {
          logEl.textContent += data.log;
          logEl.scrollTop = logEl.scrollHeight;
        }
        offset = data.offset;
        statusLine.textContent = data.status;

        if (data.status === "done") {
          setBusy(false);
          renderResults(data.result);
        } else if (data.status === "failed") {
          setBusy(false);
          showError(data.error || "run failed");
        } else if (data.status === "starting" || data.status === "running") {
          setBusy(true);
        }
      } catch (e) {
        showError("lost connection to server");
      }
    }

    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      showError("");
      logEl.textContent = "";
      resultsEl.innerHTML = "";
      offset = 0;
      const formData = new FormData(form);
      formData.set("captions", document.getElementById("captions").checked ? "true" : "false");
      formData.set("word_highlight", document.getElementById("word_highlight").checked ? "true" : "false");

      const resp = await fetch("/run", { method: "POST", body: formData });
      const data = await resp.json();
      if (!resp.ok) {
        showError(data.error || "failed to start run");
        return;
      }
      setBusy(true);
      statusLine.textContent = data.status;
    });

    poll();
    setInterval(poll, 5000);
  </script>
</body>
</html>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_webapp.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/webapp.py shorts_generator/templates/index.html tests/test_webapp.py
git commit -m "feat: add dashboard app skeleton and page"
```

---

## Task 4: `POST /run`

**Files:**
- Modify: `shorts_generator/webapp.py`
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_webapp.py`:

```python
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
```

Add this helper near the top of `tests/test_webapp.py`, alongside the fixtures:

```python
from shorts_generator.run_output import RunPaths


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
```

and add `import os` at the top of the file if not already present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_webapp.py -v`
Expected: the three new tests FAIL — `/run` doesn't exist yet, so `client.post("/run", ...)` returns 404, not the expected status codes.

- [ ] **Step 3: Implement `POST /run`**

Add to `shorts_generator/webapp.py`, after the `Job`/`job`/`_job_lock` definitions and before `index()`:

```python
def _run_job(
    url: str,
    mode: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    captions: bool,
    caption_fade_duration: float,
    word_highlight: bool,
    framing: str,
) -> None:
    try:
        paths = resolve_output_dir(url)
        with _job_lock:
            job.progress_log = paths.progress_log
            job.shorts_dir = paths.shorts_dir
            job.status = "running"
        result = generate_shorts(
            url,
            num_clips=num_clips,
            aspect_ratio=aspect_ratio,
            download_format=download_format,
            language=language,
            mode=mode,
            captions=captions,
            caption_fade_duration=caption_fade_duration,
            word_highlight=word_highlight,
            framing=framing,
            paths=paths,
        )
        with _job_lock:
            job.result = result
            job.status = "done"
    except Exception as e:
        with _job_lock:
            job.error = str(e)
            job.status = "failed"
```

Add after `index()`:

```python
@app.route("/run", methods=["POST"])
def start_run():
    url = request.form.get("url", "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    with _job_lock:
        if job.status in ("starting", "running"):
            return jsonify({"error": "a run is already in progress"}), 409
        job.status = "starting"
        job.url = url
        job.progress_log = None
        job.shorts_dir = None
        job.result = None
        job.error = None

    kwargs = dict(
        mode=request.form.get("mode", "api"),
        num_clips=int(request.form.get("num_clips", 3)),
        aspect_ratio=request.form.get("aspect_ratio", "9:16"),
        download_format=request.form.get("format", "720"),
        language=(request.form.get("language") or "").strip() or None,
        captions=request.form.get("captions", "true") == "true",
        caption_fade_duration=float(request.form.get("caption_fade_duration", 0.3)),
        word_highlight=request.form.get("word_highlight", "true") == "true",
        framing=request.form.get("framing", "locked"),
    )
    threading.Thread(target=_run_job, args=(url,), kwargs=kwargs, daemon=True).start()
    return jsonify({"status": "starting"}), 202
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_webapp.py -v`
Expected: PASS (all tests, including Task 3's)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/webapp.py tests/test_webapp.py
git commit -m "feat: add POST /run to start a background pipeline job"
```

---

## Task 5: `GET /status`

**Files:**
- Modify: `shorts_generator/webapp.py`
- Test: `tests/test_webapp.py`

**Why:** Reports job status and tails `progress.log` from a byte offset. Also translates each short's `clip_url` into something the browser can actually fetch — for local-mode shorts (and captioned api-mode shorts) `clip_url` is a local file path that must be proxied through `/download`, but for api-mode shorts with captions off or caption burn-in failed, `clip_url` is MuAPI's own hosted URL and the browser can fetch that directly (see `shorts_generator/clipper.py:66,83-87`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_webapp.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_webapp.py -v`
Expected: the three new tests FAIL — `/status` doesn't exist yet (404).

- [ ] **Step 3: Implement `GET /status`**

Add to `shorts_generator/webapp.py`, after `start_run()`:

```python
def _clip_display_url(shorts_dir: Optional[str], clip_url: Optional[str]) -> Optional[str]:
    if not clip_url:
        return None
    if clip_url.startswith("http://") or clip_url.startswith("https://"):
        return clip_url
    return f"/download/{os.path.basename(clip_url)}"


def _serialize_result(result: dict, shorts_dir: Optional[str]) -> dict:
    shorts = [
        {**s, "download_url": _clip_display_url(shorts_dir, s.get("clip_url"))}
        for s in result.get("shorts", [])
    ]
    return {**result, "shorts": shorts}


@app.route("/status")
def status():
    offset = int(request.args.get("offset", 0))
    with _job_lock:
        current_status = job.status
        progress_log = job.progress_log
        shorts_dir = job.shorts_dir
        result = job.result
        error = job.error

    log_text = ""
    new_offset = offset
    if progress_log and os.path.exists(progress_log):
        with open(progress_log, "rb") as f:
            f.seek(offset)
            chunk = f.read()
            new_offset = f.tell()
        log_text = chunk.decode("utf-8", errors="replace")

    return jsonify({
        "status": current_status,
        "log": log_text,
        "offset": new_offset,
        "result": _serialize_result(result, shorts_dir) if result else None,
        "error": error,
    })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_webapp.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/webapp.py tests/test_webapp.py
git commit -m "feat: add GET /status to tail progress and report results"
```

---

## Task 6: `GET /download/<name>`

**Files:**
- Modify: `shorts_generator/webapp.py`
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_webapp.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_webapp.py -v`
Expected: the five new tests FAIL — `_safe_join` doesn't exist (`AttributeError`) and `/download/...` 404s for the wrong reason (no route).

- [ ] **Step 3: Implement `_safe_join` and `GET /download/<name>`**

Add to `shorts_generator/webapp.py`, after `_serialize_result`:

```python
def _safe_join(base_dir: str, name: str) -> Optional[str]:
    """Resolve `name` under `base_dir`, refusing to escape it (blocks '../')."""
    base_real = os.path.realpath(base_dir)
    target = os.path.realpath(os.path.join(base_real, name))
    if target == base_real or target.startswith(base_real + os.sep):
        return target
    return None


@app.route("/download/<path:name>")
def download(name):
    with _job_lock:
        shorts_dir = job.shorts_dir
    if not shorts_dir:
        abort(404)
    target = _safe_join(shorts_dir, name)
    if not target or not os.path.isfile(target):
        abort(404)
    return send_from_directory(os.path.dirname(target), os.path.basename(target))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_webapp.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/webapp.py tests/test_webapp.py
git commit -m "feat: add GET /download/<name> with a path-traversal guard"
```

---

## Task 7: `dashboard.py` entry point

**Files:**
- Create: `dashboard.py`
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_webapp.py`:

```python
def test_dashboard_entrypoint_exposes_the_flask_app():
    import dashboard
    assert dashboard.app is webapp.app
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_webapp.py::test_dashboard_entrypoint_exposes_the_flask_app -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dashboard'`

- [ ] **Step 3: Implement the entry point**

Create `dashboard.py` (repo root, alongside `main.py`):

```python
"""Entry point for the local dashboard.

Usage:
    python dashboard.py

Then open http://127.0.0.1:5000 in a browser.
"""
from shorts_generator.webapp import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
```

The `if __name__ == "__main__":` guard is what makes the test safe — importing
`dashboard` as a module (as the test does) never calls `app.run()`, only
running it directly (`python dashboard.py`) does.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_webapp.py -v`
Expected: PASS (full file)

- [ ] **Step 5: Commit**

```bash
git add dashboard.py tests/test_webapp.py
git commit -m "feat: add dashboard.py entry point"
```

---

## Task 8: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a Dashboard section**

Insert a new section right after the `### API mode vs Local mode` table (ends at the `| Required keys | ... |` line, README.md:212) and before `## How It Works` (README.md:214):

```markdown
## Dashboard (Web UI)

Prefer a browser to a terminal? Install the extra dependency and run:

```bash
pip install -r requirements-web.txt
python dashboard.py
```

Then open `http://127.0.0.1:5000`. Paste a YouTube URL, set any options
(mode, num clips, aspect ratio, captions, etc. — the same knobs as the CLI
flags below), and submit. A live progress log streams while the pipeline
runs; once it finishes, each short shows an inline preview with a download
button (or, for a failed clip, the error that killed it).

Single-user, one run at a time — starting a new run while one is in progress
returns an error until the current one finishes.

```

- [ ] **Step 2: Update the Project Structure block**

In the `## Project Structure` section (README.md:297-317), add these lines. After `├── requirements-local.txt        optional deps for --mode local` add:

```
├── requirements-web.txt          optional deps for the dashboard (dashboard.py)
├── dashboard.py                   dashboard entry point (python dashboard.py)
```

And inside the `shorts_generator/` tree, after `├── pipeline.py               mode dispatcher (api ↔ local)` add:

```
    ├── webapp.py                 Flask dashboard: job state + routes
    ├── templates/
    │   └── index.html            dashboard page (form, live log, results)
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document the web dashboard"
```

---

## Task 9: Full suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `.venv/bin/pytest -v`
Expected: PASS, no failures, no errors — includes every pre-existing test plus everything added in Tasks 1–7.

- [ ] **Step 2: Manually smoke-test the dashboard**

Run: `.venv/bin/python dashboard.py` (leave running), then in another terminal:
`curl -s http://127.0.0.1:5000/ | grep -o 'id="run-form"'`
Expected: prints `id="run-form"`, confirming the page serves. Stop the server (Ctrl-C) when done — do not leave it running.
