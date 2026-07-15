# History Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "History" tab to the dashboard listing every `output/<Title>/` run folder (name, date, source-video size, shorts count/size) with per-run delete buttons for the source video and the shorts clips.

**Architecture:** A new pure-filesystem `list_runs()`/`summarize_run()` pair in `run_output.py` (stat + directory listing only — never parses the multi-MB `result.json`), three new Flask routes in `webapp.py` (`GET /history`, `POST /history/<name>/delete-source`, `POST /history/<name>/delete-shorts`) guarded by the existing one-run-at-a-time lock and the existing `_safe_join` traversal guard, and a tab bar + history panel added to the single-page `index.html` template, fetched on tab switch (no polling — it's a static listing).

**Tech Stack:** Python 3, Flask, pytest, vanilla JS/CSS (no build step, matches the existing dashboard).

Spec: `docs/superpowers/specs/2026-07-15-history-tab-design.md`

---

### Task 1: `RunSummary` + `list_runs()` in `run_output.py`

**Files:**
- Modify: `shorts_generator/run_output.py:133-137`
- Test: `tests/test_run_output.py` (append at end)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_run_output.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run_output.py -k list_runs -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.run_output' has no attribute 'list_runs'`

- [ ] **Step 3: Implement `RunSummary`, `summarize_run()`, `list_runs()`**

In `shorts_generator/run_output.py`, find this block (end of `resolve_output_dir`, right before `write_descriptions`):

```python
        progress_log=os.path.join(root, "progress.log"),
    )


def write_descriptions(shorts_dir: str, shorts: List[Dict]) -> str:
```

Replace it with:

```python
        progress_log=os.path.join(root, "progress.log"),
    )


@dataclass
class RunSummary:
    name: str
    mtime: float
    source_exists: bool
    source_size: int
    shorts_count: int
    shorts_size: int


def _run_mtime(root: str) -> float:
    """Newest mtime across every file in `root` (falls back to the dir's own)."""
    mtimes = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            try:
                mtimes.append(os.path.getmtime(path))
            except OSError:
                continue
    return max(mtimes) if mtimes else os.path.getmtime(root)


def summarize_run(name: str, root: str) -> RunSummary:
    """Stat a single run folder — no result.json parsing (it can be several MB)."""
    source_video = os.path.join(root, "full_source.mp4")
    source_exists = os.path.isfile(source_video)
    source_size = os.path.getsize(source_video) if source_exists else 0

    shorts_dir = os.path.join(root, "Shorts")
    shorts_names = []
    if os.path.isdir(shorts_dir):
        shorts_names = sorted(
            n for n in os.listdir(shorts_dir)
            if n.startswith("Short-") and n.endswith(".mp4")
        )
    shorts_size = sum(os.path.getsize(os.path.join(shorts_dir, n)) for n in shorts_names)

    return RunSummary(
        name=name,
        mtime=_run_mtime(root),
        source_exists=source_exists,
        source_size=source_size,
        shorts_count=len(shorts_names),
        shorts_size=shorts_size,
    )


def list_runs(base_dir: Optional[str] = None) -> List[RunSummary]:
    """List every run folder under `base_dir`, newest first."""
    base_dir = base_dir or LOCAL_OUTPUT_DIR
    if not os.path.isdir(base_dir):
        return []
    runs = [
        summarize_run(name, os.path.join(base_dir, name))
        for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ]
    runs.sort(key=lambda r: r.mtime, reverse=True)
    return runs


def write_descriptions(shorts_dir: str, shorts: List[Dict]) -> str:
```

(`List`, `Dict`, `Optional`, `dataclass`, and `LOCAL_OUTPUT_DIR` are all already imported at the top of this file — no new imports needed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_output.py -k list_runs -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py
git commit -m "feat: add list_runs() for scanning past output/ run folders"
```

---

### Task 2: `GET /history` route

**Files:**
- Modify: `shorts_generator/webapp.py:10-22` (imports), append route at end of file
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_webapp.py`, change the import line:

```python
from shorts_generator.run_output import RunPaths
```

to:

```python
from shorts_generator.run_output import RunPaths, RunSummary
```

Then append this test:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_webapp.py -k test_history_returns_serialized_run_list -v`
Expected: FAIL with 404 (no `/history` route yet)

- [ ] **Step 3: Add imports and the route**

In `shorts_generator/webapp.py`, replace the top of the file:

```python
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from typing import Optional

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from .pipeline import generate_shorts
from .run_output import resolve_output_dir

app = Flask(__name__)
```

with:

```python
import os
import sys
import threading
import traceback
from dataclasses import asdict, dataclass
from typing import Optional

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from .config import LOCAL_OUTPUT_DIR
from .pipeline import generate_shorts
from .run_output import list_runs, resolve_output_dir, summarize_run

app = Flask(__name__)
```

(`summarize_run` and `LOCAL_OUTPUT_DIR` aren't used until Task 3, but importing them now avoids touching this block again.)

Then append at the end of the file (after the `/download/<path:name>` route):

```python
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


@app.route("/history")
def history():
    return jsonify({"runs": [asdict(r) for r in list_runs()]})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_webapp.py -k test_history_returns_serialized_run_list -v`
Expected: PASS

Also run the full webapp test file to make sure the import/route changes didn't break anything existing:

Run: `pytest tests/test_webapp.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/webapp.py tests/test_webapp.py
git commit -m "feat: add GET /history route listing past runs"
```

---

### Task 3: Shared guard helper + `POST /history/<name>/delete-source`

**Files:**
- Modify: `shorts_generator/webapp.py` (append at end of file)
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_webapp.py`:

```python
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
```

(Path-traversal protection for `name` is already covered by the existing `_safe_join` unit tests — `_resolve_history_run` below reuses that same function, so it isn't re-tested at the route level, matching how `/download/<path:name>` is tested today.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_webapp.py -k delete_source -v`
Expected: FAIL with 404 (no route yet) on all four

- [ ] **Step 3: Implement the guard helper and the route**

In `shorts_generator/webapp.py`, find the `/history` route added in Task 2:

```python
@app.route("/history")
def history():
    return jsonify({"runs": [asdict(r) for r in list_runs()]})
```

Replace it with (keeping the original route and adding the helper + new route after it):

```python
@app.route("/history")
def history():
    return jsonify({"runs": [asdict(r) for r in list_runs()]})


def _resolve_history_run(name: str):
    """Validate `name` as a run folder under LOCAL_OUTPUT_DIR.

    Returns `(root, None)` on success or `(None, error_response)` when the
    request should be rejected — mirrors the one-run-at-a-time guard already
    enforced by `POST /run`, since deleting files out from under an active
    pipeline run would corrupt it.
    """
    with _job_lock:
        active = job.status in ("starting", "running")
    if active:
        return None, (jsonify({"error": "a run is in progress"}), 409)
    root = _safe_join(LOCAL_OUTPUT_DIR, name)
    if not root or not os.path.isdir(root):
        return None, (jsonify({"error": "run not found"}), 404)
    return root, None


@app.route("/history/<name>/delete-source", methods=["POST"])
def delete_history_source(name):
    root, error = _resolve_history_run(name)
    if error:
        return error
    source_video = os.path.join(root, "full_source.mp4")
    if os.path.isfile(source_video):
        os.remove(source_video)
    return jsonify(asdict(summarize_run(name, root)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_webapp.py -k delete_source -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/webapp.py tests/test_webapp.py
git commit -m "feat: add POST /history/<name>/delete-source route"
```

---

### Task 4: `POST /history/<name>/delete-shorts`

**Files:**
- Modify: `shorts_generator/webapp.py` (append at end of file)
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_webapp.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_webapp.py -k delete_shorts -v`
Expected: FAIL with 404 (no route yet) on all three

- [ ] **Step 3: Implement the route**

In `shorts_generator/webapp.py`, find the `delete_history_source` route added in Task 3:

```python
@app.route("/history/<name>/delete-source", methods=["POST"])
def delete_history_source(name):
    root, error = _resolve_history_run(name)
    if error:
        return error
    source_video = os.path.join(root, "full_source.mp4")
    if os.path.isfile(source_video):
        os.remove(source_video)
    return jsonify(asdict(summarize_run(name, root)))
```

Replace it with (keeping the original route and adding the new route after it):

```python
@app.route("/history/<name>/delete-source", methods=["POST"])
def delete_history_source(name):
    root, error = _resolve_history_run(name)
    if error:
        return error
    source_video = os.path.join(root, "full_source.mp4")
    if os.path.isfile(source_video):
        os.remove(source_video)
    return jsonify(asdict(summarize_run(name, root)))


@app.route("/history/<name>/delete-shorts", methods=["POST"])
def delete_history_shorts(name):
    root, error = _resolve_history_run(name)
    if error:
        return error
    shorts_dir = os.path.join(root, "Shorts")
    if os.path.isdir(shorts_dir):
        for filename in os.listdir(shorts_dir):
            if filename.startswith("Short-") and filename.endswith(".mp4"):
                os.remove(os.path.join(shorts_dir, filename))
    return jsonify(asdict(summarize_run(name, root)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_webapp.py -k delete_shorts -v`
Expected: 3 passed

Run the whole webapp test file once more since this is the last backend change:

Run: `pytest tests/test_webapp.py tests/test_run_output.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/webapp.py tests/test_webapp.py
git commit -m "feat: add POST /history/<name>/delete-shorts route"
```

---

### Task 5: Tab bar + History panel markup and CSS

**Files:**
- Modify: `shorts_generator/templates/index.html`
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_webapp.py`:

```python
def test_index_includes_history_tab_markup(client):
    resp = client.get("/")
    assert b'data-tab="history"' in resp.data
    assert b'id="tab-generate"' in resp.data
    assert b'id="tab-history"' in resp.data
    assert b'id="history-list"' in resp.data
    assert b'id="history-refresh"' in resp.data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_webapp.py -k test_index_includes_history_tab_markup -v`
Expected: FAIL (markup not present yet)

- [ ] **Step 3: Add the CSS**

In `shorts_generator/templates/index.html`, find:

```css
    .card-error {
      font-size: 0.78rem;
      color: var(--muted);
    }
  </style>
```

Replace with:

```css
    .card-error {
      font-size: 0.78rem;
      color: var(--muted);
    }

    /* --- tabs --- */
    .tabs { display: flex; gap: 0.5rem; margin-bottom: 1.25rem; }
    .tab-btn {
      width: auto;
      margin-top: 0;
      font-family: var(--mono);
      font-size: 0.78rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      background: transparent;
      color: var(--muted);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 0.45rem 1rem;
      cursor: pointer;
    }
    .tab-btn.is-active { color: var(--text); border-color: var(--text); }

    /* --- history --- */
    .history-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1rem;
    }
    .history-refresh-btn {
      width: auto;
      margin-top: 0;
      padding: 0.35rem 0.8rem;
      font-family: var(--mono);
      font-size: 0.72rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      background: transparent;
      color: var(--muted);
      border: 1px solid var(--line);
      border-radius: 3px;
    }
    .history-refresh-btn:hover { border-color: var(--text); color: var(--text); }
    .history-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.6rem 1.2rem;
      padding: 0.7rem 0;
      border-bottom: 1px solid var(--line);
      font-family: var(--mono);
      font-size: 0.8rem;
    }
    .history-row:last-child { border-bottom: none; }
    .history-name { flex: 1 1 220px; font-weight: 600; color: var(--text); }
    .history-date { color: var(--muted); }
    .history-item { display: flex; align-items: center; gap: 0.5rem; }
    .history-delete-btn {
      width: auto;
      margin-top: 0;
      padding: 0.25rem 0.6rem;
      font-family: var(--mono);
      font-size: 0.68rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      background: transparent;
      color: var(--warn);
      border: 1px solid var(--line);
      border-radius: 3px;
      cursor: pointer;
    }
    .history-delete-btn:hover { border-color: var(--warn); }
    .history-delete-btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .history-empty { color: var(--muted); font-size: 0.85rem; }
  </style>
```

- [ ] **Step 4: Add the tab bar and give `<main>` an id**

Find:

```html
  <div id="error-banner" role="alert"></div>

  <main class="layout">
    <section class="panel frame">
```

Replace with:

```html
  <div id="error-banner" role="alert"></div>

  <nav class="tabs">
    <button type="button" class="tab-btn is-active" data-tab="generate">Generate</button>
    <button type="button" class="tab-btn" data-tab="history">History</button>
  </nav>

  <main class="layout" id="tab-generate">
    <section class="panel frame">
```

- [ ] **Step 5: Add the History panel after `</main>`**

Find:

```html
      </section>
    </section>
  </main>

  <script>
```

Replace with:

```html
      </section>
    </section>
  </main>

  <section class="panel frame" id="tab-history" hidden>
    <div class="history-header">
      <h2 class="eyebrow">History</h2>
      <button type="button" id="history-refresh" class="history-refresh-btn">Refresh</button>
    </div>
    <div id="history-list"></div>
  </section>

  <script>
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_webapp.py -k test_index_includes_history_tab_markup -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add shorts_generator/templates/index.html tests/test_webapp.py
git commit -m "feat: add History tab markup and styles to the dashboard"
```

---

### Task 6: Tab switching + History fetch/delete JS

**Files:**
- Modify: `shorts_generator/templates/index.html`

There's no JS test runner in this repo (the existing dashboard JS — polling, copy buttons, form submit — isn't unit tested either; it's verified by hand in a browser). This task is verified manually in Task 7.

- [ ] **Step 1: Add the tab/history JS**

In `shorts_generator/templates/index.html`, find the end of the `<script>` block:

```javascript
      } catch (e) {
        showError("failed to start run");
      }
    });

    poll();
  </script>
```

Replace with:

```javascript
      } catch (e) {
        showError("failed to start run");
      }
    });

    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabGenerate = document.getElementById("tab-generate");
    const tabHistory = document.getElementById("tab-history");
    const historyList = document.getElementById("history-list");
    const historyRefreshBtn = document.getElementById("history-refresh");

    function formatBytes(bytes) {
      if (!bytes) return "0 B";
      const units = ["B", "KB", "MB", "GB"];
      let value = bytes;
      let i = 0;
      while (value >= 1024 && i < units.length - 1) {
        value /= 1024;
        i++;
      }
      return value.toFixed(i === 0 ? 0 : 1) + " " + units[i];
    }

    function formatDate(epochSeconds) {
      return new Date(epochSeconds * 1000).toLocaleString();
    }

    function updateHistoryItem(labelEl, btnEl, kind, run) {
      if (kind === "source") {
        labelEl.textContent = run.source_exists
          ? "Source: " + formatBytes(run.source_size)
          : "Source: deleted";
        btnEl.disabled = !run.source_exists;
      } else {
        labelEl.textContent = run.shorts_count > 0
          ? "Shorts: " + run.shorts_count + " clips, " + formatBytes(run.shorts_size)
          : "Shorts: none";
        btnEl.disabled = run.shorts_count === 0;
      }
    }

    async function deleteHistoryFiles(name, action, kind, labelEl, btnEl) {
      const what = kind === "source" ? "the source video" : "the shorts clips";
      if (!confirm("Delete " + what + " for \"" + name + "\"? This can't be undone.")) return;
      btnEl.disabled = true;
      try {
        const resp = await fetch("/history/" + encodeURIComponent(name) + "/" + action, { method: "POST" });
        const data = await resp.json();
        if (!resp.ok) {
          showError(data.error || "delete failed");
          btnEl.disabled = false;
          return;
        }
        showError("");
        updateHistoryItem(labelEl, btnEl, kind, data);
      } catch (e) {
        showError("lost connection to server");
        btnEl.disabled = false;
      }
    }

    function renderHistoryRow(run) {
      const row = document.createElement("div");
      row.className = "history-row";

      const nameEl = document.createElement("div");
      nameEl.className = "history-name";
      nameEl.textContent = run.name;
      row.appendChild(nameEl);

      const dateEl = document.createElement("div");
      dateEl.className = "history-date";
      dateEl.textContent = formatDate(run.mtime);
      row.appendChild(dateEl);

      const sourceItem = document.createElement("div");
      sourceItem.className = "history-item";
      const sourceLabel = document.createElement("span");
      sourceItem.appendChild(sourceLabel);
      const sourceBtn = document.createElement("button");
      sourceBtn.type = "button";
      sourceBtn.className = "history-delete-btn";
      sourceBtn.textContent = "Delete";
      sourceItem.appendChild(sourceBtn);
      updateHistoryItem(sourceLabel, sourceBtn, "source", run);
      sourceBtn.addEventListener("click", () => deleteHistoryFiles(run.name, "delete-source", "source", sourceLabel, sourceBtn));
      row.appendChild(sourceItem);

      const shortsItem = document.createElement("div");
      shortsItem.className = "history-item";
      const shortsLabel = document.createElement("span");
      shortsItem.appendChild(shortsLabel);
      const shortsBtn = document.createElement("button");
      shortsBtn.type = "button";
      shortsBtn.className = "history-delete-btn";
      shortsBtn.textContent = "Delete";
      shortsItem.appendChild(shortsBtn);
      updateHistoryItem(shortsLabel, shortsBtn, "shorts", run);
      shortsBtn.addEventListener("click", () => deleteHistoryFiles(run.name, "delete-shorts", "shorts", shortsLabel, shortsBtn));
      row.appendChild(shortsItem);

      return row;
    }

    async function loadHistory() {
      historyList.innerHTML = "";
      try {
        const resp = await fetch("/history");
        const data = await resp.json();
        if (!data.runs || data.runs.length === 0) {
          const empty = document.createElement("div");
          empty.className = "history-empty";
          empty.textContent = "No runs yet.";
          historyList.appendChild(empty);
          return;
        }
        for (const run of data.runs) {
          historyList.appendChild(renderHistoryRow(run));
        }
      } catch (e) {
        showError("failed to load history");
      }
    }

    function setActiveTab(tab) {
      for (const btn of tabButtons) {
        btn.classList.toggle("is-active", btn.dataset.tab === tab);
      }
      tabGenerate.hidden = tab !== "generate";
      tabHistory.hidden = tab !== "history";
      if (tab === "history") loadHistory();
    }

    for (const btn of tabButtons) {
      btn.addEventListener("click", () => setActiveTab(btn.dataset.tab));
    }
    historyRefreshBtn.addEventListener("click", loadHistory);

    poll();
  </script>
```

- [ ] **Step 2: Run the full test suite to make sure nothing broke**

Run: `pytest -v`
Expected: all passed

- [ ] **Step 3: Commit**

```bash
git add shorts_generator/templates/index.html
git commit -m "feat: wire up History tab fetch/delete behavior in the dashboard"
```

---

### Task 7: Manual browser verification

**Do not touch or delete anything under the real `output/Neil_deGrasse_Tyson_The_Whistleblowers_Were_Right_About_Aliens/` folder during this task** — it holds real generated shorts. Verify against a disposable dummy run folder instead.

- [ ] **Step 1: Create a disposable dummy run folder**

```bash
mkdir -p output/_manual_test_run/Shorts
head -c 1000000 /dev/urandom > output/_manual_test_run/full_source.mp4
head -c 500000 /dev/urandom > output/_manual_test_run/Shorts/Short-01.mp4
head -c 300000 /dev/urandom > output/_manual_test_run/Shorts/Short-02.mp4
echo "short 01 - test -- test" > output/_manual_test_run/Shorts/descriptions.txt
du -sh output/_manual_test_run/full_source.mp4 output/_manual_test_run/Shorts
```

Note the printed sizes for comparison in Step 3.

- [ ] **Step 2: Start the dashboard**

```bash
python dashboard.py
```

Open `http://127.0.0.1:5050` in a browser.

- [ ] **Step 3: Verify the History tab list**

Click "History". Confirm:
- `_manual_test_run` appears in the list with a recent date.
- Its source size and shorts size (2 clips) roughly match the `du -sh` output from Step 1.
- The real Neil deGrasse Tyson run also appears in the list, untouched.

- [ ] **Step 4: Delete the dummy run's shorts, then its source**

Click "Delete" next to `_manual_test_run`'s Shorts entry. Confirm the browser `confirm()` dialog appears; accept it. Verify:
- The row updates to "Shorts: none" without a page reload.
- `output/_manual_test_run/Shorts/Short-01.mp4` and `Short-02.mp4` are gone from disk; `descriptions.txt` is still there.

Click "Delete" next to the Source entry. Verify:
- The row updates to "Source: deleted".
- `output/_manual_test_run/full_source.mp4` is gone from disk.

- [ ] **Step 5: Verify idempotent delete in the live server**

Both delete buttons for `_manual_test_run` were already clicked in Step 4, so its files are already gone. Confirm clicking "Delete" again on either (still-visible, now-disabled per Step 4) doesn't apply — instead hit the route directly to confirm the no-op-success behavior end to end:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:5050/history/_manual_test_run/delete-source
```

Expected: `200` — deleting an already-deleted source is a no-op success, not an error. (The 409-while-running guard itself is already covered by the automated tests in Tasks 3-4, which monkeypatch `job.status` directly — that state isn't reachable from outside the server process, so it isn't re-verified manually here.)

- [ ] **Step 6: Clean up the dummy run folder**

```bash
rm -rf output/_manual_test_run
```

Stop the dashboard (Ctrl-C).

- [ ] **Step 7: Final full test suite run**

Run: `pytest -v`
Expected: all passed
