# Shorts dashboard — design

## Problem

Today the only way to run the pipeline is the CLI (`python main.py <url> ...`).
There's no way to kick off a run, watch its progress, or grab the resulting
shorts without a terminal and a filesystem browser.

## Goal

A local web dashboard: paste a YouTube link (plus the full set of CLI
options), submit, watch a live progress log while the pipeline runs, then see
the resulting shorts — successful ones with a preview and download button,
failed ones marked as failed.

Single-user, local tool, one run at a time. No auth, no run history browsing
(shows whatever is currently running or most recently finished — closing and
reopening the page loses the "most recent" view since state is in-memory,
which is acceptable for this tool's scope).

## Prerequisite fix: `generate_shorts()` needs an optional `paths` param

`generate_shorts()` currently calls `resolve_output_dir(youtube_url)`
*internally* (`pipeline.py:211`), which does a network oEmbed call and only
then creates `output/<Title>/` and starts writing `progress.log`. The web
layer needs to know `progress.log`'s path *before* the pipeline finishes, and
resolving it is itself a network call that must not block the HTTP request
thread.

Fix: add `paths: Optional[RunPaths] = None` to `generate_shorts()`. When
`None` (the CLI's case, unchanged), it resolves paths internally exactly as
today. The background job resolves paths itself first (inside the background
thread, not the request handler), publishes `progress_log`'s path into job
state, and only then calls `generate_shorts(url, paths=paths, ...)`.

```python
def generate_shorts(youtube_url: str, ..., paths: Optional[RunPaths] = None) -> Dict:
    ...
    paths = paths or resolve_output_dir(youtube_url)
    with capture_progress_log(paths.progress_log):
        ...
```

## Architecture

- **`shorts_generator/webapp.py`** — Flask app: routes, job state, background
  thread launcher.
- **`shorts_generator/templates/index.html`** — single-page dashboard: form,
  `<pre>` log viewer, results section. Inline `<script>`/`<style>` in the one
  file — no build step, no separate static bundle.
- **`dashboard.py`** (repo root, mirrors `main.py`) — entry point:
  ```python
  from shorts_generator.webapp import app
  app.run(host="127.0.0.1", port=5000)
  ```
- **New dependency:** Flask. Added via a new **`requirements-web.txt`**
  (`-r requirements.txt` + `flask`), following the existing
  `requirements-local.txt` convention — CLI-only users don't need it.

## Job state

One module-level job object guarded by a `threading.Lock`, since this is a
single-user, one-run-at-a-time tool:

```python
@dataclass
class Job:
    status: str  # "idle" | "starting" | "running" | "done" | "failed"
    url: str = ""
    progress_log: Optional[str] = None   # set once paths are resolved
    shorts_dir: Optional[str] = None     # set once paths are resolved
    result: Optional[dict] = None        # set on "done"
    error: Optional[str] = None          # set on "failed"
```

`status="starting"` covers the window between "thread launched" and "paths
resolved" (the oEmbed call) — the log viewer shows "starting…" during this
window since there's no file to tail yet.

**Threading note:** `capture_progress_log` swaps `sys.stdout`/`sys.stderr`
process-globally, not per-thread. That's safe *only* because at most one job
runs at a time — this constraint is load-bearing and must not be relaxed
(e.g. to support concurrent runs) without redesigning progress capture.
Werkzeug's own request logging is unaffected since its handler holds a
reference to the original stderr from before any swap.

## Routes

- **`GET /`** — renders `index.html`. No server-side state injected; the page
  JS calls `/status` on load to pick up whatever's in progress or finished.

- **`POST /run`** — form fields: `url` (required), `mode` (api/local),
  `num_clips`, `aspect_ratio`, `format`, `language`, `captions` (bool),
  `caption_fade_duration`, `word_highlight` (bool), `framing`
  (locked/adaptive) — the full CLI option set.
  - 400 if `url` is blank.
  - 409 if a job is already `starting` or `running`.
  - Otherwise resets job state to `status="starting"`, spawns a background
    thread, returns 202 immediately (does not block on oEmbed or the
    pipeline).

- **`GET /status?offset=N`** — JSON:
  ```json
  {
    "status": "running",
    "log": "<new progress.log content since byte offset N>",
    "offset": 1234,
    "result": null,
    "error": null
  }
  ```
  Reads `progress_log` from byte `N` to EOF (returns `"log": ""` and
  unchanged `offset` while `status="starting"`, since the file doesn't exist
  yet). Once `status="done"`, `result` carries the shorts list; once
  `status="failed"`, `error` carries the message.

- **`GET /download/<path:name>`** — serves a file from the *current job's*
  `shorts_dir` only. Resolves `os.path.realpath(os.path.join(shorts_dir,
  name))` and 404s unless it's still inside `shorts_dir` (blocks `../`
  traversal), then 404s if the file doesn't exist, else
  `send_from_directory`.

## Frontend behavior

- Form submit → `fetch POST /run`. On 409, show "a run is already in
  progress" instead of submitting. Disable the submit button while
  `status` is `starting`/`running`.
- Poll `GET /status?offset=<lastOffset>` every 5s (per chosen interval).
  Append `log` to the `<pre>` viewer, auto-scroll to bottom, update
  `lastOffset`.
- On `status="done"`: render `result.shorts` —
  - short with `clip_url` set → title, hook, score, `<video controls
    src="/download/Short-NN.mp4">` preview, download link.
  - short with `error` set (no `clip_url`) → marked "failed: `<error>`", no
    preview/download.
  - Re-enable the form for a new run.
- On `status="failed"`: show an error banner with `error`. Re-enable the
  form.

## Error handling

- Backend: pipeline exceptions in the background thread are caught, stored
  as `job.error`, `status` set to `"failed"` — mirrors `main.py`'s existing
  `except Exception as e` handling, just surfaced over HTTP instead of
  stderr.
- Download route never trusts the path segment blindly (traversal guard
  above).
- Frontend: fetch failures during polling show an inline "lost connection to
  server" note but keep retrying — don't wipe the log already shown.

## Tests

- **`tests/test_webapp.py`** (new) — Flask test client, `generate_shorts`
  monkeypatched to a fast fake (writes a couple lines to the given
  `paths.progress_log`, returns a canned result dict with one successful
  short and one failed short — no real download/transcribe/crop). Covers:
  - `GET /` → 200
  - `POST /run` with blank `url` → 400
  - `POST /run` while a job is `running` → 409
  - `GET /status` offset advances across polls and reflects `running` →
    `done`
  - `GET /download/<name>` serves a real file inside `shorts_dir`, 404s for
    a name outside it (`../../etc/passwd`-style) and for a nonexistent file
  - results payload distinguishes the successful short (has `clip_url`) from
    the failed one (has `error`, no `clip_url`)
- **`tests/test_pipeline.py`** — small regression test: calling
  `generate_shorts(url, paths=custom_paths, ...)` uses the given paths
  without calling `resolve_output_dir` again (patch/spy it and assert
  not-called).
