# History tab — design

## Problem

Every `generate_shorts()` run writes its own `output/<Title>/` folder
(`full_source.mp4`, `full_source.json`, `Shorts/Short-NN.mp4`,
`descriptions.txt`, `result.json`, `progress.log`) and nothing ever cleans it
up. `full_source.mp4` and the `Shorts/` clips are large — over time `output/`
grows unbounded and there's no way to reclaim space without a terminal and
`rm`.

## Goal

A "History" tab in the dashboard, alongside the existing run form, listing
every past run folder with its source-video size and shorts count/size, and a
delete button for each (source video, shorts clips) so disk space can be
reclaimed from the browser.

Out of scope: no video playback/preview of past runs (that's the live
Results panel's job for the *current* run only), no parsing of `result.json`
for the list (it can carry a multi-MB transcript — the list view never reads
it), no "delete entire run" action, no history persistence beyond what's
already on disk (the list is always a live directory scan, not a database).

## Architecture

- **`shorts_generator/run_output.py`** — add `list_runs(base_dir=None) ->
  List[RunSummary]`, a pure filesystem scan.
- **`shorts_generator/webapp.py`** — three new routes: `GET /history`,
  `POST /history/<name>/delete-source`, `POST /history/<name>/delete-shorts`.
- **`shorts_generator/templates/index.html`** — a tab bar ("Generate" /
  "History") toggling which panel is visible; History panel is populated via
  `fetch("/history")` and re-fetched after each delete. No polling — this is
  a static listing, refreshed on tab switch or after a delete action.

## Data: `list_runs()`

```python
@dataclass
class RunSummary:
    name: str              # folder name under output/, also the route id
    mtime: float            # max mtime across all files in the run, epoch seconds
    source_exists: bool
    source_size: int        # bytes; 0 if source_exists is False
    shorts_count: int        # count of Shorts/Short-*.mp4
    shorts_size: int         # total bytes of Shorts/Short-*.mp4
```

For each immediate subdirectory of `output/`: stat `full_source.mp4` for
`source_exists`/`source_size`; glob `Shorts/Short-*.mp4` for `shorts_count`/
`shorts_size` (summed). `mtime` is the max `st_mtime` across every file found
directly in the run root plus `Shorts/` (falls back to the directory's own
mtime if the folder is empty) — this is what drives "newest first" sort
order, and using file mtimes rather than the directory's own mtime means it
still reflects the latest activity after a partial delete.

No `result.json` read anywhere in this path — deliberate, since that file can
be several MB (it inlines the full transcript) and the list view only needs
sizes/counts derivable from `stat`/`glob`.

Sort: newest `mtime` first.

## Routes

- **`GET /history`** → JSON:
  ```json
  {"runs": [{"name": "...", "mtime": 1752600000.0, "source_exists": true,
             "source_size": 1288490188, "shorts_count": 8,
             "shorts_size": 152000000}, ...]}
  ```

- **`POST /history/<name>/delete-source`** — deletes `output/<name>/full_source.mp4`
  if present. Idempotent: missing file is a no-op, not an error. Returns the
  updated `RunSummary` for that run (200), so the frontend patches just that
  row instead of re-fetching the whole list.

- **`POST /history/<name>/delete-shorts`** — deletes
  `output/<name>/Shorts/Short-*.mp4` only; `descriptions.txt` is left alone
  (still copy-pasteable after the clips are gone). Same idempotent/response
  shape as above.

Both delete routes:
- Resolve `name` under `LOCAL_OUTPUT_DIR` with the same escape-guard as the
  existing `_safe_join` (`os.path.realpath` + prefix check) — rejects `../`
  traversal with 400.
- 404 if `output/<name>/` doesn't exist.
- 409 if `job.status` is `"starting"` or `"running"` — mirrors the existing
  one-run-at-a-time guard in `POST /run`; blocks *all* deletes while any run
  is active rather than trying to figure out whether the requested folder is
  the active job's own folder, since this is a single-user tool and the
  extra precision isn't worth the added surface.

## Frontend behavior

- Tab bar with two buttons, "Generate" and "History"; clicking toggles a
  `hidden` attribute on the two panel containers (form+monitor+results vs.
  history). Default tab on page load: Generate (unchanged behavior).
- Switching to History tab (or clicking a "Refresh" button in it) calls
  `GET /history` and renders one row per run: folder name, formatted date
  from `mtime`, "Source: `<size>` [Delete]" (button hidden/disabled if
  `source_exists` is false), "Shorts: `<count>` clips, `<size>` [Delete]"
  (hidden/disabled if `shorts_count` is 0).
- Delete button click → native `confirm("Delete <what> for <name>?")` →
  `fetch POST` to the matching route → on success, patch that row's numbers
  from the response body and disable the button if the corresponding count
  is now 0; on 409, show the existing error banner ("a run is in progress —
  finish or wait before deleting"); on 404, remove the row (folder's gone).

## Error handling

- Delete routes never trust `name` blindly — same traversal guard pattern
  already used by `/download/<path:name>`.
- Filesystem errors during delete (e.g. permission denied) return 500 with
  the exception message; frontend surfaces it in the existing error banner
  rather than silently failing.
- `list_runs()` skips (doesn't crash on) a run folder it can't stat — e.g.
  removed mid-scan — logging nothing, just omitting it from that response.

## Tests

- **`tests/test_run_output.py`** — `list_runs()`: empty `output/` → `[]`;
  folder with only `full_source.mp4` → `source_exists=True`,
  `shorts_count=0`; folder with only `Shorts/Short-01.mp4` → `source_exists=False`,
  `shorts_count=1`; folder with both; sort order across multiple runs with
  distinct mtimes.
- **`tests/test_webapp.py`** — `GET /history` shape against a temp
  `LOCAL_OUTPUT_DIR` with fixture run folders; `delete-source` removes the
  file and returns updated summary; re-calling `delete-source` after it's
  already gone still returns 200 with `source_exists=False` (idempotent);
  `delete-shorts` removes only `Short-*.mp4`, leaves `descriptions.txt`
  in place; traversal name (`../../etc`) → 400; delete while
  `job.status="running"` (monkeypatched) → 409.
- Manual: run the dashboard, open History tab against real `output/`
  contents, confirm sizes match `du -sh`, delete shorts then source for one
  run, confirm files gone on disk and the row reflects it without a page
  reload.
