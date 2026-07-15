# descriptions.txt + copyable title/description — design

## Problem

Once shorts are rendered, the only metadata visible is on the dashboard card
(title, hook, score). Posting a batch of clips means retyping each title and
caption by hand — no file to copy from, no copy button in the UI.

## Goal

1. After each pipeline run, write a `descriptions.txt` alongside the rendered
   clips with one line per successful short: index, title, description.
2. In the dashboard, add "Copy title" / "Copy description" buttons on each
   result card so the same text can be grabbed without opening the file.

## Format

One line per successful short in `output/<Title>/Shorts/descriptions.txt`:

```
short 01 - <title> -- <hook_sentence>
```

- Index is 2-digit, zero-padded, and matches the short's position in the
  pipeline's `shorts` list — the same `i` used to name `Short-{i:02d}.mp4`
  (both `clipper.py` and `local/clipper.py` number that way). This keeps
  `short 03` lined up with `Short-03.mp4` even if an earlier clip in the
  batch failed to crop.
- **Description = `hook_sentence`** (the LLM's chosen opening line for that
  clip) — not `virality_reason`, which is meta-commentary about *why* it's
  viral rather than clip content.
- Shorts with no `clip_url` (cropping failed) are skipped — nothing to post,
  nothing to describe. Their index number is still not reused by anything
  else, so no renumbering is needed.
- Title falls back to `"Untitled"` if empty, matching the existing dashboard
  card behavior (`s.title || "Untitled"`).

## Architecture

New function in `shorts_generator/run_output.py`:

```python
def write_descriptions(shorts_dir: str, shorts: List[Dict]) -> str:
    """Write descriptions.txt into shorts_dir; returns its path."""
```

Called once from `pipeline.generate_shorts()`, after the mode-specific
`result` dict is built (so it covers both `_run_api` and `_run_local`
without duplicating the call), before `result.json` is written:

```python
write_descriptions(paths.shorts_dir, result["shorts"])
```

No new dependency, no new file format — plain text, `\n`-joined, UTF-8,
trailing newline if non-empty.

## UI — copy buttons

Dashboard cards (`templates/index.html`, `renderResults()`) already show
`s.title` and `s.hook_sentence` for successful shorts. Add two small buttons
in that success branch, styled like the existing `.download` link (small,
mono, uppercase, bordered) so they fit the 170px-min card width:

- **Copy title** → `navigator.clipboard.writeText(s.title || "Untitled")`
- **Copy description** → `navigator.clipboard.writeText(s.hook_sentence)`,
  only rendered when `hook_sentence` is present (same condition already used
  to render the hook text itself)

Both buttons flash their label to "Copied" for ~1.2s on click as feedback,
then revert. `navigator.clipboard` is available here because the dashboard
is served from `http://127.0.0.1`, a secure context by spec even over plain
HTTP.

Placement: a `.copy-row` under the title (containing "Copy title"), and
"Copy description" directly under the hook text — each button copies the
text block immediately above it rather than living in one combined row, so
it's unambiguous which button copies what.

## Error handling

- `write_descriptions` does no network/subprocess work — a plain local file
  write. If it raises (disk full, permissions), that propagates like any
  other write in `generate_shorts()` (matches how `result.json` write
  failures are already handled — no special-casing).
- Clipboard write failures (permissions, non-secure context) are caught in
  the frontend; on rejection, fall back to `document.execCommand("copy")`
  via a temporary off-screen `<textarea>` rather than failing silently.

## Tests

- **`tests/test_run_output.py`** (existing or new) — `write_descriptions`:
  - writes one correctly formatted line per short with a `clip_url`
  - skips shorts with no `clip_url`, but a later short keeps its original
    index (e.g. short 1 fails, short 2 succeeds → file contains `short 02 -
    ...`, not renumbered to `short 01`)
  - empty `shorts` list → file exists and is empty
  - missing `hook_sentence` / `title` → falls back to `""` / `"Untitled"`
