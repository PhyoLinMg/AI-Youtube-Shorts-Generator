# Thread output folder reorganization

## Problem

`output/_Threads/<slug>/` is confusing today in two ways:

1. **Folder names are unreadable.** `resolve_thread_run_dir()` (`run_output.py:199`) slugs the folder as `sanitize_title(title_a)_x_sanitize_title(title_b)`, each side up to 100 chars — the full YouTube title of both episodes concatenated. Example: `Godfather_of_AI_We_Have_2_Years_Before_Everything_Changes_x_Why_AI_CEOs_Are_Building_Bunkers_-_Tristan_Harris`.
2. **Files are flat and mixed.** Inside one thread folder, raw per-episode clips (`clip_N_a.mp4`, `clip_N_b.mp4`, `clip_N_a.json`, `clip_N_b.json`), narration audio (`thesis_N.mp3`, `bridge_N.mp3`), assembled narration cards (`intro_card_N.mp4`, `bridge_card_N.mp4`), the generically-named final deliverable (`clip_N.mp4`), and run metadata (`thread_results.json`, `descriptions.txt`, `progress.log`) all sit in one directory with no separation between disposable intermediates and the shippable output. A thread run producing multiple theses (N = 1, 2, ...) interleaves all of their files together.

Additionally, re-running `generate_threads` on the same pair of episodes reuses the same folder (`os.makedirs(exist_ok=True)`) — if a later run uses a different `num_clips`, leftover files from the earlier run's higher-numbered theses linger untouched, orphaned. (Observed today: users have been manually moving old files into a hand-made `stale/` folder to cope.)

## Goals

- Thread folder names are short and human-scannable at a glance.
- Inside a thread folder, the shippable final video(s) are immediately visible at the top level; everything else lives under `raw/`.
- Multiple theses from one run are grouped, not interleaved.
- Re-running the same pair doesn't silently mix two runs' files together.
- Existing 17 thread folders get migrated into the new scheme (not left behind in the old one).

## Design

### 1. Folder naming

`output/_Threads/<YYYY-MM-DD>_<short-slug-a>_x_<short-slug-b>/`

- `YYYY-MM-DD` = the date the thread run started (local date).
- `short-slug-{a,b}`: a new, more aggressive slugifier distinct from the existing `sanitize_title()` (which stays as-is for per-episode `output/<Title>/` folders — those aren't changing). New helper `short_slug(title: str, max_length: int = 25) -> str`: lowercase, non-alphanumeric runs collapsed to a single `-`, truncated to `max_length`, trimmed of trailing `-`.

Example: `output/_Threads/2026-08-18_godfather-of-ai-we-have-2_x_why-ai-ceos-are-building/`.

`resolve_thread_run_dir(title_a, title_b, base_dir=None)` changes to build this slug and prepend the date. Its signature is unchanged; callers (`pipeline.generate_threads`) don't need to change how they call it.

### 2. Internal layout

```
2026-08-18_godfather-of-ai-we-have-2_x_why-ai-ceos-are-building/
  thesis_1_Is_AI_an_existential_threat.mp4   # final, ready to upload
  thesis_2_AI_trust_test.mp4
  descriptions.txt
  thread_results.json
  progress.log
  raw/
    thesis_1/
      clip_1_a.mp4  clip_1_b.mp4  clip_1_a.json  clip_1_b.json
      thesis_1.mp3  bridge_1.mp3  bridge_card_1.mp4  intro_card_1.mp4
    thesis_2/
      clip_2_a.mp4  clip_2_b.mp4  clip_2_a.json  clip_2_b.json
      thesis_2.mp3  bridge_2.mp3  bridge_card_2.mp4  intro_card_2.mp4
    stale/
      <HHMMSS>/          # only present after a same-day re-run; see below
        ...entire previous run's tree...
```

`pipeline.generate_threads` changes its per-thesis path construction (currently all flat under `out_dir`) to write intermediates under `out_dir/raw/thesis_{i}/` and the final assembled video directly under `out_dir/`.

### 3. Final filename

Final video filename becomes `thesis_{i}_<sanitize_title(thread["title"])>.mp4`, reusing the per-thesis clickbait `title` that `thread_builder.pick_thread_clips` already produces (`thread_builder.py:66`, capped at 100 chars) and the existing `sanitize_title()` from `run_output.py`. This mirrors how `unique_short_filename()` already names regular Shorts from their title.

### 4. Same-day re-run handling

At the top of `generate_threads`, after `resolve_thread_run_dir` returns `out_dir`: if `out_dir` already contains a `thread_results.json` from a prior completed run, move every existing top-level entry (final mp4s, `descriptions.txt`, `progress.log`, `raw/` minus any pre-existing `raw/stale/`) into a fresh `raw/stale/<HHMMSS>/` (current time) before any new file is written. `thread_results.json` itself moves too — a fresh one gets written at the end of the new run. This only fires when a *completed* prior run is detected (presence of `thread_results.json`), not on a resumed/crashed run.

### 5. Download route compatibility

`webapp.py`'s `_clip_display_url()` currently does `f"/download/{os.path.basename(clip_url)}"` — for thread clips this must change to preserve the path *relative to `out_dir`* (e.g. `raw/thesis_1/clip_1_a.mp4`) instead of collapsing to a bare basename, otherwise `episode_a`/`episode_b` preview downloads break once those files move under `raw/thesis_N/`. `_safe_join()` (`webapp.py:217`) already resolves and validates arbitrary relative subpaths under `base_dir`, and the route is already declared `@app.route("/download/<path:name>")` (`webapp.py:343`) — the `path` converter already accepts slashes, so no route change is needed. Only `_clip_display_url()`'s basename-collapsing needs to change for thread clips.

This only affects thread-mode display URLs. Regular Shorts/Chapters keep basename-only (their files are already flat in `Shorts/`/`Chapters/`).

### 6. Migration script

One-off script (not part of the app, run manually once): for each existing `output/_Threads/<old-slug>/`:
- Read `thread_results.json` to get the real `episode_a.title` / `episode_b.title` and each thesis's `title`.
- Compute the new date-prefixed short-slug folder name; date = the oldest file mtime found in the old folder (best available proxy for run start, since it isn't stored elsewhere).
- Create the new folder, move `clip_{i}.mp4` → `thesis_{i}_<slug(title)>.mp4` at the new root; move `clip_{i}_a.mp4`, `clip_{i}_b.mp4`, `clip_{i}_a.json`, `clip_{i}_b.json`, `thesis_{i}.mp3`, `bridge_{i}.mp3`, `intro_card_{i}.mp4`, `bridge_card_{i}.mp4` → `raw/thesis_{i}/`.
- Move `descriptions.txt`, `progress.log`, `thread_results.json` to the new root unchanged.
- If an old folder already has a hand-made `stale/` subfolder, move its contents into `raw/stale/<oldfoldername>/` (best-effort; these are already-abandoned files, so filename collisions inside `stale/` aren't a correctness concern the way they are for live thesis N).
- Remove the old (now-empty) folder.
- Print a per-folder summary; do not silently continue past an unexpected file (anything not matching a known pattern) — list it and skip that folder's cleanup rather than guessing, then report the leftover folders at the end.

## Error handling

- Migration script: any old folder with files it doesn't recognize is reported and left untouched rather than partially migrated.
- `generate_threads` stale-archive step: if the move fails partway (disk error), it should fail loudly (raise) rather than let the new run start writing into a half-archived directory.

## Testing

- Unit tests for `short_slug()` (truncation, unicode/punctuation stripping, empty-title fallback).
- Unit test for `resolve_thread_run_dir()` producing the new date-prefixed path.
- Unit test for the stale-archive step: given a folder with a pre-existing `thread_results.json`, confirm old files land under `raw/stale/<timestamp>/` and the root is clear before the new run writes.
- Unit test for `_clip_display_url` producing a relative (not basename-only) download URL for thread clip paths, and `_safe_join` accepting that nested relative path.
- Migration script gets a small fixture copy of 2-3 real thread folders (not run against the live `output/` dir in tests) to verify the move logic end to end.
