# Per-run output folder — design

## Problem

Today, every run of the pipeline (either `--mode api` or `--mode local`) writes
into a single flat `output/` directory: `short_01.mp4`, `short_02.mp4`, ...
plus, in local mode, a transcript cache json and a downloaded
`source_<id>.mp4`. Nothing ties a run's files together, and running the tool
twice against different videos mixes their outputs in the same directory.

## Goal

Given a YouTube link (or local file, in `--mode local`), create one
self-contained subfolder per video under `output/`, named after the video,
holding everything that run produces:

```
output/<Title>/
  Shorts/
    Short-01.mp4
    Short-02.mp4
    ...
  full_source.mp4
  full_source.json
  result.json
  progress.log
```

This applies to **both** `--mode api` and `--mode local`.

## Title resolution

New module `shorts_generator/run_output.py` owns this.

- For a YouTube URL: fetch the title via YouTube's public oEmbed endpoint
  (`https://www.youtube.com/oembed?url=<url>&format=json`), using the
  already-required `requests` dependency — no new dependency, and it works
  in api mode too (which otherwise has no local yt-dlp install).
- For a local file path / `file://` URL, or if oEmbed fails (non-YouTube URL,
  network error): fall back to the input's filename stem, or `"video"` if
  none is available.
- Sanitize the resolved title for filesystem safety: replace characters
  outside `[A-Za-z0-9 _-]` with `_`, collapse repeated whitespace/underscores,
  strip leading/trailing separators, truncate to ~100 chars, and fall back to
  `"untitled"` if sanitizing leaves nothing.

`resolve_output_dir(url_or_path, base_dir=LOCAL_OUTPUT_DIR) -> RunPaths` returns
a small structure with the resolved paths:

```python
@dataclass
class RunPaths:
    root: str            # output/<Title>
    shorts_dir: str       # output/<Title>/Shorts
    source_video: str     # output/<Title>/full_source.mp4
    source_json: str      # output/<Title>/full_source.json
    result_json: str      # output/<Title>/result.json
    progress_log: str     # output/<Title>/progress.log
```

`os.makedirs(paths.shorts_dir, exist_ok=True)` creates the whole tree (making
`shorts_dir` also makes `root`).

**Known tradeoff:** two different videos that happen to produce the exact
same sanitized title will land in the same folder (no id suffix is appended).
This was an explicit choice — the alternative (title + video id) was
considered and rejected in favor of a plain, human-readable folder name.

## Rerun / resume behavior

Running the same URL again reuses `output/<Title>/` rather than creating a
new numbered folder. Whether steps can be skipped differs by mode:

**local mode** — every step is a local file operation, so this is
straightforward and mirrors logic that already exists today, just now scoped
to the run folder instead of a flat global directory:
- skip download if `full_source.mp4` already exists
- skip transcription if `full_source.json` already exists and is non-empty
  (existing cache-validity check in `local/transcriber.py`, unchanged)
- highlight ranking, cropping, and `result.json` are always redone

**api mode** — MuAPI's `/autocrop` endpoint needs a *hosted* URL for each
highlight, and that URL is only produced by MuAPI's own `/youtube-download`
call. A cached local `full_source.mp4` can't be fed back into MuAPI, so the
download call itself **cannot** be skipped on rerun — it must run every time
to get a working hosted URL for cropping. What *can* be skipped:
- the local streaming-copy into `full_source.mp4`, if that file already
  exists (saves local bandwidth/disk churn, not the MuAPI call itself)
- the transcribe call, if `full_source.json` already exists and is valid
  (saves the more expensive Whisper cost/time)
- highlight ranking, cropping, and `result.json` are always redone

## progress.log

A small `Tee` stream class duplicates every `print(...)` (which the pipeline
already does liberally, e.g. `[download]`, `[transcribe]`, `[clip]`) to both
the console and `progress.log`, so behavior on the terminal is unchanged.

- Opened in append mode with a timestamped `=== run start ... ===` banner, so
  reruns accumulate history in the same file rather than truncating it.
- Scoped inside `generate_shorts()`: starts right after the output folder is
  resolved (first thing), stops right before `generate_shorts()` returns. If
  the run raises partway through, the exception message is written to the log
  before propagating, so failures are visible in `progress.log` too.
- `main.py`'s own final summary block (printed after `generate_shorts()`
  returns) is not captured — it's a terminal-only report of data that's
  already in `result.json`.

## Code changes

- **New:** `shorts_generator/run_output.py` — `RunPaths`, `resolve_output_dir`,
  title resolution + sanitizing, the `Tee` class and `capture_progress_log`
  context manager.
- **`pipeline.py`:** `generate_shorts()` resolves `RunPaths` once (mode-agnostic)
  before branching into `_run_api` / `_run_local`, wraps the whole call in
  `capture_progress_log(paths.progress_log)`, and threads `paths` through both
  mode functions instead of the current ad-hoc `out_dir` parameters. Adds
  `"output_dir": paths.root` to the returned result dict, and both mode
  functions write `result.json` there before returning.
- **`shorts_generator/local/downloader.py`:** download straight to
  `full_source.mp4` inside the run folder (via the `outtmpl` pattern) instead
  of the flat `source_<id>.<ext>` naming; drop the video-id-based
  `_existing_download` cache lookup (superseded by the folder-level
  existence check in `pipeline.py`). Local file / `file://` input that isn't
  already at `paths.source_video` gets copied (or remuxed to `.mp4` via
  `ffmpeg -c copy` if the container isn't already mp4) into the run folder so
  `full_source.mp4` is always present, matching the api-mode behavior.
- **`shorts_generator/local/transcriber.py`:** `_transcript_cache_path` currently
  resolves the cache directory from the *global* `LOCAL_OUTPUT_DIR` regardless
  of where the media file lives. Change it to use the media file's own parent
  directory, so the cache naturally lands at `<run_dir>/full_source.json`
  with no other change needed to the existing caching logic.
- **`shorts_generator/clipper.py`** (api mode) **and**
  **`shorts_generator/local/clipper.py`** (local mode): write cropped clips
  into `<run_dir>/Shorts/Short-{i:02d}.mp4` instead of `<out_dir>/short_{i:02d}.mp4`.
  api mode's `crop_highlights` also gains the "skip local copy if
  `full_source.mp4` exists" / "skip transcribe if `full_source.json` exists"
  checks described above.
- **`main.py`:** prints `Output folder: <result['output_dir']>` in the
  summary block. `--output-json` is kept as an *additional* optional copy
  path (backward compatible) — `result.json` is now always written inside the
  run folder regardless of whether `--output-json` is passed.

## Tests

Existing tests that assert on `short_01.mp4` naming or a flat/monkeypatched
`LOCAL_OUTPUT_DIR` (`tests/test_local_clipper.py`, `tests/test_clipper_api.py`,
`tests/test_pipeline.py`, `tests/test_local_transcriber.py`) will need
updating to the new paths/naming as part of implementation. New tests should
cover: title sanitization edge cases, folder reuse across two calls (resume
skips the right steps), and the api-mode partial-skip behavior.
