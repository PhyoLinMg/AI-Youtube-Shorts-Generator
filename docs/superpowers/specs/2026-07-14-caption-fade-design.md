# Fade-in subtitle burn-in — design spec

Date: 2026-07-14
Status: Approved, pending implementation plan

## Goal

Burn phrase-length captions onto generated shorts, with a simple fade-in
animation per caption line. Available in both `--mode api` and `--mode local`,
on by default, with one configurable knob (fade duration) exposed as a CLI
flag.

## Scope decisions (from brainstorming)

- **Both modes** get caption burn-in, not just local mode.
- **On by default**; `--no-captions` disables it.
- **Minimal styling surface**: only fade duration is a CLI flag. Position,
  font, and color are fixed, sensible defaults — not configurable.
- **Caption grouping**: transcript segments (sentence-length, from Whisper)
  are split into ~6-8 word phrase chunks for readability. Whisper only gives
  segment-level timestamps (no per-word timing) in either mode, so each
  chunk's start/end is interpolated within its parent segment, proportional
  to word count.
- **Render technique**: an ASS subtitle file per clip, burned in via ffmpeg's
  `subtitles` filter (libass). Each line carries a `\fad(ms,0)` override tag
  — fade in over the configured duration, no fade out. Rejected alternatives:
  chained `drawtext` filters (fragile text-escaping, unwieldy filter graphs
  for many chunks) and per-frame OpenCV compositing (reinvents font
  wrapping/anti-aliasing, and API mode has no existing frame loop to hook
  into).

## Architecture

### New module: `shorts_generator/captions.py`

Shared, mode-agnostic. Operates on any local mp4 plus the *full* transcript
segments for that source video — it does not care which mode produced the
clip.

- `_chunk_segments(segments, clip_start, clip_end, max_words=7) -> List[Dict]`
  Slices the full transcript down to the highlight's `[clip_start, clip_end]`
  window, shifts timestamps to be clip-relative (0-based), and splits each
  segment's text into ~`max_words`-word chunks. Each chunk's duration is the
  parent segment's duration apportioned by that chunk's share of the word
  count. Segments outside the window are dropped; segments that straddle a
  window edge are clipped to it.

- `_probe_resolution(video_path) -> Tuple[int, int]`
  Runs `ffprobe` to read width/height, used to scale font size and margin
  relative to the actual output resolution.

- `_write_ass(chunks, ass_path, width, height, fade_seconds) -> None`
  Writes an ASS file: `PlayResX`/`PlayResY` set to the clip's resolution; one
  `Style` (`Alignment=2` bottom-center, `Fontsize` ≈ 4.5% of height,
  `MarginV` ≈ 6% of height, white `PrimaryColour`, black `OutlineColour`,
  `BorderStyle=1`, `Outline=2`, `Shadow=1`); one `Dialogue` line per chunk,
  text prefixed with `{\fad(<fade_ms>,0)}`. Literal `{`/`}` in caption text
  are stripped (they'd otherwise be parsed as ASS override blocks).

- `burn_captions(video_path, segments, clip_start, clip_end, out_path, fade_seconds=0.3) -> str`
  Orchestrates the above: chunk → write temp `.ass` next to the clip → run
  `ffmpeg -i video_path -vf "subtitles=<ass_path>" -c:a copy out_path`
  (re-encoding video, copying audio) → delete the temp `.ass` → return
  `out_path`. Raises `CaptionError` (new exception, subclass of
  `RuntimeError`) on any failure (ffprobe missing, ffmpeg missing/non-zero
  exit, no segments in window). Callers decide the fallback.

### CLI flags (`main.py`)

- `--no-captions` — `action="store_false"`, `dest="captions"`, default
  `True`.
- `--caption-fade-duration` — `type=float`, default `0.3` (seconds).

Both flow through `generate_shorts(..., captions=True, caption_fade_duration=0.3)`
into `_run_local` / `_run_api`, and from there into the two crop functions.

### Local mode (`shorts_generator/local/clipper.py`)

`crop_highlights_local(...)` gains `transcript_segments`, `captions`,
`caption_fade_duration` parameters. After `_reframe_vertical` produces the
plain cropped clip at `out_path`, if `captions` is true:

1. Call `burn_captions(out_path, transcript_segments, h["start_time"], h["end_time"], out_path + ".captioned.mp4", fade_seconds=caption_fade_duration)`.
2. On success, `os.replace()` the captioned file over `out_path` (same
   temp-then-swap pattern the module already uses for the cut/reframe steps).
3. On `CaptionError`, log a warning, leave the plain `out_path` in place, and
   set `captions_error` in that highlight's result dict. The clip is still
   reported as a success (`clip_url` populated).

`pipeline._run_local` passes `transcript["segments"]` through.

### API mode (`shorts_generator/clipper.py`)

`crop_highlights(...)` gains `transcript_segments`, `captions`,
`caption_fade_duration`, `out_dir` (defaults to `config.LOCAL_OUTPUT_DIR`)
parameters. `crop_clip()` (the MuAPI `/autocrop` call) is unchanged and still
returns a hosted URL. When `captions` is true, per highlight:

1. Download the hosted clip to a temp local file (`requests.get(stream=True)`,
   no new dependency).
2. `burn_captions(tmp_path, transcript_segments, start_time, end_time, os.path.join(out_dir, f"short_{i:02d}.mp4"), fade_seconds=caption_fade_duration)`.
3. Set `clip_url` to that local final path — consistent with local mode,
   where `clip_url` is already a filesystem path rather than a URL. Keep the
   original remote URL too, under `hosted_clip_url`, for reference.
4. On failure (download or burn-in), fall back to `clip_url` = the original
   hosted URL, and record `captions_error`.

`pipeline._run_api` passes `transcript["segments"]` through.

**Behavior change to flag in README**: API mode previously needed no local
heavy tooling — MuAPI did downloading, transcription, and cropping remotely.
With captions on by default, API mode now requires `ffmpeg` on `PATH` too
(only `ffmpeg`; no `opencv`/`faster-whisper`/etc. — those stay local-mode-only).

## Data flow

```
transcribe() / transcribe_local()   →  full transcript segments (mode-specific)
                                     ↓
pipeline._run_api / _run_local      →  passes highlights + full segments into crop step
                                     ↓
crop_highlights / crop_highlights_local
                                     ↓ (per highlight, after crop/reframe)
captions.burn_captions()            →  slice window → chunk → write .ass → ffmpeg subtitles filter
                                     ↓
final captioned mp4 (clip_url)
```

## Error handling

Caption burn-in is best-effort per clip: any failure is caught at the crop
function level, logged with `print(..., flush=True)` (matching existing
style), and never fails the highlight outright — it falls back to the
uncaptioned clip. The reason surfaces as `captions_error` in that clip's
result dict (visible via `--output-json`).

## Testing

- Unit: `_chunk_segments` — word-count grouping (~7/chunk), chunk durations
  sum back to the parent segment's duration, correct shifting/clipping to a
  clip window (segments partially outside the window, segments fully
  outside).
- Unit: `_write_ass` — `\fad(<ms>,0)` tag present and correctly computed from
  `fade_seconds`, one `Dialogue` per chunk, `PlayResX`/`PlayResY` match the
  probed resolution.
- Integration: `burn_captions` against a tiny synthetic clip generated via
  `ffmpeg -f lavfi -i testsrc` — asserts it returns without raising and the
  output file exists with the same duration/resolution as the input.
- Manual: one real end-to-end `--mode local` run on a short sample video,
  eyeball the fade-in and readability.
