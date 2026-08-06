# Long-form podcast chapter cuts

## Problem

The pipeline only produces viral 9:16 Shorts (`generate_shorts`): tight
20-180s clips, cropped vertical, hook-card overlays, viral-scoring rubric.
There's no way to pull longer "interesting bits" out of a long-form podcast
as landscape clips — e.g. a 12-minute exchange on one topic, full context
intact, captioned but not chopped down to a swipe-optimized fragment.

Needed: a second, parallel output type — chapter cuts. Fewer, longer
segments (up to 15min each), original 16:9 frame (no crop), bottom-margin
subtitles, each segment self-contained enough that a viewer with zero other
context from the episode understands the full topic being discussed.

Scope: **local mode only** (`--mode local`). API mode (MuAPI) is not
extended for this — no hosted-crop equivalent is needed since there's no
crop, but transcript/highlight plumbing for API mode is left untouched for
now; this can be added later as a separate follow-up if needed.

## Selection logic: `highlights.py`

New chapter rubric, parallel to the existing viral-highlight one, reusing
the shared plumbing (`chunk_transcript`, `detect_content_type`,
`_parse_json_loose`, the long-video chunking path) rather than duplicating
it.

**Chapter dict shape** (much lighter than a Shorts highlight — no
`score`/`hook_sentence`/`on_screen_hook`/`reaction_type`/`cut_segments`/
`yt_hashtags`; chapters don't get viral packaging or reaction-jail dead-air
trimming):

```json
{
  "title": "string, <=8 words",
  "start_time": 0.0,
  "end_time": 0.0,
  "summary": "string, 2-4 sentences",
  "interest_reason": "string, one sentence"
}
```

- `title`: chapter-card style, ≤8 words (matches `long-form-episode-builder`'s
  own chapter-title convention, so filenames/descriptions read consistently
  with that skill's output if the user later builds a single chaptered
  video from the same source).
- `summary`: 2-4 sentences capturing the **full context** of what's
  discussed — not a hook tag. Doubles as the human-readable content of
  `chapters_description.txt` (see below) and a review aid.
- `interest_reason`: one sentence on why this segment is worth extracting
  as its own chapter (a complete story, a strong argument, a concrete
  insight, a revelation, a funny/emotional beat — same signal families as
  Shorts' `VIRALITY_CRITERIA`, reused as prose framing but not reused as a
  literal shared constant, since the packaging goal differs).

**New prompt: `CHAPTER_SYSTEM_PROMPT`** (mirrors `HIGHLIGHT_SYSTEM_PROMPT`'s
structure — content-type/density header, rules, JSON-only response) with
rules specific to full-context chapters, the opposite of the Shorts
"open cold on the hook, skip windup" rule:

- `start_time` must land where the topic/question is actually **introduced**
  — the premise or the question that kicks off the discussion — not just
  the punchline or peak moment. The windup Shorts is told to skip is often
  exactly the context a chapter needs to keep.
- `end_time` extends to where the topic naturally **resolves** or the
  conversation visibly moves to a new topic — not cut mid-thought, not cut
  the moment the "interesting part" lands.
- Rule of thumb given directly to the model: *a viewer watching only this
  chapter, with zero other context from the rest of the episode, must fully
  understand what's being discussed and why it matters.*
- Never cut mid-sentence (same rule as Shorts, reused verbatim).
- Chapters must not overlap.
- Duration: `MIN_CHAPTER_DURATION_SECONDS = 60` (a chapter shorter than this
  isn't "full context," it's a fragment) to
  `MAX_CHAPTER_DURATION_SECONDS = 900` (15min hard ceiling). No sweet-spot
  guidance like Shorts' 20-45s — natural topic length governs, the model is
  told to let a chapter run as long as the topic actually needs, up to the
  ceiling.
- Target count: 3-8 chapters (mirrors `long-form-episode-builder`'s 3-8
  chapter convention and `call_highlight_api`'s existing
  `min_clips = min(target, natural_max, 14)` shape, reused with the new
  bounds).

**Sanitize (`_sanitize_chapters`, parallel to `_sanitize_highlights`)**:
same float/string coercion pattern, clamps `end_time` to
`start_time + MAX_CHAPTER_DURATION_SECONDS`, drops entries shorter than
`MIN_CHAPTER_DURATION_SECONDS`. No `cut_segments`/`reaction_type`/hashtag
handling — those fields don't exist on this shape.

**Dedupe (`dedupe_chapters`, NOT reusing `dedupe_highlights`)**: chapters
have no `score` to rank by, and the goal is a clean sequential set of
non-overlapping segments, not "best N candidates." Sort by `start_time`
ascending, walk forward, drop any chapter whose `start_time` is before the
previously-kept chapter's `end_time` (any overlap at all is dropped, unlike
Shorts' >50%-overlap tolerance — chapters are meant to tile the episode's
interesting parts, not compete for the same moment).

**Entry points**: `get_chapters(transcript, num_chapters, llm_fn)` and
`get_chapters_cached(transcript, num_chapters, cache_path, llm_fn)`,
structurally identical to `get_highlights`/`get_highlights_cached`
(fingerprint + num_chapters + a new `CHAPTER_SCHEMA_VERSION` cache-key
triple, long-video chunking reused as-is since `chunk_transcript` doesn't
care what downstream does with the chunks).

## Clip rendering: `local/clipper.py`

New `crop_chapters_local(source_path, chapters, out_dir, transcript_segments,
captions, caption_fade_duration, word_highlight, filename_style) -> List[Dict]`,
parallel to `crop_highlights_local` but deliberately does less:

- Trim only — calls the existing `_cut_subclip(source_path, start, end,
  out_path)` directly (already exactly "ffmpeg -ss/-to, re-encoded, audio
  kept, no reframe"). **No** `_reframe_vertical`/`_reframe_vertical_adaptive`
  call — landscape in, landscape out.
- **No** jump-cut excision (`excise_cut_segments`) — chapters have no
  `cut_segments` field; the whole selected span is kept intact by design
  (that's the "full context" requirement).
- **No** hook-card / end-card overlay calls — those are Shorts-specific
  viral packaging; chapters don't carry `on_screen_hook`/`end_card_text`
  fields to feed them anyway.
- Captions: calls `burn_captions` with the new `bottom_margin_frac=0.06`
  (see below) instead of the Shorts default.
- Same per-item try/except-and-continue error pattern as
  `crop_highlights_local` — one chapter's trim/caption failure records an
  error on that entry and the run continues.

## Captions: `captions.py`

`_write_ass`'s `margin_v` is currently hardcoded to `round(height * 0.30)`
— tuned for 9:16 Shorts to clear the platform's reply/like UI column.
Landscape chapter clips have no such UI to dodge; the user wants subtitles
near the bottom edge.

Add `bottom_margin_frac: float = 0.30` as a parameter threaded through the
call chain: `_write_ass` → `_burn_chunks` → `burn_captions` /
`burn_captions_segments`. Default stays `0.30` so every existing Shorts call
site is untouched (no call-site changes needed there). The new chapters call
site in `local/clipper.py` passes `bottom_margin_frac=0.06`.

## Orchestration: `pipeline.py`

New `generate_chapters(youtube_url, num_chapters=5, download_format="1080",
language=None, captions=True, caption_fade_duration=0.3, word_highlight=True,
filename_style=None, paths=None) -> Dict`, parallel to `generate_shorts`.
Local-mode only, so no `mode` param and no `_run_api` counterpart — it calls
a new `_run_local_chapters` helper directly (mirrors `_run_local`'s
download-cache / transcribe-cache / `get_chapters_cached` / `crop_chapters_local`
/ `_trim_to_num_chapters` shape; no `CROP_FAILURE_BUFFER` equivalent needed
since a trim-only ffmpeg call has no failure mode analogous to a remote
autocrop job erroring — trims either work or the source file is bad).

Return shape mirrors `generate_shorts`'s result dict but with `"chapters"`
in place of `"shorts"`, and no `"mode"` key (always local).

## Output paths: `run_output.py`

- `RunPaths` gains `chapters_dir` (`output/<Title>/Chapters/`) and
  `chapters_json` (chapter-selection cache, parallel to `highlights_json`).
  `resolve_output_dir` creates `chapters_dir` alongside `shorts_dir` — both
  always exist per run, regardless of which output type a given call
  produces (unused dir stays empty, matching how `shorts_dir` already
  exists even for a chapters-only run).
- New `unique_chapter_filename(title, index, used_names) -> str`: always
  `f"{index:02d}_{sanitize_title(title)}.mp4"` — zero-padded numeric prefix
  regardless of `SHORT_FILENAME_STYLE`, so chapter files sort into episode
  order in a plain file browser (the numbering IS the chapter order, unlike
  Shorts where `SHORT_FILENAME_STYLE=generic`'s `video{index}.mp4` exists
  for a different reason — anonymizing clickbait titles, not ordering).
  Collision suffixing (`_2`, `_3`, ...) reuses the same loop shape as
  `unique_short_filename`.
- New `write_chapter_descriptions(chapters_dir, chapters) -> str`: writes
  `chapters_description.txt`, one block per chapter with a clip_url —
  `title`, original-video timestamp range (`start_time`–`end_time`, useful
  reference even though each chapter is its own file, not a marker in one
  long video), and the full `summary`. No yt_title/hashtags/hook_strength
  fields — those don't exist on this shape.

## CLI: `main.py`

- `--clip-type shorts|chapters` (default `shorts`).
- `--num-chapters` (default `5`), only consulted when `--clip-type chapters`.
- When `--clip-type chapters`: `--aspect-ratio`, `--framing`, `--hook-card`,
  `--end-card` are accepted but ignored with a one-line stderr notice if
  explicitly passed (no crop, no card overlays exist in this path) — not
  hard errors, since a user might leave a shell alias with old flags around.
  `--mode` is forced to `local` regardless of what's passed (chapters is
  local-only per scope) — print a notice if `--mode api` was explicitly
  requested with `--clip-type chapters`.
- `--num-clips`, `--caption-fade-duration`, `--no-captions`,
  `--no-word-highlight`, `--filename-style` are shared as-is (captions/
  filename knobs apply identically to both output types; `--num-clips` is
  simply unused when `--clip-type chapters`).

## Error handling

Same pattern throughout as the existing Shorts path: a failure on one
chapter (trim, caption burn) is caught, recorded as an `*_error` field on
that chapter's result entry, and the run continues to the next chapter —
never aborts the whole run for one bad segment.

## Testing

Mirrors the existing test files' patterns:
- `test_highlights.py`-style: `_sanitize_chapters` duration clamp/min-drop,
  `dedupe_chapters` chronological overlap-drop, cache fingerprint/version
  mismatch behavior for `get_chapters_cached`.
- `test_local_clipper.py`-style: `crop_chapters_local` produces a
  trim-only (no reframe) output; verifies no hook-card/end-card/excision
  calls happen for a chapter entry.
- `test_captions.py`-style: `_write_ass` respects a passed
  `bottom_margin_frac`, defaults to `0.30` when omitted (Shorts call sites
  unaffected).
- `test_run_output.py`-style: `unique_chapter_filename` numbering/collision
  behavior, `write_chapter_descriptions` output shape.
