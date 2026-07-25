# Escape all three shorts jails

## Problem

The "shorts jail" framework (swipe jail / reaction jail / format jail) maps
cleanly onto three gaps in how this tool picks and cuts clips:

- **Jail 1 (swipe, <1K views)** — needs a hook in the first ~1s. Verbal and
  textual hooks are already scored (`hook_sentence`, `HOOK_STRENGTH_RUBRIC`,
  `hook_strength`, `hook_self_contained`, `hook_reason` in highlights.py;
  `on_screen_hook` + `hook_card.py` overlay). The **visual** hook — does the
  opening frame itself grab attention, independent of text/audio — is not
  scored at all. The highlight LLM only ever sees transcript text
  (`build_transcript_text`), never frames.
- **Jail 2 (reaction, 1K-10K views)** — needs every clip to build toward one
  specific viewer reaction (LOL/WOW/OMG/etc.) with no dead weight. Today a
  highlight is one `[start_time, end_time]` span chosen by the LLM; there's
  no mechanism to excise a slow or off-target passage in the middle of an
  otherwise-good span, and nothing forces the model to commit to a single
  target reaction before scoring.
- **Jail 3 (format, 30K-100K views)** — needs the clip to read as one
  legible, self-contained idea. This tool only recuts existing footage, so
  it cannot retrofit a *shooting* format the way the source material
  demonstrates (outlier discovery on other channels is a separate, unrelated
  subsystem — explicitly out of scope). What it *can* do: score whether the
  span it already extracted reads as one clean idea.

## Fix

### Jail 1 — visual hook score

New pass in `highlights.py` (or a new `visual_hook.py`), invoked from both
`_run_api` and `_run_local` in `pipeline.py` right after
`top = sorted(all_highlights, ...)[:2 * num_clips]` and before
`crop_highlights`/`crop_highlights_local` — both functions already have a
local copy of the source video on disk at that point (`paths.source_video`
in `_run_api`, `source_path` in `_run_local`), so no extra download.

`score_visual_hooks(source_video_path, highlights, llm_fn) -> highlights`:
for each of the (already-selected) `top` highlights, ffmpeg-extract 2 frames
from `[start_time, start_time + 0.8s]` to temp JPEGs, send to a
vision-capable MuAPI model with a rubric prompt ("does this frame alone,
with zero audio/text, stop a scroll?"). Attach `visual_hook_score` (0-100)
and `visual_hook_reason` (one sentence), same shape as the existing
`hook_strength`/`hook_reason` pair.

Informational only in this round — doesn't change which candidates are
selected, matching how `hook_strength` already coexists with the primary
`score` field without gating it. Runs on `top` *after*
`get_highlights_cached` returns, so `visual_hook_score`/`visual_hook_reason`
are never written into `highlights.json` — no `HIGHLIGHT_SCHEMA_VERSION`
bump needed for this phase, and a cache hit from before this phase lands
still works unchanged.

**Open dependency to resolve during implementation:** which MuAPI model
accepts image input and its exact payload shape (base64 vs. hosted URL).
`muapi.py` has no vision precedent today — check MuAPI's docs/schema before
writing `score_visual_hooks`, this is not assumed by this spec.

### Jail 2 — reaction jail: single reaction + jump cuts

**Schema (`highlights.py`):**
- `reaction_type`: enum `LOL | WOW | OMG | FINALLY | WTF | WHOLESOME`. Model
  picks exactly one per highlight, before generating the rest of that
  highlight's fields.
- `cut_segments`: list of `{start_time, end_time}` sub-spans inside the
  highlight's overall range, 1-6 entries, each ≥1.5s, each boundary snapped
  to an actual transcript segment edge (never mid-sentence — this extends
  the existing "never cut mid-sentence or mid-thought" rule at
  highlights.py:95 to apply *inside* a highlight too, not just at its outer
  edges). 1 entry = clip is already tight, no jump cut needed. Named
  `cut_segments` rather than `segments` to avoid colliding with the
  `segments` parameter name already used throughout `captions.py` for
  transcript segments.
- `tightness_reason`: one sentence — what got cut and why. Same pattern as
  `hook_reason`, a human-review note, not shown to viewers.
- Top-level `start_time`/`end_time` remain the envelope:
  `min(cut segment starts)` / `max(cut segment ends)`. `dedupe_highlights`
  is untouched — it already only reads the envelope fields.
- Prompt (`HIGHLIGHT_SYSTEM_PROMPT` / a new jail-2 block alongside
  `VIRALITY_CRITERIA`): pick the reaction first; every kept `cut_segments`
  entry must build toward it; cut anything that doesn't, including
  mid-span; never lengthen a clip for retention.
- `_sanitize_highlights`: validate `cut_segments` (sorted, non-overlapping,
  clamped to `[start_time, end_time]`); if missing or invalid, fall back to
  `[{start_time, end_time}]` — identical to today's single-span behavior.
  `reaction_type` outside the enum (or missing) defaults to `"WOW"` and the
  highlight is kept — matches every other field's coerce-with-default
  pattern (`_coerce_int` → 0, `_coerce_bool` → False, missing title →
  `"Untitled Highlight"`); only a bad `start`/`end` ever drops a highlight
  (`continue`), and this doesn't change that. Bump `HIGHLIGHT_SCHEMA_VERSION`
  2 → 3 (cache-invalidating field addition, same mechanism as every prior
  bump).

**api mode (`clipper.py`):** unchanged single `crop_clip` call on the
envelope. When `len(cut_segments) > 1`, after the existing download step
(already happens whenever captions or hook_card are on, which is the
default), excise the gaps locally with ffmpeg on the file already on disk —
not additional `/autocrop` calls. One API call per highlight regardless of
cut count.

**local mode (`local/clipper.py`):** `_cut_subclip` still cuts the envelope
once; when `cut_segments` has more than one entry, excise the gaps locally
(ffmpeg concat demuxer) before `_reframe_vertical` runs. Same shape as api
mode.

**Captions (`captions.py`):** `burn_captions` gains a `cut_segments`-aware
path. Instead of chunking the full envelope and then remapping/dropping
words that land in a gap (which risks a caption line straddling a cut),
call `_chunk_segments(transcript_segments, seg["start_time"],
seg["end_time"])` **once per kept `cut_segments` entry**, using that
entry's own absolute bounds as the window — each chunk pass is scoped to a
single kept span, so no chunk can ever straddle an excised gap by
construction. Then offset each resulting chunk's clip-relative
`start`/`end` by the cumulative duration of the *previously kept* segments
(not the excised gaps) to place it correctly on the concatenated output
timeline, and concatenate the per-segment chunk lists before `_write_ass`.
Single-`cut_segments` clips take the same code path with one segment — no
special-cased branch, no behavior change for existing single-span clips.

### Jail 3 — format-legibility score

No new subsystem. Piggybacks on the existing text-only highlight LLM call
already producing `hook_strength` etc. Add `format_clarity_score` (0-100:
does this span read as ONE self-contained idea a viewer immediately grasps
— single Q&A, single before/after, single narrated event — vs. a
meandering excerpt) and `format_reason` (one sentence), same JSON-schema
and prompt-block pattern as `hook_strength`/`hook_reason`. Informational
only, no change to candidate selection. Adds two more fields to
`_sanitize_highlights`'s cached output, so this phase bumps
`HIGHLIGHT_SCHEMA_VERSION` 3 → 4 — a separate bump from jail 2's, since a
cache written before this phase (version 3, no `format_clarity_score`)
must not be accepted once the code expects that field.

### webapp (`templates/index.html`)

New badges/meters for `reaction_type`, `visual_hook_score`,
`format_clarity_score`, plus `tightness_reason`/`format_reason`/
`visual_hook_reason` text blocks — reuses the existing meter markup
(`scoreColor`, `.score-row`) and `appendLabeledText` helper already used
for `score`/`hook_strength`/`virality_reason`.

## Files touched

- `shorts_generator/highlights.py` — `reaction_type`, `cut_segments`,
  `tightness_reason`, `format_clarity_score`, `format_reason` added to the
  prompt/schema/`_sanitize_highlights`; `HIGHLIGHT_SCHEMA_VERSION` bump; new
  `score_visual_hooks` (or new `visual_hook.py` module).
- `shorts_generator/pipeline.py` — `_run_api` and `_run_local` both call
  `score_visual_hooks` on `top` before cropping.
- `shorts_generator/clipper.py` — `crop_clip`/`crop_highlights` excise
  `cut_segments` gaps locally (ffmpeg) after download when count > 1.
- `shorts_generator/local/clipper.py` — same excision step before
  `_reframe_vertical`.
- `shorts_generator/captions.py` — `burn_captions` (or a new
  `burn_captions_segments`) takes `cut_segments`, chunks per-segment, offsets
  onto the concatenated timeline.
- `shorts_generator/muapi.py` — likely needs an image-capable call path if
  the vision model requires a different payload shape than the existing
  text-only `run()`.
- `shorts_generator/templates/index.html` — new badges/meters/text blocks.
- `tests/test_highlights.py`, `tests/test_pipeline.py`,
  `tests/test_captions.py`, `tests/test_clipper_api.py`,
  `tests/test_local_clipper.py`, `tests/test_webapp.py` — all touch schema
  or call shapes changed above; see Testing below.

## Out of scope

- Outlier-format *discovery* (searching other channels/videos for
  high-performing formats) — a separate subsystem unrelated to this
  clipping pipeline, explicitly ruled out for jail 3 this round.
- Using `visual_hook_score` or `format_clarity_score` to change which
  candidates get selected into `top` — both stay informational/display-only
  in this round, same as `hook_strength` today.
- Any change to `dedupe_highlights`'s overlap logic — it keeps reading only
  the envelope `start_time`/`end_time`, which stays authoritative regardless
  of `cut_segments`.
- Re-tuning `MAX_HIGHLIGHT_API_ATTEMPTS`, chunking thresholds, or the
  content-type-detect call — unrelated to this round's schema/pipeline
  additions.

## Testing

- `tests/test_highlights.py`: `_sanitize_highlights` accepts valid
  `cut_segments` (sorted, clamped, non-overlapping) and falls back to a
  single envelope segment when missing/invalid/out-of-order; `reaction_type`
  outside the enum is rejected/defaulted; schema-version bump forces a cache
  miss on old cache files (extends the existing cache tests from the
  2026-07-19 highlights-cache design).
- `tests/test_clipper_api.py` / `tests/test_local_clipper.py`: single
  `cut_segments` entry produces byte-identical behavior to today's
  single-span path (regression guard); multi-entry `cut_segments` triggers
  the excise-then-concat path and produces one output file spanning only
  the kept spans.
- `tests/test_captions.py`: per-segment chunking never produces a chunk
  whose transcript words originated on both sides of an excised gap; a
  single-entry `cut_segments` list reproduces the exact chunk output
  `burn_captions` produces today (regression guard); offset math places a
  second kept segment's captions correctly after the first segment's full
  duration, not at its own absolute transcript time.
- New `tests/test_visual_hook.py` (or extend `test_highlights.py`):
  `score_visual_hooks` attaches `visual_hook_score`/`visual_hook_reason` per
  highlight given a stubbed vision `llm_fn`; failure of the vision call
  degrades gracefully (leaves the fields absent/default rather than failing
  the whole pipeline run), mirroring how `detect_content_type` already
  swallows LLM failures.
- `tests/test_pipeline.py`: both `_run_api` and `_run_local` call
  `score_visual_hooks` on `top` before crop; a stubbed failure there doesn't
  abort the run (per the graceful-degradation point above).
- `tests/test_webapp.py`: new fields render without crashing when present
  and when absent (older cached `result.json` from before this change).
