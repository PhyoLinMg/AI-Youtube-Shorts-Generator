# First-frame hook card — design spec

Date: 2026-07-17
Status: Draft

## Goal

Kill the static talking-head frame 0. Force the viewer's eye with bold
on-screen hook text over a "striking" (non-resting) frame for the clip's
first ~1.5 seconds, and tune the highlight-generation prompt so clips open
cold and carry a short, punchy on-screen hook distinct from the existing
full-sentence `hook_sentence`. Deferred to a future spec: cold-open
re-cut/splice (re-cutting so the clip starts on the payoff line) — bigger
scope, non-chronological editing, out of scope here.

## Scope decisions (from brainstorming)

- **Mechanic**: in-place freeze-overlay, not a prepended card. The card
  replaces the visual for `[0, 1.5s)` of the existing clip; it does not add
  new duration. This means audio, total clip length, and caption timestamps
  are all untouched — no retiming logic needed anywhere. At the 1.5s mark
  the video hard-cuts from the still+text card into live footage already in
  progress; that hard cut is the intended "kill frame 0" effect.
- **Applies to both modes**: `--mode api` and `--mode local` both end up
  with a final local mp4 before this step, regardless of how the vertical
  crop was produced — the hook card operates purely on that local file, so
  one implementation covers both.
- **New flag**: `--no-hook-card` (CLI) / `hook_card` form field (webapp),
  mirroring the existing `--no-captions` / `--no-word-highlight` pattern.
  Default on.
- **Card duration**: fixed constant, `HOOK_CARD_DURATION = 1.5` seconds. Not
  user-configurable (same "minimal styling surface" call as the caption-fade
  spec).
- **Text source**: new LLM field `on_screen_hook` (≤7 words). The existing
  `hook_sentence` is a full spoken sentence (10-15 words in practice — see
  sample runs) and is too long for a bold on-screen card; it stays
  unchanged and continues to serve its existing purpose (title/description
  copy, CLI/UI display).
- **Striking-frame heuristic**: highest motion-diff frame among samples
  taken across the whole clip, first 0.5s excluded (to guarantee we never
  land back on the resting opening frame). Reuses the frame-diff technique
  already implemented in `local/clipper.py::_detect_cursor` — no new CV
  dependency.
- **Still capture timing (correction from initial draft)**: the still frame
  must be picked and extracted from the clip *before* caption burn-in, not
  after. Picking it from the captioned file would freeze that timestamp's
  burned-in caption line into the card, behind the hook text. So: crop →
  pick+extract still (clean frame) → caption burn → composite the card
  (still + drawtext, overlaid) onto the *captioned* clip as the last pass.
  The still is captured early; only the final compositing step runs last.
- **Z-order**: the composited hook card sits on top of the captioned clip
  for `[0, 1.5s)`, fully occluding any caption text that would otherwise
  render in that window.
- **Font**: `drawtext` has no bold flag and does not reliably resolve
  system font names the way libass (used for captions) does — needs an
  explicit `fontfile=`. Bundle **Anton** (Google Fonts, OFL-licensed,
  single-weight display font, already maximally bold — no separate
  "bold" variant to manage) at `shorts_generator/assets/fonts/Anton-Regular.ttf`
  and point `fontfile=` at the packaged path. Fixed choice, not
  user-configurable — sidesteps OS/fontconfig dependence entirely, which
  matters since this also needs to run on the Linux VPS target, not just
  local dev.
- **Out of scope**: item 4 (cold-open re-cut/splice) — separate future spec
  if wanted.

## Architecture

### `shorts_generator/highlights.py`

- `HIGHLIGHT_SYSTEM_PROMPT`: add a rule + JSON field for `on_screen_hook` —
  a short punchy fragment (≤7 words), not necessarily a verbatim transcript
  line, distinct from `hook_sentence`, written to work as bold on-screen
  text.
- Reinforce the existing "every highlight must open with a strong HOOK ...
  within the first 3 seconds" rule: `start_time` must land on the hook line
  itself, not on preamble/silence/filler before it. This is the "open cold"
  requirement from item 3.
- `_sanitize_highlights`: add `on_screen_hook` to the cleaned dict, with a
  defensive `[:60]` character cap (the ≤7-word instruction is a prompt
  ask, not a hard guarantee).

### `shorts_generator/hook_card.py` (new module)

Two-phase API — phase 1 runs on the clean, pre-caption crop; phase 2 runs
last, on the captioned clip:

- `pick_striking_frame(video_path, skip_seconds=0.5, sample_fps=5) -> float`
  — opens the (already vertically-cropped, *not yet captioned*) clip with
  OpenCV, samples frames at `sample_fps`, skips the first `skip_seconds`,
  scores each sampled frame by `cv2.absdiff` magnitude against the previous
  sampled frame (same primitive as `local/clipper.py::_detect_cursor`,
  scored over the whole frame instead of a small blob). Takes the
  top-quartile of frames by motion score (candidates for "something is
  happening"), then within that set picks the one with the highest
  sharpness — `cv2.Laplacian(frame, cv2.CV_64F).var()` — as a tiebreaker,
  since the single highest-motion frame is frequently motion-blurred (see
  Risks). Returns that frame's timestamp. Falls back to `skip_seconds` if
  the clip is too short to yield 2+ samples.
- `extract_frame(video_path, timestamp, out_path) -> str` — one-frame
  `ffmpeg -ss <timestamp> -i <video_path> -vframes 1 <out_path>`, run
  against the same pre-caption clip.
- `render_card_overlay(video_path, still_path, hook_text, out_path, duration=HOOK_CARD_DURATION) -> str`
  — the last-pass compositor, run against the *captioned* clip:
  1. Probe resolution (reuse `captions._probe_resolution`).
  2. Word-wrap `hook_text` into up to 2 lines (simple word-count half
     split).
  3. One ffmpeg call: loop `still_path`, `drawtext` each line (`fontfile=`
     pointing at the bundled bold TTF, boxed background for legibility,
     centered), then `overlay=enable='between(t,0,duration)'` onto
     `video_path`'s video stream; audio stream passthrough (`-c:a copy`).
- Both phases raise `HookCardError(RuntimeError)` on any ffmpeg/ffprobe/cv2
  failure — mirrors the existing `CaptionError` contract in `captions.py`.

### Wiring (`clipper.py`, `local/clipper.py`, `pipeline.py`, `main.py`, `webapp.py`)

- `crop_highlights()` / `crop_highlights_local()` gain a `hook_card: bool =
  True` parameter. Sequence per clip, when `hook_card` is on and the
  highlight's `on_screen_hook` is non-empty:
  1. Immediately after crop produces the clean local file (`out_path` in
     local mode; `downloaded_path` in API mode, *before* `burn_captions` is
     called on it) — `pick_striking_frame` + `extract_frame` → a temp still
     image.
  2. Caption burn-in proceeds exactly as today (unchanged).
  3. After caption burn (success or skipped), `render_card_overlay` runs
     against the captioned file and replaces it in place (temp path +
     `os.replace`, same pattern already used for caption burn-in).
  4. The temp still image is removed once step 3 completes (or in a
     `finally`, mirroring the existing `downloaded_path` cleanup in API
     mode).
  On `HookCardError` from either phase, print a warning, record
  `entry["hook_card_error"]`, and keep the pre-card file — a hook-card
  failure never fails the whole clip.
- `pipeline._run_api` / `_run_local` thread a `hook_card` argument through
  from `generate_shorts`.
- `generate_shorts()`: new `hook_card: bool = True` parameter, passed to
  both mode branches.
- `main.py`: new `--no-hook-card` flag (`action="store_false"`,
  `dest="hook_card"`, `default=True`), same shape as `--no-word-highlight`.
- `webapp.py`: `request.form.get("hook_card", "true") == "true"`, threaded
  through exactly like `word_highlight`.
- `templates/index.html`: one checkbox alongside the existing
  captions/word-highlight toggles.

## Data flow

```
highlights.get_highlights()       →  each highlight now carries "on_screen_hook" (≤7 words)
                                   ↓
crop_highlights[_local]()         →  crop (clean, pre-caption file)
                                   ↓
hook_card.pick_striking_frame()   →  motion-diff scan of the CLEAN crop (no burned-in caption)
hook_card.extract_frame()         →  still image saved to temp path
                                   ↓
burn_captions() (existing)        →  captions burned onto the crop, unchanged
                                   ↓
hook_card.render_card_overlay()   →  still + drawtext, overlaid onto captioned clip [0, 1.5s); last pass
                                   ↓
final Short-NN.mp4                →  frame 0 = striking frame + bold hook text; live footage resumes at 1.5s
```

## Error handling

New `HookCardError`, following the existing `CaptionError` pattern exactly.
Any failure (ffprobe/ffmpeg/cv2 nonzero exit or exception, unreadable
video, empty `on_screen_hook`) falls back to the clip as it stood before
the hook-card pass. Never blocks caption burn-in or the rest of the run.

## Risks / open questions (for review)

- **Font rendering is unverified on this stack.** `drawtext` needs an
  explicit `fontfile=`. Bundling Anton (see Scope decisions) removes the
  fontconfig/OS uncertainty, but the *first* implementation step should
  still be a throwaway `ffmpeg drawtext` smoke test against the bundled
  font file before building the rest of `hook_card.py` on top of it —
  confirms the packaged TTF path resolves and renders bold on this
  machine/VPS before anything depends on it.
- **Max-motion ≠ best-looking still** — resolved in the design (not
  deferred): raw frame-to-frame `absdiff` finds the moment with the most
  *change*, which on a talking head is often mid-gesture or mid-blink and
  therefore motion-blurred once frozen as a still — undercutting the
  "striking frame" goal it's meant to serve. `pick_striking_frame` now
  takes the top-quartile of frames by motion score, then picks the
  sharpest one (Laplacian variance) among those as a tiebreaker, so the
  chosen still is both dynamic (came from a high-motion moment) and clear
  (not blurred). Still worth an eyeball pass on real output once
  implemented — sharpness-among-motion-candidates is a reasonable heuristic
  but not a guarantee of a flattering frame (e.g. an open-mouth mid-word
  shot can be sharp but still an odd expression).

## Testing

- `hook_card.pick_striking_frame`: synthetic fixture video (a few
  solid-color frames with one deliberate brightness jump) — assert it picks
  the jump frame, and that it respects `skip_seconds`.
- `hook_card.render_card_overlay`: mock `subprocess.run`; assert the ffmpeg
  command includes `drawtext` with the bundled `fontfile=` and
  `overlay=enable='between(t,0,...)'`, and that `HookCardError` is raised
  on `CalledProcessError`.
- `highlights._sanitize_highlights`: new case asserting `on_screen_hook` is
  present and capped at 60 characters.
- `clipper.py` / `local/clipper.py`: existing caption-burn tests extended
  with `hook_card=True/False` cases — verify the still is picked/extracted
  from the pre-caption file (not the captioned one), `render_card_overlay`
  is called only when the flag is on and `on_screen_hook` is non-empty, and
  the `hook_card_error` fallback path leaves `clip_url` pointing at the
  pre-card (but still captioned) file.
- Manual: one real end-to-end run; eyeball that frame 0 is visually
  distinct from the raw footage's resting first second, the hook text is
  legible and bold, and no stray caption text is baked into the still.
