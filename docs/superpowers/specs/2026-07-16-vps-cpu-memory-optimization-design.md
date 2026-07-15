# VPS CPU/memory optimization (local mode) — design

## Problem

The user is planning to host this on a VPS with **2 vCPUs / 8GB RAM**, running
**`--mode local`** (self-hosted: yt-dlp + faster-whisper + OpenCV framing +
ffmpeg), using both `--framing locked` and `--framing adaptive`. A prior
codebase audit (`mds/scan_result.md`) found several CPU-cost issues in the
local clipping path; this design scopes which of those (plus additional
transcription-side findings surfaced by a design review) are worth fixing
before hosting on a 2-core box, and in what order.

## Goal

Reduce CPU time per run in `mode=local` without touching output quality,
covering both halves of the CPU bill: per-clip framing/encoding *and*
whole-video transcription (the latter was under-weighted in the initial pass
and corrected after an advisor review). Confirm — not "optimize" — that
memory is a non-issue at 8GB, so no memory workstream is invented.

## Explicitly out of scope

- **Auth / CSRF / access control.** The dashboard will be reachable at a
  public IP with no login. The user has chosen to handle access control
  (reverse proxy, allowlist, etc.) separately — this design does not touch
  `webapp.py` routes or add authentication.
- **Deployment ops** — swapping the Flask dev server for a production WSGI
  server, systemd units, swapfile sizing. Worth doing before real hosting, but
  a separate follow-up pass, not CPU/memory code optimization.
- **`mode=api`** — untouched. All changes are scoped to `shorts_generator/local/*`
  and `shorts_generator/captions.py`.
- **Memory optimization work.** See "Memory — non-goal" below.

## Threat/constraint model

- 2 vCPUs, 8GB RAM, single VPS, single concurrent job (the existing
  `_job_lock` one-run-at-a-time constraint in `webapp.py` is correct for this
  box and is not being changed).
- Both `locked` (speaker-centered, static crop) and `adaptive`
  (cursor/person-aware, screen-recording) framing will be used in production,
  so fixes to both paths are in scope.
- `faster-whisper` model stays at the current default (`base`, `int8` on
  CPU) — confirmed as the right choice for 2 cores, not changing.

## Components, in priority order

Order reflects actual CPU payoff, corrected after an advisor review found the
initial draft under-weighted transcription (which runs once over the *whole*
source video, not per-clip, making it the other major cost center in local
mode alongside per-clip framing/encoding).

### 1. `_cut_subclip` seek-order fix — `shorts_generator/local/clipper.py:46-58`

**Problem:** `-ss` is placed after `-i` (output seeking) — ffmpeg decodes
every frame from `t=0` to `start` and discards it before encoding the wanted
range. For a highlight late in a long source, this is a large, repeated,
wasted decode (once per highlight clipped from that source).

**Fix:**
```
# before
["ffmpeg", "-y", "-loglevel", "error",
 "-i", source_path,
 "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
 "-c:v", "libx264", "-preset", "fast", "-crf", "20",
 "-c:a", "aac", "-b:a", "128k",
 out_path]

# after
["ffmpeg", "-y", "-loglevel", "error",
 "-ss", f"{start:.3f}", "-i", source_path,
 "-t", f"{end - start:.3f}",
 "-c:v", "libx264", "-preset", "fast", "-crf", "20",
 "-c:a", "aac", "-b:a", "128k",
 out_path]
```

**Verified caveat:** with `-ss` before `-i`, `-to` is measured from the seek
point, not the source's absolute timeline — keeping `-to end` after moving
`-ss` would produce a clip of length `end` seconds (not `end - start`). This
was confirmed empirically in this session (ffmpeg 8.1.2, synthetic 30s
source): `-ss 10 -i src -to 15` produced a ~15s output, not ~5s. The fix uses
`-t (end - start)`, which is unambiguous regardless of `-ss` placement.

**Impact:** input seeking is keyframe-based, not full-decode — this is the
single largest CPU win in the plan, especially for highlights that land late
in long sources.

**Risk:** low. Output-equivalent, seek mechanism only; verified independently
of the rest of this design.

### 2. Crop + caption single-pass fusion (locked framing) — `local/clipper.py`, `captions.py`

**Problem:** the current locked-framing path re-encodes the same footage
three times: (1) `_cut_subclip` (ffmpeg libx264), (2) `_reframe_vertical`'s
OpenCV `VideoWriter` pass (mp4v) + a separate ffmpeg mux-audio-back call, (3)
`burn_captions`'s ffmpeg libx264 subtitle burn-in. Since captions are on by
default, nearly every locked-mode clip pays for all three.

**Fix:** once pass 1 of `_reframe_vertical` determines the static crop box
`(x0, y0, crop_w, crop_h)`, skip the OpenCV `VideoWriter` pass and the
separate mux entirely. Build the `.ass` caption file (reusing the chunking
logic already in `captions.py`) sized to `crop_w × crop_h` — not the
`_probe_resolution`-detected size, since the cropped file doesn't exist yet,
it's being created in the same ffmpeg call — then run one ffmpeg pass:

```
ffmpeg -y -loglevel error -i cut.mp4 \
  -vf "crop=crop_w:crop_h:x0:y0,subtitles=ass_path" \
  -c:v libx264 -preset fast -crf 20 -c:a aac -b:a 128k \
  out_path
```

This drops video re-encode count on the default path from **3 → 2**
(cut, then one crop+caption pass) and removes the intermediate
`.silent.mp4` file and its separate mux step.

**New interface needed:** a public helper in `captions.py`, e.g.
`build_ass_file(segments, clip_start, clip_end, width, height, fade_seconds, word_highlight, ass_path) -> str`,
factored out of the existing `_chunk_segments` + `_write_ass` internals, so
`local/clipper.py` can build the ASS file directly from known crop
dimensions instead of going through `burn_captions` (which currently probes
the video file's resolution — the wrong order once crop and caption happen
in the same pass).

**Fallback behavior:** if the fused crop+subtitles ffmpeg call fails (e.g. a
malformed ASS edge case), retry with the `crop` filter alone (no
`subtitles`), producing an uncaptioned clip and recording `captions_error` —
mirroring the existing fallback contract in `crop_highlights_local` /
`crop_highlights` (API mode) rather than failing the whole clip.

**Scope note:** this fusion applies to **locked framing only**. Adaptive
framing's crop moves per-frame, so it cannot be expressed as a static ffmpeg
`crop` filter argument — it keeps its existing OpenCV write pass, followed by
the existing separate `burn_captions` call, unchanged by this item (it does
get the stride/`INTER_AREA` fixes in item 5).

**Risk:** medium — this is the most invasive change in the plan (new
`captions.py` public function, restructured call path through
`crop_clip_local`/`crop_highlights_local` for the locked+captions-on case).
Mitigated by the testing plan below (dimension assertion + fallback test).

### 3. `beam_size` 5 → 1 (configurable) — `shorts_generator/local/transcriber.py:92`, `shorts_generator/config.py`

**Problem:** `faster-whisper`'s `beam_size=5` is hardcoded. Transcription
runs once over the *entire* source video (not per-clip), making it a major
share of total CPU time for a run — a share the initial draft of this design
under-weighted by focusing only on the per-clip framing/encoding path.

**Fix:** switch the default to `beam_size=1` (greedy decoding) — a real
~1.5x CPU reduction on the transcription pass, with negligible accuracy loss
on clear speech (podcasts/interviews/talking-head content, which is this
tool's primary use case). Expose as a new env var `LOCAL_WHISPER_BEAM_SIZE`
(default `1`), following the existing pattern of tunable whisper settings
(`LOCAL_WHISPER_MODEL`, `LOCAL_WHISPER_DEVICE`, `LOCAL_WHISPER_VAD_FILTER`).

**No code change needed for VAD:** `LOCAL_WHISPER_VAD_FILTER` (`config.py:31`)
is already an opt-in toggle, off by default. Document in the VPS notes
(README) that enabling it can skip silence for a speed win on content with
long pauses, with the existing caveat that it's aggressive on music/mixed
content — a documentation addition, not a code change.

**Risk:** low. Config-level default change, independently toggleable.

### 4. `cap.grab()` skip-decode in locked-mode pass 1 — `local/clipper.py:102-113`

**Problem:** pass 1 of `_reframe_vertical` calls `cap.read()` (full decode)
on every frame, even though the face cascade only runs on every
`sample_stride`-th frame (`sample_stride = max(1, int(fps // 5))`, line 100)
— i.e. most decoded frames are discarded unused.

**Fix:** call `cap.grab()` (decode-free frame advance) on non-sampled
frames, only fully decoding (`retrieve()`) on stride hits. Cuts pass-1 decode
cost by roughly `sample_stride`x (typically ~5-6x at 30fps).

**Risk:** low. Behavior-preserving — same sampled frames feed the same
median-position calculation.

### 5. Adaptive framing: stride + cheaper resize — `local/clipper.py:178-204, 378`

In scope because the user confirmed `--framing adaptive` will be used in
production (screen-recording content).

**5a. `_classify_frames` has no stride.** Unlike locked mode's pass 1, it
runs Haar cascade face detection *and* cursor blob detection on **every**
frame. Fix: add the same `fps // 5` stride, holding the last classified
`(cls, anchor)` between samples. The existing hysteresis
(`MODE_DWELL_SECONDS = 0.75s`) already tolerates coarser sampling than
per-frame, so this should cost negligible quality for a large cut in the
heaviest per-frame work in the adaptive path.

**5b. `INTER_LANCZOS4` → `INTER_AREA`** at `local/clipper.py:378`. Lanczos is
the most expensive standard OpenCV interpolation kernel; `INTER_AREA` is
OpenCV's recommended choice for downscaling and materially cheaper, with
comparable visual quality for this use case.

**Risk:** low-medium. 5a changes sampling density (mitigated by existing
hysteresis tolerance); 5b is a drop-in interpolation swap.

### 6. Thread pinning (insurance, not a speedup) — `config.py` + ffmpeg/whisper call sites

**Clarification from review:** on a VPS whose 2 vCPUs are correctly reported
to the OS, libx264 and ctranslate2 (faster-whisper's backend) already use
both cores by default — explicit thread pinning does **not** make anything
faster there. Its value is as insurance against a container/cgroup
misreporting the available core count (common on some VPS/container
setups), which could otherwise cause oversubscription and contention.

**Fix:** add a `VPS_CPU_THREADS` env var (default: unset → let ffmpeg/
ctranslate2 auto-detect as today). When set, thread it into ffmpeg calls as
`-threads N` and into `WhisperModel(..., cpu_threads=N)` in
`local/transcriber.py`. The user sets it to `2` explicitly in the VPS `.env`
if they observe oversubscription; it is not expected to change performance
on a box that reports its core count correctly.

**Risk:** low. Additive, opt-in, no behavior change when unset.

## Memory — non-goal (stated explicitly, not left implied)

The pipeline is confirmed **CPU-bound, not memory-bound**, at the 8GB target:

- `faster-whisper` `base` model at `int8` compute type uses roughly ~1GB RAM.
  This was already the right choice before this design (advisor review
  confirmed it, not changing it).
- The framing/reframe pipeline streams frame-by-frame
  (`cap.read()`/`grab()` → transform → `writer.write()`); retained state
  across a whole clip is O(frame-count) *numbers* (centers, zooms, classes —
  floats and small tuples), never buffered frames.
- Single-job-at-a-time (`_job_lock` in `webapp.py`) means there is never
  concurrent memory pressure from overlapping runs.

8GB has large headroom under this profile. No memory-focused changes are
part of this design, and none should be invented to fill out the plan.

## Data flow (per clip, locked framing, captions on — the fused path)

1. `_cut_subclip(source, start, end)` → `cut_path` (fast, input-seeked).
2. Pass 1 over `cut_path`: grab-skip sampling → median face position →
   static crop box `(x0, y0, crop_w, crop_h)`.
3. Build `.ass` file from `crop_w × crop_h` + transcript segments clipped to
   `[start, end]` (new `captions.py` helper).
4. Single ffmpeg pass: `crop` + `subtitles` filter → `out_path`. On failure,
   retry `crop`-only → `out_path`, record `captions_error`.
5. Cleanup: remove `cut_path`, the `.ass` file.

Adaptive framing keeps its existing two-pass classify-then-write OpenCV
flow (with the item 5 fixes applied), followed by the existing separate
`burn_captions()` call — unchanged in structure by this design.

## Error handling

- Fused ffmpeg pass failure → retry crop-only, same `captions_error`
  contract as today (caller already handles this — see
  `crop_highlights_local`'s existing `try/except (CaptionError, ...)`
  pattern).
- New env vars (`LOCAL_WHISPER_BEAM_SIZE`, `VPS_CPU_THREADS`) parse
  defensively: non-numeric or out-of-range values fall back to the current
  hardcoded default, consistent with existing `config.py` patterns (e.g.
  `POLL_INTERVAL_SECONDS`, `LOCAL_WHISPER_VAD_FILTER`).
- Per-highlight `try/except` in `crop_highlights_local` is unchanged — one
  highlight failing still doesn't kill the batch.

## Testing

- **Dimension assertion (new, cheap):** add an `ffprobe`-based check to
  `tests/test_local_clipper.py` asserting output width:height matches the
  target aspect ratio (±1px for even-dimension rounding). Existing tests are
  integration-style (real ffmpeg/OpenCV against the `synthetic_source`
  fixture, no mocking) — this fits the same style and is the cheapest guard
  against a botched crop/caption fusion.
- **Fusion fallback test:** monkeypatch the fused ffmpeg call to fail once,
  assert fallback to a crop-only clip with `captions_error` recorded —
  mirrors the existing `test_caption_failure_falls_back_to_plain_clip`
  pattern.
- **Adaptive stride check:** a lightweight test/count assertion that Haar
  cascade + cursor detection calls scale with the stride, not the raw frame
  count, for a known-length synthetic clip.
- No new test needed for `beam_size` or thread-pinning — both are config
  plumbing (passed straight to library calls), not independently testable
  behavior.

## Documentation updates

- **README:** new "Running on a resource-constrained VPS" section —
  recommended `.env` settings (`LOCAL_WHISPER_BEAM_SIZE=1`,
  `VPS_CPU_THREADS=2` if oversubscription is observed, optionally
  `LOCAL_WHISPER_VAD_FILTER=true` for content with long silences), and an
  explicit note that auth/access-control hardening is a separate concern not
  covered by this work (the dashboard has no login).
- **`.env.example`:** add `LOCAL_WHISPER_BEAM_SIZE` and `VPS_CPU_THREADS`
  with brief comments.

## Work order

1. `_cut_subclip` seek-order fix
2. Crop + caption single-pass fusion (locked framing)
3. `beam_size` → 1 (configurable)
4. `cap.grab()` skip-decode (locked pass 1)
5. Adaptive stride + `INTER_AREA` swap
6. Thread pinning (insurance)
