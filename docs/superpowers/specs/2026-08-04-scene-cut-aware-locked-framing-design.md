# Scene-Cut-Aware Locked Framing — Design

**Problem:** `framing="locked"` (the default) computes ONE face anchor for an
entire highlight clip, from a global pool of face detections sampled across
the whole clip. When the source video cuts mid-clip to a different camera
layout, that global pool mixes detections from incompatible layouts and the
resulting anchor is wrong for part (sometimes most) of the clip.

Concrete case: `The_Algae_Lake_Exponential_Thinking_Test.mp4` (from the
2026-08-03 12-clip Neil deGrasse Tyson run). First ~40s is a continuous
single-person close-up — locked crop frames correctly. Around `t=45s` the
source cuts to a two-panel split-screen (both speakers shown simultaneously,
divided by a vertical line) for the remainder of the clip. The clip's single
global anchor — computed from the close-up portion, which dominates the
sample pool — is wrong for the split-screen portion: both crop halves show
shoulders/mic, no face in either.

This matches a known gap already logged from the earlier active-speaker work
(`memory/project_active_speaker_locked_framing_shipped.md`): the design
assumes one continuous shot per clip. It was accepted as a documented
limitation at the time; this spec closes it.

**Non-goals:**
- `framing="adaptive"` is untouched — it already re-evaluates per rolling
  window and isn't implicated in this bug.
- No audio/diarization. No face-identity/naming. Still frame-diff/Haar only,
  same tools as today.
- No per-frame tracking within a shot — the "one fixed anchor per shot, hard
  cut between shots" philosophy is preserved; only shot *boundaries* are now
  detected instead of assumed to not exist.

## Current flow (for reference)

`_reframe_vertical` (`shorts_generator/local/clipper.py`):
1. Pass 1: sample ~5 frames/sec across the whole clip, run Haar face
   detection each sampled frame, collect `all_detections` (every face box)
   and `largest_per_frame` (biggest face's center per sampled frame).
2. `_cluster_face_centers(all_detections, src_w)` → 1 or 2 clusters.
3. `< 2` clusters: anchor = median of `largest_per_frame` over the **whole
   clip**.
   `2` clusters: per-frame mouth-energy labeling (`_mouth_region_energy`)
   over the **whole clip**, hysteresis-smoothed, hard-cut between two fixed
   positions (`_two_speaker_positions`).
4. Pass 2: render every frame using the position(s) from step 3.

## New flow

Insert a segmentation step between Pass 1's sampling and anchor computation.
Steps 2-3 above become **per-segment** instead of whole-clip; step 4 is
unchanged except it looks up a position from the concatenated multi-segment
list.

### `_detect_scene_cuts`

```python
SCENE_CUT_DIFF_THRESHOLD = 40.0   # mean abs pixel diff (0-255 scale) between
                                   # consecutive sampled frames to call it a cut

def _detect_scene_cuts(sampled_grays: List[np.ndarray], threshold: float = SCENE_CUT_DIFF_THRESHOLD) -> List[int]:
    """Indices into `sampled_grays` where frame i differs sharply from frame
    i-1 (mean abs pixel diff over threshold) -- a hard cut. Pure numpy, no
    cv2 dependency, testable without real video decode (same pattern as
    _mouth_region_energy). First frame can never be a cut (nothing to diff
    against), so the returned list only contains indices >= 1."""
```

Computed inline in Pass 1's existing sample loop (`sampled_grays` is the
same per-sample grayscale frame already produced for face detection) — no
extra decode pass.

### `_merge_short_segments`

```python
MIN_SEGMENT_SECONDS = 2.0   # a segment shorter than this gets folded into
                             # its neighbor rather than getting its own
                             # (statistically unreliable) anchor
MAX_SEGMENTS_PER_CLIP = 6   # degrade-gracefully cap; beyond this the cut
                             # detector is probably a poor fit for this
                             # footage (e.g. fast handheld motion) and we
                             # fall back to single-global-anchor behavior

def _merge_short_segments(cut_sample_indices: List[int], total_samples: int, sample_seconds: float, min_segment_seconds: float = MIN_SEGMENT_SECONDS) -> List[int]:
    """Drop cuts that would create a segment (in sampled-frame count *
    sample_seconds) shorter than min_segment_seconds, folding it into the
    preceding segment. Returns the filtered cut list."""
```

If `len(merged_cuts) + 1 > MAX_SEGMENTS_PER_CLIP` after merging, drop all
cuts (return `[]`) — this is the fallback-to-today's-behavior path, so a
single segment spans the whole clip exactly as it does now.

### Per-segment anchor computation

Refactor of existing logic, not new logic: `_reframe_vertical`'s current
steps 2-3 move into a loop over segments (each segment = a contiguous slice
of sampled-frame indices, derived from the merged cut list). Each iteration
calls the *existing* `_cluster_face_centers` / median / `_mouth_region_energy`
/ `_two_speaker_positions` functions, scoped to that segment's own
`all_detections` / `largest_per_frame` subset and its own frame-index range
for the mouth-energy pass. Output per segment: either one fixed `(x0, y0)`
or a `positions` list covering that segment's frame range — concatenated
across segments (in frame order) into the same flat per-frame position list
Pass 2 already consumes.

**Thin-segment fallback:** if a (post-merge) segment still has too few face
detections to cluster reliably (empty `all_detections`), carry forward the
previous segment's anchor rather than defaulting to frame-center — avoids a
jarring center-crop flash. The clip's first segment, if thin, falls back to
frame-center (today's existing no-detection behavior — nothing to carry
forward from).

## Error handling / edge cases

| Case | Behavior |
|---|---|
| Zero cuts detected | Identical to today: one segment, one global anchor. Regression safety net. |
| Cuts detected, all survive merge | Independent anchor per segment, hard-cut at boundaries. |
| Excess cuts (`> MAX_SEGMENTS_PER_CLIP` after merge) | All cuts dropped; falls back to single-global-anchor (today's behavior) — never worse than baseline. |
| A segment has no face detections | Carry forward previous segment's anchor; first segment falls back to frame-center. |

## Testing

- `_detect_scene_cuts`: synthetic `sampled_grays` arrays (numpy, no video) —
  no diff → `[]`; one sharp diff → single index; diff below threshold →
  not detected; first frame never flagged.
- `_merge_short_segments`: cuts closer together than `min_segment_seconds`
  get dropped; cuts far enough apart survive; cut count exceeding
  `MAX_SEGMENTS_PER_CLIP` after merge → `[]`.
- Per-segment anchor assignment: synthetic multi-segment `all_detections`
  where segment A's faces are far from segment B's — assert segment B's
  computed anchor is NOT dragged toward segment A's positions (this is the
  actual bug, asserted directly).
- Regression: zero-cut synthetic input produces byte-identical positions to
  the current single-global-anchor code path (same fixture data, before vs.
  after this change).
- Manual verification: re-crop `The_Algae_Lake_Exponential_Thinking_Test.mp4`
  from the real source with the fix, extract frames from both the close-up
  portion (~t=1-40s) and the split-screen portion (~t=45-52s), visually
  confirm faces are framed in both (same method used to diagnose this).

## Definition of done

- `python -m pytest tests/ -q` passes, including new tests above, zero
  regressions on existing framing tests (single-person byte-identical
  guarantee holds).
- Re-cropping clip 3's source range produces a crop with a visible face in
  both the close-up and split-screen portions.
- `framing="adaptive"` code path untouched.
