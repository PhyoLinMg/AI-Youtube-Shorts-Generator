# Fix locked-framing face cutoff on two-person clips via active-speaker switching

## Context

**Problem.** In `--mode local` (the default local pipeline), `framing="locked"` (the default framing) crops every highlight to a single, whole-clip-median face position (`shorts_generator/local/clipper.py:_reframe_vertical`, lines 77-170). This works well for single-camera talking-head content. It breaks on any clip filmed as a continuous two-person wide shot (e.g. a podcast host + guest seated side by side) — a 9:16 crop out of a 16:9 frame is only ~32% of the source width, so no single fixed crop window can show two people ~1200px apart at once. The current per-frame logic (`max(faces, key=area)`, highlights.py:127) then whole-clip-medians across samples, so it locks onto whichever person wins the "largest detected face" vote most often across the clip, cutting the other person off at the crop edge for however much of the clip they're on screen.

**Confirmed with real evidence** (2026-07-24 session): pulled frames from the most recent real run (`output/Does_The_Universe_Need_A_Creator/`, a Neil deGrasse Tyson podcast clip featuring guest "Chuck"). The source `full_source.mp4` at the corresponding timestamp is a genuine simultaneous two-shot (both men in frame together, ~1920px source width, one seated far left, one far right). The locked crop output for that clip shows Chuck's face bisected at the crop edge — literally half a face — while other frames of the same clip show Tyson reasonably framed. Confirmed this is a pre-existing limitation, not a regression from the 2026-07-24 hook-strength branch (that branch never touched `local/clipper.py` or any face-detection logic).

**Goal.** Detect when a locked-mode clip contains two distinct people and, for those clips only, switch the crop between each person's own locked anchor position based on who is actively speaking (via mouth-motion energy — no external ML model, matching this codebase's existing no-external-model philosophy for `framing="adaptive"`). Switches are a **hard cut**, not a pan (matches real multi-cam podcast editing conventions; a lateral pan on a static two-shot would read as an unmotivated camera move). Single-person clips — the majority of existing output, already working correctly — must get byte-identical crops to today; this is strictly an additive fallback-safe behavior inside `framing="locked"`, not a new CLI flag.

**Decisions locked with user (2026-07-24):**
- Fix direction: active-speaker switching (not a split-screen/stacked layout — that would introduce an unvalidated new visual format; not a "smarter single anchor" — that doesn't solve genuine back-and-forth dialogue clips like the confirmed example).
- Switch style: hard cut at speaker-change boundaries, no eased pan.
- Scope: `framing="locked"` only, `--mode local` only. `framing="adaptive"` and `--mode api` (MuAPI's own autocrop) are untouched.

---

## Phase 1 — Detect all faces per sampled frame (not just the largest)

In `_reframe_vertical`'s Pass 1 (`local/clipper.py:114-129`, the sampling loop), change from keeping only `max(faces, key=area)` to keeping **every** detected face's center per sampled frame: `List[List[Tuple[int, int]]]`, one list of (x,y) centers per sampled frame index (empty list if no faces that frame). This is the only change to the existing sampling loop itself — same cascade, same `sample_stride` cadence (~5fps).

## Phase 2 — Cluster face centers into up to 2 people

New pure function `_cluster_face_centers(sample_centers: List[List[Tuple[int, int]]], src_w: float) -> Tuple[List[Tuple[float, float]], List[int]]` (or similar shape) that:
- Flattens all detected centers across all sampled frames.
- Attempts a simple 1D split on the x-coordinate into up to 2 clusters (e.g. sort by x, find the largest gap; if the gap exceeds some fraction of `src_w` — a new constant, e.g. `TWO_PERSON_MIN_SEPARATION_FRAC = 0.25` — treat as 2 clusters, else 1).
- Requires each cluster to carry at least a minimum share of total detections (new constant `MIN_CLUSTER_SAMPLE_FRAC = 0.15`) to count as a real second person, not a stray misdetection.
- Returns the median (x, y) anchor per accepted cluster (same median logic as today's single-anchor calc, lines 131-136, just scoped per cluster) and a per-sampled-frame cluster-index assignment (nearest-cluster-by-x, or `None` if no face that frame).
- **Fallback:** if clustering doesn't resolve cleanly into exactly 2 well-supported clusters (0 faces at all, 1 cluster, or 3+ ambiguous clusters), return a single cluster — this is exactly today's existing single-anchor path, wired through unchanged. This is the no-regression guarantee for every clip that already works.

Write this as a pure function over plain lists/tuples (no cv2/video I/O inside it) so it's directly unit-testable with synthetic input.

## Phase 3 — Active-speaker classification via mouth-motion energy

Only runs when Phase 2 found 2 real clusters. New function, structurally parallel to the existing `_classify_frames`/`_apply_hysteresis` pair (lines 195-241) that already does exactly this shape of "raw per-frame class → hysteresis-smoothed class" for the adaptive mode's person/cursor decision:

- During a frame-by-frame decode pass, for each frame compute a grayscale frame-to-frame diff (`cv2.absdiff`, same primitive `_detect_cursor` already uses, lines 173-192) restricted to the lower third of each cluster's fixed face-box region (the box size comes from the median face detections in that cluster) — this is the mouth-motion-energy proxy for "is this person talking right now."
- Per short window, whichever cluster has higher summed motion energy is the raw "active speaker" for that window.
- **Reuse `_apply_hysteresis` as-is** (lines 224-241) — it already operates generically on a list of string class labels with a dwell threshold, nothing person/cursor-specific about its signature. Feed it `["A", "B", "A", ...]` per-frame raw labels; it returns the same shape hysteresis-smoothed timeline. No new smoothing primitive needed here. **New constant** `SPEAKER_DWELL_SECONDS`, separate from `MODE_DWELL_SECONDS` — conversational turn-taking cadence is a different thing from the screen/person mode-switch cadence `MODE_DWELL_SECONDS` was tuned for, so it gets its own value rather than reusing that one. Exact tuned value is an implementation-time detail, verified empirically against a real two-person clip (a lower starting guess than 0.75s is likely right, since speech turns can be quick).
- No face detected that frame in either cluster's region → hold last known active speaker (same "persist last known" pattern already used for `cursor_raw_points`, lines 359-363), never flip to "neither."

## Phase 4 — Render pass: hard-cut between static anchors

In the final write pass (today: lines 143-152, single fixed `x0,y0` for every frame), when in two-speaker mode: look up each frame's hysteresis-smoothed active-cluster label and crop at *that* cluster's fixed median anchor (from Phase 2) — no `_smooth_center`/EMA interpolation between the two anchors, since the user confirmed a hard cut is the desired feel and both anchors are already static locked points. Single-cluster (fallback) clips keep using the one fixed `x0,y0` exactly as today, unchanged code path.

## Phase 5 — Tests

New tests (add to wherever this repo's local-clipper tests currently live, e.g. `tests/test_local_clipper.py`):
- `_cluster_face_centers`: given a synthetic list of per-frame center-lists forming two well-separated x-clusters, returns 2 clusters with correct medians. Given centers all in one tight x-range, returns 1 cluster (fallback path). Given one dominant cluster plus a handful of stray outlier detections below `MIN_CLUSTER_SAMPLE_FRAC`, still returns 1 cluster (outliers don't spuriously trigger 2-speaker mode).
- Active-speaker classifier: given a synthetic motion-energy timeline that clearly favors cluster A then clearly favors cluster B, returns the correct raw labels; confirm `_apply_hysteresis` (already covered by existing tests, if any — check) suppresses a brief single-frame flicker.
- An explicit end-to-end-ish test (mocking `cv2` calls as the existing `test_local_clipper.py` tests already do) confirming a single-person synthetic scenario produces the exact same crop position as before this change (the no-regression guarantee).

---

## Verification

1. Unit tests green (`pytest tests/ -q`).
2. Re-render the confirmed-broken clip (`I_Was_a_Pentecostal_Minister_Chuck_s_Shocking_Confession_to_Neil_deGrasse_Tyson`, or re-run the pipeline on the same source video) and visually confirm: no bisected face at any sampled timestamp, and a visible hard-cut crop change at the point the active speaker changes.
3. Spot-check 2-3 existing single-person clips from prior runs are byte-identical (or visually identical) before/after — confirms the fallback path is truly a no-op for the common case.

## Out of scope

- `framing="adaptive"` and `--mode api` — untouched.
- 3+ person shots (panel/roundtable) — falls back to today's single-anchor behavior; not attempting multi-person clustering beyond 2.
- Split-screen/stacked visual layout — explicitly rejected in favor of active-speaker switching (see Decisions above).
- Any change to the hook-strength/highlight-picking logic (unrelated system, already shipped 2026-07-24).

## Critical files

- `shorts_generator/local/clipper.py` — all of the above (`_reframe_vertical`, new clustering/classification functions, reuse of `_apply_hysteresis`).
- `tests/test_local_clipper.py` — new tests per Phase 5.
