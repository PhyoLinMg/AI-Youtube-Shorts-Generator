# Scene-Cut-Aware Locked Framing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `framing="locked"` vertical cropping so a mid-clip camera cut (e.g. a close-up cutting to a split-screen, as happened in `The_Algae_Lake_Exponential_Thinking_Test.mp4`) doesn't drag a single whole-clip face anchor to a position that's wrong for part of the clip.

**Architecture:** Segment the clip at detected hard camera cuts (frame-to-frame grayscale diff spike), then run the existing single/two-speaker anchor logic independently per segment instead of once globally. Segments hard-cut at boundaries — no interpolation, matching the existing "no per-frame tracking" philosophy. Zero detected cuts = one segment = today's exact behavior (regression-safe by construction).

**Tech Stack:** Python, OpenCV (`cv2`), numpy — no new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-04-scene-cut-aware-locked-framing-design.md`

---

### Task 1: `_detect_scene_cuts`

**Files:**
- Modify: `shorts_generator/local/clipper.py` (add constant + function; insertion points below)
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_local_clipper.py`, after `test_cluster_face_centers_empty_input` (currently ends at line 104):

```python
def test_detect_scene_cuts_no_cuts_on_static_frames():
    grays = [np.full((50, 50), 100, dtype=np.uint8) for _ in range(5)]
    assert local_clipper_module._detect_scene_cuts(grays) == []


def test_detect_scene_cuts_detects_single_sharp_change():
    grays = [np.full((50, 50), 100, dtype=np.uint8) for _ in range(3)]
    grays += [np.full((50, 50), 220, dtype=np.uint8) for _ in range(3)]
    assert local_clipper_module._detect_scene_cuts(grays) == [3]


def test_detect_scene_cuts_ignores_diff_below_threshold():
    grays = [np.full((50, 50), 100, dtype=np.uint8) for _ in range(3)]
    grays += [np.full((50, 50), 110, dtype=np.uint8) for _ in range(3)]  # diff=10 < default 40
    assert local_clipper_module._detect_scene_cuts(grays) == []


def test_detect_scene_cuts_first_frame_never_flagged():
    grays = [np.full((50, 50), 255, dtype=np.uint8)] + [np.full((50, 50), 0, dtype=np.uint8) for _ in range(2)]
    cuts = local_clipper_module._detect_scene_cuts(grays)
    assert 0 not in cuts
    assert cuts == [1]


def test_detect_scene_cuts_empty_input_returns_empty():
    assert local_clipper_module._detect_scene_cuts([]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k detect_scene_cuts -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.local.clipper' has no attribute '_detect_scene_cuts'`

- [ ] **Step 3: Implement**

In `shorts_generator/local/clipper.py`, insert this constant into the tunables block, right after `CENTER_MAX_STEP = 10.0` (line 41) and before `OUTPUT_CANVAS_H = 1920` (line 43):

```python

# --- scene-cut segmentation tunables (locked framing) --------------------------
SCENE_CUT_DIFF_THRESHOLD = 40.0  # mean abs pixel diff (0-255) between consecutive
                                  # sampled frames to call it a hard camera cut
```

Insert the function right after `_two_speaker_positions` ends (line 165) and before `def _cut_subclip` (line 168):

```python

def _detect_scene_cuts(
    sampled_grays: List[np.ndarray], threshold: float = SCENE_CUT_DIFF_THRESHOLD,
) -> List[int]:
    """Indices into `sampled_grays` where frame i differs sharply from frame
    i - 1 (mean abs pixel diff over `threshold`) -- a hard camera cut. Pure
    numpy, no cv2 dependency, testable without a real video decode (same
    pattern as `_mouth_region_energy`). The first frame can never be a cut
    (nothing to diff against)."""
    cuts: List[int] = []
    for i in range(1, len(sampled_grays)):
        diff = np.abs(sampled_grays[i].astype(np.int16) - sampled_grays[i - 1].astype(np.int16))
        if float(diff.mean()) >= threshold:
            cuts.append(i)
    return cuts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k detect_scene_cuts -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "feat: add _detect_scene_cuts for locked-framing segmentation"
```

---

### Task 2: `_merge_short_segments`

**Files:**
- Modify: `shorts_generator/local/clipper.py`
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_local_clipper.py`, after the `_detect_scene_cuts` tests from Task 1:

```python
def test_merge_short_segments_keeps_well_separated_cuts():
    # sample_seconds=0.2, min_segment_seconds=2.0 -> need >=10 samples between cuts
    cuts = [15, 40]
    result = local_clipper_module._merge_short_segments(cuts, total_samples=80, sample_seconds=0.2)
    assert result == [15, 40]


def test_merge_short_segments_drops_cut_too_close_to_previous():
    cuts = [15, 17, 40]  # 17 is only 0.4s after 15 -> merged away
    result = local_clipper_module._merge_short_segments(cuts, total_samples=80, sample_seconds=0.2)
    assert result == [15, 40]


def test_merge_short_segments_drops_cut_leaving_short_tail():
    cuts = [15, 78]  # tail from sample 78 to 80 = 0.4s -> too short, drop 78
    result = local_clipper_module._merge_short_segments(cuts, total_samples=80, sample_seconds=0.2)
    assert result == [15]


def test_merge_short_segments_falls_back_to_empty_when_too_many_segments():
    # 6 cuts spaced 10 samples (2.0s) apart -> 7 segments, exceeds MAX_SEGMENTS_PER_CLIP=6
    cuts = [10, 20, 30, 40, 50, 60]
    result = local_clipper_module._merge_short_segments(cuts, total_samples=100, sample_seconds=0.2)
    assert result == []


def test_merge_short_segments_empty_input_returns_empty():
    assert local_clipper_module._merge_short_segments([], total_samples=50, sample_seconds=0.2) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k merge_short_segments -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.local.clipper' has no attribute '_merge_short_segments'`

- [ ] **Step 3: Implement**

In `shorts_generator/local/clipper.py`, add these two constants right after `SCENE_CUT_DIFF_THRESHOLD` (added in Task 1):

```python
MIN_SEGMENT_SECONDS = 2.0        # a segment shorter than this gets folded into
                                  # its neighbor rather than getting its own
                                  # (statistically unreliable) anchor
MAX_SEGMENTS_PER_CLIP = 6        # degrade-gracefully cap; beyond this the cut
                                  # detector is probably a poor fit for this
                                  # footage (e.g. fast handheld motion) and we
                                  # fall back to single-global-anchor behavior
```

Add the function right after `_detect_scene_cuts`:

```python

def _merge_short_segments(
    cut_sample_indices: List[int],
    total_samples: int,
    sample_seconds: float,
    min_segment_seconds: float = MIN_SEGMENT_SECONDS,
) -> List[int]:
    """Drop cuts that would create a segment shorter than
    `min_segment_seconds`, folding it into the preceding segment (including
    a short trailing segment after the last accepted cut). If more than
    `MAX_SEGMENTS_PER_CLIP` segments remain even after merging, drop all
    cuts -- the cut detector is probably a poor fit for this footage, and
    falling back to one whole-clip segment is never worse than today's
    baseline."""
    if not cut_sample_indices or sample_seconds <= 0:
        return list(cut_sample_indices)

    accepted: List[int] = []
    prev = 0
    for c in cut_sample_indices:
        if (c - prev) * sample_seconds >= min_segment_seconds:
            accepted.append(c)
            prev = c

    if accepted and (total_samples - accepted[-1]) * sample_seconds < min_segment_seconds:
        accepted.pop()

    if len(accepted) + 1 > MAX_SEGMENTS_PER_CLIP:
        return []
    return accepted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k merge_short_segments -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "feat: add _merge_short_segments to filter unreliable scene cuts"
```

---

### Task 3: Segment `_reframe_vertical`

This is the integration task: replace the whole-clip single-anchor computation
with a per-segment loop, using `_detect_scene_cuts` + `_merge_short_segments`
from Tasks 1-2. A zero-cut clip must produce byte-identical output to the
current code (verified by the existing regression tests continuing to pass
unchanged).

**Files:**
- Modify: `shorts_generator/local/clipper.py:183-314` (full function replacement)
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Add a fixture and a failing test for the multi-segment path**

Add to `tests/test_local_clipper.py`, right after the `synthetic_source` fixture (ends at line 30):

```python
@pytest.fixture(scope="module")
def synthetic_source_with_cut(tmp_path_factory):
    """An 8s clip with a hard cut at the midpoint: 4s solid white, then 4s
    solid black -- a large frame-to-frame pixel diff at the boundary, used
    to exercise scene-cut segmentation in _reframe_vertical."""
    tmp_dir = tmp_path_factory.mktemp("source_cut")
    path = str(tmp_dir / "source_cut.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=white:size=640x360:rate=24:duration=4",
            "-f", "lavfi", "-i", "color=c=black:size=640x360:rate=24:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map", "[v]", "-map", "2:a",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            path,
        ],
        check=True,
    )
    return path
```

Add this test after `test_reframe_vertical_two_speakers_completes_and_outputs_correct_size` (currently ends at line 253):

```python
def test_reframe_vertical_scene_cut_computes_independent_anchor_per_segment(
    tmp_path, synthetic_source_with_cut, monkeypatch,
):
    def _fake_detect(self, gray, *args, **kwargs):
        # white segment (mean ~255) -> face centered near x=105
        # black segment (mean ~0)   -> face centered near x=505
        if gray.mean() > 127:
            return [(60, 80, 90, 110)]
        return [(460, 80, 90, 110)]

    monkeypatch.setattr(cv2.CascadeClassifier, "detectMultiScale", _fake_detect)

    real_clamp = local_clipper_module._clamp_crop_origin
    centers = []

    def _spy_clamp(center, crop_size, src_size):
        centers.append(center)
        return real_clamp(center, crop_size, src_size)

    monkeypatch.setattr(local_clipper_module, "_clamp_crop_origin", _spy_clamp)

    out_path = str(tmp_path / "out_scene_cut.mp4")
    local_clipper_module._reframe_vertical(synthetic_source_with_cut, out_path, "9:16")

    assert os.path.exists(out_path)
    # one hard cut (white -> black) -> two independent anchor computations,
    # not one global anchor blended from both segments' face positions
    assert len(centers) == 2
    assert centers[0][0] < 300   # segment 1 anchored near its own face (x~105)
    assert centers[1][0] > 300   # segment 2 anchored near ITS face (x~505),
                                  # not dragged toward segment 1's position
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k scene_cut_computes_independent -v`
Expected: FAIL — `len(centers) == 1` (today's code computes one global anchor), assertion error on `len(centers) == 2`

- [ ] **Step 3: Replace `_reframe_vertical`**

Replace the entire function body in `shorts_generator/local/clipper.py` (lines 183-314) with:

```python
def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Crop the cut clip to the target aspect ratio, centered on the speaker.

    Segments the clip at detected camera cuts (see `_detect_scene_cuts`) and
    computes an independent locked anchor per segment -- a mid-clip cut to a
    different camera layout (e.g. close-up -> split-screen) no longer drags
    a single whole-clip anchor to a position that's wrong for part of the
    clip. Within a segment, framing is still one locked position (or a
    hard-cut two-speaker pair) for that segment's whole span -- per-frame
    tracking still reads as camera shake on a talking head, same reasoning
    as before this change; only segment *boundaries* are now detected
    instead of assumed not to exist. A clip with zero detected cuts is one
    segment spanning the whole clip -- byte-identical to the
    pre-segmentation code.
    """
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Pass 1 -- sample the clip (~5 frames/sec), collecting per-sample face
    # detections AND the grayscale sampled frame itself (for scene-cut
    # detection). A slot is appended for every sample, even with no face
    # detected, so sample index stays aligned with sample_grays for later
    # segment slicing. Sample i always lands at frame i * sample_stride,
    # since sampling is a fixed stride from frame 0.
    sample_stride = max(1, int(fps // 5))
    sample_grays: List[np.ndarray] = []
    sample_largest: List[Optional[Tuple[int, int]]] = []
    sample_detections: List[List[Tuple[float, float, float, float]]] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            sample_grays.append(gray)
            if len(faces) > 0:
                dets = [(fx + fw / 2, fy + fh / 2, float(fw), float(fh)) for (fx, fy, fw, fh) in faces]
                sample_detections.append(dets)
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                sample_largest.append((x + w // 2, y + h // 2))
            else:
                sample_detections.append([])
                sample_largest.append(None)
        frame_idx += 1
    total_frames = frame_idx

    out_w, out_h = _output_size(target_ratio)

    # Segment the clip at detected scene cuts.
    sample_seconds = sample_stride / fps
    raw_cuts = _detect_scene_cuts(sample_grays)
    cuts = _merge_short_segments(raw_cuts, total_samples=len(sample_grays), sample_seconds=sample_seconds)
    segment_bounds = [0] + [c * sample_stride for c in cuts] + [total_frames]

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (out_w, out_h))

    prev_anchor_center: Tuple[float, float] = (src_w / 2, src_h / 2)
    for seg_i in range(len(segment_bounds) - 1):
        seg_start, seg_end = segment_bounds[seg_i], segment_bounds[seg_i + 1]
        seg_sample_start = seg_start // sample_stride
        seg_sample_end = -(-seg_end // sample_stride)  # ceil division, exclusive
        seg_detections = [d for dets in sample_detections[seg_sample_start:seg_sample_end] for d in dets]
        seg_largest = [c for c in sample_largest[seg_sample_start:seg_sample_end] if c is not None]

        clusters = _cluster_face_centers(seg_detections, src_w)

        if len(clusters) < 2:
            two_speaker = False
            if seg_largest:
                xs = sorted(c[0] for c in seg_largest)
                ys = sorted(c[1] for c in seg_largest)
                cx, cy = xs[len(xs) // 2], ys[len(ys) // 2]
            elif seg_i > 0:
                # thin/no-detection segment: carry forward the previous
                # segment's anchor rather than snapping to frame-center
                cx, cy = prev_anchor_center
            else:
                cx, cy = src_w // 2, src_h // 2
            x0, y0 = _clamp_crop_origin((cx, cy), (crop_w, crop_h), (src_w, src_h))
            prev_anchor_center = (cx, cy)
        else:
            two_speaker = True
            anchor_a, anchor_b = clusters
            cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
            raw_labels: List[str] = []
            prev_gray = None
            for _ in range(seg_end - seg_start):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is None:
                    raw_labels.append("A")
                else:
                    energy_a = _mouth_region_energy(gray, prev_gray, anchor_a)
                    energy_b = _mouth_region_energy(gray, prev_gray, anchor_b)
                    raw_labels.append("A" if energy_a >= energy_b else "B")
                prev_gray = gray
            positions = _two_speaker_positions(
                anchor_a, anchor_b, raw_labels, fps, (crop_w, crop_h), (src_w, src_h),
            )
            prev_anchor_center = (anchor_a[0], anchor_a[1])

        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        for i in range(seg_end - seg_start):
            ret, frame = cap.read()
            if not ret:
                break
            if two_speaker:
                fx0, fy0 = positions[i] if i < len(positions) else positions[-1]
            else:
                fx0, fy0 = x0, y0
            cropped = frame[fy0:fy0 + crop_h, fx0:fx0 + crop_w]
            writer.write(cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4))

    cap.release()
    writer.release()

    # Mux audio from the cut clip back onto the silent reframed video.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path
```

- [ ] **Step 4: Run the full local-clipper test suite**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -v`
Expected: all tests pass, including:
- `test_reframe_vertical_falls_back_to_frame_center_when_no_faces` (no cuts on a no-face fixture)
- `test_reframe_vertical_single_person_locks_to_detected_face` (constant face position -> zero cuts -> one segment, same as today)
- `test_reframe_vertical_two_speakers_completes_and_outputs_correct_size` (`two_speaker_calls["n"] == 1` still holds -- zero cuts on this fixture, confirmed empirically before writing this plan: max frame diff ~3.5, threshold is 40)
- `test_reframe_vertical_scene_cut_computes_independent_anchor_per_segment` (new, from Step 1)

- [ ] **Step 5: Run the full project test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 0 failures (no other module imports or calls `_reframe_vertical`'s internals in a way this refactor could break — `crop_clip_local`/`crop_highlights_local` call it only through its public signature, unchanged)

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "feat: segment locked framing at scene cuts for independent per-shot anchors"
```

---

### Task 4: Manual verification on the real failing clip

**Files:** none (verification only, no code changes)

- [ ] **Step 1: Re-crop the real source range that exposed this bug**

```bash
.venv/bin/python3 -c "
from shorts_generator.local.clipper import crop_clip_local

crop_clip_local(
    'output/Alien_Bodies_Are_In_The_Lab_Astrophysicist_On_UFOs_DMT_Life_After_Death_Neil_deGrasse_Tyson/full_source.mp4',
    1977.6, 2029.8,  # The Algae Lake clip's start/end, from result_12.json
    '9:16',
    '/tmp/algae_lake_fixed.mp4',
    framing='locked',
)
print('done')
"
```

- [ ] **Step 2: Extract frames from both the close-up and split-screen portions**

```bash
mkdir -p /tmp/algae_check
for t in 1 25 45 50; do
  ffmpeg -y -loglevel error -ss $t -i /tmp/algae_lake_fixed.mp4 -frames:v 1 /tmp/algae_check/t_${t}.png
done
```

- [ ] **Step 3: Visually confirm**

Read `/tmp/algae_check/t_45.png` and `/tmp/algae_check/t_50.png` (the previously-broken split-screen portion) and confirm a face is visible in frame, not just shoulders/mic as in the original bug report. Confirm `/tmp/algae_check/t_1.png` and `/tmp/algae_check/t_25.png` (the close-up portion) still frame correctly, matching pre-fix behavior.

- [ ] **Step 4: Re-crop the fixed clip into the actual output (optional, only if verification passes and user wants the shipped clip updated)**

Same approach as the earlier clip-5 jump-cuts fix: call `crop_clip_local` (or the full `crop_highlights_local`, if captions/hook-card need re-applying too) writing directly to the clip's real path in `output/.../Shorts/`, then re-verify duration/frames.
