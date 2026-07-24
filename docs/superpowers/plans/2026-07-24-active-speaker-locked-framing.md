# Active-Speaker-Aware Locked Framing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `framing="locked"` (the default, `--mode local`) cutting off one speaker's face on two-person side-by-side clips, by detecting when a clip has two distinct people and hard-cutting the crop between each person's own locked anchor based on mouth-motion-energy-detected active speaker — while leaving every single-person clip's output byte-identical to today.

**Architecture:** All changes live in `shorts_generator/local/clipper.py`. A new clustering step splits sampled face detections into 1 or 2 people; a new mouth-motion-energy pass (only run when 2 people are detected) decides who's talking per frame; the existing `_apply_hysteresis` dwell-smoother (already used by `framing="adaptive"`) is extended with a configurable dwell and reused unchanged otherwise. The single-person path is refactored to call the same new clustering step but is mathematically guaranteed identical output when only 1 cluster is found — no new code runs for the common case.

**Tech Stack:** Python, OpenCV (`cv2`), NumPy — no new dependencies, no external ML models (matches this codebase's existing philosophy for `framing="adaptive"`).

**Spec:** `mds/active-speaker-locked-framing.md`

---

### Task 1: Extract `_clamp_crop_origin` (DRY prep, no behavior change)

The crop-origin clamp math (`x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))`, same for y0) is currently duplicated in `_reframe_vertical` (`shorts_generator/local/clipper.py:138-139`) and `_reframe_vertical_adaptive` (`shorts_generator/local/clipper.py:391-392`). Extracting it into one pure, testable function now means every later task can reuse it instead of re-deriving the clamp math, and gives us a directly unit-testable piece of the crop logic without needing any video/cv2 mocking.

**Files:**
- Modify: `shorts_generator/local/clipper.py`
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_local_clipper.py` (near the top, after the existing imports — add `from shorts_generator.local.clipper import _clamp_crop_origin` to the import block):

```python
def test_clamp_crop_origin_centers_when_room_on_both_sides():
    # src 1000x1000, crop 200x200, center at 500,500 -> origin should center it
    assert _clamp_crop_origin((500.0, 500.0), (200, 200), (1000, 1000)) == (400, 400)


def test_clamp_crop_origin_clamps_to_left_edge():
    # center near x=0 would push origin negative -> clamp to 0
    assert _clamp_crop_origin((10.0, 500.0), (200, 200), (1000, 1000)) == (0, 400)


def test_clamp_crop_origin_clamps_to_right_edge():
    # center near x=src_w would push origin past src_w - crop_w -> clamp there
    assert _clamp_crop_origin((990.0, 500.0), (200, 200), (1000, 1000)) == (800, 400)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k clamp_crop_origin -v`
Expected: FAIL with `ImportError: cannot import name '_clamp_crop_origin'`

- [ ] **Step 3: Implement `_clamp_crop_origin` and wire both call sites**

In `shorts_generator/local/clipper.py`, add this function right before `_reframe_vertical` (i.e. just after `_ratio`, before `_cut_subclip`):

```python
def _clamp_crop_origin(
    center: Tuple[float, float],
    crop_size: Tuple[int, int],
    src_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Top-left origin of a `crop_size` window centered on `center`, clamped
    so the window never runs off the `src_size` frame."""
    cx, cy = center
    crop_w, crop_h = crop_size
    src_w, src_h = src_size
    x0 = max(0, min(src_w - crop_w, int(cx - crop_w // 2)))
    y0 = max(0, min(src_h - crop_h, int(cy - crop_h // 2)))
    return x0, y0
```

Then in `_reframe_vertical`, replace (around line 138-139):

```python
    x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
    y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
```

with:

```python
    x0, y0 = _clamp_crop_origin((cx, cy), (crop_w, crop_h), (src_w, src_h))
```

And in `_reframe_vertical_adaptive`, replace (around line 391-392):

```python
        x0 = max(0, min(src_w - crop_w, int(cx - crop_w // 2)))
        y0 = max(0, min(src_h - crop_h, int(cy - crop_h // 2)))
```

with:

```python
        x0, y0 = _clamp_crop_origin((cx, cy), (crop_w, crop_h), (src_w, src_h))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k clamp_crop_origin -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all tests pass (same count as before this task, since this is a pure refactor)

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "refactor: extract _clamp_crop_origin from duplicated crop clamp math"
```

---

### Task 2: Add configurable dwell to `_apply_hysteresis`

`_apply_hysteresis` (`shorts_generator/local/clipper.py:224-241`) hardcodes `MODE_DWELL_SECONDS` inside its body. The active-speaker switch needs its own, faster dwell (conversational turn-taking is quicker than the screen/person mode-switch cadence this constant was tuned for), so this task makes the dwell configurable while defaulting to today's exact behavior for the existing caller.

**Files:**
- Modify: `shorts_generator/local/clipper.py`
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_local_clipper.py` (add `_apply_hysteresis` to the import from `shorts_generator.local.clipper`):

```python
def test_apply_hysteresis_default_dwell_matches_mode_dwell_seconds():
    # fps=10, MODE_DWELL_SECONDS=0.75 -> dwell=8 frames; a 5-frame flip (< dwell) must not stick
    raw = ["person"] * 10 + ["cursor"] * 5 + ["person"] * 10
    result = local_clipper_module._apply_hysteresis(raw, fps=10.0)
    assert result == ["person"] * 25  # the 5-frame cursor blip never persisted long enough to flip


def test_apply_hysteresis_custom_dwell_flips_faster():
    # same raw sequence, but dwell_seconds=0.3 -> dwell=3 frames; a 5-frame blip DOES flip
    raw = ["A"] * 10 + ["B"] * 5 + ["A"] * 10
    result = local_clipper_module._apply_hysteresis(raw, fps=10.0, dwell_seconds=0.3)
    assert result == ["A"] * 12 + ["B"] * 3 + ["A"] * 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k apply_hysteresis -v`
Expected: FAIL — `test_apply_hysteresis_custom_dwell_flips_faster` fails with `TypeError: _apply_hysteresis() got an unexpected keyword argument 'dwell_seconds'`

- [ ] **Step 3: Implement the parameter**

In `shorts_generator/local/clipper.py`, change the signature and first line of `_apply_hysteresis` (lines 224-226):

```python
def _apply_hysteresis(raw_classes: List[str], fps: float) -> List[str]:
    """Only flip mode once the opposite raw class persists >= dwell frames."""
    dwell = max(1, int(round(MODE_DWELL_SECONDS * fps)))
```

to:

```python
def _apply_hysteresis(
    raw_classes: List[str], fps: float, dwell_seconds: float = MODE_DWELL_SECONDS,
) -> List[str]:
    """Only flip mode once the opposite raw class persists >= dwell frames."""
    dwell = max(1, int(round(dwell_seconds * fps)))
```

The rest of the function body is unchanged. The existing call site in `_reframe_vertical_adaptive` (`_apply_hysteresis(raw_classes, fps)`) does not need to change — it keeps using the default.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k apply_hysteresis -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "feat: make _apply_hysteresis's dwell window configurable"
```

---

### Task 3: Implement `_cluster_face_centers`

New pure function that decides whether a clip's sampled face detections represent 1 or 2 distinct people, and returns each person's median anchor. This is the core "is this a two-person clip" decision.

**Files:**
- Modify: `shorts_generator/local/clipper.py`
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_local_clipper.py` (add `_cluster_face_centers` to the import):

```python
def test_cluster_face_centers_single_person_tight_range():
    # all detections clustered around x=500 on a 1000-wide frame -> 1 cluster
    detections = [(495.0, 300.0, 100.0, 120.0), (505.0, 305.0, 98.0, 118.0), (500.0, 298.0, 102.0, 121.0)]
    clusters = local_clipper_module._cluster_face_centers(detections, src_w=1000.0)
    assert len(clusters) == 1
    assert clusters[0][0] == pytest.approx(500.0, abs=10)


def test_cluster_face_centers_two_well_separated_people():
    # 10 detections around x=150, 10 around x=850 on a 1000-wide frame -> 2 clusters
    left = [(150.0 + i, 300.0, 100.0, 120.0) for i in range(10)]
    right = [(850.0 + i, 320.0, 100.0, 120.0) for i in range(10)]
    clusters = local_clipper_module._cluster_face_centers(left + right, src_w=1000.0)
    assert len(clusters) == 2
    assert clusters[0][0] < 300  # left cluster first (sorted by x ascending)
    assert clusters[1][0] > 700


def test_cluster_face_centers_stray_outlier_does_not_split():
    # 19 detections around x=500 plus 1 stray outlier at x=950 -> outlier is
    # below MIN_CLUSTER_SAMPLE_FRAC, must NOT be treated as a second person
    main_group = [(500.0 + i, 300.0, 100.0, 120.0) for i in range(19)]
    outlier = [(950.0, 300.0, 100.0, 120.0)]
    clusters = local_clipper_module._cluster_face_centers(main_group + outlier, src_w=1000.0)
    assert len(clusters) == 1


def test_cluster_face_centers_empty_input():
    assert local_clipper_module._cluster_face_centers([], src_w=1000.0) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k cluster_face_centers -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.local.clipper' has no attribute '_cluster_face_centers'`

- [ ] **Step 3: Implement `_cluster_face_centers` and its constants**

In `shorts_generator/local/clipper.py`, add these two constants to the "adaptive framing tunables" block (near `PERSON_FACE_MIN_W_FRAC`, lines 28-36):

```python
TWO_PERSON_MIN_SEPARATION_FRAC = 0.25   # min x-gap (as a fraction of src_w) to call it 2 distinct people
MIN_CLUSTER_SAMPLE_FRAC = 0.15          # each cluster must own at least this share of all detections
SPEAKER_DWELL_SECONDS = 0.35            # active-speaker switch dwell — faster than MODE_DWELL_SECONDS,
                                         # since conversational turn-taking is quicker than screen/person mode switches
```

Add the function itself right after `_clamp_crop_origin` (from Task 1):

```python
def _cluster_face_centers(
    detections: List[Tuple[float, float, float, float]],
    src_w: float,
) -> List[Tuple[float, float, float, float]]:
    """Cluster (cx, cy, w, h) face detections into 1 or 2 people by x-position.

    Returns one (cx, cy, w, h) median entry per accepted cluster, sorted by cx
    ascending. Falls back to a single cluster (the median of everything)
    unless the split is both well-separated (>= TWO_PERSON_MIN_SEPARATION_FRAC
    * src_w gap) and well-supported on both sides (>= MIN_CLUSTER_SAMPLE_FRAC
    of all detections each) — this is what keeps single-person clips and
    stray misdetections on the single-anchor path.
    """
    if not detections:
        return []

    def _median(vals: List[float]) -> float:
        s = sorted(vals)
        return s[len(s) // 2]

    def _cluster_median(group: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
        return (
            _median([d[0] for d in group]),
            _median([d[1] for d in group]),
            _median([d[2] for d in group]),
            _median([d[3] for d in group]),
        )

    by_x = sorted(detections, key=lambda d: d[0])
    best_gap = 0.0
    best_split = None
    for i in range(1, len(by_x)):
        gap = by_x[i][0] - by_x[i - 1][0]
        if gap > best_gap:
            best_gap = gap
            best_split = i

    if best_split is not None and best_gap >= TWO_PERSON_MIN_SEPARATION_FRAC * src_w:
        left, right = by_x[:best_split], by_x[best_split:]
        min_count = MIN_CLUSTER_SAMPLE_FRAC * len(by_x)
        if len(left) >= min_count and len(right) >= min_count:
            return [_cluster_median(left), _cluster_median(right)]

    return [_cluster_median(by_x)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k cluster_face_centers -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "feat: add _cluster_face_centers for two-person detection"
```

---

### Task 4: Implement `_mouth_region_energy`

New pure function (numpy only, no cv2 needed) that measures how much a face's lower-third region changed between two consecutive frames — the mouth-motion-energy proxy for "is this person talking."

**Files:**
- Modify: `shorts_generator/local/clipper.py`
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_local_clipper.py` does not import `numpy` yet — add `import numpy as np` to the import block at the top of the file (after `import pytest`). Add `_mouth_region_energy` to the existing `from shorts_generator.local.clipper import ...` import.

```python
def test_mouth_region_energy_zero_when_region_unchanged():
    gray = np.full((200, 200), 100, dtype=np.uint8)
    prev_gray = gray.copy()
    # face box centered at (100, 100), 80x80
    energy = local_clipper_module._mouth_region_energy(gray, prev_gray, (100.0, 100.0, 80.0, 80.0))
    assert energy == 0.0


def test_mouth_region_energy_positive_when_mouth_region_changed():
    gray = np.full((200, 200), 100, dtype=np.uint8)
    prev_gray = gray.copy()
    # mouth region is the lower half of the face box: y in [cy, cy+h/2] = [100, 140], x in [60, 140]
    gray[110:130, 80:120] = 200
    energy = local_clipper_module._mouth_region_energy(gray, prev_gray, (100.0, 100.0, 80.0, 80.0))
    assert energy > 0.0


def test_mouth_region_energy_ignores_change_outside_face_box():
    gray = np.full((200, 200), 100, dtype=np.uint8)
    prev_gray = gray.copy()
    gray[0:20, 0:20] = 200  # far corner, outside the face box entirely
    energy = local_clipper_module._mouth_region_energy(gray, prev_gray, (100.0, 100.0, 80.0, 80.0))
    assert energy == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k mouth_region_energy -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.local.clipper' has no attribute '_mouth_region_energy'`

- [ ] **Step 3: Implement `_mouth_region_energy`**

Add to `shorts_generator/local/clipper.py`, right after `_cluster_face_centers`:

```python
def _mouth_region_energy(
    gray: np.ndarray,
    prev_gray: np.ndarray,
    anchor: Tuple[float, float, float, float],
) -> float:
    """Sum of abs pixel difference from `prev_gray` in the lower half of the
    face box (cx, cy, w, h) — a cheap, model-free proxy for "is this
    person's mouth moving right now." Plain numpy (no cv2) so this is
    testable without a real video decode.
    """
    cx, cy, w, h = anchor
    x0 = int(max(0, cx - w / 2))
    x1 = int(min(gray.shape[1], cx + w / 2))
    y0 = int(max(0, cy))
    y1 = int(min(gray.shape[0], cy + h / 2))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    region = gray[y0:y1, x0:x1].astype(np.int16)
    prev_region = prev_gray[y0:y1, x0:x1].astype(np.int16)
    return float(np.abs(region - prev_region).sum())
```

`numpy` is already imported at the top of this file (`import numpy as np`, line 21) — no new import needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k mouth_region_energy -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "feat: add _mouth_region_energy motion proxy for active-speaker detection"
```

---

### Task 5: Implement `_two_speaker_positions`

Composes clustering output + raw per-frame speaker labels + hysteresis smoothing + crop-origin clamping into the final per-frame crop-position timeline for a two-person clip. This is the decision logic that Task 7's real video wiring will call — kept here as a function that takes already-computed raw labels, so it's fully testable with scripted input, no video/cascade mocking required.

**Files:**
- Modify: `shorts_generator/local/clipper.py`
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_local_clipper.py` (add `_two_speaker_positions` to the import):

```python
def test_two_speaker_positions_hard_cuts_at_speaker_change():
    anchor_a = (200.0, 500.0, 100.0, 120.0)
    anchor_b = (800.0, 500.0, 100.0, 120.0)
    # 10 frames of A, then 10 frames of B -- fps=10, dwell_seconds=0.3 -> dwell=3 frames,
    # well below the 10-frame runs, so hysteresis lets both flips through cleanly
    raw_labels = ["A"] * 10 + ["B"] * 10
    positions = local_clipper_module._two_speaker_positions(
        anchor_a, anchor_b, raw_labels, fps=10.0,
        crop_size=(200, 200), src_size=(1000, 1000),
    )
    assert len(positions) == 20
    expected_a = local_clipper_module._clamp_crop_origin((200.0, 500.0), (200, 200), (1000, 1000))
    expected_b = local_clipper_module._clamp_crop_origin((800.0, 500.0), (200, 200), (1000, 1000))
    assert positions[:10] == [expected_a] * 10
    assert positions[10:] == [expected_b] * 10
    # confirm it's a hard cut: the position at the switch boundary jumps directly,
    # no intermediate value between expected_a and expected_b
    assert positions[9] == expected_a
    assert positions[10] == expected_b


def test_two_speaker_positions_suppresses_brief_flicker():
    anchor_a = (200.0, 500.0, 100.0, 120.0)
    anchor_b = (800.0, 500.0, 100.0, 120.0)
    # a 2-frame "B" blip inside a long "A" run, with dwell_seconds=0.3 @ fps=10 -> dwell=3 frames,
    # so the 2-frame blip must NOT flip the result
    raw_labels = ["A"] * 10 + ["B"] * 2 + ["A"] * 10
    positions = local_clipper_module._two_speaker_positions(
        anchor_a, anchor_b, raw_labels, fps=10.0,
        crop_size=(200, 200), src_size=(1000, 1000),
    )
    expected_a = local_clipper_module._clamp_crop_origin((200.0, 500.0), (200, 200), (1000, 1000))
    assert positions == [expected_a] * 22
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k two_speaker_positions -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.local.clipper' has no attribute '_two_speaker_positions'`

- [ ] **Step 3: Implement `_two_speaker_positions`**

Add to `shorts_generator/local/clipper.py`, right after `_mouth_region_energy`:

```python
def _two_speaker_positions(
    anchor_a: Tuple[float, float, float, float],
    anchor_b: Tuple[float, float, float, float],
    raw_labels: List[str],
    fps: float,
    crop_size: Tuple[int, int],
    src_size: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Per-frame crop origin for a two-speaker clip: hysteresis-smooth the
    raw per-frame "A"/"B" active-speaker labels, then hard-cut between each
    speaker's own fixed, clamped crop origin — no interpolation between them.
    """
    smoothed = _apply_hysteresis(raw_labels, fps, dwell_seconds=SPEAKER_DWELL_SECONDS)
    pos_a = _clamp_crop_origin((anchor_a[0], anchor_a[1]), crop_size, src_size)
    pos_b = _clamp_crop_origin((anchor_b[0], anchor_b[1]), crop_size, src_size)
    return [pos_a if label == "A" else pos_b for label in smoothed]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k two_speaker_positions -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "feat: add _two_speaker_positions hard-cut crop timeline"
```

---

### Task 6: Wire clustering into `_reframe_vertical`'s single-anchor path (no behavior change for single-person clips)

This task replaces `_reframe_vertical`'s Pass 1 with a version that also collects clustering-eligible data, and adds the branch point — but for this task, only the single-cluster (fallback) branch is wired up, and it must produce **exactly** today's output. The two-speaker branch is added in Task 7. This separation lets us prove the no-regression guarantee before adding the new behavior.

**Files:**
- Modify: `shorts_generator/local/clipper.py`
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write the failing test**

This test locks in the no-regression guarantee: with the cascade mocked to return zero faces (matching what actually happens on the existing `synthetic_source` fixture, a faceless `testsrc` pattern), the crop must fall back to the exact frame-center position, exactly as today.

Add to `tests/test_local_clipper.py`:

```python
def test_reframe_vertical_falls_back_to_frame_center_when_no_faces(tmp_path, synthetic_source):
    out_path = str(tmp_path / "out.mp4")
    local_clipper_module._reframe_vertical(synthetic_source, out_path, "9:16")
    assert os.path.exists(out_path)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", out_path],
        capture_output=True, text=True, check=True,
    )
    width, height = (int(v) for v in probe.stdout.strip().split(","))
    assert (width, height) == local_clipper_module._output_size(9 / 16)
```

Note: this test doesn't require any new code to pass on its own (the existing fallback path already does this) — it's here to give Step 3's refactor a concrete pre-existing-behavior tripwire. Confirm it already passes before refactoring:

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k falls_back_to_frame_center -v`
Expected: PASS (this confirms today's baseline before the refactor)

- [ ] **Step 2: Refactor `_reframe_vertical`'s Pass 1 to collect both signals**

In `shorts_generator/local/clipper.py`, replace `_reframe_vertical`'s existing Pass 1 loop and single-anchor computation (currently lines ~112-139):

```python
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Pass 1 — sample the clip (~5 frames/sec is plenty) and take the median
    # face position as a single, stable anchor for the whole clip.
    sample_stride = max(1, int(fps // 5))
    sample_centers: List[Tuple[int, int]] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                sample_centers.append((x + w // 2, y + h // 2))
        frame_idx += 1

    if sample_centers:
        xs = sorted(c[0] for c in sample_centers)
        ys = sorted(c[1] for c in sample_centers)
        cx, cy = xs[len(xs) // 2], ys[len(ys) // 2]
    else:
        cx, cy = src_w // 2, src_h // 2

    x0, y0 = _clamp_crop_origin((cx, cy), (crop_w, crop_h), (src_w, src_h))
    out_w, out_h = _output_size(target_ratio)
```

with:

```python
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Pass 1 — sample the clip (~5 frames/sec is plenty). Collect both:
    #   - largest_per_frame: today's exact single-largest-face-per-frame series,
    #     used unchanged for the single-person fallback median.
    #   - all_detections: every detected face across every sampled frame, used
    #     only to decide whether this clip actually has two distinct people.
    sample_stride = max(1, int(fps // 5))
    largest_per_frame: List[Tuple[int, int]] = []
    all_detections: List[Tuple[float, float, float, float]] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                for (fx, fy, fw, fh) in faces:
                    all_detections.append((fx + fw / 2, fy + fh / 2, float(fw), float(fh)))
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                largest_per_frame.append((x + w // 2, y + h // 2))
        frame_idx += 1

    clusters = _cluster_face_centers(all_detections, src_w)
    out_w, out_h = _output_size(target_ratio)

    if len(clusters) < 2:
        # Single-person (or no-face) path -- byte-identical to before this change.
        if largest_per_frame:
            xs = sorted(c[0] for c in largest_per_frame)
            ys = sorted(c[1] for c in largest_per_frame)
            cx, cy = xs[len(xs) // 2], ys[len(ys) // 2]
        else:
            cx, cy = src_w // 2, src_h // 2
        x0, y0 = _clamp_crop_origin((cx, cy), (crop_w, crop_h), (src_w, src_h))
```

(Task 7 adds an `else:` branch immediately after this `if len(clusters) < 2:` block for the two-speaker case. The code above is standalone-valid Python as written for this task: `x0, y0` are defined whenever `len(clusters) < 2`, which covers every clip in the current test suite, so the render pass below — unchanged in this task — still has the values it needs.)

- [ ] **Step 3: Run the full suite to confirm no regression**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass, including `test_reframe_vertical_falls_back_to_frame_center_when_no_faces` and every pre-existing `test_local_clipper.py`/`test_captions.py`/etc. test that exercises `crop_highlights_local` on the faceless synthetic video.

- [ ] **Step 4: Add a single-person-with-a-real-detection regression test**

This confirms the fallback math is correct when there genuinely *is* one detected face (not just the no-face path), by mocking the cascade to return one consistent detection per sampled frame. Add `import cv2` to the top-level imports in `tests/test_local_clipper.py` (this is the first test in this file that needs to patch it directly; Task 7 reuses the same import):

```python
def test_reframe_vertical_single_person_locks_to_detected_face(tmp_path, synthetic_source, monkeypatch):
    def _fake_detect(self, gray, *args, **kwargs):
        # one consistent face box every call: src is 640x360 (see synthetic_source fixture)
        return [(220, 100, 100, 120)]

    monkeypatch.setattr(cv2.CascadeClassifier, "detectMultiScale", _fake_detect)

    out_path = str(tmp_path / "out_single.mp4")
    local_clipper_module._reframe_vertical(synthetic_source, out_path, "9:16")
    assert os.path.exists(out_path)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", out_path],
        capture_output=True, text=True, check=True,
    )
    width, height = (int(v) for v in probe.stdout.strip().split(","))
    assert (width, height) == local_clipper_module._output_size(9 / 16)
```

`shorts_generator.local.clipper` imports `cv2` lazily inside each function (there is no module-level `import cv2` in `clipper.py`, so `local_clipper_module.cv2` does not exist — do not reference it). Add a plain top-level `import cv2` to `tests/test_local_clipper.py` instead, and patch the class directly: `monkeypatch.setattr(cv2.CascadeClassifier, "detectMultiScale", _fake_detect)`. Python caches `cv2` in `sys.modules`, so this is the exact same class object `_reframe_vertical`'s own lazy `import cv2` binds to — patching it here affects it there too.

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k single_person_locks -v`
Expected: PASS

- [ ] **Step 5: Run the full suite again**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "refactor: route _reframe_vertical's single-anchor path through clustering, no behavior change"
```

---

### Task 7: Wire the two-speaker branch into `_reframe_vertical`

Adds the `else` branch for when `_cluster_face_centers` finds 2 people: a second full-frame decode pass computes per-frame mouth-motion-energy labels, `_two_speaker_positions` turns those into a hard-cut position timeline, and the final render pass uses per-frame positions instead of one fixed `x0, y0`.

**Files:**
- Modify: `shorts_generator/local/clipper.py`
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write the failing test**

This scripts the cascade to return two well-separated, well-supported face boxes across the sampled frames of the `synthetic_source` fixture (640x360, 6s @ 24fps), forcing the two-speaker branch, and confirms the render completes and produces a correctly-sized output. (Pixel-level verification of which half of the output is used is already covered by Task 5's `_two_speaker_positions` unit tests — this test's job is only to confirm the two-speaker branch is reachable and wired correctly end-to-end.)

Add to `tests/test_local_clipper.py`:

```python
def test_reframe_vertical_two_speakers_completes_and_outputs_correct_size(tmp_path, synthetic_source, monkeypatch):
    call_count = {"n": 0}

    def _fake_detect(self, gray, *args, **kwargs):
        call_count["n"] += 1
        # alternate which side is reported "biggest" across sampled calls so
        # all_detections ends up with two well-separated, well-supported
        # clusters (left ~x=100, right ~x=540 on a 640-wide frame)
        if call_count["n"] % 2 == 0:
            return [(60, 80, 90, 110), (500, 90, 90, 110)]
        return [(70, 85, 88, 108), (510, 95, 88, 108)]

    monkeypatch.setattr(cv2.CascadeClassifier, "detectMultiScale", _fake_detect)

    out_path = str(tmp_path / "out_two_speaker.mp4")
    local_clipper_module._reframe_vertical(synthetic_source, out_path, "9:16")
    assert os.path.exists(out_path)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", out_path],
        capture_output=True, text=True, check=True,
    )
    width, height = (int(v) for v in probe.stdout.strip().split(","))
    assert (width, height) == local_clipper_module._output_size(9 / 16)
```

- [ ] **Step 2: Run the test to verify it currently produces single-anchor output (not yet two-speaker), or errors**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k two_speakers_completes -v`
Expected: at this point (before Step 3's implementation), `_cluster_face_centers` will correctly find 2 clusters (this part already works from Task 3/6), but `_reframe_vertical` has no `else` branch yet to handle it — the function falls through without setting `x0, y0` for the two-speaker case, so this should FAIL with a `NameError` or `UnboundLocalError` on `x0`/`y0` in the render pass.

- [ ] **Step 3: Implement the two-speaker branch and per-frame render loop**

In `shorts_generator/local/clipper.py`, add the `else` branch immediately after the `if len(clusters) < 2:` block from Task 6:

```python
    else:
        anchor_a, anchor_b = clusters
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        raw_labels: List[str] = []
        prev_gray = None
        while True:
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
```

Now update Pass 2 (the final write pass, currently a fixed `x0, y0` for every frame — around what was originally lines 142-152) to branch on whether `positions` was computed:

Replace:

```python
    # Pass 2 — write the locked crop; x0/y0 never change within the clip.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (out_w, out_h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4))
```

with:

```python
    # Pass 2 (or 3, for two-speaker clips) — write the crop. Single-person
    # clips use one fixed x0/y0 for the whole clip; two-speaker clips hard-cut
    # per frame between each speaker's own fixed position (`positions`).
    two_speaker = len(clusters) >= 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (out_w, out_h))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if two_speaker:
            fx0, fy0 = positions[idx] if idx < len(positions) else positions[-1]
        else:
            fx0, fy0 = x0, y0
        cropped = frame[fy0:fy0 + crop_h, fx0:fx0 + crop_w]
        writer.write(cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4))
        idx += 1
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_local_clipper.py -k two_speakers_completes -v`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass, including every Task 1-6 test and every pre-existing test (single-person/no-face paths are untouched by this task's `else` branch).

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "feat: hard-cut crop between active speakers on two-person locked-mode clips"
```

---

### Task 8: Manual verification against the real confirmed-broken clip

Automated tests cover the decision logic and the branch wiring; this task confirms the fix actually resolves the originally-reported bug on real footage.

**Files:** none (verification only)

- [ ] **Step 1: Re-run the local pipeline on the source video that produced the confirmed-broken clip**

```bash
.venv/bin/python main.py "<the original YouTube URL for output/Does_The_Universe_Need_A_Creator>" --mode local --num-clips 10
```

(If the original URL isn't handy, the existing cached `output/Does_The_Universe_Need_A_Creator/full_source.mp4` can be re-cropped directly against the known bad clip's timestamps — `start_time=1484.5, end_time=1525.5` from that run's `result.json` — via `shorts_generator.local.clipper.crop_clip_local` in a throwaway script, to avoid re-downloading/re-transcribing.)

- [ ] **Step 2: Extract a frame at the timestamp that previously showed the bisected face and inspect it**

```bash
ffmpeg -y -loglevel error -ss 25 -i "<new output clip path>" -frames:v 1 /tmp/verify_frame.png
```

Open `/tmp/verify_frame.png` and confirm: no face is cut in half; whichever person is on screen at that moment is framed the same way single-person clips already look (full head visible, reasonably centered).

- [ ] **Step 3: Confirm the full test suite is green**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass

- [x] **Step 4: Report back**

**Result: cutoff NOT resolved for this clip.** Re-cropped
`I_Was_a_Pentecostal_Minister_Chuck_s_Shocking_Confession_to_Neil_deGrasse_Tyson`
(1484.5-1525.5) from the cached `full_source.mp4` via `crop_clip_local` on
this branch's HEAD (commit `72af99d`). The face is still bisected at the same
timestamps as the pre-fix output — pixel-identical crop framing to the old
code.

Root cause (diagnosed via a throwaway script dumping raw face detections):
this clip isn't a continuous two-shot — it cuts between a wide two-shot and
solo close-ups of each speaker. The two dominant face-position clusters
(both from close-up segments) are only ~13% of `src_w` apart, under the 25%
`TWO_PERSON_MIN_SEPARATION_FRAC` threshold, so `_cluster_face_centers` never
splits into 2 and the clip falls through to the unchanged single-anchor
path. Full writeup added to `mds/active-speaker-locked-framing.md` under
"Known limitation."

Full test suite: 174 passed. Spot-check (`Neil_deGrasse_Tyson_Defines_God_as_a_Pocket_of_Scientific_Ignorance`,
400.2-407.1, genuinely single-person): re-cropped frame is visually identical
to the original output (only diff is the caption overlay the original had
burned in, which `crop_clip_local` doesn't apply) — confirms the
no-regression guarantee holds for the common case.

Decision (user, 2026-07-24): merge as-is with the limitation documented,
rather than expanding scope to handle scene-cut/multi-framing sources now.
The branch still correctly fixes genuine continuous side-by-side clips and
is provably a no-op for single-person clips.

---

## Self-Review Notes

- **Spec coverage:** Phase 1 (clustering) → Task 3. Phase 2 (per-cluster anchor) → included inside Task 3's `_cluster_face_centers`. Phase 3 (active-speaker classification) → Task 4 (`_mouth_region_energy`) + Task 7 (raw label collection loop). Phase 4 (hard-cut render) → Task 5 (`_two_speaker_positions`) + Task 7 (render wiring). Phase 5 (tests) → a test is added in every task, plus Task 8's manual real-clip verification per the spec's Verification section.
- **No-regression guarantee:** explicitly proven in Task 6 by keeping the single-cluster branch's math textually identical to the pre-refactor code, and covered by both the pre-existing faceless-fixture tests and a new mocked-single-face test.
- **Type/signature consistency:** `_cluster_face_centers` returns `(cx, cy, w, h)` tuples throughout (Tasks 3, 5, 6, 7 all use this shape consistently). `_apply_hysteresis`'s new `dwell_seconds` parameter is used consistently in Task 5's `_two_speaker_positions` and nowhere else changes its call signature.
