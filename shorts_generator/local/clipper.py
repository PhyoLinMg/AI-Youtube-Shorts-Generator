"""Local clipping: ffmpeg subclip + OpenCV face-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio. Two framing modes:
       - "locked" (default): find the speaker's median position (Haar
         cascade, no external models) and crop a locked vertical window
         there — static for the whole clip, since any per-frame tracking
         reads as camera shake on a talking head.
       - "adaptive": for screen-recording content that alternates between
         facecam and screen/cursor activity. Classifies a rolling window as
         person-centric (big face -> stable locked center, tight zoom) or
         cursor-heavy (follow the cursor per-frame, zoom out to keep the
         full source height visible), with hysteresis and smoothed zoom/pan
         so mode switches ease instead of snap.
"""
import os
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..captions import CaptionError, burn_captions, burn_captions_segments
from ..config import LOCAL_OUTPUT_DIR
from ..hook_card import HookCardError, render_card_overlay
from ..jump_cuts import excise_cut_segments, JumpCutError
from ..run_output import unique_short_filename

# --- adaptive framing tunables -------------------------------------------------
PERSON_FACE_MIN_W_FRAC = 0.12   # face width as a fraction of src width to count as "main person"
MODE_DWELL_SECONDS = 0.75       # raw class must persist this long before the mode flips
TWO_PERSON_MIN_SEPARATION_FRAC = 0.25   # min x-gap (as a fraction of src_w) to call it 2 distinct people
MIN_CLUSTER_SAMPLE_FRAC = 0.15          # each cluster must own at least this share of all detections
SPEAKER_DWELL_SECONDS = 0.35            # active-speaker switch dwell — faster than MODE_DWELL_SECONDS,
                                         # since conversational turn-taking is quicker than screen/person mode switches
ZOOM_PERSON = 0.62              # crop_h as a fraction of src_h when person-centric (tight)
ZOOM_CURSOR = 1.0               # crop_h as a fraction of src_h when cursor-heavy (full height)
ZOOM_EMA_ALPHA = 0.08           # smoothing for the zoom scalar (slow ramp, no pumping)
CENTER_EMA_ALPHA = 0.12         # smoothing for the cursor-follow center
CENTER_MA_WINDOW = 7            # moving-average window (frames) for extra center stability
CENTER_MAX_STEP = 10.0          # px/frame velocity clamp for the center

# --- scene-cut segmentation tunables (locked framing) --------------------------
SCENE_CUT_DIFF_THRESHOLD = 40.0  # mean abs pixel diff (0-255) between consecutive
                                  # sampled frames to call it a hard camera cut
MIN_SEGMENT_SECONDS = 2.0        # a segment shorter than this gets folded into
                                  # its neighbor rather than getting its own
                                  # (statistically unreliable) anchor
MAX_SEGMENTS_PER_CLIP = 6        # degrade-gracefully cap; beyond this the cut
                                  # detector is probably a poor fit for this
                                  # footage (e.g. fast handheld motion) and we
                                  # fall back to single-global-anchor behavior

OUTPUT_CANVAS_H = 1920          # final render height regardless of source resolution


def _output_size(target_ratio: float) -> Tuple[int, int]:
    """Fixed vertical-HD output canvas (e.g. 1080x1920 for 9:16) — every clip
    gets upscaled/downscaled to this regardless of source resolution, since a
    plain native-pixel crop off a sub-1080p source (or a narrow 9:16 slice of
    even a 1080p landscape frame) reads as "not HD" once played back full-screen.
    """
    out_h = OUTPUT_CANVAS_H
    out_w = max(2, int(round(out_h * target_ratio)))
    out_w -= out_w % 2
    return out_w, out_h


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


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


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Crop the cut clip to the target aspect ratio, centered on the speaker.

    Uses a single locked crop position for the whole clip instead of
    per-frame tracking — any tracker (even heavily smoothed) still reads as
    camera shake on a talking head, since the subject is always drifting a
    little (nodding, gesturing). A static, well-centered shot doesn't.
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

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

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


def _detect_cursor(gray, prev_gray) -> Optional[Tuple[int, int]]:
    """Largest small moving blob between frames ~ the mouse cursor."""
    import cv2  # type: ignore

    if prev_gray is None:
        return None
    diff = cv2.absdiff(gray, prev_gray)
    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, None
    for c in contours:
        a = cv2.contourArea(c)
        if 20 <= a <= 4000:
            if best_area is None or a > best_area:
                m = cv2.moments(c)
                if m["m00"] > 0:
                    best = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
                    best_area = a
    return best


def _classify_frames(cap, src_w: int) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
    """Per frame: raw class ("person" | "cursor") + its raw anchor point."""
    import cv2  # type: ignore

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    raw: List[Tuple[str, Optional[Tuple[int, int]]]] = []
    prev_gray, last_cursor = None, None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        anchor = None
        cls = "cursor"
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            if fw > PERSON_FACE_MIN_W_FRAC * src_w:
                cls, anchor = "person", (x + fw // 2, y + fh // 2)
        if cls == "cursor":
            cur = _detect_cursor(gray, prev_gray)
            if cur:
                last_cursor = cur
            anchor = last_cursor
        raw.append((cls, anchor))
        prev_gray = gray
    return raw


def _apply_hysteresis(
    raw_classes: List[str], fps: float, dwell_seconds: float = MODE_DWELL_SECONDS,
) -> List[str]:
    """Only flip mode once the opposite raw class persists >= dwell frames."""
    dwell = max(1, int(round(dwell_seconds * fps)))
    if not raw_classes:
        return []
    modes = [raw_classes[0]]
    current = raw_classes[0]
    run_len = 0
    for cls in raw_classes[1:]:
        if cls == current:
            run_len = 0
        else:
            run_len += 1
            if run_len >= dwell:
                current = cls
                run_len = 0
        modes.append(current)
    return modes


def _smooth_scalar(values: List[float], alpha: float, max_step: Optional[float] = None) -> List[float]:
    """EMA smoothing, optionally followed by a velocity clamp."""
    if not values:
        return []
    out = [values[0]]
    v = values[0]
    for x in values[1:]:
        v = v * (1 - alpha) + x * alpha
        out.append(v)
    if max_step is not None:
        clamped = [out[0]]
        c = out[0]
        for x in out[1:]:
            d = x - c
            if abs(d) > max_step:
                c += max_step if d > 0 else -max_step
            else:
                c = x
            clamped.append(c)
        return clamped
    return out


def _smooth_center(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """EMA -> moving-average -> velocity clamp, matching the crop_tips_v2 prototype."""
    if not points:
        return []
    px, py = points[0]
    ema = []
    for x, y in points:
        px, py = px * (1 - CENTER_EMA_ALPHA) + x * CENTER_EMA_ALPHA, py * (1 - CENTER_EMA_ALPHA) + y * CENTER_EMA_ALPHA
        ema.append((px, py))

    win = CENTER_MA_WINDOW
    sm = []
    for i in range(len(ema)):
        lo, hi = max(0, i - win // 2), min(len(ema), i + win // 2 + 1)
        chunk = ema[lo:hi]
        sm.append((sum(c[0] for c in chunk) / len(chunk), sum(c[1] for c in chunk) / len(chunk)))

    out = []
    cx, cy = sm[0]
    for x, y in sm:
        dx, dy = x - cx, y - cy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > CENTER_MAX_STEP:
            cx += dx * CENTER_MAX_STEP / dist
            cy += dy * CENTER_MAX_STEP / dist
        else:
            cx, cy = x, y
        out.append((cx, cy))
    return out


def _reframe_vertical_adaptive(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Cursor/person-adaptive crop: follows the cursor on screen content
    (zoomed out, full source height) and holds a stable center on the
    speaker when a big face is present (zoomed in tight). Mode switches use
    hysteresis and both zoom + center are smoothed so transitions ease
    instead of snapping — a clip that never switches degrades to a single
    stable mode with no per-frame motion.
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

    # Fixed output size (VideoWriter needs a constant frame size even though
    # the crop box itself changes per frame).
    out_w, out_h = _output_size(target_ratio)

    raw = _classify_frames(cap, src_w)
    if not raw:
        cap.release()
        raise RuntimeError(f"no frames read from {in_path}")
    raw_classes = [c for c, _ in raw]
    modes = _apply_hysteresis(raw_classes, fps)

    # Person windows: stable locked center = median face position over that
    # contiguous person run (no per-frame follow -> no shake).
    person_centers: List[Optional[Tuple[float, float]]] = [None] * len(raw)
    i = 0
    while i < len(modes):
        if modes[i] != "person":
            i += 1
            continue
        j = i
        while j < len(modes) and modes[j] == "person":
            j += 1
        run_pts = [a for c, a in raw[i:j] if c == "person" and a is not None]
        if run_pts:
            xs = sorted(p[0] for p in run_pts)
            ys = sorted(p[1] for p in run_pts)
            median = (xs[len(xs) // 2], ys[len(ys) // 2])
        else:
            median = (src_w / 2, src_h / 2)
        for k in range(i, j):
            person_centers[k] = median
        i = j

    # Cursor windows: per-frame follow through the full smoothing chain.
    fallback = (src_w / 2, src_h / 2)
    last_known = fallback
    cursor_raw_points: List[Tuple[float, float]] = []
    for anchor in (a for _, a in raw):
        if anchor is not None:
            last_known = anchor
        cursor_raw_points.append(last_known)
    cursor_centers = _smooth_center(cursor_raw_points)

    centers: List[Tuple[float, float]] = [
        person_centers[k] if modes[k] == "person" else cursor_centers[k]
        for k in range(len(modes))
    ]
    # Ease across mode transitions too (locked-center jump -> cursor follow).
    centers = _smooth_center(centers)

    zoom_raw = [ZOOM_PERSON if m == "person" else ZOOM_CURSOR for m in modes]
    zooms = _smooth_scalar(zoom_raw, ZOOM_EMA_ALPHA)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (out_w, out_h))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cx, cy = centers[idx] if idx < len(centers) else fallback
        z = zooms[idx] if idx < len(zooms) else ZOOM_CURSOR
        crop_h = max(2, min(src_h, int(round(src_h * z))))
        crop_w = max(2, min(src_w, int(round(crop_h * target_ratio))))
        crop_w -= crop_w % 2
        crop_h -= crop_h % 2
        x0, y0 = _clamp_crop_origin((cx, cy), (crop_w, crop_h), (src_w, src_h))
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4))
        idx += 1

    cap.release()
    writer.release()

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


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    framing: str = "locked",
    cut_segments: Optional[List[Dict]] = None,
    errors: Optional[Dict] = None,
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path.

    framing="locked" (default): static speaker-centered crop for the whole
    clip. framing="adaptive": cursor/person-aware crop for screen-recording
    content that alternates between facecam and screen activity.

    cut_segments (optional): when it has more than one entry, the gaps
    between kept spans are excised (jump_cuts.excise_cut_segments) before
    reframing, so a reaction-jail dead-air trim survives the vertical crop.
    If excision fails, this falls back to the un-excised envelope cut
    (mirroring api mode's clipper.crop_highlights) rather than failing the
    whole highlight -- a bug in dead-air excision shouldn't kill an
    otherwise-fine clip. When `errors` is passed, a failure is recorded at
    errors["excision_error"] so the caller can both surface it and know to
    route captions through the un-excised path.
    """
    cut_path = out_path + ".cut.mp4"
    excised_path = out_path + ".excised.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        working_path = cut_path
        if cut_segments and len(cut_segments) > 1:
            try:
                excise_cut_segments(cut_path, cut_segments, start_time, excised_path)
                working_path = excised_path
            except JumpCutError as e:
                print(f"[clip/local] jump-cut excision skipped: {e}", flush=True)
                if errors is not None:
                    errors["excision_error"] = str(e)
        if framing == "adaptive":
            _reframe_vertical_adaptive(working_path, out_path, aspect_ratio)
        else:
            _reframe_vertical(working_path, out_path, aspect_ratio)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
        if os.path.exists(excised_path):
            os.remove(excised_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    transcript_segments: Optional[List[Dict]] = None,
    captions: bool = True,
    caption_fade_duration: float = 0.3,
    word_highlight: bool = True,
    framing: str = "locked",
    hook_card: bool = True,
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    used_names: set = set()
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, unique_short_filename(h.get("title"), used_names))
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            cut_segments = h.get("cut_segments") or [
                {"start_time": float(h["start_time"]), "end_time": float(h["end_time"])}
            ]
            crop_errors: Dict = {}
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                framing=framing,
                cut_segments=cut_segments,
                errors=crop_errors,
            )
            entry = {**h, "clip_url": out_path, **crop_errors}
            want_excision = len(cut_segments) > 1 and "excision_error" not in crop_errors

            hook_text = str(h.get("on_screen_hook") or "").strip()
            want_hook_card = hook_card and bool(hook_text)

            if captions and transcript_segments:
                captioned_path = out_path + ".captioned.mp4"
                try:
                    if want_excision:
                        burn_captions_segments(
                            out_path,
                            transcript_segments,
                            cut_segments,
                            captioned_path,
                            fade_seconds=caption_fade_duration,
                            word_highlight=word_highlight,
                        )
                    else:
                        burn_captions(
                            out_path,
                            transcript_segments,
                            float(h["start_time"]),
                            float(h["end_time"]),
                            captioned_path,
                            fade_seconds=caption_fade_duration,
                            word_highlight=word_highlight,
                        )
                    os.replace(captioned_path, out_path)
                except CaptionError as e:
                    print(f"[clip/local] {i} captions skipped: {e}", flush=True)
                    entry["captions_error"] = str(e)
                    if os.path.exists(captioned_path):
                        os.remove(captioned_path)

            if want_hook_card:
                try:
                    card_path = out_path + ".card.mp4"
                    render_card_overlay(out_path, hook_text, card_path)
                    os.replace(card_path, out_path)
                except HookCardError as e:
                    print(f"[clip/local] {i} hook-card overlay skipped: {e}", flush=True)
                    entry["hook_card_error"] = str(e)

            results.append(entry)
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
