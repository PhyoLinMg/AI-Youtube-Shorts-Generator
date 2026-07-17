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

from ..captions import CaptionError, burn_captions
from ..config import LOCAL_OUTPUT_DIR
from ..hook_card import HookCardError, extract_frame, pick_striking_frame, render_card_overlay

# --- adaptive framing tunables -------------------------------------------------
PERSON_FACE_MIN_W_FRAC = 0.12   # face width as a fraction of src width to count as "main person"
MODE_DWELL_SECONDS = 0.75       # raw class must persist this long before the mode flips
ZOOM_PERSON = 0.62              # crop_h as a fraction of src_h when person-centric (tight)
ZOOM_CURSOR = 1.0               # crop_h as a fraction of src_h when cursor-heavy (full height)
ZOOM_EMA_ALPHA = 0.08           # smoothing for the zoom scalar (slow ramp, no pumping)
CENTER_EMA_ALPHA = 0.12         # smoothing for the cursor-follow center
CENTER_MA_WINDOW = 7            # moving-average window (frames) for extra center stability
CENTER_MAX_STEP = 10.0          # px/frame velocity clamp for the center


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


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

    x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
    y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))

    # Pass 2 — write the locked crop; x0/y0 never change within the clip.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cropped)

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


def _apply_hysteresis(raw_classes: List[str], fps: float) -> List[str]:
    """Only flip mode once the opposite raw class persists >= dwell frames."""
    dwell = max(1, int(round(MODE_DWELL_SECONDS * fps)))
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
    out_h = 1920
    out_w = max(2, int(round(out_h * target_ratio)) - (int(round(out_h * target_ratio)) % 2))

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
        x0 = max(0, min(src_w - crop_w, int(cx - crop_w // 2)))
        y0 = max(0, min(src_h - crop_h, int(cy - crop_h // 2)))
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
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path.

    framing="locked" (default): static speaker-centered crop for the whole
    clip. framing="adaptive": cursor/person-aware crop for screen-recording
    content that alternates between facecam and screen activity.
    """
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        if framing == "adaptive":
            _reframe_vertical_adaptive(cut_path, out_path, aspect_ratio)
        else:
            _reframe_vertical(cut_path, out_path, aspect_ratio)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
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
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"Short-{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                framing=framing,
            )
            entry = {**h, "clip_url": out_path}

            hook_text = str(h.get("on_screen_hook") or "").strip()
            still_path = out_path + ".still.jpg"
            have_still = False
            if hook_card and hook_text:
                try:
                    ts = pick_striking_frame(out_path)
                    extract_frame(out_path, ts, still_path)
                    have_still = True
                except HookCardError as e:
                    print(f"[clip/local] {i} hook-card frame skipped: {e}", flush=True)
                    entry["hook_card_error"] = str(e)

            if captions and transcript_segments:
                captioned_path = out_path + ".captioned.mp4"
                try:
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

            if have_still:
                try:
                    card_path = out_path + ".card.mp4"
                    render_card_overlay(out_path, still_path, hook_text, card_path)
                    os.replace(card_path, out_path)
                except HookCardError as e:
                    print(f"[clip/local] {i} hook-card overlay skipped: {e}", flush=True)
                    entry["hook_card_error"] = str(e)
                finally:
                    if os.path.exists(still_path):
                        os.remove(still_path)

            results.append(entry)
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
