"""Pick a striking frame from a clip and composite a bold hook-text card
over its first ~1.5 seconds.

Two-phase API:
  - pick_striking_frame + extract_frame run against the CLEAN, pre-caption
    crop, so the chosen still never has caption text baked into it.
  - render_card_overlay runs last, against the captioned clip, and is the
    only step whose output actually lands in the final mp4.
"""
import os
import subprocess
from typing import List, Tuple

import cv2  # type: ignore

from .captions import _probe_resolution

HOOK_CARD_DURATION = 1.5        # seconds the card stays on screen
SKIP_SECONDS = 0.5              # never pick a frame from the resting opening
SAMPLE_FPS = 5                  # motion/sharpness sampling rate
MOTION_TOP_QUANTILE = 0.75      # only the top 25% of frames by motion score are sharpness candidates
STILL_INPUT_BUFFER = 0.5        # extra seconds on the looped still input beyond `duration`

FONT_PATH = os.path.join(os.path.dirname(__file__), "assets", "fonts", "Anton-Regular.ttf")


class HookCardError(RuntimeError):
    """Raised when hook-card frame selection or compositing fails; callers fall back to the pre-card clip."""


def pick_striking_frame(
    video_path: str,
    skip_seconds: float = SKIP_SECONDS,
    sample_fps: float = SAMPLE_FPS,
) -> float:
    """Return the timestamp of the most "striking" frame in `video_path`:
    among the top-quartile motion-scoring sampled frames (frame-to-frame
    pixel diff), the sharpest one (Laplacian variance) — this avoids
    landing on a motion-blurred frame just because it had the single
    highest motion score. Falls back to `skip_seconds` if the clip is too
    short to yield 2+ samples after skipping the opening.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HookCardError(f"could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / sample_fps)))
    skip_frames = int(round(skip_seconds * fps))

    samples: List[Tuple[float, object]] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= skip_frames and frame_idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            samples.append((frame_idx / fps, gray))
        frame_idx += 1
    cap.release()

    if len(samples) < 2:
        return skip_seconds

    motion_scores = [0.0]
    for i in range(1, len(samples)):
        diff = cv2.absdiff(samples[i][1], samples[i - 1][1])
        motion_scores.append(float(diff.sum()))

    ranked = sorted(range(len(samples)), key=lambda i: motion_scores[i], reverse=True)
    top_n = max(1, int(round(len(ranked) * (1 - MOTION_TOP_QUANTILE))))
    candidates = ranked[:top_n]

    best_idx = max(candidates, key=lambda i: cv2.Laplacian(samples[i][1], cv2.CV_64F).var())
    return samples[best_idx][0]


def extract_frame(video_path: str, timestamp: float, out_path: str) -> str:
    """Grab a single frame at `timestamp` as a still image."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-ss", f"{timestamp:.3f}",
        "-i", video_path,
        "-vframes", "1",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HookCardError(f"ffmpeg frame extraction failed: {e.stderr}") from e
    except OSError as e:
        raise HookCardError(f"ffmpeg frame extraction failed: {e}") from e
    return out_path
