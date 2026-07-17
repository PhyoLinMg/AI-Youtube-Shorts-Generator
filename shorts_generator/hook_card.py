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

from .captions import CaptionError, _probe_resolution

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


def _wrap_two_lines(text: str) -> List[str]:
    words = text.split()
    if len(words) <= 1:
        return [text]
    mid = (len(words) + 1) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def _escape_drawtext(text: str) -> str:
    """Escape for a single-quoted ffmpeg drawtext `text=` value. Colons
    still need escaping even inside single quotes (ffmpeg's filtergraph
    quoting doesn't shield them). Apostrophes are swapped for a
    typographic quote to dodge the filtergraph's fiddly nested-single-quote
    escaping entirely.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "’")
    return text


def render_card_overlay(
    video_path: str,
    still_path: str,
    hook_text: str,
    out_path: str,
    duration: float = HOOK_CARD_DURATION,
) -> str:
    """Composite `still_path` + bold `hook_text` over `video_path`'s first
    `duration` seconds. Run this last, against the already-captioned clip.
    Audio passes through untouched; total duration and resolution are
    preserved — only the video is affected, and only inside the window.
    """
    try:
        width, height = _probe_resolution(video_path)
    except CaptionError as e:
        raise HookCardError(f"failed to probe resolution of {video_path}: {e}") from e

    lines = _wrap_two_lines(hook_text)
    fontsize = max(24, round(height * 0.07))
    line_height = fontsize * 1.25
    block_height = line_height * len(lines)

    drawtext_filters = []
    for i, line in enumerate(lines):
        y_expr = f"(h-{block_height:.1f})/2+{i * line_height:.1f}"
        drawtext_filters.append(
            "drawtext="
            f"fontfile='{FONT_PATH}':text='{_escape_drawtext(line)}':"
            f"fontsize={fontsize}:fontcolor=white:"
            "box=1:boxcolor=black@0.55:boxborderw=16:"
            f"x=(w-text_w)/2:y={y_expr}:expansion=none"
        )

    filter_complex = (
        f"[1:v]scale={width}:{height}[stillv];"
        f"[stillv]{','.join(drawtext_filters)}[card];"
        f"[0:v][card]overlay=enable='between(t,0,{duration})'[v]"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-i", video_path,
        "-loop", "1", "-t", f"{duration + STILL_INPUT_BUFFER:.3f}", "-i", still_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        "-shortest",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HookCardError(f"ffmpeg hook-card overlay failed: {e.stderr}") from e
    except OSError as e:
        raise HookCardError(f"ffmpeg hook-card overlay failed: {e}") from e
    return out_path
