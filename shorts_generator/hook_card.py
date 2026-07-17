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
