"""Composite a bold hook-text card over a clip's first ~1.5 seconds,
drawn directly on top of the live footage (the video keeps playing
underneath the text — it never freezes).
"""
import os
import subprocess
from typing import List

from .captions import CaptionError, _probe_resolution

HOOK_CARD_DURATION = 1.5        # seconds the hook text stays on screen

FONT_PATH = os.path.join(os.path.dirname(__file__), "assets", "fonts", "Anton-Regular.ttf")


class HookCardError(RuntimeError):
    """Raised when hook-card compositing fails; callers fall back to the pre-card clip."""


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
    hook_text: str,
    out_path: str,
    duration: float = HOOK_CARD_DURATION,
) -> str:
    """Draw bold `hook_text` over `video_path`'s first `duration` seconds,
    directly on top of the live, still-playing footage. Run this last,
    against the already-captioned clip. Audio passes through untouched;
    total duration and resolution are preserved — only the video is
    affected, and only inside the window.
    """
    try:
        _, height = _probe_resolution(video_path)
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
            f"x=(w-text_w)/2:y={y_expr}:expansion=none:"
            f"enable='between(t,0,{duration})'"
        )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-i", video_path,
        "-vf", ",".join(drawtext_filters),
        "-map", "0:v", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HookCardError(f"ffmpeg hook-card overlay failed: {e.stderr}") from e
    except OSError as e:
        raise HookCardError(f"ffmpeg hook-card overlay failed: {e}") from e
    return out_path
