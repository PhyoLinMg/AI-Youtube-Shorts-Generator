"""Per-clip cropping via MuAPI /autocrop, with optional local caption burn-in.

Given the source video URL plus a highlight's start/end and a target aspect
ratio, MuAPI returns a vertically-cropped short ready for posting. When
captions are enabled (the default), that hosted clip is downloaded locally
and burned with fade-in captions via ffmpeg (shorts_generator.captions) —
the one place API mode now needs a local ffmpeg on PATH.
"""
import os
from typing import Dict, List, Optional

import requests

from . import muapi
from .captions import CaptionError, burn_captions
from .config import LOCAL_OUTPUT_DIR
from .downloader import _extract_video_url


def crop_clip(source_video_url: str, start_time: float, end_time: float, aspect_ratio: str = "9:16") -> str:
    """Submit one autocrop job and return the URL of the rendered short."""
    payload = {
        "video_url": source_video_url,
        "start_time": float(start_time),
        "end_time": float(end_time),
        "aspect_ratio": aspect_ratio,
    }
    print(f"[clip] {start_time:.1f}s → {end_time:.1f}s @ {aspect_ratio}", flush=True)
    result = muapi.run("autocrop", payload, label=f"autocrop({start_time:.0f}-{end_time:.0f})")
    return _extract_video_url(result)


def _download_to(url: str, dest_path: str) -> str:
    """Stream a hosted clip to a local file."""
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 16):
            if chunk:
                f.write(chunk)
    return dest_path


def crop_highlights(
    source_video_url: str,
    highlights: list,
    aspect_ratio: str = "9:16",
    transcript_segments: Optional[List[Dict]] = None,
    captions: bool = True,
    caption_fade_duration: float = 0.3,
    word_highlight: bool = True,
    out_dir: Optional[str] = None,
) -> list:
    """Crop every highlight, attaching the resulting URL back onto the dict."""
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    out = []
    for i, h in enumerate(highlights, 1):
        print(f"[clip] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            url = crop_clip(
                source_video_url,
                h["start_time"],
                h["end_time"],
                aspect_ratio=aspect_ratio,
            )
            entry = {**h, "clip_url": url}

            if captions and transcript_segments:
                os.makedirs(out_dir, exist_ok=True)
                final_path = os.path.join(out_dir, f"Short-{i:02d}.mp4")
                downloaded_path = final_path + ".download.mp4"
                try:
                    _download_to(url, downloaded_path)
                    burn_captions(
                        downloaded_path,
                        transcript_segments,
                        float(h["start_time"]),
                        float(h["end_time"]),
                        final_path,
                        fade_seconds=caption_fade_duration,
                        word_highlight=word_highlight,
                    )
                    entry["clip_url"] = final_path
                    entry["hosted_clip_url"] = url
                except (CaptionError, requests.RequestException) as e:
                    print(f"[clip] {i} captions skipped: {e}", flush=True)
                    entry["captions_error"] = str(e)
                finally:
                    if os.path.exists(downloaded_path):
                        os.remove(downloaded_path)

            out.append(entry)
        except Exception as e:
            print(f"[clip] {i} failed: {e}", flush=True)
            out.append({**h, "clip_url": None, "error": str(e)})
    return out
