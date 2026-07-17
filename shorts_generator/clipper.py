"""Per-clip cropping via MuAPI /autocrop, with optional local caption
burn-in and hook-card overlay.

Given the source video URL plus a highlight's start/end and a target aspect
ratio, MuAPI returns a vertically-cropped short ready for posting. When
captions or the hook card are enabled (both on by default), that hosted
clip is downloaded locally and processed with ffmpeg (shorts_generator.
captions, shorts_generator.hook_card) — the one place API mode now needs a
local ffmpeg and OpenCV on PATH/installed.
"""
import os
from typing import Dict, List, Optional

import requests

from . import muapi
from .captions import CaptionError, burn_captions
from .hook_card import HookCardError, extract_frame, pick_striking_frame, render_card_overlay
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
    hook_card: bool = True,
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

            want_captions = captions and bool(transcript_segments)
            hook_text = str(h.get("on_screen_hook") or "").strip()
            want_hook_card = hook_card and bool(hook_text)

            if want_captions or want_hook_card:
                os.makedirs(out_dir, exist_ok=True)
                final_path = os.path.join(out_dir, f"Short-{i:02d}.mp4")
                downloaded_path = final_path + ".download.mp4"
                still_path = final_path + ".still.jpg"
                have_still = False
                try:
                    _download_to(url, downloaded_path)

                    if want_hook_card:
                        try:
                            ts = pick_striking_frame(downloaded_path)
                            extract_frame(downloaded_path, ts, still_path)
                            have_still = True
                        except HookCardError as e:
                            print(f"[clip] {i} hook-card frame skipped: {e}", flush=True)
                            entry["hook_card_error"] = str(e)

                    if want_captions:
                        burn_captions(
                            downloaded_path,
                            transcript_segments,
                            float(h["start_time"]),
                            float(h["end_time"]),
                            final_path,
                            fade_seconds=caption_fade_duration,
                            word_highlight=word_highlight,
                        )
                    else:
                        os.replace(downloaded_path, final_path)

                    if have_still:
                        try:
                            card_path = final_path + ".card.mp4"
                            render_card_overlay(final_path, still_path, hook_text, card_path)
                            os.replace(card_path, final_path)
                        except HookCardError as e:
                            print(f"[clip] {i} hook-card overlay skipped: {e}", flush=True)
                            entry["hook_card_error"] = str(e)

                    entry["clip_url"] = final_path
                    entry["hosted_clip_url"] = url
                except (CaptionError, requests.RequestException) as e:
                    print(f"[clip] {i} captions skipped: {e}", flush=True)
                    entry["captions_error"] = str(e)
                finally:
                    if os.path.exists(downloaded_path):
                        os.remove(downloaded_path)
                    if os.path.exists(still_path):
                        os.remove(still_path)

            out.append(entry)
        except Exception as e:
            print(f"[clip] {i} failed: {e}", flush=True)
            out.append({**h, "clip_url": None, "error": str(e)})
    return out
