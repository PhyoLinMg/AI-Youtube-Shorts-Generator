"""Per-clip cropping via MuAPI /autocrop, with optional local caption
burn-in and hook-card overlay.

Given the source video URL plus a highlight's start/end and a target aspect
ratio, MuAPI returns a vertically-cropped short ready for posting. When
captions or the hook card are enabled (both on by default), that hosted
clip is downloaded locally and processed with ffmpeg (shorts_generator.
captions, shorts_generator.hook_card) — the one place API mode now needs a
local ffmpeg on PATH/installed.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import requests

from . import muapi
from .jump_cuts import excise_cut_segments, JumpCutError
from .captions import CaptionError, burn_captions, burn_captions_segments
from .hook_card import HookCardError, render_card_overlay, render_end_card_overlay
from .config import CROP_PARALLELISM, LOCAL_OUTPUT_DIR
from .downloader import _extract_video_url
from .run_output import unique_short_filename


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


def _plan_highlight_output(
    i: int,
    h: Dict,
    used_names: set,
    out_dir: str,
    captions: bool,
    transcript_segments: Optional[List[Dict]],
    hook_card: bool,
    end_card: bool,
    filename_style: Optional[str],
) -> Dict:
    """Decide (serially, before any worker thread starts) whether this
    highlight needs local post-processing and, if so, resolve its output
    filename. Must run serially: unique_short_filename mutates the shared
    used_names set, and check-then-add across threads could hand two
    highlights the same path."""
    want_captions = captions and bool(transcript_segments)
    hook_text = str(h.get("on_screen_hook") or "").strip()
    want_hook_card = hook_card and bool(hook_text)
    end_card_text = str(h.get("end_card_text") or "").strip()
    want_end_card = end_card and bool(end_card_text)
    cut_segments = h.get("cut_segments") or [
        {"start_time": h.get("start_time"), "end_time": h.get("end_time")}
    ]
    want_excision = len(cut_segments) > 1

    final_path = None
    if want_captions or want_hook_card or want_end_card or want_excision:
        filename = unique_short_filename(h.get("title"), used_names, index=i, style=filename_style)
        final_path = os.path.join(out_dir, filename)

    return {
        "final_path": final_path,
        "want_captions": want_captions,
        "hook_text": hook_text,
        "want_hook_card": want_hook_card,
        "end_card_text": end_card_text,
        "want_end_card": want_end_card,
        "cut_segments": cut_segments,
        "want_excision": want_excision,
    }


def _crop_one_highlight(
    i: int,
    n: int,
    h: Dict,
    plan: Dict,
    source_video_url: str,
    aspect_ratio: str,
    transcript_segments: Optional[List[Dict]],
    caption_fade_duration: float,
    word_highlight: bool,
) -> Dict:
    """Render one highlight end-to-end. Never raises -- any failure is
    caught and folded into the returned entry, so one bad highlight can't
    abort the others when this runs inside a thread-pool worker."""
    print(f"[clip] {i}/{n}: {h.get('title', '(untitled)')}", flush=True)
    try:
        url = crop_clip(
            source_video_url,
            h["start_time"],
            h["end_time"],
            aspect_ratio=aspect_ratio,
        )
        entry = {**h, "clip_url": url}

        final_path = plan["final_path"]
        if final_path is not None:
            want_excision = plan["want_excision"]
            cut_segments = plan["cut_segments"]
            downloaded_path = final_path + ".download.mp4"
            try:
                _download_to(url, downloaded_path)

                if want_excision:
                    try:
                        excised_path = final_path + ".excised.mp4"
                        excise_cut_segments(
                            downloaded_path, cut_segments, float(h["start_time"]), excised_path,
                        )
                        os.replace(excised_path, downloaded_path)
                    except JumpCutError as e:
                        print(f"[clip] {i} jump-cut excision skipped: {e}", flush=True)
                        entry["excision_error"] = str(e)
                        want_excision = False

                if plan["want_captions"]:
                    try:
                        if want_excision:
                            burn_captions_segments(
                                downloaded_path,
                                transcript_segments,
                                cut_segments,
                                final_path,
                                fade_seconds=caption_fade_duration,
                                word_highlight=word_highlight,
                            )
                        else:
                            burn_captions(
                                downloaded_path,
                                transcript_segments,
                                float(h["start_time"]),
                                float(h["end_time"]),
                                final_path,
                                fade_seconds=caption_fade_duration,
                                word_highlight=word_highlight,
                            )
                    except CaptionError as e:
                        # Caption burn-in failed, but the download itself
                        # succeeded (and the hook card may already have
                        # too) -- fall back to the plain download rather
                        # than discarding everything back to the hosted
                        # URL, matching local mode's behavior.
                        print(f"[clip] {i} captions skipped: {e}", flush=True)
                        entry["captions_error"] = str(e)
                        os.replace(downloaded_path, final_path)
                else:
                    os.replace(downloaded_path, final_path)

                if plan["want_hook_card"]:
                    try:
                        card_path = final_path + ".card.mp4"
                        render_card_overlay(final_path, plan["hook_text"], card_path)
                        os.replace(card_path, final_path)
                    except HookCardError as e:
                        print(f"[clip] {i} hook-card overlay skipped: {e}", flush=True)
                        entry["hook_card_error"] = str(e)

                if plan["want_end_card"]:
                    try:
                        end_card_path = final_path + ".endcard.mp4"
                        render_end_card_overlay(final_path, plan["end_card_text"], end_card_path)
                        os.replace(end_card_path, final_path)
                    except HookCardError as e:
                        print(f"[clip] {i} end-card overlay skipped: {e}", flush=True)
                        entry["end_card_error"] = str(e)

                entry["clip_url"] = final_path
                entry["hosted_clip_url"] = url
            except requests.RequestException as e:
                # The download itself failed -- no local file exists at
                # all, so there's nothing to fall back to except the
                # hosted URL.
                print(f"[clip] {i} download failed, falling back to hosted url: {e}", flush=True)
                entry["captions_error"] = str(e)
            finally:
                if os.path.exists(downloaded_path):
                    os.remove(downloaded_path)

        return entry
    except Exception as e:
        print(f"[clip] {i} failed: {e}", flush=True)
        return {**h, "clip_url": None, "error": str(e)}


def crop_highlights(
    source_video_url: str,
    highlights: list,
    aspect_ratio: str = "9:16",
    transcript_segments: Optional[List[Dict]] = None,
    captions: bool = True,
    caption_fade_duration: float = 0.3,
    word_highlight: bool = True,
    hook_card: bool = True,
    end_card: bool = False,
    out_dir: Optional[str] = None,
    filename_style: Optional[str] = None,
) -> list:
    """Crop every highlight (in parallel, up to CROP_PARALLELISM at a time --
    each highlight is an independent MuAPI job + download + local burn-in,
    dominated by network/poll wait), attaching the resulting URL back onto
    the dict. Results preserve input order regardless of completion order,
    since callers (see pipeline._trim_to_num_clips) rely on that order
    matching highlight rank."""
    if not highlights:
        return []
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    used_names: set = set()
    plans = [
        _plan_highlight_output(
            i, h, used_names, out_dir, captions, transcript_segments, hook_card, end_card, filename_style,
        )
        for i, h in enumerate(highlights, 1)
    ]

    n = len(highlights)
    with ThreadPoolExecutor(max_workers=min(CROP_PARALLELISM, n)) as pool:
        return list(pool.map(
            lambda args: _crop_one_highlight(
                args[0], n, args[1], args[2], source_video_url, aspect_ratio,
                transcript_segments, caption_fade_duration, word_highlight,
            ),
            zip(range(1, n + 1), highlights, plans),
        ))
