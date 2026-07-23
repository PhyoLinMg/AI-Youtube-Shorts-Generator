"""End-to-end orchestrator.

Two modes:
  * mode="api"   (default) — MuAPI does download / transcribe / LLM / autocrop.
                              Fast, no local deps, pay-per-call.
  * mode="local"            — yt-dlp + faster-whisper + OpenAI or Gemini + ffmpeg/opencv.
                              Self-hosted, LLM_PROVIDER selects OpenAI or Gemini.

Both modes burn fade-in captions onto the final clips by default (see
shorts_generator.captions); pass captions=False to disable.

Every call writes into its own output/<Title>/ folder (see run_output.py):
Shorts/, full_source.mp4, full_source.json, result.json, progress.log.
"""
import json
import os
from typing import Dict, List, Optional

from .clipper import _download_to, crop_highlights
from .downloader import download_youtube
from .highlights import call_muapi_llm, get_highlights_cached
from .run_output import RunPaths, capture_progress_log, resolve_output_dir, write_descriptions
from .transcriber import transcribe


def _run_local(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    captions: bool,
    caption_fade_duration: float,
    paths: RunPaths,
    word_highlight: bool = True,
    framing: str = "locked",
    hook_card: bool = True,
) -> Dict:
    from .local.clipper import crop_highlights_local
    from .local.downloader import download_youtube_local
    from .local.llm import call_local_llm
    from .local.transcriber import transcribe_local

    if os.path.exists(paths.source_video):
        print(f"[pipeline/local] reusing cached source: {paths.source_video}", flush=True)
        source_path = paths.source_video
    else:
        source_path = download_youtube_local(youtube_url, target_path=paths.source_video, fmt=download_format)

    transcript = transcribe_local(source_path, language=language)
    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights_cached(
        transcript, num_clips=num_clips, cache_path=paths.highlights_json, llm_fn=call_local_llm,
    )
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:2 * num_clips]
    print(f"[pipeline/local] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights_local(
        source_path,
        top,
        aspect_ratio=aspect_ratio,
        out_dir=paths.shorts_dir,
        transcript_segments=transcript["segments"],
        captions=captions,
        caption_fade_duration=caption_fade_duration,
        word_highlight=word_highlight,
        framing=framing,
        hook_card=hook_card,
    )

    return {
        "mode": "local",
        "output_dir": paths.root,
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def _run_api(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    captions: bool,
    caption_fade_duration: float,
    paths: RunPaths,
    word_highlight: bool = True,
    hook_card: bool = True,
) -> Dict:
    # MuAPI /autocrop needs a fresh hosted URL for every crop, and that URL
    # only comes from /youtube-download — so this call can't be skipped on
    # rerun even if we already have a local copy of the video.
    source_url = download_youtube(youtube_url, fmt=download_format)

    if os.path.exists(paths.source_video):
        print(f"[pipeline] reusing cached local copy: {paths.source_video}", flush=True)
    else:
        # Download to a temp path and rename into place so a crash/interrupt
        # mid-download can never leave a truncated file at the cache path
        # (which a rerun would otherwise treat as a valid cached source).
        tmp_video_path = paths.source_video + ".part"
        _download_to(source_url, tmp_video_path)
        os.replace(tmp_video_path, paths.source_video)
        print(f"[pipeline] saved local copy: {paths.source_video}", flush=True)

    transcript = None
    if os.path.exists(paths.source_json):
        try:
            with open(paths.source_json, "r", encoding="utf-8") as f:
                transcript = json.load(f)
            print(f"[pipeline] reusing cached transcript: {paths.source_json}", flush=True)
        except json.JSONDecodeError:
            print(f"[pipeline] cached transcript is corrupted, re-transcribing: {paths.source_json}", flush=True)

    if transcript is None:
        transcript = transcribe(source_url, language=language)
        tmp_json_path = paths.source_json + ".part"
        with open(tmp_json_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False)
        os.replace(tmp_json_path, paths.source_json)

    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights_cached(
        transcript, num_clips=num_clips, cache_path=paths.highlights_json, llm_fn=call_muapi_llm,
    )
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:2 * num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights(
        source_url,
        top,
        aspect_ratio=aspect_ratio,
        transcript_segments=transcript["segments"],
        captions=captions,
        caption_fade_duration=caption_fade_duration,
        word_highlight=word_highlight,
        hook_card=hook_card,
        out_dir=paths.shorts_dir,
    )

    return {
        "mode": "api",
        "output_dir": paths.root,
        "source_video_url": source_url,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def generate_shorts(
    youtube_url: str,
    num_clips: int = 3,
    aspect_ratio: str = "9:16",
    download_format: str = "1080",
    language: Optional[str] = None,
    mode: str = "api",
    captions: bool = True,
    caption_fade_duration: float = 0.3,
    word_highlight: bool = True,
    framing: str = "locked",
    hook_card: bool = True,
    paths: Optional[RunPaths] = None,
) -> Dict:
    """Run the full pipeline and return a structured result.

    Args:
        youtube_url: source URL.
        num_clips: how many shorts to render.
        aspect_ratio: e.g. "9:16", "1:1".
        download_format: source resolution ("360" / "480" / "720" / "1080").
        language: ISO-639-1 to force Whisper language detection.
        mode: "api" (default, MuAPI) or "local" (yt-dlp + faster-whisper +
            OpenAI or Gemini + ffmpeg).
        captions: burn fade-in captions onto each clip (default True).
        caption_fade_duration: caption fade-in duration in seconds (default 0.3).
        word_highlight: highlight the currently-spoken word in each caption (default True).
        framing: "locked" (default, static speaker-centered crop) or
            "adaptive" (cursor/person-aware crop for screen-recording content
            that alternates between facecam and screen activity). Only
            applies to mode="local" — mode="api" always uses MuAPI's autocrop.
        hook_card: composite a bold on-screen hook (from each highlight's
            "on_screen_hook") over the clip's live footage for its first
            1.5 seconds (default True).
        paths: pre-resolved RunPaths to use instead of resolving them from
            youtube_url. Callers that need to know progress_log's path before
            the pipeline starts (e.g. a background job) should resolve it
            themselves and pass it here.

    Returns:
        {
          "mode": "api" | "local",
          "output_dir": str,         # output/<Title> for this run
          "source_video_url": str,   # hosted URL (api) or local path (local)
          "transcript": {...},
          "highlights": [...],       # all candidates ranked
          "shorts": [...],           # top `num_clips`, each with:
                                      #   clip_url: local path (Shorts/<title>.mp4)
                                      #   hosted_clip_url: original MuAPI URL (api mode,
                                      #     only present when captions or the hook card
                                      #     triggered a local download)
                                      #   hook_card_error: present if the hook-card overlay
                                      #     failed for that clip (falls back to the clip as
                                      #     it stood before the hook-card pass)
                                      #   captions_error: present if caption burn-in failed
                                      #     for that clip (falls back to the uncaptioned clip)
        }
    """
    mode = (mode or "api").lower()
    if mode not in ("api", "local"):
        raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")

    paths = paths or resolve_output_dir(youtube_url)
    with capture_progress_log(paths.progress_log):
        if mode == "local":
            result = _run_local(
                youtube_url, num_clips, aspect_ratio, download_format, language, captions, caption_fade_duration,
                paths, word_highlight=word_highlight, framing=framing, hook_card=hook_card,
            )
        else:
            result = _run_api(
                youtube_url, num_clips, aspect_ratio, download_format, language, captions, caption_fade_duration,
                paths, word_highlight=word_highlight, hook_card=hook_card,
            )

        write_descriptions(paths.shorts_dir, result["shorts"])

        with open(paths.result_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result
