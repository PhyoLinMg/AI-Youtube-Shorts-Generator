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
from typing import Callable, Dict, List, Optional

from .clipper import _download_to, crop_highlights
from .corpus import get_abstract_cached
from .downloader import download_youtube
from .highlights import call_muapi_llm, get_chapters_cached, get_highlights_cached, select_final_highlights
from .local.caption_ingest import ingest_captions
from .local.llm import call_openai_vision_llm
from .local.narration import render_narration_card, synthesize_narration
from .local.thread_assembler import assemble_thread
from .local.thread_source import _probe_local_duration, acquire_clip
from .run_output import RunPaths, archive_stale_thread_run, capture_progress_log, resolve_output_dir, resolve_thread_run_dir, sanitize_title, write_chapter_descriptions, write_descriptions, write_source_url, write_thread_descriptions
from .thread_builder import find_same_topic_pairs, ground_thread_clips
from .transcriber import transcribe
from .visual_hook import call_muapi_vision_llm, score_visual_hooks

CROP_FAILURE_BUFFER = 1  # extra candidates cropped beyond num_clips so a
                         # single crop failure (MuAPI autocrop erroring, or
                         # the local ffmpeg crop raising) doesn't silently
                         # under-deliver fewer than num_clips shorts.


def _trim_to_num_clips(shorts: List[Dict], num_clips: int) -> List[Dict]:
    """If enough crops succeeded, drop the extra buffer successes -- deleting
    their rendered local files so a trimmed-away buffer clip doesn't linger
    on disk indefinitely -- so output matches num_clips exactly. If not
    enough succeeded even with the buffer, return every entry as-is
    (including failures) so the shortfall stays visible as "Failed" cards
    downstream, instead of being hidden."""
    successes = [s for s in shorts if s.get("clip_url")]
    if len(successes) < num_clips:
        return shorts

    kept = successes[:num_clips]
    dropped = successes[num_clips:]
    for s in dropped:
        clip_path = s.get("clip_url")
        if clip_path and os.path.isfile(clip_path):
            try:
                os.remove(clip_path)
            except OSError as e:
                print(f"[pipeline] could not remove trimmed buffer clip {clip_path}: {e}", flush=True)
    return kept


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
    end_card: bool = False,
    filename_style: Optional[str] = None,
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

    top = select_final_highlights(all_highlights, num_clips + CROP_FAILURE_BUFFER)
    print(f"[pipeline/local] cropping {len(top)} of {len(all_highlights)} candidates ({num_clips} requested + {CROP_FAILURE_BUFFER} failure buffer)", flush=True)

    try:
        top = score_visual_hooks(source_path, top, llm_fn=call_openai_vision_llm)
    except Exception as e:
        print(f"[pipeline/local] visual-hook scoring skipped: {e}", flush=True)

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
        end_card=end_card,
        filename_style=filename_style,
    )
    shorts = _trim_to_num_clips(shorts, num_clips)

    return {
        "mode": "local",
        "output_dir": paths.root,
        "source_video_url": source_path,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def _run_local_chapters(
    youtube_url: str,
    num_chapters: int,
    download_format: str,
    language: Optional[str],
    captions: bool,
    caption_fade_duration: float,
    paths: RunPaths,
    word_highlight: bool = True,
) -> Dict:
    from .local.clipper import crop_chapters_local
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

    chapters_result = get_chapters_cached(
        transcript, num_chapters=num_chapters, cache_path=paths.chapters_json, llm_fn=call_local_llm,
    )
    all_chapters: List[Dict] = chapters_result.get("chapters", [])
    if not all_chapters:
        raise RuntimeError("Chapter generator returned zero chapters.")
    print(f"[pipeline/local] cropping {len(all_chapters)} chapters", flush=True)

    chapters = crop_chapters_local(
        source_path,
        all_chapters,
        out_dir=paths.chapters_dir,
        transcript_segments=transcript["segments"],
        captions=captions,
        caption_fade_duration=caption_fade_duration,
        word_highlight=word_highlight,
    )
    # No _trim_to_num_clips here (unlike the Shorts path): num_chapters is a
    # target/floor hint fed into the LLM prompt, not a post-render slice --
    # chapters have no score to rank by, so a hard cutoff here would always
    # discard whichever chapters happen to sort last chronologically, after
    # already paying the full crop/caption-burn cost for them. The actual
    # count ceiling (MAX_CHAPTERS_PER_EPISODE) is enforced in
    # highlights.get_chapters, before any chapter reaches this function.

    return {
        "output_dir": paths.root,
        "source_video_url": source_path,
        "transcript": transcript,
        "all_chapters": all_chapters,
        "chapters": chapters,
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
    end_card: bool = False,
    filename_style: Optional[str] = None,
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

    top = select_final_highlights(all_highlights, num_clips + CROP_FAILURE_BUFFER)
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates ({num_clips} requested + {CROP_FAILURE_BUFFER} failure buffer)", flush=True)

    try:
        top = score_visual_hooks(paths.source_video, top, llm_fn=call_muapi_vision_llm)
    except Exception as e:
        print(f"[pipeline] visual-hook scoring skipped: {e}", flush=True)

    shorts = crop_highlights(
        source_url,
        top,
        aspect_ratio=aspect_ratio,
        transcript_segments=transcript["segments"],
        captions=captions,
        caption_fade_duration=caption_fade_duration,
        word_highlight=word_highlight,
        hook_card=hook_card,
        end_card=end_card,
        out_dir=paths.shorts_dir,
        filename_style=filename_style,
    )
    shorts = _trim_to_num_clips(shorts, num_clips)

    return {
        "mode": "api",
        "output_dir": paths.root,
        "source_video_url": source_url,
        "transcript": transcript,
        "highlights": all_highlights,
        "shorts": shorts,
    }


def _cache_topic_abstract(run_dir: str, transcript: Dict, llm_fn) -> None:
    """Best-effort: pre-compute this episode's corpus-search abstract right
    after its transcript is ready (see corpus.get_abstract_cached), so a
    later cross-episode topic search never pays a fresh LLM call for an
    episode that's already been processed. Never aborts the run -- same
    log-and-continue pattern as score_visual_hooks in _run_api. llm_fn only
    matters on a cache miss (get_abstract_cached checks the cache first), so
    api-mode and local-mode runs can end up with abstracts from different
    providers across the corpus -- expected, not a bug."""
    try:
        get_abstract_cached(run_dir, transcript, llm_fn=llm_fn)
    except Exception as e:
        print(f"[pipeline] topic-abstract caching skipped: {e}", flush=True)


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
    end_card: bool = False,
    filename_style: Optional[str] = None,
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
        end_card: composite a bold on-screen closing line (from each
            highlight's "end_card_text") over the clip's last ~2 seconds
            (default False).
        filename_style: "specific" (default, slugified highlight title,
            e.g. My_Big_Moment.mp4) or "generic" (positional, video1.mp4,
            video2.mp4, ...). Falls back to the SHORT_FILENAME_STYLE env
            var (config.py) when None.
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
                                      #   excision_error: present if jump-cut excision of a
                                      #     highlight's cut_segments failed (falls back to the
                                      #     un-excised envelope clip)
        }
    """
    mode = (mode or "api").lower()
    if mode not in ("api", "local"):
        raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")

    paths = paths or resolve_output_dir(youtube_url)
    write_source_url(paths, youtube_url)
    with capture_progress_log(paths.progress_log):
        if mode == "local":
            result = _run_local(
                youtube_url, num_clips, aspect_ratio, download_format, language, captions, caption_fade_duration,
                paths, word_highlight=word_highlight, framing=framing, hook_card=hook_card, end_card=end_card,
                filename_style=filename_style,
            )
        else:
            result = _run_api(
                youtube_url, num_clips, aspect_ratio, download_format, language, captions, caption_fade_duration,
                paths, word_highlight=word_highlight, hook_card=hook_card, end_card=end_card,
                filename_style=filename_style,
            )

        if mode == "local":
            from .local.llm import call_local_llm
            _cache_topic_abstract(paths.root, result["transcript"], call_local_llm)
        else:
            _cache_topic_abstract(paths.root, result["transcript"], call_muapi_llm)

        write_descriptions(paths.shorts_dir, result["shorts"])

        with open(paths.result_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def generate_chapters(
    youtube_url: str,
    num_chapters: int = 5,
    download_format: str = "1080",
    language: Optional[str] = None,
    captions: bool = True,
    caption_fade_duration: float = 0.3,
    word_highlight: bool = True,
    paths: Optional[RunPaths] = None,
) -> Dict:
    """Run the chapter-cuts pipeline (local mode only) and return a
    structured result. See generate_shorts for the parallel Shorts entry
    point; this one has no `mode` param since chapters is local-only.

    Args:
        youtube_url: source URL.
        num_chapters: target chapter count (the model may return 3-8 based
            on natural topic boundaries; this is a target, not a hard slice).
        download_format: source resolution ("360" / "480" / "720" / "1080").
        language: ISO-639-1 to force Whisper language detection.
        captions: burn near-bottom-edge captions onto each chapter (default True).
        caption_fade_duration: caption fade-in duration in seconds (default 0.3).
        word_highlight: highlight the currently-spoken word in each caption (default True).
        paths: pre-resolved RunPaths to use instead of resolving them from youtube_url.

    Returns:
        {
          "output_dir": str,         # output/<Title> for this run
          "source_video_url": str,   # local path to the downloaded source
          "transcript": {...},
          "all_chapters": [...],     # every chapter get_chapters returned (already
                                      #   capped at MAX_CHAPTERS_PER_EPISODE, pre-render)
          "chapters": [...],         # same chapters as all_chapters, post-crop, each with
                                      #   clip_url / *_error fields -- num_chapters is a
                                      #   target hint fed into the LLM prompt, not a slice
        }
    """
    paths = paths or resolve_output_dir(youtube_url)
    write_source_url(paths, youtube_url)
    with capture_progress_log(paths.progress_log):
        result = _run_local_chapters(
            youtube_url, num_chapters, download_format, language, captions, caption_fade_duration,
            paths, word_highlight=word_highlight,
        )

        from .local.llm import call_local_llm
        _cache_topic_abstract(paths.root, result["transcript"], call_local_llm)

        write_chapter_descriptions(paths.chapters_dir, result["chapters"])

        with open(paths.chapters_result_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def _ingest_and_abstract(url: str, base_dir: Optional[str], llm_fn) -> Dict:
    """Caption-only ingest (no video download) + a cached topical abstract
    for one thread episode -- see local/caption_ingest.py and corpus.py."""
    ingested = ingest_captions(url, base_dir=base_dir)
    run_dir = ingested["run_dir"]
    with open(os.path.join(run_dir, "full_source.json"), "r", encoding="utf-8") as f:
        transcript = json.load(f)
    abstract = get_abstract_cached(run_dir, transcript, llm_fn=llm_fn)
    return {"run_dir": run_dir, "title": ingested["title"], "source_url": url, "abstract": abstract}


def _warn_if_under_tiktok_minimum(clip_path: str) -> None:
    """TikTok's Creator Rewards Program only pays out on videos 60s or
    longer -- thread_builder.PLATFORM_BOUNDS["tiktok"] steers the LLM's
    clip picks to make a sub-60s assembly unlikely, but narration audio
    length isn't independently bounded, so this is a defense-in-depth
    check on the actual assembled file, not a substitute for the picker's
    own bounds. Non-fatal: logs only, never deletes the file or aborts the
    run."""
    try:
        duration = _probe_local_duration(clip_path)
    except Exception as e:
        print(
            f"[pipeline/local] could not probe duration of {os.path.basename(clip_path)} "
            f"for TikTok-minimum check: {e}",
            flush=True,
        )
        return
    if duration < 60.0:
        print(
            f"[pipeline/local] WARNING: TikTok cut {os.path.basename(clip_path)} is "
            f"{duration:.1f}s, under the 60s Creator Rewards minimum",
            flush=True,
        )


def generate_threads(
    url_a: str,
    url_b: str,
    num_clips: int = 1,
    platform: str = "youtube",
    base_dir: Optional[str] = None,
    on_output_dir: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    """Build up to num_clips distinct-topic threads from exactly the two
    given episodes (see thread_builder.ground_thread_clips). Local-mode
    only, like generate_chapters -- there is no MuAPI equivalent of this
    feature. Both URLs are ingested caption-only (no video download; see
    local/caption_ingest.py) and idempotently reused if already in the
    corpus. Returns [] if no shared question is groundable between the two
    episodes, or if every requested platform grounds zero clip pairs for
    every shared question -- this is the expected, correct result when
    they don't genuinely cover the same topic (or, for tiktok, when no
    span in the transcript is long enough), not a failure to work around.

    platform selects the output cut(s): "youtube" (default, 45-60s
    target), "tiktok" (65-90s target, code-enforced by
    thread_builder.PLATFORM_BOUNDS to clear TikTok's 60s Creator Rewards
    Program minimum), or "both" (produce one file of each). The
    same-topic-question scan (thread_builder.find_same_topic_pairs) runs
    exactly once regardless of platform -- it has no dependency on clip
    length, so its result is reused across every requested platform's own
    grounding pass (thread_builder.ground_thread_clips). Output files are
    named thesis_{i}_{platform}_{title}.mp4, with raw intermediates under
    raw/thesis_{i}_{platform}/, where i is 1-indexed within that
    platform's own results (not the flattened combined list).

    To get youtube and tiktok cuts that coexist on disk, pass platform=
    "both" in a single call -- resolve_thread_run_dir/archive_stale_thread_run
    key off the two episode titles only, not platform, so a second same-day
    call for the same pair (even with a different platform) archives the
    first call's entire output dir into raw/stale/ rather than merging with
    it.

    Unlike generate_shorts/generate_chapters, the output dir is knowable up
    front from the two episode titles (see resolve_thread_run_dir) -- but
    on_output_dir, if given, still fires before any per-clip render work
    starts, matching the old single-clip contract, so a caller like the
    dashboard can start tailing progress.log immediately.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .local.downloader import download_youtube_local
    from .local.llm import call_local_llm

    platforms = ["youtube", "tiktok"] if platform == "both" else [platform]

    # url_a and url_b are two unrelated episodes -- each ingest is its own
    # yt-dlp caption fetch plus, on a cache miss, its own LLM abstract call,
    # writing into that episode's own run_dir (see corpus.get_abstract_cached)
    # -- so there's no shared state between them and no reason to pay both
    # round trips back-to-back.
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(_ingest_and_abstract, url_a, base_dir, call_local_llm)
        future_b = pool.submit(_ingest_and_abstract, url_b, base_dir, call_local_llm)
        entry_a = future_a.result()
        entry_b = future_b.result()

    out_dir = resolve_thread_run_dir(entry_a["title"], entry_b["title"], base_dir=base_dir)
    archive_stale_thread_run(out_dir)
    if on_output_dir:
        on_output_dir(out_dir)

    with capture_progress_log(os.path.join(out_dir, "progress.log")):
        print(f"[pipeline/local] ingested episode A: {entry_a['title']!r}", flush=True)
        print(f"[pipeline/local] ingested episode B: {entry_b['title']!r}", flush=True)

        with open(os.path.join(entry_a["run_dir"], "full_source.json"), "r", encoding="utf-8") as f:
            transcript_a = json.load(f)
        with open(os.path.join(entry_b["run_dir"], "full_source.json"), "r", encoding="utf-8") as f:
            transcript_b = json.load(f)

        print(f"[pipeline/local] scanning for up to {num_clips} shared-question thread(s)...", flush=True)
        shared_questions = find_same_topic_pairs(entry_a, entry_b, num_clips, call_local_llm)
        if not shared_questions:
            return []

        pairs_by_platform = {}
        for p in platforms:
            pairs_by_platform[p] = ground_thread_clips(
                entry_a, entry_b, transcript_a, transcript_b, shared_questions, num_clips, call_local_llm, platform=p,
            )
        if not any(pairs_by_platform.values()):
            return []

        # Download each episode's full video exactly once up front instead of
        # letting every clip in the loop below re-download its own padded
        # span (acquire_clip's fast path already prefers a full_source.mp4 on
        # disk over a fresh yt-dlp call) -- one full download per episode is
        # both fewer network round trips than num_clips separate section
        # downloads and less exposed to the section-download-specific CDN
        # flakiness seen with --download-sections. Only delete the videos we
        # downloaded here, never a full_source.mp4 that predates this run
        # (e.g. left over from a prior Shorts/chapters run on the same URL).
        full_source_a = os.path.join(entry_a["run_dir"], "full_source.mp4")
        full_source_b = os.path.join(entry_b["run_dir"], "full_source.mp4")
        downloaded_full_sources = []
        try:
            # Inside the try/finally from the start: if A's download
            # succeeds but B's raises, the finally below must still remove
            # A's now-orphaned multi-GB file rather than leak it.
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = []
                if not os.path.exists(full_source_a):
                    print(f"[pipeline/local] downloading full source video A: {entry_a['title']!r}...", flush=True)
                    futures.append(pool.submit(download_youtube_local, entry_a["source_url"], full_source_a))
                    downloaded_full_sources.append(full_source_a)
                if not os.path.exists(full_source_b):
                    print(f"[pipeline/local] downloading full source video B: {entry_b['title']!r}...", flush=True)
                    futures.append(pool.submit(download_youtube_local, entry_b["source_url"], full_source_b))
                    downloaded_full_sources.append(full_source_b)
                for future in futures:
                    future.result()

            results = []
            for p in platforms:
                pairs = pairs_by_platform[p]
                for i, thread in enumerate(pairs, 1):
                    episode_a, episode_b = thread["episode_a"], thread["episode_b"]
                    print(f"[pipeline/local] [{p}] clip {i}/{len(pairs)}: {thread['shared_question']!r}", flush=True)

                    thesis_dir = os.path.join(out_dir, "raw", f"thesis_{i}_{p}")
                    os.makedirs(thesis_dir, exist_ok=True)

                    clip_a_path = os.path.join(thesis_dir, f"clip_{i}_a.mp4")
                    clip_b_path = os.path.join(thesis_dir, f"clip_{i}_b.mp4")
                    print(f"[pipeline/local] acquiring clip A from {episode_a['title']!r}...", flush=True)
                    acquire_clip(
                        episode_a["run_dir"], episode_a["source_url"], cached_duration=transcript_a.get("duration") or 0.0,
                        start_time=episode_a["start_time"], end_time=episode_a["end_time"], out_path=clip_a_path,
                    )
                    print(f"[pipeline/local] acquiring clip B from {episode_b['title']!r}...", flush=True)
                    acquire_clip(
                        episode_b["run_dir"], episode_b["source_url"], cached_duration=transcript_b.get("duration") or 0.0,
                        start_time=episode_b["start_time"], end_time=episode_b["end_time"], out_path=clip_b_path,
                    )

                    intro_audio = os.path.join(thesis_dir, f"thesis_{i}.mp3")
                    bridge_audio = os.path.join(thesis_dir, f"bridge_{i}.mp3")
                    print("[pipeline/local] synthesizing narration (thesis + bridge)...", flush=True)
                    synthesize_narration(thread["thesis"], intro_audio)
                    synthesize_narration(thread["bridge"], bridge_audio)

                    intro_card = os.path.join(thesis_dir, f"intro_card_{i}.mp4")
                    bridge_card = os.path.join(thesis_dir, f"bridge_card_{i}.mp4")
                    print("[pipeline/local] rendering narration cards...", flush=True)
                    render_narration_card(intro_audio, thread["thesis"], intro_card)
                    render_narration_card(bridge_audio, thread["bridge"], bridge_card)

                    final_title = thread.get("title") or thread["shared_question"]
                    final_path = os.path.join(out_dir, f"thesis_{i}_{p}_{sanitize_title(final_title)}.mp4")
                    print("[pipeline/local] assembling final thread (intro -> clip A -> bridge -> clip B)...", flush=True)
                    assemble_thread([intro_card, clip_a_path, bridge_card, clip_b_path], final_path)
                    if p == "tiktok":
                        _warn_if_under_tiktok_minimum(final_path)

                    results.append({
                        **thread,
                        "platform": p,
                        "platform_index": i,
                        "output_dir": out_dir,
                        "clip_url": final_path,
                        "episode_a": {**episode_a, "clip_url": clip_a_path},
                        "episode_b": {**episode_b, "clip_url": clip_b_path},
                    })
                    print(f"[pipeline/local] done: {final_path}", flush=True)
        finally:
            # Clean up regardless of success/failure mid-loop -- an
            # abandoned multi-GB full_source.mp4 from a crashed run is worse
            # than losing the (re-downloadable) cache on a genuine error.
            for path in downloaded_full_sources:
                if os.path.exists(path):
                    print(f"[pipeline/local] removing downloaded full source video: {path}", flush=True)
                    os.remove(path)

        with open(os.path.join(out_dir, "thread_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        write_thread_descriptions(out_dir, results)
        return results
