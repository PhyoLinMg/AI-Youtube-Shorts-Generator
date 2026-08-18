"""Acquire the exact audio/video span thread_builder picked for one episode
of a thread, whether or not that episode's full_source.mp4 is still on disk.

The common case: full_source.mp4 is still cached -- cut directly from it via
the same crop_clip_local/burn_captions path Shorts already use, using the
cached transcript, no network call needed.

The fallback case: full_source.mp4 was deleted (typical once a channel has
100+ episodes and disk space matters) -- re-download just the needed span
via yt-dlp, but ONLY after verifying the live video's duration matches the
cached transcript's duration. This check exists because of a real incident
(see docs/superpowers/specs/2026-08-09-thread-compilation-design.md): a
mismatched source URL silently produced a downloaded span whose audio had
nothing to do with the cached transcript's timestamps, and captions burned
from that stale transcript looked plausible but described different audio
entirely. A duration mismatch is fatal, never a warning to route around.
"""
import os
import subprocess
import tempfile
import time
from typing import Dict, List

from .. import config
from ..captions import CaptionError, burn_captions
from .clipper import crop_clip_local
from .transcriber import transcribe_local

# Signed googlevideo CDN URLs intermittently 403 (short-lived/IP-bound
# tokens, edge throttling) independent of yt-dlp's own auth -- a fixed
# retry count matches the pattern already used for muapi.py's transient
# HTTP errors rather than surfacing a one-off network flake as a hard
# pipeline failure.
DOWNLOAD_SECTION_RETRIES = 3
DOWNLOAD_SECTION_RETRY_DELAY_SECONDS = 3.0

PAD_SECONDS = 3.0
# Heuristic, not proof of content identity: two different uploads of the
# same length would slip past this check. It exists to catch the specific,
# common failure mode from the design-spec incident (a wrong/stale URL
# pointing at a differently-lengthed video), not to guarantee the bytes are
# the exact same upload.
DURATION_MISMATCH_TOLERANCE_SECONDS = 2.0


class SourceMismatchError(RuntimeError):
    """Raised when a re-acquired source's live duration doesn't match the
    cached transcript's duration -- see module docstring."""


def _cookie_args() -> List[str]:
    if not config.YT_DLP_COOKIES_FROM_BROWSER:
        return []
    return ["--cookies-from-browser", config.YT_DLP_COOKIES_FROM_BROWSER]


def _probe_local_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _probe_source_duration(source_url: str) -> float:
    result = subprocess.run(
        ["yt-dlp", *_cookie_args(), "--skip-download", "--print", "duration", source_url],
        capture_output=True, text=True, check=True,
    )
    stdout = result.stdout.strip()
    try:
        return float(stdout.splitlines()[-1])
    except (IndexError, ValueError) as e:
        raise RuntimeError(
            f"could not parse a duration from yt-dlp output for {source_url!r}: "
            f"stdout={stdout!r}"
        ) from e


def _download_padded_section(source_url: str, start_time: float, end_time: float, out_path: str) -> None:
    padded_start = max(0.0, start_time - PAD_SECONDS)
    padded_end = end_time + PAD_SECONDS
    download_stem = out_path + ".download"
    downloaded_path = None
    try:
        last_error = None
        for attempt in range(1, DOWNLOAD_SECTION_RETRIES + 1):
            try:
                subprocess.run(
                    [
                        "yt-dlp",
                        *_cookie_args(),
                        "--download-sections", f"*{padded_start}-{padded_end}",
                        "-f", "bv*[height<=720]+ba/b[height<=720]",
                        "--force-keyframes-at-cuts",
                        "-o", download_stem + ".%(ext)s",
                        source_url,
                    ],
                    check=True, capture_output=True, text=True,
                )
                last_error = None
                break
            except subprocess.CalledProcessError as e:
                last_error = e
                if attempt < DOWNLOAD_SECTION_RETRIES:
                    print(
                        f"[thread_source] yt-dlp --download-sections attempt {attempt}/"
                        f"{DOWNLOAD_SECTION_RETRIES} failed for {source_url!r}, retrying: "
                        f"{e.stderr.strip().splitlines()[-1] if e.stderr else e}",
                        flush=True,
                    )
                    time.sleep(DOWNLOAD_SECTION_RETRY_DELAY_SECONDS)
        if last_error is not None:
            # Bare CalledProcessError.__str__ omits stderr, so a failure here
            # otherwise reaches progress.log as just "exit status 1" with no
            # way to tell a YouTube bot-check/429 apart from a signed CDN URL
            # 403 or any other cause -- surface the tail of yt-dlp's own stderr.
            stderr_tail = (
                "\n".join(last_error.stderr.strip().splitlines()[-15:])
                if last_error.stderr else "(no stderr captured)"
            )
            raise RuntimeError(
                f"yt-dlp --download-sections failed for {source_url!r} after "
                f"{DOWNLOAD_SECTION_RETRIES} attempts: {stderr_tail}"
            ) from last_error
        # --force-keyframes-at-cuts re-encodes the cut via yt-dlp's own
        # ffmpeg merge step, which can change the container (observed:
        # source webm -> merged mkv) regardless of the outtmpl extension
        # requested above -- so the actual output extension isn't known
        # ahead of time and has to be discovered after the fact.
        for ext in (".mkv", ".webm", ".mp4"):
            candidate = download_stem + ext
            if os.path.exists(candidate):
                downloaded_path = candidate
                break
        if downloaded_path is None:
            raise RuntimeError(f"yt-dlp did not produce an output file at {download_stem}.*")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", downloaded_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                out_path,
            ],
            check=True,
        )
    finally:
        # Self-sufficient cleanup: don't rely on the caller wrapping this in
        # a TemporaryDirectory. Runs whether the yt-dlp step, the ffmpeg
        # transcode step, or neither failed.
        if downloaded_path and os.path.exists(downloaded_path):
            os.remove(downloaded_path)


def _find_word_start(segments: List[Dict], min_time: float) -> float:
    """First word timestamp at or after min_time -- lands a clip's start
    exactly on a spoken word instead of mid-word or on dead air. Falls back
    to min_time itself if no word starts at or after it in these segments."""
    for seg in segments:
        for w in seg.get("words", []):
            if float(w["start"]) >= min_time:
                return float(w["start"])
    return min_time


def _caption_in_place(
    out_path: str,
    start_time: float,
    end_time: float,
    segments: List[Dict],
    log_label: str,
) -> Dict:
    """Burn captions onto an already-cropped clip at out_path, in place --
    a caption failure is logged and leaves the uncaptioned-but-valid clip in
    place rather than raising, and any partial .captioned.mp4 is cleaned up.
    On success the caption result is atomically swapped into out_path."""
    result = {"clip_path": out_path}
    captioned_path = out_path + ".captioned.mp4"
    try:
        burn_captions(
            out_path, segments, start_time, end_time, captioned_path,
            fade_seconds=0.3, word_highlight=True,
        )
        os.replace(captioned_path, out_path)
    except CaptionError as e:
        print(f"[thread_source] {log_label} captions skipped: {e}", flush=True)
        result["captions_error"] = str(e)
        if os.path.exists(captioned_path):
            os.remove(captioned_path)
    return result


def _crop_and_caption_fresh(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    log_label: str,
) -> Dict:
    """Crop [start_time, end_time] out of source_path to out_path, then
    caption it using a FRESH Whisper transcription of the cropped clip
    itself -- real per-word timestamps, measured from the actual clip audio,
    rather than whatever transcript the caller has cached.

    This matters specifically for the full_source.mp4 fast path: thread-mode
    episodes are ingested caption-only (YouTube auto-captions via
    yt-dlp --write-auto-sub, see caption_ingest.py) to avoid downloading
    full videos just to check topic overlap. YouTube's auto-captions only
    give segment-level timing -- captions.py estimates each word's position
    within a segment proportionally by character count, which can visibly
    drift from where the word is actually spoken. Re-transcribing the short
    (already-cut) clip fresh costs one small-model Whisper call, not a
    second download, and gets the same real-word-timestamp accuracy the
    re-download fallback path below has always had.
    """
    crop_clip_local(
        source_path, start_time, end_time, aspect_ratio, out_path,
        framing="locked", cut_segments=[{"start_time": start_time, "end_time": end_time}],
    )
    fresh_transcript = transcribe_local(out_path, model_size="small")
    return _caption_in_place(
        out_path, 0.0, fresh_transcript["duration"], fresh_transcript["segments"], log_label,
    )


def acquire_clip(
    run_dir: str,
    source_url: str,
    cached_duration: float,
    start_time: float,
    end_time: float,
    out_path: str,
    aspect_ratio: str = "9:16",
) -> Dict:
    """Cut, reframe, and caption one episode's clip for a thread.

    Returns {"clip_path": out_path}. Raises SourceMismatchError if the
    source at full_source.mp4 (or a re-download) doesn't match
    cached_duration -- see module docstring for why this check is fatal,
    never a warning to route around, regardless of which path produced
    the local file (pre-existing from a prior run, downloaded ahead of
    the whole thread by the caller, or downloaded here as a fallback).
    """
    full_source = os.path.join(run_dir, "full_source.mp4")

    if os.path.exists(full_source):
        on_disk_duration = _probe_local_duration(full_source)
        if abs(on_disk_duration - cached_duration) > DURATION_MISMATCH_TOLERANCE_SECONDS:
            raise SourceMismatchError(
                f"full_source.mp4 duration ({on_disk_duration:.1f}s) does not match cached "
                f"transcript duration ({cached_duration:.1f}s) for {run_dir} -- "
                "refusing to caption a possibly-wrong source. Confirm source_url.txt "
                "points at the same upload that was originally transcribed."
            )
        return _crop_and_caption_fresh(
            full_source, start_time, end_time, aspect_ratio, out_path, "full_source",
        )

    full_duration = _probe_source_duration(source_url)
    if abs(full_duration - cached_duration) > DURATION_MISMATCH_TOLERANCE_SECONDS:
        raise SourceMismatchError(
            f"live video duration ({full_duration:.1f}s) does not match cached "
            f"transcript duration ({cached_duration:.1f}s) for {run_dir} -- "
            "refusing to caption a possibly-wrong source. Confirm source_url.txt "
            "points at the same upload that was originally transcribed."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        padded_path = os.path.join(tmp_dir, "padded.mp4")
        _download_padded_section(source_url, start_time, end_time, padded_path)

        padded_start = max(0.0, start_time - PAD_SECONDS)
        fresh_transcript = transcribe_local(padded_path, model_size="small")
        relative_start = _find_word_start(fresh_transcript["segments"], start_time - padded_start)
        relative_end = min(end_time - padded_start, fresh_transcript["duration"])

        crop_clip_local(
            padded_path, relative_start, relative_end, aspect_ratio, out_path,
            framing="locked", cut_segments=[{"start_time": relative_start, "end_time": relative_end}],
        )
        return _caption_in_place(
            out_path, relative_start, relative_end, fresh_transcript["segments"], "re-acquired",
        )
