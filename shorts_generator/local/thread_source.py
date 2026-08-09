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
import json
import os
import subprocess
import tempfile
from typing import Dict, List

from ..captions import CaptionError, burn_captions
from .clipper import crop_clip_local
from .transcriber import transcribe_local

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


def _probe_source_duration(source_url: str) -> float:
    result = subprocess.run(
        ["yt-dlp", "--skip-download", "--print", "duration", source_url],
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
    webm_path = out_path + ".webm"
    try:
        subprocess.run(
            [
                "yt-dlp",
                "--download-sections", f"*{padded_start}-{padded_end}",
                "-f", "bv*[height<=720]+ba/b[height<=720]",
                "--force-keyframes-at-cuts",
                "-o", webm_path,
                source_url,
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", webm_path,
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
        if os.path.exists(webm_path):
            os.remove(webm_path)


def _find_word_start(segments: List[Dict], min_time: float) -> float:
    """First word timestamp at or after min_time -- lands a clip's start
    exactly on a spoken word instead of mid-word or on dead air. Falls back
    to min_time itself if no word starts at or after it in these segments."""
    for seg in segments:
        for w in seg.get("words", []):
            if float(w["start"]) >= min_time:
                return float(w["start"])
    return min_time


def _crop_and_caption(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    segments: List[Dict],
    log_label: str,
) -> Dict:
    """Crop [start_time, end_time] out of source_path to out_path, then try
    to burn captions onto it -- matching the established crop_highlights_local
    / crop_chapters_local pattern (local/clipper.py): a caption failure is
    logged and leaves the uncaptioned-but-valid clip in place at out_path
    rather than raising, and any partial .captioned.mp4 is cleaned up. On
    success the caption result is atomically swapped into out_path.
    """
    crop_clip_local(
        source_path, start_time, end_time, aspect_ratio, out_path,
        framing="locked", cut_segments=[{"start_time": start_time, "end_time": end_time}],
    )
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

    Returns {"clip_path": out_path}. Raises SourceMismatchError if a
    re-download's live source duration doesn't match cached_duration.
    """
    full_source = os.path.join(run_dir, "full_source.mp4")
    full_transcript_path = os.path.join(run_dir, "full_source.json")

    if os.path.exists(full_source):
        with open(full_transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        return _crop_and_caption(
            full_source, start_time, end_time, aspect_ratio, out_path,
            transcript["segments"], "full_source",
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

        return _crop_and_caption(
            padded_path, relative_start, relative_end, aspect_ratio, out_path,
            fresh_transcript["segments"], "re-acquired",
        )
