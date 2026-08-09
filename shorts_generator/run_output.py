"""Per-run output folder: title resolution, fixed paths, progress logging.

Every call to generate_shorts() (either mode) gets its own output/<Title>/
folder holding full_source.mp4, full_source.json (transcript), the Shorts/
subfolder, result.json, and progress.log. This module owns resolving the
title into a folder name and building those fixed paths; pipeline.py wires
it into both modes. It also owns listing and summarizing past runs
(`list_runs`/`summarize_run`) for the dashboard's History tab.
"""
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

import requests

from .config import LOCAL_OUTPUT_DIR, SHORT_FILENAME_STYLE

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9 _-]")
_WHITESPACE = re.compile(r"\s+")
_UNDERSCORE_RUNS = re.compile(r"_+")


@dataclass
class RunPaths:
    root: str
    shorts_dir: str
    chapters_dir: str
    source_video: str
    source_json: str
    highlights_json: str
    chapters_json: str
    result_json: str
    chapters_result_json: str
    source_url_txt: str
    progress_log: str


def sanitize_title(title: str, max_length: int = 100) -> str:
    """Turn a video title into a filesystem-safe folder name."""
    cleaned = _UNSAFE_CHARS.sub("_", title or "")
    cleaned = _WHITESPACE.sub("_", cleaned)
    cleaned = _UNDERSCORE_RUNS.sub("_", cleaned)
    cleaned = cleaned.strip("_-")
    cleaned = cleaned[:max_length].strip("_-")
    return cleaned or "untitled"


def unique_short_filename(
    title: str, used_names: set, index: Optional[int] = None, style: Optional[str] = None
) -> str:
    """Build a `.mp4` filename for a highlight, deduping against `used_names`
    (mutated in place) when two highlights collide within the same run.

    style="specific" (default, SHORT_FILENAME_STYLE env var) slugifies the
    highlight's own title (e.g. my_big_moment.mp4). style="generic" ignores
    the title and numbers clips positionally (video1.mp4, video2.mp4, ...)
    using `index`, which callers must pass for generic style.
    """
    style = (style or SHORT_FILENAME_STYLE).strip().lower()
    if style == "generic":
        if index is None:
            raise ValueError("index is required when style='generic'")
        base = f"video{index}"
    else:
        base = sanitize_title(title)
    name = f"{base}.mp4"
    n = 2
    while name in used_names:
        name = f"{base}_{n}.mp4"
        n += 1
    used_names.add(name)
    return name


def unique_chapter_filename(title: str, index: int, used_names: set) -> str:
    """Build a `.mp4` filename for a chapter: always a zero-padded numeric
    prefix + the chapter's own slugified title, so files sort into episode
    order in a plain file browser regardless of SHORT_FILENAME_STYLE (that
    style knob exists to anonymize clickbait Shorts titles, not to order
    them -- chapters have the opposite need)."""
    base = f"{index:02d}_{sanitize_title(title)}"
    name = f"{base}.mp4"
    n = 2
    while name in used_names:
        name = f"{base}_{n}.mp4"
        n += 1
    used_names.add(name)
    return name


def _title_via_oembed(url: str) -> Optional[str]:
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=10,
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    title = resp.json().get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def _extract_youtube_video_id(url: str) -> Optional[str]:
    """Best-effort extraction of a YouTube video id from a URL."""
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    if host == "youtu.be":
        video_id = parsed.path.lstrip("/").split("/", 1)[0]
        return video_id or None

    if "youtube.com" in host:
        if parsed.path.startswith("/watch"):
            qs = parse_qs(parsed.query)
            video_id = qs.get("v", [""])[0]
            return video_id or None
        match = re.search(r"/(?:shorts|embed|live)/([^/?#&]+)", parsed.path)
        if match:
            return match.group(1)

    return None


def _fallback_title_from_url(url: str) -> str:
    # Prefer the YouTube video id over the URL path: for a /watch URL the
    # path is just "/watch" (the id lives in the query string), so falling
    # back to a plain path-stem would collapse every watch-URL video onto
    # the same folder name whenever oEmbed fails.
    video_id = _extract_youtube_video_id(url)
    if video_id:
        return video_id
    parsed = urlparse(url)
    stem = Path(unquote(parsed.path)).stem
    return stem or "video"


def _fallback_title_from_path(path: str) -> str:
    parsed = urlparse(path)
    if parsed.scheme == "file":
        raw = unquote(parsed.path)
        return Path(raw).stem or "video"
    return Path(path).stem or "video"


def resolve_title(url_or_path: str) -> str:
    """Best-effort human-readable title for a YouTube URL, other URL, or local path."""
    parsed = urlparse(url_or_path)
    if parsed.scheme in ("http", "https"):
        title = _title_via_oembed(url_or_path)
        if title:
            return title
        return _fallback_title_from_url(url_or_path)
    return _fallback_title_from_path(url_or_path)


def resolve_output_dir(url_or_path: str, base_dir: Optional[str] = None) -> RunPaths:
    """Resolve url_or_path into a per-run RunPaths tree, creating the folders."""
    base_dir = base_dir or LOCAL_OUTPUT_DIR
    title = sanitize_title(resolve_title(url_or_path))
    root = os.path.join(base_dir, title)
    shorts_dir = os.path.join(root, "Shorts")
    chapters_dir = os.path.join(root, "Chapters")
    os.makedirs(shorts_dir, exist_ok=True)
    os.makedirs(chapters_dir, exist_ok=True)
    return RunPaths(
        root=root,
        shorts_dir=shorts_dir,
        chapters_dir=chapters_dir,
        source_video=os.path.join(root, "full_source.mp4"),
        source_json=os.path.join(root, "full_source.json"),
        highlights_json=os.path.join(root, "highlights.json"),
        chapters_json=os.path.join(root, "chapters.json"),
        result_json=os.path.join(root, "result.json"),
        # Deliberately separate from result_json: generate_shorts and
        # generate_chapters can both run against the same video (that's the
        # normal workflow this feature exists for), and their result shapes
        # are incompatible ("shorts"/"mode" vs "chapters", no "mode" key) --
        # sharing one path means whichever pipeline runs second silently
        # clobbers the other's result.json.
        chapters_result_json=os.path.join(root, "chapters_result.json"),
        source_url_txt=os.path.join(root, "source_url.txt"),
        progress_log=os.path.join(root, "progress.log"),
    )


def resolve_thread_output_dir(thesis: str, base_dir: Optional[str] = None) -> str:
    """A thread's output lives outside any single episode's RunPaths tree --
    it draws footage from two existing episode runs, so it gets its own
    output/_Threads/<slug>/ folder keyed by the thread's own thesis text."""
    base_dir = base_dir or LOCAL_OUTPUT_DIR
    slug = sanitize_title(thesis)
    root = os.path.join(base_dir, "_Threads", slug)
    os.makedirs(root, exist_ok=True)
    return root


def write_source_url(paths: RunPaths, youtube_url: str) -> None:
    """Persist the original source URL so a pruned full_source.mp4 (deleted
    to save disk once a channel has 100+ episodes) can still be
    re-acquired later -- see corpus.py / local/thread_source.py, which need
    to re-download a specific clip's source long after the original run."""
    with open(paths.source_url_txt, "w", encoding="utf-8") as f:
        f.write(youtube_url.strip())


def read_source_url(paths: RunPaths) -> Optional[str]:
    if not os.path.exists(paths.source_url_txt):
        return None
    with open(paths.source_url_txt, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content or None


@dataclass
class RunSummary:
    name: str
    mtime: float
    source_exists: bool
    source_size: int
    shorts_count: int
    shorts_size: int


def _run_mtime(root: str) -> float:
    """Newest mtime across every file in `root` (falls back to the dir's own)."""
    mtimes = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            try:
                mtimes.append(os.path.getmtime(path))
            except OSError:
                continue
    return max(mtimes) if mtimes else os.path.getmtime(root)


def summarize_run(name: str, root: str) -> RunSummary:
    """Stat a single run folder — no result.json parsing (it can be several MB)."""
    source_video = os.path.join(root, "full_source.mp4")
    source_exists = os.path.isfile(source_video)
    source_size = os.path.getsize(source_video) if source_exists else 0

    shorts_dir = os.path.join(root, "Shorts")
    shorts_names = []
    if os.path.isdir(shorts_dir):
        shorts_names = sorted(
            n for n in os.listdir(shorts_dir)
            if n.endswith(".mp4")
        )
    shorts_size = sum(os.path.getsize(os.path.join(shorts_dir, n)) for n in shorts_names)

    return RunSummary(
        name=name,
        mtime=_run_mtime(root),
        source_exists=source_exists,
        source_size=source_size,
        shorts_count=len(shorts_names),
        shorts_size=shorts_size,
    )


def list_runs(base_dir: Optional[str] = None) -> List[RunSummary]:
    """List every run folder under `base_dir`, newest first.

    A run folder that disappears mid-scan (e.g. deleted by a concurrent
    History-tab request) is silently skipped rather than raising — every
    other entry is still returned.
    """
    base_dir = base_dir or LOCAL_OUTPUT_DIR
    if not os.path.isdir(base_dir):
        return []
    runs = []
    for name in os.listdir(base_dir):
        root = os.path.join(base_dir, name)
        if not os.path.isdir(root):
            continue
        try:
            runs.append(summarize_run(name, root))
        except OSError:
            continue
    runs.sort(key=lambda r: r.mtime, reverse=True)
    return runs


def write_descriptions(shorts_dir: str, shorts: List[Dict]) -> str:
    """Write a copy-paste-ready descriptions.txt next to the short clip files.

    One block per short that actually has a clip_url, numbered by position in
    `shorts` regardless of the clip's own title-derived filename. Prefers the
    Shorts-optimized yt_title / yt_hashtags (emitted by the highlight step in
    highlights.py) when present, falling back to the highlight-step `title`
    otherwise. Each block also carries a hook-grade line (`hook_strength` /
    `hook_self_contained`, a human-review-only signal, not used for ranking)
    so this file doubles as a pick-list, not just a copy-paste source.
    """
    path = os.path.join(shorts_dir, "descriptions.txt")
    blocks = []
    for i, s in enumerate(shorts, 1):
        if not s.get("clip_url"):
            continue
        title = (s.get("yt_title") or s.get("title") or "Untitled").strip()
        description = (s.get("description") or "").strip()
        hashtags = s.get("yt_hashtags") or []
        hashtags_text = " ".join(hashtags)
        if hashtags_text and hashtags_text not in description:
            description = (description + "\n\n" + hashtags_text).strip()
        hook_strength = s.get("hook_strength") or 0
        self_contained = "yes" if s.get("hook_self_contained") else "no"
        hook_line = f"hook: {hook_strength}  self-contained: {self_contained}"
        blocks.append(f"short {i:02d} - {title}\n{hook_line}\n{description}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks))
        if blocks:
            f.write("\n")

    return path


def write_chapter_descriptions(chapters_dir: str, chapters: List[Dict]) -> str:
    """Write a copy-paste-ready chapters_description.txt next to the chapter
    clip files. One block per chapter that actually has a clip_url, numbered
    by position in `chapters` -- the same index unique_chapter_filename used
    for that chapter's own filename, so "chapter 02" here lines up with
    "02_..." on disk (unlike Shorts' write_descriptions, whose block number
    has no relation to the short's own slugified-title filename). Each
    block carries the original video's timestamp range (each chapter is its
    own file, not a marker in one long video, but the range is still useful
    context) plus the full `summary` -- unlike Shorts' write_descriptions,
    there's no yt_title/hashtags/hook_strength here, those fields don't
    exist on the chapter shape.
    """
    path = os.path.join(chapters_dir, "chapters_description.txt")
    blocks = []
    for i, c in enumerate(chapters, 1):
        if not c.get("clip_url"):
            continue
        title = (c.get("title") or "Untitled Chapter").strip()
        start = float(c.get("start_time") or 0.0)
        end = float(c.get("end_time") or 0.0)
        summary = (c.get("summary") or "").strip()
        blocks.append(f"chapter {i:02d} - {title} ({start:.1f}s - {end:.1f}s)\n{summary}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks))
        if blocks:
            f.write("\n")

    return path


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextmanager
def capture_progress_log(path: str):
    """Duplicate stdout/stderr to `path` (appended) for the duration of the block."""
    log_file = open(path, "a", encoding="utf-8")
    log_file.write(f"\n=== run start {datetime.now().isoformat(timespec='seconds')} ===\n")
    log_file.flush()

    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(old_out, log_file)
    sys.stderr = _Tee(old_err, log_file)
    try:
        yield
    except Exception as e:
        log_file.write(f"FAILED: {e}\n")
        log_file.flush()
        raise
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log_file.close()
