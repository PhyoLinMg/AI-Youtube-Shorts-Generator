"""Per-run output folder: title resolution, fixed paths, progress logging.

Every call to generate_shorts() (either mode) gets its own output/<Title>/
folder holding full_source.mp4, full_source.json (transcript), the Shorts/
subfolder, result.json, and progress.log. This module owns resolving the
title into a folder name and building those fixed paths; pipeline.py wires
it into both modes.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import requests

from .config import LOCAL_OUTPUT_DIR

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9 _-]")
_WHITESPACE = re.compile(r"\s+")
_UNDERSCORE_RUNS = re.compile(r"_+")


@dataclass
class RunPaths:
    root: str
    shorts_dir: str
    source_video: str
    source_json: str
    result_json: str
    progress_log: str


def sanitize_title(title: str, max_length: int = 100) -> str:
    """Turn a video title into a filesystem-safe folder name."""
    cleaned = _UNSAFE_CHARS.sub("_", title or "")
    cleaned = _WHITESPACE.sub("_", cleaned)
    cleaned = _UNDERSCORE_RUNS.sub("_", cleaned)
    cleaned = cleaned.strip("_-")
    cleaned = cleaned[:max_length].strip("_-")
    return cleaned or "untitled"


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


def _fallback_title_from_url(url: str) -> str:
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
    os.makedirs(shorts_dir, exist_ok=True)
    return RunPaths(
        root=root,
        shorts_dir=shorts_dir,
        source_video=os.path.join(root, "full_source.mp4"),
        source_json=os.path.join(root, "full_source.json"),
        result_json=os.path.join(root, "result.json"),
        progress_log=os.path.join(root, "progress.log"),
    )
