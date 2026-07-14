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
