# Per-Run Output Folder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Group every pipeline run's outputs under `output/<Title>/` (source video, transcript, shorts, result json, progress log) instead of a flat `output/` directory, for both `--mode api` and `--mode local`.

**Architecture:** A new mode-agnostic module `shorts_generator/run_output.py` resolves the video's title (YouTube oEmbed, with filename-stem fallbacks) into a sanitized folder name, builds the fixed set of paths for that run, and provides a stdout/stderr-duplicating context manager for `progress.log`. `pipeline.py` resolves this once per `generate_shorts()` call and threads it through both `_run_local` and `_run_api`, which each gain skip-if-cached logic for the (mode-specific) parts of resume that are actually safe to skip.

**Tech Stack:** Python 3, pytest, `requests` (already a dependency, used for the YouTube oEmbed lookup — no new dependency), `ffmpeg`/`ffprobe` on PATH (already required for existing tests).

**Spec:** `docs/superpowers/specs/2026-07-15-per-run-output-folder-design.md`

---

## Task 1: `run_output.py` — RunPaths + title sanitizing

**Files:**
- Create: `shorts_generator/run_output.py`
- Test: `tests/test_run_output.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_output.py`:

```python
from shorts_generator import run_output


def test_sanitize_title_replaces_spaces_with_underscores():
    assert run_output.sanitize_title("How to Build a Startup") == "How_to_Build_a_Startup"


def test_sanitize_title_strips_unsafe_characters():
    assert run_output.sanitize_title("A/B: Test?!") == "A_B_Test"


def test_sanitize_title_empty_input_falls_back_to_untitled():
    assert run_output.sanitize_title("") == "untitled"
    assert run_output.sanitize_title("???") == "untitled"


def test_sanitize_title_truncates_long_titles():
    result = run_output.sanitize_title("x" * 150)
    assert len(result) == 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run_output.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shorts_generator.run_output'`

- [ ] **Step 3: Implement `RunPaths` and `sanitize_title`**

Create `shorts_generator/run_output.py`:

```python
"""Per-run output folder: title resolution, fixed paths, progress logging.

Every call to generate_shorts() (either mode) gets its own output/<Title>/
folder holding full_source.mp4, full_source.json (transcript), the Shorts/
subfolder, result.json, and progress.log. This module owns resolving the
title into a folder name and building those fixed paths; pipeline.py wires
it into both modes.
"""
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_output.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py
git commit -m "feat: add RunPaths + title sanitizing for per-run output folders"
```

---

## Task 2: `run_output.py` — title resolution + `resolve_output_dir`

**Files:**
- Modify: `shorts_generator/run_output.py`
- Test: `tests/test_run_output.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_run_output.py`:

```python
import os


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


def test_resolve_title_uses_oembed_title(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        assert "oembed" in url
        assert params["url"] == "https://www.youtube.com/watch?v=abc123"
        return _FakeResponse(200, {"title": "My Cool Video!"})

    monkeypatch.setattr(run_output.requests, "get", fake_get)
    assert run_output.resolve_title("https://www.youtube.com/watch?v=abc123") == "My Cool Video!"


def test_resolve_title_falls_back_on_oembed_network_error(monkeypatch):
    def fake_get(*args, **kwargs):
        raise run_output.requests.RequestException("network down")

    monkeypatch.setattr(run_output.requests, "get", fake_get)
    title = run_output.resolve_title("https://www.youtube.com/watch?v=abc123")
    assert title  # falls back to a non-empty name derived from the URL


def test_resolve_title_falls_back_on_non_200(monkeypatch):
    monkeypatch.setattr(run_output.requests, "get", lambda *a, **k: _FakeResponse(404, {}))
    title = run_output.resolve_title("https://www.youtube.com/watch?v=abc123")
    assert title


def test_resolve_title_for_local_path_uses_filename_stem(tmp_path):
    media = tmp_path / "my_video_file.mp4"
    media.write_bytes(b"x")
    assert run_output.resolve_title(str(media)) == "my_video_file"


def test_resolve_output_dir_builds_expected_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_output.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"title": "How To Build A Startup"}),
    )

    paths = run_output.resolve_output_dir(
        "https://www.youtube.com/watch?v=abc123", base_dir=str(tmp_path)
    )

    assert paths.root == str(tmp_path / "How_To_Build_A_Startup")
    assert paths.shorts_dir == os.path.join(paths.root, "Shorts")
    assert paths.source_video == os.path.join(paths.root, "full_source.mp4")
    assert paths.source_json == os.path.join(paths.root, "full_source.json")
    assert paths.result_json == os.path.join(paths.root, "result.json")
    assert paths.progress_log == os.path.join(paths.root, "progress.log")
    assert os.path.isdir(paths.shorts_dir)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run_output.py -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.run_output' has no attribute 'resolve_title'` (and similar for `resolve_output_dir`)

- [ ] **Step 3: Implement title resolution + `resolve_output_dir`**

Append to `shorts_generator/run_output.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_output.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py
git commit -m "feat: resolve video title (oEmbed + fallbacks) into a per-run output tree"
```

---

## Task 3: `run_output.py` — `progress.log` via a stdout/stderr Tee

**Files:**
- Modify: `shorts_generator/run_output.py`
- Test: `tests/test_run_output.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_run_output.py`:

```python
from pathlib import Path

import pytest


def test_capture_progress_log_duplicates_stdout_to_file(tmp_path, capsys):
    log_path = str(tmp_path / "progress.log")
    with run_output.capture_progress_log(log_path):
        print("hello from pipeline")

    captured = capsys.readouterr()
    assert "hello from pipeline" in captured.out

    content = Path(log_path).read_text()
    assert "hello from pipeline" in content
    assert "run start" in content


def test_capture_progress_log_records_failure_and_reraises(tmp_path):
    log_path = str(tmp_path / "progress.log")
    with pytest.raises(RuntimeError):
        with run_output.capture_progress_log(log_path):
            raise RuntimeError("boom")

    content = Path(log_path).read_text()
    assert "FAILED: boom" in content


def test_capture_progress_log_restores_stdout_after(tmp_path):
    import sys
    log_path = str(tmp_path / "progress.log")
    original_stdout = sys.stdout
    with run_output.capture_progress_log(log_path):
        pass
    assert sys.stdout is original_stdout


def test_capture_progress_log_appends_across_calls(tmp_path):
    log_path = str(tmp_path / "progress.log")
    with run_output.capture_progress_log(log_path):
        print("first run")
    with run_output.capture_progress_log(log_path):
        print("second run")

    content = Path(log_path).read_text()
    assert "first run" in content
    assert "second run" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_run_output.py -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.run_output' has no attribute 'capture_progress_log'`

- [ ] **Step 3: Implement the `Tee` + `capture_progress_log`**

Append to `shorts_generator/run_output.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_output.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py
git commit -m "feat: add progress.log capture via a stdout/stderr tee"
```

---

## Task 4: transcript cache follows the media file's own directory

**Files:**
- Modify: `shorts_generator/local/transcriber.py:11,14-18`
- Modify: `tests/test_local_transcriber.py`

**Why:** `_transcript_cache_path` currently resolves the cache directory from
the *global* `LOCAL_OUTPUT_DIR`, ignoring where the media file actually
lives. Once sources live at `output/<Title>/full_source.mp4`, this must
cache at `output/<Title>/full_source.json` — i.e. next to the media file,
not in a global flat directory.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_local_transcriber.py`:

```python
def test_cache_path_follows_media_directory_not_global_default(tmp_path, monkeypatch):
    # Deliberately wrong global default — cache must NOT use this.
    monkeypatch.setattr(tr, "LOCAL_OUTPUT_DIR", "should_not_be_used")
    nested = tmp_path / "run_folder"
    nested.mkdir()
    media = nested / "full_source.mp4"
    media.write_bytes(b"x")

    cache_path = tr._transcript_cache_path(str(media))

    assert cache_path.parent == nested.resolve()
    assert cache_path.name == "full_source.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_local_transcriber.py::test_cache_path_follows_media_directory_not_global_default -v`
Expected: FAIL — `cache_path.parent` is `Path("should_not_be_used")`, not `nested.resolve()`

- [ ] **Step 3: Fix `_transcript_cache_path`**

In `shorts_generator/local/transcriber.py`, change:

```python
from ..config import LOCAL_OUTPUT_DIR, LOCAL_WHISPER_DEVICE, LOCAL_WHISPER_MODEL
```

to:

```python
from ..config import LOCAL_WHISPER_DEVICE, LOCAL_WHISPER_MODEL
```

and change:

```python
def _transcript_cache_path(media_path: str) -> Path:
    """Return the .json cache path for a media file."""
    cache_dir = Path(LOCAL_OUTPUT_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (Path(media_path).stem + ".json")
```

to:

```python
def _transcript_cache_path(media_path: str) -> Path:
    """Return the .json cache path for a media file, alongside the media itself."""
    cache_dir = Path(media_path).resolve().parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (Path(media_path).stem + ".json")
```

- [ ] **Step 4: Drop the now-dead `LOCAL_OUTPUT_DIR` monkeypatches**

In `tests/test_local_transcriber.py`, the three existing tests
(`test_json_cache_roundtrip_preserves_words`, `test_cache_path_is_json`,
`test_transcribe_local_requests_word_timestamps_and_collects_words`) each
start with `monkeypatch.setattr(tr, "LOCAL_OUTPUT_DIR", str(tmp_path))`.
Delete that line from all three — it's no longer read by
`_transcript_cache_path`, and the module no longer imports the name.

- [ ] **Step 5: Run all transcriber tests to verify they pass**

Run: `pytest tests/test_local_transcriber.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/transcriber.py tests/test_local_transcriber.py
git commit -m "fix: transcript cache follows the media file's own directory"
```

---

## Task 5: `local/downloader.py` — download/copy straight to a target path

**Files:**
- Modify: `shorts_generator/local/downloader.py`
- Test: `tests/test_local_downloader.py` (new)

**Why:** Folder-level existence checks (in `pipeline.py`, Task 7) now decide
whether to skip downloading at all. `download_youtube_local` no longer needs
its own id-based cache lookup — it just needs to always land the resolved
video at an exact caller-supplied path, as `full_source.mp4` (remuxing to
mp4 if the input isn't already one, e.g. a local `.mkv`/`.webm` file).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_local_downloader.py`:

```python
import os
import subprocess

import pytest

from shorts_generator.local import downloader


def test_local_file_input_is_copied_to_target(tmp_path):
    src = tmp_path / "input.mp4"
    src.write_bytes(b"fake-mp4-bytes")
    target = str(tmp_path / "run" / "full_source.mp4")

    result = downloader.download_youtube_local(str(src), target_path=target)

    assert result == target
    assert os.path.exists(target)
    assert open(target, "rb").read() == b"fake-mp4-bytes"


def test_local_file_already_at_target_is_left_alone(tmp_path):
    target = str(tmp_path / "full_source.mp4")
    with open(target, "wb") as f:
        f.write(b"already-here")

    result = downloader.download_youtube_local(target, target_path=target)

    assert result == target
    assert open(target, "rb").read() == b"already-here"


@pytest.fixture(scope="module")
def synthetic_mkv(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("mkv_source")
    path = str(tmp_dir / "input.mkv")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            path,
        ],
        check=True,
    )
    return path


def test_non_mp4_local_input_is_remuxed_to_mp4(tmp_path, synthetic_mkv):
    target = str(tmp_path / "run" / "full_source.mp4")

    result = downloader.download_youtube_local(synthetic_mkv, target_path=target)

    assert result == target
    assert os.path.exists(target)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "csv=p=0", target],
        capture_output=True, text=True, check=True,
    )
    assert "mp4" in probe.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_local_downloader.py -v`
Expected: FAIL with `TypeError: download_youtube_local() got an unexpected keyword argument 'target_path'`

- [ ] **Step 3: Rewrite `download_youtube_local` around `target_path`**

In `shorts_generator/local/downloader.py`, change the imports at the top from:

```python
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from typing import Optional

from ..config import LOCAL_OUTPUT_DIR
```

to:

```python
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from typing import Optional
```

Delete the `_existing_download` function entirely (no longer used):

```python
def _existing_download(out_dir: str, video_id: str) -> Optional[str]:
    """Return a cached download path if we already have this YouTube id."""
    for ext in (".mp4", ".mkv", ".webm"):
        candidate = os.path.join(out_dir, f"source_{video_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None
```

Add a helper just above `download_youtube_local`:

```python
def _ensure_mp4_at(src: str, dest: str) -> None:
    """Make sure `dest` exists as an mp4 copy of `src` (remux if needed)."""
    if os.path.abspath(src) == os.path.abspath(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if src.lower().endswith(".mp4"):
        shutil.copyfile(src, dest)
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c", "copy", dest],
        check=True,
    )
```

Replace the whole `download_youtube_local` function with:

```python
def download_youtube_local(video_url: str, target_path: str, fmt: str = "720") -> str:
    """Resolve video_url (YouTube URL, other remote URL, local path, or
    file:// URL) into a local mp4 at exactly `target_path`."""
    local_path = _resolve_local_path(video_url)
    if local_path:
        print(f"[download/local] using local file: {local_path} -> {target_path}", flush=True)
        _ensure_mp4_at(local_path, target_path)
        return target_path

    yt_dlp = _import_ytdlp()
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    print(f"[download/local] {video_url} @ {fmt}p -> {target_path}", flush=True)
    ydl_opts = {
        "format": _format_for(fmt),
        "outtmpl": target_path + ".download.%(ext)s",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        downloaded = ydl.prepare_filename(info)
        # merge_output_format may rename the extension after merge
        if not os.path.exists(downloaded):
            stem, _ = os.path.splitext(downloaded)
            for ext in (".mp4", ".mkv", ".webm"):
                if os.path.exists(stem + ext):
                    downloaded = stem + ext
                    break

    _ensure_mp4_at(downloaded, target_path)
    if os.path.abspath(downloaded) != os.path.abspath(target_path) and os.path.exists(downloaded):
        os.remove(downloaded)

    print(f"[download/local] ready: {target_path}", flush=True)
    return target_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_local_downloader.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/local/downloader.py tests/test_local_downloader.py
git commit -m "feat: download/copy local-mode sources straight to full_source.mp4"
```

---

## Task 6: `Short-NN.mp4` naming in both clippers

**Files:**
- Modify: `shorts_generator/local/clipper.py:441`
- Modify: `shorts_generator/clipper.py:70`
- Modify: `tests/test_local_clipper.py`
- Modify: `tests/test_clipper_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_local_clipper.py`:

```python
def test_output_filename_uses_short_dash_prefix(tmp_path, synthetic_source):
    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source,
        [_highlight()],
        aspect_ratio="9:16",
        out_dir=out_dir,
        transcript_segments=_segments(),
    )
    assert os.path.basename(results[0]["clip_url"]) == "Short-01.mp4"
```

Append to `tests/test_clipper_api.py`:

```python
def test_output_filename_uses_short_dash_prefix(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
    )

    assert os.path.basename(results[0]["clip_url"]) == "Short-01.mp4"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_local_clipper.py::test_output_filename_uses_short_dash_prefix tests/test_clipper_api.py::test_output_filename_uses_short_dash_prefix -v`
Expected: FAIL — both assert `"Short-01.mp4"` against the current `"short_01.mp4"`

- [ ] **Step 3: Rename the output pattern in both clippers**

In `shorts_generator/local/clipper.py`, change:

```python
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
```

to:

```python
        out_path = os.path.join(out_dir, f"Short-{i:02d}.mp4")
```

In `shorts_generator/clipper.py`, change:

```python
                final_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
```

to:

```python
                final_path = os.path.join(out_dir, f"Short-{i:02d}.mp4")
```

- [ ] **Step 4: Run the full clipper test suites to verify everything passes**

Run: `pytest tests/test_local_clipper.py tests/test_clipper_api.py -v`
Expected: PASS (all tests in both files)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/local/clipper.py shorts_generator/clipper.py tests/test_local_clipper.py tests/test_clipper_api.py
git commit -m "feat: name rendered shorts Short-NN.mp4"
```

---

## Task 7: `pipeline.py` — thread `RunPaths` through both modes

**Files:**
- Modify: `shorts_generator/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Why:** This is where the per-run folder actually gets used: resolved once
in `generate_shorts()`, progress-logged for the whole call, and threaded
into `_run_local` / `_run_api` so each writes into the right places and
applies the skip-if-cached rules from the spec (full skip in local mode;
partial skip — local mp4 copy + transcript only — in api mode, since MuAPI's
`/autocrop` needs a fresh hosted URL every time).

- [ ] **Step 1: Write the failing tests**

Replace the full contents of `tests/test_pipeline.py` with:

```python
import os
from unittest.mock import Mock

import shorts_generator.local.clipper as local_clipper_module
import shorts_generator.local.downloader as local_downloader_module
import shorts_generator.local.transcriber as local_transcriber_module
import shorts_generator.pipeline as pipeline_module
from shorts_generator.run_output import RunPaths


def _fake_transcript():
    return {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi there"}]}


def _fake_highlights_result():
    return {"highlights": [{"start_time": 0.0, "end_time": 3.0, "score": 90, "title": "Clip"}]}


def _paths(tmp_path):
    root = str(tmp_path / "Video_Title")
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


def test_run_local_threads_captions_params(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())

    crop_mock = Mock(return_value=[{"clip_url": "/tmp/out/Short-01.mp4"}])
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", crop_mock)

    result = pipeline_module._run_local(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.7,
        paths=_paths(tmp_path),
        word_highlight=False,
    )

    assert result["mode"] == "local"
    assert result["shorts"] == [{"clip_url": "/tmp/out/Short-01.mp4"}]

    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is False
    assert kwargs["caption_fade_duration"] == 0.7
    assert kwargs["word_highlight"] is False
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]


def test_run_local_skips_download_when_source_already_exists(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    with open(paths.source_video, "wb") as f:
        f.write(b"already downloaded")

    def _fail_if_called(*a, **k):
        raise AssertionError("download_youtube_local should not be called when full_source.mp4 exists")

    monkeypatch.setattr(local_downloader_module, "download_youtube_local", _fail_if_called)
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", Mock(return_value=[]))

    result = pipeline_module._run_local(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.3,
        paths=paths,
        word_highlight=True,
    )

    assert result["source_video_url"] == paths.source_video


def test_run_api_threads_captions_params(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", lambda url, dest: dest)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())

    crop_mock = Mock(return_value=[{"clip_url": "https://hosted.example/Short-1.mp4"}])
    monkeypatch.setattr(pipeline_module, "crop_highlights", crop_mock)

    result = pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=False,
    )

    assert result["mode"] == "api"
    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is True
    assert kwargs["caption_fade_duration"] == 0.3
    assert kwargs["word_highlight"] is False
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]


def test_run_api_skips_local_copy_and_transcribe_when_cached(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    with open(paths.source_video, "wb") as f:
        f.write(b"cached mp4")
    import json
    with open(paths.source_json, "w") as f:
        json.dump(_fake_transcript(), f)

    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")

    def _fail_download_to(*a, **k):
        raise AssertionError("_download_to should not be called when full_source.mp4 is cached")
    monkeypatch.setattr(pipeline_module, "_download_to", _fail_download_to)

    def _fail_transcribe(*a, **k):
        raise AssertionError("transcribe should not be called when full_source.json is cached")
    monkeypatch.setattr(pipeline_module, "transcribe", _fail_transcribe)

    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())
    monkeypatch.setattr(pipeline_module, "crop_highlights", Mock(return_value=[]))

    result = pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=paths,
        word_highlight=True,
    )

    assert result["transcript"] == _fake_transcript()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `TypeError: _run_local() missing 1 required positional argument: 'paths'` (and similarly for `_run_api`)

- [ ] **Step 3: Rewrite `pipeline.py`**

Replace the full contents of `shorts_generator/pipeline.py` with:

```python
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
from .highlights import call_muapi_llm, get_highlights
from .run_output import RunPaths, capture_progress_log, resolve_output_dir
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

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_local_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
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
) -> Dict:
    # MuAPI /autocrop needs a fresh hosted URL for every crop, and that URL
    # only comes from /youtube-download — so this call can't be skipped on
    # rerun even if we already have a local copy of the video.
    source_url = download_youtube(youtube_url, fmt=download_format)

    if os.path.exists(paths.source_video):
        print(f"[pipeline] reusing cached local copy: {paths.source_video}", flush=True)
    else:
        _download_to(source_url, paths.source_video)
        print(f"[pipeline] saved local copy: {paths.source_video}", flush=True)

    if os.path.exists(paths.source_json):
        print(f"[pipeline] reusing cached transcript: {paths.source_json}", flush=True)
        with open(paths.source_json, "r", encoding="utf-8") as f:
            transcript = json.load(f)
    else:
        transcript = transcribe(source_url, language=language)
        with open(paths.source_json, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False)

    if not transcript["segments"]:
        raise RuntimeError(
            "Whisper produced no segments. The video may have no detectable speech."
        )

    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_muapi_llm)
    all_highlights: List[Dict] = highlights_result.get("highlights", [])
    if not all_highlights:
        raise RuntimeError("Highlight generator returned zero clips.")

    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)

    shorts = crop_highlights(
        source_url,
        top,
        aspect_ratio=aspect_ratio,
        transcript_segments=transcript["segments"],
        captions=captions,
        caption_fade_duration=caption_fade_duration,
        word_highlight=word_highlight,
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
    download_format: str = "720",
    language: Optional[str] = None,
    mode: str = "api",
    captions: bool = True,
    caption_fade_duration: float = 0.3,
    word_highlight: bool = True,
    framing: str = "locked",
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

    Returns:
        {
          "mode": "api" | "local",
          "output_dir": str,         # output/<Title> for this run
          "source_video_url": str,   # hosted URL (api) or local path (local)
          "transcript": {...},
          "highlights": [...],       # all candidates ranked
          "shorts": [...],           # top `num_clips`, each with:
                                      #   clip_url: local path (Shorts/Short-NN.mp4)
                                      #   hosted_clip_url: original MuAPI URL (api mode,
                                      #     only present when captions were burned in)
                                      #   captions_error: present if caption burn-in failed
                                      #     for that clip (falls back to the uncaptioned clip)
        }
    """
    mode = (mode or "api").lower()
    if mode not in ("api", "local"):
        raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")

    paths = resolve_output_dir(youtube_url)
    with capture_progress_log(paths.progress_log):
        if mode == "local":
            result = _run_local(
                youtube_url, num_clips, aspect_ratio, download_format, language, captions, caption_fade_duration,
                paths, word_highlight=word_highlight, framing=framing,
            )
        else:
            result = _run_api(
                youtube_url, num_clips, aspect_ratio, download_format, language, captions, caption_fade_duration,
                paths, word_highlight=word_highlight,
            )

        with open(paths.result_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/pipeline.py tests/test_pipeline.py
git commit -m "feat: thread per-run output paths through both pipeline modes"
```

---

## Task 8: `main.py` — surface the output folder

**Files:**
- Modify: `main.py:86-90`

- [ ] **Step 1: Add the output-folder line to the summary block**

In `main.py`, change:

```python
    print("\n" + "=" * 72)
    print(f"Mode:          {result.get('mode', args.mode)}")
    print(f"Source video:  {result['source_video_url']}")
```

to:

```python
    print("\n" + "=" * 72)
    print(f"Mode:          {result.get('mode', args.mode)}")
    print(f"Output folder: {result.get('output_dir')}")
    print(f"Source video:  {result['source_video_url']}")
```

- [ ] **Step 2: Run the existing CLI tests to verify nothing broke**

Run: `pytest tests/test_main.py -v`
Expected: PASS (5 passed) — these only exercise `build_parser`, unaffected by this change

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: print the per-run output folder in the CLI summary"
```

---

## Task 9: README — document the new folder structure

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the env var comment**

Change:

```
   LOCAL_OUTPUT_DIR=output           # where local mp4s land
```

to:

```
   LOCAL_OUTPUT_DIR=output           # base folder; each run gets output/<Title>/
```

- [ ] **Step 2: Replace the flat-output-path line**

Change:

```
Local mode writes the rendered shorts to `./output/short_01.mp4`, `short_02.mp4`, … (override with `LOCAL_OUTPUT_DIR`).
```

to:

```
Every run (both modes) writes into its own folder named after the video, under `LOCAL_OUTPUT_DIR` (default `output/`):

```
output/<Video Title>/
  Shorts/
    Short-01.mp4
    Short-02.mp4
  full_source.mp4      # the downloaded/copied source video
  full_source.json     # the transcript (cached — see below)
  result.json           # full pipeline result (same shape as --output-json)
  progress.log          # everything printed to the console during the run
```

The title comes from YouTube's oEmbed metadata (falls back to the input filename for local files or non-YouTube URLs), sanitized for the filesystem. Running the same URL again reuses the same folder.
```

- [ ] **Step 3: Replace the caching paragraphs**

Change:

```
Local transcription is cached as an `.srt` file in `LOCAL_OUTPUT_DIR` using the
video's base name. If the cache already exists and is newer than the source
file, the app reuses it instead of running Whisper again.

Local downloads are also cached in `LOCAL_OUTPUT_DIR` as
`source_<youtube_id>.mp4` when the input is a YouTube URL. If that file already
exists, the app skips `yt-dlp` and reuses the cached video.
```

to:

```
Rerunning the same URL reuses its `output/<Title>/` folder and skips the
expensive parts that are already done:

- **local mode**: skips the download if `full_source.mp4` already exists, and
  skips transcription if `full_source.json` already exists (cached as JSON
  next to the source video). Highlight ranking, cropping, and `result.json`
  are always redone.
- **api mode**: MuAPI's `/autocrop` needs a fresh hosted URL from
  `/youtube-download` for every crop, so that call always re-runs — but the
  local `full_source.mp4` copy and the transcription call are skipped if
  they're already cached.
```

- [ ] **Step 4: Update the console-output and JSON examples**

Change:

```
     clip:   output/short_01.mp4
```

to:

```
     clip:   output/My_Video_Title/Shorts/Short-01.mp4
```

Change:

```json
{
  "source_video_url": "...",
  "transcript": { "duration": 1873.4, "segments": [...] },
  "highlights": [ {...}, {...}, ... ],
  "shorts": [
    {
      "title": "...",
      "start_time": 124.3,
      "end_time": 187.6,
      "score": 92,
      "hook_sentence": "...",
      "virality_reason": "...",
      "clip_url": "output/short_01.mp4",
      "hosted_clip_url": "https://.../short_1.mp4"
    }
  ]
}
```

to:

```json
{
  "mode": "api",
  "output_dir": "output/My_Video_Title",
  "source_video_url": "...",
  "transcript": { "duration": 1873.4, "segments": [...] },
  "highlights": [ {...}, {...}, ... ],
  "shorts": [
    {
      "title": "...",
      "start_time": 124.3,
      "end_time": 187.6,
      "score": 92,
      "hook_sentence": "...",
      "virality_reason": "...",
      "clip_url": "output/My_Video_Title/Shorts/Short-01.mp4",
      "hosted_clip_url": "https://.../short_1.mp4"
    }
  ]
}
```

Note just below it says `--output-json result.json` "produces" that shape —
add a one-line clarification right before that JSON block:

```
`result.json` (containing this same shape) is now always written inside
`output_dir` automatically; `--output-json result.json` additionally writes
a copy to whatever path you give it.
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document the per-run output folder structure"
```

---

## Task 10: Full test suite + manual smoke check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -v`
Expected: All tests pass (existing suite + all new tests added in Tasks 1–7)

- [ ] **Step 2: Manual smoke check with a local file (no network required beyond the LLM call)**

If you have a local mp4 handy and `requirements-local.txt` installed with an
`OPENAI_API_KEY`/`GEMINI_API_KEY` set:

```bash
python main.py "/path/to/any/local/video.mp4" --mode local --num-clips 1
```

Verify:
- `output/<stem-of-video.mp4>/Shorts/Short-01.mp4` exists and plays
- `output/<stem-of-video.mp4>/full_source.mp4` exists
- `output/<stem-of-video.mp4>/full_source.json` exists and has `segments`
- `output/<stem-of-video.mp4>/result.json` exists and matches the printed summary
- `output/<stem-of-video.mp4>/progress.log` contains the same `[download/local]` / `[transcribe/local]` / `[clip/local]` lines printed to the console
- Running the exact same command again logs `reusing cached source` / `reusing cached transcript` and does not re-download or re-transcribe

- [ ] **Step 3: Report results**

If anything fails, stop and debug before considering the plan complete — do
not mark this task done on a failing run.
