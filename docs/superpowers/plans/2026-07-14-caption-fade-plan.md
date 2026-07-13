# Fade-in Subtitle Burn-in Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Burn phrase-length, fade-in-animated captions onto generated shorts in both `--mode api` and `--mode local`, on by default, with fade duration exposed as a CLI flag (`--caption-fade-duration`) and an opt-out flag (`--no-captions`).

**Architecture:** A new mode-agnostic module `shorts_generator/captions.py` slices the full transcript to a clip's time window, splits it into ~7-word phrase chunks, writes an ASS subtitle file (one `\fad(ms,0)` fade-in tag per line), and burns it in via ffmpeg's `subtitles` (libass) filter. Local mode calls it right after its existing cut+reframe step. API mode downloads MuAPI's hosted clip locally first, then calls it the same way — this is the one real behavior change: API mode now needs `ffmpeg` on `PATH` too when captions are on (the default).

**Tech Stack:** Python 3, ffmpeg (`subtitles`/libass filter, already present in this environment — `ffmpeg -filters | grep subtitles` confirmed it), pytest (new dev dependency), `requests` (already a dependency, used for a plain HTTP download in API mode).

**Spec:** `docs/superpowers/specs/2026-07-14-caption-fade-design.md`

---

## Task 1: `captions.py` — transcript chunking (`_chunk_segments`)

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/conftest.py`
- Create: `tests/test_captions.py`
- Create: `shorts_generator/captions.py`

- [ ] **Step 1: Add the dev/test dependency**

Create `requirements-dev.txt`:

```
pytest>=8.0
```

- [ ] **Step 2: Make the repo root importable from `tests/`**

`main.py` lives at the repo root (not inside a package), so pytest needs the
repo root on `sys.path` to import it later. Create `tests/conftest.py`:

```python
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
```

- [ ] **Step 3: Install dev dependencies**

Run: `source .venv/bin/activate && pip install -r requirements-dev.txt`
Expected: `pytest` installs successfully.

- [ ] **Step 4: Write the failing test**

Create `tests/test_captions.py`:

```python
from shorts_generator.captions import _chunk_segments


def test_chunk_segments_splits_by_word_count_and_time_share():
    segments = [
        {
            "start": 10.0,
            "end": 12.0,
            "text": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen",
        }
    ]

    chunks = _chunk_segments(segments, clip_start=0.0, clip_end=100.0, max_words=7)

    assert chunks == [
        {"start": 10.0, "end": 11.0, "text": "one two three four five six seven"},
        {"start": 11.0, "end": 12.0, "text": "eight nine ten eleven twelve thirteen fourteen"},
    ]


def test_chunk_segments_drops_segments_outside_window():
    segments = [{"start": 5.0, "end": 6.0, "text": "not in the clip"}]

    chunks = _chunk_segments(segments, clip_start=10.0, clip_end=20.0)

    assert chunks == []


def test_chunk_segments_clips_and_shifts_straddling_segment():
    segments = [{"start": 8.0, "end": 12.0, "text": "alpha beta gamma delta"}]

    chunks = _chunk_segments(segments, clip_start=10.0, clip_end=20.0, max_words=7)

    assert chunks == [{"start": 0.0, "end": 2.0, "text": "alpha beta gamma delta"}]
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_captions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shorts_generator.captions'`

- [ ] **Step 6: Write the minimal implementation**

Create `shorts_generator/captions.py`:

```python
"""Caption chunking, ASS authoring, and ffmpeg burn-in — shared by both modes.

Given the full transcript for a source video and one highlight's clip
window, this module slices the relevant transcript segments, splits them
into short phrase chunks, writes an ASS subtitle file with a fade-in
override tag per line, and burns it onto a local video file via ffmpeg's
`subtitles` (libass) filter.
"""
from typing import Dict, List


def _chunk_segments(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    max_words: int = 7,
) -> List[Dict]:
    """Slice full-video transcript segments to a clip window and split each
    into ~max_words-word chunks, timed proportionally to word count within
    the segment's own duration, then clipped to the clip window.

    Returns clip-relative chunks: [{"start": float, "end": float, "text": str}, ...]
    """
    chunks: List[Dict] = []
    for seg in segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        words = str(seg.get("text", "")).split()
        if not words:
            continue

        total_words = len(words)
        seg_duration = seg_end - seg_start
        word_groups = [words[i:i + max_words] for i in range(0, len(words), max_words)]

        cursor = seg_start
        for group in word_groups:
            share = len(group) / total_words
            duration = seg_duration * share
            start = cursor
            end = start + duration
            cursor = end

            clipped_start = max(start, clip_start)
            clipped_end = min(end, clip_end)
            if clipped_end <= clipped_start:
                continue

            chunks.append({
                "start": clipped_start - clip_start,
                "end": clipped_end - clip_start,
                "text": " ".join(group),
            })

    return chunks
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `pytest tests/test_captions.py -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Commit**

```bash
git add requirements-dev.txt tests/conftest.py tests/test_captions.py shorts_generator/captions.py
git commit -m "feat: add transcript chunking for caption burn-in"
```

---

## Task 2: `captions.py` — ASS subtitle authoring (`_write_ass`)

**Files:**
- Modify: `tests/test_captions.py`
- Modify: `shorts_generator/captions.py`

- [ ] **Step 1: Write the failing test**

Replace `tests/test_captions.py` with (adds two new tests at the bottom):

```python
from shorts_generator.captions import _chunk_segments, _write_ass


def test_chunk_segments_splits_by_word_count_and_time_share():
    segments = [
        {
            "start": 10.0,
            "end": 12.0,
            "text": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen",
        }
    ]

    chunks = _chunk_segments(segments, clip_start=0.0, clip_end=100.0, max_words=7)

    assert chunks == [
        {"start": 10.0, "end": 11.0, "text": "one two three four five six seven"},
        {"start": 11.0, "end": 12.0, "text": "eight nine ten eleven twelve thirteen fourteen"},
    ]


def test_chunk_segments_drops_segments_outside_window():
    segments = [{"start": 5.0, "end": 6.0, "text": "not in the clip"}]

    chunks = _chunk_segments(segments, clip_start=10.0, clip_end=20.0)

    assert chunks == []


def test_chunk_segments_clips_and_shifts_straddling_segment():
    segments = [{"start": 8.0, "end": 12.0, "text": "alpha beta gamma delta"}]

    chunks = _chunk_segments(segments, clip_start=10.0, clip_end=20.0, max_words=7)

    assert chunks == [{"start": 0.0, "end": 2.0, "text": "alpha beta gamma delta"}]


def test_write_ass_contains_resolution_and_fade_tag(tmp_path):
    chunks = [
        {"start": 0.0, "end": 1.0, "text": "hello world"},
        {"start": 1.0, "end": 2.5, "text": "second line here"},
    ]
    ass_path = str(tmp_path / "captions.ass")

    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3)

    content = open(ass_path, encoding="utf-8").read()
    assert "PlayResX: 608" in content
    assert "PlayResY: 1080" in content
    assert content.count("Dialogue:") == 2
    assert "\\fad(300,0)" in content
    assert "hello world" in content
    assert "second line here" in content


def test_write_ass_strips_braces_from_text(tmp_path):
    chunks = [{"start": 0.0, "end": 1.0, "text": "watch this {glitch} moment"}]
    ass_path = str(tmp_path / "captions.ass")

    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3)

    content = open(ass_path, encoding="utf-8").read()
    assert "watch this glitch moment" in content
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_captions.py -v`
Expected: FAIL — `ImportError: cannot import name '_write_ass' from 'shorts_generator.captions'`

- [ ] **Step 3: Write the minimal implementation**

Replace `shorts_generator/captions.py` with (adds `_format_ass_timestamp` and `_write_ass`):

```python
"""Caption chunking, ASS authoring, and ffmpeg burn-in — shared by both modes.

Given the full transcript for a source video and one highlight's clip
window, this module slices the relevant transcript segments, splits them
into short phrase chunks, writes an ASS subtitle file with a fade-in
override tag per line, and burns it onto a local video file via ffmpeg's
`subtitles` (libass) filter.
"""
from typing import Dict, List


def _chunk_segments(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    max_words: int = 7,
) -> List[Dict]:
    """Slice full-video transcript segments to a clip window and split each
    into ~max_words-word chunks, timed proportionally to word count within
    the segment's own duration, then clipped to the clip window.

    Returns clip-relative chunks: [{"start": float, "end": float, "text": str}, ...]
    """
    chunks: List[Dict] = []
    for seg in segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        words = str(seg.get("text", "")).split()
        if not words:
            continue

        total_words = len(words)
        seg_duration = seg_end - seg_start
        word_groups = [words[i:i + max_words] for i in range(0, len(words), max_words)]

        cursor = seg_start
        for group in word_groups:
            share = len(group) / total_words
            duration = seg_duration * share
            start = cursor
            end = start + duration
            cursor = end

            clipped_start = max(start, clip_start)
            clipped_end = min(end, clip_end)
            if clipped_end <= clipped_start:
                continue

            chunks.append({
                "start": clipped_start - clip_start,
                "end": clipped_end - clip_start,
                "text": " ".join(group),
            })

    return chunks


def _format_ass_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_cs = int(round(seconds * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _write_ass(chunks: List[Dict], ass_path: str, width: int, height: int, fade_seconds: float) -> None:
    """Write an ASS subtitle file: one bottom-center style, one Dialogue line
    per chunk, each carrying a fade-in-only \\fad override tag."""
    fontsize = max(12, round(height * 0.045))
    margin_v = max(10, round(height * 0.06))
    fade_ms = max(0, round(fade_seconds * 1000))

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,Arial,{fontsize},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        f"0,0,0,0,100,100,0,0,1,2,1,2,20,20,{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for chunk in chunks:
        text = chunk["text"].replace("{", "").replace("}", "").replace("\n", " ")
        start_ts = _format_ass_timestamp(chunk["start"])
        end_ts = _format_ass_timestamp(chunk["end"])
        lines.append(
            f"Dialogue: 0,{start_ts},{end_ts},Caption,,0,0,0,,{{\\fad({fade_ms},0)}}{text}\n"
        )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_captions.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_captions.py shorts_generator/captions.py
git commit -m "feat: write ASS subtitles with per-line fade-in tag"
```

---

## Task 3: `captions.py` — ffmpeg burn-in (`burn_captions`, `CaptionError`)

**Files:**
- Modify: `tests/test_captions.py`
- Modify: `shorts_generator/captions.py`

- [ ] **Step 1: Write the failing test**

Replace `tests/test_captions.py` with (adds `os`/`subprocess`/`pytest` imports, a
shared synthetic-clip fixture, and four new tests at the bottom):

```python
import os
import subprocess

import pytest

from shorts_generator.captions import CaptionError, _chunk_segments, _probe_resolution, _write_ass, burn_captions


def test_chunk_segments_splits_by_word_count_and_time_share():
    segments = [
        {
            "start": 10.0,
            "end": 12.0,
            "text": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen",
        }
    ]

    chunks = _chunk_segments(segments, clip_start=0.0, clip_end=100.0, max_words=7)

    assert chunks == [
        {"start": 10.0, "end": 11.0, "text": "one two three four five six seven"},
        {"start": 11.0, "end": 12.0, "text": "eight nine ten eleven twelve thirteen fourteen"},
    ]


def test_chunk_segments_drops_segments_outside_window():
    segments = [{"start": 5.0, "end": 6.0, "text": "not in the clip"}]

    chunks = _chunk_segments(segments, clip_start=10.0, clip_end=20.0)

    assert chunks == []


def test_chunk_segments_clips_and_shifts_straddling_segment():
    segments = [{"start": 8.0, "end": 12.0, "text": "alpha beta gamma delta"}]

    chunks = _chunk_segments(segments, clip_start=10.0, clip_end=20.0, max_words=7)

    assert chunks == [{"start": 0.0, "end": 2.0, "text": "alpha beta gamma delta"}]


def test_write_ass_contains_resolution_and_fade_tag(tmp_path):
    chunks = [
        {"start": 0.0, "end": 1.0, "text": "hello world"},
        {"start": 1.0, "end": 2.5, "text": "second line here"},
    ]
    ass_path = str(tmp_path / "captions.ass")

    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3)

    content = open(ass_path, encoding="utf-8").read()
    assert "PlayResX: 608" in content
    assert "PlayResY: 1080" in content
    assert content.count("Dialogue:") == 2
    assert "\\fad(300,0)" in content
    assert "hello world" in content
    assert "second line here" in content


def test_write_ass_strips_braces_from_text(tmp_path):
    chunks = [{"start": 0.0, "end": 1.0, "text": "watch this {glitch} moment"}]
    ass_path = str(tmp_path / "captions.ass")

    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3)

    content = open(ass_path, encoding="utf-8").read()
    assert "watch this glitch moment" in content


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory):
    """A tiny 3s 9:16-ish clip generated once for this test module."""
    tmp_dir = tmp_path_factory.mktemp("captions_src")
    path = str(tmp_dir / "clip.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x568:rate=24:duration=3",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            path,
        ],
        check=True,
    )
    return path


def test_probe_resolution_reads_dimensions(synthetic_clip):
    assert _probe_resolution(synthetic_clip) == (320, 568)


def test_burn_captions_produces_output_file(tmp_path, synthetic_clip):
    out_path = str(tmp_path / "burned.mp4")
    segments = [{"start": 0.0, "end": 3.0, "text": "hello there this is a caption test"}]

    result = burn_captions(
        synthetic_clip, segments, clip_start=0.0, clip_end=3.0, out_path=out_path, fade_seconds=0.3
    )

    assert result == out_path
    assert os.path.exists(out_path)
    assert _probe_resolution(out_path) == (320, 568)
    assert not os.path.exists(out_path + ".ass")


def test_burn_captions_raises_when_no_transcript_overlaps(tmp_path, synthetic_clip):
    out_path = str(tmp_path / "burned.mp4")
    segments = [{"start": 100.0, "end": 103.0, "text": "way outside the clip"}]

    with pytest.raises(CaptionError):
        burn_captions(synthetic_clip, segments, clip_start=0.0, clip_end=3.0, out_path=out_path)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_captions.py -v`
Expected: FAIL — `ImportError: cannot import name 'CaptionError' from 'shorts_generator.captions'`

- [ ] **Step 3: Write the minimal implementation**

Replace `shorts_generator/captions.py` with (adds `CaptionError`, `_probe_resolution`,
`burn_captions`, and the `os`/`subprocess`/`Tuple` imports):

```python
"""Caption chunking, ASS authoring, and ffmpeg burn-in — shared by both modes.

Given the full transcript for a source video and one highlight's clip
window, this module slices the relevant transcript segments, splits them
into short phrase chunks, writes an ASS subtitle file with a fade-in
override tag per line, and burns it onto a local video file via ffmpeg's
`subtitles` (libass) filter.
"""
import os
import subprocess
from typing import Dict, List, Tuple


class CaptionError(RuntimeError):
    """Raised when caption burn-in fails; callers should fall back to the plain clip."""


def _chunk_segments(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    max_words: int = 7,
) -> List[Dict]:
    """Slice full-video transcript segments to a clip window and split each
    into ~max_words-word chunks, timed proportionally to word count within
    the segment's own duration, then clipped to the clip window.

    Returns clip-relative chunks: [{"start": float, "end": float, "text": str}, ...]
    """
    chunks: List[Dict] = []
    for seg in segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        words = str(seg.get("text", "")).split()
        if not words:
            continue

        total_words = len(words)
        seg_duration = seg_end - seg_start
        word_groups = [words[i:i + max_words] for i in range(0, len(words), max_words)]

        cursor = seg_start
        for group in word_groups:
            share = len(group) / total_words
            duration = seg_duration * share
            start = cursor
            end = start + duration
            cursor = end

            clipped_start = max(start, clip_start)
            clipped_end = min(end, clip_end)
            if clipped_end <= clipped_start:
                continue

            chunks.append({
                "start": clipped_start - clip_start,
                "end": clipped_end - clip_start,
                "text": " ".join(group),
            })

    return chunks


def _format_ass_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_cs = int(round(seconds * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _write_ass(chunks: List[Dict], ass_path: str, width: int, height: int, fade_seconds: float) -> None:
    """Write an ASS subtitle file: one bottom-center style, one Dialogue line
    per chunk, each carrying a fade-in-only \\fad override tag."""
    fontsize = max(12, round(height * 0.045))
    margin_v = max(10, round(height * 0.06))
    fade_ms = max(0, round(fade_seconds * 1000))

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,Arial,{fontsize},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        f"0,0,0,0,100,100,0,0,1,2,1,2,20,20,{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for chunk in chunks:
        text = chunk["text"].replace("{", "").replace("}", "").replace("\n", " ")
        start_ts = _format_ass_timestamp(chunk["start"])
        end_ts = _format_ass_timestamp(chunk["end"])
        lines.append(
            f"Dialogue: 0,{start_ts},{end_ts},Caption,,0,0,0,,{{\\fad({fade_ms},0)}}{text}\n"
        )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _probe_resolution(video_path: str) -> Tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise CaptionError(f"ffprobe failed on {video_path}: {e}") from e

    try:
        width_str, height_str = result.stdout.strip().split("x")
        return int(width_str), int(height_str)
    except ValueError as e:
        raise CaptionError(f"could not parse ffprobe output for {video_path}: {result.stdout!r}") from e


def burn_captions(
    video_path: str,
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    out_path: str,
    fade_seconds: float = 0.3,
) -> str:
    """Burn phrase-chunked, fade-in captions onto a local clip.

    Raises CaptionError on any failure; the caller decides whether to fall
    back to the uncaptioned clip.
    """
    chunks = _chunk_segments(segments, clip_start, clip_end, max_words=7)
    if not chunks:
        raise CaptionError(f"no transcript overlaps clip window [{clip_start}, {clip_end}]")

    width, height = _probe_resolution(video_path)

    ass_path = out_path + ".ass"
    _write_ass(chunks, ass_path, width, height, fade_seconds)

    try:
        escaped_ass_path = ass_path.replace("\\", "/").replace(":", "\\:")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"subtitles={escaped_ass_path}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "copy",
            out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise CaptionError(f"ffmpeg subtitles burn-in failed: {e.stderr}") from e
    finally:
        os.remove(ass_path)

    return out_path
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_captions.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_captions.py shorts_generator/captions.py
git commit -m "feat: burn ASS captions onto clips via ffmpeg subtitles filter"
```

---

## Task 4: CLI flags (`main.py`)

**Files:**
- Create: `tests/test_main.py`
- Modify: `main.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_main.py`:

```python
from main import build_parser


def test_captions_on_by_default():
    args = build_parser().parse_args(["https://example.com/video"])
    assert args.captions is True
    assert args.caption_fade_duration == 0.3


def test_no_captions_flag_disables_captions():
    args = build_parser().parse_args(["https://example.com/video", "--no-captions"])
    assert args.captions is False


def test_caption_fade_duration_flag_overrides_default():
    args = build_parser().parse_args(
        ["https://example.com/video", "--caption-fade-duration", "0.5"]
    )
    assert args.caption_fade_duration == 0.5
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_parser' from 'main'`

- [ ] **Step 3: Write the minimal implementation**

Replace `main.py` with:

```python
"""CLI entry point.

Usage:
    python main.py "https://www.youtube.com/watch?v=..." \
        --num-clips 3 --aspect-ratio 9:16
"""
import argparse
import json
import sys

# Windows uses 'charmap' by default, which can't encode Unicode characters
# like →. Reconfigure stdout/stderr to UTF-8 so output works on all platforms.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from shorts_generator import generate_shorts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI YouTube Shorts Generator")
    parser.add_argument("url", help="YouTube URL, file:// URL, or local file path")
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="api (default, MuAPI) or local (remote URL, file://, or local path + faster-whisper + LLM provider + ffmpeg).",
    )
    parser.add_argument("--num-clips", type=int, default=3, help="How many shorts to render (default: 3)")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--format", default="720", help="Source download resolution: 360 / 480 / 720 / 1080 (default: 720)")
    parser.add_argument("--language", default=None, help="Force Whisper language code, e.g. 'en' (default: auto-detect)")
    parser.add_argument("--output-json", default=None, help="Write the full result JSON to this path")
    parser.add_argument(
        "--no-captions",
        dest="captions",
        action="store_false",
        default=True,
        help="Disable fade-in caption burn-in (captions are on by default in both modes).",
    )
    parser.add_argument(
        "--caption-fade-duration",
        type=float,
        default=0.3,
        help="Caption fade-in duration in seconds (default: 0.3)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        result = generate_shorts(
            youtube_url=args.url,
            num_clips=args.num_clips,
            aspect_ratio=args.aspect_ratio,
            download_format=args.format,
            language=args.language,
            mode=args.mode,
            captions=args.captions,
            caption_fade_duration=args.caption_fade_duration,
        )
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    print(f"Mode:          {result.get('mode', args.mode)}")
    print(f"Source video:  {result['source_video_url']}")
    print(f"Highlights:    {len(result['highlights'])} candidates → kept top {len(result['shorts'])}")
    print("=" * 72)
    for i, s in enumerate(result["shorts"], 1):
        print(f"\n#{i}  score={s.get('score')}  {s.get('start_time'):.1f}s → {s.get('end_time'):.1f}s")
        print(f"     title:  {s.get('title')}")
        print(f"     hook:   {s.get('hook_sentence')}")
        if s.get("clip_url"):
            print(f"     clip:   {s['clip_url']}")
        else:
            print(f"     clip:   FAILED ({s.get('error')})")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_main.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_main.py main.py
git commit -m "feat: add --no-captions and --caption-fade-duration CLI flags"
```

---

## Task 5: Thread captions through `pipeline.py`

**Files:**
- Create: `tests/test_pipeline.py`
- Modify: `shorts_generator/pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline.py`:

```python
from unittest.mock import Mock

import shorts_generator.local.clipper as local_clipper_module
import shorts_generator.local.downloader as local_downloader_module
import shorts_generator.local.transcriber as local_transcriber_module
import shorts_generator.pipeline as pipeline_module


def _fake_transcript():
    return {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi there"}]}


def _fake_highlights_result():
    return {"highlights": [{"start_time": 0.0, "end_time": 3.0, "score": 90, "title": "Clip"}]}


def test_run_local_threads_captions_params(monkeypatch):
    monkeypatch.setattr(local_downloader_module, "download_youtube_local", lambda url, fmt: "/tmp/source.mp4")
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())

    crop_mock = Mock(return_value=[{"clip_url": "/tmp/out/short_01.mp4"}])
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", crop_mock)

    result = pipeline_module._run_local(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.7,
    )

    assert result["mode"] == "local"
    assert result["shorts"] == [{"clip_url": "/tmp/out/short_01.mp4"}]

    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is False
    assert kwargs["caption_fade_duration"] == 0.7
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]


def test_run_api_threads_captions_params(monkeypatch):
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())

    crop_mock = Mock(return_value=[{"clip_url": "https://hosted.example/short_1.mp4"}])
    monkeypatch.setattr(pipeline_module, "crop_highlights", crop_mock)

    result = pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
    )

    assert result["mode"] == "api"
    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is True
    assert kwargs["caption_fade_duration"] == 0.3
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]
```

Note the two different monkeypatch targets: `_run_local` imports `download_youtube_local`,
`transcribe_local`, and `crop_highlights_local` **inside the function body**, so those must
be patched on their *defining* modules (`shorts_generator.local.downloader`,
`shorts_generator.local.transcriber`, `shorts_generator.local.clipper`). `_run_api` uses
top-level imports in `pipeline.py`, so those are patched on `pipeline_module` directly.
`get_highlights` is top-level-imported in `pipeline.py` for both paths, so it's always
patched on `pipeline_module`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — `TypeError: _run_local() got an unexpected keyword argument 'captions'`

- [ ] **Step 3: Write the minimal implementation**

Replace `shorts_generator/pipeline.py` with:

```python
"""End-to-end orchestrator.

Two modes:
  * mode="api"   (default) — MuAPI does download / transcribe / LLM / autocrop.
                              Fast, no local deps, pay-per-call.
  * mode="local"            — yt-dlp + faster-whisper + OpenAI or Gemini + ffmpeg/opencv.
                              Self-hosted, LLM_PROVIDER selects OpenAI or Gemini.

Both modes burn fade-in captions onto the final clips by default (see
shorts_generator.captions); pass captions=False to disable.
"""
from typing import Dict, List, Optional

from .clipper import crop_highlights
from .downloader import download_youtube
from .highlights import call_muapi_llm, get_highlights
from .transcriber import transcribe


def _run_local(
    youtube_url: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    captions: bool,
    caption_fade_duration: float,
) -> Dict:
    from .local.clipper import crop_highlights_local
    from .local.downloader import download_youtube_local
    from .local.llm import call_local_llm
    from .local.transcriber import transcribe_local

    source_path = download_youtube_local(youtube_url, fmt=download_format)

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
        transcript_segments=transcript["segments"],
        captions=captions,
        caption_fade_duration=caption_fade_duration,
    )

    return {
        "mode": "local",
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
) -> Dict:
    source_url = download_youtube(youtube_url, fmt=download_format)

    transcript = transcribe(source_url, language=language)
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
    )

    return {
        "mode": "api",
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

    Returns:
        {
          "mode": "api" | "local",
          "source_video_url": str,   # hosted URL (api) or local path (local)
          "transcript": {...},
          "highlights": [...],       # all candidates ranked
          "shorts": [...],           # top `num_clips` with clip_url / local path
        }
    """
    mode = (mode or "api").lower()
    if mode == "local":
        return _run_local(
            youtube_url, num_clips, aspect_ratio, download_format, language, captions, caption_fade_duration
        )
    if mode == "api":
        return _run_api(
            youtube_url, num_clips, aspect_ratio, download_format, language, captions, caption_fade_duration
        )
    raise ValueError(f"Unknown mode: {mode!r}. Use 'api' or 'local'.")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline.py shorts_generator/pipeline.py
git commit -m "feat: thread caption options through the pipeline"
```

---

## Task 6: Local mode integration (`shorts_generator/local/clipper.py`)

**Files:**
- Create: `tests/test_local_clipper.py`
- Modify: `shorts_generator/local/clipper.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_local_clipper.py`:

```python
import os
import subprocess

import pytest

from shorts_generator import captions as captions_module
from shorts_generator.local.clipper import crop_highlights_local


@pytest.fixture(scope="module")
def synthetic_source(tmp_path_factory):
    """A tiny 6s clip with video + audio, generated once for this module."""
    tmp_dir = tmp_path_factory.mktemp("source")
    path = str(tmp_dir / "source.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=24:duration=6",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            path,
        ],
        check=True,
    )
    return path


def _highlight():
    return {"title": "Test Clip", "start_time": 1.0, "end_time": 4.0, "score": 90}


def _segments():
    return [
        {"start": 0.5, "end": 2.5, "text": "hello there this is a test caption"},
        {"start": 2.5, "end": 4.5, "text": "and here is a second phrase for good measure"},
    ]


def test_captions_burned_in_by_default(tmp_path, synthetic_source):
    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source,
        [_highlight()],
        aspect_ratio="9:16",
        out_dir=out_dir,
        transcript_segments=_segments(),
    )

    assert len(results) == 1
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])
    assert "captions_error" not in results[0]


def test_captions_disabled_skips_burn_in(tmp_path, synthetic_source, monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("burn_captions should not be called when captions=False")

    monkeypatch.setattr("shorts_generator.local.clipper.burn_captions", _fail_if_called)

    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source,
        [_highlight()],
        aspect_ratio="9:16",
        out_dir=out_dir,
        transcript_segments=_segments(),
        captions=False,
    )

    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])


def test_caption_failure_falls_back_to_plain_clip(tmp_path, synthetic_source, monkeypatch):
    def _raise(*args, **kwargs):
        raise captions_module.CaptionError("boom")

    monkeypatch.setattr("shorts_generator.local.clipper.burn_captions", _raise)

    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source,
        [_highlight()],
        aspect_ratio="9:16",
        out_dir=out_dir,
        transcript_segments=_segments(),
    )

    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])
    assert results[0]["captions_error"] == "boom"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_local_clipper.py -v`
Expected: FAIL — `TypeError: crop_highlights_local() got an unexpected keyword argument 'transcript_segments'`

- [ ] **Step 3: Write the minimal implementation**

Modify `shorts_generator/local/clipper.py`. First, update the import block near the
top (currently `from ..config import LOCAL_OUTPUT_DIR`):

```python
from ..captions import CaptionError, burn_captions
from ..config import LOCAL_OUTPUT_DIR
```

Then replace the `crop_highlights_local` function (currently the last function in the
file) with:

```python
def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    transcript_segments: Optional[List[Dict]] = None,
    captions: bool = True,
    caption_fade_duration: float = 0.3,
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
            )
            entry = {**h, "clip_url": out_path}

            if captions and transcript_segments:
                captioned_path = out_path + ".captioned.mp4"
                try:
                    burn_captions(
                        out_path,
                        transcript_segments,
                        float(h["start_time"]),
                        float(h["end_time"]),
                        captioned_path,
                        fade_seconds=caption_fade_duration,
                    )
                    os.replace(captioned_path, out_path)
                except CaptionError as e:
                    print(f"[clip/local] {i} captions skipped: {e}", flush=True)
                    entry["captions_error"] = str(e)
                    if os.path.exists(captioned_path):
                        os.remove(captioned_path)

            results.append(entry)
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_local_clipper.py -v`
Expected: PASS (3 tests) — this one takes a few real seconds since it runs actual
ffmpeg cut/reframe/burn-in and OpenCV face detection on the synthetic clip.

- [ ] **Step 5: Commit**

```bash
git add tests/test_local_clipper.py shorts_generator/local/clipper.py
git commit -m "feat: burn fade-in captions onto local-mode clips"
```

---

## Task 7: API mode integration (`shorts_generator/clipper.py`)

**Files:**
- Create: `tests/test_clipper_api.py`
- Modify: `shorts_generator/clipper.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_clipper_api.py`:

```python
import os
import shutil
import subprocess

import pytest
import requests

from shorts_generator import clipper


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory):
    """Stands in for the mp4 MuAPI would host at the returned clip URL."""
    tmp_dir = tmp_path_factory.mktemp("hosted")
    path = str(tmp_dir / "hosted.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=608x1080:rate=24:duration=4",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            path,
        ],
        check=True,
    )
    return path


def _highlight():
    return {"title": "Test Clip", "start_time": 0.0, "end_time": 3.0, "score": 90}


def _segments():
    return [{"start": 0.0, "end": 3.0, "text": "hello there this is a caption test line"}]


def test_captions_burned_in_by_default(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    out_dir = str(tmp_path / "out")
    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=out_dir,
    )

    assert results[0]["hosted_clip_url"] == "https://hosted.example/short_1.mp4"
    assert os.path.exists(results[0]["clip_url"])
    assert results[0]["clip_url"] != results[0]["hosted_clip_url"]


def test_captions_disabled_keeps_hosted_url(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")

    def _fail_if_called(*a, **k):
        raise AssertionError("_download_to should not be called when captions=False")

    monkeypatch.setattr(clipper, "_download_to", _fail_if_called)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        captions=False,
        out_dir=str(tmp_path / "out"),
    )

    assert results[0]["clip_url"] == "https://hosted.example/short_1.mp4"
    assert "hosted_clip_url" not in results[0]


def test_download_failure_falls_back_to_hosted_url(tmp_path, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")

    def _raise(*a, **k):
        raise requests.RequestException("network down")

    monkeypatch.setattr(clipper, "_download_to", _raise)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
    )

    assert results[0]["clip_url"] == "https://hosted.example/short_1.mp4"
    assert results[0]["captions_error"] == "network down"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_clipper_api.py -v`
Expected: FAIL — `TypeError: crop_highlights() got an unexpected keyword argument 'transcript_segments'`

- [ ] **Step 3: Write the minimal implementation**

Replace `shorts_generator/clipper.py` with:

```python
"""Per-clip cropping via MuAPI /autocrop, with optional local caption burn-in.

Given the source video URL plus a highlight's start/end and a target aspect
ratio, MuAPI returns a vertically-cropped short ready for posting. When
captions are enabled (the default), that hosted clip is downloaded locally
and burned with fade-in captions via ffmpeg (shorts_generator.captions) —
the one place API mode now needs a local ffmpeg on PATH.
"""
import os
from typing import Dict, List, Optional

import requests

from . import muapi
from .captions import CaptionError, burn_captions
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

            if captions and transcript_segments:
                os.makedirs(out_dir, exist_ok=True)
                final_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
                downloaded_path = final_path + ".download.mp4"
                try:
                    _download_to(url, downloaded_path)
                    burn_captions(
                        downloaded_path,
                        transcript_segments,
                        float(h["start_time"]),
                        float(h["end_time"]),
                        final_path,
                        fade_seconds=caption_fade_duration,
                    )
                    entry["clip_url"] = final_path
                    entry["hosted_clip_url"] = url
                except (CaptionError, requests.RequestException) as e:
                    print(f"[clip] {i} captions skipped: {e}", flush=True)
                    entry["captions_error"] = str(e)
                finally:
                    if os.path.exists(downloaded_path):
                        os.remove(downloaded_path)

            out.append(entry)
        except Exception as e:
            print(f"[clip] {i} failed: {e}", flush=True)
            out.append({**h, "clip_url": None, "error": str(e)})
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_clipper_api.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_clipper_api.py shorts_generator/clipper.py
git commit -m "feat: burn fade-in captions onto api-mode clips"
```

---

## Task 8: Documentation (`README.md`)

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update Prerequisites**

Find this block under `### Prerequisites`:

```
- Python 3.10+
- For **API mode (default)**: a MuAPI key — powers download, transcription, highlight ranking, and clipping in a single dependency
- For **Local mode** (`--mode local`): `ffmpeg` on your PATH and an LLM API key (`OPENAI_API_KEY` or `GEMINI_API_KEY`; only the LLM step is remote)
```

Replace with:

```
- Python 3.10+
- `ffmpeg` on your PATH — required in **both** modes for caption burn-in (captions are on by default; pass `--no-captions` to skip and, in API mode only, avoid needing ffmpeg at all)
- For **API mode (default)**: a MuAPI key — powers download, transcription, highlight ranking, and clipping in a single dependency
- For **Local mode** (`--mode local`): an LLM API key (`OPENAI_API_KEY` or `GEMINI_API_KEY`; only the LLM step is remote)
```

- [ ] **Step 2: Update the CLI flags table**

Find this table under `### CLI flags`:

```
| Flag | Default | Notes |
|------|---------|-------|
| `--mode` | `api` | `api` (MuAPI, fast, no setup) or `local` (remote URL, `file://`, or local path + faster-whisper + LLM provider + ffmpeg) |
| `--num-clips` | `3` | How many shorts to render |
| `--aspect-ratio` | `9:16` | Any ratio; `9:16` for TikTok/Reels, `1:1` for square |
| `--format` | `720` | Source download resolution: `360` / `480` / `720` / `1080` |
| `--language` | auto | Force Whisper language code (e.g. `en`) |
| `--output-json` | — | Dump the full result (transcript + all candidates) to a file |
```

Replace with:

```
| Flag | Default | Notes |
|------|---------|-------|
| `--mode` | `api` | `api` (MuAPI, fast, no setup) or `local` (remote URL, `file://`, or local path + faster-whisper + LLM provider + ffmpeg) |
| `--num-clips` | `3` | How many shorts to render |
| `--aspect-ratio` | `9:16` | Any ratio; `9:16` for TikTok/Reels, `1:1` for square |
| `--format` | `720` | Source download resolution: `360` / `480` / `720` / `1080` |
| `--language` | auto | Force Whisper language code (e.g. `en`) |
| `--output-json` | — | Dump the full result (transcript + all candidates) to a file |
| `--no-captions` | captions on | Disable fade-in caption burn-in (on by default in both modes) |
| `--caption-fade-duration` | `0.3` | Caption fade-in duration in seconds |
```

- [ ] **Step 3: Update the API mode vs Local mode table**

Find this row under `### API mode vs Local mode`:

```
| Required keys | `MUAPI_API_KEY` | `OPENAI_API_KEY` or `GEMINI_API_KEY` (+ `ffmpeg` on PATH) |
```

Replace with:

```
| Required keys | `MUAPI_API_KEY` (+ `ffmpeg` on PATH for caption burn-in) | `OPENAI_API_KEY` or `GEMINI_API_KEY` (+ `ffmpeg` on PATH) |
```

- [ ] **Step 4: Update Project Structure**

Find this line under `## Project Structure`:

```
    ├── clipper.py                 API mode: MuAPI /autocrop
```

Replace with:

```
    ├── clipper.py                 API mode: MuAPI /autocrop (+ local caption burn-in)
    ├── captions.py                shared: phrase-chunked fade-in caption burn-in (ffmpeg/libass)
```

- [ ] **Step 5: Add a short Development section**

At the end of the file, right before `## Related Projects`, add a `## Development`
section containing this fenced `bash` block:

~~~
## Development

Install dev dependencies and run the test suite:

```bash
pip install -r requirements-dev.txt
pytest
```
~~~

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: document caption flags and ffmpeg requirement for api mode"
```

---

## Task 9: Full suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `source .venv/bin/activate && pytest -v`
Expected: PASS — all tests across `test_captions.py`, `test_main.py`,
`test_pipeline.py`, `test_local_clipper.py`, and `test_clipper_api.py` green.

- [ ] **Step 2: Manual end-to-end check (deferred note, not a blocking step)**

A real `python main.py "<youtube-url>"` run exercises the LLM highlight-ranking
step, which needs a working `GEMINI_API_KEY` (or `OPENAI_API_KEY`) in `.env` —
not yet provided as of this plan. Once that key is pasted in, run:

```bash
python main.py "<a-real-youtube-url>" --mode local --num-clips 1
```

and open the resulting `output/short_01.mp4` to eyeball the caption fade-in.
This is a follow-up manual check, not a plan step to execute now.

- [ ] **Step 3: No code changes expected in this task** — if the full-suite run
surfaces any regressions from earlier tasks, fix them in the relevant task's
files and re-run before considering the plan complete. If everything is
already green, there is nothing to commit here.
