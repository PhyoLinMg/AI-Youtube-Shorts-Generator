# Long-form Podcast Chapter Cuts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add a second, parallel output type — chapter cuts — that pulls fewer, longer (up to 15min), full-context, landscape (no-crop) segments out of a long-form podcast, captioned near the bottom edge. Local mode only.

**Architecture:** new chapter-selection rubric in `highlights.py` (parallel to the existing viral-highlight one, reusing chunking/caching plumbing), a trim-only clip renderer in `local/clipper.py` (reuses `_cut_subclip`, skips crop/hook-card/excision entirely), a configurable caption bottom-margin in `captions.py`, new orchestration in `pipeline.py` (`generate_chapters`), new output-path helpers in `run_output.py`, and a `--clip-type` flag in `main.py`.

**Tech Stack:** Python (`shorts_generator/highlights.py`, `shorts_generator/captions.py`, `shorts_generator/local/clipper.py`, `shorts_generator/pipeline.py`, `shorts_generator/run_output.py`, `main.py`), pytest, ffmpeg/ffprobe.

**Spec:** `docs/superpowers/specs/2026-08-07-long-form-chapter-cuts-design.md`

---

### Task 1: `RunPaths` gains `chapters_dir` + `chapters_json`

**Files:**
- Modify: `shorts_generator/run_output.py`
- Test: `tests/test_run_output.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_run_output.py`, right after `test_resolve_output_dir_builds_expected_tree` (around line 126):

```python
def test_resolve_output_dir_builds_chapters_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_output.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"title": "How To Build A Startup"}),
    )

    paths = run_output.resolve_output_dir(
        "https://www.youtube.com/watch?v=abc123", base_dir=str(tmp_path)
    )

    assert paths.chapters_dir == os.path.join(paths.root, "Chapters")
    assert paths.chapters_json == os.path.join(paths.root, "chapters.json")
    assert os.path.isdir(paths.chapters_dir)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_output.py -k chapters_paths -v`
Expected: FAIL with `AttributeError: 'RunPaths' object has no attribute 'chapters_dir'`

- [ ] **Step 3: Add the fields and wire them into `resolve_output_dir`**

In `shorts_generator/run_output.py`, modify the `RunPaths` dataclass (lines 29-37):

```python
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
    progress_log: str
```

Then modify `resolve_output_dir` (lines 149-164):

```python
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
        progress_log=os.path.join(root, "progress.log"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_output.py -k chapters_paths -v`
Expected: PASS

- [ ] **Step 5: Update the other `RunPaths(...)` call sites that will now break**

`tests/test_pipeline.py`'s `_paths()` helper (lines 31-43) constructs `RunPaths` positionally-by-keyword and must be updated or every existing pipeline test will fail with a missing-argument `TypeError`. Modify it:

```python
def _paths(tmp_path):
    root = str(tmp_path / "Video_Title")
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
        progress_log=os.path.join(root, "progress.log"),
    )
```

- [ ] **Step 6: Run the full test suite to check for other breakage**

`RunPaths(...)` is constructed in exactly two places in this codebase: `resolve_output_dir` (fixed in Step 3) and this test helper (fixed in Step 5) — confirmed via `grep -rn "RunPaths(" shorts_generator/ tests/`. No other call site needs updating.

Run: `python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py tests/test_pipeline.py
git commit -m "feat: add chapters_dir/chapters_json to RunPaths"
```

---

### Task 2: `unique_chapter_filename`

**Files:**
- Modify: `shorts_generator/run_output.py`
- Test: `tests/test_run_output.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_run_output.py`, right after `test_unique_short_filename_generic_style_requires_index` (around line 51):

```python
def test_unique_chapter_filename_numbers_and_slugifies():
    used = set()
    assert run_output.unique_chapter_filename("The Big Reveal", 1, used) == "01_The_Big_Reveal.mp4"
    assert run_output.unique_chapter_filename("A Second Topic", 2, used) == "02_A_Second_Topic.mp4"


def test_unique_chapter_filename_pads_double_digit_index():
    used = set()
    assert run_output.unique_chapter_filename("Topic Ten", 10, used) == "10_Topic_Ten.mp4"


def test_unique_chapter_filename_dedupes_collisions():
    used = set()
    first = run_output.unique_chapter_filename("Same Title", 1, used)
    second = run_output.unique_chapter_filename("Same Title", 1, used)
    assert [first, second] == ["01_Same_Title.mp4", "01_Same_Title_2.mp4"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_output.py -k unique_chapter_filename -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.run_output' has no attribute 'unique_chapter_filename'`

- [ ] **Step 3: Implement it**

In `shorts_generator/run_output.py`, add right after `unique_short_filename` (after line 74):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_output.py -k unique_chapter_filename -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py
git commit -m "feat: add unique_chapter_filename for ordered chapter output"
```

---

### Task 3: `write_chapter_descriptions`

**Files:**
- Modify: `shorts_generator/run_output.py`
- Test: `tests/test_run_output.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_run_output.py`, right after `test_write_descriptions_does_not_duplicate_hashtags_already_in_description` (around line 249):

```python
def test_write_chapter_descriptions_formats_one_block_per_chapter(tmp_path):
    chapters = [
        {
            "clip_url": "01_Topic_One.mp4", "title": "Topic One",
            "start_time": 12.5, "end_time": 340.0,
            "summary": "They discuss the origin of the idea and where it went wrong.",
        },
        {
            "clip_url": "02_Topic_Two.mp4", "title": "Topic Two",
            "start_time": 340.0, "end_time": 610.25,
            "summary": "A concrete example of the technique in practice.",
        },
    ]
    path = run_output.write_chapter_descriptions(str(tmp_path), chapters)
    content = Path(path).read_text()
    assert content == (
        "chapter 01 - Topic One (12.5s - 340.0s)\n"
        "They discuss the origin of the idea and where it went wrong.\n\n"
        "chapter 02 - Topic Two (340.0s - 610.2s)\n"
        "A concrete example of the technique in practice.\n"
    )


def test_write_chapter_descriptions_skips_failed_clips_without_renumbering(tmp_path):
    chapters = [
        {"clip_url": None, "title": "Failed", "error": "boom"},
        {
            "clip_url": "02_Survivor.mp4", "title": "Survivor",
            "start_time": 0.0, "end_time": 60.0, "summary": "It made it through.",
        },
    ]
    path = run_output.write_chapter_descriptions(str(tmp_path), chapters)
    content = Path(path).read_text()
    assert content == "chapter 02 - Survivor (0.0s - 60.0s)\nIt made it through.\n"


def test_write_chapter_descriptions_empty_list_writes_empty_file(tmp_path):
    path = run_output.write_chapter_descriptions(str(tmp_path), [])
    assert Path(path).read_text() == ""


def test_write_chapter_descriptions_falls_back_on_missing_fields(tmp_path):
    chapters = [{"clip_url": "01_X.mp4"}]
    path = run_output.write_chapter_descriptions(str(tmp_path), chapters)
    content = Path(path).read_text()
    assert content == "chapter 01 - Untitled Chapter (0.0s - 0.0s)\n\n"
```

Note: `Path` and `pytest` are already imported lower in this file (around lines 128-130); no new imports needed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_output.py -k write_chapter_descriptions -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.run_output' has no attribute 'write_chapter_descriptions'`

- [ ] **Step 3: Implement it**

In `shorts_generator/run_output.py`, add right after `write_descriptions` (after line 270):

```python
def write_chapter_descriptions(chapters_dir: str, chapters: List[Dict]) -> str:
    """Write a copy-paste-ready chapters_description.txt next to the chapter
    clip files. One block per chapter that actually has a clip_url, numbered
    by position in `chapters` regardless of the clip's own filename. Each
    block carries the ORIGINAL video's timestamp range as a reference (each
    chapter is its own file, not a marker in one long video, but the range
    is still useful context) plus the full `summary` -- unlike Shorts'
    write_descriptions, there's no yt_title/hashtags/hook_strength here,
    those fields don't exist on the chapter shape.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_output.py -k write_chapter_descriptions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py
git commit -m "feat: add write_chapter_descriptions"
```

---

### Task 4: `captions.py` — configurable bottom margin on `_write_ass`

**Files:**
- Modify: `shorts_generator/captions.py`
- Test: `tests/test_captions.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_captions.py`, right after `test_write_ass_uses_montserrat_black_font` (around line 229):

```python
def test_write_ass_default_margin_v_is_30_percent_of_height(tmp_path):
    chunks = [{"start": 0.0, "end": 1.0, "text": "hello world"}]
    ass_path = str(tmp_path / "c.ass")

    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3)

    content = open(ass_path, encoding="utf-8").read()
    style_line = next(l for l in content.splitlines() if l.startswith("Style:"))
    margin_v = int(style_line.split(",")[-2])
    assert margin_v == round(1080 * 0.30)


def test_write_ass_custom_bottom_margin_frac_changes_margin_v(tmp_path):
    chunks = [{"start": 0.0, "end": 1.0, "text": "hello world"}]
    ass_path = str(tmp_path / "c.ass")

    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3, bottom_margin_frac=0.06)

    content = open(ass_path, encoding="utf-8").read()
    style_line = next(l for l in content.splitlines() if l.startswith("Style:"))
    margin_v = int(style_line.split(",")[-2])
    assert margin_v == round(1080 * 0.06)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_captions.py -k bottom_margin -v`
Expected: first test PASSES already (0.30 is today's hardcoded behavior); second test FAILS with `TypeError: _write_ass() got an unexpected keyword argument 'bottom_margin_frac'`

- [ ] **Step 3: Add the parameter**

In `shorts_generator/captions.py`, modify `_write_ass`'s signature and margin line (lines 230-248):

```python
def _write_ass(
    chunks: List[Dict],
    ass_path: str,
    width: int,
    height: int,
    fade_seconds: float,
    word_highlight: bool = True,
    bottom_margin_frac: float = 0.30,
) -> None:
    """Write an ASS subtitle file: one bottom-center style.

    When `word_highlight` is True and a chunk carries a `"words"` list, one
    Dialogue line is emitted per word, with the active word wrapped in a
    color+bold+bounce override; only the chunk's first word carries the
    fade-in \\fad tag. Chunks without `"words"` (or when `word_highlight` is
    False) fall back to one plain Dialogue line per chunk with a fade-in-only
    \\fad override tag.

    `bottom_margin_frac` controls how far up from the bottom edge the
    caption sits, as a fraction of frame height. Default 0.30 is tuned for
    9:16 Shorts to clear the platform's reply/like UI column; landscape
    content with no such UI to dodge should pass a much smaller value.
    """
    fontsize = max(12, round(height * 0.045))
    margin_v = max(10, round(height * bottom_margin_frac))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_captions.py -k bottom_margin -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/captions.py tests/test_captions.py
git commit -m "feat: make caption bottom margin configurable in _write_ass"
```

---

### Task 5: thread `bottom_margin_frac` through `_burn_chunks`/`burn_captions`/`burn_captions_segments`

**Files:**
- Modify: `shorts_generator/captions.py`
- Test: `tests/test_captions.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_captions.py`, right after `test_burn_captions_produces_output_file` (around line 309):

```python
def test_burn_captions_forwards_bottom_margin_frac(tmp_path, synthetic_clip, monkeypatch):
    captured = {}
    real_write_ass = _write_ass

    def _spy_write_ass(*args, **kwargs):
        captured.update(kwargs)
        return real_write_ass(*args, **kwargs)

    monkeypatch.setattr("shorts_generator.captions._write_ass", _spy_write_ass)

    out_path = str(tmp_path / "burned.mp4")
    segments = [{"start": 0.0, "end": 3.0, "text": "hello there this is a caption test"}]

    burn_captions(
        synthetic_clip, segments, clip_start=0.0, clip_end=3.0, out_path=out_path,
        fade_seconds=0.3, bottom_margin_frac=0.06,
    )

    assert captured["bottom_margin_frac"] == 0.06
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_captions.py -k forwards_bottom_margin -v`
Expected: FAIL with `TypeError: burn_captions() got an unexpected keyword argument 'bottom_margin_frac'`

- [ ] **Step 3: Thread the parameter through the three functions**

In `shorts_generator/captions.py`, modify `_burn_chunks` (lines 338-365):

```python
def _burn_chunks(
    video_path: str, chunks: List[Dict], out_path: str, fade_seconds: float, word_highlight: bool,
    bottom_margin_frac: float = 0.30,
) -> str:
    width, height = _probe_resolution(video_path)

    ass_path = out_path + ".ass"
    _write_ass(chunks, ass_path, width, height, fade_seconds, word_highlight=word_highlight, bottom_margin_frac=bottom_margin_frac)
```

(rest of the function body is unchanged)

Modify `burn_captions` (lines 368-385):

```python
def burn_captions(
    video_path: str,
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    out_path: str,
    fade_seconds: float = 0.3,
    word_highlight: bool = True,
    bottom_margin_frac: float = 0.30,
) -> str:
    """Burn phrase-chunked, fade-in captions onto a local clip.

    `bottom_margin_frac` (default 0.30, tuned for 9:16 Shorts) is forwarded
    to `_write_ass` -- see its docstring.

    Raises CaptionError on any failure; the caller decides whether to fall
    back to the uncaptioned clip.
    """
    chunks = _chunk_segments(segments, clip_start, clip_end, max_words=7)
    if not chunks:
        raise CaptionError(f"no transcript overlaps clip window [{clip_start}, {clip_end}]")
    return _burn_chunks(video_path, chunks, out_path, fade_seconds, word_highlight, bottom_margin_frac=bottom_margin_frac)
```

Modify `burn_captions_segments` (lines 388-407):

```python
def burn_captions_segments(
    video_path: str,
    transcript_segments: List[Dict],
    cut_segments: List[Dict],
    out_path: str,
    fade_seconds: float = 0.3,
    word_highlight: bool = True,
    bottom_margin_frac: float = 0.30,
) -> str:
    """Like burn_captions, but for a video already excised down to
    `cut_segments` (see jump_cuts.excise_cut_segments) — captions are
    chunked per kept span so none straddle a cut, then placed on the
    concatenated timeline.

    Raises CaptionError on any failure; the caller decides whether to fall
    back to the uncaptioned clip.
    """
    chunks = _chunk_cut_segments(transcript_segments, cut_segments, max_words=7)
    if not chunks:
        raise CaptionError(f"no transcript overlaps cut_segments {cut_segments}")
    return _burn_chunks(video_path, chunks, out_path, fade_seconds, word_highlight, bottom_margin_frac=bottom_margin_frac)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_captions.py -k forwards_bottom_margin -v`
Expected: PASS

- [ ] **Step 5: Run the full captions test file to confirm no regression**

Run: `python -m pytest tests/test_captions.py -v`
Expected: PASS (every existing call site omits `bottom_margin_frac`, so default `0.30` keeps prior behavior byte-identical)

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/captions.py tests/test_captions.py
git commit -m "feat: thread bottom_margin_frac through burn_captions/burn_captions_segments"
```

---

### Task 6: `highlights.py` — chapter constants + `_sanitize_chapters`

**Files:**
- Modify: `shorts_generator/highlights.py`
- Test: `tests/test_highlights.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_highlights.py`, at the end of the file:

```python
from shorts_generator.highlights import (
    CHAPTER_SCHEMA_VERSION,
    MAX_CHAPTER_DURATION_SECONDS,
    MIN_CHAPTER_DURATION_SECONDS,
    _sanitize_chapters,
)


def _raw_chapter(**overrides):
    base = {
        "title": "The Origin Story",
        "start_time": 10.0,
        "end_time": 400.0,
        "summary": "They trace the idea back to a late-night argument and explain how it evolved.",
        "interest_reason": "a complete, self-contained origin story with a clear arc",
    }
    base.update(overrides)
    return base


def test_sanitize_chapters_keeps_valid_chapter():
    cleaned = _sanitize_chapters([_raw_chapter()], duration=1000.0)
    assert len(cleaned) == 1
    assert cleaned[0]["title"] == "The Origin Story"
    assert cleaned[0]["start_time"] == 10.0
    assert cleaned[0]["end_time"] == 400.0
    assert cleaned[0]["summary"] == _raw_chapter()["summary"]
    assert cleaned[0]["interest_reason"] == _raw_chapter()["interest_reason"]


def test_sanitize_chapters_drops_shorter_than_min_duration():
    raw = _raw_chapter(start_time=0.0, end_time=30.0)  # 30s < MIN_CHAPTER_DURATION_SECONDS (60)
    cleaned = _sanitize_chapters([raw], duration=1000.0)
    assert cleaned == []


def test_sanitize_chapters_clamps_end_time_to_max_duration():
    raw = _raw_chapter(start_time=0.0, end_time=2000.0)  # way over MAX_CHAPTER_DURATION_SECONDS (900)
    cleaned = _sanitize_chapters([raw], duration=5000.0)
    assert cleaned[0]["end_time"] == 900.0


def test_sanitize_chapters_clamps_to_video_duration():
    raw = _raw_chapter(start_time=90.0, end_time=200.0)
    cleaned = _sanitize_chapters([raw], duration=150.0)
    assert cleaned[0]["end_time"] == 150.0


def test_sanitize_chapters_drops_invalid_start_end():
    raw = _raw_chapter(start_time=100.0, end_time=50.0)  # end before start
    cleaned = _sanitize_chapters([raw], duration=1000.0)
    assert cleaned == []


def test_sanitize_chapters_defaults_missing_fields():
    raw = {"start_time": 0.0, "end_time": 200.0}
    cleaned = _sanitize_chapters([raw], duration=1000.0)
    assert cleaned[0]["title"] == "Untitled Chapter"
    assert cleaned[0]["summary"] == ""
    assert cleaned[0]["interest_reason"] == ""


def test_sanitize_chapters_ignores_non_list_input():
    assert _sanitize_chapters(None, duration=1000.0) == []
    assert _sanitize_chapters("not a list", duration=1000.0) == []


def test_sanitize_chapters_skips_non_dict_entries():
    cleaned = _sanitize_chapters(["not a dict", _raw_chapter()], duration=1000.0)
    assert len(cleaned) == 1


def test_chapter_duration_bounds_are_60_and_900():
    assert MIN_CHAPTER_DURATION_SECONDS == 60
    assert MAX_CHAPTER_DURATION_SECONDS == 900


def test_chapter_schema_version_is_1():
    assert CHAPTER_SCHEMA_VERSION == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_highlights.py -k sanitize_chapters -v`
Expected: FAIL with `ImportError: cannot import name '_sanitize_chapters'`

- [ ] **Step 3: Implement the constants and sanitizer**

In `shorts_generator/highlights.py`, add right after the `MAX_HIGHLIGHT_DURATION_SECONDS` constant block (after line 184):

```python
MIN_CHAPTER_DURATION_SECONDS = 60    # shorter than this isn't "full context," it's a fragment
MAX_CHAPTER_DURATION_SECONDS = 900   # 15min hard ceiling
CHAPTER_SCHEMA_VERSION = 1           # bump whenever the chapter dict shape changes
```

Then add `_sanitize_chapters` right after `_sanitize_highlights` (after line 357):

```python
def _sanitize_chapters(raw_chapters: object, duration: float) -> List[Dict]:
    """Normalize model output into the chapter shape; skip invalid entries.

    Unlike _sanitize_highlights, there's no score/hook/reaction/cut_segments
    handling here -- chapters don't carry viral-packaging fields, and the
    whole selected span is kept intact by design (no reaction-jail dead-air
    trimming for a "full context" chapter).
    """
    if not isinstance(raw_chapters, list):
        return []

    max_end = duration if duration > 0 else float("inf")
    cleaned: List[Dict] = []
    for item in raw_chapters:
        if not isinstance(item, dict):
            continue

        start = _coerce_float(item.get("start_time"), default=-1.0)
        end = _coerce_float(item.get("end_time"), default=-1.0)
        if start < 0 or end <= start:
            continue

        if max_end != float("inf"):
            start = min(start, max_end)
            end = min(end, max_end)
            if end <= start:
                continue

        end = min(end, start + MAX_CHAPTER_DURATION_SECONDS)
        if end - start < MIN_CHAPTER_DURATION_SECONDS:
            continue

        cleaned.append(
            {
                "title": str(item.get("title") or "Untitled Chapter").strip()[:100],
                "start_time": start,
                "end_time": end,
                "summary": str(item.get("summary") or "").strip(),
                "interest_reason": str(item.get("interest_reason") or "").strip(),
            }
        )

    return cleaned
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_highlights.py -k "sanitize_chapters or chapter_duration_bounds or chapter_schema_version" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/highlights.py tests/test_highlights.py
git commit -m "feat: add chapter constants and _sanitize_chapters"
```

---

### Task 7: `highlights.py` — `dedupe_chapters`

**Files:**
- Modify: `shorts_generator/highlights.py`
- Test: `tests/test_highlights.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_highlights.py`, at the end of the file:

```python
from shorts_generator.highlights import dedupe_chapters


def test_dedupe_chapters_keeps_non_overlapping_chapters_in_chronological_order():
    chapters = [
        {"title": "B", "start_time": 500.0, "end_time": 800.0},
        {"title": "A", "start_time": 0.0, "end_time": 300.0},
    ]
    result = dedupe_chapters(chapters)
    assert [c["title"] for c in result] == ["A", "B"]


def test_dedupe_chapters_drops_any_overlap_with_previously_kept():
    chapters = [
        {"title": "A", "start_time": 0.0, "end_time": 300.0},
        {"title": "B", "start_time": 250.0, "end_time": 600.0},  # overlaps A by 50s -> dropped
        {"title": "C", "start_time": 600.0, "end_time": 900.0},  # starts exactly where A ended (via B's end) -- no overlap with A
    ]
    result = dedupe_chapters(chapters)
    assert [c["title"] for c in result] == ["A", "C"]


def test_dedupe_chapters_adjacent_chapters_both_kept():
    # B starts exactly when A ends -- zero overlap, both kept
    chapters = [
        {"title": "A", "start_time": 0.0, "end_time": 300.0},
        {"title": "B", "start_time": 300.0, "end_time": 600.0},
    ]
    result = dedupe_chapters(chapters)
    assert [c["title"] for c in result] == ["A", "B"]


def test_dedupe_chapters_empty_input_returns_empty():
    assert dedupe_chapters([]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_highlights.py -k dedupe_chapters -v`
Expected: FAIL with `ImportError: cannot import name 'dedupe_chapters'`

- [ ] **Step 3: Implement it**

In `shorts_generator/highlights.py`, add right after `dedupe_highlights` (after line 469):

```python
def dedupe_chapters(chapters: List[Dict]) -> List[Dict]:
    """Sort chapters chronologically and drop any chapter that overlaps the
    previously-kept one at all (unlike dedupe_highlights' >50%-overlap
    tolerance for ranked Shorts candidates -- chapters have no score to rank
    by, and the goal is a clean sequential set that tiles the episode's
    interesting parts, not competing candidates for the same moment)."""
    ordered = sorted(chapters, key=lambda c: float(c["start_time"]))
    kept: List[Dict] = []
    for c in ordered:
        if kept and float(c["start_time"]) < float(kept[-1]["end_time"]):
            continue
        kept.append(c)
    return kept
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_highlights.py -k dedupe_chapters -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/highlights.py tests/test_highlights.py
git commit -m "feat: add dedupe_chapters (chronological, zero-overlap-tolerance)"
```

---

### Task 8: `highlights.py` — `CHAPTER_SYSTEM_PROMPT` + `call_chapter_api`

**Files:**
- Modify: `shorts_generator/highlights.py`
- Test: `tests/test_highlights.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_highlights.py`, at the end of the file:

```python
from shorts_generator.highlights import call_chapter_api


def test_call_chapter_api_returns_sanitized_chapters():
    def fake_llm_fn(prompt):
        return json.dumps({
            "chapters": [
                {
                    "title": "The Origin Story",
                    "start_time": 0.0,
                    "end_time": 300.0,
                    "summary": "Full context on how the idea started.",
                    "interest_reason": "complete arc",
                }
            ]
        })

    result = call_chapter_api(
        "transcript text", {"content_type": "podcast", "density": "high"},
        duration=1000.0, num_chapters=5, llm_fn=fake_llm_fn,
    )
    assert result["chapters"][0]["title"] == "The Origin Story"


def test_call_chapter_api_retries_on_invalid_json_then_succeeds():
    calls = {"n": 0}

    def flaky_llm_fn(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return json.dumps({
            "chapters": [{"title": "Recovered", "start_time": 0.0, "end_time": 200.0}]
        })

    result = call_chapter_api(
        "transcript text", {"content_type": "podcast", "density": "high"},
        duration=1000.0, num_chapters=5, llm_fn=flaky_llm_fn,
    )
    assert result["chapters"][0]["title"] == "Recovered"
    assert calls["n"] == 2


def test_call_chapter_api_raises_after_max_attempts_with_real_error():
    def always_fails(prompt):
        raise TimeoutError("request timed out after 180s")

    with pytest.raises(RuntimeError) as exc_info:
        call_chapter_api(
            "transcript text", {"content_type": "podcast", "density": "high"},
            duration=1000.0, num_chapters=5, llm_fn=always_fails,
        )
    assert "request timed out after 180s" in str(exc_info.value)
```

Note: `pytest` is not imported at the top of `tests/test_highlights.py` today — add `import pytest` alongside the existing `import json` / `import os` lines at the top of the file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_highlights.py -k call_chapter_api -v`
Expected: FAIL with `ImportError: cannot import name 'call_chapter_api'`

- [ ] **Step 3: Add the prompt constant**

In `shorts_generator/highlights.py`, add right after `HIGHLIGHT_SYSTEM_PROMPT` (after line 162):

```python
CHAPTER_INTEREST_CRITERIA = """
Interest signals to prioritize (ranked by impact):
1. A COMPLETE STORY OR ANECDOTE — has a beginning, a middle, and a payoff or twist
2. A STRONG ARGUMENT OR DEBATE EXCHANGE — a real back-and-forth on a substantive point
3. A CONCRETE INSIGHT OR HOW-TO — a specific technique, framework, or piece of advice
4. A REVELATION — a surprising fact, confession, or reframing moment
5. AN EMOTIONAL OR FUNNY BEAT — genuine surprise, laughter, vulnerability, excitement
6. A CONFLICT OR TENSION — disagreement, pushback, a problem confronted head-on
"""


CHAPTER_SYSTEM_PROMPT = """You are a podcast editor building a chapter index of the most substantive, interesting segments in a long-form conversation. Unlike a highlight reel, your job is NOT to extract a tight, swipe-optimized fragment — it's to extract the FULL discussion of one topic, with all its context intact, as a standalone segment someone could watch with zero other knowledge of the episode.

{chapter_interest_criteria}

Content type: {content_type} | Density: {density}

Your task: identify the most substantive, self-contained chapters in this transcript.

Rules:
- start_time must land where the topic or question is actually INTRODUCED — the premise or the question that kicks off the discussion, not just the punchline or peak moment. The "skip the windup" rule a highlight-reel editor follows does NOT apply here: the windup is often exactly the context a chapter needs to keep.
- end_time must extend to where the topic naturally RESOLVES or the conversation visibly moves to a new topic — never cut mid-thought, never cut the moment the "interesting part" lands.
- The rule of thumb: a viewer watching ONLY this chapter, with zero other context from the rest of the episode, must fully understand what's being discussed and why it matters.
- Never cut mid-sentence or mid-thought — each chapter must feel complete and self-contained.
- Chapters must not overlap with each other.
- Duration: no fixed sweet spot — let a chapter run as long as the topic actually needs, from at least 60 seconds up to a hard ceiling of 15 minutes (900 seconds).
- {num_chapters_instruction}
- Write a "title" — max 8 words, chapter-card style (this renders on a title card, so keep it short and accurate, not clickbait)
- Write a "summary" — 2-4 sentences capturing the FULL context of what's discussed: the question/premise, the substance of the discussion, and how it resolves. This is not a hook tag, it's a complete recap.
- Write an "interest_reason" — one sentence on why this segment is worth extracting as its own chapter (which signal from the list above it satisfies)

Respond ONLY with valid JSON (no markdown, no explanation):
{{"chapters":[{{"title":"string","start_time":float,"end_time":float,"summary":"string","interest_reason":"string"}}]}}"""
```

- [ ] **Step 4: Implement `call_chapter_api`**

Add right after `call_highlight_api` (after line 448):

```python
def call_chapter_api(
    transcript_text: str,
    content_info: Dict,
    duration: float,
    num_chapters: int,
    is_chunk: bool = False,
    llm_fn: LLMFn = call_muapi_llm,
) -> Dict:
    target = max(num_chapters, 3)
    natural_max = max(2 if is_chunk else 3, int(duration / 300))  # roughly one chapter per 5min of content
    min_chapters = min(target, natural_max, 8)
    system = CHAPTER_SYSTEM_PROMPT.format(
        chapter_interest_criteria=CHAPTER_INTEREST_CRITERIA,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        num_chapters_instruction=f"Identify at least {min_chapters} chapters",
    )
    base_prompt = f"{system}\n\nTranscript:\n{transcript_text}"
    prompt = base_prompt
    last_error = "unknown"

    for attempt in range(1, MAX_HIGHLIGHT_API_ATTEMPTS + 1):
        try:
            raw = llm_fn(prompt)
            parsed = _parse_json_loose(raw)
            chapters = _sanitize_chapters(parsed.get("chapters"), duration=duration)
            if chapters:
                return {"chapters": chapters}
            last_error = "no valid chapters in response"
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_HIGHLIGHT_API_ATTEMPTS:
            print(
                f"[chapters] attempt {attempt}/{MAX_HIGHLIGHT_API_ATTEMPTS} failed ({last_error}); retrying",
                flush=True,
            )
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Return ONLY valid JSON with a top-level 'chapters' array."
                + " Each item must include: title, start_time, end_time, summary, interest_reason."
                + " No markdown fences, no commentary."
            )

    raise RuntimeError(
        f"Chapter generator produced invalid output after {MAX_HIGHLIGHT_API_ATTEMPTS} attempts: {last_error}"
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_highlights.py -k call_chapter_api -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/highlights.py tests/test_highlights.py
git commit -m "feat: add CHAPTER_SYSTEM_PROMPT and call_chapter_api"
```

---

### Task 9: `highlights.py` — `get_chapters` + `get_chapters_cached`

**Files:**
- Modify: `shorts_generator/highlights.py`
- Test: `tests/test_highlights.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_highlights.py`, at the end of the file:

```python
from shorts_generator.highlights import get_chapters, get_chapters_cached


def _fake_chapter_llm_responses(chapter_title):
    def fake_llm_fn(prompt):
        if "Analyze this video transcript" in prompt:
            return '{"content_type": "podcast", "density": "high"}'
        return (
            '{"chapters": [{"title": "%s", "start_time": 0.0, "end_time": 300.0, '
            '"summary": "full context here", "interest_reason": "reason"}]}' % chapter_title
        )
    return fake_llm_fn


def test_get_chapters_returns_deduped_chapters():
    transcript = {"duration": 1000.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi there"}]}
    result = get_chapters(transcript, num_chapters=3, llm_fn=_fake_chapter_llm_responses("Chapter One"))
    assert result["chapters"][0]["title"] == "Chapter One"


def test_get_chapters_cached_calls_llm_and_writes_cache_on_miss(tmp_path):
    cache_path = str(tmp_path / "chapters.json")
    transcript = {"duration": 1000.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi there"}]}

    result = get_chapters_cached(
        transcript, num_chapters=3, cache_path=cache_path, llm_fn=_fake_chapter_llm_responses("Chapter One")
    )

    assert result["chapters"][0]["title"] == "Chapter One"
    assert os.path.exists(cache_path)
    with open(cache_path) as f:
        cached = json.load(f)
    assert cached["num_chapters"] == 3
    assert cached["schema_version"] == CHAPTER_SCHEMA_VERSION
    assert cached["transcript_fingerprint"] == _transcript_fingerprint(transcript)


def test_get_chapters_cached_skips_llm_on_matching_cache(tmp_path):
    cache_path = str(tmp_path / "chapters.json")
    transcript = {"duration": 1000.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi there"}]}
    with open(cache_path, "w") as f:
        json.dump({
            "transcript_fingerprint": _transcript_fingerprint(transcript),
            "num_chapters": 3,
            "schema_version": CHAPTER_SCHEMA_VERSION,
            "chapters": [{"title": "Cached Chapter", "start_time": 0.0, "end_time": 300.0}],
        }, f)

    def fail_if_called(prompt):
        raise AssertionError("llm_fn should not be called on a cache hit")

    result = get_chapters_cached(transcript, num_chapters=3, cache_path=cache_path, llm_fn=fail_if_called)
    assert result["chapters"][0]["title"] == "Cached Chapter"


def test_get_chapters_cached_recomputes_on_schema_version_mismatch(tmp_path):
    cache_path = str(tmp_path / "chapters.json")
    transcript = {"duration": 1000.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi there"}]}
    with open(cache_path, "w") as f:
        json.dump({
            "transcript_fingerprint": _transcript_fingerprint(transcript),
            "num_chapters": 3,
            "schema_version": CHAPTER_SCHEMA_VERSION - 1,
            "chapters": [{"title": "Stale Chapter", "start_time": 0.0, "end_time": 300.0}],
        }, f)

    result = get_chapters_cached(
        transcript, num_chapters=3, cache_path=cache_path, llm_fn=_fake_chapter_llm_responses("Fresh Chapter")
    )
    assert result["chapters"][0]["title"] == "Fresh Chapter"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_highlights.py -k get_chapters -v`
Expected: FAIL with `ImportError: cannot import name 'get_chapters'`

- [ ] **Step 3: Implement `get_chapters`**

In `shorts_generator/highlights.py`, add right after `get_highlights` (after line 531):

```python
def get_chapters(
    transcript: Dict,
    num_chapters: int = 5,
    llm_fn: Optional[LLMFn] = None,
) -> Dict:
    """Main entry point for chapter selection — returns {chapters: [...]}
    sorted chronologically. Mirrors get_highlights' chunking/caching shape;
    reuses detect_content_type/chunk_transcript/build_transcript_text as-is.
    """
    llm_fn = llm_fn or call_muapi_llm
    duration = transcript.get("duration", 0)
    content_info = detect_content_type(transcript, llm_fn=llm_fn)
    print(f"[chapters] content={content_info.get('content_type')} density={content_info.get('density')} duration={duration:.0f}s", flush=True)

    if duration >= LONG_VIDEO_THRESHOLD:
        chunks = chunk_transcript(transcript)
        print(f"[chapters] long video — splitting into {len(chunks)} chunks", flush=True)
        all_chapters: List[Dict] = []
        for i, chunk in enumerate(chunks):
            offset = chunk.get("_offset", 0)
            text = build_transcript_text(chunk)
            print(f"[chapters] chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)", flush=True)
            chunk_abs_end = offset + chunk["duration"]
            t0 = time.time()
            result = call_chapter_api(text, content_info, chunk_abs_end, num_chapters=num_chapters, is_chunk=True, llm_fn=llm_fn)
            print(f"[chapters] chunk {i + 1}/{len(chunks)} done in {time.time() - t0:.1f}s", flush=True)
            all_chapters.extend(result.get("chapters", []))
        chapters = dedupe_chapters(all_chapters)
    else:
        text = build_transcript_text(transcript)
        result = call_chapter_api(text, content_info, duration, num_chapters=num_chapters, llm_fn=llm_fn)
        chapters = dedupe_chapters(result.get("chapters", []))

    return {"chapters": chapters}


def get_chapters_cached(
    transcript: Dict,
    num_chapters: int,
    cache_path: str,
    llm_fn: Optional[LLMFn] = None,
) -> Dict:
    """Wraps get_chapters with an on-disk cache keyed by a transcript
    content fingerprint + num_chapters, mirroring get_highlights_cached."""
    fingerprint = _transcript_fingerprint(transcript)

    if os.path.exists(cache_path):
        cached = None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except json.JSONDecodeError:
            print(f"[chapters] cached chapters corrupted, recomputing: {cache_path}", flush=True)

        if (
            isinstance(cached, dict)
            and cached.get("transcript_fingerprint") == fingerprint
            and cached.get("num_chapters") == num_chapters
            and cached.get("schema_version") == CHAPTER_SCHEMA_VERSION
            and isinstance(cached.get("chapters"), list)
        ):
            print(f"[chapters] reusing cached chapters: {cache_path}", flush=True)
            return {"chapters": cached["chapters"]}

    result = get_chapters(transcript, num_chapters=num_chapters, llm_fn=llm_fn or call_muapi_llm)

    tmp_path = cache_path + ".part"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "transcript_fingerprint": fingerprint,
                "num_chapters": num_chapters,
                "schema_version": CHAPTER_SCHEMA_VERSION,
                "chapters": result.get("chapters", []),
            },
            f,
            ensure_ascii=False,
        )
    os.replace(tmp_path, cache_path)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_highlights.py -k get_chapters -v`
Expected: PASS

- [ ] **Step 5: Run the full highlights test file to confirm no regression**

Run: `python -m pytest tests/test_highlights.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/highlights.py tests/test_highlights.py
git commit -m "feat: add get_chapters/get_chapters_cached entry points"
```

---

### Task 10: `local/clipper.py` — `crop_chapters_local`

**Files:**
- Modify: `shorts_generator/local/clipper.py`
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_local_clipper.py`, at the end of the file:

```python
from shorts_generator.local.clipper import crop_chapters_local


def _chapter():
    return {
        "title": "Test Chapter", "start_time": 1.0, "end_time": 4.0,
        "summary": "A short test chapter.",
    }


def test_crop_chapters_local_produces_landscape_output_no_crop(tmp_path, synthetic_source):
    out_dir = str(tmp_path / "out")
    results = crop_chapters_local(
        synthetic_source, [_chapter()], out_dir=out_dir, transcript_segments=_segments(),
    )

    assert len(results) == 1
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", results[0]["clip_url"]],
        capture_output=True, text=True, check=True,
    )
    width, height = (int(v) for v in probe.stdout.strip().split(","))
    # synthetic_source is 640x360 -- output must match the SOURCE frame size,
    # not a vertical-HD canvas like the Shorts crop path produces.
    assert (width, height) == (640, 360)


def test_crop_chapters_local_output_duration_matches_trim_window(tmp_path, synthetic_source):
    out_dir = str(tmp_path / "out")
    results = crop_chapters_local(
        synthetic_source, [_chapter()], out_dir=out_dir, transcript_segments=_segments(), captions=False,
    )
    duration = _probe_duration(results[0]["clip_url"])
    assert abs(duration - 3.0) < 0.3  # end_time(4.0) - start_time(1.0)


def test_crop_chapters_local_filename_uses_numbered_prefix(tmp_path, synthetic_source):
    out_dir = str(tmp_path / "out")
    results = crop_chapters_local(
        synthetic_source, [_chapter()], out_dir=out_dir, transcript_segments=_segments(),
    )
    assert os.path.basename(results[0]["clip_url"]) == "01_Test_Chapter.mp4"


def test_crop_chapters_local_second_chapter_gets_prefix_02(tmp_path, synthetic_source):
    out_dir = str(tmp_path / "out")
    chapter_two = {**_chapter(), "title": "Second Chapter", "start_time": 0.5, "end_time": 2.0}
    results = crop_chapters_local(
        synthetic_source, [_chapter(), chapter_two], out_dir=out_dir, transcript_segments=_segments(),
    )
    basenames = [os.path.basename(r["clip_url"]) for r in results]
    assert basenames == ["01_Test_Chapter.mp4", "02_Second_Chapter.mp4"]


def test_crop_chapters_local_captions_burned_by_default(tmp_path, synthetic_source):
    out_dir = str(tmp_path / "out")
    results = crop_chapters_local(
        synthetic_source, [_chapter()], out_dir=out_dir, transcript_segments=_segments(),
    )
    assert "captions_error" not in results[0]


def test_crop_chapters_local_captions_disabled_skips_burn_in(tmp_path, synthetic_source, monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("burn_captions should not be called when captions=False")
    monkeypatch.setattr("shorts_generator.local.clipper.burn_captions", _fail_if_called)

    out_dir = str(tmp_path / "out")
    results = crop_chapters_local(
        synthetic_source, [_chapter()], out_dir=out_dir, transcript_segments=_segments(), captions=False,
    )
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])


def test_crop_chapters_local_caption_failure_falls_back_to_plain_clip(tmp_path, synthetic_source, monkeypatch):
    def _raise(*args, **kwargs):
        raise captions_module.CaptionError("boom")
    monkeypatch.setattr("shorts_generator.local.clipper.burn_captions", _raise)

    out_dir = str(tmp_path / "out")
    results = crop_chapters_local(
        synthetic_source, [_chapter()], out_dir=out_dir, transcript_segments=_segments(),
    )
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])
    assert results[0]["captions_error"] == "boom"


def test_crop_chapters_local_uses_chapter_bottom_margin(tmp_path, synthetic_source, monkeypatch):
    # Same spy shape as test_local_clipper.py's existing
    # test_word_highlight_flag_forwarded_to_burn: burn_captions is called
    # positionally (out_path, transcript_segments, start, end, captioned_path),
    # so args[4] is always the destination path to fake-write.
    captured = {}

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        import shutil
        shutil.copyfile(args[0], args[4])
        return args[4]

    monkeypatch.setattr("shorts_generator.local.clipper.burn_captions", _spy)
    crop_chapters_local(
        synthetic_source, [_chapter()], out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
    )
    assert captured["bottom_margin_frac"] == local_clipper_module.CHAPTER_CAPTION_BOTTOM_MARGIN_FRAC
    assert local_clipper_module.CHAPTER_CAPTION_BOTTOM_MARGIN_FRAC == 0.06


def test_crop_chapters_local_failure_is_recorded_and_run_continues(tmp_path, synthetic_source):
    bad_chapter = {"title": "Bad", "start_time": 900.0, "end_time": 950.0}  # way past the 6s source
    good_chapter = _chapter()
    out_dir = str(tmp_path / "out")
    results = crop_chapters_local(
        synthetic_source, [bad_chapter, good_chapter], out_dir=out_dir, transcript_segments=_segments(),
    )
    assert len(results) == 2
    assert results[0]["clip_url"] is None
    assert "error" in results[0]
    assert results[1]["clip_url"] is not None
    assert os.path.exists(results[1]["clip_url"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_local_clipper.py -k crop_chapters_local -v`
Expected: FAIL with `ImportError: cannot import name 'crop_chapters_local'`

- [ ] **Step 3: Implement `crop_chapters_local`**

In `shorts_generator/local/clipper.py`, first update the import line (line 27) to also pull in `unique_chapter_filename`:

```python
from ..run_output import unique_chapter_filename, unique_short_filename
```

Then add a new constant right after `OUTPUT_CANVAS_H` (after line 60):

```python
CHAPTER_CAPTION_BOTTOM_MARGIN_FRAC = 0.06  # landscape chapter clips have no
                                            # platform UI to dodge (no Shorts
                                            # reply/like column) -- captions
                                            # sit near the bottom edge, not
                                            # pushed up to Shorts' 0.30
```

Then add `crop_chapters_local` right after `crop_highlights_local` (at the end of the file, after line 809):

```python
def crop_chapters_local(
    source_path: str,
    chapters: List[Dict],
    out_dir: Optional[str] = None,
    transcript_segments: Optional[List[Dict]] = None,
    captions: bool = True,
    caption_fade_duration: float = 0.3,
    word_highlight: bool = True,
) -> List[Dict]:
    """Trim every chapter to its own landscape mp4 -- no crop, no hook/end
    card, no jump-cut excision (the whole selected span is kept intact by
    design). Captions burn near the bottom edge via
    CHAPTER_CAPTION_BOTTOM_MARGIN_FRAC instead of Shorts' 0.30.
    """
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    used_names: set = set()
    for i, c in enumerate(chapters, 1):
        out_path = os.path.join(out_dir, unique_chapter_filename(c.get("title"), i, used_names))
        print(f"[chapter/local] {i}/{len(chapters)}: {c.get('title', '(untitled)')}", flush=True)
        try:
            _cut_subclip(source_path, float(c["start_time"]), float(c["end_time"]), out_path)
            entry = {**c, "clip_url": out_path}

            if captions and transcript_segments:
                captioned_path = out_path + ".captioned.mp4"
                try:
                    burn_captions(
                        out_path,
                        transcript_segments,
                        float(c["start_time"]),
                        float(c["end_time"]),
                        captioned_path,
                        fade_seconds=caption_fade_duration,
                        word_highlight=word_highlight,
                        bottom_margin_frac=CHAPTER_CAPTION_BOTTOM_MARGIN_FRAC,
                    )
                    os.replace(captioned_path, out_path)
                except CaptionError as e:
                    print(f"[chapter/local] {i} captions skipped: {e}", flush=True)
                    entry["captions_error"] = str(e)
                    if os.path.exists(captioned_path):
                        os.remove(captioned_path)

            results.append(entry)
        except Exception as e:
            print(f"[chapter/local] {i} failed: {e}", flush=True)
            results.append({**c, "clip_url": None, "error": str(e)})
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_local_clipper.py -k crop_chapters_local -v`
Expected: PASS

- [ ] **Step 5: Run the full local_clipper test file to confirm no regression**

Run: `python -m pytest tests/test_local_clipper.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "feat: add crop_chapters_local (trim-only, no crop/card/excision)"
```

---

### Task 11: `pipeline.py` — `_run_local_chapters` + `generate_chapters`

**Files:**
- Modify: `shorts_generator/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline.py`, at the end of the file:

```python
import shorts_generator.highlights as highlights_module


def _fake_chapters_result():
    return {"chapters": [{"start_time": 0.0, "end_time": 300.0, "title": "Chapter", "summary": "s"}]}


def _fake_chapters_result_many(count):
    return {
        "chapters": [
            {"start_time": float(i * 400), "end_time": float(i * 400) + 300.0, "title": f"Chapter {i}", "summary": "s"}
            for i in range(count)
        ]
    }


def test_run_local_chapters_threads_captions_params(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_chapters_cached",
        lambda transcript, num_chapters, cache_path, llm_fn: _fake_chapters_result(),
    )

    crop_mock = Mock(return_value=[{"clip_url": "/tmp/out/01_Chapter.mp4"}])
    monkeypatch.setattr(local_clipper_module, "crop_chapters_local", crop_mock)

    result = pipeline_module._run_local_chapters(
        "https://youtube.example/x",
        num_chapters=1,
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.7,
        paths=_paths(tmp_path),
        word_highlight=False,
    )

    assert result["chapters"] == [{"clip_url": "/tmp/out/01_Chapter.mp4"}]

    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is False
    assert kwargs["caption_fade_duration"] == 0.7
    assert kwargs["word_highlight"] is False
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]


def test_run_local_chapters_skips_download_when_source_already_exists(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    with open(paths.source_video, "wb") as f:
        f.write(b"already downloaded")

    def _fail_if_called(*a, **k):
        raise AssertionError("download_youtube_local should not be called when full_source.mp4 exists")

    monkeypatch.setattr(local_downloader_module, "download_youtube_local", _fail_if_called)
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_chapters_cached",
        lambda transcript, num_chapters, cache_path, llm_fn: _fake_chapters_result(),
    )
    monkeypatch.setattr(local_clipper_module, "crop_chapters_local", Mock(return_value=[]))

    result = pipeline_module._run_local_chapters(
        "https://youtube.example/x", num_chapters=1, download_format="720", language=None,
        captions=False, caption_fade_duration=0.3, paths=paths, word_highlight=True,
    )
    assert result["source_video_url"] == paths.source_video


def test_run_local_chapters_raises_on_zero_chapters(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_chapters_cached",
        lambda transcript, num_chapters, cache_path, llm_fn: {"chapters": []},
    )

    with pytest.raises(RuntimeError):
        pipeline_module._run_local_chapters(
            "https://youtube.example/x", num_chapters=1, download_format="720", language=None,
            captions=False, caption_fade_duration=0.3, paths=_paths(tmp_path), word_highlight=True,
        )


def test_run_local_chapters_trims_extra_successes_to_num_chapters(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_chapters_cached",
        lambda transcript, num_chapters, cache_path, llm_fn: _fake_chapters_result_many(5),
    )

    def all_succeed_crop(source_path, chapters, **kwargs):
        return [{**c, "clip_url": f"/tmp/out/{c['title']}.mp4"} for c in chapters]

    monkeypatch.setattr(local_clipper_module, "crop_chapters_local", all_succeed_crop)

    result = pipeline_module._run_local_chapters(
        "https://youtube.example/x", num_chapters=2, download_format="720", language=None,
        captions=False, caption_fade_duration=0.3, paths=_paths(tmp_path), word_highlight=True,
    )
    assert len(result["chapters"]) == 2


def test_generate_chapters_writes_chapter_descriptions_and_result_json(tmp_path, monkeypatch):
    paths = _paths(tmp_path)

    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_chapters_cached",
        lambda transcript, num_chapters, cache_path, llm_fn: _fake_chapters_result(),
    )
    monkeypatch.setattr(
        local_clipper_module, "crop_chapters_local",
        Mock(return_value=[{"start_time": 0.0, "end_time": 300.0, "title": "Chapter", "summary": "s", "clip_url": os.path.join(paths.chapters_dir, "01_Chapter.mp4")}]),
    )

    result = pipeline_module.generate_chapters("https://youtube.example/x", paths=paths)

    assert result["output_dir"] == paths.root
    assert os.path.exists(os.path.join(paths.chapters_dir, "chapters_description.txt"))
    assert os.path.exists(paths.result_json)
    assert os.path.exists(paths.progress_log)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -k chapters -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.pipeline' has no attribute '_run_local_chapters'`

- [ ] **Step 3: Add the imports**

In `shorts_generator/pipeline.py`, modify the import block (lines 19-25):

```python
from .clipper import _download_to, crop_highlights
from .downloader import download_youtube
from .highlights import call_muapi_llm, get_chapters_cached, get_highlights_cached, select_final_highlights
from .local.llm import call_openai_vision_llm
from .run_output import RunPaths, capture_progress_log, resolve_output_dir, write_chapter_descriptions, write_descriptions
from .transcriber import transcribe
from .visual_hook import call_muapi_vision_llm, score_visual_hooks
```

- [ ] **Step 4: Implement `_run_local_chapters`**

Add right after `_run_local` (after line 126):

```python
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
    chapters = _trim_to_num_clips(chapters, num_chapters)

    return {
        "output_dir": paths.root,
        "source_video_url": source_path,
        "transcript": transcript,
        "all_chapters": all_chapters,
        "chapters": chapters,
    }
```

- [ ] **Step 5: Implement `generate_chapters`**

Add right after `generate_shorts` (at the end of the file):

```python
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
          "all_chapters": [...],     # every candidate chapter before trimming to num_chapters
          "chapters": [...],         # top `num_chapters`, each with clip_url / *_error fields
        }
    """
    paths = paths or resolve_output_dir(youtube_url)
    with capture_progress_log(paths.progress_log):
        result = _run_local_chapters(
            youtube_url, num_chapters, download_format, language, captions, caption_fade_duration,
            paths, word_highlight=word_highlight,
        )

        write_chapter_descriptions(paths.chapters_dir, result["chapters"])

        with open(paths.result_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -k chapters -v`
Expected: PASS

- [ ] **Step 7: Run the full pipeline test file to confirm no regression**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add shorts_generator/pipeline.py tests/test_pipeline.py
git commit -m "feat: add _run_local_chapters and generate_chapters orchestration"
```

---

### Task 12: export `generate_chapters` from the package

**Files:**
- Modify: `shorts_generator/__init__.py`

- [ ] **Step 1: Update the export**

```python
from .pipeline import generate_chapters, generate_shorts

__all__ = ["generate_chapters", "generate_shorts"]
```

- [ ] **Step 2: Verify the import works**

Run: `python -c "from shorts_generator import generate_chapters, generate_shorts; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add shorts_generator/__init__.py
git commit -m "feat: export generate_chapters from shorts_generator package"
```

---

### Task 13: `main.py` — `--clip-type` CLI flag

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_main.py`, at the end of the file:

```python
def test_clip_type_defaults_to_shorts():
    args = build_parser().parse_args(["https://example.com/video"])
    assert args.clip_type == "shorts"


def test_clip_type_chapters_flag():
    args = build_parser().parse_args(["https://example.com/video", "--clip-type", "chapters"])
    assert args.clip_type == "chapters"


def test_num_chapters_defaults_to_5():
    args = build_parser().parse_args(["https://example.com/video"])
    assert args.num_chapters == 5


def test_num_chapters_flag_overrides_default():
    args = build_parser().parse_args(["https://example.com/video", "--num-chapters", "8"])
    assert args.num_chapters == 8


def test_clip_type_rejects_invalid_value():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["https://example.com/video", "--clip-type", "bogus"])
```

Note: add `import pytest` at the top of `tests/test_main.py` (not present today).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_main.py -k "clip_type or num_chapters" -v`
Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'clip_type'`

- [ ] **Step 3: Add the flags to `build_parser`**

In `main.py`, add right after the `--filename-style` argument (after line 86, before the `return parser` on line 87):

```python
    parser.add_argument(
        "--clip-type",
        choices=["shorts", "chapters"],
        default="shorts",
        help="shorts (default): viral 9:16 Shorts. chapters: long-form landscape "
             "chapter cuts, up to 15min each, full topic context, --mode local only.",
    )
    parser.add_argument(
        "--num-chapters",
        type=int,
        default=5,
        help="Target chapter count for --clip-type chapters (default: 5); the model "
             "may return 3-8 based on natural topic boundaries.",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add --clip-type and --num-chapters CLI flags"
```

---

### Task 14: `main.py` — wire `--clip-type chapters` into `main()`

**Files:**
- Modify: `main.py`

This task has no isolated unit test (it's the CLI's `main()` glue, exercised end-to-end manually in Step 3) — it wires flags already tested in Task 13 to a pipeline function already tested in Task 11.

- [ ] **Step 1: Update the import**

In `main.py`, modify line 18:

```python
from shorts_generator import generate_chapters, generate_shorts
```

- [ ] **Step 2: Branch `main()` on `args.clip_type`**

Replace the whole `main()` function (lines 90-137) with:

```python
def main() -> int:
    args = build_parser().parse_args()

    if args.clip_type == "chapters":
        if args.mode != "local":
            print(f"[main] --clip-type chapters is local-only; ignoring --mode {args.mode!r} and using local", file=sys.stderr)
        ignored_flags = []
        if args.aspect_ratio != "9:16":
            ignored_flags.append(f"--aspect-ratio {args.aspect_ratio}")
        if args.framing != "locked":
            ignored_flags.append(f"--framing {args.framing}")
        if args.hook_card is False:
            ignored_flags.append("--no-hook-card")
        if args.end_card is True:
            ignored_flags.append("--end-card")
        if ignored_flags:
            print(
                f"[main] --clip-type chapters ignores: {', '.join(ignored_flags)} "
                "(no crop, no card overlays in this path)",
                file=sys.stderr,
            )

    try:
        if args.clip_type == "chapters":
            result = generate_chapters(
                youtube_url=args.url,
                num_chapters=args.num_chapters,
                download_format=args.format,
                language=args.language,
                captions=args.captions,
                caption_fade_duration=args.caption_fade_duration,
                word_highlight=args.word_highlight,
            )
        else:
            result = generate_shorts(
                youtube_url=args.url,
                num_clips=args.num_clips,
                aspect_ratio=args.aspect_ratio,
                download_format=args.format,
                language=args.language,
                mode=args.mode,
                captions=args.captions,
                caption_fade_duration=args.caption_fade_duration,
                word_highlight=args.word_highlight,
                framing=args.framing,
                hook_card=args.hook_card,
                end_card=args.end_card,
                filename_style=args.filename_style,
            )
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 72)
    if args.clip_type == "chapters":
        print(f"Output folder: {result.get('output_dir')}")
        print(f"Source video:  {result['source_video_url']}")
        print(f"Chapters:      {len(result['all_chapters'])} candidates -> kept top {len(result['chapters'])}")
        print("=" * 72)
        for i, c in enumerate(result["chapters"], 1):
            print(f"\n#{i}  {c.get('start_time'):.1f}s -> {c.get('end_time'):.1f}s")
            print(f"     title:   {c.get('title')}")
            if c.get("summary"):
                print(f"     summary: {c.get('summary')}")
            if c.get("clip_url"):
                print(f"     clip:    {c['clip_url']}")
            else:
                print(f"     clip:    FAILED ({c.get('error')})")
    else:
        print(f"Mode:          {result.get('mode', args.mode)}")
        print(f"Output folder: {result.get('output_dir')}")
        print(f"Source video:  {result['source_video_url']}")
        print(f"Highlights:    {len(result['highlights'])} candidates → kept top {len(result['shorts'])}")
        print("=" * 72)
        for i, s in enumerate(result["shorts"], 1):
            print(f"\n#{i}  score={s.get('score')}  {s.get('start_time'):.1f}s → {s.get('end_time'):.1f}s")
            print(f"     title:  {s.get('yt_title') or s.get('title')}")
            print(f"     hook:   {s.get('hook_sentence')}")
            if s.get("description"):
                print(f"     desc:   {s.get('description')}")
            if s.get("yt_hashtags"):
                print(f"     tags:   {' '.join(s.get('yt_hashtags'))}")
            if s.get("clip_url"):
                print(f"     clip:   {s['clip_url']}")
            else:
                print(f"     clip:   FAILED ({s.get('error')})")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull JSON written to {args.output_json}")

    return 0
```

- [ ] **Step 3: Manually verify the CLI end-to-end**

This step needs a real (short) local video file and a working local-mode setup (faster-whisper + an LLM provider key), so it's a manual check, not part of the automated suite. Run:

```bash
python main.py "/path/to/some/local_podcast_clip.mp4" --clip-type chapters --mode local --num-chapters 2
```

Expected: prints `Chapters:      N candidates -> kept top 2`, each with a `title`/`summary`/`clip:` line pointing at a real file under `output/<Title>/Chapters/`; play one of the output files and confirm it's landscape (not cropped) with captions sitting near the bottom edge.

- [ ] **Step 4: Run the full automated test suite**

Run: `python -m pytest tests/ -v`
Expected: PASS (all tests, including the ones added in Tasks 1-13)

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: wire --clip-type chapters into main()"
```

---

## Self-Review Notes

- **Spec coverage:** every section of `docs/superpowers/specs/2026-08-07-long-form-chapter-cuts-design.md` maps to a task — chapter dict shape/prompt/sanitize/dedupe (Tasks 6-8), entry points (Task 9), clip rendering (Task 10), captions margin (Tasks 4-5), orchestration (Task 11-12), output paths (Tasks 1-3), CLI (Tasks 13-14).
- **`_trim_to_num_clips` reuse:** the spec sketches a `_trim_to_num_chapters` shape, but `_trim_to_num_clips` (pipeline.py, unchanged) only ever reads a dict's `clip_url` key — nothing Shorts-specific — so Task 11 reuses it directly on the chapters list rather than duplicating it. Same trim/delete-extras/keep-failures-visible behavior, zero new code.
- **`api` mode:** intentionally untouched anywhere in this plan — chapters is local-only per the spec's scope section; `main.py`'s `--mode` flag is only ever read for the shorts path or to print the local-only notice.
- **Type/name consistency check:** `crop_chapters_local` (Task 10) is imported and called by name in `_run_local_chapters` (Task 11); `get_chapters_cached`/`CHAPTER_SCHEMA_VERSION`/`dedupe_chapters`/`_sanitize_chapters`/`CHAPTER_INTEREST_CRITERIA`/`CHAPTER_SYSTEM_PROMPT` names match between their defining task and every later task that imports/calls them; `unique_chapter_filename(title, index, used_names)` signature matches between Task 2's definition and Task 10's call site; `write_chapter_descriptions(chapters_dir, chapters)` matches between Task 3 and Task 11.
