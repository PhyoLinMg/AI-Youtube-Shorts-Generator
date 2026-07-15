# VPS CPU/Memory Optimization (Local Mode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut CPU cost of `mode=local` runs (both `--framing locked` and `--framing adaptive`) so the pipeline runs comfortably on a 2-vCPU/8GB VPS, without changing output quality or the public function contracts other modules depend on.

**Architecture:** Six surgical changes to `shorts_generator/local/clipper.py`, `shorts_generator/local/transcriber.py`, `shorts_generator/captions.py`, and `shorts_generator/config.py`, in the priority order established by `docs/superpowers/specs/2026-07-16-vps-cpu-memory-optimization-design.md`: ffmpeg seek-order fix → crop+caption fusion (locked framing) → grab()-skip decode → whisper `beam_size` tuning → adaptive-path stride + interpolation swap → thread pinning (insurance). `mode=api` is untouched.

**Tech Stack:** Python, ffmpeg (subprocess), OpenCV (`cv2`), faster-whisper, pytest (existing integration-style tests using real ffmpeg/OpenCV against synthetic `testsrc` fixtures — no new mocking framework).

---

## Before you start

Read these three files in full — every task below references exact line ranges and existing behavior in them:
- `shorts_generator/local/clipper.py` (the file most tasks modify)
- `shorts_generator/captions.py`
- `tests/test_local_clipper.py` and `tests/test_captions.py`

Run the existing test suite once and confirm it's green before making any change:

```bash
cd /Users/linnmaung/project/AI-Youtube-Shorts-Generator
python -m pytest tests/test_local_clipper.py tests/test_captions.py tests/test_local_transcriber.py -v
```
Expected: all tests PASS (this is your baseline — if anything is already red, stop and report it before continuing).

---

### Task 1: Fix `ffmpeg -ss`/`-i` seek order in `_cut_subclip`

**Problem:** `_cut_subclip` (`shorts_generator/local/clipper.py:46-58`) places `-ss` after `-i` (output seeking) — ffmpeg decodes and discards every frame from `t=0` to `start` before it starts encoding. For a highlight late in a long source, this is a large, repeated, wasted decode.

**Files:**
- Modify: `shorts_generator/local/clipper.py:46-58`
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write a duration-correctness regression test**

This test is a **correctness guard for the seek-order refactor, not a red/green performance test** — `_cut_subclip` already produces the right duration today (just slower), so this test is expected to **pass both before and after** the change. Its job is to catch a future regression in the tricky `-ss`/`-to`/`-t` seek-semantics interaction (verified empirically during design: with `-ss` before `-i`, `-to end` is measured from the seek point, not the source's absolute timeline, so it must become `-t (end - start)`).

Add to `tests/test_local_clipper.py` (near the top, after the existing imports and fixtures):

```python
def _probe_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def test_cut_subclip_produces_correct_duration(tmp_path, synthetic_source):
    from shorts_generator.local.clipper import _cut_subclip

    out_path = str(tmp_path / "cut.mp4")
    _cut_subclip(synthetic_source, start=1.0, end=4.0, out_path=out_path)

    duration = _probe_duration(out_path)
    assert abs(duration - 3.0) < 0.5
```

- [ ] **Step 2: Run it to confirm it passes on the current (unfixed) code**

Run: `python -m pytest tests/test_local_clipper.py::test_cut_subclip_produces_correct_duration -v`
Expected: PASS (duration ~3.0s) — this establishes the baseline the refactor must not break.

- [ ] **Step 3: Fix the seek order**

In `shorts_generator/local/clipper.py`, replace `_cut_subclip`:

```python
def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -t (end-start) → re-encoded mp4 with audio.

    -ss is placed before -i (input seeking, keyframe-based) rather than
    after (output seeking, which would decode and discard every frame from
    0 to `start` before encoding anything — expensive for a highlight late
    in a long source). Once -ss precedes -i, -to is measured from the seek
    point rather than the source's absolute timeline, so duration must be
    expressed as -t (end - start), not -to end.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", source_path,
        "-t", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path
```

- [ ] **Step 4: Run the test again to confirm it still passes**

Run: `python -m pytest tests/test_local_clipper.py::test_cut_subclip_produces_correct_duration -v`
Expected: PASS

- [ ] **Step 5: Run the full local-clipper suite to confirm no regression**

Run: `python -m pytest tests/test_local_clipper.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "perf: input-seek in _cut_subclip instead of output-seek

Avoids decoding and discarding every frame before the highlight's start
time on every clip cut."
```

---

### Task 2: Extract `build_ass_file` from `captions.py` (pure refactor)

**Problem:** Task 3 needs to build an `.ass` caption file from crop dimensions that are known *before* the cropped video file exists (so `burn_captions`'s `_probe_resolution(video_path)` call — which requires the file to already exist — can't be reused as-is). This task extracts the ASS-building logic into a standalone public function with no behavior change to `burn_captions`.

**Files:**
- Modify: `shorts_generator/captions.py:248-289`
- Test: `tests/test_captions.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_captions.py` (add `build_ass_file` to the existing import block at the top):

```python
from shorts_generator.captions import (
    CaptionError,
    _chunk_segments,
    _format_ass_timestamp,
    _probe_resolution,
    _write_ass,
    build_ass_file,
    burn_captions,
)
```

Then add this test:

```python
def test_build_ass_file_writes_file_at_given_dimensions(tmp_path):
    segments = [{"start": 0.0, "end": 3.0, "text": "hello there this is a caption test"}]
    ass_path = str(tmp_path / "captions.ass")

    result = build_ass_file(segments, clip_start=0.0, clip_end=3.0, width=608, height=1080, ass_path=ass_path)

    assert result == ass_path
    assert os.path.exists(ass_path)
    content = open(ass_path, encoding="utf-8").read()
    assert "PlayResX: 608" in content
    assert "PlayResY: 1080" in content


def test_build_ass_file_raises_when_no_transcript_overlaps(tmp_path):
    segments = [{"start": 100.0, "end": 103.0, "text": "way outside the clip"}]
    ass_path = str(tmp_path / "captions.ass")

    with pytest.raises(CaptionError):
        build_ass_file(segments, clip_start=0.0, clip_end=3.0, width=608, height=1080, ass_path=ass_path)
```

(`os` is already imported at the top of `tests/test_captions.py`.)

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_captions.py::test_build_ass_file_writes_file_at_given_dimensions -v`
Expected: FAIL with `ImportError: cannot import name 'build_ass_file'`

- [ ] **Step 3: Extract `build_ass_file` and refactor `burn_captions` to use it**

In `shorts_generator/captions.py`, replace the `burn_captions` function (lines 248-289) with:

```python
def build_ass_file(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    width: int,
    height: int,
    ass_path: str,
    fade_seconds: float = 0.3,
    word_highlight: bool = True,
) -> str:
    """Build an .ass caption file for a clip window at explicit dimensions.

    Callers that already know the output frame size (e.g. a crop box that
    hasn't been rendered to a file yet) should use this directly instead of
    burn_captions, which probes an existing video file's resolution.

    Raises CaptionError if no transcript segments overlap the clip window.
    """
    chunks = _chunk_segments(segments, clip_start, clip_end, max_words=7)
    if not chunks:
        raise CaptionError(f"no transcript overlaps clip window [{clip_start}, {clip_end}]")

    _write_ass(chunks, ass_path, width, height, fade_seconds, word_highlight=word_highlight)
    return ass_path


def burn_captions(
    video_path: str,
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    out_path: str,
    fade_seconds: float = 0.3,
    word_highlight: bool = True,
) -> str:
    """Burn phrase-chunked, fade-in captions onto a local clip.

    Raises CaptionError on any failure; the caller decides whether to fall
    back to the uncaptioned clip.
    """
    width, height = _probe_resolution(video_path)

    ass_path = out_path + ".ass"
    build_ass_file(segments, clip_start, clip_end, width, height, ass_path, fade_seconds=fade_seconds, word_highlight=word_highlight)

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
    except OSError as e:
        raise CaptionError(f"ffmpeg subtitles burn-in failed: {e}") from e
    finally:
        os.remove(ass_path)

    return out_path
```

Note: `build_ass_file` can raise `CaptionError` before `ass_path` exists (no chunks) — `burn_captions` doesn't need its own empty-chunks check anymore, `build_ass_file` owns that.

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_captions.py -v`
Expected: all PASS, including the two new tests and every pre-existing `test_burn_captions_*` test (behavior-preserving refactor).

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/captions.py tests/test_captions.py
git commit -m "refactor: extract build_ass_file from burn_captions

No behavior change — burn_captions still probes the video file's
resolution and burns in one ffmpeg pass. This lets a caller that already
knows the target frame size (added in the next commit) build the .ass
file without needing the cropped video file to exist first."
```

---

### Task 3: Fuse crop + caption burn-in into one ffmpeg pass for locked framing

**Problem:** the locked-framing path currently re-encodes the same footage three times: `_cut_subclip` (ffmpeg), `_reframe_vertical`'s OpenCV `VideoWriter` pass + a separate ffmpeg mux, and `burn_captions`'s ffmpeg burn-in. Since captions are on by default, most locked-mode clips pay for all three. This task collapses the reframe + caption steps into one ffmpeg `crop`+`subtitles` filter pass when transcript segments are available, keeping a crop-only ffmpeg fallback for the no-captions case and leaving adaptive framing untouched.

**Files:**
- Modify: `shorts_generator/local/clipper.py:1-24` (imports), `:61-153` (`_reframe_vertical`), `:399-478` (`crop_clip_local`, `crop_highlights_local`)
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Update the import line**

In `shorts_generator/local/clipper.py`, change:

```python
from ..captions import CaptionError, burn_captions
```

to:

```python
from ..captions import CaptionError, build_ass_file, burn_captions
```

- [ ] **Step 2: Replace `_reframe_vertical` with three functions**

Replace the entire `_reframe_vertical` function (lines 61-153) with:

```python
def _compute_locked_crop_box(in_path: str, aspect_ratio: str) -> Tuple[int, int, int, int]:
    """Pass 1: sample the clip (~5 frames/sec) and take the median face
    position as a single, stable crop anchor for the whole clip.

    Returns (x0, y0, crop_w, crop_h) — a static box that never changes
    within the clip, since any per-frame tracking (even heavily smoothed)
    reads as camera shake on a talking head.
    """
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    sample_stride = max(1, int(fps // 5))
    sample_centers: List[Tuple[int, int]] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                sample_centers.append((x + w // 2, y + h // 2))
        frame_idx += 1

    cap.release()

    if sample_centers:
        xs = sorted(c[0] for c in sample_centers)
        ys = sorted(c[1] for c in sample_centers)
        cx, cy = xs[len(xs) // 2], ys[len(ys) // 2]
    else:
        cx, cy = src_w // 2, src_h // 2

    x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
    y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
    return x0, y0, crop_w, crop_h


def _apply_crop(
    in_path: str,
    out_path: str,
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    ass_path: Optional[str] = None,
) -> str:
    """Single ffmpeg pass: crop to the given static box, optionally burning
    in captions from `ass_path` in the same pass. Audio is carried straight
    through from `in_path` (single input, default stream mapping) since
    there's no second silent-video file to mux anymore.

    Raises CaptionError if a captioned pass fails (caller should retry
    without ass_path). A crop-only failure propagates as
    subprocess.CalledProcessError, same as the pre-fusion behavior.
    """
    vf = f"crop={crop_w}:{crop_h}:{x0}:{y0}"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    if not ass_path:
        subprocess.run(cmd, check=True)
        return out_path

    escaped_ass_path = ass_path.replace("\\", "/").replace(":", "\\:")
    cmd[cmd.index("-vf") + 1] = f"{vf},subtitles={escaped_ass_path}"
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise CaptionError(f"ffmpeg crop+subtitles burn-in failed: {e.stderr}") from e
    return out_path


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Crop-only pass (no captions) — locked framing, static box from pass 1."""
    x0, y0, crop_w, crop_h = _compute_locked_crop_box(in_path, aspect_ratio)
    return _apply_crop(in_path, out_path, x0, y0, crop_w, crop_h)


def _reframe_vertical_with_captions(
    in_path: str,
    out_path: str,
    aspect_ratio: str,
    transcript_segments: List[Dict],
    clip_start: float,
    clip_end: float,
    fade_seconds: float,
    word_highlight: bool,
) -> Tuple[str, Optional[str]]:
    """Locked framing + caption burn-in fused into a single ffmpeg pass.

    Returns (out_path, captions_error). captions_error is None on success;
    on any caption-related failure, out_path is still produced via a
    crop-only fallback pass and captions_error carries the failure reason —
    mirroring the existing captions_error contract used elsewhere in this
    module (see crop_highlights_local).
    """
    x0, y0, crop_w, crop_h = _compute_locked_crop_box(in_path, aspect_ratio)

    ass_path = out_path + ".ass"
    try:
        build_ass_file(
            transcript_segments, clip_start, clip_end,
            crop_w, crop_h, ass_path,
            fade_seconds=fade_seconds, word_highlight=word_highlight,
        )
    except CaptionError as e:
        _apply_crop(in_path, out_path, x0, y0, crop_w, crop_h)
        return out_path, str(e)

    try:
        _apply_crop(in_path, out_path, x0, y0, crop_w, crop_h, ass_path=ass_path)
        return out_path, None
    except CaptionError as e:
        _apply_crop(in_path, out_path, x0, y0, crop_w, crop_h)
        return out_path, str(e)
    finally:
        if os.path.exists(ass_path):
            os.remove(ass_path)
```

- [ ] **Step 3: Rewire `crop_clip_local`**

Replace `crop_clip_local` (lines 399-423 of the original file) with:

```python
def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    framing: str = "locked",
    transcript_segments: Optional[List[Dict]] = None,
    caption_fade_duration: float = 0.3,
    word_highlight: bool = True,
) -> Tuple[str, Optional[str]]:
    """Cut + reframe one highlight, returning (local mp4 path, captions_error).

    captions_error is None when no caption burn-in was attempted, or it
    succeeded; it carries an error message when transcript_segments were
    given for locked framing but the fused caption burn-in failed (out_path
    is still a valid, uncaptioned clip via a crop-only fallback).

    framing="locked" (default): static speaker-centered crop for the whole
    clip. When transcript_segments is given, captions are burned in the
    same ffmpeg pass as the crop.
    framing="adaptive": cursor/person-aware crop for screen-recording
    content that alternates between facecam and screen activity; captions
    (if any) are the caller's responsibility, burned in a separate pass.
    """
    cut_path = out_path + ".cut.mp4"
    captions_error: Optional[str] = None
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        if framing == "adaptive":
            _reframe_vertical_adaptive(cut_path, out_path, aspect_ratio)
        elif transcript_segments:
            out_path, captions_error = _reframe_vertical_with_captions(
                cut_path, out_path, aspect_ratio,
                transcript_segments, start_time, end_time,
                caption_fade_duration, word_highlight,
            )
        else:
            _reframe_vertical(cut_path, out_path, aspect_ratio)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path, captions_error
```

- [ ] **Step 4: Rewire `crop_highlights_local`**

Replace the body of the `for i, h in enumerate(highlights, 1):` loop in `crop_highlights_local` (lines 440-476 of the original file) with:

```python
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"Short-{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            # Locked framing fuses captions into the crop pass; adaptive
            # framing burns them separately below (its crop moves per-frame,
            # so it can't be expressed as a static ffmpeg crop filter).
            segs_for_fusion = transcript_segments if (captions and framing != "adaptive") else None
            out_path, captions_error = crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                framing=framing,
                transcript_segments=segs_for_fusion,
                caption_fade_duration=caption_fade_duration,
                word_highlight=word_highlight,
            )
            entry = {**h, "clip_url": out_path}
            if captions_error:
                entry["captions_error"] = captions_error

            if captions and transcript_segments and framing == "adaptive":
                captioned_path = out_path + ".captioned.mp4"
                try:
                    burn_captions(
                        out_path,
                        transcript_segments,
                        float(h["start_time"]),
                        float(h["end_time"]),
                        captioned_path,
                        fade_seconds=caption_fade_duration,
                        word_highlight=word_highlight,
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

- [ ] **Step 5: Update the two existing tests that assumed locked-mode captions went through `burn_captions`**

These two tests in `tests/test_local_clipper.py` monkeypatch `shorts_generator.local.clipper.burn_captions` to simulate caption success/failure for the **default (locked) framing** — after this task, locked framing no longer calls `burn_captions` at all (it calls `build_ass_file` fused into the crop pass). Update both to target the new fusion entry point.

Replace `test_caption_failure_falls_back_to_plain_clip`:

```python
def test_caption_failure_falls_back_to_plain_clip(tmp_path, synthetic_source, monkeypatch):
    def _raise(*args, **kwargs):
        raise captions_module.CaptionError("boom")

    monkeypatch.setattr("shorts_generator.local.clipper.build_ass_file", _raise)

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

Replace `test_word_highlight_flag_forwarded_to_burn`:

```python
def test_word_highlight_flag_forwarded_to_burn(tmp_path, synthetic_source, monkeypatch):
    from shorts_generator.captions import build_ass_file as real_build_ass_file

    captured = {}

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        return real_build_ass_file(*args, **kwargs)

    monkeypatch.setattr("shorts_generator.local.clipper.build_ass_file", _spy)
    crop_highlights_local(
        synthetic_source, [_highlight()], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
        word_highlight=False,
    )
    assert captured["word_highlight"] is False
```

- [ ] **Step 6: Add a dimension-correctness test for the fused output**

Add to `tests/test_local_clipper.py` (needs `_probe_resolution` imported — add `from shorts_generator.captions import _probe_resolution` near the top alongside the other imports):

```python
def test_locked_framing_fused_output_matches_target_aspect_ratio(tmp_path, synthetic_source):
    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source,
        [_highlight()],
        aspect_ratio="9:16",
        out_dir=out_dir,
        transcript_segments=_segments(),
    )

    width, height = _probe_resolution(results[0]["clip_url"])
    assert abs((width / height) - (9 / 16)) < 0.01
```

- [ ] **Step 7: Run the local-clipper and captions suites**

Run: `python -m pytest tests/test_local_clipper.py tests/test_captions.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "perf: fuse crop + caption burn-in into one ffmpeg pass (locked framing)

Locked framing (the default) now does cut -> crop[+subtitles] -> done,
dropping the OpenCV VideoWriter reframe pass and its separate audio-mux
step. Cuts video re-encode count on the default captioned path from 3 to
2. Adaptive framing (per-frame moving crop) keeps its existing OpenCV
path and separate caption burn-in, unchanged."
```

---

### Task 4: `cap.grab()` skip-decode in `_compute_locked_crop_box`

**Problem:** pass 1 (extracted in Task 3) calls `cap.read()` — a full decode — on every frame, even though the face cascade only runs on every `sample_stride`-th frame. Most decoded frames are thrown away unused.

**Files:**
- Modify: `shorts_generator/local/clipper.py` (`_compute_locked_crop_box`, added in Task 3)
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write a test asserting `grab()` is used for skipped frames**

Add to `tests/test_local_clipper.py`:

```python
def test_compute_locked_crop_box_uses_grab_for_skipped_frames(synthetic_source, monkeypatch):
    import cv2
    from shorts_generator.local.clipper import _compute_locked_crop_box

    counts = {"read": 0, "grab": 0}
    original_read = cv2.VideoCapture.read
    original_grab = cv2.VideoCapture.grab

    def counting_read(self, *a, **kw):
        counts["read"] += 1
        return original_read(self, *a, **kw)

    def counting_grab(self, *a, **kw):
        counts["grab"] += 1
        return original_grab(self, *a, **kw)

    monkeypatch.setattr(cv2.VideoCapture, "read", counting_read)
    monkeypatch.setattr(cv2.VideoCapture, "grab", counting_grab)

    _compute_locked_crop_box(synthetic_source, "9:16")

    # synthetic_source is a 6s @ 24fps clip (~144 frames); stride = max(1, 24 // 5) = 4
    # -> full decode (read) should happen on ~36 sampled frames, everything
    # else should be a cheap grab().
    assert counts["grab"] > 0
    assert counts["read"] <= 40
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_local_clipper.py::test_compute_locked_crop_box_uses_grab_for_skipped_frames -v`
Expected: FAIL — `counts["grab"] > 0` fails (current code never calls `grab()`, so `counts["read"]` will be ~144, comfortably over the `<= 40` bound too).

- [ ] **Step 3: Implement grab-skip decoding**

In `shorts_generator/local/clipper.py`, replace the sampling loop inside `_compute_locked_crop_box`:

```python
    sample_stride = max(1, int(fps // 5))
    sample_centers: List[Tuple[int, int]] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                sample_centers.append((x + w // 2, y + h // 2))
        frame_idx += 1
```

with:

```python
    sample_stride = max(1, int(fps // 5))
    sample_centers: List[Tuple[int, int]] = []
    frame_idx = 0
    while True:
        if frame_idx % sample_stride == 0:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                sample_centers.append((x + w // 2, y + h // 2))
        else:
            ret = cap.grab()
            if not ret:
                break
        frame_idx += 1
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `python -m pytest tests/test_local_clipper.py::test_compute_locked_crop_box_uses_grab_for_skipped_frames -v`
Expected: PASS

- [ ] **Step 5: Run the full local-clipper suite**

Run: `python -m pytest tests/test_local_clipper.py -v`
Expected: all PASS (crop box result is unchanged — same frames sampled, same order, only the decode mechanism for skipped frames changed).

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "perf: use cap.grab() for skipped frames in locked-mode pass 1

Only fully decodes the ~1-in-N frames the face cascade actually samples;
skipped frames are advanced with grab() instead of a full read()."
```

---

### Task 5: `LOCAL_WHISPER_BEAM_SIZE` config (faster-whisper `beam_size` 5 → 1)

**Problem:** `local/transcriber.py:92` hardcodes `beam_size=5`. Transcription runs once over the *whole* source video, making it a major share of total CPU time for a run in `mode=local` — `beam_size=1` (greedy decoding) is a real ~1.5x CPU reduction with negligible accuracy loss on clear speech.

**Files:**
- Create: `tests/test_config.py`
- Modify: `shorts_generator/config.py`
- Modify: `shorts_generator/local/transcriber.py:1-12` (imports), `:89-95` (`transcribe_kwargs`)
- Modify: `tests/test_local_transcriber.py`

- [ ] **Step 1: Write the failing config test**

Create `tests/test_config.py`:

```python
import shorts_generator.config as config
from shorts_generator.config import _parse_positive_int


def test_parse_positive_int_valid():
    assert _parse_positive_int("3", default=1) == 3


def test_parse_positive_int_invalid_falls_back():
    assert _parse_positive_int("abc", default=1) == 1


def test_parse_positive_int_zero_or_negative_falls_back():
    assert _parse_positive_int("0", default=5) == 5
    assert _parse_positive_int("-2", default=5) == 5


def test_local_whisper_beam_size_defaults_to_one():
    assert config.LOCAL_WHISPER_BEAM_SIZE >= 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ImportError: cannot import name '_parse_positive_int'`

- [ ] **Step 3: Add the helper and `LOCAL_WHISPER_BEAM_SIZE` to config.py**

In `shorts_generator/config.py`, add this helper near the top (after the `load_dotenv()` call, before its first use):

```python
def _parse_positive_int(value: str, default: int) -> int:
    """Parse an env-var string as a positive int; fall back to `default` on
    anything non-numeric or <= 0."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default
```

Then add, right after the existing `LOCAL_WHISPER_DEVICE` line:

```python
LOCAL_WHISPER_BEAM_SIZE = _parse_positive_int(os.getenv("LOCAL_WHISPER_BEAM_SIZE", "1"), default=1)
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 5: Write the failing transcriber test**

Add to `tests/test_local_transcriber.py`:

```python
def test_transcribe_local_forwards_beam_size_from_config(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")

    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWhisperModel)
    monkeypatch.setattr(tr, "LOCAL_WHISPER_BEAM_SIZE", 1)

    tr.transcribe_local(str(media))

    assert _FakeWhisperModel.last_transcribe_kwargs["beam_size"] == 1
```

- [ ] **Step 6: Run it to verify it fails**

Run: `python -m pytest tests/test_local_transcriber.py::test_transcribe_local_forwards_beam_size_from_config -v`
Expected: FAIL — `monkeypatch.setattr(tr, "LOCAL_WHISPER_BEAM_SIZE", 1)` raises `AttributeError` because `local/transcriber.py` doesn't import that name yet.

- [ ] **Step 7: Wire `LOCAL_WHISPER_BEAM_SIZE` into `local/transcriber.py`**

Change the import line:

```python
from ..config import LOCAL_WHISPER_DEVICE, LOCAL_WHISPER_MODEL
```

to:

```python
from ..config import LOCAL_WHISPER_BEAM_SIZE, LOCAL_WHISPER_DEVICE, LOCAL_WHISPER_MODEL
```

Then in `transcribe_kwargs` (inside `transcribe_local`), change:

```python
    transcribe_kwargs = {
        "audio": media_path,
        "language": language,
        "beam_size": 5,
        "condition_on_previous_text": False,
        "word_timestamps": True,
    }
```

to:

```python
    transcribe_kwargs = {
        "audio": media_path,
        "language": language,
        "beam_size": LOCAL_WHISPER_BEAM_SIZE,
        "condition_on_previous_text": False,
        "word_timestamps": True,
    }
```

- [ ] **Step 8: Run the new test to verify it passes**

Run: `python -m pytest tests/test_local_transcriber.py::test_transcribe_local_forwards_beam_size_from_config -v`
Expected: PASS

- [ ] **Step 9: Run the full transcriber and config suites**

Run: `python -m pytest tests/test_local_transcriber.py tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 10: Commit**

```bash
git add shorts_generator/config.py shorts_generator/local/transcriber.py tests/test_config.py tests/test_local_transcriber.py
git commit -m "perf: make faster-whisper beam_size configurable, default 1

Greedy decoding (beam_size=1) cuts CPU time on the whole-video
transcription pass ~1.5x vs the previous hardcoded beam_size=5, with
negligible accuracy loss on clear speech. Override with
LOCAL_WHISPER_BEAM_SIZE if a specific source needs more careful decoding."
```

---

### Task 6: Adaptive framing — stride face/cursor detection, cheaper resize

**Problem:** `_classify_frames` (screen-recording/adaptive framing) runs Haar cascade face detection and cursor-blob detection on **every** frame, unlike the locked path's pass 1 which already strides. Separately, `_reframe_vertical_adaptive` resizes every output frame with `cv2.INTER_LANCZOS4`, the most expensive standard OpenCV interpolation kernel.

**Files:**
- Modify: `shorts_generator/local/clipper.py:178-204` (`_classify_frames`), `:281-396` (`_reframe_vertical_adaptive`)
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write a test asserting strided detection**

Add to `tests/test_local_clipper.py`:

```python
def test_classify_frames_strides_face_detection(synthetic_source, monkeypatch):
    import cv2
    from shorts_generator.local.clipper import _classify_frames

    counts = {"n": 0}
    original = cv2.CascadeClassifier.detectMultiScale

    def counting(self, *a, **kw):
        counts["n"] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(cv2.CascadeClassifier, "detectMultiScale", counting)

    cap = cv2.VideoCapture(synthetic_source)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    _classify_frames(cap, src_w, fps)
    cap.release()

    # synthetic_source is 6s @ 24fps (~144 frames); stride = max(1, 24 // 5) = 4
    assert 0 < counts["n"] <= (total_frames // 4) + 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_local_clipper.py::test_classify_frames_strides_face_detection -v`
Expected: FAIL — two failure modes are both acceptable evidence the test is red for the right reason: either a `TypeError` (current `_classify_frames(cap, src_w)` doesn't accept a third `fps` argument yet) or, if you temporarily call it with two args to check, `counts["n"]` will be ~144 (every frame), well over the stride bound.

- [ ] **Step 3: Add stride to `_classify_frames`**

Replace `_classify_frames` (lines 178-204):

```python
def _classify_frames(cap, src_w: int, fps: float) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
    """Per frame: raw class ("person" | "cursor") + its raw anchor point.

    Face/cursor detection only runs every `sample_stride`-th frame (the
    same ~5/sec cadence as locked framing's pass 1); skipped frames hold
    the last sampled (cls, anchor) instead of re-running detection — the
    existing hysteresis (MODE_DWELL_SECONDS) already tolerates sampling
    this much coarser than per-frame.
    """
    import cv2  # type: ignore

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    sample_stride = max(1, int(fps // 5))
    raw: List[Tuple[str, Optional[Tuple[int, int]]]] = []
    prev_gray, last_cursor = None, None
    last_cls, last_anchor = "cursor", None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            anchor = None
            cls = "cursor"
            if len(faces):
                x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                if fw > PERSON_FACE_MIN_W_FRAC * src_w:
                    cls, anchor = "person", (x + fw // 2, y + fh // 2)
            if cls == "cursor":
                cur = _detect_cursor(gray, prev_gray)
                if cur:
                    last_cursor = cur
                anchor = last_cursor
            prev_gray = gray
            last_cls, last_anchor = cls, anchor
        raw.append((last_cls, last_anchor))
        frame_idx += 1
    return raw
```

- [ ] **Step 4: Update the one call site**

In `_reframe_vertical_adaptive`, change:

```python
    raw = _classify_frames(cap, src_w)
```

to:

```python
    raw = _classify_frames(cap, src_w, fps)
```

(`fps` is already computed a few lines earlier in `_reframe_vertical_adaptive` — no new variable needed.)

- [ ] **Step 5: Swap the interpolation kernel**

In `_reframe_vertical_adaptive`, change:

```python
        writer.write(cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4))
```

to:

```python
        writer.write(cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA))
```

- [ ] **Step 6: Run the new test to verify it passes**

Run: `python -m pytest tests/test_local_clipper.py::test_classify_frames_strides_face_detection -v`
Expected: PASS

- [ ] **Step 7: Run the full local-clipper suite**

Run: `python -m pytest tests/test_local_clipper.py -v`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "perf: stride adaptive-mode face/cursor detection, cheaper resize

_classify_frames now samples at the same ~5/sec cadence as locked
framing's pass 1 instead of running Haar cascade + cursor-blob detection
on every frame. cv2.INTER_LANCZOS4 (most expensive OpenCV kernel) swapped
for cv2.INTER_AREA (OpenCV's recommended, cheaper choice for
downscaling) in the adaptive resize."
```

---

### Task 7: Thread pinning (insurance against cgroup/container core-count misreporting)

**Problem:** on a VPS whose 2 vCPUs are correctly reported to the OS, libx264 and faster-whisper's ctranslate2 backend already use both cores — this task does **not** speed anything up on a correctly-reporting box. Its value is guarding against a container/cgroup that misreports available core count, which can cause thread oversubscription and contention. Opt-in, off by default.

**Files:**
- Modify: `shorts_generator/config.py`
- Modify: `shorts_generator/local/clipper.py` (`_cut_subclip`, `_apply_crop`, `_reframe_vertical_adaptive`'s mux call)
- Modify: `shorts_generator/captions.py` (`burn_captions`'s ffmpeg call)
- Modify: `shorts_generator/local/transcriber.py` (`WhisperModel(...)` call)
- Test: `tests/test_config.py`, `tests/test_local_transcriber.py`

- [ ] **Step 1: Write the failing config tests**

Add to `tests/test_config.py`:

```python
def test_ffmpeg_thread_args_empty_when_unset(monkeypatch):
    monkeypatch.setattr(config, "VPS_CPU_THREADS", 0)
    assert config.ffmpeg_thread_args() == []


def test_ffmpeg_thread_args_set_when_configured(monkeypatch):
    monkeypatch.setattr(config, "VPS_CPU_THREADS", 2)
    assert config.ffmpeg_thread_args() == ["-threads", "2"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.config' has no attribute 'VPS_CPU_THREADS'`

- [ ] **Step 3: Add `VPS_CPU_THREADS` and `ffmpeg_thread_args()` to config.py**

In `shorts_generator/config.py`, add right after `LOCAL_WHISPER_BEAM_SIZE`:

```python
# 0 = auto (let ffmpeg / ctranslate2 pick, matches their own default
# convention). Set explicitly only if a container/cgroup misreports the
# available core count and you observe thread oversubscription.
VPS_CPU_THREADS = _parse_positive_int(os.getenv("VPS_CPU_THREADS", "0"), default=0)


def ffmpeg_thread_args() -> "list[str]":
    """`-threads N` for ffmpeg's argv if VPS_CPU_THREADS is set, else []."""
    return ["-threads", str(VPS_CPU_THREADS)] if VPS_CPU_THREADS > 0 else []
```

(Note: trace this against `_parse_positive_int`'s body from Task 5 before moving on — `_parse_positive_int("0", default=0)` parses `"0"` to the int `0` cleanly, then hits the `parsed if parsed >= 1 else default` line: `0 >= 1` is False, so it falls through to `default`, which is also `0` here. Either way the result is `0`, which is exactly the "unset/auto" sentinel `ffmpeg_thread_args()` checks for — this is correct as written, not a bug.)

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 5: Wire into `local/transcriber.py`**

Change the import line:

```python
from ..config import LOCAL_WHISPER_BEAM_SIZE, LOCAL_WHISPER_DEVICE, LOCAL_WHISPER_MODEL
```

to:

```python
from ..config import LOCAL_WHISPER_BEAM_SIZE, LOCAL_WHISPER_DEVICE, LOCAL_WHISPER_MODEL, VPS_CPU_THREADS
```

Change:

```python
    model = WhisperModel(LOCAL_WHISPER_MODEL, device=device, compute_type=compute_type)
```

to:

```python
    model = WhisperModel(LOCAL_WHISPER_MODEL, device=device, compute_type=compute_type, cpu_threads=VPS_CPU_THREADS)
```

- [ ] **Step 6: Write the failing transcriber test**

First, extend the `_FakeWhisperModel` fixture in `tests/test_local_transcriber.py` to capture constructor kwargs too:

```python
class _FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel; records the kwargs it was
    constructed and called with so tests can assert on them."""

    def __init__(self, *args, **kwargs):
        _FakeWhisperModel.last_init_kwargs = kwargs

    def transcribe(self, **kwargs):
        _FakeWhisperModel.last_transcribe_kwargs = kwargs
        segments = [
            _FakeSegment(
                0.0, 2.0, "the quick fox",
                [
                    _FakeWord(0.0, 0.5, "the"),
                    _FakeWord(0.5, 1.2, "quick"),
                    _FakeWord(1.2, 2.0, "fox"),
                ],
            )
        ]
        info = SimpleNamespace(duration=2.0)
        return iter(segments), info
```

Then add:

```python
def test_transcribe_local_forwards_cpu_threads_from_config(tmp_path, monkeypatch):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")

    import faster_whisper
    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWhisperModel)
    monkeypatch.setattr(tr, "VPS_CPU_THREADS", 2)

    tr.transcribe_local(str(media))

    assert _FakeWhisperModel.last_init_kwargs["cpu_threads"] == 2
```

- [ ] **Step 7: Run it to verify it fails, then passes**

Run: `python -m pytest tests/test_local_transcriber.py -v`
Expected: first run FAILS (`AttributeError` on `tr.VPS_CPU_THREADS` before Step 5, or `KeyError`/`AttributeError: last_init_kwargs` if Step 5 was already done but the fixture wasn't updated) — confirm you did Step 5 and the fixture update above, then re-run and expect all PASS.

- [ ] **Step 8: Wire into `local/clipper.py`'s ffmpeg calls**

Add to the imports at the top of `shorts_generator/local/clipper.py`:

```python
from ..config import LOCAL_OUTPUT_DIR, ffmpeg_thread_args
```

(replacing the existing `from ..config import LOCAL_OUTPUT_DIR` line).

In `_cut_subclip`, change:

```python
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", source_path,
        "-t", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
```

to:

```python
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *ffmpeg_thread_args(),
        "-ss", f"{start:.3f}",
        "-i", source_path,
        "-t", f"{end - start:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
```

In `_apply_crop`, change:

```python
    vf = f"crop={crop_w}:{crop_h}:{x0}:{y0}"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", in_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
```

to:

```python
    vf = f"crop={crop_w}:{crop_h}:{x0}:{y0}"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *ffmpeg_thread_args(),
        "-i", in_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
```

In `_reframe_vertical_adaptive`'s final mux command, change:

```python
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path
```

(this appears twice — once in `_reframe_vertical_adaptive`, once previously in the old `_reframe_vertical` which Task 3 already removed; only the `_reframe_vertical_adaptive` copy remains) to:

```python
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *ffmpeg_thread_args(),
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path
```

- [ ] **Step 9: Wire into `captions.py`'s `burn_captions` (benefits both local adaptive-mode captions and API-mode captions, which share this function)**

Add `ffmpeg_thread_args` to the import in `shorts_generator/captions.py`:

```python
from .config import ffmpeg_thread_args
```

In `burn_captions`, change:

```python
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"subtitles={escaped_ass_path}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "copy",
            out_path,
        ]
```

to:

```python
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            *ffmpeg_thread_args(),
            "-i", video_path,
            "-vf", f"subtitles={escaped_ass_path}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "copy",
            out_path,
        ]
```

- [ ] **Step 10: Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: all PASS (thread args are empty by default, so every existing ffmpeg invocation is byte-for-byte unchanged unless `VPS_CPU_THREADS` is set).

- [ ] **Step 11: Commit**

```bash
git add shorts_generator/config.py shorts_generator/local/clipper.py shorts_generator/local/transcriber.py shorts_generator/captions.py tests/test_config.py tests/test_local_transcriber.py
git commit -m "perf: opt-in thread pinning via VPS_CPU_THREADS

Insurance against a container/cgroup misreporting available core count,
not a speedup on a correctly-reporting box (ffmpeg/ctranslate2 already
use all visible cores by default). Off (0 = auto) unless set."
```

---

### Task 8: Document VPS deployment settings

**Files:**
- Modify: `README.md`
- Modify: `.env.example`

- [ ] **Step 1: Add the new env vars to `.env.example`**

Add after the existing `LOCAL_OUTPUT_DIR=output` line:

```
LOCAL_WHISPER_BEAM_SIZE=1         # 1=greedy (fastest), up to 5=more careful/slower
VPS_CPU_THREADS=0                 # 0=auto; set to your VPS's vCPU count only if you see thread oversubscription
```

- [ ] **Step 2: Add a "Running on a resource-constrained VPS" section to README.md**

Add a new section (place it near the existing `--mode local` documentation — search the file for the local-mode section and add this as a sibling subsection):

```markdown
### Running on a resource-constrained VPS

`--mode local` runs the whole pipeline (yt-dlp, faster-whisper, OpenCV
framing, ffmpeg) on the machine you're hosting on. On a small VPS (e.g. 2
vCPUs / 8GB RAM), a few `.env` settings matter:

- `LOCAL_WHISPER_MODEL=base` (the default) is the right size for 2 cores —
  larger models cost meaningfully more CPU time and RAM for marginal
  accuracy gains on clear speech.
- `LOCAL_WHISPER_BEAM_SIZE=1` (the default) uses greedy decoding, ~1.5x
  faster than beam search on CPU, with negligible accuracy loss on clear
  speech. Raise it only if you're seeing bad transcripts on a specific
  source.
- `LOCAL_WHISPER_VAD_FILTER=true` skips silence during transcription — a
  real speed win on content with long pauses, but it's aggressive on
  music/mixed audio, so it's off by default. Worth trying if your source
  has a lot of dead air.
- `VPS_CPU_THREADS` is insurance, not a speed setting — on a VPS that
  correctly reports its core count, ffmpeg and faster-whisper already use
  every core by default. Only set it (to your vCPU count) if you notice
  thread oversubscription, which can happen in some container/cgroup
  setups.
- Memory is not a concern at 8GB with these settings — `faster-whisper`'s
  `base` model at `int8` uses roughly 1GB, and the framing/reframe pipeline
  streams frame-by-frame rather than buffering the video in memory.
- **The dashboard has no authentication.** If you're exposing it on a
  public IP, anyone who finds the URL can trigger a pipeline run or delete
  your run history. Put it behind a reverse proxy with auth, a VPN, or an
  IP allowlist — this is not handled by the application itself.
```

- [ ] **Step 3: Commit**

```bash
git add README.md .env.example
git commit -m "docs: add VPS deployment notes for mode=local

Recommended .env settings for a 2-core/8GB box, and an explicit callout
that the dashboard has no built-in authentication."
```

---

## Final verification

- [ ] Run the entire test suite once more end to end:

```bash
python -m pytest tests/ -v
```
Expected: all PASS.

- [ ] Sanity-check the fused locked-framing path end-to-end with real captions using the CLI (adjust the path to any local test video you have, or skip if none is available):

```bash
python main.py /path/to/a/short/local/test-video.mp4 --mode local --num-clips 1 --framing locked
```
Expected: completes without error, produces `output/<title>/Shorts/Short-01.mp4` with burned-in captions, and the console output shows one ffmpeg crop+subtitles pass per clip (no `.silent.mp4` intermediate file left behind).
