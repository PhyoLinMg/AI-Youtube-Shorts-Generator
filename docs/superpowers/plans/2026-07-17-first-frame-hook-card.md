# First-Frame Hook Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Kill the static talking-head frame 0 by compositing a bold on-screen hook card (a motion-picked "striking" still + drawtext) over the first 1.5 seconds of every generated short, and tune the highlight-generation prompt to supply the short on-screen hook text this needs.

**Architecture:** A new `shorts_generator/hook_card.py` module does frame selection (OpenCV motion + sharpness scoring, reusing the frame-diff primitive already in `local/clipper.py`) and ffmpeg compositing (drawtext + overlay with a time-gated `enable` window, so total duration/audio/caption timestamps never change). It is wired into both `clipper.py` (API mode) and `local/clipper.py` (local mode) as two phases: pick+extract the still from the clean pre-caption crop, then composite the card as the last pass onto the captioned clip. `highlights.py` gains a new `on_screen_hook` LLM field (≤7 words) that supplies the card's text.

**Tech Stack:** Python, OpenCV (`cv2`), ffmpeg (`drawtext`, `overlay`, `loop` filters), pytest with real synthetic ffmpeg fixtures (no subprocess mocking, matching this repo's existing test style).

**Spec:** `docs/superpowers/specs/2026-07-17-first-frame-hook-card-design.md`

**Validated before writing this plan:** the exact drawtext+overlay filter graph below was smoke-tested against a real synthetic clip on this machine (Anton font downloaded and rendered correctly, bold, boxed, two-line wrap; card visible in `[0, 1.5s)` and cleanly gone after; colon/apostrophe escaping tested; the `pick_striking_frame` motion+sharpness algorithm was run against a synthetic clip with a deliberately blurry high-motion segment and a sharp high-motion segment, and correctly picked the sharp one). The commands in this plan reproduce that validated behavior.

---

### Task 1: Bundle the Anton bold font

**Files:**
- Create: `shorts_generator/assets/fonts/Anton-Regular.ttf`
- Create: `shorts_generator/assets/fonts/OFL.txt`

- [ ] **Step 1: Create the directory and download the font + its license**

```bash
mkdir -p shorts_generator/assets/fonts
curl -sL -o shorts_generator/assets/fonts/Anton-Regular.ttf \
  "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf"
curl -sL -o shorts_generator/assets/fonts/OFL.txt \
  "https://github.com/google/fonts/raw/main/ofl/anton/OFL.txt"
```

- [ ] **Step 2: Verify the download**

Run: `file shorts_generator/assets/fonts/Anton-Regular.ttf && wc -c shorts_generator/assets/fonts/Anton-Regular.ttf`
Expected: `TrueType Font data` and a size around `170812` bytes (not an HTML error page — GitHub raw URLs return HTML on a bad path).

- [ ] **Step 3: Smoke-test the font actually renders bold via ffmpeg drawtext**

```bash
ffmpeg -y -loglevel error -f lavfi -i "color=c=gray:size=320x568:rate=24:duration=1" \
  -vf "drawtext=fontfile='shorts_generator/assets/fonts/Anton-Regular.ttf':text='TEST':fontsize=64:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10:x=(w-text_w)/2:y=(h-text_h)/2" \
  -frames:v 1 /tmp/hookcard_font_smoketest.png
```

Run: `file /tmp/hookcard_font_smoketest.png`
Expected: `PNG image data`, exit code 0. If this fails with a fontconfig/freetype error, stop and re-check the ffmpeg build (`ffmpeg -version` must show `--enable-libfreetype`) before continuing to any later task.

- [ ] **Step 4: Commit**

```bash
git add shorts_generator/assets/fonts/Anton-Regular.ttf shorts_generator/assets/fonts/OFL.txt
git commit -m "$(cat <<'EOF'
feat: bundle Anton bold font for the hook-card overlay

Single-weight OFL display font, already maximally bold, packaged so
drawtext's fontfile= doesn't depend on OS/fontconfig font resolution.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `hook_card.py` — errors, constants, `extract_frame`

**Files:**
- Create: `shorts_generator/hook_card.py`
- Test: `tests/test_hook_card.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hook_card.py
import os
import subprocess

import cv2
import pytest

from shorts_generator.hook_card import HookCardError, extract_frame


def _run(cmd):
    subprocess.run(cmd, check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def red_clip(tmp_path_factory):
    """A tiny solid-red 3s clip w/ audio — stands in for a final vertical crop."""
    tmp_dir = tmp_path_factory.mktemp("hookcard_src")
    path = str(tmp_dir / "clip.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=red:size=270x480:rate=24:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", "-shortest",
        path,
    ])
    return path


def test_extract_frame_writes_readable_image(red_clip, tmp_path):
    out_path = str(tmp_path / "frame.jpg")
    result = extract_frame(red_clip, 1.0, out_path)

    assert result == out_path
    assert os.path.exists(out_path)
    img = cv2.imread(out_path)
    assert img is not None
    assert img.shape[:2] == (480, 270)


def test_extract_frame_raises_hook_card_error_on_bad_video(tmp_path):
    out_path = str(tmp_path / "frame.jpg")
    with pytest.raises(HookCardError):
        extract_frame(str(tmp_path / "missing.mp4"), 1.0, out_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hook_card.py -v`
Expected: `ModuleNotFoundError: No module named 'shorts_generator.hook_card'`

- [ ] **Step 3: Write the module**

```python
# shorts_generator/hook_card.py
"""Pick a striking frame from a clip and composite a bold hook-text card
over its first ~1.5 seconds.

Two-phase API:
  - pick_striking_frame + extract_frame run against the CLEAN, pre-caption
    crop, so the chosen still never has caption text baked into it.
  - render_card_overlay runs last, against the captioned clip, and is the
    only step whose output actually lands in the final mp4.
"""
import os
import subprocess
from typing import List, Tuple

import cv2  # type: ignore

from .captions import _probe_resolution

HOOK_CARD_DURATION = 1.5        # seconds the card stays on screen
SKIP_SECONDS = 0.5              # never pick a frame from the resting opening
SAMPLE_FPS = 5                  # motion/sharpness sampling rate
MOTION_TOP_QUANTILE = 0.75      # only the top 25% of frames by motion score are sharpness candidates
STILL_INPUT_BUFFER = 0.5        # extra seconds on the looped still input beyond `duration`

FONT_PATH = os.path.join(os.path.dirname(__file__), "assets", "fonts", "Anton-Regular.ttf")


class HookCardError(RuntimeError):
    """Raised when hook-card frame selection or compositing fails; callers fall back to the pre-card clip."""


def extract_frame(video_path: str, timestamp: float, out_path: str) -> str:
    """Grab a single frame at `timestamp` as a still image."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-ss", f"{timestamp:.3f}",
        "-i", video_path,
        "-vframes", "1",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HookCardError(f"ffmpeg frame extraction failed: {e.stderr}") from e
    except OSError as e:
        raise HookCardError(f"ffmpeg frame extraction failed: {e}") from e
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hook_card.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/hook_card.py tests/test_hook_card.py
git commit -m "$(cat <<'EOF'
feat: add hook_card.extract_frame

First piece of the hook-card module: grab a single still frame from a
clip at a given timestamp via ffmpeg.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `hook_card.py` — `pick_striking_frame`

**Files:**
- Modify: `shorts_generator/hook_card.py`
- Test: `tests/test_hook_card.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hook_card.py`:

```python
def _run(cmd):
    subprocess.run(cmd, check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def motion_clip(tmp_path_factory):
    """0-2s static gray (resting), 2-3s a heavily blurred bright flash
    (high motion, low sharpness), 3-4s a sharp high-contrast test pattern
    (high motion, high sharpness), 4-6s back to static gray."""
    tmp_dir = tmp_path_factory.mktemp("hookcard_motion")

    seg0 = str(tmp_dir / "seg0.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=gray:size=320x568:rate=24:duration=2",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", seg0,
    ])

    seg1 = str(tmp_dir / "seg1.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=white:size=320x568:rate=24:duration=1",
        "-vf", "gblur=sigma=20",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", seg1,
    ])

    seg2 = str(tmp_dir / "seg2.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x568:rate=24:duration=1",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", seg2,
    ])

    seg3 = str(tmp_dir / "seg3.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=gray:size=320x568:rate=24:duration=2",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", seg3,
    ])

    list_path = str(tmp_dir / "list.txt")
    with open(list_path, "w") as f:
        for seg in (seg0, seg1, seg2, seg3):
            f.write(f"file '{seg}'\n")

    out_path = str(tmp_dir / "motion.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_path,
    ])
    return out_path


def test_pick_striking_frame_prefers_sharp_over_blurry_motion(motion_clip):
    from shorts_generator.hook_card import pick_striking_frame

    ts = pick_striking_frame(motion_clip)

    # 2-3s is high-motion but blurred (sharpness ~0); 3-4s is high-motion
    # AND sharp. The sharpness tiebreaker must pick from the sharp window.
    assert 3.0 <= ts < 4.0


def test_pick_striking_frame_falls_back_when_too_short_to_sample(tmp_path):
    from shorts_generator.hook_card import pick_striking_frame

    short_path = str(tmp_path / "short.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:size=320x568:rate=24:duration=0.3",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", short_path,
    ])

    # 0.3s @ 24fps = 8 total frames; skip_seconds=0.5 -> skip_frames=12 > 8,
    # so zero frames survive the skip and the fallback must fire (a 1s clip
    # was tried here originally and turned out to still yield exactly 2
    # post-skip samples, which does NOT trigger the <2 fallback -- verified
    # live before locking this duration in).
    assert pick_striking_frame(short_path, skip_seconds=0.5) == 0.5


def test_pick_striking_frame_raises_hook_card_error_on_bad_video(tmp_path):
    from shorts_generator.hook_card import pick_striking_frame

    with pytest.raises(HookCardError):
        pick_striking_frame(str(tmp_path / "missing.mp4"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hook_card.py -v -k pick_striking_frame`
Expected: `ImportError: cannot import name 'pick_striking_frame'`

- [ ] **Step 3: Implement `pick_striking_frame`**

Add to `shorts_generator/hook_card.py`, after the imports/constants and before `extract_frame`:

```python
def pick_striking_frame(
    video_path: str,
    skip_seconds: float = SKIP_SECONDS,
    sample_fps: float = SAMPLE_FPS,
) -> float:
    """Return the timestamp of the most "striking" frame in `video_path`:
    among the top-quartile motion-scoring sampled frames (frame-to-frame
    pixel diff), the sharpest one (Laplacian variance) — this avoids
    landing on a motion-blurred frame just because it had the single
    highest motion score. Falls back to `skip_seconds` if the clip is too
    short to yield 2+ samples after skipping the opening.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HookCardError(f"could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / sample_fps)))
    skip_frames = int(round(skip_seconds * fps))

    samples: List[Tuple[float, object]] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= skip_frames and frame_idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            samples.append((frame_idx / fps, gray))
        frame_idx += 1
    cap.release()

    if len(samples) < 2:
        return skip_seconds

    motion_scores = [0.0]
    for i in range(1, len(samples)):
        diff = cv2.absdiff(samples[i][1], samples[i - 1][1])
        motion_scores.append(float(diff.sum()))

    ranked = sorted(range(len(samples)), key=lambda i: motion_scores[i], reverse=True)
    top_n = max(1, int(round(len(ranked) * (1 - MOTION_TOP_QUANTILE))))
    candidates = ranked[:top_n]

    best_idx = max(candidates, key=lambda i: cv2.Laplacian(samples[i][1], cv2.CV_64F).var())
    return samples[best_idx][0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hook_card.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/hook_card.py tests/test_hook_card.py
git commit -m "$(cat <<'EOF'
feat: add hook_card.pick_striking_frame

Motion-diff scan (reusing the frame-diff primitive already in
local/clipper.py) with a sharpness tiebreaker among the top-motion
candidates, so the chosen still is dynamic but not motion-blurred.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `hook_card.py` — `render_card_overlay`

**Files:**
- Modify: `shorts_generator/hook_card.py`
- Test: `tests/test_hook_card.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hook_card.py`:

```python
@pytest.fixture(scope="module")
def white_still(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("hookcard_still")
    path = str(tmp_dir / "still.jpg")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=white:size=270x480",
        "-frames:v", "1", path,
    ])
    return path


def _center_pixel_bgr(video_path, timestamp, tmp_path, name):
    frame_path = str(tmp_path / name)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{timestamp}",
          "-i", video_path, "-vframes", "1", frame_path])
    img = cv2.imread(frame_path)
    h, w = img.shape[:2]
    region = img[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5]
    return region.reshape(-1, 3).mean(axis=0)  # BGR


def test_render_card_overlay_shows_card_then_reveals_live_footage(red_clip, white_still, tmp_path):
    from shorts_generator.hook_card import render_card_overlay

    out_path = str(tmp_path / "out.mp4")
    result = render_card_overlay(red_clip, white_still, "TEST HOOK", out_path, duration=1.0)

    assert result == out_path
    assert os.path.exists(out_path)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out_path],
        capture_output=True, text=True, check=True,
    )
    assert float(probe.stdout.strip()) == pytest.approx(3.0, abs=0.2)

    during_card = _center_pixel_bgr(out_path, 0.3, tmp_path, "during.jpg")
    after_card = _center_pixel_bgr(out_path, 1.5, tmp_path, "after.jpg")

    # red_clip is pure red (BGR ~ [0, 0, 255]); the boxed white-still card
    # composited on top pulls the center pixel well away from pure red.
    assert during_card[2] < 215 or during_card[0] > 40 or during_card[1] > 40
    # After the card window the live red footage is back with nothing
    # composited on top of it.
    assert after_card[2] > 200 and after_card[0] < 40 and after_card[1] < 40


def test_render_card_overlay_wraps_long_hook_text_onto_two_lines(red_clip, white_still, tmp_path):
    from shorts_generator.hook_card import render_card_overlay

    out_path = str(tmp_path / "out.mp4")
    # Should not raise even with a 7-word hook (two-line wrap path).
    render_card_overlay(red_clip, white_still, "You Won't Believe: This Happened Today", out_path, duration=1.0)
    assert os.path.exists(out_path)


def test_render_card_overlay_raises_hook_card_error_on_missing_video(white_still, tmp_path):
    from shorts_generator.hook_card import render_card_overlay

    out_path = str(tmp_path / "out.mp4")
    with pytest.raises(HookCardError):
        render_card_overlay(str(tmp_path / "missing.mp4"), white_still, "HOOK", out_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hook_card.py -v -k render_card_overlay`
Expected: `ImportError: cannot import name 'render_card_overlay'`

- [ ] **Step 3: Implement `render_card_overlay`**

Add to `shorts_generator/hook_card.py`, after `pick_striking_frame` and before `extract_frame` (or after `extract_frame` — order among the three public functions doesn't matter, keep `extract_frame` where it already is):

```python
def _wrap_two_lines(text: str) -> List[str]:
    words = text.split()
    if len(words) <= 1:
        return [text]
    mid = (len(words) + 1) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def _escape_drawtext(text: str) -> str:
    """Escape for a single-quoted ffmpeg drawtext `text=` value. Colons
    still need escaping even inside single quotes (ffmpeg's filtergraph
    quoting doesn't shield them). Apostrophes are swapped for a
    typographic quote to dodge the filtergraph's fiddly nested-single-quote
    escaping entirely.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "’")
    return text


def render_card_overlay(
    video_path: str,
    still_path: str,
    hook_text: str,
    out_path: str,
    duration: float = HOOK_CARD_DURATION,
) -> str:
    """Composite `still_path` + bold `hook_text` over `video_path`'s first
    `duration` seconds. Run this last, against the already-captioned clip.
    Audio passes through untouched; total duration and resolution are
    preserved — only the video is affected, and only inside the window.
    """
    width, height = _probe_resolution(video_path)
    lines = _wrap_two_lines(hook_text)
    fontsize = max(24, round(height * 0.07))
    line_height = fontsize * 1.25
    block_height = line_height * len(lines)

    drawtext_filters = []
    for i, line in enumerate(lines):
        y_expr = f"(h-{block_height:.1f})/2+{i * line_height:.1f}"
        drawtext_filters.append(
            "drawtext="
            f"fontfile='{FONT_PATH}':text='{_escape_drawtext(line)}':"
            f"fontsize={fontsize}:fontcolor=white:"
            "box=1:boxcolor=black@0.55:boxborderw=16:"
            f"x=(w-text_w)/2:y={y_expr}:expansion=none"
        )

    filter_complex = (
        f"[1:v]scale={width}:{height}[stillv];"
        f"[stillv]{','.join(drawtext_filters)}[card];"
        f"[0:v][card]overlay=enable='between(t,0,{duration})'[v]"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-i", video_path,
        "-loop", "1", "-t", f"{duration + STILL_INPUT_BUFFER:.3f}", "-i", still_path,
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        "-shortest",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HookCardError(f"ffmpeg hook-card overlay failed: {e.stderr}") from e
    except OSError as e:
        raise HookCardError(f"ffmpeg hook-card overlay failed: {e}") from e
    return out_path
```

**Important implementation note:** do NOT use `-loop 1` with the `loop=loop=-1:size=1` video filter together — that combination (infinite input stream + infinite filter loop) was tried during spec validation and hung ffmpeg indefinitely (never terminated even with `-shortest` on the output). Bounding the still input with `-loop 1 -t <duration + buffer>` (finite input, no `loop` filter needed) is the version that was verified to work.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hook_card.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/hook_card.py tests/test_hook_card.py
git commit -m "$(cat <<'EOF'
feat: add hook_card.render_card_overlay

Last-pass compositor: loops the still image (bounded duration, not an
infinite loop — that combination hangs ffmpeg), draws up to 2 lines of
boxed bold hook text via the bundled Anton font, and overlays it onto
the clip only for [0, duration) via a time-gated enable expression.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Promote `opencv-python` to the base requirements

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements-local.txt`

**Context:** `hook_card.py` needs `cv2` in both `--mode api` and `--mode local` (it operates on the final local mp4 regardless of mode). `opencv-python` is currently only in `requirements-local.txt`. Per the spec decision, move it to the base file so `--mode api` installs stay correct (this does make API-mode installs heavier — accepted tradeoff, see spec).

- [ ] **Step 1: Edit `requirements.txt`**

Current content:
```
requests>=2.31
python-dotenv>=1.0
```

New content:
```
requests>=2.31
python-dotenv>=1.0
opencv-python>=4.8.0,<5
```

- [ ] **Step 2: Edit `requirements-local.txt`**

Current content:
```
-r requirements.txt

# Optional dependencies for --mode local.
yt-dlp>=2024.1.1
faster-whisper>=1.0.0
openai>=1.0.0
google-genai>=1.0.0
opencv-python>=4.8.0,<5
# torch is only needed if you want CUDA Whisper. CPU works without it.
# torch>=2.0
```

New content (drop the now-redundant `opencv-python` line — `-r requirements.txt` already pulls it in):
```
-r requirements.txt

# Optional dependencies for --mode local.
yt-dlp>=2024.1.1
faster-whisper>=1.0.0
openai>=1.0.0
google-genai>=1.0.0
# opencv-python is a base dependency now (shorts_generator/requirements.txt) —
# both modes need it for the hook-card frame picker.
# torch is only needed if you want CUDA Whisper. CPU works without it.
# torch>=2.0
```

- [ ] **Step 3: Verify the installed environment already satisfies this**

Run: `python3 -c "import cv2; print(cv2.__version__)"`
Expected: prints a version like `4.13.0` (already true in this dev environment; this step just confirms the requirements files now accurately describe what the code needs — no new install required here).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt requirements-local.txt
git commit -m "$(cat <<'EOF'
build: move opencv-python to base requirements

The hook-card frame picker needs cv2 in --mode api too, not just
--mode local. API-mode installs get heavier but stay correct.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `highlights.py` — `on_screen_hook` field

**Files:**
- Modify: `shorts_generator/highlights.py:42-63` (prompt), `:129-163` (`_sanitize_highlights`)
- Test: `tests/test_highlights.py` (new file — none exists yet for this module)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_highlights.py
from shorts_generator.highlights import _sanitize_highlights


def _raw_highlight(**overrides):
    base = {
        "title": "Big News",
        "start_time": 1.0,
        "end_time": 5.0,
        "score": 90,
        "hook_sentence": "This is the full hook sentence spoken in the clip.",
        "on_screen_hook": "WAIT FOR IT",
        "virality_reason": "because",
        "description": "desc",
    }
    base.update(overrides)
    return base


def test_sanitize_highlights_includes_on_screen_hook():
    cleaned = _sanitize_highlights([_raw_highlight()], duration=100.0)
    assert cleaned[0]["on_screen_hook"] == "WAIT FOR IT"


def test_sanitize_highlights_caps_on_screen_hook_length():
    cleaned = _sanitize_highlights([_raw_highlight(on_screen_hook="x" * 200)], duration=100.0)
    assert len(cleaned[0]["on_screen_hook"]) == 60


def test_sanitize_highlights_defaults_on_screen_hook_to_empty_string():
    raw = {"start_time": 1.0, "end_time": 5.0}
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["on_screen_hook"] == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_highlights.py -v`
Expected: `KeyError: 'on_screen_hook'`

- [ ] **Step 3: Update the prompt**

In `shorts_generator/highlights.py`, replace the `HIGHLIGHT_SYSTEM_PROMPT` string (lines 42-63) with:

```python
HIGHLIGHT_SYSTEM_PROMPT = """You are an elite short-form video editor who has studied thousands of viral clips on TikTok, Instagram Reels, and YouTube Shorts. You know exactly what makes viewers stop scrolling, watch to the end, and share.

{virality_criteria}

Content type: {content_type} | Density: {density}

Your task: identify the most viral-worthy highlights from the transcript.

Rules:
- Every highlight must open with a strong HOOK — a line that grabs attention within the first 3 seconds. start_time must land ON that hook line itself, never on preamble, silence, or filler before it — the clip opens cold, mid-energy, not with a slow windup
- Duration sweet spot: 45-90 seconds. Go shorter (20-44s) only for a perfect standalone one-liner. Go longer (91-180s) only when a story arc needs full context to land
- Never cut mid-sentence or mid-thought — each clip must feel complete and self-contained
- Clips must not overlap significantly with each other
- Score 0-100 on viral potential (not general quality)
- {num_clips_instruction}
- For each highlight, identify the single best "hook_sentence" — the opening line that would make someone stop scrolling
- Write an "on_screen_hook" — a short punchy fragment, 7 words or fewer, distinct from hook_sentence (it does NOT need to be a verbatim transcript line). This is bold text that gets overlaid on screen for the first 1.5 seconds, so it must work standalone with zero context: think thumbnail text, not a sentence
- Explain in one sentence why this clip is viral ("virality_reason")
- Write a "title" — max 100 characters, aggressive clickbait style (curiosity gap, numbers, shock value, "you won't believe", etc.) optimized to maximize clicks and views, while still being accurate to the clip's content
- Write a "description" — up to 300 words, original marketing copy (NOT a transcript line) built to maximize views and clicks: open with a strong curiosity- or CTA-driven hook, up to a few relevant emoji, then end with 15-30 relevant hashtags mixing broad reach tags (#shorts #viral #fyp #trending) with niche tags specific to the clip's topic

Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","on_screen_hook":"string","virality_reason":"string","description":"string"}}]}}"""
```

- [ ] **Step 4: Update `_sanitize_highlights`**

In `shorts_generator/highlights.py`, inside the `cleaned.append({...})` dict (around line 151-161), add one line after `"hook_sentence"`:

```python
        cleaned.append(
            {
                "title": str(item.get("title") or "Untitled Highlight").strip()[:100],
                "start_time": start,
                "end_time": end,
                "score": max(0, min(100, _coerce_int(item.get("score"), default=0))),
                "hook_sentence": str(item.get("hook_sentence") or "").strip(),
                "on_screen_hook": str(item.get("on_screen_hook") or "").strip()[:60],
                "virality_reason": str(item.get("virality_reason") or "").strip(),
                "description": str(item.get("description") or "").strip(),
            }
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_highlights.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/highlights.py tests/test_highlights.py
git commit -m "$(cat <<'EOF'
feat: add on_screen_hook field to highlight generation

New LLM field (<=7 words, distinct from the full-sentence hook_sentence)
feeds the hook-card overlay's on-screen text. Also reinforces the
existing hook-opens-in-first-3s rule: start_time must land on the hook
line itself, not on preamble/filler.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Wire into `local/clipper.py`

**Files:**
- Modify: `shorts_generator/local/clipper.py:1-24` (imports), `:440-491` (`crop_highlights_local`)
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_local_clipper.py`:

```python
from shorts_generator.hook_card import HookCardError


def _highlight_with_hook():
    return {**_highlight(), "on_screen_hook": "WATCH THIS"}


def test_hook_card_still_picked_before_caption_burn(tmp_path, synthetic_source, monkeypatch):
    """Regression test: the still MUST come from the clean pre-caption
    crop, not the captioned file (a still picked after caption burn would
    freeze that timestamp's burned-in caption line into the card)."""
    order = []

    def _fake_pick(video_path):
        order.append("pick")
        return 0.5

    def _fake_extract(video_path, ts, out_path):
        order.append("extract")
        with open(out_path, "wb") as f:
            f.write(b"fake still")
        return out_path

    def _fake_burn(*args, **kwargs):
        order.append("burn")
        import shutil
        shutil.copyfile(args[0], args[4])
        return args[4]

    def _fake_render(video_path, still_path, hook_text, out_path, duration=1.5):
        order.append("render")
        import shutil
        shutil.copyfile(video_path, out_path)
        return out_path

    monkeypatch.setattr(local_clipper_module, "pick_striking_frame", _fake_pick)
    monkeypatch.setattr(local_clipper_module, "extract_frame", _fake_extract)
    monkeypatch.setattr(local_clipper_module, "burn_captions", _fake_burn)
    monkeypatch.setattr(local_clipper_module, "render_card_overlay", _fake_render)

    crop_highlights_local(
        synthetic_source, [_highlight_with_hook()], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
    )

    assert order == ["pick", "extract", "burn", "render"]


def test_hook_card_skipped_when_flag_off(tmp_path, synthetic_source, monkeypatch):
    def _fail_if_called(*a, **k):
        raise AssertionError("pick_striking_frame should not be called when hook_card=False")
    monkeypatch.setattr(local_clipper_module, "pick_striking_frame", _fail_if_called)

    results = crop_highlights_local(
        synthetic_source, [_highlight_with_hook()], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
        hook_card=False,
    )
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])


def test_hook_card_skipped_when_on_screen_hook_missing(tmp_path, synthetic_source, monkeypatch):
    def _fail_if_called(*a, **k):
        raise AssertionError("pick_striking_frame should not be called without on_screen_hook")
    monkeypatch.setattr(local_clipper_module, "pick_striking_frame", _fail_if_called)

    results = crop_highlights_local(
        synthetic_source, [_highlight()], aspect_ratio="9:16",  # no on_screen_hook
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
    )
    assert results[0]["clip_url"] is not None


def test_hook_card_failure_falls_back_to_captioned_clip(tmp_path, synthetic_source, monkeypatch):
    def _raise(*a, **k):
        raise HookCardError("boom")
    monkeypatch.setattr(local_clipper_module, "pick_striking_frame", _raise)

    results = crop_highlights_local(
        synthetic_source, [_highlight_with_hook()], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
    )
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])
    assert results[0]["hook_card_error"] == "boom"
    assert "captions_error" not in results[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_local_clipper.py -v -k hook_card`
Expected: `ImportError: cannot import name 'HookCardError' from 'shorts_generator.hook_card'` or `TypeError: crop_highlights_local() got an unexpected keyword argument 'hook_card'`

- [ ] **Step 3: Add the import**

In `shorts_generator/local/clipper.py`, after the existing `from ..captions import CaptionError, burn_captions` line (line 23):

```python
from ..hook_card import HookCardError, extract_frame, pick_striking_frame, render_card_overlay
```

- [ ] **Step 4: Rewrite `crop_highlights_local`**

Replace the function body (lines 440-491) with:

```python
def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    transcript_segments: Optional[List[Dict]] = None,
    captions: bool = True,
    caption_fade_duration: float = 0.3,
    word_highlight: bool = True,
    framing: str = "locked",
    hook_card: bool = True,
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"Short-{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')}", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                framing=framing,
            )
            entry = {**h, "clip_url": out_path}

            hook_text = str(h.get("on_screen_hook") or "").strip()
            still_path = out_path + ".still.jpg"
            have_still = False
            if hook_card and hook_text:
                try:
                    ts = pick_striking_frame(out_path)
                    extract_frame(out_path, ts, still_path)
                    have_still = True
                except HookCardError as e:
                    print(f"[clip/local] {i} hook-card frame skipped: {e}", flush=True)
                    entry["hook_card_error"] = str(e)

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
                        word_highlight=word_highlight,
                    )
                    os.replace(captioned_path, out_path)
                except CaptionError as e:
                    print(f"[clip/local] {i} captions skipped: {e}", flush=True)
                    entry["captions_error"] = str(e)
                    if os.path.exists(captioned_path):
                        os.remove(captioned_path)

            if have_still:
                try:
                    card_path = out_path + ".card.mp4"
                    render_card_overlay(out_path, still_path, hook_text, card_path)
                    os.replace(card_path, out_path)
                except HookCardError as e:
                    print(f"[clip/local] {i} hook-card overlay skipped: {e}", flush=True)
                    entry["hook_card_error"] = str(e)
                finally:
                    if os.path.exists(still_path):
                        os.remove(still_path)

            results.append(entry)
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_local_clipper.py -v`
Expected: all pass (existing tests + 4 new ones)

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "$(cat <<'EOF'
feat: wire hook card into local-mode clip cropping

pick+extract the still from the clean pre-caption crop, burn captions
as before, then composite the card as the final pass. A hook-card
failure at either phase falls back to the clip as it stood, recorded
via hook_card_error — never fails the whole clip.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Wire into `clipper.py` (API mode)

**Files:**
- Modify: `shorts_generator/clipper.py` (module docstring, imports, `crop_highlights`)
- Test: `tests/test_clipper_api.py`

**Context:** unlike local mode, API mode only downloads the hosted clip locally today when `captions and transcript_segments` — otherwise `clip_url` stays a remote MuAPI URL. The hook card needs a local file to run `cv2`/ffmpeg on, so the download condition must widen to `captions-wanted OR hook-card-wanted`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_clipper_api.py`:

```python
from shorts_generator.hook_card import HookCardError


def _highlight_with_hook():
    return {**_highlight(), "on_screen_hook": "WATCH THIS"}


def _stub_hook_card(monkeypatch):
    def _fake_pick(video_path):
        return 0.5

    def _fake_extract(video_path, ts, out_path):
        with open(out_path, "wb") as f:
            f.write(b"x")
        return out_path

    def _fake_render(video_path, still_path, hook_text, out_path, duration=1.5):
        shutil.copyfile(video_path, out_path)
        return out_path

    monkeypatch.setattr(clipper, "pick_striking_frame", _fake_pick)
    monkeypatch.setattr(clipper, "extract_frame", _fake_extract)
    monkeypatch.setattr(clipper, "render_card_overlay", _fake_render)


def test_hook_card_triggers_local_download_even_when_captions_disabled(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper, "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )
    _stub_hook_card(monkeypatch)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight_with_hook()],
        aspect_ratio="9:16",
        transcript_segments=None,
        captions=False,
        out_dir=str(tmp_path / "out"),
    )

    assert results[0]["clip_url"] != "https://hosted.example/short_1.mp4"
    assert os.path.exists(results[0]["clip_url"])
    assert results[0]["hosted_clip_url"] == "https://hosted.example/short_1.mp4"


def test_no_local_download_when_captions_and_hook_card_both_off(tmp_path, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")

    def _fail_if_called(*a, **k):
        raise AssertionError("_download_to should not be called")
    monkeypatch.setattr(clipper, "_download_to", _fail_if_called)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight_with_hook()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        captions=False,
        hook_card=False,
        out_dir=str(tmp_path / "out"),
    )
    assert results[0]["clip_url"] == "https://hosted.example/short_1.mp4"


def test_hook_card_skipped_when_on_screen_hook_missing(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper, "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    def _fail_if_called(*a, **k):
        raise AssertionError("pick_striking_frame should not be called without on_screen_hook")
    monkeypatch.setattr(clipper, "pick_striking_frame", _fail_if_called)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],  # no on_screen_hook
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
    )
    assert results[0]["clip_url"] is not None
    assert "hook_card_error" not in results[0]


def test_hook_card_failure_falls_back_to_captioned_clip(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper, "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    def _raise(*a, **k):
        raise HookCardError("boom")
    monkeypatch.setattr(clipper, "pick_striking_frame", _raise)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight_with_hook()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
    )
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])
    assert results[0]["hook_card_error"] == "boom"
    assert "captions_error" not in results[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_clipper_api.py -v -k hook_card`
Expected: `TypeError: crop_highlights() got an unexpected keyword argument 'hook_card'`

- [ ] **Step 3: Add the import**

In `shorts_generator/clipper.py`, after the existing `from .captions import CaptionError, burn_captions` line (line 15):

```python
from .hook_card import HookCardError, extract_frame, pick_striking_frame, render_card_overlay
```

- [ ] **Step 4: Rewrite `crop_highlights`**

Replace the function body (lines 44-96) with:

```python
def crop_highlights(
    source_video_url: str,
    highlights: list,
    aspect_ratio: str = "9:16",
    transcript_segments: Optional[List[Dict]] = None,
    captions: bool = True,
    caption_fade_duration: float = 0.3,
    word_highlight: bool = True,
    hook_card: bool = True,
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

            want_captions = captions and bool(transcript_segments)
            hook_text = str(h.get("on_screen_hook") or "").strip()
            want_hook_card = hook_card and bool(hook_text)

            if want_captions or want_hook_card:
                os.makedirs(out_dir, exist_ok=True)
                final_path = os.path.join(out_dir, f"Short-{i:02d}.mp4")
                downloaded_path = final_path + ".download.mp4"
                still_path = final_path + ".still.jpg"
                have_still = False
                try:
                    _download_to(url, downloaded_path)

                    if want_hook_card:
                        try:
                            ts = pick_striking_frame(downloaded_path)
                            extract_frame(downloaded_path, ts, still_path)
                            have_still = True
                        except HookCardError as e:
                            print(f"[clip] {i} hook-card frame skipped: {e}", flush=True)
                            entry["hook_card_error"] = str(e)

                    if want_captions:
                        burn_captions(
                            downloaded_path,
                            transcript_segments,
                            float(h["start_time"]),
                            float(h["end_time"]),
                            final_path,
                            fade_seconds=caption_fade_duration,
                            word_highlight=word_highlight,
                        )
                    else:
                        os.replace(downloaded_path, final_path)

                    if have_still:
                        try:
                            card_path = final_path + ".card.mp4"
                            render_card_overlay(final_path, still_path, hook_text, card_path)
                            os.replace(card_path, final_path)
                        except HookCardError as e:
                            print(f"[clip] {i} hook-card overlay skipped: {e}", flush=True)
                            entry["hook_card_error"] = str(e)

                    entry["clip_url"] = final_path
                    entry["hosted_clip_url"] = url
                except (CaptionError, requests.RequestException) as e:
                    print(f"[clip] {i} captions skipped: {e}", flush=True)
                    entry["captions_error"] = str(e)
                finally:
                    if os.path.exists(downloaded_path):
                        os.remove(downloaded_path)
                    if os.path.exists(still_path):
                        os.remove(still_path)

            out.append(entry)
        except Exception as e:
            print(f"[clip] {i} failed: {e}", flush=True)
            out.append({**h, "clip_url": None, "error": str(e)})
    return out
```

- [ ] **Step 5: Update the module docstring**

Replace the module docstring (lines 1-8) with:

```python
"""Per-clip cropping via MuAPI /autocrop, with optional local caption
burn-in and hook-card overlay.

Given the source video URL plus a highlight's start/end and a target aspect
ratio, MuAPI returns a vertically-cropped short ready for posting. When
captions or the hook card are enabled (both on by default), that hosted
clip is downloaded locally and processed with ffmpeg (shorts_generator.
captions, shorts_generator.hook_card) — the one place API mode now needs a
local ffmpeg and OpenCV on PATH/installed.
"""
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_clipper_api.py -v`
Expected: all pass (existing tests + 4 new ones)

- [ ] **Step 7: Commit**

```bash
git add shorts_generator/clipper.py tests/test_clipper_api.py
git commit -m "$(cat <<'EOF'
feat: wire hook card into API-mode clip cropping

Widens the local-download condition from "captions wanted" to
"captions or hook-card wanted", since the hook card needs a local file
to run cv2/ffmpeg on even when captions are off. Same pick-before-burn,
composite-last ordering and error-fallback contract as local mode.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Thread `hook_card` through `pipeline.py`

**Files:**
- Modify: `shorts_generator/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Modify the existing threading tests**

In `tests/test_pipeline.py`, update `test_run_local_threads_captions_params` (around line 36) — add `hook_card=False` to the call and assert it:

```python
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
        hook_card=False,
    )

    assert result["mode"] == "local"
    assert result["shorts"] == [{"clip_url": "/tmp/out/Short-01.mp4"}]

    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is False
    assert kwargs["caption_fade_duration"] == 0.7
    assert kwargs["word_highlight"] is False
    assert kwargs["hook_card"] is False
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]
```

And `test_run_api_threads_captions_params` (around line 103) similarly:

```python
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
        hook_card=False,
    )

    assert result["mode"] == "api"
    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is True
    assert kwargs["caption_fade_duration"] == 0.3
    assert kwargs["word_highlight"] is False
    assert kwargs["hook_card"] is False
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]
```

The other tests in this file that call `_run_local`/`_run_api` without `word_highlight` covered (e.g. `test_run_local_skips_download_when_source_already_exists`, `test_run_api_skips_local_copy_and_transcribe_when_cached`, `test_run_api_interrupted_download_does_not_leave_partial_source_video`, `test_run_api_recovers_from_corrupted_transcript_cache`) already pass `word_highlight=...` explicitly and don't assert on it — leave those as-is; `hook_card` will use its default (`True`) in those calls, which is fine since they don't inspect it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v -k threads_captions_params`
Expected: `TypeError: _run_local() got an unexpected keyword argument 'hook_card'`

- [ ] **Step 3: Update `pipeline.py`**

In `shorts_generator/pipeline.py`, `_run_local` signature (lines 26-37) — add the parameter:

```python
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
) -> Dict:
```

and thread it into the `crop_highlights_local(...)` call (around line 63-73):

```python
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
    )
```

`_run_api` signature (lines 85-95) — add the parameter:

```python
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
) -> Dict:
```

and thread it into the `crop_highlights(...)` call (around line 141-150):

```python
    shorts = crop_highlights(
        source_url,
        top,
        aspect_ratio=aspect_ratio,
        transcript_segments=transcript["segments"],
        captions=captions,
        caption_fade_duration=caption_fade_duration,
        word_highlight=word_highlight,
        hook_card=hook_card,
        out_dir=paths.shorts_dir,
    )
```

`generate_shorts` signature (lines 162-173) — add the parameter:

```python
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
    paths: Optional[RunPaths] = None,
) -> Dict:
```

Update its docstring's Args section (add after the `framing` bullet, around line 191):

```python
        hook_card: composite a bold on-screen hook (from each highlight's
            "on_screen_hook") over a motion-picked striking still for the
            clip's first 1.5 seconds (default True).
```

And update the Returns section's `hosted_clip_url` bullet (around line 206) to reflect the new API-mode trigger:

```python
                                      #   hosted_clip_url: original MuAPI URL (api mode,
                                      #     only present when captions or the hook card
                                      #     triggered a local download)
                                      #   hook_card_error: present if the hook-card overlay
                                      #     failed for that clip (falls back to the clip as
                                      #     it stood before the hook-card pass)
```

And thread `hook_card` into both branch calls (lines 218-227):

```python
        if mode == "local":
            result = _run_local(
                youtube_url, num_clips, aspect_ratio, download_format, language, captions, caption_fade_duration,
                paths, word_highlight=word_highlight, framing=framing, hook_card=hook_card,
            )
        else:
            result = _run_api(
                youtube_url, num_clips, aspect_ratio, download_format, language, captions, caption_fade_duration,
                paths, word_highlight=word_highlight, hook_card=hook_card,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/pipeline.py tests/test_pipeline.py
git commit -m "$(cat <<'EOF'
feat: thread hook_card through the pipeline

generate_shorts() gains a hook_card flag (default True), passed down
through both _run_api and _run_local to their respective crop_highlights
calls.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: `main.py` CLI flag

**Files:**
- Modify: `main.py:48-62` (parser), `:70-81` (generate_shorts call)
- Test: `tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_main.py`:

```python
def test_hook_card_on_by_default():
    args = build_parser().parse_args(["https://example.com/video"])
    assert args.hook_card is True


def test_no_hook_card_flag_disables():
    args = build_parser().parse_args(["https://example.com/video", "--no-hook-card"])
    assert args.hook_card is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_main.py -v -k hook_card`
Expected: `AttributeError: 'Namespace' object has no attribute 'hook_card'`

- [ ] **Step 3: Add the flag**

In `main.py`, after the `--no-word-highlight` block (lines 48-54) and before `--framing`:

```python
    parser.add_argument(
        "--no-hook-card",
        dest="hook_card",
        action="store_false",
        default=True,
        help="Disable the first-frame hook-card overlay (bold on-screen hook text over a "
             "motion-picked striking still for the first 1.5s; on by default).",
    )
```

- [ ] **Step 4: Thread it into the `generate_shorts` call**

In `main.py`, in `main()` (around line 70-81), add `hook_card=args.hook_card,`:

```python
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
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "$(cat <<'EOF'
feat: add --no-hook-card CLI flag

Same shape as --no-word-highlight; hook card is on by default.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Webapp form field + dashboard checkbox

**Files:**
- Modify: `shorts_generator/webapp.py:39-68` (`_run_job`), `:117-134` (`start_run`)
- Modify: `shorts_generator/templates/index.html:549-556` (checkbox), `:818-827` (JS)

No dedicated test — existing simple form-passthrough fields (`mode`, `aspect_ratio`, `framing`) aren't unit-tested in `tests/test_webapp.py` today either; this follows that same convention. Verified manually in Step 5.

- [ ] **Step 1: Update `_run_job`**

In `shorts_generator/webapp.py`, add `hook_card: bool` to the signature (around line 39-50):

```python
def _run_job(
    url: str,
    mode: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    captions: bool,
    caption_fade_duration: float,
    word_highlight: bool,
    framing: str,
    hook_card: bool,
) -> None:
```

and thread it into the `generate_shorts(...)` call (around line 55-68):

```python
        result = generate_shorts(
            url,
            num_clips=num_clips,
            aspect_ratio=aspect_ratio,
            download_format=download_format,
            language=language,
            mode=mode,
            captions=captions,
            caption_fade_duration=caption_fade_duration,
            word_highlight=word_highlight,
            framing=framing,
            hook_card=hook_card,
            paths=paths,
        )
```

- [ ] **Step 2: Update `start_run`**

In `shorts_generator/webapp.py`, add to the `kwargs` dict (around line 124-134), after `word_highlight`:

```python
            word_highlight=request.form.get("word_highlight", "true") == "true",
            hook_card=request.form.get("hook_card", "true") == "true",
            framing=request.form.get("framing", "locked"),
```

- [ ] **Step 3: Add the dashboard checkbox**

In `shorts_generator/templates/index.html`, in the checkbox row (lines 549-556), add a third checkbox:

```html
          <div class="row">
            <div class="checkbox-row">
              <label><input type="checkbox" id="captions" name="captions" checked> Burn captions</label>
            </div>
            <div class="checkbox-row">
              <label><input type="checkbox" id="word_highlight" name="word_highlight" checked> Word highlight</label>
            </div>
            <div class="checkbox-row">
              <label><input type="checkbox" id="hook_card" name="hook_card" checked> Hook card</label>
            </div>
          </div>
```

- [ ] **Step 4: Wire the checkbox into the submit handler**

In `shorts_generator/templates/index.html`, in the form submit listener (around line 826-827), add a third line:

```javascript
      formData.set("captions", document.getElementById("captions").checked ? "true" : "false");
      formData.set("word_highlight", document.getElementById("word_highlight").checked ? "true" : "false");
      formData.set("hook_card", document.getElementById("hook_card").checked ? "true" : "false");
```

- [ ] **Step 5: Manual verification**

Run: `python3 -c "from shorts_generator.webapp import app; print([r.rule for r in app.url_map.iter_rules()])"`
Expected: prints the route list without error (confirms `webapp.py` still imports cleanly after the edits — a syntax/signature error here would raise immediately). Then start the dashboard (`python dashboard.py` or however it's normally launched) and visually confirm the "Hook card" checkbox renders next to "Word highlight" and is checked by default.

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/webapp.py shorts_generator/templates/index.html
git commit -m "$(cat <<'EOF'
feat: add hook-card toggle to the dashboard

Checkbox alongside the existing captions/word-highlight toggles,
threaded through start_run -> _run_job -> generate_shorts. On by
default, matching the CLI flag's default.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: README documentation

**Files:**
- Modify: `README.md:191-201` (flags table), `:203-212` (mode comparison table)

- [ ] **Step 1: Add the flag row**

In `README.md`, in the flags table, add a row after `--no-word-highlight` (line 201):

```markdown
| `--no-hook-card` | hook card on | Disable the first-frame hook-card overlay (bold on-screen hook text over a motion-picked striking still for the first 1.5s; on by default in both modes) |
```

- [ ] **Step 2: Update the mode-comparison table's Output row**

In `README.md`, the "API mode vs Local mode" table's `Output` row (line 211) currently reads:

```markdown
| Output | local mp4 path with captions burned in (default); hosted MuAPI URL if `--no-captions` | local mp4 paths |
```

Replace with:

```markdown
| Output | local mp4 path with captions and hook card burned in (default); hosted MuAPI URL only if both `--no-captions` and `--no-hook-card` | local mp4 paths |
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: document the --no-hook-card flag

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all tests pass, none skipped/errored. If `test_webapp.py` or any other pre-existing test fails due to a signature change made in this plan, fix the call site before proceeding — do not leave a broken test.

- [ ] **Step 2: Manual end-to-end smoke check**

Run one real short clip through `--mode local` on a short local video (a few seconds is enough), e.g.:

```bash
python main.py "file:///path/to/a/short/local/test/video.mp4" --mode local --num-clips 1
```

Expected: the run completes, and the produced `Short-01.mp4`'s first ~1.5 seconds visibly differ from a plain crop of the source's opening frame (bold text over a still, not the raw talking-head start), with live footage and audio in sync from ~1.5s onward. If no local test video is handy, this step can instead re-run the `render_card_overlay` smoke test from Task 4 against a real (non-synthetic) short clip already present in `output/` from a prior run, if one exists.

- [ ] **Step 3: Confirm nothing was left uncommitted**

Run: `git status --short`
Expected: no changes related to this feature left unstaged (unrelated pre-existing dirty files from other in-progress work are fine to leave as they were found).
