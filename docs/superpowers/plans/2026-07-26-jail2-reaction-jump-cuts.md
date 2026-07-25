# Jail 2 — Reaction Jail Escape Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** every generated highlight commits to one target viewer reaction and excises any mid-span dead weight via jump cuts, so clips escape "reaction jail" (stuck 1K-10K views).

**Architecture:** the highlight LLM now returns `reaction_type` (enum) and `cut_segments` (1-6 sub-spans, replacing the implicit single span) per highlight. A new shared `jump_cuts.py` module trims-and-concats the kept spans via ffmpeg, used by both api mode (`clipper.py`, operating on the already-cropped download) and local mode (`local/clipper.py`, operating on the raw envelope cut before reframing). `captions.py` chunks each kept span independently so no caption line can ever straddle an excised gap, then offsets chunks onto the concatenated timeline.

**Tech Stack:** Python, ffmpeg (trim + concat demuxer), pytest, existing MuAPI/local LLM prompt plumbing.

**Spec:** `docs/superpowers/specs/2026-07-26-three-jails-escape-design.md`

---

### Task 1: Capture the pre-change test baseline

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite and record the result**

Run: `python -m pytest tests/ -q`
Expected: note the exact pass count and any pre-existing failures/skips verbatim (e.g. "142 passed, 2 skipped"). This is the baseline every later "no behavior change for existing single-span clips" claim in this plan is checked against — if a test that passed here fails after a later task, that's a regression, not a pre-existing issue.

- [ ] **Step 2: Note it in the working notes (not committed)**

No commit for this task — it's a reference point for Tasks 3, 5, 6 below, not a code change.

---

### Task 2: `cut_segments` + `reaction_type` validation in `highlights.py`

**Files:**
- Modify: `shorts_generator/highlights.py`
- Test: `tests/test_highlights.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_highlights.py` (after `test_sanitize_highlights_coerces_string_hook_self_contained`, using the existing `_raw_highlight()` helper at the top of the file):

```python
def test_sanitize_highlights_keeps_valid_cut_segments():
    raw = _raw_highlight(
        start_time=1.0, end_time=10.0,
        cut_segments=[
            {"start_time": 1.0, "end_time": 3.0},
            {"start_time": 6.0, "end_time": 10.0},
        ],
    )
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["cut_segments"] == [
        {"start_time": 1.0, "end_time": 3.0},
        {"start_time": 6.0, "end_time": 10.0},
    ]


def test_sanitize_highlights_falls_back_to_envelope_when_cut_segments_missing():
    raw = _raw_highlight(start_time=1.0, end_time=5.0)
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["cut_segments"] == [{"start_time": 1.0, "end_time": 5.0}]


def test_sanitize_highlights_falls_back_to_envelope_when_cut_segments_overlap():
    raw = _raw_highlight(
        start_time=1.0, end_time=10.0,
        cut_segments=[
            {"start_time": 1.0, "end_time": 5.0},
            {"start_time": 4.0, "end_time": 10.0},
        ],
    )
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["cut_segments"] == [{"start_time": 1.0, "end_time": 10.0}]


def test_sanitize_highlights_clamps_cut_segments_to_envelope():
    raw = _raw_highlight(
        start_time=2.0, end_time=8.0,
        cut_segments=[{"start_time": 0.0, "end_time": 20.0}],
    )
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["cut_segments"] == [{"start_time": 2.0, "end_time": 8.0}]


def test_sanitize_highlights_caps_cut_segments_at_six():
    raw = _raw_highlight(
        start_time=0.0, end_time=100.0,
        cut_segments=[{"start_time": float(i * 10), "end_time": float(i * 10 + 5)} for i in range(8)],
    )
    cleaned = _sanitize_highlights([raw], duration=200.0)
    assert len(cleaned[0]["cut_segments"]) == 6


def test_sanitize_highlights_keeps_valid_reaction_type():
    raw = _raw_highlight(reaction_type="LOL")
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["reaction_type"] == "LOL"


def test_sanitize_highlights_reaction_type_case_insensitive():
    raw = _raw_highlight(reaction_type="lol")
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["reaction_type"] == "LOL"


def test_sanitize_highlights_defaults_reaction_type_when_invalid():
    raw = _raw_highlight(reaction_type="HYPE")
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["reaction_type"] == "WOW"


def test_sanitize_highlights_defaults_reaction_type_when_missing():
    raw = {"start_time": 1.0, "end_time": 5.0}
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["reaction_type"] == "WOW"


def test_sanitize_highlights_defaults_tightness_reason_to_empty_string():
    raw = {"start_time": 1.0, "end_time": 5.0}
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["tightness_reason"] == ""


def test_sanitize_highlights_includes_tightness_reason():
    raw = _raw_highlight(tightness_reason="cut the walk-back-in, kept the punchline")
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["tightness_reason"] == "cut the walk-back-in, kept the punchline"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_highlights.py -k "cut_segments or reaction_type or tightness_reason" -v`
Expected: FAIL — `KeyError: 'cut_segments'` (and similar for the other new fields), since `_sanitize_highlights` doesn't produce them yet.

- [ ] **Step 3: Implement `_sanitize_cut_segments` and wire both new fields into `_sanitize_highlights`**

In `shorts_generator/highlights.py`, add a module-level constant right after `_GENERIC_SPAM_TAGS` (around line 31):

```python
_REACTION_TYPES = {"LOL", "WOW", "OMG", "FINALLY", "WTF", "WHOLESOME"}
```

Add a new helper function right before `_sanitize_highlights` (which currently starts at line 207):

```python
def _sanitize_cut_segments(raw_segments: object, envelope_start: float, envelope_end: float) -> Optional[List[Dict]]:
    """Validate/clamp a highlight's cut_segments against its envelope.
    Returns None (caller falls back to the single-span envelope) if the
    input isn't a non-empty list of well-formed, non-overlapping spans."""
    if not isinstance(raw_segments, list) or not raw_segments:
        return None

    cleaned: List[Dict] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            return None
        s = _coerce_float(item.get("start_time"), default=-1.0)
        e = _coerce_float(item.get("end_time"), default=-1.0)
        if s < 0 or e <= s:
            return None
        s = max(s, envelope_start)
        e = min(e, envelope_end)
        if e <= s:
            return None
        cleaned.append({"start_time": s, "end_time": e})

    cleaned.sort(key=lambda c: c["start_time"])
    for i in range(1, len(cleaned)):
        if cleaned[i]["start_time"] < cleaned[i - 1]["end_time"]:
            return None  # overlapping spans -- ambiguous, fall back to envelope

    return cleaned[:6]
```

This uses `Optional` from `typing`, already imported at the top of the file (`from typing import Callable, Dict, List, Optional`).

Now edit `_sanitize_highlights` (highlights.py:207-247). The existing body clamps `start`/`end` at lines 218-227, right before building the `cleaned.append({...})` dict. Insert right after that clamping block (still inside the `for item in raw_highlights:` loop, before `cleaned.append(...)`):

```python
        cut_segments = _sanitize_cut_segments(item.get("cut_segments"), start, end) or [
            {"start_time": start, "end_time": end}
        ]

        reaction_type = str(item.get("reaction_type") or "").strip().upper()
        if reaction_type not in _REACTION_TYPES:
            reaction_type = "WOW"
```

Then add three keys to the existing `cleaned.append({...})` dict literal (which currently ends with `"yt_hashtags": _sanitize_hashtags(item.get("yt_hashtags")),`):

```python
                "cut_segments": cut_segments,
                "reaction_type": reaction_type,
                "tightness_reason": str(item.get("tightness_reason") or "").strip(),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_highlights.py -v`
Expected: PASS — all tests including the pre-existing ones (compare count against Task 1's baseline; should be baseline + 11 new passing tests, zero regressions).

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/highlights.py tests/test_highlights.py
git commit -m "feat: validate cut_segments and reaction_type on highlights"
```

---

### Task 3: Reaction-jail prompt block + schema version bump

**Files:**
- Modify: `shorts_generator/highlights.py`

- [ ] **Step 1: Add the `REACTION_JAIL_CRITERIA` prompt block**

In `shorts_generator/highlights.py`, add a new module-level string right after `VIRALITY_CRITERIA` (which ends at line 61):

```python
REACTION_JAIL_CRITERIA = """
Reaction-jail escape (views stuck 1K-10K, the "does anyone care" test):
every highlight must build toward ONE specific viewer reaction, chosen
before anything else about the clip. Pick exactly one:
- LOL: comedic payoff
- WOW: visual or factual astonishment
- OMG: shocking revelation
- FINALLY: viewer relief/agreement ("someone said it")
- WTF: bewildering, confusing-in-a-compelling-way moment
- WHOLESOME: warmth, tenderness, feel-good

Once the reaction is picked, cut everything that doesn't build toward it —
even a passage in the middle of the highlight's overall time range if it's
dead air, a tangent, or setup that doesn't pay off. Never lengthen a clip
"for retention." Express this as `cut_segments`: a list of 1 to 6
`{"start_time": float, "end_time": float}` spans to keep, in order. One
entry means the clip is already tight end-to-end and needs no internal cut.
Every `cut_segments` boundary MUST land exactly on a transcript line's
start or end timestamp from the transcript below — never mid-sentence,
never mid-word, same rule as the highlight's own outer start_time/end_time.
"""
```

- [ ] **Step 2: Wire the new block into `HIGHLIGHT_SYSTEM_PROMPT`**

In `HIGHLIGHT_SYSTEM_PROMPT` (highlights.py:82-111), change the header:

```python
HIGHLIGHT_SYSTEM_PROMPT = """You are an elite short-form video editor who has studied thousands of viral clips on TikTok, Instagram Reels, and YouTube Shorts. You know exactly what makes viewers stop scrolling, watch to the end, and share.

{virality_criteria}

{reaction_jail_criteria}

Content type: {content_type} | Density: {density}
```

Add two new bullets to the `Rules:` list, right after the existing `on_screen_hook` bullet (the one starting "Write an \"on_screen_hook\"..."):

```
- Set "reaction_type" to exactly one of: LOL, WOW, OMG, FINALLY, WTF, WHOLESOME — the single reaction this clip is built to trigger
- Set "cut_segments" to the list of kept spans described above (1-6 entries); write a "tightness_reason" — one sentence on what got cut and why, or why nothing needed cutting
```

Update the final JSON schema line (the one starting `Respond ONLY with valid JSON`) to include the three new keys in the example object:

```python
Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","on_screen_hook":"string","virality_reason":"string","hook_strength":int,"hook_self_contained":bool,"hook_reason":"string","description":"string","yt_title":"string","yt_hashtags":["#Shorts","#topic1","#topic2"],"reaction_type":"string","cut_segments":[{{"start_time":float,"end_time":float}}],"tightness_reason":"string"}}]}}"""
```

Now call `HIGHLIGHT_SYSTEM_PROMPT.format(...)` in `call_highlight_api` (highlights.py:300-306) with the new argument:

```python
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        reaction_jail_criteria=REACTION_JAIL_CRITERIA,
        hook_strength_rubric=HOOK_STRENGTH_RUBRIC,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        num_clips_instruction=f"Generate at least {min_clips} highlights",
    )
```

Also update the retry-prompt field list in the same function (the `IMPORTANT: Return ONLY valid JSON...` string a few lines below), which currently lists required fields — append `, reaction_type, cut_segments, tightness_reason` to that list:

```python
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Return ONLY valid JSON with a top-level 'highlights' array."
                + " Each item must include: title, start_time, end_time, score, hook_sentence, on_screen_hook, virality_reason, hook_strength, hook_self_contained, hook_reason, description, yt_title, yt_hashtags, reaction_type, cut_segments, tightness_reason."
                + " No markdown fences, no commentary."
            )
```

- [ ] **Step 3: Bump the schema version**

Change `HIGHLIGHT_SCHEMA_VERSION = 2` to `HIGHLIGHT_SCHEMA_VERSION = 3` (highlights.py:119), and update its comment to note why:

```python
HIGHLIGHT_SCHEMA_VERSION = 3    # bump whenever the highlight dict shape changes,
                                # so a stale on-disk cache (missing new fields)
                                # is treated as a miss instead of silently reused.
                                # v3: added cut_segments, reaction_type, tightness_reason.
```

- [ ] **Step 4: Run the full highlights test suite**

Run: `python -m pytest tests/test_highlights.py -v`
Expected: PASS, same count as Task 2's Step 4 (this task only changes prompt strings and the version constant, both already covered by existing schema-version-mismatch tests like `test_get_highlights_cached_recomputes_on_schema_version_mismatch`).

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/highlights.py
git commit -m "feat: add reaction-jail prompt block, bump highlight schema to v3"
```

---

### Task 4: Shared `jump_cuts.py` excision module

**Files:**
- Create: `shorts_generator/jump_cuts.py`
- Test: `tests/test_jump_cuts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_jump_cuts.py`:

```python
import subprocess

import pytest

from shorts_generator.jump_cuts import excise_cut_segments


def _probe_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


@pytest.fixture(scope="module")
def synthetic_envelope(tmp_path_factory):
    """A 10s clip standing in for a highlight's envelope cut (already trimmed
    to [start_time, end_time] by an upstream ffmpeg/-autocrop step)."""
    tmp_dir = tmp_path_factory.mktemp("jump_cuts_src")
    path = str(tmp_dir / "envelope.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x568:rate=24:duration=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            path,
        ],
        check=True,
    )
    return path


def test_excise_cut_segments_drops_the_gap(tmp_path, synthetic_envelope):
    out_path = str(tmp_path / "excised.mp4")
    # Envelope spans absolute [100.0, 110.0]; keep [100,102] and [107,110],
    # drop the [102,107] gap in the middle.
    cut_segments = [
        {"start_time": 100.0, "end_time": 102.0},
        {"start_time": 107.0, "end_time": 110.0},
    ]

    result = excise_cut_segments(synthetic_envelope, cut_segments, envelope_start=100.0, out_path=out_path)

    assert result == out_path
    duration = _probe_duration(out_path)
    assert abs(duration - 5.0) < 0.3  # 2s + 3s kept, 5s of gap dropped


def test_excise_cut_segments_single_span_matches_input_duration(tmp_path, synthetic_envelope):
    out_path = str(tmp_path / "excised.mp4")
    cut_segments = [{"start_time": 100.0, "end_time": 106.0}]

    excise_cut_segments(synthetic_envelope, cut_segments, envelope_start=100.0, out_path=out_path)

    assert abs(_probe_duration(out_path) - 6.0) < 0.3


def test_excise_cut_segments_cleans_up_temp_dir(tmp_path, synthetic_envelope):
    import os

    out_path = str(tmp_path / "excised.mp4")
    cut_segments = [
        {"start_time": 100.0, "end_time": 102.0},
        {"start_time": 105.0, "end_time": 108.0},
    ]

    excise_cut_segments(synthetic_envelope, cut_segments, envelope_start=100.0, out_path=out_path)

    assert not os.path.exists(out_path + ".parts")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_jump_cuts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shorts_generator.jump_cuts'`.

- [ ] **Step 3: Implement `shorts_generator/jump_cuts.py`**

```python
"""ffmpeg trim + concat: excise the gaps between a highlight's cut_segments,
keeping only the spans the highlight generator marked as building toward its
target reaction (see highlights.py's REACTION_JAIL_CRITERIA).

Shared by both api mode (clipper.py, which runs this on the already
aspect-ratio-cropped download) and local mode (local/clipper.py, which runs
this on the raw envelope cut before reframing) — the excision itself is
identical ffmpeg trim/concat regardless of what's already been done to the
video.
"""
import os
import shutil
import subprocess
from typing import Dict, List


def excise_cut_segments(
    source_path: str,
    cut_segments: List[Dict],
    envelope_start: float,
    out_path: str,
) -> str:
    """Keep only `cut_segments` (absolute transcript times) from
    `source_path` (already trimmed to the highlight's envelope, starting at
    `envelope_start`), drop everything else, and write the concatenated
    result to `out_path`. Assumes `cut_segments` has at least one entry."""
    tmp_dir = out_path + ".parts"
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        part_paths = []
        for i, seg in enumerate(cut_segments):
            rel_start = float(seg["start_time"]) - envelope_start
            rel_end = float(seg["end_time"]) - envelope_start
            part_path = os.path.join(tmp_dir, f"part{i}.mp4")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", source_path,
                    "-ss", f"{rel_start:.3f}", "-to", f"{rel_end:.3f}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "128k",
                    part_path,
                ],
                check=True, capture_output=True, text=True,
            )
            part_paths.append(part_path)

        concat_list_path = os.path.join(tmp_dir, "concat.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for p in part_paths:
                f.write(f"file '{p}'\n")

        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", concat_list_path,
                "-c", "copy",
                out_path,
            ],
            check=True, capture_output=True, text=True,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_jump_cuts.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/jump_cuts.py tests/test_jump_cuts.py
git commit -m "feat: add jump_cuts.excise_cut_segments for shared trim+concat"
```

---

### Task 5: Per-segment caption chunking in `captions.py`

**Files:**
- Modify: `shorts_generator/captions.py`
- Test: `tests/test_captions.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_captions.py`, near the existing `_chunk_segments` tests:

```python
from shorts_generator.captions import _chunk_cut_segments, burn_captions_segments


def test_chunk_cut_segments_offsets_second_span_after_first():
    transcript_segments = [
        {"start": 0.0, "end": 2.0, "text": "alpha beta"},
        {"start": 10.0, "end": 12.0, "text": "gamma delta"},
    ]
    cut_segments = [
        {"start_time": 0.0, "end_time": 2.0},
        {"start_time": 10.0, "end_time": 12.0},
    ]

    chunks = _chunk_cut_segments(transcript_segments, cut_segments, max_words=7)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "alpha beta"
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 2.0
    # second span starts at output-timeline offset 2.0 (end of first kept span),
    # not at its own absolute transcript time of 10.0
    assert chunks[1]["text"] == "gamma delta"
    assert chunks[1]["start"] == 2.0
    assert chunks[1]["end"] == 4.0


def test_chunk_cut_segments_no_chunk_straddles_the_gap():
    transcript_segments = [
        {"start": 0.0, "end": 4.0, "text": "one two three four five six seven eight"},
    ]
    # Keep only [0,2] and [2.5,4] of this single 4s transcript segment --
    # a naive whole-envelope chunk pass could produce a chunk spanning the
    # dropped [2,2.5] gap; per-segment chunking must not.
    cut_segments = [
        {"start_time": 0.0, "end_time": 2.0},
        {"start_time": 2.5, "end_time": 4.0},
    ]

    chunks = _chunk_cut_segments(transcript_segments, cut_segments, max_words=7)

    for c in chunks:
        # every chunk's word set must come entirely from one kept span:
        # duration of any single chunk can't exceed the longest kept span (1.5s)
        assert (c["end"] - c["start"]) <= 1.5 + 1e-6


def test_burn_captions_segments_produces_output_file(tmp_path, synthetic_clip):
    out_path = str(tmp_path / "burned.mp4")
    transcript_segments = [{"start": 0.0, "end": 3.0, "text": "hello there this is a caption test"}]
    cut_segments = [{"start_time": 0.0, "end_time": 3.0}]

    result = burn_captions_segments(
        synthetic_clip, transcript_segments, cut_segments, out_path, fade_seconds=0.3,
    )

    assert result == out_path
    assert os.path.exists(out_path)


def test_burn_captions_segments_raises_when_no_transcript_overlaps(tmp_path, synthetic_clip):
    out_path = str(tmp_path / "burned.mp4")
    transcript_segments = [{"start": 100.0, "end": 103.0, "text": "way outside"}]
    cut_segments = [{"start_time": 0.0, "end_time": 3.0}]

    with pytest.raises(CaptionError):
        burn_captions_segments(synthetic_clip, transcript_segments, cut_segments, out_path)
```

(`synthetic_clip`, `os`, `pytest`, and `CaptionError` are already imported/fixtured in `tests/test_captions.py`; only `_chunk_cut_segments` and `burn_captions_segments` need adding to the import line at the top of the file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_captions.py -k "cut_segments" -v`
Expected: FAIL — `ImportError: cannot import name '_chunk_cut_segments'`.

- [ ] **Step 3: Implement `_chunk_cut_segments` and `burn_captions_segments`, refactor `burn_captions` to share the ffmpeg-burn tail**

In `shorts_generator/captions.py`, add `_chunk_cut_segments` right after `_chunk_segments` (which ends at line 139):

```python
def _chunk_cut_segments(
    transcript_segments: List[Dict], cut_segments: List[Dict], max_words: int = 7,
) -> List[Dict]:
    """Chunk each kept `cut_segments` span independently against
    `transcript_segments` (so no chunk can ever straddle an excised gap by
    construction), then offset each span's chunks onto the concatenated
    output timeline by the cumulative duration of the *previously kept*
    spans (not the excised gaps)."""
    chunks: List[Dict] = []
    offset = 0.0
    for seg in cut_segments:
        seg_start = float(seg["start_time"])
        seg_end = float(seg["end_time"])
        for c in _chunk_segments(transcript_segments, seg_start, seg_end, max_words=max_words):
            chunks.append({
                "start": c["start"] + offset,
                "end": c["end"] + offset,
                "text": c["text"],
                "words": [
                    {"start": w["start"] + offset, "end": w["end"] + offset, "text": w["text"]}
                    for w in c["words"]
                ],
            })
        offset += seg_end - seg_start
    return chunks
```

Now refactor `burn_captions` (captions.py:260-302) to extract the shared ffmpeg-burn tail into `_burn_chunks`, and add `burn_captions_segments`. Replace the entire existing `burn_captions` function with:

```python
def _burn_chunks(
    video_path: str, chunks: List[Dict], out_path: str, fade_seconds: float, word_highlight: bool,
) -> str:
    width, height = _probe_resolution(video_path)

    ass_path = out_path + ".ass"
    _write_ass(chunks, ass_path, width, height, fade_seconds, word_highlight=word_highlight)

    try:
        escaped_ass_path = _escape_ffmpeg_path(ass_path)
        escaped_font_dir = _escape_ffmpeg_path(FONT_DIR)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"subtitles={escaped_ass_path}:fontsdir={escaped_font_dir}",
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
    chunks = _chunk_segments(segments, clip_start, clip_end, max_words=7)
    if not chunks:
        raise CaptionError(f"no transcript overlaps clip window [{clip_start}, {clip_end}]")
    return _burn_chunks(video_path, chunks, out_path, fade_seconds, word_highlight)


def burn_captions_segments(
    video_path: str,
    transcript_segments: List[Dict],
    cut_segments: List[Dict],
    out_path: str,
    fade_seconds: float = 0.3,
    word_highlight: bool = True,
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
    return _burn_chunks(video_path, chunks, out_path, fade_seconds, word_highlight)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_captions.py -v`
Expected: PASS — same pre-existing test count as Task 1's baseline (the `burn_captions` refactor must be behavior-preserving: same signature, same chunking, same error strings) plus the new tests from Step 1.

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/captions.py tests/test_captions.py
git commit -m "feat: add per-segment caption chunking for jump-cut clips"
```

---

### Task 6: Wire excision + segment captions into api mode (`clipper.py`)

**Files:**
- Modify: `shorts_generator/clipper.py`
- Test: `tests/test_clipper_api.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_clipper_api.py` (it already has `synthetic_clip`, `_segments()` helpers and the `clipper`/`shutil`/`os` imports):

```python
def _probe_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def test_multi_cut_segments_excises_the_gap(tmp_path, synthetic_clip, monkeypatch):
    # synthetic_clip is a 4s clip standing in for the hosted [0,4] envelope download.
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    highlight = {
        "title": "Test Clip", "start_time": 0.0, "end_time": 4.0, "score": 90,
        "cut_segments": [
            {"start_time": 0.0, "end_time": 1.0},
            {"start_time": 3.0, "end_time": 4.0},
        ],
    }

    out_dir = str(tmp_path / "out")
    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [highlight],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        captions=False,
        hook_card=False,
        out_dir=out_dir,
    )

    assert "excision_error" not in results[0]
    duration = _probe_duration(results[0]["clip_url"])
    assert abs(duration - 2.0) < 0.3  # kept 1s + 1s, dropped the 2s middle


def test_single_cut_segment_skips_excision(tmp_path, synthetic_clip, monkeypatch):
    # A single-entry cut_segments list must take the exact same path as
    # today's single-span highlights -- no excision step, no size change.
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    highlight = {
        **_highlight(),
        "cut_segments": [{"start_time": 0.0, "end_time": 3.0}],
    }

    out_dir = str(tmp_path / "out")
    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [highlight],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=out_dir,
    )

    assert "excision_error" not in results[0]
    duration = _probe_duration(results[0]["clip_url"])
    assert abs(duration - 3.0) < 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_clipper_api.py -k "cut_segment" -v`
Expected: FAIL — `abs(duration - 2.0) < 0.3` assertion fails (excision never happens yet, so the clip stays 4s / includes the middle).

- [ ] **Step 3: Wire excision + segment captions into `crop_highlights`**

In `shorts_generator/clipper.py`, add the import at the top:

```python
from .jump_cuts import excise_cut_segments
from .captions import CaptionError, burn_captions, burn_captions_segments
```

(replacing the existing `from .captions import CaptionError, burn_captions` line) and add `import subprocess` to the top-level imports (alongside the existing `import os`).

Replace the body of the `for i, h in enumerate(highlights, 1):` loop in `crop_highlights` (clipper.py:63-133) with:

```python
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
            cut_segments = h.get("cut_segments") or [
                {"start_time": h["start_time"], "end_time": h["end_time"]}
            ]
            want_excision = len(cut_segments) > 1

            if want_captions or want_hook_card or want_excision:
                os.makedirs(out_dir, exist_ok=True)
                filename = unique_short_filename(h.get("title"), used_names)
                final_path = os.path.join(out_dir, filename)
                downloaded_path = final_path + ".download.mp4"
                try:
                    _download_to(url, downloaded_path)

                    if want_excision:
                        try:
                            excised_path = final_path + ".excised.mp4"
                            excise_cut_segments(
                                downloaded_path, cut_segments, float(h["start_time"]), excised_path,
                            )
                            os.replace(excised_path, downloaded_path)
                        except subprocess.CalledProcessError as e:
                            print(f"[clip] {i} jump-cut excision skipped: {e}", flush=True)
                            entry["excision_error"] = str(e)
                            want_excision = False

                    if want_captions:
                        try:
                            if want_excision:
                                burn_captions_segments(
                                    downloaded_path,
                                    transcript_segments,
                                    cut_segments,
                                    final_path,
                                    fade_seconds=caption_fade_duration,
                                    word_highlight=word_highlight,
                                )
                            else:
                                burn_captions(
                                    downloaded_path,
                                    transcript_segments,
                                    float(h["start_time"]),
                                    float(h["end_time"]),
                                    final_path,
                                    fade_seconds=caption_fade_duration,
                                    word_highlight=word_highlight,
                                )
                        except CaptionError as e:
                            # Caption burn-in failed, but the download itself
                            # succeeded (and the hook card may already have
                            # too) -- fall back to the plain download rather
                            # than discarding everything back to the hosted
                            # URL, matching local mode's behavior.
                            print(f"[clip] {i} captions skipped: {e}", flush=True)
                            entry["captions_error"] = str(e)
                            os.replace(downloaded_path, final_path)
                    else:
                        os.replace(downloaded_path, final_path)

                    if want_hook_card:
                        try:
                            card_path = final_path + ".card.mp4"
                            render_card_overlay(final_path, hook_text, card_path)
                            os.replace(card_path, final_path)
                        except HookCardError as e:
                            print(f"[clip] {i} hook-card overlay skipped: {e}", flush=True)
                            entry["hook_card_error"] = str(e)

                    entry["clip_url"] = final_path
                    entry["hosted_clip_url"] = url
                except requests.RequestException as e:
                    # The download itself failed -- no local file exists at
                    # all, so there's nothing to fall back to except the
                    # hosted URL.
                    print(f"[clip] {i} download failed, falling back to hosted url: {e}", flush=True)
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

This threads `downloaded_path` as the single working file through excision → captions → hook-card, exactly like today's download → captions → hook-card chain, just with one new step spliced in before captions. A highlight with a single `cut_segments` entry (or none at all, via the `or [...]` fallback) sets `want_excision = False` and takes the exact pre-existing code path.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_clipper_api.py -v`
Expected: PASS — baseline count (Task 1) + 2 new tests, zero regressions in the pre-existing captions/hook-card tests.

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/clipper.py tests/test_clipper_api.py
git commit -m "feat: excise jump-cut gaps in api-mode clip download"
```

---

### Task 7: Wire excision + segment captions into local mode (`local/clipper.py`)

**Files:**
- Modify: `shorts_generator/local/clipper.py`
- Test: `tests/test_local_clipper.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_local_clipper.py` (it already has `synthetic_source`, `_highlight()`, `_segments()` helpers):

```python
def _probe_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def test_multi_cut_segments_excises_the_gap(tmp_path, synthetic_source):
    # synthetic_source spans [0,6]; highlight envelope is [1,4] (3s).
    # Keep [1,2] and [3,4], drop the [2,3] middle -> 2s output.
    highlight = {
        "title": "Test Clip", "start_time": 1.0, "end_time": 4.0, "score": 90,
        "cut_segments": [
            {"start_time": 1.0, "end_time": 2.0},
            {"start_time": 3.0, "end_time": 4.0},
        ],
    }

    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source,
        [highlight],
        aspect_ratio="9:16",
        out_dir=out_dir,
        captions=False,
        hook_card=False,
    )

    assert "error" not in results[0]
    duration = _probe_duration(results[0]["clip_url"])
    assert abs(duration - 2.0) < 0.3


def test_single_cut_segment_skips_excision(tmp_path, synthetic_source):
    highlight = {**_highlight(), "cut_segments": [{"start_time": 1.0, "end_time": 4.0}]}

    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source, [highlight], aspect_ratio="9:16", out_dir=out_dir,
        captions=False, hook_card=False,
    )

    assert "error" not in results[0]
    duration = _probe_duration(results[0]["clip_url"])
    assert abs(duration - 3.0) < 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_local_clipper.py -k "cut_segment" -v`
Expected: FAIL — output stays 3s (excision never runs), so the 2.0s assertion fails.

- [ ] **Step 3: Wire excision into `crop_clip_local` and segment captions into `crop_highlights_local`**

In `shorts_generator/local/clipper.py`, add the import at the top:

```python
from ..jump_cuts import excise_cut_segments
```

and change:

```python
from ..captions import CaptionError, burn_captions
```

to:

```python
from ..captions import CaptionError, burn_captions, burn_captions_segments
```

Replace `crop_clip_local` (local/clipper.py:559-583) with:

```python
def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    framing: str = "locked",
    cut_segments: Optional[List[Dict]] = None,
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path.

    framing="locked" (default): static speaker-centered crop for the whole
    clip. framing="adaptive": cursor/person-aware crop for screen-recording
    content that alternates between facecam and screen activity.

    cut_segments (optional): when it has more than one entry, the gaps
    between kept spans are excised (jump_cuts.excise_cut_segments) before
    reframing, so a reaction-jail dead-air trim survives the vertical crop.
    """
    cut_path = out_path + ".cut.mp4"
    excised_path = out_path + ".excised.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        working_path = cut_path
        if cut_segments and len(cut_segments) > 1:
            excise_cut_segments(cut_path, cut_segments, start_time, excised_path)
            working_path = excised_path
        if framing == "adaptive":
            _reframe_vertical_adaptive(working_path, out_path, aspect_ratio)
        else:
            _reframe_vertical(working_path, out_path, aspect_ratio)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
        if os.path.exists(excised_path):
            os.remove(excised_path)
    return out_path
```

Now update `crop_highlights_local` (local/clipper.py:586-651). In the `crop_clip_local(...)` call (currently lines 606-613), add the `cut_segments` argument:

```python
            cut_segments = h.get("cut_segments") or [
                {"start_time": float(h["start_time"]), "end_time": float(h["end_time"])}
            ]
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                framing=framing,
                cut_segments=cut_segments,
            )
```

And update the captions block (currently lines 619-636) to route through `burn_captions_segments` when there's more than one kept span:

```python
            if captions and transcript_segments:
                captioned_path = out_path + ".captioned.mp4"
                try:
                    if len(cut_segments) > 1:
                        burn_captions_segments(
                            out_path,
                            transcript_segments,
                            cut_segments,
                            captioned_path,
                            fade_seconds=caption_fade_duration,
                            word_highlight=word_highlight,
                        )
                    else:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_local_clipper.py -v`
Expected: PASS — baseline count (Task 1) + 2 new tests, zero regressions.

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/local/clipper.py tests/test_local_clipper.py
git commit -m "feat: excise jump-cut gaps in local-mode clip pipeline"
```

---

### Task 8: Surface `reaction_type` and `tightness_reason` in the webapp

**Files:**
- Modify: `shorts_generator/templates/index.html`

- [ ] **Step 1: Add a reaction badge and tightness-reason text block**

In `shorts_generator/templates/index.html`, in `buildShortCard` (around line 815-817), right after the existing `hook_reason` block:

```javascript
        if (s.hook_reason) {
          appendLabeledText(card, "Hook read", "reason", s.hook_reason);
        }

        if (s.reaction_type) {
          const reactionRow = document.createElement("div");
          reactionRow.className = "score-row";
          const reactionLabel = document.createElement("span");
          reactionLabel.textContent = "Reaction: " + s.reaction_type;
          reactionRow.appendChild(reactionLabel);
          card.appendChild(reactionRow);
        }

        if (s.tightness_reason) {
          appendLabeledText(card, "Tightness", "reason", s.tightness_reason);
        }
```

This reuses the existing `.score-row` class for the badge and the existing `appendLabeledText`/`.reason` pattern already used for `virality_reason` and `hook_reason` — no new CSS needed.

- [ ] **Step 2: Manually verify in the browser**

Run: `python -m shorts_generator.webapp` (or however this project's dev server is started — check `README.md`'s "Run" section if unsure), generate or open a past run with highlights that have `reaction_type`/`tightness_reason` set, and confirm the new badge/text render without layout breakage. If no real run has the new fields yet (expected, since Tasks 2-3 only changed what future LLM calls produce), temporarily paste a fake `reaction_type`/`tightness_reason` into a `result.json` from a past run, load its history page, and confirm rendering, then discard the edit.

- [ ] **Step 3: Commit**

```bash
git add shorts_generator/templates/index.html
git commit -m "feat: show reaction type and tightness reason on short cards"
```

---

## Definition of done

- [ ] `python -m pytest tests/ -q` passes with the same baseline count from Task 1 plus all new tests from Tasks 2-7 (0 regressions).
- [ ] A highlight with a single `cut_segments` entry (or none) produces byte-for-byte the same pipeline behavior as before this plan (verified by `test_single_cut_segment_skips_excision` in both `test_clipper_api.py` and `test_local_clipper.py`).
- [ ] A highlight with multiple `cut_segments` entries produces a shorter output clip with the gap(s) removed, in both api and local mode, with captions correctly offset onto the concatenated timeline.
- [ ] `reaction_type` and `tightness_reason` are visible in the webapp UI.
- [ ] `HIGHLIGHT_SCHEMA_VERSION` is 3; a `highlights.json` cache written before this plan is treated as a miss and recomputed.
