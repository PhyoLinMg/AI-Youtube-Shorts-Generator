# Caption Font & Style Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix caption "rearranging" jank (active-word scale bounce shifts the centered line) and replace the unbundled Arial base font with a bundled, reliable Montserrat Black.

**Architecture:** `shorts_generator/captions.py` builds an ASS subtitle file and burns it via ffmpeg's `subtitles` libass filter. Three surgical changes: (1) drop the `\fscx`/`\fscy` scale animation from the active-word highlight override tag, keeping color+bold only; (2) bundle `Montserrat-Black.ttf` under `shorts_generator/assets/fonts/` (mirrors the existing `Anton-Regular.ttf` used by `hook_card.py`) and set it as the ASS style's `Fontname`; (3) point ffmpeg at that bundled fonts directory via the `subtitles` filter's `fontsdir=` option so rendering no longer depends on the host machine having any particular font installed.

**Tech Stack:** Python, ffmpeg/ffprobe (subprocess), ASS/libass subtitle format, pytest.

---

### Task 1: License hygiene — split OFL.txt into per-font files

**Files:**
- Modify: `shorts_generator/assets/fonts/OFL.txt` → rename to `shorts_generator/assets/fonts/OFL-Anton.txt`
- Create: `shorts_generator/assets/fonts/OFL-Montserrat.txt`

The existing `OFL.txt` carries an Anton-specific copyright header but a generic SIL OFL 1.1 body. Once a second OFL font is bundled, one shared `OFL.txt` is ambiguous about which font it covers — split per font, same pattern e.g. `google/fonts` uses per-family OFL files.

- [ ] **Step 1: Rename the existing license file**

```bash
git mv shorts_generator/assets/fonts/OFL.txt shorts_generator/assets/fonts/OFL-Anton.txt
```

- [ ] **Step 2: Create the Montserrat license file**

Same SIL OFL 1.1 body as `OFL-Anton.txt`, with Montserrat's copyright header:

```
Copyright 2011 The Montserrat Project Authors (https://github.com/JulietaUla/Montserrat)

This Font Software is licensed under the SIL Open Font License, Version 1.1.
This license is copied below, and is also available with a FAQ at:
http://scripts.sil.org/OFL


-----------------------------------------------------------
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
-----------------------------------------------------------

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded, 
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
```

Write this exact content to `shorts_generator/assets/fonts/OFL-Montserrat.txt`.

- [ ] **Step 3: Commit**

```bash
git add shorts_generator/assets/fonts/OFL-Anton.txt shorts_generator/assets/fonts/OFL-Montserrat.txt
git commit -m "chore: split font OFL license into per-family files"
```

---

### Task 2: Bundle Montserrat-Black.ttf

**Files:**
- Create: `shorts_generator/assets/fonts/Montserrat-Black.ttf`

The static Black-weight file is already present on this machine at
`/Users/linnmaung/Library/Fonts/Montserrat-Black.ttf` (confirmed via `fc-scan`:
family `Montserrat,Montserrat Black`, style `Black,Regular`). Montserrat is
SIL OFL licensed (Task 1 adds the license text), so bundling it into the repo
is permitted.

- [ ] **Step 1: Copy the font file into the repo**

```bash
cp /Users/linnmaung/Library/Fonts/Montserrat-Black.ttf shorts_generator/assets/fonts/Montserrat-Black.ttf
```

- [ ] **Step 2: Verify the family name libass will resolve**

```bash
fc-scan --format '%{family}\t%{style}\n' shorts_generator/assets/fonts/Montserrat-Black.ttf
```

Expected output: `Montserrat,Montserrat Black\tBlack,Regular` — confirms the
font's family name includes `Montserrat Black`, which is what the ASS
`Fontname` field will reference in Task 4.

- [ ] **Step 3: Commit**

```bash
git add shorts_generator/assets/fonts/Montserrat-Black.ttf
git commit -m "chore: bundle Montserrat Black font for captions"
```

---

### Task 3: Drop the active-word scale bounce

**Files:**
- Modify: `shorts_generator/captions.py:13-18` (the `_HIGHLIGHT_OPEN` constant and its comment)
- Test: `tests/test_captions.py`

The bounce (`\fscx125\fscy125` → `\fscx100\fscy100`) changes the active
word's rendered width. Because the caption style is centered
(`Alignment: 2`), a width change on any word re-centers the whole line —
this is the "rearranging" jank. Dropping the scale tags and keeping
`\c&H00FFFF&\b1` (yellow + bold) gives a color pop with zero width change:
synthetic bold thickens strokes in place without altering advance width.

- [ ] **Step 1: Update the existing highlight test to assert no bounce**

In `tests/test_captions.py`, find `test_write_ass_emits_one_dialogue_per_word_with_highlight`
(around line 99) and replace:

```python
    assert "\\t(0,80,\\fscx125\\fscy125)" in content  # bounce
```

with:

```python
    assert "\\b1" in content                         # bold
    assert "\\fscx" not in content                    # no scale bounce
    assert "\\fscy" not in content                    # no scale bounce
```

- [ ] **Step 2: Add a direct regression test on the constant**

Add to `tests/test_captions.py`'s import block (currently `from shorts_generator.captions import (...)`):

```python
from shorts_generator.captions import (
    CaptionError,
    _HIGHLIGHT_OPEN,
    _chunk_segments,
    _format_ass_timestamp,
    _probe_resolution,
    _write_ass,
    burn_captions,
)
```

Then add this new test function anywhere after the imports:

```python
def test_highlight_open_has_no_scale_bounce():
    """Scale animation on the active word changes its rendered width, which
    re-centers the whole (centered) caption line as focus moves word to
    word. The highlight must pop via color/bold only, never scale."""
    assert "\\fscx" not in _HIGHLIGHT_OPEN
    assert "\\fscy" not in _HIGHLIGHT_OPEN
    assert "\\c&H00FFFF&" in _HIGHLIGHT_OPEN
    assert "\\b1" in _HIGHLIGHT_OPEN
```

- [ ] **Step 3: Run the tests to confirm they fail**

```bash
pytest tests/test_captions.py -k "highlight" -v
```

Expected: both tests FAIL — `test_write_ass_emits_one_dialogue_per_word_with_highlight`
still finds `\fscx` in the output, and `test_highlight_open_has_no_scale_bounce`
finds `\fscx` still present in `_HIGHLIGHT_OPEN`.

- [ ] **Step 4: Remove the scale bounce**

In `shorts_generator/captions.py`, replace:

```python
# ASS override tags for the karaoke-style active-word highlight: pop to
# yellow + bold, bounce to 125% scale over the first 80ms of the word's own
# Dialogue line, then settle back to 100% by 160ms. `{\r}` resets back to
# the line's base `Caption` style for the remainder of the text.
_HIGHLIGHT_OPEN = "{\\c&H00FFFF&\\b1\\t(0,80,\\fscx125\\fscy125)\\t(80,160,\\fscx100\\fscy100)}"
_HIGHLIGHT_CLOSE = "{\\r}"
```

with:

```python
# ASS override tags for the karaoke-style active-word highlight: pop to
# yellow + bold. No scale/size change — the caption line is centered, so
# resizing the active word would shift the whole line's rendered width
# and re-center it every word (visible as the line jumping side to side).
# `{\r}` resets back to the line's base `Caption` style for the remainder
# of the text.
_HIGHLIGHT_OPEN = "{\\c&H00FFFF&\\b1}"
_HIGHLIGHT_CLOSE = "{\\r}"
```

- [ ] **Step 5: Run the tests to confirm they pass**

```bash
pytest tests/test_captions.py -k "highlight" -v
```

Expected: PASS for both tests.

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/captions.py tests/test_captions.py
git commit -m "fix: drop active-word scale bounce that re-centered caption lines"
```

---

### Task 4: Switch caption base font to Montserrat Black

**Files:**
- Modify: `shorts_generator/captions.py` (`_write_ass`'s `[V4+ Styles]` line)
- Test: `tests/test_captions.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_captions.py`:

```python
def test_write_ass_uses_montserrat_black_font(tmp_path):
    chunks = [{"start": 0.0, "end": 1.0, "text": "hello world"}]
    ass_path = str(tmp_path / "c.ass")

    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3)

    content = open(ass_path, encoding="utf-8").read()
    assert "Style: Caption,Montserrat Black," in content
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
pytest tests/test_captions.py::test_write_ass_uses_montserrat_black_font -v
```

Expected: FAIL — style line currently reads `Style: Caption,Arial,...`.

- [ ] **Step 3: Change the style's Fontname**

In `shorts_generator/captions.py`, inside `_write_ass`, find:

```python
        f"Style: Caption,Arial,{fontsize},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
```

Replace with:

```python
        f"Style: Caption,Montserrat Black,{fontsize},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
```

- [ ] **Step 4: Run it to confirm it passes**

```bash
pytest tests/test_captions.py::test_write_ass_uses_montserrat_black_font -v
```

Expected: PASS.

- [ ] **Step 5: Run the full caption test file to check nothing else broke**

```bash
pytest tests/test_captions.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/captions.py tests/test_captions.py
git commit -m "feat: set caption style font to Montserrat Black"
```

---

### Task 5: Bundle the font at render time via ffmpeg fontsdir

**Files:**
- Modify: `shorts_generator/captions.py` (new `FONT_DIR` constant, new `_escape_ffmpeg_path` helper, `burn_captions`'s ffmpeg command)
- Test: `tests/test_captions.py`

Setting `Fontname: Montserrat Black` (Task 4) only works if libass can find
that font. Right now it depends on host fontconfig, same problem the
project already solved for `hook_card.py` via an explicit `FONT_PATH`. The
`subtitles` filter's `fontsdir=` option is the equivalent fix here: point
it at `shorts_generator/assets/fonts/` so libass loads the bundled ttf
directly, independent of the host's installed fonts.

The existing `ass_path` escaping (`.replace("\\", "/").replace(":", "\\:")`)
gets extracted into a small helper since it's now needed for two ffmpeg
filter arguments (the `.ass` path and the fonts dir) — DRY, and testable in
isolation.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_captions.py`:

```python
def test_escape_ffmpeg_path_escapes_backslashes_and_colons():
    assert _escape_ffmpeg_path("C:\\videos\\out.ass") == "C:/videos/out.ass".replace(":", "\\:")


def test_burn_captions_vf_includes_fontsdir(tmp_path, synthetic_clip, monkeypatch):
    """The subtitles filter must point at the bundled fonts directory so
    Montserrat Black renders even on a host with no matching system font."""
    captured = {}
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffmpeg":
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr("shorts_generator.captions.subprocess.run", fake_run)

    out_path = str(tmp_path / "burned.mp4")
    segments = [{"start": 0.0, "end": 3.0, "text": "hello there caption test"}]

    burn_captions(synthetic_clip, segments, clip_start=0.0, clip_end=3.0, out_path=out_path, fade_seconds=0.3)

    vf_arg = captured["cmd"][captured["cmd"].index("-vf") + 1]
    assert vf_arg.startswith("subtitles=")
    assert ":fontsdir=" in vf_arg
    assert vf_arg.endswith(_escape_ffmpeg_path(FONT_DIR))
```

Add `FONT_DIR` and `_escape_ffmpeg_path` to the existing import block at the
top of `tests/test_captions.py`:

```python
from shorts_generator.captions import (
    CaptionError,
    FONT_DIR,
    _HIGHLIGHT_OPEN,
    _chunk_segments,
    _escape_ffmpeg_path,
    _format_ass_timestamp,
    _probe_resolution,
    _write_ass,
    burn_captions,
)
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
pytest tests/test_captions.py -k "escape_ffmpeg_path or fontsdir" -v
```

Expected: FAIL with `ImportError` (`FONT_DIR` / `_escape_ffmpeg_path` don't
exist yet) or `ModuleNotFoundError`-style collection error.

- [ ] **Step 3: Add FONT_DIR and the escaping helper**

In `shorts_generator/captions.py`, near the top after the existing imports
and `_HIGHLIGHT_OPEN`/`_HIGHLIGHT_CLOSE` constants, add:

```python
FONT_DIR = os.path.join(os.path.dirname(__file__), "assets", "fonts")


def _escape_ffmpeg_path(path: str) -> str:
    """Escape a filesystem path for use as an ffmpeg filter argument value
    (backslashes to forward slashes, colons escaped so they aren't parsed
    as a filter option separator)."""
    return path.replace("\\", "/").replace(":", "\\:")
```

- [ ] **Step 4: Wire fontsdir into the ffmpeg command**

In `shorts_generator/captions.py`, inside `burn_captions`, replace:

```python
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
```

with:

```python
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
```

- [ ] **Step 5: Run the tests to confirm they pass**

```bash
pytest tests/test_captions.py -k "escape_ffmpeg_path or fontsdir" -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/captions.py tests/test_captions.py
git commit -m "feat: bundle caption font via ffmpeg fontsdir instead of host fontconfig"
```

---

### Task 6: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all PASS, including the pre-existing `test_burn_captions_produces_output_file`
(now exercising the real Montserrat Black + fontsdir path end to end).

- [ ] **Step 2: Manual visual check**

Burn captions on a real short clip through the normal pipeline entry point
(`shorts_generator/pipeline.py` or `shorts_generator/local/clipper.py`,
whichever this project run uses) and inspect the output:

- Caption text renders in Montserrat Black (thick, not thin default Arial).
- Active word turns yellow/bold in place — the line does not shift left or
  right as focus moves from word to word.

- [ ] **Step 3: Confirm final git status is clean**

```bash
git status
git log --oneline -6
```

Expected: working tree clean (aside from any unrelated pre-existing dirty
files noted at session start — `shorts_generator/config.py`,
`shorts_generator/highlights.py`, `shorts_generator/local/llm.py`,
`tests/test_highlights.py`, and untracked `AGENTS.md`, `result.json`,
`run.log`, `tests/test_local_llm.py`, which are out of scope for this
plan), and 5 new commits from Tasks 1–5 on top of the earlier spec-doc
commit (6 total related commits).
