# Word-Highlight Caption Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Highlight+bounce the spoken word in burned-in captions, both modes, on by default.

**Architecture:** faster-whisper emits per-word times (local); char-length estimate synthesizes them (api). `_chunk_segments` unifies both into `{start,end,text,words[]}`. `_write_ass` emits one Dialogue line per word with ASS override tags on the active word. New `word_highlight` bool threads main.py → pipeline → both crop functions → `burn_captions` → `_write_ass`.

**Tech Stack:** Python, faster-whisper, ffmpeg/libass (ASS override tags), pytest.

---

## Context

Current captions (`shorts_generator/captions.py`) burn one bottom-center phrase per chunk with a plain `\fad` fade-in. User wants a "fancier" caption: the word currently being spoken pops to a bright accent color, goes bold, and does a quick scale bounce (CapCut/TikTok karaoke look). Approved design spec: `docs/superpowers/specs/2026-07-14-caption-word-highlight-design.md`.

Two timing sources feed the same renderer:
- `--mode local` (faster-whisper) → real per-word timestamps via `word_timestamps=True`.
- `--mode api` (MuAPI `/openai-whisper`) → no per-word data → estimate per-word windows by character length within each phrase chunk's known duration.

Both converge on a chunk shape where **every chunk carries a `"words"` list** (clip-relative, clipped, non-empty), so `_write_ass` never branches on mode. On by default; new `--no-word-highlight` flag restores the old plain-phrase look.

## File Structure

- `shorts_generator/local/transcriber.py` — enable `word_timestamps=True`, collect `.words`, switch cache `.srt` → `.json`.
- `shorts_generator/captions.py` — word-level chunking (real + estimate paths), per-word Dialogue rendering with highlight tags, `word_highlight` param.
- `shorts_generator/clipper.py`, `shorts_generator/local/clipper.py` — add `word_highlight` param, pass through to `burn_captions`.
- `shorts_generator/pipeline.py` — thread `word_highlight` through `_run_api`/`_run_local`/`generate_shorts`.
- `main.py` — `--no-word-highlight` flag.
- Tests: `tests/test_captions.py`, `tests/test_local_clipper.py`, `tests/test_clipper_api.py`, `tests/test_pipeline.py`, `tests/test_main.py`, plus new `tests/test_local_transcriber.py`.

## Key design details

**ASS highlight tag** (active word), wrapped around the word, reset with `{\r}` after:
```
{\c&H00FFFF&\b1\t(0,80,\fscx125\fscy125)\t(80,160,\fscx100\fscy100)}WORD{\r}
```
- `&H00FFFF&` = ASS BGR yellow (R255 G255 B0). `\b1` bold. `\t` timings are relative to that Dialogue line's own Start, so bounce (100→125 over 80ms, 125→100 by 160ms) lands exactly when the word goes active. Fixed constants, not CLI-configurable.
- Only a chunk's **first** word line additionally gets the existing `{\fad(fade_ms,0)}` prefix — phrase fades in once, no flicker on word swaps.

**Unified chunk contract** from `_chunk_segments`: every chunk `{"start","end","text","words":[{"start","end","text"}, ...]}`, all clip-relative, all windows non-empty. `text` is derived from the surviving words (`" ".join`). Real-words path clips each word to the window and drops empty ones; estimate path synthesizes words inside each already-clipped chunk.

---

## Task 1: faster-whisper per-word timestamps + JSON cache

**Files:**
- Modify: `shorts_generator/local/transcriber.py`
- Create: `tests/test_local_transcriber.py`

- [ ] **Step 1: Write failing test** — `tests/test_local_transcriber.py`:
```python
import json
from pathlib import Path

import shorts_generator.local.transcriber as tr


def test_json_cache_roundtrip_preserves_words(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "LOCAL_OUTPUT_DIR", str(tmp_path))
    transcript = {
        "duration": 4.0,
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "hello world",
             "words": [
                 {"start": 0.0, "end": 1.0, "word": "hello"},
                 {"start": 1.0, "end": 2.0, "word": "world"},
             ]},
        ],
    }
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    cache_path = tr._write_json_cache(str(media), transcript)
    assert cache_path.suffix == ".json"

    loaded = tr._load_json_cache(cache_path)
    assert loaded["segments"][0]["words"][1]["word"] == "world"
    assert loaded["duration"] == 4.0


def test_cache_path_is_json(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "LOCAL_OUTPUT_DIR", str(tmp_path))
    p = tr._transcript_cache_path("/some/video.mp4")
    assert p.name == "video.json"
```

- [ ] **Step 2: Run — expect FAIL** (`_write_json_cache`/`_load_json_cache` undefined, `.srt` suffix):
`.venv/bin/python -m pytest tests/test_local_transcriber.py -v`

- [ ] **Step 3: Implement** in `shorts_generator/local/transcriber.py`:
  - `_transcript_cache_path`: change suffix from `.srt` to `.json`.
  - Replace `_write_srt_cache`/`_load_srt_cache` (and the SRT timestamp helpers `_format_srt_timestamp`/`_parse_srt_timestamp`, now unused) with:
```python
import json

def _write_json_cache(media_path: str, transcript: Dict) -> Path:
    cache_path = _transcript_cache_path(media_path)
    cache_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
    return cache_path


def _load_json_cache(cache_path: Path) -> Dict:
    content = cache_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return {"duration": 0.0, "segments": []}
    data = json.loads(content)
    return {
        "duration": float(data.get("duration", 0.0)),
        "segments": data.get("segments", []),
    }
```
  - In `transcribe_local`, point the cache-hit branch at `_load_json_cache` and the write branch at `_write_json_cache` (keep the existing mtime freshness check and empty-cache invalidation logic unchanged, just swapped loader/writer names).
  - Add `"word_timestamps": True` to `transcribe_kwargs`.
  - In the segment collection loop, attach words:
```python
for s in segments_iter:
    words = [
        {"start": float(w.start), "end": float(w.end), "word": (w.word or "").strip()}
        for w in (getattr(s, "words", None) or [])
    ]
    segments.append({
        "start": float(s.start),
        "end": float(s.end),
        "text": (s.text or "").strip(),
        "words": words,
    })
```

- [ ] **Step 4: Run — expect PASS**: `.venv/bin/python -m pytest tests/test_local_transcriber.py -v`

- [ ] **Step 5: Commit**:
```bash
git add shorts_generator/local/transcriber.py tests/test_local_transcriber.py
git commit -m "feat: per-word whisper timestamps + json transcript cache"
```

---

## Task 2: Word-level chunking in captions.py

**Files:**
- Modify: `shorts_generator/captions.py`
- Modify: `tests/test_captions.py`

- [ ] **Step 1: Write failing tests** — add to `tests/test_captions.py`:
```python
def test_chunk_segments_uses_real_word_timestamps():
    segments = [{
        "start": 10.0, "end": 12.0, "text": "alpha beta gamma",
        "words": [
            {"start": 10.0, "end": 10.5, "word": "alpha"},
            {"start": 10.5, "end": 11.2, "word": "beta"},
            {"start": 11.2, "end": 12.0, "word": "gamma"},
        ],
    }]
    chunks = _chunk_segments(segments, clip_start=10.0, clip_end=20.0, max_words=7)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["text"] == "alpha beta gamma"
    assert [w["text"] for w in c["words"]] == ["alpha", "beta", "gamma"]
    assert c["words"][0]["start"] == 0.0          # 10.0 - clip_start
    assert c["words"][2]["end"] == 2.0            # 12.0 - clip_start
    assert c["start"] == 0.0 and c["end"] == 2.0


def test_chunk_segments_estimates_words_when_absent():
    segments = [{"start": 0.0, "end": 4.0, "text": "hi supercalifragilistic"}]
    chunks = _chunk_segments(segments, clip_start=0.0, clip_end=100.0, max_words=7)
    words = chunks[0]["words"]
    assert [w["text"] for w in words] == ["hi", "supercalifragilistic"]
    # char-length weighted: 2 vs 20 chars over 4.0s → ~0.36s vs ~3.64s
    assert words[1]["end"] - words[1]["start"] > words[0]["end"] - words[0]["start"]
    assert words[0]["start"] == 0.0
    assert words[-1]["end"] == pytest.approx(4.0)
```
  Update the existing `test_chunk_segments_splits_by_word_count_and_time_share`, `test_chunk_segments_drops_segments_outside_window`, `test_chunk_segments_clips_and_shifts_straddling_segment` to also assert `"words"` is present on each returned chunk (and word windows fall within `[chunk start, chunk end]`); keep their existing `start`/`end`/`text` assertions.

- [ ] **Step 2: Run — expect FAIL**: `.venv/bin/python -m pytest tests/test_captions.py -k chunk -v`

- [ ] **Step 3: Implement** in `shorts_generator/captions.py`. Add helper and rework `_chunk_segments`:
```python
def _estimate_word_windows(words: List[str], start: float, end: float) -> List[Dict]:
    """Apportion [start, end] across words by character length."""
    total_chars = sum(len(w) for w in words) or 1
    span = end - start
    out = []
    cursor = start
    for w in words:
        share = len(w) / total_chars
        w_end = cursor + span * share
        out.append({"start": cursor, "end": w_end, "text": w})
        cursor = w_end
    if out:
        out[-1]["end"] = end  # absorb float drift
    return out


def _chunk_from_real_words(
    seg_words: List[Dict], clip_start: float, clip_end: float, max_words: int
) -> List[Dict]:
    chunks = []
    for i in range(0, len(seg_words), max_words):
        group = seg_words[i:i + max_words]
        kept = []
        for w in group:
            ws = max(float(w["start"]), clip_start)
            we = min(float(w["end"]), clip_end)
            if we <= ws:
                continue
            kept.append({
                "start": ws - clip_start,
                "end": we - clip_start,
                "text": str(w.get("word", "")).strip(),
            })
        if not kept:
            continue
        chunks.append({
            "start": kept[0]["start"],
            "end": kept[-1]["end"],
            "text": " ".join(w["text"] for w in kept),
            "words": kept,
        })
    return chunks
```
  Rework the body of `_chunk_segments` so each segment either:
  - has a non-empty `"words"` list → extend chunks with `_chunk_from_real_words(seg["words"], clip_start, clip_end, max_words)`; **or**
  - has none → run the existing proportional-by-word-count split (unchanged) to get each chunk's `start`/`end`/`text`, then set `chunk["words"] = _estimate_word_windows(group, chunk["start"], chunk["end"])` where `group` is that chunk's word-strings (relative windows since `chunk["start"]/["end"]` are already clip-relative).

  Keep the return contract: list of `{"start","end","text","words"}`.

- [ ] **Step 4: Run — expect PASS**: `.venv/bin/python -m pytest tests/test_captions.py -k chunk -v`

- [ ] **Step 5: Commit**:
```bash
git add shorts_generator/captions.py tests/test_captions.py
git commit -m "feat: word-level caption chunking (real + estimated timestamps)"
```

---

## Task 3: Per-word highlight rendering in _write_ass

**Files:**
- Modify: `shorts_generator/captions.py`
- Modify: `tests/test_captions.py`

- [ ] **Step 1: Write failing tests** — add to `tests/test_captions.py`:
```python
def test_write_ass_emits_one_dialogue_per_word_with_highlight(tmp_path):
    chunks = [{
        "start": 0.0, "end": 2.0, "text": "alpha beta",
        "words": [
            {"start": 0.0, "end": 1.0, "text": "alpha"},
            {"start": 1.0, "end": 2.0, "text": "beta"},
        ],
    }]
    ass_path = str(tmp_path / "c.ass")
    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3)
    content = open(ass_path, encoding="utf-8").read()
    assert content.count("Dialogue:") == 2          # one per word
    assert "\\c&H00FFFF&" in content                # yellow highlight
    assert "\\t(0,80,\\fscx125\\fscy125)" in content  # bounce
    assert content.count("\\fad(300,0)") == 1       # fade on first word only


def test_write_ass_word_highlight_false_is_one_line_per_chunk(tmp_path):
    chunks = [{
        "start": 0.0, "end": 2.0, "text": "alpha beta",
        "words": [
            {"start": 0.0, "end": 1.0, "text": "alpha"},
            {"start": 1.0, "end": 2.0, "text": "beta"},
        ],
    }]
    ass_path = str(tmp_path / "c.ass")
    _write_ass(chunks, ass_path, width=608, height=1080, fade_seconds=0.3, word_highlight=False)
    content = open(ass_path, encoding="utf-8").read()
    assert content.count("Dialogue:") == 1
    assert "\\c&H00FFFF&" not in content
    assert "\\fad(300,0)" in content
```
  Update `test_write_ass_contains_resolution_and_fade_tag` and `test_write_ass_strips_braces_from_text` to pass chunks that include a `"words"` list (default `word_highlight=True` now emits per-word lines; adjust the `Dialogue:` count assertion accordingly, or pass `word_highlight=False` to keep asserting the one-line-per-chunk shape).

- [ ] **Step 2: Run — expect FAIL**: `.venv/bin/python -m pytest tests/test_captions.py -k write_ass -v`

- [ ] **Step 3: Implement** — change `_write_ass` signature to `(..., fade_seconds, word_highlight: bool = True)`. Add a highlight builder and branch the Dialogue emission:
```python
_HIGHLIGHT_OPEN = "{\\c&H00FFFF&\\b1\\t(0,80,\\fscx125\\fscy125)\\t(80,160,\\fscx100\\fscy100)}"
_HIGHLIGHT_CLOSE = "{\\r}"


def _clean(text: str) -> str:
    return text.replace("{", "").replace("}", "").replace("\n", " ")


def _render_word_line(words: List[str], active: int) -> str:
    parts = []
    for i, w in enumerate(words):
        if i == active:
            parts.append(f"{_HIGHLIGHT_OPEN}{w}{_HIGHLIGHT_CLOSE}")
        else:
            parts.append(w)
    return " ".join(parts)
```
  In the events loop:
```python
for chunk in chunks:
    words = chunk["words"]
    word_texts = [_clean(w["text"]) for w in words]
    if word_highlight:
        for i, w in enumerate(words):
            start_ts = _format_ass_timestamp(w["start"])
            end_ts = _format_ass_timestamp(w["end"])
            fad = f"{{\\fad({fade_ms},0)}}" if i == 0 else ""
            text = fad + _render_word_line(word_texts, i)
            lines.append(f"Dialogue: 0,{start_ts},{end_ts},Caption,,0,0,0,,{text}\n")
    else:
        start_ts = _format_ass_timestamp(chunk["start"])
        end_ts = _format_ass_timestamp(chunk["end"])
        text = _clean(chunk["text"])
        lines.append(
            f"Dialogue: 0,{start_ts},{end_ts},Caption,,0,0,0,,{{\\fad({fade_ms},0)}}{text}\n"
        )
```

- [ ] **Step 4: Run — expect PASS**: `.venv/bin/python -m pytest tests/test_captions.py -v`

- [ ] **Step 5: Commit**:
```bash
git add shorts_generator/captions.py tests/test_captions.py
git commit -m "feat: per-word highlight + bounce rendering in ASS captions"
```

---

## Task 4: Thread word_highlight through burn_captions + crop functions

**Files:**
- Modify: `shorts_generator/captions.py` (`burn_captions`)
- Modify: `shorts_generator/local/clipper.py`, `shorts_generator/clipper.py`
- Modify: `tests/test_local_clipper.py`, `tests/test_clipper_api.py`

- [ ] **Step 1: Write failing tests** — add to `tests/test_local_clipper.py`:
```python
def test_word_highlight_flag_forwarded_to_burn(tmp_path, synthetic_source, monkeypatch):
    captured = {}

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        # produce a real output so crop still "succeeds"
        import shutil
        shutil.copyfile(args[0], args[4])
        return args[4]

    monkeypatch.setattr("shorts_generator.local.clipper.burn_captions", _spy)
    crop_highlights_local(
        synthetic_source, [_highlight()], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
        word_highlight=False,
    )
    assert captured["word_highlight"] is False
```
  Add the analogous test to `tests/test_clipper_api.py` (spy on `clipper.burn_captions`, assert `word_highlight` forwarded; reuse the existing `_download_to` monkeypatch pattern).

- [ ] **Step 2: Run — expect FAIL**: `.venv/bin/python -m pytest tests/test_local_clipper.py tests/test_clipper_api.py -k word_highlight -v`

- [ ] **Step 3: Implement**:
  - `burn_captions(...)` gains `word_highlight: bool = True`, passed into `_write_ass(chunks, ass_path, width, height, fade_seconds, word_highlight=word_highlight)`.
  - `crop_highlights_local(...)` gains `word_highlight: bool = True`; pass `word_highlight=word_highlight` in its `burn_captions(...)` call.
  - `crop_highlights(...)` (api) gains `word_highlight: bool = True`; pass it into its `burn_captions(...)` call.

- [ ] **Step 4: Run — expect PASS**: `.venv/bin/python -m pytest tests/test_local_clipper.py tests/test_clipper_api.py -v`

- [ ] **Step 5: Commit**:
```bash
git add shorts_generator/captions.py shorts_generator/local/clipper.py shorts_generator/clipper.py tests/test_local_clipper.py tests/test_clipper_api.py
git commit -m "feat: thread word_highlight flag through crop + burn_captions"
```

---

## Task 5: Pipeline + CLI wiring

**Files:**
- Modify: `shorts_generator/pipeline.py`, `main.py`
- Modify: `tests/test_pipeline.py`, `tests/test_main.py`

- [ ] **Step 1: Write failing tests**:
  - `tests/test_main.py` — add:
```python
def test_word_highlight_on_by_default():
    args = build_parser().parse_args(["https://example.com/video"])
    assert args.word_highlight is True


def test_no_word_highlight_flag_disables():
    args = build_parser().parse_args(["https://example.com/video", "--no-word-highlight"])
    assert args.word_highlight is False
```
  - `tests/test_pipeline.py` — extend both `_run_local`/`_run_api` calls to pass `word_highlight=False` and assert `crop_mock.call_args.kwargs["word_highlight"] is False`.

- [ ] **Step 2: Run — expect FAIL**: `.venv/bin/python -m pytest tests/test_main.py tests/test_pipeline.py -v`

- [ ] **Step 3: Implement**:
  - `main.py`: add flag mirroring `--no-captions`:
```python
parser.add_argument(
    "--no-word-highlight",
    dest="word_highlight",
    action="store_false",
    default=True,
    help="Disable per-word highlight animation; caption shows a plain fading phrase instead.",
)
```
    and pass `word_highlight=args.word_highlight` into `generate_shorts(...)`.
  - `shorts_generator/pipeline.py`: add `word_highlight: bool = True` param to `generate_shorts`, `_run_local`, `_run_api`; pass `word_highlight=word_highlight` into `crop_highlights_local(...)` and `crop_highlights(...)`; forward it in the two `generate_shorts` dispatch calls.

- [ ] **Step 4: Run — expect PASS**: `.venv/bin/python -m pytest tests/test_main.py tests/test_pipeline.py -v`

- [ ] **Step 5: Commit**:
```bash
git add main.py shorts_generator/pipeline.py tests/test_pipeline.py tests/test_main.py
git commit -m "feat: --no-word-highlight CLI flag wired through pipeline"
```

---

## Task 6: Full suite + docs + manual verification

- [ ] **Step 1: Full test suite** — expect all green:
`.venv/bin/python -m pytest -q`

- [ ] **Step 2: README** — update the caption section to mention word-highlight is default and document `--no-word-highlight`. Update the spec doc status to Implemented.

- [ ] **Step 3: Manual E2E** on an existing local source (already downloaded at `output/source__g4l7YkDQwA.mp4`). Note: cache is now JSON, so delete the stale `output/source__g4l7YkDQwA.srt` first (it will be ignored anyway) and let it re-transcribe once:
```bash
.venv/bin/python main.py "file://$PWD/output/source__g4l7YkDQwA.mp4" \
    --mode local --num-clips 1 --output-json result.json
```
  Open the produced `output/short_01.mp4`, confirm each spoken word pops yellow/bold with a bounce in time with the audio. Then confirm the fallback:
```bash
.venv/bin/python main.py "file://$PWD/output/source__g4l7YkDQwA.mp4" \
    --mode local --num-clips 1 --no-word-highlight
```
  (plain fading phrases, no per-word color).

- [ ] **Step 4: Commit docs**:
```bash
git add README.md docs/superpowers/specs/2026-07-14-caption-word-highlight-design.md
git commit -m "docs: document word-highlight captions and --no-word-highlight"
```

- [ ] **Step 5: Push to fork branch**:
```bash
git push fork caption-word-highlight
```

---

## Verification

- **Automated:** `.venv/bin/python -m pytest -q` — all pass, including new word-chunking, per-word render, cache round-trip, and flag-forwarding tests.
- **Manual:** local-mode E2E run above; eyeball highlight color, bold, bounce timing against spoken audio; confirm `--no-word-highlight` restores plain phrases. (API mode uses the char-length estimate — same render path, verified structurally by tests since it needs live MuAPI.)
- **Regression watch:** existing `test_captions.py` fade/brace/probe/burn tests still pass; `.srt`→`.json` cache switch doesn't break `transcribe_local` freshness/empty-cache logic.

## Notes / risks

- ASS color is BGR: yellow = `&H00FFFF&`. Bounce `\t` timings are line-relative, so they fire when each word's Dialogue line starts — correct by construction.
- Old `.srt` caches are silently ignored (different extension); first run per source re-transcribes once. Acceptable per design.
- Uncommitted working-tree changes (config/llm/crop OpenRouter edits, `result.json`, `run.log`) are unrelated and stay out of these commits — stage only the listed files per task.
