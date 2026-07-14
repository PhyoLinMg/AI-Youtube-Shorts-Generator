# Word-highlight caption animation — design spec

Date: 2026-07-14
Status: Approved, pending implementation plan

## Goal

Make the burned-in captions (`shorts_generator/captions.py`) more "fancy":
highlight the word currently being spoken, with a color-pop + bounce
animation, instead of the current plain fade-in-per-phrase. Available in
both `--mode api` and `--mode local`; on by default; one new CLI escape
hatch (`--no-word-highlight`) to fall back to the old plain-phrase look.

## Scope decisions (from brainstorming)

- **Animation style**: active word switches to a fixed accent color (bright
  yellow) + bold, and does a quick scale-up/settle bounce (125% → 100% over
  160ms) exactly when it starts being spoken. Rest of the phrase stays the
  existing plain white style. Not configurable (color/bounce values are
  fixed constants) — consistent with the existing fade spec's "minimal
  styling surface" call: only on/off is a flag, not the look itself.
- **Word timing source**:
  - `--mode local` (faster-whisper): turn on `word_timestamps=True` and use
    the model's real per-word start/end times.
  - `--mode api` (MuAPI `/openai-whisper`): no per-word data available from
    that endpoint. Falls back to splitting each phrase chunk's known
    duration across its words proportional to word character length
    (longer words get more time).
  - Both paths converge on the same chunk shape (every chunk carries a
    `"words"` list), so `_write_ass` doesn't need to know which mode
    produced the timing.
- **Local transcript cache format**: switches from `.srt` to `.json` (needs
  to store per-word timestamps, which SRT can't represent). Existing `.srt`
  caches on disk are simply ignored going forward; the next run re-
  transcribes once and writes the new `.json` cache. No migration needed.
- **Escape hatch**: `--no-word-highlight` (mirrors the existing
  `--no-captions` pattern) disables the per-word split and restores the
  original single-Dialogue-line-per-chunk, phrase-level-fade-only look.

## Architecture

### `shorts_generator/local/transcriber.py`

- `transcribe_local()`: add `"word_timestamps": True` to the
  `model.transcribe(...)` kwargs. Each `faster_whisper` segment object
  exposes `.words` (a list of word objects with `.start`, `.end`, `.word`)
  when this flag is on. Collect them:
  ```python
  words = [{"start": float(w.start), "end": float(w.end), "word": w.word.strip()} for w in (s.words or [])]
  ```
  and store as `segments[i]["words"]`.
- `_transcript_cache_path()`: extension changes from `.srt` to `.json`.
- `_write_srt_cache` / `_load_srt_cache` are replaced by
  `_write_json_cache` / `_load_json_cache`, which `json.dump`/`json.load`
  the full `{"duration": ..., "segments": [...]}` shape (segments now
  including their `"words"` lists) directly — no timestamp text-format
  round-trip needed.
- `transcribe_local()`'s cache-freshness check (mtime comparison) and
  empty-cache invalidation logic stay the same, just pointed at the new
  loader/writer.

### `shorts_generator/transcriber.py` (API/MuAPI mode)

Untouched. `verbose_json` segments have no per-word data; segments continue
to be built without a `"words"` key.

### `shorts_generator/captions.py`

- `_chunk_segments(segments, clip_start, clip_end, max_words=7)`:
  - If a segment has a non-empty `"words"` list, group those words directly
    into ≤`max_words`-word chunks. A chunk's `start`/`end` become its first
    and last word's actual times (still clipped/shifted to the clip window
    exactly as today). Each chunk also carries a clip-relative `"words"`
    list (one entry per word: `{start, end, text}`).
  - If a segment has no `"words"` (API mode, or a local segment that
    somehow lacks them), keep the existing proportional-by-word-count
    chunk split unchanged, and additionally synthesize each chunk's
    `"words"` list via a new helper, `_estimate_word_windows(text, start,
    end)`, which apportions `[start, end]` across the chunk's words
    proportional to character length (mirrors the existing chunk-level
    proportional-by-word-count logic, one level deeper).
  - Either path: every returned chunk has `start`, `end`, `text`, `words`.
- `_write_ass(chunks, ass_path, width, height, fade_seconds, word_highlight=True)`:
  - `word_highlight=True` (default): for each chunk, emit one `Dialogue`
    line per word in `chunk["words"]`, `Start`/`End` = that word's own
    (clip-relative) window. Line text = the full chunk text with the
    active word wrapped in
    `{\c&H00FFFF&\b1\t(0,80,\fscx125\fscy125)\t(80,160,\fscx100\fscy100)}...word...{\r}`
    (yellow, bold, pop to 125% over 80ms then settle to 100% by 160ms —
    `\t` timings are relative to that Dialogue line's own `Start`, so the
    bounce always lands exactly when the word goes active). Only the
    chunk's *first* word line additionally gets the existing
    `{\fad(fade_ms,0)}` prefix, so the phrase still fades in once instead
    of flashing on every word swap. Literal `{`/`}` stripped from text as
    today.
  - `word_highlight=False`: unchanged original behavior — one `Dialogue`
    line per chunk, `{\fad(fade_ms,0)}` prefix, no per-word override tags.
- `burn_captions(..., word_highlight: bool = True)`: threads the flag into
  `_write_ass`. No other change to its orchestration, error handling, or
  `CaptionError` contract.

### Wiring (`clipper.py`, `local/clipper.py`, `pipeline.py`, `main.py`)

- `crop_highlights()` / `crop_highlights_local()` gain a `word_highlight:
  bool = True` parameter, passed straight through to `burn_captions`.
- `pipeline._run_api` / `_run_local` pass `args.word_highlight` through.
- `main.py`: new flag
  `--no-word-highlight` (`action="store_false"`, `dest="word_highlight"`,
  default `True`), same pattern as `--no-captions`.

## Data flow

```
transcribe_local()              →  segments now include "words" (real, local mode)
transcribe()                    →  segments unchanged, no "words" (api mode)
                                 ↓
captions._chunk_segments()      →  every chunk gets "words" (real or char-length estimate)
                                 ↓
captions._write_ass()           →  one Dialogue line per word, active word highlighted+bounced
                                 ↓
ffmpeg subtitles filter         →  burned-in mp4
```

## Error handling

No new failure modes. Missing/malformed `"words"` on a segment just takes
the estimate path instead of the real-timestamp path — never raises.
`CaptionError` contract (raised only on ffprobe/ffmpeg failure or zero
overlapping transcript) is unchanged.

## Testing

- `_chunk_segments`: existing tests updated for the new `"words"` key on
  every returned chunk; new case with a segment carrying real `"words"`
  (chunk boundaries = first/last real word time, not proportional); new
  case confirming the char-length estimate path when `"words"` is absent
  (windows sum back to chunk duration, longer words get proportionally
  more time).
- `_write_ass`: `Dialogue:` count now equals total word count (not chunk
  count); highlight override tags (`\c&H00FFFF&`, `\b1`, `\t(`) present on
  each word line; `\fad(...)` present only once per chunk (its first word);
  `word_highlight=False` reproduces the old one-line-per-chunk output
  exactly.
- `local/transcriber.py`: mock a `faster_whisper` segment with `.words`,
  assert `transcribe_local()` returns segments with a `"words"` list; cache
  round-trip test (write `.json` cache, reload, words preserved,
  `.srt`-suffixed leftovers from before this change are ignored).
- Integration `burn_captions` test: unchanged — still just asserts the
  output file exists with matching resolution.
- Manual: one real end-to-end `--mode local` run on a short sample video,
  eyeball the per-word highlight/bounce timing against the spoken audio.
