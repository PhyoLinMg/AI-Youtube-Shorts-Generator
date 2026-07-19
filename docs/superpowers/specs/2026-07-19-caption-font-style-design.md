# Caption font & style enhancement

## Problem

`shorts_generator/captions.py` burns word-highlighted ASS captions via ffmpeg's
`subtitles` filter. Two issues:

1. **Jank / "rearranging" text.** The active-word highlight tag
   (`_HIGHLIGHT_OPEN`) applies `\t(0,80,\fscx125\fscy125)\t(80,160,\fscx100\fscy100)`
   — a scale bounce on the active word. Each word gets its own `Dialogue`
   line (word-highlight mode), and the caption style is centered
   (`Alignment: 2`). Scaling one word's width changes the whole line's
   rendered width, which re-centers the line — so the caption visibly
   shifts left/right as focus moves word to word.
2. **Plain/unreliable font.** Base style is `Fontname: Arial`, resolved by
   libass via host fontconfig at render time — no bundled font file, so
   render depends on whatever the host machine has installed, and Arial
   reads thin/generic for short-form caption style.

## Fix

1. **Drop the scale bounce.** Remove `\fscx125\fscy125` / the `\t(...)`
   scale animation from `_HIGHLIGHT_OPEN`. Keep `\c&H00FFFF&\b1` (yellow +
   bold). Synthetic bold (`\b1`) thickens strokes in place without changing
   advance width, so it doesn't reflow the line. Highlight becomes a clean
   color pop with no positional shift.
2. **Bundle Montserrat Black.** Add `Montserrat-Black.ttf` (SIL OFL
   licensed, family name `Montserrat Black`) to
   `shorts_generator/assets/fonts/`. Set the ASS `Style` line's `Fontname`
   to `Montserrat Black`.
3. **Bundle for reliability.** Point ffmpeg's `subtitles` filter at the
   fonts dir via the `fontsdir=` option (same escaping treatment as the
   existing `ass_path`), so caption rendering no longer depends on the host
   having any particular font installed — same reliability class as
   `hook_card.py`'s explicit `FONT_PATH` use for `drawtext`.
4. **License hygiene.** `assets/fonts/OFL.txt` currently carries an
   Anton-specific copyright header. Rename it to `OFL-Anton.txt` and add
   `OFL-Montserrat.txt` (Montserrat Project Authors' OFL text) alongside
   the new ttf.

## Files touched

- `shorts_generator/captions.py`
  - `_HIGHLIGHT_OPEN` constant: drop scale tags.
  - New module-level `FONT_DIR` constant pointing at `assets/fonts`
    (mirrors `hook_card.py`'s `FONT_PATH` pattern).
  - `_write_ass`: `Style` line `Fontname` → `Montserrat Black`.
  - `burn_captions`: `-vf` string adds `:fontsdir=<escaped dir>` to the
    `subtitles` filter.
- `shorts_generator/assets/fonts/`
  - Add `Montserrat-Black.ttf`.
  - Rename `OFL.txt` → `OFL-Anton.txt`.
  - Add `OFL-Montserrat.txt`.

## Out of scope

- No change to chunking, timing, fade-in, outline/shadow, font size,
  margins, or base/highlight colors — those aren't part of the reported
  jank or font complaint.
- No change to `hook_card.py` (already uses a bundled font correctly).

## Testing

- `tests/test_captions.py` doesn't assert on font name or highlight-tag
  string content today, so no existing test breaks. Add one assertion that
  `_HIGHLIGHT_OPEN` contains no `\fscx`/`\fscy` (regression guard against
  the bounce coming back) and that `_write_ass` output contains
  `Montserrat Black`.
- Manual: burn captions on a short clip, visually confirm the line no
  longer shifts horizontally as words highlight, and that font renders as
  Montserrat Black (not a fallback) even without the font installed
  system-wide.
