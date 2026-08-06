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

# ASS override tags for the karaoke-style active-word highlight: pop to
# yellow + bold. No size scaling — a scaled word changes the line's
# rendered width and re-centers the whole line every word, which reads as
# the caption jumping/switching too fast even though only the highlight
# moves.
# `{\r}` resets back to the line's base `Caption` style for the remainder
# of the text.
_HIGHLIGHT_OPEN = "{\\c&H00FFFF&\\b1}"
_HIGHLIGHT_CLOSE = "{\\r}"

FONT_DIR = os.path.join(os.path.dirname(__file__), "assets", "fonts")


def _escape_ffmpeg_path(path: str) -> str:
    """Escape a filesystem path for use as an ffmpeg filter argument value
    (backslashes to forward slashes, colons escaped so they aren't parsed
    as a filter option separator)."""
    return path.replace("\\", "/").replace(":", "\\:")


class CaptionError(RuntimeError):
    """Raised when caption burn-in fails; callers should fall back to the plain clip."""


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


def _chunk_segments(
    segments: List[Dict],
    clip_start: float,
    clip_end: float,
    max_words: int = 7,
) -> List[Dict]:
    """Slice full-video transcript segments to a clip window and split each
    into ~max_words-word chunks, timed proportionally to word count within
    the segment's own duration, then clipped to the clip window.

    Returns clip-relative chunks with per-word timings:
    [{"start": float, "end": float, "text": str, "words": [...]}, ...]

    The estimate path (no real per-word timestamps on the segment) clips
    each word's own estimated window to [clip_start, clip_end] and rebuilds
    text from only the surviving words -- this applies to every caller, not
    just jump-cut clips, so a highlight's leading/trailing caption line can
    read slightly differently than before cut_segments existed (words whose
    estimated window falls entirely outside the window are now dropped
    rather than shown with a compressed duration). API mode hits this path
    for essentially all captions (MuAPI transcription never returns
    per-word timestamps); local mode mostly avoids it via faster-whisper's
    real word timestamps, except for any segment where that returns an
    empty word list.
    """
    chunks: List[Dict] = []
    for seg in segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        # Real-timestamp path: use actual per-word timestamps
        seg_words = seg.get("words")
        if seg_words and isinstance(seg_words, list) and len(seg_words) > 0:
            chunks.extend(_chunk_from_real_words(seg_words, clip_start, clip_end, max_words))
            continue

        # Estimate path: proportional split with character-weighted word timing
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

            # Per-word estimated windows (character-weighted), in the
            # segment's original absolute time — computed before clipping so
            # we can drop/trim individual words rather than the whole group.
            word_windows = _estimate_word_windows(group, start, end)

            kept = []
            for w in word_windows:
                ws = max(w["start"], clip_start)
                we = min(w["end"], clip_end)
                if we <= ws:
                    continue
                kept.append({
                    "start": ws - clip_start,
                    "end": we - clip_start,
                    "text": w["text"],
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


def _chunk_cut_segments(
    transcript_segments: List[Dict], cut_segments: List[Dict], max_words: int = 7,
) -> List[Dict]:
    """Chunk each kept `cut_segments` span independently against
    `transcript_segments` (so no chunk can ever straddle an excised gap by
    construction), then offset each span's chunks onto the concatenated
    output timeline by the cumulative duration of the *previously kept*
    spans (not the excised gaps).

    This offset uses each span's *nominal* requested duration
    (`end_time - start_time`), while jump_cuts.excise_cut_segments's actual
    re-encoded parts run up to roughly one video frame longer than nominal
    each (encoder/keyframe rounding) — never shorter. So captions drift
    slightly *early* relative to speech as a clip progresses, worst case
    on the order of one frame per kept span (up to 6 spans). Accepted as a
    minor, unmeasured-in-practice skew rather than threading measured
    per-part durations back from jump_cuts through this offset math.
    """
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


def _clean_caption_text(text: str) -> str:
    return text.replace("{", "").replace("}", "").replace("\n", " ")


def _render_word_line(word_texts: List[str], active_index: int) -> str:
    """Render a chunk's full text with the word at `active_index` wrapped in
    the highlight override tags."""
    parts = []
    for i, w in enumerate(word_texts):
        if i == active_index:
            parts.append(f"{_HIGHLIGHT_OPEN}{w}{_HIGHLIGHT_CLOSE}")
        else:
            parts.append(w)
    return " ".join(parts)


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
    margin_h = max(10, round(width * 0.06))
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
        f"Style: Caption,Montserrat Black,{fontsize},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
        f"0,0,0,0,100,100,0,0,1,2,1,2,{margin_h},{margin_h},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    for chunk in chunks:
        words = chunk.get("words")
        if word_highlight and words:
            word_texts = [_clean_caption_text(w["text"]) for w in words]
            for i, word in enumerate(words):
                # Extend each word's line to the next word's start (chunk's own
                # end for the last word) rather than the word's own end — real
                # whisper timestamps can have small gaps between words, which
                # would otherwise blink the whole phrase off between them.
                line_end = words[i + 1]["start"] if i + 1 < len(words) else chunk["end"]
                start_ts = _format_ass_timestamp(word["start"])
                end_ts = _format_ass_timestamp(line_end)
                line_text = _render_word_line(word_texts, i)
                fad_prefix = f"{{\\fad({fade_ms},0)}}" if i == 0 else ""
                lines.append(
                    f"Dialogue: 0,{start_ts},{end_ts},Caption,,0,0,0,,{fad_prefix}{line_text}\n"
                )
        else:
            text = _clean_caption_text(chunk["text"])
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


def _probe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise CaptionError(f"ffprobe failed on {video_path}: {e}") from e

    try:
        return float(result.stdout.strip())
    except ValueError as e:
        raise CaptionError(f"could not parse ffprobe duration for {video_path}: {result.stdout!r}") from e


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
