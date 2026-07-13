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
