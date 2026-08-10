"""Add an episode to the local corpus (see corpus.py, thread_builder.py)
from YouTube's own auto-captions, with no video/audio download at all --
just enough for the same-topic gate and, if the episode is later picked
for a thread, thread_source.acquire_clip's re-download-just-the-span path
(full_source.mp4 deliberately never gets written here).

Uses srv1 (not vtt/json3): YouTube's auto-caption vtt/json3 are a rolling
karaoke stream where each cue re-sends the prior line's words as context,
so a naive parse duplicates most of the transcript. srv1 gives one clean
<text start dur>...</text> cue per caption card.
"""
import html
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from ..run_output import RunPaths, resolve_output_dir, write_source_url

DEFAULT_SEGMENT_SECONDS = 12.0


def _fetch_duration(youtube_url: str) -> float:
    # Deliberately the same source generate_threads()'s acquire_clip will
    # later probe (thread_source._probe_source_duration) and compare against
    # this value with a 2.0s tolerance -- captions can end before the video
    # itself does (outros, music), so the cues' own timestamps are not a
    # safe stand-in for this.
    result = subprocess.run(
        ["yt-dlp", "--skip-download", "--print", "duration", youtube_url],
        capture_output=True, text=True, check=True,
    )
    stdout = result.stdout.strip()
    try:
        return float(stdout.splitlines()[-1])
    except (IndexError, ValueError) as e:
        raise RuntimeError(
            f"could not parse a duration from yt-dlp output for {youtube_url!r}: stdout={stdout!r}"
        ) from e


def _fetch_srv1_captions(youtube_url: str, lang: str = "en") -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_template = os.path.join(tmp_dir, "captions.%(ext)s")
        subprocess.run(
            [
                "yt-dlp", "--skip-download", "--write-auto-sub",
                "--sub-lang", lang, "--sub-format", "srv1",
                "-o", out_template, youtube_url,
            ],
            check=True, capture_output=True, text=True,
        )
        caption_path = os.path.join(tmp_dir, f"captions.{lang}.srv1")
        if not os.path.exists(caption_path):
            raise RuntimeError(f"no {lang!r} auto-captions available for {youtube_url!r}")
        with open(caption_path, "r", encoding="utf-8") as f:
            return f.read()


def _parse_srv1(xml_text: str, target_segment_seconds: float = DEFAULT_SEGMENT_SECONDS) -> List[Dict]:
    """One <text start dur>cue</text> per caption card -> merged into
    ~target_segment_seconds segments (cue-level cards run 2-4s each; that's
    too fine-grained for thread_builder's stage B, which feeds the FULL
    transcript of two whole episodes into one LLM prompt -- thousands of
    tiny segments there bloats that prompt for no benefit, since srv1 has no
    word-level timing to lose by merging).

    html.unescape runs once, after ElementTree's own entity resolution: the
    raw file double-escapes (e.g. "&amp;#39;"), so ET leaves "&#39;" in
    .text and only the explicit unescape() call resolves that down to "'".
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise RuntimeError(f"could not parse srv1 captions: {e}") from e

    cues = []
    for elem in root.findall("text"):
        start = float(elem.get("start", "0"))
        dur = float(elem.get("dur", "0"))
        text = html.unescape(elem.text or "").strip()
        if not text:
            continue
        cues.append((start, start + dur, text))

    if not cues:
        return []

    segments = []
    seg_start, seg_end, seg_texts = cues[0][0], cues[0][1], [cues[0][2]]
    for start, end, text in cues[1:]:
        if end - seg_start <= target_segment_seconds:
            seg_end = end
            seg_texts.append(text)
        else:
            segments.append({"start": seg_start, "end": seg_end, "text": " ".join(seg_texts)})
            seg_start, seg_end, seg_texts = start, end, [text]
    segments.append({"start": seg_start, "end": seg_end, "text": " ".join(seg_texts)})
    return segments


def ingest_captions(youtube_url: str, base_dir: Optional[str] = None) -> Dict:
    """Add one episode to the corpus from its auto-captions only. Writes
    full_source.json + source_url.txt (see corpus.list_corpus_run_dirs --
    both are required for corpus eligibility) but never full_source.mp4.

    Idempotent: if this URL's run dir already has a full_source.json --
    whether from a prior full pipeline run (real Whisper transcript,
    possibly with full_source.mp4 still on disk) or a prior caption-only
    ingest -- the fetch is skipped entirely and the existing transcript is
    reused as-is. Without this guard, re-ingesting an already-fully-
    processed episode would silently downgrade its real transcript to
    lower-fidelity YouTube auto-captions."""
    paths: RunPaths = resolve_output_dir(youtube_url, base_dir=base_dir)

    if os.path.exists(paths.source_json):
        print(f"[caption_ingest] {youtube_url!r} already in the corpus at {paths.root} -- skipping caption fetch", flush=True)
        with open(paths.source_json, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {
            "run_dir": paths.root,
            "title": os.path.basename(paths.root),
            "duration": existing.get("duration", 0.0),
            "segment_count": len(existing.get("segments", [])),
        }

    duration = _fetch_duration(youtube_url)
    xml_text = _fetch_srv1_captions(youtube_url)
    segments = _parse_srv1(xml_text)
    if not segments:
        raise RuntimeError(f"no caption text parsed for {youtube_url!r}")

    with open(paths.source_json, "w", encoding="utf-8") as f:
        json.dump({"duration": duration, "segments": segments}, f, ensure_ascii=False, indent=2)
    write_source_url(paths, youtube_url)

    return {
        "run_dir": paths.root,
        "title": os.path.basename(paths.root),
        "duration": duration,
        "segment_count": len(segments),
    }
