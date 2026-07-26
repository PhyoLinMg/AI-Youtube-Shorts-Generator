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
