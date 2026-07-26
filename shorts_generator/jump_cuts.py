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


class JumpCutError(RuntimeError):
    """Raised when trim/concat excision fails; callers should fall back to
    the un-excised clip."""


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
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-loglevel", "error",
                        # -ss before -i is a fast keyframe-seek on the
                        # demuxer instead of a full decode-from-start; -t
                        # (duration, relative) is used instead of -to
                        # because -to before -i is an ABSOLUTE input
                        # timestamp, not relative to -ss, when seeking
                        # occurs before -i.
                        "-ss", f"{rel_start:.3f}", "-i", source_path,
                        "-t", f"{rel_end - rel_start:.3f}",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "128k",
                        part_path,
                    ],
                    check=True, capture_output=True, text=True,
                )
            except subprocess.CalledProcessError as e:
                raise JumpCutError(f"ffmpeg trim of segment {i} failed: {e.stderr}") from e
            except OSError as e:
                raise JumpCutError(f"ffmpeg trim of segment {i} failed: {e}") from e
            part_paths.append(part_path)

        concat_list_path = os.path.join(tmp_dir, "concat.txt")
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for p in part_paths:
                f.write(f"file '{p}'\n")

        # Write the concat output inside tmp_dir (not directly to out_path)
        # so that if this subprocess is killed or crashes partway (OOM,
        # disk-full, hard kill), the partial file lands somewhere the
        # existing `finally: shutil.rmtree(tmp_dir)` already cleans up,
        # instead of leaving a truncated/corrupt file sitting at out_path
        # where callers' crash-recovery scanners could pick it up.
        tmp_output_path = os.path.join(tmp_dir, "concat_output.mp4")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "concat", "-safe", "0", "-i", concat_list_path,
                    "-c", "copy",
                    tmp_output_path,
                ],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            raise JumpCutError(f"ffmpeg concat of excised segments failed: {e.stderr}") from e
        except OSError as e:
            raise JumpCutError(f"ffmpeg concat of excised segments failed: {e}") from e

        # Only now that the concat subprocess has fully succeeded do we
        # atomically publish the result at out_path. os.replace is atomic
        # on POSIX and Windows, so out_path either ends up with the full
        # correct file or is untouched — never a partial write.
        try:
            os.replace(tmp_output_path, out_path)
        except OSError as e:
            raise JumpCutError(f"publishing excised output to {out_path} failed: {e}") from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path
