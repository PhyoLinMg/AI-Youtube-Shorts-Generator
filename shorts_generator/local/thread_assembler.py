"""Re-encode ffmpeg concat for a thread's ordered segments (narration cards
+ live-footage clips). A plain stream-copy concat (as jump_cuts.py uses for
same-source excision) assumes matching codec/resolution/fps/audio across
every input; a thread's clips come from different source episodes with
different native fps/audio, so every segment is normalized to one target
spec before concatenation, matching the spec crop_clip_local's vertical
output and narration cards already use.
"""
import subprocess
from typing import List

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_FPS = "30000/1001"
TARGET_AUDIO_RATE = 44100


class ThreadAssemblyError(RuntimeError):
    """Raised when the final ffmpeg concat fails."""


def assemble_thread(segment_paths: List[str], out_path: str) -> str:
    """Concatenate segment_paths in order into one vertical video at
    out_path, re-encoding every stream to a common spec first."""
    if len(segment_paths) < 2:
        raise ValueError("assemble_thread needs at least 2 segments")

    inputs = []
    for p in segment_paths:
        inputs += ["-i", p]

    filter_parts = []
    concat_refs = []
    for i in range(len(segment_paths)):
        filter_parts.append(
            f"[{i}:v]fps={TARGET_FPS},scale={TARGET_WIDTH}:{TARGET_HEIGHT},setsar=1,format=yuv420p[v{i}]"
        )
        filter_parts.append(f"[{i}:a]aresample={TARGET_AUDIO_RATE},aformat=channel_layouts=stereo[a{i}]")
        concat_refs += [f"[v{i}]", f"[a{i}]"]

    filter_complex = (
        ";".join(filter_parts)
        + ";" + "".join(concat_refs)
        + f"concat=n={len(segment_paths)}:v=1:a=1[outv][outa]"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise ThreadAssemblyError(f"ffmpeg concat failed: {e.stderr}") from e
    return out_path
