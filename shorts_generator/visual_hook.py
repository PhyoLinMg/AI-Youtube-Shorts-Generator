"""Visual-hook scoring: does the opening frame(s) of a candidate highlight
stop a scroll on their own, independent of text or audio -- the one gap in
the already-shipped verbal/textual hook system (see hook_strength/hook_card
in highlights.py/hook_card.py and
docs/superpowers/specs/2026-07-26-three-jails-escape-design.md).

Runs as a post-selection pass over the already-chosen `top` candidates in
pipeline.py, using the local source video both modes already have on disk
at that point. Results are informational only -- never written back into
the highlights cache -- so a failed or unavailable vision backend degrades
one highlight to "no score" rather than blocking the pipeline, exactly like
highlights.detect_content_type already degrades on LLM failure.
"""
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List

from .config import VISUAL_HOOK_PARALLELISM

VisionLLMFn = Callable[[str, List[str]], str]

HOOK_FRAME_OFFSETS = (0.0, 0.6)

VISUAL_HOOK_PROMPT = """You are scoring the VISUAL hook of a short-form video clip -- does the very first frame or two, with NO audio and NO on-screen text, grab attention and make someone stop scrolling?

Score 0-100:
- High (80+): a striking, unusual, or immediately intriguing image on its own -- unexpected action, a visually surprising scene, strong composition.
- Low (<30): a static talking-head frame, a blank/neutral background, nothing visually distinct from any other video.

Respond ONLY with valid JSON: {"visual_hook_score": int, "visual_hook_reason": "one sentence"}"""


def _extract_hook_frames(video_path: str, start_time: float, out_dir: str) -> List[str]:
    """ffmpeg-extract one JPEG per HOOK_FRAME_OFFSETS timestamp, relative to
    start_time. Returns the frames that were successfully extracted --
    silently skips any offset ffmpeg can't produce (e.g. past the end of
    the source video), never raises."""
    paths = []
    for i, offset in enumerate(HOOK_FRAME_OFFSETS):
        frame_path = f"{out_dir}/hook_frame_{i}.jpg"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start_time + offset:.3f}",
            "-i", video_path,
            "-frames:v", "1", "-q:v", "2",
            frame_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            paths.append(frame_path)
    return paths


def _parse_visual_hook_response(raw: str) -> Dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    score = max(0, min(100, int(float(data.get("visual_hook_score", 0)))))
    reason = str(data.get("visual_hook_reason") or "").strip()
    return {"visual_hook_score": score, "visual_hook_reason": reason}


def call_muapi_vision_llm(prompt: str, image_paths: List[str]) -> str:
    """MuAPI vision backend for score_visual_hooks, used by --mode api.

    MuAPI has no usable vision-capable endpoint as of this writing (see the
    Task 1 spike in docs/superpowers/plans/2026-07-26-jail1-visual-hook.md).
    This raises unconditionally so score_visual_hooks's per-highlight
    try/except degrades api mode to "no visual hook score" rather than
    blocking the pipeline. Replace this body with a real muapi.run(...)
    call once MuAPI ships a vision-capable endpoint.
    """
    raise RuntimeError("no MuAPI vision endpoint available")


def _score_one_highlight(i: int, h: Dict, source_video_path: str, tmp_dir: str, llm_fn: VisionLLMFn) -> Dict:
    """Never raises: any per-highlight failure (frame extraction, vision
    call, bad JSON) is logged and that highlight is returned unmodified --
    one bad candidate must never abort the rest of the pipeline."""
    # Frames land in a per-highlight subdir so concurrent workers never
    # write the same hook_frame_*.jpg path at once.
    highlight_dir = f"{tmp_dir}/{i}"
    os.makedirs(highlight_dir, exist_ok=True)
    try:
        entry = dict(h)
        frame_paths = _extract_hook_frames(source_video_path, float(h["start_time"]), highlight_dir)
        if not frame_paths:
            raise RuntimeError("no frames extracted")
        raw = llm_fn(VISUAL_HOOK_PROMPT, frame_paths)
        entry.update(_parse_visual_hook_response(raw))
        return entry
    except Exception as e:
        print(f"[visual_hook] {i} skipped: {e}", flush=True)
        return h


def score_visual_hooks(
    source_video_path: str, highlights: List[Dict], llm_fn: VisionLLMFn,
) -> List[Dict]:
    """Attach visual_hook_score/visual_hook_reason to each highlight, in
    parallel (up to VISUAL_HOOK_PARALLELISM at a time) -- each highlight's
    frame extraction + vision call is independent of the others. Result
    order matches input order regardless of completion order."""
    import tempfile

    if not highlights:
        return []

    with tempfile.TemporaryDirectory() as tmp_dir:
        n = len(highlights)
        with ThreadPoolExecutor(max_workers=min(VISUAL_HOOK_PARALLELISM, n)) as pool:
            return list(pool.map(
                lambda args: _score_one_highlight(args[0], args[1], source_video_path, tmp_dir, llm_fn),
                zip(range(1, n + 1), highlights),
            ))
