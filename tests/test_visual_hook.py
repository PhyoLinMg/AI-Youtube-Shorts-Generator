import os
import subprocess

import pytest

from shorts_generator.visual_hook import (
    HOOK_FRAME_OFFSETS,
    _extract_hook_frames,
    _parse_visual_hook_response,
    score_visual_hooks,
)


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("visual_hook_src")
    path = str(tmp_dir / "source.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=24:duration=6",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            path,
        ],
        check=True,
    )
    return path


def test_extract_hook_frames_returns_one_file_per_offset(tmp_path, synthetic_video):
    frames = _extract_hook_frames(synthetic_video, start_time=1.0, out_dir=str(tmp_path))
    assert len(frames) == len(HOOK_FRAME_OFFSETS)
    for f in frames:
        assert os.path.exists(f)
        assert os.path.getsize(f) > 0


def test_extract_hook_frames_skips_offsets_past_video_end(tmp_path, synthetic_video):
    # synthetic_video is 6s; starting at 5.9s, the +0.6s offset frame (6.5s)
    # doesn't exist -- ffmpeg fails for that one offset, must not raise.
    frames = _extract_hook_frames(synthetic_video, start_time=5.9, out_dir=str(tmp_path))
    assert len(frames) < len(HOOK_FRAME_OFFSETS)


def test_parse_visual_hook_response_plain_json():
    parsed = _parse_visual_hook_response('{"visual_hook_score": 72, "visual_hook_reason": "unusual framing"}')
    assert parsed == {"visual_hook_score": 72, "visual_hook_reason": "unusual framing"}


def test_parse_visual_hook_response_strips_markdown_fence():
    parsed = _parse_visual_hook_response('```json\n{"visual_hook_score": 40, "visual_hook_reason": "flat shot"}\n```')
    assert parsed == {"visual_hook_score": 40, "visual_hook_reason": "flat shot"}


def test_parse_visual_hook_response_clamps_score_above_range():
    parsed = _parse_visual_hook_response('{"visual_hook_score": 150, "visual_hook_reason": "x"}')
    assert parsed["visual_hook_score"] == 100


def test_parse_visual_hook_response_clamps_score_below_range():
    parsed = _parse_visual_hook_response('{"visual_hook_score": -10, "visual_hook_reason": "x"}')
    assert parsed["visual_hook_score"] == 0


def test_score_visual_hooks_attaches_score_and_reason(synthetic_video):
    highlights = [{"title": "Clip A", "start_time": 1.0, "end_time": 3.0}]

    def stub_llm(prompt, image_paths):
        assert len(image_paths) > 0
        return '{"visual_hook_score": 88, "visual_hook_reason": "surprising opener"}'

    result = score_visual_hooks(synthetic_video, highlights, llm_fn=stub_llm)

    assert result[0]["visual_hook_score"] == 88
    assert result[0]["visual_hook_reason"] == "surprising opener"
    assert result[0]["title"] == "Clip A"  # original fields preserved


def test_score_visual_hooks_degrades_gracefully_on_llm_failure(synthetic_video):
    highlights = [{"title": "Clip A", "start_time": 1.0, "end_time": 3.0}]

    def failing_llm(prompt, image_paths):
        raise RuntimeError("vision backend unavailable")

    result = score_visual_hooks(synthetic_video, highlights, llm_fn=failing_llm)

    assert "visual_hook_score" not in result[0]
    assert result[0]["title"] == "Clip A"  # highlight itself survives, unmodified


def test_score_visual_hooks_one_failure_does_not_block_others(synthetic_video):
    highlights = [
        {"title": "Clip A", "start_time": 1.0, "end_time": 3.0},
        {"title": "Clip B", "start_time": 2.0, "end_time": 4.0},
    ]
    calls = {"count": 0}

    def flaky_llm(prompt, image_paths):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient failure")
        return '{"visual_hook_score": 50, "visual_hook_reason": "ok"}'

    result = score_visual_hooks(synthetic_video, highlights, llm_fn=flaky_llm)

    assert "visual_hook_score" not in result[0]
    assert result[1]["visual_hook_score"] == 50
