import os
import subprocess

import cv2
import pytest

from shorts_generator.hook_card import HookCardError, extract_frame


def _run(cmd):
    subprocess.run(cmd, check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def red_clip(tmp_path_factory):
    """A tiny solid-red 3s clip w/ audio — stands in for a final vertical crop."""
    tmp_dir = tmp_path_factory.mktemp("hookcard_src")
    path = str(tmp_dir / "clip.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=red:size=270x480:rate=24:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", "-shortest",
        path,
    ])
    return path


def test_extract_frame_writes_readable_image(red_clip, tmp_path):
    out_path = str(tmp_path / "frame.jpg")
    result = extract_frame(red_clip, 1.0, out_path)

    assert result == out_path
    assert os.path.exists(out_path)
    img = cv2.imread(out_path)
    assert img is not None
    assert img.shape[:2] == (480, 270)


def test_extract_frame_raises_hook_card_error_on_bad_video(tmp_path):
    out_path = str(tmp_path / "frame.jpg")
    with pytest.raises(HookCardError):
        extract_frame(str(tmp_path / "missing.mp4"), 1.0, out_path)


@pytest.fixture(scope="module")
def motion_clip(tmp_path_factory):
    """0-2s static gray (resting), 2-3s a heavily blurred bright flash
    (high motion, low sharpness), 3-4s a sharp high-contrast test pattern
    (high motion, high sharpness), 4-6s back to static gray."""
    tmp_dir = tmp_path_factory.mktemp("hookcard_motion")

    seg0 = str(tmp_dir / "seg0.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=gray:size=320x568:rate=24:duration=2",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", seg0,
    ])

    seg1 = str(tmp_dir / "seg1.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=white:size=320x568:rate=24:duration=1",
        "-vf", "gblur=sigma=20",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", seg1,
    ])

    seg2 = str(tmp_dir / "seg2.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=320x568:rate=24:duration=1",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", seg2,
    ])

    seg3 = str(tmp_dir / "seg3.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=gray:size=320x568:rate=24:duration=2",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", seg3,
    ])

    list_path = str(tmp_dir / "list.txt")
    with open(list_path, "w") as f:
        for seg in (seg0, seg1, seg2, seg3):
            f.write(f"file '{seg}'\n")

    out_path = str(tmp_dir / "motion.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_path,
    ])
    return out_path


def test_pick_striking_frame_prefers_sharp_over_blurry_motion(motion_clip):
    from shorts_generator.hook_card import pick_striking_frame

    ts = pick_striking_frame(motion_clip)

    # 2-3s is high-motion but blurred (sharpness ~0); 3-4s is high-motion
    # AND sharp. The sharpness tiebreaker must pick from the sharp window.
    assert 3.0 <= ts < 4.0


def test_pick_striking_frame_falls_back_when_too_short_to_sample(tmp_path):
    from shorts_generator.hook_card import pick_striking_frame

    short_path = str(tmp_path / "short.mp4")
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=blue:size=320x568:rate=24:duration=0.3",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", short_path,
    ])

    assert pick_striking_frame(short_path, skip_seconds=0.5) == 0.5


def test_pick_striking_frame_raises_hook_card_error_on_bad_video(tmp_path):
    from shorts_generator.hook_card import pick_striking_frame

    with pytest.raises(HookCardError):
        pick_striking_frame(str(tmp_path / "missing.mp4"))
