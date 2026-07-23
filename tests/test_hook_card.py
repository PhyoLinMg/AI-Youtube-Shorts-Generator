import os
import subprocess

import cv2
import pytest

from shorts_generator.hook_card import HookCardError, render_card_overlay


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


def _corner_pixel_bgr(video_path, timestamp, tmp_path, name):
    """Sample a corner (outside the centered hook-text box) so we can tell
    whether the underlying footage is still changing frame to frame."""
    frame_path = str(tmp_path / name)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{timestamp}",
          "-i", video_path, "-vframes", "1", frame_path])
    img = cv2.imread(frame_path)
    region = img[0:10, 0:10]
    return region.reshape(-1, 3).mean(axis=0)  # BGR


def _center_pixel_bgr(video_path, timestamp, tmp_path, name):
    frame_path = str(tmp_path / name)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{timestamp}",
          "-i", video_path, "-vframes", "1", frame_path])
    img = cv2.imread(frame_path)
    h, w = img.shape[:2]
    region = img[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5]
    return region.reshape(-1, 3).mean(axis=0)  # BGR


def test_render_card_overlay_does_not_freeze_moving_footage(motion_clip, tmp_path):
    """A direct freeze/no-freeze check: sample the source clip's corner
    pixel at two timestamps inside the hook window where the source itself
    is changing (the gray->flash transition just after 2s), with a hook
    window long enough to span it. If the card froze the opening frame,
    both output samples would match the t=0 frame instead of tracking the
    source."""
    out_path = str(tmp_path / "out.mp4")
    duration = 2.3
    render_card_overlay(motion_clip, "TEST HOOK", out_path, duration=duration)

    source_at_0 = _corner_pixel_bgr(motion_clip, 0.1, tmp_path, "src0.jpg")
    source_at_2_2 = _corner_pixel_bgr(motion_clip, 2.2, tmp_path, "src22.jpg")
    out_at_0 = _corner_pixel_bgr(out_path, 0.1, tmp_path, "out0.jpg")
    out_at_2_2 = _corner_pixel_bgr(out_path, 2.2, tmp_path, "out22.jpg")

    # The source itself changes a lot between these two timestamps (gray -> blurred white).
    assert abs(float(source_at_2_2[0]) - float(source_at_0[0])) > 30
    # The output tracks the source at each timestamp instead of staying pinned to the t=0 frame.
    assert out_at_0 == pytest.approx(source_at_0, abs=10)
    assert out_at_2_2 == pytest.approx(source_at_2_2, abs=10)


def test_render_card_overlay_wraps_long_hook_text_onto_two_lines(red_clip, tmp_path):
    out_path = str(tmp_path / "out.mp4")
    # Should not raise even with a 7-word hook (two-line wrap path).
    render_card_overlay(red_clip, "You Won't Believe: This Happened Today", out_path, duration=1.0)
    assert os.path.exists(out_path)


def test_render_card_overlay_escapes_special_characters_in_hook_text(red_clip, tmp_path):
    """LLM-generated hook text flows straight into an ffmpeg drawtext filter
    string -- colon, apostrophe, comma, semicolon, and backslash are all
    filtergraph-significant and must not break the command."""
    out_path = str(tmp_path / "out.mp4")
    render_card_overlay(red_clip, "Wait, this: really; happened\\shocking", out_path, duration=1.0)
    assert os.path.exists(out_path)


def test_render_card_overlay_raises_hook_card_error_on_missing_video(tmp_path):
    out_path = str(tmp_path / "out.mp4")
    with pytest.raises(HookCardError):
        render_card_overlay(str(tmp_path / "missing.mp4"), "HOOK", out_path)


def test_render_card_overlay_preserves_duration(red_clip, tmp_path):
    out_path = str(tmp_path / "out.mp4")
    render_card_overlay(red_clip, "TEST HOOK", out_path, duration=1.0)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out_path],
        capture_output=True, text=True, check=True,
    )
    assert float(probe.stdout.strip()) == pytest.approx(3.0, abs=0.2)

    # Text shows up during the window...
    during_card = _center_pixel_bgr(out_path, 0.3, tmp_path, "during.jpg")
    # red_clip is pure red (BGR ~ [0, 0, 255]); the boxed hook text pulls
    # the center pixel well away from pure red.
    assert during_card[2] < 215 or during_card[0] > 40 or during_card[1] > 40
    # ...and is gone after the window, with the live footage back underneath.
    after_card = _center_pixel_bgr(out_path, 1.5, tmp_path, "after.jpg")
    assert after_card[2] > 200 and after_card[0] < 40 and after_card[1] < 40
