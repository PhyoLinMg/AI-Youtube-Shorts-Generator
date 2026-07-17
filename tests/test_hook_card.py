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
