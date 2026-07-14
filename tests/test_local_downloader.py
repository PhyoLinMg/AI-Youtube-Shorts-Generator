import os
import subprocess

import pytest

from shorts_generator.local import downloader


def test_local_file_input_is_copied_to_target(tmp_path):
    src = tmp_path / "input.mp4"
    src.write_bytes(b"fake-mp4-bytes")
    target = str(tmp_path / "run" / "full_source.mp4")

    result = downloader.download_youtube_local(str(src), target_path=target)

    assert result == target
    assert os.path.exists(target)
    assert open(target, "rb").read() == b"fake-mp4-bytes"


def test_local_file_already_at_target_is_left_alone(tmp_path):
    target = str(tmp_path / "full_source.mp4")
    with open(target, "wb") as f:
        f.write(b"already-here")

    result = downloader.download_youtube_local(target, target_path=target)

    assert result == target
    assert open(target, "rb").read() == b"already-here"


@pytest.fixture(scope="module")
def synthetic_mkv(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("mkv_source")
    path = str(tmp_dir / "input.mkv")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            path,
        ],
        check=True,
    )
    return path


def test_non_mp4_local_input_is_remuxed_to_mp4(tmp_path, synthetic_mkv):
    target = str(tmp_path / "run" / "full_source.mp4")

    result = downloader.download_youtube_local(synthetic_mkv, target_path=target)

    assert result == target
    assert os.path.exists(target)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "csv=p=0", target],
        capture_output=True, text=True, check=True,
    )
    assert "mp4" in probe.stdout
