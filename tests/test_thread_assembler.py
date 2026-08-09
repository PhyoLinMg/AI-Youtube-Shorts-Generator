import re
import subprocess

import pytest

from shorts_generator.local import thread_assembler


@pytest.fixture(scope="module")
def synthetic_clip_a(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("assembler_a")
    path = str(tmp_dir / "a.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=24:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", "-shortest", path,
        ],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def synthetic_clip_b(tmp_path_factory):
    # Deliberately different resolution/fps/audio rate from clip_a, matching
    # the real cross-episode case this module exists to normalize.
    tmp_dir = tmp_path_factory.mktemp("assembler_b")
    path = str(tmp_dir / "b.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
            "-ar", "48000",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", "-shortest", path,
        ],
        check=True,
    )
    return path


def test_assemble_thread_requires_at_least_two_segments(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        thread_assembler.assemble_thread(["/tmp/only_one.mp4"], str(tmp_path / "out.mp4"))


def test_assemble_thread_normalizes_mismatched_sources_into_one_output(synthetic_clip_a, synthetic_clip_b, tmp_path):
    out_path = str(tmp_path / "assembled.mp4")

    thread_assembler.assemble_thread([synthetic_clip_a, synthetic_clip_b], out_path)

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=width,height,r_frame_rate",
            "-of", "default=noprint_wrappers=1",
            out_path,
        ],
        capture_output=True, text=True, check=True,
    )
    output = probe.stdout
    assert f"width={thread_assembler.TARGET_WIDTH}" in output
    assert f"height={thread_assembler.TARGET_HEIGHT}" in output
    assert "r_frame_rate=30000/1001" in output

    # Guards against a filter-graph regression (e.g. a mismatched concat `n`)
    # that silently drops a segment while ffmpeg still exits 0 and the
    # surviving segment(s) still probe at the correct dims/fps above.
    duration_match = re.search(r"duration=([\d.]+)", output)
    assert duration_match is not None
    duration = float(duration_match.group(1))
    assert duration == pytest.approx(5.0, abs=0.5)  # clip_a (2s) + clip_b (3s)


def test_assemble_thread_raises_thread_assembly_error_on_ffmpeg_failure(tmp_path):
    with pytest.raises(thread_assembler.ThreadAssemblyError):
        thread_assembler.assemble_thread(
            ["/nonexistent/a.mp4", "/nonexistent/b.mp4"], str(tmp_path / "out.mp4")
        )
