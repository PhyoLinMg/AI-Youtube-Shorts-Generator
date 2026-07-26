import subprocess

import pytest

from shorts_generator.jump_cuts import excise_cut_segments


def _probe_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


@pytest.fixture(scope="module")
def synthetic_envelope(tmp_path_factory):
    """A 10s clip standing in for a highlight's envelope cut (already trimmed
    to [start_time, end_time] by an upstream ffmpeg/-autocrop step)."""
    tmp_dir = tmp_path_factory.mktemp("jump_cuts_src")
    path = str(tmp_dir / "envelope.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=320x568:rate=24:duration=10",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            path,
        ],
        check=True,
    )
    return path


def test_excise_cut_segments_drops_the_gap(tmp_path, synthetic_envelope):
    out_path = str(tmp_path / "excised.mp4")
    # Envelope spans absolute [100.0, 110.0]; keep [100,102] and [107,110],
    # drop the [102,107] gap in the middle.
    cut_segments = [
        {"start_time": 100.0, "end_time": 102.0},
        {"start_time": 107.0, "end_time": 110.0},
    ]

    result = excise_cut_segments(synthetic_envelope, cut_segments, envelope_start=100.0, out_path=out_path)

    assert result == out_path
    duration = _probe_duration(out_path)
    assert abs(duration - 5.0) < 0.3  # 2s + 3s kept, 5s of gap dropped


def test_excise_cut_segments_single_span_matches_input_duration(tmp_path, synthetic_envelope):
    out_path = str(tmp_path / "excised.mp4")
    cut_segments = [{"start_time": 100.0, "end_time": 106.0}]

    excise_cut_segments(synthetic_envelope, cut_segments, envelope_start=100.0, out_path=out_path)

    assert abs(_probe_duration(out_path) - 6.0) < 0.3


def test_excise_cut_segments_cleans_up_temp_dir(tmp_path, synthetic_envelope):
    import os

    out_path = str(tmp_path / "excised.mp4")
    cut_segments = [
        {"start_time": 100.0, "end_time": 102.0},
        {"start_time": 105.0, "end_time": 108.0},
    ]

    excise_cut_segments(synthetic_envelope, cut_segments, envelope_start=100.0, out_path=out_path)

    assert not os.path.exists(out_path + ".parts")
