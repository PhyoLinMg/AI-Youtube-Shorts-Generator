import os
import subprocess

import numpy as np
import pytest

import shorts_generator.local.clipper as local_clipper_module
from shorts_generator import captions as captions_module
from shorts_generator.local.clipper import _clamp_crop_origin, crop_highlights_local


@pytest.fixture(scope="module")
def synthetic_source(tmp_path_factory):
    """A tiny 6s clip with video + audio, generated once for this module."""
    tmp_dir = tmp_path_factory.mktemp("source")
    path = str(tmp_dir / "source.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=24:duration=6",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            path,
        ],
        check=True,
    )
    return path


def _highlight():
    return {"title": "Test Clip", "start_time": 1.0, "end_time": 4.0, "score": 90}


def _segments():
    return [
        {"start": 0.5, "end": 2.5, "text": "hello there this is a test caption"},
        {"start": 2.5, "end": 4.5, "text": "and here is a second phrase for good measure"},
    ]


def test_clamp_crop_origin_centers_when_room_on_both_sides():
    # src 1000x1000, crop 200x200, center at 500,500 -> origin should center it
    assert _clamp_crop_origin((500.0, 500.0), (200, 200), (1000, 1000)) == (400, 400)


def test_clamp_crop_origin_clamps_to_left_edge():
    # center near x=0 would push origin negative -> clamp to 0
    assert _clamp_crop_origin((10.0, 500.0), (200, 200), (1000, 1000)) == (0, 400)


def test_clamp_crop_origin_clamps_to_right_edge():
    # center near x=src_w would push origin past src_w - crop_w -> clamp there
    assert _clamp_crop_origin((990.0, 500.0), (200, 200), (1000, 1000)) == (800, 400)


def test_apply_hysteresis_default_dwell_matches_mode_dwell_seconds():
    # fps=10, MODE_DWELL_SECONDS=0.75 -> dwell=8 frames; a 5-frame flip (< dwell) must not stick
    raw = ["person"] * 10 + ["cursor"] * 5 + ["person"] * 10
    result = local_clipper_module._apply_hysteresis(raw, fps=10.0)
    assert result == ["person"] * 25  # the 5-frame cursor blip never persisted long enough to flip


def test_apply_hysteresis_custom_dwell_flips_faster():
    # same raw sequence, but dwell_seconds=0.3 -> dwell=3 frames; a 5-frame blip DOES flip.
    # Hysteresis is symmetric: entry into "B" is delayed 2 frames (indices 10-11 still read
    # "A" while the run builds to dwell=3), and exit back to "A" is delayed 2 frames the same
    # way (indices 15-16 still read "B"). The 5-frame B run just shifts 2 later -> 12 A / 5 B / 8 A.
    raw = ["A"] * 10 + ["B"] * 5 + ["A"] * 10
    result = local_clipper_module._apply_hysteresis(raw, fps=10.0, dwell_seconds=0.3)
    assert result == ["A"] * 12 + ["B"] * 5 + ["A"] * 8


def test_cluster_face_centers_single_person_tight_range():
    # all detections clustered around x=500 on a 1000-wide frame -> 1 cluster
    detections = [(495.0, 300.0, 100.0, 120.0), (505.0, 305.0, 98.0, 118.0), (500.0, 298.0, 102.0, 121.0)]
    clusters = local_clipper_module._cluster_face_centers(detections, src_w=1000.0)
    assert len(clusters) == 1
    assert clusters[0][0] == pytest.approx(500.0, abs=10)


def test_cluster_face_centers_two_well_separated_people():
    # 10 detections around x=150, 10 around x=850 on a 1000-wide frame -> 2 clusters
    left = [(150.0 + i, 300.0, 100.0, 120.0) for i in range(10)]
    right = [(850.0 + i, 320.0, 100.0, 120.0) for i in range(10)]
    clusters = local_clipper_module._cluster_face_centers(left + right, src_w=1000.0)
    assert len(clusters) == 2
    assert clusters[0][0] < 300  # left cluster first (sorted by x ascending)
    assert clusters[1][0] > 700


def test_cluster_face_centers_stray_outlier_does_not_split():
    # 19 detections around x=500 plus 1 stray outlier at x=950 -> outlier is
    # below MIN_CLUSTER_SAMPLE_FRAC, must NOT be treated as a second person
    main_group = [(500.0 + i, 300.0, 100.0, 120.0) for i in range(19)]
    outlier = [(950.0, 300.0, 100.0, 120.0)]
    clusters = local_clipper_module._cluster_face_centers(main_group + outlier, src_w=1000.0)
    assert len(clusters) == 1


def test_cluster_face_centers_empty_input():
    assert local_clipper_module._cluster_face_centers([], src_w=1000.0) == []


def test_mouth_region_energy_zero_when_region_unchanged():
    gray = np.full((200, 200), 100, dtype=np.uint8)
    prev_gray = gray.copy()
    # face box centered at (100, 100), 80x80
    energy = local_clipper_module._mouth_region_energy(gray, prev_gray, (100.0, 100.0, 80.0, 80.0))
    assert energy == 0.0


def test_mouth_region_energy_positive_when_mouth_region_changed():
    gray = np.full((200, 200), 100, dtype=np.uint8)
    prev_gray = gray.copy()
    # mouth region is the lower half of the face box: y in [cy, cy+h/2] = [100, 140], x in [60, 140]
    gray[110:130, 80:120] = 200
    energy = local_clipper_module._mouth_region_energy(gray, prev_gray, (100.0, 100.0, 80.0, 80.0))
    assert energy > 0.0


def test_mouth_region_energy_ignores_change_outside_face_box():
    gray = np.full((200, 200), 100, dtype=np.uint8)
    prev_gray = gray.copy()
    gray[0:20, 0:20] = 200  # far corner, outside the face box entirely
    energy = local_clipper_module._mouth_region_energy(gray, prev_gray, (100.0, 100.0, 80.0, 80.0))
    assert energy == 0.0


def test_mouth_region_energy_handles_decrease_without_uint8_wraparound():
    # prev brighter than current: on uint8 this underflows/wraps unless the
    # implementation upcasts to a signed dtype before subtracting.
    gray = np.full((200, 200), 100, dtype=np.uint8)
    prev_gray = np.full((200, 200), 100, dtype=np.uint8)
    # same patch as the "changed" test above: 20 rows x 40 cols = 800 px
    prev_gray[110:130, 80:120] = 200
    gray[110:130, 80:120] = 50
    energy = local_clipper_module._mouth_region_energy(gray, prev_gray, (100.0, 100.0, 80.0, 80.0))
    # correct signed diff: |50 - 200| = 150 per pixel, 800 px in the patch
    assert energy == 800 * 150


def test_captions_burned_in_by_default(tmp_path, synthetic_source):
    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source,
        [_highlight()],
        aspect_ratio="9:16",
        out_dir=out_dir,
        transcript_segments=_segments(),
    )

    assert len(results) == 1
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])
    assert "captions_error" not in results[0]


def test_captions_disabled_skips_burn_in(tmp_path, synthetic_source, monkeypatch):
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("burn_captions should not be called when captions=False")

    monkeypatch.setattr("shorts_generator.local.clipper.burn_captions", _fail_if_called)

    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source,
        [_highlight()],
        aspect_ratio="9:16",
        out_dir=out_dir,
        transcript_segments=_segments(),
        captions=False,
    )

    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])


def test_caption_failure_falls_back_to_plain_clip(tmp_path, synthetic_source, monkeypatch):
    def _raise(*args, **kwargs):
        raise captions_module.CaptionError("boom")

    monkeypatch.setattr("shorts_generator.local.clipper.burn_captions", _raise)

    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source,
        [_highlight()],
        aspect_ratio="9:16",
        out_dir=out_dir,
        transcript_segments=_segments(),
    )

    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])
    assert results[0]["captions_error"] == "boom"


def test_word_highlight_flag_forwarded_to_burn(tmp_path, synthetic_source, monkeypatch):
    captured = {}

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        import shutil
        shutil.copyfile(args[0], args[4])
        return args[4]

    monkeypatch.setattr("shorts_generator.local.clipper.burn_captions", _spy)
    crop_highlights_local(
        synthetic_source, [_highlight()], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
        word_highlight=False,
    )
    assert captured["word_highlight"] is False


def test_output_filename_uses_highlight_title(tmp_path, synthetic_source):
    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source,
        [_highlight()],
        aspect_ratio="9:16",
        out_dir=out_dir,
        transcript_segments=_segments(),
    )
    assert os.path.basename(results[0]["clip_url"]) == "Test_Clip.mp4"


def test_output_filename_dedupes_repeated_titles(tmp_path, synthetic_source):
    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source,
        [_highlight(), _highlight()],
        aspect_ratio="9:16",
        out_dir=out_dir,
        transcript_segments=_segments(),
    )
    basenames = sorted(os.path.basename(r["clip_url"]) for r in results)
    assert basenames == ["Test_Clip.mp4", "Test_Clip_2.mp4"]


from shorts_generator.hook_card import HookCardError


def _highlight_with_hook():
    return {**_highlight(), "on_screen_hook": "WATCH THIS"}


def test_hook_card_runs_after_caption_burn(tmp_path, synthetic_source, monkeypatch):
    """The card overlay must run against the already-captioned clip (it's
    the last step), not the clean pre-caption crop."""
    order = []

    def _fake_burn(*args, **kwargs):
        order.append("burn")
        import shutil
        shutil.copyfile(args[0], args[4])
        return args[4]

    def _fake_render(video_path, hook_text, out_path, duration=1.5):
        order.append("render")
        import shutil
        shutil.copyfile(video_path, out_path)
        return out_path

    monkeypatch.setattr(local_clipper_module, "burn_captions", _fake_burn)
    monkeypatch.setattr(local_clipper_module, "render_card_overlay", _fake_render)

    crop_highlights_local(
        synthetic_source, [_highlight_with_hook()], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
    )

    assert order == ["burn", "render"]


def test_hook_card_skipped_when_flag_off(tmp_path, synthetic_source, monkeypatch):
    def _fail_if_called(*a, **k):
        raise AssertionError("render_card_overlay should not be called when hook_card=False")
    monkeypatch.setattr(local_clipper_module, "render_card_overlay", _fail_if_called)

    results = crop_highlights_local(
        synthetic_source, [_highlight_with_hook()], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
        hook_card=False,
    )
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])


def test_hook_card_skipped_when_on_screen_hook_missing(tmp_path, synthetic_source, monkeypatch):
    def _fail_if_called(*a, **k):
        raise AssertionError("render_card_overlay should not be called without on_screen_hook")
    monkeypatch.setattr(local_clipper_module, "render_card_overlay", _fail_if_called)

    results = crop_highlights_local(
        synthetic_source, [_highlight()], aspect_ratio="9:16",  # no on_screen_hook
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
    )
    assert results[0]["clip_url"] is not None


def test_hook_card_failure_falls_back_to_captioned_clip(tmp_path, synthetic_source, monkeypatch):
    def _raise(*a, **k):
        raise HookCardError("boom")
    monkeypatch.setattr(local_clipper_module, "render_card_overlay", _raise)

    results = crop_highlights_local(
        synthetic_source, [_highlight_with_hook()], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
    )
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])
    assert results[0]["hook_card_error"] == "boom"
    assert "captions_error" not in results[0]
