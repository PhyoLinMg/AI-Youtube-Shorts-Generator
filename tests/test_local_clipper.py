import os
import subprocess

import cv2
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


def test_two_speaker_positions_hard_cuts_at_speaker_change():
    anchor_a = (200.0, 500.0, 100.0, 120.0)
    anchor_b = (800.0, 500.0, 100.0, 120.0)
    # 10 frames of A, then 10 frames of B. fps=10, SPEAKER_DWELL_SECONDS=0.35 ->
    # dwell = round(0.35*10) = 4 frames. The hysteresis flip only fires once 4
    # consecutive opposite-class raw frames have been seen, so the smoothed
    # switch lands 3 frames after the raw switch: raw B starts at index 10,
    # smoothed flip happens at index 10 + dwell - 1 = 13. Result: 13 frames of
    # A's position, then 7 frames of B's position.
    raw_labels = ["A"] * 10 + ["B"] * 10
    positions = local_clipper_module._two_speaker_positions(
        anchor_a, anchor_b, raw_labels, fps=10.0,
        crop_size=(200, 200), src_size=(1000, 1000),
    )
    assert len(positions) == 20
    expected_a = local_clipper_module._clamp_crop_origin((200.0, 500.0), (200, 200), (1000, 1000))
    expected_b = local_clipper_module._clamp_crop_origin((800.0, 500.0), (200, 200), (1000, 1000))
    assert positions[:13] == [expected_a] * 13
    assert positions[13:] == [expected_b] * 7
    # confirm it's a hard cut: the position at the switch boundary jumps directly,
    # no intermediate value between expected_a and expected_b
    assert positions[12] == expected_a
    assert positions[13] == expected_b


def test_two_speaker_positions_suppresses_brief_flicker():
    anchor_a = (200.0, 500.0, 100.0, 120.0)
    anchor_b = (800.0, 500.0, 100.0, 120.0)
    # a 2-frame "B" blip inside a long "A" run. dwell=4 frames (see above), and
    # 2 < 4, so the blip never persists long enough to flip the smoothed result.
    raw_labels = ["A"] * 10 + ["B"] * 2 + ["A"] * 10
    positions = local_clipper_module._two_speaker_positions(
        anchor_a, anchor_b, raw_labels, fps=10.0,
        crop_size=(200, 200), src_size=(1000, 1000),
    )
    expected_a = local_clipper_module._clamp_crop_origin((200.0, 500.0), (200, 200), (1000, 1000))
    assert positions == [expected_a] * 22


def test_reframe_vertical_falls_back_to_frame_center_when_no_faces(tmp_path, synthetic_source):
    out_path = str(tmp_path / "out.mp4")
    local_clipper_module._reframe_vertical(synthetic_source, out_path, "9:16")
    assert os.path.exists(out_path)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", out_path],
        capture_output=True, text=True, check=True,
    )
    width, height = (int(v) for v in probe.stdout.strip().split(","))
    assert (width, height) == local_clipper_module._output_size(9 / 16)


def test_reframe_vertical_single_person_locks_to_detected_face(tmp_path, synthetic_source, monkeypatch):
    def _fake_detect(self, gray, *args, **kwargs):
        # one consistent face box every call: src is 640x360 (see synthetic_source fixture)
        return [(220, 100, 100, 120)]

    monkeypatch.setattr(cv2.CascadeClassifier, "detectMultiScale", _fake_detect)

    out_path = str(tmp_path / "out_single.mp4")
    local_clipper_module._reframe_vertical(synthetic_source, out_path, "9:16")
    assert os.path.exists(out_path)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", out_path],
        capture_output=True, text=True, check=True,
    )
    width, height = (int(v) for v in probe.stdout.strip().split(","))
    assert (width, height) == local_clipper_module._output_size(9 / 16)


def test_reframe_vertical_two_speakers_completes_and_outputs_correct_size(tmp_path, synthetic_source, monkeypatch):
    call_count = {"n": 0}

    def _fake_detect(self, gray, *args, **kwargs):
        call_count["n"] += 1
        # alternate which side is reported "biggest" across sampled calls so
        # all_detections ends up with two well-separated, well-supported
        # clusters (left ~x=100, right ~x=540 on a 640-wide frame)
        if call_count["n"] % 2 == 0:
            return [(60, 80, 90, 110), (500, 90, 90, 110)]
        return [(70, 85, 88, 108), (510, 95, 88, 108)]

    monkeypatch.setattr(cv2.CascadeClassifier, "detectMultiScale", _fake_detect)

    # Spy on _two_speaker_positions (wraps the real implementation) so the
    # test actually proves the two-speaker branch ran, rather than merely
    # checking output shape -- which a silent fallback to the single-person
    # path would also satisfy.
    real_two_speaker_positions = local_clipper_module._two_speaker_positions
    two_speaker_calls = {"n": 0}

    def _spy_two_speaker_positions(*args, **kwargs):
        two_speaker_calls["n"] += 1
        return real_two_speaker_positions(*args, **kwargs)

    monkeypatch.setattr(local_clipper_module, "_two_speaker_positions", _spy_two_speaker_positions)

    out_path = str(tmp_path / "out_two_speaker.mp4")
    local_clipper_module._reframe_vertical(synthetic_source, out_path, "9:16")
    assert two_speaker_calls["n"] == 1
    assert os.path.exists(out_path)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", out_path],
        capture_output=True, text=True, check=True,
    )
    width, height = (int(v) for v in probe.stdout.strip().split(","))
    assert (width, height) == local_clipper_module._output_size(9 / 16)


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


def test_multi_cut_segments_excises_the_gap(tmp_path, synthetic_source):
    # synthetic_source spans [0,6]; highlight envelope is [1,4] (3s).
    # Keep [1,2] and [3,4], drop the [2,3] middle -> 2s output.
    highlight = {
        "title": "Test Clip", "start_time": 1.0, "end_time": 4.0, "score": 90,
        "cut_segments": [
            {"start_time": 1.0, "end_time": 2.0},
            {"start_time": 3.0, "end_time": 4.0},
        ],
    }

    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source,
        [highlight],
        aspect_ratio="9:16",
        out_dir=out_dir,
        captions=False,
        hook_card=False,
    )

    assert "error" not in results[0]
    duration = _probe_duration(results[0]["clip_url"])
    assert abs(duration - 2.0) < 0.3


def test_single_cut_segment_skips_excision(tmp_path, synthetic_source):
    highlight = {**_highlight(), "cut_segments": [{"start_time": 1.0, "end_time": 4.0}]}

    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source, [highlight], aspect_ratio="9:16", out_dir=out_dir,
        captions=False, hook_card=False,
    )

    assert "error" not in results[0]
    duration = _probe_duration(results[0]["clip_url"])
    assert abs(duration - 3.0) < 0.3


def test_multi_cut_segments_routes_captions_through_segments_burner(tmp_path, synthetic_source, monkeypatch):
    """Excision succeeds -> captions must go through burn_captions_segments
    (chunked per kept span on the concatenated timeline), not burn_captions
    (which would place phrases against the un-excised absolute timeline)."""
    segments_calls = []
    plain_calls = []

    def _spy_segments(*args, **kwargs):
        segments_calls.append((args, kwargs))
        import shutil
        shutil.copyfile(args[0], args[3])
        return args[3]

    def _spy_plain(*args, **kwargs):
        plain_calls.append((args, kwargs))
        import shutil
        shutil.copyfile(args[0], args[4])
        return args[4]

    monkeypatch.setattr(local_clipper_module, "burn_captions_segments", _spy_segments)
    monkeypatch.setattr(local_clipper_module, "burn_captions", _spy_plain)

    highlight = {
        "title": "Test Clip", "start_time": 1.0, "end_time": 4.0, "score": 90,
        "cut_segments": [
            {"start_time": 1.0, "end_time": 2.0},
            {"start_time": 3.0, "end_time": 4.0},
        ],
    }
    results = crop_highlights_local(
        synthetic_source, [highlight], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
        hook_card=False,
    )

    assert len(segments_calls) == 1
    assert len(plain_calls) == 0
    called_cut_segments = segments_calls[0][0][2]
    assert called_cut_segments == highlight["cut_segments"]
    assert "captions_error" not in results[0]
    assert "excision_error" not in results[0]


def test_excision_failure_falls_back_to_plain_captions_on_unexcised_clip(tmp_path, synthetic_source, monkeypatch):
    """If excise_cut_segments blows up, crop_clip_local must fall back to the
    un-excised envelope cut (not fail the whole highlight), and captions
    must route through plain burn_captions against that un-excised clip --
    routing through burn_captions_segments here would place captions against
    a timeline that was never actually excised."""
    def _raise(*args, **kwargs):
        raise local_clipper_module.JumpCutError("boom")

    monkeypatch.setattr(local_clipper_module, "excise_cut_segments", _raise)

    segments_calls = []
    plain_calls = []

    def _spy_segments(*args, **kwargs):
        segments_calls.append((args, kwargs))
        import shutil
        shutil.copyfile(args[0], args[3])
        return args[3]

    def _spy_plain(*args, **kwargs):
        plain_calls.append((args, kwargs))
        import shutil
        shutil.copyfile(args[0], args[4])
        return args[4]

    monkeypatch.setattr(local_clipper_module, "burn_captions_segments", _spy_segments)
    monkeypatch.setattr(local_clipper_module, "burn_captions", _spy_plain)

    highlight = {
        "title": "Test Clip", "start_time": 1.0, "end_time": 4.0, "score": 90,
        "cut_segments": [
            {"start_time": 1.0, "end_time": 2.0},
            {"start_time": 3.0, "end_time": 4.0},
        ],
    }
    results = crop_highlights_local(
        synthetic_source, [highlight], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
        hook_card=False,
    )

    assert results[0]["excision_error"] == "boom"
    assert "error" not in results[0]
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])
    duration = _probe_duration(results[0]["clip_url"])
    assert abs(duration - 3.0) < 0.3  # un-excised envelope duration
    assert len(plain_calls) == 1
    assert len(segments_calls) == 0
