import os
import shutil
import subprocess

import pytest
import requests

from shorts_generator import clipper


@pytest.fixture(scope="module")
def synthetic_clip(tmp_path_factory):
    """Stands in for the mp4 MuAPI would host at the returned clip URL."""
    tmp_dir = tmp_path_factory.mktemp("hosted")
    path = str(tmp_dir / "hosted.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=608x1080:rate=24:duration=4",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            path,
        ],
        check=True,
    )
    return path


def _highlight():
    return {"title": "Test Clip", "start_time": 0.0, "end_time": 3.0, "score": 90}


def _segments():
    return [{"start": 0.0, "end": 3.0, "text": "hello there this is a caption test line"}]


def test_captions_burned_in_by_default(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    out_dir = str(tmp_path / "out")
    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=out_dir,
    )

    assert results[0]["hosted_clip_url"] == "https://hosted.example/short_1.mp4"
    assert os.path.exists(results[0]["clip_url"])
    assert results[0]["clip_url"] != results[0]["hosted_clip_url"]


def test_captions_disabled_keeps_hosted_url(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")

    def _fail_if_called(*a, **k):
        raise AssertionError("_download_to should not be called when captions=False")

    monkeypatch.setattr(clipper, "_download_to", _fail_if_called)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        captions=False,
        out_dir=str(tmp_path / "out"),
    )

    assert results[0]["clip_url"] == "https://hosted.example/short_1.mp4"
    assert "hosted_clip_url" not in results[0]


def test_download_failure_falls_back_to_hosted_url(tmp_path, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")

    def _raise(*a, **k):
        raise requests.RequestException("network down")

    monkeypatch.setattr(clipper, "_download_to", _raise)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
    )

    assert results[0]["clip_url"] == "https://hosted.example/short_1.mp4"
    assert results[0]["captions_error"] == "network down"


def test_word_highlight_flag_forwarded_to_burn(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )
    captured = {}

    def _spy(*args, **kwargs):
        captured.update(kwargs)
        shutil.copyfile(args[0], args[4])
        return args[4]

    monkeypatch.setattr(clipper, "burn_captions", _spy)

    clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
        word_highlight=False,
    )

    assert captured["word_highlight"] is False


def test_output_filename_uses_highlight_title(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
    )

    assert os.path.basename(results[0]["clip_url"]) == "Test_Clip.mp4"


def test_output_filename_dedupes_repeated_titles(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight(), _highlight()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
    )

    basenames = sorted(os.path.basename(r["clip_url"]) for r in results)
    assert basenames == ["Test_Clip.mp4", "Test_Clip_2.mp4"]


from shorts_generator.hook_card import HookCardError


def _highlight_with_hook():
    return {**_highlight(), "on_screen_hook": "WATCH THIS"}


def _stub_hook_card(monkeypatch):
    def _fake_render(video_path, hook_text, out_path, duration=1.5):
        shutil.copyfile(video_path, out_path)
        return out_path

    monkeypatch.setattr(clipper, "render_card_overlay", _fake_render)


def test_hook_card_triggers_local_download_even_when_captions_disabled(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper, "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )
    _stub_hook_card(monkeypatch)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight_with_hook()],
        aspect_ratio="9:16",
        transcript_segments=None,
        captions=False,
        out_dir=str(tmp_path / "out"),
    )

    assert results[0]["clip_url"] != "https://hosted.example/short_1.mp4"
    assert os.path.exists(results[0]["clip_url"])
    assert results[0]["hosted_clip_url"] == "https://hosted.example/short_1.mp4"


def test_no_local_download_when_captions_and_hook_card_both_off(tmp_path, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")

    def _fail_if_called(*a, **k):
        raise AssertionError("_download_to should not be called")
    monkeypatch.setattr(clipper, "_download_to", _fail_if_called)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight_with_hook()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        captions=False,
        hook_card=False,
        out_dir=str(tmp_path / "out"),
    )
    assert results[0]["clip_url"] == "https://hosted.example/short_1.mp4"


def test_hook_card_skipped_when_on_screen_hook_missing(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper, "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    def _fail_if_called(*a, **k):
        raise AssertionError("render_card_overlay should not be called without on_screen_hook")
    monkeypatch.setattr(clipper, "render_card_overlay", _fail_if_called)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight()],  # no on_screen_hook
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
    )
    assert results[0]["clip_url"] is not None
    assert "hook_card_error" not in results[0]


def test_hook_card_failure_falls_back_to_captioned_clip(tmp_path, synthetic_clip, monkeypatch):
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper, "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    def _raise(*a, **k):
        raise HookCardError("boom")
    monkeypatch.setattr(clipper, "render_card_overlay", _raise)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight_with_hook()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
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


def test_multi_cut_segments_excises_the_gap(tmp_path, synthetic_clip, monkeypatch):
    # synthetic_clip is a 4s clip standing in for the hosted [0,4] envelope download.
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    highlight = {
        "title": "Test Clip", "start_time": 0.0, "end_time": 4.0, "score": 90,
        "cut_segments": [
            {"start_time": 0.0, "end_time": 1.0},
            {"start_time": 3.0, "end_time": 4.0},
        ],
    }

    out_dir = str(tmp_path / "out")
    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [highlight],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        captions=False,
        hook_card=False,
        out_dir=out_dir,
    )

    assert "excision_error" not in results[0]
    duration = _probe_duration(results[0]["clip_url"])
    assert abs(duration - 2.0) < 0.3  # kept 1s + 1s, dropped the 2s middle


def test_single_cut_segment_skips_excision(tmp_path, synthetic_clip, monkeypatch):
    # A single-entry cut_segments list must take the exact same path as
    # today's single-span highlights -- no excision step at all.
    #
    # NOTE: duration can't be used as the observable here. synthetic_clip is
    # a fixed 4s fixture, while this highlight's envelope is only 3s; in
    # production the downloaded clip already matches the highlight's span
    # (MuAPI's autocrop trims to start_time/end_time), but the mock download
    # doesn't simulate that. So we assert the invariant the excision step
    # was never invoked, rather than inferring it from output size.
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )
    excise_calls = []
    monkeypatch.setattr(
        clipper, "excise_cut_segments",
        lambda *a, **k: excise_calls.append(a) or a[-1],
    )

    highlight = {
        **_highlight(),
        "cut_segments": [{"start_time": 0.0, "end_time": 3.0}],
    }

    out_dir = str(tmp_path / "out")
    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [highlight],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=out_dir,
    )

    assert excise_calls == []
    assert "excision_error" not in results[0]
    assert os.path.exists(results[0]["clip_url"])
    # No excision happened, so the fixture's own duration should survive untouched.
    duration = _probe_duration(results[0]["clip_url"])
    assert abs(duration - 4.0) < 0.3


def test_multi_cut_segments_uses_segment_aware_captions(tmp_path, synthetic_clip, monkeypatch):
    # The excised path must burn captions with burn_captions_segments (which
    # chunks per kept span so nothing straddles a cut), never the plain
    # burn_captions used by the single-span path.
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    def _fail_if_called(*a, **k):
        raise AssertionError("burn_captions (span-based) must not be used on the excised path")
    monkeypatch.setattr(clipper, "burn_captions", _fail_if_called)

    captured = {}

    def _spy_segments(video_path, transcript_segments, cut_segments, out_path, **kwargs):
        captured["cut_segments"] = cut_segments
        captured.update(kwargs)
        shutil.copyfile(video_path, out_path)
        return out_path

    monkeypatch.setattr(clipper, "burn_captions_segments", _spy_segments)

    cut_segments = [
        {"start_time": 0.0, "end_time": 1.0},
        {"start_time": 3.0, "end_time": 4.0},
    ]
    highlight = {
        "title": "Test Clip", "start_time": 0.0, "end_time": 4.0, "score": 90,
        "cut_segments": cut_segments,
    }

    out_dir = str(tmp_path / "out")
    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [highlight],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        hook_card=False,
        out_dir=out_dir,
        word_highlight=False,
    )

    assert "excision_error" not in results[0]
    assert "captions_error" not in results[0]
    assert captured["cut_segments"] == cut_segments
    assert captured["word_highlight"] is False


def test_excision_failure_falls_back_to_plain_captions_on_unexcised_clip(tmp_path, synthetic_clip, monkeypatch):
    """If excise_cut_segments blows up, crop_highlights must fall back to
    the un-excised download (not fail the whole highlight), and captions
    must route through plain burn_captions against that un-excised clip --
    routing through burn_captions_segments here would place captions
    against a timeline that was never actually excised. Mirrors local
    mode's test_excision_failure_falls_back_to_plain_captions_on_unexcised_clip."""
    def _raise(*args, **kwargs):
        raise clipper.JumpCutError("boom")

    monkeypatch.setattr(clipper, "excise_cut_segments", _raise)
    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper,
        "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )

    segments_calls = []
    plain_calls = []

    def _spy_segments(*args, **kwargs):
        segments_calls.append((args, kwargs))
        shutil.copyfile(args[0], args[3])
        return args[3]

    def _spy_plain(*args, **kwargs):
        plain_calls.append((args, kwargs))
        shutil.copyfile(args[0], args[4])
        return args[4]

    monkeypatch.setattr(clipper, "burn_captions_segments", _spy_segments)
    monkeypatch.setattr(clipper, "burn_captions", _spy_plain)

    highlight = {
        "title": "Test Clip", "start_time": 0.0, "end_time": 4.0, "score": 90,
        "cut_segments": [
            {"start_time": 0.0, "end_time": 1.0},
            {"start_time": 3.0, "end_time": 4.0},
        ],
    }

    out_dir = str(tmp_path / "out")
    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [highlight],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        hook_card=False,
        out_dir=out_dir,
    )

    assert results[0]["excision_error"] == "boom"
    assert "error" not in results[0]
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])
    duration = _probe_duration(results[0]["clip_url"])
    assert abs(duration - 4.0) < 0.3  # un-excised download duration, untouched
    assert len(plain_calls) == 1
    assert len(segments_calls) == 0


def test_caption_failure_preserves_a_successful_hook_card(tmp_path, synthetic_clip, monkeypatch):
    """A caption-burn failure must not discard an already-successful hook
    card -- fall back to the plain (uncaptioned) download and still
    composite the card onto it, matching local mode's behavior."""
    from shorts_generator.captions import CaptionError

    monkeypatch.setattr(clipper, "crop_clip", lambda *a, **k: "https://hosted.example/short_1.mp4")
    monkeypatch.setattr(
        clipper, "_download_to",
        lambda url, dest_path: shutil.copyfile(synthetic_clip, dest_path) or dest_path,
    )
    _stub_hook_card(monkeypatch)

    def _raise_caption_error(*a, **k):
        raise CaptionError("no overlapping transcript")
    monkeypatch.setattr(clipper, "burn_captions", _raise_caption_error)

    results = clipper.crop_highlights(
        "https://source.example/video.mp4",
        [_highlight_with_hook()],
        aspect_ratio="9:16",
        transcript_segments=_segments(),
        out_dir=str(tmp_path / "out"),
    )

    assert results[0]["clip_url"] != "https://hosted.example/short_1.mp4"
    assert os.path.exists(results[0]["clip_url"])
    assert results[0]["captions_error"] == "no overlapping transcript"
    assert "hook_card_error" not in results[0]
