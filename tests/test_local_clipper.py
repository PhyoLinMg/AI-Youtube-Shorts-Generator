import os
import subprocess

import pytest

import shorts_generator.local.clipper as local_clipper_module
from shorts_generator import captions as captions_module
from shorts_generator.local.clipper import crop_highlights_local


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


def test_output_filename_uses_short_dash_prefix(tmp_path, synthetic_source):
    out_dir = str(tmp_path / "out")
    results = crop_highlights_local(
        synthetic_source,
        [_highlight()],
        aspect_ratio="9:16",
        out_dir=out_dir,
        transcript_segments=_segments(),
    )
    assert os.path.basename(results[0]["clip_url"]) == "Short-01.mp4"


from shorts_generator.hook_card import HookCardError


def _highlight_with_hook():
    return {**_highlight(), "on_screen_hook": "WATCH THIS"}


def test_hook_card_still_picked_before_caption_burn(tmp_path, synthetic_source, monkeypatch):
    """Regression test: the still MUST come from the clean pre-caption
    crop, not the captioned file (a still picked after caption burn would
    freeze that timestamp's burned-in caption line into the card)."""
    order = []

    def _fake_pick(video_path):
        order.append("pick")
        return 0.5

    def _fake_extract(video_path, ts, out_path):
        order.append("extract")
        with open(out_path, "wb") as f:
            f.write(b"fake still")
        return out_path

    def _fake_burn(*args, **kwargs):
        order.append("burn")
        import shutil
        shutil.copyfile(args[0], args[4])
        return args[4]

    def _fake_render(video_path, still_path, hook_text, out_path, duration=1.5):
        order.append("render")
        import shutil
        shutil.copyfile(video_path, out_path)
        return out_path

    monkeypatch.setattr(local_clipper_module, "pick_striking_frame", _fake_pick)
    monkeypatch.setattr(local_clipper_module, "extract_frame", _fake_extract)
    monkeypatch.setattr(local_clipper_module, "burn_captions", _fake_burn)
    monkeypatch.setattr(local_clipper_module, "render_card_overlay", _fake_render)

    crop_highlights_local(
        synthetic_source, [_highlight_with_hook()], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
    )

    assert order == ["pick", "extract", "burn", "render"]


def test_hook_card_skipped_when_flag_off(tmp_path, synthetic_source, monkeypatch):
    def _fail_if_called(*a, **k):
        raise AssertionError("pick_striking_frame should not be called when hook_card=False")
    monkeypatch.setattr(local_clipper_module, "pick_striking_frame", _fail_if_called)

    results = crop_highlights_local(
        synthetic_source, [_highlight_with_hook()], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
        hook_card=False,
    )
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])


def test_hook_card_skipped_when_on_screen_hook_missing(tmp_path, synthetic_source, monkeypatch):
    def _fail_if_called(*a, **k):
        raise AssertionError("pick_striking_frame should not be called without on_screen_hook")
    monkeypatch.setattr(local_clipper_module, "pick_striking_frame", _fail_if_called)

    results = crop_highlights_local(
        synthetic_source, [_highlight()], aspect_ratio="9:16",  # no on_screen_hook
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
    )
    assert results[0]["clip_url"] is not None


def test_hook_card_failure_falls_back_to_captioned_clip(tmp_path, synthetic_source, monkeypatch):
    def _raise(*a, **k):
        raise HookCardError("boom")
    monkeypatch.setattr(local_clipper_module, "pick_striking_frame", _raise)

    results = crop_highlights_local(
        synthetic_source, [_highlight_with_hook()], aspect_ratio="9:16",
        out_dir=str(tmp_path / "out"), transcript_segments=_segments(),
    )
    assert results[0]["clip_url"] is not None
    assert os.path.exists(results[0]["clip_url"])
    assert results[0]["hook_card_error"] == "boom"
    assert "captions_error" not in results[0]
