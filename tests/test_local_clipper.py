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
