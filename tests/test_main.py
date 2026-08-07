import sys

import pytest

import main as main_module
from main import build_parser


def test_captions_on_by_default():
    args = build_parser().parse_args(["https://example.com/video"])
    assert args.captions is True
    assert args.caption_fade_duration == 0.3


def test_no_captions_flag_disables_captions():
    args = build_parser().parse_args(["https://example.com/video", "--no-captions"])
    assert args.captions is False


def test_caption_fade_duration_flag_overrides_default():
    args = build_parser().parse_args(
        ["https://example.com/video", "--caption-fade-duration", "0.5"]
    )
    assert args.caption_fade_duration == 0.5


def test_word_highlight_on_by_default():
    args = build_parser().parse_args(["https://example.com/video"])
    assert args.word_highlight is True


def test_no_word_highlight_flag_disables():
    args = build_parser().parse_args(["https://example.com/video", "--no-word-highlight"])
    assert args.word_highlight is False


def test_hook_card_on_by_default():
    args = build_parser().parse_args(["https://example.com/video"])
    assert args.hook_card is True


def test_no_hook_card_flag_disables():
    args = build_parser().parse_args(["https://example.com/video", "--no-hook-card"])
    assert args.hook_card is False


def test_clip_type_defaults_to_shorts():
    args = build_parser().parse_args(["https://example.com/video"])
    assert args.clip_type == "shorts"


def test_clip_type_chapters_flag():
    args = build_parser().parse_args(["https://example.com/video", "--clip-type", "chapters"])
    assert args.clip_type == "chapters"


def test_num_chapters_defaults_to_5():
    args = build_parser().parse_args(["https://example.com/video"])
    assert args.num_chapters == 5


def test_num_chapters_flag_overrides_default():
    args = build_parser().parse_args(["https://example.com/video", "--num-chapters", "8"])
    assert args.num_chapters == 8


def test_clip_type_rejects_invalid_value():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["https://example.com/video", "--clip-type", "bogus"])


def _fake_chapters_result():
    return {
        "output_dir": "output/Fake",
        "source_video_url": "/tmp/source.mp4",
        "transcript": {"duration": 10.0, "segments": []},
        "all_chapters": [{"start_time": 0.0, "end_time": 300.0, "title": "T", "summary": "s"}],
        "chapters": [{"start_time": 0.0, "end_time": 300.0, "title": "T", "summary": "s", "clip_url": "output/Fake/Chapters/01_T.mp4"}],
    }


def _fake_shorts_result():
    return {
        "mode": "local",
        "output_dir": "output/Fake",
        "source_video_url": "/tmp/source.mp4",
        "highlights": [{"start_time": 0.0, "end_time": 5.0, "title": "H", "score": 90}],
        "shorts": [{"start_time": 0.0, "end_time": 5.0, "title": "H", "score": 90, "clip_url": "output/Fake/Shorts/H.mp4"}],
    }


def test_main_dispatches_to_generate_chapters_for_clip_type_chapters(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(main_module, "generate_chapters", lambda **kwargs: (calls.append(kwargs), _fake_chapters_result())[1])
    monkeypatch.setattr(main_module, "generate_shorts", lambda **kwargs: pytest.fail("generate_shorts should not be called"))
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/video", "--clip-type", "chapters", "--mode", "local"])

    exit_code = main_module.main()

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["num_chapters"] == 5
    out = capsys.readouterr().out
    assert "Chapters:      1 produced (target was 5)" in out


def test_main_dispatches_to_generate_shorts_by_default(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(main_module, "generate_shorts", lambda **kwargs: (calls.append(kwargs), _fake_shorts_result())[1])
    monkeypatch.setattr(main_module, "generate_chapters", lambda **kwargs: pytest.fail("generate_chapters should not be called"))
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/video"])

    exit_code = main_module.main()

    assert exit_code == 0
    assert len(calls) == 1
    out = capsys.readouterr().out
    assert "Highlights:" in out


def test_main_warns_on_shorts_only_flags_with_clip_type_chapters(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "generate_chapters", lambda **kwargs: _fake_chapters_result())
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "https://example.com/video", "--clip-type", "chapters", "--mode", "local",
         "--num-clips", "7", "--filename-style", "generic"],
    )

    main_module.main()

    err = capsys.readouterr().err
    assert "--num-clips 7" in err
    assert "--filename-style generic" in err
