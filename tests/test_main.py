import pytest

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
