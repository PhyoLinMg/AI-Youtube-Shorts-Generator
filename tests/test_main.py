import sys

import pytest

import main as main_module
from main import build_parser, main


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


def test_main_does_not_warn_about_mode_when_mode_flag_omitted(monkeypatch, capsys):
    # args.mode defaults to "api" when --mode is never typed -- the plain,
    # most natural invocation ("--clip-type chapters" with no --mode at all)
    # must not spuriously warn on every single run just because the default
    # differs from what chapters always uses regardless.
    monkeypatch.setattr(main_module, "generate_chapters", lambda **kwargs: _fake_chapters_result())
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/video", "--clip-type", "chapters"])

    main_module.main()

    err = capsys.readouterr().err
    assert "--mode" not in err


def test_main_warns_on_mode_when_explicitly_passed_non_local(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "generate_chapters", lambda **kwargs: _fake_chapters_result())
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "https://example.com/video", "--clip-type", "chapters", "--mode", "api"],
    )

    main_module.main()

    err = capsys.readouterr().err
    assert "ignoring --mode 'api'" in err


def test_clip_type_accepts_thread():
    args = build_parser().parse_args(["--clip-type", "thread"])
    assert args.clip_type == "thread"


def test_url_is_optional_for_clip_type_thread():
    args = build_parser().parse_args(["--clip-type", "thread"])
    assert args.url is None


def test_url_still_required_positional_when_provided():
    args = build_parser().parse_args(["https://example.com/video"])
    assert args.url == "https://example.com/video"


def test_main_fails_cleanly_without_url_for_shorts(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py"])
    exit_code = main()
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "url is required" in captured.err


def test_main_dispatches_to_generate_threads_for_clip_type_thread(monkeypatch, capsys):
    calls = []
    fake_results = [{
        "output_dir": "output/_Threads/A_x_B",
        "shared_question": "Does X cause Y?",
        "thesis": "Two guests disagree.",
        "bridge": "Here's the other side.",
        "episode_a": {"title": "Episode A", "start_time": 10.0, "end_time": 30.0},
        "episode_b": {"title": "Episode B", "start_time": 5.0, "end_time": 25.0},
        "clip_url": "output/_Threads/A_x_B/clip_1.mp4",
    }]

    def _fake_generate_threads(url_a, url_b, **kwargs):
        calls.append((url_a, url_b, kwargs))
        return fake_results

    monkeypatch.setattr(main_module, "generate_threads", _fake_generate_threads)
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread"])

    exit_code = main()

    assert exit_code == 0
    assert calls == [("https://example.com/a", "https://example.com/b", {"num_clips": 3, "platform": "youtube"})]
    captured = capsys.readouterr()
    assert "Does X cause Y?" in captured.out


def test_main_fails_when_clip_type_thread_missing_url_b(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/a", "--clip-type", "thread"])

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--url-b" in captured.err


def _fake_thread_results():
    return [{
        "output_dir": "d", "shared_question": "q?", "thesis": "t", "bridge": "b",
        "episode_a": {"title": "A", "start_time": 0.0, "end_time": 1.0},
        "episode_b": {"title": "B", "start_time": 0.0, "end_time": 1.0},
        "clip_url": "d/clip_1.mp4",
    }]


def test_main_reports_no_thread_available_when_generate_threads_returns_empty_list(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "generate_threads", lambda url_a, url_b, **kwargs: [])
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread"])

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No shared-question thread found" in captured.err


def test_main_warns_on_shorts_only_flags_with_clip_type_thread(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "generate_threads", lambda url_a, url_b, **kwargs: _fake_thread_results())
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread",
         "--filename-style", "generic", "--mode", "local", "--aspect-ratio", "16:9", "--format", "720",
         "--language", "en", "--framing", "adaptive", "--no-captions", "--caption-fade-duration", "0.5",
         "--no-word-highlight", "--no-hook-card", "--end-card", "--num-chapters", "9"],
    )

    main()

    err = capsys.readouterr().err
    assert "--filename-style generic" in err
    assert "--mode local" in err
    assert "--aspect-ratio 16:9" in err
    assert "--format 720" in err
    assert "--language en" in err
    assert "--framing adaptive" in err
    assert "--no-captions" in err
    assert "--caption-fade-duration 0.5" in err
    assert "--no-word-highlight" in err
    assert "--no-hook-card" in err
    assert "--end-card" in err
    assert "--num-chapters 9" in err
    assert "--num-clips" not in err  # now a live flag for thread mode, not ignored


def test_main_does_not_warn_when_no_extra_flags_passed_with_clip_type_thread(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "generate_threads", lambda url_a, url_b, **kwargs: _fake_thread_results())
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread"])

    main()

    err = capsys.readouterr().err
    assert "ignores" not in err


def test_main_warns_on_mode_equals_form_with_clip_type_thread(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "generate_threads", lambda url_a, url_b, **kwargs: _fake_thread_results())
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread", "--mode=api"])

    main()

    err = capsys.readouterr().err
    assert "--mode api" in err


def test_main_no_thread_message_is_actionable(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "generate_threads", lambda url_a, url_b, **kwargs: [])
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread"])

    exit_code = main()

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "different pair" in err


def test_platform_flag_defaults_to_youtube():
    args = build_parser().parse_args(["--clip-type", "thread"])
    assert args.platform == "youtube"


def test_platform_flag_accepts_tiktok_and_both():
    args = build_parser().parse_args(["--clip-type", "thread", "--platform", "tiktok"])
    assert args.platform == "tiktok"
    args = build_parser().parse_args(["--clip-type", "thread", "--platform", "both"])
    assert args.platform == "both"


def test_main_passes_platform_through_to_generate_threads(monkeypatch, capsys):
    captured_kwargs = {}

    def _fake_generate_threads(url_a, url_b, **kwargs):
        captured_kwargs.update(kwargs)
        return _fake_thread_results()

    monkeypatch.setattr(main_module, "generate_threads", _fake_generate_threads)
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread", "--platform", "tiktok"],
    )

    main()

    assert captured_kwargs["platform"] == "tiktok"


def test_main_labels_thread_summary_by_platform_for_platform_both(monkeypatch, capsys):
    fake_results = [
        {
            "output_dir": "d", "shared_question": "q?", "thesis": "t", "bridge": "b",
            "episode_a": {"title": "A", "start_time": 0.0, "end_time": 1.0},
            "episode_b": {"title": "B", "start_time": 0.0, "end_time": 1.0},
            "clip_url": "d/clip_1.mp4",
            "platform": "youtube", "platform_index": 1,
        },
        {
            "output_dir": "d", "shared_question": "q?", "thesis": "t", "bridge": "b",
            "episode_a": {"title": "A", "start_time": 0.0, "end_time": 1.0},
            "episode_b": {"title": "B", "start_time": 0.0, "end_time": 1.0},
            "clip_url": "d/clip_1.mp4",
            "platform": "tiktok", "platform_index": 1,
        },
    ]
    monkeypatch.setattr(main_module, "generate_threads", lambda url_a, url_b, **kwargs: fake_results)
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread", "--platform", "both"],
    )

    exit_code = main()

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "#1 [youtube]" in captured.out
    assert "#1 [tiktok]" in captured.out
    assert "across platforms" in captured.out


def test_main_warns_when_platform_passed_with_clip_type_shorts(monkeypatch, capsys):
    monkeypatch.setattr(
        main_module, "generate_shorts",
        lambda **kwargs: {"mode": "api", "output_dir": "d", "source_video_url": "u", "highlights": [], "shorts": []},
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "https://youtube.example/x", "--platform", "tiktok"])

    main()

    err = capsys.readouterr().err
    assert "--platform tiktok" in err


def test_main_does_not_warn_when_platform_omitted_with_clip_type_shorts(monkeypatch, capsys):
    monkeypatch.setattr(
        main_module, "generate_shorts",
        lambda **kwargs: {"mode": "api", "output_dir": "d", "source_video_url": "u", "highlights": [], "shorts": []},
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "https://youtube.example/x"])

    main()

    err = capsys.readouterr().err
    assert "--platform" not in err
