import json
import os
from unittest.mock import Mock

import pytest

import shorts_generator.local.clipper as local_clipper_module
import shorts_generator.local.downloader as local_downloader_module
import shorts_generator.local.transcriber as local_transcriber_module
import shorts_generator.pipeline as pipeline_module
from shorts_generator.run_output import RunPaths


def _fake_transcript():
    return {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi there"}]}


def _fake_highlights_result():
    return {"highlights": [{"start_time": 0.0, "end_time": 3.0, "score": 90, "title": "Clip"}]}


def _fake_highlights_result_many(count):
    return {
        "highlights": [
            {"start_time": float(i), "end_time": float(i) + 3.0, "score": 100 - i, "title": f"Clip {i}"}
            for i in range(count)
        ]
    }


def _paths(tmp_path):
    root = str(tmp_path / "Video_Title")
    shorts_dir = os.path.join(root, "Shorts")
    chapters_dir = os.path.join(root, "Chapters")
    os.makedirs(shorts_dir, exist_ok=True)
    os.makedirs(chapters_dir, exist_ok=True)
    return RunPaths(
        root=root,
        shorts_dir=shorts_dir,
        chapters_dir=chapters_dir,
        source_video=os.path.join(root, "full_source.mp4"),
        source_json=os.path.join(root, "full_source.json"),
        highlights_json=os.path.join(root, "highlights.json"),
        chapters_json=os.path.join(root, "chapters.json"),
        result_json=os.path.join(root, "result.json"),
        progress_log=os.path.join(root, "progress.log"),
    )


def test_run_local_threads_captions_params(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())

    crop_mock = Mock(return_value=[{"clip_url": "/tmp/out/Short-01.mp4"}])
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", crop_mock)

    result = pipeline_module._run_local(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.7,
        paths=_paths(tmp_path),
        word_highlight=False,
        hook_card=False,
    )

    assert result["mode"] == "local"
    assert result["shorts"] == [{"clip_url": "/tmp/out/Short-01.mp4"}]

    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is False
    assert kwargs["caption_fade_duration"] == 0.7
    assert kwargs["word_highlight"] is False
    assert kwargs["hook_card"] is False
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]


def test_run_local_crops_num_clips_candidates_via_claim_specificity_gate(tmp_path, monkeypatch):
    # _fake_highlights_result_many gives no candidate a claim_specificity
    # field, so every candidate is a non-passer (defaults to 0) and the
    # gate falls back to plain top-num_clips-by-score.
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_highlights_cached",
        lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result_many(5),
    )

    crop_mock = Mock(return_value=[])
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", crop_mock)

    pipeline_module._run_local(
        "https://youtube.example/x",
        num_clips=2,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    args, _ = crop_mock.call_args
    top = args[1]
    assert len(top) == 2 + pipeline_module.CROP_FAILURE_BUFFER


def test_run_local_skips_download_when_source_already_exists(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    with open(paths.source_video, "wb") as f:
        f.write(b"already downloaded")

    def _fail_if_called(*a, **k):
        raise AssertionError("download_youtube_local should not be called when full_source.mp4 exists")

    monkeypatch.setattr(local_downloader_module, "download_youtube_local", _fail_if_called)
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", Mock(return_value=[]))

    result = pipeline_module._run_local(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.3,
        paths=paths,
        word_highlight=True,
    )

    assert result["source_video_url"] == paths.source_video


def _fake_download_to(url, dest):
    with open(dest, "wb") as f:
        f.write(b"fake downloaded mp4")
    return dest


def test_run_api_threads_captions_params(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())

    crop_mock = Mock(return_value=[{"clip_url": "https://hosted.example/Short-1.mp4"}])
    monkeypatch.setattr(pipeline_module, "crop_highlights", crop_mock)

    result = pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=False,
        hook_card=False,
    )

    assert result["mode"] == "api"
    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is True
    assert kwargs["caption_fade_duration"] == 0.3
    assert kwargs["word_highlight"] is False
    assert kwargs["hook_card"] is False
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]


def test_run_api_crops_num_clips_candidates_via_claim_specificity_gate(tmp_path, monkeypatch):
    # Same rationale as the local-mode version above: no candidate has a
    # claim_specificity field, so the gate falls back to top-num_clips-by-score.
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_highlights_cached",
        lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result_many(5),
    )

    crop_mock = Mock(return_value=[])
    monkeypatch.setattr(pipeline_module, "crop_highlights", crop_mock)

    pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=2,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    args, _ = crop_mock.call_args
    top = args[1]
    assert len(top) == 2 + pipeline_module.CROP_FAILURE_BUFFER


def _fake_highlights_result_mixed_specificity():
    return {
        "highlights": [
            {"start_time": 0.0, "end_time": 3.0, "score": 99, "claim_specificity": 30, "title": "High score, vague"},
            {"start_time": 10.0, "end_time": 13.0, "score": 70, "claim_specificity": 85, "title": "Lower score, specific"},
        ]
    }


def test_run_api_gate_prefers_claim_specificity_over_raw_score(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_highlights_cached",
        lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result_mixed_specificity(),
    )

    crop_mock = Mock(return_value=[])
    monkeypatch.setattr(pipeline_module, "crop_highlights", crop_mock)

    pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    args, _ = crop_mock.call_args
    top = args[1]
    assert len(top) == 1 + pipeline_module.CROP_FAILURE_BUFFER
    assert top[0]["title"] == "Lower score, specific"


def test_run_api_calls_score_visual_hooks_before_crop(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())

    calls = []

    def fake_score_visual_hooks(source_video_path, highlights, llm_fn):
        calls.append((source_video_path, len(highlights)))
        return highlights

    monkeypatch.setattr(pipeline_module, "score_visual_hooks", fake_score_visual_hooks)
    monkeypatch.setattr(pipeline_module, "crop_highlights", Mock(return_value=[]))

    paths = _paths(tmp_path)
    pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=paths,
        word_highlight=True,
    )

    assert len(calls) == 1
    assert calls[0] == (paths.source_video, 1)


def test_run_api_visual_hook_failure_does_not_abort_run(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())

    def raising_score_visual_hooks(source_video_path, highlights, llm_fn):
        raise RuntimeError("vision backend down")

    monkeypatch.setattr(pipeline_module, "score_visual_hooks", raising_score_visual_hooks)
    crop_mock = Mock(return_value=[{"clip_url": "https://hosted.example/Short-1.mp4"}])
    monkeypatch.setattr(pipeline_module, "crop_highlights", crop_mock)

    result = pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    assert result["mode"] == "api"
    assert crop_mock.called  # pipeline continued past the failed visual-hook scoring


def test_run_local_calls_score_visual_hooks_before_crop(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())

    calls = []

    def fake_score_visual_hooks(source_video_path, highlights, llm_fn):
        calls.append((source_video_path, len(highlights)))
        return highlights

    monkeypatch.setattr(pipeline_module, "score_visual_hooks", fake_score_visual_hooks)
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", Mock(return_value=[]))

    pipeline_module._run_local(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    assert len(calls) == 1
    assert calls[0] == ("/tmp/source.mp4", 1)


def test_run_api_skips_local_copy_and_transcribe_when_cached(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    with open(paths.source_video, "wb") as f:
        f.write(b"cached mp4")
    with open(paths.source_json, "w") as f:
        json.dump(_fake_transcript(), f)

    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")

    def _fail_download_to(*a, **k):
        raise AssertionError("_download_to should not be called when full_source.mp4 is cached")
    monkeypatch.setattr(pipeline_module, "_download_to", _fail_download_to)

    def _fail_transcribe(*a, **k):
        raise AssertionError("transcribe should not be called when full_source.json is cached")
    monkeypatch.setattr(pipeline_module, "transcribe", _fail_transcribe)

    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())
    monkeypatch.setattr(pipeline_module, "crop_highlights", Mock(return_value=[]))

    result = pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=paths,
        word_highlight=True,
    )

    assert result["transcript"] == _fake_transcript()


def test_run_api_interrupted_download_does_not_leave_partial_source_video(tmp_path, monkeypatch):
    paths = _paths(tmp_path)

    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")

    def _write_partial_then_raise(url, dest_path):
        with open(dest_path, "wb") as f:
            f.write(b"only half a file")
        raise ConnectionError("connection dropped")

    monkeypatch.setattr(pipeline_module, "_download_to", _write_partial_then_raise)

    with pytest.raises(ConnectionError):
        pipeline_module._run_api(
            "https://youtube.example/x",
            num_clips=1,
            aspect_ratio="9:16",
            download_format="720",
            language=None,
            captions=True,
            caption_fade_duration=0.3,
            paths=paths,
            word_highlight=True,
        )

    # The interrupted write must not land at the final path, or a rerun
    # would treat the truncated file as a valid cached source.
    assert not os.path.exists(paths.source_video)


def test_generate_shorts_uses_provided_paths_without_resolving(tmp_path, monkeypatch):
    paths = _paths(tmp_path)

    def _fail_resolve(*a, **k):
        raise AssertionError("resolve_output_dir should not be called when paths is provided")
    monkeypatch.setattr(pipeline_module, "resolve_output_dir", _fail_resolve)

    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", Mock(return_value=[]))

    result = pipeline_module.generate_shorts(
        "https://youtube.example/x",
        mode="local",
        paths=paths,
    )

    assert result["output_dir"] == paths.root
    assert os.path.exists(paths.progress_log)


def test_run_api_recovers_from_corrupted_transcript_cache(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    with open(paths.source_video, "wb") as f:
        f.write(b"cached mp4")
    with open(paths.source_json, "w") as f:
        f.write("{not valid json")  # simulates a truncated/corrupted cache

    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())
    monkeypatch.setattr(pipeline_module, "crop_highlights", Mock(return_value=[]))

    result = pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=paths,
        word_highlight=True,
    )

    assert result["transcript"] == _fake_transcript()


def test_run_api_trims_extra_successful_crops_to_num_clips(tmp_path, monkeypatch):
    # Every buffered candidate crops successfully -- the pipeline must trim
    # back to exactly num_clips rather than over-delivering.
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_highlights_cached",
        lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result_many(5),
    )

    def all_succeed_crop(source_url, top, **kwargs):
        return [{**h, "clip_url": f"https://hosted.example/{h['title']}.mp4"} for h in top]

    monkeypatch.setattr(pipeline_module, "crop_highlights", all_succeed_crop)

    result = pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=2,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    assert len(result["shorts"]) == 2
    assert all(s.get("clip_url") for s in result["shorts"])


def test_run_api_buffer_covers_a_single_crop_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_highlights_cached",
        lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result_many(5),
    )

    def flaky_crop(source_url, top, **kwargs):
        out = []
        for i, h in enumerate(top):
            if i == 0:
                out.append({**h, "clip_url": None, "error": "autocrop failed"})
            else:
                out.append({**h, "clip_url": f"https://hosted.example/{h['title']}.mp4"})
        return out

    monkeypatch.setattr(pipeline_module, "crop_highlights", flaky_crop)

    result = pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=2,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    assert len(result["shorts"]) == 2
    assert all(s.get("clip_url") for s in result["shorts"])


def test_run_api_returns_available_successes_when_buffer_insufficient(tmp_path, monkeypatch):
    # More failures than the buffer can cover -- the shortfall must stay
    # visible (as "Failed" cards downstream in the webapp) rather than being
    # silently hidden.
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_highlights_cached",
        lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result_many(5),
    )

    def mostly_failing_crop(source_url, top, **kwargs):
        out = []
        for i, h in enumerate(top):
            if i < len(top) - 1:
                out.append({**h, "clip_url": None, "error": "autocrop failed"})
            else:
                out.append({**h, "clip_url": f"https://hosted.example/{h['title']}.mp4"})
        return out

    monkeypatch.setattr(pipeline_module, "crop_highlights", mostly_failing_crop)

    result = pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=2,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    assert len(result["shorts"]) == 2 + pipeline_module.CROP_FAILURE_BUFFER
    assert sum(1 for s in result["shorts"] if s.get("clip_url")) == 1


def test_run_api_deletes_orphaned_buffer_clip_after_trim(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_highlights_cached",
        lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result_many(5),
    )

    paths = _paths(tmp_path)
    # num_clips=2 + CROP_FAILURE_BUFFER=1 -> 3 candidates get cropped, each
    # backed by a real file on disk (matching what crop_highlights actually
    # does -- writes a file for every candidate it's given, before any
    # trimming happens).
    clip_paths = {}
    for i in range(3):
        p = os.path.join(paths.shorts_dir, f"clip_{i}.mp4")
        with open(p, "wb") as f:
            f.write(b"fake mp4 bytes")
        clip_paths[f"Clip {i}"] = p

    def all_succeed_crop(source_url, top, **kwargs):
        return [{**h, "clip_url": clip_paths[h["title"]]} for h in top]

    monkeypatch.setattr(pipeline_module, "crop_highlights", all_succeed_crop)

    pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=2,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=paths,
        word_highlight=True,
    )

    # _fake_highlights_result_many gives "Clip 0".."Clip 4" scores 100..96
    # (descending), none with claim_specificity, so select_final_highlights
    # falls back to plain top-3-by-score: Clip 0, Clip 1, Clip 2 (in that
    # order). The two highest-scored (Clip 0, Clip 1) are kept.
    assert os.path.exists(clip_paths["Clip 0"])
    assert os.path.exists(clip_paths["Clip 1"])
    # Clip 2 is the trimmed-away buffer candidate -- its file must be
    # deleted, not left orphaned on disk indefinitely.
    assert not os.path.exists(clip_paths["Clip 2"])


def test_run_local_buffer_covers_a_single_crop_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_highlights_cached",
        lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result_many(5),
    )

    def flaky_crop_local(source_path, top, **kwargs):
        out = []
        for i, h in enumerate(top):
            if i == 0:
                out.append({**h, "clip_url": None, "error": "crop failed"})
            else:
                out.append({**h, "clip_url": f"/tmp/out/{h['title']}.mp4"})
        return out

    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", flaky_crop_local)

    result = pipeline_module._run_local(
        "https://youtube.example/x",
        num_clips=2,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    assert len(result["shorts"]) == 2
    assert all(s.get("clip_url") for s in result["shorts"])
