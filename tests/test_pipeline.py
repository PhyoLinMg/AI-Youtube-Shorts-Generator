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
        chapters_result_json=os.path.join(root, "chapters_result.json"),
        source_url_txt=os.path.join(root, "source_url.txt"),
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
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: None)

    result = pipeline_module.generate_shorts(
        "https://youtube.example/x",
        mode="local",
        paths=paths,
    )

    assert result["output_dir"] == paths.root
    assert os.path.exists(paths.progress_log)


def test_generate_shorts_persists_source_url(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline_module, "resolve_output_dir", lambda url: paths)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", lambda *a, **k: [{"clip_url": "/tmp/out/Short-01.mp4"}])
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: None)

    pipeline_module.generate_shorts("https://www.youtube.com/watch?v=xyz", mode="local", num_clips=1)

    with open(paths.source_url_txt) as f:
        assert f.read().strip() == "https://www.youtube.com/watch?v=xyz"


def test_generate_shorts_caches_topic_abstract_for_local_mode(tmp_path, monkeypatch):
    # Eager abstract caching lets a future topic search skip a fresh LLM
    # call for every episode already processed -- see corpus.py.
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline_module, "resolve_output_dir", lambda url: paths)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", lambda *a, **k: [])

    calls = []
    monkeypatch.setattr(
        pipeline_module, "get_abstract_cached",
        lambda run_dir, transcript, llm_fn=None: calls.append((run_dir, transcript, llm_fn)),
    )

    pipeline_module.generate_shorts("https://www.youtube.com/watch?v=xyz", mode="local", num_clips=1)

    from shorts_generator.local.llm import call_local_llm
    assert calls == [(paths.root, _fake_transcript(), call_local_llm)]


def test_generate_shorts_caches_topic_abstract_for_api_mode(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    with open(paths.source_video, "wb") as f:
        f.write(b"cached mp4")
    monkeypatch.setattr(pipeline_module, "resolve_output_dir", lambda url: paths)
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())
    monkeypatch.setattr(pipeline_module, "crop_highlights", Mock(return_value=[]))

    calls = []
    monkeypatch.setattr(
        pipeline_module, "get_abstract_cached",
        lambda run_dir, transcript, llm_fn=None: calls.append((run_dir, transcript, llm_fn)),
    )

    pipeline_module.generate_shorts("https://www.youtube.com/watch?v=xyz", mode="api", num_clips=1)

    assert calls == [(paths.root, _fake_transcript(), pipeline_module.call_muapi_llm)]


def test_generate_shorts_does_not_abort_when_abstract_caching_fails(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline_module, "resolve_output_dir", lambda url: paths)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", lambda *a, **k: [{"clip_url": "/tmp/out/Short-01.mp4"}])

    def _raise(*a, **k):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", _raise)

    result = pipeline_module.generate_shorts("https://www.youtube.com/watch?v=xyz", mode="local", num_clips=1)

    assert result["shorts"] == [{"clip_url": "/tmp/out/Short-01.mp4"}]


def test_generate_shorts_writes_a_reusable_abstract_cache_file(tmp_path, monkeypatch):
    # End-to-end check that _cache_topic_abstract's call actually lands a
    # corpus_abstract.json at the path corpus.list_corpus_run_dirs/
    # build_corpus read from -- the earlier tests all stub get_abstract_cached
    # out entirely, so none of them prove a file reaches disk.
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline_module, "resolve_output_dir", lambda url: paths)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", lambda *a, **k: [])

    import shorts_generator.local.llm as local_llm_module
    llm_calls = []

    def _stub_llm(prompt):
        llm_calls.append(prompt)
        return "a topical abstract"

    monkeypatch.setattr(local_llm_module, "call_local_llm", _stub_llm)

    pipeline_module.generate_shorts("https://www.youtube.com/watch?v=xyz", mode="local", num_clips=1)

    abstract_cache_path = os.path.join(paths.root, "corpus_abstract.json")
    assert os.path.isfile(abstract_cache_path)
    assert len(llm_calls) == 1

    # A transcript loaded fresh from full_source.json (as build_corpus does)
    # must fingerprint identically to the in-memory one _cache_topic_abstract
    # was given, or the eager cache is never actually hit later. transcribe_local
    # is stubbed above (it doesn't write the real cache file in this test), so
    # round-trip the transcript through JSON directly rather than relying on a
    # file the pipeline never wrote.
    reloaded_transcript = json.loads(json.dumps(_fake_transcript()))
    from shorts_generator.corpus import get_abstract_cached
    get_abstract_cached(paths.root, reloaded_transcript, llm_fn=_stub_llm)
    assert len(llm_calls) == 1  # cache hit -- no second LLM call


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


def _fake_chapters_result():
    return {"chapters": [{"start_time": 0.0, "end_time": 300.0, "title": "Chapter", "summary": "s"}]}


def _fake_chapters_result_many(count):
    return {
        "chapters": [
            {"start_time": float(i * 400), "end_time": float(i * 400) + 300.0, "title": f"Chapter {i}", "summary": "s"}
            for i in range(count)
        ]
    }


def test_run_local_chapters_threads_captions_params(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_chapters_cached",
        lambda transcript, num_chapters, cache_path, llm_fn: _fake_chapters_result(),
    )

    crop_mock = Mock(return_value=[{"clip_url": "/tmp/out/01_Chapter.mp4"}])
    monkeypatch.setattr(local_clipper_module, "crop_chapters_local", crop_mock)

    result = pipeline_module._run_local_chapters(
        "https://youtube.example/x",
        num_chapters=1,
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.7,
        paths=_paths(tmp_path),
        word_highlight=False,
    )

    assert result["chapters"] == [{"clip_url": "/tmp/out/01_Chapter.mp4"}]

    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is False
    assert kwargs["caption_fade_duration"] == 0.7
    assert kwargs["word_highlight"] is False
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]


def test_run_local_chapters_skips_download_when_source_already_exists(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    with open(paths.source_video, "wb") as f:
        f.write(b"already downloaded")

    def _fail_if_called(*a, **k):
        raise AssertionError("download_youtube_local should not be called when full_source.mp4 exists")

    monkeypatch.setattr(local_downloader_module, "download_youtube_local", _fail_if_called)
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_chapters_cached",
        lambda transcript, num_chapters, cache_path, llm_fn: _fake_chapters_result(),
    )
    monkeypatch.setattr(local_clipper_module, "crop_chapters_local", Mock(return_value=[]))

    result = pipeline_module._run_local_chapters(
        "https://youtube.example/x", num_chapters=1, download_format="720", language=None,
        captions=False, caption_fade_duration=0.3, paths=paths, word_highlight=True,
    )
    assert result["source_video_url"] == paths.source_video


def test_run_local_chapters_raises_on_zero_chapters(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_chapters_cached",
        lambda transcript, num_chapters, cache_path, llm_fn: {"chapters": []},
    )

    with pytest.raises(RuntimeError):
        pipeline_module._run_local_chapters(
            "https://youtube.example/x", num_chapters=1, download_format="720", language=None,
            captions=False, caption_fade_duration=0.3, paths=_paths(tmp_path), word_highlight=True,
        )


def test_run_local_chapters_returns_every_cropped_chapter_no_post_render_trim(tmp_path, monkeypatch):
    # num_chapters is a target/floor hint fed into the LLM prompt (and the
    # count ceiling lives in highlights.get_chapters, pre-render) -- this
    # function must NOT slice the result down to num_chapters afterward.
    # Requesting 2 while 5 chapters were cropped must return all 5, not 2:
    # a post-render trim would always discard whichever chapters sort last
    # chronologically, after already paying the full crop/caption-burn cost
    # for them.
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_chapters_cached",
        lambda transcript, num_chapters, cache_path, llm_fn: _fake_chapters_result_many(5),
    )

    def all_succeed_crop(source_path, chapters, **kwargs):
        return [{**c, "clip_url": f"/tmp/out/{c['title']}.mp4"} for c in chapters]

    monkeypatch.setattr(local_clipper_module, "crop_chapters_local", all_succeed_crop)

    result = pipeline_module._run_local_chapters(
        "https://youtube.example/x", num_chapters=2, download_format="720", language=None,
        captions=False, caption_fade_duration=0.3, paths=_paths(tmp_path), word_highlight=True,
    )
    assert len(result["chapters"]) == 5


def test_generate_chapters_writes_chapter_descriptions_and_result_json(tmp_path, monkeypatch):
    paths = _paths(tmp_path)

    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_chapters_cached",
        lambda transcript, num_chapters, cache_path, llm_fn: _fake_chapters_result(),
    )
    monkeypatch.setattr(
        local_clipper_module, "crop_chapters_local",
        Mock(return_value=[{"start_time": 0.0, "end_time": 300.0, "title": "Chapter", "summary": "s", "clip_url": os.path.join(paths.chapters_dir, "01_Chapter.mp4")}]),
    )
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: None)

    result = pipeline_module.generate_chapters("https://youtube.example/x", paths=paths)

    assert result["output_dir"] == paths.root
    assert os.path.exists(os.path.join(paths.chapters_dir, "chapters_description.txt"))
    assert os.path.exists(paths.chapters_result_json)
    assert os.path.exists(paths.progress_log)


def test_generate_chapters_caches_topic_abstract(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_chapters_cached",
        lambda transcript, num_chapters, cache_path, llm_fn: _fake_chapters_result(),
    )
    monkeypatch.setattr(local_clipper_module, "crop_chapters_local", Mock(return_value=[]))

    calls = []
    monkeypatch.setattr(
        pipeline_module, "get_abstract_cached",
        lambda run_dir, transcript, llm_fn=None: calls.append((run_dir, transcript, llm_fn)),
    )

    pipeline_module.generate_chapters("https://youtube.example/x", paths=paths)

    from shorts_generator.local.llm import call_local_llm
    assert calls == [(paths.root, _fake_transcript(), call_local_llm)]


def test_generate_chapters_does_not_abort_when_abstract_caching_fails(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_chapters_cached",
        lambda transcript, num_chapters, cache_path, llm_fn: _fake_chapters_result(),
    )
    monkeypatch.setattr(
        local_clipper_module, "crop_chapters_local",
        Mock(return_value=[{"start_time": 0.0, "end_time": 300.0, "title": "Chapter", "summary": "s", "clip_url": os.path.join(paths.chapters_dir, "01_Chapter.mp4")}]),
    )

    def _raise(*a, **k):
        raise RuntimeError("LLM unavailable")
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", _raise)

    result = pipeline_module.generate_chapters("https://youtube.example/x", paths=paths)

    assert len(result["chapters"]) == 1


def test_generate_chapters_does_not_clobber_a_prior_generate_shorts_result(tmp_path, monkeypatch):
    # generate_shorts and generate_chapters can both run against the same
    # video (that's the normal workflow this feature exists for) -- they
    # must write to separate result files, or whichever runs second silently
    # destroys the other's result.json.
    paths = _paths(tmp_path)

    with open(paths.result_json, "w", encoding="utf-8") as f:
        json.dump({"mode": "local", "shorts": [{"clip_url": "Shorts/x.mp4"}]}, f)

    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_chapters_cached",
        lambda transcript, num_chapters, cache_path, llm_fn: _fake_chapters_result(),
    )
    monkeypatch.setattr(
        local_clipper_module, "crop_chapters_local",
        Mock(return_value=[{"start_time": 0.0, "end_time": 300.0, "title": "Chapter", "summary": "s", "clip_url": os.path.join(paths.chapters_dir, "01_Chapter.mp4")}]),
    )
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: None)

    pipeline_module.generate_chapters("https://youtube.example/x", paths=paths)

    with open(paths.result_json) as f:
        shorts_result = json.load(f)
    assert shorts_result["shorts"] == [{"clip_url": "Shorts/x.mp4"}]

    with open(paths.chapters_result_json) as f:
        chapters_result = json.load(f)
    assert "chapters" in chapters_result


def _fake_ingest_captions(entries):
    def _ingest(url, base_dir=None):
        return entries[url]
    return _ingest


def test_generate_threads_returns_empty_list_when_no_shared_questions(tmp_path, monkeypatch):
    episode_a_dir = tmp_path / "Episode_A"
    episode_b_dir = tmp_path / "Episode_B"
    episode_a_dir.mkdir()
    episode_b_dir.mkdir()
    (episode_a_dir / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))
    (episode_b_dir / "full_source.json").write_text(json.dumps({"duration": 200.0, "segments": []}))

    monkeypatch.setattr(pipeline_module, "ingest_captions", _fake_ingest_captions({
        "https://example.com/a": {"run_dir": str(episode_a_dir), "title": "Episode A", "duration": 100.0, "segment_count": 0},
        "https://example.com/b": {"run_dir": str(episode_b_dir), "title": "Episode B", "duration": 200.0, "segment_count": 0},
    }))
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: "an abstract")
    monkeypatch.setattr(pipeline_module, "select_thread_pairs", lambda entry_a, entry_b, transcript_a, transcript_b, num_clips, llm_fn: [])

    result = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=2, base_dir=str(tmp_path))

    assert result == []


def test_generate_threads_assembles_and_writes_results_for_each_pair(tmp_path, monkeypatch):
    episode_a_dir = tmp_path / "Episode_A"
    episode_b_dir = tmp_path / "Episode_B"
    episode_a_dir.mkdir()
    episode_b_dir.mkdir()
    (episode_a_dir / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))
    (episode_b_dir / "full_source.json").write_text(json.dumps({"duration": 200.0, "segments": []}))

    monkeypatch.setattr(pipeline_module, "ingest_captions", _fake_ingest_captions({
        "https://example.com/a": {"run_dir": str(episode_a_dir), "title": "Episode A", "duration": 100.0, "segment_count": 0},
        "https://example.com/b": {"run_dir": str(episode_b_dir), "title": "Episode B", "duration": 200.0, "segment_count": 0},
    }))
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: "an abstract")

    fake_pairs = [
        {
            "shared_question": "Does X cause Y?", "thesis": "t1", "bridge": "b1", "title": "Title One #Shorts",
            "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 10.0, "end_time": 30.0},
            "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 5.0, "end_time": 25.0},
        },
        {
            "shared_question": "Does A cause B?", "thesis": "t2", "bridge": "b2", "title": "Title Two #Shorts",
            "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 40.0, "end_time": 60.0},
            "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 35.0, "end_time": 55.0},
        },
    ]
    monkeypatch.setattr(pipeline_module, "select_thread_pairs", lambda entry_a, entry_b, transcript_a, transcript_b, num_clips, llm_fn: fake_pairs)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt="720": open(target_path, "wb").write(b"full source") or target_path,
    )
    monkeypatch.setattr(pipeline_module, "acquire_clip", lambda run_dir, source_url, cached_duration, start_time, end_time, out_path: open(out_path, "wb").write(b"clip") or {"clip_path": out_path})
    monkeypatch.setattr(pipeline_module, "synthesize_narration", lambda text, out_path, **k: open(out_path, "wb").write(b"audio") or out_path)
    monkeypatch.setattr(pipeline_module, "render_narration_card", lambda audio_path, text, out_path: open(out_path, "wb").write(b"card") or out_path)
    assemble_calls = []
    monkeypatch.setattr(pipeline_module, "assemble_thread", lambda segment_paths, out_path: (assemble_calls.append(segment_paths), open(out_path, "wb").write(b"final"))[1] or out_path)

    result = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=2, base_dir=str(tmp_path))

    assert len(result) == 2
    out_dir = result[0]["output_dir"]
    assert result[0]["clip_url"] == os.path.join(out_dir, "thesis_1_Title_One_Shorts.mp4")
    assert result[1]["clip_url"] == os.path.join(out_dir, "thesis_2_Title_Two_Shorts.mp4")
    assert result[0]["episode_a"]["clip_url"] == os.path.join(out_dir, "raw", "thesis_1", "clip_1_a.mp4")
    assert result[0]["episode_b"]["clip_url"] == os.path.join(out_dir, "raw", "thesis_1", "clip_1_b.mp4")
    assert result[1]["episode_a"]["clip_url"] == os.path.join(out_dir, "raw", "thesis_2", "clip_2_a.mp4")
    assert assemble_calls[0] == [
        os.path.join(out_dir, "raw", "thesis_1", "intro_card_1.mp4"), os.path.join(out_dir, "raw", "thesis_1", "clip_1_a.mp4"),
        os.path.join(out_dir, "raw", "thesis_1", "bridge_card_1.mp4"), os.path.join(out_dir, "raw", "thesis_1", "clip_1_b.mp4"),
    ]
    assert os.path.isfile(os.path.join(out_dir, "thread_results.json"))
    with open(os.path.join(out_dir, "thread_results.json")) as f:
        written = json.load(f)
    assert len(written) == 2


def test_generate_threads_calls_on_output_dir_before_the_slow_work(tmp_path, monkeypatch):
    episode_a_dir = tmp_path / "Episode_A"
    episode_b_dir = tmp_path / "Episode_B"
    episode_a_dir.mkdir()
    episode_b_dir.mkdir()
    (episode_a_dir / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))
    (episode_b_dir / "full_source.json").write_text(json.dumps({"duration": 200.0, "segments": []}))

    monkeypatch.setattr(pipeline_module, "ingest_captions", _fake_ingest_captions({
        "https://example.com/a": {"run_dir": str(episode_a_dir), "title": "Episode A", "duration": 100.0, "segment_count": 0},
        "https://example.com/b": {"run_dir": str(episode_b_dir), "title": "Episode B", "duration": 200.0, "segment_count": 0},
    }))
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: "an abstract")

    fake_pairs = [{
        "shared_question": "Does X cause Y?", "thesis": "t1", "bridge": "b1",
        "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 10.0, "end_time": 30.0},
        "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 5.0, "end_time": 25.0},
    }]
    monkeypatch.setattr(pipeline_module, "select_thread_pairs", lambda entry_a, entry_b, transcript_a, transcript_b, num_clips, llm_fn: fake_pairs)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt="720": open(target_path, "wb").write(b"full source") or target_path,
    )

    calls = []

    def _fake_acquire_clip(run_dir, source_url, cached_duration, start_time, end_time, out_path):
        assert calls, "on_output_dir was not called before acquire_clip"
        open(out_path, "wb").write(b"clip")
        return {"clip_path": out_path}

    monkeypatch.setattr(pipeline_module, "acquire_clip", _fake_acquire_clip)
    monkeypatch.setattr(pipeline_module, "synthesize_narration", lambda text, out_path, **k: open(out_path, "wb").write(b"audio") or out_path)
    monkeypatch.setattr(pipeline_module, "render_narration_card", lambda audio_path, text, out_path: open(out_path, "wb").write(b"card") or out_path)
    monkeypatch.setattr(pipeline_module, "assemble_thread", lambda segment_paths, out_path: open(out_path, "wb").write(b"final") or out_path)

    result = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=1, base_dir=str(tmp_path), on_output_dir=calls.append)

    assert calls == [result[0]["output_dir"]]
    assert os.path.isfile(os.path.join(result[0]["output_dir"], "progress.log"))


def _setup_thread_run(tmp_path, monkeypatch, num_pairs=2):
    episode_a_dir = tmp_path / "Episode_A"
    episode_b_dir = tmp_path / "Episode_B"
    episode_a_dir.mkdir()
    episode_b_dir.mkdir()
    (episode_a_dir / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))
    (episode_b_dir / "full_source.json").write_text(json.dumps({"duration": 200.0, "segments": []}))

    monkeypatch.setattr(pipeline_module, "ingest_captions", _fake_ingest_captions({
        "https://example.com/a": {"run_dir": str(episode_a_dir), "title": "Episode A", "duration": 100.0, "segment_count": 0},
        "https://example.com/b": {"run_dir": str(episode_b_dir), "title": "Episode B", "duration": 200.0, "segment_count": 0},
    }))
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: "an abstract")

    fake_pairs = [{
        "shared_question": f"Question {i}?", "thesis": f"t{i}", "bridge": f"b{i}",
        "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 10.0 * i, "end_time": 10.0 * i + 5},
        "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 10.0 * i, "end_time": 10.0 * i + 5},
    } for i in range(1, num_pairs + 1)]
    monkeypatch.setattr(pipeline_module, "select_thread_pairs", lambda entry_a, entry_b, transcript_a, transcript_b, num_clips, llm_fn: fake_pairs)
    monkeypatch.setattr(pipeline_module, "acquire_clip", lambda run_dir, source_url, cached_duration, start_time, end_time, out_path: open(out_path, "wb").write(b"clip") or {"clip_path": out_path})
    monkeypatch.setattr(pipeline_module, "synthesize_narration", lambda text, out_path, **k: open(out_path, "wb").write(b"audio") or out_path)
    monkeypatch.setattr(pipeline_module, "render_narration_card", lambda audio_path, text, out_path: open(out_path, "wb").write(b"card") or out_path)
    monkeypatch.setattr(pipeline_module, "assemble_thread", lambda segment_paths, out_path: open(out_path, "wb").write(b"final") or out_path)
    return episode_a_dir, episode_b_dir


def test_generate_threads_downloads_full_source_once_and_deletes_after_all_clips(tmp_path, monkeypatch):
    episode_a_dir, episode_b_dir = _setup_thread_run(tmp_path, monkeypatch, num_pairs=2)

    download_calls = []

    def _fake_download(url, target_path, fmt="720"):
        download_calls.append((url, target_path))
        open(target_path, "wb").write(b"full source")
        return target_path

    monkeypatch.setattr(local_downloader_module, "download_youtube_local", _fake_download)

    pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=2, base_dir=str(tmp_path))

    # One download per episode, not one per clip, even though num_clips=2.
    assert len(download_calls) == 2
    assert {c[0] for c in download_calls} == {"https://example.com/a", "https://example.com/b"}

    # Downloaded once all clips are cut and assembled, the full videos are
    # removed rather than left on disk.
    assert not os.path.exists(episode_a_dir / "full_source.mp4")
    assert not os.path.exists(episode_b_dir / "full_source.mp4")


def test_generate_threads_cleans_up_partial_download_when_one_episode_fails(tmp_path, monkeypatch):
    """A downloads fine, B's download raises -- A's now-orphaned
    full_source.mp4 must still be removed, not leaked, even though the
    overall run fails."""
    episode_a_dir, episode_b_dir = _setup_thread_run(tmp_path, monkeypatch, num_pairs=1)

    def _fake_download(url, target_path, fmt="720"):
        if "/b" in url:
            raise RuntimeError("simulated network failure for B")
        open(target_path, "wb").write(b"full source")
        return target_path

    monkeypatch.setattr(local_downloader_module, "download_youtube_local", _fake_download)

    with pytest.raises(RuntimeError, match="simulated network failure for B"):
        pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=1, base_dir=str(tmp_path))

    assert not os.path.exists(episode_a_dir / "full_source.mp4")
    assert not os.path.exists(episode_b_dir / "full_source.mp4")


def test_generate_threads_does_not_download_or_delete_preexisting_full_source(tmp_path, monkeypatch):
    episode_a_dir, episode_b_dir = _setup_thread_run(tmp_path, monkeypatch, num_pairs=1)
    # Simulate a full_source.mp4 already on disk from a prior Shorts/chapters
    # run on the same URL -- it should be reused, not re-downloaded, and
    # must survive this thread run's cleanup since we didn't create it.
    (episode_a_dir / "full_source.mp4").write_bytes(b"preexisting")
    (episode_b_dir / "full_source.mp4").write_bytes(b"preexisting")

    def _fail_if_called(url, target_path, fmt="720"):
        raise AssertionError(f"download_youtube_local should not be called for {url!r}")

    monkeypatch.setattr(local_downloader_module, "download_youtube_local", _fail_if_called)
    # acquire_clip is patched by _setup_thread_run to always succeed
    # regardless of whether full_source.mp4 exists, so this only exercises
    # generate_threads' own download-then-cleanup bookkeeping.

    pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=1, base_dir=str(tmp_path))

    assert (episode_a_dir / "full_source.mp4").read_bytes() == b"preexisting"
    assert (episode_b_dir / "full_source.mp4").read_bytes() == b"preexisting"


def test_generate_threads_final_filename_falls_back_to_shared_question_when_title_missing(tmp_path, monkeypatch):
    """_setup_thread_run's fake pairs (used by most of the tests below this
    one) don't set a "title" key -- generate_threads must not crash on that,
    and should fall back to shared_question for the final filename, same
    fallback write_thread_descriptions already uses."""
    _setup_thread_run(tmp_path, monkeypatch, num_pairs=1)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt="720": open(target_path, "wb").write(b"full source") or target_path,
    )

    result = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=1, base_dir=str(tmp_path))

    out_dir = result[0]["output_dir"]
    assert result[0]["clip_url"] == os.path.join(out_dir, "thesis_1_Question_1.mp4")


def test_generate_threads_archives_prior_same_slug_run_before_second_call(tmp_path, monkeypatch):
    """Integration check that generate_threads actually wires up
    archive_stale_thread_run (unit-tested on its own): calling it twice for
    the same episode pair on the same day must not mix the two runs' files
    -- the first run's thread_results.json (and everything else) should
    land under raw/stale/<timestamp>/ before the second run writes."""
    _setup_thread_run(tmp_path, monkeypatch, num_pairs=1)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt="720": open(target_path, "wb").write(b"full source") or target_path,
    )

    first = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=1, base_dir=str(tmp_path))
    out_dir = first[0]["output_dir"]

    second = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=1, base_dir=str(tmp_path))

    assert second[0]["output_dir"] == out_dir
    stale_root = os.path.join(out_dir, "raw", "stale")
    assert os.path.isdir(stale_root)
    archived_dirs = os.listdir(stale_root)
    assert len(archived_dirs) == 1
    assert os.path.isfile(os.path.join(stale_root, archived_dirs[0], "thread_results.json"))
    # The fresh run's own final file must not have been swept into the archive.
    assert os.path.isfile(second[0]["clip_url"])
