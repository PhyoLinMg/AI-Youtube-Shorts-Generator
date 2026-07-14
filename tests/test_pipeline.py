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


def _paths(tmp_path):
    root = str(tmp_path / "Video_Title")
    shorts_dir = os.path.join(root, "Shorts")
    os.makedirs(shorts_dir, exist_ok=True)
    return RunPaths(
        root=root,
        shorts_dir=shorts_dir,
        source_video=os.path.join(root, "full_source.mp4"),
        source_json=os.path.join(root, "full_source.json"),
        result_json=os.path.join(root, "result.json"),
        progress_log=os.path.join(root, "progress.log"),
    )


def test_run_local_threads_captions_params(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())

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
    )

    assert result["mode"] == "local"
    assert result["shorts"] == [{"clip_url": "/tmp/out/Short-01.mp4"}]

    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is False
    assert kwargs["caption_fade_duration"] == 0.7
    assert kwargs["word_highlight"] is False
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]


def test_run_local_skips_download_when_source_already_exists(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    with open(paths.source_video, "wb") as f:
        f.write(b"already downloaded")

    def _fail_if_called(*a, **k):
        raise AssertionError("download_youtube_local should not be called when full_source.mp4 exists")

    monkeypatch.setattr(local_downloader_module, "download_youtube_local", _fail_if_called)
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())
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
    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())

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
    )

    assert result["mode"] == "api"
    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is True
    assert kwargs["caption_fade_duration"] == 0.3
    assert kwargs["word_highlight"] is False
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]


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

    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())
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


def test_run_api_recovers_from_corrupted_transcript_cache(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    with open(paths.source_video, "wb") as f:
        f.write(b"cached mp4")
    with open(paths.source_json, "w") as f:
        f.write("{not valid json")  # simulates a truncated/corrupted cache

    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())
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
