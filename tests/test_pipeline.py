from unittest.mock import Mock

import shorts_generator.local.clipper as local_clipper_module
import shorts_generator.local.downloader as local_downloader_module
import shorts_generator.local.transcriber as local_transcriber_module
import shorts_generator.pipeline as pipeline_module


def _fake_transcript():
    return {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi there"}]}


def _fake_highlights_result():
    return {"highlights": [{"start_time": 0.0, "end_time": 3.0, "score": 90, "title": "Clip"}]}


def test_run_local_threads_captions_params(monkeypatch):
    monkeypatch.setattr(local_downloader_module, "download_youtube_local", lambda url, fmt: "/tmp/source.mp4")
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())

    crop_mock = Mock(return_value=[{"clip_url": "/tmp/out/short_01.mp4"}])
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", crop_mock)

    result = pipeline_module._run_local(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.7,
        word_highlight=False,
    )

    assert result["mode"] == "local"
    assert result["shorts"] == [{"clip_url": "/tmp/out/short_01.mp4"}]

    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is False
    assert kwargs["caption_fade_duration"] == 0.7
    assert kwargs["word_highlight"] is False
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]


def test_run_api_threads_captions_params(monkeypatch):
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())

    crop_mock = Mock(return_value=[{"clip_url": "https://hosted.example/short_1.mp4"}])
    monkeypatch.setattr(pipeline_module, "crop_highlights", crop_mock)

    result = pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        word_highlight=False,
    )

    assert result["mode"] == "api"
    _, kwargs = crop_mock.call_args
    assert kwargs["captions"] is True
    assert kwargs["caption_fade_duration"] == 0.3
    assert kwargs["word_highlight"] is False
    assert kwargs["transcript_segments"] == _fake_transcript()["segments"]
