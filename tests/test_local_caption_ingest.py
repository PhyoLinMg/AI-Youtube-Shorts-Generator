import json
import os
import subprocess

import pytest

from shorts_generator.local import caption_ingest as caption_ingest_module
from shorts_generator.local.caption_ingest import _parse_srv1, ingest_captions

# Real cue shape from `yt-dlp --sub-format srv1` on a YouTube auto-caption
# track -- double-escaped entities ("&amp;#39;", "&amp;gt;") are what the
# file actually contains, not a test artifact.
SAMPLE_SRV1 = (
    '<?xml version="1.0" encoding="utf-8" ?><transcript>'
    '<text start="0" dur="3.68">The scary open secret in the AI industry</text>'
    '<text start="2.2" dur="2.96">right now is that it&amp;#39;s possible that</text>'
    '<text start="12.48" dur="2.64">&amp;gt;&amp;gt; It&amp;#39;s quite chilling what you&amp;#39;re saying.</text>'
    '</transcript>'
)


def test_parse_srv1_resolves_double_escaped_entities():
    segments = _parse_srv1(SAMPLE_SRV1, target_segment_seconds=1000.0)  # one merged segment
    text = segments[0]["text"]
    assert "It's quite chilling" in text
    assert ">>" in text
    assert "&#" not in text
    assert "&amp;" not in text


def test_parse_srv1_merges_cues_into_target_length_segments():
    segments = _parse_srv1(SAMPLE_SRV1, target_segment_seconds=6.0)
    # cue 1 (0-3.68) + cue 2 (2.2-5.16) fit within a 6s window measured from
    # the first cue's own start (5.16 - 0 = 5.16 <= 6.0); cue 3 starts far
    # enough past that (12.48) to open a new window on its own.
    assert len(segments) == 2
    assert segments[0]["start"] == 0.0
    assert "scary open secret" in segments[0]["text"]
    assert "chilling" in segments[1]["text"]


def test_parse_srv1_skips_blank_cues():
    xml_text = (
        '<?xml version="1.0"?><transcript>'
        '<text start="0" dur="1"> </text>'
        '<text start="1" dur="2">hello</text>'
        '</transcript>'
    )
    segments = _parse_srv1(xml_text)
    assert len(segments) == 1
    assert segments[0]["text"] == "hello"


def test_parse_srv1_returns_empty_list_for_no_cues():
    assert _parse_srv1('<?xml version="1.0"?><transcript></transcript>') == []


def test_parse_srv1_raises_narration_error_on_malformed_xml():
    with pytest.raises(RuntimeError, match="could not parse srv1"):
        _parse_srv1("not xml at all <<<")


def test_ingest_captions_writes_transcript_and_source_url(tmp_path, monkeypatch):
    monkeypatch.setattr(caption_ingest_module, "_fetch_duration", lambda url: 123.4)
    monkeypatch.setattr(caption_ingest_module, "_fetch_srv1_captions", lambda url, lang="en": SAMPLE_SRV1)

    result = ingest_captions("https://www.youtube.com/watch?v=abc123", base_dir=str(tmp_path))

    run_dir = result["run_dir"]
    assert os.path.isdir(run_dir)
    with open(os.path.join(run_dir, "full_source.json"), "r", encoding="utf-8") as f:
        transcript = json.load(f)
    assert transcript["duration"] == 123.4
    assert len(transcript["segments"]) == result["segment_count"]
    assert all("start" in s and "end" in s and "text" in s for s in transcript["segments"])

    with open(os.path.join(run_dir, "source_url.txt"), "r", encoding="utf-8") as f:
        assert f.read().strip() == "https://www.youtube.com/watch?v=abc123"

    # No full_source.mp4 -- the whole point is no video download.
    assert not os.path.exists(os.path.join(run_dir, "full_source.mp4"))


def test_ingest_captions_raises_when_no_captions_available(tmp_path, monkeypatch):
    monkeypatch.setattr(caption_ingest_module, "_fetch_duration", lambda url: 100.0)

    def _no_captions(url, lang="en"):
        raise RuntimeError(f"no {lang!r} auto-captions available for {url!r}")

    monkeypatch.setattr(caption_ingest_module, "_fetch_srv1_captions", _no_captions)

    with pytest.raises(RuntimeError, match="auto-captions available"):
        ingest_captions("https://www.youtube.com/watch?v=nocaps", base_dir=str(tmp_path))


def test_ingest_captions_skips_fetch_when_already_ingested(tmp_path, monkeypatch):
    run_dir = os.path.join(str(tmp_path), "already")
    os.makedirs(run_dir, exist_ok=True)
    existing = {"duration": 555.0, "segments": [{"start": 0.0, "end": 1.0, "text": "already here"}]}
    with open(os.path.join(run_dir, "full_source.json"), "w", encoding="utf-8") as f:
        json.dump(existing, f)
    with open(os.path.join(run_dir, "source_url.txt"), "w", encoding="utf-8") as f:
        f.write("https://www.youtube.com/watch?v=already")

    def _fail(*a, **k):
        pytest.fail("fetch functions should not be called when the episode is already ingested")

    monkeypatch.setattr(caption_ingest_module, "_fetch_duration", _fail)
    monkeypatch.setattr(caption_ingest_module, "_fetch_srv1_captions", _fail)

    result = ingest_captions("https://www.youtube.com/watch?v=already", base_dir=str(tmp_path))

    assert result["run_dir"] == run_dir
    assert result["duration"] == 555.0
    assert result["segment_count"] == 1


def test_ingest_captions_refetches_when_cached_transcript_is_corrupted(tmp_path, monkeypatch):
    run_dir = os.path.join(str(tmp_path), "corrupt")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "full_source.json"), "w", encoding="utf-8") as f:
        f.write("{not valid json")

    monkeypatch.setattr(caption_ingest_module, "_fetch_duration", lambda url: 42.0)
    monkeypatch.setattr(caption_ingest_module, "_fetch_srv1_captions", lambda url, lang="en": SAMPLE_SRV1)

    result = ingest_captions("https://www.youtube.com/watch?v=corrupt", base_dir=str(tmp_path))

    assert result["run_dir"] == run_dir
    assert result["duration"] == 42.0
    assert result["segment_count"] > 0
    with open(os.path.join(run_dir, "full_source.json"), "r", encoding="utf-8") as f:
        transcript = json.load(f)
    assert transcript["duration"] == 42.0


def test_ingest_captions_refetches_when_cached_transcript_has_wrong_shape(tmp_path, monkeypatch):
    run_dir = os.path.join(str(tmp_path), "wrongshape")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "full_source.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": 10.0, "segments": "not-a-list"}, f)

    monkeypatch.setattr(caption_ingest_module, "_fetch_duration", lambda url: 99.0)
    monkeypatch.setattr(caption_ingest_module, "_fetch_srv1_captions", lambda url, lang="en": SAMPLE_SRV1)

    result = ingest_captions("https://www.youtube.com/watch?v=wrongshape", base_dir=str(tmp_path))

    assert result["run_dir"] == run_dir
    assert result["duration"] == 99.0
    assert result["segment_count"] > 0


def test_ingest_captions_refetches_when_cached_segments_list_is_empty(tmp_path, monkeypatch):
    run_dir = os.path.join(str(tmp_path), "emptyseg")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "full_source.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": 120.0, "segments": []}, f)

    monkeypatch.setattr(caption_ingest_module, "_fetch_duration", lambda url: 77.0)
    monkeypatch.setattr(caption_ingest_module, "_fetch_srv1_captions", lambda url, lang="en": SAMPLE_SRV1)

    result = ingest_captions("https://www.youtube.com/watch?v=emptyseg", base_dir=str(tmp_path))

    assert result["run_dir"] == run_dir
    assert result["duration"] == 77.0
    assert result["segment_count"] > 0


def test_fetch_duration_parses_yt_dlp_stdout(monkeypatch):
    def _fake_run(cmd, **kwargs):
        assert cmd[:3] == ["yt-dlp", "--skip-download", "--print"]
        return subprocess.CompletedProcess(cmd, 0, stdout="7250\n", stderr="")

    monkeypatch.setattr(caption_ingest_module.subprocess, "run", _fake_run)
    assert caption_ingest_module._fetch_duration("https://example.com/x") == 7250.0


def test_fetch_duration_raises_on_unparseable_output(monkeypatch):
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(caption_ingest_module.subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError, match="could not parse a duration"):
        caption_ingest_module._fetch_duration("https://example.com/x")
