import os
import subprocess

import pytest

from shorts_generator import config
from shorts_generator.local import narration as narration_module


def test_synthesize_narration_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(narration_module, "ELEVENLABS_API_KEY", "")
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        narration_module.synthesize_narration("hello", str(tmp_path / "out.mp3"))


def test_synthesize_narration_writes_audio_from_fake_client(monkeypatch, tmp_path):
    monkeypatch.setattr(narration_module, "ELEVENLABS_API_KEY", "fake-key")

    class _FakeTTS:
        def convert(self, **kwargs):
            assert kwargs["text"] == "hello there"
            return iter([b"chunk1", b"chunk2"])

    class _FakeClient:
        def __init__(self, api_key):
            self.text_to_speech = _FakeTTS()

    monkeypatch.setattr(narration_module, "_get_elevenlabs_client_class", lambda: _FakeClient)

    out_path = str(tmp_path / "out.mp3")
    narration_module.synthesize_narration("hello there", out_path)

    with open(out_path, "rb") as f:
        assert f.read() == b"chunk1chunk2"


def test_synthesize_narration_wraps_client_errors(monkeypatch, tmp_path):
    class _FakeTTS:
        def convert(self, **kwargs):
            raise RuntimeError("api down")

    class _FakeClient:
        def __init__(self, api_key):
            self.text_to_speech = _FakeTTS()

    monkeypatch.setattr(narration_module, "ELEVENLABS_API_KEY", "fake-key")
    monkeypatch.setattr(narration_module, "_get_elevenlabs_client_class", lambda: _FakeClient)

    with pytest.raises(narration_module.NarrationError, match="api down"):
        narration_module.synthesize_narration("hi", str(tmp_path / "out.mp3"))


def test_wrap_text_splits_long_sentence_into_multiple_lines():
    wrapped = narration_module._wrap_text("The Vice President wants to go find proof himself.", max_chars_per_line=28)
    lines = wrapped.split("\n")
    assert len(lines) > 1
    assert all(len(line) <= 28 or " " not in line for line in lines)


@pytest.fixture(scope="module")
def synthetic_audio(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("narration_audio")
    path = str(tmp_dir / "line.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", path],
        check=True,
    )
    return path


def test_render_narration_card_produces_vertical_video_matching_audio_duration(synthetic_audio, tmp_path):
    out_path = str(tmp_path / "card.mp4")
    narration_module.render_narration_card(synthetic_audio, "Test narration line here.", out_path)

    assert os.path.exists(out_path)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height", "-of", "csv=p=0", out_path],
        capture_output=True, text=True, check=True,
    )
    assert "1080" in probe.stdout
    assert "1920" in probe.stdout
