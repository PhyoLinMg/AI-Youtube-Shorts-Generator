import openai

from shorts_generator import config
from shorts_generator.local import llm as local_llm


class _FakeMessage:
    content = "ok"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


class _FakeCompletions:
    def create(self, **kwargs):
        return _FakeResponse()


class _FakeChat:
    completions = _FakeCompletions()


class _FakeOpenAI:
    """Captures the kwargs it's constructed with, like the real OpenAI client."""

    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.chat = _FakeChat()


def test_call_openai_llm_sets_timeout(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(local_llm, "require_openai_key", lambda: "test-key")

    local_llm.call_openai_llm("prompt")

    assert _FakeOpenAI.last_kwargs["timeout"] == config.LOCAL_LLM_TIMEOUT_SECONDS


def test_call_openrouter_llm_sets_timeout(monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(local_llm, "require_openrouter_key", lambda: "test-key")

    local_llm.call_openrouter_llm("prompt")

    assert _FakeOpenAI.last_kwargs["timeout"] == config.LOCAL_LLM_TIMEOUT_SECONDS


class _FakeVisionCompletions:
    last_kwargs = None

    def create(self, **kwargs):
        type(self).last_kwargs = kwargs
        return _FakeResponse()


class _FakeVisionChat:
    completions = _FakeVisionCompletions()


class _FakeVisionOpenAI:
    def __init__(self, **kwargs):
        self.chat = _FakeVisionChat()


def test_call_openai_vision_llm_sends_text_and_image_content(tmp_path, monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", _FakeVisionOpenAI)
    monkeypatch.setattr(local_llm, "require_openai_key", lambda: "test-key")

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0fakejpegbytes")

    result = local_llm.call_openai_vision_llm("describe this", [str(image_path)])

    assert result == "ok"
    kwargs = _FakeVisionCompletions.last_kwargs
    assert kwargs["model"] == config.OPENAI_MODEL
    content = kwargs["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_call_openai_vision_llm_one_image_block_per_path(tmp_path, monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", _FakeVisionOpenAI)
    monkeypatch.setattr(local_llm, "require_openai_key", lambda: "test-key")

    p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    p1.write_bytes(b"one")
    p2.write_bytes(b"two")

    local_llm.call_openai_vision_llm("prompt", [str(p1), str(p2)])

    content = _FakeVisionCompletions.last_kwargs["messages"][0]["content"]
    assert len(content) == 3  # 1 text block + 2 image blocks
