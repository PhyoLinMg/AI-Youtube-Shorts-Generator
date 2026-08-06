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
    """Captures the kwargs it's constructed with, like the real OpenAI client."""

    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.chat = _FakeVisionChat()


def test_call_openai_vision_llm_sends_text_and_image_content(tmp_path, monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", _FakeVisionOpenAI)
    monkeypatch.setattr(local_llm, "require_openai_key", lambda: "test-key")

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0fakejpegbytes")

    result = local_llm.call_openai_vision_llm("describe this", [str(image_path)])

    assert result == "ok"
    assert _FakeVisionOpenAI.last_kwargs["timeout"] == config.LOCAL_LLM_TIMEOUT_SECONDS
    assert _FakeVisionOpenAI.last_kwargs["api_key"] == "test-key"
    kwargs = _FakeVisionCompletions.last_kwargs
    assert kwargs["model"] == config.OPENAI_MODEL
    assert kwargs["temperature"] == 0.2
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


def test_call_openrouter_vision_llm_uses_vision_model_and_base_url(tmp_path, monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", _FakeVisionOpenAI)
    monkeypatch.setattr(local_llm, "require_openrouter_key", lambda: "test-key")
    monkeypatch.setattr(local_llm, "OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini")

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0fakejpegbytes")

    result = local_llm.call_openrouter_vision_llm("describe this", [str(image_path)])

    assert result == "ok"
    assert _FakeVisionOpenAI.last_kwargs["base_url"] == config.OPENROUTER_BASE_URL
    kwargs = _FakeVisionCompletions.last_kwargs
    assert kwargs["model"] == "openai/gpt-4o-mini"
    content = kwargs["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe this"}
    assert content[1]["type"] == "image_url"


def test_call_local_vision_llm_dispatches_openai(monkeypatch):
    monkeypatch.setattr(local_llm, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(local_llm, "call_openai_vision_llm", lambda p, imgs: "openai-result")

    assert local_llm.call_local_vision_llm("prompt", ["a.jpg"]) == "openai-result"


def test_call_local_vision_llm_dispatches_openrouter(monkeypatch):
    monkeypatch.setattr(local_llm, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(local_llm, "call_openrouter_vision_llm", lambda p, imgs: "openrouter-result")

    assert local_llm.call_local_vision_llm("prompt", ["a.jpg"]) == "openrouter-result"


def test_call_local_vision_llm_raises_on_gemini_without_openai_key(monkeypatch):
    monkeypatch.setattr(local_llm, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(local_llm, "OPENAI_API_KEY", "")

    try:
        local_llm.call_local_vision_llm("prompt", ["a.jpg"])
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "gemini" in str(e)


def test_call_local_vision_llm_gemini_falls_back_to_openai_when_key_present(monkeypatch):
    monkeypatch.setattr(local_llm, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(local_llm, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(local_llm, "call_openai_vision_llm", lambda p, imgs: "openai-fallback-result")

    assert local_llm.call_local_vision_llm("prompt", ["a.jpg"]) == "openai-fallback-result"
