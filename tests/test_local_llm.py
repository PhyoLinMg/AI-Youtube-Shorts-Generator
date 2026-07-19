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
