"""Local LLM backend — OpenAI, Gemini, or OpenRouter, selected by LLM_PROVIDER."""
from ..config import (
    GEMINI_MODEL,
    LLM_PROVIDER,
    OPENAI_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    require_gemini_key,
    require_openai_key,
    require_openrouter_key,
)


def call_openai_llm(prompt: str) -> str:
    """OpenAI Chat Completions backend used by --mode local."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key=require_openai_key())
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_gemini_llm(prompt: str) -> str:
    """Gemini backend used by --mode local when LLM_PROVIDER=gemini."""
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-genai is required for LLM_PROVIDER=gemini. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = genai.Client(api_key=require_gemini_key())
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "max_output_tokens": 16384,
            # Flash models spend output-token budget on invisible "thinking" by
            # default; on dense chunks that can eat the whole budget and leave
            # zero tokens for the actual JSON answer. Turn it off so the full
            # budget goes to the response.
            "thinking_config": {"thinking_budget": 0},
        },
    )
    return response.text or ""


def call_openrouter_llm(prompt: str) -> str:
    """OpenRouter backend used by --mode local when LLM_PROVIDER=openrouter.

    OpenRouter exposes an OpenAI-compatible Chat Completions API, so this
    reuses the `openai` SDK pointed at OpenRouter's base URL.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for LLM_PROVIDER=openrouter (used as an OpenAI-"
            "compatible client). Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(
        api_key=require_openrouter_key(),
        base_url=OPENROUTER_BASE_URL,
        # OpenRouter requires/expects these to identify the app, especially
        # for routing and rate-limiting free-tier models.
        default_headers={
            "HTTP-Referer": "https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator",
            "X-Title": "AI YouTube Shorts Generator",
        },
    )
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        temperature=0.2,
        # Not all OpenRouter-routed models support response_format=json_object
        # (some only support json_schema, some none at all) — the prompts
        # already demand JSON-only output and _parse_json_loose strips fences.
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def call_local_llm(prompt: str) -> str:
    """Dispatch to the configured local LLM provider."""
    provider = (LLM_PROVIDER or "openai").strip().lower()
    if provider == "openai":
        return call_openai_llm(prompt)
    if provider == "gemini":
        return call_gemini_llm(prompt)
    if provider == "openrouter":
        return call_openrouter_llm(prompt)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'openai', 'gemini', or 'openrouter'."
    )
