"""Local LLM backend — OpenAI, Gemini, or OpenRouter, selected by LLM_PROVIDER."""
import base64
from typing import List

from ..config import (
    GEMINI_MODEL,
    LLM_PROVIDER,
    LOCAL_LLM_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_VISION_MODEL,
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

    client = OpenAI(api_key=require_openai_key(), timeout=LOCAL_LLM_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _build_vision_content(prompt: str, image_paths: List[str]) -> list:
    content = [{"type": "text", "text": prompt}]
    for path in image_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return content


def call_openai_vision_llm(prompt: str, image_paths: List[str]) -> str:
    """OpenAI vision backend for visual_hook.score_visual_hooks, used by
    --mode local when LLM_PROVIDER=openai (the default). A missing/invalid
    OPENAI_API_KEY here just means score_visual_hooks degrades that
    highlight to "no visual hook score," it doesn't fail the run, per
    score_visual_hooks's per-highlight try/except."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for visual-hook scoring. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    client = OpenAI(api_key=require_openai_key(), timeout=LOCAL_LLM_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[{"role": "user", "content": _build_vision_content(prompt, image_paths)}],
    )
    return response.choices[0].message.content or ""


def call_openrouter_vision_llm(prompt: str, image_paths: List[str]) -> str:
    """OpenRouter vision backend for visual_hook.score_visual_hooks, used by
    --mode local when LLM_PROVIDER=openrouter. Uses OPENROUTER_VISION_MODEL
    (default openai/gpt-4o-mini) rather than OPENROUTER_MODEL, since the
    text model picked for highlight ranking (e.g. a cheap deepseek model)
    may not be vision-capable."""
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
        timeout=LOCAL_LLM_TIMEOUT_SECONDS,
        default_headers={
            "HTTP-Referer": "https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator",
            "X-Title": "AI YouTube Shorts Generator",
        },
    )
    response = client.chat.completions.create(
        model=OPENROUTER_VISION_MODEL,
        temperature=0.2,
        messages=[{"role": "user", "content": _build_vision_content(prompt, image_paths)}],
    )
    return response.choices[0].message.content or ""


def call_local_vision_llm(prompt: str, image_paths: List[str]) -> str:
    """Dispatch to the configured local vision LLM provider, mirroring
    call_local_llm's LLM_PROVIDER dispatch. Gemini has no vision backend
    implemented yet, so it falls back to OpenAI vision when OPENAI_API_KEY
    is set (matching pre-OpenRouter behavior, where local mode always used
    OpenAI for vision regardless of LLM_PROVIDER); otherwise it raises and
    score_visual_hooks's per-highlight try/except degrades that to "no
    visual hook score" rather than failing the run."""
    provider = (LLM_PROVIDER or "openai").strip().lower()
    if provider == "openai":
        return call_openai_vision_llm(prompt, image_paths)
    if provider == "openrouter":
        return call_openrouter_vision_llm(prompt, image_paths)
    if provider == "gemini" and OPENAI_API_KEY:
        return call_openai_vision_llm(prompt, image_paths)
    raise RuntimeError(
        f"No vision backend implemented for LLM_PROVIDER={provider!r}. "
        "Use 'openai' or 'openrouter', or set OPENAI_API_KEY for visual-hook scoring."
    )


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
        timeout=LOCAL_LLM_TIMEOUT_SECONDS,
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


def describe_local_llm_models() -> dict:
    """Report which text/vision models --mode local will actually call, given
    the current LLM_PROVIDER. Gemini has no vision backend, so its vision
    slot is None rather than a model name."""
    provider = (LLM_PROVIDER or "openai").strip().lower()
    if provider == "openai":
        return {"provider": provider, "text_model": OPENAI_MODEL, "vision_model": OPENAI_MODEL}
    if provider == "gemini":
        return {"provider": provider, "text_model": GEMINI_MODEL, "vision_model": None}
    if provider == "openrouter":
        return {"provider": provider, "text_model": OPENROUTER_MODEL, "vision_model": OPENROUTER_VISION_MODEL}
    return {"provider": provider, "text_model": None, "vision_model": None}


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
