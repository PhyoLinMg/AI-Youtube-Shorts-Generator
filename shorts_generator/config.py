import os

from dotenv import load_dotenv

load_dotenv()

MUAPI_API_KEY = os.getenv("MUAPI_API_KEY", "").strip()
MUAPI_BASE_URL = os.getenv("MUAPI_BASE_URL", "https://api.muapi.ai/api/v1").rstrip("/")

POLL_INTERVAL_SECONDS = float(os.getenv("MUAPI_POLL_INTERVAL", "5"))
POLL_TIMEOUT_SECONDS = float(os.getenv("MUAPI_POLL_TIMEOUT", "600"))

# Local-mode (--mode local) settings — only consulted when running offline.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
# Separate from OPENROUTER_MODEL: text ranking can run on a cheap non-vision
# model (e.g. deepseek), but visual-hook scoring needs a vision-capable one.
OPENROUTER_VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
# OpenAI/OpenRouter SDK default (600s read x up to 3 tries) lets a single
# stalled provider-side request block the whole chunk loop for the better
# part of an hour with zero output. Cap it so a stall fails fast and the
# SDK's own retry kicks in instead of hanging silently.
LOCAL_LLM_TIMEOUT_SECONDS = float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "180"))
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "base")
LOCAL_WHISPER_DEVICE = os.getenv("LOCAL_WHISPER_DEVICE", "auto")  # auto / cpu / cuda

# YouTube increasingly 429s / bot-checks unauthenticated yt-dlp requests.
# Set to a browser name ("chrome", "firefox", ...) to pass
# --cookies-from-browser through every yt-dlp call; empty (default) skips
# auth entirely. Decrypting a browser's cookie store requires OS keychain
# access, which isn't available in a sandboxed/headless shell -- only set
# this when running somewhere with real Keychain/DBus access.
YT_DLP_COOKIES_FROM_BROWSER = os.getenv("YT_DLP_COOKIES_FROM_BROWSER", "").strip()

# Per-clip work (crop_clip API call + download + caption/hook-card burn, or
# the local cut/reframe/caption pipeline) is independent per highlight, so
# it fans out across a thread pool instead of running one clip at a time.
# api mode is network/poll-wait dominated -> higher default parallelism is
# safe. local mode's reframe pass is CPU-bound (OpenCV, single-threaded per
# clip), so a large pool just oversubscribes cores -> lower default.
CROP_PARALLELISM = int(os.getenv("CROP_PARALLELISM", "4"))
CROP_PARALLELISM_LOCAL = int(os.getenv("CROP_PARALLELISM_LOCAL", "2"))
VISUAL_HOOK_PARALLELISM = int(os.getenv("VISUAL_HOOK_PARALLELISM", "4"))
LOCAL_OUTPUT_DIR = os.getenv("LOCAL_OUTPUT_DIR", "output")
# "specific" -> slugified highlight title (e.g. my_big_moment.mp4)
# "generic"  -> positional (video1.mp4, video2.mp4, ...)
SHORT_FILENAME_STYLE = os.getenv("SHORT_FILENAME_STYLE", "specific").strip().lower()

# Thread-compilation narration (--clip-type thread, local mode only).
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
# "George - Warm, Captivating Storyteller" -- validated by hand for narrator
# tone; override via env if a different channel voice is wanted later.
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb").strip()

# VAD (Voice Activity Detection) settings for faster-whisper
# Default threshold is 0.5; lower = more sensitive, higher = less sensitive
# Default min_speech_duration_ms is 250ms; increase to avoid tiny false positives
# Default min_silence_duration_ms is 2000ms; increase to avoid splitting mid-sentence
# DISABLED by default because VAD is too aggressive on mixed speech/music content
LOCAL_WHISPER_VAD_FILTER = os.getenv("LOCAL_WHISPER_VAD_FILTER", "false").strip().lower() == "true"
_vad_params_env = os.getenv("LOCAL_WHISPER_VAD_PARAMETERS", "")
if _vad_params_env:
    import json
    LOCAL_WHISPER_VAD_PARAMETERS = json.loads(_vad_params_env)
else:
    # Match faster-whisper defaults when VAD is enabled
    LOCAL_WHISPER_VAD_PARAMETERS = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": float("inf"),
        "min_silence_duration_ms": 2000,
        "speech_pad_ms": 400,
    }


def require_api_key() -> str:
    if not MUAPI_API_KEY:
        raise RuntimeError(
            "MUAPI_API_KEY is not set. Add it to your .env file or export it as an env var."
        )
    return MUAPI_API_KEY


def require_openai_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Local mode needs an OpenAI key for highlight ranking. "
            "Add it to your .env or export it, or switch back to --mode api."
        )
    return OPENAI_API_KEY


def require_gemini_key() -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Local mode needs a Gemini key when LLM_PROVIDER=gemini. "
            "Add it to your .env or export it, or switch LLM_PROVIDER back to openai."
        )
    return GEMINI_API_KEY


def require_openrouter_key() -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Local mode needs an OpenRouter key when "
            "LLM_PROVIDER=openrouter. Add it to your .env or export it."
        )
    return OPENROUTER_API_KEY


def require_elevenlabs_key() -> str:
    if not ELEVENLABS_API_KEY:
        raise RuntimeError(
            "ELEVENLABS_API_KEY is not set. Thread narration needs an ElevenLabs "
            "key. Add it to your .env file or export it as an env var."
        )
    return ELEVENLABS_API_KEY
