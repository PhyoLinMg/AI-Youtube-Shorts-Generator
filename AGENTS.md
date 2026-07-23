# AI YouTube Shorts Generator

## Entrypoints
- **CLI**: `python main.py <URL> [--mode api|local] [--num-clips N] [--aspect-ratio 9:16] [--no-captions] [--framing locked|adaptive]`
- **Python**: `from shorts_generator import generate_shorts`
- **Dashboard**: `python dashboard.py` (needs `pip install -r requirements-web.txt`)

## Two modes
- `--mode api` (default) — MuAPI for download/transcribe/LLM/crop. Requires `MUAPI_API_KEY` in `.env`.
- `--mode local` — yt-dlp + faster-whisper + LLM provider + ffmpeg/OpenCV. Requires `LLM_PROVIDER` (openai|gemini|openrouter) + matching API key.

## Requirements files
- `requirements.txt` — core (api mode only)
- `requirements-local.txt` — local mode extras (yt-dlp, faster-whisper, openai, google-genai, ffmpeg-python, opencv-python)
- `requirements-web.txt` — dashboard (Flask)
- `requirements-dev.txt` — testing (pytest)

## Commands
- **Test**: `pip install -r requirements-dev.txt && pytest`
- **Install all**: `pip install -r requirements.txt -r requirements-local.txt -r requirements-web.txt`

## Pipeline (in order)
download → transcribe → classify content type → chunk if >30min → rank highlights (virality framework, 8 criteria) → dedupe (>50% overlap→keep higher score) → top-N → vertical crop

## Gotchas
- Captions ON by default in both modes; `--no-captions` to disable. Caption burn-in requires ffmpeg on PATH.
- `--framing adaptive` is local-mode only. `--mode api` always uses MuAPI autocrop.
- Caching: rerun skips download + transcription if cached files exist. Highlight ranking + crop always re-run.
- Each run writes `output/<Title>/Shorts/Short-NN.mp4` + `result.json` + `progress.log`.
- LLM provider also supports OpenRouter (`LLM_PROVIDER=openrouter` + `OPENROUTER_API_KEY`).
- Gemini Flash models need `thinking_config: {"thinking_budget": 0}` (already in code).
- MuAPI key, API keys, and model names come from `.env` (see `.env.example`).

## Skill reference
Detailed pipeline execution steps in `.claude/skills/youtube-shorts-generator/SKILL.md`.

## Git remote policy
- `origin` = `SamurAIGPT/AI-Youtube-Shorts-Generator` (upstream). NEVER push here.
- `fork` = `PhyoLinMg/AI-Youtube-Shorts-Generator` (your fork). ONLY push here.
- `main` tracks `fork/main`. After merging upstream PRs, pull from `fork/main`, then push to `fork main` only.
- Rule: never `git push origin`. Only `git push fork`.
