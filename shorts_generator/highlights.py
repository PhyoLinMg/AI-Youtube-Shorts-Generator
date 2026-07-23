"""Find the most viral-worthy highlights in a transcript.

Logic ported from ViralVadoo's transcript_analysis/highlight_generator.py:
  - content-type / density detection
  - chunking for long videos with overlap
  - virality-criteria prompt
  - score-based dedupe with overlap suppression

The LLM call is pluggable via the `llm_fn` argument so the same prompts can
drive either MuAPI (default, --mode api) or a direct local LLM client
(--mode local).
"""
import hashlib
import json
import os
import re
import time
from typing import Callable, Dict, List, Optional

from . import muapi


LLMFn = Callable[[str], str]

# Generic spam tags that hurt Shorts reach — the model is told to avoid these,
# but we drop them defensively so they never reach YouTube.
_GENERIC_SPAM_TAGS = {
    "#fyp", "#foryou", "#foryoupage", "#viral", "#viralvideo",
    "#trending", "#trendingshorts", "#shortsfeed", "#ytshorts",
    "#youtube", "#fypシ",
}


def _transcript_fingerprint(transcript: Dict) -> str:
    """Stable content hash used to invalidate the highlights cache when the
    transcript actually changes — independent of *how* it was obtained
    (freshly transcribed vs. read from either transcriber's own cache)."""
    payload = json.dumps(
        {"duration": transcript.get("duration"), "segments": transcript.get("segments")},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


CONTENT_TYPE_PROMPT = """Analyze this video transcript sample and classify the content type.
Choose one: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
Also estimate content density: low (mostly filler/chit-chat), medium, or high (dense info/stories).
Respond with JSON only: {"content_type": "...", "density": "..."}"""


VIRALITY_CRITERIA = """
Virality signals to prioritize (ranked by impact):
1. HOOK MOMENTS — statements that create immediate curiosity ("The secret is...", "Nobody talks about...", "I was completely wrong about...")
2. EMOTIONAL PEAKS — genuine surprise, laughter, anger, vulnerability, excitement; raw unscripted reactions
3. OPINION BOMBS — strong, polarizing or counter-intuitive statements that trigger agree/disagree
4. REVELATION MOMENTS — surprising facts, stats, or confessions that reframe how the viewer thinks
5. CONFLICT/TENSION — disagreement, pushback, or a problem being confronted head-on
6. QUOTABLE ONE-LINERS — a sentence that works as a standalone quote card
7. STORY PEAKS — the climax or twist of an anecdote; the payoff moment
8. PRACTICAL VALUE — a concrete tip, hack, or insight the viewer can immediately apply
"""


HOOK_STRENGTH_RUBRIC = """
Hook-landing speed (the ~1-second swipe-decision window):
- High (90+): the first spoken line states a COMPLETE, surprising or contrarian
  idea that stands alone with zero prior context — e.g. "If I made the universe,
  I wouldn't need your praise." A viewer hearing only the first ~1 second grasps
  a full claim.
- Low (<40): the opener is a thesis that requires processing — holding two
  concepts, or setup before the point lands — e.g. "The Ultimate Irony:
  Capitalism Will Create the Communist Utopia." Also low: windup/throat-clearing
  ("So the thing is...", "What people don't realize...") and curiosity-gap
  teases ("You won't believe what happened").
- Reward: a declarative self-contained contrarian statement, a named-authority
  factual claim, or a complete vivid image.
- Penalize: dependency on prior context, multi-step reasoning, or a
  promise-of-payoff-later.
"""


HIGHLIGHT_SYSTEM_PROMPT = """You are an elite short-form video editor who has studied thousands of viral clips on TikTok, Instagram Reels, and YouTube Shorts. You know exactly what makes viewers stop scrolling, watch to the end, and share.

{virality_criteria}

Content type: {content_type} | Density: {density}

Your task: identify the most viral-worthy highlights from the transcript.

{hook_strength_rubric}

Rules:
- Every highlight must open with a strong HOOK — a line that grabs attention within the first 3 seconds. start_time must land ON that hook line itself, never on preamble, silence, or filler before it — the clip opens cold, mid-energy, not with a slow windup
- Duration sweet spot: 45-90 seconds. Go shorter (20-44s) only for a perfect standalone one-liner. Go longer (91-180s) only when a story arc needs full context to land
- Never cut mid-sentence or mid-thought — each clip must feel complete and self-contained
- Clips must not overlap significantly with each other
- Score 0-100 on viral potential (not general quality)
- {num_clips_instruction}
- For each highlight, identify the single best "hook_sentence" — the opening line that would make someone stop scrolling
- Write an "on_screen_hook" — a short punchy fragment, 7 words or fewer, distinct from hook_sentence (it does NOT need to be a verbatim transcript line). This is bold text that gets overlaid on screen for the first 1.5 seconds, so it must work standalone with zero context. It must STATE the complete claim or payoff itself — never tease it. Bad (curiosity-gap teases, do NOT do this): "You won't believe what happened", "The truth about X", "Wait for it...". Good: a full contrarian statement or the punchline itself, e.g. "I'd rather be wrong than boring." Think thumbnail text that gives away the point, not a cliffhanger.
- Explain in one sentence why this clip is viral ("virality_reason")
- Score "hook_strength" 0-100 on how completely and immediately the opening line lands within the ~1-second swipe-decision window, per the hook-landing rubric above (this is independent from the overall viral "score")
- Set "hook_self_contained" (true/false) — true only if the opener needs zero prior context to land
- Write a "hook_reason" — one sentence on why the opener lands or fails within that first second (this is a human-review note, not shown to viewers)
- Write a "title" — max 100 characters, aggressive clickbait style (curiosity gap, numbers, shock value, "you won't believe", etc.) optimized to maximize clicks and views, while still being accurate to the clip's content
- Write a "description" — SHORTS-optimized, max 220 characters, original copy (NOT a transcript line) built to maximize views from BOTH the Shorts feed and YouTube search: line 1 is a hook line (<=150 chars, the only line most viewers see before "more"); line 2 works your primary keyword + 1-2 related terms in naturally (no keyword lists); line 3 is a short punchy CTA that drives session watch time — prefer a specific next-action tied to this content ("watch part 2" / "full video on my channel" / a pointed comment prompt) over a generic "follow/subscribe" ask. Do NOT include any emojis.
- Write a "yt_title" — max 60 characters, a sharp Shorts title: Result/Hook + specific topic + for [audience]. Accurate to the clip, no emojis.
- Write "yt_hashtags" — a JSON array of exactly 2-3 highly relevant NICHE hashtags (lowercase, leading #, no spaces). Always include "#Shorts" plus 1-2 topic-specific tags. Do NOT use generic spam tags (#fyp, #viral, #trending).

Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","on_screen_hook":"string","virality_reason":"string","hook_strength":int,"hook_self_contained":bool,"hook_reason":"string","description":"string","yt_title":"string","yt_hashtags":["#Shorts","#topic1","#topic2"]}}]}}"""


CHUNK_SIZE_SECONDS = 1200       # 20-min chunks for long videos
LONG_VIDEO_THRESHOLD = 1800     # chunk videos longer than 30 min
CHUNK_OVERLAP_SECONDS = 60
GPT_CALL_TIMEOUT_SECONDS = 300  # cap LLM polls at 5 min — a wedged call should fail fast
MAX_HIGHLIGHT_API_ATTEMPTS = 3
HIGHLIGHT_SCHEMA_VERSION = 2    # bump whenever the highlight dict shape changes,
                                # so a stale on-disk cache (missing new fields)
                                # is treated as a miss instead of silently reused


def call_muapi_llm(prompt: str) -> str:
    """Default LLM backend: MuAPI gpt-5-mini."""
    result = muapi.run(
        "gpt-5-mini",
        {"prompt": prompt},
        label="gpt-5-mini",
        timeout=GPT_CALL_TIMEOUT_SECONDS,
    )

    outputs = result.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], str) and outputs[0].strip():
        return outputs[0]

    for key in ("output", "text", "response", "result", "content"):
        v = result.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, dict):
            inner = v.get("text") or v.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0]

    raise RuntimeError(f"Could not extract gpt-5-mini text from response: {result}")


def _parse_json_loose(raw: str) -> Dict:
    """gpt-5-4 sometimes wraps JSON in markdown fences — strip and parse."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        raise


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes")
    return default


def _sanitize_hashtags(raw_tags: object) -> List[str]:
    """Keep 2-3 niche hashtags; always include #Shorts; drop generic spam."""
    hashtags: List[str] = []
    if isinstance(raw_tags, list):
        for t in raw_tags:
            if not isinstance(t, str):
                continue
            tag = t.strip().lower()
            if tag and not tag.startswith("#"):
                tag = "#" + tag
            if not tag or tag in _GENERIC_SPAM_TAGS:
                continue
            if tag not in hashtags and len(hashtags) < 4:
                hashtags.append(tag)
    if "#shorts" not in [h.lower() for h in hashtags]:
        hashtags.insert(0, "#Shorts")
    return hashtags[:4]


def _sanitize_highlights(raw_highlights: object, duration: float) -> List[Dict]:
    """Normalize model output into the expected shape; skip invalid entries."""
    if not isinstance(raw_highlights, list):
        return []

    max_end = duration if duration > 0 else float("inf")
    cleaned: List[Dict] = []
    for item in raw_highlights:
        if not isinstance(item, dict):
            continue

        start = _coerce_float(item.get("start_time"), default=-1.0)
        end = _coerce_float(item.get("end_time"), default=-1.0)
        if start < 0 or end <= start:
            continue

        if max_end != float("inf"):
            start = min(start, max_end)
            end = min(end, max_end)
            if end <= start:
                continue

        cleaned.append(
            {
                "title": str(item.get("title") or "Untitled Highlight").strip()[:100],
                "start_time": start,
                "end_time": end,
                "score": max(0, min(100, _coerce_int(item.get("score"), default=0))),
                "hook_sentence": str(item.get("hook_sentence") or "").strip(),
                "on_screen_hook": str(item.get("on_screen_hook") or "").strip()[:60],
                "virality_reason": str(item.get("virality_reason") or "").strip(),
                "hook_strength": max(0, min(100, _coerce_int(item.get("hook_strength"), default=0))),
                "hook_self_contained": _coerce_bool(item.get("hook_self_contained"), default=False),
                "hook_reason": str(item.get("hook_reason") or "").strip(),
                "description": str(item.get("description") or "").strip()[:220],
                "yt_title": str(item.get("yt_title") or "").strip()[:60],
                "yt_hashtags": _sanitize_hashtags(item.get("yt_hashtags")),
            }
        )

    return cleaned


def detect_content_type(transcript: Dict, llm_fn: LLMFn = call_muapi_llm) -> Dict[str, str]:
    segments = transcript.get("segments", [])
    sample = " ".join(s["text"] for s in segments[:25])[:3000]
    prompt = f"{CONTENT_TYPE_PROMPT}\n\nTranscript sample:\n{sample}"
    try:
        raw = llm_fn(prompt)
        return _parse_json_loose(raw)
    except Exception:
        return {"content_type": "other", "density": "medium"}


def build_transcript_text(transcript: Dict) -> str:
    segments = transcript.get("segments", [])
    return "\n".join(f"[{s['start']:.1f}s] {s['text'].strip()}" for s in segments)


def chunk_transcript(transcript: Dict) -> List[Dict]:
    segments = transcript.get("segments", [])
    duration = transcript.get("duration", segments[-1]["end"] if segments else 0)
    chunks = []
    start = 0
    while start < duration:
        end = min(start + CHUNK_SIZE_SECONDS, duration)
        chunk_segs = [
            s for s in segments
            if s["start"] >= start and s["end"] <= end + CHUNK_OVERLAP_SECONDS
        ]
        if chunk_segs:
            chunk = dict(transcript)
            chunk["segments"] = chunk_segs
            chunk["duration"] = end - start
            chunk["_offset"] = start
            chunks.append(chunk)
        start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
    return chunks


def call_highlight_api(
    transcript_text: str,
    content_info: Dict,
    duration: float,
    num_clips: int,
    is_chunk: bool = False,
    llm_fn: LLMFn = call_muapi_llm,
) -> Dict:
    # Ask for ~2× the user's target so dedupe has headroom, but cap so the model
    # doesn't have to generate a huge JSON payload (which times out gpt-5-mini).
    target = max(num_clips * 2, 5)
    natural_max = max(2 if is_chunk else 3, int(duration / 60))
    min_clips = min(target, natural_max, 14)
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        hook_strength_rubric=HOOK_STRENGTH_RUBRIC,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        num_clips_instruction=f"Generate at least {min_clips} highlights",
    )
    base_prompt = f"{system}\n\nTranscript:\n{transcript_text}"
    prompt = base_prompt
    last_error = "unknown"

    for attempt in range(1, MAX_HIGHLIGHT_API_ATTEMPTS + 1):
        try:
            raw = llm_fn(prompt)
            parsed = _parse_json_loose(raw)
            highlights = _sanitize_highlights(parsed.get("highlights"), duration=duration)
            if highlights:
                return {"highlights": highlights}
            last_error = "no valid highlights in response"
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_HIGHLIGHT_API_ATTEMPTS:
            print(
                f"[highlights] attempt {attempt}/{MAX_HIGHLIGHT_API_ATTEMPTS} failed ({last_error}); retrying",
                flush=True,
            )
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Return ONLY valid JSON with a top-level 'highlights' array."
                + " Each item must include: title, start_time, end_time, score, hook_sentence, on_screen_hook, virality_reason, hook_strength, hook_self_contained, hook_reason, description, yt_title, yt_hashtags."
                + " No markdown fences, no commentary."
            )

    raise RuntimeError(
        f"Highlight generator produced invalid output after {MAX_HIGHLIGHT_API_ATTEMPTS} attempts: {last_error}"
    )


def dedupe_highlights(highlights: List[Dict]) -> List[Dict]:
    """Drop a highlight if it overlaps >50% with a higher-scoring one already kept."""
    highlights = sorted(highlights, key=lambda x: int(x.get("score", 0)), reverse=True)
    kept: List[Dict] = []
    for h in highlights:
        h_start = float(h["start_time"])
        h_end = float(h["end_time"])
        h_dur = h_end - h_start
        overlapping = False
        for k in kept:
            latest_start = max(h_start, float(k["start_time"]))
            earliest_end = min(h_end, float(k["end_time"]))
            overlap = earliest_end - latest_start
            if overlap > 0 and overlap > 0.5 * h_dur:
                overlapping = True
                break
        if not overlapping:
            kept.append(h)
    return kept


def get_highlights(
    transcript: Dict,
    num_clips: int = 3,
    llm_fn: Optional[LLMFn] = None,
) -> Dict:
    """Main entry point — returns {highlights: [...]} sorted by score.

    `llm_fn` swaps the underlying LLM. Defaults to MuAPI gpt-5-mini; local
    mode passes in a local LLM-backed callable.
    """
    llm_fn = llm_fn or call_muapi_llm
    duration = transcript.get("duration", 0)
    content_info = detect_content_type(transcript, llm_fn=llm_fn)
    print(f"[highlights] content={content_info.get('content_type')} density={content_info.get('density')} duration={duration:.0f}s", flush=True)

    if duration >= LONG_VIDEO_THRESHOLD:
        chunks = chunk_transcript(transcript)
        print(f"[highlights] long video — splitting into {len(chunks)} chunks", flush=True)
        all_highlights: List[Dict] = []
        for i, chunk in enumerate(chunks):
            offset = chunk.get("_offset", 0)
            text = build_transcript_text(chunk)
            print(f"[highlights] chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)", flush=True)
            # build_transcript_text labels each line with the segment's absolute
            # timestamp, so the model replies in absolute time too — clamp against
            # the chunk's absolute end (not its relative length) and don't re-offset.
            chunk_abs_end = offset + chunk["duration"]
            t0 = time.time()
            result = call_highlight_api(text, content_info, chunk_abs_end, num_clips=num_clips, is_chunk=True, llm_fn=llm_fn)
            print(f"[highlights] chunk {i + 1}/{len(chunks)} done in {time.time() - t0:.1f}s", flush=True)
            all_highlights.extend(result.get("highlights", []))
        highlights = dedupe_highlights(all_highlights)
    else:
        text = build_transcript_text(transcript)
        result = call_highlight_api(text, content_info, duration, num_clips=num_clips, llm_fn=llm_fn)
        highlights = dedupe_highlights(result.get("highlights", []))

    return {"highlights": highlights}


def get_highlights_cached(
    transcript: Dict,
    num_clips: int,
    cache_path: str,
    llm_fn: Optional[LLMFn] = None,
) -> Dict:
    """Wraps get_highlights with an on-disk cache keyed by a transcript
    content fingerprint + num_clips, so rerunning the pipeline on a video
    whose transcript hasn't changed skips the LLM call(s) entirely.

    A fingerprint mismatch, num_clips mismatch, or unparseable cache file
    all fall back to a full recompute (which then overwrites the cache) —
    no partial reuse.
    """
    fingerprint = _transcript_fingerprint(transcript)

    if os.path.exists(cache_path):
        cached = None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except json.JSONDecodeError:
            print(f"[highlights] cached highlights corrupted, recomputing: {cache_path}", flush=True)

        if (
            isinstance(cached, dict)
            and cached.get("transcript_fingerprint") == fingerprint
            and cached.get("num_clips") == num_clips
            and cached.get("schema_version") == HIGHLIGHT_SCHEMA_VERSION
            and isinstance(cached.get("highlights"), list)
        ):
            print(f"[highlights] reusing cached highlights: {cache_path}", flush=True)
            return {"highlights": cached["highlights"]}

    result = get_highlights(transcript, num_clips=num_clips, llm_fn=llm_fn or call_muapi_llm)

    tmp_path = cache_path + ".part"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "transcript_fingerprint": fingerprint,
                "num_clips": num_clips,
                "schema_version": HIGHLIGHT_SCHEMA_VERSION,
                "highlights": result.get("highlights", []),
            },
            f,
            ensure_ascii=False,
        )
    os.replace(tmp_path, cache_path)

    return result
