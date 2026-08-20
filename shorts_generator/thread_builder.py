"""Two-stage picker for thread compilation: first a multi-question same-topic
gate over a FIXED pair of episodes (see docs/superpowers/specs/2026-08-09-
thread-compilation-design.md for the hard same-topic requirement, and
2026-08-10-thread-two-url-multi-clip-design.md for why the pair is fixed by
the caller instead of scanned from the whole corpus), then, for each
qualifying shared question, exact clip spans + narration text grounded in
the two chosen full transcripts.
"""
import json
import math
from typing import Dict, List, Optional, Tuple

from .highlights import LLMFn, _parse_json_loose, build_transcript_text, call_muapi_llm

SAME_TOPIC_SYSTEM_PROMPT = """You are a strict fact-checker deciding whether two podcast episodes can be combined into a single short video built around ONE shared question.

You will be given a numbered list of episode abstracts. Your ONLY job is to find, if one exists, exactly one pair of episodes that:
1. Discuss the SAME specific topic (not just an adjacent or loosely related one)
2. Each independently make a claim, argument, or statement that answers or responds to the SAME underlying question -- not just the same subject area

Rules -- read carefully, these are hard requirements, not preferences:
- Sharing a broad subject (e.g. both mention "aliens", both mention "the economy") is NOT enough. Two episodes both mentioning AI is not a match; two episodes both making a claim about whether AI will take creative jobs IS a match.
- If you are not certain both episodes answer the exact same question, you MUST return no_match: true. A weak or "kind of related" pair is a wrong answer, not a partial credit answer.
- Do not invent a connection. Only report a pair if the abstracts themselves make the shared question obvious.
- State the shared question explicitly, in one sentence, phrased as a real question a viewer would recognize both clips are answering.

Episodes:
{abstracts_block}

Respond ONLY with valid JSON (no markdown, no explanation):
{{"no_match": bool, "episode_a_index": int or null, "episode_b_index": int or null, "shared_question": "string or empty"}}"""

SAME_TOPIC_MULTI_PROMPT = """You are a strict fact-checker deciding whether two podcast episodes share any genuine same-question topics that could each become a short video.

You will be given the abstracts for exactly two episodes, A and B. Find EVERY genuinely distinct shared question where:
1. Both episodes discuss the SAME specific topic (not just an adjacent or loosely related one)
2. Each episode independently makes a claim, argument, or statement that answers or responds to that SAME underlying question -- not just the same subject area

Rules -- read carefully, these are hard requirements, not preferences:
- Sharing a broad subject (e.g. both mention "aliens", both mention "the economy") is NOT enough. Two episodes both mentioning AI is not a match; two episodes both making a claim about whether AI will take creative jobs IS a match.
- Only report a question if the abstracts themselves make the shared question obvious. Do not invent a connection.
- Each shared question must be genuinely distinct from the others you report -- do not report near-duplicate phrasings of the same question twice.
- Return at most {num_pairs} questions, ordered from strongest/most obvious match to weakest.
- If there is no genuine shared question at all, return an empty list.

Episode A: {abstract_a}

Episode B: {abstract_b}

Respond ONLY with valid JSON (no markdown, no explanation):
{{"shared_questions": ["question 1?", "question 2?"]}}"""

THREAD_PICK_SYSTEM_PROMPT = """You are editing a short video that puts two podcast guests' answers to the same question side by side.

Shared question both episodes are answering: {shared_question}
{avoid_block}
You will be given the full transcript of TWO episodes, each labeled with absolute timestamps. Find, in EACH transcript, one span where the speaker directly answers or responds to the shared question above.

Rules:
- start_time must land on the exact line where the speaker begins answering the shared question -- never on preamble, a host's question, or unrelated chat before it.
- end_time must land where that specific answer finishes -- never mid-sentence, never trailing into the next unrelated topic.
- Duration per clip: {clip_range}. If the answer runs longer, pick the single most complete self-contained span within it -- do not use extra length just because it's available.
- The whole video (thesis + clip A + bridge + clip B) must land in the {total_range} range, so both narration lines and both clips need to be tight.
- Write a "thesis" -- ONE punchy sentence (max ~12 words) a narrator would say BEFORE either clip plays. This is the scroll-stopping HOOK -- it must grab the viewer in the first 1-3 seconds. Name the shared question and hint at the tension or disagreement between the two answers, but do NOT reveal what either speaker actually said, which side they land on, or any number/percentage/statistic from either transcript. Stay mysterious -- the viewer has to watch both clips to find out. No throat-clearing, no setup, no filler words.
- Write a "bridge" -- ONE punchy sentence (max ~12 words) a narrator would say BETWEEN the two clips, pivoting from what episode A's speaker just said to what episode B's speaker is about to say -- without giving away episode B's actual answer or any number/percentage/statistic from either transcript. Same energy as the thesis: fast, sharp, zero filler, still mysterious.
- Write a "title" -- max 100 characters TOTAL, including 1-2 trailing hashtags (e.g. "#Shorts #ai"). Aggressive clickbait style (curiosity gap, "vs", stakes) that hooks a scroller without revealing the outcome or either speaker's actual position. Accurate to the shared question, never spoils the answer. Do NOT include any number, percentage, or statistic pulled from either transcript -- a number IS the spoiler.
- Write a "description" -- max 200 characters, Shorts-optimized, original copy (not a transcript line). Builds curiosity around the shared question and the fact the two guests clash or agree on it, without spoiling what either one says. Do NOT quote or paraphrase either speaker's specific claim, and do NOT include any number, percentage, statistic, or named finding from either transcript -- if you catch yourself writing a digit, cut it and rephrase around the question instead. End with a short watch-through CTA (e.g. "Watch both takes"). No emojis, no spoilers.
- If you cannot find a genuine on-topic answer in BOTH transcripts, set "grounded" to false and leave the clip/title/description fields empty -- do not force a loose match.

Episode A transcript:
{transcript_a}

Episode B transcript:
{transcript_b}

Respond ONLY with valid JSON (no markdown, no explanation):
{{"grounded": bool, "thesis": "string", "bridge": "string", "title": "string", "description": "string", "clip_a": {{"start_time": float, "end_time": float}}, "clip_b": {{"start_time": float, "end_time": float}}}}"""

# Per-platform clip-span bounds. TikTok's Creator Rewards Program only pays
# out on videos 60s+ (duets/stitches/sub-60s videos earn nothing regardless
# of views) -- min_clip=28.0 is chosen so the worst-case assembly (two
# clips at the floor + a conservative ~8s narration floor) still clears 60s
# with margin: 2*28 + 8 = 64s. max_clip=40.0 keeps the ceiling inside the
# 2026 "1-3 minute but don't pad past what completion rate can support"
# sweet spot (worst-case ceiling 2*40+14=94s). See
# docs/superpowers/specs/2026-08-20-thread-dual-platform-design.md Section 1.
# clip_range is deliberately narrower than [min_clip, max_clip] for both
# platforms -- it steers the LLM toward the middle of the accepted band so
# a slightly-off pick still clears the hard bound instead of getting
# silently dropped by _sanitize_clip_span.
PLATFORM_BOUNDS = {
    "youtube": {"min_clip": 8.0, "max_clip": 25.0, "clip_range": "12-22 seconds", "total_range": "45-60 second"},
    "tiktok": {"min_clip": 28.0, "max_clip": 40.0, "clip_range": "30-38 seconds", "total_range": "65-90 second"},
}


def _coerce_float_or_none(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _sanitize_topic_pick(raw: object, corpus_len: int) -> Optional[Dict]:
    if not isinstance(raw, dict) or raw.get("no_match"):
        return None
    a = raw.get("episode_a_index")
    b = raw.get("episode_b_index")
    if not isinstance(a, int) or not isinstance(b, int):
        return None
    if a == b or not (0 <= a < corpus_len) or not (0 <= b < corpus_len):
        return None
    raw_question = raw.get("shared_question")
    if not isinstance(raw_question, str):
        return None
    question = raw_question.strip()
    if not question:
        return None
    return {"episode_a_index": a, "episode_b_index": b, "shared_question": question}


def _sanitize_topic_picks_multi(raw: object, num_pairs: int) -> List[str]:
    if not isinstance(raw, dict):
        return []
    raw_questions = raw.get("shared_questions")
    if not isinstance(raw_questions, list):
        return []
    seen = set()
    questions = []
    for item in raw_questions:
        if not isinstance(item, str):
            continue
        question = item.strip()
        if not question or question.lower() in seen:
            continue
        seen.add(question.lower())
        questions.append(question)
        if len(questions) >= num_pairs:
            break
    return questions


def _sanitize_clip_span(raw: object, duration: float, min_clip: float, max_clip: float) -> Optional[Dict]:
    if not isinstance(raw, dict):
        return None
    start = _coerce_float_or_none(raw.get("start_time"))
    end = _coerce_float_or_none(raw.get("end_time"))
    if start is None or end is None or start < 0 or end <= start:
        return None
    if duration > 0:
        end = min(end, duration)
        if end <= start:
            return None
    span = end - start
    if span < min_clip or span > max_clip:
        return None
    return {"start_time": start, "end_time": end}


def _sanitize_clip_pick(raw: object, duration_a: float, duration_b: float, min_clip: float, max_clip: float) -> Optional[Dict]:
    if not isinstance(raw, dict) or not raw.get("grounded"):
        return None
    raw_thesis = raw.get("thesis")
    raw_bridge = raw.get("bridge")
    raw_title = raw.get("title")
    raw_description = raw.get("description")
    if not isinstance(raw_thesis, str) or not isinstance(raw_bridge, str):
        return None
    if not isinstance(raw_title, str) or not isinstance(raw_description, str):
        return None
    thesis = raw_thesis.strip()
    bridge = raw_bridge.strip()
    title = raw_title.strip()[:100]
    description = raw_description.strip()[:200]
    if not thesis or not bridge or not title or not description:
        return None
    clip_a = _sanitize_clip_span(raw.get("clip_a"), duration_a, min_clip, max_clip)
    clip_b = _sanitize_clip_span(raw.get("clip_b"), duration_b, min_clip, max_clip)
    if clip_a is None or clip_b is None:
        return None
    return {"thesis": thesis, "bridge": bridge, "title": title, "description": description, "clip_a": clip_a, "clip_b": clip_b}


def find_same_topic_pairs(entry_a: Dict, entry_b: Dict, num_pairs: int, llm_fn: LLMFn) -> List[str]:
    """Stage A, fixed-pair variant (see select_thread_pairs). Given exactly
    two corpus entries (each needing "abstract"), returns up to num_pairs
    distinct shared-question strings both abstracts genuinely answer -- []
    whenever none exist. Same hard same-topic gate as find_same_topic_pair,
    just returning every qualifying question instead of the single best
    one, since the pair itself is already fixed by the caller."""
    if num_pairs < 1:
        return []
    prompt = SAME_TOPIC_MULTI_PROMPT.format(
        num_pairs=num_pairs, abstract_a=entry_a["abstract"], abstract_b=entry_b["abstract"],
    )
    try:
        parsed = _parse_json_loose(llm_fn(prompt))
    except Exception:
        return []
    return _sanitize_topic_picks_multi(parsed, num_pairs)


def _format_avoid_block(
    avoid_ranges_a: Optional[List[Tuple[float, float]]],
    avoid_ranges_b: Optional[List[Tuple[float, float]]],
) -> str:
    if not avoid_ranges_a and not avoid_ranges_b:
        return ""
    lines = ["Spans already used by an earlier clip in this same thread run -- pick a DIFFERENT span, do not reuse or overlap any of these:"]
    if avoid_ranges_a:
        ranges_text = ", ".join(f"{s:.1f}s-{e:.1f}s" for s, e in avoid_ranges_a)
        lines.append(f"- Episode A already-used spans: {ranges_text}")
    if avoid_ranges_b:
        ranges_text = ", ".join(f"{s:.1f}s-{e:.1f}s" for s, e in avoid_ranges_b)
        lines.append(f"- Episode B already-used spans: {ranges_text}")
    return "\n".join(lines)


def pick_thread_clips(
    episode_a: Dict, episode_b: Dict, shared_question: str, llm_fn: LLMFn,
    avoid_ranges_a: Optional[List[Tuple[float, float]]] = None,
    avoid_ranges_b: Optional[List[Tuple[float, float]]] = None,
    platform: str = "youtube",
) -> Optional[Dict]:
    """Stage B. episode_a/episode_b must each have a "transcript" key (full
    {duration, segments} shape). Returns None if the model can't ground a
    clip answering shared_question in BOTH transcripts. avoid_ranges_a/b,
    if given, are (start_time, end_time) spans already used by an earlier
    accepted pick in this same thread run (see select_thread_pairs) -- told
    to the model as a soft steer; the caller still enforces non-overlap
    itself as a hard backstop, since the model can ignore this. platform
    selects the clip-span length bounds from PLATFORM_BOUNDS -- "youtube"
    (default, 45-60s target) or "tiktok" (65-90s target, code-enforced to
    clear TikTok's 60s Creator Rewards minimum)."""
    bounds = PLATFORM_BOUNDS[platform]
    prompt = THREAD_PICK_SYSTEM_PROMPT.format(
        shared_question=shared_question,
        avoid_block=_format_avoid_block(avoid_ranges_a, avoid_ranges_b),
        clip_range=bounds["clip_range"],
        total_range=bounds["total_range"],
        transcript_a=build_transcript_text(episode_a["transcript"]),
        transcript_b=build_transcript_text(episode_b["transcript"]),
    )
    try:
        parsed = _parse_json_loose(llm_fn(prompt))
    except Exception:
        return None
    return _sanitize_clip_pick(
        parsed,
        duration_a=episode_a["transcript"].get("duration", 0.0),
        duration_b=episode_b["transcript"].get("duration", 0.0),
        min_clip=bounds["min_clip"],
        max_clip=bounds["max_clip"],
    )


def _overlaps_any(span: Tuple[float, float], ranges: List[Tuple[float, float]]) -> bool:
    start, end = span
    return any(start < r_end and end > r_start for r_start, r_end in ranges)


def select_thread_pairs(
    entry_a: Dict, entry_b: Dict, transcript_a: Dict, transcript_b: Dict,
    num_clips: int, llm_fn: LLMFn,
) -> List[Dict]:
    """Multi-pair picker for a FIXED pair of episodes (see
    pipeline.generate_threads, which ingests entry_a/entry_b and their
    transcripts up front). entry_a/entry_b need "run_dir", "title",
    "source_url", "abstract"; transcript_a/transcript_b are the full
    {duration, segments} shape.

    Returns up to num_clips grounded, non-overlapping thread dicts, each
    shaped like {"shared_question", "thesis", "bridge", "title",
    "description", "episode_a", "episode_b"} where episode_a/b carry
    run_dir/title/source_url plus the picked start_time/end_time. Returns
    [] rather than raising whenever
    fewer than num_clips (including zero) are groundable -- refuse rather
    than force, same philosophy as the old whole-corpus build_thread."""
    if num_clips < 1:
        return []

    shared_questions = find_same_topic_pairs(entry_a, entry_b, num_clips, llm_fn)
    if not shared_questions:
        print("[thread_builder] no same-topic questions found between the two episodes -- refusing to build a thread", flush=True)
        return []

    results: List[Dict] = []
    used_ranges_a: List[Tuple[float, float]] = []
    used_ranges_b: List[Tuple[float, float]] = []

    for shared_question in shared_questions:
        if len(results) >= num_clips:
            break
        try:
            clips = pick_thread_clips(
                {**entry_a, "transcript": transcript_a},
                {**entry_b, "transcript": transcript_b},
                shared_question, llm_fn,
                avoid_ranges_a=used_ranges_a, avoid_ranges_b=used_ranges_b,
            )
        except Exception as e:
            print(f"[thread_builder] skipping {shared_question!r}: {e}", flush=True)
            continue
        if clips is None:
            print(f"[thread_builder] no groundable clip pair for {shared_question!r} -- skipping", flush=True)
            continue

        span_a = (clips["clip_a"]["start_time"], clips["clip_a"]["end_time"])
        span_b = (clips["clip_b"]["start_time"], clips["clip_b"]["end_time"])
        if _overlaps_any(span_a, used_ranges_a) or _overlaps_any(span_b, used_ranges_b):
            print(f"[thread_builder] discarding {shared_question!r}: span reused from an earlier pick", flush=True)
            continue

        used_ranges_a.append(span_a)
        used_ranges_b.append(span_b)
        results.append({
            "shared_question": shared_question,
            "thesis": clips["thesis"],
            "bridge": clips["bridge"],
            "title": clips["title"],
            "description": clips["description"],
            "episode_a": {"run_dir": entry_a["run_dir"], "title": entry_a["title"], "source_url": entry_a["source_url"], **clips["clip_a"]},
            "episode_b": {"run_dir": entry_b["run_dir"], "title": entry_b["title"], "source_url": entry_b["source_url"], **clips["clip_b"]},
        })

    return results
