"""Two-stage picker for thread compilation: first a hard same-topic gate
across the whole local corpus (refuses outright if no pair shares a genuine
question -- see docs/superpowers/specs/2026-08-09-thread-compilation-design.md),
then, only for a qualifying pair, exact clip spans + narration text grounded
in the two chosen full transcripts.
"""
import json
import math
from typing import Dict, List, Optional, Tuple

from .corpus import build_corpus
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
- Duration per clip: 15-40 seconds. If the answer runs longer, pick the single most complete self-contained span within it.
- Write a "thesis" -- one sentence a narrator would say BEFORE either clip plays, naming the shared question and hinting that the two answers differ or agree in an interesting way. This must state the actual tension or agreement, not a vague tease.
- Write a "bridge" -- one sentence a narrator would say BETWEEN the two clips, explicitly connecting what episode A's speaker just said to what episode B's speaker is about to say.
- If you cannot find a genuine on-topic answer in BOTH transcripts, set "grounded" to false and leave the clip fields empty -- do not force a loose match.

Episode A transcript:
{transcript_a}

Episode B transcript:
{transcript_b}

Respond ONLY with valid JSON (no markdown, no explanation):
{{"grounded": bool, "thesis": "string", "bridge": "string", "clip_a": {{"start_time": float, "end_time": float}}, "clip_b": {{"start_time": float, "end_time": float}}}}"""

MIN_CLIP_SECONDS = 8.0
MAX_CLIP_SECONDS = 60.0


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


def _sanitize_clip_span(raw: object, duration: float) -> Optional[Dict]:
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
    if span < MIN_CLIP_SECONDS or span > MAX_CLIP_SECONDS:
        return None
    return {"start_time": start, "end_time": end}


def _sanitize_clip_pick(raw: object, duration_a: float, duration_b: float) -> Optional[Dict]:
    if not isinstance(raw, dict) or not raw.get("grounded"):
        return None
    raw_thesis = raw.get("thesis")
    raw_bridge = raw.get("bridge")
    if not isinstance(raw_thesis, str) or not isinstance(raw_bridge, str):
        return None
    thesis = raw_thesis.strip()
    bridge = raw_bridge.strip()
    if not thesis or not bridge:
        return None
    clip_a = _sanitize_clip_span(raw.get("clip_a"), duration_a)
    clip_b = _sanitize_clip_span(raw.get("clip_b"), duration_b)
    if clip_a is None or clip_b is None:
        return None
    return {"thesis": thesis, "bridge": bridge, "clip_a": clip_a, "clip_b": clip_b}


def find_same_topic_pair(corpus: List[Dict], llm_fn: LLMFn) -> Optional[Dict]:
    """Stage A. Returns None -- the expected, correct result whenever no
    pair shares a genuine question -- or {"episode_a_index",
    "episode_b_index", "shared_question"} for a qualifying pair."""
    if len(corpus) < 2:
        return None
    abstracts_block = "\n".join(
        f"{i}. {entry['title']}: {entry['abstract']}" for i, entry in enumerate(corpus)
    )
    prompt = SAME_TOPIC_SYSTEM_PROMPT.format(abstracts_block=abstracts_block)
    try:
        parsed = _parse_json_loose(llm_fn(prompt))
    except Exception:
        return None
    return _sanitize_topic_pick(parsed, corpus_len=len(corpus))


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
) -> Optional[Dict]:
    """Stage B. episode_a/episode_b must each have a "transcript" key (full
    {duration, segments} shape). Returns None if the model can't ground a
    clip answering shared_question in BOTH transcripts. avoid_ranges_a/b,
    if given, are (start_time, end_time) spans already used by an earlier
    accepted pick in this same thread run (see select_thread_pairs) -- told
    to the model as a soft steer; the caller still enforces non-overlap
    itself as a hard backstop, since the model can ignore this."""
    prompt = THREAD_PICK_SYSTEM_PROMPT.format(
        shared_question=shared_question,
        avoid_block=_format_avoid_block(avoid_ranges_a, avoid_ranges_b),
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
    )


def build_thread(base_dir: Optional[str] = None, llm_fn: Optional[LLMFn] = None) -> Optional[Dict]:
    """Full stage A + B pipeline. Returns None whenever the corpus doesn't
    support a genuine same-topic pairing today -- this is success (nothing
    to build), not a failure to retry or work around."""
    llm_fn = llm_fn or call_muapi_llm
    corpus = build_corpus(base_dir=base_dir, llm_fn=llm_fn)

    pick = find_same_topic_pair(corpus, llm_fn)
    if pick is None:
        print("[thread_builder] no same-topic pair found in corpus -- refusing to build a thread", flush=True)
        return None

    entry_a = corpus[pick["episode_a_index"]]
    entry_b = corpus[pick["episode_b_index"]]
    try:
        with open(f"{entry_a['run_dir']}/full_source.json", "r", encoding="utf-8") as f:
            transcript_a = json.load(f)
        with open(f"{entry_b['run_dir']}/full_source.json", "r", encoding="utf-8") as f:
            transcript_b = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[thread_builder] failed to load full transcript for picked pair: {e} -- refusing to build a thread", flush=True)
        return None

    # build_corpus only validates that full_source.json parses as JSON, not
    # that it has the {duration, segments: [{start, end, text}, ...]} shape
    # pick_thread_clips/build_transcript_text assume -- a present-but-null
    # "duration" or a segment missing "start"/"text" is valid JSON but
    # malformed shape, and would otherwise raise (TypeError/KeyError/
    # AttributeError) past this point instead of refusing like every other
    # "corpus doesn't support a pairing" case in this module.
    try:
        clips = pick_thread_clips(
            {**entry_a, "transcript": transcript_a},
            {**entry_b, "transcript": transcript_b},
            pick["shared_question"],
            llm_fn,
        )
    except Exception as e:
        print(f"[thread_builder] malformed transcript shape for picked pair: {e} -- refusing to build a thread", flush=True)
        return None
    if clips is None:
        print("[thread_builder] no groundable clip pair for the shared question -- refusing to build a thread", flush=True)
        return None

    return {
        "shared_question": pick["shared_question"],
        "thesis": clips["thesis"],
        "bridge": clips["bridge"],
        "episode_a": {
            "run_dir": entry_a["run_dir"], "title": entry_a["title"], "source_url": entry_a["source_url"],
            **clips["clip_a"],
        },
        "episode_b": {
            "run_dir": entry_b["run_dir"], "title": entry_b["title"], "source_url": entry_b["source_url"],
            **clips["clip_b"],
        },
    }
