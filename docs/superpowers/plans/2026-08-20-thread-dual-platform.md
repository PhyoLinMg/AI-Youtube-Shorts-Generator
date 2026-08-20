# Thread Clips: YouTube + TikTok Dual-Platform Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `platform` axis (`youtube` / `tiktok` / `both`) to thread-clip generation so TikTok cuts hit its 60s+ Creator Rewards monetization floor, without changing YouTube's existing 45-60s default.

**Architecture:** `thread_builder.py` gains a `PLATFORM_BOUNDS` dict and threads a `platform` param through its LLM-picker functions; its expensive same-topic discovery step (`find_same_topic_pairs`) is extracted so it runs once and is shared across platforms via a new `ground_thread_clips`. `pipeline.generate_threads` loops per requested platform, writing platform-suffixed filenames, and probes TikTok output duration as a non-fatal safety net. `run_output.write_thread_descriptions`, `main.py`, and `webapp.py` follow.

**Tech Stack:** Python, pytest, Flask, ffmpeg/ffprobe (subprocess).

Spec: `docs/superpowers/specs/2026-08-20-thread-dual-platform-design.md`

---

## Commit 1: Core generation + CLI (Tasks 1-6)

### Task 1: Platform-aware clip-span bounds (`thread_builder.py`)

**Files:**
- Modify: `shorts_generator/thread_builder.py:53-80`
- Test: `tests/test_thread_builder.py`

- [ ] **Step 1: Write failing tests for platform-specific bounds**

Add to `tests/test_thread_builder.py`:

```python
def test_pick_thread_clips_tiktok_platform_rejects_span_below_28_seconds():
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    llm_fn = lambda prompt: json.dumps({
        "grounded": True, "thesis": "t", "bridge": "b", "title": "Title #Shorts", "description": "d",
        "clip_a": {"start_time": 0.0, "end_time": 27.9},
        "clip_b": {"start_time": 0.0, "end_time": 30.0},
    })

    assert thread_builder.pick_thread_clips(episode_a, episode_b, "q?", llm_fn, platform="tiktok") is None


def test_pick_thread_clips_tiktok_platform_accepts_span_at_28_seconds():
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    llm_fn = lambda prompt: json.dumps({
        "grounded": True, "thesis": "t", "bridge": "b", "title": "Title #Shorts", "description": "d",
        "clip_a": {"start_time": 0.0, "end_time": 28.0},
        "clip_b": {"start_time": 0.0, "end_time": 30.0},
    })

    result = thread_builder.pick_thread_clips(episode_a, episode_b, "q?", llm_fn, platform="tiktok")

    assert result["clip_a"] == {"start_time": 0.0, "end_time": 28.0}


def test_pick_thread_clips_tiktok_platform_rejects_span_above_40_seconds():
    episode_a = _episode(100.0, [(0.0, 45.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    llm_fn = lambda prompt: json.dumps({
        "grounded": True, "thesis": "t", "bridge": "b", "title": "Title #Shorts", "description": "d",
        "clip_a": {"start_time": 0.0, "end_time": 40.1},
        "clip_b": {"start_time": 0.0, "end_time": 30.0},
    })

    assert thread_builder.pick_thread_clips(episode_a, episode_b, "q?", llm_fn, platform="tiktok") is None


def test_pick_thread_clips_defaults_to_youtube_platform_bounds():
    """A 30s clip_a span exceeds youtube's 8-25s bound (thread_builder.
    PLATFORM_BOUNDS) but sits well inside tiktok's 28-40s bound -- it must
    still be rejected when platform is omitted, proving "youtube" is the
    implicit default rather than "tiktok" (a bug here would silently pass
    this pick through instead of rejecting it)."""
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    llm_fn = lambda prompt: json.dumps({
        "grounded": True, "thesis": "t", "bridge": "b", "title": "Title #Shorts", "description": "d",
        "clip_a": {"start_time": 0.0, "end_time": 30.0},
        "clip_b": {"start_time": 0.0, "end_time": 20.0},
    })

    assert thread_builder.pick_thread_clips(episode_a, episode_b, "q?", llm_fn) is None


def test_pick_thread_clips_prompt_uses_tiktok_length_instructions():
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    seen_prompts = []

    def llm_fn(prompt):
        seen_prompts.append(prompt)
        return json.dumps({
            "grounded": True, "thesis": "t", "bridge": "b", "title": "Title #Shorts", "description": "d",
            "clip_a": {"start_time": 0.0, "end_time": 30.0},
            "clip_b": {"start_time": 0.0, "end_time": 30.0},
        })

    thread_builder.pick_thread_clips(episode_a, episode_b, "q?", llm_fn, platform="tiktok")

    assert "28-40 seconds" in seen_prompts[0]
    assert "65-90 second" in seen_prompts[0]
    assert "do not use extra length just because it's available" in seen_prompts[0]


def test_pick_thread_clips_prompt_uses_youtube_length_instructions_by_default():
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    seen_prompts = []

    def llm_fn(prompt):
        seen_prompts.append(prompt)
        return json.dumps({
            "grounded": True, "thesis": "t", "bridge": "b", "title": "Title #Shorts", "description": "d",
            "clip_a": {"start_time": 0.0, "end_time": 20.0},
            "clip_b": {"start_time": 0.0, "end_time": 20.0},
        })

    thread_builder.pick_thread_clips(episode_a, episode_b, "q?", llm_fn)

    assert "12-22 seconds" in seen_prompts[0]
    assert "45-60 second" in seen_prompts[0]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_thread_builder.py -k "tiktok_platform or prompt_uses" -v`
Expected: FAIL — `pick_thread_clips() got an unexpected keyword argument 'platform'`

- [ ] **Step 3: Replace the module constants and prompt with platform bounds**

In `shorts_generator/thread_builder.py`, replace lines 53-80 (from `THREAD_PICK_SYSTEM_PROMPT = """...` through `MAX_CLIP_SECONDS = 25.0  # ...`) with:

```python
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
PLATFORM_BOUNDS = {
    "youtube": {"min_clip": 8.0, "max_clip": 25.0, "clip_range": "12-22 seconds", "total_range": "45-60 second"},
    "tiktok": {"min_clip": 28.0, "max_clip": 40.0, "clip_range": "28-40 seconds", "total_range": "65-90 second"},
}
```

- [ ] **Step 4: Thread `platform` through `_sanitize_clip_span` and `pick_thread_clips`**

In `shorts_generator/thread_builder.py`, replace the `_sanitize_clip_span` function (originally lines 132-146):

```python
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
```

Replace `_sanitize_clip_pick` (originally lines 149-170) so it accepts and forwards bounds:

```python
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
```

Replace `pick_thread_clips` (originally lines 208-234):

```python
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
```

- [ ] **Step 5: Run the full thread_builder test suite**

Run: `pytest tests/test_thread_builder.py -v`
Expected: PASS (all existing tests unaffected — they call `pick_thread_clips` without `platform`, defaulting to `"youtube"`, identical bounds to before)

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/thread_builder.py tests/test_thread_builder.py
git commit -m "$(cat <<'EOF'
feat: platform-aware clip-span bounds for thread picker

TikTok's Creator Rewards Program only pays 60s+ videos. Adds
PLATFORM_BOUNDS (youtube: 8-25s/45-60s total, tiktok: 28-40s/65-90s
total) and threads platform through pick_thread_clips so the LLM
picker targets the right length per platform.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Extract `ground_thread_clips` so stage A runs once per platform set (`thread_builder.py`)

**Files:**
- Modify: `shorts_generator/thread_builder.py:242-306` (original line numbers, `select_thread_pairs`)
- Test: `tests/test_thread_builder.py`

- [ ] **Step 1: Write failing tests for `ground_thread_clips`**

Add to `tests/test_thread_builder.py`:

```python
def test_ground_thread_clips_does_not_call_find_same_topic_pairs(monkeypatch):
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])

    def _fail_if_called(entry_a, entry_b, num_pairs, llm_fn):
        pytest.fail("ground_thread_clips must not call find_same_topic_pairs -- shared_questions is an argument")

    monkeypatch.setattr(thread_builder, "find_same_topic_pairs", _fail_if_called)

    llm_fn = lambda prompt: json.dumps({
        "grounded": True, "thesis": "t", "bridge": "b", "title": "Title #Shorts", "description": "d",
        "clip_a": {"start_time": 0.0, "end_time": 20.0}, "clip_b": {"start_time": 0.0, "end_time": 20.0},
    })

    result = thread_builder.ground_thread_clips(
        entry_a, entry_b, transcript_a, transcript_b, ["Question one?"], num_clips=1, llm_fn=llm_fn,
    )

    assert len(result) == 1


def test_ground_thread_clips_uses_tiktok_bounds_when_given_platform():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 60.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 60.0, "hello from b")])
    llm_fn = lambda prompt: json.dumps({
        "grounded": True, "thesis": "t", "bridge": "b", "title": "Title #Shorts", "description": "d",
        "clip_a": {"start_time": 0.0, "end_time": 20.0},  # valid for youtube, too short for tiktok
        "clip_b": {"start_time": 0.0, "end_time": 20.0},
    })

    result = thread_builder.ground_thread_clips(
        entry_a, entry_b, transcript_a, transcript_b, ["Question one?"], num_clips=1, llm_fn=llm_fn, platform="tiktok",
    )

    assert result == []


def test_select_thread_pairs_calls_ground_thread_clips_with_default_platform(monkeypatch):
    entry_a = _corpus_entry(0, "Ep A", "argues remote work increases productivity", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "argues remote work decreases productivity", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    seen_platforms = []

    def _fake_ground(entry_a, entry_b, transcript_a, transcript_b, shared_questions, num_clips, llm_fn, platform="youtube"):
        seen_platforms.append(platform)
        return []

    monkeypatch.setattr(thread_builder, "ground_thread_clips", _fake_ground)
    llm_fn = lambda prompt: json.dumps({"shared_questions": ["Does remote work increase or decrease productivity?"]})

    thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=1, llm_fn=llm_fn)

    assert seen_platforms == ["youtube"]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_thread_builder.py -k "ground_thread_clips or calls_ground_thread_clips" -v`
Expected: FAIL — `AttributeError: module 'shorts_generator.thread_builder' has no attribute 'ground_thread_clips'`

- [ ] **Step 3: Extract `ground_thread_clips` and make `select_thread_pairs` a thin wrapper**

In `shorts_generator/thread_builder.py`, replace `select_thread_pairs` (originally lines 242-306) with:

```python
def ground_thread_clips(
    entry_a: Dict, entry_b: Dict, transcript_a: Dict, transcript_b: Dict,
    shared_questions: List[str], num_clips: int, llm_fn: LLMFn, platform: str = "youtube",
) -> List[Dict]:
    """Stage B driver. Given a pre-computed shared_questions list (see
    find_same_topic_pairs), grounds up to num_clips non-overlapping clip
    pairs for the given platform. Split out from select_thread_pairs so a
    caller building both a "youtube" and a "tiktok" cut of the same thread
    run can call find_same_topic_pairs (the expensive same-topic LLM scan)
    exactly once and reuse its shared_questions for both platform passes --
    see pipeline.generate_threads.

    Returns up to num_clips grounded, non-overlapping thread dicts, each
    shaped like {"shared_question", "thesis", "bridge", "title",
    "description", "episode_a", "episode_b"} where episode_a/b carry
    run_dir/title/source_url plus the picked start_time/end_time. Returns
    [] rather than raising whenever fewer than num_clips (including zero)
    are groundable -- refuse rather than force, same philosophy as the old
    whole-corpus build_thread."""
    if num_clips < 1:
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
                platform=platform,
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


def select_thread_pairs(
    entry_a: Dict, entry_b: Dict, transcript_a: Dict, transcript_b: Dict,
    num_clips: int, llm_fn: LLMFn,
) -> List[Dict]:
    """Multi-pair picker for a FIXED pair of episodes (see
    pipeline.generate_threads, which ingests entry_a/entry_b and their
    transcripts up front). entry_a/entry_b need "run_dir", "title",
    "source_url", "abstract"; transcript_a/transcript_b are the full
    {duration, segments} shape.

    Single-platform ("youtube") convenience wrapper around
    find_same_topic_pairs + ground_thread_clips -- see ground_thread_clips
    for the return shape. pipeline.generate_threads calls the two stages
    directly instead of this wrapper when it needs to share one
    find_same_topic_pairs call across multiple platforms."""
    if num_clips < 1:
        return []

    shared_questions = find_same_topic_pairs(entry_a, entry_b, num_clips, llm_fn)
    if not shared_questions:
        print("[thread_builder] no same-topic questions found between the two episodes -- refusing to build a thread", flush=True)
        return []

    return ground_thread_clips(entry_a, entry_b, transcript_a, transcript_b, shared_questions, num_clips, llm_fn)
```

- [ ] **Step 4: Run the full thread_builder test suite**

Run: `pytest tests/test_thread_builder.py -v`
Expected: PASS — all existing `select_thread_pairs` tests still pass unchanged (same signature, same behavior), plus the new tests from this task and Task 1.

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/thread_builder.py tests/test_thread_builder.py
git commit -m "$(cat <<'EOF'
refactor: extract ground_thread_clips from select_thread_pairs

Splits stage A (same-topic question discovery) from stage B (clip
grounding) so a caller producing both a youtube and a tiktok cut of
the same thread run can share one find_same_topic_pairs call across
both platform passes instead of paying for it twice.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `pipeline.generate_threads` platform loop and naming

**Files:**
- Modify: `shorts_generator/pipeline.py:29` (import), `shorts_generator/pipeline.py:473-618` (`generate_threads`)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Update the import**

In `shorts_generator/pipeline.py`, replace line 29:

```python
from .thread_builder import select_thread_pairs
```

with:

```python
from .thread_builder import find_same_topic_pairs, ground_thread_clips
```

- [ ] **Step 2: Replace `generate_threads`**

In `shorts_generator/pipeline.py`, replace the entire `generate_threads` function (originally lines 473-618) with:

```python
def generate_threads(
    url_a: str,
    url_b: str,
    num_clips: int = 1,
    platform: str = "youtube",
    base_dir: Optional[str] = None,
    on_output_dir: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    """Build up to num_clips distinct-topic threads from exactly the two
    given episodes (see thread_builder.ground_thread_clips). Local-mode
    only, like generate_chapters -- there is no MuAPI equivalent of this
    feature. Both URLs are ingested caption-only (no video download; see
    local/caption_ingest.py) and idempotently reused if already in the
    corpus. Returns [] if no shared question is groundable between the two
    episodes, or if every requested platform grounds zero clip pairs for
    every shared question -- this is the expected, correct result when
    they don't genuinely cover the same topic (or, for tiktok, when no
    span in the transcript is long enough), not a failure to work around.

    platform selects the output cut(s): "youtube" (default, 45-60s
    target), "tiktok" (65-90s target, code-enforced by
    thread_builder.PLATFORM_BOUNDS to clear TikTok's 60s Creator Rewards
    Program minimum), or "both" (produce one file of each). The
    same-topic-question scan (thread_builder.find_same_topic_pairs) runs
    exactly once regardless of platform -- it has no dependency on clip
    length, so its result is reused across every requested platform's own
    grounding pass (thread_builder.ground_thread_clips). Output files are
    named thesis_{i}_{platform}_{title}.mp4, with raw intermediates under
    raw/thesis_{i}_{platform}/, where i is 1-indexed within that
    platform's own results (not the flattened combined list).

    Unlike generate_shorts/generate_chapters, the output dir is knowable up
    front from the two episode titles (see resolve_thread_run_dir) -- but
    on_output_dir, if given, still fires before any per-clip render work
    starts, matching the old single-clip contract, so a caller like the
    dashboard can start tailing progress.log immediately.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .local.downloader import download_youtube_local
    from .local.llm import call_local_llm

    platforms = ["youtube", "tiktok"] if platform == "both" else [platform]

    # url_a and url_b are two unrelated episodes -- each ingest is its own
    # yt-dlp caption fetch plus, on a cache miss, its own LLM abstract call,
    # writing into that episode's own run_dir (see corpus.get_abstract_cached)
    # -- so there's no shared state between them and no reason to pay both
    # round trips back-to-back.
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(_ingest_and_abstract, url_a, base_dir, call_local_llm)
        future_b = pool.submit(_ingest_and_abstract, url_b, base_dir, call_local_llm)
        entry_a = future_a.result()
        entry_b = future_b.result()

    out_dir = resolve_thread_run_dir(entry_a["title"], entry_b["title"], base_dir=base_dir)
    archive_stale_thread_run(out_dir)
    if on_output_dir:
        on_output_dir(out_dir)

    with capture_progress_log(os.path.join(out_dir, "progress.log")):
        print(f"[pipeline/local] ingested episode A: {entry_a['title']!r}", flush=True)
        print(f"[pipeline/local] ingested episode B: {entry_b['title']!r}", flush=True)

        with open(os.path.join(entry_a["run_dir"], "full_source.json"), "r", encoding="utf-8") as f:
            transcript_a = json.load(f)
        with open(os.path.join(entry_b["run_dir"], "full_source.json"), "r", encoding="utf-8") as f:
            transcript_b = json.load(f)

        print(f"[pipeline/local] scanning for up to {num_clips} shared-question thread(s)...", flush=True)
        shared_questions = find_same_topic_pairs(entry_a, entry_b, num_clips, call_local_llm)
        if not shared_questions:
            return []

        pairs_by_platform = {}
        for p in platforms:
            pairs_by_platform[p] = ground_thread_clips(
                entry_a, entry_b, transcript_a, transcript_b, shared_questions, num_clips, call_local_llm, platform=p,
            )
        if not any(pairs_by_platform.values()):
            return []

        # Download each episode's full video exactly once up front instead of
        # letting every clip in the loop below re-download its own padded
        # span (acquire_clip's fast path already prefers a full_source.mp4 on
        # disk over a fresh yt-dlp call) -- one full download per episode is
        # both fewer network round trips than num_clips separate section
        # downloads and less exposed to the section-download-specific CDN
        # flakiness seen with --download-sections. Only delete the videos we
        # downloaded here, never a full_source.mp4 that predates this run
        # (e.g. left over from a prior Shorts/chapters run on the same URL).
        full_source_a = os.path.join(entry_a["run_dir"], "full_source.mp4")
        full_source_b = os.path.join(entry_b["run_dir"], "full_source.mp4")
        downloaded_full_sources = []
        try:
            # Inside the try/finally from the start: if A's download
            # succeeds but B's raises, the finally below must still remove
            # A's now-orphaned multi-GB file rather than leak it.
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = []
                if not os.path.exists(full_source_a):
                    print(f"[pipeline/local] downloading full source video A: {entry_a['title']!r}...", flush=True)
                    futures.append(pool.submit(download_youtube_local, entry_a["source_url"], full_source_a))
                    downloaded_full_sources.append(full_source_a)
                if not os.path.exists(full_source_b):
                    print(f"[pipeline/local] downloading full source video B: {entry_b['title']!r}...", flush=True)
                    futures.append(pool.submit(download_youtube_local, entry_b["source_url"], full_source_b))
                    downloaded_full_sources.append(full_source_b)
                for future in futures:
                    future.result()

            results = []
            for p in platforms:
                pairs = pairs_by_platform[p]
                for i, thread in enumerate(pairs, 1):
                    episode_a, episode_b = thread["episode_a"], thread["episode_b"]
                    print(f"[pipeline/local] [{p}] clip {i}/{len(pairs)}: {thread['shared_question']!r}", flush=True)

                    thesis_dir = os.path.join(out_dir, "raw", f"thesis_{i}_{p}")
                    os.makedirs(thesis_dir, exist_ok=True)

                    clip_a_path = os.path.join(thesis_dir, f"clip_{i}_a.mp4")
                    clip_b_path = os.path.join(thesis_dir, f"clip_{i}_b.mp4")
                    print(f"[pipeline/local] acquiring clip A from {episode_a['title']!r}...", flush=True)
                    acquire_clip(
                        episode_a["run_dir"], episode_a["source_url"], cached_duration=transcript_a.get("duration") or 0.0,
                        start_time=episode_a["start_time"], end_time=episode_a["end_time"], out_path=clip_a_path,
                    )
                    print(f"[pipeline/local] acquiring clip B from {episode_b['title']!r}...", flush=True)
                    acquire_clip(
                        episode_b["run_dir"], episode_b["source_url"], cached_duration=transcript_b.get("duration") or 0.0,
                        start_time=episode_b["start_time"], end_time=episode_b["end_time"], out_path=clip_b_path,
                    )

                    intro_audio = os.path.join(thesis_dir, f"thesis_{i}.mp3")
                    bridge_audio = os.path.join(thesis_dir, f"bridge_{i}.mp3")
                    print("[pipeline/local] synthesizing narration (thesis + bridge)...", flush=True)
                    synthesize_narration(thread["thesis"], intro_audio)
                    synthesize_narration(thread["bridge"], bridge_audio)

                    intro_card = os.path.join(thesis_dir, f"intro_card_{i}.mp4")
                    bridge_card = os.path.join(thesis_dir, f"bridge_card_{i}.mp4")
                    print("[pipeline/local] rendering narration cards...", flush=True)
                    render_narration_card(intro_audio, thread["thesis"], intro_card)
                    render_narration_card(bridge_audio, thread["bridge"], bridge_card)

                    final_title = thread.get("title") or thread["shared_question"]
                    final_path = os.path.join(out_dir, f"thesis_{i}_{p}_{sanitize_title(final_title)}.mp4")
                    print("[pipeline/local] assembling final thread (intro -> clip A -> bridge -> clip B)...", flush=True)
                    assemble_thread([intro_card, clip_a_path, bridge_card, clip_b_path], final_path)

                    results.append({
                        **thread,
                        "platform": p,
                        "platform_index": i,
                        "output_dir": out_dir,
                        "clip_url": final_path,
                        "episode_a": {**episode_a, "clip_url": clip_a_path},
                        "episode_b": {**episode_b, "clip_url": clip_b_path},
                    })
                    print(f"[pipeline/local] done: {final_path}", flush=True)
        finally:
            # Clean up regardless of success/failure mid-loop -- an
            # abandoned multi-GB full_source.mp4 from a crashed run is worse
            # than losing the (re-downloadable) cache on a genuine error.
            for path in downloaded_full_sources:
                if os.path.exists(path):
                    print(f"[pipeline/local] removing downloaded full source video: {path}", flush=True)
                    os.remove(path)

        with open(os.path.join(out_dir, "thread_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        write_thread_descriptions(out_dir, results)
        return results
```

- [ ] **Step 3: Update the existing tests that mock the old `select_thread_pairs` call site**

In `tests/test_pipeline.py`, replace `test_generate_threads_returns_empty_list_when_no_shared_questions` (originally lines 1024-1041):

```python
def test_generate_threads_returns_empty_list_when_no_shared_questions(tmp_path, monkeypatch):
    episode_a_dir = tmp_path / "Episode_A"
    episode_b_dir = tmp_path / "Episode_B"
    episode_a_dir.mkdir()
    episode_b_dir.mkdir()
    (episode_a_dir / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))
    (episode_b_dir / "full_source.json").write_text(json.dumps({"duration": 200.0, "segments": []}))

    monkeypatch.setattr(pipeline_module, "ingest_captions", _fake_ingest_captions({
        "https://example.com/a": {"run_dir": str(episode_a_dir), "title": "Episode A", "duration": 100.0, "segment_count": 0},
        "https://example.com/b": {"run_dir": str(episode_b_dir), "title": "Episode B", "duration": 200.0, "segment_count": 0},
    }))
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: "an abstract")
    monkeypatch.setattr(pipeline_module, "find_same_topic_pairs", lambda entry_a, entry_b, num_clips, llm_fn: [])

    result = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=2, base_dir=str(tmp_path))

    assert result == []
```

Replace `test_generate_threads_assembles_and_writes_results_for_each_pair` (originally lines 1044-1097):

```python
def test_generate_threads_assembles_and_writes_results_for_each_pair(tmp_path, monkeypatch):
    episode_a_dir = tmp_path / "Episode_A"
    episode_b_dir = tmp_path / "Episode_B"
    episode_a_dir.mkdir()
    episode_b_dir.mkdir()
    (episode_a_dir / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))
    (episode_b_dir / "full_source.json").write_text(json.dumps({"duration": 200.0, "segments": []}))

    monkeypatch.setattr(pipeline_module, "ingest_captions", _fake_ingest_captions({
        "https://example.com/a": {"run_dir": str(episode_a_dir), "title": "Episode A", "duration": 100.0, "segment_count": 0},
        "https://example.com/b": {"run_dir": str(episode_b_dir), "title": "Episode B", "duration": 200.0, "segment_count": 0},
    }))
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: "an abstract")

    fake_pairs = [
        {
            "shared_question": "Does X cause Y?", "thesis": "t1", "bridge": "b1", "title": "Title One #Shorts",
            "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 10.0, "end_time": 30.0},
            "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 5.0, "end_time": 25.0},
        },
        {
            "shared_question": "Does A cause B?", "thesis": "t2", "bridge": "b2", "title": "Title Two #Shorts",
            "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 40.0, "end_time": 60.0},
            "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 35.0, "end_time": 55.0},
        },
    ]
    monkeypatch.setattr(pipeline_module, "find_same_topic_pairs", lambda entry_a, entry_b, num_clips, llm_fn: ["Does X cause Y?", "Does A cause B?"])
    monkeypatch.setattr(pipeline_module, "ground_thread_clips", lambda entry_a, entry_b, transcript_a, transcript_b, shared_questions, num_clips, llm_fn, platform="youtube": fake_pairs)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt="720": open(target_path, "wb").write(b"full source") or target_path,
    )
    monkeypatch.setattr(pipeline_module, "acquire_clip", lambda run_dir, source_url, cached_duration, start_time, end_time, out_path: open(out_path, "wb").write(b"clip") or {"clip_path": out_path})
    monkeypatch.setattr(pipeline_module, "synthesize_narration", lambda text, out_path, **k: open(out_path, "wb").write(b"audio") or out_path)
    monkeypatch.setattr(pipeline_module, "render_narration_card", lambda audio_path, text, out_path: open(out_path, "wb").write(b"card") or out_path)
    assemble_calls = []
    monkeypatch.setattr(pipeline_module, "assemble_thread", lambda segment_paths, out_path: (assemble_calls.append(segment_paths), open(out_path, "wb").write(b"final"))[1] or out_path)

    result = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=2, base_dir=str(tmp_path))

    assert len(result) == 2
    out_dir = result[0]["output_dir"]
    assert result[0]["platform"] == "youtube"
    assert result[0]["platform_index"] == 1
    assert result[1]["platform_index"] == 2
    assert result[0]["clip_url"] == os.path.join(out_dir, "thesis_1_youtube_Title_One_Shorts.mp4")
    assert result[1]["clip_url"] == os.path.join(out_dir, "thesis_2_youtube_Title_Two_Shorts.mp4")
    assert result[0]["episode_a"]["clip_url"] == os.path.join(out_dir, "raw", "thesis_1_youtube", "clip_1_a.mp4")
    assert result[0]["episode_b"]["clip_url"] == os.path.join(out_dir, "raw", "thesis_1_youtube", "clip_1_b.mp4")
    assert result[1]["episode_a"]["clip_url"] == os.path.join(out_dir, "raw", "thesis_2_youtube", "clip_2_a.mp4")
    assert assemble_calls[0] == [
        os.path.join(out_dir, "raw", "thesis_1_youtube", "intro_card_1.mp4"), os.path.join(out_dir, "raw", "thesis_1_youtube", "clip_1_a.mp4"),
        os.path.join(out_dir, "raw", "thesis_1_youtube", "bridge_card_1.mp4"), os.path.join(out_dir, "raw", "thesis_1_youtube", "clip_1_b.mp4"),
    ]
    assert os.path.isfile(os.path.join(out_dir, "thread_results.json"))
    with open(os.path.join(out_dir, "thread_results.json")) as f:
        written = json.load(f)
    assert len(written) == 2
```

Replace `test_generate_threads_calls_on_output_dir_before_the_slow_work` (originally lines 1100-1140):

```python
def test_generate_threads_calls_on_output_dir_before_the_slow_work(tmp_path, monkeypatch):
    episode_a_dir = tmp_path / "Episode_A"
    episode_b_dir = tmp_path / "Episode_B"
    episode_a_dir.mkdir()
    episode_b_dir.mkdir()
    (episode_a_dir / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))
    (episode_b_dir / "full_source.json").write_text(json.dumps({"duration": 200.0, "segments": []}))

    monkeypatch.setattr(pipeline_module, "ingest_captions", _fake_ingest_captions({
        "https://example.com/a": {"run_dir": str(episode_a_dir), "title": "Episode A", "duration": 100.0, "segment_count": 0},
        "https://example.com/b": {"run_dir": str(episode_b_dir), "title": "Episode B", "duration": 200.0, "segment_count": 0},
    }))
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: "an abstract")

    fake_pairs = [{
        "shared_question": "Does X cause Y?", "thesis": "t1", "bridge": "b1",
        "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 10.0, "end_time": 30.0},
        "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 5.0, "end_time": 25.0},
    }]
    monkeypatch.setattr(pipeline_module, "find_same_topic_pairs", lambda entry_a, entry_b, num_clips, llm_fn: ["Does X cause Y?"])
    monkeypatch.setattr(pipeline_module, "ground_thread_clips", lambda entry_a, entry_b, transcript_a, transcript_b, shared_questions, num_clips, llm_fn, platform="youtube": fake_pairs)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt="720": open(target_path, "wb").write(b"full source") or target_path,
    )

    calls = []

    def _fake_acquire_clip(run_dir, source_url, cached_duration, start_time, end_time, out_path):
        assert calls, "on_output_dir was not called before acquire_clip"
        open(out_path, "wb").write(b"clip")
        return {"clip_path": out_path}

    monkeypatch.setattr(pipeline_module, "acquire_clip", _fake_acquire_clip)
    monkeypatch.setattr(pipeline_module, "synthesize_narration", lambda text, out_path, **k: open(out_path, "wb").write(b"audio") or out_path)
    monkeypatch.setattr(pipeline_module, "render_narration_card", lambda audio_path, text, out_path: open(out_path, "wb").write(b"card") or out_path)
    monkeypatch.setattr(pipeline_module, "assemble_thread", lambda segment_paths, out_path: open(out_path, "wb").write(b"final") or out_path)

    result = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=1, base_dir=str(tmp_path), on_output_dir=calls.append)

    assert calls == [result[0]["output_dir"]]
    assert os.path.isfile(os.path.join(result[0]["output_dir"], "progress.log"))
```

Replace `_setup_thread_run` (originally lines 1143-1167):

```python
def _setup_thread_run(tmp_path, monkeypatch, num_pairs=2):
    episode_a_dir = tmp_path / "Episode_A"
    episode_b_dir = tmp_path / "Episode_B"
    episode_a_dir.mkdir()
    episode_b_dir.mkdir()
    (episode_a_dir / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))
    (episode_b_dir / "full_source.json").write_text(json.dumps({"duration": 200.0, "segments": []}))

    monkeypatch.setattr(pipeline_module, "ingest_captions", _fake_ingest_captions({
        "https://example.com/a": {"run_dir": str(episode_a_dir), "title": "Episode A", "duration": 100.0, "segment_count": 0},
        "https://example.com/b": {"run_dir": str(episode_b_dir), "title": "Episode B", "duration": 200.0, "segment_count": 0},
    }))
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: "an abstract")

    fake_pairs = [{
        "shared_question": f"Question {i}?", "thesis": f"t{i}", "bridge": f"b{i}",
        "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 10.0 * i, "end_time": 10.0 * i + 5},
        "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 10.0 * i, "end_time": 10.0 * i + 5},
    } for i in range(1, num_pairs + 1)]
    monkeypatch.setattr(pipeline_module, "find_same_topic_pairs", lambda entry_a, entry_b, num_clips, llm_fn: [f"Question {i}?" for i in range(1, num_pairs + 1)])
    monkeypatch.setattr(pipeline_module, "ground_thread_clips", lambda entry_a, entry_b, transcript_a, transcript_b, shared_questions, num_clips, llm_fn, platform="youtube": fake_pairs)
    monkeypatch.setattr(pipeline_module, "acquire_clip", lambda run_dir, source_url, cached_duration, start_time, end_time, out_path: open(out_path, "wb").write(b"clip") or {"clip_path": out_path})
    monkeypatch.setattr(pipeline_module, "synthesize_narration", lambda text, out_path, **k: open(out_path, "wb").write(b"audio") or out_path)
    monkeypatch.setattr(pipeline_module, "render_narration_card", lambda audio_path, text, out_path: open(out_path, "wb").write(b"card") or out_path)
    monkeypatch.setattr(pipeline_module, "assemble_thread", lambda segment_paths, out_path: open(out_path, "wb").write(b"final") or out_path)
    return episode_a_dir, episode_b_dir
```

In `test_generate_threads_final_filename_falls_back_to_shared_question_when_title_missing` (originally lines 1237-1251), update the assertion:

```python
def test_generate_threads_final_filename_falls_back_to_shared_question_when_title_missing(tmp_path, monkeypatch):
    """_setup_thread_run's fake pairs (used by most of the tests below this
    one) don't set a "title" key -- generate_threads must not crash on that,
    and should fall back to shared_question for the final filename, same
    fallback write_thread_descriptions already uses."""
    _setup_thread_run(tmp_path, monkeypatch, num_pairs=1)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt="720": open(target_path, "wb").write(b"full source") or target_path,
    )

    result = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=1, base_dir=str(tmp_path))

    out_dir = result[0]["output_dir"]
    assert result[0]["clip_url"] == os.path.join(out_dir, "thesis_1_youtube_Question_1.mp4")
```

`test_generate_threads_downloads_full_source_once_and_deletes_after_all_clips`, `test_generate_threads_cleans_up_partial_download_when_one_episode_fails`, `test_generate_threads_does_not_download_or_delete_preexisting_full_source`, and `test_generate_threads_archives_prior_same_slug_run_before_second_call` need no assertion changes — they only depend on `_setup_thread_run`'s behavior (now updated above), not on filenames.

- [ ] **Step 4: Add new tests for the platform loop itself**

Add to `tests/test_pipeline.py`:

```python
def _setup_both_platform_thread_run(tmp_path, monkeypatch, tiktok_pairs=None):
    """Like _setup_thread_run but drives find_same_topic_pairs/ground_thread_clips
    directly so a test can vary ground_thread_clips' per-platform return
    value (e.g. simulate tiktok grounding nothing for a question youtube
    still grounds)."""
    episode_a_dir = tmp_path / "Episode_A"
    episode_b_dir = tmp_path / "Episode_B"
    episode_a_dir.mkdir()
    episode_b_dir.mkdir()
    (episode_a_dir / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))
    (episode_b_dir / "full_source.json").write_text(json.dumps({"duration": 200.0, "segments": []}))

    monkeypatch.setattr(pipeline_module, "ingest_captions", _fake_ingest_captions({
        "https://example.com/a": {"run_dir": str(episode_a_dir), "title": "Episode A", "duration": 100.0, "segment_count": 0},
        "https://example.com/b": {"run_dir": str(episode_b_dir), "title": "Episode B", "duration": 200.0, "segment_count": 0},
    }))
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: "an abstract")
    monkeypatch.setattr(pipeline_module, "find_same_topic_pairs", lambda entry_a, entry_b, num_clips, llm_fn: ["Does X cause Y?"])

    def _youtube_pair():
        return [{
            "shared_question": "Does X cause Y?", "thesis": "t", "bridge": "b", "title": "Title youtube",
            "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 10.0, "end_time": 30.0},
            "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 5.0, "end_time": 25.0},
        }]

    def _fake_ground(entry_a, entry_b, transcript_a, transcript_b, shared_questions, num_clips, llm_fn, platform="youtube"):
        if platform == "tiktok":
            return tiktok_pairs if tiktok_pairs is not None else [{
                "shared_question": "Does X cause Y?", "thesis": "t", "bridge": "b", "title": "Title tiktok",
                "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 10.0, "end_time": 45.0},
                "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 5.0, "end_time": 40.0},
            }]
        return _youtube_pair()

    monkeypatch.setattr(pipeline_module, "ground_thread_clips", _fake_ground)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt="720": open(target_path, "wb").write(b"full source") or target_path,
    )
    monkeypatch.setattr(pipeline_module, "acquire_clip", lambda run_dir, source_url, cached_duration, start_time, end_time, out_path: open(out_path, "wb").write(b"clip") or {"clip_path": out_path})
    monkeypatch.setattr(pipeline_module, "synthesize_narration", lambda text, out_path, **k: open(out_path, "wb").write(b"audio") or out_path)
    monkeypatch.setattr(pipeline_module, "render_narration_card", lambda audio_path, text, out_path: open(out_path, "wb").write(b"card") or out_path)
    monkeypatch.setattr(pipeline_module, "assemble_thread", lambda segment_paths, out_path: open(out_path, "wb").write(b"final") or out_path)


def test_generate_threads_both_platforms_produce_distinct_named_files(tmp_path, monkeypatch):
    _setup_both_platform_thread_run(tmp_path, monkeypatch)

    result = pipeline_module.generate_threads(
        "https://example.com/a", "https://example.com/b", num_clips=1, platform="both", base_dir=str(tmp_path),
    )

    assert len(result) == 2
    out_dir = result[0]["output_dir"]
    assert {r["platform"] for r in result} == {"youtube", "tiktok"}
    youtube_result = next(r for r in result if r["platform"] == "youtube")
    tiktok_result = next(r for r in result if r["platform"] == "tiktok")
    assert youtube_result["clip_url"] == os.path.join(out_dir, "thesis_1_youtube_Title_youtube.mp4")
    assert tiktok_result["clip_url"] == os.path.join(out_dir, "thesis_1_tiktok_Title_tiktok.mp4")
    assert youtube_result["platform_index"] == 1
    assert tiktok_result["platform_index"] == 1
    assert youtube_result["episode_a"]["clip_url"] == os.path.join(out_dir, "raw", "thesis_1_youtube", "clip_1_a.mp4")
    assert tiktok_result["episode_a"]["clip_url"] == os.path.join(out_dir, "raw", "thesis_1_tiktok", "clip_1_a.mp4")


def test_generate_threads_both_platforms_call_find_same_topic_pairs_once(tmp_path, monkeypatch):
    _setup_both_platform_thread_run(tmp_path, monkeypatch)
    stage_a_calls = []
    original = pipeline_module.find_same_topic_pairs

    def _counting_find(entry_a, entry_b, num_clips, llm_fn):
        stage_a_calls.append(1)
        return ["Does X cause Y?"]

    monkeypatch.setattr(pipeline_module, "find_same_topic_pairs", _counting_find)

    pipeline_module.generate_threads(
        "https://example.com/a", "https://example.com/b", num_clips=1, platform="both", base_dir=str(tmp_path),
    )

    assert len(stage_a_calls) == 1


def test_generate_threads_returns_results_for_one_platform_when_other_grounds_nothing(tmp_path, monkeypatch):
    _setup_both_platform_thread_run(tmp_path, monkeypatch, tiktok_pairs=[])

    result = pipeline_module.generate_threads(
        "https://example.com/a", "https://example.com/b", num_clips=1, platform="both", base_dir=str(tmp_path),
    )

    assert len(result) == 1
    assert result[0]["platform"] == "youtube"


def test_generate_threads_returns_empty_list_when_all_platforms_ground_nothing(tmp_path, monkeypatch):
    episode_a_dir = tmp_path / "Episode_A"
    episode_b_dir = tmp_path / "Episode_B"
    episode_a_dir.mkdir()
    episode_b_dir.mkdir()
    (episode_a_dir / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))
    (episode_b_dir / "full_source.json").write_text(json.dumps({"duration": 200.0, "segments": []}))

    monkeypatch.setattr(pipeline_module, "ingest_captions", _fake_ingest_captions({
        "https://example.com/a": {"run_dir": str(episode_a_dir), "title": "Episode A", "duration": 100.0, "segment_count": 0},
        "https://example.com/b": {"run_dir": str(episode_b_dir), "title": "Episode B", "duration": 200.0, "segment_count": 0},
    }))
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: "an abstract")
    monkeypatch.setattr(pipeline_module, "find_same_topic_pairs", lambda entry_a, entry_b, num_clips, llm_fn: ["Does X cause Y?"])
    monkeypatch.setattr(pipeline_module, "ground_thread_clips", lambda entry_a, entry_b, transcript_a, transcript_b, shared_questions, num_clips, llm_fn, platform="youtube": [])

    result = pipeline_module.generate_threads(
        "https://example.com/a", "https://example.com/b", num_clips=1, platform="both", base_dir=str(tmp_path),
    )

    assert result == []
```

- [ ] **Step 5: Run the full pipeline test suite**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/pipeline.py tests/test_pipeline.py
git commit -m "$(cat <<'EOF'
feat: dual-platform loop in generate_threads

Adds platform="youtube"|"tiktok"|"both" to generate_threads. Runs
find_same_topic_pairs once regardless of platform count and reuses
it across each platform's own ground_thread_clips pass. Output
naming becomes thesis_{i}_{platform}_{title}.mp4 with raw
intermediates under raw/thesis_{i}_{platform}/ -- i is 1-indexed per
platform, not per flattened result list. A run aborts (returns [])
only when every requested platform grounds zero pairs.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: TikTok duration verification after assembly

**Files:**
- Modify: `shorts_generator/pipeline.py` (imports + `generate_threads` render loop)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests for the duration-warning helper**

Add to `tests/test_pipeline.py`:

```python
def test_warn_if_under_tiktok_minimum_logs_for_short_clip(monkeypatch, capsys):
    monkeypatch.setattr(pipeline_module, "_probe_local_duration", lambda path: 45.0)

    pipeline_module._warn_if_under_tiktok_minimum("/some/out/thesis_1_tiktok_Title.mp4")

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "thesis_1_tiktok_Title.mp4" in out
    assert "45.0s" in out


def test_warn_if_under_tiktok_minimum_silent_for_long_clip(monkeypatch, capsys):
    monkeypatch.setattr(pipeline_module, "_probe_local_duration", lambda path: 75.0)

    pipeline_module._warn_if_under_tiktok_minimum("/some/out/thesis_1_tiktok_Title.mp4")

    out = capsys.readouterr().out
    assert "WARNING" not in out


def test_warn_if_under_tiktok_minimum_swallows_probe_failure(monkeypatch, capsys):
    def _fail(path):
        raise RuntimeError("ffprobe not found")

    monkeypatch.setattr(pipeline_module, "_probe_local_duration", _fail)

    pipeline_module._warn_if_under_tiktok_minimum("/some/out/thesis_1_tiktok_Title.mp4")  # must not raise

    out = capsys.readouterr().out
    assert "WARNING" not in out
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_pipeline.py -k "warn_if_under_tiktok" -v`
Expected: FAIL — `AttributeError: module 'shorts_generator.pipeline' has no attribute '_warn_if_under_tiktok_minimum'`

- [ ] **Step 3: Add the import and the helper function**

In `shorts_generator/pipeline.py`, replace line 27:

```python
from .local.thread_source import acquire_clip
```

with:

```python
from .local.thread_source import _probe_local_duration, acquire_clip
```

Add this function directly above `def generate_threads(` in `shorts_generator/pipeline.py`:

```python
def _warn_if_under_tiktok_minimum(clip_path: str) -> None:
    """TikTok's Creator Rewards Program only pays out on videos 60s or
    longer -- thread_builder.PLATFORM_BOUNDS["tiktok"] steers the LLM's
    clip picks to make a sub-60s assembly unlikely, but narration audio
    length isn't independently bounded, so this is a defense-in-depth
    check on the actual assembled file, not a substitute for the picker's
    own bounds. Non-fatal: logs only, never deletes the file or aborts the
    run."""
    try:
        duration = _probe_local_duration(clip_path)
    except Exception:
        return
    if duration < 60.0:
        print(
            f"[pipeline/local] WARNING: TikTok cut {os.path.basename(clip_path)} is "
            f"{duration:.1f}s, under the 60s Creator Rewards minimum",
            flush=True,
        )
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `pytest tests/test_pipeline.py -k "warn_if_under_tiktok" -v`
Expected: PASS

- [ ] **Step 5: Wire the check into the render loop**

In `shorts_generator/pipeline.py`, inside `generate_threads`, immediately after the line:

```python
                    assemble_thread([intro_card, clip_a_path, bridge_card, clip_b_path], final_path)
```

add:

```python
                    if p == "tiktok":
                        _warn_if_under_tiktok_minimum(final_path)
```

- [ ] **Step 6: Write an integration test confirming the check only fires for tiktok clips**

Add to `tests/test_pipeline.py`:

```python
def test_generate_threads_only_probes_duration_for_tiktok_clips(tmp_path, monkeypatch):
    _setup_both_platform_thread_run(tmp_path, monkeypatch)
    probed_paths = []
    monkeypatch.setattr(pipeline_module, "_probe_local_duration", lambda path: probed_paths.append(path) or 75.0)

    pipeline_module.generate_threads(
        "https://example.com/a", "https://example.com/b", num_clips=1, platform="both", base_dir=str(tmp_path),
    )

    assert len(probed_paths) == 1
    assert "tiktok" in probed_paths[0]
```

- [ ] **Step 7: Run the full pipeline test suite**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add shorts_generator/pipeline.py tests/test_pipeline.py
git commit -m "$(cat <<'EOF'
feat: warn when an assembled TikTok cut lands under 60s

thread_builder's tiktok bounds steer the picker but don't bound
narration audio length, so a defense-in-depth ffprobe check runs
after assembly for tiktok-platform clips only. Non-fatal -- logs to
progress.log, doesn't delete the file or fail the run.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Per-platform numbering in `write_thread_descriptions`

**Files:**
- Modify: `shorts_generator/run_output.py:421-449`
- Test: `tests/test_run_output.py`

- [ ] **Step 1: Update the existing tests for the new format**

In `tests/test_run_output.py`, replace `test_write_thread_descriptions_formats_one_block_per_clip` (originally lines 370-380):

```python
def test_write_thread_descriptions_formats_one_block_per_clip(tmp_path):
    threads = [
        {"clip_url": "clip_1.mp4", "title": "Title One #Shorts", "description": "Watch clip one.", "platform": "youtube", "platform_index": 1},
        {"clip_url": "clip_2.mp4", "title": "Title Two #Shorts", "description": "Watch clip two.", "platform": "youtube", "platform_index": 2},
    ]
    path = run_output.write_thread_descriptions(str(tmp_path), threads)
    content = Path(path).read_text()
    assert content == (
        "clip 1 [youtube] (clip_1.mp4)\nTitle: Title One #Shorts\nDescription: Watch clip one.\n\n"
        "clip 2 [youtube] (clip_2.mp4)\nTitle: Title Two #Shorts\nDescription: Watch clip two.\n"
    )
```

Replace `test_write_thread_descriptions_skips_ungrounded_threads_without_renumbering` (originally lines 383-390):

```python
def test_write_thread_descriptions_skips_ungrounded_threads_without_renumbering(tmp_path):
    threads = [
        {"clip_url": None, "shared_question": "no clip made"},
        {"clip_url": "clip_2.mp4", "title": "Title Two #Shorts", "description": "Watch clip two.", "platform": "youtube", "platform_index": 2},
    ]
    path = run_output.write_thread_descriptions(str(tmp_path), threads)
    content = Path(path).read_text()
    assert content == "clip 2 [youtube] (clip_2.mp4)\nTitle: Title Two #Shorts\nDescription: Watch clip two.\n"
```

Replace `test_write_thread_descriptions_falls_back_to_shared_question` (originally lines 393-397):

```python
def test_write_thread_descriptions_falls_back_to_shared_question(tmp_path):
    threads = [{"clip_url": "clip_1.mp4", "shared_question": "Does X cause Y?", "description": "d", "platform": "youtube", "platform_index": 1}]
    path = run_output.write_thread_descriptions(str(tmp_path), threads)
    content = Path(path).read_text()
    assert content == "clip 1 [youtube] (clip_1.mp4)\nTitle: Does X cause Y?\nDescription: d\n"
```

Replace `test_write_thread_descriptions_uses_actual_clip_url_basename` (originally lines 400-409):

```python
def test_write_thread_descriptions_uses_actual_clip_url_basename(tmp_path):
    threads = [{
        "clip_url": "/some/output/_Threads/2026-08-18_a_x_b/thesis_1_tiktok_Is_AI_a_threat.mp4",
        "title": "Is AI a threat? #Shorts", "description": "Watch both takes.",
        "platform": "tiktok", "platform_index": 1,
    }]
    path = run_output.write_thread_descriptions(str(tmp_path), threads)
    content = Path(path).read_text()
    assert content == (
        "clip 1 [tiktok] (thesis_1_tiktok_Is_AI_a_threat.mp4)\nTitle: Is AI a threat? #Shorts\nDescription: Watch both takes.\n"
    )
```

Add a new test for restarted numbering across platforms:

```python
def test_write_thread_descriptions_numbers_restart_per_platform(tmp_path):
    threads = [
        {"clip_url": "thesis_1_youtube_A.mp4", "title": "A", "description": "d", "platform": "youtube", "platform_index": 1},
        {"clip_url": "thesis_1_tiktok_A.mp4", "title": "A", "description": "d", "platform": "tiktok", "platform_index": 1},
    ]
    path = run_output.write_thread_descriptions(str(tmp_path), threads)
    content = Path(path).read_text()
    assert content == (
        "clip 1 [youtube] (thesis_1_youtube_A.mp4)\nTitle: A\nDescription: d\n\n"
        "clip 1 [tiktok] (thesis_1_tiktok_A.mp4)\nTitle: A\nDescription: d\n"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_run_output.py -k thread_descriptions -v`
Expected: FAIL — assertion mismatches (old format has no `[platform]` tag)

- [ ] **Step 3: Update `write_thread_descriptions`**

In `shorts_generator/run_output.py`, replace `write_thread_descriptions` (originally lines 421-449):

```python
def write_thread_descriptions(out_dir: str, threads: List[Dict]) -> str:
    """Write a copy-paste-ready descriptions.txt next to a thread run's clip
    files. One block per thread that actually has a clip_url. Numbered by
    each entry's own "platform_index" (falling back to its position in
    `threads` when that key is absent, e.g. for a caller that never went
    through pipeline.generate_threads) rather than flat list position --
    generate_threads numbers each platform's own results from 1, so a
    flattened multi-platform list can contain two entries both carrying
    "platform_index": 1 (one youtube, one tiktok); using flat position here
    would drift from the "i" in each entry's own filename
    (thesis_{i}_{platform}_*.mp4), the exact class of bug fixed once
    already in commit ddda8cc. Each block is tagged with "[{platform}]" so
    same-numbered blocks from different platforms stay unambiguous. The
    filename shown is the clip_url's actual basename rather than a
    reconstructed pattern. Threads carry title/description at the top
    level (see thread_builder.ground_thread_clips) instead of
    per-highlight yt_title/yt_hashtags. Labeled "Title:"/"Description:" so
    each block can be copy-pasted straight into YouTube's or TikTok's
    upload form for that clip."""
    path = os.path.join(out_dir, "descriptions.txt")
    blocks = []
    for pos, t in enumerate(threads, 1):
        clip_url = t.get("clip_url")
        if not clip_url:
            continue
        index = t.get("platform_index") or pos
        platform = t.get("platform") or "youtube"
        title = (t.get("title") or t.get("shared_question") or "Untitled").strip()
        description = (t.get("description") or "").strip()
        blocks.append(f"clip {index} [{platform}] ({os.path.basename(clip_url)})\nTitle: {title}\nDescription: {description}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks))
        if blocks:
            f.write("\n")

    return path
```

- [ ] **Step 4: Run the full run_output test suite**

Run: `pytest tests/test_run_output.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py
git commit -m "$(cat <<'EOF'
fix: number thread descriptions.txt blocks per platform, not flat position

Prevents a multi-platform run's descriptions.txt numbering from
drifting from the filename index it's supposed to match (same bug
class as ddda8cc) now that a flattened result list can hold two
entries both indexed "1" for different platforms. Each block also
gets a [platform] tag to disambiguate.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `--platform` CLI flag

**Files:**
- Modify: `shorts_generator/../main.py` (repo root `main.py`)
- Test: `tests/test_main.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_main.py`:

```python
def test_platform_flag_defaults_to_youtube():
    args = build_parser().parse_args(["--clip-type", "thread"])
    assert args.platform == "youtube"


def test_platform_flag_accepts_tiktok_and_both():
    args = build_parser().parse_args(["--clip-type", "thread", "--platform", "tiktok"])
    assert args.platform == "tiktok"
    args = build_parser().parse_args(["--clip-type", "thread", "--platform", "both"])
    assert args.platform == "both"


def test_main_passes_platform_through_to_generate_threads(monkeypatch, capsys):
    captured_kwargs = {}

    def _fake_generate_threads(url_a, url_b, **kwargs):
        captured_kwargs.update(kwargs)
        return _fake_thread_results()

    monkeypatch.setattr(main_module, "generate_threads", _fake_generate_threads)
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread", "--platform", "tiktok"],
    )

    main()

    assert captured_kwargs["platform"] == "tiktok"


def test_main_warns_when_platform_passed_with_clip_type_shorts(monkeypatch, capsys):
    monkeypatch.setattr(
        main_module, "generate_shorts",
        lambda **kwargs: {"mode": "api", "output_dir": "d", "source_video_url": "u", "highlights": [], "shorts": []},
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "https://youtube.example/x", "--platform", "tiktok"])

    main()

    err = capsys.readouterr().err
    assert "--platform tiktok" in err


def test_main_does_not_warn_when_platform_omitted_with_clip_type_shorts(monkeypatch, capsys):
    monkeypatch.setattr(
        main_module, "generate_shorts",
        lambda **kwargs: {"mode": "api", "output_dir": "d", "source_video_url": "u", "highlights": [], "shorts": []},
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "https://youtube.example/x"])

    main()

    err = capsys.readouterr().err
    assert "--platform" not in err
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_main.py -k platform -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'platform'`

- [ ] **Step 3: Add the `--platform` argument**

In `main.py`, insert immediately after the `--clip-type` `parser.add_argument(...)` block (originally lines 97-106, ending `)`), before the `--num-chapters` block:

```python
    parser.add_argument(
        "--platform",
        choices=["youtube", "tiktok", "both"],
        default="youtube",
        help="--clip-type thread only: youtube (default, 45-60s target) or "
             "tiktok (65-90s target, code-enforced to clear TikTok's 60s "
             "Creator Rewards Program minimum), or both (produce one file "
             "of each).",
    )
```

- [ ] **Step 4: Pass `platform` through to `generate_threads` and add the mirror warning**

In `main.py`, replace the thread-dispatch line (originally line 196):

```python
            result = generate_threads(url_a=args.url, url_b=args.url_b, num_clips=args.num_clips)
```

with:

```python
            result = generate_threads(url_a=args.url, url_b=args.url_b, num_clips=args.num_clips, platform=args.platform)
```

In `main.py`, in the `if args.clip_type == "chapters":` block, add `--platform` to its `ignored_flags` list. Find this block (originally lines 166-192):

```python
    if args.clip_type == "chapters":
        # Only warn if --mode was explicitly typed and isn't "local" -- args.mode
        # defaults to "api" when omitted, and the plain, most natural invocation
        # (`--clip-type chapters` with no --mode at all) must not spuriously warn
        # on every single run just because the shorts-path default happens to
        # differ from what chapters always uses anyway.
        if "--mode" in sys.argv and args.mode != "local":
            print(f"[main] --clip-type chapters is local-only; ignoring --mode {args.mode!r} and using local", file=sys.stderr)
        ignored_flags = []
        if args.aspect_ratio != "9:16":
            ignored_flags.append(f"--aspect-ratio {args.aspect_ratio}")
        if args.framing != "locked":
            ignored_flags.append(f"--framing {args.framing}")
        if args.hook_card is False:
            ignored_flags.append("--no-hook-card")
        if args.end_card is True:
            ignored_flags.append("--end-card")
        if args.num_clips != 3:
            ignored_flags.append(f"--num-clips {args.num_clips}")
        if args.filename_style is not None:
            ignored_flags.append(f"--filename-style {args.filename_style}")
        if ignored_flags:
            print(
                f"[main] --clip-type chapters ignores: {', '.join(ignored_flags)} "
                "(no crop, no card overlays in this path)",
                file=sys.stderr,
            )
```

Replace it with (adding the `--platform` check to `ignored_flags` and a new `elif` branch for `shorts` right after):

```python
    if args.clip_type == "chapters":
        # Only warn if --mode was explicitly typed and isn't "local" -- args.mode
        # defaults to "api" when omitted, and the plain, most natural invocation
        # (`--clip-type chapters` with no --mode at all) must not spuriously warn
        # on every single run just because the shorts-path default happens to
        # differ from what chapters always uses anyway.
        if "--mode" in sys.argv and args.mode != "local":
            print(f"[main] --clip-type chapters is local-only; ignoring --mode {args.mode!r} and using local", file=sys.stderr)
        ignored_flags = []
        if args.aspect_ratio != "9:16":
            ignored_flags.append(f"--aspect-ratio {args.aspect_ratio}")
        if args.framing != "locked":
            ignored_flags.append(f"--framing {args.framing}")
        if args.hook_card is False:
            ignored_flags.append("--no-hook-card")
        if args.end_card is True:
            ignored_flags.append("--end-card")
        if args.num_clips != 3:
            ignored_flags.append(f"--num-clips {args.num_clips}")
        if args.filename_style is not None:
            ignored_flags.append(f"--filename-style {args.filename_style}")
        if args.platform != "youtube":
            ignored_flags.append(f"--platform {args.platform}")
        if ignored_flags:
            print(
                f"[main] --clip-type chapters ignores: {', '.join(ignored_flags)} "
                "(no crop, no card overlays in this path)",
                file=sys.stderr,
            )
    elif args.clip_type == "shorts" and args.platform != "youtube":
        # --platform is thread-only -- everywhere else, silently doing
        # nothing with an explicitly-passed flag would be surprising, so
        # warn the same way the thread/chapters blocks above warn about
        # their own out-of-scope flags.
        print(
            f"[main] --clip-type shorts ignores: --platform {args.platform} "
            "(platform only applies to --clip-type thread)",
            file=sys.stderr,
        )
```

- [ ] **Step 5: Run the full main test suite**

Run: `pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "$(cat <<'EOF'
feat: --platform CLI flag for thread mode

--platform youtube|tiktok|both, thread-only, default youtube (no
behavior change for existing invocations). Mirrors the existing
ignored-flags warning pattern in the other direction: passing
--platform with --clip-type shorts/chapters now warns instead of
silently no-op'ing.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Commit 2: Webapp / dashboard UI (Tasks 7-8)

Lands after Commit 1 is verified on its own — see spec's Rollout section. Only start this once Tasks 1-6 are committed and green.

### Task 7: `webapp.py` platform support

**Files:**
- Modify: `shorts_generator/webapp.py:93-224`
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Update existing fake `generate_threads` signatures and write new tests**

In `tests/test_webapp.py`, update all four `_fake_generate_threads` definitions (originally at lines 126, 161, 204, 244) to accept `platform`. Each currently reads:

```python
    def _fake_generate_threads(url_a, url_b, num_clips=1, base_dir=None, on_output_dir=None):
```

Replace each occurrence with:

```python
    def _fake_generate_threads(url_a, url_b, num_clips=1, platform="youtube", base_dir=None, on_output_dir=None):
```

Update the inline lambda in `test_run_thread_fails_with_a_helpful_message_when_no_pair_found` (originally line 143):

```python
    monkeypatch.setattr(webapp, "generate_threads", lambda url_a, url_b, num_clips=1, base_dir=None, on_output_dir=None: [])
```

Replace with:

```python
    monkeypatch.setattr(webapp, "generate_threads", lambda url_a, url_b, num_clips=1, platform="youtube", base_dir=None, on_output_dir=None: [])
```

In `test_status_serializes_thread_results_and_omits_run_name` (originally lines 152-188), add a `"platform"` key to the fake result dict and assert it round-trips. The fake result dict (inside `_fake_generate_threads`, originally lines 164-172) becomes:

```python
        return [{
            "shared_question": "Does X cause Y?",
            "thesis": "Two guests disagree.",
            "bridge": "Here's the other side.",
            "platform": "youtube",
            "episode_a": {"title": "Episode A", "clip_url": clip_a_path},
            "episode_b": {"title": "Episode B", "clip_url": clip_b_path},
            "output_dir": out_dir,
            "clip_url": clip_path,
        }]
```

and, after the existing assertions in that test (originally lines 185-188), add:

```python
    assert thread["platform"] == "youtube"
```

Add new tests for platform pass-through and validation:

```python
def test_run_thread_passes_platform_through_to_generate_threads(client, monkeypatch):
    captured = {}

    def _fake_generate_threads(url_a, url_b, num_clips=1, platform="youtube", base_dir=None, on_output_dir=None):
        captured["platform"] = platform
        return []

    monkeypatch.setattr(webapp, "generate_threads", _fake_generate_threads)
    monkeypatch.setattr(webapp.threading, "Thread", _SyncThread)

    client.post("/run", data={
        "clip_type": "thread", "url_a": "https://example.com/a", "url_b": "https://example.com/b", "platform": "tiktok",
    })

    assert captured["platform"] == "tiktok"


def test_run_thread_defaults_platform_to_youtube(client, monkeypatch):
    captured = {}

    def _fake_generate_threads(url_a, url_b, num_clips=1, platform="youtube", base_dir=None, on_output_dir=None):
        captured["platform"] = platform
        return []

    monkeypatch.setattr(webapp, "generate_threads", _fake_generate_threads)
    monkeypatch.setattr(webapp.threading, "Thread", _SyncThread)

    client.post("/run", data={"clip_type": "thread", "url_a": "https://example.com/a", "url_b": "https://example.com/b"})

    assert captured["platform"] == "youtube"


def test_run_thread_rejects_invalid_platform(client):
    resp = client.post("/run", data={
        "clip_type": "thread", "url_a": "https://example.com/a", "url_b": "https://example.com/b", "platform": "instagram",
    })
    assert resp.status_code == 400
    assert "platform" in resp.get_json()["error"]


def test_status_thread_source_clip_under_platform_suffixed_raw_subfolder_gets_relative_download_url(client, monkeypatch, tmp_path):
    """episode_a/episode_b source clips now live under
    out_dir/raw/thesis_N_{platform}/ (see pipeline.generate_threads'
    platform-suffixed naming) -- the download URL must carry that relative
    path. _relative_clip_path/_clip_display_url were already fixed once for
    nested raw/thesis_N/ paths (95fa8b1); this pins the same fix under the
    deeper platform-suffixed nesting so it doesn't silently regress."""
    out_dir = str(tmp_path / "_Threads" / "2026-08-20_a_x_b")
    raw_dir = os.path.join(out_dir, "raw", "thesis_1_tiktok")
    os.makedirs(raw_dir, exist_ok=True)
    clip_path = os.path.join(out_dir, "thesis_1_tiktok_Title.mp4")
    clip_a_path = os.path.join(raw_dir, "clip_1_a.mp4")
    clip_b_path = os.path.join(raw_dir, "clip_1_b.mp4")
    for p in (clip_path, clip_a_path, clip_b_path):
        open(p, "wb").write(b"data")

    def _fake_generate_threads(url_a, url_b, num_clips=1, platform="youtube", base_dir=None, on_output_dir=None):
        if on_output_dir:
            on_output_dir(out_dir)
        return [{
            "shared_question": "Does X cause Y?",
            "thesis": "Two guests disagree.",
            "bridge": "Here's the other side.",
            "platform": "tiktok",
            "episode_a": {"title": "Episode A", "clip_url": clip_a_path},
            "episode_b": {"title": "Episode B", "clip_url": clip_b_path},
            "output_dir": out_dir,
            "clip_url": clip_path,
        }]

    monkeypatch.setattr(webapp, "generate_threads", _fake_generate_threads)
    monkeypatch.setattr(webapp.threading, "Thread", _SyncThread)

    client.post("/run", data={"clip_type": "thread", "url_a": "https://example.com/a", "url_b": "https://example.com/b", "platform": "tiktok"})
    resp = client.get("/status")
    data = resp.get_json()

    thread = data["result"]["threads"][0]
    assert thread["download_url"] == "/download/thesis_1_tiktok_Title.mp4"
    assert thread["episode_a_download_url"] == "/download/raw/thesis_1_tiktok/clip_1_a.mp4"
    assert thread["episode_b_download_url"] == "/download/raw/thesis_1_tiktok/clip_1_b.mp4"
```

- [ ] **Step 2: Run the new/updated tests to verify they fail**

Run: `pytest tests/test_webapp.py -k "thread" -v`
Expected: FAIL — `_run_thread_job() got an unexpected keyword argument 'platform'` and the new tests' `platform` assertions/400s not yet implemented

- [ ] **Step 3: Update `_run_thread_job` and the `/run` route**

In `shorts_generator/webapp.py`, replace `_run_thread_job` (originally lines 93-123):

```python
def _run_thread_job(url_a: str, url_b: str, num_clips: int, platform: str = "youtube") -> None:
    """Ingests url_a/url_b caption-only (no video download) and builds up
    to num_clips distinct shared-question threads between them -- see
    generate_threads in pipeline.py. platform selects "youtube" (default),
    "tiktok", or "both" -- see generate_threads. Like _run_job, the output
    dir isn't known until generate_threads has resolved it from the two
    episode titles, so job.progress_log/shorts_dir are set via the
    on_output_dir callback."""
    def _on_output_dir(out_dir: str) -> None:
        with _job_lock:
            job.status = "running"
            job.progress_log = os.path.join(out_dir, "progress.log")
            job.shorts_dir = out_dir

    try:
        result = generate_threads(url_a, url_b, num_clips=num_clips, platform=platform, on_output_dir=_on_output_dir)
        with _job_lock:
            if not result:
                job.error = (
                    "No shared-question thread found between these two episodes -- "
                    "try a different pair, or make sure both URLs genuinely cover the "
                    "same topic."
                )
                job.status = "failed"
            else:
                job.result = result
                job.status = "done"
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        with _job_lock:
            job.error = str(e)
            job.status = "failed"
```

In `shorts_generator/webapp.py`, replace the thread branch of `start_run` (originally lines 254-276):

```python
    if clip_type == "thread":
        url_a = request.form.get("url_a", "").strip()
        url_b = request.form.get("url_b", "").strip()
        if not url_a or not url_b:
            return jsonify({"error": "url_a and url_b are both required for clip_type=thread"}), 400
        try:
            num_clips = int(request.form.get("num_clips", 2))
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"invalid input: {e}"}), 400

        with _job_lock:
            if job.status in ("starting", "running"):
                return jsonify({"error": "a run is already in progress"}), 409
            job.status = "starting"
            job.url = ""
            job.clip_type = "thread"
            job.progress_log = None
            job.shorts_dir = None
            job.result = None
            job.error = None

        threading.Thread(target=_run_thread_job, args=(url_a, url_b, num_clips), daemon=True).start()
        return jsonify({"status": "starting"}), 202
```

with:

```python
    if clip_type == "thread":
        url_a = request.form.get("url_a", "").strip()
        url_b = request.form.get("url_b", "").strip()
        if not url_a or not url_b:
            return jsonify({"error": "url_a and url_b are both required for clip_type=thread"}), 400
        try:
            num_clips = int(request.form.get("num_clips", 2))
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"invalid input: {e}"}), 400
        platform = request.form.get("platform", "youtube")
        if platform not in ("youtube", "tiktok", "both"):
            return jsonify({"error": f"invalid platform: {platform!r}"}), 400

        with _job_lock:
            if job.status in ("starting", "running"):
                return jsonify({"error": "a run is already in progress"}), 409
            job.status = "starting"
            job.url = ""
            job.clip_type = "thread"
            job.progress_log = None
            job.shorts_dir = None
            job.result = None
            job.error = None

        threading.Thread(target=_run_thread_job, args=(url_a, url_b, num_clips, platform), daemon=True).start()
        return jsonify({"status": "starting"}), 202
```

- [ ] **Step 4: Add `platform` to serialized thread results**

In `shorts_generator/webapp.py`, in `_serialize_thread_results` (originally lines 192-224), replace the `threads.append({...})` block:

```python
        threads.append({
            "shared_question": r.get("shared_question"),
            "thesis": r.get("thesis"),
            "bridge": r.get("bridge"),
            "title": r.get("title"),
            "description": r.get("description"),
            "episode_a": r.get("episode_a"),
            "episode_b": r.get("episode_b"),
            "download_url": _clip_display_url(out_dir, clip_url),
            "episode_a_download_url": episode_a_download_url,
            "episode_b_download_url": episode_b_download_url,
        })
```

with:

```python
        threads.append({
            "shared_question": r.get("shared_question"),
            "thesis": r.get("thesis"),
            "bridge": r.get("bridge"),
            "title": r.get("title"),
            "description": r.get("description"),
            "platform": r.get("platform"),
            "episode_a": r.get("episode_a"),
            "episode_b": r.get("episode_b"),
            "download_url": _clip_display_url(out_dir, clip_url),
            "episode_a_download_url": episode_a_download_url,
            "episode_b_download_url": episode_b_download_url,
        })
```

- [ ] **Step 5: Run the full webapp test suite**

Run: `pytest tests/test_webapp.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/webapp.py tests/test_webapp.py
git commit -m "$(cat <<'EOF'
feat: platform support in the thread dashboard job/route

/run accepts platform=youtube|tiktok|both for clip_type=thread
(default youtube, 400 on an invalid value), threaded through
_run_thread_job into generate_threads. _serialize_thread_results now
carries platform per thread.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Platform selector in the dashboard UI

**Files:**
- Modify: `shorts_generator/templates/index.html:607-617`
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write a failing test for the new form control**

Add to `tests/test_webapp.py`:

```python
def test_index_thread_platform_select_present(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b'id="platform"' in resp.data
    assert b'value="tiktok"' in resp.data
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_webapp.py -k platform_select -v`
Expected: FAIL — `AssertionError: b'id="platform"' not found`

- [ ] **Step 3: Add the platform select to the thread-fields block**

In `shorts_generator/templates/index.html`, replace the `thread-fields` div (originally lines 607-617):

```html
          <div id="thread-fields" hidden>
            <label for="url_a">Episode A URL</label>
            <input type="text" id="url_a" name="url_a" placeholder="https://www.youtube.com/watch?v=...">
            <label for="url_b">Episode B URL</label>
            <input type="text" id="url_b" name="url_b" placeholder="https://www.youtube.com/watch?v=...">
            <p style="opacity:0.7; font-size:0.9em;">
              Captions only are fetched for both URLs (no video download) to find shared-question
              topics between them. Up to "Num clips" distinct threads are built, each cutting a
              matched span from both episodes around a narrated bridge.
            </p>
          </div>
```

with:

```html
          <div id="thread-fields" hidden>
            <label for="url_a">Episode A URL</label>
            <input type="text" id="url_a" name="url_a" placeholder="https://www.youtube.com/watch?v=...">
            <label for="url_b">Episode B URL</label>
            <input type="text" id="url_b" name="url_b" placeholder="https://www.youtube.com/watch?v=...">
            <label for="platform">Platform</label>
            <select id="platform" name="platform">
              <option value="youtube">YouTube (45-60s)</option>
              <option value="tiktok">TikTok (65-90s, Creator Rewards eligible)</option>
              <option value="both">Both</option>
            </select>
            <p style="opacity:0.7; font-size:0.9em;">
              Captions only are fetched for both URLs (no video download) to find shared-question
              topics between them. Up to "Num clips" distinct threads are built, each cutting a
              matched span from both episodes around a narrated bridge. TikTok's Creator Rewards
              Program only pays out on videos 60s or longer, so the TikTok cut targets a longer
              span than the YouTube one.
            </p>
          </div>
```

- [ ] **Step 4: Run the webapp test suite**

Run: `pytest tests/test_webapp.py -v`
Expected: PASS

- [ ] **Step 5: Manually verify in the browser**

Run: `python -m shorts_generator.webapp` (or however the project normally starts the dev server — check `README.md` for the exact command), open the dashboard, select "Thread" as the clip type, and confirm the new "Platform" dropdown appears with YouTube/TikTok/Both options and submits correctly (check the `/run` POST body in devtools network tab includes `platform`).

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/templates/index.html tests/test_webapp.py
git commit -m "$(cat <<'EOF'
feat: platform selector in the thread dashboard form

YouTube / TikTok / Both dropdown next to the existing episode A/B
URL fields, submitted as `platform` in the /run POST body.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Plan self-review notes

- **Spec coverage:** Section 1 (bounds) → Task 1. Section 2 (`ground_thread_clips`) → Task 2. Section 3 (`generate_threads` loop/naming/abort condition) → Task 3. Section 4 (output naming/descriptions numbering) → Tasks 3 + 5. Section 5 (duration verification) → Task 4. Section 6 (CLI) → Task 6. Section 7 (webapp) → Tasks 7-8. Rollout (two commits) → Commit 1 / Commit 2 split above. Testing plan → covered by each task's Step list.
- **Type consistency checked:** `pick_thread_clips(..., platform: str = "youtube")` (Task 1) → `ground_thread_clips(..., platform: str = "youtube")` (Task 2) → `pairs_by_platform[p] = ground_thread_clips(..., platform=p)` (Task 3) all use the same parameter name and default. `platform`/`platform_index` keys set in Task 3's `results.append` match the keys read in Task 5's `write_thread_descriptions`. `_run_thread_job(..., platform: str = "youtube")` (Task 7) matches `generate_threads(..., platform=platform, ...)`'s call in the same task.
- **No placeholders:** every step above shows the complete function/test body being written, not a description of it.
