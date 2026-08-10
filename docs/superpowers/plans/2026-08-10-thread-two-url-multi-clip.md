# Thread: Two-URL Input + Multi-Clip Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace thread mode's no-input "scan the whole local corpus" flow with two explicit YouTube URL fields (dashboard + CLI), reusing caption-only ingest so no full video download is needed, and produce up to N distinct shared-question thread clips per run (reusing the existing "Num clips" field) instead of always exactly one.

**Architecture:** `thread_builder.py` gets a fixed-pair multi-question stage A (`find_same_topic_pairs`) and a driving loop (`select_thread_pairs`) that calls stage B (`pick_thread_clips`, now overlap-aware) once per question, discarding spans that collide with an earlier accepted pick. `pipeline.generate_threads(url_a, url_b, num_clips, ...)` ingests both URLs via `local/caption_ingest.py`, hands the fixed pair to `select_thread_pairs`, then renders each grounded pair into one shared `output/_Threads/<slug>/` run directory as `clip_{i}.mp4` + `clip_{i}_a.mp4` + `clip_{i}_b.mp4`. The dashboard, CLI, and their existing tests are updated to the new signature; the old no-URL corpus-scan path (`build_thread`, `find_same_topic_pair` singular, `resolve_thread_output_dir`) is deleted, not kept as a fallback.

**Tech Stack:** Python (Flask dashboard, argparse CLI, pytest), vanilla JS/HTML template, ffmpeg (unchanged, via existing `local/thread_source.py` / `local/thread_assembler.py`).

**Spec:** `docs/superpowers/specs/2026-08-10-thread-two-url-multi-clip-design.md`

---

## Before you start

- **No real download, but not free.** `local/thread_source.py`'s `acquire_clip` only skips its network fallback when `full_source.mp4` already exists locally. A caption-ingested episode has no `full_source.mp4` by design, so every clip in this feature takes `acquire_clip`'s existing re-download-just-the-span + local Whisper (`model_size="small"`) fallback path — that part is unchanged by this plan, just now the *normal* path for thread clips instead of the rare one. N=3 clips means up to 6 padded-span downloads and 6 local Whisper transcriptions. This is correct, expected behavior (the duration-mismatch check will pass, since the cached duration and the live-probe duration both come from the same `yt-dlp --print duration` call) — not a regression to fix.
- **Task 0 must run before any other task touches git.** The working tree already has an uncommitted one-line change in `webapp.py` and three untracked files (`ingest_corpus.py`, `shorts_generator/local/caption_ingest.py`, `tests/test_local_caption_ingest.py`) from a prior session. If left dirty, the first commit any later task makes will accidentally bundle them in. Task 0 commits them on their own first.

---

### Task 0: Commit pre-existing uncommitted work

**Files:**
- (none created/modified — this stages and commits what's already on disk)

- [ ] **Step 1: Confirm the exact dirty state**

Run: `git status --short`
Expected:
```
 M shorts_generator/webapp.py
?? ingest_corpus.py
?? shorts_generator/local/caption_ingest.py
?? tests/test_local_caption_ingest.py
```
If the output differs from this, stop and check with the user before proceeding — do not blind-commit unexpected files.

- [ ] **Step 2: Stage and commit exactly these four files**

```bash
git add shorts_generator/webapp.py ingest_corpus.py shorts_generator/local/caption_ingest.py tests/test_local_caption_ingest.py
git commit -m "$(cat <<'EOF'
feat: add caption-only corpus ingest, fix thread job status race

ingest_corpus.py / local/caption_ingest.py grow the local corpus from
YouTube auto-captions only, no video download, for --clip-type thread.
webapp.py's thread job now sets status="running" from the on_output_dir
callback, matching the shorts job's own status transition.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify a clean tree**

Run: `git status --short`
Expected: no output (clean).

---

### Task 1: Ingest guard — don't downgrade an already-ingested episode

**Files:**
- Modify: `shorts_generator/local/caption_ingest.py`
- Test: `tests/test_local_caption_ingest.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_local_caption_ingest.py`:

```python
def test_ingest_captions_skips_fetch_when_already_ingested(tmp_path, monkeypatch):
    run_dir = os.path.join(str(tmp_path), "already")
    os.makedirs(run_dir, exist_ok=True)
    existing = {"duration": 555.0, "segments": [{"start": 0.0, "end": 1.0, "text": "already here"}]}
    with open(os.path.join(run_dir, "full_source.json"), "w", encoding="utf-8") as f:
        json.dump(existing, f)
    with open(os.path.join(run_dir, "source_url.txt"), "w", encoding="utf-8") as f:
        f.write("https://www.youtube.com/watch?v=already")

    def _fail(*a, **k):
        pytest.fail("fetch functions should not be called when the episode is already ingested")

    monkeypatch.setattr(caption_ingest_module, "_fetch_duration", _fail)
    monkeypatch.setattr(caption_ingest_module, "_fetch_srv1_captions", _fail)

    result = ingest_captions("https://www.youtube.com/watch?v=already", base_dir=str(tmp_path))

    assert result["run_dir"] == run_dir
    assert result["duration"] == 555.0
    assert result["segment_count"] == 1
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_local_caption_ingest.py::test_ingest_captions_skips_fetch_when_already_ingested -v`
Expected: FAIL (the fake `_fetch_duration`/`_fetch_srv1_captions` get called, `pytest.fail` raises).

- [ ] **Step 3: Add the guard**

In `shorts_generator/local/caption_ingest.py`, replace the body of `ingest_captions`:

```python
def ingest_captions(youtube_url: str, base_dir: Optional[str] = None) -> Dict:
    """Add one episode to the corpus from its auto-captions only. Writes
    full_source.json + source_url.txt (see corpus.list_corpus_run_dirs --
    both are required for corpus eligibility) but never full_source.mp4.

    Idempotent: if this URL's run dir already has a full_source.json --
    whether from a prior full pipeline run (real Whisper transcript,
    possibly with full_source.mp4 still on disk) or a prior caption-only
    ingest -- the fetch is skipped entirely and the existing transcript is
    reused as-is. Without this guard, re-ingesting an already-fully-
    processed episode would silently downgrade its real transcript to
    lower-fidelity YouTube auto-captions."""
    paths: RunPaths = resolve_output_dir(youtube_url, base_dir=base_dir)

    if os.path.exists(paths.source_json):
        print(f"[caption_ingest] {youtube_url!r} already in the corpus at {paths.root} -- skipping caption fetch", flush=True)
        with open(paths.source_json, "r", encoding="utf-8") as f:
            existing = json.load(f)
        return {
            "run_dir": paths.root,
            "title": os.path.basename(paths.root),
            "duration": existing.get("duration", 0.0),
            "segment_count": len(existing.get("segments", [])),
        }

    duration = _fetch_duration(youtube_url)
    xml_text = _fetch_srv1_captions(youtube_url)
    segments = _parse_srv1(xml_text)
    if not segments:
        raise RuntimeError(f"no caption text parsed for {youtube_url!r}")

    with open(paths.source_json, "w", encoding="utf-8") as f:
        json.dump({"duration": duration, "segments": segments}, f, ensure_ascii=False, indent=2)
    write_source_url(paths, youtube_url)

    return {
        "run_dir": paths.root,
        "title": os.path.basename(paths.root),
        "duration": duration,
        "segment_count": len(segments),
    }
```

- [ ] **Step 4: Run the full ingest test file to verify everything passes**

Run: `pytest tests/test_local_caption_ingest.py -v`
Expected: all PASS, including the new test and the pre-existing `test_ingest_captions_writes_transcript_and_source_url` (which still exercises the fresh-fetch path since it starts from an empty `tmp_path`).

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/local/caption_ingest.py tests/test_local_caption_ingest.py
git commit -m "$(cat <<'EOF'
fix: skip caption re-fetch for an already-ingested episode

ingest_captions() always overwrote full_source.json, which would
silently downgrade a real Whisper transcript to auto-captions if the
same URL got caption-ingested a second time. Now a no-op reusing the
cached transcript when one already exists.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `thread_builder.py` — multi-question stage A for a fixed pair

**Files:**
- Modify: `shorts_generator/thread_builder.py`
- Test: `tests/test_thread_builder.py`

- [ ] **Step 1: Remove the tests for the whole-corpus-scan stage A (being replaced)**

In `tests/test_thread_builder.py`, delete these 7 test functions (they test `find_same_topic_pair`, the singular whole-corpus scanner being replaced by the fixed-pair `find_same_topic_pairs` below):
- `test_find_same_topic_pair_returns_none_on_no_match_response`
- `test_find_same_topic_pair_returns_none_with_fewer_than_two_entries`
- `test_find_same_topic_pair_returns_pick_on_valid_match`
- `test_find_same_topic_pair_rejects_out_of_range_indices`
- `test_find_same_topic_pair_rejects_missing_shared_question`
- `test_find_same_topic_pair_rejects_non_string_shared_question`
- `test_find_same_topic_pair_returns_none_on_malformed_llm_output`

Leave `_corpus_entry`, `_episode`, and every `pick_thread_clips`/`build_thread` test in place for now — `build_thread` tests are removed in Task 4 once `select_thread_pairs` replaces it.

- [ ] **Step 2: Write the new failing tests**

Add to `tests/test_thread_builder.py`:

```python
def test_find_same_topic_pairs_returns_up_to_num_pairs_questions():
    entry_a = _corpus_entry(0, "Ep A", "discusses remote work and also housing policy", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "discusses remote work and also housing policy", "/tmp/b")
    llm_fn = lambda prompt: json.dumps({"shared_questions": [
        "Does remote work increase productivity?",
        "Does zoning reform lower housing costs?",
        "A third question that should be dropped",
    ]})

    result = thread_builder.find_same_topic_pairs(entry_a, entry_b, num_pairs=2, llm_fn=llm_fn)

    assert result == [
        "Does remote work increase productivity?",
        "Does zoning reform lower housing costs?",
    ]


def test_find_same_topic_pairs_drops_non_string_items():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    llm_fn = lambda prompt: json.dumps({"shared_questions": ["A real question?", ["not", "a", "string"], "", "   "]})

    result = thread_builder.find_same_topic_pairs(entry_a, entry_b, num_pairs=5, llm_fn=llm_fn)

    assert result == ["A real question?"]


def test_find_same_topic_pairs_dedupes_case_insensitive_duplicates():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    llm_fn = lambda prompt: json.dumps({"shared_questions": [
        "Does X cause Y?", "does x cause y?", "Does X Cause Y?",
    ]})

    result = thread_builder.find_same_topic_pairs(entry_a, entry_b, num_pairs=5, llm_fn=llm_fn)

    assert result == ["Does X cause Y?"]


def test_find_same_topic_pairs_returns_empty_list_on_malformed_llm_output():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    llm_fn = lambda prompt: "not json at all"

    assert thread_builder.find_same_topic_pairs(entry_a, entry_b, num_pairs=3, llm_fn=llm_fn) == []


def test_find_same_topic_pairs_returns_empty_list_when_num_pairs_less_than_one():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    llm_fn = lambda prompt: pytest.fail("llm_fn should not be called when num_pairs < 1")

    assert thread_builder.find_same_topic_pairs(entry_a, entry_b, num_pairs=0, llm_fn=llm_fn) == []
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_thread_builder.py -k find_same_topic_pairs -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.thread_builder' has no attribute 'find_same_topic_pairs'`.

- [ ] **Step 4: Implement `find_same_topic_pairs`**

In `shorts_generator/thread_builder.py`, add the new prompt constant right after `SAME_TOPIC_SYSTEM_PROMPT` and before `THREAD_PICK_SYSTEM_PROMPT`:

```python
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
```

Add this sanitizer right after `_sanitize_topic_pick`:

```python
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
```

Add this function right after `find_same_topic_pair`:

```python
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
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_thread_builder.py -k find_same_topic_pairs -v`
Expected: all 5 PASS.

- [ ] **Step 6: Run the whole file to make sure nothing else broke**

Run: `pytest tests/test_thread_builder.py -v`
Expected: PASS (the removed 7 tests are gone; `pick_thread_clips`/`build_thread` tests untouched still pass).

- [ ] **Step 7: Commit**

```bash
git add shorts_generator/thread_builder.py tests/test_thread_builder.py
git commit -m "$(cat <<'EOF'
feat: add find_same_topic_pairs, a fixed-pair multi-question stage A

Replaces find_same_topic_pair's whole-corpus single-best-pair scan
with a version that takes exactly two episodes (already fixed by the
caller) and returns every genuinely distinct shared question between
them, up to a requested count -- feeds the new multi-clip thread loop.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `thread_builder.py` — overlap-aware stage B

**Files:**
- Modify: `shorts_generator/thread_builder.py`
- Test: `tests/test_thread_builder.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_thread_builder.py`:

```python
def test_pick_thread_clips_includes_avoid_ranges_in_prompt_when_given():
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    seen_prompts = []

    def llm_fn(prompt):
        seen_prompts.append(prompt)
        return json.dumps({
            "grounded": True, "thesis": "t", "bridge": "b",
            "clip_a": {"start_time": 5.0, "end_time": 25.0},
            "clip_b": {"start_time": 2.0, "end_time": 20.0},
        })

    thread_builder.pick_thread_clips(
        episode_a, episode_b, "q?", llm_fn,
        avoid_ranges_a=[(0.0, 10.0)], avoid_ranges_b=[(50.0, 60.0)],
    )

    assert "0.0s-10.0s" in seen_prompts[0]
    assert "50.0s-60.0s" in seen_prompts[0]


def test_pick_thread_clips_prompt_has_no_avoid_block_when_none_given():
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    seen_prompts = []

    def llm_fn(prompt):
        seen_prompts.append(prompt)
        return json.dumps({
            "grounded": True, "thesis": "t", "bridge": "b",
            "clip_a": {"start_time": 5.0, "end_time": 25.0},
            "clip_b": {"start_time": 2.0, "end_time": 20.0},
        })

    thread_builder.pick_thread_clips(episode_a, episode_b, "q?", llm_fn)

    assert "already-used" not in seen_prompts[0]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_thread_builder.py -k avoid_ranges -v`
Expected: FAIL with `TypeError: pick_thread_clips() got an unexpected keyword argument 'avoid_ranges_a'`.

- [ ] **Step 3: Implement**

In `shorts_generator/thread_builder.py`, change the `typing` import line at the top:

```python
from typing import Dict, List, Optional, Tuple
```

Update `THREAD_PICK_SYSTEM_PROMPT` to insert an `{avoid_block}` placeholder right after the shared-question line:

```python
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
```

Add this helper right before `pick_thread_clips`:

```python
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
```

Update `pick_thread_clips`'s signature and prompt build:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_thread_builder.py -v`
Expected: all PASS (the two new tests, plus every pre-existing `pick_thread_clips` test still passing unchanged since the new params are optional).

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/thread_builder.py tests/test_thread_builder.py
git commit -m "$(cat <<'EOF'
feat: make pick_thread_clips overlap-aware via optional avoid_ranges

Steers stage B away from spans an earlier accepted pick in the same
thread run already used, via explicit prompt text. The caller (Task 4)
still enforces non-overlap itself as a hard backstop.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `thread_builder.py` — the multi-pair selection loop, remove the old whole-corpus path

**Files:**
- Modify: `shorts_generator/thread_builder.py`
- Test: `tests/test_thread_builder.py`

- [ ] **Step 1: Remove the tests for the whole-corpus `build_thread` (being replaced)**

In `tests/test_thread_builder.py`, delete every test whose name starts with `test_build_thread_` (there are 6: `_returns_none_when_no_same_topic_pair`, `_returns_none_when_corpus_has_fewer_than_two_episodes`, `_returns_full_shape_on_qualifying_pair`, `_returns_none_when_picked_run_dir_missing_full_source_json`, `_returns_none_when_full_source_json_is_unparseable`, `_returns_none_when_transcript_shape_is_malformed`, `_returns_none_when_transcript_duration_is_null`) — that's 7, all of them. Also delete the now-unused `_topic_gate_llm_response` helper if nothing else references it after this removal.

- [ ] **Step 2: Write the new failing tests**

Add to `tests/test_thread_builder.py`:

```python
def _transcript(duration, texts_with_times):
    segments = [{"start": s, "end": e, "text": t} for s, e, t in texts_with_times]
    return {"duration": duration, "segments": segments}


def test_select_thread_pairs_returns_empty_list_when_no_shared_questions():
    entry_a = _corpus_entry(0, "Ep A", "unrelated topic one", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "unrelated topic two", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "world")])
    llm_fn = lambda prompt: json.dumps({"shared_questions": []})

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=2, llm_fn=llm_fn)

    assert result == []


def test_select_thread_pairs_returns_one_grounded_pair():
    entry_a = _corpus_entry(0, "Ep A", "argues remote work increases productivity", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "argues remote work decreases productivity", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    responses = [
        json.dumps({"shared_questions": ["Does remote work increase or decrease productivity?"]}),
        json.dumps({
            "grounded": True, "thesis": "Two guests, one question.", "bridge": "Here is the other side.",
            "clip_a": {"start_time": 5.0, "end_time": 25.0},
            "clip_b": {"start_time": 2.0, "end_time": 20.0},
        }),
    ]

    def llm_fn(prompt):
        return responses.pop(0)

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=1, llm_fn=llm_fn)

    assert result == [{
        "shared_question": "Does remote work increase or decrease productivity?",
        "thesis": "Two guests, one question.",
        "bridge": "Here is the other side.",
        "episode_a": {"run_dir": "/tmp/a", "title": "Ep A", "source_url": "https://example.com/0", "start_time": 5.0, "end_time": 25.0},
        "episode_b": {"run_dir": "/tmp/b", "title": "Ep B", "source_url": "https://example.com/1", "start_time": 2.0, "end_time": 20.0},
    }]


def test_select_thread_pairs_returns_multiple_grounded_pairs_for_multiple_questions():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    responses = [
        json.dumps({"shared_questions": ["Question one?", "Question two?"]}),
        json.dumps({
            "grounded": True, "thesis": "t1", "bridge": "b1",
            "clip_a": {"start_time": 0.0, "end_time": 20.0},
            "clip_b": {"start_time": 0.0, "end_time": 20.0},
        }),
        json.dumps({
            "grounded": True, "thesis": "t2", "bridge": "b2",
            "clip_a": {"start_time": 40.0, "end_time": 60.0},
            "clip_b": {"start_time": 40.0, "end_time": 60.0},
        }),
    ]

    def llm_fn(prompt):
        return responses.pop(0)

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=2, llm_fn=llm_fn)

    assert [r["shared_question"] for r in result] == ["Question one?", "Question two?"]


def test_select_thread_pairs_discards_pair_whose_span_overlaps_an_earlier_pick():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    responses = [
        json.dumps({"shared_questions": ["Question one?", "Question two?"]}),
        json.dumps({
            "grounded": True, "thesis": "t1", "bridge": "b1",
            "clip_a": {"start_time": 0.0, "end_time": 20.0},
            "clip_b": {"start_time": 0.0, "end_time": 20.0},
        }),
        # Question two's clip_a overlaps question one's accepted clip_a (0-20 vs 10-30) -- must be discarded.
        json.dumps({
            "grounded": True, "thesis": "t2", "bridge": "b2",
            "clip_a": {"start_time": 10.0, "end_time": 30.0},
            "clip_b": {"start_time": 40.0, "end_time": 60.0},
        }),
    ]

    def llm_fn(prompt):
        return responses.pop(0)

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=2, llm_fn=llm_fn)

    assert len(result) == 1
    assert result[0]["shared_question"] == "Question one?"


def test_select_thread_pairs_skips_ungroundable_question_and_continues():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    responses = [
        json.dumps({"shared_questions": ["Question one?", "Question two?"]}),
        json.dumps({"grounded": False, "thesis": "", "bridge": "", "clip_a": {}, "clip_b": {}}),
        json.dumps({
            "grounded": True, "thesis": "t2", "bridge": "b2",
            "clip_a": {"start_time": 0.0, "end_time": 20.0},
            "clip_b": {"start_time": 0.0, "end_time": 20.0},
        }),
    ]

    def llm_fn(prompt):
        return responses.pop(0)

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=2, llm_fn=llm_fn)

    assert len(result) == 1
    assert result[0]["shared_question"] == "Question two?"


def test_select_thread_pairs_stops_once_num_clips_reached():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    transcript_a = _transcript(100.0, [(0.0, 30.0, "hello from a")])
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    responses = [
        json.dumps({"shared_questions": ["Question one?", "Question two?", "Question three?"]}),
        json.dumps({
            "grounded": True, "thesis": "t1", "bridge": "b1",
            "clip_a": {"start_time": 0.0, "end_time": 20.0}, "clip_b": {"start_time": 0.0, "end_time": 20.0},
        }),
    ]

    def llm_fn(prompt):
        if not responses:
            pytest.fail("select_thread_pairs must stop calling pick_thread_clips once num_clips is reached")
        return responses.pop(0)

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=1, llm_fn=llm_fn)

    assert len(result) == 1


def test_select_thread_pairs_tolerates_malformed_transcript_segment_shape():
    entry_a = _corpus_entry(0, "Ep A", "abstract a", "/tmp/a")
    entry_b = _corpus_entry(1, "Ep B", "abstract b", "/tmp/b")
    # Segment missing "text"/"start" -- build_transcript_text (called inside
    # pick_thread_clips) will raise; select_thread_pairs must catch that per
    # question and return whatever it has ([] here), not propagate.
    transcript_a = {"duration": 100.0, "segments": [{"end": 10.0}]}
    transcript_b = _transcript(100.0, [(0.0, 30.0, "hello from b")])
    llm_fn = lambda prompt: json.dumps({"shared_questions": ["Question one?"]})

    result = thread_builder.select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips=1, llm_fn=llm_fn)

    assert result == []
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_thread_builder.py -k select_thread_pairs -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.thread_builder' has no attribute 'select_thread_pairs'`.

- [ ] **Step 4: Implement, and delete the old whole-corpus functions**

In `shorts_generator/thread_builder.py`:

1. Change the import line to drop the now-unused `build_corpus`:
```python
from .highlights import LLMFn, _parse_json_loose, build_transcript_text, call_muapi_llm
```

2. Delete `find_same_topic_pair` (singular) and `build_thread` in their entirety (both defined after `_sanitize_clip_pick`), along with the module docstring's mention of "first a hard same-topic gate across the whole local corpus" — replace the module docstring with:

```python
"""Two-stage picker for thread compilation: first a multi-question same-topic
gate over a FIXED pair of episodes (see docs/superpowers/specs/2026-08-09-
thread-compilation-design.md for the hard same-topic requirement, and
2026-08-10-thread-two-url-multi-clip-design.md for why the pair is fixed by
the caller instead of scanned from the whole corpus), then, for each
qualifying shared question, exact clip spans + narration text grounded in
the two chosen full transcripts.
"""
```

3. Add `_overlaps_any` right before `select_thread_pairs`:

```python
def _overlaps_any(span: Tuple[float, float], ranges: List[Tuple[float, float]]) -> bool:
    start, end = span
    return any(start < r_end and end > r_start for r_start, r_end in ranges)
```

4. Add `select_thread_pairs` at the end of the file:

```python
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
    shaped like {"shared_question", "thesis", "bridge", "episode_a",
    "episode_b"} where episode_a/b carry run_dir/title/source_url plus the
    picked start_time/end_time. Returns [] rather than raising whenever
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
            "episode_a": {"run_dir": entry_a["run_dir"], "title": entry_a["title"], "source_url": entry_a["source_url"], **clips["clip_a"]},
            "episode_b": {"run_dir": entry_b["run_dir"], "title": entry_b["title"], "source_url": entry_b["source_url"], **clips["clip_b"]},
        })

    return results
```

- [ ] **Step 5: Run the whole file**

Run: `pytest tests/test_thread_builder.py -v`
Expected: all PASS.

- [ ] **Step 6: Confirm nothing else in the codebase still references the removed functions**

Run: `grep -rn "build_thread\b\|find_same_topic_pair\b" --include="*.py" . | grep -v __pycache__`
Expected: no matches (Task 6 will remove `pipeline.py`'s import of `build_thread` — if this grep still shows a hit there, that's expected until Task 6 lands; re-run this check again after Task 6).

- [ ] **Step 7: Commit**

```bash
git add shorts_generator/thread_builder.py tests/test_thread_builder.py
git commit -m "$(cat <<'EOF'
feat: add select_thread_pairs, remove whole-corpus build_thread

select_thread_pairs finds up to num_clips grounded, non-overlapping
shared-question pairs between a caller-fixed pair of episodes -- the
multi-clip replacement for build_thread's single-pick whole-corpus
scan, which is now dead code (nothing calls generate_threads() with
no URLs anymore -- see pipeline.py/main.py/webapp.py updates).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `run_output.py` — output dir keyed by both episode titles

**Files:**
- Modify: `shorts_generator/run_output.py`
- Test: `tests/test_run_output.py`

- [ ] **Step 1: Replace the failing test**

In `tests/test_run_output.py`, replace `test_resolve_thread_output_dir_slugifies_thesis` with:

```python
def test_resolve_thread_run_dir_slugifies_both_titles(tmp_path):
    result = run_output.resolve_thread_run_dir("Episode A Title", "Episode B Title", base_dir=str(tmp_path))
    assert result == str(tmp_path / "_Threads" / "Episode_A_Title_x_Episode_B_Title")
    assert os.path.isdir(result)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_run_output.py::test_resolve_thread_run_dir_slugifies_both_titles -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.run_output' has no attribute 'resolve_thread_run_dir'`.

- [ ] **Step 3: Implement**

In `shorts_generator/run_output.py`, replace `resolve_thread_output_dir`:

```python
def resolve_thread_run_dir(title_a: str, title_b: str, base_dir: Optional[str] = None) -> str:
    """A thread run's output lives outside any single episode's RunPaths
    tree -- it draws footage from two existing episode runs, so it gets its
    own output/_Threads/<slug>/ folder. Slugged from both episode titles
    (fixed by the caller up front, see generate_threads in pipeline.py) --
    not from a thesis, since one run can now produce more than one
    thesis (see thread_builder.select_thread_pairs)."""
    base_dir = base_dir or LOCAL_OUTPUT_DIR
    slug = f"{sanitize_title(title_a)}_x_{sanitize_title(title_b)}"
    root = os.path.join(base_dir, "_Threads", slug)
    os.makedirs(root, exist_ok=True)
    return root
```

Update the comment above `list_runs`'s `_Threads` skip (it currently says "keyed by thread slug, not by episode title" — now it *is* keyed by episode titles):

```python
        # resolve_thread_run_dir() (see above) creates output/_Threads/ as a
        # sibling of per-episode run folders, keyed by both episode titles,
        # not a single episode's own title -- it has no full_source.mp4/
        # Shorts shape of its own and isn't a run the History tab should list.
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_run_output.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py
git commit -m "$(cat <<'EOF'
refactor: key thread run dirs by both episode titles, not a thesis

resolve_thread_output_dir(thesis) -> resolve_thread_run_dir(title_a,
title_b). The output dir is now knowable before any LLM call runs (the
two episodes are fixed by the caller), and a single run can produce
more than one thesis, so a thesis-derived slug no longer fits.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `pipeline.py` — rewrite `generate_threads`

**Files:**
- Modify: `shorts_generator/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Remove the 4 tests for the old signature**

In `tests/test_pipeline.py`, delete: `test_generate_threads_returns_none_when_build_thread_returns_none`, `test_generate_threads_assembles_and_writes_result`, `test_generate_threads_calls_on_output_dir_before_the_slow_work`, `test_generate_threads_tolerates_missing_duration_key`.

- [ ] **Step 2: Write the new failing tests**

Add to `tests/test_pipeline.py`:

```python
def _fake_ingest_captions(entries):
    def _ingest(url, base_dir=None):
        return entries[url]
    return _ingest


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
    monkeypatch.setattr(pipeline_module, "select_thread_pairs", lambda entry_a, entry_b, transcript_a, transcript_b, num_clips, llm_fn: [])

    result = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=2, base_dir=str(tmp_path))

    assert result == []


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
            "shared_question": "Does X cause Y?", "thesis": "t1", "bridge": "b1",
            "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 10.0, "end_time": 30.0},
            "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 5.0, "end_time": 25.0},
        },
        {
            "shared_question": "Does A cause B?", "thesis": "t2", "bridge": "b2",
            "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 40.0, "end_time": 60.0},
            "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 35.0, "end_time": 55.0},
        },
    ]
    monkeypatch.setattr(pipeline_module, "select_thread_pairs", lambda entry_a, entry_b, transcript_a, transcript_b, num_clips, llm_fn: fake_pairs)
    monkeypatch.setattr(pipeline_module, "acquire_clip", lambda run_dir, source_url, cached_duration, start_time, end_time, out_path: open(out_path, "wb").write(b"clip") or {"clip_path": out_path})
    monkeypatch.setattr(pipeline_module, "synthesize_narration", lambda text, out_path, **k: open(out_path, "wb").write(b"audio") or out_path)
    monkeypatch.setattr(pipeline_module, "render_narration_card", lambda audio_path, text, out_path: open(out_path, "wb").write(b"card") or out_path)
    assemble_calls = []
    monkeypatch.setattr(pipeline_module, "assemble_thread", lambda segment_paths, out_path: (assemble_calls.append(segment_paths), open(out_path, "wb").write(b"final"))[1] or out_path)

    result = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=2, base_dir=str(tmp_path))

    assert len(result) == 2
    out_dir = result[0]["output_dir"]
    assert result[0]["clip_url"] == os.path.join(out_dir, "clip_1.mp4")
    assert result[1]["clip_url"] == os.path.join(out_dir, "clip_2.mp4")
    assert result[0]["episode_a"]["clip_url"] == os.path.join(out_dir, "clip_1_a.mp4")
    assert result[0]["episode_b"]["clip_url"] == os.path.join(out_dir, "clip_1_b.mp4")
    assert result[1]["episode_a"]["clip_url"] == os.path.join(out_dir, "clip_2_a.mp4")
    assert assemble_calls[0] == [
        os.path.join(out_dir, "intro_card_1.mp4"), os.path.join(out_dir, "clip_1_a.mp4"),
        os.path.join(out_dir, "bridge_card_1.mp4"), os.path.join(out_dir, "clip_1_b.mp4"),
    ]
    assert os.path.isfile(os.path.join(out_dir, "thread_results.json"))
    with open(os.path.join(out_dir, "thread_results.json")) as f:
        written = json.load(f)
    assert len(written) == 2


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
    monkeypatch.setattr(pipeline_module, "select_thread_pairs", lambda entry_a, entry_b, transcript_a, transcript_b, num_clips, llm_fn: fake_pairs)

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

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_pipeline.py -k generate_threads -v`
Expected: FAIL (old `generate_threads(base_dir=...)` call signature doesn't match / `ingest_captions`/`get_abstract_cached`/`select_thread_pairs` don't exist on `pipeline_module` yet).

- [ ] **Step 4: Implement**

In `shorts_generator/pipeline.py`, update the import block (lines 19-29):

```python
from .clipper import _download_to, crop_highlights
from .corpus import get_abstract_cached
from .downloader import download_youtube
from .highlights import call_muapi_llm, get_chapters_cached, get_highlights_cached, select_final_highlights
from .local.caption_ingest import ingest_captions
from .local.llm import call_openai_vision_llm
from .local.narration import render_narration_card, synthesize_narration
from .local.thread_assembler import assemble_thread
from .local.thread_source import acquire_clip
from .run_output import RunPaths, capture_progress_log, resolve_output_dir, resolve_thread_run_dir, write_chapter_descriptions, write_descriptions, write_source_url
from .thread_builder import select_thread_pairs
from .transcriber import transcribe
from .visual_hook import call_muapi_vision_llm, score_visual_hooks
```

Replace the entire `generate_threads` function (currently lines 436-506) with:

```python
def _ingest_and_abstract(url: str, base_dir: Optional[str], llm_fn) -> Dict:
    """Caption-only ingest (no video download) + a cached topical abstract
    for one thread episode -- see local/caption_ingest.py and corpus.py."""
    ingested = ingest_captions(url, base_dir=base_dir)
    run_dir = ingested["run_dir"]
    with open(os.path.join(run_dir, "full_source.json"), "r", encoding="utf-8") as f:
        transcript = json.load(f)
    abstract = get_abstract_cached(run_dir, transcript, llm_fn=llm_fn)
    return {"run_dir": run_dir, "title": ingested["title"], "source_url": url, "abstract": abstract}


def generate_threads(
    url_a: str,
    url_b: str,
    num_clips: int = 1,
    base_dir: Optional[str] = None,
    on_output_dir: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    """Build up to num_clips distinct-topic threads from exactly the two
    given episodes (see thread_builder.select_thread_pairs). Local-mode
    only, like generate_chapters -- there is no MuAPI equivalent of this
    feature. Both URLs are ingested caption-only (no video download; see
    local/caption_ingest.py) and idempotently reused if already in the
    corpus. Returns [] if no shared question is groundable between the two
    episodes -- this is the expected, correct result when they don't
    genuinely cover the same topic, not a failure to work around.

    Unlike generate_shorts/generate_chapters, the output dir is knowable up
    front from the two episode titles (see resolve_thread_run_dir) -- but
    on_output_dir, if given, still fires before any per-clip render work
    starts, matching the old single-clip contract, so a caller like the
    dashboard can start tailing progress.log immediately.
    """
    from .local.llm import call_local_llm

    entry_a = _ingest_and_abstract(url_a, base_dir, call_local_llm)
    entry_b = _ingest_and_abstract(url_b, base_dir, call_local_llm)

    out_dir = resolve_thread_run_dir(entry_a["title"], entry_b["title"], base_dir=base_dir)
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
        pairs = select_thread_pairs(entry_a, entry_b, transcript_a, transcript_b, num_clips, call_local_llm)
        if not pairs:
            return []

        results = []
        for i, thread in enumerate(pairs, 1):
            episode_a, episode_b = thread["episode_a"], thread["episode_b"]
            print(f"[pipeline/local] clip {i}/{len(pairs)}: {thread['shared_question']!r}", flush=True)

            clip_a_path = os.path.join(out_dir, f"clip_{i}_a.mp4")
            clip_b_path = os.path.join(out_dir, f"clip_{i}_b.mp4")
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

            intro_audio = os.path.join(out_dir, f"thesis_{i}.mp3")
            bridge_audio = os.path.join(out_dir, f"bridge_{i}.mp3")
            print("[pipeline/local] synthesizing narration (thesis + bridge)...", flush=True)
            synthesize_narration(thread["thesis"], intro_audio)
            synthesize_narration(thread["bridge"], bridge_audio)

            intro_card = os.path.join(out_dir, f"intro_card_{i}.mp4")
            bridge_card = os.path.join(out_dir, f"bridge_card_{i}.mp4")
            print("[pipeline/local] rendering narration cards...", flush=True)
            render_narration_card(intro_audio, thread["thesis"], intro_card)
            render_narration_card(bridge_audio, thread["bridge"], bridge_card)

            final_path = os.path.join(out_dir, f"clip_{i}.mp4")
            print("[pipeline/local] assembling final thread (intro -> clip A -> bridge -> clip B)...", flush=True)
            assemble_thread([intro_card, clip_a_path, bridge_card, clip_b_path], final_path)

            results.append({
                **thread,
                "output_dir": out_dir,
                "clip_url": final_path,
                "episode_a": {**episode_a, "clip_url": clip_a_path},
                "episode_b": {**episode_b, "clip_url": clip_b_path},
            })
            print(f"[pipeline/local] done: {final_path}", flush=True)

        with open(os.path.join(out_dir, "thread_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_pipeline.py -k generate_threads -v`
Expected: all PASS.

- [ ] **Step 6: Run the full pipeline test file**

Run: `pytest tests/test_pipeline.py -v`
Expected: all PASS (nothing else in this file touches `generate_threads`/`build_thread`/`resolve_thread_output_dir`).

- [ ] **Step 7: Confirm the dead-code grep from Task 4 is now clean**

Run: `grep -rn "build_thread\b\|find_same_topic_pair\b\|resolve_thread_output_dir\b" --include="*.py" . | grep -v __pycache__`
Expected: no matches.

- [ ] **Step 8: Commit**

```bash
git add shorts_generator/pipeline.py tests/test_pipeline.py
git commit -m "$(cat <<'EOF'
feat: generate_threads(url_a, url_b, num_clips) replaces the corpus scan

Ingests both URLs caption-only (no download), asks
thread_builder.select_thread_pairs for up to num_clips grounded
shared-question pairs between exactly those two episodes, and renders
each into one shared output/_Threads/<slug>/ run dir as
clip_{i}.mp4 / clip_{i}_a.mp4 / clip_{i}_b.mp4. Replaces the old
no-argument, always-one-clip, whole-corpus-scan generate_threads().

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `main.py` CLI — two URLs, multi-clip output

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Update the affected tests**

In `tests/test_main.py`:

Delete `test_main_warns_when_url_given_with_clip_type_thread` (url is now required and used for thread mode, never "ignored").

Replace `test_main_dispatches_to_generate_threads_for_clip_type_thread`, `test_main_reports_no_thread_available_when_generate_threads_returns_none`, `_fake_thread_result`, `test_main_warns_on_shorts_only_flags_with_clip_type_thread`, `test_main_does_not_warn_when_no_flags_passed_with_clip_type_thread`, `test_main_warns_on_mode_equals_form_with_clip_type_thread`, and `test_main_no_thread_message_is_actionable` with:

```python
def test_main_dispatches_to_generate_threads_for_clip_type_thread(monkeypatch, capsys):
    calls = []
    fake_results = [{
        "output_dir": "output/_Threads/A_x_B",
        "shared_question": "Does X cause Y?",
        "thesis": "Two guests disagree.",
        "bridge": "Here's the other side.",
        "episode_a": {"title": "Episode A", "start_time": 10.0, "end_time": 30.0},
        "episode_b": {"title": "Episode B", "start_time": 5.0, "end_time": 25.0},
        "clip_url": "output/_Threads/A_x_B/clip_1.mp4",
    }]

    def _fake_generate_threads(url_a, url_b, **kwargs):
        calls.append((url_a, url_b, kwargs))
        return fake_results

    monkeypatch.setattr(main_module, "generate_threads", _fake_generate_threads)
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread"])

    exit_code = main()

    assert exit_code == 0
    assert calls == [("https://example.com/a", "https://example.com/b", {"num_clips": 3})]
    captured = capsys.readouterr()
    assert "Does X cause Y?" in captured.out


def test_main_fails_when_clip_type_thread_missing_url_b(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/a", "--clip-type", "thread"])

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "--url-b" in captured.err


def _fake_thread_results():
    return [{
        "output_dir": "d", "shared_question": "q?", "thesis": "t", "bridge": "b",
        "episode_a": {"title": "A", "start_time": 0.0, "end_time": 1.0},
        "episode_b": {"title": "B", "start_time": 0.0, "end_time": 1.0},
        "clip_url": "d/clip_1.mp4",
    }]


def test_main_reports_no_thread_available_when_generate_threads_returns_empty_list(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "generate_threads", lambda url_a, url_b, **kwargs: [])
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread"])

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No shared-question thread found" in captured.err


def test_main_warns_on_shorts_only_flags_with_clip_type_thread(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "generate_threads", lambda url_a, url_b, **kwargs: _fake_thread_results())
    monkeypatch.setattr(
        sys, "argv",
        ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread",
         "--filename-style", "generic", "--mode", "local", "--aspect-ratio", "16:9", "--format", "720",
         "--language", "en", "--framing", "adaptive", "--no-captions", "--caption-fade-duration", "0.5",
         "--no-word-highlight", "--no-hook-card", "--end-card", "--num-chapters", "9"],
    )

    main()

    err = capsys.readouterr().err
    assert "--filename-style generic" in err
    assert "--mode local" in err
    assert "--aspect-ratio 16:9" in err
    assert "--format 720" in err
    assert "--language en" in err
    assert "--framing adaptive" in err
    assert "--no-captions" in err
    assert "--caption-fade-duration 0.5" in err
    assert "--no-word-highlight" in err
    assert "--no-hook-card" in err
    assert "--end-card" in err
    assert "--num-chapters 9" in err
    assert "--num-clips" not in err  # now a live flag for thread mode, not ignored


def test_main_does_not_warn_when_no_extra_flags_passed_with_clip_type_thread(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "generate_threads", lambda url_a, url_b, **kwargs: _fake_thread_results())
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread"])

    main()

    err = capsys.readouterr().err
    assert "ignores" not in err


def test_main_warns_on_mode_equals_form_with_clip_type_thread(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "generate_threads", lambda url_a, url_b, **kwargs: _fake_thread_results())
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread", "--mode=api"])

    main()

    err = capsys.readouterr().err
    assert "--mode api" in err


def test_main_no_thread_message_is_actionable(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "generate_threads", lambda url_a, url_b, **kwargs: [])
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/a", "--url-b", "https://example.com/b", "--clip-type", "thread"])

    exit_code = main()

    err = capsys.readouterr().err
    assert exit_code == 1
    assert "different pair" in err
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_main.py -k thread -v`
Expected: FAIL — `--url-b` is not a recognized argument yet, and `generate_threads` is still called with no args in `main.py`.

- [ ] **Step 3: Implement**

In `main.py`:

1. Update the positional `url` help text and add `--url-b`:

```python
    parser.add_argument(
        "url", nargs="?", default=None,
        help="YouTube URL, file:// URL, or local file path. For --clip-type "
             "thread, this is episode A -- pair it with --url-b for episode B.",
    )
```

Add right after the `--filename-style` argument (before `--clip-type`):

```python
    parser.add_argument(
        "--url-b",
        default=None,
        help="Second episode URL, required together with the positional url "
             "when --clip-type thread (paired as episode A + episode B).",
    )
```

2. Update the `--clip-type thread` help text:

```python
    parser.add_argument(
        "--clip-type",
        choices=["shorts", "chapters", "thread"],
        default="shorts",
        help="shorts (default): viral 9:16 Shorts. chapters: long-form landscape "
             "chapter cuts, up to 15min each, full topic context, --mode local only. "
             "thread: up to --num-clips same-topic compilations built from the "
             "positional url (episode A) and --url-b (episode B), captions only, "
             "no video download.",
    )
```

3. Replace the url-required validation block:

```python
    if args.clip_type == "thread":
        if not args.url or not args.url_b:
            print("\nFAILED: --clip-type thread requires both the positional url (episode A) and --url-b (episode B)", file=sys.stderr)
            return 1
    elif not args.url:
        print("\nFAILED: url is required for --clip-type shorts/chapters", file=sys.stderr)
        return 1
```

4. Replace the thread-specific ignored-flags block (drop `--num-clips`, it's live now; drop the "ignores the url argument" warning, url is now required/used):

```python
    if args.clip_type == "thread":
        # generate_threads() ingests url/url_b caption-only and uses
        # num_clips -- every other shorts/chapters-only flag below is
        # silently discarded, so tell the user which ones (if any) they
        # explicitly passed but that won't do anything here.
        thread_mode_explicit = any(a == "--mode" or a.startswith("--mode=") for a in sys.argv)
        ignored_flags = []
        if thread_mode_explicit:
            ignored_flags.append(f"--mode {args.mode}")
        if args.aspect_ratio != "9:16":
            ignored_flags.append(f"--aspect-ratio {args.aspect_ratio}")
        if args.format != "1080":
            ignored_flags.append(f"--format {args.format}")
        if args.language is not None:
            ignored_flags.append(f"--language {args.language}")
        if args.framing != "locked":
            ignored_flags.append(f"--framing {args.framing}")
        if args.filename_style is not None:
            ignored_flags.append(f"--filename-style {args.filename_style}")
        if args.captions is False:
            ignored_flags.append("--no-captions")
        if args.caption_fade_duration != 0.3:
            ignored_flags.append(f"--caption-fade-duration {args.caption_fade_duration}")
        if args.word_highlight is False:
            ignored_flags.append("--no-word-highlight")
        if args.hook_card is False:
            ignored_flags.append("--no-hook-card")
        if args.end_card is True:
            ignored_flags.append("--end-card")
        if args.num_chapters != 5:
            ignored_flags.append(f"--num-chapters {args.num_chapters}")
        if ignored_flags:
            print(
                f"[main] --clip-type thread ignores: {', '.join(ignored_flags)} "
                "(captions are fetched from YouTube directly, no download/transcribe step)",
                file=sys.stderr,
            )
```

5. Update the dispatch call:

```python
        if args.clip_type == "thread":
            result = generate_threads(args.url, args.url_b, num_clips=args.num_clips)
```

6. Replace the "no thread" empty-result handling:

```python
    if args.clip_type == "thread" and not result:
        print("\nNo shared-question thread found between these two episodes -- nothing to build.", file=sys.stderr)
        print(
            "This can happen if the two episodes don't genuinely answer the same question -- "
            "try a different pair, or lower --num-clips.",
            file=sys.stderr,
        )
        return 1
```

7. Replace the thread result-printing block:

```python
    if args.clip_type == "thread":
        print(f"Threads built:   {len(result)} (requested {args.num_clips})")
        for i, t in enumerate(result, 1):
            print(f"\n#{i}  {t.get('shared_question')}")
            print(f"     thesis:  {t.get('thesis')}")
            print(f"     bridge:  {t.get('bridge')}")
            ea, eb = t["episode_a"], t["episode_b"]
            print(f"     episode A: {ea['title']} ({ea['start_time']:.1f}s -> {ea['end_time']:.1f}s)")
            print(f"     episode B: {eb['title']} ({eb['start_time']:.1f}s -> {eb['end_time']:.1f}s)")
            print(f"     clip:      {t.get('clip_url')}")
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_main.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "$(cat <<'EOF'
feat: CLI --clip-type thread takes url + --url-b, prints all clips

Matches the new generate_threads(url_a, url_b, num_clips) contract:
the positional url is episode A, --url-b is episode B, both required
for thread mode. num_clips is now a live flag instead of an ignored
one, and the result printer loops over every built clip.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `webapp.py` — dashboard backend

**Files:**
- Modify: `shorts_generator/webapp.py`
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Update the affected tests**

In `tests/test_webapp.py`, replace `test_run_thread_starts_a_job_with_no_url_and_reaches_done`, `test_run_thread_fails_with_a_helpful_message_when_no_pair_found`, and `test_status_serializes_a_thread_result_and_omits_run_name` with:

```python
def test_run_thread_requires_both_urls(client):
    resp = client.post("/run", data={"clip_type": "thread", "url_a": "https://example.com/a"})
    assert resp.status_code == 400
    assert "url_a and url_b" in resp.get_json()["error"]


def test_run_thread_starts_a_job_and_reaches_done(client, monkeypatch, tmp_path):
    out_dir = str(tmp_path / "_Threads" / "A_x_B")
    os.makedirs(out_dir, exist_ok=True)
    clip_path = os.path.join(out_dir, "clip_1.mp4")
    clip_a_path = os.path.join(out_dir, "clip_1_a.mp4")
    clip_b_path = os.path.join(out_dir, "clip_1_b.mp4")
    for p in (clip_path, clip_a_path, clip_b_path):
        open(p, "wb").write(b"data")

    fake_results = [{
        "shared_question": "Does X cause Y?",
        "thesis": "Two guests disagree.",
        "bridge": "Here's the other side.",
        "episode_a": {"title": "Episode A", "clip_url": clip_a_path},
        "episode_b": {"title": "Episode B", "clip_url": clip_b_path},
        "output_dir": out_dir,
        "clip_url": clip_path,
    }]

    def _fake_generate_threads(url_a, url_b, num_clips=1, base_dir=None, on_output_dir=None):
        if on_output_dir:
            on_output_dir(out_dir)
        return fake_results

    monkeypatch.setattr(webapp, "generate_threads", _fake_generate_threads)
    monkeypatch.setattr(webapp.threading, "Thread", _SyncThread)

    resp = client.post("/run", data={"clip_type": "thread", "url_a": "https://example.com/a", "url_b": "https://example.com/b", "num_clips": "2"})
    assert resp.status_code == 202
    assert webapp.job.status == "done"
    assert webapp.job.clip_type == "thread"
    assert webapp.job.result == fake_results
    assert webapp.job.progress_log == os.path.join(out_dir, "progress.log")


def test_run_thread_fails_with_a_helpful_message_when_no_pair_found(client, monkeypatch):
    monkeypatch.setattr(webapp, "generate_threads", lambda url_a, url_b, num_clips=1, base_dir=None, on_output_dir=None: [])
    monkeypatch.setattr(webapp.threading, "Thread", _SyncThread)

    resp = client.post("/run", data={"clip_type": "thread", "url_a": "https://example.com/a", "url_b": "https://example.com/b"})
    assert resp.status_code == 202
    assert webapp.job.status == "failed"
    assert "no shared-question thread" in webapp.job.error.lower()


def test_status_serializes_thread_results_and_omits_run_name(client, monkeypatch, tmp_path):
    out_dir = str(tmp_path / "_Threads" / "A_x_B")
    os.makedirs(out_dir, exist_ok=True)
    clip_path = os.path.join(out_dir, "clip_1.mp4")
    clip_a_path = os.path.join(out_dir, "clip_1_a.mp4")
    clip_b_path = os.path.join(out_dir, "clip_1_b.mp4")
    for p in (clip_path, clip_a_path, clip_b_path):
        open(p, "wb").write(b"data")

    def _fake_generate_threads(url_a, url_b, num_clips=1, base_dir=None, on_output_dir=None):
        if on_output_dir:
            on_output_dir(out_dir)
        return [{
            "shared_question": "Does X cause Y?",
            "thesis": "Two guests disagree.",
            "bridge": "Here's the other side.",
            "episode_a": {"title": "Episode A", "clip_url": clip_a_path},
            "episode_b": {"title": "Episode B", "clip_url": clip_b_path},
            "output_dir": out_dir,
            "clip_url": clip_path,
        }]

    monkeypatch.setattr(webapp, "generate_threads", _fake_generate_threads)
    monkeypatch.setattr(webapp.threading, "Thread", _SyncThread)

    client.post("/run", data={"clip_type": "thread", "url_a": "https://example.com/a", "url_b": "https://example.com/b"})
    resp = client.get("/status")
    data = resp.get_json()

    assert data["status"] == "done"
    assert data["run_name"] is None
    assert len(data["result"]["threads"]) == 1
    thread = data["result"]["threads"][0]
    assert thread["thesis"] == "Two guests disagree."
    assert thread["download_url"] == "/download/clip_1.mp4"
    assert thread["episode_a_download_url"] == "/download/clip_1_a.mp4"
    assert thread["episode_b_download_url"] == "/download/clip_1_b.mp4"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_webapp.py -k thread -v`
Expected: FAIL — `/run` still accepts a no-URL thread request, `_serialize_thread_result` (singular) doesn't produce a `"threads"` list.

- [ ] **Step 3: Implement**

In `shorts_generator/webapp.py`:

Replace `_run_thread_job`:

```python
def _run_thread_job(url_a: str, url_b: str, num_clips: int) -> None:
    """Ingests url_a/url_b caption-only (no video download) and builds up
    to num_clips distinct shared-question threads between them -- see
    generate_threads in pipeline.py. Like _run_job, the output dir isn't
    known until generate_threads has resolved it from the two episode
    titles, so job.progress_log/shorts_dir are set via the on_output_dir
    callback."""
    def _on_output_dir(out_dir: str) -> None:
        with _job_lock:
            job.status = "running"
            job.progress_log = os.path.join(out_dir, "progress.log")
            job.shorts_dir = out_dir

    try:
        result = generate_threads(url_a, url_b, num_clips=num_clips, on_output_dir=_on_output_dir)
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

Replace `_serialize_thread_result`:

```python
def _serialize_thread_results(results: List[Dict], out_dir: Optional[str]) -> Dict:
    """Thread results have a different shape than shorts/chapters (multiple
    two-source clips, no score/hook fields) -- see generate_threads' return
    in pipeline.py. One entry per grounded shared-question pair (up to the
    requested num_clips, possibly fewer -- see select_thread_pairs)."""
    threads = []
    for r in results:
        clip_url = r.get("clip_url")
        if clip_url and not _clip_file_exists(out_dir, clip_url):
            # Clip was generated but its file has since been deleted -- drop
            # it instead of re-rendering it as a "Failed" card, which it isn't.
            continue
        episode_a_clip = (r.get("episode_a") or {}).get("clip_url")
        episode_b_clip = (r.get("episode_b") or {}).get("clip_url")
        threads.append({
            "shared_question": r.get("shared_question"),
            "thesis": r.get("thesis"),
            "bridge": r.get("bridge"),
            "episode_a": r.get("episode_a"),
            "episode_b": r.get("episode_b"),
            "download_url": _clip_display_url(out_dir, clip_url),
            "episode_a_download_url": _clip_display_url(out_dir, episode_a_clip),
            "episode_b_download_url": _clip_display_url(out_dir, episode_b_clip),
        })
    return {"threads": threads}
```

Add the `Dict`/`List` import if not already present (check the top of the file — it currently imports `from typing import Any, Optional, Tuple`; change to):

```python
from typing import Any, Dict, List, Optional, Tuple
```

Replace the `/run` route's thread branch:

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

Update the `/status` route's thread branch:

```python
    if result and clip_type == "thread":
        serialized_result = _serialize_thread_results(result, shorts_dir)
        # A thread run has no per-episode run folder (shorts_dir is
        # _Threads/<slug> here, not output/<Title>/Shorts) -- there's no
        # History-tab run to name, and _run_name_from_shorts_dir(shorts_dir)
        # would wrongly resolve to "_Threads".
        run_name = None
```

Update the two module docstrings referencing the old contract: the `Job` dataclass's `shorts_dir` comment (currently says "reused as the download route's serve-from directory" — still true, no change needed there) and the removed reference to `resolve_thread_output_dir` at the top of `_run_thread_job` (already rewritten above).

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_webapp.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/webapp.py tests/test_webapp.py
git commit -m "$(cat <<'EOF'
feat: dashboard thread mode takes url_a/url_b, serializes multi-clip results

/run now requires url_a + url_b (+ optional num_clips) for
clip_type=thread instead of taking no input. _serialize_thread_results
returns a "threads" list (was a single "thread" object), each entry
now also carrying episode_a_download_url/episode_b_download_url for
the two source clips alongside the final assembled one.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: `templates/index.html` — dashboard form + results UI

**Files:**
- Modify: `shorts_generator/templates/index.html`

No automated test file covers this template's JS (the existing suite only checks static markers like `id="run-form"` and `id="url"` render — Task 10 covers a manual smoke check). Edit directly and verify by hand in Step 3.

- [ ] **Step 1: Replace the thread-mode form fields**

Find the `<fieldset>` block starting at the `clip_type` select. Replace from `<label for="clip_type">Clip type</label>` through the closing `</div>` of `id="shorts-fields"` and the button, with:

```html
          <label for="clip_type">Clip type</label>
          <select id="clip_type" name="clip_type">
            <option value="shorts" selected>Shorts</option>
            <option value="thread">Thread</option>
          </select>

          <label for="num_clips">Num clips</label>
          <input type="number" id="num_clips" name="num_clips" value="3" min="1">

          <div id="shorts-fields">
            <label for="url">YouTube URL</label>
            <input type="text" id="url" name="url" placeholder="https://www.youtube.com/watch?v=..." required>

            <div class="row">
              <div>
                <label for="mode">Mode</label>
                <select id="mode" name="mode">
                  <option value="api">api</option>
                  <option value="local">local</option>
                </select>
              </div>
              <div>
                <label for="aspect_ratio">Aspect ratio</label>
                <input type="text" id="aspect_ratio" name="aspect_ratio" value="9:16">
              </div>
            </div>

            <div class="row">
              <div>
                <label for="format">Download resolution</label>
                <select id="format" name="format">
                  <option value="360">360</option>
                  <option value="480">480</option>
                  <option value="720">720</option>
                  <option value="1080" selected>1080</option>
                </select>
              </div>
              <div>
                <label for="language">Language (blank = auto-detect)</label>
                <input type="text" id="language" name="language" placeholder="en">
              </div>
            </div>

            <div class="row">
              <div>
                <label for="framing">Framing (local mode only)</label>
                <select id="framing" name="framing">
                  <option value="locked">locked</option>
                  <option value="adaptive">adaptive</option>
                </select>
              </div>
              <div>
                <label for="filename_style">Clip filenames</label>
                <select id="filename_style" name="filename_style">
                  <option value="specific" selected>specific (title1.mp4)</option>
                  <option value="generic">generic (video1.mp4)</option>
                </select>
              </div>
            </div>

            <div class="row">
              <div class="checkbox-row">
                <label><input type="checkbox" id="captions" name="captions" checked> Burn captions</label>
              </div>
              <div class="checkbox-row">
                <label><input type="checkbox" id="word_highlight" name="word_highlight" checked> Word highlight</label>
              </div>
              <div class="checkbox-row">
                <label><input type="checkbox" id="hook_card" name="hook_card" checked> Hook card</label>
              </div>
              <div class="checkbox-row">
                <label><input type="checkbox" id="end_card" name="end_card"> End card</label>
              </div>
            </div>
            <label for="caption_fade_duration">Caption fade (s)</label>
            <input type="number" id="caption_fade_duration" name="caption_fade_duration" value="0.3" step="0.1" min="0">
          </div>

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

          <button type="submit" id="submit-btn">Generate shorts</button>
        </fieldset>
      </form>
```

- [ ] **Step 2: Update the JS**

Find the block starting `const clipTypeSelect = document.getElementById("clip_type");` through `updateClipTypeVisibility();` and replace it with:

```javascript
    const clipTypeSelect = document.getElementById("clip_type");
    const shortsFields = document.getElementById("shorts-fields");
    const threadFields = document.getElementById("thread-fields");
    const urlInput = document.getElementById("url");
    const urlAInput = document.getElementById("url_a");
    const urlBInput = document.getElementById("url_b");
    const numClipsInput = document.getElementById("num_clips");

    const STATUS_LABEL = { idle: "STANDBY", starting: "STARTING", running: "REC", done: "DONE", failed: "FAILED" };
    const STATUS_DOT = { idle: "", starting: "is-rec", running: "is-rec", done: "is-ok", failed: "is-warn" };

    function updateClipTypeVisibility() {
      const isThread = clipTypeSelect.value === "thread";
      shortsFields.hidden = isThread;
      threadFields.hidden = !isThread;
      urlInput.required = !isThread;
      urlAInput.required = isThread;
      urlBInput.required = isThread;
      numClipsInput.value = isThread ? "2" : "3";
      submitBtn.textContent = isThread ? "Build threads" : "Generate shorts";
    }
    clipTypeSelect.addEventListener("change", updateClipTypeVisibility);
    updateClipTypeVisibility();
```

(This keeps the pre-existing `STATUS_LABEL`/`STATUS_DOT` constants right after the new field lookups, in the same position they were in before — do not duplicate them if they already appear elsewhere in the surrounding code you're editing.)

- [ ] **Step 3: Update `buildThreadCard` and `renderResults`**

Replace the existing `buildThreadCard` function and the `renderResults` function's thread branch with:

```javascript
    function appendThreadSourceVideo(card, label, episode, downloadUrl) {
      if (!episode || !episode.title) return;
      appendLabeledText(card, label, "reason", episode.title);
      if (!downloadUrl) return;
      const videoFrame = document.createElement("div");
      videoFrame.className = "video-frame frame";
      const video = document.createElement("video");
      video.controls = true;
      video.src = downloadUrl;
      videoFrame.appendChild(video);
      card.appendChild(videoFrame);
    }

    function buildThreadCard(t) {
      const card = document.createElement("div");
      card.className = "card frame" + (t.download_url ? "" : " failed");

      const titleEl = document.createElement("div");
      titleEl.className = "card-title";
      titleEl.textContent = t.shared_question || "Thread";
      card.appendChild(titleEl);

      if (t.thesis) appendLabeledText(card, "Thesis", "pitch", t.thesis);
      if (t.bridge) appendLabeledText(card, "Bridge", "pitch", t.bridge);

      if (t.download_url) {
        const videoFrame = document.createElement("div");
        videoFrame.className = "video-frame frame";
        const video = document.createElement("video");
        video.controls = true;
        video.src = t.download_url;
        videoFrame.appendChild(video);
        card.appendChild(videoFrame);

        const link = document.createElement("a");
        link.className = "download";
        link.href = t.download_url;
        link.setAttribute("download", "");
        link.textContent = "Download";
        card.appendChild(link);
      } else {
        const statusEl = document.createElement("div");
        statusEl.className = "card-status";
        statusEl.textContent = "Failed";
        card.appendChild(statusEl);
      }

      appendThreadSourceVideo(card, "Episode A", t.episode_a, t.episode_a_download_url);
      appendThreadSourceVideo(card, "Episode B", t.episode_b, t.episode_b_download_url);

      return card;
    }

    function renderResults(result, runName) {
      const hasShorts = result && Array.isArray(result.shorts) && result.shorts.length > 0;
      const hasThreads = result && Array.isArray(result.threads) && result.threads.length > 0;
      resultsPanel.hidden = !hasShorts && !hasThreads;
      if (hasThreads) {
        resultsEl.innerHTML = "";
        result.threads.forEach(t => resultsEl.appendChild(buildThreadCard(t)));
        return;
      }
      if (!hasShorts) {
        resultsEl.innerHTML = "";
        return;
      }
      renderShortsGrid(resultsEl, result.shorts, runName);
    }
```

- [ ] **Step 4: Run the existing dashboard test to confirm the static markers still render**

Run: `pytest tests/test_webapp.py::test_index_returns_the_dashboard_page -v`
Expected: PASS.

- [ ] **Step 5: Manual smoke check**

Start the dashboard (`python -m shorts_generator.webapp` or however it's normally run locally — check `webapp.py`'s bottom for the exact entry point if unsure) and in a browser:
1. Load `/`, confirm "Num clips" is visible with clip_type defaulted to Shorts.
2. Switch clip type to "Thread" — confirm the Shorts fields hide, "Episode A URL"/"Episode B URL" fields appear, "Num clips" value changes to 2, and the submit button reads "Build threads".
3. Switch back to "Shorts" — confirm "Num clips" resets to 3 and the URL field reappears required.

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/templates/index.html
git commit -m "$(cat <<'EOF'
feat: dashboard thread mode UI -- two URL fields, multi-clip results

Replaces the no-input "builds from local corpus" note with Episode A/B
URL fields and shows the existing Num clips field for thread mode too
(reset to 2 by default, since a strict same-topic gate rarely finds 3
distinct questions between two episodes). Each result card now also
shows the two source clips a thread was cut from, not just the final
assembled video.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Full verification pass

**Files:**
- (none — verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `pytest -q`
Expected: all tests PASS, zero failures/errors. If anything fails, stop and fix it before proceeding — do not commit around a red suite.

- [ ] **Step 2: Confirm no leftover references to removed APIs anywhere in the repo**

Run: `grep -rn "build_thread\b\|find_same_topic_pair\b\|resolve_thread_output_dir\b\|_serialize_thread_result\b" --include="*.py" --include="*.html" . | grep -v __pycache__`
Expected: no matches. (`_serialize_thread_result` singular should be fully gone, replaced by `_serialize_thread_results`.)

- [ ] **Step 3: Confirm the working tree is clean**

Run: `git status --short`
Expected: no output. If anything is unstaged, review it with `git diff` before deciding whether it belongs in a follow-up commit or was accidentally missed by an earlier task.

- [ ] **Step 4: Report completion**

No commit needed for this task — it's verification only. Summarize for the user: full suite green, dead code removed, dashboard + CLI both exercise the new two-URL multi-clip contract.
