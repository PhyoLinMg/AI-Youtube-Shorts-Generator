# Claim-Specificity Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** score every highlight candidate on whether its hook states a concrete, specific claim (vs. a vague topic gesture), and use that score to gate + backfill which candidates actually get cropped into Shorts.

**Architecture:** add `claim_specificity`/`claim_specificity_reason` to the existing 18-field highlight schema in `shorts_generator/highlights.py` (new rubric, prompt fields, sanitize logic, schema-version bump). Add a new `select_final_highlights()` function that replaces the pipeline's `top = sorted(...)[:2*num_clips]` slice with a gate (claim_specificity >= 80) + backfill-from-best-score. Surface the new score/reason on the webapp short card, same visual pattern as `hook_strength`.

**Tech Stack:** Python (`shorts_generator/highlights.py`, `shorts_generator/pipeline.py`), vanilla JS in `shorts_generator/templates/index.html`, pytest.

**Spec:** `docs/superpowers/specs/2026-07-26-claim-specificity-gate-design.md`

---

### Task 1: Add `claim_specificity` scoring to the highlight prompt + sanitizer

**Files:**
- Modify: `shorts_generator/highlights.py`
- Test: `tests/test_highlights.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_highlights.py`, after `test_sanitize_highlights_includes_format_reason` (around line 178):

```python
def test_sanitize_highlights_clamps_claim_specificity_above_range():
    cleaned = _sanitize_highlights([_raw_highlight(claim_specificity=150)], duration=100.0)
    assert cleaned[0]["claim_specificity"] == 100


def test_sanitize_highlights_clamps_claim_specificity_below_range():
    cleaned = _sanitize_highlights([_raw_highlight(claim_specificity=-20)], duration=100.0)
    assert cleaned[0]["claim_specificity"] == 0


def test_sanitize_highlights_defaults_claim_specificity_fields_when_missing():
    raw = {"start_time": 1.0, "end_time": 5.0}
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["claim_specificity"] == 0
    assert cleaned[0]["claim_specificity_reason"] == ""


def test_sanitize_highlights_includes_claim_specificity_reason():
    raw = _raw_highlight(claim_specificity=88, claim_specificity_reason="names a specific dollar figure")
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["claim_specificity"] == 88
    assert cleaned[0]["claim_specificity_reason"] == "names a specific dollar figure"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_highlights.py -k claim_specificity -v`
Expected: FAIL — `KeyError: 'claim_specificity'` (the sanitizer doesn't produce this key yet).

- [ ] **Step 3: Add the rubric constant**

In `shorts_generator/highlights.py`, immediately after the `HOOK_STRENGTH_RUBRIC` constant (ends at line 104, right before `HIGHLIGHT_SYSTEM_PROMPT = """You are an elite...`), insert:

```python
CLAIM_SPECIFICITY_RUBRIC = """
Claim specificity (does the hook state something concrete, or just gesture at a topic):
- High (80+): a specific, surprising fact, number, or claim a viewer could
  repeat verbatim -- e.g. "95% of the universe is dark matter and dark energy,"
  "I lost $40,000 in one trade before I turned 20." Names a number, a named
  mechanism, or a falsifiable claim.
- Low (<40): a vague topic gesture or a generic opinion with no concrete
  payload -- e.g. "He had a good point about success," "We talked about
  what really matters in life." Sounds insightful but says nothing a viewer
  could repeat.
- Reward: a stat, a dollar figure, a named fact, a concrete contrarian
  assertion.
- Penalize: abstractions ("mindset," "success," "the truth about X") with
  no concrete instantiation attached.
"""
```

- [ ] **Step 4: Wire the rubric into the system prompt template**

In `HIGHLIGHT_SYSTEM_PROMPT` (the big triple-quoted string), find:

```
{hook_strength_rubric}

Rules:
```

Replace with:

```
{hook_strength_rubric}

{claim_specificity_rubric}

Rules:
```

- [ ] **Step 5: Add the new rule bullet**

In the same `HIGHLIGHT_SYSTEM_PROMPT`, find the bullet:

```
- Score "format_clarity_score" 0-100 on whether this span reads as ONE self-contained idea a viewer immediately grasps — a single Q&A, a single before/after, a single narrated event — versus a meandering excerpt that needs outside context. Write a "format_reason" — one sentence on what makes the format legible or muddy.
```

Add this new bullet immediately after it:

```
- Score "claim_specificity" 0-100 on whether this highlight states a concrete, specific fact, number, or claim a viewer could repeat verbatim — versus a vague topic gesture or generic opinion — per the claim-specificity rubric above. Write a "claim_specificity_reason" — one sentence on what makes the claim concrete or vague.
```

- [ ] **Step 6: Add the fields to the JSON schema example**

In the same `HIGHLIGHT_SYSTEM_PROMPT`, find the schema line:

```
Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","on_screen_hook":"string","virality_reason":"string","hook_strength":int,"hook_self_contained":bool,"hook_reason":"string","description":"string","yt_title":"string","yt_hashtags":["#Shorts","#topic1","#topic2"],"reaction_type":"string","cut_segments":[{{"start_time":float,"end_time":float}}],"tightness_reason":"string","format_clarity_score":int,"format_reason":"string"}}]}}"""
```

Replace with (added `claim_specificity` and `claim_specificity_reason` right after `format_reason`):

```
Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","on_screen_hook":"string","virality_reason":"string","hook_strength":int,"hook_self_contained":bool,"hook_reason":"string","description":"string","yt_title":"string","yt_hashtags":["#Shorts","#topic1","#topic2"],"reaction_type":"string","cut_segments":[{{"start_time":float,"end_time":float}}],"tightness_reason":"string","format_clarity_score":int,"format_reason":"string","claim_specificity":int,"claim_specificity_reason":"string"}}]}}"""
```

- [ ] **Step 7: Pass the rubric into `.format()`**

In `call_highlight_api`, find:

```python
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        reaction_jail_criteria=REACTION_JAIL_CRITERIA,
        hook_strength_rubric=HOOK_STRENGTH_RUBRIC,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        num_clips_instruction=f"Generate at least {min_clips} highlights",
    )
```

Replace with:

```python
    system = HIGHLIGHT_SYSTEM_PROMPT.format(
        virality_criteria=VIRALITY_CRITERIA,
        reaction_jail_criteria=REACTION_JAIL_CRITERIA,
        hook_strength_rubric=HOOK_STRENGTH_RUBRIC,
        claim_specificity_rubric=CLAIM_SPECIFICITY_RUBRIC,
        content_type=content_info.get("content_type", "other"),
        density=content_info.get("density", "medium"),
        num_clips_instruction=f"Generate at least {min_clips} highlights",
    )
```

- [ ] **Step 8: Add the fields to the retry-prompt's required-fields list**

In `call_highlight_api`'s retry block, find:

```python
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Return ONLY valid JSON with a top-level 'highlights' array."
                + " Each item must include: title, start_time, end_time, score, hook_sentence, on_screen_hook, virality_reason, hook_strength, hook_self_contained, hook_reason, description, yt_title, yt_hashtags, reaction_type, cut_segments, tightness_reason, format_clarity_score, format_reason."
                + " No markdown fences, no commentary."
            )
```

Replace with:

```python
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Return ONLY valid JSON with a top-level 'highlights' array."
                + " Each item must include: title, start_time, end_time, score, hook_sentence, on_screen_hook, virality_reason, hook_strength, hook_self_contained, hook_reason, description, yt_title, yt_hashtags, reaction_type, cut_segments, tightness_reason, format_clarity_score, format_reason, claim_specificity, claim_specificity_reason."
                + " No markdown fences, no commentary."
            )
```

- [ ] **Step 9: Add the fields to `_sanitize_highlights`**

In `_sanitize_highlights`, find:

```python
                "format_clarity_score": max(0, min(100, _coerce_int(item.get("format_clarity_score"), default=0))),
                "format_reason": str(item.get("format_reason") or "").strip(),
            }
        )
```

Replace with:

```python
                "format_clarity_score": max(0, min(100, _coerce_int(item.get("format_clarity_score"), default=0))),
                "format_reason": str(item.get("format_reason") or "").strip(),
                "claim_specificity": max(0, min(100, _coerce_int(item.get("claim_specificity"), default=0))),
                "claim_specificity_reason": str(item.get("claim_specificity_reason") or "").strip(),
            }
        )
```

- [ ] **Step 10: Bump the schema version**

Find:

```python
HIGHLIGHT_SCHEMA_VERSION = 4    # bump whenever the highlight dict shape changes,
                                # so a stale on-disk cache (missing new fields)
                                # is treated as a miss instead of silently reused.
                                # v3: added cut_segments, reaction_type, tightness_reason.
                                # v4: added format_clarity_score, format_reason.
```

Replace with:

```python
HIGHLIGHT_SCHEMA_VERSION = 5    # bump whenever the highlight dict shape changes,
                                # so a stale on-disk cache (missing new fields)
                                # is treated as a miss instead of silently reused.
                                # v3: added cut_segments, reaction_type, tightness_reason.
                                # v4: added format_clarity_score, format_reason.
                                # v5: added claim_specificity, claim_specificity_reason.
```

- [ ] **Step 11: Run tests to verify they pass**

Run: `python -m pytest tests/test_highlights.py -v`
Expected: PASS, all tests including the 4 new ones, zero regressions.

- [ ] **Step 12: Commit**

```bash
git add shorts_generator/highlights.py tests/test_highlights.py
git commit -m "feat: score claim specificity on each highlight candidate"
```

---

### Task 2: `select_final_highlights` — gate + backfill

**Files:**
- Modify: `shorts_generator/highlights.py`
- Test: `tests/test_highlights.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_highlights.py`, after the `dedupe_highlights`/`get_highlights` tests (end of file, before any `if __name__` block if present — otherwise just append):

```python
from shorts_generator.highlights import CLAIM_SPECIFICITY_THRESHOLD, select_final_highlights


def test_select_final_highlights_keeps_top_passers_by_score():
    highlights = [
        {"title": "A", "score": 90, "claim_specificity": 85},
        {"title": "B", "score": 95, "claim_specificity": 82},
        {"title": "C", "score": 99, "claim_specificity": 50},  # highest score, fails the gate
    ]
    result = select_final_highlights(highlights, num_clips=2)
    assert [h["title"] for h in result] == ["B", "A"]


def test_select_final_highlights_backfills_when_too_few_passers():
    highlights = [
        {"title": "A", "score": 90, "claim_specificity": 85},  # passes
        {"title": "B", "score": 80, "claim_specificity": 40},  # fails
        {"title": "C", "score": 70, "claim_specificity": 30},  # fails
    ]
    result = select_final_highlights(highlights, num_clips=2)
    assert [h["title"] for h in result] == ["A", "B"]


def test_select_final_highlights_zero_passers_matches_score_only_ranking():
    highlights = [
        {"title": "A", "score": 90, "claim_specificity": 10},
        {"title": "B", "score": 95, "claim_specificity": 20},
        {"title": "C", "score": 70, "claim_specificity": 5},
    ]
    result = select_final_highlights(highlights, num_clips=2)
    assert [h["title"] for h in result] == ["B", "A"]


def test_select_final_highlights_returns_all_when_fewer_than_num_clips():
    highlights = [{"title": "A", "score": 90, "claim_specificity": 85}]
    result = select_final_highlights(highlights, num_clips=3)
    assert [h["title"] for h in result] == ["A"]


def test_select_final_highlights_missing_claim_specificity_defaults_to_non_passer():
    highlights = [
        {"title": "A", "score": 90},  # no claim_specificity key at all
        {"title": "B", "score": 80, "claim_specificity": 85},
    ]
    result = select_final_highlights(highlights, num_clips=2)
    assert [h["title"] for h in result] == ["B", "A"]


def test_claim_specificity_threshold_is_80():
    assert CLAIM_SPECIFICITY_THRESHOLD == 80
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_highlights.py -k select_final_highlights -v`
Expected: FAIL — `ImportError: cannot import name 'select_final_highlights'`.

- [ ] **Step 3: Implement `select_final_highlights`**

In `shorts_generator/highlights.py`, add after the `dedupe_highlights` function (after its closing `return kept`):

```python
CLAIM_SPECIFICITY_THRESHOLD = 80


def select_final_highlights(
    all_highlights: List[Dict], num_clips: int, threshold: int = CLAIM_SPECIFICITY_THRESHOLD,
) -> List[Dict]:
    """Select up to num_clips highlights, preferring ones whose claim is
    concrete enough to survive the swipe test (claim_specificity >= threshold).
    Backfills from the best-scoring remaining candidates when too few clear
    the bar, so a strict gate never shrinks output below what score-only
    ranking would have produced -- zero passers degrades to plain
    top-N-by-score, never raises."""
    ranked = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)
    passers: List[Dict] = []
    rest: List[Dict] = []
    for h in ranked:
        target = passers if int(h.get("claim_specificity", 0)) >= threshold else rest
        target.append(h)

    final = passers[:num_clips]
    if len(final) < num_clips:
        final += rest[: num_clips - len(final)]
    return final
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_highlights.py -v`
Expected: PASS, all tests including the 6 new ones, zero regressions.

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/highlights.py tests/test_highlights.py
git commit -m "feat: add claim-specificity gate with score-based backfill"
```

---

### Task 3: Wire the gate into both pipeline modes

**Files:**
- Modify: `shorts_generator/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Update the two candidate-count tests to reflect the new behavior**

These two existing tests assert today's `2 * num_clips` slice. That behavior is being replaced by this task, so the tests must change to describe the new contract, not stay red. In `tests/test_pipeline.py`, replace:

```python
def test_run_local_crops_double_num_clips_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_highlights_cached",
        lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result_many(5),
    )

    crop_mock = Mock(return_value=[])
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", crop_mock)

    pipeline_module._run_local(
        "https://youtube.example/x",
        num_clips=2,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    args, _ = crop_mock.call_args
    top = args[1]
    assert len(top) == 4
```

with:

```python
def test_run_local_crops_num_clips_candidates_via_claim_specificity_gate(tmp_path, monkeypatch):
    # _fake_highlights_result_many gives no candidate a claim_specificity
    # field, so every candidate is a non-passer (defaults to 0) and the
    # gate falls back to plain top-num_clips-by-score.
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_highlights_cached",
        lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result_many(5),
    )

    crop_mock = Mock(return_value=[])
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", crop_mock)

    pipeline_module._run_local(
        "https://youtube.example/x",
        num_clips=2,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    args, _ = crop_mock.call_args
    top = args[1]
    assert len(top) == 2
```

And replace:

```python
def test_run_api_crops_double_num_clips_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_highlights_cached",
        lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result_many(5),
    )

    crop_mock = Mock(return_value=[])
    monkeypatch.setattr(pipeline_module, "crop_highlights", crop_mock)

    pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=2,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    args, _ = crop_mock.call_args
    top = args[1]
    assert len(top) == 4
```

with:

```python
def test_run_api_crops_num_clips_candidates_via_claim_specificity_gate(tmp_path, monkeypatch):
    # Same rationale as the local-mode version above: no candidate has a
    # claim_specificity field, so the gate falls back to top-num_clips-by-score.
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_highlights_cached",
        lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result_many(5),
    )

    crop_mock = Mock(return_value=[])
    monkeypatch.setattr(pipeline_module, "crop_highlights", crop_mock)

    pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=2,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    args, _ = crop_mock.call_args
    top = args[1]
    assert len(top) == 2
```

- [ ] **Step 2: Add a test proving the gate actually prefers claim_specificity over raw score**

Add to `tests/test_pipeline.py`, after the two tests just rewritten:

```python
def _fake_highlights_result_mixed_specificity():
    return {
        "highlights": [
            {"start_time": 0.0, "end_time": 3.0, "score": 99, "claim_specificity": 30, "title": "High score, vague"},
            {"start_time": 10.0, "end_time": 13.0, "score": 70, "claim_specificity": 85, "title": "Lower score, specific"},
        ]
    }


def test_run_api_gate_prefers_claim_specificity_over_raw_score(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(
        pipeline_module, "get_highlights_cached",
        lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result_mixed_specificity(),
    )

    crop_mock = Mock(return_value=[])
    monkeypatch.setattr(pipeline_module, "crop_highlights", crop_mock)

    pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    args, _ = crop_mock.call_args
    top = args[1]
    assert len(top) == 1
    assert top[0]["title"] == "Lower score, specific"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -k "claim_specificity" -v`
Expected: FAIL — the two renamed tests still see `len(top) == 4` (old behavior), and the new mixed-specificity test picks the high-score-but-vague clip instead. `AttributeError: module 'shorts_generator.pipeline' has no attribute 'select_final_highlights'` is NOT expected here since we only reference it via pipeline's internal call, not import it in the test — confirm failures are assertion mismatches, not import errors.

- [ ] **Step 4: Wire `select_final_highlights` into pipeline.py**

In `shorts_generator/pipeline.py`, find:

```python
from .highlights import call_muapi_llm, get_highlights_cached
```

Replace with:

```python
from .highlights import call_muapi_llm, get_highlights_cached, select_final_highlights
```

In `_run_local`, find:

```python
    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:2 * num_clips]
    print(f"[pipeline/local] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)
```

Replace with:

```python
    top = select_final_highlights(all_highlights, num_clips)
    print(f"[pipeline/local] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)
```

In `_run_api`, find:

```python
    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:2 * num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)
```

Replace with:

```python
    top = select_final_highlights(all_highlights, num_clips)
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS — all tests including the 3 touched/added in this task, zero regressions.

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/pipeline.py tests/test_pipeline.py
git commit -m "feat: gate highlight selection on claim specificity, not a fixed 2x pool"
```

---

### Task 4: Surface `claim_specificity` on the webapp short card

**Files:**
- Modify: `shorts_generator/templates/index.html`

- [ ] **Step 1: Add the meter + reason block**

In `shorts_generator/templates/index.html`'s `buildShortCard`, find the closing of the `hook_strength` block:

```javascript
        if (typeof s.hook_strength === "number") {
          const hookStrength = Number(s.hook_strength) || 0;
          const hookRow = document.createElement("div");
          hookRow.className = "score-row";
          const hookLabel = document.createElement("span");
          hookLabel.textContent = "Hook score";
          hookRow.appendChild(hookLabel);
          const hookMeter = document.createElement("div");
          hookMeter.className = "meter";
          const hookMeterFill = document.createElement("span");
          hookMeterFill.style.width = hookStrength + "%";
          hookMeterFill.style.background = scoreColor(hookStrength);
          hookMeter.appendChild(hookMeterFill);
          hookRow.appendChild(hookMeter);
          const hookNum = document.createElement("span");
          hookNum.textContent = hookStrength + (s.hook_self_contained ? " (self-contained)" : "");
          hookRow.appendChild(hookNum);
          card.appendChild(hookRow);
        }

        if (s.description) {
```

Replace with (inserting the new block between them):

```javascript
        if (typeof s.hook_strength === "number") {
          const hookStrength = Number(s.hook_strength) || 0;
          const hookRow = document.createElement("div");
          hookRow.className = "score-row";
          const hookLabel = document.createElement("span");
          hookLabel.textContent = "Hook score";
          hookRow.appendChild(hookLabel);
          const hookMeter = document.createElement("div");
          hookMeter.className = "meter";
          const hookMeterFill = document.createElement("span");
          hookMeterFill.style.width = hookStrength + "%";
          hookMeterFill.style.background = scoreColor(hookStrength);
          hookMeter.appendChild(hookMeterFill);
          hookRow.appendChild(hookMeter);
          const hookNum = document.createElement("span");
          hookNum.textContent = hookStrength + (s.hook_self_contained ? " (self-contained)" : "");
          hookRow.appendChild(hookNum);
          card.appendChild(hookRow);
        }

        if (typeof s.claim_specificity === "number") {
          const claimScore = Number(s.claim_specificity) || 0;
          const claimRow = document.createElement("div");
          claimRow.className = "score-row";
          const claimLabel = document.createElement("span");
          claimLabel.textContent = "Claim specificity";
          claimRow.appendChild(claimLabel);
          const claimMeter = document.createElement("div");
          claimMeter.className = "meter";
          const claimMeterFill = document.createElement("span");
          claimMeterFill.style.width = claimScore + "%";
          claimMeterFill.style.background = scoreColor(claimScore);
          claimMeter.appendChild(claimMeterFill);
          claimRow.appendChild(claimMeter);
          const claimNum = document.createElement("span");
          claimNum.textContent = claimScore;
          claimRow.appendChild(claimNum);
          card.appendChild(claimRow);
        }

        if (s.claim_specificity_reason) {
          appendLabeledText(card, "Claim reason", "reason", s.claim_specificity_reason);
        }

        if (s.description) {
```

- [ ] **Step 2: Manually verify in the browser**

Start the dev server (see project README for the exact command — `python main.py --mode local ...` or the webapp entrypoint), run it against a short local video, or temporarily paste a `claim_specificity`/`claim_specificity_reason` pair into an existing `result.json` in `output/<Title>/` and reload the webapp. Confirm:
- the "Claim specificity" meter renders directly below "Hook score", with the same color-by-score styling as the other meters
- the "Claim reason" text block renders below it
- a card whose highlight predates this change (no `claim_specificity` key) still renders cleanly with the block simply omitted, no console errors

- [ ] **Step 3: Commit**

```bash
git add shorts_generator/templates/index.html
git commit -m "feat: show claim-specificity score on short cards"
```

---

## Definition of done

- [ ] `python -m pytest tests/ -q` passes with zero regressions.
- [ ] `highlights.json` for a fresh run includes `claim_specificity`/`claim_specificity_reason` on every highlight.
- [ ] `HIGHLIGHT_SCHEMA_VERSION == 5`; an old cached `highlights.json` (schema v4) misses and recomputes cleanly (already covered by existing cache-miss tests in `test_highlights.py` that compare against `HIGHLIGHT_SCHEMA_VERSION` symbolically).
- [ ] Both `_run_api` and `_run_local` crop `min(num_clips, len(all_highlights))` clips via `select_final_highlights`, not a fixed `2 * num_clips`.
- [ ] A candidate with high `score` but low `claim_specificity` loses its slot to a lower-`score`/high-`claim_specificity` candidate when slots are scarce (`test_run_api_gate_prefers_claim_specificity_over_raw_score`).
- [ ] "Claim specificity" meter + "Claim reason" text visible on every short card in the webapp; older results without the field render unaffected.
