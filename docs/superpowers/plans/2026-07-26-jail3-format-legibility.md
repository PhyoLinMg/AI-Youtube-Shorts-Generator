# Jail 3 — Format Legibility Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** score whether each extracted clip reads as one legible, self-contained idea (jail 3 / "format jail," 30K-100K views) — piggybacking on the existing text-only highlight LLM call, no new subsystem.

**Architecture:** two new fields, `format_clarity_score` (0-100) and `format_reason`, added to the same JSON schema `hook_strength`/`hook_reason` already live in. Prompt gains one new rule; `_sanitize_highlights` validates the two fields the same way it already validates `hook_strength`/`hook_reason`. Informational only — surfaced in the webapp, doesn't change candidate selection.

**Tech Stack:** Python, existing highlight-LLM prompt plumbing.

**Prerequisite:** this plan assumes `docs/superpowers/plans/2026-07-26-jail2-reaction-jump-cuts.md` has landed — `HIGHLIGHT_SCHEMA_VERSION` starts this plan at 3 (set by that plan's Task 3) and this plan bumps it to 4.

**Spec:** `docs/superpowers/specs/2026-07-26-three-jails-escape-design.md`

---

### Task 1: `format_clarity_score` + `format_reason` validation

**Files:**
- Modify: `shorts_generator/highlights.py`
- Test: `tests/test_highlights.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_highlights.py`:

```python
def test_sanitize_highlights_clamps_format_clarity_score_above_range():
    cleaned = _sanitize_highlights([_raw_highlight(format_clarity_score=150)], duration=100.0)
    assert cleaned[0]["format_clarity_score"] == 100


def test_sanitize_highlights_clamps_format_clarity_score_below_range():
    cleaned = _sanitize_highlights([_raw_highlight(format_clarity_score=-20)], duration=100.0)
    assert cleaned[0]["format_clarity_score"] == 0


def test_sanitize_highlights_defaults_format_fields_when_missing():
    raw = {"start_time": 1.0, "end_time": 5.0}
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["format_clarity_score"] == 0
    assert cleaned[0]["format_reason"] == ""


def test_sanitize_highlights_includes_format_reason():
    raw = _raw_highlight(format_clarity_score=85, format_reason="single clean before/after beat")
    cleaned = _sanitize_highlights([raw], duration=100.0)
    assert cleaned[0]["format_clarity_score"] == 85
    assert cleaned[0]["format_reason"] == "single clean before/after beat"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_highlights.py -k "format_clarity or format_reason" -v`
Expected: FAIL — `KeyError: 'format_clarity_score'`.

- [ ] **Step 3: Add the two fields to `_sanitize_highlights`**

In `shorts_generator/highlights.py`, in the `cleaned.append({...})` dict inside `_sanitize_highlights` (the same dict Task 2 of the jail-2 plan already extended with `cut_segments`/`reaction_type`/`tightness_reason`), add two more keys:

```python
                "format_clarity_score": max(0, min(100, _coerce_int(item.get("format_clarity_score"), default=0))),
                "format_reason": str(item.get("format_reason") or "").strip(),
```

This mirrors the existing `hook_strength` clamp (`max(0, min(100, _coerce_int(item.get("hook_strength"), default=0)))`) and `hook_reason` default (`str(item.get("hook_reason") or "").strip()`) one-for-one.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_highlights.py -v`
Expected: PASS, all tests including the 4 new ones, zero regressions.

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/highlights.py tests/test_highlights.py
git commit -m "feat: validate format_clarity_score and format_reason on highlights"
```

---

### Task 2: Format-legibility prompt rule + schema version bump

**Files:**
- Modify: `shorts_generator/highlights.py`

- [ ] **Step 1: Add a format-legibility rule to `HIGHLIGHT_SYSTEM_PROMPT`'s `Rules:` list**

Add this bullet right after the `reaction_type`/`cut_segments` bullets added by the jail-2 plan (or, if jail-2's prompt changes haven't landed, right after the `on_screen_hook` bullet):

```
- Score "format_clarity_score" 0-100 on whether this span reads as ONE self-contained idea a viewer immediately grasps — a single Q&A, a single before/after, a single narrated event — versus a meandering excerpt that needs outside context. Write a "format_reason" — one sentence on what makes the format legible or muddy.
```

Add the two new keys to the JSON schema example at the end of `HIGHLIGHT_SYSTEM_PROMPT`:

```python
Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_sentence":"string","on_screen_hook":"string","virality_reason":"string","hook_strength":int,"hook_self_contained":bool,"hook_reason":"string","description":"string","yt_title":"string","yt_hashtags":["#Shorts","#topic1","#topic2"],"reaction_type":"string","cut_segments":[{{"start_time":float,"end_time":float}}],"tightness_reason":"string","format_clarity_score":int,"format_reason":"string"}}]}}"""
```

Add `, format_clarity_score, format_reason` to the retry-prompt required-fields list in `call_highlight_api` (the same string Task 3 of the jail-2 plan already extended with `reaction_type, cut_segments, tightness_reason`):

```python
                + " Each item must include: title, start_time, end_time, score, hook_sentence, on_screen_hook, virality_reason, hook_strength, hook_self_contained, hook_reason, description, yt_title, yt_hashtags, reaction_type, cut_segments, tightness_reason, format_clarity_score, format_reason."
```

- [ ] **Step 2: Bump the schema version**

Change `HIGHLIGHT_SCHEMA_VERSION = 3` to `HIGHLIGHT_SCHEMA_VERSION = 4`, and extend its comment:

```python
HIGHLIGHT_SCHEMA_VERSION = 4    # bump whenever the highlight dict shape changes,
                                # so a stale on-disk cache (missing new fields)
                                # is treated as a miss instead of silently reused.
                                # v3: added cut_segments, reaction_type, tightness_reason.
                                # v4: added format_clarity_score, format_reason.
```

A separate bump from jail 2's v2→v3 is required here, not a reuse of v3 — a cache written by jail-2-only code (schema v3, no `format_clarity_score`) must not be silently accepted once this plan's code expects that field to be present.

- [ ] **Step 3: Run the full highlights test suite**

Run: `python -m pytest tests/test_highlights.py -v`
Expected: PASS, same count as Task 1's Step 4 (prompt-string and version-constant changes only, already covered by the existing schema-version-mismatch tests).

- [ ] **Step 4: Commit**

```bash
git add shorts_generator/highlights.py
git commit -m "feat: add format-legibility prompt rule, bump highlight schema to v4"
```

---

### Task 3: Surface `format_clarity_score` and `format_reason` in the webapp

**Files:**
- Modify: `shorts_generator/templates/index.html`

- [ ] **Step 1: Add a format-clarity meter and reason text block**

In `shorts_generator/templates/index.html`'s `buildShortCard`, right after the `hook_strength` meter block (or after the jail-2 plan's `reaction_type` badge, if that's already landed):

```javascript
        if (typeof s.format_clarity_score === "number") {
          const formatScore = Number(s.format_clarity_score) || 0;
          const formatRow = document.createElement("div");
          formatRow.className = "score-row";
          const formatLabel = document.createElement("span");
          formatLabel.textContent = "Format clarity";
          formatRow.appendChild(formatLabel);
          const formatMeter = document.createElement("div");
          formatMeter.className = "meter";
          const formatMeterFill = document.createElement("span");
          formatMeterFill.style.width = formatScore + "%";
          formatMeterFill.style.background = scoreColor(formatScore);
          formatMeter.appendChild(formatMeterFill);
          formatRow.appendChild(formatMeter);
          const formatNum = document.createElement("span");
          formatNum.textContent = formatScore;
          formatRow.appendChild(formatNum);
          card.appendChild(formatRow);
        }

        if (s.format_reason) {
          appendLabeledText(card, "Format read", "reason", s.format_reason);
        }
```

This is a straight copy of the existing `hook_strength` meter block's structure (variable names changed), reusing `scoreColor`, `.meter`, `.score-row` — no new CSS.

- [ ] **Step 2: Manually verify in the browser**

Same manual-check approach as the jail-2 plan's webapp task: start the dev server, load a run (or a past `result.json` with fake `format_clarity_score`/`format_reason` values pasted in temporarily) and confirm the meter renders correctly, then discard any temporary edit.

- [ ] **Step 3: Commit**

```bash
git add shorts_generator/templates/index.html
git commit -m "feat: show format clarity score on short cards"
```

---

## Definition of done

- [ ] `python -m pytest tests/ -q` passes with zero regressions against the pre-Task-1 count.
- [ ] `HIGHLIGHT_SCHEMA_VERSION` is 4; a cache written under v3 (jail-2-only) is treated as a miss and recomputed.
- [ ] `format_clarity_score` and `format_reason` are visible in the webapp UI.
