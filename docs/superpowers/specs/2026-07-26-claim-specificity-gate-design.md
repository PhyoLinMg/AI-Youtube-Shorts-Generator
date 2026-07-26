# Claim-Specificity Gate — Design

**Problem:** some published Shorts get views, some don't, even though all pass
today's `hook_strength`/`format_clarity_score` bar. Manual review of the
generator's output (see `result.json`) plus a proposed stricter selection
prompt suggested the missing signal: today's rubric scores how *fast* a hook
lands and how *self-contained* it is, but nothing scores whether the hook's
content is a **concrete, specific claim** versus a vague topic gesture that
merely sounds insightful.

**Non-goals:** this does not replace `hook_strength`, `format_clarity_score`,
`hook_self_contained`, or `on_screen_hook`'s anti-curiosity-gap rules — those
already cover opening-reversal, standalone-strength, and title-ability
respectively (verified against `shorts_generator/highlights.py` during
design). Only claim specificity is genuinely new.

## Scoring: `claim_specificity`

New rubric block, same shape/placement as the existing `HOOK_STRENGTH_RUBRIC`,
injected into `HIGHLIGHT_SYSTEM_PROMPT`:

```
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

Two new fields per highlight, scored/written by the model alongside the
existing 18:
- `claim_specificity`: int 0-100 (same scale/clamp convention as
  `hook_strength`/`format_clarity_score`)
- `claim_specificity_reason`: one-sentence justification (same convention as
  `hook_reason`/`format_reason`)

Both get added to: the JSON schema line in `HIGHLIGHT_SYSTEM_PROMPT`, the
retry-prompt's required-fields list, and `_sanitize_highlights` (int
clamp 0-100 default 0; string default `""`).

`HIGHLIGHT_SCHEMA_VERSION` bumps 4 → 5 — unlike `visual_hook_score` (which is
explicitly never cached), these two fields land in the same highlight dict
that `get_highlights_cached` persists to `highlights.json`, so a version bump
is required to invalidate stale caches missing the new fields.

## Gate + backfill: `select_final_highlights`

New function in `highlights.py`, replacing the line

```python
top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:2 * num_clips]
```

in both `pipeline._run_api` and `pipeline._run_local`, with:

```python
top = select_final_highlights(all_highlights, num_clips)
```

```python
CLAIM_SPECIFICITY_THRESHOLD = 80

def select_final_highlights(
    all_highlights: List[Dict], num_clips: int, threshold: int = CLAIM_SPECIFICITY_THRESHOLD,
) -> List[Dict]:
    """Select up to num_clips highlights, preferring ones whose claim is
    concrete enough to survive the swipe test (claim_specificity >= threshold).
    Backfills from the best-scoring remaining candidates when too few clear
    the bar, so a strict gate never shrinks output below what score-only
    ranking would have produced -- zero passers degrades to today's
    pure-top-N-by-score behavior instead of raising."""
    ranked = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)
    passers, rest = [], []
    for h in ranked:
        (passers if int(h.get("claim_specificity", 0)) >= threshold else rest).append(h)

    final = passers[:num_clips]
    if len(final) < num_clips:
        final += rest[: num_clips - len(final)]
    return final
```

This is a **behavior change**, approved during design: the pipeline currently
crops `2 * num_clips` candidates and shows all of them in the webapp gallery
(browse-and-pick UX). After this change it crops exactly `min(num_clips,
len(all_highlights))` — the gate is meant to shrink what gets rendered, not
just reorder a browsable pool. `score_visual_hooks` still runs after
`select_final_highlights`, unchanged, on the now-smaller `top` list.

Zero-passer and empty-`all_highlights` behavior is unchanged from today
(empty `all_highlights` still raises `RuntimeError` in `_run_api`/`_run_local`
before `select_final_highlights` is ever called).

## UI

`shorts_generator/templates/index.html`, `buildShortCard` — immediately after
the existing `hook_strength` meter block, add a "Claim specificity" score-row
(meter + number, same `scoreColor()` styling) and a "Claim reason" labeled
text block, following the exact pattern the `hook_strength`/`hook_reason`
pair already uses. Guard the meter with `typeof s.claim_specificity ===
"number"`, matching `format_clarity_score`'s guard — `_sanitize_highlights`
always populates the field going forward, but a `result.json` written before
this change (schema v4 or earlier) won't have it, and the guard keeps that
card rendering cleanly instead of showing a stray "0%" meter.

## Testing

- `_sanitize_highlights`: defaults claim_specificity to 0 and
  claim_specificity_reason to `""` when absent; clamps out-of-range scores.
- `select_final_highlights`: enough passers (>num_clips) → top num_clips by
  score among passers only; too few passers → backfill from best-scoring
  non-passers to reach num_clips; zero passers → identical output to today's
  `sorted(...)[:num_clips]`; fewer than num_clips candidates total → returns
  all of them, no error.
- `pipeline.py`: update `test_run_api_crops_double_num_clips_candidates` (and
  its local-mode equivalent) — rename and rewrite to assert `len(top) ==
  num_clips` (not `2 * num_clips`), since that's the behavior this design
  intentionally changes.
- Webapp: manual check that the new meter/reason render for a highlight with
  the new fields present, no console errors.

## Definition of done

- `python -m pytest tests/ -q` passes with zero regressions (after updating
  the now-intentionally-outdated 2x-candidates test).
- A run with `--mode local` or `--mode api` produces `claim_specificity`/
  `claim_specificity_reason` on every highlight in `highlights.json`.
- Pipeline crops exactly `min(num_clips, len(all_highlights))` clips, gated
  by claim specificity with backfill, not a fixed `2 * num_clips`.
- New meter + reason text visible on every short card in the webapp.
- `HIGHLIGHT_SCHEMA_VERSION == 5`; old cached `highlights.json` files miss and
  recompute cleanly.
