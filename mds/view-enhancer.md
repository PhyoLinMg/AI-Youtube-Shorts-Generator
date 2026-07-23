# Improve highlight-picker: rank & grade on hook-landing-speed + human review gate

## Context

**Problem.** The channel's Shorts performance is driven almost entirely by *stayed-to-watch %* (feed swipe-decision), not impressions/CTR. Diagnostic finding this session: winners open with a **complete, self-contained, contrarian claim graspable in ~1 second** ("If I made the universe, I wouldn't need your praise"); losers open with a **thesis that needs processing** ("The Ultimate Irony: Capitalism Will Create the Communist Utopia"). Timing and thumbnails were tested and are not the bottleneck. The suspected root cause is that the AI highlight-picker selects for topical virality, not for "does this open with an immediate, complete hook."

**What the code actually does today** (from exploration of `shorts_generator/highlights.py` + `pipeline.py`):
- The picker *does* have hook language: `VIRALITY_CRITERIA` (highlights.py:51-61) lists "HOOK MOMENTS" as signal #1 of 8, and `HIGHLIGHT_SYSTEM_PROMPT` (highlights.py:64-88, rule at line 73) enforces "open with a strong HOOK within the first **3 seconds**, `start_time` on the hook line, self-contained."
- **But** the single `score` (0-100) is a blend of all 8 signals — hook is one input, never isolated, never a separate scored axis, never verified. A clip can score high on revelation/practical-value while opening slow.
- The overlay/title copy actively optimizes the *losing* pattern: `title`/`yt_title` rules say "aggressive clickbait / curiosity gap" (highlights.py:85-86). A curiosity gap withholds the payoff → requires processing → the exact anti-pattern the data flags.
- Selection is pure top-N by blended `score` (`pipeline.py:145` api / `:63` local): `sorted(..., key=score)[:num_clips]`. No score floor, no hook axis, **no human curation** — every top-N clip auto-finalizes into an mp4. `webapp.py` is submit/monitor/browse only, not a review gate.

**Goal.** Make the picker (1) score the **opener** on a dedicated hook-landing axis defined by the user's own labeled examples, (2) rank hook-first, (3) reshape the on-screen overlay copy away from curiosity-gap teasing toward stating the complete claim, and (4) surface a per-clip hook grade so the human reviews and hand-picks the absolute hits (no auto-cut). Validate the scorer by **backtesting against known winners/losers before any YouTube re-test.**

**Decisions locked with user:** small N / "only want absolute hits" is fine; user will supply labeled winner/loser clips; **human review only** (grade + surface, human decides — do NOT auto-drop).

---

## Phase 0 — Backtest (DONE — result changed the design)

Ran a real backtest, not a simulated one: pulled the user's actual YouTube Studio export (`Table data.csv`, 62 published Shorts with real "Stayed to watch (%)"), matched 38 of them by title back to their `result.json` opener text (`hook_sentence`), and blind-scored every opener with a draft `hook_strength` rubric via the project's configured local LLM (OpenRouter — the MuAPI key in `.env` turned out to be expired/403, so `shorts_generator/local/llm.py:call_openrouter_llm` was used as the judge instead). The scorer never saw the stayed-to-watch number or the old `score`.

**Finding 1 — the existing blended `score` is dead weight.** Across all 38 real shipped clips it ranged 95-100 (spread of 5 points). It carries no ranking signal at all — within-podcast Spearman rho against stayed-to-watch was *negative* in 4 of 5 podcasts (mean rho ≈ -0.19). This alone justifies replacing it as the thing a human trusts to judge clip quality.

**Finding 2 — hook_strength is real but content-dependent, not a clean universal ranker.** For declarative-fact content (Neil deGrasse Tyson, MrBeast) precision-at-top was strong: top-ranked-by-hook clips were also the actual top performers. But it **inverts** for emotional/confessional content: in the Karen Hao (AI whistleblower) podcast, the top-2 clips by hook_strength ("I screamed at my child for distracting me" → scored 95, "Do some of these AI CEOs realize they are quite literally summoning the demon?" → scored 85) were that podcast's two *worst* performers (23% and 13.5% stayed-to-watch). Worse: in the Hinton podcast, the podcast's actual best performer — "I think that's complete nonsense." (61.6% stayed, 6th-best of all 38 clips tested) — scored hook_strength=10, dead last. A rhetorical question with vivid imagery, or a punchy but context-dependent reaction line, fools the rubric into a high score while a plain declarative fact-check line that plays well gets buried. This isn't a wording bug — a rubric tweak that fixes the Karen Hao false positives (e.g. penalize rhetorical questions) doesn't touch the Hinton false negative, which is exactly the opposite failure mode.

**Decision — hook_strength does NOT drive the automated crop/rank cut.** If it sorted the top-N slice at small N (the user's stated preference), it would silently discard real hits before a human ever sees them — the exact thing curation was supposed to prevent. Instead: keep it as a **surfaced signal for human review** (Phase 3) — displayed alongside score/virality_reason so the human's own judgment, informed but not overridden, makes the cut. This matches the user's own "human review only" decision better than the original hook-first-sort plan did. Phase 1 (add the field, rubric, sanitize) still stands; **Phase 1e below is revised** from the original "rank hook-first" to "do not reorder the crop-selection cut."

Backtest data + scorer script are in the session scratchpad (not committed — throwaway validation artifacts, not product code).

---

## Phase 1 — Add the hook-landing scoring axis (`shorts_generator/highlights.py`)

**1a. New rubric block.** Add a `HOOK_STRENGTH_RUBRIC` string and interpolate it into `HIGHLIGHT_SYSTEM_PROMPT`. Define hook_strength precisely, anchored to the **~1-second swipe-decision window** (tighten the existing "first 3 seconds" language), using the user's verbatim examples:
   - *High (90+):* first spoken line states a COMPLETE, surprising/contrarian idea that stands alone with zero prior context — e.g. "If I made the universe, I wouldn't need your praise." A viewer hearing only the first ~1s grasps a full claim.
   - *Low (<40):* opener is a thesis requiring processing / holding two concepts / setup before the point lands — e.g. "The Ultimate Irony: Capitalism Will Create the Communist Utopia." Also low: windup / throat-clearing ("So the thing is…", "What people don't realize…") and curiosity-gap teases ("You won't believe what happened").
   - Reward: declarative self-contained contrarian statement, named-authority factual claim, a complete vivid image. Penalize: dependency on prior context, multi-step reasoning, promise-of-payoff-later.

**1b. New JSON schema fields** (extend the schema at highlights.py:88 and the retry reminder at :289-294):
   - `hook_strength`: int 0-100 — how completely & immediately the opener lands in the ~1s window.
   - `hook_self_contained`: bool — opener needs zero prior context.
   - `hook_reason`: str — one sentence on why the opener lands or fails (human-review aid).

**1c. Reshape overlay copy (the high-leverage surface).** Per the data, the swipe decision lands on the opening spoken line + `on_screen_hook` overlay — NOT `yt_title` (title/CTR near-meaningless on the feed). Rewrite the `on_screen_hook` rule (highlights.py:80) to require **stating the complete claim/payoff itself, not teasing it**: add explicit anti-patterns ("You won't believe…", "The truth about X") and good patterns (a full contrarian statement). Leave `yt_title`/`title` mostly as-is (low leverage), only softening the "curiosity gap" wording so title and overlay don't fight.

**1d. Sanitize new fields** in `_sanitize_highlights` (highlights.py:173-210): clamp `hook_strength` to 0-100 (mirror the existing `score` clamp at :200), coerce `hook_self_contained` to bool, default `hook_reason` to "". Missing `hook_strength` → default 0 (so a stale/blank entry sorts to the bottom, never silently mid-pack).

**1e. REVISED per Phase 0 result — do not let hook_strength drive the crop cut.** The backtest showed hook_strength inverts on emotional/confessional content (see Phase 0 above), so sorting the top-N slice by it would silently discard real hits at small N before a human sees them. Do NOT reorder by hook_strength in `get_highlights` (highlights.py:359) or the pipeline top-N slice (`pipeline.py:145` api, `:63` local). `hook_strength` is computed and attached to every highlight, but is consumed only in Phase 3 as a **displayed signal**, not a sort/cut key.

Separately, the backtest also showed the *old* `score` is uninformative (constant 95-100 across real clips — see Phase 0), which means the current `sorted(..., key=score)[:num_clips]` cut is already close to arbitrary among near-tied candidates — it isn't reliably picking the best content today either. Given the user's stated goal ("I only want the absolute hit," human-review-only), the fix is to **widen what gets physically cropped** so the human is choosing from real material instead of a near-random pre-filtered slice.

**Correction on pool size — the ~14 cap is per-chunk, not global.** `min(target, natural_max, 14)` (highlights.py:260-262) caps each individual LLM call at ~14, but long videos (>30 min) get split into multiple chunks (`CHUNK_SIZE_SECONDS=1200`, `LONG_VIDEO_THRESHOLD=1800`, `CHUNK_OVERLAP_SECONDS=60` — highlights.py:91-93), each making its own capped call. Worked example for the user's real case — a 2hr (7200s) video at `num_clips=10`: chunking produces 7 chunks (20-min each, 1-min overlap); for a video this long, `natural_max` exceeds the outer cap in every chunk, so each of the 7 chunks asks for "at least 14" independently → **~90-100 raw candidates**, and since the 7 chunks cover mostly-disjoint 20-min windows, dedupe (which only removes >50%-overlapping pairs) leaves most of that pool intact. So "crop the full deduped set" was a mis-sized recommendation for long/chunked videos — it means ~90 clips, hours of local ffmpeg/OpenCV time, and a 90-clip review job, not a handful.

**Decision (user confirmed): crop a fixed multiple of `num_clips`, not the full deduped pool.** Crop pool size = `2 * num_clips` (e.g. 20 candidates cropped for a `num_clips=10` request), taken from the deduped candidate list — since `score` doesn't discriminate, take them in dedupe's existing score-sorted order (arbitrary among near-ties, but stable) or, simpler, unsorted/as-returned; ordering of *which* 20 doesn't matter much given score's uninformativeness, but the count must not silently balloon to the full 90-100 pool. Implement `pipeline.py:63` (local) / `:145` (api) as `[: 2 * num_clips]` instead of `[:num_clips]`.

**Cost (user runs `--mode local`):** the LLM highlight-picking call cost is unchanged by this — it already generates the ~90-100-candidate pool per chunk regardless of how many get cropped afterward (highlights.py:260-262 sizing doesn't look at the crop-pool multiplier). Cropping in local mode (`local/clipper.py`: ffmpeg subclip + OpenCV vertical reframe + caption burn + hook-card render) is local compute, not a paid API — going from 10 to 20 cropped clips costs $0 extra, just more wall-clock local processing time (~2x). (Note for completeness: `--mode api` cropping *does* go through paid MuAPI autocrop, so if the api path is ever used this same 2x multiplier would ~2x MuAPI cost there too — keep that in mind if api mode is used later.)

---

## Phase 2 — Cache versioning (`shorts_generator/highlights.py`)

`_transcript_fingerprint` (highlights.py:34-42) only covers transcript content, so a prompt/schema change would silently reuse a stale `highlights.json` with no `hook_strength`. Add a module constant `HIGHLIGHT_SCHEMA_VERSION` (bump it now), write it into the cache dict alongside `transcript_fingerprint`/`num_clips`/`highlights` (highlights.py:397-408), and require it to match on read (`get_highlights_cached`, :386-393) — mismatch → full recompute. This also lets future rubric edits invalidate cleanly.

---

## Phase 3 — Surface the grade for human review (no auto-cut)

Human-review-only means the pipeline grades and displays; the human picks. Make the hook grade visible everywhere a clip is reviewed:

- **`result.json`** already carries all per-clip fields through `pipeline.py:247` — confirm `hook_strength`, `hook_self_contained`, `hook_reason` flow into each `shorts[]` entry (they ride inside the highlight dict).
- **`shorts_generator/run_output.py` `write_descriptions`** (:225-252): prepend each block with its hook grade, e.g. `hook: 82  self-contained: yes`, so the copy-paste file doubles as a pick-list.
- **`shorts_generator/webapp.py` + `templates/index.html`**: in the status/history clip listing, show `hook_strength` alongside `score` and `virality_reason` as plain sortable columns/badges — **do not default-sort by hook_strength** (Phase 0 showed it inverts on some content; a human eyeballing all signals together outperforms trusting either number alone). This is the review affordance: more signal, same human judgment. No checkboxes/auto-drop in this phase.

**Optional Phase 3b (only if backtest validates and user wants to save compute):** a two-phase gate — pipeline emits ranked graded candidates *before* cropping, human selects a subset in the dashboard, only selected clips get cropped (MuAPI autocrop + caption + hook-card baking are the expensive steps). Bigger webapp change; defer until the scorer is proven.

---

## Phase 4 — Tests

- `tests/` (add `test_highlights.py` if absent): `_sanitize_highlights` clamps/defaults the new fields; cache read rejects a dict with mismatched `HIGHLIGHT_SCHEMA_VERSION`; ranking orders by `hook_strength` then `score`.
- Adjust `tests/test_run_output.py` for the new `write_descriptions` grade line, and `tests/test_webapp.py` for the surfaced grade.
- Keep existing MuAPI/LLM calls mocked (tests already inject `llm_fn`).

---

## Verification (end-to-end)

1. **Backtest (done):** blind-scored 38 real shipped clips against real stayed-to-watch from the user's YouTube Studio export. Result: old `score` carries no signal (constant 95-100, negative within-podcast correlation); `hook_strength` is a real but content-dependent signal (strong for declarative-fact content, inverts for emotional/confessional content) — so it ships as a review signal, not a rank/cut key. See Phase 0 above.
2. **Unit:** `pytest tests/ -q` green, including the new sanitize/cache/crop-pool tests.
3. **Pipeline dry-run:** re-run on a cached transcript (bump `HIGHLIGHT_SCHEMA_VERSION` so the LLM re-runs); inspect `output/<run>/highlights.json` and `result.json` — every highlight has a plausible `hook_strength`, the surviving `on_screen_hook`s read as self-contained claims (not teases), and the cropped set is not silently pre-filtered to a near-arbitrary top-N.
4. **Review surface:** open the dashboard History for that run; confirm `hook_strength`/`score`/`virality_reason` all show per clip; confirm `descriptions.txt` carries the grade line.
5. **Forward test (post-ship):** after the user posts a hand-picked batch, re-pull stayed-to-watch and compare against the old batch to confirm the change moved the metric.

## Out of scope (noted, not planned)

- Content-pillar pruning (DaVinci tutorials dead) — that's a channel/split decision the user is still weighing, not a picker change. Could later feed `content_type` into a down-rank, but not now.
- Any auto-publish/upload step — the human still hand-posts.

## Critical files

- `shorts_generator/highlights.py` — rubric, schema fields, sanitize, ranking, cache version (core).
- `shorts_generator/pipeline.py` — top-N ranking order (:63, :145).
- `shorts_generator/run_output.py` — `write_descriptions` grade line.
- `shorts_generator/webapp.py` + `templates/index.html` — review surface.
- `tests/test_highlights.py` (new/edit), `tests/test_run_output.py`, `tests/test_webapp.py`.
