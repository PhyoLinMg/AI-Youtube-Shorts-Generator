# Thread Clips: YouTube + TikTok Dual-Platform Output — Design Spec

## Problem

Thread mode (`docs/superpowers/specs/2026-08-09-thread-compilation-design.md`, extended by `2026-08-10-thread-two-url-multi-clip-design.md`) targets a single hardcoded 45-60s runtime (`thread_builder.MIN_CLIP_SECONDS=8.0`, `MAX_CLIP_SECONDS=25.0`, prompt lines "Duration per clip: 12-22 seconds" / "must land in the 45-60 second range"). That target was picked for YouTube Shorts, but TikTok's Creator Rewards Program (which replaced the old Creator Fund) only pays out on videos **60 seconds or longer** — duets, stitches, and anything under 60s earn nothing regardless of views. TikTok's 2026 algorithm also actively favors 1-3 minute content over the old sub-30s norm.

At the current 45-60s target, every thread clip produced today is TikTok-monetization-ineligible by construction. There is no per-platform concept anywhere in the codebase (`shorts_generator/config.py` and a full-tree grep confirm no `--platform` flag, no platform enum, no platform-conditional output logic).

## Solution shape

Add a `platform` axis to thread generation: `"youtube"` (today's 45-60s target, unchanged default) or `"tiktok"` (a new 65-90s target, code-enforced to never assemble under 60s), or `"both"` (produce one file of each, sharing the expensive same-topic discovery step). Platform only changes clip-span length bounds and the prompt's length instructions — it does not change aspect ratio (both stay 1080×1920 vertical, since TikTok also wants vertical) or the assembly structure (intro card → clip A → bridge card → clip B stays the same for both).

## 1. Platform bounds (`shorts_generator/thread_builder.py`)

Replace the module-level `MIN_CLIP_SECONDS`/`MAX_CLIP_SECONDS` constants with:

```python
PLATFORM_BOUNDS = {
    "youtube": {"min_clip": 8.0,  "max_clip": 25.0, "clip_range": "12-22 seconds", "total_range": "45-60 second"},
    "tiktok":  {"min_clip": 28.0, "max_clip": 40.0, "clip_range": "28-40 seconds", "total_range": "65-90 second"},
}
```

**Why `min_clip=28.0` for TikTok, not a lower number:** the code-enforced floor determines the worst case the LLM can legally produce, and the worst case is what decides monetization eligibility. Two clips at the floor (2×28=56s) plus a conservative narration floor (~8s: two ~12-word lines at TTS-normal pace) is ~64s — safely clear of the 60s cutoff with margin. A floor of 20s (2×20=40s + 8s narration ≈ 48s) would let a technically-valid pick still land under 60s, defeating the feature's whole purpose. `max_clip=40.0` keeps the upper bound inside the 2026 "1-3 minute, but don't pad past what completion rate can support" sweet spot from research (worst-case ceiling 2×40+14≈94s).

`THREAD_PICK_SYSTEM_PROMPT` (`thread_builder.py:53-77`) gets two new format placeholders, `{clip_range}` and `{total_range}`, substituted from `PLATFORM_BOUNDS[platform]`, replacing the hardcoded "12-22 seconds" / "45-60 second" text at lines 62-63. The existing line "If the answer runs longer, pick the single most complete self-contained span within it — do not use extra length just because it's available" (line 62) is preserved verbatim for both platforms — it's what keeps TikTok cuts from maxing out to 90s by default and hurting completion rate.

Signature changes (all take `platform: str = "youtube"`, validated against `PLATFORM_BOUNDS.keys()`):
- `_sanitize_clip_span(raw, duration, min_clip, max_clip)` — bounds now passed as arguments instead of read from module constants.
- `pick_thread_clips(episode_a, episode_b, shared_question, llm_fn, avoid_ranges_a=None, avoid_ranges_b=None, platform="youtube")` — looks up `PLATFORM_BOUNDS[platform]`, formats the prompt with `clip_range`/`total_range`, passes `min_clip`/`max_clip` into `_sanitize_clip_span`.

## 2. Avoid duplicating stage A (`thread_builder.py`)

`find_same_topic_pairs` (shared-question discovery between two episode abstracts) has no dependency on clip length — the same shared-question list is valid input for both a 45-60s and a 65-90s grounding pass. Running it twice for `platform="both"` would double an LLM call for no benefit.

Extract the existing per-shared-question grounding loop (today inlined in `select_thread_pairs`, `thread_builder.py:267-306`) into a new function:

```python
def ground_thread_clips(
    entry_a: Dict, entry_b: Dict, transcript_a: Dict, transcript_b: Dict,
    shared_questions: List[str], num_clips: int, llm_fn: LLMFn, platform: str = "youtube",
) -> List[Dict]:
```

Same body as today's loop (calls `pick_thread_clips` per question with `avoid_ranges_a/b` non-overlap tracking, same early-stop at `num_clips`), just taking `shared_questions` as a parameter instead of computing it, and threading `platform` into each `pick_thread_clips` call.

`select_thread_pairs` becomes a thin wrapper: call `find_same_topic_pairs` once, then `ground_thread_clips` once — identical behavior to today for existing single-platform callers, no signature break.

## 3. `pipeline.generate_threads()` (`shorts_generator/pipeline.py`)

New parameter: `platform: str = "youtube"`, one of `"youtube"`, `"tiktok"`, `"both"`.

Flow change at `pipeline.py:525-528` (today: one `select_thread_pairs` call, abort on empty):

1. Ingest + abstracts unchanged (episode-parallel, `pipeline.py:505-509`).
2. Call `find_same_topic_pairs(entry_a, entry_b, num_clips, call_local_llm)` **once**, regardless of `platform` value.
3. If empty, `return []` (unchanged — no shared question means nothing groundable for either platform).
4. For each platform in the requested set (`["youtube"]`, `["tiktok"]`, or `["youtube", "tiktok"]` for `"both"`): call `ground_thread_clips(entry_a, entry_b, transcript_a, transcript_b, shared_questions, num_clips, call_local_llm, platform=platform)` → that platform's own `pairs` list.
5. **Abort condition changes:** `return []` only if *every* requested platform's `pairs` list is empty. Today's single-platform behavior (abort if the one list is empty) is the `platform="both"` behavior's degenerate single-item case, so this is a strict generalization, not a behavior change for existing callers. A platform failing to ground while another succeeds (e.g. a shared question grounds a valid 20s span for YouTube but no valid 28s+ span exists in the transcript for TikTok) is not a failure — that platform's list is just shorter, same "refuse rather than force" philosophy already used per-question.
6. Full-source video download (`pipeline.py:539-557`) is unaffected — still one download per episode total, done once before any platform's render loop, reused by both.
7. Render loop runs once per platform, over that platform's `pairs` list. Per-clip body is unchanged (`acquire_clip` ×2 → `synthesize_narration` ×2 → `render_narration_card` ×2 → `assemble_thread`) except output paths (§4) and a new post-assembly duration check for TikTok (§5). Results from all platforms accumulate into one flat `results` list, each entry gets a new `"platform"` key and a `"platform_index"` key (the 1-indexed position *within that platform's own list*, needed by §4 since flat list position and per-platform index diverge once there's more than one platform).

## 4. Output naming (`pipeline.py`, `shorts_generator/run_output.py`)

- Raw intermediates: `out_dir/raw/thesis_{i}_{platform}/` (was `raw/thesis_{i}/`), containing `clip_{i}_a.mp4`, `clip_{i}_b.mp4`, `thesis_{i}.mp3`, `bridge_{i}.mp3`, `intro_card_{i}.mp4`, `bridge_card_{i}.mp4` — unchanged leaf names, just nested one level deeper under the platform-suffixed dir. Here `i` is the platform-local index (`platform_index` from §3), matching how today's `i` is local to the single list.
- Final file: `out_dir/thesis_{i}_{platform}_{sanitize_title(title)}.mp4` (was `out_dir/thesis_{i}_{title}.mp4`).
- `write_thread_descriptions(out_dir, threads)` (`run_output.py:421-449`): today numbers blocks by flat `enumerate(threads, 1)`, and its docstring states that position must match the `i` used in the filename. With results flattened across platforms, flat-list position and per-platform filename index diverge (2 threads × 2 platforms = a 4-element list numbered 1-4, while filenames use `thesis_1_youtube`/`thesis_2_youtube`/`thesis_1_tiktok`/`thesis_2_tiktok`) — reintroducing the exact class of bug `ddda8cc` fixed. **Fix:** number using each entry's own `platform_index` (not flat list position), and tag the block with the platform:
  ```
  clip {platform_index} [{platform}] ({basename of clip_url})
  Title: {title}
  Description: {description}
  ```
  Numbering restarts at 1 for each platform group; the `[{platform}]` tag disambiguates "clip 1" between groups.
- `thread_results.json`: each entry gets the new `"platform"` and `"platform_index"` fields; otherwise unchanged shape.

## 5. TikTok duration verification (`pipeline.py`)

Code-enforced bounds (§1) are a steer on the LLM's pick, not a guarantee on the final assembled file — narration audio length is TTS-derived and not independently bounded. After `assemble_thread` writes a TikTok-platform final file, ffprobe its duration (ffmpeg is already a hard dependency via `thread_assembler.py`/`narration.py`) and log a warning to `progress.log` if it's under 60s:

```
[pipeline/local] WARNING: TikTok cut thesis_{i}_tiktok is {duration:.1f}s, under the 60s Creator Rewards minimum
```

Non-fatal (does not delete the file or abort the run) — this is a defense-in-depth signal for the operator, not a hard gate, since a retry-until-60s loop is out of scope for this change and the §1 bounds should make this warning rare in practice.

## 6. CLI (`main.py`)

New flag: `--platform {youtube,tiktok,both}`, default `youtube`. Thread-only, following the existing pattern at `main.py:97-106` where `--clip-type` choices are documented together.

Two mirror-image validations, matching the existing ignored-flags block at `main.py:128-159` (which lists every shorts/chapters flag that thread mode ignores):
- `--clip-type thread` with `--platform` omitted or explicit: passed straight to `generate_threads(platform=args.platform)`.
- `--clip-type shorts` or `chapters` with `--platform` explicitly passed (non-default): add `--platform {args.platform}` to the existing ignored-flags warning list — `--platform` is the first flag that's meaningful for thread and meaningless everywhere else, the inverse of every other entry in that block, so it needs the same warning in the other direction or it will silently no-op for shorts/chapters users.

## 7. Webapp (`shorts_generator/webapp.py`, `templates/index.html`)

Landed as a follow-up commit after §1-6 (core + CLI) are verified independently — see Rollout below.

- `_run_thread_job(url_a, url_b, num_clips, platform="youtube")` → `generate_threads(url_a, url_b, num_clips=num_clips, platform=platform, on_output_dir=_on_output_dir)`.
- `/run` route (`webapp.py:254-276`): reads `platform = request.form.get("platform", "youtube")`, validates membership in `{"youtube", "tiktok", "both"}` (400 on invalid, same pattern as the existing `num_clips` int-parse error handling at `webapp.py:259-262`), passes into the `_run_thread_job` thread args.
- `_serialize_thread_results` (`webapp.py:192-224`): each serialized thread item includes `"platform"`.
- `templates/index.html`: thread mode's form gets a platform `<select>` (YouTube / TikTok / Both) next to the existing `num_clips` field, submitted as `platform` in the POST body.
- `_relative_clip_path`/`_clip_display_url` (`webapp.py:~130`, fixed once already in `95fa8b1` for nested `raw/thesis_N/` paths): the nesting changes from `raw/thesis_{i}/` to `raw/thesis_{i}_{platform}/` — one path segment's name changes, not the nesting depth, so the existing fix should cover it, but this is explicitly called out for a regression test (§8) since this exact code has already broken once on a path-shape change.

## Removed from scope

- No retry-until-≥60s loop for TikTok cuts that land short after assembly — warning only (§5).
- No change to aspect ratio, assembly structure, or narration-length prompt limits (the "max ~12 words" thesis/bridge lines stay the same for both platforms).
- No platform concept added to the regular (non-thread) Shorts/chapters pipelines — `generate_shorts`'s existing `--aspect-ratio` flag is untouched and out of scope here.

## Rollout

Two commits: (1) `thread_builder.py` + `pipeline.py` + `run_output.py` + `main.py` (core generation + CLI), verified with its own test pass before (2) `webapp.py` + `templates/index.html` (dashboard UI) lands on top. Keeps the picker/pipeline change — the risky part, since it touches LLM prompt bounds and output-file identity — independently verifiable before the Flask/HTML layer touches it.

## Testing plan

- `thread_builder.py`: `_sanitize_clip_span` accepts/rejects at each platform's exact `min_clip`/`max_clip` boundary (e.g. TikTok: 27.9s rejected, 28.0s accepted, 40.0s accepted, 40.1s rejected). Prompt formatting test confirms `{clip_range}`/`{total_range}` substitute correctly per platform and the "do not use extra length" line is present in both. `ground_thread_clips` given a pre-built `shared_questions` list produces grounded results without calling stage A (mock/spy on `find_same_topic_pairs` asserting zero calls).
- `pipeline.generate_threads`: mocked-LLM test for `platform="both"` — asserts `find_same_topic_pairs` called exactly once, two platforms' worth of files produced with correct `thesis_{i}_{platform}_*` naming, `thread_results.json` entries carry correct `platform`/`platform_index`, and a run where TikTok grounds zero pairs but YouTube grounds one still returns the YouTube result instead of aborting. Existing `platform="youtube"`-default tests (implicit today) must stay green unchanged.
- `run_output.write_thread_descriptions`: numbering restarts per platform group and matches each entry's `platform_index`/filename exactly (regression test directly targeting the `ddda8cc` bug class).
- New test: ffprobe-based duration check emits the warning log line for a stubbed sub-60s TikTok assembly and does *not* emit it for a stubbed ≥60s one; confirms non-fatal (file and run result untouched either way).
- `main.py`: `--platform` accepted for `--clip-type thread`; `--platform tiktok --clip-type shorts` triggers the ignored-flags warning.
- `webapp.py`: `_relative_clip_path`/`_clip_display_url` resolve correctly against a `raw/thesis_1_tiktok/clip_1_a.mp4`-shaped path (regression test for the `95fa8b1` fix under the new nesting); `/run` rejects an invalid `platform` value with 400.
