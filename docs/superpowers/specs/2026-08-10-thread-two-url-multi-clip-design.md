# Thread: Two-URL Input + Multi-Clip Output — Design Spec

## Problem

Thread mode (`docs/superpowers/specs/2026-08-09-thread-compilation-design.md`) currently only works by auto-scanning the whole local corpus (`output/<Title>/` folders that already have a cached transcript) for a same-topic pair, and it always produces exactly one clip. Two problems in practice:

1. **No way to pick the pair.** The user often already knows which two episodes should be threaded together, but the dashboard's thread mode takes no input at all — it's entirely at the mercy of whatever `find_same_topic_pair` picks out of the full corpus, which requires the corpus to already contain both episodes (via a full pipeline run) and requires the LLM scan to land on that exact pair among however many others are in the corpus.
2. **Only one clip per run.** A same-topic pair between two full episodes often supports more than one distinct shared-question thread, but today's pipeline stops at the first grounded pick.

## Solution shape

Replace the no-input "scan the corpus" thread mode with two explicit YouTube URL fields in the dashboard (episode A, episode B), reusing caption-only ingest (`shorts_generator/local/caption_ingest.py`, already built for `ingest_corpus.py`) so no full video download is needed just to thread two episodes. The existing "Num clips" field (already in the UI for Shorts mode) now also applies to threads: the pipeline looks for up to N distinct shared-question pairs between the same two episodes and renders a clip for each one it can ground.

## 1. Ingest guard (`shorts_generator/local/caption_ingest.py`)

`ingest_captions(url)` currently always fetches auto-captions and overwrites `full_source.json` at `resolve_output_dir(url)`. This is safe for a URL that has never been processed, but unsafe for a URL that was already run through the full pipeline (real Whisper transcript, possibly with `full_source.mp4` still present) — a caption-only re-ingest would silently downgrade that episode's cached transcript to lower-fidelity YouTube auto-captions.

**Fix:** before fetching, check whether `paths.source_json` already exists. If it does, skip the `yt-dlp`/parse work entirely, log that the episode is already in the corpus, and return metadata read from the existing file (duration, segment count) instead of freshly-fetched ones. This applies regardless of how the existing `full_source.json` got there (prior full run, or a prior caption-only ingest) — idempotent either way.

This guard is a correctness fix independent of the dashboard change and benefits `ingest_corpus.py` too (re-running it over a URL list becomes a safe no-op for URLs already ingested).

## 2. Multi-pair stage A (`shorts_generator/thread_builder.py`)

Today's stage A (`find_same_topic_pair`) screens an arbitrary-length corpus for the single best-matching pair by index. With the pair already fixed by the dashboard's two URL fields, stage A's job changes: given exactly episode A and episode B's abstracts, find **up to N distinct shared questions** both episodes independently address.

New function: `find_same_topic_pairs(entry_a: Dict, entry_b: Dict, num_pairs: int, llm_fn: LLMFn) -> List[str]` — returns a list of shared-question strings (possibly empty, possibly shorter than `num_pairs`). New prompt variant (`SAME_TOPIC_MULTI_PROMPT`) adapted from `SAME_TOPIC_SYSTEM_PROMPT`: same hard-gate rules (broad subject overlap doesn't count, must be a real shared question, no forcing a stretch), but framed as "list every genuinely distinct shared question these two abstracts both answer, up to {num_pairs}" instead of "find the one best pair across this whole list." Sanitization mirrors `_sanitize_topic_pick`'s strictness per item (non-empty string, dedupe near-identical questions) and drops the whole list down to `[]` if the response doesn't parse.

Stage B (`pick_thread_clips`) is unchanged as a function, but the driving loop calls it once per shared question from stage A, in order, and enforces **no span reuse**: if a grounded pick's `clip_a` or `clip_b` span overlaps (by time range) a span already used by an earlier accepted pick in the same episode, that pick is discarded (not retried) and the loop moves to the next shared question. This stops the same "best" quote in an episode from being recycled across multiple threads.

The loop stops once `num_clips` grounded, non-overlapping pairs are collected, or the shared-question list is exhausted — whichever comes first. **Fewer than `num_clips` groundable pairs is not an error**: same refuse-rather-than-force philosophy as the existing `no_match`/`grounded` checks in `thread_builder.py`. Zero groundable pairs is the existing "no thread" outcome, just phrased for the two-URL case instead of "grow the corpus."

## 3. Output layout (`shorts_generator/run_output.py`, `pipeline.py`)

One parent run dir per dashboard run: `output/_Threads/<run-slug>/`, slugged from episode A + B's titles (not from a single thesis, since a run can now produce multiple theses). For each accepted pair `i` (1-indexed, in the order stage A returned them):

- `clip_{i}.mp4` — final assembled thread (intro card → clip A → bridge card → clip B), same assembly as today
- `clip_{i}_a.mp4`, `clip_{i}_b.mp4` — the two source spans acquired from episode A/B (already produced today as `clip_a.mp4`/`clip_b.mp4` per thread, just index-suffixed now that a run can hold several)
- one shared `progress.log` for the whole run (all N clips' pipeline output, same `capture_progress_log` mechanism as today, opened once before the first clip starts rendering)
- `thread_results.json` — a list (was previously a single-object `thread_result.json`)

This mirrors the existing multi-clip Shorts layout (one run dir, `short1.mp4..shortN.mp4`), which the dashboard's per-clip download/delete UI already knows how to render — no new plumbing needed for that part.

## 4. `generate_threads()` signature (`shorts_generator/pipeline.py`)

```
generate_threads(url_a: str, url_b: str, num_clips: int = 1, base_dir=None, on_output_dir=None) -> List[Dict]
```

Behavior:
1. `ingest_captions(url_a)`, `ingest_captions(url_b)` (guarded per §1) → two run dirs.
2. Compute/reuse abstracts for just these two run dirs via `corpus.get_abstract_cached` directly (not `build_corpus`, which walks the whole `output/` tree — unneeded here since the pair is already fixed).
3. `find_same_topic_pairs(entry_a, entry_b, num_clips, llm_fn)`. If empty, return `[]` — caller (webapp) treats this the same as today's "no thread" case.
4. Resolve the parent run dir (§3), call `on_output_dir(run_dir)` once, before any per-clip rendering starts (same early-callback contract as today, so the dashboard can start tailing `progress.log` right away).
5. For each accepted, non-overlapping grounded pair (in order, up to `num_clips`): run the existing per-clip steps (`acquire_clip` ×2 → `synthesize_narration` ×2 → `render_narration_card` ×2 → `assemble_thread`) against `clip_{i}_a.mp4`/`clip_{i}_b.mp4`/`clip_{i}.mp4`, same as today's single-clip body just index-suffixed and looped.
6. Return the list of per-clip result dicts (each shaped like today's single result: `shared_question`, `thesis`, `bridge`, `episode_a`, `episode_b`, `clip_url`), write the same list to `thread_results.json`.

The old no-argument corpus-auto-scan path (`build_thread(base_dir=...)` called with no episode selection) is removed — `generate_threads` no longer has a zero-URL mode. `thread_builder.build_thread` and `find_same_topic_pair` (singular, whole-corpus scan) can stay in place unused by the dashboard, or be removed if nothing else calls them — confirm during implementation whether `build_thread`/`find_same_topic_pair` have any other caller before deleting.

## 5. Webapp (`shorts_generator/webapp.py`, `templates/index.html`)

**Form:** thread mode's `shorts-fields`-style block gets two URL inputs (`url_a`, `url_b`, both required when `clip_type=thread`) replacing the current no-URL note. The existing `num_clips` field is no longer hidden for thread mode — it's shown for both `shorts` and `thread`, same input, same default (3).

**Job/backend:** `_run_thread_job` takes `url_a`, `url_b`, `num_clips` from the form (mirroring how `_run_job` already reads `num_clips` for shorts), calls `generate_threads(url_a, url_b, num_clips, on_output_dir=_on_output_dir)`. Empty-list result (no groundable pair) sets `job.error` with dashboard-appropriate wording (paste two episodes that actually share a question) instead of today's "grow the corpus" message.

**Result serialization:** `_serialize_thread_result` (singular) → `_serialize_thread_results` (list), same shape/existence-check pattern as `_serialize_result` for shorts. Each item additionally carries `episode_a_download_url` / `episode_b_download_url` — computed via the existing `_clip_display_url`/`_clip_file_exists` helpers against `clip_{i}_a.mp4`/`clip_{i}_b.mp4`, same pattern already used for the final clip.

**Frontend:** `buildThreadCard` renders per result item: the final thread clip as the primary `<video>`, plus two secondary `<video>` elements for the episode A / episode B source spans, each labeled with that episode's title (`episode_a.title` / `episode_b.title`, already in the result). Source-clip videos are view-only — no independent delete button; deleting a thread result item removes `clip_{i}.mp4` + `clip_{i}_a.mp4` + `clip_{i}_b.mp4` together. Results panel renders one card per item in the returned list, same loop structure the Shorts results grid already uses.

## Removed from scope

- The old no-URL "scan whole local corpus automatically" thread mode and its dashboard UI (note text, hidden-field toggling) — replaced entirely, not kept as a fallback.
- Any UI for choosing *which* corpus-wide pair to thread — moot once the two URLs are explicit input.

## Testing plan

- `tests/test_local_caption_ingest.py`: new case — `ingest_captions` on a URL whose run dir already has `full_source.json` skips the fetch and returns the cached metadata untouched.
- `thread_builder.py`: new tests for `find_same_topic_pairs` (returns ≤N questions, drops malformed/empty items, empty list on a no-match response) and for the overlap-rejection logic in the pair-selection loop (a second pick whose span overlaps an already-accepted pick in the same episode is discarded, loop continues to the next candidate).
- `pipeline.generate_threads`: mocked-LLM/mocked-render test for the multi-clip loop — asserts N clips requested but only M < N groundable produces exactly M results without error, and that `on_output_dir` fires once before any per-clip render work.
