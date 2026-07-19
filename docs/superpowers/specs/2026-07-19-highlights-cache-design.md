# Cache highlights across pipeline reruns

## Problem

`generate_shorts()` already avoids redundant work on a rerun of the same
video: `_run_api` skips redownload if `paths.source_video` exists and skips
re-transcription if `paths.source_json` parses cleanly; `_run_local`'s
`transcribe_local()` has its own mtime-keyed cache and skips re-whisper-ing
a file it's already transcribed. Neither mode does the equivalent for
`get_highlights()` — every call to `generate_shorts()` on the same video
re-invokes the LLM (1 content-type-detect call + 1 call per 20-minute
transcript chunk, each with up to 3 retries on malformed JSON), even when
the transcript is byte-identical to a previous run.

This matters most for `_run_local` with `LLM_PROVIDER=openrouter` and
`OPENROUTER_MODEL` set to a DeepSeek model: rerunning the pipeline while
iterating on something downstream (captions, hook card, crop framing)
burns a fresh paid DeepSeek call every time, for output that would be
identical to what's already sitting in `result.json` from the last run.

## Fix

Add a highlights cache, sibling to the existing transcript cache, keyed by
a content fingerprint of the transcript plus the requested `num_clips` —
not by "was the transcript cache hit," since `_run_local`'s
`transcribe_local()` doesn't expose that to the caller (it silently
returns either a cached or freshly-transcribed dict with no marker
distinguishing the two). A content fingerprint works identically for both
modes with no changes to either transcriber.

1. **New path.** `RunPaths` gets a `highlights_json` field:
   `os.path.join(root, "highlights.json")`, set in `resolve_output_dir`.

2. **New function in `highlights.py`: `get_highlights_cached`.** Wraps
   `get_highlights`:
   - Compute `_transcript_fingerprint(transcript)` — a SHA-256 hex digest
     of `{"duration": ..., "segments": ...}` serialized with sorted keys.
   - If `cache_path` exists, parses as JSON, and its stored
     `transcript_fingerprint` and `num_clips` both match the current
     request → return its cached `highlights` list directly, skip the LLM
     entirely. `num_clips` mismatch (or any other mismatch/corruption)
     falls through to a full recompute — no partial reuse.
   - Otherwise call `get_highlights(transcript, num_clips, llm_fn)` as
     today, then atomically write `{"transcript_fingerprint", "num_clips",
     "highlights"}` to `cache_path` (temp-file-then-`os.replace`, matching
     `_run_api`'s existing `.part` pattern for `source_json`) before
     returning the result.
   - A corrupted/unparseable cache file is treated as a miss (log and
     recompute), same as `_run_api` already does for a corrupted
     `source_json`.

3. **Wire into both pipeline paths.** `_run_local` and `_run_api` each
   replace their `get_highlights(transcript, num_clips=num_clips,
   llm_fn=...)` call with `get_highlights_cached(transcript,
   num_clips=num_clips, cache_path=paths.highlights_json, llm_fn=...)`.
   No other change to either function — the return shape is identical
   (`{"highlights": [...]}"`), so downstream `top = sorted(...)[:num_clips]`
   and everything after it is untouched.

No new flag. Matches the existing zero-flag automatic caching UX for
source video and transcript: deleting `highlights.json` (or the whole run
folder) forces a fresh LLM call, same as deleting `full_source.json`
forces a fresh transcribe.

## Files touched

- `shorts_generator/run_output.py` — add `highlights_json` to `RunPaths`
  and `resolve_output_dir`.
- `shorts_generator/highlights.py` — add `_transcript_fingerprint` and
  `get_highlights_cached`.
- `shorts_generator/pipeline.py` — `_run_local` and `_run_api` call
  `get_highlights_cached` instead of `get_highlights`.
- `tests/test_pipeline.py` and `tests/test_webapp.py` — both construct
  `RunPaths(...)` directly (`_paths()` / `_fake_run_paths()` helpers);
  adding a required dataclass field breaks both unless updated. All 6
  `monkeypatch.setattr(pipeline_module, "get_highlights", ...)` call sites
  in `test_pipeline.py` must move to `get_highlights_cached` (with the
  lambda's signature gaining `cache_path`) — patching the old name would
  go silently inert once `pipeline.py` no longer calls it, since
  `get_highlights_cached` calls the real `get_highlights` from inside
  `highlights.py`'s own module namespace, not through `pipeline_module`.

## Out of scope

- Not touching `call_highlight_api`'s retry count, the content-type-detect
  call, or chunking — this is purely about skipping repeat work across
  separate `generate_shorts()` invocations on the same video, not reducing
  calls within a single invocation (that was explicitly ruled out this
  round).
- Not adding a force-refresh flag (explicitly declined — manual file
  deletion is the existing mechanism for the transcript cache and stays
  consistent here).
- Not touching MuAPI's `call_muapi_llm` path specially — `_run_api` gets
  the same caching wrapper as `_run_local`, since the mechanism is
  provider-agnostic, even though the motivating cost concern (DeepSeek via
  OpenRouter) only applies to `_run_local`.

## Testing

- `tests/test_highlights.py`: `_transcript_fingerprint` is stable for
  identical transcripts and changes when segments/duration change.
  `get_highlights_cached` — cache miss (no file) calls the LLM and writes
  the cache; cache hit (matching fingerprint + num_clips) returns cached
  highlights without calling the LLM (assert the stub `llm_fn` was never
  invoked); fingerprint mismatch and num_clips mismatch both fall back to
  a fresh call and overwrite the cache; corrupted cache file falls back to
  a fresh call.
- `tests/test_pipeline.py`: update `_paths()` and all 6
  `get_highlights`-monkeypatch sites (see Files touched) so the existing
  behavioral tests keep exercising the real code path through
  `get_highlights_cached` rather than silently no-op-patching a name
  `pipeline.py` no longer calls.
- `tests/test_webapp.py`: update `_fake_run_paths()` for the new field —
  no behavioral assertions there depend on it, just construction.
- Confirm `resolve_output_dir`'s new `highlights_json` path doesn't
  collide with anything `test_run_output.py` asserts about folder
  contents (`_run_mtime`'s `os.walk` picks it up automatically; no special
  handling needed there).
