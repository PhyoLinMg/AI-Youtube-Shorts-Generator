# Thread Compilation — Design Spec

## Problem

The existing pipeline (Shorts, Chapters) always operates on ONE source video at a time: download → transcribe → select → render. Every output, however it's packaged, is still footage trimmed from a single source — a human reviewer (or YouTube's reused-content policy) can always ask "could this have been produced by just trimming the one video?" and get "yes."

A **thread** is a new output type built from TWO existing episodes already in the local corpus (`output/<Title>/`), combined around one shared question neither episode answers alone. The pairing itself — not just the editing — is the thing that can't be produced by trimming a single source.
 
## Hard constraint: same topic, no negotiation

Earlier iteration paired a JD Vance clip (a VP's promise to look for alien evidence) with a Neil deGrasse Tyson clip (an astrophysicist's general epistemology about unidentified phenomena). Both mention aliens/UFOs, but they don't answer the same question — one is a personal pledge, the other is a methodological point. The user rejected this as a stretch and set a hard rule:

**Topic match is a gate checked BEFORE the thesis is written, not a quality the thesis argues for afterward.** If no pair of episodes in the corpus shares a genuine same question, the correct output is **no thread at all** — not a loosely-connected pair. A thread builder that always returns *something* will always return something stretched. The refusal path is not an edge case to handle; it's the default outcome whenever the corpus doesn't support a real pairing.

Concretely: the LLM stage that screens episode pairs must be able to say `no_match: true`, and the pipeline must treat that as success (nothing to build today), not as an error to route around.

## Incident: source/caption misalignment (2026-08-09)

While building a manual proof-of-concept thread, captions were burned onto a re-downloaded clip using a *cached* transcript's absolute timestamps, without verifying the re-downloaded source actually matched the video that transcript was made from. It didn't — the URL given for the second episode was for a different-length upload (5778s cached vs. 6530s actual at the wrong URL). The result: real, professionally-rendered captions describing content that had nothing to do with the audio actually playing under them ("So many pilot sightings, they don't know" burned over a segment about cows and photosynthesis). The mismatch was invisible without cross-checking video duration.

**Rule going forward:** any time a clip is re-acquired from a source URL instead of cut from an already-local `full_source.mp4`, the live source's duration MUST be checked against the cached transcript's duration before any timestamp from that transcript is trusted. A mismatch beyond a couple of seconds is fatal (raise), never a warning to route around. This is implemented as `SourceMismatchError` in `local/thread_source.py`.

A second, smaller lesson from the same incident: cached highlight/segment timestamps land on Whisper segment boundaries, which don't always land on the actual start of a spoken word (a "hook" line can start mid-segment). When cutting a clip fresh, prefer the nearest word-level timestamp at or after the intended start over the raw segment boundary.

## Why episodes need a persisted source URL

`full_source.mp4` (700MB+) gets pruned for disk space over time — expected at 100+ episodes. Once pruned, the only way to re-acquire a specific span is the original YouTube URL, and today's pipeline never saves it anywhere retrievable (`result.json`'s `source_video_url` is a local path in local mode, not the origin URL). This makes any future corpus-wide feature — thread compilation, or anything else operating across old episodes — dependent on the operator's memory. Fixed by persisting `source_url.txt` per run, written once, unconditionally, regardless of mode.

## Corpus reality check

As of this design, the local corpus has 3 cached episodes on three unrelated topics (US politics, UFO/alien skepticism, longevity science). A strict same-topic gate against this corpus will legitimately return "no match" — this is not a bug to work around. Thread compilation becomes usable as the corpus grows (naturally, as more episodes get processed locally) or via deliberate backfill (re-transcribing more of the existing channel catalog). The feature is built now so it's ready the moment two same-topic episodes exist; it is not expected to produce a real thread on day one.
