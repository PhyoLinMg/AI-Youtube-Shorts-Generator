"""Cross-run corpus index for multi-episode thread compilation (see
thread_builder.py). Walks every run folder under LOCAL_OUTPUT_DIR that has
both a cached transcript (full_source.json) and a persisted source URL
(source_url.txt -- see run_output.write_source_url) and produces one short
topical abstract per episode, cached alongside the transcript, so
thread_builder can screen every episode for a same-topic pair without
feeding N full transcripts into one LLM call.
"""
import json
import os
from typing import Dict, List, Optional

from .config import LOCAL_OUTPUT_DIR
from .highlights import LLMFn, _transcript_fingerprint, call_muapi_llm

ABSTRACT_SCHEMA_VERSION = 1

ABSTRACT_PROMPT = """Summarize this podcast transcript sample in one paragraph (120-200 words) covering only its SUBSTANTIVE topics and any specific claims, opinions, or arguments made -- not the format, not the guest's name, not general praise. A reader should be able to tell from this abstract alone whether this episode discusses the same specific question as another episode's abstract.

Transcript sample:
{sample}

Respond with plain text only, no markdown, no JSON."""


def _abstract_cache_path(run_dir: str) -> str:
    return os.path.join(run_dir, "corpus_abstract.json")


def _sample_transcript_text(transcript: Dict, max_chars: int = 6000) -> str:
    segments = transcript.get("segments", [])
    texts = [s.get("text", "") for s in segments]
    total = len(texts)
    if total == 0:
        return ""
    # Sample evenly across the whole episode (opening, middle, closing) rather
    # than just the first N chars -- a podcast's real topic often isn't
    # established until after intro/sponsor-read chatter, so head-truncation
    # biases the abstract toward the wrong part of the episode.
    budget_per_section = max_chars // 3
    thirds = [
        texts[: total // 3],
        texts[total // 3 : 2 * total // 3],
        texts[2 * total // 3 :],
    ]
    parts = [" ".join(section)[:budget_per_section] for section in thirds]
    return " ... ".join(parts)


def get_abstract_cached(run_dir: str, transcript: Dict, llm_fn: Optional[LLMFn] = None) -> str:
    """Compute (or reuse) a topical abstract for one episode's transcript,
    cached alongside it and invalidated the same way highlights/chapters
    caches are: by a content fingerprint of the transcript itself."""
    llm_fn = llm_fn or call_muapi_llm
    fingerprint = _transcript_fingerprint(transcript)
    cache_path = _abstract_cache_path(run_dir)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if (
                isinstance(cached, dict)
                and cached.get("transcript_fingerprint") == fingerprint
                and cached.get("schema_version") == ABSTRACT_SCHEMA_VERSION
                and cached.get("abstract")
            ):
                print(f"[corpus] reusing cached abstract: {cache_path}", flush=True)
                return cached["abstract"]
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            print(f"[corpus] cached abstract corrupted, recomputing: {cache_path}", flush=True)

    abstract = llm_fn(ABSTRACT_PROMPT.format(sample=_sample_transcript_text(transcript))).strip()

    tmp_path = cache_path + ".part"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "transcript_fingerprint": fingerprint,
                "schema_version": ABSTRACT_SCHEMA_VERSION,
                "abstract": abstract,
            },
            f,
            ensure_ascii=False,
        )
    os.replace(tmp_path, cache_path)
    return abstract


def list_corpus_run_dirs(base_dir: Optional[str] = None) -> List[str]:
    """Every run folder under base_dir that has both a cached transcript and
    a persisted source URL -- the two things thread-building needs from an
    episode regardless of whether its full_source.mp4 is still on disk."""
    base_dir = base_dir or LOCAL_OUTPUT_DIR
    if not os.path.isdir(base_dir):
        return []
    run_dirs = []
    for name in sorted(os.listdir(base_dir)):
        run_dir = os.path.join(base_dir, name)
        if not os.path.isdir(run_dir):
            continue
        source_url_path = os.path.join(run_dir, "source_url.txt")
        if not os.path.exists(os.path.join(run_dir, "full_source.json")):
            continue
        if not os.path.exists(source_url_path):
            continue
        # Match run_output.read_source_url's empty-value contract: a
        # whitespace-only source_url.txt counts as "no source URL", so
        # treat the run as ineligible rather than surfacing source_url: "".
        # Unreadable/non-UTF-8 files are treated the same way -- this read
        # happens before build_corpus's own per-run try/except ever runs,
        # so it needs its own error handling rather than propagating.
        try:
            with open(source_url_path, "r", encoding="utf-8") as f:
                if not f.read().strip():
                    continue
        except (OSError, UnicodeDecodeError) as e:
            print(f"[corpus] skipping {run_dir}: {e}", flush=True)
            continue
        run_dirs.append(run_dir)
    return run_dirs


def build_corpus(base_dir: Optional[str] = None, llm_fn: Optional[LLMFn] = None) -> List[Dict]:
    """[{"run_dir", "title", "source_url", "abstract"}] for every eligible
    run, computing/reusing each abstract as needed. Does NOT load full
    transcripts for the caller -- callers load the full transcript only for
    the episodes actually needed (see pipeline.generate_threads)."""
    entries = []
    for run_dir in list_corpus_run_dirs(base_dir):
        try:
            with open(os.path.join(run_dir, "full_source.json"), "r", encoding="utf-8") as f:
                transcript = json.load(f)
            with open(os.path.join(run_dir, "source_url.txt"), "r", encoding="utf-8") as f:
                source_url = f.read().strip()
            # get_abstract_cached lives inside this same try block: a live
            # llm_fn call across 100+ episodes can fail transiently (network
            # error, rate limit) just as easily as a file can be corrupted,
            # and either failure should skip only this one episode rather
            # than aborting the whole corpus build.
            abstract = get_abstract_cached(run_dir, transcript, llm_fn=llm_fn)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            print(f"[corpus] skipping {run_dir}: {e}", flush=True)
            continue
        except Exception as e:
            print(f"[corpus] skipping {run_dir}: abstract generation failed: {e}", flush=True)
            continue
        entries.append({
            "run_dir": run_dir,
            "title": os.path.basename(run_dir),
            "source_url": source_url,
            "abstract": abstract,
        })
    return entries
