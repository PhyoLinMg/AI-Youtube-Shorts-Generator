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
    text = " ".join(s.get("text", "") for s in segments)
    return text[:max_chars]


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
                return cached["abstract"]
        except json.JSONDecodeError:
            pass

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
        if (
            os.path.exists(os.path.join(run_dir, "full_source.json"))
            and os.path.exists(os.path.join(run_dir, "source_url.txt"))
        ):
            run_dirs.append(run_dir)
    return run_dirs


def build_corpus(base_dir: Optional[str] = None, llm_fn: Optional[LLMFn] = None) -> List[Dict]:
    """[{"run_dir", "title", "source_url", "abstract"}] for every eligible
    run, computing/reusing each abstract as needed. Does NOT load full
    transcripts for the caller -- thread_builder.build_thread loads the
    full transcript only for the two episodes actually picked."""
    entries = []
    for run_dir in list_corpus_run_dirs(base_dir):
        with open(os.path.join(run_dir, "full_source.json"), "r", encoding="utf-8") as f:
            transcript = json.load(f)
        with open(os.path.join(run_dir, "source_url.txt"), "r", encoding="utf-8") as f:
            source_url = f.read().strip()
        entries.append({
            "run_dir": run_dir,
            "title": os.path.basename(run_dir),
            "source_url": source_url,
            "abstract": get_abstract_cached(run_dir, transcript, llm_fn=llm_fn),
        })
    return entries
