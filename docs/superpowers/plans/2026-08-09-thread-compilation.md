# Thread Compilation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add a third output type — a "thread" — that combines two already-processed episodes from the local corpus around one shared question, narrated by a consistent ElevenLabs voice, with a hard same-topic gate that refuses to build anything when no genuine pairing exists in the corpus.

**Architecture:** a corpus index (`corpus.py`) turns every locally-cached episode into a short topical abstract; a two-stage LLM picker (`thread_builder.py`) first gates on same-topic (refusing outright if no pair qualifies), then grounds exact clip spans + thesis/bridge narration text in the two chosen full transcripts; a source-acquisition helper (`local/thread_source.py`) cuts each clip from the still-local `full_source.mp4` when present, or re-downloads just the needed span with a hard duration-mismatch check before trusting any cached timestamp; a narration module (`local/narration.py`) turns thesis/bridge text into ElevenLabs audio composited onto a text card matching the channel's existing hook-card typography; an assembler (`local/thread_assembler.py`) re-encode-concats the ordered segments to one normalized spec; `pipeline.py` wires all of it into `generate_threads()`; `main.py` exposes `--clip-type thread`.

**Tech Stack:** Python (`shorts_generator/corpus.py`, `shorts_generator/thread_builder.py`, `shorts_generator/local/narration.py`, `shorts_generator/local/thread_assembler.py`, `shorts_generator/local/thread_source.py`, `shorts_generator/pipeline.py`, `shorts_generator/run_output.py`, `shorts_generator/config.py`, `main.py`), pytest, ffmpeg/ffprobe, yt-dlp, ElevenLabs Python SDK.

**Spec:** `docs/superpowers/specs/2026-08-09-thread-compilation-design.md`

---

### Task 1: Persist each run's original source URL

**Files:**
- Modify: `shorts_generator/run_output.py`
- Test: `tests/test_run_output.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_run_output.py`, after `test_resolve_output_dir_builds_chapters_paths`:

```python
def test_resolve_output_dir_includes_source_url_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_output.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"title": "How To Build A Startup"}),
    )
    paths = run_output.resolve_output_dir(
        "https://www.youtube.com/watch?v=abc123", base_dir=str(tmp_path)
    )
    assert paths.source_url_txt == os.path.join(paths.root, "source_url.txt")


def test_write_source_url_then_read_source_url_roundtrips(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_output.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"title": "How To Build A Startup"}),
    )
    paths = run_output.resolve_output_dir(
        "https://www.youtube.com/watch?v=abc123", base_dir=str(tmp_path)
    )
    run_output.write_source_url(paths, "https://www.youtube.com/watch?v=abc123")
    assert run_output.read_source_url(paths) == "https://www.youtube.com/watch?v=abc123"


def test_read_source_url_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_output.requests, "get",
        lambda *a, **k: _FakeResponse(200, {"title": "How To Build A Startup"}),
    )
    paths = run_output.resolve_output_dir(
        "https://www.youtube.com/watch?v=abc123", base_dir=str(tmp_path)
    )
    assert run_output.read_source_url(paths) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_output.py -k source_url -v`
Expected: FAIL with `AttributeError: 'RunPaths' object has no attribute 'source_url_txt'`

- [ ] **Step 3: Add the field, wire it into `resolve_output_dir`, add read/write helpers**

In `shorts_generator/run_output.py`, modify the `RunPaths` dataclass:

```python
@dataclass
class RunPaths:
    root: str
    shorts_dir: str
    chapters_dir: str
    source_video: str
    source_json: str
    highlights_json: str
    chapters_json: str
    result_json: str
    chapters_result_json: str
    source_url_txt: str
    progress_log: str
```

Modify `resolve_output_dir`'s return statement to add the new path:

```python
        chapters_result_json=os.path.join(root, "chapters_result.json"),
        source_url_txt=os.path.join(root, "source_url.txt"),
        progress_log=os.path.join(root, "progress.log"),
    )
```

Add two new functions after `resolve_output_dir`:

```python
def write_source_url(paths: RunPaths, youtube_url: str) -> None:
    """Persist the original source URL so a pruned full_source.mp4 (deleted
    to save disk once a channel has 100+ episodes) can still be
    re-acquired later -- see corpus.py / local/thread_source.py, which need
    to re-download a specific clip's source long after the original run."""
    with open(paths.source_url_txt, "w", encoding="utf-8") as f:
        f.write(youtube_url.strip())


def read_source_url(paths: RunPaths) -> Optional[str]:
    if not os.path.exists(paths.source_url_txt):
        return None
    with open(paths.source_url_txt, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content or None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_output.py -k "source_url" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Fix the `RunPaths(...)` construction helper in other test files**

`RunPaths` is a dataclass with no defaults, constructed manually in `tests/test_pipeline.py`'s `_paths()` helper — it will now fail with a missing-argument `TypeError`. Modify `tests/test_pipeline.py`'s `_paths()`:

```python
def _paths(tmp_path):
    root = str(tmp_path / "Video_Title")
    shorts_dir = os.path.join(root, "Shorts")
    chapters_dir = os.path.join(root, "Chapters")
    os.makedirs(shorts_dir, exist_ok=True)
    os.makedirs(chapters_dir, exist_ok=True)
    return RunPaths(
        root=root,
        shorts_dir=shorts_dir,
        chapters_dir=chapters_dir,
        source_video=os.path.join(root, "full_source.mp4"),
        source_json=os.path.join(root, "full_source.json"),
        highlights_json=os.path.join(root, "highlights.json"),
        chapters_json=os.path.join(root, "chapters.json"),
        result_json=os.path.join(root, "result.json"),
        chapters_result_json=os.path.join(root, "chapters_result.json"),
        source_url_txt=os.path.join(root, "source_url.txt"),
        progress_log=os.path.join(root, "progress.log"),
    )
```

Run `grep -rn "RunPaths(" tests/ shorts_generator/` and fix every other manual construction the same way (add `source_url_txt=os.path.join(root, "source_url.txt"),` in the same position).

- [ ] **Step 6: Run the full test file to verify nothing else broke**

Run: `python -m pytest tests/test_run_output.py tests/test_pipeline.py -v`
Expected: PASS (all tests, including pre-existing ones)

- [ ] **Step 7: Wire `write_source_url` into both pipeline entry points**

In `shorts_generator/pipeline.py`, modify `generate_shorts` — find:

```python
    paths = paths or resolve_output_dir(youtube_url)
    with capture_progress_log(paths.progress_log):
```

Replace with:

```python
    paths = paths or resolve_output_dir(youtube_url)
    write_source_url(paths, youtube_url)
    with capture_progress_log(paths.progress_log):
```

Do the same in `generate_chapters` (identical `paths = paths or resolve_output_dir(youtube_url)` line exists there too).

Update the import at the top of `pipeline.py`:

```python
from .run_output import RunPaths, capture_progress_log, resolve_output_dir, write_chapter_descriptions, write_descriptions, write_source_url
```

- [ ] **Step 8: Write a test confirming the pipeline writes the source URL**

Add to `tests/test_pipeline.py`, near the other `generate_shorts`-level tests (check the file for where full-pipeline tests live and match that style — they monkeypatch `pipeline_module.resolve_output_dir` to return the `_paths(tmp_path)` fixture):

```python
def test_generate_shorts_persists_source_url(tmp_path, monkeypatch):
    paths = _paths(tmp_path)
    monkeypatch.setattr(pipeline_module, "resolve_output_dir", lambda url: paths)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", lambda *a, **k: [{"clip_url": "/tmp/out/Short-01.mp4"}])

    pipeline_module.generate_shorts("https://www.youtube.com/watch?v=xyz", mode="local", num_clips=1)

    with open(paths.source_url_txt) as f:
        assert f.read().strip() == "https://www.youtube.com/watch?v=xyz"
```

If `_fake_transcript`/`_fake_highlights_result`/`local_downloader_module`/`local_transcriber_module`/`local_clipper_module` aren't already imported/defined at module level in `tests/test_pipeline.py`, they already are (see the top of that file) — reuse them as-is.

- [ ] **Step 9: Run test to verify it passes**

Run: `python -m pytest tests/test_pipeline.py -k persists_source_url -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add shorts_generator/run_output.py shorts_generator/pipeline.py tests/test_run_output.py tests/test_pipeline.py
git commit -m "feat: persist each run's original source URL

Full_source.mp4 gets pruned for disk space once a channel has 100+
episodes, and today nothing records the original YouTube URL anywhere
retrievable once that happens. This blocks any future corpus-wide
feature from re-acquiring a specific episode's audio. Write it once,
unconditionally, per run."
```

---

### Task 2: `transcribe_local` accepts an explicit model-size override

**Files:**
- Modify: `shorts_generator/local/transcriber.py`
- Test: `tests/test_local_transcriber.py`

- [ ] **Step 1: Write the failing test**

Check `tests/test_local_transcriber.py` for the existing fake-`WhisperModel` monkeypatch pattern (it monkeypatches `faster_whisper.WhisperModel` to a fake class capturing constructor args). Add a test following that same pattern:

```python
def test_transcribe_local_model_size_overrides_config_default(tmp_path, monkeypatch):
    captured = {}

    class _FakeModel:
        def __init__(self, model_name, device, compute_type):
            captured["model_name"] = model_name

        def transcribe(self, **kwargs):
            return iter([]), type("Info", (), {"duration": 1.0})()

    monkeypatch.setattr(local_transcriber_module.config, "LOCAL_WHISPER_MODEL", "base")
    monkeypatch.setitem(sys.modules, "faster_whisper", type("FW", (), {"WhisperModel": _FakeModel}))

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake")
    local_transcriber_module.transcribe_local(str(media), model_size="small")

    assert captured["model_name"] == "small"
```

Adjust the fake-`WhisperModel` injection to match whatever monkeypatch style the existing tests in that file already use (import `sys` and `local_transcriber_module` at the top the same way the rest of the file does) — don't reinvent it if a fixture already exists for this.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_local_transcriber.py -k model_size_overrides -v`
Expected: FAIL with `TypeError: transcribe_local() got an unexpected keyword argument 'model_size'`

- [ ] **Step 3: Add the parameter**

In `shorts_generator/local/transcriber.py`, modify `transcribe_local`'s signature and body:

```python
def transcribe_local(media_path: str, language: Optional[str] = None, model_size: Optional[str] = None) -> Dict:
    """Run faster-whisper on a local file path, caching the result as .json.

    model_size overrides LOCAL_WHISPER_MODEL for this call only -- used by
    thread compilation's source re-acquisition path, which re-transcribes a
    short (~30s) span and can afford a larger, more accurate model than the
    pipeline-wide default without changing the cost of every Shorts run.
    """
    cache_path = _transcript_cache_path(media_path)
    if cache_path.exists():
        source_mtime = os.path.getmtime(media_path)
        cache_mtime = cache_path.stat().st_mtime
        if cache_mtime >= source_mtime:
            print(f"[transcribe/local] reusing cached transcript: {cache_path}", flush=True)
            cached = _load_json_cache(cache_path)
            if not cached["segments"] or cached["duration"] <= 0.0:
                print(f"[transcribe/local] cache is empty/invalid, deleting: {cache_path}", flush=True)
                cache_path.unlink(missing_ok=True)
            else:
                print(
                    f"[transcribe/local] {len(cached['segments'])} cached segments, "
                    f"{cached['duration']:.0f}s of audio",
                    flush=True,
                )
                return cached

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    device = _resolve_device()
    compute_type = "float16" if device == "cuda" else "int8"
    model_name = model_size or LOCAL_WHISPER_MODEL
    print(f"[transcribe/local] faster-whisper model={model_name} device={device}", flush=True)

    from ..config import LOCAL_WHISPER_VAD_FILTER, LOCAL_WHISPER_VAD_PARAMETERS

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
```

The rest of the function body (from `transcribe_kwargs = {...}` onward) is unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_local_transcriber.py -v`
Expected: PASS (all tests, including pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/local/transcriber.py tests/test_local_transcriber.py
git commit -m "feat: let transcribe_local override the whisper model size per call

Thread compilation's source re-acquisition path re-transcribes a short
span and needs better accuracy than the pipeline-wide base-model
default, without paying that cost on every Shorts run."
```

---

### Task 3: Corpus index (`corpus.py`)

**Files:**
- Create: `shorts_generator/corpus.py`
- Test: `tests/test_corpus.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_corpus.py`:

```python
import json
import os

from shorts_generator import corpus


def _write_episode(base_dir, name, duration=100.0, source_url="https://example.com/v1"):
    run_dir = os.path.join(base_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "full_source.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": duration, "segments": [{"start": 0.0, "end": 5.0, "text": "hello world"}]}, f)
    with open(os.path.join(run_dir, "source_url.txt"), "w", encoding="utf-8") as f:
        f.write(source_url)
    return run_dir


def test_list_corpus_run_dirs_requires_both_transcript_and_source_url(tmp_path):
    complete = _write_episode(str(tmp_path), "Complete_Episode")
    incomplete_dir = os.path.join(str(tmp_path), "Incomplete_Episode")
    os.makedirs(incomplete_dir, exist_ok=True)
    with open(os.path.join(incomplete_dir, "full_source.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": 50.0, "segments": []}, f)
    # no source_url.txt written for Incomplete_Episode

    run_dirs = corpus.list_corpus_run_dirs(base_dir=str(tmp_path))

    assert run_dirs == [complete]


def test_list_corpus_run_dirs_empty_base_dir_returns_empty_list(tmp_path):
    assert corpus.list_corpus_run_dirs(base_dir=str(tmp_path / "does_not_exist")) == []


def test_get_abstract_cached_calls_llm_once_then_reuses_cache(tmp_path):
    run_dir = _write_episode(str(tmp_path), "Episode_One")
    transcript = json.load(open(os.path.join(run_dir, "full_source.json")))

    calls = []
    def fake_llm(prompt):
        calls.append(prompt)
        return "an abstract about hello world"

    first = corpus.get_abstract_cached(run_dir, transcript, llm_fn=fake_llm)
    second = corpus.get_abstract_cached(run_dir, transcript, llm_fn=fake_llm)

    assert first == "an abstract about hello world"
    assert second == "an abstract about hello world"
    assert len(calls) == 1


def test_get_abstract_cached_invalidates_on_transcript_change(tmp_path):
    run_dir = _write_episode(str(tmp_path), "Episode_Two")
    transcript_v1 = json.load(open(os.path.join(run_dir, "full_source.json")))

    calls = []
    def fake_llm(prompt):
        calls.append(prompt)
        return f"abstract {len(calls)}"

    corpus.get_abstract_cached(run_dir, transcript_v1, llm_fn=fake_llm)

    transcript_v2 = {**transcript_v1, "segments": [{"start": 0.0, "end": 5.0, "text": "a different episode entirely"}]}
    result = corpus.get_abstract_cached(run_dir, transcript_v2, llm_fn=fake_llm)

    assert result == "abstract 2"
    assert len(calls) == 2


def test_build_corpus_returns_title_source_url_and_abstract(tmp_path):
    _write_episode(str(tmp_path), "My_Episode", source_url="https://example.com/my-episode")

    entries = corpus.build_corpus(base_dir=str(tmp_path), llm_fn=lambda prompt: "summary text")

    assert len(entries) == 1
    assert entries[0]["title"] == "My_Episode"
    assert entries[0]["source_url"] == "https://example.com/my-episode"
    assert entries[0]["abstract"] == "summary text"
    assert entries[0]["run_dir"].endswith("My_Episode")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shorts_generator.corpus'`

- [ ] **Step 3: Create `shorts_generator/corpus.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/corpus.py tests/test_corpus.py
git commit -m "feat: add cross-run corpus index for thread compilation

One cached topical abstract per locally-transcribed episode, so
thread_builder can screen the whole corpus for a same-topic pair
without feeding every full transcript into one LLM call."
```

---

### Task 4: Thread builder — same-topic gate + clip picker (`thread_builder.py`)

**Files:**
- Create: `shorts_generator/thread_builder.py`
- Test: `tests/test_thread_builder.py`

This is the task that makes "no negos" real: `find_same_topic_pair` must return `None` (not a weak match) whenever the corpus doesn't support a genuine same-question pairing, and `build_thread` must propagate that `None` all the way out rather than substituting a fallback.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_thread_builder.py`:

```python
import json
import os

import pytest

from shorts_generator import thread_builder


def _corpus_entry(idx, title, abstract, run_dir):
    return {"run_dir": run_dir, "title": title, "source_url": f"https://example.com/{idx}", "abstract": abstract}


def test_find_same_topic_pair_returns_none_on_no_match_response():
    corpus = [
        _corpus_entry(0, "Politics Ep", "discusses tax policy", "/tmp/a"),
        _corpus_entry(1, "Science Ep", "discusses black holes", "/tmp/b"),
    ]
    llm_fn = lambda prompt: json.dumps({"no_match": True, "episode_a_index": None, "episode_b_index": None, "shared_question": ""})

    assert thread_builder.find_same_topic_pair(corpus, llm_fn) is None


def test_find_same_topic_pair_returns_none_with_fewer_than_two_entries():
    corpus = [_corpus_entry(0, "Solo Ep", "abstract", "/tmp/a")]
    llm_fn = lambda prompt: pytest.fail("llm_fn should not be called with < 2 corpus entries")

    assert thread_builder.find_same_topic_pair(corpus, llm_fn) is None


def test_find_same_topic_pair_returns_pick_on_valid_match():
    corpus = [
        _corpus_entry(0, "Ep A", "argues remote work increases productivity", "/tmp/a"),
        _corpus_entry(1, "Ep B", "argues remote work decreases productivity", "/tmp/b"),
    ]
    llm_fn = lambda prompt: json.dumps({
        "no_match": False, "episode_a_index": 0, "episode_b_index": 1,
        "shared_question": "Does remote work increase or decrease productivity?",
    })

    result = thread_builder.find_same_topic_pair(corpus, llm_fn)

    assert result == {
        "episode_a_index": 0, "episode_b_index": 1,
        "shared_question": "Does remote work increase or decrease productivity?",
    }


def test_find_same_topic_pair_rejects_out_of_range_indices():
    corpus = [
        _corpus_entry(0, "Ep A", "abstract a", "/tmp/a"),
        _corpus_entry(1, "Ep B", "abstract b", "/tmp/b"),
    ]
    llm_fn = lambda prompt: json.dumps({"no_match": False, "episode_a_index": 0, "episode_b_index": 5, "shared_question": "x?"})

    assert thread_builder.find_same_topic_pair(corpus, llm_fn) is None


def test_find_same_topic_pair_rejects_missing_shared_question():
    corpus = [
        _corpus_entry(0, "Ep A", "abstract a", "/tmp/a"),
        _corpus_entry(1, "Ep B", "abstract b", "/tmp/b"),
    ]
    llm_fn = lambda prompt: json.dumps({"no_match": False, "episode_a_index": 0, "episode_b_index": 1, "shared_question": ""})

    assert thread_builder.find_same_topic_pair(corpus, llm_fn) is None


def test_find_same_topic_pair_returns_none_on_malformed_llm_output():
    corpus = [
        _corpus_entry(0, "Ep A", "abstract a", "/tmp/a"),
        _corpus_entry(1, "Ep B", "abstract b", "/tmp/b"),
    ]
    llm_fn = lambda prompt: "not json at all"

    assert thread_builder.find_same_topic_pair(corpus, llm_fn) is None


def _episode(duration, texts_with_times):
    segments = [{"start": s, "end": e, "text": t} for s, e, t in texts_with_times]
    return {"transcript": {"duration": duration, "segments": segments}}


def test_pick_thread_clips_returns_none_when_not_grounded():
    episode_a = _episode(100.0, [(0.0, 10.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 10.0, "world")])
    llm_fn = lambda prompt: json.dumps({"grounded": False, "thesis": "", "bridge": "", "clip_a": {}, "clip_b": {}})

    assert thread_builder.pick_thread_clips(episode_a, episode_b, "shared question?", llm_fn) is None


def test_pick_thread_clips_returns_clips_and_narration_on_valid_response():
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    llm_fn = lambda prompt: json.dumps({
        "grounded": True,
        "thesis": "Two guests, one question.",
        "bridge": "Here is the other side.",
        "clip_a": {"start_time": 5.0, "end_time": 25.0},
        "clip_b": {"start_time": 2.0, "end_time": 20.0},
    })

    result = thread_builder.pick_thread_clips(episode_a, episode_b, "shared question?", llm_fn)

    assert result == {
        "thesis": "Two guests, one question.",
        "bridge": "Here is the other side.",
        "clip_a": {"start_time": 5.0, "end_time": 25.0},
        "clip_b": {"start_time": 2.0, "end_time": 20.0},
    }


def test_pick_thread_clips_rejects_span_shorter_than_8_seconds():
    episode_a = _episode(100.0, [(0.0, 30.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    llm_fn = lambda prompt: json.dumps({
        "grounded": True, "thesis": "t", "bridge": "b",
        "clip_a": {"start_time": 5.0, "end_time": 7.0},
        "clip_b": {"start_time": 2.0, "end_time": 20.0},
    })

    assert thread_builder.pick_thread_clips(episode_a, episode_b, "q?", llm_fn) is None


def test_pick_thread_clips_clamps_end_time_to_episode_duration():
    episode_a = _episode(20.0, [(0.0, 20.0, "hello")])
    episode_b = _episode(100.0, [(0.0, 30.0, "world")])
    llm_fn = lambda prompt: json.dumps({
        "grounded": True, "thesis": "t", "bridge": "b",
        "clip_a": {"start_time": 5.0, "end_time": 50.0},
        "clip_b": {"start_time": 2.0, "end_time": 20.0},
    })

    result = thread_builder.pick_thread_clips(episode_a, episode_b, "q?", llm_fn)

    assert result["clip_a"]["end_time"] == 20.0


def test_build_thread_returns_none_when_no_same_topic_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(
        thread_builder, "build_corpus",
        lambda base_dir=None, llm_fn=None: [
            _corpus_entry(0, "Ep A", "unrelated topic one", "/tmp/a"),
            _corpus_entry(1, "Ep B", "unrelated topic two", "/tmp/b"),
        ],
    )
    llm_fn = lambda prompt: json.dumps({"no_match": True, "episode_a_index": None, "episode_b_index": None, "shared_question": ""})

    assert thread_builder.build_thread(base_dir=str(tmp_path), llm_fn=llm_fn) is None


def test_build_thread_returns_none_when_corpus_has_fewer_than_two_episodes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        thread_builder, "build_corpus",
        lambda base_dir=None, llm_fn=None: [_corpus_entry(0, "Only Ep", "abstract", "/tmp/a")],
    )
    llm_fn = lambda prompt: pytest.fail("llm_fn should not be called for topic gate with < 2 episodes")

    assert thread_builder.build_thread(base_dir=str(tmp_path), llm_fn=llm_fn) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_thread_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shorts_generator.thread_builder'`

- [ ] **Step 3: Create `shorts_generator/thread_builder.py`**

```python
"""Two-stage picker for thread compilation: first a hard same-topic gate
across the whole local corpus (refuses outright if no pair shares a genuine
question -- see docs/superpowers/specs/2026-08-09-thread-compilation-design.md),
then, only for a qualifying pair, exact clip spans + narration text grounded
in the two chosen full transcripts.
"""
import json
from typing import Dict, List, Optional

from .corpus import build_corpus
from .highlights import LLMFn, _parse_json_loose, build_transcript_text, call_muapi_llm

SAME_TOPIC_SYSTEM_PROMPT = """You are a strict fact-checker deciding whether two podcast episodes can be combined into a single short video built around ONE shared question.

You will be given a numbered list of episode abstracts. Your ONLY job is to find, if one exists, exactly one pair of episodes that:
1. Discuss the SAME specific topic (not just an adjacent or loosely related one)
2. Each independently make a claim, argument, or statement that answers or responds to the SAME underlying question -- not just the same subject area

Rules -- read carefully, these are hard requirements, not preferences:
- Sharing a broad subject (e.g. both mention "aliens", both mention "the economy") is NOT enough. Two episodes both mentioning AI is not a match; two episodes both making a claim about whether AI will take creative jobs IS a match.
- If you are not certain both episodes answer the exact same question, you MUST return no_match: true. A weak or "kind of related" pair is a wrong answer, not a partial credit answer.
- Do not invent a connection. Only report a pair if the abstracts themselves make the shared question obvious.
- State the shared question explicitly, in one sentence, phrased as a real question a viewer would recognize both clips are answering.

Episodes:
{abstracts_block}

Respond ONLY with valid JSON (no markdown, no explanation):
{{"no_match": bool, "episode_a_index": int or null, "episode_b_index": int or null, "shared_question": "string or empty"}}"""

THREAD_PICK_SYSTEM_PROMPT = """You are editing a short video that puts two podcast guests' answers to the same question side by side.

Shared question both episodes are answering: {shared_question}

You will be given the full transcript of TWO episodes, each labeled with absolute timestamps. Find, in EACH transcript, one span where the speaker directly answers or responds to the shared question above.

Rules:
- start_time must land on the exact line where the speaker begins answering the shared question -- never on preamble, a host's question, or unrelated chat before it.
- end_time must land where that specific answer finishes -- never mid-sentence, never trailing into the next unrelated topic.
- Duration per clip: 15-40 seconds. If the answer runs longer, pick the single most complete self-contained span within it.
- Write a "thesis" -- one sentence a narrator would say BEFORE either clip plays, naming the shared question and hinting that the two answers differ or agree in an interesting way. This must state the actual tension or agreement, not a vague tease.
- Write a "bridge" -- one sentence a narrator would say BETWEEN the two clips, explicitly connecting what episode A's speaker just said to what episode B's speaker is about to say.
- If you cannot find a genuine on-topic answer in BOTH transcripts, set "grounded" to false and leave the clip fields empty -- do not force a loose match.

Episode A transcript:
{transcript_a}

Episode B transcript:
{transcript_b}

Respond ONLY with valid JSON (no markdown, no explanation):
{{"grounded": bool, "thesis": "string", "bridge": "string", "clip_a": {{"start_time": float, "end_time": float}}, "clip_b": {{"start_time": float, "end_time": float}}}}"""

MIN_CLIP_SECONDS = 8.0
MAX_CLIP_SECONDS = 60.0


def _coerce_float_or_none(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sanitize_topic_pick(raw: object, corpus_len: int) -> Optional[Dict]:
    if not isinstance(raw, dict) or raw.get("no_match"):
        return None
    a = raw.get("episode_a_index")
    b = raw.get("episode_b_index")
    if not isinstance(a, int) or not isinstance(b, int):
        return None
    if a == b or not (0 <= a < corpus_len) or not (0 <= b < corpus_len):
        return None
    question = str(raw.get("shared_question") or "").strip()
    if not question:
        return None
    return {"episode_a_index": a, "episode_b_index": b, "shared_question": question}


def _sanitize_clip_span(raw: object, duration: float) -> Optional[Dict]:
    if not isinstance(raw, dict):
        return None
    start = _coerce_float_or_none(raw.get("start_time"))
    end = _coerce_float_or_none(raw.get("end_time"))
    if start is None or end is None or start < 0 or end <= start:
        return None
    if duration > 0:
        end = min(end, duration)
        if end <= start:
            return None
    span = end - start
    if span < MIN_CLIP_SECONDS or span > MAX_CLIP_SECONDS:
        return None
    return {"start_time": start, "end_time": end}


def _sanitize_clip_pick(raw: object, duration_a: float, duration_b: float) -> Optional[Dict]:
    if not isinstance(raw, dict) or not raw.get("grounded"):
        return None
    thesis = str(raw.get("thesis") or "").strip()
    bridge = str(raw.get("bridge") or "").strip()
    if not thesis or not bridge:
        return None
    clip_a = _sanitize_clip_span(raw.get("clip_a"), duration_a)
    clip_b = _sanitize_clip_span(raw.get("clip_b"), duration_b)
    if clip_a is None or clip_b is None:
        return None
    return {"thesis": thesis, "bridge": bridge, "clip_a": clip_a, "clip_b": clip_b}


def find_same_topic_pair(corpus: List[Dict], llm_fn: LLMFn) -> Optional[Dict]:
    """Stage A. Returns None -- the expected, correct result whenever no
    pair shares a genuine question -- or {"episode_a_index",
    "episode_b_index", "shared_question"} for a qualifying pair."""
    if len(corpus) < 2:
        return None
    abstracts_block = "\n".join(
        f"{i}. {entry['title']}: {entry['abstract']}" for i, entry in enumerate(corpus)
    )
    prompt = SAME_TOPIC_SYSTEM_PROMPT.format(abstracts_block=abstracts_block)
    try:
        parsed = _parse_json_loose(llm_fn(prompt))
    except Exception:
        return None
    return _sanitize_topic_pick(parsed, corpus_len=len(corpus))


def pick_thread_clips(episode_a: Dict, episode_b: Dict, shared_question: str, llm_fn: LLMFn) -> Optional[Dict]:
    """Stage B. episode_a/episode_b must each have a "transcript" key (full
    {duration, segments} shape). Returns None if the model can't ground a
    clip answering shared_question in BOTH transcripts."""
    prompt = THREAD_PICK_SYSTEM_PROMPT.format(
        shared_question=shared_question,
        transcript_a=build_transcript_text(episode_a["transcript"]),
        transcript_b=build_transcript_text(episode_b["transcript"]),
    )
    try:
        parsed = _parse_json_loose(llm_fn(prompt))
    except Exception:
        return None
    return _sanitize_clip_pick(
        parsed,
        duration_a=episode_a["transcript"].get("duration", 0.0),
        duration_b=episode_b["transcript"].get("duration", 0.0),
    )


def build_thread(base_dir: Optional[str] = None, llm_fn: Optional[LLMFn] = None) -> Optional[Dict]:
    """Full stage A + B pipeline. Returns None whenever the corpus doesn't
    support a genuine same-topic pairing today -- this is success (nothing
    to build), not a failure to retry or work around."""
    llm_fn = llm_fn or call_muapi_llm
    corpus = build_corpus(base_dir=base_dir, llm_fn=llm_fn)

    pick = find_same_topic_pair(corpus, llm_fn)
    if pick is None:
        print("[thread_builder] no same-topic pair found in corpus -- refusing to build a thread", flush=True)
        return None

    entry_a = corpus[pick["episode_a_index"]]
    entry_b = corpus[pick["episode_b_index"]]
    with open(f"{entry_a['run_dir']}/full_source.json", "r", encoding="utf-8") as f:
        transcript_a = json.load(f)
    with open(f"{entry_b['run_dir']}/full_source.json", "r", encoding="utf-8") as f:
        transcript_b = json.load(f)

    clips = pick_thread_clips(
        {**entry_a, "transcript": transcript_a},
        {**entry_b, "transcript": transcript_b},
        pick["shared_question"],
        llm_fn,
    )
    if clips is None:
        print("[thread_builder] no groundable clip pair for the shared question -- refusing to build a thread", flush=True)
        return None

    return {
        "shared_question": pick["shared_question"],
        "thesis": clips["thesis"],
        "bridge": clips["bridge"],
        "episode_a": {
            "run_dir": entry_a["run_dir"], "title": entry_a["title"], "source_url": entry_a["source_url"],
            **clips["clip_a"],
        },
        "episode_b": {
            "run_dir": entry_b["run_dir"], "title": entry_b["title"], "source_url": entry_b["source_url"],
            **clips["clip_b"],
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_thread_builder.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/thread_builder.py tests/test_thread_builder.py
git commit -m "feat: add thread_builder with a hard same-topic refusal gate

Stage A screens the whole corpus and returns None outright whenever no
pair of episodes shares a genuine question -- not a loosely-related
pair, per the explicit no-negotiation rule on topic matching. Stage B
only runs for a qualifying pair and grounds exact clip spans + thesis/
bridge narration text in the two full transcripts."
```

---

### Task 5: ElevenLabs config + narration module (`local/narration.py`)

**Files:**
- Modify: `shorts_generator/config.py`
- Create: `shorts_generator/local/narration.py`
- Test: `tests/test_narration.py`

- [ ] **Step 1: Add ElevenLabs settings to config.py**

In `shorts_generator/config.py`, add after the `SHORT_FILENAME_STYLE` line:

```python
# Thread-compilation narration (--clip-type thread, local mode only).
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
# "George - Warm, Captivating Storyteller" -- validated by hand for narrator
# tone; override via env if a different channel voice is wanted later.
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb").strip()
```

Add after `require_openrouter_key`:

```python
def require_elevenlabs_key() -> str:
    if not ELEVENLABS_API_KEY:
        raise RuntimeError(
            "ELEVENLABS_API_KEY is not set. Thread narration needs an ElevenLabs "
            "key. Add it to your .env file or export it as an env var."
        )
    return ELEVENLABS_API_KEY
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_narration.py`:

```python
import os
import subprocess

import pytest

from shorts_generator import config
from shorts_generator.local import narration as narration_module


def test_synthesize_narration_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ELEVENLABS_API_KEY", "")
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        narration_module.synthesize_narration("hello", str(tmp_path / "out.mp3"))


def test_synthesize_narration_writes_audio_from_fake_client(monkeypatch, tmp_path):
    monkeypatch.setattr(narration_module, "ELEVENLABS_API_KEY", "fake-key")

    class _FakeTTS:
        def convert(self, **kwargs):
            assert kwargs["text"] == "hello there"
            return iter([b"chunk1", b"chunk2"])

    class _FakeClient:
        def __init__(self, api_key):
            self.text_to_speech = _FakeTTS()

    monkeypatch.setattr(narration_module, "_get_elevenlabs_client_class", lambda: _FakeClient)

    out_path = str(tmp_path / "out.mp3")
    narration_module.synthesize_narration("hello there", out_path)

    with open(out_path, "rb") as f:
        assert f.read() == b"chunk1chunk2"


def test_synthesize_narration_wraps_client_errors():
    class _FakeTTS:
        def convert(self, **kwargs):
            raise RuntimeError("api down")

    class _FakeClient:
        def __init__(self, api_key):
            self.text_to_speech = _FakeTTS()

    def run(monkeypatch, tmp_path):
        monkeypatch.setattr(narration_module, "ELEVENLABS_API_KEY", "fake-key")
        monkeypatch.setattr(narration_module, "_get_elevenlabs_client_class", lambda: _FakeClient)
        with pytest.raises(narration_module.NarrationError, match="api down"):
            narration_module.synthesize_narration("hi", str(tmp_path / "out.mp3"))

    return run


def test_synthesize_narration_wraps_client_errors_actual(monkeypatch, tmp_path):
    class _FakeTTS:
        def convert(self, **kwargs):
            raise RuntimeError("api down")

    class _FakeClient:
        def __init__(self, api_key):
            self.text_to_speech = _FakeTTS()

    monkeypatch.setattr(narration_module, "ELEVENLABS_API_KEY", "fake-key")
    monkeypatch.setattr(narration_module, "_get_elevenlabs_client_class", lambda: _FakeClient)

    with pytest.raises(narration_module.NarrationError, match="api down"):
        narration_module.synthesize_narration("hi", str(tmp_path / "out.mp3"))


def test_wrap_text_splits_long_sentence_into_multiple_lines():
    wrapped = narration_module._wrap_text("The Vice President wants to go find proof himself.", max_chars_per_line=28)
    lines = wrapped.split("\n")
    assert len(lines) > 1
    assert all(len(line) <= 28 or " " not in line for line in lines)


@pytest.fixture(scope="module")
def synthetic_audio(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("narration_audio")
    path = str(tmp_dir / "line.mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", path],
        check=True,
    )
    return path


def test_render_narration_card_produces_vertical_video_matching_audio_duration(synthetic_audio, tmp_path):
    out_path = str(tmp_path / "card.mp4")
    narration_module.render_narration_card(synthetic_audio, "Test narration line here.", out_path)

    assert os.path.exists(out_path)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=width,height", "-of", "csv=p=0", out_path],
        capture_output=True, text=True, check=True,
    )
    assert "1080" in probe.stdout
    assert "1920" in probe.stdout
```

Remove the two throwaway helper functions `test_synthesize_narration_wraps_client_errors` (the closure-returning one) before saving — that was a scratch draft; keep only `test_synthesize_narration_wraps_client_errors_actual` and rename it to `test_synthesize_narration_wraps_client_errors`. The final file should have exactly one test for that behavior.

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_narration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shorts_generator.local.narration'`

- [ ] **Step 4: Create `shorts_generator/local/narration.py`**

```python
"""ElevenLabs narrator voice for thread bridges -- see thread_builder.py for
where "thesis" and "bridge" text comes from. Renders each line as audio via
ElevenLabs, then composites it onto a plain title card matching the
channel's existing hook-card typography (Anton font, white text on a
translucent black box -- see hook_card.py) so it drops into the same
ffmpeg-concat assembly as the live-footage clips (see thread_assembler.py).
"""
import os
import subprocess
from typing import Type

from ..config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID, require_elevenlabs_key
from ..hook_card import FONT_PATH

DEFAULT_VOICE_ID = ELEVENLABS_VOICE_ID
CARD_WIDTH = 1080
CARD_HEIGHT = 1920
CARD_FPS = "30000/1001"
CARD_BG_COLOR = "0x0d0d0d"


class NarrationError(RuntimeError):
    """Raised when ElevenLabs synthesis or card rendering fails."""


def _get_elevenlabs_client_class() -> Type:
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise NarrationError(
            "elevenlabs is required for thread narration. Install it with:\n"
            "    pip install elevenlabs"
        ) from e
    return ElevenLabs


def synthesize_narration(text: str, out_path: str, voice_id: str = DEFAULT_VOICE_ID) -> str:
    """Call ElevenLabs TTS and write the audio to out_path (mp3)."""
    require_elevenlabs_key()
    client_cls = _get_elevenlabs_client_class()
    client = client_cls(api_key=ELEVENLABS_API_KEY)
    try:
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
        )
        with open(out_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
    except Exception as e:
        raise NarrationError(f"ElevenLabs synthesis failed: {e}") from e
    return out_path


def _wrap_text(text: str, max_chars_per_line: int = 28) -> str:
    words = text.split()
    lines = []
    current: list = []
    current_len = 0
    for word in words:
        added_len = len(word) + (1 if current else 0)
        if current and current_len + added_len > max_chars_per_line:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += added_len
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def render_narration_card(audio_path: str, text: str, out_path: str) -> str:
    """Composite `text` (Anton font, boxed, auto-wrapped) over a plain dark
    card, muxed with the narration audio at audio_path, sized/timed to match
    the thread's live-footage clips (see thread_assembler.py's TARGET_*)."""
    text_file = out_path + ".txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(_wrap_text(text))

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path],
            capture_output=True, text=True, check=True,
        )
        duration = float(probe.stdout.strip())

        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", f"color=c={CARD_BG_COLOR}:s={CARD_WIDTH}x{CARD_HEIGHT}:r={CARD_FPS}:d={duration}",
                "-i", audio_path,
                "-vf",
                f"drawtext=fontfile='{FONT_PATH}':textfile='{text_file}':fontsize=64:fontcolor=white:"
                "box=1:boxcolor=black@0.55:boxborderw=16:x=(w-text_w)/2:y=(h-text_h)/2:line_spacing=20",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", CARD_FPS,
                "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "192k",
                "-shortest", out_path,
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise NarrationError(f"narration card render failed: {e.stderr}") from e
    finally:
        if os.path.exists(text_file):
            os.remove(text_file)
    return out_path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_narration.py -v`
Expected: PASS (5 tests). Requires `ffmpeg`/`ffprobe` on PATH (already a hard dependency of this repo) and does not require a real `ELEVENLABS_API_KEY` (the client is faked).

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/config.py shorts_generator/local/narration.py tests/test_narration.py
git commit -m "feat: add ElevenLabs narration module for thread bridges

Renders thesis/bridge text to audio via ElevenLabs, then composites it
onto a title card matching the channel's existing hook-card typography
so it drops into the same concat assembly as live-footage clips."
```

---

### Task 6: Thread assembler (`local/thread_assembler.py`)

**Files:**
- Create: `shorts_generator/local/thread_assembler.py`
- Test: `tests/test_thread_assembler.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_thread_assembler.py`:

```python
import subprocess

import pytest

from shorts_generator.local import thread_assembler


@pytest.fixture(scope="module")
def synthetic_clip_a(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("assembler_a")
    path = str(tmp_dir / "a.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=24:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", "-shortest", path,
        ],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def synthetic_clip_b(tmp_path_factory):
    # Deliberately different resolution/fps/audio rate from clip_a, matching
    # the real cross-episode case this module exists to normalize.
    tmp_dir = tmp_path_factory.mktemp("assembler_b")
    path = str(tmp_dir / "b.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
            "-ar", "48000",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", "-shortest", path,
        ],
        check=True,
    )
    return path


def test_assemble_thread_requires_at_least_two_segments(tmp_path):
    with pytest.raises(ValueError, match="at least 2"):
        thread_assembler.assemble_thread(["/tmp/only_one.mp4"], str(tmp_path / "out.mp4"))


def test_assemble_thread_normalizes_mismatched_sources_into_one_output(synthetic_clip_a, synthetic_clip_b, tmp_path):
    out_path = str(tmp_path / "assembled.mp4")

    thread_assembler.assemble_thread([synthetic_clip_a, synthetic_clip_b], out_path)

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=width,height,r_frame_rate",
            "-of", "default=noprint_wrappers=1",
            out_path,
        ],
        capture_output=True, text=True, check=True,
    )
    output = probe.stdout
    assert f"width={thread_assembler.TARGET_WIDTH}" in output
    assert f"height={thread_assembler.TARGET_HEIGHT}" in output
    assert "r_frame_rate=30000/1001" in output


def test_assemble_thread_raises_thread_assembly_error_on_ffmpeg_failure(tmp_path):
    with pytest.raises(thread_assembler.ThreadAssemblyError):
        thread_assembler.assemble_thread(
            ["/nonexistent/a.mp4", "/nonexistent/b.mp4"], str(tmp_path / "out.mp4")
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_thread_assembler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shorts_generator.local.thread_assembler'`

- [ ] **Step 3: Create `shorts_generator/local/thread_assembler.py`**

```python
"""Re-encode ffmpeg concat for a thread's ordered segments (narration cards
+ live-footage clips). A plain stream-copy concat (as jump_cuts.py uses for
same-source excision) assumes matching codec/resolution/fps/audio across
every input; a thread's clips come from different source episodes with
different native fps/audio, so every segment is normalized to one target
spec before concatenation, matching the spec crop_clip_local's vertical
output and narration cards already use.
"""
import subprocess
from typing import List

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_FPS = "30000/1001"
TARGET_AUDIO_RATE = 44100


class ThreadAssemblyError(RuntimeError):
    """Raised when the final ffmpeg concat fails."""


def assemble_thread(segment_paths: List[str], out_path: str) -> str:
    """Concatenate segment_paths in order into one vertical video at
    out_path, re-encoding every stream to a common spec first."""
    if len(segment_paths) < 2:
        raise ValueError("assemble_thread needs at least 2 segments")

    inputs = []
    for p in segment_paths:
        inputs += ["-i", p]

    filter_parts = []
    concat_refs = []
    for i in range(len(segment_paths)):
        filter_parts.append(
            f"[{i}:v]fps={TARGET_FPS},scale={TARGET_WIDTH}:{TARGET_HEIGHT},setsar=1,format=yuv420p[v{i}]"
        )
        filter_parts.append(f"[{i}:a]aresample={TARGET_AUDIO_RATE},aformat=channel_layouts=stereo[a{i}]")
        concat_refs += [f"[v{i}]", f"[a{i}]"]

    filter_complex = (
        ";".join(filter_parts)
        + ";" + "".join(concat_refs)
        + f"concat=n={len(segment_paths)}:v=1:a=1[outv][outa]"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise ThreadAssemblyError(f"ffmpeg concat failed: {e.stderr}") from e
    return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_thread_assembler.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/local/thread_assembler.py tests/test_thread_assembler.py
git commit -m "feat: add re-encode concat assembler for cross-episode threads

Stream-copy concat (as jump_cuts.py uses) assumes matching source
specs; a thread's segments come from different episodes with different
native fps/audio, so every input is normalized to one target spec
before concatenation."
```

---

### Task 7: Source acquisition with duration-mismatch guard (`local/thread_source.py`)

**Files:**
- Create: `shorts_generator/local/thread_source.py`
- Test: `tests/test_thread_source.py`

This directly encodes the incident from the design spec: never trust a re-downloaded source's timestamps without checking its duration against the cached transcript first.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_thread_source.py`:

```python
import json
import os
import subprocess

import pytest

from shorts_generator.local import thread_source as thread_source_module
from shorts_generator.local.thread_source import SourceMismatchError, acquire_clip


def test_find_word_start_returns_first_word_at_or_after_min_time():
    segments = [
        {"start": 0.0, "end": 4.0, "words": [
            {"start": 0.0, "end": 0.5, "word": "hello"},
            {"start": 1.5, "end": 2.0, "word": "world"},
        ]},
        {"start": 4.0, "end": 6.0, "words": [
            {"start": 4.2, "end": 4.5, "word": "next"},
        ]},
    ]
    assert thread_source_module._find_word_start(segments, 1.0) == 1.5
    assert thread_source_module._find_word_start(segments, 3.0) == 4.2


def test_find_word_start_falls_back_to_min_time_when_no_word_found():
    segments = [{"start": 0.0, "end": 1.0, "words": [{"start": 0.0, "end": 0.5, "word": "hi"}]}]
    assert thread_source_module._find_word_start(segments, 10.0) == 10.0


def test_acquire_clip_cuts_directly_when_full_source_present(tmp_path, monkeypatch):
    run_dir = tmp_path / "episode"
    run_dir.mkdir()
    (run_dir / "full_source.mp4").write_bytes(b"fake video bytes")
    (run_dir / "full_source.json").write_text(json.dumps({
        "duration": 100.0,
        "segments": [{"start": 0.0, "end": 10.0, "text": "hello world", "words": []}],
    }))

    calls = {}
    monkeypatch.setattr(
        thread_source_module, "crop_clip_local",
        lambda *a, **k: calls.setdefault("crop_args", (a, k)),
    )
    monkeypatch.setattr(
        thread_source_module, "burn_captions",
        lambda src, segs, start, end, out, **k: open(out, "wb").write(b"captioned"),
    )

    def _fail(*a, **k):
        pytest.fail("network re-acquisition should not run when full_source.mp4 is present")
    monkeypatch.setattr(thread_source_module, "_probe_source_duration", _fail)
    monkeypatch.setattr(thread_source_module, "_download_padded_section", _fail)

    out_path = str(tmp_path / "clip.mp4")
    # crop_clip_local is faked to a no-op, so pre-create the file
    # burn_captions replaces -- burn_captions itself writes out_path fresh.
    result = acquire_clip(
        str(run_dir), "https://example.com/video", cached_duration=100.0,
        start_time=1.0, end_time=8.0, out_path=out_path,
    )

    assert result == {"clip_path": out_path}
    assert calls["crop_args"][0][:3] == (str(run_dir / "full_source.mp4"), 1.0, 8.0)


def test_acquire_clip_raises_on_duration_mismatch_before_downloading(tmp_path, monkeypatch):
    run_dir = tmp_path / "episode"
    run_dir.mkdir()
    (run_dir / "full_source.json").write_text(json.dumps({"duration": 5778.0, "segments": []}))
    # full_source.mp4 deliberately absent -- forces the re-acquire fallback path

    monkeypatch.setattr(thread_source_module, "_probe_source_duration", lambda url: 6530.0)

    def _fail_if_called(*a, **k):
        pytest.fail("_download_padded_section should not be called on a duration mismatch")
    monkeypatch.setattr(thread_source_module, "_download_padded_section", _fail_if_called)

    with pytest.raises(SourceMismatchError):
        acquire_clip(
            str(run_dir), "https://example.com/video", cached_duration=5778.0,
            start_time=100.0, end_time=120.0, out_path=str(tmp_path / "out.mp4"),
        )


def test_acquire_clip_proceeds_when_duration_matches_within_tolerance(tmp_path, monkeypatch):
    run_dir = tmp_path / "episode"
    run_dir.mkdir()
    (run_dir / "full_source.json").write_text(json.dumps({"duration": 5778.0, "segments": []}))

    monkeypatch.setattr(thread_source_module, "_probe_source_duration", lambda url: 5778.9)

    padded_path_holder = {}
    def _fake_download(source_url, start_time, end_time, out_path):
        padded_path_holder["out_path"] = out_path
        with open(out_path, "wb") as f:
            f.write(b"fake padded video")
    monkeypatch.setattr(thread_source_module, "_download_padded_section", _fake_download)
    monkeypatch.setattr(
        thread_source_module, "transcribe_local",
        lambda path, model_size=None: {"duration": 30.0, "segments": [
            {"start": 0.0, "end": 5.0, "text": "hi", "words": [{"start": 1.5, "end": 2.0, "word": "hi"}]}
        ]},
    )
    monkeypatch.setattr(thread_source_module, "crop_clip_local", lambda *a, **k: None)
    monkeypatch.setattr(
        thread_source_module, "burn_captions",
        lambda src, segs, start, end, out, **k: open(out, "wb").write(b"captioned"),
    )

    result = acquire_clip(
        str(run_dir), "https://example.com/video", cached_duration=5778.0,
        start_time=956.2, end_time=977.3, out_path=str(tmp_path / "out.mp4"),
    )

    assert result == {"clip_path": str(tmp_path / "out.mp4")}
    assert padded_path_holder["out_path"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_thread_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'shorts_generator.local.thread_source'`

- [ ] **Step 3: Create `shorts_generator/local/thread_source.py`**

```python
"""Acquire the exact audio/video span thread_builder picked for one episode
of a thread, whether or not that episode's full_source.mp4 is still on disk.

The common case: full_source.mp4 is still cached -- cut directly from it via
the same crop_clip_local/burn_captions path Shorts already use, using the
cached transcript, no network call needed.

The fallback case: full_source.mp4 was deleted (typical once a channel has
100+ episodes and disk space matters) -- re-download just the needed span
via yt-dlp, but ONLY after verifying the live video's duration matches the
cached transcript's duration. This check exists because of a real incident
(see docs/superpowers/specs/2026-08-09-thread-compilation-design.md): a
mismatched source URL silently produced a downloaded span whose audio had
nothing to do with the cached transcript's timestamps, and captions burned
from that stale transcript looked plausible but described different audio
entirely. A duration mismatch is fatal, never a warning to route around.
"""
import os
import subprocess
import tempfile
from typing import Dict, List

from ..captions import burn_captions
from .clipper import crop_clip_local
from .transcriber import transcribe_local

PAD_SECONDS = 3.0
DURATION_MISMATCH_TOLERANCE_SECONDS = 2.0


class SourceMismatchError(RuntimeError):
    """Raised when a re-acquired source's live duration doesn't match the
    cached transcript's duration -- see module docstring."""


def _probe_source_duration(source_url: str) -> float:
    result = subprocess.run(
        ["yt-dlp", "--skip-download", "--print", "duration", source_url],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip().splitlines()[-1])


def _download_padded_section(source_url: str, start_time: float, end_time: float, out_path: str) -> None:
    padded_start = max(0.0, start_time - PAD_SECONDS)
    padded_end = end_time + PAD_SECONDS
    webm_path = out_path + ".webm"
    subprocess.run(
        [
            "yt-dlp",
            "--download-sections", f"*{padded_start}-{padded_end}",
            "-f", "bv*[height<=720]+ba/b[height<=720]",
            "--force-keyframes-at-cuts",
            "-o", webm_path,
            source_url,
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", webm_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            out_path,
        ],
        check=True,
    )
    os.remove(webm_path)


def _find_word_start(segments: List[Dict], min_time: float) -> float:
    """First word timestamp at or after min_time -- lands a clip's start
    exactly on a spoken word instead of mid-word or on dead air. Falls back
    to min_time itself if no word starts at or after it in these segments."""
    for seg in segments:
        for w in seg.get("words", []):
            if float(w["start"]) >= min_time:
                return float(w["start"])
    return min_time


def acquire_clip(
    run_dir: str,
    source_url: str,
    cached_duration: float,
    start_time: float,
    end_time: float,
    out_path: str,
    aspect_ratio: str = "9:16",
) -> Dict:
    """Cut, reframe, and caption one episode's clip for a thread.

    Returns {"clip_path": out_path}. Raises SourceMismatchError if a
    re-download's live source duration doesn't match cached_duration.
    """
    full_source = os.path.join(run_dir, "full_source.mp4")
    full_transcript_path = os.path.join(run_dir, "full_source.json")

    if os.path.exists(full_source):
        import json
        with open(full_transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        crop_clip_local(
            full_source, start_time, end_time, aspect_ratio, out_path,
            framing="locked", cut_segments=[{"start_time": start_time, "end_time": end_time}],
        )
        captioned_path = out_path + ".captioned.mp4"
        burn_captions(
            out_path, transcript["segments"], start_time, end_time, captioned_path,
            fade_seconds=0.3, word_highlight=True,
        )
        os.replace(captioned_path, out_path)
        return {"clip_path": out_path}

    full_duration = _probe_source_duration(source_url)
    if abs(full_duration - cached_duration) > DURATION_MISMATCH_TOLERANCE_SECONDS:
        raise SourceMismatchError(
            f"live video duration ({full_duration:.1f}s) does not match cached "
            f"transcript duration ({cached_duration:.1f}s) for {run_dir} -- "
            "refusing to caption a possibly-wrong source. Confirm source_url.txt "
            "points at the same upload that was originally transcribed."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        padded_path = os.path.join(tmp_dir, "padded.mp4")
        _download_padded_section(source_url, start_time, end_time, padded_path)

        padded_start = max(0.0, start_time - PAD_SECONDS)
        fresh_transcript = transcribe_local(padded_path, model_size="small")
        relative_start = _find_word_start(fresh_transcript["segments"], start_time - padded_start)
        relative_end = min(end_time - padded_start, fresh_transcript["duration"])

        crop_clip_local(
            padded_path, relative_start, relative_end, aspect_ratio, out_path,
            framing="locked", cut_segments=[{"start_time": relative_start, "end_time": relative_end}],
        )
        captioned_path = out_path + ".captioned.mp4"
        burn_captions(
            out_path, fresh_transcript["segments"], relative_start, relative_end, captioned_path,
            fade_seconds=0.3, word_highlight=True,
        )
        os.replace(captioned_path, out_path)
        return {"clip_path": out_path}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_thread_source.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/local/thread_source.py tests/test_thread_source.py
git commit -m "feat: add duration-guarded source acquisition for thread clips

Cuts directly from a still-local full_source.mp4 when present (no
network needed); otherwise re-downloads just the needed span, but only
after verifying the live source's duration matches the cached
transcript's -- a hard guard against the exact mismatch incident that
produced captions describing audio that wasn't actually playing."
```

---

### Task 8: Pipeline orchestration — `generate_threads()`

**Files:**
- Modify: `shorts_generator/pipeline.py`
- Modify: `shorts_generator/run_output.py`
- Modify: `shorts_generator/__init__.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Add `resolve_thread_output_dir` to `run_output.py`**

A thread draws footage from two existing episode runs, so its output can't live inside either one's `RunPaths` tree. Add to `shorts_generator/run_output.py`, after `resolve_output_dir`:

```python
def resolve_thread_output_dir(thesis: str, base_dir: Optional[str] = None) -> str:
    """A thread's output lives outside any single episode's RunPaths tree --
    it draws footage from two existing episode runs, so it gets its own
    output/_Threads/<slug>/ folder keyed by the thread's own thesis text."""
    base_dir = base_dir or LOCAL_OUTPUT_DIR
    slug = sanitize_title(thesis)
    root = os.path.join(base_dir, "_Threads", slug)
    os.makedirs(root, exist_ok=True)
    return root
```

- [ ] **Step 2: Write the failing test for `resolve_thread_output_dir`**

Add to `tests/test_run_output.py`:

```python
def test_resolve_thread_output_dir_slugifies_thesis(tmp_path):
    result = run_output.resolve_thread_output_dir(
        "Two founders, two answers to the same question.", base_dir=str(tmp_path)
    )
    assert result == str(tmp_path / "_Threads" / "Two_founders_two_answers_to_the_same_question")
    assert os.path.isdir(result)
```

- [ ] **Step 3: Run test to verify it fails, then passes**

Run: `python -m pytest tests/test_run_output.py -k thread_output_dir -v`
Expected: first FAIL with `AttributeError`, then PASS after Step 1's code is in place (Step 1 and Step 3 together form the standard red/green cycle — Step 1's code already exists by the time you run this, so this should show PASS immediately; if it doesn't, fix Step 1 before continuing).

- [ ] **Step 4: Write the failing test for `generate_threads`**

Add to `tests/test_pipeline.py`. Match this file's existing monkeypatch style (imports at top of file already include `pipeline_module`; add these too):

```python
def test_generate_threads_returns_none_when_build_thread_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "build_thread", lambda base_dir=None, llm_fn=None: None)

    result = pipeline_module.generate_threads(base_dir=str(tmp_path))

    assert result is None


def test_generate_threads_assembles_and_writes_result(tmp_path, monkeypatch):
    episode_a_dir = tmp_path / "Episode_A"
    episode_b_dir = tmp_path / "Episode_B"
    episode_a_dir.mkdir()
    episode_b_dir.mkdir()
    (episode_a_dir / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))
    (episode_b_dir / "full_source.json").write_text(json.dumps({"duration": 200.0, "segments": []}))

    fake_thread = {
        "shared_question": "Does X cause Y?",
        "thesis": "Two guests disagree about X causing Y.",
        "bridge": "Here's the other side.",
        "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 10.0, "end_time": 30.0},
        "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 5.0, "end_time": 25.0},
    }
    monkeypatch.setattr(pipeline_module, "build_thread", lambda base_dir=None, llm_fn=None: fake_thread)
    monkeypatch.setattr(pipeline_module, "acquire_clip", lambda run_dir, source_url, cached_duration, start_time, end_time, out_path: open(out_path, "wb").write(b"clip") or {"clip_path": out_path})
    monkeypatch.setattr(pipeline_module, "synthesize_narration", lambda text, out_path, **k: open(out_path, "wb").write(b"audio") or out_path)
    monkeypatch.setattr(pipeline_module, "render_narration_card", lambda audio_path, text, out_path: open(out_path, "wb").write(b"card") or out_path)
    assemble_calls = []
    monkeypatch.setattr(pipeline_module, "assemble_thread", lambda segment_paths, out_path: (assemble_calls.append(segment_paths), open(out_path, "wb").write(b"final"))[1] or out_path)

    result = pipeline_module.generate_threads(base_dir=str(tmp_path))

    assert result is not None
    assert result["shared_question"] == "Does X cause Y?"
    assert result["clip_url"].endswith("thread.mp4")
    assert os.path.isfile(os.path.join(result["output_dir"], "thread_result.json"))
    # intro card, clip A, bridge card, clip B, in that order
    assert len(assemble_calls[0]) == 4
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -k generate_threads -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.pipeline' has no attribute 'generate_threads'`

- [ ] **Step 6: Implement `generate_threads` in `pipeline.py`**

Add these imports near the top of `shorts_generator/pipeline.py`, alongside the existing ones:

```python
from .local.narration import render_narration_card, synthesize_narration
from .local.thread_assembler import assemble_thread
from .local.thread_source import acquire_clip
from .thread_builder import build_thread
```

`call_local_llm` is deliberately NOT imported at module level here — match the existing `_run_local`/`_run_local_chapters` convention of importing it inside the function body, so pure API-mode runs never import `local/llm.py` (and its `openai` dependency) at all.

Add the function at the end of `pipeline.py`, after `generate_chapters`:

```python
def generate_threads(base_dir: Optional[str] = None) -> Optional[Dict]:
    """Build one multi-episode thread from the local corpus (see corpus.py,
    thread_builder.py). Local-mode only, like generate_chapters -- there is
    no MuAPI equivalent of this feature. Returns None if no same-topic pair
    currently exists in the corpus -- this is the expected, correct result
    when the corpus is too thin or too topically scattered, not a failure
    to work around.
    """
    from .local.llm import call_local_llm

    thread = build_thread(base_dir=base_dir, llm_fn=call_local_llm)
    if thread is None:
        return None

    out_dir = resolve_thread_output_dir(thread["thesis"], base_dir=base_dir)

    episode_a = thread["episode_a"]
    episode_b = thread["episode_b"]

    with open(os.path.join(episode_a["run_dir"], "full_source.json"), "r", encoding="utf-8") as f:
        duration_a = json.load(f)["duration"]
    with open(os.path.join(episode_b["run_dir"], "full_source.json"), "r", encoding="utf-8") as f:
        duration_b = json.load(f)["duration"]

    clip_a_path = os.path.join(out_dir, "clip_a.mp4")
    clip_b_path = os.path.join(out_dir, "clip_b.mp4")
    acquire_clip(
        episode_a["run_dir"], episode_a["source_url"], cached_duration=duration_a,
        start_time=episode_a["start_time"], end_time=episode_a["end_time"], out_path=clip_a_path,
    )
    acquire_clip(
        episode_b["run_dir"], episode_b["source_url"], cached_duration=duration_b,
        start_time=episode_b["start_time"], end_time=episode_b["end_time"], out_path=clip_b_path,
    )

    intro_audio = os.path.join(out_dir, "thesis.mp3")
    bridge_audio = os.path.join(out_dir, "bridge.mp3")
    synthesize_narration(thread["thesis"], intro_audio)
    synthesize_narration(thread["bridge"], bridge_audio)

    intro_card = os.path.join(out_dir, "intro_card.mp4")
    bridge_card = os.path.join(out_dir, "bridge_card.mp4")
    render_narration_card(intro_audio, thread["thesis"], intro_card)
    render_narration_card(bridge_audio, thread["bridge"], bridge_card)

    final_path = os.path.join(out_dir, "thread.mp4")
    assemble_thread([intro_card, clip_a_path, bridge_card, clip_b_path], final_path)

    result = {**thread, "output_dir": out_dir, "clip_url": final_path}
    with open(os.path.join(out_dir, "thread_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
```

Update the `run_output` import line at the top of `pipeline.py` to include `resolve_thread_output_dir`:

```python
from .run_output import RunPaths, capture_progress_log, resolve_output_dir, resolve_thread_output_dir, write_chapter_descriptions, write_descriptions, write_source_url
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -k "generate_threads or thread_output_dir" -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Run the full pipeline test file to confirm nothing else broke**

Run: `python -m pytest tests/test_pipeline.py tests/test_run_output.py -v`
Expected: PASS (all tests)

- [ ] **Step 9: Export `generate_threads` from the package**

Modify `shorts_generator/__init__.py`:

```python
from .pipeline import generate_chapters, generate_shorts, generate_threads

__all__ = ["generate_chapters", "generate_shorts", "generate_threads"]
```

- [ ] **Step 10: Commit**

```bash
git add shorts_generator/pipeline.py shorts_generator/run_output.py shorts_generator/__init__.py tests/test_pipeline.py tests/test_run_output.py
git commit -m "feat: wire generate_threads() end-to-end

Corpus -> same-topic gate -> clip acquisition -> ElevenLabs narration
-> re-encode concat assembly, writing thread_result.json into its own
output/_Threads/<slug>/ folder (a thread draws from two existing
episode runs, so it can't live in either one's own RunPaths tree).
Returns None, not an error, whenever the corpus doesn't support a
genuine same-topic pairing yet."
```

---

### Task 9: CLI wiring — `--clip-type thread`

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_main.py`, near the existing `--clip-type` tests:

```python
def test_clip_type_accepts_thread():
    args = build_parser().parse_args(["--clip-type", "thread"])
    assert args.clip_type == "thread"


def test_url_is_optional_for_clip_type_thread():
    args = build_parser().parse_args(["--clip-type", "thread"])
    assert args.url is None


def test_url_still_required_positional_when_provided():
    args = build_parser().parse_args(["https://example.com/video"])
    assert args.url == "https://example.com/video"


def test_main_fails_cleanly_without_url_for_shorts(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py"])
    exit_code = main()
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "url is required" in captured.err


def test_main_dispatches_to_generate_threads_for_clip_type_thread(monkeypatch, capsys):
    calls = []
    fake_result = {
        "output_dir": "output/_Threads/Some_Thesis",
        "shared_question": "Does X cause Y?",
        "thesis": "Two guests disagree.",
        "bridge": "Here's the other side.",
        "episode_a": {"title": "Episode A", "start_time": 10.0, "end_time": 30.0},
        "episode_b": {"title": "Episode B", "start_time": 5.0, "end_time": 25.0},
        "clip_url": "output/_Threads/Some_Thesis/thread.mp4",
    }
    monkeypatch.setattr(main_module, "generate_threads", lambda **kwargs: (calls.append(kwargs), fake_result)[1])
    monkeypatch.setattr(sys, "argv", ["main.py", "--clip-type", "thread"])

    exit_code = main()

    assert exit_code == 0
    assert calls == [{}]
    captured = capsys.readouterr()
    assert "Does X cause Y?" in captured.out


def test_main_reports_no_thread_available_when_generate_threads_returns_none(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "generate_threads", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["main.py", "--clip-type", "thread"])

    exit_code = main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No same-topic episode pair" in captured.err


def test_main_warns_when_url_given_with_clip_type_thread(monkeypatch, capsys):
    fake_result = {
        "output_dir": "d", "shared_question": "q?", "thesis": "t", "bridge": "b",
        "episode_a": {"title": "A", "start_time": 0.0, "end_time": 1.0},
        "episode_b": {"title": "B", "start_time": 0.0, "end_time": 1.0},
        "clip_url": "d/thread.mp4",
    }
    monkeypatch.setattr(main_module, "generate_threads", lambda **kwargs: fake_result)
    monkeypatch.setattr(sys, "argv", ["main.py", "https://example.com/video", "--clip-type", "thread"])

    main()

    captured = capsys.readouterr()
    assert "ignores the url argument" in captured.err
```

Check the top of `tests/test_main.py` for how `main_module`, `build_parser`, and `main` are already imported (the existing chapters tests use `main_module.generate_chapters` and call a bare `main()` after setting `sys.argv`) and match that pattern exactly — don't add a second import style.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_main.py -k thread -v`
Expected: FAIL — `--clip-type thread` rejected as an invalid choice, and `url` is currently a required positional so parsing `["--clip-type", "thread"]` alone raises `SystemExit`.

- [ ] **Step 3: Update `main.py`**

Modify the `url` argument to be optional:

```python
    parser.add_argument(
        "url", nargs="?", default=None,
        help="YouTube URL, file:// URL, or local file path. Not used for "
             "--clip-type thread, which draws from the existing local corpus instead.",
    )
```

Modify the `--clip-type` choices and help text:

```python
    parser.add_argument(
        "--clip-type",
        choices=["shorts", "chapters", "thread"],
        default="shorts",
        help="shorts (default): viral 9:16 Shorts. chapters: long-form landscape "
             "chapter cuts, up to 15min each, full topic context, --mode local only. "
             "thread: a two-episode same-topic compilation built from the existing "
             "local corpus, no url needed, --mode local only.",
    )
```

Update the top-of-file import:

```python
from shorts_generator import generate_chapters, generate_shorts, generate_threads
```

Add URL validation right after `args = build_parser().parse_args()` in `main()`:

```python
def main() -> int:
    args = build_parser().parse_args()

    if args.clip_type != "thread" and not args.url:
        print("\nFAILED: url is required for --clip-type shorts/chapters", file=sys.stderr)
        return 1
    if args.clip_type == "thread" and args.url:
        print(
            f"[main] --clip-type thread ignores the url argument ({args.url!r}); "
            "it draws from the existing local corpus instead",
            file=sys.stderr,
        )
```

This replaces the current first line of `main()` (`args = build_parser().parse_args()` alone) — keep the existing `if args.clip_type == "chapters": ...` block that follows it unchanged, just inserted after this new validation.

Modify the dispatch `try` block to add the thread branch first:

```python
    try:
        if args.clip_type == "thread":
            result = generate_threads()
        elif args.clip_type == "chapters":
            result = generate_chapters(
                youtube_url=args.url,
                num_chapters=args.num_chapters,
                download_format=args.format,
                language=args.language,
                captions=args.captions,
                caption_fade_duration=args.caption_fade_duration,
                word_highlight=args.word_highlight,
            )
        else:
            result = generate_shorts(
                youtube_url=args.url,
                num_clips=args.num_clips,
                aspect_ratio=args.aspect_ratio,
                download_format=args.format,
                language=args.language,
                mode=args.mode,
                captions=args.captions,
                caption_fade_duration=args.caption_fade_duration,
                word_highlight=args.word_highlight,
                framing=args.framing,
                hook_card=args.hook_card,
                end_card=args.end_card,
                filename_style=args.filename_style,
            )
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

    if args.clip_type == "thread" and result is None:
        print("\nNo same-topic episode pair found in the local corpus -- nothing to build.", file=sys.stderr)
        print("This is expected until enough related episodes have been transcribed locally.", file=sys.stderr)
        return 1
```

Add a thread branch to the results-printing section (right before the existing `if args.clip_type == "chapters":` print block):

```python
    print("\n" + "=" * 72)
    if args.clip_type == "thread":
        print(f"Output folder:   {result.get('output_dir')}")
        print(f"Shared question: {result.get('shared_question')}")
        print(f"Thesis:          {result.get('thesis')}")
        print(f"Bridge:          {result.get('bridge')}")
        ea, eb = result["episode_a"], result["episode_b"]
        print(f"Episode A:       {ea['title']} ({ea['start_time']:.1f}s -> {ea['end_time']:.1f}s)")
        print(f"Episode B:       {eb['title']} ({eb['start_time']:.1f}s -> {eb['end_time']:.1f}s)")
        print(f"Clip:            {result.get('clip_url')}")
    elif args.clip_type == "chapters":
        print(f"Output folder: {result.get('output_dir')}")
```

Note the `elif` — the existing `if args.clip_type == "chapters":` becomes `elif`, and its sibling `else:` (the shorts branch) is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_main.py -v`
Expected: PASS (all tests, including every pre-existing one)

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: add --clip-type thread CLI flag

Draws from the existing local corpus rather than a fresh url -- url
becomes optional and is ignored (with a warning) when passed alongside
--clip-type thread. Exits 1 with a clear message when the corpus
doesn't support a same-topic pairing yet, distinct from an actual
error."
```

---

### Task 10: `requirements-local.txt` gains `elevenlabs`

**Files:**
- Modify: `requirements-local.txt`

- [ ] **Step 1: Check current contents**

Run: `cat requirements-local.txt`

- [ ] **Step 2: Add the dependency**

Add a line `elevenlabs` to `requirements-local.txt` (alphabetical position if the file is sorted; otherwise append). Match whatever version-pinning style the rest of the file already uses (unpinned vs. pinned) rather than introducing a new convention.

- [ ] **Step 3: Verify it installs cleanly**

Run: `pip install -r requirements-local.txt`
Expected: no errors (the package was already installed by hand earlier in this session, so this should be a no-op confirming the pin, if any, matches what's installed).

- [ ] **Step 4: Commit**

```bash
git add requirements-local.txt
git commit -m "chore: add elevenlabs to requirements-local.txt

Thread narration (local/narration.py) depends on it; it was installed
ad hoc earlier and was missing from the tracked requirements file."
```

---

### Task 11: `.env.example` documents the new settings

**Files:**
- Modify: `.env.example` (create if it doesn't exist — check first with `ls .env.example`)

- [ ] **Step 1: Check whether `.env.example` exists and its current format**

Run: `cat .env.example 2>/dev/null || echo "does not exist"`

- [ ] **Step 2: Add the new variables**

If the file exists, add near the other local-mode settings (matching the file's existing section-comment style):

```
# Thread compilation (--clip-type thread, local mode only)
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb
```

If `.env.example` doesn't exist in this repo at all, skip this task entirely — don't introduce a new convention the project doesn't already have.

- [ ] **Step 3: Commit** (only if Step 2 made a change)

```bash
git add .env.example
git commit -m "docs: document ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID in .env.example"
```

---

## Self-Review Notes

**Spec coverage:**
- Corpus index with abstract caching → Task 3.
- Hard same-topic gate with explicit refusal (`None`, not a loose match) → Task 4.
- Source-URL persistence so a pruned `full_source.mp4` can be re-acquired → Task 1.
- Duration-mismatch guard before trusting any re-downloaded source's timestamps → Task 7.
- Word-level cut-point precision instead of raw segment boundaries → Task 7 (`_find_word_start`).
- ElevenLabs narration matching validated channel typography → Task 5.
- Cross-episode spec normalization for concat → Task 6.
- `generate_threads()` returning `None` (not raising) when the corpus doesn't support a thread → Task 8, propagated to a clean CLI exit in Task 9.
- CLI exposure → Task 9.

**Known limitation, stated on purpose, not a gap:** with the corpus at 3 unrelated-topic episodes (per the design spec), running `--clip-type thread` today will print "No same-topic episode pair found" and exit 1. That's correct behavior, not a bug to fix in this plan — the feature is ready for the moment the corpus supports it.

**Out of scope for this plan** (flag to the user, don't silently build): a retrofit pass over the 100+ existing channel videos, bulk corpus backfill tooling (re-transcribing old uploads specifically to grow the thread corpus), and switching `LOCAL_WHISPER_MODEL` from `base` to `small` project-wide — all previously discussed, none are part of this feature's file set.
