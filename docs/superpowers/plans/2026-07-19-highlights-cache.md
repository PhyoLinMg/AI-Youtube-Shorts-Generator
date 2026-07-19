# Cache Highlights Across Pipeline Reruns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rerunning `generate_shorts()` on a video whose transcript hasn't changed skips the LLM highlight call(s) entirely, instead of re-hitting the (potentially paid, e.g. DeepSeek-via-OpenRouter) LLM every time.

**Architecture:** A new `get_highlights_cached()` wrapper in `highlights.py` fingerprints the transcript's content (SHA-256 of duration+segments), and reads/writes a small `highlights.json` cache file (new `RunPaths.highlights_json` field) keyed by that fingerprint plus `num_clips`. Both `pipeline.py`'s `_run_local` and `_run_api` call this wrapper instead of `get_highlights` directly — provider-agnostic, works identically for MuAPI (api mode) and local-LLM (local mode, including OpenRouter/DeepSeek).

**Tech Stack:** Python, hashlib, JSON, pytest, unittest.mock.

---

### Task 1: Add `highlights_json` to RunPaths

**Files:**
- Modify: `shorts_generator/run_output.py` (`RunPaths` dataclass, `resolve_output_dir`)
- Test: `tests/test_run_output.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_run_output.py`, extend `test_resolve_output_dir_builds_expected_tree` (find it via `grep -n "def test_resolve_output_dir_builds_expected_tree"`) by adding one line after the existing `paths.progress_log` assertion:

```python
    assert paths.progress_log == os.path.join(paths.root, "progress.log")
    assert paths.highlights_json == os.path.join(paths.root, "highlights.json")
    assert os.path.isdir(paths.shorts_dir)
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
.venv/bin/python -m pytest tests/test_run_output.py::test_resolve_output_dir_builds_expected_tree -v
```

Expected: FAIL — `AttributeError: 'RunPaths' object has no attribute 'highlights_json'`.

- [ ] **Step 3: Add the field**

In `shorts_generator/run_output.py`, the `RunPaths` dataclass currently reads:

```python
@dataclass
class RunPaths:
    root: str
    shorts_dir: str
    source_video: str
    source_json: str
    result_json: str
    progress_log: str
```

Add `highlights_json` after `source_json`:

```python
@dataclass
class RunPaths:
    root: str
    shorts_dir: str
    source_video: str
    source_json: str
    highlights_json: str
    result_json: str
    progress_log: str
```

And in `resolve_output_dir`, the return statement currently reads:

```python
    return RunPaths(
        root=root,
        shorts_dir=shorts_dir,
        source_video=os.path.join(root, "full_source.mp4"),
        source_json=os.path.join(root, "full_source.json"),
        result_json=os.path.join(root, "result.json"),
        progress_log=os.path.join(root, "progress.log"),
    )
```

Add the new path:

```python
    return RunPaths(
        root=root,
        shorts_dir=shorts_dir,
        source_video=os.path.join(root, "full_source.mp4"),
        source_json=os.path.join(root, "full_source.json"),
        highlights_json=os.path.join(root, "highlights.json"),
        result_json=os.path.join(root, "result.json"),
        progress_log=os.path.join(root, "progress.log"),
    )
```

- [ ] **Step 4: Fix the two test-file `RunPaths(...)` construction sites**

`RunPaths` is now missing a required positional/keyword field in two other places that build it directly, both of which will fail to import/collect once Step 3 lands. Fix both now so the suite still collects (their own tests come later in Tasks 4 and 5):

In `tests/test_pipeline.py`, `_paths()` currently reads:

```python
def _paths(tmp_path):
    root = str(tmp_path / "Video_Title")
    shorts_dir = os.path.join(root, "Shorts")
    os.makedirs(shorts_dir, exist_ok=True)
    return RunPaths(
        root=root,
        shorts_dir=shorts_dir,
        source_video=os.path.join(root, "full_source.mp4"),
        source_json=os.path.join(root, "full_source.json"),
        result_json=os.path.join(root, "result.json"),
        progress_log=os.path.join(root, "progress.log"),
    )
```

Change to:

```python
def _paths(tmp_path):
    root = str(tmp_path / "Video_Title")
    shorts_dir = os.path.join(root, "Shorts")
    os.makedirs(shorts_dir, exist_ok=True)
    return RunPaths(
        root=root,
        shorts_dir=shorts_dir,
        source_video=os.path.join(root, "full_source.mp4"),
        source_json=os.path.join(root, "full_source.json"),
        highlights_json=os.path.join(root, "highlights.json"),
        result_json=os.path.join(root, "result.json"),
        progress_log=os.path.join(root, "progress.log"),
    )
```

In `tests/test_webapp.py`, `_fake_run_paths()` currently reads:

```python
def _fake_run_paths(tmp_path):
    root = str(tmp_path / "Video_Title")
    shorts_dir = os.path.join(root, "Shorts")
    os.makedirs(shorts_dir, exist_ok=True)
    return RunPaths(
        root=root,
        shorts_dir=shorts_dir,
        source_video=os.path.join(root, "full_source.mp4"),
        source_json=os.path.join(root, "full_source.json"),
        result_json=os.path.join(root, "result.json"),
        progress_log=os.path.join(root, "progress.log"),
    )
```

Change to:

```python
def _fake_run_paths(tmp_path):
    root = str(tmp_path / "Video_Title")
    shorts_dir = os.path.join(root, "Shorts")
    os.makedirs(shorts_dir, exist_ok=True)
    return RunPaths(
        root=root,
        shorts_dir=shorts_dir,
        source_video=os.path.join(root, "full_source.mp4"),
        source_json=os.path.join(root, "full_source.json"),
        highlights_json=os.path.join(root, "highlights.json"),
        result_json=os.path.join(root, "result.json"),
        progress_log=os.path.join(root, "progress.log"),
    )
```

- [ ] **Step 5: Run the tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_run_output.py tests/test_pipeline.py tests/test_webapp.py -v
```

Expected: all PASS (the `test_pipeline.py` tests still pass `llm_fn` to the old `get_highlights` name at this point — that's fixed in Task 4).

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py tests/test_pipeline.py tests/test_webapp.py
git commit -m "feat: add highlights_json to RunPaths"
```

---

### Task 2: Transcript content fingerprint

**Files:**
- Modify: `shorts_generator/highlights.py` (imports, new `_transcript_fingerprint`)
- Test: `tests/test_highlights.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_highlights.py`. First extend the import line at the top of the file:

```python
from shorts_generator.highlights import _sanitize_highlights, _transcript_fingerprint, call_highlight_api
```

Then add these tests anywhere after the imports:

```python
def test_transcript_fingerprint_stable_for_identical_transcripts():
    t1 = {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi"}]}
    t2 = {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi"}]}
    assert _transcript_fingerprint(t1) == _transcript_fingerprint(t2)


def test_transcript_fingerprint_changes_when_segments_change():
    t1 = {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi"}]}
    t2 = {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "bye"}]}
    assert _transcript_fingerprint(t1) != _transcript_fingerprint(t2)


def test_transcript_fingerprint_changes_when_duration_changes():
    t1 = {"duration": 10.0, "segments": []}
    t2 = {"duration": 20.0, "segments": []}
    assert _transcript_fingerprint(t1) != _transcript_fingerprint(t2)
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_highlights.py -k fingerprint -v
```

Expected: FAIL — `ImportError: cannot import name '_transcript_fingerprint'`.

- [ ] **Step 3: Implement `_transcript_fingerprint`**

In `shorts_generator/highlights.py`, the imports currently read:

```python
import json
import re
import time
from typing import Callable, Dict, List, Optional

from . import muapi
```

Change to:

```python
import hashlib
import json
import os
import re
import time
from typing import Callable, Dict, List, Optional

from . import muapi
```

Then add this function — a reasonable spot is right after the `LLMFn` type alias and before `CONTENT_TYPE_PROMPT`:

```python
def _transcript_fingerprint(transcript: Dict) -> str:
    """Stable content hash used to invalidate the highlights cache when the
    transcript actually changes — independent of *how* it was obtained
    (freshly transcribed vs. read from either transcriber's own cache)."""
    payload = json.dumps(
        {"duration": transcript.get("duration"), "segments": transcript.get("segments")},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_highlights.py -k fingerprint -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/highlights.py tests/test_highlights.py
git commit -m "feat: add transcript content fingerprint helper"
```

---

### Task 3: `get_highlights_cached`

**Files:**
- Modify: `shorts_generator/highlights.py` (new `get_highlights_cached`)
- Test: `tests/test_highlights.py`

- [ ] **Step 1: Write the failing tests**

Extend the import line in `tests/test_highlights.py` again:

```python
from shorts_generator.highlights import (
    _sanitize_highlights,
    _transcript_fingerprint,
    call_highlight_api,
    get_highlights_cached,
)
```

Add `import json` and `import os` at the very top of `tests/test_highlights.py` (the file currently has no imports besides the `from shorts_generator.highlights import ...` line):

```python
import json
import os

from shorts_generator.highlights import (
    _sanitize_highlights,
    _transcript_fingerprint,
    call_highlight_api,
    get_highlights_cached,
)
```

Then add these tests:

```python
def _fake_short_transcript():
    return {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi there"}]}


def _fake_llm_responses(highlight_title):
    def fake_llm_fn(prompt):
        if "Analyze this video transcript" in prompt:
            return '{"content_type": "podcast", "density": "medium"}'
        return (
            '{"highlights": [{"title": "%s", "start_time": 0.0, "end_time": 3.0, "score": 90}]}'
            % highlight_title
        )
    return fake_llm_fn


def test_get_highlights_cached_calls_llm_and_writes_cache_on_miss(tmp_path):
    cache_path = str(tmp_path / "highlights.json")
    transcript = _fake_short_transcript()

    result = get_highlights_cached(
        transcript, num_clips=1, cache_path=cache_path, llm_fn=_fake_llm_responses("Clip")
    )

    assert result["highlights"][0]["title"] == "Clip"
    assert os.path.exists(cache_path)
    with open(cache_path) as f:
        cached = json.load(f)
    assert cached["num_clips"] == 1
    assert cached["transcript_fingerprint"] == _transcript_fingerprint(transcript)
    assert cached["highlights"][0]["title"] == "Clip"


def test_get_highlights_cached_skips_llm_on_matching_cache(tmp_path):
    cache_path = str(tmp_path / "highlights.json")
    transcript = _fake_short_transcript()
    with open(cache_path, "w") as f:
        json.dump({
            "transcript_fingerprint": _transcript_fingerprint(transcript),
            "num_clips": 1,
            "highlights": [{"title": "Cached Clip", "start_time": 0.0, "end_time": 3.0, "score": 80}],
        }, f)

    def fail_if_called(prompt):
        raise AssertionError("llm_fn should not be called on a cache hit")

    result = get_highlights_cached(transcript, num_clips=1, cache_path=cache_path, llm_fn=fail_if_called)

    assert result["highlights"][0]["title"] == "Cached Clip"


def test_get_highlights_cached_recomputes_on_num_clips_mismatch(tmp_path):
    cache_path = str(tmp_path / "highlights.json")
    transcript = _fake_short_transcript()
    with open(cache_path, "w") as f:
        json.dump({
            "transcript_fingerprint": _transcript_fingerprint(transcript),
            "num_clips": 1,
            "highlights": [{"title": "Cached Clip", "start_time": 0.0, "end_time": 3.0, "score": 80}],
        }, f)

    result = get_highlights_cached(
        transcript, num_clips=2, cache_path=cache_path, llm_fn=_fake_llm_responses("Fresh Clip")
    )

    assert result["highlights"][0]["title"] == "Fresh Clip"


def test_get_highlights_cached_recomputes_on_fingerprint_mismatch(tmp_path):
    cache_path = str(tmp_path / "highlights.json")
    transcript = _fake_short_transcript()
    with open(cache_path, "w") as f:
        json.dump({
            "transcript_fingerprint": "stale-fingerprint",
            "num_clips": 1,
            "highlights": [{"title": "Cached Clip", "start_time": 0.0, "end_time": 3.0, "score": 80}],
        }, f)

    result = get_highlights_cached(
        transcript, num_clips=1, cache_path=cache_path, llm_fn=_fake_llm_responses("Fresh Clip")
    )

    assert result["highlights"][0]["title"] == "Fresh Clip"


def test_get_highlights_cached_recomputes_on_corrupted_cache_file(tmp_path):
    cache_path = str(tmp_path / "highlights.json")
    with open(cache_path, "w") as f:
        f.write("{not valid json")

    transcript = _fake_short_transcript()

    result = get_highlights_cached(
        transcript, num_clips=1, cache_path=cache_path, llm_fn=_fake_llm_responses("Fresh Clip")
    )

    assert result["highlights"][0]["title"] == "Fresh Clip"
    with open(cache_path) as f:
        cached = json.load(f)
    assert cached["highlights"][0]["title"] == "Fresh Clip"
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_highlights.py -k get_highlights_cached -v
```

Expected: FAIL — `ImportError: cannot import name 'get_highlights_cached'`.

- [ ] **Step 3: Implement `get_highlights_cached`**

In `shorts_generator/highlights.py`, add this function after `get_highlights` (at the end of the file):

```python
def get_highlights_cached(
    transcript: Dict,
    num_clips: int,
    cache_path: str,
    llm_fn: Optional[LLMFn] = None,
) -> Dict:
    """Wraps get_highlights with an on-disk cache keyed by a transcript
    content fingerprint + num_clips, so rerunning the pipeline on a video
    whose transcript hasn't changed skips the LLM call(s) entirely.

    A fingerprint mismatch, num_clips mismatch, or unparseable cache file
    all fall back to a full recompute (which then overwrites the cache) —
    no partial reuse.
    """
    fingerprint = _transcript_fingerprint(transcript)

    if os.path.exists(cache_path):
        cached = None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except json.JSONDecodeError:
            print(f"[highlights] cached highlights corrupted, recomputing: {cache_path}", flush=True)

        if (
            isinstance(cached, dict)
            and cached.get("transcript_fingerprint") == fingerprint
            and cached.get("num_clips") == num_clips
            and isinstance(cached.get("highlights"), list)
        ):
            print(f"[highlights] reusing cached highlights: {cache_path}", flush=True)
            return {"highlights": cached["highlights"]}

    result = get_highlights(transcript, num_clips=num_clips, llm_fn=llm_fn or call_muapi_llm)

    tmp_path = cache_path + ".part"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "transcript_fingerprint": fingerprint,
                "num_clips": num_clips,
                "highlights": result.get("highlights", []),
            },
            f,
            ensure_ascii=False,
        )
    os.replace(tmp_path, cache_path)

    return result
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_highlights.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/highlights.py tests/test_highlights.py
git commit -m "feat: add get_highlights_cached"
```

---

### Task 4: Wire the cache into both pipeline paths

**Files:**
- Modify: `shorts_generator/pipeline.py` (`_run_local`, `_run_api`, import line)
- Test: `tests/test_pipeline.py` (6 monkeypatch sites)

- [ ] **Step 1: Update the 6 monkeypatch sites in test_pipeline.py first**

These currently patch a name (`get_highlights`) that `pipeline.py` is about to stop calling — patching it would go silently inert (the real `get_highlights_cached` calls the real `get_highlights` from inside `highlights.py`'s own namespace, untouched by a patch on `pipeline_module.get_highlights`). Fix the patch target and the stub's signature (adding `cache_path`) in the same pass. Every occurrence in `tests/test_pipeline.py` reads exactly:

```python
    monkeypatch.setattr(pipeline_module, "get_highlights", lambda transcript, num_clips, llm_fn: _fake_highlights_result())
```

Replace **every occurrence** (there are 6 — lines 42, 81, 109, 153, 213, 235 as of this writing) with:

```python
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())
```

- [ ] **Step 2: Run the pipeline tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_pipeline.py -v
```

Expected: FAIL — `pipeline_module` has no attribute `get_highlights_cached` yet (the monkeypatch target doesn't exist), or the tests error before reaching their assertions.

- [ ] **Step 3: Wire `get_highlights_cached` into pipeline.py**

The import line at the top of `shorts_generator/pipeline.py` currently reads:

```python
from .highlights import call_muapi_llm, get_highlights
```

Change to:

```python
from .highlights import call_muapi_llm, get_highlights_cached
```

In `_run_local`, this line:

```python
    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_local_llm)
```

becomes:

```python
    highlights_result = get_highlights_cached(
        transcript, num_clips=num_clips, cache_path=paths.highlights_json, llm_fn=call_local_llm,
    )
```

In `_run_api`, this line:

```python
    highlights_result = get_highlights(transcript, num_clips=num_clips, llm_fn=call_muapi_llm)
```

becomes:

```python
    highlights_result = get_highlights_cached(
        transcript, num_clips=num_clips, cache_path=paths.highlights_json, llm_fn=call_muapi_llm,
    )
```

- [ ] **Step 4: Run the pipeline tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_pipeline.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/pipeline.py tests/test_pipeline.py
git commit -m "feat: cache highlights across pipeline reruns"
```

---

### Task 5: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 2: Manual end-to-end check of the actual cache-skip behavior**

The unit tests cover the caching logic in isolation; this step proves it fires for real inside `generate_shorts()`. Run a two-call sequence against `get_highlights_cached` directly (no need for a real video/LLM):

```bash
.venv/bin/python - <<'EOF'
import json, tempfile, os
from shorts_generator.highlights import get_highlights_cached

tmp = tempfile.mkdtemp()
cache_path = os.path.join(tmp, "highlights.json")
transcript = {"duration": 10.0, "segments": [{"start": 0.0, "end": 5.0, "text": "hi there"}]}

calls = {"n": 0}
def llm_fn(prompt):
    calls["n"] += 1
    if "Analyze this video transcript" in prompt:
        return '{"content_type": "podcast", "density": "medium"}'
    return '{"highlights": [{"title": "Clip", "start_time": 0.0, "end_time": 3.0, "score": 90}]}'

r1 = get_highlights_cached(transcript, num_clips=1, cache_path=cache_path, llm_fn=llm_fn)
calls_after_first_run = calls["n"]
r2 = get_highlights_cached(transcript, num_clips=1, cache_path=cache_path, llm_fn=llm_fn)
calls_after_second_run = calls["n"]

print("calls after first run:", calls_after_first_run)
print("calls after second run (should be unchanged):", calls_after_second_run)
assert calls_after_first_run > 0
assert calls_after_second_run == calls_after_first_run, "second call must not hit the LLM again"
assert r1 == r2
print("OK: second run served entirely from cache")
EOF
```

Expected output ends with `OK: second run served entirely from cache`, and the second `calls` count equals the first.

- [ ] **Step 3: Confirm git status is clean**

```bash
git status
git log --oneline -8
```

Expected: working tree clean aside from the pre-existing unrelated dirty files noted at session start (`shorts_generator/config.py`, `shorts_generator/local/llm.py`, and whatever remains of the LLM-timeout work — `shorts_generator/highlights.py` and `tests/test_highlights.py` are no longer "unrelated dirty" since this plan's commits now include their prior uncommitted diff), plus untracked `AGENTS.md`, `result.json`, `run.log`, `tests/test_local_llm.py`; and 4 new commits from Tasks 1–4 on top of the spec-doc and plan-doc commits.
