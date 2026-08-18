# Thread Output Folder Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `output/_Threads/<full-title>_x_<full-title>/` (unreadable names, flat mixed raw/final files) with `output/_Threads/<date>_<short-slug>_x_<short-slug>/` (short names, `raw/thesis_N/` split, final videos named from their own clickbait title at the folder root), and migrate the 17 existing thread folders into the new scheme.

**Architecture:** Two new helpers in `run_output.py` (`short_slug`, `archive_stale_thread_run`) plus a changed `resolve_thread_run_dir`; `pipeline.generate_threads` writes intermediates under `raw/thesis_N/` and names the final video from the thesis's own title; `webapp.py`'s clip-URL helpers switch from bare-basename to base-dir-relative paths so nested `raw/` files stay downloadable; a standalone dry-run-by-default script migrates the existing 17 folders.

**Tech Stack:** Python 3.14, pytest, Flask (webapp.py), stdlib `shutil`/`datetime`/`json` only — no new dependencies.

---

## Design reference

Full rationale: `docs/superpowers/specs/2026-08-19-thread-folder-reorg-design.md`

## File Structure

- Modify `shorts_generator/run_output.py`: add `short_slug()`, add `archive_stale_thread_run()`, change `resolve_thread_run_dir()`, fix `write_thread_descriptions()`.
- Modify `shorts_generator/pipeline.py`: change `generate_threads()`'s path construction and call `archive_stale_thread_run`.
- Modify `shorts_generator/webapp.py`: add `_relative_clip_path()`, use it in `_clip_display_url()` and `_clip_file_exists()`.
- Create `migrate_thread_folders.py` (repo root, alongside the existing `ingest_corpus.py`): one-off migration script for the 17 existing folders.
- Modify `tests/test_run_output.py`, `tests/test_pipeline.py`, `tests/test_webapp.py`; create `tests/test_migrate_thread_folders.py`.

---

### Task 1: `short_slug()` helper

**Files:**
- Modify: `shorts_generator/run_output.py`
- Test: `tests/test_run_output.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_run_output.py` (near the other `sanitize_title` tests, after line 22):

```python
def test_short_slug_lowercases_hyphenates_and_truncates():
    result = run_output.short_slug("Godfather of AI: We Have 2 Years Before Everything Changes!")
    assert result == "godfather-of-ai-we-have-2"


def test_short_slug_strips_unsafe_characters():
    assert run_output.short_slug("A/B: Test?!") == "a-b-test"


def test_short_slug_empty_input_falls_back_to_untitled():
    assert run_output.short_slug("") == "untitled"
    assert run_output.short_slug("???") == "untitled"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_run_output.py -k short_slug -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.run_output' has no attribute 'short_slug'`

- [ ] **Step 3: Implement `short_slug()`**

In `shorts_generator/run_output.py`, add a new regex constant next to the existing ones (after line 26, `_UNDERSCORE_RUNS = re.compile(r"_+")`):

```python
_SHORT_SLUG_UNSAFE = re.compile(r"[^a-z0-9]+")
```

Then add the function right after `sanitize_title()` (after line 51):

```python
def short_slug(title: str, max_length: int = 25) -> str:
    """Aggressive, eyeball-friendly slug for thread folder names -- distinct
    from sanitize_title() above, which stays spaced-and-capitalized for
    single-episode output/<Title>/ folders and isn't changing."""
    lowered = (title or "").lower()
    slug = _SHORT_SLUG_UNSAFE.sub("-", lowered).strip("-")
    slug = slug[:max_length].strip("-")
    return slug or "untitled"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_run_output.py -k short_slug -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py
git commit -m "feat: add short_slug() helper for thread folder names"
```

---

### Task 2: Date-prefixed `resolve_thread_run_dir()`

**Files:**
- Modify: `shorts_generator/run_output.py:199-210`
- Test: `tests/test_run_output.py:213-215`

- [ ] **Step 1: Write the failing test**

In `tests/test_run_output.py`, add `from datetime import datetime` to the imports at the top of the file (after `import shutil`), then replace the existing test at line 213-215:

```python
def test_resolve_thread_run_dir_slugifies_both_titles(tmp_path):
    result = run_output.resolve_thread_run_dir("Episode A Title", "Episode B Title", base_dir=str(tmp_path))
    assert result == str(tmp_path / "_Threads" / "Episode_A_Title_x_Episode_B_Title")
```

with:

```python
def test_resolve_thread_run_dir_uses_date_and_short_slugs(tmp_path):
    fixed_now = datetime(2026, 8, 18, 14, 30, 0)
    result = run_output.resolve_thread_run_dir(
        "Episode A Title", "Episode B Title", base_dir=str(tmp_path), now=fixed_now,
    )
    assert result == str(tmp_path / "_Threads" / "2026-08-18_episode-a-title_x_episode-b-title")


def test_resolve_thread_run_dir_defaults_now_to_current_time(tmp_path):
    result = run_output.resolve_thread_run_dir("Episode A Title", "Episode B Title", base_dir=str(tmp_path))
    today = datetime.now().strftime("%Y-%m-%d")
    assert os.path.basename(result).startswith(today + "_")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_run_output.py -k resolve_thread_run_dir -v`
Expected: FAIL — `test_resolve_thread_run_dir_uses_date_and_short_slugs` fails with `TypeError: resolve_thread_run_dir() got an unexpected keyword argument 'now'`; the old-format assertion in the same test (before you replace it) would otherwise fail on a path mismatch.

- [ ] **Step 3: Implement the date-prefixed slug**

In `shorts_generator/run_output.py`, replace `resolve_thread_run_dir()` (lines 199-210):

```python
def resolve_thread_run_dir(
    title_a: str, title_b: str, base_dir: Optional[str] = None, now: Optional[datetime] = None,
) -> str:
    """A thread run's output lives outside any single episode's RunPaths
    tree -- it draws footage from two existing episode runs, so it gets its
    own output/_Threads/<date>_<slug>/ folder. Slugged from both episode
    titles (fixed by the caller up front, see generate_threads in
    pipeline.py) using short_slug() -- a more aggressive slugifier than
    sanitize_title() above, since two full episode titles concatenated
    together (the old scheme) made these folders unreadable. Date-prefixed
    so same-day thread runs sort together and folders stay short-lived and
    scannable in a plain file browser."""
    base_dir = base_dir or LOCAL_OUTPUT_DIR
    date_prefix = (now or datetime.now()).strftime("%Y-%m-%d")
    slug = f"{date_prefix}_{short_slug(title_a)}_x_{short_slug(title_b)}"
    root = os.path.join(base_dir, "_Threads", slug)
    os.makedirs(root, exist_ok=True)
    return root
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_run_output.py -k resolve_thread_run_dir -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full run_output test file to check for other breakage**

Run: `.venv/bin/pytest tests/test_run_output.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py
git commit -m "feat: date-prefix thread folder names with short slugs"
```

---

### Task 3: `archive_stale_thread_run()`

**Files:**
- Modify: `shorts_generator/run_output.py`
- Test: `tests/test_run_output.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_run_output.py`, right after the `resolve_thread_run_dir` tests from Task 2:

```python
def test_archive_stale_thread_run_returns_none_when_no_prior_run(tmp_path):
    out_dir = tmp_path / "thread"
    out_dir.mkdir()
    assert run_output.archive_stale_thread_run(str(out_dir)) is None


def test_archive_stale_thread_run_moves_prior_run_into_raw_stale(tmp_path):
    out_dir = tmp_path / "thread"
    (out_dir / "raw" / "thesis_1").mkdir(parents=True)
    (out_dir / "thread_results.json").write_text("[]")
    (out_dir / "descriptions.txt").write_text("d")
    (out_dir / "thesis_1_Old_Title.mp4").write_bytes(b"old final")
    (out_dir / "raw" / "thesis_1" / "clip_1_a.mp4").write_bytes(b"old raw")

    fixed_now = datetime(2026, 8, 18, 14, 30, 22)
    stale_dir = run_output.archive_stale_thread_run(str(out_dir), now=fixed_now)

    assert stale_dir == str(out_dir / "raw" / "stale" / "143022")
    assert not (out_dir / "thread_results.json").exists()
    assert not (out_dir / "thesis_1_Old_Title.mp4").exists()
    assert (Path(stale_dir) / "thread_results.json").exists()
    assert (Path(stale_dir) / "thesis_1_Old_Title.mp4").exists()
    assert (Path(stale_dir) / "thesis_1" / "clip_1_a.mp4").exists()
    assert set(os.listdir(out_dir)) == {"raw"}
    assert set(os.listdir(out_dir / "raw")) == {"stale"}


def test_archive_stale_thread_run_keeps_earlier_stale_archives(tmp_path):
    """A second same-day re-run must not clobber the first re-run's
    archive -- both timestamps should coexist under raw/stale/."""
    out_dir = tmp_path / "thread"
    (out_dir / "raw" / "stale" / "090000").mkdir(parents=True)
    (out_dir / "raw" / "stale" / "090000" / "old.mp4").write_bytes(b"first archive")
    (out_dir / "thread_results.json").write_text("[]")

    fixed_now = datetime(2026, 8, 18, 14, 30, 22)
    run_output.archive_stale_thread_run(str(out_dir), now=fixed_now)

    assert (out_dir / "raw" / "stale" / "090000" / "old.mp4").exists()
    assert (out_dir / "raw" / "stale" / "143022" / "thread_results.json").exists()
```

`Path` is already imported at module scope in this test file (used later for `write_descriptions` tests); `datetime` was added in Task 2.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_run_output.py -k archive_stale_thread_run -v`
Expected: FAIL with `AttributeError: module 'shorts_generator.run_output' has no attribute 'archive_stale_thread_run'`

- [ ] **Step 3: Implement `archive_stale_thread_run()`**

In `shorts_generator/run_output.py`, add `import shutil` to the imports at the top of the file (after `import re`), then add the function right after `resolve_thread_run_dir()`:

```python
def archive_stale_thread_run(out_dir: str, now: Optional[datetime] = None) -> Optional[str]:
    """If out_dir already holds a completed thread run (a thread_results.json
    from an earlier call that picked the same date+slug -- i.e. a same-day
    re-run of the same episode pair), move everything currently in it into
    raw/stale/<HHMMSS>/ before the new run writes anything, so the two
    runs' files can't silently mix. Returns the stale dir it archived into,
    or None if out_dir had no prior completed run to archive."""
    results_json = os.path.join(out_dir, "thread_results.json")
    if not os.path.isfile(results_json):
        return None

    timestamp = (now or datetime.now()).strftime("%H%M%S")
    stale_dir = os.path.join(out_dir, "raw", "stale", timestamp)
    os.makedirs(stale_dir, exist_ok=True)

    for name in os.listdir(out_dir):
        if name == "raw":
            continue
        shutil.move(os.path.join(out_dir, name), os.path.join(stale_dir, name))

    raw_dir = os.path.join(out_dir, "raw")
    for name in os.listdir(raw_dir):
        if name == "stale":
            continue
        shutil.move(os.path.join(raw_dir, name), os.path.join(stale_dir, name))

    return stale_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_run_output.py -k archive_stale_thread_run -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py
git commit -m "feat: archive a same-day thread re-run's files before overwriting"
```

---

### Task 4: `write_thread_descriptions()` uses the real final filename

**Files:**
- Modify: `shorts_generator/run_output.py:374-398`
- Test: `tests/test_run_output.py`

**Context:** `write_thread_descriptions` currently hardcodes the filename shown in `descriptions.txt` as `clip_{i}.mp4` (matching position `i`, not the actual `clip_url` in the dict) — the final filename is about to stop being `clip_{i}.mp4` (Task 5), so this label needs to reflect the real file. The existing tests all happen to use `"clip_1.mp4"` as their literal `clip_url` values, so `os.path.basename()` of those is unchanged and those tests keep passing unmodified — only a new test with a realistic filename is needed to prove the fix.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_run_output.py`, right after `test_write_thread_descriptions_falls_back_to_shared_question` (after line 331):

```python
def test_write_thread_descriptions_uses_actual_clip_url_basename(tmp_path):
    threads = [{
        "clip_url": "/some/output/_Threads/2026-08-18_a_x_b/thesis_1_Is_AI_a_threat.mp4",
        "title": "Is AI a threat? #Shorts", "description": "Watch both takes.",
    }]
    path = run_output.write_thread_descriptions(str(tmp_path), threads)
    content = Path(path).read_text()
    assert content == (
        "clip 1 (thesis_1_Is_AI_a_threat.mp4)\nTitle: Is AI a threat? #Shorts\nDescription: Watch both takes.\n"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_run_output.py -k test_write_thread_descriptions_uses_actual_clip_url_basename -v`
Expected: FAIL — content shows `clip 1 (clip_1.mp4)` instead of the real basename

- [ ] **Step 3: Fix the implementation**

In `shorts_generator/run_output.py`, in `write_thread_descriptions()` (line 391), change:

```python
        blocks.append(f"clip {i} (clip_{i}.mp4)\nTitle: {title}\nDescription: {description}")
```

to:

```python
        blocks.append(f"clip {i} ({os.path.basename(clip_url)})\nTitle: {title}\nDescription: {description}")
```

(`clip_url` is already bound by the existing `if not t.get("clip_url"): continue` check two lines above — capture it there instead of calling `.get()` twice: change that line from `if not t.get("clip_url"): continue` to `clip_url = t.get("clip_url")` / `if not clip_url: continue` on two lines.)

The full updated loop body:

```python
    for i, t in enumerate(threads, 1):
        clip_url = t.get("clip_url")
        if not clip_url:
            continue
        title = (t.get("title") or t.get("shared_question") or "Untitled").strip()
        description = (t.get("description") or "").strip()
        blocks.append(f"clip {i} ({os.path.basename(clip_url)})\nTitle: {title}\nDescription: {description}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_run_output.py -k thread_descriptions -v`
Expected: PASS (all `write_thread_descriptions` tests, including the 3 pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/run_output.py tests/test_run_output.py
git commit -m "fix: descriptions.txt shows a thread clip's real filename"
```

---

### Task 5: `pipeline.generate_threads()` writes `raw/thesis_N/` and names finals from their title

**Files:**
- Modify: `shorts_generator/pipeline.py:24, 511-513, 561-597`
- Test: `tests/test_pipeline.py:1044-1097`, add one new test after line 1234

- [ ] **Step 1: Update the failing tests first**

In `tests/test_pipeline.py`, replace `test_generate_threads_assembles_and_writes_results_for_each_pair` (lines 1044-1097) in full:

```python
def test_generate_threads_assembles_and_writes_results_for_each_pair(tmp_path, monkeypatch):
    episode_a_dir = tmp_path / "Episode_A"
    episode_b_dir = tmp_path / "Episode_B"
    episode_a_dir.mkdir()
    episode_b_dir.mkdir()
    (episode_a_dir / "full_source.json").write_text(json.dumps({"duration": 100.0, "segments": []}))
    (episode_b_dir / "full_source.json").write_text(json.dumps({"duration": 200.0, "segments": []}))

    monkeypatch.setattr(pipeline_module, "ingest_captions", _fake_ingest_captions({
        "https://example.com/a": {"run_dir": str(episode_a_dir), "title": "Episode A", "duration": 100.0, "segment_count": 0},
        "https://example.com/b": {"run_dir": str(episode_b_dir), "title": "Episode B", "duration": 200.0, "segment_count": 0},
    }))
    monkeypatch.setattr(pipeline_module, "get_abstract_cached", lambda run_dir, transcript, llm_fn=None: "an abstract")

    fake_pairs = [
        {
            "shared_question": "Does X cause Y?", "thesis": "t1", "bridge": "b1", "title": "Title One #Shorts",
            "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 10.0, "end_time": 30.0},
            "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 5.0, "end_time": 25.0},
        },
        {
            "shared_question": "Does A cause B?", "thesis": "t2", "bridge": "b2", "title": "Title Two #Shorts",
            "episode_a": {"run_dir": str(episode_a_dir), "title": "Episode A", "source_url": "https://example.com/a", "start_time": 40.0, "end_time": 60.0},
            "episode_b": {"run_dir": str(episode_b_dir), "title": "Episode B", "source_url": "https://example.com/b", "start_time": 35.0, "end_time": 55.0},
        },
    ]
    monkeypatch.setattr(pipeline_module, "select_thread_pairs", lambda entry_a, entry_b, transcript_a, transcript_b, num_clips, llm_fn: fake_pairs)
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt="720": open(target_path, "wb").write(b"full source") or target_path,
    )
    monkeypatch.setattr(pipeline_module, "acquire_clip", lambda run_dir, source_url, cached_duration, start_time, end_time, out_path: open(out_path, "wb").write(b"clip") or {"clip_path": out_path})
    monkeypatch.setattr(pipeline_module, "synthesize_narration", lambda text, out_path, **k: open(out_path, "wb").write(b"audio") or out_path)
    monkeypatch.setattr(pipeline_module, "render_narration_card", lambda audio_path, text, out_path: open(out_path, "wb").write(b"card") or out_path)
    assemble_calls = []
    monkeypatch.setattr(pipeline_module, "assemble_thread", lambda segment_paths, out_path: (assemble_calls.append(segment_paths), open(out_path, "wb").write(b"final"))[1] or out_path)

    result = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=2, base_dir=str(tmp_path))

    assert len(result) == 2
    out_dir = result[0]["output_dir"]
    assert result[0]["clip_url"] == os.path.join(out_dir, "thesis_1_Title_One_Shorts.mp4")
    assert result[1]["clip_url"] == os.path.join(out_dir, "thesis_2_Title_Two_Shorts.mp4")
    assert result[0]["episode_a"]["clip_url"] == os.path.join(out_dir, "raw", "thesis_1", "clip_1_a.mp4")
    assert result[0]["episode_b"]["clip_url"] == os.path.join(out_dir, "raw", "thesis_1", "clip_1_b.mp4")
    assert result[1]["episode_a"]["clip_url"] == os.path.join(out_dir, "raw", "thesis_2", "clip_2_a.mp4")
    assert assemble_calls[0] == [
        os.path.join(out_dir, "raw", "thesis_1", "intro_card_1.mp4"), os.path.join(out_dir, "raw", "thesis_1", "clip_1_a.mp4"),
        os.path.join(out_dir, "raw", "thesis_1", "bridge_card_1.mp4"), os.path.join(out_dir, "raw", "thesis_1", "clip_1_b.mp4"),
    ]
    assert os.path.isfile(os.path.join(out_dir, "thread_results.json"))
    with open(os.path.join(out_dir, "thread_results.json")) as f:
        written = json.load(f)
    assert len(written) == 2
```

Then add a new test right after `test_generate_threads_does_not_download_or_delete_preexisting_full_source` (after line 1234, the end of the file):

```python
def test_generate_threads_final_filename_falls_back_to_shared_question_when_title_missing(tmp_path, monkeypatch):
    """_setup_thread_run's fake pairs (used by most of the tests below this
    one) don't set a "title" key -- generate_threads must not crash on that,
    and should fall back to shared_question for the final filename, same
    fallback write_thread_descriptions already uses."""
    _setup_thread_run(tmp_path, monkeypatch, num_pairs=1)

    result = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=1, base_dir=str(tmp_path))

    out_dir = result[0]["output_dir"]
    assert result[0]["clip_url"] == os.path.join(out_dir, "thesis_1_Question_1.mp4")


def test_generate_threads_archives_prior_same_slug_run_before_second_call(tmp_path, monkeypatch):
    """Integration check that generate_threads actually wires up
    archive_stale_thread_run (unit-tested on its own in Task 3): calling it
    twice for the same episode pair on the same day must not mix the two
    runs' files -- the first run's thread_results.json (and everything else)
    should land under raw/stale/<timestamp>/ before the second run writes."""
    _setup_thread_run(tmp_path, monkeypatch, num_pairs=1)

    first = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=1, base_dir=str(tmp_path))
    out_dir = first[0]["output_dir"]

    second = pipeline_module.generate_threads("https://example.com/a", "https://example.com/b", num_clips=1, base_dir=str(tmp_path))

    assert second[0]["output_dir"] == out_dir
    stale_root = os.path.join(out_dir, "raw", "stale")
    assert os.path.isdir(stale_root)
    archived_dirs = os.listdir(stale_root)
    assert len(archived_dirs) == 1
    assert os.path.isfile(os.path.join(stale_root, archived_dirs[0], "thread_results.json"))
    # The fresh run's own final file must not have been swept into the archive.
    assert os.path.isfile(second[0]["clip_url"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -k generate_threads -v`
Expected: FAIL — filename/path assertions don't match the current flat `clip_{i}.mp4` / `clip_{i}_a.mp4` layout

- [ ] **Step 3: Implement the new path construction in `generate_threads()`**

In `shorts_generator/pipeline.py`, update the import line (line 24) to add `archive_stale_thread_run` and `sanitize_title`:

```python
from .run_output import RunPaths, archive_stale_thread_run, capture_progress_log, resolve_output_dir, resolve_thread_run_dir, sanitize_title, write_chapter_descriptions, write_descriptions, write_source_url, write_thread_descriptions
```

Then, right after `out_dir = resolve_thread_run_dir(...)` (line 511), archive any prior same-day run before the callback and before any writes:

```python
    out_dir = resolve_thread_run_dir(entry_a["title"], entry_b["title"], base_dir=base_dir)
    archive_stale_thread_run(out_dir)
    if on_output_dir:
        on_output_dir(out_dir)
```

Then, inside the `for i, thread in enumerate(pairs, 1):` loop (lines 561-590), replace the whole body from `clip_a_path = ...` through `assemble_thread(...)` with:

```python
                thesis_dir = os.path.join(out_dir, "raw", f"thesis_{i}")
                os.makedirs(thesis_dir, exist_ok=True)

                clip_a_path = os.path.join(thesis_dir, f"clip_{i}_a.mp4")
                clip_b_path = os.path.join(thesis_dir, f"clip_{i}_b.mp4")
                print(f"[pipeline/local] acquiring clip A from {episode_a['title']!r}...", flush=True)
                acquire_clip(
                    episode_a["run_dir"], episode_a["source_url"], cached_duration=transcript_a.get("duration") or 0.0,
                    start_time=episode_a["start_time"], end_time=episode_a["end_time"], out_path=clip_a_path,
                )
                print(f"[pipeline/local] acquiring clip B from {episode_b['title']!r}...", flush=True)
                acquire_clip(
                    episode_b["run_dir"], episode_b["source_url"], cached_duration=transcript_b.get("duration") or 0.0,
                    start_time=episode_b["start_time"], end_time=episode_b["end_time"], out_path=clip_b_path,
                )

                intro_audio = os.path.join(thesis_dir, f"thesis_{i}.mp3")
                bridge_audio = os.path.join(thesis_dir, f"bridge_{i}.mp3")
                print("[pipeline/local] synthesizing narration (thesis + bridge)...", flush=True)
                synthesize_narration(thread["thesis"], intro_audio)
                synthesize_narration(thread["bridge"], bridge_audio)

                intro_card = os.path.join(thesis_dir, f"intro_card_{i}.mp4")
                bridge_card = os.path.join(thesis_dir, f"bridge_card_{i}.mp4")
                print("[pipeline/local] rendering narration cards...", flush=True)
                render_narration_card(intro_audio, thread["thesis"], intro_card)
                render_narration_card(bridge_audio, thread["bridge"], bridge_card)

                final_title = thread.get("title") or thread["shared_question"]
                final_path = os.path.join(out_dir, f"thesis_{i}_{sanitize_title(final_title)}.mp4")
                print("[pipeline/local] assembling final thread (intro -> clip A -> bridge -> clip B)...", flush=True)
                assemble_thread([intro_card, clip_a_path, bridge_card, clip_b_path], final_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline.py -k generate_threads -v`
Expected: PASS (all `generate_threads` tests, including the new fallback test)

- [ ] **Step 5: Run the full pipeline and run_output test files to check for other breakage**

Run: `.venv/bin/pytest tests/test_pipeline.py tests/test_run_output.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/pipeline.py tests/test_pipeline.py
git commit -m "feat: write thread intermediates to raw/thesis_N/, name finals from their title"
```

---

### Task 6: `webapp.py` download URLs survive nested `raw/` paths

**Files:**
- Modify: `shorts_generator/webapp.py:126-153`
- Test: `tests/test_webapp.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_webapp.py`, right after `test_status_omits_a_source_clip_download_url_whose_file_is_missing` (after line 228):

```python
def test_status_thread_source_clip_under_raw_subfolder_gets_relative_download_url(client, monkeypatch, tmp_path):
    """episode_a/episode_b source clips now live under out_dir/raw/thesis_N/
    (see run_output.py's new thread folder layout) -- the download URL must
    carry that relative path, not collapse to a bare basename, or the link
    404s and _clip_file_exists wrongly reports the file as missing."""
    out_dir = str(tmp_path / "_Threads" / "2026-08-18_a_x_b")
    raw_dir = os.path.join(out_dir, "raw", "thesis_1")
    os.makedirs(raw_dir, exist_ok=True)
    clip_path = os.path.join(out_dir, "thesis_1_Title.mp4")
    clip_a_path = os.path.join(raw_dir, "clip_1_a.mp4")
    clip_b_path = os.path.join(raw_dir, "clip_1_b.mp4")
    for p in (clip_path, clip_a_path, clip_b_path):
        open(p, "wb").write(b"data")

    def _fake_generate_threads(url_a, url_b, num_clips=1, base_dir=None, on_output_dir=None):
        if on_output_dir:
            on_output_dir(out_dir)
        return [{
            "shared_question": "Does X cause Y?",
            "thesis": "Two guests disagree.",
            "bridge": "Here's the other side.",
            "episode_a": {"title": "Episode A", "clip_url": clip_a_path},
            "episode_b": {"title": "Episode B", "clip_url": clip_b_path},
            "output_dir": out_dir,
            "clip_url": clip_path,
        }]

    monkeypatch.setattr(webapp, "generate_threads", _fake_generate_threads)
    monkeypatch.setattr(webapp.threading, "Thread", _SyncThread)

    client.post("/run", data={"clip_type": "thread", "url_a": "https://example.com/a", "url_b": "https://example.com/b"})
    resp = client.get("/status")
    data = resp.get_json()

    thread = data["result"]["threads"][0]
    assert thread["download_url"] == "/download/thesis_1_Title.mp4"
    assert thread["episode_a_download_url"] == "/download/raw/thesis_1/clip_1_a.mp4"
    assert thread["episode_b_download_url"] == "/download/raw/thesis_1/clip_1_b.mp4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_webapp.py -k raw_subfolder -v`
Expected: FAIL — `episode_a_download_url`/`episode_b_download_url` come back `None` (today's `os.path.basename`-based `_clip_file_exists` looks for `clip_1_a.mp4` directly in `out_dir`, not `out_dir/raw/thesis_1/`, so the file is reported missing)

- [ ] **Step 3: Implement `_relative_clip_path()` and use it**

In `shorts_generator/webapp.py`, add a new helper right before `_clip_display_url` (before line 126):

```python
def _relative_clip_path(base_dir: Optional[str], clip_url: str) -> str:
    """Path to use in a /download/<path:name> URL and in _safe_join lookups
    -- relative to base_dir when clip_url lives under it. Works whether
    clip_url is flat (regular Shorts/Chapters clips, where this equals the
    bare basename) or nested (thread clips under raw/thesis_N/), falling
    back to the bare basename when it isn't under base_dir at all."""
    if base_dir:
        try:
            rel = os.path.relpath(clip_url, base_dir)
        except ValueError:
            rel = None
        if rel and rel != "." and not rel.startswith(".."):
            return rel
    return os.path.basename(clip_url)
```

Then change `_clip_display_url` (line 131):

```python
    return f"/download/{os.path.basename(clip_url)}"
```

to:

```python
    return f"/download/{_relative_clip_path(shorts_dir, clip_url)}"
```

And change `_clip_file_exists` (line 152):

```python
    target = _safe_join(shorts_dir, os.path.basename(clip_url))
```

to:

```python
    target = _safe_join(shorts_dir, _relative_clip_path(shorts_dir, clip_url))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_webapp.py -k raw_subfolder -v`
Expected: PASS

- [ ] **Step 5: Run the full webapp test file to check for other breakage**

Run: `.venv/bin/pytest tests/test_webapp.py -v`
Expected: PASS (all tests, including the pre-existing flat-path thread tests and the regular-Shorts tests — both still get a plain basename since `os.path.relpath` of a flat file against its own containing dir is just the basename)

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/webapp.py tests/test_webapp.py
git commit -m "fix: thread clip download URLs survive nested raw/ paths"
```

---

### Task 7: Migration script for the 17 existing thread folders

**Files:**
- Create: `migrate_thread_folders.py` (repo root)
- Test: `tests/test_migrate_thread_folders.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_migrate_thread_folders.py`:

```python
import json
import os
from pathlib import Path

import migrate_thread_folders as migrate_module


def _write(path: Path, content: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _old_thread_folder(tmp_path: Path, name: str, num_theses: int = 1) -> Path:
    old = tmp_path / "_Threads" / name
    results = []
    for i in range(1, num_theses + 1):
        results.append({
            "shared_question": f"Question {i}?",
            "title": f"Title {i} #Shorts",
            "episode_a": {"title": "Episode A Full Title"},
            "episode_b": {"title": "Episode B Full Title"},
            "clip_url": str(old / f"clip_{i}.mp4"),
        })
        _write(old / f"clip_{i}.mp4")
        _write(old / f"clip_{i}_a.mp4")
        _write(old / f"clip_{i}_b.mp4")
        _write(old / f"clip_{i}_a.json")
        _write(old / f"clip_{i}_b.json")
        _write(old / f"thesis_{i}.mp3")
        _write(old / f"bridge_{i}.mp3")
        _write(old / f"intro_card_{i}.mp4")
        _write(old / f"bridge_card_{i}.mp4")
    _write(old / "descriptions.txt")
    _write(old / "progress.log")
    (old / "thread_results.json").write_text(json.dumps(results))
    return old


def test_build_plan_maps_final_and_raw_files(tmp_path):
    old = _old_thread_folder(tmp_path, "Episode_A_x_Episode_B", num_theses=1)
    threads_root = str(tmp_path / "_Threads")

    plan = migrate_module.build_plan(str(old), threads_root)

    assert plan is not None
    assert plan.new_path.startswith(threads_root)
    assert plan.new_path.endswith("episode-a-full-title_x_episode-b-full-title")
    dst_names = {os.path.relpath(dst, plan.new_path) for _src, dst in plan.moves}
    assert "thesis_1_Title_1_Shorts.mp4" in dst_names
    assert os.path.join("raw", "thesis_1", "clip_1_a.mp4") in dst_names
    assert "descriptions.txt" in dst_names
    assert "thread_results.json" in dst_names


def test_build_plan_returns_none_for_folder_without_thread_results(tmp_path):
    old = tmp_path / "_Threads" / "Not_A_Thread"
    old.mkdir(parents=True)
    (old / "random.txt").write_text("x")

    assert migrate_module.build_plan(str(old), str(tmp_path / "_Threads")) is None


def test_build_plan_returns_none_and_flags_unrecognized_files(tmp_path, capsys):
    old = _old_thread_folder(tmp_path, "Episode_A_x_Episode_B", num_theses=1)
    (old / "mystery_file.mp4").write_bytes(b"?")

    plan = migrate_module.build_plan(str(old), str(tmp_path / "_Threads"))

    assert plan is None
    assert "mystery_file.mp4" in capsys.readouterr().out


def test_build_plan_moves_existing_stale_folder_wholesale(tmp_path):
    old = _old_thread_folder(tmp_path, "Episode_A_x_Episode_B", num_theses=1)
    _write(old / "stale" / "clip_3_a.mp4")
    _write(old / "stale" / "thesis_3.mp3")

    plan = migrate_module.build_plan(str(old), str(tmp_path / "_Threads"))

    assert plan is not None
    stale_moves = [(s, d) for s, d in plan.moves if s == str(old / "stale")]
    assert len(stale_moves) == 1
    _src, dst = stale_moves[0]
    assert dst == os.path.join(plan.new_path, "raw", "stale", "Episode_A_x_Episode_B")


def test_apply_plan_moves_files_and_removes_old_folder(tmp_path):
    old = _old_thread_folder(tmp_path, "Episode_A_x_Episode_B", num_theses=1)
    threads_root = str(tmp_path / "_Threads")
    plan = migrate_module.build_plan(str(old), threads_root)

    migrate_module.apply_plan(plan)

    assert not old.exists()
    assert os.path.isfile(os.path.join(plan.new_path, "thread_results.json"))
    assert os.path.isfile(os.path.join(plan.new_path, "raw", "thesis_1", "clip_1_a.mp4"))
    final_files = [n for n in os.listdir(plan.new_path) if n.startswith("thesis_1_")]
    assert final_files == ["thesis_1_Title_1_Shorts.mp4"]


def test_main_dry_run_does_not_touch_disk(tmp_path, capsys):
    old = _old_thread_folder(tmp_path, "Episode_A_x_Episode_B", num_theses=1)

    exit_code = migrate_module.main([str(tmp_path)])

    assert exit_code == 0
    assert old.exists()
    assert "would be migrated" in capsys.readouterr().out


def test_main_apply_migrates_and_reports_count(tmp_path, capsys):
    old = _old_thread_folder(tmp_path, "Episode_A_x_Episode_B", num_theses=1)

    exit_code = migrate_module.main(["--apply", str(tmp_path)])

    assert exit_code == 0
    assert not old.exists()
    assert "Migrated 1 folder(s)." in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_migrate_thread_folders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'migrate_thread_folders'`

- [ ] **Step 3: Implement `migrate_thread_folders.py`**

Create `migrate_thread_folders.py` at the repo root:

```python
"""One-off migration for existing output/_Threads/<slug>/ folders (created
before the 2026-08-19 folder reorg -- see docs/superpowers/specs/
2026-08-19-thread-folder-reorg-design.md) into the new date+short-slug
naming with a raw/thesis_N/ split.

Dry-run by default: prints the planned moves for every folder under
output/_Threads/ without touching disk. Pass --apply to actually perform
them.

Usage:
    python migrate_thread_folders.py [--apply] [<base_dir>]

<base_dir> defaults to LOCAL_OUTPUT_DIR (normally "output"); its
_Threads/ subfolder is what gets scanned.
"""
import json
import os
import shutil
import sys
from datetime import datetime
from typing import List, Optional, Tuple

from shorts_generator.config import LOCAL_OUTPUT_DIR
from shorts_generator.run_output import sanitize_title, short_slug

ROOT_LEVEL_PASSTHROUGH = {"descriptions.txt", "progress.log", "thread_results.json"}
IGNORED_FILES = {".DS_Store"}
RAW_SUFFIXES_PER_THESIS = [
    "clip_{i}_a.mp4", "clip_{i}_b.mp4", "clip_{i}_a.json", "clip_{i}_b.json",
    "thesis_{i}.mp3", "bridge_{i}.mp3", "intro_card_{i}.mp4", "bridge_card_{i}.mp4",
]


class MigrationPlan:
    def __init__(self, old_path: str, new_path: str):
        self.old_path = old_path
        self.new_path = new_path
        self.moves: List[Tuple[str, str]] = []

    def add_move(self, src: str, dst: str) -> None:
        self.moves.append((src, dst))


def _oldest_mtime_date(folder: str) -> str:
    """Best available proxy for run-start date -- old thread folders don't
    store a creation date anywhere, so this uses the oldest file mtime
    found anywhere in the folder."""
    oldest = None
    for dirpath, _dirnames, filenames in os.walk(folder):
        for name in filenames:
            try:
                mtime = os.path.getmtime(os.path.join(dirpath, name))
            except OSError:
                continue
            if oldest is None or mtime < oldest:
                oldest = mtime
    if oldest is None:
        oldest = os.path.getmtime(folder)
    return datetime.fromtimestamp(oldest).strftime("%Y-%m-%d")


def build_plan(old_path: str, threads_root: str) -> Optional[MigrationPlan]:
    """Returns None (after printing why) if old_path can't be safely
    migrated -- caller should leave it untouched in that case."""
    results_path = os.path.join(old_path, "thread_results.json")
    if not os.path.isfile(results_path):
        print(f"SKIP {old_path}: no thread_results.json, not a recognized thread folder")
        return None

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    if not isinstance(results, list) or not results:
        print(f"SKIP {old_path}: thread_results.json is empty or malformed")
        return None

    episode_a_title = ((results[0].get("episode_a") or {}).get("title") or "").strip()
    episode_b_title = ((results[0].get("episode_b") or {}).get("title") or "").strip()
    if not episode_a_title or not episode_b_title:
        print(f"SKIP {old_path}: thread_results.json is missing episode titles")
        return None

    date_prefix = _oldest_mtime_date(old_path)
    new_slug = f"{date_prefix}_{short_slug(episode_a_title)}_x_{short_slug(episode_b_title)}"
    new_path = os.path.join(threads_root, new_slug)
    plan = MigrationPlan(old_path, new_path)

    remaining = set(os.listdir(old_path)) - IGNORED_FILES

    if "stale" in remaining and os.path.isdir(os.path.join(old_path, "stale")):
        remaining.discard("stale")
        plan.add_move(
            os.path.join(old_path, "stale"),
            os.path.join(new_path, "raw", "stale", os.path.basename(old_path)),
        )

    for i, thread in enumerate(results, 1):
        final_title = thread.get("title") or thread.get("shared_question") or "Untitled"
        final_name = f"clip_{i}.mp4"
        if final_name in remaining:
            remaining.discard(final_name)
            plan.add_move(
                os.path.join(old_path, final_name),
                os.path.join(new_path, f"thesis_{i}_{sanitize_title(final_title)}.mp4"),
            )
        for pattern in RAW_SUFFIXES_PER_THESIS:
            name = pattern.format(i=i)
            if name in remaining:
                remaining.discard(name)
                plan.add_move(os.path.join(old_path, name), os.path.join(new_path, "raw", f"thesis_{i}", name))

    for name in ROOT_LEVEL_PASSTHROUGH:
        if name in remaining:
            remaining.discard(name)
            plan.add_move(os.path.join(old_path, name), os.path.join(new_path, name))

    if remaining:
        print(f"SKIP {old_path}: unrecognized files, leaving untouched: {sorted(remaining)}")
        return None

    return plan


def apply_plan(plan: MigrationPlan) -> None:
    os.makedirs(plan.new_path, exist_ok=True)
    for src, dst in plan.moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    for name in os.listdir(plan.old_path):
        # Only IGNORED_FILES (e.g. .DS_Store) should remain -- clear them so
        # the now-empty old folder can be removed.
        os.remove(os.path.join(plan.old_path, name))
    os.rmdir(plan.old_path)


def main(argv: List[str]) -> int:
    apply = "--apply" in argv
    positional = [a for a in argv if a != "--apply"]
    base_dir = positional[0] if positional else LOCAL_OUTPUT_DIR
    threads_root = os.path.join(base_dir, "_Threads")

    if not os.path.isdir(threads_root):
        print(f"No _Threads folder at {threads_root!r} -- nothing to migrate.")
        return 0

    plans = []
    for name in sorted(os.listdir(threads_root)):
        old_path = os.path.join(threads_root, name)
        if not os.path.isdir(old_path):
            continue
        plan = build_plan(old_path, threads_root)
        if plan:
            plans.append(plan)

    for plan in plans:
        print(f"{'APPLY' if apply else 'PLAN'} {plan.old_path}\n  -> {plan.new_path}")
        for src, dst in plan.moves:
            print(f"    {os.path.relpath(src, plan.old_path)} -> {os.path.relpath(dst, plan.new_path)}")

    if not apply:
        print(f"\n{len(plans)} folder(s) would be migrated. Re-run with --apply to perform the moves.")
        return 0

    for plan in plans:
        apply_plan(plan)
    print(f"\nMigrated {len(plans)} folder(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_migrate_thread_folders.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add migrate_thread_folders.py tests/test_migrate_thread_folders.py
git commit -m "feat: add dry-run-by-default migration script for existing thread folders"
```

---

### Task 8: Full test suite check, then a real dry run

**Files:** none modified

- [ ] **Step 1: Run the entire test suite**

Run: `.venv/bin/pytest -q`
Expected: PASS, no failures, no new warnings about the files touched in Tasks 1-7

- [ ] **Step 2: Dry-run the migration against the real `output/_Threads/`**

Run: `.venv/bin/python migrate_thread_folders.py`

This only prints a plan (default is dry-run) — it does not touch any files. Read the output:
- Confirm all 17 folders show up as `PLAN` (none as `SKIP`).
- If any show as `SKIP` with "unrecognized files", inspect that folder by hand — it likely has a manually-added file (e.g. a renamed final export) the script's pattern list doesn't know about; decide with the user whether to extend `RAW_SUFFIXES_PER_THESIS`/`ROOT_LEVEL_PASSTHROUGH` or leave that folder for manual cleanup.

- [ ] **Step 3: Stop here and hand back to the user**

Do **not** run `python migrate_thread_folders.py --apply` in this task — actually moving/renaming the user's 17 real thread folders (some containing final videos they may have already uploaded or manually renamed) is a real, hard-to-fully-reverse filesystem change outside this plan's automatic scope. Show the user the dry-run output from Step 2 and let them confirm before anyone runs `--apply`.

---

## Self-review notes

- **Spec coverage:** naming (Task 2), internal layout (Task 5), final filename (Task 5), same-day re-run archiving (Task 3, wired in Task 5), download-route compatibility (Task 6), migration (Task 7) — every section of the design doc has a task.
- **`sanitize_title` vs `short_slug`:** intentionally two different functions per the design doc — `sanitize_title` (existing, unchanged) still names single-episode `output/<Title>/` folders and now also thread final filenames; `short_slug` (new) only names the thread folder itself.
- **`archive_stale_thread_run` ordering:** called in `generate_threads` right after `resolve_thread_run_dir` and before `on_output_dir` fires, so a dashboard tailing `progress.log` immediately after `on_output_dir` never reads a half-archived file.
- **No renumbering on `write_thread_descriptions`:** the fix in Task 4 deliberately keeps `enumerate(threads, 1)` as the position source for the `clip {i}` label (matching the pre-existing "without_renumbering" test) — only the filename inside the parens changes to the real basename.
