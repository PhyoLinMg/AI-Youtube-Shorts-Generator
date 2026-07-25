# Jail 1 — Visual Hook Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** score whether the opening frame(s) of each selected clip grab attention on their own, independent of text/audio — the one gap in the already-shipped jail-1 (swipe jail) hook system.

**Architecture:** a new `visual_hook.py` module extracts 2 frames near a highlight's `start_time` via ffmpeg and sends them to a vision-capable LLM with a scoring rubric, attaching `visual_hook_score`/`visual_hook_reason` to each highlight. Runs as a post-selection pass in `pipeline.py`, after `get_highlights_cached` returns and before cropping — never written into the highlights cache, so no schema-version bump. Never blocks the pipeline: any failure (missing frames, vision call error, bad JSON) degrades that one highlight to "no score," same pattern `detect_content_type` already uses for LLM failures.

**Tech Stack:** Python, ffmpeg (frame extraction), OpenAI vision API (`gpt-4o-mini`, local mode), MuAPI (api mode — gated on Task 1's spike), pytest.

**Spec:** `docs/superpowers/specs/2026-07-26-three-jails-escape-design.md`

---

### Task 1: Spike — find a MuAPI vision-capable endpoint (api mode)

**Files:** none (investigation only; produces findings that Task 5 implements against)

This task exists because `shorts_generator/muapi.py`'s `run()` dispatches to MuAPI model endpoints by name (e.g. `"gpt-5-mini"`, `"autocrop"`), but nothing in this codebase has ever called a vision/image-input endpoint — the payload shape for image input is unknown. Local mode is NOT blocked by this (Task 4 uses the well-documented OpenAI vision API directly) — this spike only gates api mode's `call_muapi_vision_llm` (Task 5).

- [ ] **Step 1: Check the MuAPI playground for a vision-capable model**

Open `https://muapi.ai/playground` (already referenced in this repo's `README.md` for the existing `ai-clipping` and `autocrop` endpoints) and look for an image-input or vision/VQA-capable model — something that accepts an image plus a text prompt and returns text (a captioning model, a VQA model, or a general multimodal chat model like a hosted GPT-4o/Gemini-vision equivalent).

- [ ] **Step 2: If a candidate endpoint exists, probe it directly**

Write a throwaway script (don't commit it) using the existing `shorts_generator.muapi.run` helper, e.g.:

```python
from shorts_generator import muapi

result = muapi.run(
    "<candidate-endpoint-name>",
    {"prompt": "Describe this image in one sentence.", "image": "<a small test image, however that endpoint expects it -- URL vs base64, single 'image' key vs an 'images' list>"},
    label="vision-spike",
)
print(result)
```

Run it (`python <script>.py`) against a real MuAPI API key and inspect the raw response shape. Record: the exact endpoint name, the exact payload key(s) for image input, whether it wants a hosted URL or base64/data-URI, and where the text response lands in the result dict (mirroring how `call_muapi_llm` in `highlights.py` already searches `outputs`/`output`/`text`/`response`/`result`/`content` for the gpt-5-mini text endpoint).

- [ ] **Step 3: Decide go/no-go**

If a working vision endpoint was found: note its exact name and payload shape (you'll need it verbatim for Task 5). If no vision-capable endpoint exists on MuAPI, or none returns usable text: Task 5 ships `call_muapi_vision_llm` as an intentional always-fails stub (see that task) — api mode gets no visual-hook score, local mode still gets one via Task 4, and a comment marks where to wire in a real MuAPI call once one becomes available. Either outcome is a valid, complete result for this task — do not skip Tasks 2-7 waiting on MuAPI to add a feature it doesn't have.

---

### Task 2: Frame extraction + response parsing (`visual_hook.py`)

**Files:**
- Create: `shorts_generator/visual_hook.py`
- Test: `tests/test_visual_hook.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_visual_hook.py`:

```python
import os
import subprocess

import pytest

from shorts_generator.visual_hook import (
    HOOK_FRAME_OFFSETS,
    _extract_hook_frames,
    _parse_visual_hook_response,
)


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("visual_hook_src")
    path = str(tmp_dir / "source.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=24:duration=6",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            path,
        ],
        check=True,
    )
    return path


def test_extract_hook_frames_returns_one_file_per_offset(tmp_path, synthetic_video):
    frames = _extract_hook_frames(synthetic_video, start_time=1.0, out_dir=str(tmp_path))
    assert len(frames) == len(HOOK_FRAME_OFFSETS)
    for f in frames:
        assert os.path.exists(f)
        assert os.path.getsize(f) > 0


def test_extract_hook_frames_skips_offsets_past_video_end(tmp_path, synthetic_video):
    # synthetic_video is 6s; starting at 5.9s, the +0.6s offset frame (6.5s)
    # doesn't exist -- ffmpeg fails for that one offset, must not raise.
    frames = _extract_hook_frames(synthetic_video, start_time=5.9, out_dir=str(tmp_path))
    assert len(frames) < len(HOOK_FRAME_OFFSETS)


def test_parse_visual_hook_response_plain_json():
    parsed = _parse_visual_hook_response('{"visual_hook_score": 72, "visual_hook_reason": "unusual framing"}')
    assert parsed == {"visual_hook_score": 72, "visual_hook_reason": "unusual framing"}


def test_parse_visual_hook_response_strips_markdown_fence():
    parsed = _parse_visual_hook_response('```json\n{"visual_hook_score": 40, "visual_hook_reason": "flat shot"}\n```')
    assert parsed == {"visual_hook_score": 40, "visual_hook_reason": "flat shot"}


def test_parse_visual_hook_response_clamps_score_above_range():
    parsed = _parse_visual_hook_response('{"visual_hook_score": 150, "visual_hook_reason": "x"}')
    assert parsed["visual_hook_score"] == 100


def test_parse_visual_hook_response_clamps_score_below_range():
    parsed = _parse_visual_hook_response('{"visual_hook_score": -10, "visual_hook_reason": "x"}')
    assert parsed["visual_hook_score"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_visual_hook.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'shorts_generator.visual_hook'`.

- [ ] **Step 3: Implement `shorts_generator/visual_hook.py`**

```python
"""Visual-hook scoring: does the opening frame(s) of a candidate highlight
stop a scroll on their own, independent of text or audio -- the one gap in
the already-shipped verbal/textual hook system (see hook_strength/hook_card
in highlights.py/hook_card.py and
docs/superpowers/specs/2026-07-26-three-jails-escape-design.md).

Runs as a post-selection pass over the already-chosen `top` candidates in
pipeline.py, using the local source video both modes already have on disk
at that point. Results are informational only -- never written back into
the highlights cache -- so a failed or unavailable vision backend degrades
one highlight to "no score" rather than blocking the pipeline, exactly like
highlights.detect_content_type already degrades on LLM failure.
"""
import json
import re
import subprocess
from typing import Callable, Dict, List

VisionLLMFn = Callable[[str, List[str]], str]

HOOK_FRAME_OFFSETS = (0.0, 0.6)

VISUAL_HOOK_PROMPT = """You are scoring the VISUAL hook of a short-form video clip -- does the very first frame or two, with NO audio and NO on-screen text, grab attention and make someone stop scrolling?

Score 0-100:
- High (80+): a striking, unusual, or immediately intriguing image on its own -- unexpected action, a visually surprising scene, strong composition.
- Low (<30): a static talking-head frame, a blank/neutral background, nothing visually distinct from any other video.

Respond ONLY with valid JSON: {"visual_hook_score": int, "visual_hook_reason": "one sentence"}"""


def _extract_hook_frames(video_path: str, start_time: float, out_dir: str) -> List[str]:
    """ffmpeg-extract one JPEG per HOOK_FRAME_OFFSETS timestamp, relative to
    start_time. Returns the frames that were successfully extracted --
    silently skips any offset ffmpeg can't produce (e.g. past the end of
    the source video), never raises."""
    paths = []
    for i, offset in enumerate(HOOK_FRAME_OFFSETS):
        frame_path = f"{out_dir}/hook_frame_{i}.jpg"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start_time + offset:.3f}",
            "-i", video_path,
            "-frames:v", "1", "-q:v", "2",
            frame_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            paths.append(frame_path)
    return paths


def _parse_visual_hook_response(raw: str) -> Dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    score = max(0, min(100, int(float(data.get("visual_hook_score", 0)))))
    reason = str(data.get("visual_hook_reason") or "").strip()
    return {"visual_hook_score": score, "visual_hook_reason": reason}


def score_visual_hooks(
    source_video_path: str, highlights: List[Dict], llm_fn: VisionLLMFn,
) -> List[Dict]:
    """Attach visual_hook_score/visual_hook_reason to each highlight.

    Never raises: any per-highlight failure (frame extraction, vision call,
    bad JSON) is logged and that highlight is returned unmodified -- one bad
    candidate must never abort the rest of the pipeline."""
    import tempfile

    out = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, h in enumerate(highlights, 1):
            entry = dict(h)
            try:
                frame_paths = _extract_hook_frames(source_video_path, float(h["start_time"]), tmp_dir)
                if not frame_paths:
                    raise RuntimeError("no frames extracted")
                raw = llm_fn(VISUAL_HOOK_PROMPT, frame_paths)
                entry.update(_parse_visual_hook_response(raw))
            except Exception as e:
                print(f"[visual_hook] {i} skipped: {e}", flush=True)
            out.append(entry)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visual_hook.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add shorts_generator/visual_hook.py tests/test_visual_hook.py
git commit -m "feat: add frame extraction and response parsing for visual-hook scoring"
```

---

### Task 3: `score_visual_hooks` orchestration and graceful degradation

**Files:**
- Test: `tests/test_visual_hook.py`

(No new implementation in this task — `score_visual_hooks` was already written in Task 2 alongside the pieces it depends on, since it's a thin loop over already-tested helpers. This task adds the orchestration-level tests that exercise it end-to-end with a stub `llm_fn`, so the try/except degradation is actually verified rather than assumed.)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_visual_hook.py`:

```python
from shorts_generator.visual_hook import score_visual_hooks


def test_score_visual_hooks_attaches_score_and_reason(synthetic_video):
    highlights = [{"title": "Clip A", "start_time": 1.0, "end_time": 3.0}]

    def stub_llm(prompt, image_paths):
        assert len(image_paths) > 0
        return '{"visual_hook_score": 88, "visual_hook_reason": "surprising opener"}'

    result = score_visual_hooks(synthetic_video, highlights, llm_fn=stub_llm)

    assert result[0]["visual_hook_score"] == 88
    assert result[0]["visual_hook_reason"] == "surprising opener"
    assert result[0]["title"] == "Clip A"  # original fields preserved


def test_score_visual_hooks_degrades_gracefully_on_llm_failure(synthetic_video):
    highlights = [{"title": "Clip A", "start_time": 1.0, "end_time": 3.0}]

    def failing_llm(prompt, image_paths):
        raise RuntimeError("vision backend unavailable")

    result = score_visual_hooks(synthetic_video, highlights, llm_fn=failing_llm)

    assert "visual_hook_score" not in result[0]
    assert result[0]["title"] == "Clip A"  # highlight itself survives, unmodified


def test_score_visual_hooks_one_failure_does_not_block_others(synthetic_video):
    highlights = [
        {"title": "Clip A", "start_time": 1.0, "end_time": 3.0},
        {"title": "Clip B", "start_time": 2.0, "end_time": 4.0},
    ]
    calls = {"count": 0}

    def flaky_llm(prompt, image_paths):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("transient failure")
        return '{"visual_hook_score": 50, "visual_hook_reason": "ok"}'

    result = score_visual_hooks(synthetic_video, highlights, llm_fn=flaky_llm)

    assert "visual_hook_score" not in result[0]
    assert result[1]["visual_hook_score"] == 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_visual_hook.py -k score_visual_hooks -v`
Expected: these should mostly already PASS since `score_visual_hooks` was implemented in Task 2 — run this step to confirm rather than to find a failure. If any fail, that's a real bug in Task 2's implementation to fix now (e.g. verify `entry = dict(h)` correctly preserves original keys and that a raised exception inside the `try` block correctly leaves `entry` unmodified).

- [ ] **Step 3: Fix anything the tests turned up, otherwise no code change**

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_visual_hook.py -v`
Expected: PASS, 9 tests total (6 from Task 2 + 3 from this task).

- [ ] **Step 5: Commit**

```bash
git add tests/test_visual_hook.py
git commit -m "test: verify score_visual_hooks degrades gracefully on vision-call failure"
```

---

### Task 4: OpenAI vision backend for local mode

**Files:**
- Modify: `shorts_generator/local/llm.py`
- Test: `tests/test_local_llm.py`

- [ ] **Step 1: Note the existing test conventions**

`tests/test_local_llm.py` mocks the `openai` SDK client directly (`_FakeOpenAI`/`_FakeChat`/`_FakeCompletions`/`_FakeResponse`/`_FakeMessage`, monkeypatching `openai.OpenAI`) rather than hitting a real API — see `test_call_openai_llm_sets_timeout`. The existing `_FakeCompletions.create` doesn't capture its kwargs, so the new tests below add a small dedicated fake chain (`_FakeVisionCompletions` etc.) that does, rather than modifying the shared one other tests already depend on.

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_local_llm.py`:

```python
class _FakeVisionCompletions:
    last_kwargs = None

    def create(self, **kwargs):
        type(self).last_kwargs = kwargs
        return _FakeResponse()


class _FakeVisionChat:
    completions = _FakeVisionCompletions()


class _FakeVisionOpenAI:
    def __init__(self, **kwargs):
        self.chat = _FakeVisionChat()


def test_call_openai_vision_llm_sends_text_and_image_content(tmp_path, monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", _FakeVisionOpenAI)
    monkeypatch.setattr(local_llm, "require_openai_key", lambda: "test-key")

    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xe0fakejpegbytes")

    result = local_llm.call_openai_vision_llm("describe this", [str(image_path)])

    assert result == "ok"
    kwargs = _FakeVisionCompletions.last_kwargs
    assert kwargs["model"] == config.OPENAI_MODEL
    content = kwargs["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_call_openai_vision_llm_one_image_block_per_path(tmp_path, monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", _FakeVisionOpenAI)
    monkeypatch.setattr(local_llm, "require_openai_key", lambda: "test-key")

    p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    p1.write_bytes(b"one")
    p2.write_bytes(b"two")

    local_llm.call_openai_vision_llm("prompt", [str(p1), str(p2)])

    content = _FakeVisionCompletions.last_kwargs["messages"][0]["content"]
    assert len(content) == 3  # 1 text block + 2 image blocks
```

`_FakeResponse` (returning `.choices[0].message.content == "ok"`) is already defined at the top of `tests/test_local_llm.py` and is reused here unchanged.

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_local_llm.py -k vision -v`
Expected: FAIL — `ImportError: cannot import name 'call_openai_vision_llm'`.

- [ ] **Step 4: Implement `call_openai_vision_llm`**

In `shorts_generator/local/llm.py`, add `import base64` to the top-level imports, and add this function after `call_openai_llm`:

```python
def call_openai_vision_llm(prompt: str, image_paths: List[str]) -> str:
    """OpenAI vision backend for visual_hook.score_visual_hooks, used by
    --mode local regardless of LLM_PROVIDER (Gemini's local text path stays
    on call_local_llm; this is the one place --mode local always uses
    OpenAI, since it's the only vision backend implemented so far -- a
    missing/invalid OPENAI_API_KEY here just means score_visual_hooks
    degrades that highlight to "no visual hook score," it doesn't fail the
    run, per score_visual_hooks's per-highlight try/except)."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "openai is required for visual-hook scoring. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    content = [{"type": "text", "text": prompt}]
    for path in image_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    client = OpenAI(api_key=require_openai_key(), timeout=LOCAL_LLM_TIMEOUT_SECONDS)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content or ""
```

Add `List` to the file's imports if not already present (`from typing import List` — check the top of `local/llm.py`; if it currently has no `typing` import, add `from typing import List`).

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_local_llm.py -v`
Expected: PASS, all tests including the new one, zero regressions.

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/local/llm.py tests/test_local_llm.py
git commit -m "feat: add OpenAI vision backend for local-mode visual-hook scoring"
```

---

### Task 5: MuAPI vision backend for api mode (per Task 1's spike findings)

**Files:**
- Modify: `shorts_generator/visual_hook.py`
- Test: `tests/test_visual_hook.py`

- [ ] **Step 1: Branch on Task 1's outcome**

**If Task 1 found a working MuAPI vision endpoint:** write `call_muapi_vision_llm(prompt: str, image_paths: List[str]) -> str` in `shorts_generator/visual_hook.py` using `muapi.run(<endpoint name from Task 1>, <payload shape from Task 1>, label="visual-hook")`, extracting the text response the same way `call_muapi_llm` in `highlights.py` does (search `outputs[0]`, then `output`/`text`/`response`/`result`/`content`). Add a corresponding test in `tests/test_visual_hook.py` monkeypatching `shorts_generator.visual_hook.muapi.run` and asserting the payload shape matches what Task 1 found.

**If Task 1 found no usable endpoint,** implement the stub instead:

```python
def call_muapi_vision_llm(prompt: str, image_paths: List[str]) -> str:
    """MuAPI vision backend for score_visual_hooks, used by --mode api.

    MuAPI has no usable vision-capable endpoint as of this writing (see the
    Task 1 spike in docs/superpowers/plans/2026-07-26-jail1-visual-hook.md).
    This raises unconditionally so score_visual_hooks's per-highlight
    try/except degrades api mode to "no visual hook score" rather than
    blocking the pipeline. Replace this body with a real muapi.run(...)
    call once MuAPI ships a vision-capable endpoint.
    """
    raise RuntimeError("no MuAPI vision endpoint available")
```

with this test:

```python
from shorts_generator.visual_hook import call_muapi_vision_llm


def test_call_muapi_vision_llm_raises_until_muapi_has_a_vision_endpoint():
    with pytest.raises(RuntimeError):
        call_muapi_vision_llm("prompt", ["frame.jpg"])
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_visual_hook.py -v`
Expected: PASS, all tests.

- [ ] **Step 3: Commit**

```bash
git add shorts_generator/visual_hook.py tests/test_visual_hook.py
git commit -m "feat: add api-mode vision backend for visual-hook scoring"
```

---

### Task 6: Wire `score_visual_hooks` into the pipeline

**Files:**
- Modify: `shorts_generator/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Note the existing test conventions**

`tests/test_pipeline.py` monkeypatches `pipeline_module.get_highlights_cached`, `pipeline_module.crop_highlights`/`local_clipper_module.crop_highlights_local`, `pipeline_module.download_youtube`/`pipeline_module._download_to`, etc. — always at the module level where `pipeline.py` looks the name up (per the 2026-07-19 highlights-cache design doc's note: patching the definition site instead goes silently inert). `_fake_download_to(url, dest)` writes real bytes to `dest` so `paths.source_video` exists on disk afterward. The new tests below monkeypatch `pipeline_module.score_visual_hooks` the same way, reusing the file's existing `_fake_transcript()`, `_fake_highlights_result()`, and `_paths(tmp_path)` helpers verbatim.

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_pipeline.py`, after `test_run_api_crops_double_num_clips_candidates`:

```python
def test_run_api_calls_score_visual_hooks_before_crop(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())

    calls = []

    def fake_score_visual_hooks(source_video_path, highlights, llm_fn):
        calls.append((source_video_path, len(highlights)))
        return highlights

    monkeypatch.setattr(pipeline_module, "score_visual_hooks", fake_score_visual_hooks)
    monkeypatch.setattr(pipeline_module, "crop_highlights", Mock(return_value=[]))

    paths = _paths(tmp_path)
    pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=paths,
        word_highlight=True,
    )

    assert len(calls) == 1
    assert calls[0] == (paths.source_video, 1)


def test_run_api_visual_hook_failure_does_not_abort_run(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "download_youtube", lambda url, fmt: "https://hosted.example/source.mp4")
    monkeypatch.setattr(pipeline_module, "_download_to", _fake_download_to)
    monkeypatch.setattr(pipeline_module, "transcribe", lambda url, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())

    def raising_score_visual_hooks(source_video_path, highlights, llm_fn):
        raise RuntimeError("vision backend down")

    monkeypatch.setattr(pipeline_module, "score_visual_hooks", raising_score_visual_hooks)
    crop_mock = Mock(return_value=[{"clip_url": "https://hosted.example/Short-1.mp4"}])
    monkeypatch.setattr(pipeline_module, "crop_highlights", crop_mock)

    result = pipeline_module._run_api(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=True,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    assert result["mode"] == "api"
    assert crop_mock.called  # pipeline continued past the failed visual-hook scoring


def test_run_local_calls_score_visual_hooks_before_crop(tmp_path, monkeypatch):
    monkeypatch.setattr(
        local_downloader_module, "download_youtube_local",
        lambda url, target_path, fmt: "/tmp/source.mp4",
    )
    monkeypatch.setattr(local_transcriber_module, "transcribe_local", lambda path, language=None: _fake_transcript())
    monkeypatch.setattr(pipeline_module, "get_highlights_cached", lambda transcript, num_clips, cache_path, llm_fn: _fake_highlights_result())

    calls = []

    def fake_score_visual_hooks(source_video_path, highlights, llm_fn):
        calls.append((source_video_path, len(highlights)))
        return highlights

    monkeypatch.setattr(pipeline_module, "score_visual_hooks", fake_score_visual_hooks)
    monkeypatch.setattr(local_clipper_module, "crop_highlights_local", Mock(return_value=[]))

    pipeline_module._run_local(
        "https://youtube.example/x",
        num_clips=1,
        aspect_ratio="9:16",
        download_format="720",
        language=None,
        captions=False,
        caption_fade_duration=0.3,
        paths=_paths(tmp_path),
        word_highlight=True,
    )

    assert len(calls) == 1
    assert calls[0] == ("/tmp/source.mp4", 1)
```

These reuse `_fake_download_to`, `_fake_transcript`, `_fake_highlights_result`, and `_paths` exactly as defined at the top of `tests/test_pipeline.py` (lines 14-43 as of this writing) — no new fixtures needed.

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -k visual_hook -v`
Expected: FAIL — `AttributeError: module 'shorts_generator.pipeline' has no attribute 'score_visual_hooks'`.

- [ ] **Step 4: Wire it in**

In `shorts_generator/pipeline.py`, add to the imports at the top:

```python
from .visual_hook import score_visual_hooks
```

and import the two vision backends where each mode needs them:

```python
from .local.llm import call_openai_vision_llm
from .visual_hook import call_muapi_vision_llm, score_visual_hooks
```

In `_run_api` (pipeline.py:89-167), right after the existing line:

```python
    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:2 * num_clips]
    print(f"[pipeline] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)
```

insert:

```python
    try:
        top = score_visual_hooks(paths.source_video, top, llm_fn=call_muapi_vision_llm)
    except Exception as e:
        print(f"[pipeline] visual-hook scoring skipped: {e}", flush=True)
```

In `_run_local` (pipeline.py:26-86), right after the equivalent line:

```python
    top = sorted(all_highlights, key=lambda h: int(h.get("score", 0)), reverse=True)[:2 * num_clips]
    print(f"[pipeline/local] cropping {len(top)} of {len(all_highlights)} candidates", flush=True)
```

insert:

```python
    try:
        top = score_visual_hooks(source_path, top, llm_fn=call_openai_vision_llm)
    except Exception as e:
        print(f"[pipeline/local] visual-hook scoring skipped: {e}", flush=True)
```

Both call sites wrap `score_visual_hooks` in their own `try/except` even though `score_visual_hooks` already catches per-highlight failures internally — this outer guard covers a failure in the call itself (e.g. `tempfile.TemporaryDirectory()` failing, or `llm_fn` raising before the loop even starts iterating in some future refactor), consistent with this codebase's existing belt-and-suspenders pattern (`detect_content_type` degrades internally too, but nothing calling it assumes that alone is sufficient).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS — baseline count (per the jail-2 plan's Task 1, re-verified here since that plan and this one may run independently) plus the 3 new tests, zero regressions.

- [ ] **Step 6: Commit**

```bash
git add shorts_generator/pipeline.py tests/test_pipeline.py
git commit -m "feat: wire visual-hook scoring into both pipeline modes"
```

---

### Task 7: Surface `visual_hook_score` in the webapp

**Files:**
- Modify: `shorts_generator/templates/index.html`

- [ ] **Step 1: Add a visual-hook meter and reason text block**

In `shorts_generator/templates/index.html`'s `buildShortCard`, right after the `hook_strength` meter block (or after whichever jail-2/jail-3 badges already landed):

```javascript
        if (typeof s.visual_hook_score === "number") {
          const visualScore = Number(s.visual_hook_score) || 0;
          const visualRow = document.createElement("div");
          visualRow.className = "score-row";
          const visualLabel = document.createElement("span");
          visualLabel.textContent = "Visual hook";
          visualRow.appendChild(visualLabel);
          const visualMeter = document.createElement("div");
          visualMeter.className = "meter";
          const visualMeterFill = document.createElement("span");
          visualMeterFill.style.width = visualScore + "%";
          visualMeterFill.style.background = scoreColor(visualScore);
          visualMeter.appendChild(visualMeterFill);
          visualRow.appendChild(visualMeter);
          const visualNum = document.createElement("span");
          visualNum.textContent = visualScore;
          visualRow.appendChild(visualNum);
          card.appendChild(visualRow);
        }

        if (s.visual_hook_reason) {
          appendLabeledText(card, "Visual read", "reason", s.visual_hook_reason);
        }
```

Note the `typeof s.visual_hook_score === "number"` guard (matching the existing `hook_strength` block's guard) — this field is genuinely optional per-highlight (absent whenever `score_visual_hooks` degraded that highlight), unlike the jail-2/jail-3 fields which `_sanitize_highlights` always populates with a default.

- [ ] **Step 2: Manually verify in the browser**

Same manual-check approach as the other two plans: start the dev server, load a run (or paste a fake `visual_hook_score`/`visual_hook_reason` into a past `result.json` temporarily) and confirm the meter renders, including the case where the field is absent (older results, or a run where vision scoring degraded) — the card must render cleanly with the block simply omitted, not throw.

- [ ] **Step 3: Commit**

```bash
git add shorts_generator/templates/index.html
git commit -m "feat: show visual hook score on short cards"
```

---

## Definition of done

- [ ] `python -m pytest tests/ -q` passes with zero regressions.
- [ ] Local mode (`--mode local`) produces a `visual_hook_score`/`visual_hook_reason` on each `top` candidate via OpenAI vision, when `OPENAI_API_KEY` is configured.
- [ ] Api mode either does the same via a real MuAPI vision endpoint (if Task 1 found one) or cleanly no-ops with a log line (if it didn't) — either way, api mode's pipeline run completes successfully.
- [ ] A vision-call failure on any single highlight never aborts the run (`test_score_visual_hooks_one_failure_does_not_block_others`, `test_run_api_visual_hook_failure_does_not_abort_run`).
- [ ] `visual_hook_score`/`visual_hook_reason` are visible in the webapp UI when present, and absent gracefully when not.
- [ ] No `HIGHLIGHT_SCHEMA_VERSION` bump — confirm `highlights.json` caches are untouched by this plan (`visual_hook_score` is never written to that file).
