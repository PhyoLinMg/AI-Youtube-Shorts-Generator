"""Flask dashboard: submit a YouTube URL, watch progress, grab the shorts.

Single-user, local tool — at most one pipeline run at a time. State lives in
the module-level `job` object, guarded by `_job_lock` since a background
thread and Flask's request threads touch it concurrently. This one-run-at-a-
time constraint is load-bearing: `capture_progress_log` (run_output.py) swaps
sys.stdout/sys.stderr process-globally, not per-thread, so two concurrent
runs would interleave each other's progress logs.
"""
import json
import os
import sys
import threading
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from .config import LOCAL_OUTPUT_DIR
from .pipeline import generate_shorts, generate_threads
from .run_output import list_runs, resolve_output_dir, summarize_run

app = Flask(__name__)


@dataclass
class Job:
    status: str = "idle"  # "idle" | "starting" | "running" | "done" | "failed"
    url: str = ""
    clip_type: str = "shorts"  # "shorts" | "thread"
    progress_log: Optional[str] = None
    # For clip_type="thread" this holds the thread run's own output/_Threads/<slug>
    # dir (see resolve_thread_run_dir), reused as the download route's
    # serve-from directory -- but _run_name_from_shorts_dir must not be
    # applied to it (there's no per-episode run name for a thread; see /status).
    shorts_dir: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


job = Job()
_job_lock = threading.Lock()


def _run_job(
    url: str,
    mode: str,
    num_clips: int,
    aspect_ratio: str,
    download_format: str,
    language: Optional[str],
    captions: bool,
    caption_fade_duration: float,
    word_highlight: bool,
    framing: str,
    hook_card: bool,
    end_card: bool,
    filename_style: str,
) -> None:
    try:
        paths = resolve_output_dir(url)
        with _job_lock:
            job.progress_log = paths.progress_log
            job.shorts_dir = paths.shorts_dir
            job.status = "running"
        result = generate_shorts(
            url,
            num_clips=num_clips,
            aspect_ratio=aspect_ratio,
            download_format=download_format,
            language=language,
            mode=mode,
            captions=captions,
            caption_fade_duration=caption_fade_duration,
            word_highlight=word_highlight,
            framing=framing,
            hook_card=hook_card,
            end_card=end_card,
            filename_style=filename_style,
            paths=paths,
        )
        with _job_lock:
            job.result = result
            job.status = "done"
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        with _job_lock:
            job.error = str(e)
            job.status = "failed"


def _run_thread_job(url_a: str, url_b: str, num_clips: int, platform: str = "youtube") -> None:
    """Ingests url_a/url_b caption-only (no video download) and builds up
    to num_clips distinct shared-question threads between them -- see
    generate_threads in pipeline.py. platform selects "youtube" (default),
    "tiktok", or "both" -- see generate_threads. Like _run_job, the output
    dir isn't known until generate_threads has resolved it from the two
    episode titles, so job.progress_log/shorts_dir are set via the
    on_output_dir callback."""
    def _on_output_dir(out_dir: str) -> None:
        with _job_lock:
            job.status = "running"
            job.progress_log = os.path.join(out_dir, "progress.log")
            job.shorts_dir = out_dir

    try:
        result = generate_threads(url_a, url_b, num_clips=num_clips, platform=platform, on_output_dir=_on_output_dir)
        with _job_lock:
            if not result:
                job.error = (
                    "No thread built between these two episodes -- either they don't "
                    "genuinely cover the same topic, or a shared question was found but "
                    f"no span satisfying platform={platform!r}'s length bounds could be "
                    "grounded for it. Try a different pair, a different platform, or lower "
                    "num_clips."
                )
                job.status = "failed"
            else:
                job.result = result
                job.status = "done"
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        with _job_lock:
            job.error = str(e)
            job.status = "failed"


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


def _clip_display_url(shorts_dir: Optional[str], clip_url: Optional[str]) -> Optional[str]:
    if not clip_url:
        return None
    if clip_url.startswith("http://") or clip_url.startswith("https://"):
        return clip_url
    return f"/download/{_relative_clip_path(shorts_dir, clip_url)}"


def _clip_filename_for_delete(clip_url: Optional[str]) -> Optional[str]:
    """The bare filename a delete-clip request should target, or None when
    there's no local file to delete (no clip, or a remote-hosted URL)."""
    if not clip_url or clip_url.startswith("http://") or clip_url.startswith("https://"):
        return None
    return os.path.basename(clip_url)


def _clip_file_exists(shorts_dir: Optional[str], clip_url: Optional[str]) -> bool:
    """Whether a local clip_url still has a backing file. Remote URLs are
    always treated as existing — we have no way to check them here, and the
    pre-delete-clip behavior never checked them either."""
    if not clip_url:
        return False
    if clip_url.startswith("http://") or clip_url.startswith("https://"):
        return True
    if not shorts_dir:
        return False
    target = _safe_join(shorts_dir, _relative_clip_path(shorts_dir, clip_url))
    return bool(target and os.path.isfile(target))


def _serialize_result(result: dict, shorts_dir: Optional[str]) -> dict:
    # Only "shorts" is ever rendered by the dashboard — the full pipeline
    # result also carries the whole transcript and every highlight candidate,
    # which /status would otherwise re-serialize and re-send on every poll.
    shorts = []
    for s in result.get("shorts", []):
        clip_url = s.get("clip_url")
        if clip_url and not _clip_file_exists(shorts_dir, clip_url):
            # Clip was generated but its file has since been deleted (e.g.
            # via the per-clip delete button) — drop it instead of
            # re-rendering it as a "Failed" card, which it isn't.
            continue
        shorts.append({
            **s,
            "download_url": _clip_display_url(shorts_dir, clip_url),
            "clip_filename": _clip_filename_for_delete(clip_url),
        })
    return {"shorts": shorts}


def _serialize_thread_results(results: List[Dict], out_dir: Optional[str]) -> Dict:
    """Thread results have a different shape than shorts/chapters (multiple
    two-source clips, no score/hook fields) -- see generate_threads' return
    in pipeline.py. One entry per grounded shared-question pair (up to the
    requested num_clips, possibly fewer -- see select_thread_pairs)."""
    threads = []
    for r in results:
        clip_url = r.get("clip_url")
        if clip_url and not _clip_file_exists(out_dir, clip_url):
            # Clip was generated but its file has since been deleted -- drop
            # it instead of re-rendering it as a "Failed" card, which it isn't.
            continue
        episode_a_clip = (r.get("episode_a") or {}).get("clip_url")
        episode_b_clip = (r.get("episode_b") or {}).get("clip_url")
        episode_a_download_url = (
            _clip_display_url(out_dir, episode_a_clip) if _clip_file_exists(out_dir, episode_a_clip) else None
        )
        episode_b_download_url = (
            _clip_display_url(out_dir, episode_b_clip) if _clip_file_exists(out_dir, episode_b_clip) else None
        )
        threads.append({
            "shared_question": r.get("shared_question"),
            "thesis": r.get("thesis"),
            "bridge": r.get("bridge"),
            "title": r.get("title"),
            "description": r.get("description"),
            "platform": r.get("platform"),
            "episode_a": r.get("episode_a"),
            "episode_b": r.get("episode_b"),
            "download_url": _clip_display_url(out_dir, clip_url),
            "episode_a_download_url": episode_a_download_url,
            "episode_b_download_url": episode_b_download_url,
        })
    return {"threads": threads}


def _run_name_from_shorts_dir(shorts_dir: Optional[str]) -> Optional[str]:
    if not shorts_dir:
        return None
    return os.path.basename(os.path.dirname(shorts_dir))


def _safe_join(base_dir: str, name: str) -> Optional[str]:
    """Resolve `name` under `base_dir`, refusing to escape it (blocks '../')."""
    try:
        base_real = os.path.realpath(base_dir)
        target = os.path.realpath(os.path.join(base_real, name))
    except (ValueError, OSError):
        return None
    if target == base_real or target.startswith(base_real + os.sep):
        return target
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def start_run():
    clip_type = request.form.get("clip_type", "shorts")

    if clip_type == "thread":
        url_a = request.form.get("url_a", "").strip()
        url_b = request.form.get("url_b", "").strip()
        if not url_a or not url_b:
            return jsonify({"error": "url_a and url_b are both required for clip_type=thread"}), 400
        try:
            num_clips = int(request.form.get("num_clips", 2))
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"invalid input: {e}"}), 400
        platform = request.form.get("platform", "youtube")
        if platform not in ("youtube", "tiktok", "both"):
            return jsonify({"error": f"invalid platform: {platform!r}"}), 400

        with _job_lock:
            if job.status in ("starting", "running"):
                return jsonify({"error": "a run is already in progress"}), 409
            job.status = "starting"
            job.url = ""
            job.clip_type = "thread"
            job.progress_log = None
            job.shorts_dir = None
            job.result = None
            job.error = None

        threading.Thread(target=_run_thread_job, args=(url_a, url_b, num_clips, platform), daemon=True).start()
        return jsonify({"status": "starting"}), 202

    url = request.form.get("url", "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    try:
        kwargs = dict(
            mode=request.form.get("mode", "api"),
            num_clips=int(request.form.get("num_clips", 3)),
            aspect_ratio=request.form.get("aspect_ratio", "9:16"),
            download_format=request.form.get("format", "1080"),
            language=(request.form.get("language") or "").strip() or None,
            captions=request.form.get("captions", "true") == "true",
            caption_fade_duration=float(request.form.get("caption_fade_duration", 0.3)),
            word_highlight=request.form.get("word_highlight", "true") == "true",
            hook_card=request.form.get("hook_card", "true") == "true",
            end_card=request.form.get("end_card", "false") == "true",
            framing=request.form.get("framing", "locked"),
            filename_style=request.form.get("filename_style", "specific"),
        )
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"invalid input: {e}"}), 400

    with _job_lock:
        if job.status in ("starting", "running"):
            return jsonify({"error": "a run is already in progress"}), 409
        job.status = "starting"
        job.url = url
        job.clip_type = "shorts"
        job.progress_log = None
        job.shorts_dir = None
        job.result = None
        job.error = None

    threading.Thread(target=_run_job, args=(url,), kwargs=kwargs, daemon=True).start()
    return jsonify({"status": "starting"}), 202


@app.route("/status")
def status():
    offset = int(request.args.get("offset", 0))
    with _job_lock:
        current_status = job.status
        clip_type = job.clip_type
        progress_log = job.progress_log
        shorts_dir = job.shorts_dir
        result = job.result
        error = job.error

    log_text = ""
    new_offset = offset
    if progress_log and os.path.exists(progress_log):
        with open(progress_log, "rb") as f:
            f.seek(offset)
            chunk = f.read()
            new_offset = f.tell()
        log_text = chunk.decode("utf-8", errors="replace")

    if result and clip_type == "thread":
        serialized_result = _serialize_thread_results(result, shorts_dir)
        # A thread run has no per-episode run folder (shorts_dir is
        # _Threads/<slug> here, not output/<Title>/Shorts) -- there's no
        # History-tab run to name, and _run_name_from_shorts_dir(shorts_dir)
        # would wrongly resolve to "_Threads".
        run_name = None
    elif result:
        serialized_result = _serialize_result(result, shorts_dir)
        run_name = _run_name_from_shorts_dir(shorts_dir)
    else:
        serialized_result = None
        run_name = None

    return jsonify({
        "status": current_status,
        "log": log_text,
        "offset": new_offset,
        "result": serialized_result,
        "run_name": run_name,
        "error": error,
    })


@app.route("/download/<path:name>")
def download(name):
    with _job_lock:
        shorts_dir = job.shorts_dir
    if not shorts_dir:
        abort(404)
    target = _safe_join(shorts_dir, name)
    if not target or not os.path.isfile(target):
        abort(404)
    return send_from_directory(os.path.dirname(target), os.path.basename(target))


@app.route("/history")
def history():
    return jsonify({"runs": [asdict(r) for r in list_runs()]})


def _history_clip_display_url(name: str, shorts_dir: str, clip_url: Optional[str]) -> Optional[str]:
    if not clip_url:
        return None
    filename = os.path.basename(clip_url)
    if not _safe_join(shorts_dir, filename) or not os.path.isfile(os.path.join(shorts_dir, filename)):
        return None
    return f"/history/{name}/download/{filename}"


def _shorts_from_clip_files(shorts_dir: str) -> list:
    """Reconstruct a minimal shorts list straight from Shorts/*.mp4.

    Used when result.json is missing — e.g. a run crashed after cropping
    finished but before the result was written (see write_descriptions'
    former hashtags-list bug). Filenames are title-derived (unique_short_filename),
    so this recovers a usable title even with no description/hashtags.
    """
    if not os.path.isdir(shorts_dir):
        return []
    return [
        {"clip_url": n, "title": os.path.splitext(n)[0].replace("_", " ")}
        for n in sorted(os.listdir(shorts_dir))
        if n.endswith(".mp4")
    ]


@app.route("/history/<name>/shorts")
def history_shorts(name):
    """Past shorts for a run — title/description/score plus a playable URL.

    Read-only: unlike delete-source/delete-shorts, reviewing an old run's
    clips doesn't touch files an in-progress run could be writing, so this
    isn't gated by the one-run-at-a-time lock.
    """
    root = _safe_join(LOCAL_OUTPUT_DIR, name)
    if not root or not os.path.isdir(root):
        return jsonify({"error": "run not found"}), 404
    shorts_dir = os.path.join(root, "Shorts")
    result_path = os.path.join(root, "result.json")
    if not os.path.isfile(result_path):
        shorts_source = _shorts_from_clip_files(shorts_dir)
    else:
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        except (OSError, json.JSONDecodeError):
            return jsonify({"error": "could not read result.json"}), 500
        shorts_source = result.get("shorts", [])
    shorts = []
    for s in shorts_source:
        clip_url = s.get("clip_url")
        if clip_url and not _clip_file_exists(shorts_dir, clip_url):
            # Clip was generated but its file has since been deleted — drop
            # it instead of re-rendering it as a "Failed" card, which it isn't.
            continue
        shorts.append({
            **s,
            "download_url": _history_clip_display_url(name, shorts_dir, clip_url),
            "clip_filename": _clip_filename_for_delete(clip_url),
        })
    return jsonify({"shorts": shorts})


@app.route("/history/<name>/download/<path:filename>")
def history_download(name, filename):
    root = _safe_join(LOCAL_OUTPUT_DIR, name)
    if not root:
        abort(404)
    target = _safe_join(os.path.join(root, "Shorts"), filename)
    if not target or not os.path.isfile(target):
        abort(404)
    return send_from_directory(os.path.dirname(target), os.path.basename(target))


def _resolve_history_run(name: str) -> Tuple[Optional[str], Optional[Any]]:
    """Validate `name` as a run folder under LOCAL_OUTPUT_DIR.

    Returns `(root, None)` on success or `(None, error_response)` when the
    request should be rejected — mirrors the one-run-at-a-time guard already
    enforced by `POST /run`, since deleting files out from under an active
    pipeline run would corrupt it.
    """
    with _job_lock:
        active = job.status in ("starting", "running")
    if active:
        return None, (jsonify({"error": "a run is in progress"}), 409)
    root = _safe_join(LOCAL_OUTPUT_DIR, name)
    if not root or not os.path.isdir(root):
        return None, (jsonify({"error": "run not found"}), 404)
    return root, None


@app.route("/history/<name>/delete-source", methods=["POST"])
def delete_history_source(name):
    root, error = _resolve_history_run(name)
    if error:
        return error
    source_video = os.path.join(root, "full_source.mp4")
    try:
        os.remove(source_video)
    except FileNotFoundError:
        pass  # already gone — deleting is idempotent
    try:
        return jsonify(asdict(summarize_run(name, root)))
    except OSError:
        # `root` itself vanished between the isdir() check above and here
        # (e.g. a concurrent delete-shorts request) — treat it the same as
        # "not found" rather than 500ing.
        return jsonify({"error": "run not found"}), 404


@app.route("/history/<name>/delete-shorts", methods=["POST"])
def delete_history_shorts(name):
    root, error = _resolve_history_run(name)
    if error:
        return error
    shorts_dir = os.path.join(root, "Shorts")
    if os.path.isdir(shorts_dir):
        try:
            filenames = os.listdir(shorts_dir)
        except FileNotFoundError:
            filenames = []  # dir vanished between the isdir() check and here
        for filename in filenames:
            if filename.endswith(".mp4"):
                try:
                    os.remove(os.path.join(shorts_dir, filename))
                except FileNotFoundError:
                    pass  # already gone — deleting is idempotent
    try:
        return jsonify(asdict(summarize_run(name, root)))
    except OSError:
        # `root` itself vanished between the isdir() check above and here —
        # treat it the same as "not found" rather than 500ing.
        return jsonify({"error": "run not found"}), 404


@app.route("/history/<name>/shorts/<filename>/delete", methods=["POST"])
def delete_history_clip(name, filename):
    """Delete a single clip file. Shared by both the History tab and the
    just-finished results grid — a live run's shorts_dir is `LOCAL_OUTPUT_DIR
    /<name>/Shorts`, the same tree addressed by every /history/<name>/...
    route, so one endpoint covers both views."""
    root, error = _resolve_history_run(name)
    if error:
        return error
    shorts_dir = os.path.join(root, "Shorts")
    target = _safe_join(shorts_dir, filename)
    if not target:
        return jsonify({"error": "invalid filename"}), 400
    try:
        os.remove(target)
    except FileNotFoundError:
        pass  # already gone — deleting is idempotent
    try:
        return jsonify(asdict(summarize_run(name, root)))
    except OSError:
        # `root` itself vanished between the isdir() check above and here.
        return jsonify({"error": "run not found"}), 404
