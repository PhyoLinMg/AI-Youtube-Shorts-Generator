"""Flask dashboard: submit a YouTube URL, watch progress, grab the shorts.

Single-user, local tool — at most one pipeline run at a time. State lives in
the module-level `job` object, guarded by `_job_lock` since a background
thread and Flask's request threads touch it concurrently. This one-run-at-a-
time constraint is load-bearing: `capture_progress_log` (run_output.py) swaps
sys.stdout/sys.stderr process-globally, not per-thread, so two concurrent
runs would interleave each other's progress logs.
"""
import os
import sys
import threading
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from .config import LOCAL_OUTPUT_DIR
from .pipeline import generate_shorts
from .run_output import list_runs, resolve_output_dir, summarize_run

app = Flask(__name__)


@dataclass
class Job:
    status: str = "idle"  # "idle" | "starting" | "running" | "done" | "failed"
    url: str = ""
    progress_log: Optional[str] = None
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


def _clip_display_url(shorts_dir: Optional[str], clip_url: Optional[str]) -> Optional[str]:
    if not clip_url:
        return None
    if clip_url.startswith("http://") or clip_url.startswith("https://"):
        return clip_url
    return f"/download/{os.path.basename(clip_url)}"


def _serialize_result(result: dict, shorts_dir: Optional[str]) -> dict:
    # Only "shorts" is ever rendered by the dashboard — the full pipeline
    # result also carries the whole transcript and every highlight candidate,
    # which /status would otherwise re-serialize and re-send on every poll.
    shorts = [
        {**s, "download_url": _clip_display_url(shorts_dir, s.get("clip_url"))}
        for s in result.get("shorts", [])
    ]
    return {"shorts": shorts}


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
    url = request.form.get("url", "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    try:
        kwargs = dict(
            mode=request.form.get("mode", "api"),
            num_clips=int(request.form.get("num_clips", 3)),
            aspect_ratio=request.form.get("aspect_ratio", "9:16"),
            download_format=request.form.get("format", "720"),
            language=(request.form.get("language") or "").strip() or None,
            captions=request.form.get("captions", "true") == "true",
            caption_fade_duration=float(request.form.get("caption_fade_duration", 0.3)),
            word_highlight=request.form.get("word_highlight", "true") == "true",
            hook_card=request.form.get("hook_card", "true") == "true",
            framing=request.form.get("framing", "locked"),
        )
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"invalid input: {e}"}), 400

    with _job_lock:
        if job.status in ("starting", "running"):
            return jsonify({"error": "a run is already in progress"}), 409
        job.status = "starting"
        job.url = url
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

    return jsonify({
        "status": current_status,
        "log": log_text,
        "offset": new_offset,
        "result": _serialize_result(result, shorts_dir) if result else None,
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
            if filename.startswith("Short-") and filename.endswith(".mp4"):
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
