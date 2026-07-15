"""Flask dashboard: submit a YouTube URL, watch progress, grab the shorts.

Single-user, local tool — at most one pipeline run at a time. State lives in
the module-level `job` object, guarded by `_job_lock` since a background
thread and Flask's request threads touch it concurrently. This one-run-at-a-
time constraint is load-bearing: `capture_progress_log` (run_output.py) swaps
sys.stdout/sys.stderr process-globally, not per-thread, so two concurrent
runs would interleave each other's progress logs.
"""
import os
import threading
from dataclasses import dataclass
from typing import Optional

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from .pipeline import generate_shorts
from .run_output import resolve_output_dir

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
            paths=paths,
        )
        with _job_lock:
            job.result = result
            job.status = "done"
    except Exception as e:
        with _job_lock:
            job.error = str(e)
            job.status = "failed"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def start_run():
    url = request.form.get("url", "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400

    with _job_lock:
        if job.status in ("starting", "running"):
            return jsonify({"error": "a run is already in progress"}), 409
        job.status = "starting"
        job.url = url
        job.progress_log = None
        job.shorts_dir = None
        job.result = None
        job.error = None

    kwargs = dict(
        mode=request.form.get("mode", "api"),
        num_clips=int(request.form.get("num_clips", 3)),
        aspect_ratio=request.form.get("aspect_ratio", "9:16"),
        download_format=request.form.get("format", "720"),
        language=(request.form.get("language") or "").strip() or None,
        captions=request.form.get("captions", "true") == "true",
        caption_fade_duration=float(request.form.get("caption_fade_duration", 0.3)),
        word_highlight=request.form.get("word_highlight", "true") == "true",
        framing=request.form.get("framing", "locked"),
    )
    threading.Thread(target=_run_job, args=(url,), kwargs=kwargs, daemon=True).start()
    return jsonify({"status": "starting"}), 202
