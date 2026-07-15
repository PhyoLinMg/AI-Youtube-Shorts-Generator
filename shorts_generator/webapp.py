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


@app.route("/")
def index():
    return render_template("index.html")
