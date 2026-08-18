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
