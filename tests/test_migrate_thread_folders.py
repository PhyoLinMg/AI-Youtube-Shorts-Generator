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
