"""Record FCFS, fixed heuristic, and BO heuristic videos for one comparison case."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from rware.fcfs_cross_simulation import FCFSCrossExperiment
from scripts.record_fcfs_policy_videos import (
    POLICY_LABELS,
    draw_frame,
    write_video,
)
from scripts.run_total_time_bo_policy_comparison import (
    FIXED_HEURISTIC_WEIGHTS,
    TOTAL_AGVS,
)


POLICY_VIDEO_LABELS = {
    "fcfs": "FCFS",
    "fixed_heuristic": "Basic Heuristic",
    "bo_heuristic": "BO Heuristic",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record one high-gap case from total_time BO comparison outputs."
    )
    parser.add_argument(
        "--experiment-dir",
        default=str(Path("final_output") / "total_time_bo_policy_comparison_stress"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("final_output") / "total_time_bo_policy_comparison_stress" / "videos"),
    )
    parser.add_argument("--run-index", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--admission-window-steps", type=int, default=2)
    parser.add_argument("--shared-area-capacity", type=int, default=None)
    parser.add_argument("--raw-heuristic-features", action="store_true")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = _read_csv(experiment_dir / "experiment_settings.csv")
    comparisons = _read_csv(experiment_dir / "trial_comparison.csv")
    weight_metadata = json.loads(
        (experiment_dir / "bo_best_weights.json").read_text(encoding="utf-8")
    )
    bo_weights = weight_metadata["best_bo_weights"]
    fixed_weights = weight_metadata.get(
        "fixed_greedy_best_weights",
        weight_metadata.get("fixed_heuristic_weights", FIXED_HEURISTIC_WEIGHTS),
    )
    shared_area_capacity = (
        args.shared_area_capacity
        if args.shared_area_capacity is not None
        else int(weight_metadata.get("shared_area_capacity", 2))
    )

    selected = _select_case(comparisons, args.run_index)
    setting = next(row for row in settings if int(row["run_index"]) == int(selected["run_index"]))
    scenario = _scenario_from_row(setting)

    video_rows = []
    for policy in ("fcfs", "fixed_heuristic", "bo_heuristic"):
        path, metrics = _record_video(
            policy=policy,
            scenario=scenario,
            fixed_weights=fixed_weights,
            bo_weights=bo_weights,
            max_steps=args.max_steps,
            corridor_length=args.corridor_length,
            west_exit_extension=args.west_exit_extension,
            admission_window_steps=args.admission_window_steps,
            shared_area_capacity=shared_area_capacity,
            normalize_heuristic_features=not args.raw_heuristic_features,
            fps=args.fps,
            frame_stride=args.frame_stride,
            video_format=args.format,
            output_dir=output_dir,
        )
        video_rows.append(
            {
                "run_index": selected["run_index"],
                "policy": policy,
                "total_time": metrics["total_time"],
                "completed": metrics["completed"],
                "video_path": str(path),
            }
        )
        print(f"{policy}: {path} total_time={metrics['total_time']}")

    _write_csv(video_rows, output_dir / "selected_video_case.csv")
    _write_readme(
        output_dir,
        selected,
        setting,
        video_rows,
        fixed_weights,
        bo_weights,
        shared_area_capacity,
    )
    print(f"selected run_index={selected['run_index']}")
    print(f"videos: {output_dir.resolve()}")


def _record_video(
    policy: str,
    scenario: dict,
    fixed_weights: dict[str, float],
    bo_weights: dict[str, float],
    max_steps: int,
    corridor_length: int,
    west_exit_extension: int,
    admission_window_steps: int,
    shared_area_capacity: int,
    normalize_heuristic_features: bool,
    fps: int,
    frame_stride: int,
    video_format: str,
    output_dir: Path,
) -> tuple[Path, dict]:
    if policy == "fcfs":
        policy_type = "fcfs"
        weights = None
    elif policy == "fixed_heuristic":
        policy_type = "heuristic"
        weights = fixed_weights
    else:
        policy_type = "heuristic"
        weights = bo_weights

    experiment = FCFSCrossExperiment(
        robots_by_direction=scenario["allocation"],
        random_seed=scenario["seed"],
        corridor_length=corridor_length,
        west_exit_extension=west_exit_extension,
        spawn_gap_steps=0,
        spawn_plan_by_direction=scenario["spawn_plan"],
        admission_window_steps=admission_window_steps,
        goal_plan_by_direction=scenario["goal_plan"],
        shared_area_capacity=shared_area_capacity,
        normalize_heuristic_features=normalize_heuristic_features,
        policy_type=policy_type,
        heuristic_weights=weights,
    )

    previous_heuristic_label = POLICY_LABELS.get("heuristic", "Heuristic")
    if policy_type == "heuristic":
        POLICY_LABELS["heuristic"] = POLICY_VIDEO_LABELS[policy]
    frames = [draw_frame(experiment, last_result=None)]
    last_result = None
    while not experiment.is_done and experiment.step_count < max_steps:
        last_result = experiment.step()
        if experiment.step_count % frame_stride == 0 or experiment.is_done:
            frames.append(draw_frame(experiment, last_result=last_result))
    if policy_type == "heuristic":
        POLICY_LABELS["heuristic"] = previous_heuristic_label

    metrics = experiment.metrics()
    path = output_dir / (
        f"run{scenario['run_index']:03d}_{policy}_total{metrics['total_time']}."
        f"{video_format}"
    )
    write_video(path, frames, fps=fps, video_format=video_format)
    return path, metrics


def _select_case(rows: list[dict], run_index: int) -> dict:
    if run_index >= 0:
        return next(row for row in rows if int(row["run_index"]) == run_index)
    strict_rows = [
        row
        for row in rows
        if int(row["fcfs_total_time"])
        > int(row["fixed_heuristic_total_time"])
        > int(row["bo_heuristic_total_time"])
    ]
    candidates = strict_rows or rows
    return max(
        candidates,
        key=lambda row: (
            int(row["fcfs_total_time"]) - int(row["bo_heuristic_total_time"]),
            int(row["fixed_heuristic_total_time"]) - int(row["bo_heuristic_total_time"]),
        ),
    )


def _scenario_from_row(row: dict) -> dict:
    return {
        "run_index": int(row["run_index"]),
        "seed": int(row["scenario_seed"]),
        "allocation": {
            "NORTH": int(row["north_agvs"]),
            "SOUTH": int(row["south_agvs"]),
            "WEST": int(row["west_agvs"]),
            "EAST": int(row["east_agvs"]),
        },
        "spawn_plan": json.loads(row["spawn_plan_json"]),
        "goal_plan": json.loads(row["goal_plan_json"]),
    }


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(
    output_dir: Path,
    selected: dict,
    setting: dict,
    video_rows: list[dict],
    fixed_weights: dict,
    bo_weights: dict,
    shared_area_capacity: int,
) -> None:
    lines = [
        "# Selected Total-Time Video Case",
        "",
        f"- run_index: `{selected['run_index']}`",
        f"- allocation: NORTH={setting['north_agvs']}, SOUTH={setting['south_agvs']}, "
        f"WEST={setting['west_agvs']}, EAST={setting['east_agvs']}",
        f"- spawn_plan: `{setting['spawn_plan_json']}`",
        f"- goal_plan: `{setting['goal_plan_json']}`",
        f"- shared_area_capacity: `{shared_area_capacity}`",
        f"- Fixed weights: `{json.dumps(fixed_weights, sort_keys=True)}`",
        f"- BO weights: `{json.dumps(bo_weights, sort_keys=True)}`",
        "",
        "| policy | total_time | video |",
        "|---|---:|---|",
    ]
    for row in video_rows:
        lines.append(
            f"| {row['policy']} | {row['total_time']} | `{Path(row['video_path']).name}` |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
