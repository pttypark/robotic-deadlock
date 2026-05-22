"""Package random-allocation experiment tables and representative videos."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from rware.fcfs_cross_simulation import FCFSCrossExperiment
from scripts.record_fcfs_policy_videos import draw_frame, write_video


POLICIES = ["fcfs", "heuristic"]
COMBINED_FIELDNAMES = [
    "set_id",
    "total_agvs",
    "north_agvs",
    "south_agvs",
    "west_agvs",
    "east_agvs",
    "layout",
    "corridor_length",
    "west_exit_extension",
    "spawn_gap_steps",
    "admission_window_steps",
    "runs_per_set",
    "fcfs_avg_total_time",
    "heuristic_avg_total_time",
    "delta_total_time",
    "total_time_improvement_pct",
    "fcfs_avg_total_wait_time",
    "heuristic_avg_total_wait_time",
    "delta_total_wait_time",
    "fcfs_avg_avg_wait_time",
    "heuristic_avg_avg_wait_time",
    "delta_avg_wait_time",
    "fcfs_avg_max_wait_time",
    "heuristic_avg_max_wait_time",
    "delta_max_wait_time",
    "fcfs_avg_utilization",
    "heuristic_avg_utilization",
    "delta_utilization",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one combined CSV and representative videos for random-set experiments."
    )
    parser.add_argument(
        "--input-dir",
        default=str(Path("final_output") / "fixed_layout_random_sets_15_optimized"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("final_output") / "random_set_deliverables"),
    )
    parser.add_argument("--run-seed-base", type=int, default=9000)
    parser.add_argument("--video-run-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    settings = _read_by_key(input_dir / "experiment_settings.csv", "set_id")
    comparisons = _read_by_key(input_dir / "policy_comparison_by_set.csv", "set_id")
    combined_rows = _combine_rows(settings, comparisons)
    combined_csv = output_dir / "agv_random_set_policy_comparison_combined.csv"
    _write_csv(combined_rows, combined_csv, COMBINED_FIELDNAMES)

    video_rows = []
    for row in combined_rows:
        allocation = {
            "NORTH": int(row["north_agvs"]),
            "SOUTH": int(row["south_agvs"]),
            "WEST": int(row["west_agvs"]),
            "EAST": int(row["east_agvs"]),
        }
        set_id = int(row["set_id"])
        seed = args.run_seed_base + set_id * 1000 + args.video_run_index
        for policy_type in POLICIES:
            video_path, metrics = _record_video(
                allocation=allocation,
                set_id=set_id,
                policy_type=policy_type,
                seed=seed,
                corridor_length=int(row["corridor_length"]),
                west_exit_extension=int(row["west_exit_extension"]),
                spawn_gap_steps=int(row["spawn_gap_steps"]),
                admission_window_steps=int(row["admission_window_steps"]),
                max_steps=args.max_steps,
                fps=args.fps,
                frame_stride=args.frame_stride,
                video_format=args.format,
                output_dir=video_dir,
            )
            video_rows.append(
                {
                    "set_id": set_id,
                    "policy_type": policy_type,
                    "seed": seed,
                    "video_path": str(video_path),
                    "total_time": metrics["total_time"],
                    "total_wait_time": metrics["total_wait_time"],
                    "avg_wait_time": metrics["avg_wait_time"],
                    "max_wait_time": metrics["max_wait_time"],
                    "utilization": metrics["utilization"],
                }
            )
            print(
                f"set={set_id} policy={policy_type} video={video_path} "
                f"total_time={metrics['total_time']}"
            )
    _write_csv(
        video_rows,
        output_dir / "representative_video_index.csv",
        [
            "set_id",
            "policy_type",
            "seed",
            "video_path",
            "total_time",
            "total_wait_time",
            "avg_wait_time",
            "max_wait_time",
            "utilization",
        ],
    )
    _write_readme(output_dir, combined_csv, video_rows)
    print(f"combined_csv: {combined_csv.resolve()}")
    print(f"videos: {video_dir.resolve()}")


def _record_video(
    allocation: dict[str, int],
    set_id: int,
    policy_type: str,
    seed: int,
    corridor_length: int,
    west_exit_extension: int,
    spawn_gap_steps: int,
    admission_window_steps: int,
    max_steps: int,
    fps: int,
    frame_stride: int,
    video_format: str,
    output_dir: Path,
) -> tuple[Path, dict]:
    experiment = FCFSCrossExperiment(
        robots_by_direction=allocation,
        random_seed=seed,
        corridor_length=corridor_length,
        west_exit_extension=west_exit_extension,
        spawn_gap_steps=spawn_gap_steps,
        admission_window_steps=admission_window_steps,
        policy_type=policy_type,
    )

    frames = [draw_frame(experiment, last_result=None)]
    last_result = None
    while not experiment.is_done and experiment.step_count < max_steps:
        last_result = experiment.step()
        if experiment.step_count % frame_stride == 0 or experiment.is_done:
            frames.append(draw_frame(experiment, last_result=last_result))

    metrics = experiment.metrics()
    path = output_dir / (
        f"set{set_id:02d}_{policy_type}_robots{metrics['robots']}_"
        f"seed{seed}.{video_format}"
    )
    write_video(path, frames, fps=fps, video_format=video_format)
    return path, metrics


def _read_by_key(path: Path, key: str) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def _combine_rows(settings: dict[str, dict], comparisons: dict[str, dict]) -> list[dict]:
    rows = []
    for set_id in sorted(settings, key=lambda item: int(item)):
        row = {**settings[set_id], **comparisons[set_id]}
        rows.append(row)
    return rows


def _write_csv(rows: list[dict], path: Path, preferred_fieldnames: list[str]) -> None:
    discovered = {key for row in rows for key in row}
    fieldnames = [field for field in preferred_fieldnames if field in discovered]
    fieldnames.extend(sorted(discovered - set(fieldnames)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(output_dir: Path, combined_csv: Path, video_rows: list[dict]) -> None:
    lines = [
        "# AGV Random Set Deliverables",
        "",
        "## Files",
        "",
        f"- `{combined_csv.name}`: combined experiment setting and policy comparison table",
        "- `representative_video_index.csv`: video metadata",
        "- `videos/`: representative FCFS and Heuristic videos, one pair per set",
        "",
        "## Videos",
        "",
        "| set | policy | seed | total_time | video |",
        "|---:|---|---:|---:|---|",
    ]
    for row in video_rows:
        lines.append(
            f"| {row['set_id']} | {row['policy_type']} | {row['seed']} | "
            f"{row['total_time']} | `{Path(row['video_path']).name}` |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
