"""Run the 24-AGV policy comparison and record smooth 3D videos."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AGV-24 FCFS/Basic/BO comparison and smooth 3D case videos."
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("final_output") / "agv24_policy_comparison"),
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--admission-window-steps", type=int, default=2)
    parser.add_argument("--shared-area-capacity", type=int, default=2)
    parser.add_argument("--max-planned-spawn-step", type=int, default=8)
    parser.add_argument("--min-agvs-per-direction", type=int, default=4)
    parser.add_argument("--max-agvs-per-direction", type=int, default=8)
    parser.add_argument("--scenario-seed", type=int, default=20260519)
    parser.add_argument("--run-seed-base", type=int, default=30000)
    parser.add_argument("--fixed-weight-set-count", type=int, default=50)
    parser.add_argument("--bo-trials", type=int, default=50)
    parser.add_argument("--bo-initial-local-trials", type=int, default=8)
    parser.add_argument("--bo-initial-random-trials", type=int, default=8)
    parser.add_argument("--bo-candidate-count", type=int, default=2048)
    parser.add_argument(
        "--bo-exploration-mode",
        choices=["stable", "balanced", "aggressive"],
        default="aggressive",
    )
    parser.add_argument("--skip-experiment", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-frames-per-step", type=int, default=8)
    parser.add_argument(
        "--video-selection",
        choices=["bo_better", "max_spread"],
        default="bo_better",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    experiment_script = PROJECT_DIR / "scripts" / "run_total_time_bo_policy_comparison.py"
    video_script = PROJECT_DIR / "scripts" / "record_total_time_bo_case_3d_videos.py"

    if not args.skip_experiment:
        _run(
            [
                sys.executable,
                str(experiment_script),
                "--total-agvs",
                "24",
                "--runs",
                str(args.runs),
                "--max-steps",
                str(args.max_steps),
                "--corridor-length",
                str(args.corridor_length),
                "--west-exit-extension",
                str(args.west_exit_extension),
                "--admission-window-steps",
                str(args.admission_window_steps),
                "--shared-area-capacity",
                str(args.shared_area_capacity),
                "--max-planned-spawn-step",
                str(args.max_planned_spawn_step),
                "--min-agvs-per-direction",
                str(args.min_agvs_per_direction),
                "--max-agvs-per-direction",
                str(args.max_agvs_per_direction),
                "--scenario-seed",
                str(args.scenario_seed),
                "--run-seed-base",
                str(args.run_seed_base),
                "--fixed-weight-set-count",
                str(args.fixed_weight_set_count),
                "--bo-trials",
                str(args.bo_trials),
                "--bo-initial-local-trials",
                str(args.bo_initial_local_trials),
                "--bo-initial-random-trials",
                str(args.bo_initial_random_trials),
                "--bo-candidate-count",
                str(args.bo_candidate_count),
                "--bo-exploration-mode",
                args.bo_exploration_mode,
                "--output-dir",
                str(output_dir),
            ]
        )

    if not args.skip_video:
        _run(
            [
                sys.executable,
                str(video_script),
                "--experiment-dir",
                str(output_dir),
                "--output-dir",
                str(output_dir / "videos_3d_smooth"),
                "--max-steps",
                str(args.max_steps),
                "--corridor-length",
                str(args.corridor_length),
                "--west-exit-extension",
                str(args.west_exit_extension),
                "--admission-window-steps",
                str(args.admission_window_steps),
                "--shared-area-capacity",
                str(args.shared_area_capacity),
                "--selection",
                args.video_selection,
                "--fps",
                str(args.video_fps),
                "--frames-per-step",
                str(args.video_frames_per_step),
            ]
        )

    print(f"done: {output_dir.resolve()}")


def _run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


if __name__ == "__main__":
    main()
