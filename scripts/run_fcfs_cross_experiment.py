"""Run the 12-AGV A* + FCFS cross shared-area baseline experiment."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from rware.fcfs_cross_simulation import FCFSCrossExperiment


def main() -> None:
    """Parse CLI arguments and run the FCFS baseline."""

    parser = argparse.ArgumentParser(description="Run A* + FCFS cross shared-area baseline.")
    parser.add_argument("--robots-per-direction", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    experiment = FCFSCrossExperiment(
        robots_per_direction=args.robots_per_direction,
        random_seed=args.random_seed,
    )
    experiment.print_layout_summary()

    while not experiment.is_done and experiment.step_count < args.max_steps:
        result = experiment.step()
        if args.debug:
            print(
                f"step={result['step']} spawned={result['spawned']} "
                f"admitted={result['admitted']} shared={result['shared_robot_id']} "
                f"queue={result['fcfs_queue']} completed={result['completed_count']}"
            )

    metrics = experiment.metrics()
    print("metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
