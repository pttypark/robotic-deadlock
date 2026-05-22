"""Run the 12-AGV A* + FCFS cross shared-area baseline experiment."""

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


DEFAULT_POLICIES = "fcfs"

DECISION_FIELDNAMES = [
    "decision_step",
    "candidate_agent_id",
    "candidate_agent_label",
    "arrival_time",
    "waiting_steps",
    "queue_position",
    "queue_size",
    "exit_competition",
    "path_conflict_count",
    "approach_queue_length",
    "maneuver_priority",
    "route_length",
    "shared_zone_entry_step",
    "global_active_agents",
    "shared_zone_occupied",
    "selection_order_length",
    "score",
    "selected",
    "approach_north",
    "approach_south",
    "approach_west",
    "approach_east",
    "maneuver_left",
    "maneuver_straight",
    "maneuver_right",
    "exit_north",
    "exit_south",
    "exit_west",
    "exit_east",
    "policy_type",
    "seed",
    "task_priority",
    "yielded_count",
    "downstream_priority",
    "remaining_path_length",
    "same_direction_backlog",
    "run_index",
]


def main() -> None:
    """Parse CLI arguments and run the FCFS baseline."""

    parser = argparse.ArgumentParser(description="Run A* + FCFS cross shared-area baseline.")
    parser.add_argument("--robots-per-direction", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--spawn-gap-steps", type=int, default=2)
    parser.add_argument("--admission-window-steps", type=int, default=0)
    parser.add_argument("--output-csv", default=str(Path("outputs") / "fcfs_cross_experiment" / "results.csv"))
    parser.add_argument("--decision-csv", default="")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    rows = []
    decision_rows = []
    policies = _parse_strings(args.policies)
    for run_index in range(args.runs):
        seed = args.random_seed + run_index
        for policy_type in policies:
            metrics, decisions = run_single_experiment(
                robots_per_direction=args.robots_per_direction,
                max_steps=args.max_steps,
                random_seed=seed,
                corridor_length=args.corridor_length,
                west_exit_extension=args.west_exit_extension,
                spawn_gap_steps=args.spawn_gap_steps,
                admission_window_steps=args.admission_window_steps,
                policy_type=policy_type,
                scenario_name=args.scenario,
                debug=args.debug and args.runs == 1 and len(policies) == 1,
                print_summary=args.runs == 1,
            )
            row = _metrics_to_row(run_index, metrics)
            row["decision_csv_path"] = args.decision_csv
            rows.append(row)
            for decision in decisions:
                decision["run_index"] = run_index
                decision_rows.append(decision)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_results_csv(rows, output_path, append=args.append)
    if args.decision_csv:
        decision_path = Path(args.decision_csv)
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        _write_generic_csv(decision_rows, decision_path, append=args.append)

    if args.runs == 1:
        print(f"wrote {len(rows)} rows to {output_path.resolve()}")
    else:
        completed = sum(1 for row in rows if row["completed"] == row["robots"])
        avg_total_time = sum(row["total_time"] for row in rows) / len(rows)
        print(
            f"wrote {len(rows)} rows to {output_path.resolve()} "
            f"(completed_runs={completed}, avg_total_time={avg_total_time:.2f})"
        )


def run_single_experiment(
    robots_per_direction: int,
    max_steps: int,
    random_seed: int,
    corridor_length: int,
    west_exit_extension: int,
    spawn_gap_steps: int,
    admission_window_steps: int,
    policy_type: str,
    scenario_name: str,
    debug: bool = False,
    print_summary: bool = False,
) -> tuple[dict, list[dict]]:
    """Run one FCFS episode and return its metrics."""

    experiment = FCFSCrossExperiment(
        robots_per_direction=robots_per_direction,
        random_seed=random_seed,
        corridor_length=corridor_length,
        west_exit_extension=west_exit_extension,
        spawn_gap_steps=spawn_gap_steps,
        admission_window_steps=admission_window_steps,
        policy_type=policy_type,
        scenario_name=scenario_name,
    )
    if print_summary:
        experiment.print_layout_summary()

    while not experiment.is_done and experiment.step_count < max_steps:
        result = experiment.step()
        if debug:
            print(
                f"step={result['step']} spawned={result['spawned']} "
                f"admitted={result['admitted']} shared={result['shared_robot_id']} "
                f"queue={result['fcfs_queue']} completed={result['completed_count']}"
            )

    metrics = experiment.metrics()
    if print_summary:
        print("metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    return metrics, experiment.decision_log


def _metrics_to_row(run_index: int, metrics: dict) -> dict:
    return {
        "run_index": run_index,
        "seed": metrics["random_seed"],
        "scenario_name": metrics["scenario_name"],
        "policy": metrics["policy"],
        "policy_type": metrics["policy_type"],
        "decision_strategy": metrics["policy_type"],
        "layout": metrics["layout"],
        "robots": metrics["robots"],
        "robots_per_direction": metrics["robots_per_direction"],
        "corridor_length": metrics["corridor_length"],
        "west_exit_extension": metrics["west_exit_extension"],
        "spawn_gap_steps": metrics["spawn_gap_steps"],
        "spawn_offsets_by_direction": json.dumps(metrics["spawn_offsets_by_direction"], sort_keys=True),
        "admission_window_steps": metrics["admission_window_steps"],
        "completed": metrics["completed"],
        "total_time": metrics["total_time"],
        "total_travel_time": metrics["total_travel_time"],
        "avg_travel_time": metrics["avg_travel_time"],
        "total_wait_time": metrics["total_wait_time"],
        "avg_wait_time": metrics["avg_wait_time"],
        "max_wait_time": metrics["max_wait_time"],
        "shared_rule": metrics["shared_rule"],
        "decision_rule": metrics["decision_rule"],
        "finish_steps": json.dumps(metrics["finish_steps"], sort_keys=True),
        "end_assignments": json.dumps(metrics["end_assignments"], sort_keys=True),
    }


def _write_results_csv(rows: list[dict], output_path: Path, append: bool = False) -> None:
    fieldnames = [
        "run_index",
        "seed",
        "scenario_name",
        "policy",
        "policy_type",
        "decision_strategy",
        "layout",
        "robots",
        "robots_per_direction",
        "corridor_length",
        "west_exit_extension",
        "spawn_gap_steps",
        "spawn_offsets_by_direction",
        "admission_window_steps",
        "completed",
        "total_time",
        "total_travel_time",
        "avg_travel_time",
        "total_wait_time",
        "avg_wait_time",
        "max_wait_time",
        "shared_rule",
        "decision_rule",
        "decision_csv_path",
        "finish_steps",
        "end_assignments",
    ]
    write_header = not append or not output_path.exists() or output_path.stat().st_size == 0
    mode = "a" if append else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _write_generic_csv(rows: list[dict], output_path: Path, append: bool = False) -> None:
    if not rows:
        return
    discovered = {key for row in rows for key in row}
    fieldnames = [key for key in DECISION_FIELDNAMES if key in discovered]
    fieldnames.extend(sorted(discovered - set(fieldnames)))
    write_header = not append or not output_path.exists() or output_path.stat().st_size == 0
    mode = "a" if append else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _parse_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    main()
