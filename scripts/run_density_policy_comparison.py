"""Run policy comparisons while varying AGV density per approach direction."""

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


DEFAULT_POLICIES = "fcfs,heuristic"
DEFAULT_DENSITIES = "2,3,4,5"

METRIC_FIELDNAMES = [
    "cars_per_direction",
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

SUMMARY_FIELDNAMES = [
    "cars_per_direction",
    "robots",
    "scenario_name",
    "layout",
    "fcfs_total_time",
    "heuristic_total_time",
    "delta_total_time",
    "total_time_improvement_pct",
    "fcfs_total_travel_time",
    "heuristic_total_travel_time",
    "delta_total_travel_time",
    "fcfs_avg_travel_time",
    "heuristic_avg_travel_time",
    "delta_avg_travel_time",
    "fcfs_total_wait_time",
    "heuristic_total_wait_time",
    "delta_total_wait_time",
    "fcfs_avg_wait_time",
    "heuristic_avg_wait_time",
    "delta_avg_wait_time",
    "fcfs_max_wait_time",
    "heuristic_max_wait_time",
]

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
    "cars_per_direction",
    "robots",
    "policy_type",
    "decision_strategy",
    "scenario_name",
    "seed",
    "run_index",
    "task_priority",
    "yielded_count",
    "downstream_priority",
    "remaining_path_length",
    "same_direction_backlog",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare shared-area policies over multiple AGV densities."
    )
    parser.add_argument("--robots-per-direction-values", default=DEFAULT_DENSITIES)
    parser.add_argument("--policies", default=DEFAULT_POLICIES)
    parser.add_argument("--scenario", default="density_stress")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--spawn-gap-steps", type=int, default=1)
    parser.add_argument("--admission-window-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument(
        "--output-dir",
        default=str(Path("final_output") / "density_stress"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    densities = _parse_ints(args.robots_per_direction_values)
    policies = _parse_strings(args.policies)
    metric_rows: list[dict] = []
    decision_rows: list[dict] = []

    for run_index in range(args.runs):
        seed = args.random_seed + run_index
        for cars_per_direction in densities:
            for policy_type in policies:
                metrics, decisions = _run_episode(
                    cars_per_direction=cars_per_direction,
                    max_steps=args.max_steps,
                    random_seed=seed,
                    corridor_length=args.corridor_length,
                    west_exit_extension=args.west_exit_extension,
                    spawn_gap_steps=args.spawn_gap_steps,
                    admission_window_steps=args.admission_window_steps,
                    scenario_name=args.scenario,
                    policy_type=policy_type,
                )
                decision_path = (
                    output_dir
                    / f"{policy_type}_decision_data"
                    / f"{policy_type}_decision_dataset_cars{cars_per_direction}_run{run_index + 1}.csv"
                )
                row = _metrics_to_row(
                    run_index=run_index,
                    cars_per_direction=cars_per_direction,
                    metrics=metrics,
                    decision_path=decision_path,
                )
                metric_rows.append(row)

                policy_decisions = []
                for decision in decisions:
                    enriched = dict(decision)
                    enriched["cars_per_direction"] = cars_per_direction
                    enriched["robots"] = metrics["robots"]
                    enriched["decision_strategy"] = policy_type
                    enriched["scenario_name"] = metrics["scenario_name"]
                    enriched["run_index"] = run_index
                    policy_decisions.append(enriched)
                    decision_rows.append(enriched)
                _write_csv(policy_decisions, decision_path, DECISION_FIELDNAMES)

                print(
                    f"cars={cars_per_direction} policy={policy_type} "
                    f"total_time={metrics['total_time']} "
                    f"avg_travel={metrics['avg_travel_time']:.3f} "
                    f"total_wait={metrics['total_wait_time']}"
                )

    summary_rows = _summarize(metric_rows)
    _write_csv(metric_rows, output_dir / "density_policy_comparison.csv", METRIC_FIELDNAMES)
    _write_csv(summary_rows, output_dir / "density_policy_summary.csv", SUMMARY_FIELDNAMES)
    _write_csv(decision_rows, output_dir / "decision_features.csv", DECISION_FIELDNAMES)
    print(f"metrics: {(output_dir / 'density_policy_comparison.csv').resolve()}")
    print(f"summary: {(output_dir / 'density_policy_summary.csv').resolve()}")
    print(f"features: {(output_dir / 'decision_features.csv').resolve()}")


def _run_episode(
    cars_per_direction: int,
    max_steps: int,
    random_seed: int,
    corridor_length: int,
    west_exit_extension: int,
    spawn_gap_steps: int,
    admission_window_steps: int,
    scenario_name: str,
    policy_type: str,
) -> tuple[dict, list[dict]]:
    experiment = FCFSCrossExperiment(
        robots_per_direction=cars_per_direction,
        random_seed=random_seed,
        corridor_length=corridor_length,
        west_exit_extension=west_exit_extension,
        spawn_gap_steps=spawn_gap_steps,
        admission_window_steps=admission_window_steps,
        policy_type=policy_type,
        scenario_name=scenario_name,
    )
    metrics = experiment.run(max_steps=max_steps)
    return metrics, experiment.decision_log


def _metrics_to_row(
    run_index: int,
    cars_per_direction: int,
    metrics: dict,
    decision_path: Path,
) -> dict:
    return {
        "cars_per_direction": cars_per_direction,
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
        "spawn_offsets_by_direction": json.dumps(
            metrics["spawn_offsets_by_direction"],
            sort_keys=True,
        ),
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
        "decision_csv_path": str(decision_path),
        "finish_steps": json.dumps(metrics["finish_steps"], sort_keys=True),
        "end_assignments": json.dumps(metrics["end_assignments"], sort_keys=True),
    }


def _summarize(metric_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, int], dict[str, dict]] = {}
    for row in metric_rows:
        key = (row["run_index"], row["cars_per_direction"])
        grouped.setdefault(key, {})[row["policy_type"]] = row

    summary_rows = []
    for (_run_index, cars_per_direction), policies in sorted(grouped.items()):
        if "fcfs" not in policies or "heuristic" not in policies:
            continue
        fcfs = policies["fcfs"]
        heuristic = policies["heuristic"]
        delta_total_time = fcfs["total_time"] - heuristic["total_time"]
        summary_rows.append(
            {
                "cars_per_direction": cars_per_direction,
                "robots": fcfs["robots"],
                "scenario_name": fcfs["scenario_name"],
                "layout": fcfs["layout"],
                "fcfs_total_time": fcfs["total_time"],
                "heuristic_total_time": heuristic["total_time"],
                "delta_total_time": delta_total_time,
                "total_time_improvement_pct": (
                    delta_total_time / fcfs["total_time"] * 100.0
                    if fcfs["total_time"]
                    else 0.0
                ),
                "fcfs_total_travel_time": fcfs["total_travel_time"],
                "heuristic_total_travel_time": heuristic["total_travel_time"],
                "delta_total_travel_time": fcfs["total_travel_time"]
                - heuristic["total_travel_time"],
                "fcfs_avg_travel_time": fcfs["avg_travel_time"],
                "heuristic_avg_travel_time": heuristic["avg_travel_time"],
                "delta_avg_travel_time": fcfs["avg_travel_time"]
                - heuristic["avg_travel_time"],
                "fcfs_total_wait_time": fcfs["total_wait_time"],
                "heuristic_total_wait_time": heuristic["total_wait_time"],
                "delta_total_wait_time": fcfs["total_wait_time"]
                - heuristic["total_wait_time"],
                "fcfs_avg_wait_time": fcfs["avg_wait_time"],
                "heuristic_avg_wait_time": heuristic["avg_wait_time"],
                "delta_avg_wait_time": fcfs["avg_wait_time"]
                - heuristic["avg_wait_time"],
                "fcfs_max_wait_time": fcfs["max_wait_time"],
                "heuristic_max_wait_time": heuristic["max_wait_time"],
            }
        )
    return summary_rows


def _write_csv(rows: list[dict], output_path: Path, preferred_fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    discovered = {key for row in rows for key in row}
    fieldnames = [key for key in preferred_fieldnames if key in discovered]
    fieldnames.extend(sorted(discovered - set(fieldnames)))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    main()
