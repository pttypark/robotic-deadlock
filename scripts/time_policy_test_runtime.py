from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_total_time_bo_policy_comparison import (  # noqa: E402
    BO_FEATURES,
    POLICIES,
    RUN_FIELDNAMES,
    ScenarioSpec,
    _complete_heuristic_weights,
    _metrics_to_run_row,
    _run_single_policy,
    _write_csv,
)


DEFAULT_RESULT_DIR = PROJECT_DIR / "final_output" / "test_100_by_agv_fixed_weights"
DEFAULT_SCENARIO_FILE = DEFAULT_RESULT_DIR / "test_scenarios_300_all_agv_counts.csv"

BASIC_WEIGHTS = {
    "waiting": 5.985670,
    "maneuver": -3.799914,
    "exit_competition": -0.699648,
    "path_conflict": -0.544682,
    "approach_queue": 0.0,
    "remaining_path": 0.0,
    "same_direction_backlog": 0.0,
}

BO_WEIGHTS = {
    "waiting": 5.660327,
    "maneuver": -4.0,
    "exit_competition": -4.0,
    "path_conflict": -0.656567,
    "approach_queue": 0.0,
    "remaining_path": 0.0,
    "same_direction_backlog": 0.0,
}

PSO_WEIGHTS = {
    "waiting": 8.339856,
    "maneuver": -3.688935,
    "exit_competition": -2.609410,
    "path_conflict": -3.0,
    "approach_queue": 0.0,
    "remaining_path": 0.0,
    "same_direction_backlog": 0.0,
}

RUNTIME_FIELDNAMES = [
    "policy",
    "simulations",
    "scenario_count",
    "completed_runs",
    "avg_total_time",
    "elapsed_seconds",
    "elapsed_minutes",
    "seconds_per_simulation",
]

RUNTIME_BY_AGV_FIELDNAMES = [
    "total_agvs",
    *RUNTIME_FIELDNAMES,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure wall-clock runtime by policy on the fixed test scenarios."
    )
    parser.add_argument("--scenario-file", type=Path, default=DEFAULT_SCENARIO_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--admission-window-steps", type=int, default=2)
    parser.add_argument("--shared-area-capacity", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = _load_scenarios(args.scenario_file)
    policy_weights = {
        "fcfs": None,
        "fixed_heuristic": _complete_heuristic_weights(BASIC_WEIGHTS),
        "bo_heuristic": _complete_heuristic_weights(BO_WEIGHTS),
        "pso_heuristic": _complete_heuristic_weights(PSO_WEIGHTS),
    }

    timed_rows = []
    policy_elapsed = defaultdict(float)

    overall_start = time.perf_counter()
    for policy in POLICIES:
        for scenario in scenarios:
            start = time.perf_counter()
            metrics = _run_single_policy(
                scenario=scenario,
                policy=policy,
                weights=policy_weights[policy],
                max_steps=args.max_steps,
                corridor_length=args.corridor_length,
                west_exit_extension=args.west_exit_extension,
                admission_window_steps=args.admission_window_steps,
                shared_area_capacity=args.shared_area_capacity,
            )
            elapsed = time.perf_counter() - start
            policy_elapsed[policy] += elapsed
            row = _metrics_to_run_row(
                scenario=scenario,
                policy=policy,
                metrics=metrics,
                total_agvs=scenario.total_agvs,
            )
            row["elapsed_seconds"] = elapsed
            row["seconds_per_simulation"] = elapsed
            timed_rows.append(row)
    overall_elapsed = time.perf_counter() - overall_start

    runtime_rows = _runtime_summary(timed_rows, policy_elapsed)
    runtime_by_agv_rows = _runtime_by_agv_summary(timed_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        _format_rows(timed_rows),
        args.output_dir / "test_raw_runs_with_policy_runtime.csv",
        [*RUN_FIELDNAMES, "elapsed_seconds", "seconds_per_simulation"],
    )
    _write_csv(
        _format_rows(runtime_rows),
        args.output_dir / "test_policy_runtime_by_policy.csv",
        RUNTIME_FIELDNAMES,
    )
    _write_csv(
        _format_rows(runtime_by_agv_rows),
        args.output_dir / "test_policy_runtime_by_agv.csv",
        RUNTIME_BY_AGV_FIELDNAMES,
    )

    print(f"scenarios: {len(scenarios)}")
    print(f"total simulations: {len(timed_rows)}")
    print(f"overall elapsed seconds: {overall_elapsed:.4f}")
    for row in runtime_rows:
        print(
            f"{row['policy']}: elapsed={row['elapsed_seconds']:.4f}s, "
            f"seconds/sim={row['seconds_per_simulation']:.5f}, "
            f"avg_total_time={row['avg_total_time']:.2f}"
        )


def _load_scenarios(path: Path) -> list[ScenarioSpec]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    scenarios = []
    for row in rows:
        allocation = {
            "NORTH": int(row["north_agvs"]),
            "SOUTH": int(row["south_agvs"]),
            "WEST": int(row["west_agvs"]),
            "EAST": int(row["east_agvs"]),
        }
        scenarios.append(
            ScenarioSpec(
                run_index=int(row["run_index"]),
                seed=int(row["scenario_seed"]),
                allocation=allocation,
                spawn_plan=json.loads(row["spawn_plan_json"]),
                goal_plan=json.loads(row["goal_plan_json"]),
                scenario_case_id=int(row.get("scenario_case_id") or 0),
                scenario_case_name=row.get("scenario_case_name") or "random_research",
                scenario_case_label=row.get("scenario_case_label") or "Random research scenario",
                scenario_split=row.get("scenario_split") or "train_test",
                dominant_approach=row.get("dominant_approach") or "",
                dominant_approach_count=int(row.get("dominant_approach_count") or 0),
                dominant_exit=row.get("dominant_exit") or "",
                dominant_exit_count=int(row.get("dominant_exit_count") or 0),
                early_departures=int(row.get("early_departures") or 0),
                total_agvs=int(row.get("total_agvs") or sum(allocation.values())),
            )
        )
    return scenarios


def _runtime_summary(rows: list[dict], policy_elapsed: dict[str, float]) -> list[dict]:
    output = []
    for policy in POLICIES:
        policy_rows = [row for row in rows if row["policy"] == policy]
        values = [float(row["total_time"]) for row in policy_rows]
        elapsed = float(policy_elapsed[policy])
        output.append(
            {
                "policy": policy,
                "simulations": len(policy_rows),
                "scenario_count": len(policy_rows),
                "completed_runs": sum(
                    int(row["completed"]) == int(row["total_agvs"])
                    for row in policy_rows
                ),
                "avg_total_time": statistics.fmean(values),
                "elapsed_seconds": elapsed,
                "elapsed_minutes": elapsed / 60.0,
                "seconds_per_simulation": elapsed / len(policy_rows),
            }
        )
    return output


def _runtime_by_agv_summary(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["total_agvs"]), row["policy"])].append(row)

    output = []
    for (total_agvs, policy), group_rows in sorted(grouped.items()):
        values = [float(row["total_time"]) for row in group_rows]
        elapsed = sum(float(row["elapsed_seconds"]) for row in group_rows)
        output.append(
            {
                "total_agvs": total_agvs,
                "policy": policy,
                "simulations": len(group_rows),
                "scenario_count": len(group_rows),
                "completed_runs": sum(
                    int(row["completed"]) == int(row["total_agvs"])
                    for row in group_rows
                ),
                "avg_total_time": statistics.fmean(values),
                "elapsed_seconds": elapsed,
                "elapsed_minutes": elapsed / 60.0,
                "seconds_per_simulation": elapsed / len(group_rows),
            }
        )
    return output


def _format_rows(rows: list[dict]) -> list[dict]:
    return [_format_row(row) for row in rows]


def _format_row(row: dict) -> dict:
    formatted = {}
    for key, value in row.items():
        if key in {
            "avg_total_time",
            "total_time",
            "elapsed_seconds",
            "elapsed_minutes",
            "seconds_per_simulation",
        } and isinstance(value, (float, int)):
            formatted[key] = f"{float(value):.6f}" if key.startswith("elapsed") or key.startswith("seconds") else f"{float(value):.2f}"
        else:
            formatted[key] = value
    return formatted


if __name__ == "__main__":
    main()
