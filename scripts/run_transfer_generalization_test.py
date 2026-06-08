from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rware.agv_layouts import build_fcfs_double_cross_shared_area_layout  # noqa: E402
from rware.fcfs_cross_simulation import FCFSCrossExperiment  # noqa: E402
from scripts.run_total_time_bo_policy_comparison import _complete_heuristic_weights  # noqa: E402


DEFAULT_SCENARIO_FILE = (
    PROJECT_DIR
    / "final_output"
    / "transfer_generalization_test_scenarios"
    / "transfer_test_scenarios_all_layouts.csv"
)
DEFAULT_CHECKPOINT_FILE = (
    PROJECT_DIR
    / "final_output"
    / "transfer_generalization_train_400_time_limited"
    / "train_best_checkpoints.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "final_output" / "transfer_generalization_test_results"

POLICY_LABELS = (
    "fcfs",
    "basic_5min",
    "basic_30min",
    "bo_5min",
    "bo_30min",
    "pso_5min",
    "pso_30min",
)

RAW_FIELDNAMES = [
    "layout_key",
    "policy",
    "source_train_policy",
    "checkpoint_label",
    "run_index",
    "scenario_seed",
    "scenario_case_id",
    "scenario_case_name",
    "scenario_case_label",
    "combined_features",
    "total_agvs",
    "completed",
    "total_time",
    "shared_area_capacity",
    "dominant_approach",
    "dominant_approach_count",
    "dominant_exit",
    "dominant_exit_count",
    "early_departures",
    "direction_count_json",
    "weights_json",
    "waiting",
    "maneuver",
    "exit_competition",
    "path_conflict",
]

SUMMARY_FIELDNAMES = [
    "layout_key",
    "total_agvs",
    "scenario_case_id",
    "scenario_case_name",
    "policy",
    "runs",
    "completed_runs",
    "avg_total_time",
    "median_total_time",
    "std_total_time",
    "min_total_time",
    "max_total_time",
    "p10_total_time",
    "p90_total_time",
]

RUNTIME_FIELDNAMES = [
    "scope",
    "layout_key",
    "policy",
    "scenario_count",
    "simulation_runs",
    "elapsed_seconds",
    "elapsed_minutes",
    "seconds_per_simulation",
]


@dataclass(frozen=True)
class TestScenario:
    layout_key: str
    run_index: int
    seed: int
    total_agvs: int
    case_id: int
    case_name: str
    case_label: str
    combined_features: str
    allocation: dict[str, int]
    spawn_plan: dict[str, list[int]]
    goal_plan: dict[str, list[str]]
    dominant_approach: str
    dominant_approach_count: int
    dominant_exit: str
    dominant_exit_count: int
    early_departures: int
    direction_count_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply FCFS plus Basic/BO/PSO 5-min and 30-min trained weights to "
            "single-cross and double-cross transfer test scenarios."
        )
    )
    parser.add_argument("--scenario-file", type=Path, default=DEFAULT_SCENARIO_FILE)
    parser.add_argument("--checkpoint-file", type=Path, default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--admission-window-steps", type=int, default=2)
    parser.add_argument("--shared-area-capacity", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = _load_scenarios(args.scenario_file)
    weights_by_policy = _load_checkpoint_weights(args.checkpoint_file)

    start_all = time.monotonic()
    raw_rows: list[dict] = []
    runtime_rows: list[dict] = []

    for policy in POLICY_LABELS:
        policy_start = time.monotonic()
        policy_rows = []
        weights = weights_by_policy.get(policy)
        for scenario in scenarios:
            metrics = _run_policy(
                scenario=scenario,
                policy=policy,
                weights=weights,
                args=args,
            )
            row = _raw_row(scenario, policy, weights, metrics)
            raw_rows.append(row)
            policy_rows.append(row)
        elapsed = time.monotonic() - policy_start
        runtime_rows.append(
            {
                "scope": "policy",
                "layout_key": "all",
                "policy": policy,
                "scenario_count": len(scenarios),
                "simulation_runs": len(policy_rows),
                "elapsed_seconds": elapsed,
                "elapsed_minutes": elapsed / 60.0,
                "seconds_per_simulation": elapsed / max(1, len(policy_rows)),
            }
        )

    elapsed_all = time.monotonic() - start_all
    runtime_rows.append(
        {
            "scope": "all",
            "layout_key": "all",
            "policy": "all",
            "scenario_count": len(scenarios),
            "simulation_runs": len(raw_rows),
            "elapsed_seconds": elapsed_all,
            "elapsed_minutes": elapsed_all / 60.0,
            "seconds_per_simulation": elapsed_all / max(1, len(raw_rows)),
        }
    )

    comparison_rows = _comparison_rows(raw_rows)
    summary_overall = _summary_rows(raw_rows, ("layout_key", "policy"))
    summary_by_agv = _summary_rows(raw_rows, ("layout_key", "total_agvs", "policy"))
    summary_by_case = _summary_rows(raw_rows, ("layout_key", "scenario_case_id", "scenario_case_name", "policy"))
    summary_by_case_agv = _summary_rows(
        raw_rows,
        ("layout_key", "total_agvs", "scenario_case_id", "scenario_case_name", "policy"),
    )

    _write_csv(args.output_dir / "transfer_test_raw_runs.csv", _format_rows(raw_rows), RAW_FIELDNAMES)
    _write_csv(args.output_dir / "transfer_test_trial_comparison.csv", _format_rows(comparison_rows), _comparison_fieldnames())
    _write_csv(args.output_dir / "transfer_test_summary_overall.csv", _format_rows(summary_overall), _summary_fieldnames(("layout_key", "policy")))
    _write_csv(args.output_dir / "transfer_test_summary_by_agv.csv", _format_rows(summary_by_agv), _summary_fieldnames(("layout_key", "total_agvs", "policy")))
    _write_csv(args.output_dir / "transfer_test_summary_by_scenario.csv", _format_rows(summary_by_case), _summary_fieldnames(("layout_key", "scenario_case_id", "scenario_case_name", "policy")))
    _write_csv(args.output_dir / "transfer_test_summary_by_scenario_agv.csv", _format_rows(summary_by_case_agv), _summary_fieldnames(("layout_key", "total_agvs", "scenario_case_id", "scenario_case_name", "policy")))
    _write_csv(args.output_dir / "transfer_test_runtime_summary.csv", _format_rows(runtime_rows), RUNTIME_FIELDNAMES)
    _write_settings(args.output_dir, args, weights_by_policy, len(scenarios), len(raw_rows))

    print(f"wrote output dir: {args.output_dir.resolve()}")
    print(f"scenario rows: {len(scenarios)}")
    print(f"simulation runs: {len(raw_rows)}")
    print(f"elapsed minutes: {elapsed_all / 60.0:.2f}")


def _run_policy(
    scenario: TestScenario,
    policy: str,
    weights: dict[str, float] | None,
    args: argparse.Namespace,
) -> dict:
    layout = None
    if scenario.layout_key == "double_cross":
        layout = build_fcfs_double_cross_shared_area_layout(corridor_length=args.corridor_length)
    experiment = FCFSCrossExperiment(
        layout=layout,
        robots_by_direction=scenario.allocation,
        random_seed=scenario.seed,
        corridor_length=args.corridor_length,
        west_exit_extension=args.west_exit_extension,
        spawn_gap_steps=0,
        spawn_plan_by_direction=scenario.spawn_plan,
        admission_window_steps=args.admission_window_steps,
        shared_area_capacity=args.shared_area_capacity,
        normalize_heuristic_features=True,
        goal_plan_by_direction=scenario.goal_plan,
        policy_type="fcfs" if policy == "fcfs" else "heuristic",
        heuristic_weights=weights if policy != "fcfs" else None,
    )
    return experiment.run(max_steps=args.max_steps)


def _load_checkpoint_weights(path: Path) -> dict[str, dict[str, float]]:
    rows = _read_csv(path)
    mapping = {
        ("basic_heuristic", "5min"): "basic_5min",
        ("basic_heuristic", "30min"): "basic_30min",
        ("bo_heuristic", "5min"): "bo_5min",
        ("bo_heuristic", "30min"): "bo_30min",
        ("pso_heuristic", "5min"): "pso_5min",
        ("pso_heuristic", "30min"): "pso_30min",
    }
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        key = mapping.get((row["policy"], row["checkpoint_label"]))
        if not key:
            continue
        weights = json.loads(row["best_weights_json"])
        output[key] = _complete_heuristic_weights(weights)
    missing = [policy for policy in POLICY_LABELS if policy != "fcfs" and policy not in output]
    if missing:
        raise ValueError(f"checkpoint file is missing weights for: {missing}")
    return output


def _load_scenarios(path: Path) -> list[TestScenario]:
    return [_scenario_from_row(row) for row in _read_csv(path)]


def _scenario_from_row(row: dict[str, str]) -> TestScenario:
    layout_key = row["layout_key"]
    if layout_key == "single_cross":
        allocation = {
            "NORTH": int(row["north_agvs"]),
            "SOUTH": int(row["south_agvs"]),
            "WEST": int(row["west_agvs"]),
            "EAST": int(row["east_agvs"]),
        }
    elif layout_key == "double_cross":
        allocation = {
            "L_NORTH": int(row["l_north_agvs"]),
            "L_SOUTH": int(row["l_south_agvs"]),
            "WEST": int(row["west_agvs"]),
            "R_NORTH": int(row["r_north_agvs"]),
            "R_SOUTH": int(row["r_south_agvs"]),
            "EAST": int(row["east_agvs"]),
        }
    else:
        raise ValueError(f"unknown layout_key: {layout_key}")
    return TestScenario(
        layout_key=layout_key,
        run_index=int(row["run_index"]),
        seed=int(row["scenario_seed"]),
        total_agvs=int(row["total_agvs"]),
        case_id=int(row["scenario_case_id"]),
        case_name=row["scenario_case_name"],
        case_label=row["scenario_case_label"],
        combined_features=row.get("combined_features", ""),
        allocation=allocation,
        spawn_plan=json.loads(row["spawn_plan_json"]),
        goal_plan=json.loads(row["goal_plan_json"]),
        dominant_approach=row.get("dominant_approach", ""),
        dominant_approach_count=int(row.get("dominant_approach_count") or 0),
        dominant_exit=row.get("dominant_exit", ""),
        dominant_exit_count=int(row.get("dominant_exit_count") or 0),
        early_departures=int(row.get("early_departures") or 0),
        direction_count_json=row.get("direction_count_json", json.dumps(allocation, sort_keys=True)),
    )


def _raw_row(
    scenario: TestScenario,
    policy: str,
    weights: dict[str, float] | None,
    metrics: dict,
) -> dict:
    source_policy, checkpoint_label = _policy_source(policy)
    row = {
        "layout_key": scenario.layout_key,
        "policy": policy,
        "source_train_policy": source_policy,
        "checkpoint_label": checkpoint_label,
        "run_index": scenario.run_index,
        "scenario_seed": scenario.seed,
        "scenario_case_id": scenario.case_id,
        "scenario_case_name": scenario.case_name,
        "scenario_case_label": scenario.case_label,
        "combined_features": scenario.combined_features,
        "total_agvs": scenario.total_agvs,
        "completed": metrics["completed"],
        "total_time": metrics["total_time"],
        "shared_area_capacity": metrics["shared_area_capacity"],
        "dominant_approach": scenario.dominant_approach,
        "dominant_approach_count": scenario.dominant_approach_count,
        "dominant_exit": scenario.dominant_exit,
        "dominant_exit_count": scenario.dominant_exit_count,
        "early_departures": scenario.early_departures,
        "direction_count_json": scenario.direction_count_json,
        "weights_json": "" if weights is None else json.dumps(weights, sort_keys=True),
    }
    for feature in ("waiting", "maneuver", "exit_competition", "path_conflict"):
        row[feature] = "" if weights is None else weights.get(feature, 0.0)
    return row


def _comparison_rows(raw_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(dict)
    for row in raw_rows:
        key = (row["layout_key"], int(row["run_index"]))
        grouped[key][row["policy"]] = row

    output = []
    for (layout_key, run_index), policies in sorted(grouped.items()):
        if any(policy not in policies for policy in POLICY_LABELS):
            continue
        any_row = next(iter(policies.values()))
        times = {policy: int(policies[policy]["total_time"]) for policy in POLICY_LABELS}
        best_time = min(times.values())
        best_policies = [policy for policy, total_time in times.items() if total_time == best_time]
        row = {
            "layout_key": layout_key,
            "run_index": run_index,
            "scenario_seed": any_row["scenario_seed"],
            "total_agvs": any_row["total_agvs"],
            "scenario_case_id": any_row["scenario_case_id"],
            "scenario_case_name": any_row["scenario_case_name"],
            "combined_features": any_row["combined_features"],
            "best_policy": ",".join(best_policies),
            "best_total_time": best_time,
            "num_best_policies": len(best_policies),
        }
        for policy in POLICY_LABELS:
            row[f"{policy}_total_time"] = times[policy]
        output.append(row)
    return output


def _summary_rows(raw_rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        values = [int(row["total_time"]) for row in rows]
        summary = {group_key: key[index] for index, group_key in enumerate(group_keys)}
        summary.update(
            {
                "runs": len(rows),
                "completed_runs": sum(int(row["completed"]) == int(row["total_agvs"]) for row in rows),
                "avg_total_time": statistics.fmean(values),
                "median_total_time": statistics.median(values),
                "std_total_time": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min_total_time": min(values),
                "max_total_time": max(values),
                "p10_total_time": _percentile(values, 10.0),
                "p90_total_time": _percentile(values, 90.0),
            }
        )
        output.append(summary)
    return output


def _summary_fieldnames(group_keys: tuple[str, ...]) -> list[str]:
    return [
        *group_keys,
        "runs",
        "completed_runs",
        "avg_total_time",
        "median_total_time",
        "std_total_time",
        "min_total_time",
        "max_total_time",
        "p10_total_time",
        "p90_total_time",
    ]


def _comparison_fieldnames() -> list[str]:
    return [
        "layout_key",
        "run_index",
        "scenario_seed",
        "total_agvs",
        "scenario_case_id",
        "scenario_case_name",
        "combined_features",
        *[f"{policy}_total_time" for policy in POLICY_LABELS],
        "best_policy",
        "best_total_time",
        "num_best_policies",
    ]


def _policy_source(policy: str) -> tuple[str, str]:
    if policy == "fcfs":
        return "fcfs", ""
    source, checkpoint = policy.rsplit("_", 1)
    return f"{source}_heuristic", checkpoint


def _percentile(values: list[int], percentile: float) -> float:
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _format_rows(rows: list[dict]) -> list[dict]:
    return [{key: _format_value(value) for key, value in row.items()} for row in rows]


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.2f}"
    return value


def _write_settings(
    output_dir: Path,
    args: argparse.Namespace,
    weights_by_policy: dict[str, dict[str, float]],
    scenario_count: int,
    simulation_runs: int,
) -> None:
    settings = {
        "scenario_file": str(args.scenario_file),
        "checkpoint_file": str(args.checkpoint_file),
        "scenario_count": scenario_count,
        "simulation_runs": simulation_runs,
        "policies": list(POLICY_LABELS),
        "weights_by_policy": weights_by_policy,
        "max_steps": args.max_steps,
        "corridor_length": args.corridor_length,
        "west_exit_extension": args.west_exit_extension,
        "admission_window_steps": args.admission_window_steps,
        "shared_area_capacity": args.shared_area_capacity,
        "feature_normalization": "normalize_heuristic_features=True",
        "score": "w1*waiting + w2*maneuver + w3*exit_competition + w4*path_conflict",
    }
    (output_dir / "transfer_test_settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
