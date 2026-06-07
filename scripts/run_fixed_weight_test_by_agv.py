from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_total_time_bo_policy_comparison import (  # noqa: E402
    BEST_POLICY_FIELDNAMES,
    COMPARISON_FIELDNAMES,
    PDF_CASE_TOTAL_AGV_OPTIONS,
    POLICIES,
    RUN_FIELDNAMES,
    SCENARIO_CASES,
    SETTINGS_FIELDNAMES,
    SUMMARY_FIELDNAMES,
    _best_policy_rows,
    _build_pdf_case_scenario,
    _compare_runs,
    _complete_heuristic_weights,
    _percentile,
    _run_policy_comparison,
    _scenario_to_settings_row,
    _summarize_runs,
    _write_csv,
    _write_json,
)


DEFAULT_BASIC_WEIGHTS = {
    "waiting": 5.985670,
    "maneuver": -3.799914,
    "exit_competition": -0.699648,
    "path_conflict": -0.544682,
    "approach_queue": 0.0,
    "remaining_path": 0.0,
    "same_direction_backlog": 0.0,
}

DEFAULT_BO_WEIGHTS = {
    "waiting": 5.660327,
    "maneuver": -4.0,
    "exit_competition": -4.0,
    "path_conflict": -0.656567,
    "approach_queue": 0.0,
    "remaining_path": 0.0,
    "same_direction_backlog": 0.0,
}

DEFAULT_PSO_WEIGHTS = {
    "waiting": 8.339856,
    "maneuver": -3.688935,
    "exit_competition": -2.609410,
    "path_conflict": -3.0,
    "approach_queue": 0.0,
    "remaining_path": 0.0,
    "same_direction_backlog": 0.0,
}

DEFAULT_OUTPUT_DIR = PROJECT_DIR / "final_output" / "test_100_by_agv_fixed_weights"
CASE_IDS = (1, 2, 3, 4, 5)
METRIC_COLUMNS = {
    "total_time",
    "avg_total_time",
    "median_total_time",
    "std_total_time",
    "min_total_time",
    "max_total_time",
    "p10_total_time",
    "p25_total_time",
    "p75_total_time",
    "p90_total_time",
    "fcfs_total_time",
    "fixed_heuristic_total_time",
    "bo_heuristic_total_time",
    "pso_heuristic_total_time",
    "best_total_time",
    "second_best_total_time",
    "margin_to_second",
    "elapsed_seconds",
    "elapsed_minutes",
    "seconds_per_simulation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 100 fixed test scenarios for each AGV count and evaluate "
            "FCFS, Basic-Heuristic, BO-Heuristic, and PSO-Heuristic with fixed weights."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenarios-per-case", type=int, default=20)
    parser.add_argument("--agv-counts", default="16,20,24")
    parser.add_argument("--scenario-seed", type=int, default=20260608)
    parser.add_argument("--seed-base", type=int, default=60000)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--admission-window-steps", type=int, default=2)
    parser.add_argument("--shared-area-capacity", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agv_counts = _parse_agv_counts(args.agv_counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    scenarios_by_agv = _build_test_scenarios_by_agv(
        agv_counts=agv_counts,
        scenarios_per_case=args.scenarios_per_case,
        scenario_seed=args.scenario_seed,
        seed_base=args.seed_base,
    )
    all_scenarios = [
        scenario
        for total_agvs in agv_counts
        for scenario in scenarios_by_agv[total_agvs]
    ]

    _write_scenario_files(args.output_dir, scenarios_by_agv, all_scenarios)

    weights = {
        "fixed_heuristic": _complete_heuristic_weights(DEFAULT_BASIC_WEIGHTS),
        "bo_heuristic": _complete_heuristic_weights(DEFAULT_BO_WEIGHTS),
        "pso_heuristic": _complete_heuristic_weights(DEFAULT_PSO_WEIGHTS),
    }

    all_run_rows = []
    all_comparison_rows = []
    all_best_policy_rows = []
    summary_by_agv = []

    for total_agvs in agv_counts:
        scenarios = scenarios_by_agv[total_agvs]
        run_rows = _run_policy_comparison(
            scenarios=scenarios,
            total_agvs=total_agvs,
            fixed_weights=weights["fixed_heuristic"],
            bo_weights=weights["bo_heuristic"],
            pso_weights=weights["pso_heuristic"],
            max_steps=args.max_steps,
            corridor_length=args.corridor_length,
            west_exit_extension=args.west_exit_extension,
            admission_window_steps=args.admission_window_steps,
            shared_area_capacity=args.shared_area_capacity,
        )
        comparison_rows = _compare_runs(run_rows)
        best_policy_rows = _best_policy_rows(comparison_rows, scenarios)

        for row in _summarize_runs(run_rows, comparison_rows, total_agvs):
            row["total_agvs"] = total_agvs
            summary_by_agv.append(row)

        all_run_rows.extend(run_rows)
        all_comparison_rows.extend(comparison_rows)
        all_best_policy_rows.extend(best_policy_rows)

    overall_summary = _summarize_runs(all_run_rows, all_comparison_rows, 0)
    case_summary_by_agv = _group_total_time_summary(
        all_run_rows,
        group_keys=("total_agvs", "scenario_case_id", "scenario_case_name", "policy"),
    )
    case_summary_overall = _group_total_time_summary(
        all_run_rows,
        group_keys=("scenario_case_id", "scenario_case_name", "policy"),
    )

    elapsed_seconds = time.monotonic() - start_time
    runtime_rows = [
        {
            "scope": "all_test_sets",
            "agv_counts": ",".join(str(value) for value in agv_counts),
            "scenario_count": len(all_scenarios),
            "policy_count": len(POLICIES),
            "total_simulation_runs": len(all_run_rows),
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "seconds_per_simulation": elapsed_seconds / len(all_run_rows),
        }
    ]

    _write_csv(
        _format_rows(all_run_rows),
        args.output_dir / "test_raw_runs.csv",
        RUN_FIELDNAMES,
    )
    _write_csv(
        _format_rows(all_comparison_rows),
        args.output_dir / "test_trial_comparison.csv",
        COMPARISON_FIELDNAMES,
    )
    _write_csv(
        _format_rows(all_best_policy_rows),
        args.output_dir / "test_best_policy_by_run.csv",
        BEST_POLICY_FIELDNAMES,
    )
    _write_csv(
        _format_rows(overall_summary),
        args.output_dir / "test_policy_summary_overall.csv",
        SUMMARY_FIELDNAMES,
    )
    _write_csv(
        _format_rows(summary_by_agv),
        args.output_dir / "test_policy_summary_by_agv.csv",
        ["total_agvs", *SUMMARY_FIELDNAMES],
    )
    _write_csv(
        _format_rows(case_summary_by_agv),
        args.output_dir / "test_scenario_case_summary_by_agv.csv",
        [
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
        ],
    )
    _write_csv(
        _format_rows(case_summary_overall),
        args.output_dir / "test_scenario_case_summary_overall.csv",
        [
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
        ],
    )
    _write_csv(
        _format_rows(runtime_rows),
        args.output_dir / "test_runtime_summary.csv",
        [
            "scope",
            "agv_counts",
            "scenario_count",
            "policy_count",
            "total_simulation_runs",
            "elapsed_seconds",
            "elapsed_minutes",
            "seconds_per_simulation",
        ],
    )
    _write_json(
        {
            "basic_heuristic_weights": weights["fixed_heuristic"],
            "bo_heuristic_weights": weights["bo_heuristic"],
            "pso_heuristic_weights": weights["pso_heuristic"],
            "agv_counts": agv_counts,
            "case_ids": CASE_IDS,
            "scenarios_per_case": args.scenarios_per_case,
            "scenario_seed": args.scenario_seed,
            "seed_base": args.seed_base,
            "max_steps": args.max_steps,
            "corridor_length": args.corridor_length,
            "west_exit_extension": args.west_exit_extension,
            "admission_window_steps": args.admission_window_steps,
            "shared_area_capacity": args.shared_area_capacity,
        },
        args.output_dir / "test_weights_and_settings.json",
    )

    print(f"wrote output dir: {args.output_dir.resolve()}")
    print(f"test scenarios: {len(all_scenarios)}")
    print(f"simulation runs: {len(all_run_rows)}")
    print(f"elapsed_seconds: {elapsed_seconds:.2f}")
    print("overall avg Total_Time:")
    for row in overall_summary:
        print(f"  {row['policy']}: {row['avg_total_time']:.2f}")


def _parse_agv_counts(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    invalid = sorted(set(values) - set(PDF_CASE_TOTAL_AGV_OPTIONS))
    if invalid:
        raise ValueError(f"unsupported AGV count(s): {invalid}")
    return values


def _build_test_scenarios_by_agv(
    agv_counts: list[int],
    scenarios_per_case: int,
    scenario_seed: int,
    seed_base: int,
):
    rng = random.Random(scenario_seed)
    scenarios_by_agv = {}
    run_index = 0

    for total_agvs in agv_counts:
        scenarios = []
        for case_id in CASE_IDS:
            for variant_index in range(scenarios_per_case):
                scenarios.append(
                    _build_pdf_case_scenario(
                        case_id=case_id,
                        variant_index=variant_index,
                        run_index=run_index,
                        seed=seed_base + run_index,
                        total_agvs=total_agvs,
                        rng=rng,
                        split="test",
                    )
                )
                run_index += 1
        scenarios_by_agv[total_agvs] = scenarios

    return scenarios_by_agv


def _write_scenario_files(output_dir: Path, scenarios_by_agv, all_scenarios) -> None:
    for total_agvs, scenarios in scenarios_by_agv.items():
        _write_csv(
            [_scenario_to_settings_row(scenario, total_agvs) for scenario in scenarios],
            output_dir / f"test_scenarios_100_agv{total_agvs}.csv",
            SETTINGS_FIELDNAMES,
        )
    _write_csv(
        [_scenario_to_settings_row(scenario, scenario.total_agvs) for scenario in all_scenarios],
        output_dir / "test_scenarios_300_all_agv_counts.csv",
        SETTINGS_FIELDNAMES,
    )


def _group_total_time_summary(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[key] for key in group_keys)
        grouped[key].append(row)

    summary_rows = []
    for key, group_rows in sorted(grouped.items()):
        values = [float(row["total_time"]) for row in group_rows]
        summary = {
            group_key: key[index]
            for index, group_key in enumerate(group_keys)
        }
        summary.update(
            {
                "runs": len(group_rows),
                "completed_runs": sum(
                    int(row["completed"]) == int(row["total_agvs"])
                    for row in group_rows
                ),
                "avg_total_time": statistics.fmean(values),
                "median_total_time": statistics.median(values),
                "std_total_time": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min_total_time": min(values),
                "max_total_time": max(values),
                "p10_total_time": _percentile([int(value) for value in values], 10.0),
                "p90_total_time": _percentile([int(value) for value in values], 90.0),
            }
        )
        summary_rows.append(summary)
    return summary_rows


def _format_rows(rows: list[dict]) -> list[dict]:
    return [_format_row(row) for row in rows]


def _format_row(row: dict) -> dict:
    formatted = {}
    for key, value in row.items():
        if key in METRIC_COLUMNS and isinstance(value, (float, int)):
            formatted[key] = f"{float(value):.2f}"
        elif key.endswith("_pct") and isinstance(value, (float, int)):
            formatted[key] = f"{float(value):.2f}"
        else:
            formatted[key] = value
    return formatted


if __name__ == "__main__":
    main()
