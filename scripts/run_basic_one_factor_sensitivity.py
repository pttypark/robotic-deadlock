"""Run Basic-Heuristic one-factor sensitivity analysis.

The analysis uses Scenario 1-4 with 20 AGVs. Each parameter is swept one at a
time while the other four BO/heuristic features stay fixed at 1.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from decimal import Decimal
from pathlib import Path

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts.run_total_time_bo_policy_comparison import (
    BO_FEATURES,
    DIRECTIONS,
    PDF_TRAIN_TOTAL_AGVS,
    SETTINGS_FIELDNAMES,
    ScenarioSpec,
    _complete_heuristic_weights,
    _generate_pdf_case_scenarios,
    _run_single_policy,
    _scenario_to_settings_row,
    _write_csv,
)


CASE_IDS = (1, 2, 3, 4)
DEFAULT_SCENARIOS_PER_CASE = 100
DEFAULT_OUTPUT_DIR = Path("final_output") / "basic_one_factor_sensitivity"

TRIAL_FIELDNAMES = [
    "feature",
    "weight",
    "overall_avg_total_time",
    "case_1_avg_total_time",
    "case_2_avg_total_time",
    "case_3_avg_total_time",
    "case_4_avg_total_time",
    "best_so_far_avg_total_time",
    "weights_json",
]

SUMMARY_FIELDNAMES = [
    "feature",
    "best_weight",
    "best_avg_total_time",
    "worst_weight",
    "worst_avg_total_time",
    "search_min",
    "search_max",
    "search_step",
    "candidate_count",
]

RANGE_FIELDNAMES = [
    "feature",
    "lower",
    "upper",
    "best_weight",
    "best_avg_total_time",
    "range_tolerance_pct",
    "selected_weight_count",
    "search_min",
    "search_max",
    "search_step",
]

RAW_FIELDNAMES = [
    "feature",
    "weight",
    "run_index",
    "scenario_seed",
    "scenario_case_id",
    "scenario_case_name",
    "total_agvs",
    "north_agvs",
    "south_agvs",
    "west_agvs",
    "east_agvs",
    "completed",
    "total_time",
    "waiting",
    "maneuver",
    "exit_competition",
    "path_conflict",
    "approach_queue",
    "weights_json",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Scenario 1-4 Basic-Heuristic one-factor sensitivity sweep."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scenario-file", default="")
    parser.add_argument("--force-regenerate-scenarios", action="store_true")
    parser.add_argument(
        "--generate-scenarios-only",
        action="store_true",
        help="Create/reuse the Scenario 1-4 train scenario CSV, then exit before sweeping.",
    )
    parser.add_argument("--scenarios-per-case", type=int, default=DEFAULT_SCENARIOS_PER_CASE)
    parser.add_argument("--scenario-seed", type=int, default=20260606)
    parser.add_argument("--run-seed-base", type=int, default=40000)
    parser.add_argument("--sweep-min", type=float, default=-10.0)
    parser.add_argument("--sweep-max", type=float, default=10.0)
    parser.add_argument("--sweep-step", type=float, default=0.25)
    parser.add_argument(
        "--range-tolerance-pct",
        type=float,
        default=0.0,
        help=(
            "Select weights with avg_total_time <= best * (1 + pct/100) "
            "for sensitivity_selected_ranges.csv. 0 keeps only exact best/ties."
        ),
    )
    parser.add_argument(
        "--features",
        default=",".join(BO_FEATURES),
        help="Comma-separated subset of the five heuristic parameters to sweep.",
    )
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--admission-window-steps", type=int, default=2)
    parser.add_argument("--shared-area-capacity", type=int, default=2)
    args = parser.parse_args()

    _validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_count = len(CASE_IDS) * args.scenarios_per_case
    scenario_file = (
        Path(args.scenario_file)
        if args.scenario_file
        else output_dir / f"train_scenarios_{scenario_count}.csv"
    )

    scenarios = _load_or_generate_train_scenarios(
        scenario_file=scenario_file,
        scenarios_per_case=args.scenarios_per_case,
        scenario_seed=args.scenario_seed,
        run_seed_base=args.run_seed_base,
        force_regenerate=args.force_regenerate_scenarios,
    )
    if args.generate_scenarios_only:
        print(f"wrote scenarios: {scenario_file.resolve()}")
        print(f"scenario count: {len(scenarios)}")
        return

    values = _sweep_values(args.sweep_min, args.sweep_max, args.sweep_step)
    features = _parse_features(args.features)

    trial_rows, summary_rows, range_rows, raw_rows = _run_one_factor_sensitivity(
        scenarios=scenarios,
        features=features,
        values=values,
        range_tolerance_pct=args.range_tolerance_pct,
        checkpoint_dir=output_dir,
        max_steps=args.max_steps,
        corridor_length=args.corridor_length,
        west_exit_extension=args.west_exit_extension,
        admission_window_steps=args.admission_window_steps,
        shared_area_capacity=args.shared_area_capacity,
    )

    _write_csv(trial_rows, output_dir / "sensitivity_trials.csv", TRIAL_FIELDNAMES)
    _write_csv(summary_rows, output_dir / "sensitivity_summary.csv", SUMMARY_FIELDNAMES)
    _write_csv(range_rows, output_dir / "sensitivity_selected_ranges.csv", RANGE_FIELDNAMES)
    _write_csv(raw_rows, output_dir / "sensitivity_raw_runs.csv", RAW_FIELDNAMES)
    _write_readme(
        output_dir=output_dir,
        scenario_file=scenario_file,
        scenarios=scenarios,
        features=features,
        values=values,
        range_tolerance_pct=args.range_tolerance_pct,
        summary_rows=summary_rows,
        range_rows=range_rows,
    )

    print(f"wrote scenarios: {scenario_file.resolve()}")
    print(f"wrote trials: {(output_dir / 'sensitivity_trials.csv').resolve()}")
    print(f"wrote raw runs: {(output_dir / 'sensitivity_raw_runs.csv').resolve()}")
    print(f"wrote summary: {(output_dir / 'sensitivity_summary.csv').resolve()}")
    print(f"wrote selected ranges: {(output_dir / 'sensitivity_selected_ranges.csv').resolve()}")
    for row in summary_rows:
        print(
            f"{row['feature']}: best_weight={row['best_weight']}, "
            f"best_avg_total_time={float(row['best_avg_total_time']):.3f}"
        )


def _validate_args(args: argparse.Namespace) -> None:
    if args.scenarios_per_case < 1:
        raise ValueError("--scenarios-per-case must be at least 1")
    if args.sweep_step <= 0:
        raise ValueError("--sweep-step must be positive")
    if args.sweep_min > args.sweep_max:
        raise ValueError("--sweep-min must be <= --sweep-max")
    if args.range_tolerance_pct < 0:
        raise ValueError("--range-tolerance-pct must be non-negative")
    if args.shared_area_capacity < 1:
        raise ValueError("--shared-area-capacity must be at least 1")
    _parse_features(args.features)


def _load_or_generate_train_scenarios(
    scenario_file: Path,
    scenarios_per_case: int,
    scenario_seed: int,
    run_seed_base: int,
    force_regenerate: bool,
) -> list[ScenarioSpec]:
    if scenario_file.exists() and not force_regenerate:
        return _load_scenarios_csv(scenario_file)

    scenarios = _generate_pdf_case_scenarios(
        case_ids=CASE_IDS,
        scenarios_per_case=scenarios_per_case,
        total_agv_options=(PDF_TRAIN_TOTAL_AGVS,),
        seed_base=run_seed_base,
        scenario_seed=scenario_seed,
        split="train",
    )
    rows = [_scenario_to_settings_row(scenario, scenario.total_agvs) for scenario in scenarios]
    _write_csv(rows, scenario_file, SETTINGS_FIELDNAMES)
    return scenarios


def _run_one_factor_sensitivity(
    scenarios: list[ScenarioSpec],
    features: list[str],
    values: list[float],
    range_tolerance_pct: float,
    checkpoint_dir: Path | None,
    max_steps: int,
    corridor_length: int,
    west_exit_extension: int,
    admission_window_steps: int,
    shared_area_capacity: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    trial_rows = []
    summary_rows = []
    range_rows = []
    raw_rows = []

    for feature in features:
        feature_rows = []
        best_so_far = math.inf
        for value in values:
            weights = _one_factor_weights(feature, value)
            by_case, candidate_raw_rows = _evaluate_weights_by_case(
                scenarios=scenarios,
                feature=feature,
                value=value,
                weights=weights,
                max_steps=max_steps,
                corridor_length=corridor_length,
                west_exit_extension=west_exit_extension,
                admission_window_steps=admission_window_steps,
                shared_area_capacity=shared_area_capacity,
            )
            raw_rows.extend(candidate_raw_rows)
            all_values = [
                total_time
                for case_values in by_case.values()
                for total_time in case_values
            ]
            objective = statistics.fmean(all_values)
            best_so_far = min(best_so_far, objective)
            row = {
                "feature": feature,
                "weight": value,
                "overall_avg_total_time": objective,
                "best_so_far_avg_total_time": best_so_far,
                "weights_json": json.dumps(weights, sort_keys=True),
            }
            for case_id in CASE_IDS:
                row[f"case_{case_id}_avg_total_time"] = statistics.fmean(by_case[case_id])
            feature_rows.append(row)
            trial_rows.append(row)
            print(
                f"[{feature}] weight={value} "
                f"avg_total_time={objective:.3f} best_so_far={best_so_far:.3f}",
                flush=True,
            )

        summary_rows.append(_summarize_feature(feature, feature_rows, values))
        range_rows.append(
            _selected_range_row(
                feature=feature,
                feature_rows=feature_rows,
                values=values,
                range_tolerance_pct=range_tolerance_pct,
            )
        )
        if checkpoint_dir is not None:
            _write_csv(trial_rows, checkpoint_dir / "sensitivity_trials.csv", TRIAL_FIELDNAMES)
            _write_csv(summary_rows, checkpoint_dir / "sensitivity_summary.csv", SUMMARY_FIELDNAMES)
            _write_csv(range_rows, checkpoint_dir / "sensitivity_selected_ranges.csv", RANGE_FIELDNAMES)
            _write_csv(raw_rows, checkpoint_dir / "sensitivity_raw_runs.csv", RAW_FIELDNAMES)

    return trial_rows, summary_rows, range_rows, raw_rows


def _evaluate_weights_by_case(
    scenarios: list[ScenarioSpec],
    feature: str,
    value: float,
    weights: dict[str, float],
    max_steps: int,
    corridor_length: int,
    west_exit_extension: int,
    admission_window_steps: int,
    shared_area_capacity: int,
) -> tuple[dict[int, list[int]], list[dict]]:
    by_case = {case_id: [] for case_id in CASE_IDS}
    raw_rows = []
    weights_json = json.dumps(weights, sort_keys=True)
    for scenario in scenarios:
        metrics = _run_single_policy(
            scenario=scenario,
            policy="fixed_heuristic",
            weights=weights,
            max_steps=max_steps,
            corridor_length=corridor_length,
            west_exit_extension=west_exit_extension,
            admission_window_steps=admission_window_steps,
            shared_area_capacity=shared_area_capacity,
        )
        by_case[scenario.scenario_case_id].append(int(metrics["total_time"]))
        row = {
            "feature": feature,
            "weight": value,
            "run_index": scenario.run_index,
            "scenario_seed": scenario.seed,
            "scenario_case_id": scenario.scenario_case_id,
            "scenario_case_name": scenario.scenario_case_name,
            "total_agvs": scenario.total_agvs,
            "north_agvs": scenario.allocation["NORTH"],
            "south_agvs": scenario.allocation["SOUTH"],
            "west_agvs": scenario.allocation["WEST"],
            "east_agvs": scenario.allocation["EAST"],
            "completed": metrics["completed"],
            "total_time": metrics["total_time"],
            "weights_json": weights_json,
        }
        row.update({candidate: weights[candidate] for candidate in BO_FEATURES})
        raw_rows.append(row)
    return by_case, raw_rows


def _one_factor_weights(feature: str, value: float) -> dict[str, float]:
    weights = {candidate: 1.0 for candidate in BO_FEATURES}
    weights[feature] = float(value)
    return _complete_heuristic_weights(weights)


def _summarize_feature(
    feature: str,
    feature_rows: list[dict],
    values: list[float],
) -> dict:
    best_row = min(feature_rows, key=lambda row: row["overall_avg_total_time"])
    worst_row = max(feature_rows, key=lambda row: row["overall_avg_total_time"])
    step = values[1] - values[0] if len(values) > 1 else 0.0
    return {
        "feature": feature,
        "best_weight": best_row["weight"],
        "best_avg_total_time": best_row["overall_avg_total_time"],
        "worst_weight": worst_row["weight"],
        "worst_avg_total_time": worst_row["overall_avg_total_time"],
        "search_min": values[0],
        "search_max": values[-1],
        "search_step": step,
        "candidate_count": len(values),
    }


def _selected_range_row(
    feature: str,
    feature_rows: list[dict],
    values: list[float],
    range_tolerance_pct: float,
) -> dict:
    best_row = min(feature_rows, key=lambda row: row["overall_avg_total_time"])
    threshold = best_row["overall_avg_total_time"] * (1.0 + range_tolerance_pct / 100.0)
    selected = [
        row
        for row in feature_rows
        if row["overall_avg_total_time"] <= threshold
    ]
    selected_weights = [row["weight"] for row in selected]
    step = values[1] - values[0] if len(values) > 1 else 0.0
    return {
        "feature": feature,
        "lower": min(selected_weights),
        "upper": max(selected_weights),
        "best_weight": best_row["weight"],
        "best_avg_total_time": best_row["overall_avg_total_time"],
        "range_tolerance_pct": range_tolerance_pct,
        "selected_weight_count": len(selected_weights),
        "search_min": values[0],
        "search_max": values[-1],
        "search_step": step,
    }


def _sweep_values(min_value: float, max_value: float, step: float) -> list[float]:
    current = Decimal(str(min_value))
    end = Decimal(str(max_value))
    increment = Decimal(str(step))
    values = []
    while current <= end:
        values.append(float(current))
        current += increment
    return values


def _parse_features(raw_features: str) -> list[str]:
    features = [
        feature.strip()
        for feature in raw_features.split(",")
        if feature.strip()
    ]
    unknown = sorted(set(features) - set(BO_FEATURES))
    if unknown:
        raise ValueError(f"Unknown feature(s): {unknown}")
    if not features:
        raise ValueError("At least one feature must be provided")
    return features


def _load_scenarios_csv(path: Path) -> list[ScenarioSpec]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [_scenario_from_row(row) for row in reader]


def _scenario_from_row(row: dict[str, str]) -> ScenarioSpec:
    allocation = {
        "NORTH": int(row["north_agvs"]),
        "SOUTH": int(row["south_agvs"]),
        "WEST": int(row["west_agvs"]),
        "EAST": int(row["east_agvs"]),
    }
    return ScenarioSpec(
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


def _write_readme(
    output_dir: Path,
    scenario_file: Path,
    scenarios: list[ScenarioSpec],
    features: list[str],
    values: list[float],
    range_tolerance_pct: float,
    summary_rows: list[dict],
    range_rows: list[dict],
) -> None:
    scenario_counts = {
        case_id: sum(scenario.scenario_case_id == case_id for scenario in scenarios)
        for case_id in CASE_IDS
    }
    lines = [
        "# Basic One-Factor Sensitivity Analysis",
        "",
        "## Setup",
        "",
        f"- Scenario file: `{scenario_file}`",
        f"- Scenario cases: `{list(CASE_IDS)}`",
        f"- Scenarios per case: `{scenario_counts}`",
        f"- Total scenarios: `{len(scenarios)}`",
        f"- AGVs: `{PDF_TRAIN_TOTAL_AGVS}`",
        f"- Features swept: `{features}`",
        f"- Sweep values: `{values[0]}` to `{values[-1]}` by `{values[1] - values[0] if len(values) > 1 else 0.0}`",
        f"- Controlled feature value: `1.0`",
        f"- Selected range tolerance: `{range_tolerance_pct}%`",
        "",
        "For each feature, the other four heuristic parameters are fixed at `1.0`. "
        "Only the target feature changes across the sweep grid. FCFS is excluded "
        "because it has no weight vector, and PSO/BO are not run during this "
        "sensitivity stage.",
        "",
        "## Best Weight By Feature",
        "",
        "| feature | best weight | avg total_time | selected lower | selected upper |",
        "|---|---:|---:|---:|---:|",
    ]
    ranges_by_feature = {row["feature"]: row for row in range_rows}
    for row in summary_rows:
        range_row = ranges_by_feature[row["feature"]]
        lines.append(
            f"| {row['feature']} | {row['best_weight']} | "
            f"{float(row['best_avg_total_time']):.3f} | "
            f"{range_row['lower']} | {range_row['upper']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `train_scenarios_400.csv`: Scenario 1-4 training set reused by later Train runs",
            "- `sensitivity_trials.csv`: full one-factor sweep curve",
            "- `sensitivity_raw_runs.csv`: one row per feature, weight, and scenario episode",
            "- `sensitivity_summary.csv`: best/worst value per parameter",
            "- `sensitivity_selected_ranges.csv`: range selected from near-best sweep values",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
