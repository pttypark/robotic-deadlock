"""Analyze integrated sensitivity results and recommend parameter ranges."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


DEFAULT_INPUT = (
    Path("final_output")
    / "basic_one_factor_sensitivity"
    / "sensitivity_integrated_results.csv"
)
DEFAULT_OUTPUT_DIR = Path("final_output") / "basic_one_factor_sensitivity"

STAT_FIELDNAMES = [
    "feature",
    "weight",
    "count",
    "mean_total_time",
    "std_total_time",
    "min_total_time",
    "q1_total_time",
    "median_total_time",
    "q3_total_time",
    "iqr_total_time",
    "max_total_time",
    "boxplot_upper_fence",
    "outlier_high_count",
    "case_1_mean_total_time",
    "case_2_mean_total_time",
    "case_3_mean_total_time",
    "case_4_mean_total_time",
]

RANGE_FIELDNAMES = [
    "feature",
    "best_weight",
    "best_mean_total_time",
    "best_std_total_time",
    "sweep_mean_q1",
    "sweep_mean_median",
    "sweep_mean_q3",
    "sweep_mean_iqr",
    "boxplot_upper_on_weight_means",
    "near_best_5pct_upper",
    "outlier_removed_lower",
    "outlier_removed_upper",
    "outlier_removed_count",
    "top_quartile_lower",
    "top_quartile_upper",
    "top_quartile_count",
    "hybrid_5pct_lower",
    "hybrid_5pct_upper",
    "hybrid_5pct_count",
    "recommended_lower",
    "recommended_upper",
    "recommended_rule",
    "transition_best_weight",
    "transition_best_mean_total_time",
    "transition_best_rule",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze sensitivity_integrated_results.csv by feature and weight."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = _load_grouped_total_times(input_path)
    stat_rows = _build_stat_rows(grouped)
    range_rows = _build_range_rows(stat_rows)

    stat_path = output_dir / "sensitivity_feature_weight_statistics.csv"
    range_path = output_dir / "sensitivity_range_recommendations.csv"
    _write_csv(stat_path, stat_rows, STAT_FIELDNAMES)
    _write_csv(range_path, range_rows, RANGE_FIELDNAMES)

    print(f"wrote feature/weight statistics: {stat_path.resolve()}")
    print(f"wrote range recommendations: {range_path.resolve()}")
    for row in range_rows:
        print(
            f"{row['feature']}: best={row['best_weight']} "
            f"mean={float(row['best_mean_total_time']):.2f}, "
            f"recommended=[{row['recommended_lower']}, {row['recommended_upper']}] "
            f"rule={row['recommended_rule']}"
        )


def _load_grouped_total_times(path: Path) -> dict[tuple[str, float], dict]:
    grouped: dict[tuple[str, float], dict] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            feature = row["feature"]
            weight = float(row["weight"])
            case_id = int(row["scenario_case_id"])
            total_time = float(row["total_time"])
            key = (feature, weight)
            if key not in grouped:
                grouped[key] = {
                    "feature": feature,
                    "weight": weight,
                    "values": [],
                    "case_values": defaultdict(list),
                }
            grouped[key]["values"].append(total_time)
            grouped[key]["case_values"][case_id].append(total_time)
    return grouped


def _build_stat_rows(grouped: dict[tuple[str, float], dict]) -> list[dict]:
    rows = []
    for _, group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        values = sorted(group["values"])
        q1 = _percentile(values, 25.0)
        median = _percentile(values, 50.0)
        q3 = _percentile(values, 75.0)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        row = {
            "feature": group["feature"],
            "weight": group["weight"],
            "count": len(values),
            "mean_total_time": statistics.fmean(values),
            "std_total_time": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min_total_time": values[0],
            "q1_total_time": q1,
            "median_total_time": median,
            "q3_total_time": q3,
            "iqr_total_time": iqr,
            "max_total_time": values[-1],
            "boxplot_upper_fence": upper_fence,
            "outlier_high_count": sum(value > upper_fence for value in values),
        }
        for case_id in (1, 2, 3, 4):
            case_values = group["case_values"][case_id]
            row[f"case_{case_id}_mean_total_time"] = statistics.fmean(case_values)
        rows.append(row)
    return rows


def _build_range_rows(stat_rows: list[dict]) -> list[dict]:
    rows_by_feature: dict[str, list[dict]] = defaultdict(list)
    for row in stat_rows:
        rows_by_feature[row["feature"]].append(row)

    range_rows = []
    for feature, rows in sorted(rows_by_feature.items()):
        rows = sorted(rows, key=lambda row: row["weight"])
        means = [float(row["mean_total_time"]) for row in rows]
        weights = [float(row["weight"]) for row in rows]
        best_row = min(rows, key=lambda row: row["mean_total_time"])
        best_mean = float(best_row["mean_total_time"])
        q1 = _percentile(sorted(means), 25.0)
        median = _percentile(sorted(means), 50.0)
        q3 = _percentile(sorted(means), 75.0)
        iqr = q3 - q1
        boxplot_upper = q3 + 1.5 * iqr
        near_best_5pct_upper = best_mean * 1.05

        outlier_removed = [
            weight
            for weight, mean in zip(weights, means)
            if mean <= boxplot_upper
        ]
        top_quartile = [
            weight
            for weight, mean in zip(weights, means)
            if mean <= q1
        ]
        hybrid_upper = min(boxplot_upper, near_best_5pct_upper)
        hybrid = [
            weight
            for weight, mean in zip(weights, means)
            if mean <= hybrid_upper
        ]
        recommended = _contiguous_weights_around_best(
            selected_weights=top_quartile,
            best_weight=float(best_row["weight"]),
            step=_infer_step(weights),
        )
        transition_best = _transition_best_point(rows)

        range_rows.append(
            {
                "feature": feature,
                "best_weight": best_row["weight"],
                "best_mean_total_time": best_mean,
                "best_std_total_time": best_row["std_total_time"],
                "sweep_mean_q1": q1,
                "sweep_mean_median": median,
                "sweep_mean_q3": q3,
                "sweep_mean_iqr": iqr,
                "boxplot_upper_on_weight_means": boxplot_upper,
                "near_best_5pct_upper": near_best_5pct_upper,
                "outlier_removed_lower": min(outlier_removed),
                "outlier_removed_upper": max(outlier_removed),
                "outlier_removed_count": len(outlier_removed),
                "top_quartile_lower": min(top_quartile),
                "top_quartile_upper": max(top_quartile),
                "top_quartile_count": len(top_quartile),
                "hybrid_5pct_lower": min(hybrid),
                "hybrid_5pct_upper": max(hybrid),
                "hybrid_5pct_count": len(hybrid),
                "recommended_lower": recommended[0],
                "recommended_upper": recommended[1],
                "recommended_rule": "contiguous top-quartile interval around best weight",
                "transition_best_weight": transition_best["weight"],
                "transition_best_mean_total_time": transition_best["mean_total_time"],
                "transition_best_rule": transition_best["rule"],
            }
        )
    return range_rows


def _transition_best_point(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda row: row["weight"])
    weights = [float(row["weight"]) for row in rows]
    means = [float(row["mean_total_time"]) for row in rows]
    best_mean = min(means)
    tolerance = 0.005
    selected = [
        index
        for index, mean in enumerate(means)
        if mean <= best_mean + tolerance
    ]
    if len(selected) == len(rows):
        neutral_weight = 0.0 if 0.0 in weights else weights[len(weights) // 2]
        neutral_index = weights.index(neutral_weight)
        return {
            "weight": neutral_weight,
            "mean_total_time": means[neutral_index],
            "rule": "flat curve; neutral weight selected",
        }

    groups = _contiguous_index_groups(selected)
    candidates = []
    last_index = len(rows) - 1
    for group in groups:
        if group[-1] == last_index:
            candidate_index = group[0]
            rule = "entry point into best plateau near upper boundary"
        elif group[0] == 0:
            candidate_index = group[-1]
            rule = "entry point into best plateau near lower boundary"
        else:
            candidate_index = min(group, key=lambda index: abs(weights[index]))
            rule = "interior best point closest to neutral weight"
        candidates.append((candidate_index, rule))
    candidate_index, rule = min(candidates, key=lambda item: abs(weights[item[0]]))
    return {
        "weight": weights[candidate_index],
        "mean_total_time": means[candidate_index],
        "rule": rule,
    }


def _contiguous_index_groups(indices: list[int]) -> list[list[int]]:
    groups = []
    for index in indices:
        if not groups or index != groups[-1][-1] + 1:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def _contiguous_weights_around_best(
    selected_weights: list[float],
    best_weight: float,
    step: float,
) -> tuple[float, float]:
    selected = {round(weight, 10) for weight in selected_weights}
    current = round(best_weight, 10)
    lower = current
    upper = current
    while round(lower - step, 10) in selected:
        lower = round(lower - step, 10)
    while round(upper + step, 10) in selected:
        upper = round(upper + step, 10)
    return lower, upper


def _infer_step(weights: list[float]) -> float:
    diffs = [
        round(weights[index + 1] - weights[index], 10)
        for index in range(len(weights) - 1)
    ]
    return min(diff for diff in diffs if diff > 0)


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _format_value(row[field])
                    for field in fieldnames
                }
            )


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


if __name__ == "__main__":
    main()
