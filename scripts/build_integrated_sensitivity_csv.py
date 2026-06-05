"""Build one integrated CSV from sensitivity analysis outputs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


DEFAULT_INPUT_DIR = Path("final_output") / "basic_one_factor_sensitivity"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "sensitivity_integrated_results.csv"

RAW_FILE = "sensitivity_raw_runs.csv"
TRIAL_FILE = "sensitivity_trials.csv"
SUMMARY_FILE = "sensitivity_summary.csv"
RANGE_FILE = "sensitivity_selected_ranges.csv"

RAW_FIELDS = [
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

TRIAL_FIELDS = [
    "overall_avg_total_time",
    "case_1_avg_total_time",
    "case_2_avg_total_time",
    "case_3_avg_total_time",
    "case_4_avg_total_time",
    "best_so_far_avg_total_time",
]

SUMMARY_FIELDS = [
    "best_weight",
    "best_avg_total_time",
    "worst_weight",
    "worst_avg_total_time",
    "search_min",
    "search_max",
    "search_step",
    "candidate_count",
]

RANGE_FIELDS = [
    "lower",
    "upper",
    "range_tolerance_pct",
    "selected_weight_count",
]

INTEGER_FIELDS = {
    "run_index",
    "scenario_seed",
    "scenario_case_id",
    "total_agvs",
    "north_agvs",
    "south_agvs",
    "west_agvs",
    "east_agvs",
    "completed",
    "candidate_count",
    "selected_weight_count",
}

TEXT_FIELDS = {
    "feature",
    "scenario_case_name",
    "weights_json",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge sensitivity raw/trial/summary/range CSV files."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--excel-text",
        action="store_true",
        help="Write decimal-valued cells as Excel text formulas so trailing .00 stays visible.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trials = _load_by_feature_weight(input_dir / TRIAL_FILE)
    summaries = _load_by_feature(input_dir / SUMMARY_FILE)
    ranges = _load_by_feature(input_dir / RANGE_FILE)

    fieldnames = _integrated_fieldnames()
    row_count = 0
    with (input_dir / RAW_FILE).open(newline="", encoding="utf-8") as raw_handle:
        reader = csv.DictReader(raw_handle)
        with output_path.open("w", newline="", encoding="utf-8") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()
            for raw_row in reader:
                key = _feature_weight_key(raw_row)
                feature = raw_row["feature"]
                row = dict(raw_row)
                row.update(
                    {
                        f"trial_{field}": trials[key][field]
                        for field in TRIAL_FIELDS
                    }
                )
                row.update(
                    {
                        f"summary_{field}": summaries[feature][field]
                        for field in SUMMARY_FIELDS
                    }
                )
                row.update(
                    {
                        f"range_{field}": ranges[feature][field]
                        for field in RANGE_FIELDS
                    }
                )
                writer.writerow(_format_row(row, fieldnames, excel_text=args.excel_text))
                row_count += 1

    print(f"wrote integrated csv: {output_path.resolve()}")
    print(f"rows: {row_count}")


def _integrated_fieldnames() -> list[str]:
    return (
        RAW_FIELDS
        + [f"trial_{field}" for field in TRIAL_FIELDS]
        + [f"summary_{field}" for field in SUMMARY_FIELDS]
        + [f"range_{field}" for field in RANGE_FIELDS]
    )


def _load_by_feature_weight(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            _feature_weight_key(row): row
            for row in csv.DictReader(handle)
        }


def _load_by_feature(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["feature"]: row
            for row in csv.DictReader(handle)
        }


def _feature_weight_key(row: dict[str, str]) -> tuple[str, str]:
    return (row["feature"], _format_number(row["weight"]))


def _format_row(
    row: dict[str, str],
    fieldnames: list[str],
    excel_text: bool,
) -> dict[str, str]:
    return {
        field: _format_value(field, row.get(field, ""), excel_text=excel_text)
        for field in fieldnames
    }


def _format_value(field: str, value: str, excel_text: bool) -> str:
    if value == "":
        return ""
    base_field = field.split("_", 1)[1] if field.startswith(("trial_", "summary_", "range_")) else field
    if base_field in TEXT_FIELDS:
        return value
    if base_field in INTEGER_FIELDS:
        return str(int(float(value)))
    formatted = _format_number(value)
    if excel_text:
        return f'="{formatted}"'
    return formatted


def _format_number(value: str) -> str:
    return f"{float(value):.2f}"


if __name__ == "__main__":
    main()
