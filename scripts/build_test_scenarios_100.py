from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.run_total_time_bo_policy_comparison import (  # noqa: E402
    PDF_CASE_TOTAL_AGV_OPTIONS,
    SCENARIO_CASES,
    SETTINGS_FIELDNAMES,
    _build_pdf_case_scenario,
    _scenario_to_settings_row,
)


DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "final_output"
    / "test_scenarios_100"
    / "test_scenarios_100.csv"
)
DEFAULT_CASE_IDS = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the fixed 100-row PDF test scenario set: "
            "20 scenarios for each of cases 1, 2, 3, 4, and 5."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV path to write.",
    )
    parser.add_argument(
        "--scenarios-per-case",
        type=int,
        default=20,
        help="Number of test scenarios to generate for each case.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=50000,
        help="First scenario_seed value written to the CSV.",
    )
    parser.add_argument(
        "--scenario-seed",
        type=int,
        default=20260607,
        help="Random seed used to sample burst, skew, and goal-plan variants.",
    )
    return parser.parse_args()


def build_test_scenarios(
    scenarios_per_case: int,
    seed_base: int,
    scenario_seed: int,
):
    rng = random.Random(scenario_seed)
    scenarios = []
    run_index = 0

    for case_id in DEFAULT_CASE_IDS:
        for variant_index in range(scenarios_per_case):
            total_agvs = PDF_CASE_TOTAL_AGV_OPTIONS[
                variant_index % len(PDF_CASE_TOTAL_AGV_OPTIONS)
            ]
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

    return scenarios


def write_settings_csv(path: Path, scenarios) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SETTINGS_FIELDNAMES)
        writer.writeheader()
        for scenario in scenarios:
            writer.writerow(_scenario_to_settings_row(scenario, scenario.total_agvs))


def print_summary(path: Path, scenarios) -> None:
    case_counts = Counter(scenario.scenario_case_id for scenario in scenarios)
    agv_counts_by_case: dict[int, Counter] = defaultdict(Counter)
    split_counts = Counter(scenario.scenario_split for scenario in scenarios)

    for scenario in scenarios:
        agv_counts_by_case[scenario.scenario_case_id][scenario.total_agvs] += 1

    print(f"wrote: {path}")
    print(f"rows: {len(scenarios)}")
    print("case counts:")
    for case_id in DEFAULT_CASE_IDS:
        case = SCENARIO_CASES[case_id]
        print(f"  {case_id} {case['name']}: {case_counts[case_id]}")
    print("AGV counts by case:")
    for case_id in DEFAULT_CASE_IDS:
        counts = agv_counts_by_case[case_id]
        formatted = ", ".join(
            f"{total_agvs}={counts[total_agvs]}"
            for total_agvs in PDF_CASE_TOTAL_AGV_OPTIONS
        )
        print(f"  case {case_id}: {formatted}")
    print("split counts:")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")


def main() -> None:
    args = parse_args()
    scenarios = build_test_scenarios(
        scenarios_per_case=args.scenarios_per_case,
        seed_base=args.seed_base,
        scenario_seed=args.scenario_seed,
    )
    write_settings_csv(args.output, scenarios)
    print_summary(args.output, scenarios)


if __name__ == "__main__":
    main()
