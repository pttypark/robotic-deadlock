from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rware.agv_layouts import build_fcfs_double_cross_shared_area_layout  # noqa: E402
from rware.fcfs_cross_simulation import FCFSCrossExperiment  # noqa: E402
from scripts.run_total_time_bo_policy_comparison import _complete_heuristic_weights  # noqa: E402


DIRECTIONS = ("L_NORTH", "L_SOUTH", "WEST", "R_NORTH", "R_SOUTH", "EAST")
DOMINANT_ROTATION = ("L_NORTH", "R_NORTH", "EAST", "R_SOUTH", "L_SOUTH", "WEST")
CASE_IDS = (1, 2, 3, 4, 5)
POLICIES = ("fcfs", "fixed_heuristic", "bo_heuristic", "pso_heuristic")
SCENARIO_CASES = {
    1: ("normal_balanced", "Scenario 1: Normal balanced"),
    2: ("arrival_burst", "Scenario 2: Arrival burst"),
    3: ("direction_skewed", "Scenario 3: Direction-skewed"),
    4: ("exit_concentrated", "Scenario 4: Exit-concentrated"),
    5: ("mixed_simulation", "Scenario 5: Mixed-Simulation"),
}
EXIT_NODE_BY_DIRECTION = {
    "L_NORTH": "L_N_EXIT",
    "L_SOUTH": "L_S_EXIT",
    "WEST": "W_EXIT",
    "R_NORTH": "R_N_EXIT",
    "R_SOUTH": "R_S_EXIT",
    "EAST": "E_EXIT",
}
DOMINANT_COUNT_RANGE_BY_TOTAL_AGVS = {
    24: (12, 16),
    30: (15, 20),
}
NORMAL_DEPARTURE_STEP = 4
EARLY_DEPARTURE_RANGE = (0, 2)
LATE_DEPARTURE_RANGE = (3, 8)

DEFAULT_OUTPUT_DIR = PROJECT_DIR / "final_output" / "double_cross_test_100_by_agv_fixed_weights"
SETTING_FIELDNAMES = [
    "run_index",
    "scenario_seed",
    "scenario_case_id",
    "scenario_case_name",
    "scenario_case_label",
    "scenario_split",
    "dominant_approach",
    "dominant_approach_count",
    "dominant_exit",
    "dominant_exit_count",
    "early_departures",
    "total_agvs",
    "l_north_agvs",
    "l_south_agvs",
    "west_agvs",
    "r_north_agvs",
    "r_south_agvs",
    "east_agvs",
    "spawn_plan_json",
    "goal_plan_json",
]
RAW_FIELDNAMES = [
    "run_index",
    "scenario_seed",
    "scenario_case_id",
    "scenario_case_name",
    "policy",
    "total_agvs",
    "l_north_agvs",
    "l_south_agvs",
    "west_agvs",
    "r_north_agvs",
    "r_south_agvs",
    "east_agvs",
    "shared_area_capacity",
    "completed",
    "total_time",
    "spawn_plan_json",
    "goal_plan_json",
]
SUMMARY_FIELDNAMES = [
    "scope",
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
COMPARISON_FIELDNAMES = [
    "run_index",
    "scenario_seed",
    "total_agvs",
    "scenario_case_id",
    "scenario_case_name",
    "fcfs_total_time",
    "fixed_heuristic_total_time",
    "bo_heuristic_total_time",
    "pso_heuristic_total_time",
    "best_policy",
    "num_best_policies",
]

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


@dataclass(frozen=True)
class DoubleCrossScenario:
    run_index: int
    seed: int
    case_id: int
    case_name: str
    case_label: str
    split: str
    allocation: dict[str, int]
    spawn_plan: dict[str, list[int]]
    goal_plan: dict[str, list[str]]
    dominant_approach: str
    dominant_approach_count: int
    dominant_exit: str
    dominant_exit_count: int
    early_departures: int
    total_agvs: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and test double-cross scenarios for 24 and 30 AGVs."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--agv-counts", default="24,30")
    parser.add_argument("--scenarios-per-case", type=int, default=20)
    parser.add_argument("--scenario-seed", type=int, default=20260608)
    parser.add_argument("--seed-base", type=int, default=70000)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--admission-window-steps", type=int, default=2)
    parser.add_argument("--shared-area-capacity", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agv_counts = _parse_agv_counts(args.agv_counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.monotonic()
    scenarios_by_agv = _build_scenarios_by_agv(
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
    _write_scenario_csvs(args.output_dir, scenarios_by_agv, all_scenarios)

    weights_by_policy = {
        "fcfs": None,
        "fixed_heuristic": _complete_heuristic_weights(BASIC_WEIGHTS),
        "bo_heuristic": _complete_heuristic_weights(BO_WEIGHTS),
        "pso_heuristic": _complete_heuristic_weights(PSO_WEIGHTS),
    }

    raw_rows = []
    for scenario in all_scenarios:
        for policy in POLICIES:
            metrics = _run_policy(
                scenario=scenario,
                policy=policy,
                weights=weights_by_policy[policy],
                args=args,
            )
            raw_rows.append(_raw_row(scenario, policy, metrics))

    comparison_rows = _comparison_rows(raw_rows)
    summaries = []
    summaries.extend(_summary_rows(raw_rows, scope="overall", group_keys=("policy",)))
    summaries.extend(_summary_rows(raw_rows, scope="by_agv", group_keys=("total_agvs", "policy")))
    summaries.extend(
        _summary_rows(
            raw_rows,
            scope="by_agv_case",
            group_keys=("total_agvs", "scenario_case_id", "scenario_case_name", "policy"),
        )
    )
    runtime_rows = [
        {
            "scenario_count": len(all_scenarios),
            "policy_count": len(POLICIES),
            "total_simulation_runs": len(raw_rows),
            "elapsed_seconds": time.monotonic() - start_time,
            "elapsed_minutes": (time.monotonic() - start_time) / 60.0,
        }
    ]

    _write_csv(_format_rows(raw_rows), args.output_dir / "double_cross_test_raw_runs.csv", RAW_FIELDNAMES)
    _write_csv(_format_rows(comparison_rows), args.output_dir / "double_cross_test_trial_comparison.csv", COMPARISON_FIELDNAMES)
    _write_csv(_format_rows(summaries), args.output_dir / "double_cross_test_summary.csv", SUMMARY_FIELDNAMES)
    _write_csv(_format_rows(runtime_rows), args.output_dir / "double_cross_test_runtime_summary.csv", list(runtime_rows[0]))
    _write_json(
        {
            "layout": "fcfs_double_cross_shared_area_v2",
            "directions": DIRECTIONS,
            "exit_node_by_direction": EXIT_NODE_BY_DIRECTION,
            "agv_counts": agv_counts,
            "scenarios_per_case": args.scenarios_per_case,
            "dominant_count_ranges": DOMINANT_COUNT_RANGE_BY_TOTAL_AGVS,
            "basic_heuristic_weights": weights_by_policy["fixed_heuristic"],
            "bo_heuristic_weights": weights_by_policy["bo_heuristic"],
            "pso_heuristic_weights": weights_by_policy["pso_heuristic"],
            "max_steps": args.max_steps,
            "admission_window_steps": args.admission_window_steps,
            "shared_area_capacity_per_cross": args.shared_area_capacity,
        },
        args.output_dir / "double_cross_test_settings.json",
    )

    print(f"wrote output dir: {args.output_dir.resolve()}")
    print(f"scenarios: {len(all_scenarios)}")
    print(f"simulation runs: {len(raw_rows)}")
    for row in summaries:
        if row["scope"] == "overall":
            print(f"{row['policy']}: avg Total_Time={row['avg_total_time']:.2f}")


def _parse_agv_counts(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    invalid = sorted(set(values) - set(DOMINANT_COUNT_RANGE_BY_TOTAL_AGVS))
    if invalid:
        raise ValueError(f"unsupported double-cross AGV count(s): {invalid}")
    return values


def _build_scenarios_by_agv(
    agv_counts: list[int],
    scenarios_per_case: int,
    scenario_seed: int,
    seed_base: int,
) -> dict[int, list[DoubleCrossScenario]]:
    rng = random.Random(scenario_seed)
    scenarios_by_agv = {}
    run_index = 0
    for total_agvs in agv_counts:
        scenarios = []
        for case_id in CASE_IDS:
            for variant_index in range(scenarios_per_case):
                scenarios.append(
                    _build_scenario(
                        case_id=case_id,
                        variant_index=variant_index,
                        run_index=run_index,
                        seed=seed_base + run_index,
                        total_agvs=total_agvs,
                        rng=rng,
                    )
                )
                run_index += 1
        scenarios_by_agv[total_agvs] = scenarios
    return scenarios_by_agv


def _build_scenario(
    case_id: int,
    variant_index: int,
    run_index: int,
    seed: int,
    total_agvs: int,
    rng: random.Random,
) -> DoubleCrossScenario:
    case_name, case_label = SCENARIO_CASES[case_id]
    dominant_approach = ""
    dominant_exit = ""
    dominant_approach_count = 0
    dominant_exit_count = 0
    early_departures = 0

    if case_id == 1:
        allocation = _balanced_counts(total_agvs)
        exit_counts = _balanced_counts(total_agvs)
        spawn_plan = _staggered_spawn_plan(allocation)
    elif case_id == 2:
        allocation = _balanced_counts(total_agvs)
        exit_counts = _balanced_counts(total_agvs)
        early_departures = total_agvs // 2
        spawn_plan = _burst_spawn_plan(allocation, early_departures, rng)
    elif case_id == 3:
        dominant_approach = _rotating_direction(variant_index)
        allocation = _skewed_counts(dominant_approach, total_agvs, rng)
        dominant_approach_count = allocation[dominant_approach]
        exit_counts = _balanced_counts(total_agvs)
        spawn_plan = _staggered_spawn_plan(allocation)
    elif case_id == 4:
        dominant_exit = _rotating_direction(variant_index)
        allocation = _balanced_counts(total_agvs)
        exit_counts = _skewed_counts(dominant_exit, total_agvs, rng)
        dominant_exit_count = exit_counts[dominant_exit]
        spawn_plan = _staggered_spawn_plan(allocation)
    elif case_id == 5:
        dominant_approach = _rotating_direction(variant_index)
        dominant_exit = _rotating_direction(variant_index + 1)
        allocation = _skewed_counts(dominant_approach, total_agvs, rng)
        exit_counts = _skewed_counts(dominant_exit, total_agvs, rng)
        dominant_approach_count = allocation[dominant_approach]
        dominant_exit_count = exit_counts[dominant_exit]
        early_departures = total_agvs // 2
        spawn_plan = _burst_spawn_plan(allocation, early_departures, rng)
    else:
        raise ValueError(case_id)

    return DoubleCrossScenario(
        run_index=run_index,
        seed=seed,
        case_id=case_id,
        case_name=case_name,
        case_label=case_label,
        split="test_only" if case_id == 5 else "train_test",
        allocation=allocation,
        spawn_plan=spawn_plan,
        goal_plan=_goal_plan_from_exit_counts(allocation, exit_counts, rng),
        dominant_approach=dominant_approach,
        dominant_approach_count=dominant_approach_count,
        dominant_exit=dominant_exit,
        dominant_exit_count=dominant_exit_count,
        early_departures=early_departures,
        total_agvs=total_agvs,
    )


def _balanced_counts(total_agvs: int) -> dict[str, int]:
    if total_agvs % len(DIRECTIONS) != 0:
        raise ValueError("double-cross balanced scenarios require AGVs divisible by 6")
    return {direction: total_agvs // len(DIRECTIONS) for direction in DIRECTIONS}


def _skewed_counts(dominant_direction: str, total_agvs: int, rng: random.Random) -> dict[str, int]:
    lower, upper = DOMINANT_COUNT_RANGE_BY_TOTAL_AGVS[total_agvs]
    dominant_count = rng.randint(lower, upper)
    other_directions = [direction for direction in DIRECTIONS if direction != dominant_direction]
    other_counts = _random_positive_counts(total_agvs - dominant_count, len(other_directions), rng)
    counts = {dominant_direction: dominant_count}
    counts.update(dict(zip(other_directions, other_counts)))
    return counts


def _random_positive_counts(total: int, bucket_count: int, rng: random.Random) -> list[int]:
    max_count = total // 2
    candidates = [
        counts
        for counts in _positive_compositions(total, bucket_count)
        if all(count <= max_count for count in counts)
    ]
    if not candidates:
        raise ValueError(f"cannot distribute {total} into {bucket_count} positive buckets")
    return list(rng.choice(candidates))


def _positive_compositions(total: int, bucket_count: int) -> list[tuple[int, ...]]:
    if bucket_count == 1:
        return [(total,)]
    output = []
    for count in range(1, total - bucket_count + 2):
        for tail in _positive_compositions(total - count, bucket_count - 1):
            output.append((count, *tail))
    return output


def _rotating_direction(variant_index: int) -> str:
    return DOMINANT_ROTATION[variant_index % len(DOMINANT_ROTATION)]


def _staggered_spawn_plan(allocation: dict[str, int]) -> dict[str, list[int]]:
    return {
        direction: [index * NORMAL_DEPARTURE_STEP for index in range(allocation[direction])]
        for direction in DIRECTIONS
    }


def _burst_spawn_plan(
    allocation: dict[str, int],
    early_departures: int,
    rng: random.Random,
) -> dict[str, list[int]]:
    robots = [direction for direction in DIRECTIONS for _ in range(allocation[direction])]
    rng.shuffle(robots)
    plan = {direction: [] for direction in DIRECTIONS}
    for index, direction in enumerate(robots):
        if index < early_departures:
            start = rng.randint(*EARLY_DEPARTURE_RANGE)
        else:
            start = rng.randint(*LATE_DEPARTURE_RANGE)
        plan[direction].append(start)
    return {direction: sorted(plan[direction]) for direction in DIRECTIONS}


def _goal_plan_from_exit_counts(
    allocation: dict[str, int],
    exit_counts: dict[str, int],
    rng: random.Random,
) -> dict[str, list[str]]:
    entries = [direction for direction in DIRECTIONS for _ in range(allocation[direction])]
    rng.shuffle(entries)
    quotas = dict(exit_counts)
    plan = {direction: [] for direction in DIRECTIONS}

    def feasible(next_index: int) -> bool:
        remaining_entries = entries[next_index:]
        for exit_direction, remaining_quota in quotas.items():
            available = sum(entry_direction != exit_direction for entry_direction in remaining_entries)
            if remaining_quota > available:
                return False
        return True

    def assign(index: int) -> bool:
        if index == len(entries):
            return all(value == 0 for value in quotas.values())
        entry_direction = entries[index]
        candidates = [
            direction
            for direction in DOMINANT_ROTATION
            if direction != entry_direction and quotas[direction] > 0
        ]
        rng.shuffle(candidates)
        candidates.sort(key=lambda direction: quotas[direction], reverse=True)
        for exit_direction in candidates:
            quotas[exit_direction] -= 1
            plan[entry_direction].append(EXIT_NODE_BY_DIRECTION[exit_direction])
            if feasible(index + 1) and assign(index + 1):
                return True
            plan[entry_direction].pop()
            quotas[exit_direction] += 1
        return False

    if not feasible(0) or not assign(0):
        raise ValueError(f"Could not build feasible goal plan: {allocation=} {exit_counts=}")
    return {direction: list(plan[direction]) for direction in DIRECTIONS}


def _run_policy(
    scenario: DoubleCrossScenario,
    policy: str,
    weights: dict[str, float] | None,
    args: argparse.Namespace,
) -> dict:
    layout = build_fcfs_double_cross_shared_area_layout(corridor_length=args.corridor_length)
    experiment = FCFSCrossExperiment(
        layout=layout,
        robots_by_direction=scenario.allocation,
        random_seed=scenario.seed,
        corridor_length=args.corridor_length,
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


def _setting_row(scenario: DoubleCrossScenario) -> dict:
    row = {
        "run_index": scenario.run_index,
        "scenario_seed": scenario.seed,
        "scenario_case_id": scenario.case_id,
        "scenario_case_name": scenario.case_name,
        "scenario_case_label": scenario.case_label,
        "scenario_split": scenario.split,
        "dominant_approach": scenario.dominant_approach,
        "dominant_approach_count": scenario.dominant_approach_count,
        "dominant_exit": scenario.dominant_exit,
        "dominant_exit_count": scenario.dominant_exit_count,
        "early_departures": scenario.early_departures,
        "total_agvs": scenario.total_agvs,
        "spawn_plan_json": json.dumps(scenario.spawn_plan, sort_keys=True),
        "goal_plan_json": json.dumps(scenario.goal_plan, sort_keys=True),
    }
    row.update(_allocation_columns(scenario.allocation))
    return row


def _raw_row(scenario: DoubleCrossScenario, policy: str, metrics: dict) -> dict:
    row = {
        "run_index": scenario.run_index,
        "scenario_seed": scenario.seed,
        "scenario_case_id": scenario.case_id,
        "scenario_case_name": scenario.case_name,
        "policy": policy,
        "total_agvs": scenario.total_agvs,
        "shared_area_capacity": metrics["shared_area_capacity"],
        "completed": metrics["completed"],
        "total_time": metrics["total_time"],
        "spawn_plan_json": json.dumps(scenario.spawn_plan, sort_keys=True),
        "goal_plan_json": json.dumps(scenario.goal_plan, sort_keys=True),
    }
    row.update(_allocation_columns(scenario.allocation))
    return row


def _allocation_columns(allocation: dict[str, int]) -> dict[str, int]:
    return {
        "l_north_agvs": allocation["L_NORTH"],
        "l_south_agvs": allocation["L_SOUTH"],
        "west_agvs": allocation["WEST"],
        "r_north_agvs": allocation["R_NORTH"],
        "r_south_agvs": allocation["R_SOUTH"],
        "east_agvs": allocation["EAST"],
    }


def _comparison_rows(raw_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(dict)
    for row in raw_rows:
        grouped[int(row["run_index"])][row["policy"]] = row
    output = []
    for run_index, policies in sorted(grouped.items()):
        times = {policy: int(policies[policy]["total_time"]) for policy in POLICIES}
        best_time = min(times.values())
        best_policies = [policy for policy, total_time in times.items() if total_time == best_time]
        any_row = next(iter(policies.values()))
        output.append(
            {
                "run_index": run_index,
                "scenario_seed": any_row["scenario_seed"],
                "total_agvs": any_row["total_agvs"],
                "scenario_case_id": any_row["scenario_case_id"],
                "scenario_case_name": any_row["scenario_case_name"],
                "fcfs_total_time": times["fcfs"],
                "fixed_heuristic_total_time": times["fixed_heuristic"],
                "bo_heuristic_total_time": times["bo_heuristic"],
                "pso_heuristic_total_time": times["pso_heuristic"],
                "best_policy": ",".join(best_policies),
                "num_best_policies": len(best_policies),
            }
        )
    return output


def _summary_rows(raw_rows: list[dict], scope: str, group_keys: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[tuple(row.get(key, "") for key in group_keys)].append(row)

    output = []
    for key, rows in sorted(grouped.items()):
        values = [int(row["total_time"]) for row in rows]
        summary = {
            "scope": scope,
            "total_agvs": "",
            "scenario_case_id": "",
            "scenario_case_name": "",
            "policy": "",
        }
        for index, group_key in enumerate(group_keys):
            summary[group_key] = key[index]
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


def _percentile(values: list[int], percentile: float) -> float:
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _write_scenario_csvs(
    output_dir: Path,
    scenarios_by_agv: dict[int, list[DoubleCrossScenario]],
    all_scenarios: list[DoubleCrossScenario],
) -> None:
    for total_agvs, scenarios in scenarios_by_agv.items():
        _write_csv(
            [_setting_row(scenario) for scenario in scenarios],
            output_dir / f"double_cross_test_scenarios_100_agv{total_agvs}.csv",
            SETTING_FIELDNAMES,
        )
    _write_csv(
        [_setting_row(scenario) for scenario in all_scenarios],
        output_dir / "double_cross_test_scenarios_200_all_agv_counts.csv",
        SETTING_FIELDNAMES,
    )


def _format_rows(rows: list[dict]) -> list[dict]:
    return [_format_row(row) for row in rows]


def _format_row(row: dict) -> dict:
    formatted = {}
    for key, value in row.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.2f}"
        else:
            formatted[key] = value
    return formatted


def _write_csv(rows: list[dict], output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: dict, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
