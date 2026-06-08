from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rware.fcfs_cross_simulation import FCFSCrossExperiment  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "final_output"
    / "transfer_generalization_test_scenarios"
)
DEFAULT_DESKTOP_DIR = (
    Path.home()
    / "OneDrive"
    / "바탕 화면"
    / "transfer_generalization_test_scenarios"
)

SINGLE_DIRECTIONS = ("NORTH", "SOUTH", "WEST", "EAST")
DOUBLE_DIRECTIONS = ("L_NORTH", "L_SOUTH", "WEST", "R_NORTH", "R_SOUTH", "EAST")
SINGLE_AGV_COUNTS = (20, 24, 28)
DOUBLE_AGV_COUNTS = (24, 30, 36)
NORMAL_DEPARTURE_STEP = 4
EARLY_DEPARTURE_RANGE = (0, 2)
LATE_DEPARTURE_RANGE = (3, 8)

SCENARIO_CASES = {
    5: {
        "name": "burst_direction_exit_combined",
        "label": "Scenario 5: 2+3+4 arrival burst + direction skew + exit concentration",
        "features": ("arrival_burst", "direction_skew", "exit_concentration"),
    },
    6: {
        "name": "burst_direction_combined",
        "label": "Scenario 6: 2+3 arrival burst + direction skew",
        "features": ("arrival_burst", "direction_skew"),
    },
    7: {
        "name": "burst_exit_combined",
        "label": "Scenario 7: 2+4 arrival burst + exit concentration",
        "features": ("arrival_burst", "exit_concentration"),
    },
    8: {
        "name": "direction_exit_combined",
        "label": "Scenario 8: 3+4 direction skew + exit concentration",
        "features": ("direction_skew", "exit_concentration"),
    },
}

FIELDNAMES = [
    "layout_key",
    "run_index",
    "scenario_seed",
    "total_agvs",
    "scenario_case_id",
    "scenario_case_name",
    "scenario_case_label",
    "scenario_split",
    "combined_features",
    "dominant_approach",
    "dominant_approach_count",
    "dominant_exit",
    "dominant_exit_count",
    "early_departures",
    "north_agvs",
    "south_agvs",
    "west_agvs",
    "east_agvs",
    "l_north_agvs",
    "l_south_agvs",
    "r_north_agvs",
    "r_south_agvs",
    "direction_count_json",
    "spawn_plan_json",
    "goal_plan_json",
]


@dataclass(frozen=True)
class ScenarioRow:
    layout_key: str
    run_index: int
    seed: int
    total_agvs: int
    case_id: int
    allocation: dict[str, int]
    spawn_plan: dict[str, list[int]]
    goal_plan: dict[str, list[str]]
    dominant_approach: str = ""
    dominant_approach_count: int = 0
    dominant_exit: str = ""
    dominant_exit_count: int = 0
    early_departures: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build transfer/generalization test scenarios. Train remains the "
            "existing single-cross 400 scenarios; this script creates only "
            "new test cases 5, 6, 7, and 8."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenarios-per-case", type=int, default=50)
    parser.add_argument("--single-agv-counts", default="20,24,28")
    parser.add_argument("--double-agv-counts", default="24,30,36")
    parser.add_argument("--single-seed-base", type=int, default=81000)
    parser.add_argument("--double-seed-base", type=int, default=91000)
    parser.add_argument("--scenario-seed", type=int, default=20260608)
    parser.add_argument("--copy-to-desktop", action="store_true")
    parser.add_argument("--desktop-dir", type=Path, default=DEFAULT_DESKTOP_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    single_counts = _parse_ints(args.single_agv_counts)
    double_counts = _parse_ints(args.double_agv_counts)

    single_rows = _build_layout_rows(
        layout_key="single_cross",
        directions=SINGLE_DIRECTIONS,
        agv_counts=single_counts,
        scenarios_per_case=args.scenarios_per_case,
        seed_base=args.single_seed_base,
        scenario_seed=args.scenario_seed,
        exit_node_by_direction=_single_exit_node_by_direction(),
    )
    double_rows = _build_layout_rows(
        layout_key="double_cross",
        directions=DOUBLE_DIRECTIONS,
        agv_counts=double_counts,
        scenarios_per_case=args.scenarios_per_case,
        seed_base=args.double_seed_base,
        scenario_seed=args.scenario_seed + 1009,
        exit_node_by_direction=_double_exit_node_by_direction(),
    )

    _write_layout_files(output_dir, "single_cross", single_rows, single_counts)
    _write_layout_files(output_dir, "double_cross", double_rows, double_counts)
    _write_csv(output_dir / "transfer_test_scenarios_all_layouts.csv", single_rows + double_rows)
    _write_readme(output_dir, single_rows, double_rows, args)

    if args.copy_to_desktop:
        args.desktop_dir.mkdir(parents=True, exist_ok=True)
        for path in output_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, args.desktop_dir / path.name)
        print(f"copied to desktop: {args.desktop_dir}")

    print(f"wrote scenarios: {output_dir.resolve()}")
    _print_summary("single_cross", single_rows)
    _print_summary("double_cross", double_rows)


def _build_layout_rows(
    layout_key: str,
    directions: tuple[str, ...],
    agv_counts: list[int],
    scenarios_per_case: int,
    seed_base: int,
    scenario_seed: int,
    exit_node_by_direction: dict[str, str],
) -> list[ScenarioRow]:
    rng = random.Random(scenario_seed)
    rows: list[ScenarioRow] = []
    run_index = 0
    for total_agvs in agv_counts:
        for case_id in SCENARIO_CASES:
            for variant_index in range(scenarios_per_case):
                rows.append(
                    _build_scenario(
                        layout_key=layout_key,
                        directions=directions,
                        case_id=case_id,
                        variant_index=variant_index,
                        run_index=run_index,
                        seed=seed_base + run_index,
                        total_agvs=total_agvs,
                        rng=rng,
                        exit_node_by_direction=exit_node_by_direction,
                    )
                )
                run_index += 1
    return rows


def _build_scenario(
    layout_key: str,
    directions: tuple[str, ...],
    case_id: int,
    variant_index: int,
    run_index: int,
    seed: int,
    total_agvs: int,
    rng: random.Random,
    exit_node_by_direction: dict[str, str],
) -> ScenarioRow:
    features = set(SCENARIO_CASES[case_id]["features"])
    dominant_approach = ""
    dominant_exit = ""
    dominant_approach_count = 0
    dominant_exit_count = 0
    early_departures = 0

    if "direction_skew" in features:
        dominant_approach = _rotating_direction(directions, variant_index)
        allocation = _skewed_counts(directions, dominant_approach, total_agvs, rng)
        dominant_approach_count = allocation[dominant_approach]
    else:
        allocation = _balanced_counts(directions, total_agvs)

    if "exit_concentration" in features:
        offset = 1 if "direction_skew" in features else 0
        dominant_exit = _rotating_direction(directions, variant_index + offset)
        exit_counts = _skewed_counts(directions, dominant_exit, total_agvs, rng)
        dominant_exit_count = exit_counts[dominant_exit]
    else:
        exit_counts = _balanced_counts(directions, total_agvs)

    if "arrival_burst" in features:
        early_departures = total_agvs // 2
        spawn_plan = _burst_spawn_plan(directions, allocation, early_departures, rng)
    else:
        spawn_plan = _staggered_spawn_plan(directions, allocation)

    return ScenarioRow(
        layout_key=layout_key,
        run_index=run_index,
        seed=seed,
        total_agvs=total_agvs,
        case_id=case_id,
        allocation=allocation,
        spawn_plan=spawn_plan,
        goal_plan=_goal_plan_from_exit_counts(
            directions=directions,
            allocation=allocation,
            exit_counts=exit_counts,
            rng=rng,
            exit_node_by_direction=exit_node_by_direction,
        ),
        dominant_approach=dominant_approach,
        dominant_approach_count=dominant_approach_count,
        dominant_exit=dominant_exit,
        dominant_exit_count=dominant_exit_count,
        early_departures=early_departures,
    )


def _balanced_counts(directions: tuple[str, ...], total_agvs: int) -> dict[str, int]:
    if total_agvs % len(directions) != 0:
        raise ValueError(f"total_agvs={total_agvs} is not divisible by {len(directions)}")
    return {direction: total_agvs // len(directions) for direction in directions}


def _skewed_counts(
    directions: tuple[str, ...],
    dominant_direction: str,
    total_agvs: int,
    rng: random.Random,
) -> dict[str, int]:
    lower, upper = _dominant_range(total_agvs, len(directions))
    dominant_count = rng.randint(lower, upper)
    remainder = total_agvs - dominant_count
    other_directions = [direction for direction in directions if direction != dominant_direction]
    other_counts = _random_positive_counts(remainder, len(other_directions), rng)
    counts = {dominant_direction: dominant_count}
    counts.update(dict(zip(other_directions, other_counts)))
    return {direction: counts[direction] for direction in directions}


def _dominant_range(total_agvs: int, direction_count: int) -> tuple[int, int]:
    lower = total_agvs // 2
    if direction_count == 4:
        upper = lower + 4
    else:
        upper = int(round(total_agvs * 2 / 3))
    upper = min(upper, total_agvs - (direction_count - 1))
    if lower > upper:
        lower = upper
    return lower, upper


def _random_positive_counts(total: int, bucket_count: int, rng: random.Random) -> list[int]:
    max_count = total // 2
    candidates = [
        counts
        for counts in _positive_count_compositions(total, bucket_count)
        if all(1 <= count <= max_count for count in counts)
    ]
    if not candidates:
        raise ValueError(
            f"Cannot distribute total={total} into {bucket_count} positive buckets "
            f"with every bucket <= {max_count}"
        )
    return list(rng.choice(candidates))


def _positive_count_compositions(total: int, bucket_count: int) -> list[tuple[int, ...]]:
    if bucket_count == 1:
        return [(total,)]
    rows: list[tuple[int, ...]] = []
    for head in range(1, total - bucket_count + 2):
        for tail in _positive_count_compositions(total - head, bucket_count - 1):
            rows.append((head, *tail))
    return rows


def _staggered_spawn_plan(
    directions: tuple[str, ...],
    allocation: dict[str, int],
) -> dict[str, list[int]]:
    return {
        direction: [index * NORMAL_DEPARTURE_STEP for index in range(allocation[direction])]
        for direction in directions
    }


def _burst_spawn_plan(
    directions: tuple[str, ...],
    allocation: dict[str, int],
    early_departures: int,
    rng: random.Random,
) -> dict[str, list[int]]:
    robots = [direction for direction in directions for _ in range(allocation[direction])]
    rng.shuffle(robots)
    plan = {direction: [] for direction in directions}
    for index, direction in enumerate(robots):
        if index < early_departures:
            spawn_step = rng.randint(*EARLY_DEPARTURE_RANGE)
        else:
            spawn_step = rng.randint(*LATE_DEPARTURE_RANGE)
        plan[direction].append(spawn_step)
    return {direction: sorted(plan[direction]) for direction in directions}


def _goal_plan_from_exit_counts(
    directions: tuple[str, ...],
    allocation: dict[str, int],
    exit_counts: dict[str, int],
    rng: random.Random,
    exit_node_by_direction: dict[str, str],
) -> dict[str, list[str]]:
    entries = [direction for direction in directions for _ in range(allocation[direction])]
    rng.shuffle(entries)
    quotas = dict(exit_counts)
    plan = {direction: [] for direction in directions}

    def feasible(next_index: int) -> bool:
        remaining_entries = entries[next_index:]
        for exit_direction, remaining_quota in quotas.items():
            available = sum(entry_direction != exit_direction for entry_direction in remaining_entries)
            if remaining_quota > available:
                return False
        return True

    for index, entry_direction in enumerate(entries):
        candidate_exits = [
            exit_direction
            for exit_direction in directions
            if exit_direction != entry_direction and quotas[exit_direction] > 0
        ]
        rng.shuffle(candidate_exits)
        selected_exit = None
        for exit_direction in candidate_exits:
            quotas[exit_direction] -= 1
            if feasible(index + 1):
                selected_exit = exit_direction
                break
            quotas[exit_direction] += 1
        if selected_exit is None:
            raise ValueError("Could not build feasible goal plan")
        plan[entry_direction].append(exit_node_by_direction[selected_exit])

    return {direction: list(plan[direction]) for direction in directions}


def _rotating_direction(directions: tuple[str, ...], variant_index: int) -> str:
    return directions[variant_index % len(directions)]


def _single_exit_node_by_direction() -> dict[str, str]:
    return {
        exit_direction: exit_node
        for exit_node, exit_direction in FCFSCrossExperiment.EXIT_DIRECTION_BY_NODE.items()
    }


def _double_exit_node_by_direction() -> dict[str, str]:
    return {
        exit_direction: exit_node
        for exit_node, exit_direction in FCFSCrossExperiment.DOUBLE_EXIT_DIRECTION_BY_NODE.items()
    }


def _write_layout_files(
    output_dir: Path,
    layout_key: str,
    rows: list[ScenarioRow],
    agv_counts: list[int],
) -> None:
    for total_agvs in agv_counts:
        selected = [row for row in rows if row.total_agvs == total_agvs]
        _write_csv(output_dir / f"{layout_key}_test_scenarios_5_8_agv{total_agvs}.csv", selected)
    _write_csv(output_dir / f"{layout_key}_test_scenarios_5_8_all_agv_counts.csv", rows)


def _write_csv(path: Path, rows: list[ScenarioRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(_scenario_to_row(row))


def _scenario_to_row(row: ScenarioRow) -> dict:
    case = SCENARIO_CASES[row.case_id]
    allocation = row.allocation
    return {
        "layout_key": row.layout_key,
        "run_index": row.run_index,
        "scenario_seed": row.seed,
        "total_agvs": row.total_agvs,
        "scenario_case_id": row.case_id,
        "scenario_case_name": case["name"],
        "scenario_case_label": case["label"],
        "scenario_split": "test_only",
        "combined_features": "+".join(case["features"]),
        "dominant_approach": row.dominant_approach,
        "dominant_approach_count": row.dominant_approach_count,
        "dominant_exit": row.dominant_exit,
        "dominant_exit_count": row.dominant_exit_count,
        "early_departures": row.early_departures,
        "north_agvs": allocation.get("NORTH", 0),
        "south_agvs": allocation.get("SOUTH", 0),
        "west_agvs": allocation.get("WEST", 0),
        "east_agvs": allocation.get("EAST", 0),
        "l_north_agvs": allocation.get("L_NORTH", 0),
        "l_south_agvs": allocation.get("L_SOUTH", 0),
        "r_north_agvs": allocation.get("R_NORTH", 0),
        "r_south_agvs": allocation.get("R_SOUTH", 0),
        "direction_count_json": json.dumps(row.allocation, sort_keys=True),
        "spawn_plan_json": json.dumps(row.spawn_plan, sort_keys=True),
        "goal_plan_json": json.dumps(row.goal_plan, sort_keys=True),
    }


def _write_readme(
    output_dir: Path,
    single_rows: list[ScenarioRow],
    double_rows: list[ScenarioRow],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Transfer Generalization Test Scenarios",
        "",
        "Train layout: single-cross only, using the existing 400 training scenarios.",
        "Test layouts: single-cross and double-cross.",
        "",
        "Scenario cases:",
    ]
    for case_id, case in SCENARIO_CASES.items():
        lines.append(f"- {case_id}: {case['label']}")
    lines.extend(
        [
            "",
            f"Scenarios per case: {args.scenarios_per_case}",
            f"Single-cross AGV counts: {args.single_agv_counts}",
            f"Double-cross AGV counts: {args.double_agv_counts}",
            "",
            f"Single rows: {len(single_rows)}",
            f"Double rows: {len(double_rows)}",
            "",
            "Burst scenarios set half of the AGVs to early departures sampled from 0-2 steps; the rest are sampled from 3-8 steps.",
            "Direction-skew and exit-concentration scenarios sample the dominant direction count and randomly distribute the remainder.",
            "Every non-dominant point receives at least one AGV, and no non-dominant point exceeds half of the non-dominant remainder.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(layout_key: str, rows: list[ScenarioRow]) -> None:
    print(f"{layout_key}: {len(rows)} rows")
    for total_agvs in sorted({row.total_agvs for row in rows}):
        count = sum(1 for row in rows if row.total_agvs == total_agvs)
        print(f"  AGV {total_agvs}: {count}")
    for case_id in SCENARIO_CASES:
        count = sum(1 for row in rows if row.case_id == case_id)
        print(f"  scenario {case_id}: {count}")


def _parse_ints(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("at least one AGV count is required")
    return values


if __name__ == "__main__":
    main()
