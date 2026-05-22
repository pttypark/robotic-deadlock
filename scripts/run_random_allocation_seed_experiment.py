"""Compare FCFS and heuristic on fixed layout with random AGV allocations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
from pathlib import Path

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from rware.fcfs_cross_simulation import FCFSCrossExperiment


DIRECTIONS = ["NORTH", "SOUTH", "WEST", "EAST"]
POLICIES = ["fcfs", "heuristic"]
METRICS = [
    "total_time",
    "total_wait_time",
    "avg_wait_time",
    "max_wait_time",
    "utilization",
]

SETTINGS_FIELDNAMES = [
    "set_id",
    "total_agvs",
    "north_agvs",
    "south_agvs",
    "west_agvs",
    "east_agvs",
    "allocation_json",
    "layout",
    "corridor_length",
    "west_exit_extension",
    "spawn_gap_steps",
    "admission_window_steps",
    "runs_per_set",
]

RUN_FIELDNAMES = [
    "set_id",
    "run_index",
    "seed",
    "policy_type",
    "total_agvs",
    "north_agvs",
    "south_agvs",
    "west_agvs",
    "east_agvs",
    "layout",
    "completed",
    "total_time",
    "total_wait_time",
    "avg_wait_time",
    "max_wait_time",
    "utilization",
    "shared_occupied_steps",
]

SUMMARY_FIELDNAMES = [
    "set_id",
    "policy_type",
    "total_agvs",
    "north_agvs",
    "south_agvs",
    "west_agvs",
    "east_agvs",
    "runs",
    "completed_runs",
    "avg_total_time",
    "std_total_time",
    "avg_total_wait_time",
    "std_total_wait_time",
    "avg_avg_wait_time",
    "std_avg_wait_time",
    "avg_max_wait_time",
    "std_max_wait_time",
    "avg_utilization",
    "std_utilization",
]

COMPARISON_FIELDNAMES = [
    "set_id",
    "total_agvs",
    "allocation",
    "fcfs_avg_total_time",
    "heuristic_avg_total_time",
    "delta_total_time",
    "total_time_improvement_pct",
    "fcfs_avg_total_wait_time",
    "heuristic_avg_total_wait_time",
    "delta_total_wait_time",
    "fcfs_avg_avg_wait_time",
    "heuristic_avg_avg_wait_time",
    "delta_avg_wait_time",
    "fcfs_avg_max_wait_time",
    "heuristic_avg_max_wait_time",
    "delta_max_wait_time",
    "fcfs_avg_utilization",
    "heuristic_avg_utilization",
    "delta_utilization",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed cross layout experiment: randomly sample AGV counts/allocation "
            "sets, then run each set across many route seeds."
        )
    )
    parser.add_argument("--num-sets", type=int, default=5)
    parser.add_argument("--runs-per-set", type=int, default=30)
    parser.add_argument("--total-agv-candidates", default="8,12,16,20,24")
    parser.add_argument("--total-selection-mode", choices=["shuffle", "cycle"], default="shuffle")
    parser.add_argument("--set-seed", type=int, default=2026)
    parser.add_argument("--run-seed-base", type=int, default=7000)
    parser.add_argument("--heuristic-weights-json", default="")
    parser.add_argument("--heuristic-weights-file", default="")
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--spawn-gap-steps", type=int, default=1)
    parser.add_argument("--admission-window-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--output-dir",
        default=str(Path("final_output") / "fixed_layout_random_sets"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    totals = _parse_ints(args.total_agv_candidates)
    settings = _sample_settings(
        num_sets=args.num_sets,
        total_candidates=totals,
        rng=random.Random(args.set_seed),
        total_selection_mode=args.total_selection_mode,
        corridor_length=args.corridor_length,
        west_exit_extension=args.west_exit_extension,
        spawn_gap_steps=args.spawn_gap_steps,
        admission_window_steps=args.admission_window_steps,
        runs_per_set=args.runs_per_set,
    )

    run_rows: list[dict] = []
    heuristic_weights = _load_heuristic_weights(
        raw_json=args.heuristic_weights_json,
        file_path=args.heuristic_weights_file,
    )
    for setting in settings:
        allocation = _allocation_from_setting(setting)
        for run_index in range(args.runs_per_set):
            seed = args.run_seed_base + setting["set_id"] * 1000 + run_index
            for policy_type in POLICIES:
                metrics = _run_single(
                    allocation=allocation,
                    seed=seed,
                    policy_type=policy_type,
                    corridor_length=args.corridor_length,
                    west_exit_extension=args.west_exit_extension,
                    spawn_gap_steps=args.spawn_gap_steps,
                    admission_window_steps=args.admission_window_steps,
                    max_steps=args.max_steps,
                    heuristic_weights=heuristic_weights,
                )
                run_rows.append(
                    {
                        "set_id": setting["set_id"],
                        "run_index": run_index,
                        "seed": seed,
                        "policy_type": policy_type,
                        "total_agvs": setting["total_agvs"],
                        "north_agvs": allocation["NORTH"],
                        "south_agvs": allocation["SOUTH"],
                        "west_agvs": allocation["WEST"],
                        "east_agvs": allocation["EAST"],
                        "layout": metrics["layout"],
                        "completed": metrics["completed"],
                        "total_time": metrics["total_time"],
                        "total_wait_time": metrics["total_wait_time"],
                        "avg_wait_time": metrics["avg_wait_time"],
                        "max_wait_time": metrics["max_wait_time"],
                        "utilization": metrics["utilization"],
                        "shared_occupied_steps": metrics["shared_occupied_steps"],
                    }
                )
        print(
            f"set={setting['set_id']} total={setting['total_agvs']} "
            f"allocation={allocation} runs={args.runs_per_set}"
        )

    summary_rows = _summarize(run_rows)
    comparison_rows = _compare(summary_rows)
    _write_csv(settings, output_dir / "experiment_settings.csv", SETTINGS_FIELDNAMES)
    _write_csv(run_rows, output_dir / "raw_runs.csv", RUN_FIELDNAMES)
    _write_csv(summary_rows, output_dir / "policy_summary_by_set.csv", SUMMARY_FIELDNAMES)
    _write_csv(comparison_rows, output_dir / "policy_comparison_by_set.csv", COMPARISON_FIELDNAMES)
    _write_readme(output_dir, settings, comparison_rows)

    print(f"settings: {(output_dir / 'experiment_settings.csv').resolve()}")
    print(f"raw runs: {(output_dir / 'raw_runs.csv').resolve()}")
    print(f"summary: {(output_dir / 'policy_summary_by_set.csv').resolve()}")
    print(f"comparison: {(output_dir / 'policy_comparison_by_set.csv').resolve()}")


def _sample_settings(
    num_sets: int,
    total_candidates: list[int],
    rng: random.Random,
    total_selection_mode: str,
    corridor_length: int,
    west_exit_extension: int,
    spawn_gap_steps: int,
    admission_window_steps: int,
    runs_per_set: int,
) -> list[dict]:
    if num_sets < 1:
        raise ValueError("--num-sets must be at least 1")
    if not total_candidates:
        raise ValueError("--total-agv-candidates must not be empty")
    if any(total < len(DIRECTIONS) for total in total_candidates):
        raise ValueError("total AGV candidates must be at least 4")

    shuffled_totals = list(total_candidates)
    if total_selection_mode == "shuffle":
        rng.shuffle(shuffled_totals)
    settings = []
    for index in range(num_sets):
        if total_selection_mode == "cycle":
            total_agvs = total_candidates[index % len(total_candidates)]
        else:
            total_agvs = (
                shuffled_totals[index]
                if index < len(shuffled_totals)
                else rng.choice(total_candidates)
            )
        allocation = _random_positive_allocation(total_agvs, rng)
        layout = (
            "fcfs_cross_shared_area_v1"
            if corridor_length == 5 and west_exit_extension == 0
            else f"fcfs_cross_shared_area_corridor{corridor_length}"
        )
        if west_exit_extension:
            layout += f"_westtail{west_exit_extension}"
        settings.append(
            {
                "set_id": index + 1,
                "total_agvs": total_agvs,
                "north_agvs": allocation["NORTH"],
                "south_agvs": allocation["SOUTH"],
                "west_agvs": allocation["WEST"],
                "east_agvs": allocation["EAST"],
                "allocation_json": json.dumps(allocation, sort_keys=True),
                "layout": layout,
                "corridor_length": corridor_length,
                "west_exit_extension": west_exit_extension,
                "spawn_gap_steps": spawn_gap_steps,
                "admission_window_steps": admission_window_steps,
                "runs_per_set": runs_per_set,
            }
        )
    return settings


def _random_positive_allocation(total_agvs: int, rng: random.Random) -> dict[str, int]:
    remaining = total_agvs - len(DIRECTIONS)
    counts = {direction: 1 for direction in DIRECTIONS}
    for _ in range(remaining):
        counts[rng.choice(DIRECTIONS)] += 1
    return counts


def _run_single(
    allocation: dict[str, int],
    seed: int,
    policy_type: str,
    corridor_length: int,
    west_exit_extension: int,
    spawn_gap_steps: int,
    admission_window_steps: int,
    max_steps: int,
    heuristic_weights: dict[str, float] | None,
) -> dict:
    experiment = FCFSCrossExperiment(
        robots_by_direction=allocation,
        random_seed=seed,
        corridor_length=corridor_length,
        west_exit_extension=west_exit_extension,
        spawn_gap_steps=spawn_gap_steps,
        admission_window_steps=admission_window_steps,
        policy_type=policy_type,
        heuristic_weights=heuristic_weights if policy_type == "heuristic" else None,
    )
    return experiment.run(max_steps=max_steps)


def _load_heuristic_weights(raw_json: str, file_path: str) -> dict[str, float] | None:
    if raw_json and file_path:
        raise ValueError("Use only one of --heuristic-weights-json or --heuristic-weights-file")
    if file_path:
        return json.loads(Path(file_path).read_text(encoding="utf-8"))
    if raw_json:
        return json.loads(raw_json)
    return None


def _summarize(run_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in run_rows:
        grouped.setdefault((row["set_id"], row["policy_type"]), []).append(row)

    summary_rows = []
    for (set_id, policy_type), rows in sorted(grouped.items()):
        first = rows[0]
        summary = {
            "set_id": set_id,
            "policy_type": policy_type,
            "total_agvs": first["total_agvs"],
            "north_agvs": first["north_agvs"],
            "south_agvs": first["south_agvs"],
            "west_agvs": first["west_agvs"],
            "east_agvs": first["east_agvs"],
            "runs": len(rows),
            "completed_runs": sum(1 for row in rows if row["completed"] == row["total_agvs"]),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            summary[f"avg_{metric}"] = statistics.fmean(values)
            summary[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    return summary_rows


def _compare(summary_rows: list[dict]) -> list[dict]:
    grouped: dict[int, dict[str, dict]] = {}
    for row in summary_rows:
        grouped.setdefault(row["set_id"], {})[row["policy_type"]] = row

    comparison_rows = []
    for set_id, policies in sorted(grouped.items()):
        if "fcfs" not in policies or "heuristic" not in policies:
            continue
        fcfs = policies["fcfs"]
        heuristic = policies["heuristic"]
        delta_total_time = fcfs["avg_total_time"] - heuristic["avg_total_time"]
        allocation = {
            "NORTH": fcfs["north_agvs"],
            "SOUTH": fcfs["south_agvs"],
            "WEST": fcfs["west_agvs"],
            "EAST": fcfs["east_agvs"],
        }
        comparison_rows.append(
            {
                "set_id": set_id,
                "total_agvs": fcfs["total_agvs"],
                "allocation": json.dumps(allocation, sort_keys=True),
                "fcfs_avg_total_time": fcfs["avg_total_time"],
                "heuristic_avg_total_time": heuristic["avg_total_time"],
                "delta_total_time": delta_total_time,
                "total_time_improvement_pct": (
                    delta_total_time / fcfs["avg_total_time"] * 100.0
                    if fcfs["avg_total_time"]
                    else 0.0
                ),
                "fcfs_avg_total_wait_time": fcfs["avg_total_wait_time"],
                "heuristic_avg_total_wait_time": heuristic["avg_total_wait_time"],
                "delta_total_wait_time": fcfs["avg_total_wait_time"]
                - heuristic["avg_total_wait_time"],
                "fcfs_avg_avg_wait_time": fcfs["avg_avg_wait_time"],
                "heuristic_avg_avg_wait_time": heuristic["avg_avg_wait_time"],
                "delta_avg_wait_time": fcfs["avg_avg_wait_time"]
                - heuristic["avg_avg_wait_time"],
                "fcfs_avg_max_wait_time": fcfs["avg_max_wait_time"],
                "heuristic_avg_max_wait_time": heuristic["avg_max_wait_time"],
                "delta_max_wait_time": fcfs["avg_max_wait_time"]
                - heuristic["avg_max_wait_time"],
                "fcfs_avg_utilization": fcfs["avg_utilization"],
                "heuristic_avg_utilization": heuristic["avg_utilization"],
                "delta_utilization": heuristic["avg_utilization"]
                - fcfs["avg_utilization"],
            }
        )
    return comparison_rows


def _allocation_from_setting(setting: dict) -> dict[str, int]:
    return {
        "NORTH": setting["north_agvs"],
        "SOUTH": setting["south_agvs"],
        "WEST": setting["west_agvs"],
        "EAST": setting["east_agvs"],
    }


def _write_csv(rows: list[dict], output_path: Path, preferred_fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    discovered = {key for row in rows for key in row}
    fieldnames = [field for field in preferred_fieldnames if field in discovered]
    fieldnames.extend(sorted(discovered - set(fieldnames)))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(output_dir: Path, settings: list[dict], comparison_rows: list[dict]) -> None:
    lines = [
        "# Fixed Layout Random Allocation Experiment",
        "",
        "## Common setup",
        "",
        "| item | value |",
        "|---|---|",
        "| layout | `fcfs_cross_shared_area_corridor8` |",
        "| west_exit_extension | `0` |",
        "| shared area | 2x2 cross center, one AGV at a time |",
        "| spawn_gap_steps | `1` |",
        "| admission_window_steps | `1` |",
        "| policies | `fcfs`, `heuristic` |",
        "| runs per set | `30` route/priority seeds |",
        "",
        "Each set fixes total AGV count and NORTH/SOUTH/WEST/EAST allocation. "
        "For each fixed set, route destinations and priorities are randomized by changing seed 30 times.",
        "",
        "## Random sets",
        "",
        "| set | total AGVs | north | south | west | east |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in settings:
        lines.append(
            f"| {setting['set_id']} | {setting['total_agvs']} | "
            f"{setting['north_agvs']} | {setting['south_agvs']} | "
            f"{setting['west_agvs']} | {setting['east_agvs']} |"
        )

    lines.extend(
        [
            "",
            "## Policy comparison",
            "",
            "| set | total AGVs | FCFS total_time | Heuristic total_time | delta | FCFS wait | Heuristic wait | FCFS util | Heuristic util |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison_rows:
        lines.append(
            f"| {row['set_id']} | {row['total_agvs']} | "
            f"{row['fcfs_avg_total_time']:.2f} | {row['heuristic_avg_total_time']:.2f} | "
            f"{row['delta_total_time']:.2f} | "
            f"{row['fcfs_avg_total_wait_time']:.2f} | {row['heuristic_avg_total_wait_time']:.2f} | "
            f"{row['fcfs_avg_utilization']:.3f} | {row['heuristic_avg_utilization']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `experiment_settings.csv`: fixed random set definitions",
            "- `raw_runs.csv`: all per-seed runs",
            "- `policy_summary_by_set.csv`: mean/std by set and policy",
            "- `policy_comparison_by_set.csv`: FCFS vs Heuristic mean comparison",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    main()
