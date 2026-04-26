from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from deadlock.dsd_fsd.benchmark.config import ARRIVAL_INTERVALS, CAPACITY_TO_BAYS, POLICY_TYPES, ExperimentConfig
from deadlock.dsd_fsd.benchmark.layout_builder import build_layout
from deadlock.dsd_fsd.benchmark.metrics import write_plots, write_results_csv, write_summary_table
from deadlock.dsd_fsd.benchmark.simulator import WarehouseSimulator, generate_task_arrivals


DEFAULT_OUTPUT_DIR = Path("outputs") / "dsd_fsd_benchmark"


def run_experiment_grid(
    seeds: list[int] | None = None,
    policies: list[str] | None = None,
    robot_counts: list[int] | None = None,
    aisle_counts: list[int] | None = None,
    capacities: list[str] | None = None,
    arrival_rates: list[str] | None = None,
    max_episode_steps: int = 500,
    max_wait_steps: int = 6,
    deadlock_window: int = 8,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    make_plots: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """Run fair Rule-based/DSD/FSD sweeps and write result artifacts.

    Args:
        seeds: Random seeds for deterministic task sequences.
        policies: Policy names to compare.
        robot_counts: Robot counts to sweep.
        aisle_counts: Aisle counts to sweep.
        capacities: Capacity labels to sweep.
        arrival_rates: Arrival rate labels to sweep.
        max_episode_steps: Episode hard stop.
        max_wait_steps: Wait threshold before recovery.
        deadlock_window: Repeated blocked signature length.
        output_dir: Directory for CSV and plots.
        make_plots: Whether to create PNG plots.
        verbose: Print robot states during simulation.

    Returns:
        Per-episode KPI rows.
    """

    seeds = seeds or [0, 1, 2, 3, 4]
    policies = policies or list(POLICY_TYPES)
    robot_counts = robot_counts or [3, 4, 5]
    aisle_counts = aisle_counts or [6, 12]
    capacities = capacities or list(CAPACITY_TO_BAYS)
    arrival_rates = arrival_rates or list(ARRIVAL_INTERVALS)

    rows: list[dict] = []
    for seed in seeds:
        for num_robots in robot_counts:
            for num_aisles in aisle_counts:
                for capacity in capacities:
                    for arrival_rate in arrival_rates:
                        base_config = ExperimentConfig(
                            policy_type="rule_based",
                            num_robots=num_robots,
                            num_aisles=num_aisles,
                            capacity=capacity,
                            arrival_rate=arrival_rate,
                            seed=seed,
                            max_episode_steps=max_episode_steps,
                            max_wait_steps=max_wait_steps,
                            deadlock_window=deadlock_window,
                            verbose=verbose,
                        )
                        layout = build_layout(base_config)
                        arrivals = generate_task_arrivals(base_config, layout)
                        for policy in policies:
                            config = replace(base_config, policy_type=policy)
                            simulator = WarehouseSimulator(config, layout=layout, task_arrivals=arrivals)
                            rows.append(simulator.run())

    output_dir.mkdir(parents=True, exist_ok=True)
    write_results_csv(rows, output_dir / "results.csv")
    write_summary_table(rows, output_dir / "summary_table.csv")
    if make_plots:
        write_plots(rows, output_dir)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Rule-based/DSD/FSD benchmark.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max-wait-steps", type=int, default=6)
    parser.add_argument("--deadlock-window", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--policies", default="rule_based,dsd,fsd")
    parser.add_argument("--robots", default="3,4,5")
    parser.add_argument("--aisles", default="6,12")
    parser.add_argument("--capacities", default="small,large")
    parser.add_argument("--arrival-rates", default="low,medium,high")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_experiment_grid(
        seeds=_parse_ints(args.seeds),
        policies=_parse_strings(args.policies),
        robot_counts=_parse_ints(args.robots),
        aisle_counts=_parse_ints(args.aisles),
        capacities=_parse_strings(args.capacities),
        arrival_rates=_parse_strings(args.arrival_rates),
        max_episode_steps=args.steps,
        max_wait_steps=args.max_wait_steps,
        deadlock_window=args.deadlock_window,
        output_dir=Path(args.output_dir),
        make_plots=not args.no_plots,
        verbose=args.verbose,
    )
    print(f"wrote {len(rows)} rows to {Path(args.output_dir).resolve()}")


def _parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_strings(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    main()
