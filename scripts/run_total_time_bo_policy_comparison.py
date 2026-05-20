"""Compare FCFS, fixed heuristic, and BO heuristic with total_time only.

The experiment keeps the cross-layout simulator unchanged while sampling 16 AGVs
across the NORTH/SOUTH/WEST/EAST start points. Each sampled scenario fixes the
directional allocation, per-AGV planned start times, and end-point assignments,
then reuses that exact scenario for all three policies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from rware.fcfs_cross_simulation import FCFSCrossExperiment


DIRECTIONS = ("NORTH", "SOUTH", "WEST", "EAST")
TOTAL_AGVS = 16
POLICIES = ("fcfs", "fixed_heuristic", "bo_heuristic")
BO_FEATURES = (
    "waiting",
    "maneuver",
    "exit_competition",
    "path_conflict",
    "approach_queue",
)
FIXED_HEURISTIC_WEIGHTS = {
    "waiting": 1.0,
    "maneuver": 0.8,
    "exit_competition": 1.2,
    "path_conflict": 1.5,
    "approach_queue": 0.7,
    "remaining_path": 0.0,
    "same_direction_backlog": 0.0,
}
FIXED_WEIGHT_BOUNDS = {
    "waiting": (0.4, 1.8),
    "maneuver": (0.0, 1.4),
    "exit_competition": (0.6, 2.0),
    "path_conflict": (0.8, 2.4),
    "approach_queue": (0.0, 1.6),
}
DEFAULT_BO_WEIGHT_BOUNDS = {
    "waiting": (0.0, 2.5),
    "maneuver": (-2.0, 2.0),
    "exit_competition": (-2.5, 2.5),
    "path_conflict": (-1.0, 3.0),
    "approach_queue": (-2.5, 2.0),
}
BO_PRIOR_WEIGHT_CANDIDATES = [
    {
        "waiting": 1.0,
        "maneuver": -1.0,
        "exit_competition": 0.0,
        "path_conflict": 1.5,
        "approach_queue": -1.5,
        "remaining_path": 0.0,
        "same_direction_backlog": 0.0,
    },
    {
        "waiting": 0.4,
        "maneuver": 1.6,
        "exit_competition": -1.5,
        "path_conflict": 0.5,
        "approach_queue": -2.0,
        "remaining_path": 0.0,
        "same_direction_backlog": 0.0,
    },
    {
        "waiting": 1.8,
        "maneuver": -0.6,
        "exit_competition": 1.8,
        "path_conflict": 2.4,
        "approach_queue": 0.0,
        "remaining_path": 0.0,
        "same_direction_backlog": 0.0,
    },
    {
        "waiting": 0.0,
        "maneuver": 0.0,
        "exit_competition": 0.0,
        "path_conflict": 2.8,
        "approach_queue": -2.2,
        "remaining_path": 0.0,
        "same_direction_backlog": 0.0,
    },
]
BO_EXPLORATION_MODES = {
    "stable": {
        "kappa": 0.8,
        "global_ratio": 0.45,
        "local_scale_ratio": 0.08,
    },
    "balanced": {
        "kappa": 1.96,
        "global_ratio": 0.80,
        "local_scale_ratio": 0.15,
    },
    "aggressive": {
        "kappa": 3.0,
        "global_ratio": 1.00,
        "local_scale_ratio": 0.30,
    },
}

SETTINGS_FIELDNAMES = [
    "run_index",
    "scenario_seed",
    "total_agvs",
    "north_agvs",
    "south_agvs",
    "west_agvs",
    "east_agvs",
    "spawn_plan_json",
    "goal_plan_json",
]

RUN_FIELDNAMES = [
    "run_index",
    "scenario_seed",
    "policy",
    "total_agvs",
    "north_agvs",
    "south_agvs",
    "west_agvs",
    "east_agvs",
    "shared_area_capacity",
    "completed",
    "total_time",
    "spawn_plan_json",
    "goal_plan_json",
]

SUMMARY_FIELDNAMES = [
    "policy",
    "runs",
    "completed_runs",
    "min_total_time",
    "min_run_index",
    "max_total_time",
    "max_run_index",
    "avg_total_time",
    "median_total_time",
    "std_total_time",
    "p10_total_time",
    "p25_total_time",
    "p75_total_time",
    "p90_total_time",
    "best_count",
    "unique_best_count",
    "best_rate_pct",
    "unique_best_rate_pct",
]

COMPARISON_FIELDNAMES = [
    "run_index",
    "scenario_seed",
    "fcfs_total_time",
    "fixed_heuristic_total_time",
    "bo_heuristic_total_time",
    "best_policy",
    "num_best_policies",
    "fixed_minus_fcfs",
    "fixed_improvement_pct_vs_fcfs",
    "bo_minus_fcfs",
    "bo_improvement_pct_vs_fcfs",
    "bo_minus_fixed",
    "bo_improvement_pct_vs_fixed",
]

BEST_POLICY_FIELDNAMES = [
    "run_index",
    "scenario_seed",
    "allocation_json",
    "spawn_plan_json",
    "goal_plan_json",
    "fcfs_total_time",
    "fixed_heuristic_total_time",
    "bo_heuristic_total_time",
    "best_policy",
    "best_total_time",
    "second_best_total_time",
    "margin_to_second",
    "is_tie",
]

BO_TRIAL_FIELDNAMES = [
    "trial_index",
    "source",
    "objective_avg_total_time",
    "best_so_far_avg_total_time",
    "weights_json",
    "gp_mu",
    "gp_sigma",
    "lcb",
    *BO_FEATURES,
]

FIXED_SEARCH_FIELDNAMES = [
    "candidate_index",
    "source",
    "objective_avg_total_time",
    "best_so_far_avg_total_time",
    "weights_json",
    *BO_FEATURES,
]


@dataclass(frozen=True)
class ScenarioSpec:
    """A fixed episode definition reused across every policy."""

    run_index: int
    seed: int
    allocation: dict[str, int]
    spawn_plan: dict[str, list[int]]
    goal_plan: dict[str, list[str]]


@dataclass(frozen=True)
class BOResult:
    """Bayesian optimization output."""

    best_weights: dict[str, float]
    best_objective: float
    trial_rows: list[dict]


@dataclass(frozen=True)
class FixedSearchResult:
    """Greedy search result over pre-composed fixed heuristic weight sets."""

    best_weights: dict[str, float]
    best_objective: float
    candidate_rows: list[dict]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a total_time-only comparison of FCFS, Fixed Heuristic, and "
            "Bayesian-Optimization Heuristic on 16 AGVs."
        )
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--admission-window-steps", type=int, default=2)
    parser.add_argument("--shared-area-capacity", type=int, default=2)
    parser.add_argument("--max-planned-spawn-step", type=int, default=8)
    parser.add_argument("--min-agvs-per-direction", type=int, default=2)
    parser.add_argument("--max-agvs-per-direction", type=int, default=6)
    parser.add_argument("--scenario-seed", type=int, default=20260519)
    parser.add_argument("--run-seed-base", type=int, default=30000)
    parser.add_argument("--fixed-weight-set-count", type=int, default=50)
    parser.add_argument("--bo-trials", type=int, default=50)
    parser.add_argument("--bo-seed-count", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--bo-initial-local-trials", type=int, default=8)
    parser.add_argument("--bo-initial-random-trials", type=int, default=8)
    parser.add_argument("--bo-candidate-count", type=int, default=2048)
    parser.add_argument("--bo-kappa", type=float, default=1.96)
    parser.add_argument(
        "--bo-exploration-mode",
        choices=sorted(BO_EXPLORATION_MODES),
        default="balanced",
    )
    parser.add_argument("--bo-length-scale", type=float, default=0.45)
    parser.add_argument("--bo-noise", type=float, default=1e-6)
    parser.add_argument("--bo-random-seed", type=int, default=9090)
    parser.add_argument("--bo-scenario-seed", type=int, default=20260520, help=argparse.SUPPRESS)
    parser.add_argument("--bo-seed-base", type=int, default=10000, help=argparse.SUPPRESS)
    parser.add_argument("--weight-min", type=float, default=None)
    parser.add_argument("--weight-max", type=float, default=None)
    parser.add_argument("--bo-weights-file", default="")
    parser.add_argument(
        "--output-dir",
        default=str(Path("final_output") / "total_time_bo_policy_comparison"),
    )
    args = parser.parse_args()

    _validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_config = _bo_mode_config(args.bo_exploration_mode, args.bo_kappa)

    bounds = _bo_weight_bounds(args)
    cycle_scenarios = _generate_scenarios(
        count=args.runs,
        seed_base=args.run_seed_base,
        scenario_seed=args.scenario_seed,
        max_planned_spawn_step=args.max_planned_spawn_step,
        min_agvs_per_direction=args.min_agvs_per_direction,
        max_agvs_per_direction=args.max_agvs_per_direction,
    )
    fixed_candidates = _build_fixed_weight_candidates(args.fixed_weight_set_count)
    fixed_result = _run_fixed_greedy_search(
        candidates=fixed_candidates,
        scenarios=cycle_scenarios,
        max_steps=args.max_steps,
        corridor_length=args.corridor_length,
        west_exit_extension=args.west_exit_extension,
        admission_window_steps=args.admission_window_steps,
        shared_area_capacity=args.shared_area_capacity,
    )
    if args.bo_weights_file:
        bo_weights = _complete_heuristic_weights(
            _load_bo_weights_file(Path(args.bo_weights_file))
        )
        bo_result = BOResult(best_weights=bo_weights, best_objective=math.nan, trial_rows=[])
    else:
        bo_result = _run_bayesian_optimization(
            scenarios=cycle_scenarios,
            initial_weights=fixed_result.best_weights,
            trials=args.bo_trials,
            initial_local_trials=args.bo_initial_local_trials,
            initial_random_trials=args.bo_initial_random_trials,
            candidate_count=args.bo_candidate_count,
            bounds=bounds,
            rng_seed=args.bo_random_seed,
            kappa=mode_config["kappa"],
            global_ratio=mode_config["global_ratio"],
            local_scale_ratio=mode_config["local_scale_ratio"],
            length_scale=args.bo_length_scale,
            noise=args.bo_noise,
            max_steps=args.max_steps,
            corridor_length=args.corridor_length,
            west_exit_extension=args.west_exit_extension,
            admission_window_steps=args.admission_window_steps,
            shared_area_capacity=args.shared_area_capacity,
        )

    run_rows = _run_policy_comparison(
        scenarios=cycle_scenarios,
        fixed_weights=fixed_result.best_weights,
        bo_weights=bo_result.best_weights,
        max_steps=args.max_steps,
        corridor_length=args.corridor_length,
        west_exit_extension=args.west_exit_extension,
        admission_window_steps=args.admission_window_steps,
        shared_area_capacity=args.shared_area_capacity,
    )
    comparison_rows = _compare_runs(run_rows)
    best_policy_rows = _best_policy_rows(comparison_rows, cycle_scenarios)
    summary_rows = _summarize_runs(run_rows, comparison_rows)
    setting_rows = [_scenario_to_settings_row(scenario) for scenario in cycle_scenarios]

    _write_csv(setting_rows, output_dir / "experiment_settings.csv", SETTINGS_FIELDNAMES)
    _write_csv(run_rows, output_dir / "raw_runs.csv", RUN_FIELDNAMES)
    _write_csv(summary_rows, output_dir / "policy_summary.csv", SUMMARY_FIELDNAMES)
    _write_csv(comparison_rows, output_dir / "trial_comparison.csv", COMPARISON_FIELDNAMES)
    _write_csv(best_policy_rows, output_dir / "best_policy_by_run.csv", BEST_POLICY_FIELDNAMES)
    _write_csv(fixed_result.candidate_rows, output_dir / "fixed_greedy_search.csv", FIXED_SEARCH_FIELDNAMES)
    _write_csv(bo_result.trial_rows, output_dir / "bo_trials.csv", BO_TRIAL_FIELDNAMES)
    _write_json(
        {
            "features": list(BO_FEATURES),
            "bounds": bounds,
            "fixed_heuristic_anchor_weights": FIXED_HEURISTIC_WEIGHTS,
            "fixed_weight_set_count": args.fixed_weight_set_count,
            "fixed_greedy_best_weights": fixed_result.best_weights,
            "fixed_greedy_best_avg_total_time": fixed_result.best_objective,
            "bo_prior_weight_candidates": BO_PRIOR_WEIGHT_CANDIDATES,
            "best_bo_weights": bo_result.best_weights,
            "best_bo_training_avg_total_time": bo_result.best_objective,
            "bo_training_cycle_scenario_count": args.runs,
            "bo_trials": args.bo_trials if not args.bo_weights_file else 0,
            "shared_area_capacity": args.shared_area_capacity,
            "bo_exploration_mode": args.bo_exploration_mode,
            "bo_kappa": mode_config["kappa"],
            "bo_global_ratio": mode_config["global_ratio"],
            "bo_local_scale_ratio": mode_config["local_scale_ratio"],
        },
        output_dir / "bo_best_weights.json",
    )
    _write_readme(
        output_dir=output_dir,
        args=args,
        summary_rows=summary_rows,
        fixed_result=fixed_result,
        bo_result=bo_result,
    )

    print(f"wrote settings: {(output_dir / 'experiment_settings.csv').resolve()}")
    print(f"wrote raw runs: {(output_dir / 'raw_runs.csv').resolve()}")
    print(f"wrote summary: {(output_dir / 'policy_summary.csv').resolve()}")
    print(f"wrote comparison: {(output_dir / 'trial_comparison.csv').resolve()}")
    print(f"wrote best policy by run: {(output_dir / 'best_policy_by_run.csv').resolve()}")
    print(f"wrote fixed search: {(output_dir / 'fixed_greedy_search.csv').resolve()}")
    print(
        f"fixed greedy best: avg_total_time={fixed_result.best_objective:.3f}, "
        f"weights={json.dumps(fixed_result.best_weights, sort_keys=True)}"
    )
    print(
        f"bo best: avg_total_time={bo_result.best_objective:.3f}, "
        f"weights={json.dumps(bo_result.best_weights, sort_keys=True)}"
    )
    for row in summary_rows:
        print(
            f"{row['policy']}: avg_total_time={row['avg_total_time']:.3f}, "
            f"best_count={row['best_count']}/{row['runs']}"
        )


def _validate_args(args: argparse.Namespace) -> None:
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.fixed_weight_set_count < 1:
        raise ValueError("--fixed-weight-set-count must be at least 1")
    if args.bo_trials < 1 and not args.bo_weights_file:
        raise ValueError("--bo-trials must be at least 1 unless --bo-weights-file is set")
    if args.bo_initial_local_trials < 0:
        raise ValueError("--bo-initial-local-trials must be non-negative")
    if args.bo_initial_random_trials < 0:
        raise ValueError("--bo-initial-random-trials must be non-negative")
    if args.bo_candidate_count < 1:
        raise ValueError("--bo-candidate-count must be at least 1")
    if (args.weight_min is None) != (args.weight_max is None):
        raise ValueError("--weight-min and --weight-max must be provided together")
    if args.weight_min is not None and args.weight_min >= args.weight_max:
        raise ValueError("--weight-min must be smaller than --weight-max")
    if args.max_planned_spawn_step < 0:
        raise ValueError("--max-planned-spawn-step must be non-negative")
    if args.shared_area_capacity < 1:
        raise ValueError("--shared-area-capacity must be at least 1")
    if args.min_agvs_per_direction < 1:
        raise ValueError("--min-agvs-per-direction must be at least 1")
    if args.max_agvs_per_direction < args.min_agvs_per_direction:
        raise ValueError("--max-agvs-per-direction must be >= --min-agvs-per-direction")
    if args.min_agvs_per_direction * len(DIRECTIONS) > TOTAL_AGVS:
        raise ValueError("minimum per-direction AGV count is too high for TOTAL_AGVS")
    if args.max_agvs_per_direction * len(DIRECTIONS) < TOTAL_AGVS:
        raise ValueError("maximum per-direction AGV count is too low for TOTAL_AGVS")


def _bo_mode_config(mode: str, cli_kappa: float) -> dict[str, float]:
    config = dict(BO_EXPLORATION_MODES[mode])
    if cli_kappa != 1.96:
        config["kappa"] = cli_kappa
    return config


def _bo_weight_bounds(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    if args.weight_min is not None and args.weight_max is not None:
        return {
            feature: (float(args.weight_min), float(args.weight_max))
            for feature in BO_FEATURES
        }
    return dict(DEFAULT_BO_WEIGHT_BOUNDS)


def _generate_scenarios(
    count: int,
    seed_base: int,
    scenario_seed: int,
    max_planned_spawn_step: int,
    min_agvs_per_direction: int = 2,
    max_agvs_per_direction: int = 6,
) -> list[ScenarioSpec]:
    rng = random.Random(scenario_seed)
    scenarios = []
    for run_index in range(count):
        allocation = _random_research_allocation(
            TOTAL_AGVS,
            rng,
            min_agvs_per_direction,
            max_agvs_per_direction,
        )
        spawn_plan = {
            direction: sorted(
                rng.randint(0, max_planned_spawn_step)
                for _ in range(allocation[direction])
            )
            for direction in DIRECTIONS
        }
        goal_plan = {
            direction: _random_goals(direction, allocation[direction], rng)
            for direction in DIRECTIONS
        }
        scenarios.append(
            ScenarioSpec(
                run_index=run_index,
                seed=seed_base + run_index,
                allocation=allocation,
                spawn_plan=spawn_plan,
                goal_plan=goal_plan,
            )
        )
    return scenarios


def _load_bo_weights_file(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "best_bo_weights" in payload:
        return payload["best_bo_weights"]
    return payload


def _random_research_allocation(
    total_agvs: int,
    rng: random.Random,
    min_count: int,
    max_count: int,
) -> dict[str, int]:
    counts = [min_count for _ in DIRECTIONS]
    remaining = total_agvs - min_count * len(DIRECTIONS)
    for _ in range(remaining):
        eligible = [
            index
            for index, count in enumerate(counts)
            if count < max_count
        ]
        counts[rng.choice(eligible)] += 1
    return dict(zip(DIRECTIONS, counts))


def _random_goals(direction: str, count: int, rng: random.Random) -> list[str]:
    own_exit = next(
        exit_node
        for exit_node, exit_direction in FCFSCrossExperiment.EXIT_DIRECTION_BY_NODE.items()
        if exit_direction == direction
    )
    candidates = sorted(set(FCFSCrossExperiment.EXIT_DIRECTION_BY_NODE) - {own_exit})
    return [rng.choice(candidates) for _ in range(count)]


def _build_fixed_weight_candidates(count: int) -> list[dict]:
    candidates: list[tuple[str, dict[str, float]]] = [
        ("manual_ahp_anchor", _complete_heuristic_weights(FIXED_HEURISTIC_WEIGHTS))
    ]
    sensitivity_values = {
        "waiting": (0.6, 1.4, 1.8),
        "maneuver": (0.2, 1.1, 1.4),
        "exit_competition": (0.8, 1.6, 2.0),
        "path_conflict": (1.0, 2.0, 2.4),
        "approach_queue": (0.2, 1.1, 1.6),
    }
    for feature, values in sensitivity_values.items():
        for value in values:
            weights = dict(FIXED_HEURISTIC_WEIGHTS)
            weights[feature] = value
            candidates.append((f"one_factor_{feature}", _complete_heuristic_weights(weights)))

    bottleneck_profiles = [
        {"waiting": 1.2, "maneuver": 0.4, "exit_competition": 1.6, "path_conflict": 2.0, "approach_queue": 0.4},
        {"waiting": 0.8, "maneuver": 0.6, "exit_competition": 1.8, "path_conflict": 1.8, "approach_queue": 0.8},
        {"waiting": 1.4, "maneuver": 0.3, "exit_competition": 1.0, "path_conflict": 2.2, "approach_queue": 1.0},
        {"waiting": 1.0, "maneuver": 1.2, "exit_competition": 1.4, "path_conflict": 1.2, "approach_queue": 0.5},
        {"waiting": 1.6, "maneuver": 0.8, "exit_competition": 0.8, "path_conflict": 1.6, "approach_queue": 1.2},
        {"waiting": 0.7, "maneuver": 1.0, "exit_competition": 2.0, "path_conflict": 2.0, "approach_queue": 0.2},
        {"waiting": 1.8, "maneuver": 0.0, "exit_competition": 1.2, "path_conflict": 2.4, "approach_queue": 0.0},
        {"waiting": 0.5, "maneuver": 1.4, "exit_competition": 1.8, "path_conflict": 1.0, "approach_queue": 1.4},
        {"waiting": 1.1, "maneuver": 0.7, "exit_competition": 0.6, "path_conflict": 2.4, "approach_queue": 1.6},
        {"waiting": 1.3, "maneuver": 0.5, "exit_competition": 1.6, "path_conflict": 0.8, "approach_queue": 1.5},
    ]
    candidates.extend(
        ("bottleneck_profile", _complete_heuristic_weights(profile))
        for profile in bottleneck_profiles
    )

    rng = random.Random(20260521)
    seen = {_candidate_key(weights) for _, weights in candidates}
    while len(candidates) < count:
        weights = {
            feature: round(rng.uniform(*FIXED_WEIGHT_BOUNDS[feature]), 2)
            for feature in BO_FEATURES
        }
        completed = _complete_heuristic_weights(weights)
        key = _candidate_key(completed)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(("precomposed_plausible_set", completed))

    return [
        {
            "candidate_index": index,
            "source": source,
            "weights": weights,
        }
        for index, (source, weights) in enumerate(candidates[:count])
    ]


def _candidate_key(weights: dict[str, float]) -> tuple[float, ...]:
    return tuple(round(float(weights[feature]), 6) for feature in BO_FEATURES)


def _run_fixed_greedy_search(
    candidates: list[dict],
    scenarios: list[ScenarioSpec],
    max_steps: int,
    corridor_length: int,
    west_exit_extension: int,
    admission_window_steps: int,
    shared_area_capacity: int,
) -> FixedSearchResult:
    rows = []
    objectives = []
    best_so_far = math.inf
    for candidate in candidates:
        weights = candidate["weights"]
        objective = _objective_avg_total_time(
            weights=weights,
            scenarios=scenarios,
            max_steps=max_steps,
            corridor_length=corridor_length,
            west_exit_extension=west_exit_extension,
            admission_window_steps=admission_window_steps,
            shared_area_capacity=shared_area_capacity,
            policy="fixed_heuristic",
        )
        objectives.append(objective)
        best_so_far = min(best_so_far, objective)
        row = {
            "candidate_index": candidate["candidate_index"],
            "source": candidate["source"],
            "objective_avg_total_time": objective,
            "best_so_far_avg_total_time": best_so_far,
            "weights_json": json.dumps(weights, sort_keys=True),
        }
        row.update({feature: weights[feature] for feature in BO_FEATURES})
        rows.append(row)

    best_index = min(range(len(objectives)), key=objectives.__getitem__)
    return FixedSearchResult(
        best_weights=candidates[best_index]["weights"],
        best_objective=objectives[best_index],
        candidate_rows=rows,
    )


def _run_bayesian_optimization(
    scenarios: list[ScenarioSpec],
    initial_weights: dict[str, float],
    trials: int,
    initial_local_trials: int,
    initial_random_trials: int,
    candidate_count: int,
    bounds: dict[str, tuple[float, float]],
    rng_seed: int,
    kappa: float,
    global_ratio: float,
    local_scale_ratio: float,
    length_scale: float,
    noise: float,
    max_steps: int,
    corridor_length: int,
    west_exit_extension: int,
    admission_window_steps: int,
    shared_area_capacity: int,
) -> BOResult:
    rng = np.random.default_rng(rng_seed)
    vectors: list[np.ndarray] = []
    objectives: list[float] = []
    trial_rows: list[dict] = []
    initial_vector = _weights_to_vector(initial_weights, bounds)

    for trial_index in range(trials):
        gp_mu = math.nan
        gp_sigma = math.nan
        lcb = math.nan
        if trial_index == 0:
            vector = initial_vector
            source = "fixed_greedy_best_anchor"
        elif trial_index <= len(BO_PRIOR_WEIGHT_CANDIDATES):
            vector = _weights_to_vector(
                BO_PRIOR_WEIGHT_CANDIDATES[trial_index - 1],
                bounds,
            )
            source = "structural_prior"
        elif trial_index < 1 + len(BO_PRIOR_WEIGHT_CANDIDATES) + initial_local_trials:
            vector = _local_weight_vector(
                rng,
                center=initial_vector,
                bounds=bounds,
                scale_ratio=0.18,
            )
            source = "local_perturbation"
        elif trial_index < 1 + len(BO_PRIOR_WEIGHT_CANDIDATES) + initial_local_trials + initial_random_trials:
            vector = _random_weight_vector(rng, bounds)
            source = "random"
        else:
            vector, gp_mu, gp_sigma, lcb = _suggest_lcb_candidate(
                vectors=np.vstack(vectors),
                objectives=np.array(objectives, dtype=float),
                candidate_count=candidate_count,
                bounds=bounds,
                rng=rng,
                kappa=kappa,
                global_ratio=global_ratio,
                local_scale_ratio=local_scale_ratio,
                length_scale=length_scale,
                noise=noise,
            )
            source = "lcb"

        weights = _complete_heuristic_weights(_vector_to_weights(vector))
        objective = _objective_avg_total_time(
            weights=weights,
            scenarios=scenarios,
            max_steps=max_steps,
            corridor_length=corridor_length,
            west_exit_extension=west_exit_extension,
            admission_window_steps=admission_window_steps,
            shared_area_capacity=shared_area_capacity,
        )
        vectors.append(vector)
        objectives.append(objective)
        best_so_far = min(objectives)
        row = {
            "trial_index": trial_index,
            "source": source,
            "objective_avg_total_time": objective,
            "best_so_far_avg_total_time": best_so_far,
            "weights_json": json.dumps(weights, sort_keys=True),
            "gp_mu": gp_mu,
            "gp_sigma": gp_sigma,
            "lcb": lcb,
        }
        row.update({feature: weights[feature] for feature in BO_FEATURES})
        trial_rows.append(row)

    best_index = min(range(len(objectives)), key=objectives.__getitem__)
    best_weights = _complete_heuristic_weights(_vector_to_weights(vectors[best_index]))
    return BOResult(
        best_weights=best_weights,
        best_objective=objectives[best_index],
        trial_rows=trial_rows,
    )


def _objective_avg_total_time(
    weights: dict[str, float],
    scenarios: list[ScenarioSpec],
    max_steps: int,
    corridor_length: int,
    west_exit_extension: int,
    admission_window_steps: int,
    shared_area_capacity: int,
    policy: str = "bo_heuristic",
) -> float:
    values = [
        _run_single_policy(
            scenario=scenario,
            policy=policy,
            weights=weights,
            max_steps=max_steps,
            corridor_length=corridor_length,
            west_exit_extension=west_exit_extension,
            admission_window_steps=admission_window_steps,
            shared_area_capacity=shared_area_capacity,
        )["total_time"]
        for scenario in scenarios
    ]
    return statistics.fmean(values)


def _suggest_lcb_candidate(
    vectors: np.ndarray,
    objectives: np.ndarray,
    candidate_count: int,
    bounds: dict[str, tuple[float, float]],
    rng: np.random.Generator,
    kappa: float,
    global_ratio: float,
    local_scale_ratio: float,
    length_scale: float,
    noise: float,
) -> tuple[np.ndarray, float, float, float]:
    lower, upper = _bounds_arrays(bounds)
    global_count = max(1, int(candidate_count * global_ratio))
    local_count = max(1, candidate_count - global_count)
    global_candidates = rng.uniform(lower, upper, size=(global_count, len(BO_FEATURES)))
    best_vector = vectors[int(np.argmin(objectives))]
    local_scale = (upper - lower) * local_scale_ratio
    local_candidates = best_vector + rng.normal(
        loc=0.0,
        scale=local_scale,
        size=(local_count, len(BO_FEATURES)),
    )
    local_candidates = np.clip(local_candidates, lower, upper)
    candidates = np.vstack([global_candidates, local_candidates])
    mu, sigma = _gp_predict(
        train_x=_scale_vectors(vectors, bounds),
        train_y=objectives,
        test_x=_scale_vectors(candidates, bounds),
        length_scale=length_scale,
        noise=noise,
    )
    lcb_values = mu - kappa * sigma
    best_candidate_index = int(np.argmin(lcb_values))
    return (
        candidates[best_candidate_index],
        float(mu[best_candidate_index]),
        float(sigma[best_candidate_index]),
        float(lcb_values[best_candidate_index]),
    )


def _gp_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    length_scale: float,
    noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    y_mean = float(np.mean(train_y))
    y_std = float(np.std(train_y))
    if y_std < 1e-12:
        y_std = 1.0
    normalized_y = (train_y - y_mean) / y_std

    kernel = _rbf_kernel(train_x, train_x, length_scale)
    kernel += noise * np.eye(len(train_x))
    chol = _stable_cholesky(kernel)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, normalized_y))
    cross_kernel = _rbf_kernel(test_x, train_x, length_scale)
    normalized_mu = cross_kernel @ alpha
    solve = np.linalg.solve(chol, cross_kernel.T)
    variance = np.maximum(1.0 - np.sum(solve * solve, axis=0), 1e-12)
    return normalized_mu * y_std + y_mean, np.sqrt(variance) * y_std


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float) -> np.ndarray:
    diff = (x1[:, None, :] - x2[None, :, :]) / length_scale
    return np.exp(-0.5 * np.sum(diff * diff, axis=2))


def _stable_cholesky(matrix: np.ndarray) -> np.ndarray:
    jitter = 1e-10
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            matrix = matrix + jitter * np.eye(matrix.shape[0])
            jitter *= 10.0
    return np.linalg.cholesky(matrix + jitter * np.eye(matrix.shape[0]))


def _random_weight_vector(
    rng: np.random.Generator,
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    lower, upper = _bounds_arrays(bounds)
    return rng.uniform(lower, upper, size=len(BO_FEATURES))


def _local_weight_vector(
    rng: np.random.Generator,
    center: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    scale_ratio: float,
) -> np.ndarray:
    lower, upper = _bounds_arrays(bounds)
    scale = (upper - lower) * scale_ratio
    return np.clip(center + rng.normal(0.0, scale, size=len(BO_FEATURES)), lower, upper)


def _weights_to_vector(
    weights: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    lower, upper = _bounds_arrays(bounds)
    values = np.array([float(weights[feature]) for feature in BO_FEATURES])
    return np.clip(values, lower, upper)


def _vector_to_weights(vector: np.ndarray) -> dict[str, float]:
    return {
        feature: float(value)
        for feature, value in zip(BO_FEATURES, vector)
    }


def _complete_heuristic_weights(weights: dict[str, float]) -> dict[str, float]:
    completed = {feature: float(weights.get(feature, 0.0)) for feature in BO_FEATURES}
    completed["remaining_path"] = float(weights.get("remaining_path", 0.0))
    completed["same_direction_backlog"] = float(weights.get("same_direction_backlog", 0.0))
    return completed


def _bounds_arrays(bounds: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([bounds[feature][0] for feature in BO_FEATURES], dtype=float)
    upper = np.array([bounds[feature][1] for feature in BO_FEATURES], dtype=float)
    return lower, upper


def _scale_vectors(
    vectors: np.ndarray,
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    lower, upper = _bounds_arrays(bounds)
    return (vectors - lower) / (upper - lower)


def _run_policy_comparison(
    scenarios: list[ScenarioSpec],
    fixed_weights: dict[str, float],
    bo_weights: dict[str, float],
    max_steps: int,
    corridor_length: int,
    west_exit_extension: int,
    admission_window_steps: int,
    shared_area_capacity: int,
) -> list[dict]:
    rows = []
    policy_weights = {
        "fcfs": None,
        "fixed_heuristic": fixed_weights,
        "bo_heuristic": bo_weights,
    }
    for scenario in scenarios:
        for policy in POLICIES:
            metrics = _run_single_policy(
                scenario=scenario,
                policy=policy,
                weights=policy_weights[policy],
                max_steps=max_steps,
                corridor_length=corridor_length,
                west_exit_extension=west_exit_extension,
                admission_window_steps=admission_window_steps,
                shared_area_capacity=shared_area_capacity,
            )
            rows.append(_metrics_to_run_row(scenario, policy, metrics))
    return rows


def _run_single_policy(
    scenario: ScenarioSpec,
    policy: str,
    weights: dict[str, float] | None,
    max_steps: int,
    corridor_length: int,
    west_exit_extension: int,
    admission_window_steps: int,
    shared_area_capacity: int,
) -> dict:
    policy_type = "fcfs" if policy == "fcfs" else "heuristic"
    experiment = FCFSCrossExperiment(
        robots_by_direction=scenario.allocation,
        random_seed=scenario.seed,
        corridor_length=corridor_length,
        west_exit_extension=west_exit_extension,
        spawn_gap_steps=0,
        spawn_plan_by_direction=scenario.spawn_plan,
        admission_window_steps=admission_window_steps,
        shared_area_capacity=shared_area_capacity,
        normalize_heuristic_features=True,
        goal_plan_by_direction=scenario.goal_plan,
        policy_type=policy_type,
        heuristic_weights=weights if policy_type == "heuristic" else None,
    )
    return experiment.run(max_steps=max_steps)


def _metrics_to_run_row(
    scenario: ScenarioSpec,
    policy: str,
    metrics: dict,
) -> dict:
    return {
        "run_index": scenario.run_index,
        "scenario_seed": scenario.seed,
        "policy": policy,
        "total_agvs": TOTAL_AGVS,
        "north_agvs": scenario.allocation["NORTH"],
        "south_agvs": scenario.allocation["SOUTH"],
        "west_agvs": scenario.allocation["WEST"],
        "east_agvs": scenario.allocation["EAST"],
        "shared_area_capacity": metrics["shared_area_capacity"],
        "completed": metrics["completed"],
        "total_time": metrics["total_time"],
        "spawn_plan_json": json.dumps(scenario.spawn_plan, sort_keys=True),
        "goal_plan_json": json.dumps(scenario.goal_plan, sort_keys=True),
    }


def _scenario_to_settings_row(scenario: ScenarioSpec) -> dict:
    return {
        "run_index": scenario.run_index,
        "scenario_seed": scenario.seed,
        "total_agvs": TOTAL_AGVS,
        "north_agvs": scenario.allocation["NORTH"],
        "south_agvs": scenario.allocation["SOUTH"],
        "west_agvs": scenario.allocation["WEST"],
        "east_agvs": scenario.allocation["EAST"],
        "spawn_plan_json": json.dumps(scenario.spawn_plan, sort_keys=True),
        "goal_plan_json": json.dumps(scenario.goal_plan, sort_keys=True),
    }


def _compare_runs(run_rows: list[dict]) -> list[dict]:
    grouped: dict[int, dict[str, dict]] = {}
    for row in run_rows:
        grouped.setdefault(row["run_index"], {})[row["policy"]] = row

    comparison_rows = []
    for run_index, policies in sorted(grouped.items()):
        times = {
            policy: int(policies[policy]["total_time"])
            for policy in POLICIES
        }
        best_time = min(times.values())
        best_policies = [
            policy
            for policy, total_time in times.items()
            if total_time == best_time
        ]
        fcfs = times["fcfs"]
        fixed = times["fixed_heuristic"]
        bo = times["bo_heuristic"]
        comparison_rows.append(
            {
                "run_index": run_index,
                "scenario_seed": policies["fcfs"]["scenario_seed"],
                "fcfs_total_time": fcfs,
                "fixed_heuristic_total_time": fixed,
                "bo_heuristic_total_time": bo,
                "best_policy": ",".join(best_policies),
                "num_best_policies": len(best_policies),
                "fixed_minus_fcfs": fixed - fcfs,
                "fixed_improvement_pct_vs_fcfs": _improvement_pct(fcfs, fixed),
                "bo_minus_fcfs": bo - fcfs,
                "bo_improvement_pct_vs_fcfs": _improvement_pct(fcfs, bo),
                "bo_minus_fixed": bo - fixed,
                "bo_improvement_pct_vs_fixed": _improvement_pct(fixed, bo),
            }
        )
    return comparison_rows


def _best_policy_rows(
    comparison_rows: list[dict],
    scenarios: list[ScenarioSpec],
) -> list[dict]:
    scenarios_by_index = {scenario.run_index: scenario for scenario in scenarios}
    rows = []
    for row in comparison_rows:
        scenario = scenarios_by_index[int(row["run_index"])]
        times = sorted(
            [
                int(row["fcfs_total_time"]),
                int(row["fixed_heuristic_total_time"]),
                int(row["bo_heuristic_total_time"]),
            ]
        )
        best_total_time = times[0]
        second_best_total_time = times[1]
        rows.append(
            {
                "run_index": row["run_index"],
                "scenario_seed": row["scenario_seed"],
                "allocation_json": json.dumps(scenario.allocation, sort_keys=True),
                "spawn_plan_json": json.dumps(scenario.spawn_plan, sort_keys=True),
                "goal_plan_json": json.dumps(scenario.goal_plan, sort_keys=True),
                "fcfs_total_time": row["fcfs_total_time"],
                "fixed_heuristic_total_time": row["fixed_heuristic_total_time"],
                "bo_heuristic_total_time": row["bo_heuristic_total_time"],
                "best_policy": row["best_policy"],
                "best_total_time": best_total_time,
                "second_best_total_time": second_best_total_time,
                "margin_to_second": second_best_total_time - best_total_time,
                "is_tie": int(row["num_best_policies"]) > 1,
            }
        )
    return rows


def _summarize_runs(run_rows: list[dict], comparison_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {policy: [] for policy in POLICIES}
    for row in run_rows:
        grouped[row["policy"]].append(row)

    summaries = []
    for policy in POLICIES:
        rows = grouped[policy]
        values = [int(row["total_time"]) for row in rows]
        min_value = min(values)
        max_value = max(values)
        min_run = rows[values.index(min_value)]["run_index"]
        max_run = rows[values.index(max_value)]["run_index"]
        best_count = sum(
            policy in row["best_policy"].split(",")
            for row in comparison_rows
        )
        unique_best_count = sum(
            row["best_policy"] == policy
            for row in comparison_rows
        )
        summaries.append(
            {
                "policy": policy,
                "runs": len(rows),
                "completed_runs": sum(row["completed"] == TOTAL_AGVS for row in rows),
                "min_total_time": min_value,
                "min_run_index": min_run,
                "max_total_time": max_value,
                "max_run_index": max_run,
                "avg_total_time": statistics.fmean(values),
                "median_total_time": statistics.median(values),
                "std_total_time": statistics.stdev(values) if len(values) > 1 else 0.0,
                "p10_total_time": _percentile(values, 10.0),
                "p25_total_time": _percentile(values, 25.0),
                "p75_total_time": _percentile(values, 75.0),
                "p90_total_time": _percentile(values, 90.0),
                "best_count": best_count,
                "unique_best_count": unique_best_count,
                "best_rate_pct": best_count / len(rows) * 100.0,
                "unique_best_rate_pct": unique_best_count / len(rows) * 100.0,
            }
        )
    return summaries


def _percentile(values: list[int], percentile: float) -> float:
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[int(position)])
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _improvement_pct(baseline: int, candidate: int) -> float:
    if baseline == 0:
        return 0.0
    return (baseline - candidate) / baseline * 100.0


def _write_csv(rows: list[dict], output_path: Path, preferred_fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(preferred_fieldnames)
    discovered = {key for row in rows for key in row}
    fieldnames.extend(sorted(discovered - set(fieldnames)))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: dict, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_readme(
    output_dir: Path,
    args: argparse.Namespace,
    summary_rows: list[dict],
    fixed_result: FixedSearchResult,
    bo_result: BOResult,
) -> None:
    lines = [
        "# Total-Time Policy Comparison",
        "",
        "## Common setup",
        "",
        "| item | value |",
        "|---|---:|",
        f"| AGVs | {TOTAL_AGVS} |",
        f"| scenarios per cycle | {args.runs} |",
        f"| corridor_length | {args.corridor_length} |",
        f"| west_exit_extension | {args.west_exit_extension} |",
        f"| admission_window_steps | {args.admission_window_steps} |",
        f"| shared_area_capacity | {args.shared_area_capacity} |",
        f"| max_planned_spawn_step | {args.max_planned_spawn_step} |",
        f"| per-direction AGV count range | {args.min_agvs_per_direction}-{args.max_agvs_per_direction} |",
        f"| fixed weight sets | {args.fixed_weight_set_count} |",
        "",
        "Each run samples one fixed scenario: NORTH/SOUTH/WEST/EAST AGV counts, "
        "per-AGV planned start times, and random end-point assignments. The same "
        "scenario is replayed for FCFS, Fixed Heuristic, and BO Heuristic.",
        "",
        "One cycle means running the full set of sampled scenarios once. FCFS is "
        "executed for one cycle as the deterministic baseline. Fixed Heuristic "
        "evaluates the pre-composed weight sets over the same cycle and keeps the "
        "lowest-average `total_time` set. BO starts from that Fixed best set and "
        "continues optimizing the same five weights.",
        "",
        "The only evaluation metric used in summaries is `total_time`.",
        "",
        "## Fixed heuristic search",
        "",
        f"- Fixed candidate sets: `{args.fixed_weight_set_count}`",
        f"- Fixed greedy best avg total_time: `{fixed_result.best_objective}`",
        f"- Fixed greedy best weights: `{json.dumps(fixed_result.best_weights, sort_keys=True)}`",
        "",
        "## Bayesian optimization",
        "",
        "The BO stage optimizes the five heuristic weights from the report: "
        "`waiting`, `maneuver`, `exit_competition`, `path_conflict`, and "
        "`approach_queue`. It uses a lightweight Gaussian Process surrogate and "
        "Lower Confidence Bound acquisition, `LCB(w) = mu(w) - kappa * sigma(w)`, "
        "to minimize average `total_time` over the same 100-scenario cycle.",
        "",
        f"- BO trials: `{args.bo_trials if not args.bo_weights_file else 0}`",
        f"- BO scenarios per trial: `{args.runs}`",
        f"- BO initial local trials: `{args.bo_initial_local_trials}`",
        f"- BO initial random trials: `{args.bo_initial_random_trials}`",
        f"- BO exploration mode: `{args.bo_exploration_mode}`",
        f"- BO kappa: `{_bo_mode_config(args.bo_exploration_mode, args.bo_kappa)['kappa']}`",
        f"- BO best training avg total_time: `{bo_result.best_objective}`",
        "",
        "## Summary",
        "",
        "| policy | avg total_time | min | max | std | best count | unique best count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['policy']} | {row['avg_total_time']:.3f} | "
            f"{row['min_total_time']} | {row['max_total_time']} | "
            f"{row['std_total_time']:.3f} | {row['best_count']} | "
            f"{row['unique_best_count']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `experiment_settings.csv`: sampled allocations, start plans, and end plans",
            "- `raw_runs.csv`: one row per policy per run",
            "- `policy_summary.csv`: total_time-only statistics by policy",
            "- `trial_comparison.csv`: per-run winner and pairwise total_time deltas",
            "- `best_policy_by_run.csv`: one row per scenario showing which policy had the lowest total_time",
            "- `fixed_greedy_search.csv`: one row per pre-composed Fixed weight set",
            "- `bo_trials.csv`: BO search trace",
            "- `bo_best_weights.json`: selected BO weights, bounds, and Fixed search result",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
