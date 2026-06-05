"""Compare FCFS, fixed heuristic, BO heuristic, and PSO heuristic.

The experiment keeps the cross-layout simulator unchanged while sampling AGVs
across the NORTH/SOUTH/WEST/EAST start points. Each sampled scenario fixes the
directional allocation, per-AGV planned start times, and end-point assignments,
then reuses that exact scenario for all four policies.
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
DOMINANT_ROTATION = ("NORTH", "EAST", "SOUTH", "WEST")
TOTAL_AGVS = 20
POLICIES = ("fcfs", "fixed_heuristic", "bo_heuristic", "pso_heuristic")
SCENARIO_CASES = {
    1: {
        "name": "normal_balanced",
        "label": "Scenario 1: Normal balanced",
        "use_train": True,
        "use_test": True,
    },
    2: {
        "name": "arrival_burst",
        "label": "Scenario 2: Arrival burst",
        "use_train": True,
        "use_test": True,
    },
    3: {
        "name": "direction_skewed",
        "label": "Scenario 3: Direction-skewed",
        "use_train": True,
        "use_test": True,
    },
    4: {
        "name": "exit_concentrated",
        "label": "Scenario 4: Exit-concentrated",
        "use_train": True,
        "use_test": True,
    },
    5: {
        "name": "mixed_simulation",
        "label": "Scenario 5: Mixed-Simulation",
        "use_train": False,
        "use_test": True,
    },
}
SCENARIO_SET_CASE_IDS = {
    "random": (),
    "pdf_train": (1, 2, 3, 4),
    "pdf_test": (1, 2, 3, 4, 5),
}
PDF_CASE_TOTAL_AGV_OPTIONS = (16, 20, 24)
PDF_TRAIN_TOTAL_AGVS = 20
NORMAL_DEPARTURE_STEP = 4
EARLY_DEPARTURE_RANGE = (0, 2)
LATE_DEPARTURE_RANGE = (3, 8)
DOMINANT_COUNT_RANGE_BY_TOTAL_AGVS = {
    16: (8, 12),
    20: (10, 14),
    24: (12, 16),
}
BO_FEATURES = (
    "waiting",
    "maneuver",
    "exit_competition",
    "path_conflict",
    "approach_queue",
)
COMMON_INITIAL_RANDOM_COUNT = 20
PSO_SWARM_SIZE = 20
PSO_INERTIA_START = 0.7
PSO_INERTIA_END = 0.4
PSO_COGNITIVE_COEFFICIENT = 1.5
PSO_SOCIAL_COEFFICIENT = 1.5
PSO_VELOCITY_LIMIT_RATIO = 0.2
PSO_RANDOM_SEED = 42
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
    "scenario_case_id",
    "scenario_case_name",
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
    "scenario_case_id",
    "scenario_case_name",
    "fcfs_total_time",
    "fixed_heuristic_total_time",
    "bo_heuristic_total_time",
    "pso_heuristic_total_time",
    "best_policy",
    "num_best_policies",
    "fixed_minus_fcfs",
    "fixed_improvement_pct_vs_fcfs",
    "bo_minus_fcfs",
    "bo_improvement_pct_vs_fcfs",
    "bo_minus_fixed",
    "bo_improvement_pct_vs_fixed",
    "pso_minus_fcfs",
    "pso_improvement_pct_vs_fcfs",
    "pso_minus_fixed",
    "pso_improvement_pct_vs_fixed",
    "pso_minus_bo",
    "pso_improvement_pct_vs_bo",
]

BEST_POLICY_FIELDNAMES = [
    "run_index",
    "scenario_seed",
    "scenario_case_id",
    "scenario_case_name",
    "allocation_json",
    "spawn_plan_json",
    "goal_plan_json",
    "fcfs_total_time",
    "fixed_heuristic_total_time",
    "bo_heuristic_total_time",
    "pso_heuristic_total_time",
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

PSO_TRIAL_FIELDNAMES = [
    "evaluation_index",
    "iteration_index",
    "particle_index",
    "source",
    "objective_avg_total_time",
    "personal_best_avg_total_time",
    "best_so_far_avg_total_time",
    "inertia_weight",
    "velocity_norm",
    "weights_json",
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
    scenario_case_id: int = 0
    scenario_case_name: str = "random_research"
    scenario_case_label: str = "Random research scenario"
    scenario_split: str = "train_test"
    dominant_approach: str = ""
    dominant_approach_count: int = 0
    dominant_exit: str = ""
    dominant_exit_count: int = 0
    early_departures: int = 0
    total_agvs: int = TOTAL_AGVS


@dataclass(frozen=True)
class BOResult:
    """Bayesian optimization output."""

    best_weights: dict[str, float]
    best_objective: float
    trial_rows: list[dict]


@dataclass(frozen=True)
class PSOResult:
    """Particle Swarm Optimization output."""

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
            "Run a total_time-only comparison of FCFS, Fixed Heuristic, "
            "Bayesian-Optimization Heuristic, and PSO Heuristic."
        )
    )
    parser.add_argument("--total-agvs", type=int, default=TOTAL_AGVS)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--admission-window-steps", type=int, default=2)
    parser.add_argument("--shared-area-capacity", type=int, default=2)
    parser.add_argument(
        "--scenario-set",
        choices=sorted(SCENARIO_SET_CASE_IDS),
        default="pdf_train",
    )
    parser.add_argument(
        "--scenarios-per-case",
        type=int,
        default=0,
        help="For pdf_train/pdf_test, 0 derives an equal per-case count from --runs.",
    )
    parser.add_argument(
        "--test-agv-counts",
        default="16,20,24",
        help="Comma-separated AGV counts used only by --scenario-set pdf_test.",
    )
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
    parser.add_argument("--common-initial-random-count", type=int, default=COMMON_INITIAL_RANDOM_COUNT)
    parser.add_argument("--pso-swarm-size", type=int, default=PSO_SWARM_SIZE)
    parser.add_argument(
        "--pso-evaluations",
        type=int,
        default=0,
        help="Number of PSO objective evaluations. 0 uses the BO trial count.",
    )
    parser.add_argument("--pso-inertia-start", type=float, default=PSO_INERTIA_START)
    parser.add_argument("--pso-inertia-end", type=float, default=PSO_INERTIA_END)
    parser.add_argument("--pso-cognitive", type=float, default=PSO_COGNITIVE_COEFFICIENT)
    parser.add_argument("--pso-social", type=float, default=PSO_SOCIAL_COEFFICIENT)
    parser.add_argument("--pso-velocity-limit-ratio", type=float, default=PSO_VELOCITY_LIMIT_RATIO)
    parser.add_argument("--pso-random-seed", type=int, default=PSO_RANDOM_SEED)
    parser.add_argument("--pso-weights-file", default="")
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
    common_initial_vectors = _common_initial_weight_vectors(
        count=args.common_initial_random_count,
        bounds=bounds,
        rng_seed=args.pso_random_seed,
    )
    cycle_scenarios = _build_scenarios_from_args(args)
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
            initial_random_vectors=common_initial_vectors,
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

    pso_evaluations = _effective_pso_evaluations(args)
    if args.pso_weights_file:
        pso_weights = _complete_heuristic_weights(
            _load_weights_file(Path(args.pso_weights_file), preferred_key="best_pso_weights")
        )
        pso_result = PSOResult(best_weights=pso_weights, best_objective=math.nan, trial_rows=[])
    else:
        pso_result = _run_particle_swarm_optimization(
            scenarios=cycle_scenarios,
            swarm_size=args.pso_swarm_size,
            evaluations=pso_evaluations,
            bounds=bounds,
            initial_vectors=common_initial_vectors,
            rng_seed=args.pso_random_seed,
            inertia_start=args.pso_inertia_start,
            inertia_end=args.pso_inertia_end,
            cognitive_coefficient=args.pso_cognitive,
            social_coefficient=args.pso_social,
            velocity_limit_ratio=args.pso_velocity_limit_ratio,
            max_steps=args.max_steps,
            corridor_length=args.corridor_length,
            west_exit_extension=args.west_exit_extension,
            admission_window_steps=args.admission_window_steps,
            shared_area_capacity=args.shared_area_capacity,
        )

    run_rows = _run_policy_comparison(
        scenarios=cycle_scenarios,
        total_agvs=args.total_agvs,
        fixed_weights=fixed_result.best_weights,
        bo_weights=bo_result.best_weights,
        pso_weights=pso_result.best_weights,
        max_steps=args.max_steps,
        corridor_length=args.corridor_length,
        west_exit_extension=args.west_exit_extension,
        admission_window_steps=args.admission_window_steps,
        shared_area_capacity=args.shared_area_capacity,
    )
    comparison_rows = _compare_runs(run_rows)
    best_policy_rows = _best_policy_rows(comparison_rows, cycle_scenarios)
    summary_rows = _summarize_runs(run_rows, comparison_rows, args.total_agvs)
    setting_rows = [
        _scenario_to_settings_row(scenario, args.total_agvs)
        for scenario in cycle_scenarios
    ]

    _write_csv(setting_rows, output_dir / "experiment_settings.csv", SETTINGS_FIELDNAMES)
    _write_csv(run_rows, output_dir / "raw_runs.csv", RUN_FIELDNAMES)
    _write_csv(summary_rows, output_dir / "policy_summary.csv", SUMMARY_FIELDNAMES)
    _write_csv(comparison_rows, output_dir / "trial_comparison.csv", COMPARISON_FIELDNAMES)
    _write_csv(best_policy_rows, output_dir / "best_policy_by_run.csv", BEST_POLICY_FIELDNAMES)
    _write_csv(fixed_result.candidate_rows, output_dir / "fixed_greedy_search.csv", FIXED_SEARCH_FIELDNAMES)
    _write_csv(bo_result.trial_rows, output_dir / "bo_trials.csv", BO_TRIAL_FIELDNAMES)
    _write_csv(pso_result.trial_rows, output_dir / "pso_trials.csv", PSO_TRIAL_FIELDNAMES)
    _write_json(
        {
            "features": list(BO_FEATURES),
            "bounds": bounds,
            "common_initial_random_count": args.common_initial_random_count,
            "common_initial_random_seed": args.pso_random_seed,
            "common_initial_random_weights": [
                _complete_heuristic_weights(_vector_to_weights(vector))
                for vector in common_initial_vectors
            ],
            "fixed_heuristic_anchor_weights": FIXED_HEURISTIC_WEIGHTS,
            "fixed_weight_set_count": args.fixed_weight_set_count,
            "fixed_greedy_best_weights": fixed_result.best_weights,
            "fixed_greedy_best_avg_total_time": fixed_result.best_objective,
            "bo_prior_weight_candidates": BO_PRIOR_WEIGHT_CANDIDATES,
            "best_bo_weights": bo_result.best_weights,
            "best_bo_training_avg_total_time": bo_result.best_objective,
            "bo_training_cycle_scenario_count": len(cycle_scenarios),
            "scenario_set": args.scenario_set,
            "scenario_cases": list(SCENARIO_SET_CASE_IDS[args.scenario_set]),
            "scenario_agv_counts": list(_pdf_total_agv_options(args)) if args.scenario_set != "random" else [args.total_agvs],
            "dominant_count_ranges": DOMINANT_COUNT_RANGE_BY_TOTAL_AGVS,
            "scenarios_per_case": _effective_scenarios_per_case(args),
            "bo_trials": args.bo_trials if not args.bo_weights_file else 0,
            "total_agvs": args.total_agvs,
            "shared_area_capacity": args.shared_area_capacity,
            "bo_exploration_mode": args.bo_exploration_mode,
            "bo_kappa": mode_config["kappa"],
            "bo_global_ratio": mode_config["global_ratio"],
            "bo_local_scale_ratio": mode_config["local_scale_ratio"],
            "best_pso_weights": pso_result.best_weights,
            "best_pso_training_avg_total_time": pso_result.best_objective,
            "pso_evaluations": 0 if args.pso_weights_file else pso_evaluations,
            "pso_swarm_size": args.pso_swarm_size,
            "pso_inertia_start": args.pso_inertia_start,
            "pso_inertia_end": args.pso_inertia_end,
            "pso_cognitive": args.pso_cognitive,
            "pso_social": args.pso_social,
            "pso_velocity_limit_ratio": args.pso_velocity_limit_ratio,
            "pso_random_seed": args.pso_random_seed,
        },
        output_dir / "bo_best_weights.json",
    )
    _write_readme(
        output_dir=output_dir,
        args=args,
        summary_rows=summary_rows,
        fixed_result=fixed_result,
        bo_result=bo_result,
        pso_result=pso_result,
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
    print(
        f"pso best: avg_total_time={pso_result.best_objective:.3f}, "
        f"weights={json.dumps(pso_result.best_weights, sort_keys=True)}"
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
    if args.common_initial_random_count < 0:
        raise ValueError("--common-initial-random-count must be non-negative")
    if args.bo_initial_local_trials < 0:
        raise ValueError("--bo-initial-local-trials must be non-negative")
    if args.bo_initial_random_trials < 0:
        raise ValueError("--bo-initial-random-trials must be non-negative")
    if args.bo_candidate_count < 1:
        raise ValueError("--bo-candidate-count must be at least 1")
    if args.pso_swarm_size < 1:
        raise ValueError("--pso-swarm-size must be at least 1")
    if args.pso_evaluations < 0:
        raise ValueError("--pso-evaluations must be non-negative")
    if args.pso_evaluations and args.pso_evaluations < args.pso_swarm_size:
        raise ValueError("--pso-evaluations must be 0 or at least --pso-swarm-size")
    if args.pso_velocity_limit_ratio <= 0.0:
        raise ValueError("--pso-velocity-limit-ratio must be positive")
    if args.pso_cognitive < 0.0 or args.pso_social < 0.0:
        raise ValueError("--pso-cognitive and --pso-social must be non-negative")
    if (args.weight_min is None) != (args.weight_max is None):
        raise ValueError("--weight-min and --weight-max must be provided together")
    if args.weight_min is not None and args.weight_min >= args.weight_max:
        raise ValueError("--weight-min must be smaller than --weight-max")
    if args.max_planned_spawn_step < 0:
        raise ValueError("--max-planned-spawn-step must be non-negative")
    if args.scenarios_per_case < 0:
        raise ValueError("--scenarios-per-case must be non-negative")
    if args.scenario_set not in SCENARIO_SET_CASE_IDS:
        raise ValueError(f"Unknown scenario set: {args.scenario_set}")
    if args.scenario_set != "random":
        case_count = len(SCENARIO_SET_CASE_IDS[args.scenario_set])
        agv_count_count = len(_pdf_total_agv_options(args))
        if args.scenario_set == "pdf_train" and args.total_agvs != PDF_TRAIN_TOTAL_AGVS:
            raise ValueError("pdf_train uses exactly 20 AGVs; use pdf_test for 16/20/24")
        if args.scenarios_per_case == 0 and args.runs % (case_count * agv_count_count) != 0:
            raise ValueError(
                "--runs must be divisible by the number of PDF scenario cases times AGV counts "
                "unless --scenarios-per-case is provided"
            )
    if args.shared_area_capacity < 1:
        raise ValueError("--shared-area-capacity must be at least 1")
    if args.total_agvs < len(DIRECTIONS):
        raise ValueError("--total-agvs must be at least the number of directions")
    if args.min_agvs_per_direction < 1:
        raise ValueError("--min-agvs-per-direction must be at least 1")
    if args.max_agvs_per_direction < args.min_agvs_per_direction:
        raise ValueError("--max-agvs-per-direction must be >= --min-agvs-per-direction")
    if args.min_agvs_per_direction * len(DIRECTIONS) > args.total_agvs:
        raise ValueError("minimum per-direction AGV count is too high for --total-agvs")
    if args.max_agvs_per_direction * len(DIRECTIONS) < args.total_agvs:
        raise ValueError("maximum per-direction AGV count is too low for --total-agvs")


def _effective_pso_evaluations(args: argparse.Namespace) -> int:
    if args.pso_evaluations:
        return args.pso_evaluations
    return max(args.bo_trials, args.pso_swarm_size)


def _effective_scenarios_per_case(args: argparse.Namespace) -> int:
    if args.scenario_set == "random":
        return 0
    if args.scenarios_per_case:
        return args.scenarios_per_case
    return args.runs // (
        len(SCENARIO_SET_CASE_IDS[args.scenario_set])
        * len(_pdf_total_agv_options(args))
    )


def _effective_scenario_count(args: argparse.Namespace) -> int:
    if args.scenario_set == "random":
        return args.runs
    return (
        _effective_scenarios_per_case(args)
        * len(SCENARIO_SET_CASE_IDS[args.scenario_set])
        * len(_pdf_total_agv_options(args))
    )


def _pdf_total_agv_options(args: argparse.Namespace) -> tuple[int, ...]:
    if args.scenario_set == "pdf_train":
        return (PDF_TRAIN_TOTAL_AGVS,)
    values = tuple(
        int(part.strip())
        for part in str(args.test_agv_counts).split(",")
        if part.strip()
    )
    if not values:
        raise ValueError("--test-agv-counts must include at least one AGV count")
    invalid = sorted(set(values) - set(PDF_CASE_TOTAL_AGV_OPTIONS))
    if invalid:
        raise ValueError(f"Unsupported PDF test AGV count(s): {invalid}")
    return values


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


def _build_scenarios_from_args(args: argparse.Namespace) -> list[ScenarioSpec]:
    if args.scenario_set == "random":
        return _generate_scenarios(
            count=args.runs,
            total_agvs=args.total_agvs,
            seed_base=args.run_seed_base,
            scenario_seed=args.scenario_seed,
            max_planned_spawn_step=args.max_planned_spawn_step,
            min_agvs_per_direction=args.min_agvs_per_direction,
            max_agvs_per_direction=args.max_agvs_per_direction,
        )
    case_ids = SCENARIO_SET_CASE_IDS[args.scenario_set]
    scenarios_per_case = (
        args.scenarios_per_case
        if args.scenarios_per_case
        else _effective_scenarios_per_case(args)
    )
    return _generate_pdf_case_scenarios(
        case_ids=case_ids,
        scenarios_per_case=scenarios_per_case,
        total_agv_options=_pdf_total_agv_options(args),
        seed_base=args.run_seed_base,
        scenario_seed=args.scenario_seed,
        split="train" if args.scenario_set == "pdf_train" else "test",
    )


def _generate_pdf_case_scenarios(
    case_ids: tuple[int, ...],
    scenarios_per_case: int,
    total_agv_options: tuple[int, ...] | int,
    seed_base: int,
    scenario_seed: int,
    split: str,
) -> list[ScenarioSpec]:
    rng = random.Random(scenario_seed)
    scenarios = []
    run_index = 0
    if isinstance(total_agv_options, int):
        total_agv_options = (total_agv_options,)
    for total_agvs in total_agv_options:
        for case_id in case_ids:
            for variant_index in range(scenarios_per_case):
                scenarios.append(
                    _build_pdf_case_scenario(
                        case_id=case_id,
                        variant_index=variant_index,
                        run_index=run_index,
                        seed=seed_base + run_index,
                        total_agvs=total_agvs,
                        rng=rng,
                        split=split,
                    )
                )
                run_index += 1
    return scenarios


def _build_pdf_case_scenario(
    case_id: int,
    variant_index: int,
    run_index: int,
    seed: int,
    total_agvs: int,
    rng: random.Random,
    split: str,
) -> ScenarioSpec:
    if case_id not in SCENARIO_CASES:
        raise ValueError(f"Unknown PDF scenario case: {case_id}")

    case = SCENARIO_CASES[case_id]
    dominant_approach = ""
    dominant_approach_count = 0
    dominant_exit = ""
    dominant_exit_count = 0
    early_departures = 0

    if case_id == 1:
        allocation = _balanced_direction_counts(total_agvs)
        exit_counts = _balanced_direction_counts(total_agvs)
        spawn_plan = _staggered_spawn_plan(allocation, step=NORMAL_DEPARTURE_STEP)
    elif case_id == 2:
        allocation = _balanced_direction_counts(total_agvs)
        exit_counts = _balanced_direction_counts(total_agvs)
        early_departures = _early_departure_count(total_agvs)
        spawn_plan = _burst_spawn_plan(allocation, early_departures, rng)
    elif case_id == 3:
        dominant_approach = _rotating_direction(variant_index)
        allocation = _skewed_direction_counts(dominant_approach, total_agvs, rng)
        dominant_approach_count = allocation[dominant_approach]
        exit_counts = _balanced_direction_counts(total_agvs)
        spawn_plan = _staggered_spawn_plan(allocation, step=NORMAL_DEPARTURE_STEP)
    elif case_id == 4:
        dominant_exit = _rotating_direction(variant_index)
        allocation = _balanced_direction_counts(total_agvs)
        exit_counts = _skewed_direction_counts(dominant_exit, total_agvs, rng)
        dominant_exit_count = exit_counts[dominant_exit]
        spawn_plan = _staggered_spawn_plan(allocation, step=NORMAL_DEPARTURE_STEP)
    elif case_id == 5:
        dominant_approach = _rotating_direction(variant_index)
        dominant_exit = _rotating_direction(variant_index + 1)
        allocation = _skewed_direction_counts(dominant_approach, total_agvs, rng)
        exit_counts = _skewed_direction_counts(dominant_exit, total_agvs, rng)
        dominant_approach_count = allocation[dominant_approach]
        dominant_exit_count = exit_counts[dominant_exit]
        early_departures = _early_departure_count(total_agvs)
        spawn_plan = _burst_spawn_plan(allocation, early_departures, rng)
    else:
        raise ValueError(f"Unsupported PDF scenario case: {case_id}")

    goal_plan = _goal_plan_from_exit_counts(allocation, exit_counts, rng)
    scenario_split = "test_only" if case_id == 5 else "train_test"
    if split == "train" and not case["use_train"]:
        raise ValueError(f"{case['label']} is not available for training")
    if split == "test" and not case["use_test"]:
        raise ValueError(f"{case['label']} is not available for testing")

    return ScenarioSpec(
        run_index=run_index,
        seed=seed,
        allocation=allocation,
        spawn_plan=spawn_plan,
        goal_plan=goal_plan,
        scenario_case_id=case_id,
        scenario_case_name=case["name"],
        scenario_case_label=case["label"],
        scenario_split=scenario_split,
        dominant_approach=dominant_approach,
        dominant_approach_count=dominant_approach_count,
        dominant_exit=dominant_exit,
        dominant_exit_count=dominant_exit_count,
        early_departures=early_departures,
        total_agvs=total_agvs,
    )


def _generate_scenarios(
    count: int,
    total_agvs: int,
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
            total_agvs,
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
                total_agvs=total_agvs,
            )
        )
    return scenarios


def _balanced_direction_counts(total_agvs: int) -> dict[str, int]:
    if total_agvs % len(DIRECTIONS) != 0:
        raise ValueError("balanced PDF scenarios require total_agvs divisible by 4")
    return {direction: total_agvs // len(DIRECTIONS) for direction in DIRECTIONS}


def _skewed_direction_counts(
    dominant_direction: str,
    total_agvs: int,
    rng: random.Random,
) -> dict[str, int]:
    lower, upper = DOMINANT_COUNT_RANGE_BY_TOTAL_AGVS[total_agvs]
    dominant_count = rng.randint(lower, upper)
    other_directions = [
        direction
        for direction in DIRECTIONS
        if direction != dominant_direction
    ]
    other_counts = _random_positive_counts(
        total=total_agvs - dominant_count,
        bucket_count=len(other_directions),
        rng=rng,
    )
    counts = {dominant_direction: dominant_count}
    counts.update(dict(zip(other_directions, other_counts)))
    return {direction: counts[direction] for direction in DIRECTIONS}


def _random_positive_counts(
    total: int,
    bucket_count: int,
    rng: random.Random,
) -> list[int]:
    if total < bucket_count:
        raise ValueError("total must be at least bucket_count for positive random counts")
    max_count = total // 2
    candidates = []
    for counts in _positive_count_compositions(total, bucket_count):
        if all(count <= max_count for count in counts):
            candidates.append(counts)
    if not candidates:
        raise ValueError(
            f"Could not distribute total={total} into {bucket_count} positive "
            f"counts with each count <= {max_count}"
        )
    return list(rng.choice(candidates))


def _positive_count_compositions(total: int, bucket_count: int) -> list[tuple[int, ...]]:
    if bucket_count == 1:
        return [(total,)]
    compositions = []
    for count in range(1, total - bucket_count + 2):
        for tail in _positive_count_compositions(total - count, bucket_count - 1):
            compositions.append((count, *tail))
    return compositions


def _early_departure_count(total_agvs: int) -> int:
    return total_agvs // 2


def _rotating_direction(variant_index: int) -> str:
    return DOMINANT_ROTATION[variant_index % len(DOMINANT_ROTATION)]


def _staggered_spawn_plan(
    allocation: dict[str, int],
    step: int,
) -> dict[str, list[int]]:
    return {
        direction: [index * step for index in range(allocation[direction])]
        for direction in DIRECTIONS
    }


def _burst_spawn_plan(
    allocation: dict[str, int],
    early_departures: int,
    rng: random.Random,
) -> dict[str, list[int]]:
    robots = [
        direction
        for direction in DIRECTIONS
        for _ in range(allocation[direction])
    ]
    rng.shuffle(robots)
    plan = {direction: [] for direction in DIRECTIONS}
    for index, direction in enumerate(robots):
        if index < early_departures:
            start = rng.randint(*EARLY_DEPARTURE_RANGE)
        else:
            start = rng.randint(*LATE_DEPARTURE_RANGE)
        plan[direction].append(start)
    return {
        direction: sorted(plan[direction])
        for direction in DIRECTIONS
    }


def _goal_plan_from_exit_counts(
    allocation: dict[str, int],
    exit_counts: dict[str, int],
    rng: random.Random,
) -> dict[str, list[str]]:
    total_starts = sum(allocation.values())
    total_exits = sum(exit_counts.values())
    if total_starts != total_exits:
        raise ValueError("allocation and exit_counts must have the same total")

    entries = [
        direction
        for direction in DIRECTIONS
        for _ in range(allocation[direction])
    ]
    rng.shuffle(entries)
    quotas = dict(exit_counts)
    plan = {direction: [] for direction in DIRECTIONS}
    exit_node_by_direction = {
        exit_direction: exit_node
        for exit_node, exit_direction in FCFSCrossExperiment.EXIT_DIRECTION_BY_NODE.items()
    }

    def feasible(next_index: int) -> bool:
        remaining_entries = entries[next_index:]
        for exit_direction, remaining_quota in quotas.items():
            available = sum(
                entry_direction != exit_direction
                for entry_direction in remaining_entries
            )
            if remaining_quota > available:
                return False
        return True

    def assign(index: int) -> bool:
        if index == len(entries):
            return all(value == 0 for value in quotas.values())
        entry_direction = entries[index]
        candidates = [
            exit_direction
            for exit_direction in DOMINANT_ROTATION
            if exit_direction != entry_direction and quotas[exit_direction] > 0
        ]
        rng.shuffle(candidates)
        candidates.sort(key=lambda direction: quotas[direction], reverse=True)
        for exit_direction in candidates:
            quotas[exit_direction] -= 1
            plan[entry_direction].append(exit_node_by_direction[exit_direction])
            if feasible(index + 1) and assign(index + 1):
                return True
            plan[entry_direction].pop()
            quotas[exit_direction] += 1
        return False

    if not feasible(0) or not assign(0):
        raise ValueError(
            f"Could not build feasible goal plan for allocation={allocation}, "
            f"exit_counts={exit_counts}"
        )
    return {
        direction: list(plan[direction])
        for direction in DIRECTIONS
    }


def _load_bo_weights_file(path: Path) -> dict[str, float]:
    return _load_weights_file(path, preferred_key="best_bo_weights")


def _load_weights_file(path: Path, preferred_key: str) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if preferred_key in payload:
        return payload[preferred_key]
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
    initial_random_vectors: list[np.ndarray],
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
    shared_initial_vectors = [
        _clip_vector(np.array(vector, dtype=float), bounds)
        for vector in initial_random_vectors
    ]

    for trial_index in range(trials):
        gp_mu = math.nan
        gp_sigma = math.nan
        lcb = math.nan
        shared_count = len(shared_initial_vectors)
        search_index = trial_index - shared_count
        if trial_index < shared_count:
            vector = shared_initial_vectors[trial_index]
            source = "shared_initial_random"
        elif search_index == 0:
            vector = initial_vector
            source = "fixed_greedy_best_anchor"
        elif search_index <= len(BO_PRIOR_WEIGHT_CANDIDATES):
            vector = _weights_to_vector(
                BO_PRIOR_WEIGHT_CANDIDATES[search_index - 1],
                bounds,
            )
            source = "structural_prior"
        elif search_index < 1 + len(BO_PRIOR_WEIGHT_CANDIDATES) + initial_local_trials:
            vector = _local_weight_vector(
                rng,
                center=initial_vector,
                bounds=bounds,
                scale_ratio=0.18,
            )
            source = "local_perturbation"
        elif search_index < 1 + len(BO_PRIOR_WEIGHT_CANDIDATES) + initial_local_trials + initial_random_trials:
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


def _run_particle_swarm_optimization(
    scenarios: list[ScenarioSpec],
    swarm_size: int,
    evaluations: int,
    bounds: dict[str, tuple[float, float]],
    initial_vectors: list[np.ndarray],
    rng_seed: int,
    inertia_start: float,
    inertia_end: float,
    cognitive_coefficient: float,
    social_coefficient: float,
    velocity_limit_ratio: float,
    max_steps: int,
    corridor_length: int,
    west_exit_extension: int,
    admission_window_steps: int,
    shared_area_capacity: int,
) -> PSOResult:
    """Optimize the five heuristic weights with Particle Swarm Optimization."""

    if evaluations < swarm_size:
        raise ValueError("PSO evaluations must be at least swarm_size")

    rng = np.random.default_rng(rng_seed)
    lower, upper = _bounds_arrays(bounds)
    velocity_limit = (upper - lower) * velocity_limit_ratio
    positions = _initial_swarm_positions(
        swarm_size=swarm_size,
        bounds=bounds,
        initial_vectors=initial_vectors,
        rng=rng,
    )
    velocities = rng.uniform(
        low=-velocity_limit,
        high=velocity_limit,
        size=(swarm_size, len(BO_FEATURES)),
    )
    personal_best_positions = positions.copy()
    personal_best_objectives = np.full(swarm_size, math.inf, dtype=float)
    global_best_position: np.ndarray | None = None
    global_best_objective = math.inf
    trial_rows: list[dict] = []
    evaluation_index = 0

    def evaluate_particle(
        particle_index: int,
        iteration_index: int,
        source: str,
        inertia_weight: float,
    ) -> None:
        nonlocal evaluation_index, global_best_position, global_best_objective

        weights = _complete_heuristic_weights(_vector_to_weights(positions[particle_index]))
        objective = _objective_avg_total_time(
            weights=weights,
            scenarios=scenarios,
            max_steps=max_steps,
            corridor_length=corridor_length,
            west_exit_extension=west_exit_extension,
            admission_window_steps=admission_window_steps,
            shared_area_capacity=shared_area_capacity,
            policy="pso_heuristic",
        )
        if objective < personal_best_objectives[particle_index]:
            personal_best_objectives[particle_index] = objective
            personal_best_positions[particle_index] = positions[particle_index].copy()
        if objective < global_best_objective:
            global_best_objective = objective
            global_best_position = positions[particle_index].copy()

        row = {
            "evaluation_index": evaluation_index,
            "iteration_index": iteration_index,
            "particle_index": particle_index,
            "source": source,
            "objective_avg_total_time": objective,
            "personal_best_avg_total_time": personal_best_objectives[particle_index],
            "best_so_far_avg_total_time": global_best_objective,
            "inertia_weight": inertia_weight,
            "velocity_norm": float(np.linalg.norm(velocities[particle_index])),
            "weights_json": json.dumps(weights, sort_keys=True),
        }
        row.update({feature: weights[feature] for feature in BO_FEATURES})
        trial_rows.append(row)
        evaluation_index += 1

    for particle_index in range(swarm_size):
        evaluate_particle(
            particle_index=particle_index,
            iteration_index=0,
            source="initial_swarm",
            inertia_weight=inertia_start,
        )

    update_budget = evaluations - swarm_size
    total_update_iterations = max(1, math.ceil(update_budget / swarm_size))
    iteration_index = 0
    while evaluation_index < evaluations:
        iteration_index += 1
        progress = (
            (iteration_index - 1) / max(1, total_update_iterations - 1)
        )
        inertia_weight = inertia_start + (inertia_end - inertia_start) * progress
        assert global_best_position is not None
        for particle_index in range(swarm_size):
            if evaluation_index >= evaluations:
                break
            r_personal = rng.random(len(BO_FEATURES))
            r_global = rng.random(len(BO_FEATURES))
            velocities[particle_index] = (
                inertia_weight * velocities[particle_index]
                + cognitive_coefficient
                * r_personal
                * (personal_best_positions[particle_index] - positions[particle_index])
                + social_coefficient
                * r_global
                * (global_best_position - positions[particle_index])
            )
            velocities[particle_index] = np.clip(
                velocities[particle_index],
                -velocity_limit,
                velocity_limit,
            )
            positions[particle_index] = np.clip(
                positions[particle_index] + velocities[particle_index],
                lower,
                upper,
            )
            evaluate_particle(
                particle_index=particle_index,
                iteration_index=iteration_index,
                source="swarm_update",
                inertia_weight=inertia_weight,
            )

    assert global_best_position is not None
    return PSOResult(
        best_weights=_complete_heuristic_weights(_vector_to_weights(global_best_position)),
        best_objective=global_best_objective,
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


def _common_initial_weight_vectors(
    count: int,
    bounds: dict[str, tuple[float, float]],
    rng_seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    return [
        _random_weight_vector(rng, bounds)
        for _ in range(count)
    ]


def _initial_swarm_positions(
    swarm_size: int,
    bounds: dict[str, tuple[float, float]],
    initial_vectors: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    positions = []
    for vector in initial_vectors[:swarm_size]:
        positions.append(_clip_vector(np.array(vector, dtype=float), bounds))
    while len(positions) < swarm_size:
        positions.append(_random_weight_vector(rng, bounds))
    return np.vstack(positions)


def _clip_vector(
    vector: np.ndarray,
    bounds: dict[str, tuple[float, float]],
) -> np.ndarray:
    lower, upper = _bounds_arrays(bounds)
    return np.clip(vector, lower, upper)


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
    total_agvs: int,
    fixed_weights: dict[str, float],
    bo_weights: dict[str, float],
    pso_weights: dict[str, float],
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
        "pso_heuristic": pso_weights,
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
            rows.append(_metrics_to_run_row(scenario, policy, metrics, total_agvs))
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
    total_agvs: int,
) -> dict:
    return {
        "run_index": scenario.run_index,
        "scenario_seed": scenario.seed,
        "scenario_case_id": scenario.scenario_case_id,
        "scenario_case_name": scenario.scenario_case_name,
        "policy": policy,
        "total_agvs": scenario.total_agvs,
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


def _scenario_to_settings_row(scenario: ScenarioSpec, total_agvs: int) -> dict:
    return {
        "run_index": scenario.run_index,
        "scenario_seed": scenario.seed,
        "scenario_case_id": scenario.scenario_case_id,
        "scenario_case_name": scenario.scenario_case_name,
        "scenario_case_label": scenario.scenario_case_label,
        "scenario_split": scenario.scenario_split,
        "dominant_approach": scenario.dominant_approach,
        "dominant_approach_count": scenario.dominant_approach_count,
        "dominant_exit": scenario.dominant_exit,
        "dominant_exit_count": scenario.dominant_exit_count,
        "early_departures": scenario.early_departures,
        "total_agvs": scenario.total_agvs,
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
        pso = times["pso_heuristic"]
        comparison_rows.append(
            {
                "run_index": run_index,
                "scenario_seed": policies["fcfs"]["scenario_seed"],
                "scenario_case_id": policies["fcfs"]["scenario_case_id"],
                "scenario_case_name": policies["fcfs"]["scenario_case_name"],
                "fcfs_total_time": fcfs,
                "fixed_heuristic_total_time": fixed,
                "bo_heuristic_total_time": bo,
                "pso_heuristic_total_time": pso,
                "best_policy": ",".join(best_policies),
                "num_best_policies": len(best_policies),
                "fixed_minus_fcfs": fixed - fcfs,
                "fixed_improvement_pct_vs_fcfs": _improvement_pct(fcfs, fixed),
                "bo_minus_fcfs": bo - fcfs,
                "bo_improvement_pct_vs_fcfs": _improvement_pct(fcfs, bo),
                "bo_minus_fixed": bo - fixed,
                "bo_improvement_pct_vs_fixed": _improvement_pct(fixed, bo),
                "pso_minus_fcfs": pso - fcfs,
                "pso_improvement_pct_vs_fcfs": _improvement_pct(fcfs, pso),
                "pso_minus_fixed": pso - fixed,
                "pso_improvement_pct_vs_fixed": _improvement_pct(fixed, pso),
                "pso_minus_bo": pso - bo,
                "pso_improvement_pct_vs_bo": _improvement_pct(bo, pso),
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
                int(row["pso_heuristic_total_time"]),
            ]
        )
        best_total_time = times[0]
        second_best_total_time = times[1]
        rows.append(
            {
                "run_index": row["run_index"],
                "scenario_seed": row["scenario_seed"],
                "scenario_case_id": row["scenario_case_id"],
                "scenario_case_name": row["scenario_case_name"],
                "allocation_json": json.dumps(scenario.allocation, sort_keys=True),
                "spawn_plan_json": json.dumps(scenario.spawn_plan, sort_keys=True),
                "goal_plan_json": json.dumps(scenario.goal_plan, sort_keys=True),
                "fcfs_total_time": row["fcfs_total_time"],
                "fixed_heuristic_total_time": row["fixed_heuristic_total_time"],
                "bo_heuristic_total_time": row["bo_heuristic_total_time"],
                "pso_heuristic_total_time": row["pso_heuristic_total_time"],
                "best_policy": row["best_policy"],
                "best_total_time": best_total_time,
                "second_best_total_time": second_best_total_time,
                "margin_to_second": second_best_total_time - best_total_time,
                "is_tie": int(row["num_best_policies"]) > 1,
            }
        )
    return rows


def _summarize_runs(
    run_rows: list[dict],
    comparison_rows: list[dict],
    total_agvs: int,
) -> list[dict]:
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
                "completed_runs": sum(
                    int(row["completed"]) == int(row["total_agvs"])
                    for row in rows
                ),
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
    pso_result: PSOResult,
) -> None:
    lines = [
        "# Total-Time Policy Comparison",
        "",
        "## Common setup",
        "",
        "| item | value |",
        "|---|---:|",
        f"| AGVs | {args.total_agvs} |",
        f"| scenario_set | {args.scenario_set} |",
        f"| scenario cases | {list(SCENARIO_SET_CASE_IDS[args.scenario_set])} |",
        f"| scenario AGV counts | {list(_pdf_total_agv_options(args)) if args.scenario_set != 'random' else [args.total_agvs]} |",
        f"| dominant count ranges | {DOMINANT_COUNT_RANGE_BY_TOTAL_AGVS} |",
        f"| scenarios per case | {_effective_scenarios_per_case(args)} |",
        f"| scenarios per cycle | {_effective_scenario_count(args)} |",
        f"| corridor_length | {args.corridor_length} |",
        f"| west_exit_extension | {args.west_exit_extension} |",
        f"| admission_window_steps | {args.admission_window_steps} |",
        f"| shared_area_capacity | {args.shared_area_capacity} |",
        f"| max_planned_spawn_step | {args.max_planned_spawn_step} |",
        f"| per-direction AGV count range | {args.min_agvs_per_direction}-{args.max_agvs_per_direction} |",
        f"| fixed weight sets | {args.fixed_weight_set_count} |",
        f"| common initial random weights | {args.common_initial_random_count} |",
        "",
        "Each run samples one fixed scenario: NORTH/SOUTH/WEST/EAST AGV counts, "
        "per-AGV planned start times, and random end-point assignments. The same "
        "scenario is replayed for FCFS, Fixed Heuristic, BO Heuristic, and PSO Heuristic.",
        "",
        "`pdf_train` generates Scenario 1-4 with the same number of scenarios per "
        "case using only 20 AGVs. `pdf_test` generates Scenario 1-5 across "
        "the configured test AGV counts, where Scenario 5 is marked as "
        "test-only mixed simulation.",
        "",
        "Scenario 3 and Scenario 4 do not use fixed dominant counts. The dominant "
        "approach or exit is sampled from the AGV-count-specific range "
        "`16: 8-12`, `20: 10-14`, and `24: 12-16`; the remaining AGVs are "
        "randomly distributed across the other three directions with at least "
        "one AGV per direction and no direction receiving more than half of the "
        "non-dominant remainder.",
        "",
        "One cycle means running the full set of sampled scenarios once. FCFS is "
        "executed for one cycle as the deterministic baseline. Fixed Heuristic "
        "evaluates the pre-composed weight sets over the same cycle and keeps the "
        "lowest-average `total_time` set. BO and PSO optimize the same five "
        "weights over the same scenario cycle.",
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
        f"to minimize average `total_time` over the same {_effective_scenario_count(args)}-scenario cycle.",
        "",
        f"- BO trials: `{args.bo_trials if not args.bo_weights_file else 0}`",
        f"- BO scenarios per trial: `{args.runs}`",
        f"- BO initial local trials: `{args.bo_initial_local_trials}`",
        f"- BO initial random trials: `{args.bo_initial_random_trials}`",
        f"- BO exploration mode: `{args.bo_exploration_mode}`",
        f"- BO kappa: `{_bo_mode_config(args.bo_exploration_mode, args.bo_kappa)['kappa']}`",
        f"- BO best training avg total_time: `{bo_result.best_objective}`",
        "",
        "## Particle Swarm Optimization",
        "",
        "The PSO stage uses the same priority score structure as Basic Heuristic. "
        "One particle is one five-dimensional weight vector: `waiting`, "
        "`maneuver`, `exit_competition`, `path_conflict`, and `approach_queue`. "
        "The objective is to minimize average `total_time` over the training "
        "scenario cycle. The first random weight candidates are shared with BO "
        "so both optimizers start from the same random conditions.",
        "",
        f"- PSO swarm size: `{args.pso_swarm_size}`",
        f"- PSO evaluations: `{0 if args.pso_weights_file else _effective_pso_evaluations(args)}`",
        f"- PSO inertia: `{args.pso_inertia_start}` -> `{args.pso_inertia_end}`",
        f"- PSO cognitive coefficient: `{args.pso_cognitive}`",
        f"- PSO social coefficient: `{args.pso_social}`",
        f"- PSO velocity limit ratio: `{args.pso_velocity_limit_ratio}`",
        f"- PSO random seed: `{args.pso_random_seed}`",
        f"- PSO best training avg total_time: `{pso_result.best_objective}`",
        f"- PSO best weights: `{json.dumps(pso_result.best_weights, sort_keys=True)}`",
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
            "- `pso_trials.csv`: PSO particle evaluation trace with best-so-far total_time",
            "- `bo_best_weights.json`: selected BO/PSO weights, bounds, and optimizer settings",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
