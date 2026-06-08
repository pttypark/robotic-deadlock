from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import scripts.run_total_time_bo_policy_comparison as comparison_runner  # noqa: E402
from scripts.run_total_time_bo_policy_comparison import ScenarioSpec  # noqa: E402


ACTIVE_FEATURES = ("waiting", "maneuver", "exit_competition", "path_conflict")
DEFAULT_SCENARIO_FILE = (
    PROJECT_DIR
    / "final_output"
    / "basic_one_factor_sensitivity"
    / "train_scenarios_400.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "final_output"
    / "transfer_generalization_train_400_time_limited"
)
BOUNDS = {feature: (-1.0, 1.0) for feature in ACTIVE_FEATURES}

TRIAL_FIELDNAMES = [
    "policy",
    "evaluation_index",
    "iteration_index",
    "particle_index",
    "source",
    "elapsed_seconds",
    "elapsed_minutes",
    "avg_total_time",
    "best_so_far_avg_total_time",
    "weights_json",
    "gp_mu",
    "gp_sigma",
    "lcb",
    "inertia_weight",
    "velocity_norm",
    *ACTIVE_FEATURES,
]

RAW_FIELDNAMES = [
    "policy",
    "evaluation_index",
    "source",
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
    "weights_json",
    *ACTIVE_FEATURES,
]

CHECKPOINT_FIELDNAMES = [
    "policy",
    "checkpoint_label",
    "checkpoint_seconds",
    "evaluations_completed",
    "best_avg_total_time",
    "best_weights_json",
    *ACTIVE_FEATURES,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train Basic, BO, and PSO on the existing 400 single-cross training "
            "scenarios with a 5-minute checkpoint and 30-minute per-policy limit."
        )
    )
    parser.add_argument("--scenario-file", type=Path, default=DEFAULT_SCENARIO_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-seconds", type=float, default=1800.0)
    parser.add_argument("--checkpoint-seconds", type=float, default=300.0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--admission-window-steps", type=int, default=2)
    parser.add_argument("--shared-area-capacity", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260608)
    parser.add_argument("--bo-candidate-count", type=int, default=800)
    parser.add_argument("--bo-kappa", type=float, default=1.96)
    parser.add_argument("--bo-global-ratio", type=float, default=0.65)
    parser.add_argument("--bo-local-scale-ratio", type=float, default=0.18)
    parser.add_argument("--bo-length-scale", type=float, default=0.45)
    parser.add_argument("--bo-noise", type=float, default=1e-6)
    parser.add_argument("--pso-swarm-size", type=int, default=20)
    parser.add_argument("--pso-inertia-start", type=float, default=0.7)
    parser.add_argument("--pso-inertia-end", type=float, default=0.4)
    parser.add_argument("--pso-cognitive", type=float, default=1.4)
    parser.add_argument("--pso-social", type=float, default=1.4)
    parser.add_argument("--pso-velocity-limit", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_runner.BO_FEATURES = ACTIVE_FEATURES
    scenarios = _load_scenarios(args.scenario_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_trials: list[dict] = []
    all_raw: list[dict] = []
    checkpoints: list[dict] = []

    basic = _train_basic(scenarios, args)
    _extend_outputs("basic_heuristic", basic, all_trials, all_raw, checkpoints, args)
    basic_5 = _vectors_from_top_rows(basic["top_5min"])
    basic_30 = _vectors_from_top_rows(basic["top_30min"])

    bo = _train_bo(scenarios, args, initial_5min=basic_5, initial_30min=basic_30)
    _extend_outputs("bo_heuristic", bo, all_trials, all_raw, checkpoints, args)

    pso = _train_pso(scenarios, args, initial_5min=basic_5, initial_30min=basic_30)
    _extend_outputs("pso_heuristic", pso, all_trials, all_raw, checkpoints, args)

    _write_csv(args.output_dir / "train_trials_all_policies.csv", all_trials, TRIAL_FIELDNAMES)
    _write_csv(args.output_dir / "train_raw_runs_all_policies.csv", all_raw, RAW_FIELDNAMES)
    _write_csv(args.output_dir / "train_best_checkpoints.csv", checkpoints, CHECKPOINT_FIELDNAMES)
    _write_top_files(args.output_dir, "basic_heuristic", basic, args.top_k)
    _write_top_files(args.output_dir, "bo_heuristic", bo, args.top_k)
    _write_top_files(args.output_dir, "pso_heuristic", pso, args.top_k)
    _plot_best_total_time_over_policy_time(args.output_dir, all_trials)
    _write_summary(args.output_dir, checkpoints, scenarios, args)

    print(f"wrote output dir: {args.output_dir.resolve()}")
    for row in checkpoints:
        print(
            f"{row['policy']} {row['checkpoint_label']}: "
            f"best avg Total_Time={row['best_avg_total_time']} "
            f"weights={row['best_weights_json']}"
        )


def _train_basic(scenarios: list[ScenarioSpec], args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.random_seed)
    start = time.monotonic()
    trials: list[dict] = []
    raw_rows: list[dict] = []
    checkpoint_saved = False
    checkpoint_trials: list[dict] = []
    evaluation_index = 0
    anchors = _basic_anchor_vectors()

    while _elapsed(start) < args.train_seconds:
        if evaluation_index < len(anchors):
            vector = anchors[evaluation_index]
            source = "basic_l2_anchor"
        else:
            vector = _random_l2_vector(rng)
            source = "basic_random_l2"
        result = _evaluate_vector(
            policy="basic_heuristic",
            evaluation_index=evaluation_index,
            source=source,
            vector=vector,
            scenarios=scenarios,
            args=args,
        )
        trials.append(result["trial"])
        raw_rows.extend(result["raw_rows"])
        evaluation_index += 1
        if not checkpoint_saved and _elapsed(start) >= args.checkpoint_seconds:
            checkpoint_trials = list(trials)
            checkpoint_saved = True

    if not checkpoint_trials:
        checkpoint_trials = list(trials)
    return {
        "trials": trials,
        "raw_rows": raw_rows,
        "top_5min": _top_trials(checkpoint_trials, args.top_k),
        "top_30min": _top_trials(trials, args.top_k),
        "checkpoint_evaluations": len(checkpoint_trials),
        "final_evaluations": len(trials),
    }


def _train_bo(
    scenarios: list[ScenarioSpec],
    args: argparse.Namespace,
    initial_5min: list[np.ndarray],
    initial_30min: list[np.ndarray],
) -> dict:
    rng = np.random.default_rng(args.random_seed + 101)
    start = time.monotonic()
    trials: list[dict] = []
    raw_rows: list[dict] = []
    checkpoint_saved = False
    checkpoint_trials: list[dict] = []
    injected_30min = False
    vectors: list[np.ndarray] = []
    objectives: list[float] = []
    pending_5min = list(initial_5min)
    pending_30min = list(initial_30min)
    evaluation_index = 0

    while _elapsed(start) < args.train_seconds:
        source = "bo_lcb_surrogate"
        gp_mu = gp_sigma = lcb = ""
        if pending_5min:
            vector = pending_5min.pop(0)
            source = "basic_5min_top20_warm_start"
        elif checkpoint_saved and pending_30min:
            vector = pending_30min.pop(0)
            source = "basic_30min_top20_warm_start"
        elif len(vectors) < 3:
            vector = _random_l2_vector(rng)
            source = "bo_random_l2_seed"
        else:
            suggested, gp_mu, gp_sigma, lcb = comparison_runner._suggest_lcb_candidate(
                vectors=np.vstack(vectors),
                objectives=np.array(objectives, dtype=float),
                candidate_count=args.bo_candidate_count,
                bounds=BOUNDS,
                rng=rng,
                kappa=args.bo_kappa,
                global_ratio=args.bo_global_ratio,
                local_scale_ratio=args.bo_local_scale_ratio,
                length_scale=args.bo_length_scale,
                noise=args.bo_noise,
            )
            vector = _l2_normalize(suggested)

        result = _evaluate_vector(
            policy="bo_heuristic",
            evaluation_index=evaluation_index,
            source=source,
            vector=vector,
            scenarios=scenarios,
            args=args,
            gp_mu=gp_mu,
            gp_sigma=gp_sigma,
            lcb=lcb,
        )
        trials.append(result["trial"])
        raw_rows.extend(result["raw_rows"])
        vectors.append(_vector_from_trial(result["trial"]))
        objectives.append(float(result["trial"]["avg_total_time"]))
        evaluation_index += 1

        if not checkpoint_saved and _elapsed(start) >= args.checkpoint_seconds:
            checkpoint_trials = list(trials)
            checkpoint_saved = True
        if checkpoint_saved and not injected_30min:
            injected_30min = True

    if not checkpoint_trials:
        checkpoint_trials = list(trials)
    return {
        "trials": trials,
        "raw_rows": raw_rows,
        "top_5min": _top_trials(checkpoint_trials, args.top_k),
        "top_30min": _top_trials(trials, args.top_k),
        "checkpoint_evaluations": len(checkpoint_trials),
        "final_evaluations": len(trials),
    }


def _train_pso(
    scenarios: list[ScenarioSpec],
    args: argparse.Namespace,
    initial_5min: list[np.ndarray],
    initial_30min: list[np.ndarray],
) -> dict:
    rng = np.random.default_rng(args.random_seed + 202)
    start = time.monotonic()
    trials: list[dict] = []
    raw_rows: list[dict] = []
    checkpoint_saved = False
    checkpoint_trials: list[dict] = []
    injected_30min = False
    swarm_size = args.pso_swarm_size
    positions = _initial_positions(initial_5min, swarm_size, rng)
    velocities = rng.uniform(-args.pso_velocity_limit, args.pso_velocity_limit, size=positions.shape)
    personal_best_positions = positions.copy()
    personal_best_objectives = np.full(swarm_size, math.inf)
    global_best_position = positions[0].copy()
    global_best_objective = math.inf
    evaluation_index = 0
    iteration_index = 0

    while _elapsed(start) < args.train_seconds:
        progress = min(1.0, _elapsed(start) / max(1e-9, args.train_seconds))
        inertia = args.pso_inertia_start + (args.pso_inertia_end - args.pso_inertia_start) * progress
        for particle_index in range(swarm_size):
            if _elapsed(start) >= args.train_seconds:
                break
            vector = _l2_normalize(positions[particle_index])
            result = _evaluate_vector(
                policy="pso_heuristic",
                evaluation_index=evaluation_index,
                source="pso_swarm_update" if evaluation_index >= swarm_size else "basic_5min_swarm_initial",
                vector=vector,
                scenarios=scenarios,
                args=args,
                iteration_index=iteration_index,
                particle_index=particle_index,
                inertia_weight=inertia,
                velocity_norm=float(np.linalg.norm(velocities[particle_index])),
            )
            objective = float(result["trial"]["avg_total_time"])
            trials.append(result["trial"])
            raw_rows.extend(result["raw_rows"])
            if objective < personal_best_objectives[particle_index]:
                personal_best_objectives[particle_index] = objective
                personal_best_positions[particle_index] = vector.copy()
            if objective < global_best_objective:
                global_best_objective = objective
                global_best_position = vector.copy()
            evaluation_index += 1
            if not checkpoint_saved and _elapsed(start) >= args.checkpoint_seconds:
                checkpoint_trials = list(trials)
                checkpoint_saved = True
        if checkpoint_saved and not injected_30min and initial_30min:
            for index, vector in enumerate(initial_30min[:swarm_size]):
                positions[index] = vector
                velocities[index] = 0.0
            injected_30min = True
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = (
            inertia * velocities
            + args.pso_cognitive * r1 * (personal_best_positions - positions)
            + args.pso_social * r2 * (global_best_position - positions)
        )
        velocities = np.clip(velocities, -args.pso_velocity_limit, args.pso_velocity_limit)
        positions = np.vstack([_l2_normalize(vector) for vector in positions + velocities])
        iteration_index += 1

    if not checkpoint_trials:
        checkpoint_trials = list(trials)
    return {
        "trials": trials,
        "raw_rows": raw_rows,
        "top_5min": _top_trials(checkpoint_trials, args.top_k),
        "top_30min": _top_trials(trials, args.top_k),
        "checkpoint_evaluations": len(checkpoint_trials),
        "final_evaluations": len(trials),
    }


def _evaluate_vector(
    policy: str,
    evaluation_index: int,
    source: str,
    vector: np.ndarray,
    scenarios: list[ScenarioSpec],
    args: argparse.Namespace,
    iteration_index: int | str = "",
    particle_index: int | str = "",
    gp_mu: float | str = "",
    gp_sigma: float | str = "",
    lcb: float | str = "",
    inertia_weight: float | str = "",
    velocity_norm: float | str = "",
) -> dict:
    vector = _l2_normalize(vector)
    weights = _weights_from_vector(vector)
    raw_rows: list[dict] = []
    total_times: list[float] = []
    started = time.monotonic()
    for scenario in scenarios:
        metrics = comparison_runner._run_single_policy(
            scenario=scenario,
            policy=policy,
            weights=weights,
            max_steps=args.max_steps,
            corridor_length=args.corridor_length,
            west_exit_extension=args.west_exit_extension,
            admission_window_steps=args.admission_window_steps,
            shared_area_capacity=args.shared_area_capacity,
        )
        total_time = float(metrics["total_time"])
        total_times.append(total_time)
        row = {
            "policy": policy,
            "evaluation_index": evaluation_index,
            "source": source,
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
            "total_time": _fmt(total_time),
            "weights_json": _weights_json(weights),
        }
        row.update(_weight_columns(vector))
        raw_rows.append(row)

    avg_total_time = statistics.fmean(total_times)
    trial = {
        "policy": policy,
        "evaluation_index": evaluation_index,
        "iteration_index": iteration_index,
        "particle_index": particle_index,
        "source": source,
        "elapsed_seconds": _fmt(time.monotonic() - started),
        "elapsed_minutes": _fmt((time.monotonic() - started) / 60.0),
        "avg_total_time": _fmt(avg_total_time),
        "best_so_far_avg_total_time": "",
        "weights_json": _weights_json(weights),
        "gp_mu": _fmt_optional(gp_mu),
        "gp_sigma": _fmt_optional(gp_sigma),
        "lcb": _fmt_optional(lcb),
        "inertia_weight": _fmt_optional(inertia_weight),
        "velocity_norm": _fmt_optional(velocity_norm),
    }
    trial.update(_weight_columns(vector))
    return {"trial": trial, "raw_rows": raw_rows}


def _extend_outputs(
    policy: str,
    result: dict,
    all_trials: list[dict],
    all_raw: list[dict],
    checkpoints: list[dict],
    args: argparse.Namespace,
) -> None:
    best_so_far = math.inf
    for trial in result["trials"]:
        current = float(trial["avg_total_time"])
        best_so_far = min(best_so_far, current)
        trial["best_so_far_avg_total_time"] = _fmt(best_so_far)
        all_trials.append(trial)
    all_raw.extend(result["raw_rows"])
    checkpoints.append(
        _checkpoint_row(
            policy,
            "5min",
            args.checkpoint_seconds,
            result["checkpoint_evaluations"],
            result["top_5min"],
        )
    )
    checkpoints.append(
        _checkpoint_row(
            policy,
            "30min",
            args.train_seconds,
            result["final_evaluations"],
            result["top_30min"],
        )
    )


def _checkpoint_row(
    policy: str,
    label: str,
    seconds: float,
    evaluations_completed: int,
    top_rows: list[dict],
) -> dict:
    best = top_rows[0]
    row = {
        "policy": policy,
        "checkpoint_label": label,
        "checkpoint_seconds": _fmt(seconds),
        "evaluations_completed": evaluations_completed,
        "best_avg_total_time": best["avg_total_time"],
        "best_weights_json": best["weights_json"],
    }
    for feature in ACTIVE_FEATURES:
        row[feature] = best[feature]
    return row


def _write_top_files(output_dir: Path, policy: str, result: dict, top_k: int) -> None:
    _write_csv(output_dir / f"{policy}_top{top_k}_5min.csv", result["top_5min"], TRIAL_FIELDNAMES)
    _write_csv(output_dir / f"{policy}_top{top_k}_30min.csv", result["top_30min"], TRIAL_FIELDNAMES)


def _write_summary(
    output_dir: Path,
    checkpoints: list[dict],
    scenarios: list[ScenarioSpec],
    args: argparse.Namespace,
) -> None:
    settings = {
        "scenario_file": str(args.scenario_file),
        "scenario_count": len(scenarios),
        "train_layout": "single_cross",
        "test_layouts_planned": ["single_cross", "double_cross"],
        "active_features": list(ACTIVE_FEATURES),
        "priority_score": "w1*waiting + w2*maneuver + w3*exit_competition + w4*path_conflict",
        "feature_normalization": "FCFSCrossExperiment normalize_heuristic_features=True",
        "weight_normalization": "L2-normalized vector, each weight in [-1, 1]",
        "train_seconds_per_policy": args.train_seconds,
        "checkpoint_seconds": args.checkpoint_seconds,
        "top_k": args.top_k,
        "basic_warm_start_for_bo_pso": {
            "before_checkpoint": "basic_heuristic_top20_5min",
            "after_checkpoint": "basic_heuristic_top20_30min",
        },
        "checkpoints": checkpoints,
    }
    (output_dir / "train_settings_and_best.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _plot_best_total_time_over_policy_time(output_dir: Path, trials: list[dict]) -> None:
    if not trials:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local plotting install
        (output_dir / "plot_skipped.txt").write_text(
            f"Could not create plots because matplotlib is unavailable: {exc}\n",
            encoding="utf-8",
        )
        return

    colors = {
        "basic_heuristic": "#2ca02c",
        "bo_heuristic": "#1f77b4",
        "pso_heuristic": "#d62728",
    }
    labels = {
        "basic_heuristic": "Basic-Heuristic",
        "bo_heuristic": "BO-Heuristic",
        "pso_heuristic": "PSO-Heuristic",
    }

    grouped: dict[str, list[dict]] = {}
    for row in trials:
        grouped.setdefault(row["policy"], []).append(row)

    plt.figure(figsize=(11, 6.2))
    for policy in ("basic_heuristic", "bo_heuristic", "pso_heuristic"):
        rows = sorted(grouped.get(policy, []), key=lambda item: int(item["evaluation_index"]))
        if not rows:
            continue
        elapsed = []
        cumulative_seconds = 0.0
        best_values = []
        best = math.inf
        for row in rows:
            cumulative_seconds += float(row["elapsed_seconds"])
            elapsed.append(cumulative_seconds / 60.0)
            best = min(best, float(row["avg_total_time"]))
            best_values.append(best)
        plt.plot(
            elapsed,
            best_values,
            marker="o",
            markersize=3,
            linewidth=2.0,
            color=colors[policy],
            label=labels[policy],
        )
        if elapsed:
            plt.scatter(elapsed[-1], best_values[-1], s=70, color=colors[policy], edgecolor="black", zorder=5)
            plt.text(
                elapsed[-1],
                best_values[-1],
                f"  {best_values[-1]:.2f}",
                va="center",
                fontsize=9,
                color=colors[policy],
            )

    plt.axvline(5.0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.75, label="5 min checkpoint")
    plt.xlabel("Training time within each policy (minutes)")
    plt.ylabel("Best-so-far average Total_Time over 400 train scenarios")
    plt.title("Best Total_Time by Training Time")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "best_total_time_over_training_time.png", dpi=180)
    plt.close()

    for policy in ("basic_heuristic", "bo_heuristic", "pso_heuristic"):
        rows = sorted(grouped.get(policy, []), key=lambda item: int(item["evaluation_index"]))
        if not rows:
            continue
        elapsed = []
        cumulative_seconds = 0.0
        best_values = []
        best = math.inf
        for row in rows:
            cumulative_seconds += float(row["elapsed_seconds"])
            elapsed.append(cumulative_seconds / 60.0)
            best = min(best, float(row["avg_total_time"]))
            best_values.append(best)

        plt.figure(figsize=(9, 5.2))
        plt.plot(elapsed, best_values, marker="o", markersize=3, linewidth=2.2, color=colors[policy])
        plt.axvline(5.0, color="#666666", linestyle="--", linewidth=1.2, alpha=0.75)
        plt.xlabel("Training time within policy (minutes)")
        plt.ylabel("Best-so-far average Total_Time")
        plt.title(f"{labels[policy]} Best Total_Time by Training Time")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / f"{policy}_best_total_time_over_training_time.png", dpi=180)
        plt.close()


def _load_scenarios(path: Path) -> list[ScenarioSpec]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no scenarios found: {path}")
    return [_scenario_from_row(row) for row in rows]


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
        scenario_case_name=row.get("scenario_case_name") or "train_scenario",
        scenario_case_label=row.get("scenario_case_label") or "",
        scenario_split=row.get("scenario_split") or "train_test",
        dominant_approach=row.get("dominant_approach") or "",
        dominant_approach_count=int(row.get("dominant_approach_count") or 0),
        dominant_exit=row.get("dominant_exit") or "",
        dominant_exit_count=int(row.get("dominant_exit_count") or 0),
        early_departures=int(row.get("early_departures") or 0),
        total_agvs=int(row.get("total_agvs") or sum(allocation.values())),
    )


def _basic_anchor_vectors() -> list[np.ndarray]:
    anchors = [np.ones(len(ACTIVE_FEATURES), dtype=float)]
    for index in range(len(ACTIVE_FEATURES)):
        positive = np.zeros(len(ACTIVE_FEATURES), dtype=float)
        positive[index] = 1.0
        negative = np.zeros(len(ACTIVE_FEATURES), dtype=float)
        negative[index] = -1.0
        anchors.extend([positive, negative])
    return [_l2_normalize(vector) for vector in anchors]


def _initial_positions(initial_vectors: list[np.ndarray], swarm_size: int, rng: np.random.Generator) -> np.ndarray:
    positions = [_l2_normalize(vector) for vector in initial_vectors[:swarm_size]]
    while len(positions) < swarm_size:
        positions.append(_random_l2_vector(rng))
    return np.vstack(positions)


def _random_l2_vector(rng: np.random.Generator) -> np.ndarray:
    return _l2_normalize(rng.uniform(-1.0, 1.0, size=len(ACTIVE_FEATURES)))


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.array(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        vector = np.ones(len(ACTIVE_FEATURES), dtype=float)
        norm = float(np.linalg.norm(vector))
    return np.clip(vector / norm, -1.0, 1.0)


def _weights_from_vector(vector: np.ndarray) -> dict[str, float]:
    weights = {feature: float(value) for feature, value in zip(ACTIVE_FEATURES, _l2_normalize(vector))}
    weights["approach_queue"] = 0.0
    weights["remaining_path"] = 0.0
    weights["same_direction_backlog"] = 0.0
    return weights


def _vector_from_trial(trial: dict) -> np.ndarray:
    return np.array([float(trial[feature]) for feature in ACTIVE_FEATURES], dtype=float)


def _vectors_from_top_rows(rows: list[dict]) -> list[np.ndarray]:
    return [_vector_from_trial(row) for row in rows]


def _top_trials(trials: list[dict], top_k: int) -> list[dict]:
    return sorted(trials, key=lambda row: float(row["avg_total_time"]))[:top_k]


def _weight_columns(vector: np.ndarray) -> dict[str, str]:
    vector = _l2_normalize(vector)
    return {feature: f"{float(value):.6f}" for feature, value in zip(ACTIVE_FEATURES, vector)}


def _weights_json(weights: dict[str, float]) -> str:
    return json.dumps({key: round(float(value), 6) for key, value in weights.items()}, sort_keys=True)


def _fmt(value: float) -> str:
    return f"{float(value):.2f}"


def _fmt_optional(value: float | str) -> str:
    if value == "":
        return ""
    return f"{float(value):.6f}"


def _elapsed(start: float) -> float:
    return time.monotonic() - start


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
