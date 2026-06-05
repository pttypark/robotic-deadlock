"""Tests for PSO heuristic integration in the total-time comparison script."""

import os
import sys

import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from scripts.run_total_time_bo_policy_comparison import (
    BO_FEATURES,
    DEFAULT_BO_WEIGHT_BOUNDS,
    _common_initial_weight_vectors,
    _generate_scenarios,
    _run_particle_swarm_optimization,
)


def test_common_initial_weight_vectors_are_reproducible_and_bounded():
    first = _common_initial_weight_vectors(
        count=20,
        bounds=DEFAULT_BO_WEIGHT_BOUNDS,
        rng_seed=42,
    )
    second = _common_initial_weight_vectors(
        count=20,
        bounds=DEFAULT_BO_WEIGHT_BOUNDS,
        rng_seed=42,
    )

    assert len(first) == 20
    assert all(vector.shape == (len(BO_FEATURES),) for vector in first)
    for left, right in zip(first, second):
        np.testing.assert_allclose(left, right)
        for index, feature in enumerate(BO_FEATURES):
            lower, upper = DEFAULT_BO_WEIGHT_BOUNDS[feature]
            assert lower <= left[index] <= upper


def test_pso_policy_returns_best_weights_and_particle_trace():
    scenarios = _generate_scenarios(
        count=1,
        total_agvs=4,
        seed_base=500,
        scenario_seed=600,
        max_planned_spawn_step=1,
        min_agvs_per_direction=1,
        max_agvs_per_direction=1,
    )
    initial_vectors = _common_initial_weight_vectors(
        count=3,
        bounds=DEFAULT_BO_WEIGHT_BOUNDS,
        rng_seed=42,
    )

    result = _run_particle_swarm_optimization(
        scenarios=scenarios,
        swarm_size=3,
        evaluations=5,
        bounds=DEFAULT_BO_WEIGHT_BOUNDS,
        initial_vectors=initial_vectors,
        rng_seed=42,
        inertia_start=0.7,
        inertia_end=0.4,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        velocity_limit_ratio=0.2,
        max_steps=300,
        corridor_length=5,
        west_exit_extension=0,
        admission_window_steps=1,
        shared_area_capacity=1,
    )

    assert set(BO_FEATURES).issubset(result.best_weights)
    assert result.best_objective > 0
    assert len(result.trial_rows) == 5
    assert result.trial_rows[0]["source"] == "initial_swarm"
    assert result.trial_rows[-1]["best_so_far_avg_total_time"] == result.best_objective
