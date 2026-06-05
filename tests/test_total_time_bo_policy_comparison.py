"""Tests for the total_time-only BO policy comparison runner."""

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from rware.fcfs_cross_simulation import FCFSCrossExperiment
from scripts.run_total_time_bo_policy_comparison import (
    DIRECTIONS,
    FIXED_HEURISTIC_WEIGHTS,
    TOTAL_AGVS,
    _generate_scenarios,
    _run_single_policy,
)


def test_fcfs_cross_accepts_per_agv_spawn_plan():
    experiment = FCFSCrossExperiment(
        robots_by_direction={
            "NORTH": 1,
            "SOUTH": 1,
            "WEST": 1,
            "EAST": 1,
        },
        spawn_plan_by_direction={
            "NORTH": [0],
            "SOUTH": [3],
            "WEST": [1],
            "EAST": [5],
        },
        goal_plan_by_direction={
            "NORTH": ["S_EXIT"],
            "SOUTH": ["N_EXIT"],
            "WEST": ["E_EXIT"],
            "EAST": ["W_EXIT"],
        },
        corridor_length=8,
        admission_window_steps=1,
        random_seed=123,
    )

    metrics = experiment.run(max_steps=300)

    assert metrics["completed"] == 4
    assert metrics["spawn_plan_by_direction"] == {
        "NORTH": [0],
        "SOUTH": [3],
        "WEST": [1],
        "EAST": [5],
    }


def test_generated_scenarios_fix_total_agvs_start_times_and_goals():
    scenarios = _generate_scenarios(
        count=3,
        total_agvs=TOTAL_AGVS,
        seed_base=100,
        scenario_seed=200,
        max_planned_spawn_step=24,
    )

    assert len(scenarios) == 3
    for scenario in scenarios:
        assert sum(scenario.allocation.values()) == TOTAL_AGVS
        assert set(scenario.allocation) == set(DIRECTIONS)
        for direction in DIRECTIONS:
            count = scenario.allocation[direction]
            assert count >= 1
            assert len(scenario.spawn_plan[direction]) == count
            assert scenario.spawn_plan[direction] == sorted(scenario.spawn_plan[direction])
            assert len(scenario.goal_plan[direction]) == count


def test_fixed_and_bo_with_same_weights_match_on_same_scenario():
    scenario = _generate_scenarios(
        count=1,
        total_agvs=TOTAL_AGVS,
        seed_base=100,
        scenario_seed=201,
        max_planned_spawn_step=10,
    )[0]

    fixed = _run_single_policy(
        scenario=scenario,
        policy="fixed_heuristic",
        weights=FIXED_HEURISTIC_WEIGHTS,
        max_steps=500,
        corridor_length=8,
        west_exit_extension=0,
        admission_window_steps=1,
        shared_area_capacity=2,
    )
    bo = _run_single_policy(
        scenario=scenario,
        policy="bo_heuristic",
        weights=FIXED_HEURISTIC_WEIGHTS,
        max_steps=500,
        corridor_length=8,
        west_exit_extension=0,
        admission_window_steps=1,
        shared_area_capacity=2,
    )

    assert bo["total_time"] == fixed["total_time"]
