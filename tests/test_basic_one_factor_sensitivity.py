"""Tests for Basic-Heuristic one-factor sensitivity analysis."""

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from scripts.run_basic_one_factor_sensitivity import (
    _load_or_generate_train_scenarios,
    _one_factor_weights,
    _run_one_factor_sensitivity,
    _sweep_values,
)
from scripts.run_total_time_bo_policy_comparison import BO_FEATURES


def test_sweep_values_include_endpoints_with_decimal_step():
    assert _sweep_values(-0.5, 0.5, 0.25) == [-0.5, -0.25, 0.0, 0.25, 0.5]


def test_one_factor_weights_control_other_features_at_one():
    weights = _one_factor_weights("waiting", -2.5)

    assert weights["waiting"] == -2.5
    for feature in BO_FEATURES:
        if feature != "waiting":
            assert weights[feature] == 1.0
    assert weights["remaining_path"] == 0.0
    assert weights["same_direction_backlog"] == 0.0


def test_train_scenarios_are_saved_and_reloaded(tmp_path):
    scenario_file = tmp_path / "train_scenarios.csv"
    generated = _load_or_generate_train_scenarios(
        scenario_file=scenario_file,
        scenarios_per_case=1,
        scenario_seed=200,
        run_seed_base=100,
        force_regenerate=True,
    )
    loaded = _load_or_generate_train_scenarios(
        scenario_file=scenario_file,
        scenarios_per_case=99,
        scenario_seed=999,
        run_seed_base=999,
        force_regenerate=False,
    )

    assert scenario_file.exists()
    assert len(generated) == 4
    assert [scenario.scenario_case_id for scenario in loaded] == [1, 2, 3, 4]
    assert [scenario.seed for scenario in loaded] == [100, 101, 102, 103]
    assert loaded[0].spawn_plan == generated[0].spawn_plan
    assert loaded[0].goal_plan == generated[0].goal_plan


def test_one_factor_sensitivity_uses_fixed_heuristic_and_selected_range(monkeypatch, tmp_path):
    scenarios = _load_or_generate_train_scenarios(
        scenario_file=tmp_path / "scenarios.csv",
        scenarios_per_case=1,
        scenario_seed=200,
        run_seed_base=100,
        force_regenerate=True,
    )
    calls = []

    def fake_run_single_policy(
        scenario,
        policy,
        weights,
        max_steps,
        corridor_length,
        west_exit_extension,
        admission_window_steps,
        shared_area_capacity,
    ):
        calls.append((policy, dict(weights)))
        return {
            "completed": scenario.total_agvs,
            "total_time": int(abs(weights["waiting"]) * 10) + scenario.scenario_case_id,
        }

    monkeypatch.setattr(
        "scripts.run_basic_one_factor_sensitivity._run_single_policy",
        fake_run_single_policy,
    )

    trial_rows, summary_rows, range_rows, raw_rows = _run_one_factor_sensitivity(
        scenarios=scenarios,
        features=["waiting"],
        values=[-0.25, 0.0, 0.25],
        range_tolerance_pct=0.0,
        checkpoint_dir=None,
        max_steps=10,
        corridor_length=8,
        west_exit_extension=0,
        admission_window_steps=2,
        shared_area_capacity=2,
    )

    assert len(trial_rows) == 3
    assert len(raw_rows) == 12
    assert summary_rows[0]["best_weight"] == 0.0
    assert range_rows[0]["lower"] == 0.0
    assert range_rows[0]["upper"] == 0.0
    assert {policy for policy, _ in calls} == {"fixed_heuristic"}
    assert all(call_weights["waiting"] in {-0.25, 0.0, 0.25} for _, call_weights in calls)
    assert all(call_weights["maneuver"] == 1.0 for _, call_weights in calls)
    assert {row["scenario_case_id"] for row in raw_rows} == {1, 2, 3, 4}
    assert {row["feature"] for row in raw_rows} == {"waiting"}
    assert {row["weight"] for row in raw_rows} == {-0.25, 0.0, 0.25}
