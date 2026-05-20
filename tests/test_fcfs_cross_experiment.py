"""Tests for the 12-AGV A* + FCFS cross shared-area baseline."""

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from rware.agv_layouts import build_fcfs_cross_shared_area_layout
from rware.fcfs_cross_simulation import FCFSCrossExperiment


def test_fcfs_cross_layout_has_four_starts_exits_and_shared_area():
    layout = build_fcfs_cross_shared_area_layout()
    area = layout.interaction_areas[0]

    assert layout.name == "fcfs_cross_shared_area_v1"
    assert {"N_START", "S_START", "W_START", "E_START"}.issubset(layout.stations)
    assert {"N_EXIT", "S_EXIT", "W_EXIT", "E_EXIT"}.issubset(layout.stations)
    assert area.conflict_zone_nodes == {"CP_NW", "CP_NE", "CP_SW", "CP_SE"}
    assert area.priority_rule == "fcfs"


def test_fcfs_cross_completes_12_robots_and_reports_total_time():
    experiment = FCFSCrossExperiment(robots_per_direction=3)

    metrics = experiment.run(max_steps=500)

    assert metrics["policy"] == "A*_FCFS"
    assert metrics["robots"] == 12
    assert metrics["completed"] == 12
    assert metrics["total_time"] > 0


def test_fcfs_cross_initially_places_three_robots_per_start_side():
    experiment = FCFSCrossExperiment(robots_per_direction=3, random_seed=7)

    by_direction = {}
    for robot in experiment.active.values():
        by_direction.setdefault(robot.direction, []).append(robot)

    assert set(by_direction) == {"NORTH", "SOUTH", "WEST", "EAST"}
    for direction, robots in by_direction.items():
        assert len(robots) == 3
        assert {robot.current_node for robot in robots} == {experiment.START_BY_DIRECTION[direction]}
    assert len({robot.current_node for robot in experiment.active.values()}) == 4


def test_fcfs_cross_random_end_assignments_all_pass_conflict_area():
    experiment = FCFSCrossExperiment(robots_per_direction=3, random_seed=7)
    conflict_nodes = experiment.area.conflict_zone_nodes

    for robot in experiment.active.values():
        assert robot.goal_node in {"N_EXIT", "S_EXIT", "W_EXIT", "E_EXIT"}
        assert robot.goal_node != experiment.START_BY_DIRECTION[robot.direction].replace("START", "EXIT")
        assert any(node_id in conflict_nodes for node_id in robot.path)


def test_fcfs_cross_never_has_more_than_one_robot_in_shared_area():
    experiment = FCFSCrossExperiment(robots_per_direction=3)
    conflict_nodes = experiment.area.conflict_zone_nodes

    while not experiment.is_done and experiment.step_count < 500:
        experiment.step()
        inside = [
            robot.robot_id
            for robot in experiment.active.values()
            if robot.current_node in conflict_nodes
        ]
        assert len(inside) <= 1


def test_fcfs_cross_capacity_two_only_allows_disjoint_conflict_paths():
    experiment = FCFSCrossExperiment(
        robots_by_direction={
            "NORTH": 1,
            "SOUTH": 1,
            "WEST": 1,
            "EAST": 1,
        },
        goal_plan_by_direction={
            "NORTH": ["S_EXIT"],
            "SOUTH": ["N_EXIT"],
            "WEST": ["E_EXIT"],
            "EAST": ["W_EXIT"],
        },
        corridor_length=8,
        admission_window_steps=1,
        shared_area_capacity=2,
        random_seed=7,
    )
    conflict_nodes = experiment.area.conflict_zone_nodes

    while not experiment.is_done and experiment.step_count < 200:
        experiment.step()
        inside = [
            robot
            for robot in experiment.active.values()
            if robot.current_node in conflict_nodes
        ]
        assert len(inside) <= 2
        if len(inside) == 2:
            first, second = inside
            assert not (
                set(experiment._path_conflict_points(first.path[first.path_index:]))
                & set(experiment._path_conflict_points(second.path[second.path_index:]))
            )


def test_fcfs_queue_uses_waiting_point_arrival_order():
    experiment = FCFSCrossExperiment(robots_per_direction=1)
    seen_admissions = []

    while not experiment.is_done and experiment.step_count < 100:
        result = experiment.step()
        if result["admitted"]:
            seen_admissions.append(result["admitted"])

    assert seen_admissions[0] == "AGV_N1"
    assert len(seen_admissions) == 4


def test_fcfs_cross_dynamic_corridor_length_extends_layout_and_paths():
    default_layout = build_fcfs_cross_shared_area_layout()
    extended_layout = build_fcfs_cross_shared_area_layout(corridor_length=8)
    experiment = FCFSCrossExperiment(layout=extended_layout, robots_per_direction=1, corridor_length=8)

    assert default_layout.grid_size == (13, 13)
    assert extended_layout.grid_size == (19, 19)
    assert extended_layout.name == "fcfs_cross_shared_area_corridor8"
    assert len(experiment.active["AGV_N1"].path) > len(
        FCFSCrossExperiment(layout=default_layout, robots_per_direction=1).active["AGV_N1"].path
    )


def test_fcfs_cross_spawn_gap_staggers_agents_by_direction():
    experiment = FCFSCrossExperiment(
        robots_per_direction=3,
        corridor_length=8,
        spawn_gap_steps=2,
    )

    assert len(experiment.active) == 4
    assert sum(len(queue) for queue in experiment.pending_by_direction.values()) == 8

    spawned_by_step = {}
    while not experiment.is_done and experiment.step_count < 20:
        result = experiment.step()
        if result["spawned"]:
            spawned_by_step[result["step"]] = result["spawned"]

    assert set(spawned_by_step[2]) == {"AGV_N2", "AGV_S2", "AGV_W2", "AGV_E2"}
    assert set(spawned_by_step[4]) == {"AGV_N3", "AGV_S3", "AGV_W3", "AGV_E3"}


def test_fcfs_cross_heuristic_records_decision_features():
    experiment = FCFSCrossExperiment(
        robots_per_direction=3,
        corridor_length=8,
        spawn_gap_steps=2,
        policy_type="adaptive_fairness",
    )

    metrics = experiment.run(max_steps=500)

    assert metrics["policy_type"] == "adaptive_fairness"
    assert metrics["total_wait_time"] >= metrics["max_wait_time"]
    assert experiment.decision_log
    row = experiment.decision_log[0]
    assert {
        "decision_step",
        "candidate_agent_id",
        "waiting_steps",
        "exit_competition",
        "path_conflict_count",
        "approach_queue_length",
        "maneuver_priority",
        "score",
        "selected",
    }.issubset(row)


def test_heuristic_advantage_scenario_reduces_total_time_and_travel_time():
    fcfs = FCFSCrossExperiment(
        robots_per_direction=4,
        corridor_length=8,
        spawn_gap_steps=2,
        scenario_name="heuristic_advantage",
        policy_type="fcfs",
    ).run(max_steps=1000)
    heuristic = FCFSCrossExperiment(
        robots_per_direction=4,
        corridor_length=8,
        spawn_gap_steps=2,
        scenario_name="heuristic_advantage",
        policy_type="heuristic",
    ).run(max_steps=1000)

    assert fcfs["layout"] == "fcfs_cross_shared_area_corridor8_westtail4"
    assert fcfs["robots"] == 16
    assert heuristic["completed"] == fcfs["completed"] == 16
    assert heuristic["total_time"] < fcfs["total_time"]
    assert heuristic["total_travel_time"] < fcfs["total_travel_time"]
    assert heuristic["avg_travel_time"] < fcfs["avg_travel_time"]


def test_fcfs_cross_accepts_unbalanced_robot_counts_and_reports_utilization():
    experiment = FCFSCrossExperiment(
        robots_by_direction={
            "NORTH": 1,
            "SOUTH": 3,
            "WEST": 2,
            "EAST": 4,
        },
        corridor_length=8,
        spawn_gap_steps=1,
        admission_window_steps=1,
        random_seed=11,
    )

    metrics = experiment.run(max_steps=500)

    assert metrics["robots"] == 10
    assert metrics["robots_by_direction"] == {
        "NORTH": 1,
        "SOUTH": 3,
        "WEST": 2,
        "EAST": 4,
    }
    assert metrics["completed"] == 10
    assert 0.0 <= metrics["utilization"] <= 1.0
    assert metrics["shared_occupied_steps"] > 0
