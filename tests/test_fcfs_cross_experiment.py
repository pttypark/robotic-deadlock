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


def test_fcfs_queue_uses_waiting_point_arrival_order():
    experiment = FCFSCrossExperiment(robots_per_direction=1)
    seen_admissions = []

    while not experiment.is_done and experiment.step_count < 100:
        result = experiment.step()
        if result["admitted"]:
            seen_admissions.append(result["admitted"])

    assert seen_admissions[0] == "AGV_N1"
    assert len(seen_admissions) == 4
