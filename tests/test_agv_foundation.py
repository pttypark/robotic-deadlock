"""Tests for the AGV interaction-area foundation layer."""

import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from rware.agv_layouts import AGVLayout, build_graph_rware_intersection_v1, build_warehouse_aisle_layout_v1
from rware.agv_movement import AGVMovementSimulator
from rware.agv_path_planning import astar_path
from rware.agv_topology import LaneGraph
from rware.agv_types import AGVAction, AGVState, Edge, Heading, InteractionArea, Node


def test_single_agv_straight_route_logs_wp_cp():
    layout = build_graph_rware_intersection_v1()
    sim = AGVMovementSimulator(
        layout,
        [
            AGVState(
                "AGV_1",
                "STATION_NORTH",
                None,
                None,
                Heading.SOUTH,
                priority=3,
                assigned_route_id="NORTH_TO_SOUTH",
            )
        ],
    )

    for _ in range(7):
        sim.step({"AGV_1": AGVAction.FORWARD})

    event_types = {event["event_type"] for event in sim.debug_snapshot()["event_log"]}
    assert "waiting_point_reached" in event_types
    assert "conflict_point_occupied" in event_types
    assert sim.agvs[0].occupied_conflict_point == "CP_2"


def test_turn_is_invalid_on_normal_lane():
    layout = build_graph_rware_intersection_v1()
    sim = AGVMovementSimulator(
        layout,
        [
            AGVState(
                "AGV_1",
                "N_01_08",
                None,
                None,
                Heading.SOUTH,
                priority=3,
                assigned_route_id="NORTH_TO_SOUTH",
            )
        ],
    )

    result = sim.step({"AGV_1": AGVAction.TURN_LEFT})

    assert result["proposals"]["AGV_1"]["valid"] is False
    assert result["proposals"]["AGV_1"]["reason"] == "action_not_allowed_at_node"
    assert sim.agvs[0].current_node == "N_01_08"


def test_intersection_turn_route_changes_heading():
    layout = build_graph_rware_intersection_v1()
    sim = AGVMovementSimulator(
        layout,
        [
            AGVState(
                "AGV_1",
                "CP_2",
                None,
                None,
                Heading.SOUTH,
                priority=3,
                assigned_route_id="NORTH_TO_EAST",
            )
        ],
    )

    result = sim.step({"AGV_1": AGVAction.TURN_LEFT})

    assert result["proposals"]["AGV_1"]["valid"] is True
    assert sim.agvs[0].current_node == "CP_3"
    assert sim.agvs[0].heading == Heading.EAST


def test_conflict_point_simultaneous_access_is_logged():
    layout = build_graph_rware_intersection_v1()
    sim = AGVMovementSimulator(
        layout,
        [
            AGVState(
                "AGV_1",
                "CP_2",
                None,
                None,
                Heading.SOUTH,
                priority=3,
                assigned_route_id="NORTH_TO_SOUTH",
            ),
            AGVState(
                "AGV_2",
                "CP_4",
                None,
                None,
                Heading.EAST,
                priority=5,
                assigned_route_id="WEST_TO_EAST",
            ),
        ],
    )

    result = sim.step({"AGV_1": AGVAction.FORWARD, "AGV_2": AGVAction.FORWARD})

    event_types = {event["event_type"] for event in result["conflict_events"]}
    assert "node_collision" in event_types
    assert "conflict_point_occupation" in event_types
    cp_event = next(
        event for event in result["conflict_events"] if event["event_type"] == "conflict_point_occupation"
    )
    assert cp_event["node"] == "CP_5"
    assert cp_event["priorities"] == {"AGV_1": 3, "AGV_2": 5}


def test_edge_swap_collision_is_logged():
    graph = LaneGraph()
    graph.add_node(Node("A", (0, 0), "normal_lane", ["forward", "wait"]))
    graph.add_node(Node("B", (0, 1), "normal_lane", ["forward", "wait"]))
    graph.add_edge(Edge("A->B", "A", "B", "EAST"))
    graph.add_edge(Edge("B->A", "B", "A", "WEST"))
    area = InteractionArea(
        area_id="IA_SWAP",
        area_type="bottleneck",
        communication_zone_nodes=set(),
        conflict_zone_nodes=set(),
        waiting_points=set(),
        conflict_points=set(),
        allowed_routes={},
    )
    layout = AGVLayout(
        name="swap_test",
        grid_size=(1, 2),
        graph=graph,
        interaction_areas=[area],
        rware_layout=".g",
        routes={},
        initial_nodes={},
        initial_headings={},
    )
    sim = AGVMovementSimulator(
        layout,
        [
            AGVState("AGV_1", "A", None, None, Heading.EAST, priority=2),
            AGVState("AGV_2", "B", None, None, Heading.WEST, priority=4),
        ],
    )

    result = sim.step({"AGV_1": AGVAction.FORWARD, "AGV_2": AGVAction.FORWARD})

    assert any(event["event_type"] == "edge_swap_collision" for event in result["conflict_events"])


def test_priority_is_included_in_conflict_logs():
    layout = build_graph_rware_intersection_v1()
    sim = AGVMovementSimulator(
        layout,
        [
            AGVState("AGV_1", "CP_2", None, None, Heading.SOUTH, priority=1, assigned_route_id="NORTH_TO_SOUTH"),
            AGVState("AGV_2", "CP_4", None, None, Heading.EAST, priority=5, assigned_route_id="WEST_TO_EAST"),
        ],
    )

    result = sim.step({"AGV_1": AGVAction.FORWARD, "AGV_2": AGVAction.FORWARD})

    assert result["conflict_events"][0]["priorities"]["AGV_1"] == 1
    assert result["conflict_events"][0]["priorities"]["AGV_2"] == 5


def test_warehouse_layout_has_stations_tasks_and_astar_paths():
    layout = build_warehouse_aisle_layout_v1()

    assert layout.grid_size == (19, 29)
    assert set(layout.stations) == {"STATION_1", "STATION_2", "STATION_3", "STATION_4"}
    assert {"TASK_A", "TASK_B", "TASK_C", "TASK_D"}.issubset(layout.task_nodes)

    path = astar_path(layout.graph, "STATION_1", "TASK_A")

    assert path[0] == "STATION_1"
    assert path[-1] == "TASK_A"
    assert len(path) > 2
