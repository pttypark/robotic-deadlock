"""Pytest-free smoke checks for the AGV interaction-area foundation layer."""

from __future__ import annotations

import os
import sys

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from rware.agv_layouts import build_graph_rware_intersection_v1, build_warehouse_aisle_layout_v1
from rware.agv_movement import AGVMovementSimulator
from rware.agv_path_planning import astar_path
from rware.agv_types import AGVAction, AGVState, Heading


def main() -> None:
    """Run core AGV foundation checks without requiring pytest."""

    layout = build_graph_rware_intersection_v1()

    sim = AGVMovementSimulator(
        layout,
        [AGVState("AGV_1", "STATION_NORTH", None, None, Heading.SOUTH, 3, "NORTH_TO_SOUTH")],
    )
    for _ in range(7):
        sim.step({"AGV_1": AGVAction.FORWARD})
    event_types = {event["event_type"] for event in sim.debug_snapshot()["event_log"]}
    assert "attention_point_reached" in event_types
    assert "waiting_point_reached" in event_types
    assert "conflict_point_occupied" in event_types

    sim = AGVMovementSimulator(
        layout,
        [AGVState("AGV_1", "N_01_08", None, None, Heading.SOUTH, 3, "NORTH_TO_SOUTH")],
    )
    result = sim.step({"AGV_1": AGVAction.TURN_LEFT})
    assert result["proposals"]["AGV_1"]["valid"] is False

    sim = AGVMovementSimulator(
        layout,
        [AGVState("AGV_1", "CP_2", None, None, Heading.SOUTH, 3, "NORTH_TO_EAST")],
    )
    sim.step({"AGV_1": AGVAction.TURN_LEFT})
    assert sim.agvs[0].current_node == "CP_3"
    assert sim.agvs[0].heading == Heading.EAST

    sim = AGVMovementSimulator(
        layout,
        [
            AGVState("AGV_1", "CP_2", None, None, Heading.SOUTH, 3, "NORTH_TO_SOUTH"),
            AGVState("AGV_2", "CP_4", None, None, Heading.EAST, 5, "WEST_TO_EAST"),
        ],
    )
    result = sim.step({"AGV_1": AGVAction.FORWARD, "AGV_2": AGVAction.FORWARD})
    event_types = {event["event_type"] for event in result["conflict_events"]}
    assert "conflict_point_occupation" in event_types
    assert result["conflict_events"][0]["priorities"]

    warehouse = build_warehouse_aisle_layout_v1()
    path = astar_path(warehouse.graph, "STATION_1", "TASK_A")
    assert path[0] == "STATION_1"
    assert path[-1] == "TASK_A"

    print("AGV foundation smoke checks passed.")


if __name__ == "__main__":
    main()
