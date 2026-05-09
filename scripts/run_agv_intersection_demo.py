"""Run the warehouse AGV A* task demo in the RWARE renderer."""

from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from rware.agv_layouts import build_warehouse_aisle_layout_v1
from rware.agv_path_planning import astar_path
from rware.agv_simulation_wrapper import AGVRWARESimulationWrapper
from rware.agv_topology import heading_between
from rware.agv_types import AGVAction, AGVMission, turn_left, turn_right


def scripted_actions(wrapper: AGVRWARESimulationWrapper) -> dict[str, AGVAction]:
    """Return deterministic route-following actions for the demo.

    Args:
        wrapper: Active RWARE/AGV wrapper.

    Returns:
        Action mapping keyed by robot id.
    """

    actions = {}
    sim = wrapper.simulator
    for agv in sim.agvs:
        route = sim.routes.get(agv.assigned_route_id)
        if route is None or agv.route_index >= len(route.node_sequence) - 1:
            actions[agv.robot_id] = AGVAction.WAIT
            continue
        next_node = route.node_sequence[agv.route_index + 1]
        next_heading = heading_between(
            sim.graph.nodes[agv.current_node].position,
            sim.graph.nodes[next_node].position,
        )
        if next_heading == agv.heading:
            actions[agv.robot_id] = AGVAction.FORWARD
        elif next_heading == turn_left(agv.heading):
            actions[agv.robot_id] = AGVAction.TURN_LEFT
        elif next_heading == turn_right(agv.heading):
            actions[agv.robot_id] = AGVAction.TURN_RIGHT
        else:
            actions[agv.robot_id] = AGVAction.TURN_RIGHT
    return actions


def assign_initial_missions(wrapper: AGVRWARESimulationWrapper) -> None:
    """Assign ideal warehouse missions and plan initial A* paths.

    Args:
        wrapper: Active RWARE/AGV wrapper.

    Returns:
        None.
    """

    assignments = {
        "AGV_1": ("TASK_A", "STATION_1"),
        "AGV_2": ("TASK_B", "STATION_2"),
        "AGV_3": ("TASK_E", "STATION_3"),
        "AGV_4": ("TASK_D", "STATION_4"),
    }
    for agv in wrapper.simulator.agvs:
        pickup, dropoff = assignments[agv.robot_id]
        agv.mission = AGVMission(
            mission_id=f"MISSION_{agv.robot_id}",
            pickup_node=pickup,
            dropoff_node=dropoff,
            metadata={"purpose": "station_to_shelf_access_then_return_to_G_station"},
        )
        _assign_astar_route(wrapper, agv.robot_id, pickup, "to_pickup")


def update_missions(wrapper: AGVRWARESimulationWrapper) -> None:
    """Switch AGVs from pickup travel to dropoff travel when targets are reached."""

    for agv in wrapper.simulator.agvs:
        mission = agv.mission
        if mission is None or mission.phase == "completed":
            continue
        if mission.phase == "to_pickup" and agv.current_node == mission.pickup_node:
            mission.phase = "to_dropoff"
            _assign_astar_route(wrapper, agv.robot_id, mission.dropoff_node, "to_dropoff")
        elif mission.phase == "to_dropoff" and agv.current_node == mission.dropoff_node:
            mission.phase = "completed"
            agv.assigned_route_id = None


def _assign_astar_route(
    wrapper: AGVRWARESimulationWrapper,
    robot_id: str,
    target_node: str,
    phase: str,
) -> None:
    sim = wrapper.simulator
    agv = next(item for item in sim.agvs if item.robot_id == robot_id)
    path = astar_path(sim.graph, agv.current_node, target_node)
    if not path:
        raise RuntimeError(f"No A* path for {robot_id}: {agv.current_node} -> {target_node}")
    route_id = f"ASTAR_{robot_id}_{phase}_{sim.step_count}"
    sim.set_route(robot_id, route_id, path, route_type="astar")


def main() -> None:
    """Parse CLI arguments and run the RWARE AGV demo."""

    parser = argparse.ArgumentParser(description="Run warehouse AGV A* task demo in RWARE.")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    layout = build_warehouse_aisle_layout_v1()
    wrapper = AGVRWARESimulationWrapper(layout, max_steps=args.steps)

    try:
        wrapper.reset(seed=args.seed)
        assign_initial_missions(wrapper)
        wrapper.simulator.print_layout_summary()
        if not args.no_render:
            wrapper.render()

        for _ in range(args.steps):
            update_missions(wrapper)
            result = wrapper.step(scripted_actions(wrapper))
            if args.debug or result["conflict_events"]:
                wrapper.simulator.print_step_debug(result)
            if not args.no_render:
                wrapper.render()
                time.sleep(args.delay)
    finally:
        wrapper.close()


if __name__ == "__main__":
    main()
