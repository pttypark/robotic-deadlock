"""Export FCFS cross-simulation state for the browser 3D viewer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from rware.fcfs_cross_simulation import FCFSCrossExperiment, FCFSRobotState


ROBOT_COLORS = {
    "NORTH": "#2c7ec9",
    "SOUTH": "#3a9e5b",
    "WEST": "#8e5bbf",
    "EAST": "#e28034",
}


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    experiment = FCFSCrossExperiment(
        robots_per_direction=args.robots_per_direction,
        random_seed=args.seed,
        corridor_length=args.corridor_length,
        west_exit_extension=args.west_exit_extension,
        spawn_gap_steps=args.spawn_gap_steps,
        admission_window_steps=args.admission_window_steps,
        shared_area_capacity=args.shared_area_capacity,
        scenario_name=args.scenario,
        policy_type=args.policy,
    )

    frames = [_snapshot_frame(experiment, last_result=None)]
    last_result = None
    while not experiment.is_done and experiment.step_count < args.max_steps:
        last_result = experiment.step()
        frames.append(_snapshot_frame(experiment, last_result=last_result))

    payload = {
        "schema_version": 1,
        "metadata": {
            "title": "RWARE FCFS Cross 3D Trace",
            "policy_type": args.policy,
            "scenario_name": args.scenario,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "robots_per_direction": args.robots_per_direction,
            "corridor_length": args.corridor_length,
            "west_exit_extension": args.west_exit_extension,
            "spawn_gap_steps": args.spawn_gap_steps,
            "admission_window_steps": args.admission_window_steps,
            "shared_area_capacity": args.shared_area_capacity,
        },
        "layout": _snapshot_layout(experiment),
        "frames": frames,
        "metrics": experiment.metrics(),
    }

    with output.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")

    print(f"wrote {output.resolve()}")
    print(
        f"frames={len(frames)} robots={payload['metrics']['robots']} "
        f"completed={payload['metrics']['completed']} total_time={payload['metrics']['total_time']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a 3D viewer trace for the FCFS cross experiment.")
    parser.add_argument("--output", default=str(Path("viewer3d") / "data" / "fcfs_trace.json"))
    parser.add_argument("--policy", choices=("fcfs", "heuristic", "adaptive", "adaptive_fairness"), default="heuristic")
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--robots-per-direction", type=int, default=3)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--spawn-gap-steps", type=int, default=2)
    parser.add_argument("--admission-window-steps", type=int, default=2)
    parser.add_argument("--shared-area-capacity", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    return parser.parse_args()


def _snapshot_layout(experiment: FCFSCrossExperiment) -> dict[str, Any]:
    graph = experiment.graph
    start_nodes = set(experiment.START_BY_DIRECTION.values())
    exit_nodes = set(experiment.EXIT_BY_DIRECTION.values())
    area = experiment.area

    nodes = []
    for node_id, node in sorted(graph.nodes.items()):
        role = "road"
        if node_id in start_nodes:
            role = "start"
        elif node_id in exit_nodes:
            role = "exit"
        elif node_id in area.conflict_zone_nodes:
            role = "conflict"
        elif node_id in area.waiting_points:
            role = "waiting"
        nodes.append(
            {
                "id": node_id,
                "row": node.position[0],
                "col": node.position[1],
                "node_type": node.node_type,
                "role": role,
            }
        )

    edges = [
        {
            "from": edge.from_node,
            "to": edge.to_node,
            "direction": edge.direction,
            "edge_type": edge.edge_type,
        }
        for edge in sorted(graph.edges.values(), key=lambda item: item.edge_id)
    ]

    return {
        "name": experiment.layout.name,
        "grid_size": {
            "rows": experiment.layout.grid_size[0],
            "cols": experiment.layout.grid_size[1],
        },
        "nodes": nodes,
        "edges": edges,
        "start_by_direction": dict(experiment.START_BY_DIRECTION),
        "exit_by_direction": dict(experiment.EXIT_BY_DIRECTION),
        "wait_by_direction": dict(experiment.WAIT_BY_DIRECTION),
        "robot_colors": dict(ROBOT_COLORS),
        "interaction_area": {
            "area_id": area.area_id,
            "communication_zone_nodes": sorted(area.communication_zone_nodes),
            "conflict_zone_nodes": sorted(area.conflict_zone_nodes),
            "waiting_points": sorted(area.waiting_points),
            "shared_area_capacity": experiment.shared_area_capacity,
        },
    }


def _snapshot_frame(experiment: FCFSCrossExperiment, last_result: dict | None) -> dict[str, Any]:
    robots = []
    for robot in sorted(experiment.active.values(), key=lambda item: item.robot_id):
        robots.append(_snapshot_robot(experiment, robot, status_override=robot.status))
    for robot in sorted(experiment.completed, key=lambda item: item.robot_id):
        robots.append(_snapshot_robot(experiment, robot, status_override="completed"))

    return {
        "step": experiment.step_count,
        "robots": robots,
        "queue": list(experiment.fcfs_queue),
        "shared_robot_id": experiment.shared_robot_id,
        "shared_robot_ids": sorted(experiment.shared_robot_ids),
        "completed_count": len(experiment.completed),
        "pending_count": sum(len(queue) for queue in experiment.pending_by_direction.values()),
        "pending_by_direction": {
            direction: len(queue)
            for direction, queue in sorted(experiment.pending_by_direction.items())
        },
        "event_count": len(experiment.event_log),
        "recent_events": experiment.event_log[-8:],
        "last_result": {
            "spawned": last_result.get("spawned", []) if last_result else [],
            "admitted": last_result.get("admitted") if last_result else None,
            "moved": last_result.get("moved", []) if last_result else [],
        },
    }


def _snapshot_robot(
    experiment: FCFSCrossExperiment,
    robot: FCFSRobotState,
    status_override: str,
) -> dict[str, Any]:
    node_id = robot.current_node
    position = experiment.graph.nodes[node_id].position if node_id else (None, None)
    return {
        "id": robot.robot_id,
        "direction": robot.direction,
        "color": ROBOT_COLORS[robot.direction],
        "status": status_override,
        "current_node": node_id,
        "row": position[0],
        "col": position[1],
        "heading": robot.heading.name,
        "start_node": robot.start_node,
        "goal_node": robot.goal_node,
        "priority": robot.priority,
        "path_index": robot.path_index,
        "path": list(robot.path),
        "spawn_step": robot.spawn_step,
        "planned_spawn_step": robot.metadata.get("planned_spawn_step"),
        "waiting_time": robot.waiting_time,
        "shared_wait_time": robot.shared_wait_time,
        "yielded_count": robot.yielded_count,
        "is_shared": robot.robot_id in experiment.shared_robot_ids,
    }


if __name__ == "__main__":
    main()
