"""A* + FCFS baseline simulation for cross-shaped AGV shared area."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from rware.agv_layouts import AGVLayout, build_fcfs_cross_shared_area_layout
from rware.agv_path_planning import astar_path
from rware.agv_topology import heading_between
from rware.agv_types import Heading


@dataclass
class FCFSRobotState:
    """Runtime state for one AGV in the FCFS cross experiment."""

    robot_id: str
    direction: str
    route_id: str
    path: list[str]
    heading: Heading
    start_node: str
    goal_node: str
    priority: int = 1
    current_node: str | None = None
    path_index: int = 0
    status: str = "pending"
    spawn_step: int | None = None
    waiting_point_step: int | None = None
    shared_entry_step: int | None = None
    finish_step: int | None = None
    waiting_time: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def next_node(self) -> str | None:
        """Return the next planned node, if any."""

        if self.path_index >= len(self.path) - 1:
            return None
        return self.path[self.path_index + 1]


class FCFSCrossExperiment:
    """A* path + FCFS shared-area admission baseline for 12 AGVs."""

    START_BY_DIRECTION = {
        "NORTH": "N_START",
        "SOUTH": "S_START",
        "WEST": "W_START",
        "EAST": "E_START",
    }
    EXIT_BY_DIRECTION = {
        "NORTH": "S_EXIT",
        "SOUTH": "N_EXIT",
        "WEST": "E_EXIT",
        "EAST": "W_EXIT",
    }
    EXIT_DIRECTION_BY_NODE = {
        "N_EXIT": "NORTH",
        "S_EXIT": "SOUTH",
        "W_EXIT": "WEST",
        "E_EXIT": "EAST",
    }
    WAIT_BY_DIRECTION = {
        "NORTH": "N_WAIT",
        "SOUTH": "S_WAIT",
        "WEST": "W_WAIT",
        "EAST": "E_WAIT",
    }
    ENTRY_CONFLICT_BY_DIRECTION = {
        "NORTH": "CP_NW",
        "SOUTH": "CP_SE",
        "WEST": "CP_SW",
        "EAST": "CP_NE",
    }
    EXIT_CONFLICT_BY_GOAL = {
        "N_EXIT": "CP_NE",
        "S_EXIT": "CP_SW",
        "W_EXIT": "CP_NW",
        "E_EXIT": "CP_SE",
    }
    HEADING_BY_DIRECTION = {
        "NORTH": Heading.SOUTH,
        "SOUTH": Heading.NORTH,
        "WEST": Heading.EAST,
        "EAST": Heading.WEST,
    }

    def __init__(
        self,
        layout: AGVLayout | None = None,
        robots_per_direction: int = 3,
        random_seed: int | None = 7,
    ) -> None:
        """Create the FCFS experiment.

        Args:
            layout: Optional cross shared-area layout.
            robots_per_direction: Number of AGVs fixed at each start side.
            random_seed: Seed used to assign random end points reproducibly.

        Returns:
            None.
        """

        self.layout = layout or build_fcfs_cross_shared_area_layout()
        self.graph = self.layout.graph
        self.area = self.layout.interaction_areas[0]
        self.robots_per_direction = robots_per_direction
        self.random_seed = random_seed
        self.random = random.Random(random_seed)
        self.step_count = 0
        self.pending_by_direction: dict[str, deque[FCFSRobotState]] = {}
        self.active: dict[str, FCFSRobotState] = {}
        self.completed: list[FCFSRobotState] = []
        self.fcfs_queue: deque[str] = deque()
        self.shared_robot_id: str | None = None
        self.event_log: list[dict] = []
        self.step_log: list[dict] = []
        self._build_robots()

    def run(self, max_steps: int = 500) -> dict:
        """Run until all AGVs finish or max_steps is reached.

        Args:
            max_steps: Safety limit for simulation length.

        Returns:
            Metrics dictionary containing total_time and completion details.
        """

        while not self.is_done and self.step_count < max_steps:
            self.step()
        return self.metrics()

    def step(self) -> dict:
        """Advance the FCFS simulation by one synchronous step.

        Args:
            None.

        Returns:
            Step log dictionary.
        """

        self._enqueue_waiting_robots()
        admitted = self._admit_next_if_possible()
        proposals = self._build_proposals()
        moved = self._apply_moves(proposals)
        self._release_shared_area_if_empty()
        self._complete_finished()

        result = {
            "step": self.step_count,
            "spawned": [],
            "admitted": admitted,
            "shared_robot_id": self.shared_robot_id,
            "fcfs_queue": list(self.fcfs_queue),
            "proposals": proposals,
            "moved": moved,
            "active": {
                robot_id: robot.current_node for robot_id, robot in sorted(self.active.items())
            },
            "completed_count": len(self.completed),
        }
        self.step_log.append(result)
        self.step_count += 1
        return result

    @property
    def is_done(self) -> bool:
        """Return whether every AGV has completed."""

        return (
            len(self.completed) == 4 * self.robots_per_direction
            and not self.active
        )

    def metrics(self) -> dict:
        """Return experiment metrics.

        Args:
            None.

        Returns:
            Metrics with TOTAL time and per-robot finish data.
        """

        finish_steps = {
            robot.robot_id: robot.finish_step for robot in sorted(self.completed, key=lambda item: item.robot_id)
        }
        return {
            "policy": "A*_FCFS",
            "layout": self.layout.name,
            "robots": 4 * self.robots_per_direction,
            "robots_per_direction": self.robots_per_direction,
            "random_seed": self.random_seed,
            "total_time": self.step_count,
            "completed": len(self.completed),
            "finish_steps": finish_steps,
            "end_assignments": {
                robot.robot_id: robot.goal_node
                for robot in sorted(self.completed, key=lambda item: item.robot_id)
            },
            "shared_rule": "one_agv_in_shared_area",
            "decision_rule": "FCFS at waiting point immediately before shared area",
        }

    def print_layout_summary(self) -> None:
        """Print the experimental layout and route setup."""

        print(f"layout: {self.layout.name}")
        print(f"grid size: {self.layout.grid_size}")
        print(f"robots: {4 * self.robots_per_direction}")
        print(f"robots per direction: {self.robots_per_direction}")
        print(f"shared area: {self.area.area_id}")
        print(f"conflict zone: {sorted(self.area.conflict_zone_nodes)}")
        print(f"waiting points: {sorted(self.area.waiting_points)}")
        print(f"starts: {self.START_BY_DIRECTION}")
        print(f"exits: {self.EXIT_BY_DIRECTION}")
        print(f"random seed: {self.random_seed}")
        for robot in sorted(self.active.values(), key=lambda item: item.robot_id):
            print(
                f"{robot.robot_id}: start={robot.start_node} goal={robot.goal_node} "
                f"path_len={len(robot.path)} conflict_points={self._path_conflict_points(robot.path)}"
            )

    def _build_robots(self) -> None:
        for direction in ("NORTH", "SOUTH", "WEST", "EAST"):
            start = self.START_BY_DIRECTION[direction]
            goals = self._random_goals_for_direction(direction)
            for idx in range(self.robots_per_direction):
                robot_id = f"AGV_{direction[0]}{idx + 1}"
                goal = goals[idx]
                path = self._path_via_own_waiting_point(direction, start, goal)
                if not self._path_conflict_points(path):
                    raise ValueError(f"Route for {robot_id} does not pass conflict area: {start} -> {goal}")
                robot = FCFSRobotState(
                    robot_id=robot_id,
                    direction=direction,
                    route_id=f"{direction}_TO_{self.EXIT_DIRECTION_BY_NODE[goal]}",
                    path=path,
                    heading=self.HEADING_BY_DIRECTION[direction],
                    start_node=start,
                    goal_node=goal,
                    current_node=start,
                    path_index=0,
                    status="active",
                    spawn_step=0,
                )
                self.active[robot.robot_id] = robot
                self._event("initially_placed", robot, node=start)

    def _path_via_own_waiting_point(self, direction: str, start: str, goal: str) -> list[str]:
        """Plan an A* route through the side's W point and shared area.

        Args:
            direction: Entry side name.
            start: Fixed start staging node.
            goal: Randomly assigned end node.

        Returns:
            Node sequence from start to goal.
        """

        waiting_point = self.WAIT_BY_DIRECTION[direction]
        entry_conflict = self.ENTRY_CONFLICT_BY_DIRECTION[direction]
        exit_conflict = self.EXIT_CONFLICT_BY_GOAL[goal]
        to_wait = astar_path(self.graph, start, waiting_point)
        to_conflict = astar_path(self.graph, waiting_point, entry_conflict)
        across_conflict = astar_path(self.graph, entry_conflict, exit_conflict)
        to_goal = astar_path(self.graph, exit_conflict, goal)
        return to_wait + to_conflict[1:] + across_conflict[1:] + to_goal[1:]

    def _random_goals_for_direction(self, direction: str) -> list[str]:
        """Assign random end points that still force conflict-zone traversal.

        Args:
            direction: Entry side name.

        Returns:
            Random exit node ids for every AGV on the side.
        """

        own_exit = next(
            exit_node
            for exit_node, exit_direction in self.EXIT_DIRECTION_BY_NODE.items()
            if exit_direction == direction
        )
        candidates = sorted(set(self.EXIT_DIRECTION_BY_NODE) - {own_exit})
        return [self.random.choice(candidates) for _ in range(self.robots_per_direction)]

    def _path_conflict_points(self, path: list[str]) -> list[str]:
        """Return conflict-zone nodes included in a path."""

        return [node_id for node_id in path if node_id in self.area.conflict_zone_nodes]

    def _enqueue_waiting_robots(self) -> None:
        for robot in sorted(self.active.values(), key=lambda item: item.robot_id):
            if robot.robot_id in self.fcfs_queue:
                continue
            if robot.current_node in self.area.waiting_points and robot.next_node in self.area.conflict_zone_nodes:
                robot.waiting_point_step = self.step_count
                self.fcfs_queue.append(robot.robot_id)
                self._event("fcfs_waiting_point_arrival", robot, node=robot.current_node)

    def _admit_next_if_possible(self) -> str | None:
        if self.shared_robot_id is not None:
            return None
        while self.fcfs_queue:
            robot_id = self.fcfs_queue.popleft()
            robot = self.active.get(robot_id)
            if robot is None or robot.current_node not in self.area.waiting_points:
                continue
            self.shared_robot_id = robot_id
            robot.status = "admitted"
            self._event("fcfs_admitted", robot, node=robot.current_node)
            return robot_id
        return None

    def _build_proposals(self) -> dict[str, dict]:
        robots_by_node: dict[str, list[str]] = {}
        for robot_id, robot in sorted(self.active.items()):
            robots_by_node.setdefault(robot.current_node, []).append(robot_id)
        occupied = {
            node_id: robot_ids[0]
            for node_id, robot_ids in robots_by_node.items()
        }
        proposals = {}
        for robot_id, robot in sorted(self.active.items()):
            next_node = robot.next_node
            allowed = next_node is not None
            reason = "ok"
            node_queue = robots_by_node.get(robot.current_node, [])
            if len(node_queue) > 1 and node_queue[0] != robot_id:
                allowed = False
                reason = f"stacked_behind_{node_queue[0]}"
            elif next_node is None:
                allowed = False
                reason = "route_completed"
            elif next_node in self.area.conflict_zone_nodes and robot_id != self.shared_robot_id:
                allowed = False
                reason = "waiting_for_fcfs_admission"
            elif next_node in occupied:
                allowed = False
                reason = f"blocked_by_{occupied[next_node]}"
            proposals[robot_id] = {
                "from_node": robot.current_node,
                "to_node": next_node if allowed else robot.current_node,
                "wanted_node": next_node,
                "allowed": allowed,
                "reason": reason,
                "shared_robot_id": self.shared_robot_id,
            }
        return proposals

    def _apply_moves(self, proposals: dict[str, dict]) -> list[str]:
        moved = []
        for robot_id, proposal in proposals.items():
            robot = self.active[robot_id]
            if not proposal["allowed"]:
                robot.waiting_time += 1
                continue
            previous = robot.current_node
            robot.current_node = proposal["to_node"]
            robot.path_index += 1
            moved.append(robot_id)
            self._update_heading(robot, previous, robot.current_node)
            if robot.current_node in self.area.conflict_zone_nodes and robot.shared_entry_step is None:
                robot.shared_entry_step = self.step_count
                self._event("shared_area_entered", robot, node=robot.current_node)
            self._event("moved", robot, node=robot.current_node, metadata={"from_node": previous})
        return moved

    def _release_shared_area_if_empty(self) -> None:
        if self.shared_robot_id is None:
            return
        robot = self.active.get(self.shared_robot_id)
        if robot is None or robot.current_node not in self.area.conflict_zone_nodes:
            released = self.shared_robot_id
            self.shared_robot_id = None
            if robot is not None:
                robot.status = "active"
                self._event("shared_area_released", robot, node=robot.current_node)
            else:
                self.event_log.append(
                    {"step": self.step_count, "event_type": "shared_area_released", "robot_id": released}
                )

    def _complete_finished(self) -> None:
        for robot_id, robot in list(self.active.items()):
            if robot.current_node == robot.goal_node:
                robot.status = "completed"
                robot.finish_step = self.step_count + 1
                self.completed.append(robot)
                del self.active[robot_id]
                self._event("completed", robot, node=robot.current_node)

    def _update_heading(self, robot: FCFSRobotState, from_node: str, to_node: str) -> None:
        robot.heading = heading_between(
            self.graph.nodes[from_node].position,
            self.graph.nodes[to_node].position,
        )

    def _event(
        self,
        event_type: str,
        robot: FCFSRobotState,
        node: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.event_log.append(
            {
                "step": self.step_count,
                "event_type": event_type,
                "robot_id": robot.robot_id,
                "direction": robot.direction,
                "node": node,
                "route_id": robot.route_id,
                "start_node": robot.start_node,
                "goal_node": robot.goal_node,
                "metadata": metadata or {},
            }
        )
