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
    shared_wait_time: int = 0
    yielded_count: int = 0
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
    POLICY_LABELS = {
        "fcfs": "A*_FCFS",
        "heuristic": "A*_HEURISTIC",
        "adaptive": "A*_ADAPTIVE_PRIORITY",
        "adaptive_fairness": "A*_ADAPTIVE_FAIRNESS",
    }
    BASE_HEURISTIC_WEIGHTS = {
        "waiting": 0.8,
        "maneuver": 0.3,
        "exit_competition": 0.8,
        "path_conflict": 5.0,
        "approach_queue": 0.1,
        "remaining_path": 0.0,
        "same_direction_backlog": 0.0,
    }
    ADAPTIVE_WEIGHTS = {
        "priority": 1.2,
        "yield": 1.0,
        "waiting": 0.9,
        "approach_queue": 0.35,
        "downstream_priority": 0.2,
        "exit_competition": 0.8,
        "path_conflict": 3.0,
        "maneuver": 0.3,
        "fairness": 0.0,
        "remaining_path": 0.0,
        "same_direction_backlog": 0.0,
    }
    HEURISTIC_ADVANTAGE_GOALS = {
        "NORTH": ["S_EXIT", "W_EXIT", "W_EXIT", "E_EXIT", "W_EXIT"],
        "SOUTH": ["N_EXIT", "W_EXIT", "E_EXIT", "W_EXIT", "E_EXIT"],
        "WEST": ["E_EXIT", "N_EXIT", "E_EXIT", "S_EXIT", "E_EXIT"],
        "EAST": ["W_EXIT", "W_EXIT", "N_EXIT", "W_EXIT", "S_EXIT"],
    }
    DENSITY_STRESS_GOALS = {
        "NORTH": ["W_EXIT", "W_EXIT", "W_EXIT", "W_EXIT", "S_EXIT"],
        "SOUTH": ["W_EXIT", "W_EXIT", "W_EXIT", "W_EXIT", "N_EXIT"],
        "WEST": ["E_EXIT", "E_EXIT", "E_EXIT", "N_EXIT", "S_EXIT"],
        "EAST": ["W_EXIT", "W_EXIT", "W_EXIT", "W_EXIT", "N_EXIT"],
    }
    HEURISTIC_ADVANTAGE_SPAWN_OFFSETS = {
        "NORTH": 0,
        "WEST": 0,
        "SOUTH": 1,
        "EAST": 1,
    }
    HEURISTIC_ADVANTAGE_WEIGHTS = {
        "waiting": 1.0,
        "maneuver": 0.2,
        "exit_competition": 1.8,
        "path_conflict": 0.0,
        "approach_queue": 0.0,
        "remaining_path": 0.8,
        "same_direction_backlog": 0.8,
    }
    DENSITY_STRESS_WEIGHTS = {
        "waiting": 1.2,
        "maneuver": 0.1,
        "exit_competition": 2.0,
        "path_conflict": 0.0,
        "approach_queue": 0.5,
        "remaining_path": 1.5,
        "same_direction_backlog": 1.0,
    }

    def __init__(
        self,
        layout: AGVLayout | None = None,
        robots_per_direction: int = 3,
        robots_by_direction: dict[str, int] | None = None,
        random_seed: int | None = 7,
        corridor_length: int = 5,
        spawn_gap_steps: int = 0,
        policy_type: str = "fcfs",
        scenario_name: str = "default",
        west_exit_extension: int = 0,
        admission_window_steps: int = 0,
        shared_area_capacity: int = 1,
        normalize_heuristic_features: bool = False,
        spawn_offsets_by_direction: dict[str, int] | None = None,
        spawn_plan_by_direction: dict[str, list[int]] | None = None,
        goal_plan_by_direction: dict[str, list[str]] | None = None,
        heuristic_weights: dict[str, float] | None = None,
        yield_base: float = 1.0,
        yield_count_power: float = 1.0,
        yield_priority_power: float = 1.0,
        fairness_threshold: int = 12,
        fairness_bonus: float = 3.0,
    ) -> None:
        """Create the FCFS experiment.

        Args:
            layout: Optional cross shared-area layout.
            robots_per_direction: Number of AGVs fixed at each start side.
            robots_by_direction: Optional per-direction AGV counts. When set,
                this overrides robots_per_direction and can create unbalanced
                NORTH/SOUTH/WEST/EAST demand.
            random_seed: Seed used to assign random end points reproducibly.
            corridor_length: Length of each approach corridor when layout is not provided.
            spawn_gap_steps: Steps between AGV releases from the same side.
            policy_type: Admission policy: fcfs, heuristic, adaptive, or adaptive_fairness.
            scenario_name: Optional built-in scenario. Use "heuristic_advantage"
                to compare FCFS with a congestion-aware heuristic.
            west_exit_extension: Extra outbound tail cells after W_EXIT.
            admission_window_steps: Steps to collect waiting candidates before
                admitting the next AGV when the shared area is free.
            shared_area_capacity: Maximum AGVs allowed in the shared area.
                Values above 1 still require disjoint conflict-zone paths.
            normalize_heuristic_features: Normalize traffic factors to 0-1
                within the current candidate set before heuristic scoring.
            spawn_offsets_by_direction: Per-direction release offsets.
            spawn_plan_by_direction: Optional per-AGV planned release steps for
                each direction. When set, these exact steps override the
                regular offset + gap schedule.
            goal_plan_by_direction: Fixed exit sequence per direction.
            heuristic_weights: Optional score weights for heuristic policies.
            yield_base: Base score for vehicles that have yielded before.
            yield_count_power: Exponent applied to yielded_count.
            yield_priority_power: Exponent applied to task priority.
            fairness_threshold: Waiting time before fairness protection starts.
            fairness_bonus: Score added per step beyond fairness_threshold.

        Returns:
            None.
        """

        if robots_per_direction < 1:
            raise ValueError("robots_per_direction must be at least 1")
        robots_by_direction = self._normalize_robot_counts(
            robots_by_direction,
            robots_per_direction,
        )
        if spawn_gap_steps < 0:
            raise ValueError("spawn_gap_steps must be non-negative")
        if policy_type not in self.POLICY_LABELS:
            raise ValueError(f"Unknown policy_type: {policy_type}")
        if admission_window_steps < 0:
            raise ValueError("admission_window_steps must be non-negative")
        if shared_area_capacity < 1:
            raise ValueError("shared_area_capacity must be at least 1")
        if west_exit_extension < 0:
            raise ValueError("west_exit_extension must be non-negative")

        scenario = self._scenario_defaults(scenario_name)
        if scenario and west_exit_extension == 0:
            west_exit_extension = scenario["west_exit_extension"]
        if scenario and admission_window_steps == 0:
            admission_window_steps = scenario["admission_window_steps"]
        if scenario and spawn_offsets_by_direction is None:
            spawn_offsets_by_direction = dict(scenario["spawn_offsets_by_direction"])
        if scenario and goal_plan_by_direction is None:
            goal_plan_by_direction = {
                direction: list(goals)
                for direction, goals in scenario["goal_plan_by_direction"].items()
            }
        if scenario and heuristic_weights is None:
            heuristic_weights = dict(scenario["heuristic_weights"])

        self.layout = layout or build_fcfs_cross_shared_area_layout(
            corridor_length=corridor_length,
            west_exit_extension=west_exit_extension,
        )
        if layout is not None:
            corridor_length = (layout.grid_size[0] - 3) // 2
            west_exit_extension = max(0, layout.grid_size[1] - layout.grid_size[0])
        self.graph = self.layout.graph
        self.area = self.layout.interaction_areas[0]
        self.robots_per_direction = robots_per_direction
        self.robots_by_direction = robots_by_direction
        self.total_robots = sum(robots_by_direction.values())
        self.random_seed = random_seed
        self.random = random.Random(random_seed)
        self.corridor_length = corridor_length
        self.west_exit_extension = west_exit_extension
        self.spawn_gap_steps = spawn_gap_steps
        self.policy_type = policy_type
        self.scenario_name = scenario_name
        self.admission_window_steps = admission_window_steps
        self.shared_area_capacity = shared_area_capacity
        self.normalize_heuristic_features = normalize_heuristic_features
        self.spawn_offsets_by_direction = self._normalize_direction_ints(
            spawn_offsets_by_direction,
            default=0,
            field_name="spawn_offsets_by_direction",
        )
        self.spawn_plan_by_direction = self._normalize_spawn_plan(spawn_plan_by_direction)
        self.goal_plan_by_direction = self._normalize_goal_plan(goal_plan_by_direction)
        self.heuristic_weights = heuristic_weights or {}
        self.yield_base = yield_base
        self.yield_count_power = yield_count_power
        self.yield_priority_power = yield_priority_power
        self.fairness_threshold = fairness_threshold
        self.fairness_bonus = fairness_bonus
        self.step_count = 0
        self.pending_by_direction: dict[str, deque[FCFSRobotState]] = {}
        self.active: dict[str, FCFSRobotState] = {}
        self.completed: list[FCFSRobotState] = []
        self.fcfs_queue: deque[str] = deque()
        self.shared_robot_id: str | None = None
        self.shared_robot_ids: set[str] = set()
        self.selection_count = 0
        self.shared_occupied_steps = 0
        self.event_log: list[dict] = []
        self.step_log: list[dict] = []
        self.decision_log: list[dict] = []
        self._build_robots()

    def _scenario_defaults(self, scenario_name: str) -> dict | None:
        if scenario_name in {"", "default"}:
            return None
        if scenario_name == "heuristic_advantage":
            return {
                "west_exit_extension": 4,
                "admission_window_steps": 2,
                "spawn_offsets_by_direction": self.HEURISTIC_ADVANTAGE_SPAWN_OFFSETS,
                "goal_plan_by_direction": self.HEURISTIC_ADVANTAGE_GOALS,
                "heuristic_weights": self.HEURISTIC_ADVANTAGE_WEIGHTS,
            }
        if scenario_name == "density_stress":
            return {
                "west_exit_extension": 24,
                "admission_window_steps": 1,
                "spawn_offsets_by_direction": self.HEURISTIC_ADVANTAGE_SPAWN_OFFSETS,
                "goal_plan_by_direction": self.DENSITY_STRESS_GOALS,
                "heuristic_weights": self.DENSITY_STRESS_WEIGHTS,
            }
        else:
            raise ValueError(f"Unknown scenario_name: {scenario_name}")

    def _normalize_robot_counts(
        self,
        robots_by_direction: dict[str, int] | None,
        robots_per_direction: int,
    ) -> dict[str, int]:
        if robots_by_direction is None:
            return {
                direction: robots_per_direction
                for direction in self.START_BY_DIRECTION
            }
        unknown = set(robots_by_direction) - set(self.START_BY_DIRECTION)
        if unknown:
            raise ValueError(f"Unknown direction(s) in robots_by_direction: {sorted(unknown)}")
        normalized = {
            direction: int(robots_by_direction.get(direction, 0))
            for direction in self.START_BY_DIRECTION
        }
        if any(count < 0 for count in normalized.values()):
            raise ValueError("robots_by_direction values must be non-negative")
        if sum(normalized.values()) < 1:
            raise ValueError("robots_by_direction must contain at least one AGV")
        return normalized

    def _normalize_direction_ints(
        self,
        values: dict[str, int] | None,
        default: int,
        field_name: str,
    ) -> dict[str, int]:
        normalized = {direction: default for direction in self.START_BY_DIRECTION}
        if values is None:
            return normalized
        unknown = set(values) - set(self.START_BY_DIRECTION)
        if unknown:
            raise ValueError(f"Unknown direction(s) in {field_name}: {sorted(unknown)}")
        for direction, value in values.items():
            if value < 0:
                raise ValueError(f"{field_name} values must be non-negative")
            normalized[direction] = value
        return normalized

    def _normalize_spawn_plan(
        self,
        spawn_plan_by_direction: dict[str, list[int]] | None,
    ) -> dict[str, list[int]]:
        if spawn_plan_by_direction is None:
            return {}
        unknown = set(spawn_plan_by_direction) - set(self.START_BY_DIRECTION)
        if unknown:
            raise ValueError(f"Unknown direction(s) in spawn_plan_by_direction: {sorted(unknown)}")
        normalized = {}
        for direction, plan in spawn_plan_by_direction.items():
            expected = self.robots_by_direction[direction]
            if len(plan) != expected:
                raise ValueError(
                    f"spawn_plan_by_direction[{direction!r}] must contain "
                    f"{expected} step(s), got {len(plan)}"
                )
            steps = [int(step) for step in plan]
            if any(step < 0 for step in steps):
                raise ValueError("spawn_plan_by_direction values must be non-negative")
            if steps != sorted(steps):
                raise ValueError(
                    f"spawn_plan_by_direction[{direction!r}] must be sorted in nondecreasing order"
                )
            normalized[direction] = steps
        return normalized

    def _normalize_goal_plan(
        self,
        goal_plan_by_direction: dict[str, list[str]] | None,
    ) -> dict[str, list[str]]:
        if goal_plan_by_direction is None:
            return {}
        unknown = set(goal_plan_by_direction) - set(self.START_BY_DIRECTION)
        if unknown:
            raise ValueError(f"Unknown direction(s) in goal_plan_by_direction: {sorted(unknown)}")
        valid_goals = set(self.EXIT_DIRECTION_BY_NODE)
        normalized = {}
        for direction, goals in goal_plan_by_direction.items():
            if not goals:
                raise ValueError(f"goal_plan_by_direction[{direction!r}] must not be empty")
            own_exit = next(
                exit_node
                for exit_node, exit_direction in self.EXIT_DIRECTION_BY_NODE.items()
                if exit_direction == direction
            )
            invalid = [goal for goal in goals if goal not in valid_goals or goal == own_exit]
            if invalid:
                raise ValueError(
                    f"Invalid goal(s) for {direction}: {invalid}. Goals must be "
                    "valid exits and cannot be the same side's own exit."
                )
            normalized[direction] = list(goals)
        return normalized

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

        spawned = self._spawn_due_robots()
        self._enqueue_waiting_robots()
        admitted = self._admit_next_if_possible()
        proposals = self._build_proposals()
        moved = self._apply_moves(proposals)
        self._record_shared_occupancy()
        self._release_shared_area_if_empty()
        self._complete_finished()

        result = {
            "step": self.step_count,
            "spawned": spawned,
            "admitted": admitted,
            "shared_robot_id": self.shared_robot_id,
            "shared_robot_ids": sorted(self.shared_robot_ids),
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
            len(self.completed) == self.total_robots
            and all(not queue for queue in self.pending_by_direction.values())
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
        travel_times = [
            (robot.finish_step or self.step_count) - (robot.spawn_step or 0)
            for robot in self.completed
        ]
        return {
            "policy": self.POLICY_LABELS[self.policy_type],
            "policy_type": self.policy_type,
            "scenario_name": self.scenario_name,
            "layout": self.layout.name,
            "robots": self.total_robots,
            "robots_per_direction": self.robots_per_direction,
            "robots_by_direction": dict(self.robots_by_direction),
            "random_seed": self.random_seed,
            "corridor_length": self.corridor_length,
            "west_exit_extension": self.west_exit_extension,
            "spawn_gap_steps": self.spawn_gap_steps,
            "spawn_offsets_by_direction": dict(self.spawn_offsets_by_direction),
            "spawn_plan_by_direction": {
                direction: list(plan)
                for direction, plan in self.spawn_plan_by_direction.items()
            },
            "admission_window_steps": self.admission_window_steps,
            "shared_area_capacity": self.shared_area_capacity,
            "total_time": self.step_count,
            "completed": len(self.completed),
            "total_travel_time": sum(travel_times),
            "avg_travel_time": (
                sum(travel_times) / len(travel_times)
                if travel_times
                else 0.0
            ),
            "total_wait_time": sum(robot.shared_wait_time for robot in self.completed),
            "avg_wait_time": (
                sum(robot.shared_wait_time for robot in self.completed) / len(self.completed)
                if self.completed
                else 0.0
            ),
            "max_wait_time": max((robot.shared_wait_time for robot in self.completed), default=0),
            "shared_occupied_steps": self.shared_occupied_steps,
            "utilization": (
                self.shared_occupied_steps / self.step_count
                if self.step_count
                else 0.0
            ),
            "finish_steps": finish_steps,
            "end_assignments": {
                robot.robot_id: robot.goal_node
                for robot in sorted(self.completed, key=lambda item: item.robot_id)
            },
            "shared_rule": "one_agv_in_shared_area",
            "decision_rule": self._decision_rule_label(),
        }

    def print_layout_summary(self) -> None:
        """Print the experimental layout and route setup."""

        print(f"layout: {self.layout.name}")
        print(f"grid size: {self.layout.grid_size}")
        print(f"robots: {self.total_robots}")
        print(f"robots per direction: {self.robots_per_direction}")
        print(f"robots by direction: {self.robots_by_direction}")
        print(f"corridor length: {self.corridor_length}")
        print(f"west exit extension: {self.west_exit_extension}")
        print(f"spawn gap steps: {self.spawn_gap_steps}")
        print(f"spawn offsets: {self.spawn_offsets_by_direction}")
        if self.spawn_plan_by_direction:
            print(f"spawn plan: {self.spawn_plan_by_direction}")
        print(f"admission window steps: {self.admission_window_steps}")
        print(f"shared area capacity: {self.shared_area_capacity}")
        print(f"scenario: {self.scenario_name}")
        print(f"policy: {self.policy_type}")
        print(f"shared area: {self.area.area_id}")
        print(f"conflict zone: {sorted(self.area.conflict_zone_nodes)}")
        print(f"waiting points: {sorted(self.area.waiting_points)}")
        print(f"starts: {self.START_BY_DIRECTION}")
        print(f"exits: {self.EXIT_BY_DIRECTION}")
        print(f"random seed: {self.random_seed}")
        robots = list(self.active.values())
        for queue in self.pending_by_direction.values():
            robots.extend(queue)
        for robot in sorted(robots, key=lambda item: item.robot_id):
            print(
                f"{robot.robot_id}: start={robot.start_node} goal={robot.goal_node} "
                f"path_len={len(robot.path)} conflict_points={self._path_conflict_points(robot.path)}"
            )

    def _build_robots(self) -> None:
        for direction in ("NORTH", "SOUTH", "WEST", "EAST"):
            start = self.START_BY_DIRECTION[direction]
            goals = self._goals_for_direction(direction)
            spawn_offset = self.spawn_offsets_by_direction[direction]
            spawn_plan = self.spawn_plan_by_direction.get(direction)
            self.pending_by_direction[direction] = deque()
            for idx in range(self.robots_by_direction[direction]):
                robot_id = f"AGV_{direction[0]}{idx + 1}"
                goal = goals[idx]
                path = self._path_via_own_waiting_point(direction, start, goal)
                if spawn_plan is None:
                    planned_spawn_step = spawn_offset + idx * self.spawn_gap_steps
                else:
                    planned_spawn_step = spawn_plan[idx]
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
                    current_node=None,
                    path_index=0,
                    status="pending",
                    spawn_step=None,
                    priority=self.random.randint(1, 3),
                    metadata={"planned_spawn_step": planned_spawn_step},
                )
                if self.spawn_gap_steps == 0 and planned_spawn_step == 0:
                    self._place_robot(robot, event_type="initially_placed")
                elif planned_spawn_step == 0:
                    self._place_robot(robot, event_type="spawned")
                else:
                    self.pending_by_direction[direction].append(robot)
                    self._event(
                        "queued_for_spawn",
                        robot,
                        node=start,
                        metadata={"planned_spawn_step": robot.metadata["planned_spawn_step"]},
                    )

    def _place_robot(self, robot: FCFSRobotState, event_type: str) -> None:
        robot.current_node = robot.start_node
        robot.path_index = 0
        robot.status = "active"
        robot.spawn_step = self.step_count
        self.active[robot.robot_id] = robot
        self._event(event_type, robot, node=robot.start_node)

    def _spawn_due_robots(self) -> list[str]:
        spawned = []
        occupied = {
            robot.current_node
            for robot in self.active.values()
            if robot.current_node is not None
        }
        for direction, queue in self.pending_by_direction.items():
            if not queue:
                continue
            robot = queue[0]
            planned_spawn_step = robot.metadata["planned_spawn_step"]
            if self.step_count < planned_spawn_step:
                continue
            if robot.start_node in occupied:
                continue
            queue.popleft()
            self._place_robot(robot, event_type="spawned")
            occupied.add(robot.start_node)
            spawned.append(robot.robot_id)
        return spawned

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

    def _goals_for_direction(self, direction: str) -> list[str]:
        if direction in self.goal_plan_by_direction:
            plan = self.goal_plan_by_direction[direction]
            return [plan[idx % len(plan)] for idx in range(self.robots_by_direction[direction])]
        return self._random_goals_for_direction(direction)

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
        return [self.random.choice(candidates) for _ in range(self.robots_by_direction[direction])]

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

    def _admit_next_if_possible(self) -> str | list[str] | None:
        if len(self.shared_robot_ids) >= self.shared_area_capacity:
            return None
        self._drop_stale_queue_entries()
        if not self._admission_window_ready():
            return None
        admitted = []
        while len(self.shared_robot_ids) < self.shared_area_capacity:
            previous_count = len(self.shared_robot_ids)
            if self.policy_type == "fcfs":
                robot_id = self._admit_next_fcfs()
            else:
                robot_id = self._admit_next_by_score()
            if robot_id is None:
                break
            admitted.append(robot_id)
            if len(self.shared_robot_ids) == previous_count:
                break
        if not admitted:
            return None
        if len(admitted) == 1:
            return admitted[0]
        return admitted

    def _admit_next_fcfs(self) -> str | None:
        while self.fcfs_queue:
            robot_id = self.fcfs_queue[0]
            robot = self.active.get(robot_id)
            if robot is None or robot.current_node not in self.area.waiting_points:
                self.fcfs_queue.popleft()
                continue
            if not self._compatible_with_shared_area(robot):
                return None
            candidates = self._current_waiting_candidates()
            self._record_decision(candidates, robot.robot_id)
            self.fcfs_queue.popleft()
            self._add_shared_robot(robot_id)
            robot.status = "admitted"
            robot.shared_wait_time = self._shared_wait_steps(robot)
            self.selection_count += 1
            self._event("fcfs_admitted", robot, node=robot.current_node)
            return robot_id
        return None

    def _admit_next_by_score(self) -> str | None:
        candidates = self._current_waiting_candidates()
        if not candidates:
            return None
        compatible_candidates = [
            robot
            for robot in candidates
            if self._compatible_with_shared_area(robot)
        ]
        if not compatible_candidates:
            return None
        scored = [
            (self._admission_score(robot, candidates), robot.waiting_point_step or 0, robot.robot_id, robot)
            for robot in compatible_candidates
        ]
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected = scored[0][3]
        self._record_decision(candidates, selected.robot_id)
        self.fcfs_queue = deque(robot_id for robot_id in self.fcfs_queue if robot_id != selected.robot_id)
        for robot in candidates:
            if robot.robot_id != selected.robot_id:
                robot.yielded_count += 1
        self._add_shared_robot(selected.robot_id)
        selected.status = "admitted"
        selected.shared_wait_time = self._shared_wait_steps(selected)
        self.selection_count += 1
        self._event(
            "heuristic_admitted",
            selected,
            node=selected.current_node,
            metadata={"score": scored[0][0], "policy_type": self.policy_type},
        )
        return selected.robot_id

    def _admission_window_ready(self) -> bool:
        if self.admission_window_steps == 0 or not self.fcfs_queue:
            return True
        candidates = self._current_waiting_candidates()
        if not candidates:
            return False
        first_arrival = min(
            robot.waiting_point_step
            for robot in candidates
            if robot.waiting_point_step is not None
        )
        return self.step_count - first_arrival >= self.admission_window_steps

    def _current_waiting_candidates(self) -> list[FCFSRobotState]:
        return [
            self.active[robot_id]
            for robot_id in self.fcfs_queue
            if robot_id in self.active
            and self.active[robot_id].current_node in self.area.waiting_points
        ]

    def _shared_wait_steps(self, robot: FCFSRobotState) -> int:
        if robot.waiting_point_step is None:
            return 0
        return self.step_count - robot.waiting_point_step

    def _drop_stale_queue_entries(self) -> None:
        self.fcfs_queue = deque(
            robot_id
            for robot_id in self.fcfs_queue
            if robot_id in self.active and self.active[robot_id].current_node in self.area.waiting_points
        )

    def _admission_score(self, robot: FCFSRobotState, candidates: list[FCFSRobotState]) -> float:
        if self.normalize_heuristic_features:
            features = self._normalized_decision_features(robot, candidates)
        else:
            features = self._decision_features(robot, candidates)
        if self.policy_type == "heuristic":
            weights = {**self.BASE_HEURISTIC_WEIGHTS, **self.heuristic_weights}
            return (
                weights["waiting"] * features["waiting_steps"]
                + weights["maneuver"] * features["maneuver_priority"]
                + weights["exit_competition"] * features["exit_competition"]
                + weights["path_conflict"] * features["path_conflict_count"]
                + weights["approach_queue"] * features["approach_queue_length"]
                + weights["remaining_path"] * features["remaining_path_length"]
                + weights["same_direction_backlog"] * features["same_direction_backlog"]
            )
        weights = {**self.ADAPTIVE_WEIGHTS, **self.heuristic_weights}
        fairness = 0.0
        if self.policy_type == "adaptive_fairness":
            fairness = self.fairness_bonus * max(0, features["waiting_steps"] - self.fairness_threshold)
        return (
            weights["priority"] * robot.priority
            + weights["yield"] * self._yield_score(robot)
            + weights["waiting"] * features["waiting_steps"]
            + weights["approach_queue"] * features["approach_queue_length"]
            + weights["downstream_priority"] * features["downstream_priority"]
            + weights["exit_competition"] * features["exit_competition"]
            + weights["path_conflict"] * features["path_conflict_count"]
            + weights["maneuver"] * features["maneuver_priority"]
            + weights["remaining_path"] * features["remaining_path_length"]
            + weights["same_direction_backlog"] * features["same_direction_backlog"]
            + weights["fairness"] * fairness
            + fairness
        )

    def _yield_score(self, robot: FCFSRobotState) -> float:
        if robot.yielded_count == 0:
            return 0.0
        return (
            self.yield_base
            * (robot.yielded_count ** self.yield_count_power)
            * (robot.priority ** self.yield_priority_power)
        )

    def _numeric_robot_id(self, robot: FCFSRobotState) -> int:
        direction_order = ["NORTH", "SOUTH", "WEST", "EAST"]
        sequence = int(robot.robot_id.split("_")[1][1:])
        offset = sum(
            self.robots_by_direction[direction]
            for direction in direction_order[: direction_order.index(robot.direction)]
        )
        return offset + sequence

    def _record_shared_occupancy(self) -> None:
        if any(
            robot.current_node in self.area.conflict_zone_nodes
            for robot in self.active.values()
        ):
            self.shared_occupied_steps += 1

    def _decision_features(self, robot: FCFSRobotState, candidates: list[FCFSRobotState]) -> dict:
        conflict_path = set(self._path_conflict_points(robot.path[robot.path_index:]))
        candidate_paths = {
            other.robot_id: set(self._path_conflict_points(other.path[other.path_index:]))
            for other in candidates
        }
        waiting_steps = self._shared_wait_steps(robot)
        queue = list(self.fcfs_queue)
        queue_position = queue.index(robot.robot_id) if robot.robot_id in queue else -1
        same_direction = [
            other
            for other in candidates
            if other.robot_id != robot.robot_id and other.direction == robot.direction
        ]
        same_direction_backlog = sum(
            1
            for other in self.active.values()
            if other.robot_id != robot.robot_id and other.direction == robot.direction
        ) + len(self.pending_by_direction.get(robot.direction, ()))
        return {
            "decision_step": self.step_count,
            "candidate_agent_id": self._numeric_robot_id(robot),
            "candidate_agent_label": robot.robot_id,
            "arrival_time": robot.waiting_point_step,
            "waiting_steps": waiting_steps,
            "queue_position": queue_position,
            "queue_size": len(candidates),
            "exit_competition": sum(
                1 for other in candidates if other.robot_id != robot.robot_id and other.goal_node == robot.goal_node
            ),
            "path_conflict_count": sum(
                1
                for other in candidates
                if other.robot_id != robot.robot_id and conflict_path & candidate_paths[other.robot_id]
            ),
            "approach_queue_length": len(same_direction),
            "maneuver_priority": self._maneuver_priority(robot),
            "route_length": len(robot.path),
            "remaining_path_length": len(robot.path) - robot.path_index - 1,
            "shared_zone_entry_step": robot.path_index,
            "global_active_agents": len(self.active),
            "shared_zone_occupied": int(self.shared_robot_id is not None),
            "shared_zone_occupancy": len(self.shared_robot_ids),
            "shared_zone_capacity": self.shared_area_capacity,
            "selection_order_length": self.selection_count,
            "task_priority": robot.priority,
            "yielded_count": robot.yielded_count,
            "downstream_priority": sum(other.priority for other in same_direction),
            "same_direction_backlog": same_direction_backlog,
            "approach_north": int(robot.direction == "NORTH"),
            "approach_south": int(robot.direction == "SOUTH"),
            "approach_west": int(robot.direction == "WEST"),
            "approach_east": int(robot.direction == "EAST"),
            "maneuver_left": int(self._maneuver_type(robot) == "left"),
            "maneuver_straight": int(self._maneuver_type(robot) == "straight"),
            "maneuver_right": int(self._maneuver_type(robot) == "right"),
            "exit_north": int(self.EXIT_DIRECTION_BY_NODE[robot.goal_node] == "NORTH"),
            "exit_south": int(self.EXIT_DIRECTION_BY_NODE[robot.goal_node] == "SOUTH"),
            "exit_west": int(self.EXIT_DIRECTION_BY_NODE[robot.goal_node] == "WEST"),
            "exit_east": int(self.EXIT_DIRECTION_BY_NODE[robot.goal_node] == "EAST"),
        }

    def _record_decision(self, candidates: list[FCFSRobotState], selected_robot_id: str) -> None:
        for robot in candidates:
            features = self._decision_features(robot, candidates)
            features["score"] = self._admission_score(robot, candidates) if self.policy_type != "fcfs" else 0.0
            features["selected"] = int(robot.robot_id == selected_robot_id)
            features["policy_type"] = self.policy_type
            features["seed"] = self.random_seed
            self.decision_log.append(features)

    def _normalized_decision_features(
        self,
        robot: FCFSRobotState,
        candidates: list[FCFSRobotState],
    ) -> dict:
        raw_features = {
            candidate.robot_id: self._decision_features(candidate, candidates)
            for candidate in candidates
        }
        features = dict(raw_features[robot.robot_id])
        for key in (
            "waiting_steps",
            "exit_competition",
            "path_conflict_count",
            "approach_queue_length",
        ):
            denominator = max(
                1.0,
                max(float(item[key]) for item in raw_features.values()),
            )
            features[key] = float(features[key]) / denominator
        return features

    def _maneuver_type(self, robot: FCFSRobotState) -> str:
        entry = robot.direction
        exit_direction = self.EXIT_DIRECTION_BY_NODE[robot.goal_node]
        if self.EXIT_BY_DIRECTION[entry] == robot.goal_node:
            return "straight"
        order = ["NORTH", "EAST", "SOUTH", "WEST"]
        delta = (order.index(exit_direction) - order.index(entry)) % 4
        if delta == 1:
            return "left"
        if delta == 3:
            return "right"
        return "straight"

    def _maneuver_priority(self, robot: FCFSRobotState) -> float:
        return {"left": 0.0, "straight": 0.5, "right": 1.0}[self._maneuver_type(robot)]

    def _decision_rule_label(self) -> str:
        if self.policy_type == "fcfs":
            return "FCFS at waiting point immediately before shared area"
        if self.policy_type == "heuristic":
            return "Weighted waiting/maneuver/exit/path/approach heuristic"
        if self.policy_type == "adaptive":
            return "Adaptive priority with task priority and weighted yield"
        return "Adaptive priority with weighted yield and fairness bonus"

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
            elif next_node in self.area.conflict_zone_nodes and robot_id not in self.shared_robot_ids:
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
                "shared_robot_ids": sorted(self.shared_robot_ids),
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
        if not self.shared_robot_ids:
            return
        for robot_id in list(self.shared_robot_ids):
            robot = self.active.get(robot_id)
            if robot is not None and robot.current_node in self.area.conflict_zone_nodes:
                continue
            self.shared_robot_ids.remove(robot_id)
            self._sync_shared_robot_id()
            if robot is not None:
                robot.status = "active"
                self._event("shared_area_released", robot, node=robot.current_node)
            else:
                self.event_log.append(
                    {"step": self.step_count, "event_type": "shared_area_released", "robot_id": robot_id}
                )

    def _add_shared_robot(self, robot_id: str) -> None:
        self.shared_robot_ids.add(robot_id)
        self._sync_shared_robot_id()

    def _sync_shared_robot_id(self) -> None:
        self.shared_robot_id = sorted(self.shared_robot_ids)[0] if self.shared_robot_ids else None

    def _compatible_with_shared_area(self, robot: FCFSRobotState) -> bool:
        if len(self.shared_robot_ids) >= self.shared_area_capacity:
            return False
        robot_conflicts = set(self._path_conflict_points(robot.path[robot.path_index:]))
        for shared_robot_id in self.shared_robot_ids:
            other = self.active.get(shared_robot_id)
            if other is None:
                continue
            other_conflicts = set(self._path_conflict_points(other.path[other.path_index:]))
            if robot_conflicts & other_conflicts:
                return False
        return True

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
