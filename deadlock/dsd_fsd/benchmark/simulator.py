from __future__ import annotations

import random
from collections import deque

from deadlock.dsd_fsd.benchmark.config import ExperimentConfig
from deadlock.dsd_fsd.benchmark.layout_builder import (
    BenchmarkLayout,
    buffer_for_target,
    build_layout,
    manhattan,
    nearest_aisle_index,
    zone_for_target,
)
from deadlock.dsd_fsd.benchmark.policies import DSDPolicy, FSDPolicy, RuleBasedPolicy
from deadlock.dsd_fsd.benchmark.types import Cell, MoveProposal, RobotState, SimulationStats, Task


class WarehouseSimulator:
    """Deterministic AMR warehouse simulator for Rule-based/DSD/FSD comparison.

    Args:
        config: Experiment condition.
        layout: Optional prebuilt layout. If omitted, build_layout(config) is used.
        task_arrivals: Optional prebuilt arrivals shared across policies.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        layout: BenchmarkLayout | None = None,
        task_arrivals: list[tuple[int, Cell]] | None = None,
    ) -> None:
        self.config = config
        self.layout = layout or build_layout(config)
        self.policy = self._make_policy(config.policy_type)
        self.task_arrivals = list(task_arrivals or generate_task_arrivals(config, self.layout))
        self.tasks: dict[int, Task] = {}
        self.robots = self._make_robots()
        self.stats = SimulationStats()
        self.deadlock_events: list[dict] = []
        self.trigger_events: list[dict] = []
        self._arrival_cursor = 0
        self._blocked_history: deque[tuple[int, ...]] = deque(maxlen=config.deadlock_window)
        self._last_deadlock_signature: tuple[int, ...] | None = None
        self._last_deadlock_step = -1

    def run(self) -> dict:
        """Run one bounded episode and return KPI metrics.

        Args:
            None.

        Returns:
            Flat dictionary ready to write into results.csv.
        """

        for step in range(self.config.max_episode_steps):
            self.step(step)

        return self.metrics()

    def step(self, step: int) -> dict:
        """Advance the simulator by one timestep.

        Args:
            step: Current timestep supplied by the caller.

        Returns:
            Small event dictionary with moved and blocked robot ids.
        """

        self._add_arrivals(step)
        self._advance_completed_goals(step)
        self.policy.assign_tasks(self, step)
        self._advance_completed_goals(step)

        proposals = self._build_proposals(step)
        accepted, block_info = self._resolve_moves(proposals, step)
        moved_ids = self._apply_moves(accepted, proposals)
        blocked_ids = self._record_blocking(proposals, accepted, block_info, step)
        self._advance_completed_goals(step + 1)
        self._detect_deadlock(blocked_ids, moved_ids, step)
        self._sample_queue_lengths()

        if self.config.verbose:
            self._print_verbose(step, moved_ids, blocked_ids)
        return {
            "step": step,
            "moved_ids": tuple(sorted(moved_ids)),
            "blocked_ids": blocked_ids,
        }

    def metrics(self) -> dict:
        """Compute episode-level KPI values.

        Args:
            None.

        Returns:
            Dictionary containing required and auxiliary KPI columns.
        """

        completed = [task for task in self.tasks.values() if task.is_completed]
        flows = [task.flow_time() for task in completed if task.flow_time() is not None]
        total_steps = max(1, self.config.max_episode_steps)
        avg_queue = self.stats.queue_length_sum / max(1, self.stats.queue_samples)
        utilization = self.stats.robot_busy_steps / max(1, total_steps * len(self.robots))
        avg_distance = self.stats.robot_travel_distance / max(1, len(self.robots))
        metrics = {
            "seed": self.config.seed,
            "policy_type": self.config.policy_type,
            "num_robots": self.config.num_robots,
            "num_aisles": self.config.num_aisles,
            "capacity": self.config.capacity,
            "arrival_rate": self.config.arrival_rate,
            "Favg": sum(flows) / len(flows) if flows else 0.0,
            "Fmax": max(flows) if flows else 0,
            "Throughput": len(completed),
            "BlockingTime": self.stats.blocking_time,
            "AvgQueueLength": avg_queue,
            "DeadlockCount": self.stats.deadlock_count,
            "TriggerCount": self.stats.trigger_count,
            "RerouteCount": self.stats.reroute_count,
            "CollisionAttempts": self.stats.collision_attempt_count,
            "ForcedRelocationCount": self.stats.forced_relocation_count,
            "RobotUtilization": utilization,
            "AverageTravelDistance": avg_distance,
        }
        for zone_id in range(self.config.num_robots):
            total = self.stats.zone_queue_length_sum.get(zone_id, 0)
            metrics[f"Zone{zone_id}AvgQueueLength"] = total / max(1, self.stats.queue_samples)
        return metrics

    def assign_task(self, robot_id: int, task_id: int, step: int) -> None:
        """Assign a task to a robot from policy code.

        Args:
            robot_id: Robot id.
            task_id: Task id.
            step: Assignment timestep.

        Returns:
            None.
        """

        robot = self.robots[robot_id]
        task = self.tasks[task_id]
        robot.task_id = task_id
        robot.phase = "to_bay"
        robot.forced_target = None
        robot.wait_steps = 0
        task.assigned_robot_id = robot_id
        task.assigned_step = step

    def mark_reroute(self, robot_id: int, blocked_cell: Cell, step: int) -> None:
        """Temporarily avoid a blocked cell after max_wait_steps.

        Args:
            robot_id: Robot requesting reroute.
            blocked_cell: Cell to penalize/avoid temporarily.
            step: Current timestep.

        Returns:
            None.
        """

        robot = self.robots[robot_id]
        robot.avoid_cells[blocked_cell] = step + self.config.max_wait_steps
        self.stats.reroute_count += 1

    def trigger_blocker(self, aamr_id: int, damr_id: int, step: int) -> bool:
        """Move a blocking DAMR toward a safe point for FSD.

        Args:
            aamr_id: Active robot requesting passage.
            damr_id: Blocking robot.
            step: Current timestep.

        Returns:
            True when a trigger target was installed.
        """

        if aamr_id == damr_id:
            return False
        damr = self.robots[damr_id]
        if damr.forced_target is not None:
            return False
        target = self._select_safe_point(damr, prefer_escape=True)
        if target is None:
            return False
        damr.forced_target = target
        self.stats.trigger_count += 1
        self.trigger_events.append(
            {
                "step": step,
                "aamr": aamr_id,
                "damr": damr_id,
                "damr_cell": damr.position,
                "target": target,
            }
        )
        return True

    def nearest_aisle_x(self, cell: Cell) -> int:
        """Return nearest aisle x-coordinate for a cell."""

        idx = nearest_aisle_index(self.layout.aisle_columns, cell[0])
        return self.layout.aisle_columns[idx]

    def shortest_path(
        self,
        start: Cell,
        goal: Cell,
        allowed: frozenset[Cell],
        avoid: set[Cell] | None = None,
    ) -> list[Cell]:
        """Find a deterministic shortest path over allowed cells.

        Args:
            start: Start cell.
            goal: Goal cell.
            allowed: Static cells the robot may enter.
            avoid: Temporary cells to avoid during reroute.

        Returns:
            Path including start and goal, or an empty list if unreachable.
        """

        if goal is None:
            return []
        avoid = set(avoid or set())
        avoid.discard(start)
        avoid.discard(goal)
        if start == goal:
            return [start]
        if start not in allowed or goal not in allowed:
            return []

        parents: dict[Cell, Cell | None] = {start: None}
        queue: deque[Cell] = deque([start])
        while queue:
            cell = queue.popleft()
            for nxt in self._neighbors(cell):
                if nxt in parents or nxt not in allowed or nxt in avoid:
                    continue
                parents[nxt] = cell
                if nxt == goal:
                    return _reconstruct_path(parents, goal)
                queue.append(nxt)
        return []

    def _make_policy(self, policy_type: str):
        if policy_type == "dsd":
            return DSDPolicy(max_wait_steps=self.config.max_wait_steps)
        if policy_type == "fsd":
            return FSDPolicy(max_wait_steps=self.config.max_wait_steps)
        return RuleBasedPolicy(max_wait_steps=self.config.max_wait_steps)

    def _make_robots(self) -> list[RobotState]:
        starts = self._start_positions()
        return [
            RobotState(id=idx, position=starts[idx], zone_id=idx % self.config.num_robots)
            for idx in range(self.config.num_robots)
        ]

    def _start_positions(self) -> list[Cell]:
        starts: list[Cell] = []
        used = set()
        for robot_id in range(self.config.num_robots):
            zone = self.layout.zones[robot_id % len(self.layout.zones)]
            candidates = []
            if zone:
                zone_xs = sorted({point[0] for point in zone})
                candidates = [
                    point
                    for point in self.layout.workstations
                    if point[0] in zone_xs and point not in used
                ]
            if not candidates:
                candidates = [point for point in self.layout.workstations if point not in used]
            if not candidates:
                candidates = [point for point in sorted(self.layout.decision_points) if point not in used]
            start = candidates[0]
            used.add(start)
            starts.append(start)
        return starts

    def _add_arrivals(self, step: int) -> None:
        while self._arrival_cursor < len(self.task_arrivals):
            arrival_step, target = self.task_arrivals[self._arrival_cursor]
            if arrival_step > step:
                break
            task_id = len(self.tasks)
            self.tasks[task_id] = Task(
                id=task_id,
                target=target,
                buffer=buffer_for_target(self.layout, target),
                created_step=arrival_step,
                zone_id=zone_for_target(self.layout, target),
            )
            self._arrival_cursor += 1

    def _advance_completed_goals(self, step: int) -> None:
        for robot in self.robots:
            if robot.forced_target is not None and robot.position == robot.forced_target:
                robot.forced_target = None
                robot.wait_steps = 0
                continue
            task = self.tasks.get(robot.task_id)
            if task is None:
                continue
            if robot.phase == "to_bay" and robot.position == task.target:
                task.reached_bay_step = step
                robot.phase = "to_buffer"
                robot.wait_steps = 0
            elif robot.phase == "to_buffer" and robot.position == task.buffer:
                task.completed_step = step
                robot.task_id = None
                robot.phase = "idle"
                robot.wait_steps = 0
                self.stats.completed_tasks += 1

    def _build_proposals(self, step: int) -> dict[int, MoveProposal]:
        proposals: dict[int, MoveProposal] = {}
        for robot in self.robots:
            self._expire_avoid_cells(robot, step)
            goal = self.policy.goal_for_robot(self, robot)
            if goal is None or goal == robot.position:
                proposals[robot.id] = MoveProposal(robot.id, robot.position, robot.position, False, goal)
                continue

            allowed = self.policy.allowed_cells(self, robot)
            avoid = {cell for cell, expires_at in robot.avoid_cells.items() if expires_at > step}
            path = self.shortest_path(robot.position, goal, allowed=allowed, avoid=avoid)
            if len(path) < 2:
                proposals[robot.id] = MoveProposal(robot.id, robot.position, robot.position, True, goal)
                continue
            proposals[robot.id] = MoveProposal(robot.id, robot.position, path[1], True, goal)
        return proposals

    def _resolve_moves(
        self,
        proposals: dict[int, MoveProposal],
        step: int,
    ) -> tuple[set[int], dict[int, tuple[str, int | None, Cell | None]]]:
        accepted = {robot_id for robot_id, proposal in proposals.items() if proposal.wants_move}
        block_info: dict[int, tuple[str, int | None, Cell | None]] = {}
        occupied = {robot.position: robot.id for robot in self.robots}

        changed = True
        while changed:
            changed = False
            changed |= self._block_vertex_conflicts(accepted, proposals, block_info, step)
            changed |= self._block_edge_conflicts(accepted, proposals, block_info, step)
            changed |= self._block_decision_conflicts(accepted, proposals, block_info, step)
            changed |= self._block_occupied_targets(accepted, proposals, occupied, block_info)

        return accepted, block_info

    def _block_vertex_conflicts(self, accepted, proposals, block_info, step: int) -> bool:
        by_target: dict[Cell, list[int]] = {}
        for robot_id in accepted:
            by_target.setdefault(proposals[robot_id].target, []).append(robot_id)
        changed = False
        for target, robot_ids in by_target.items():
            if len(robot_ids) <= 1:
                continue
            winner = self._winner(robot_ids, step)
            for robot_id in robot_ids:
                if robot_id == winner or robot_id not in accepted:
                    continue
                accepted.remove(robot_id)
                block_info[robot_id] = ("vertex_collision", winner, target)
                self.stats.collision_attempt_count += 1
                changed = True
        return changed

    def _block_edge_conflicts(self, accepted, proposals, block_info, step: int) -> bool:
        changed = False
        robot_ids = sorted(accepted)
        for idx, left_id in enumerate(robot_ids):
            if left_id not in accepted:
                continue
            left = proposals[left_id]
            for right_id in robot_ids[idx + 1:]:
                if right_id not in accepted:
                    continue
                right = proposals[right_id]
                if left.start == right.target and left.target == right.start:
                    winner = self._winner([left_id, right_id], step)
                    loser = right_id if winner == left_id else left_id
                    accepted.remove(loser)
                    block_info[loser] = ("edge_collision", winner, proposals[loser].target)
                    self.stats.collision_attempt_count += 1
                    changed = True
        return changed

    def _block_decision_conflicts(self, accepted, proposals, block_info, step: int) -> bool:
        changed = False
        by_decision: dict[Cell, list[int]] = {}
        for robot_id in accepted:
            target = proposals[robot_id].target
            if target in self.layout.decision_points:
                by_decision.setdefault(target, []).append(robot_id)
        for target, robot_ids in by_decision.items():
            occupants = [
                robot.id
                for robot in self.robots
                if robot.position == target and robot.id not in robot_ids
            ]
            if occupants:
                robot_ids.extend(occupants)
            if len(robot_ids) <= 1:
                continue
            entrants = [robot_id for robot_id in robot_ids if robot_id in accepted]
            if not entrants:
                continue
            winner = self._winner(robot_ids, step)
            for robot_id in entrants:
                if robot_id == winner:
                    continue
                accepted.remove(robot_id)
                block_info[robot_id] = ("decision_reserved", winner, target)
                self.stats.collision_attempt_count += 1
                changed = True
        return changed

    def _block_occupied_targets(self, accepted, proposals, occupied, block_info) -> bool:
        changed = False
        for robot_id in sorted(list(accepted)):
            target = proposals[robot_id].target
            occupant_id = occupied.get(target)
            if occupant_id is None or occupant_id == robot_id:
                continue
            occupant_moves_away = (
                occupant_id in accepted
                and proposals[occupant_id].target != proposals[occupant_id].start
                and proposals[occupant_id].target != proposals[robot_id].start
            )
            if occupant_moves_away:
                continue
            accepted.remove(robot_id)
            block_info[robot_id] = ("occupied", occupant_id, target)
            changed = True
        return changed

    def _apply_moves(self, accepted: set[int], proposals: dict[int, MoveProposal]) -> set[int]:
        moved_ids = set()
        for robot_id in sorted(accepted):
            robot = self.robots[robot_id]
            proposal = proposals[robot_id]
            if proposal.target == robot.position:
                continue
            robot.position = proposal.target
            robot.wait_steps = 0
            robot.last_block_reason = ""
            robot.last_blocked_cell = None
            robot.travel_distance += 1
            self.stats.robot_travel_distance += 1
            moved_ids.add(robot_id)
        for robot in self.robots:
            if robot.is_busy:
                robot.busy_steps += 1
                self.stats.robot_busy_steps += 1
        return moved_ids

    def _record_blocking(
        self,
        proposals: dict[int, MoveProposal],
        accepted: set[int],
        block_info: dict[int, tuple[str, int | None, Cell | None]],
        step: int,
    ) -> tuple[int, ...]:
        blocked_ids = []
        for robot_id, proposal in sorted(proposals.items()):
            robot = self.robots[robot_id]
            if not proposal.wants_move or robot_id in accepted:
                if not proposal.wants_move:
                    robot.wait_steps = 0 if robot.is_idle else robot.wait_steps
                continue
            reason, blocking_robot_id, blocked_cell = block_info.get(
                robot_id,
                ("no_path", None, proposal.goal),
            )
            robot.wait_steps += 1
            robot.blocked_steps += 1
            robot.last_block_reason = reason
            robot.last_blocked_cell = blocked_cell
            self.stats.blocking_time += 1
            blocked_ids.append(robot_id)
            self.policy.on_blocked(self, robot, step, reason, blocking_robot_id, blocked_cell)
        return tuple(blocked_ids)

    def _detect_deadlock(self, blocked_ids: tuple[int, ...], moved_ids: set[int], step: int) -> None:
        active_blocked = tuple(
            robot_id
            for robot_id in blocked_ids
            if self.robots[robot_id].is_busy and robot_id not in moved_ids
        )
        if active_blocked:
            self._blocked_history.append(active_blocked)
        else:
            self._blocked_history.clear()
            self._last_deadlock_signature = None
            return

        if len(self._blocked_history) < self.config.deadlock_window:
            return
        signature = self._blocked_history[0]
        if not signature or any(item != signature for item in self._blocked_history):
            return
        if self._last_deadlock_signature == signature and step - self._last_deadlock_step < self.config.deadlock_window:
            return
        self._last_deadlock_signature = signature
        self._last_deadlock_step = step
        self.stats.deadlock_count += 1
        event = {
            "step": step,
            "robots": list(signature),
            "cells": {robot_id: self.robots[robot_id].position for robot_id in signature},
        }
        self.deadlock_events.append(event)
        loser = min(signature, key=lambda robot_id: self.policy.priority(self, self.robots[robot_id], step))
        self._force_safe_relocation(loser, step)

    def _force_safe_relocation(self, robot_id: int, step: int) -> bool:
        robot = self.robots[robot_id]
        target = self._select_safe_point(robot, prefer_escape=True)
        if target is not None:
            robot.forced_target = target
            self.stats.forced_relocation_count += 1
            return True

        for neighbor in self._neighbors(robot.position):
            if neighbor in self.layout.traversable and neighbor not in {item.position for item in self.robots}:
                robot.position = neighbor
                robot.wait_steps = 0
                self.stats.forced_relocation_count += 1
                self.deadlock_events.append(
                    {
                        "step": step,
                        "robots": [robot_id],
                        "cells": {robot_id: neighbor},
                        "forced_direct_relocation": True,
                    }
                )
                return True
        return False

    def _select_safe_point(self, robot: RobotState, prefer_escape: bool) -> Cell | None:
        occupied = {item.position for item in self.robots if item.id != robot.id}
        candidate_groups = []
        if prefer_escape:
            candidate_groups.append(sorted(self.layout.escape_points))
        candidate_groups.extend(
            [
                sorted(self.layout.waiting_points),
                sorted(self.layout.decision_points),
                sorted(self.layout.traversable),
            ]
        )
        allowed = self.policy.allowed_cells(self, robot)
        for candidates in candidate_groups:
            reachable = []
            for target in candidates:
                if target == robot.position or target in occupied or target not in allowed:
                    continue
                path = self.shortest_path(robot.position, target, allowed=allowed, avoid=occupied)
                if len(path) > 1:
                    reachable.append((len(path), manhattan(robot.position, target), target))
            if reachable:
                return min(reachable)[-1]
        return None

    def _sample_queue_lengths(self) -> None:
        waiting = [task for task in self.tasks.values() if task.is_waiting]
        self.stats.queue_length_sum += len(waiting)
        for zone_id in range(self.config.num_robots):
            self.stats.zone_queue_length_sum.setdefault(zone_id, 0)
        for task in waiting:
            self.stats.zone_queue_length_sum[task.zone_id] = self.stats.zone_queue_length_sum.get(task.zone_id, 0) + 1
        self.stats.queue_samples += 1

    def _winner(self, robot_ids: list[int], step: int) -> int:
        return max(robot_ids, key=lambda robot_id: self.policy.priority(self, self.robots[robot_id], step))

    def _neighbors(self, cell: Cell) -> tuple[Cell, ...]:
        x, y = cell
        candidates = ((x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1))
        return tuple(
            (nx, ny)
            for nx, ny in candidates
            if 0 <= nx < self.layout.width and 0 <= ny < self.layout.height
        )

    def _expire_avoid_cells(self, robot: RobotState, step: int) -> None:
        for cell, expires_at in list(robot.avoid_cells.items()):
            if expires_at <= step:
                robot.avoid_cells.pop(cell, None)

    def _print_verbose(self, step: int, moved_ids: set[int], blocked_ids: tuple[int, ...]) -> None:
        states = []
        for robot in self.robots:
            states.append(
                f"r{robot.id}@{robot.position}:task={robot.task_id},phase={robot.phase},wait={robot.wait_steps}"
            )
        print(f"step={step} moved={sorted(moved_ids)} blocked={list(blocked_ids)} {' | '.join(states)}")


def generate_task_arrivals(config: ExperimentConfig, layout: BenchmarkLayout) -> list[tuple[int, Cell]]:
    """Generate one deterministic task-arrival sequence for a scenario.

    Args:
        config: Experiment condition. policy_type is intentionally ignored.
        layout: Layout providing storage points.

    Returns:
        List of (step, target) arrivals shared across policies.
    """

    rng = random.Random(config.seed)
    storage = list(layout.storage_points)
    arrivals: list[tuple[int, Cell]] = []
    for step in range(0, config.max_episode_steps, config.arrival_interval):
        target = _sample_target(storage, layout, rng, config.arrival_rate)
        arrivals.append((step, target))
    return arrivals


def _sample_target(storage: list[Cell], layout: BenchmarkLayout, rng: random.Random, arrival_rate: str) -> Cell:
    if arrival_rate == "high" and len(layout.aisle_columns) >= 3 and rng.random() < 0.55:
        middle = len(layout.aisle_columns) // 2
        hot_aisles = set(layout.aisle_columns[max(0, middle - 1): min(len(layout.aisle_columns), middle + 2)])
        candidates = [point for point in storage if point[0] in hot_aisles]
        return candidates[rng.randrange(len(candidates))]
    if arrival_rate == "medium" and rng.random() < 0.35:
        upper_rows = sorted({point[1] for point in storage})[: max(1, len(storage) // max(1, len(layout.aisle_columns) * 3))]
        candidates = [point for point in storage if point[1] in set(upper_rows)]
        return candidates[rng.randrange(len(candidates))]
    return storage[rng.randrange(len(storage))]


def _reconstruct_path(parents: dict[Cell, Cell | None], goal: Cell) -> list[Cell]:
    path = [goal]
    current = goal
    while parents[current] is not None:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path
