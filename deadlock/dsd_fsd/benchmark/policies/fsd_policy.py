from __future__ import annotations

from collections import defaultdict

from deadlock.dsd_fsd.benchmark.layout_builder import manhattan, nearest_aisle_index
from deadlock.dsd_fsd.benchmark.policies.base import BasePolicy, assign_robot_to_task
from deadlock.dsd_fsd.benchmark.types import RobotState


class FSDPolicy(BasePolicy):
    """Flexible System Design policy using decision/waiting/escape points."""

    policy_type = "fsd"

    def __init__(
        self,
        max_wait_steps: int = 6,
        alpha: float = 2.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        super().__init__(max_wait_steps=max_wait_steps)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def assign_tasks(self, sim, step: int) -> None:
        """Assign globally accessible tasks by wait/density/distance score.

        Args:
            sim: Running simulator.
            step: Current timestep.

        Returns:
            None.
        """

        free = sorted((robot for robot in sim.robots if robot.is_idle), key=lambda item: item.id)
        waiting = sorted(
            (task for task in sim.tasks.values() if task.is_waiting),
            key=lambda item: (item.created_step, item.id),
        )
        while free and waiting:
            density = self._aisle_density(sim)
            max_distance = max(1, sim.layout.width + sim.layout.height)
            max_density = max(1, max(density.values(), default=0), len(sim.robots))
            best_robot, best_task = max(
                ((robot, task) for robot in free for task in waiting),
                key=lambda pair: (
                    self._score(sim, pair[0], pair[1], step, density, max_distance, max_density),
                    -pair[0].id,
                    -pair[1].id,
                ),
            )
            assign_robot_to_task(sim, best_robot, best_task, step)
            free.remove(best_robot)
            waiting.remove(best_task)

    def goal_for_robot(self, sim, robot: RobotState):
        """Route major choices through decision points.

        Args:
            sim: Running simulator.
            robot: Robot state.

        Returns:
            Next decision point, buffer, bay target, or recovery target.
        """

        if robot.forced_target is not None:
            return robot.forced_target
        task = sim.tasks.get(robot.task_id)
        if task is None:
            return None
        final_goal = task.buffer if robot.phase == "to_buffer" else task.target
        if robot.position == final_goal:
            return final_goal

        if robot.phase == "to_buffer":
            if robot.position[0] == final_goal[0]:
                return final_goal
            return self._decision_routing_goal(sim, robot.position, final_goal)

        if robot.position[0] == final_goal[0]:
            return final_goal
        return self._decision_routing_goal(sim, robot.position, final_goal)

    def priority(self, sim, robot: RobotState, step: int) -> tuple:
        """Give priority to robots exiting aisles and robots waiting longer."""

        leaving_aisle = 1 if robot.phase == "to_buffer" else 0
        active_task = 1 if robot.task_id is not None else 0
        return (leaving_aisle, active_task, robot.wait_steps, -robot.id)

    def on_blocked(
        self,
        sim,
        robot: RobotState,
        step: int,
        reason: str,
        blocking_robot_id: int | None,
        blocked_cell,
    ) -> None:
        """Escalate FSD conflicts from wait to reroute to trigger.

        Args:
            sim: Running simulator.
            robot: Blocked AAMR.
            step: Current timestep.
            reason: Blocking reason.
            blocking_robot_id: DAMR id if known.
            blocked_cell: Cell requested by the AAMR.

        Returns:
            None.
        """

        if robot.wait_steps == self.max_wait_steps and blocked_cell is not None:
            sim.mark_reroute(robot.id, blocked_cell, step)
        if blocking_robot_id is None:
            return
        blocker = sim.robots[blocking_robot_id]
        blocker_is_not_clearing = blocker.is_idle or blocker.wait_steps > 0
        if robot.wait_steps >= self.max_wait_steps + 1 or blocker_is_not_clearing:
            sim.trigger_blocker(robot.id, blocking_robot_id, step)

    def _score(self, sim, robot, task, step, density, max_distance, max_density) -> float:
        waiting_time = max(0, step - task.created_step)
        wait_norm = waiting_time / max(1, step + 1)
        distance_norm = manhattan(robot.position, task.target) / max_distance
        aisle_idx = nearest_aisle_index(sim.layout.aisle_columns, task.target[0])
        density_norm = density.get(aisle_idx, 0) / max_density
        return self.alpha * wait_norm - self.beta * distance_norm - self.gamma * density_norm

    def _aisle_density(self, sim) -> dict[int, int]:
        density = defaultdict(int)
        for robot in sim.robots:
            task = sim.tasks.get(robot.task_id)
            if task is not None:
                idx = nearest_aisle_index(sim.layout.aisle_columns, task.target[0])
                density[idx] += 1
            else:
                idx = nearest_aisle_index(sim.layout.aisle_columns, robot.position[0])
                density[idx] += 1
        return dict(density)

    def _decision_routing_goal(self, sim, start, final_goal):
        current_aisle = sim.nearest_aisle_x(start)
        target_aisle = sim.nearest_aisle_x(final_goal)
        if start[1] in sim.layout.transition_rows:
            return (target_aisle, start[1])
        transition = min(
            sim.layout.transition_rows,
            key=lambda row: (abs(row - start[1]) + abs(current_aisle - start[0]), row),
        )
        return (current_aisle, transition)
