from __future__ import annotations

from typing import TYPE_CHECKING

from deadlock.dsd_fsd.benchmark.layout_builder import manhattan
from deadlock.dsd_fsd.benchmark.types import Cell, RobotState, Task

if TYPE_CHECKING:
    from deadlock.dsd_fsd.benchmark.simulator import WarehouseSimulator


class BasePolicy:
    """Base interface for deterministic warehouse policies.

    Args:
        max_wait_steps: Wait threshold before recovery actions start.
    """

    policy_type = "base"

    def __init__(self, max_wait_steps: int = 6) -> None:
        self.max_wait_steps = max_wait_steps

    def assign_tasks(self, sim: "WarehouseSimulator", step: int) -> None:
        """Assign waiting tasks to idle robots.

        Args:
            sim: Running simulator.
            step: Current timestep.

        Returns:
            None. Mutates robot/task assignment.
        """

        raise NotImplementedError

    def goal_for_robot(self, sim: "WarehouseSimulator", robot: RobotState) -> Cell | None:
        """Return the current navigation goal for a robot.

        Args:
            sim: Running simulator.
            robot: Robot state.

        Returns:
            Target cell or None when the robot should stay idle.
        """

        if robot.forced_target is not None:
            return robot.forced_target
        task = sim.tasks.get(robot.task_id)
        if task is None:
            return None
        if robot.phase == "to_buffer":
            return task.buffer
        return task.target

    def allowed_cells(self, sim: "WarehouseSimulator", robot: RobotState) -> frozenset[Cell]:
        """Return traversable cells allowed for the robot.

        Args:
            sim: Running simulator.
            robot: Robot state.

        Returns:
            Set of allowed grid cells.
        """

        return sim.layout.traversable

    def priority(self, sim: "WarehouseSimulator", robot: RobotState, step: int) -> tuple:
        """Return deterministic conflict priority.

        Higher tuple wins. Smaller robot id wins after policy-specific terms.
        """

        return (robot.wait_steps, -robot.id)

    def on_blocked(
        self,
        sim: "WarehouseSimulator",
        robot: RobotState,
        step: int,
        reason: str,
        blocking_robot_id: int | None,
        blocked_cell: Cell | None,
    ) -> None:
        """React after a robot failed to move.

        Args:
            sim: Running simulator.
            robot: Blocked robot.
            step: Current timestep.
            reason: Blocking reason.
            blocking_robot_id: Robot occupying/conflicting with the target, if known.
            blocked_cell: Requested cell that caused the block.

        Returns:
            None. May mutate recovery targets/counters.
        """

        if robot.wait_steps == self.max_wait_steps and blocked_cell is not None:
            sim.mark_reroute(robot.id, blocked_cell, step)


def assign_robot_to_task(sim: "WarehouseSimulator", robot: RobotState, task: Task, step: int) -> None:
    """Attach a task to a robot with deterministic bookkeeping."""

    robot.task_id = task.id
    robot.phase = "to_bay"
    robot.wait_steps = 0
    robot.forced_target = None
    task.assigned_robot_id = robot.id
    task.assigned_step = step


def available_robots(sim: "WarehouseSimulator") -> list[RobotState]:
    """Return idle robots sorted by robot id."""

    return sorted((robot for robot in sim.robots if robot.is_idle), key=lambda item: item.id)


def waiting_tasks(sim: "WarehouseSimulator") -> list[Task]:
    """Return waiting tasks sorted by creation time and task id."""

    return sorted(
        (task for task in sim.tasks.values() if task.is_waiting),
        key=lambda item: (item.created_step, item.id),
    )


def distance_to_task(robot: RobotState, task: Task) -> int:
    """Return a simple deterministic travel proxy for assignment."""

    return manhattan(robot.position, task.target) + manhattan(task.target, task.buffer)
