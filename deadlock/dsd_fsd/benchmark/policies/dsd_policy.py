from __future__ import annotations

from deadlock.dsd_fsd.benchmark.policies.base import BasePolicy, assign_robot_to_task, waiting_tasks


class DSDPolicy(BasePolicy):
    """Dedicated System Design policy with hard zone boundaries."""

    policy_type = "dsd"

    def assign_tasks(self, sim, step: int) -> None:
        """Assign each task only to an idle robot dedicated to its zone.

        Args:
            sim: Running simulator.
            step: Current timestep.

        Returns:
            None.
        """

        free_by_zone = {}
        for robot in sorted(sim.robots, key=lambda item: item.id):
            if robot.is_idle:
                free_by_zone.setdefault(robot.zone_id, []).append(robot)

        for task in waiting_tasks(sim):
            robots = free_by_zone.get(task.zone_id)
            if not robots:
                continue
            robot = robots.pop(0)
            assign_robot_to_task(sim, robot, task, step)

    def allowed_cells(self, sim, robot):
        """Return only cells inside the robot's dedicated zone.

        Args:
            sim: Running simulator.
            robot: Robot state.

        Returns:
            Zone-limited traversable cells.
        """

        return sim.layout.zone_allowed[robot.zone_id]

    def priority(self, sim, robot, step: int) -> tuple:
        """Favor the longest-waiting robot inside a zone, then lower id."""

        return (robot.wait_steps, -robot.id)
