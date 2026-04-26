from __future__ import annotations

from deadlock.dsd_fsd.benchmark.policies.base import (
    BasePolicy,
    assign_robot_to_task,
    available_robots,
    distance_to_task,
    waiting_tasks,
)


class RuleBasedPolicy(BasePolicy):
    """Nearest-task shortest-path baseline with wait-first recovery."""

    policy_type = "rule_based"

    def assign_tasks(self, sim, step: int) -> None:
        """Assign nearest waiting task to nearest idle robot.

        Args:
            sim: Running simulator.
            step: Current timestep.

        Returns:
            None.
        """

        free = available_robots(sim)
        waiting = waiting_tasks(sim)
        while free and waiting:
            robot, task = min(
                ((robot, task) for robot in free for task in waiting),
                key=lambda pair: (
                    distance_to_task(pair[0], pair[1]),
                    pair[1].created_step,
                    pair[0].id,
                    pair[1].id,
                ),
            )
            assign_robot_to_task(sim, robot, task, step)
            free.remove(robot)
            waiting.remove(task)
