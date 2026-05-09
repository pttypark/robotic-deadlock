"""Wrapper that syncs AGV lane-graph simulation state into a RWARE env."""

from __future__ import annotations

from rware.agv_layouts import AGVLayout
from rware.agv_movement import AGVMovementSimulator
from rware.agv_types import AGVAction, Heading
from rware.warehouse import Direction, RewardType, Warehouse


HEADING_TO_RWARE = {
    Heading.NORTH: Direction.UP,
    Heading.EAST: Direction.RIGHT,
    Heading.SOUTH: Direction.DOWN,
    Heading.WEST: Direction.LEFT,
}


class AGVRWARESimulationWrapper:
    """Run AGV topology simulation while using RWARE for visualization."""

    def __init__(
        self,
        layout: AGVLayout,
        max_steps: int = 200,
        render_mode: str = "human",
        detect_only: bool = True,
    ) -> None:
        """Create RWARE env and AGV simulator.

        Args:
            layout: AGVLayout to visualize.
            max_steps: Maximum RWARE episode steps.
            render_mode: RWARE render mode.
            detect_only: Whether conflicts are only logged.

        Returns:
            None.
        """

        self.layout = layout
        self.simulator = AGVMovementSimulator(layout, detect_only=detect_only)
        self.env = Warehouse(
            shelf_columns=1,
            column_height=1,
            shelf_rows=1,
            n_agents=len(self.simulator.agvs),
            msg_bits=0,
            sensor_range=1,
            request_queue_size=1,
            max_inactivity_steps=None,
            max_steps=max_steps,
            reward_type=RewardType.GLOBAL,
            layout=layout.rware_layout,
            render_mode=render_mode,
        )

    def reset(self, seed: int | None = None):
        """Reset RWARE and sync AGV positions.

        Args:
            seed: Optional RWARE reset seed.

        Returns:
            RWARE reset output.
        """

        output = self.env.reset(seed=seed)
        self.sync_to_rware()
        return output

    def step(self, actions: dict[str | int, AGVAction | str]) -> dict:
        """Advance AGV simulator and sync resulting positions into RWARE.

        Args:
            actions: AGV action mapping.

        Returns:
            AGV simulator step result.
        """

        result = self.simulator.step(actions)
        self.sync_to_rware()
        return result

    def render(self):
        """Render the wrapped RWARE environment."""

        return self.env.render()

    def close(self) -> None:
        """Close the RWARE renderer."""

        self.env.close()

    def sync_to_rware(self) -> None:
        """Copy lane-graph AGV state into RWARE agents for rendering."""

        for index, agv in enumerate(self.simulator.agvs):
            node = self.simulator.graph.nodes[agv.current_node]
            row, col = node.position
            agent = self.env.agents[index]
            agent.prev_x = agent.x
            agent.prev_y = agent.y
            agent.x = col
            agent.y = row
            agent.dir = HEADING_TO_RWARE[agv.heading]
        self.env._recalc_grid()
