from __future__ import annotations

from dataclasses import dataclass, field

Cell = tuple[int, int]


@dataclass
class Task:
    """Task state used by the deterministic benchmark simulator.

    Args:
        id: Stable task id.
        target: Storage/bay access cell.
        buffer: Workstation/buffer cell.
        created_step: Step when the task entered the queue.
        zone_id: DSD zone assigned from the target aisle.
    """

    id: int
    target: Cell
    buffer: Cell
    created_step: int
    zone_id: int
    assigned_robot_id: int | None = None
    assigned_step: int | None = None
    reached_bay_step: int | None = None
    completed_step: int | None = None

    @property
    def is_waiting(self) -> bool:
        return self.assigned_robot_id is None and self.completed_step is None

    @property
    def is_completed(self) -> bool:
        return self.completed_step is not None

    def flow_time(self) -> int | None:
        if self.completed_step is None:
            return None
        return self.completed_step - self.created_step


@dataclass
class RobotState:
    """Robot state tracked independently from the RWARE renderer.

    Args:
        id: Robot id used for deterministic tie-breaking.
        position: Current grid cell.
        zone_id: DSD dedicated zone for this robot.
    """

    id: int
    position: Cell
    zone_id: int
    task_id: int | None = None
    phase: str = "idle"
    wait_steps: int = 0
    blocked_steps: int = 0
    travel_distance: int = 0
    busy_steps: int = 0
    forced_target: Cell | None = None
    avoid_cells: dict[Cell, int] = field(default_factory=dict)
    last_block_reason: str = ""
    last_blocked_cell: Cell | None = None

    @property
    def is_idle(self) -> bool:
        return self.task_id is None and self.forced_target is None

    @property
    def is_busy(self) -> bool:
        return self.task_id is not None or self.forced_target is not None


@dataclass(frozen=True)
class MoveProposal:
    """Proposed one-step movement before reservation/collision resolution."""

    robot_id: int
    start: Cell
    target: Cell
    wants_move: bool
    goal: Cell | None


@dataclass
class SimulationStats:
    """Counters accumulated during one episode."""

    blocking_time: int = 0
    collision_attempt_count: int = 0
    deadlock_count: int = 0
    trigger_count: int = 0
    reroute_count: int = 0
    forced_relocation_count: int = 0
    queue_length_sum: int = 0
    zone_queue_length_sum: dict[int, int] = field(default_factory=dict)
    queue_samples: int = 0
    robot_busy_steps: int = 0
    robot_travel_distance: int = 0
    completed_tasks: int = 0
