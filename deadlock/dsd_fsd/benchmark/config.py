from __future__ import annotations

from dataclasses import dataclass


POLICY_TYPES = ("rule_based", "dsd", "fsd")
CAPACITY_TO_BAYS = {
    "small": 18,
    "large": 30,
}
ARRIVAL_INTERVALS = {
    "low": 12,
    "medium": 8,
    "high": 5,
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Single deterministic experiment condition.

    Args:
        policy_type: One of rule_based, dsd, or fsd.
        num_robots: Number of AMRs in the episode.
        num_aisles: Number of vertical access aisles.
        capacity: Warehouse size label. Maps to a fixed bay row count.
        arrival_rate: Demand intensity label. Maps to a fixed arrival interval.
        seed: Seed used only for deterministic task sequence generation.
        max_episode_steps: Hard stop that prevents infinite episodes.
        max_wait_steps: Wait threshold before reroute/trigger/relocation.
        deadlock_window: Consecutive blocked-window length for deadlock detection.
        verbose: If true, prints per-robot state each step.
    """

    policy_type: str = "rule_based"
    num_robots: int = 4
    num_aisles: int = 6
    capacity: str = "small"
    arrival_rate: str = "medium"
    seed: int = 0
    max_episode_steps: int = 500
    max_wait_steps: int = 6
    deadlock_window: int = 8
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.policy_type not in POLICY_TYPES:
            raise ValueError(f"Unknown policy_type: {self.policy_type}")
        if self.capacity not in CAPACITY_TO_BAYS:
            raise ValueError(f"Unknown capacity: {self.capacity}")
        if self.arrival_rate not in ARRIVAL_INTERVALS:
            raise ValueError(f"Unknown arrival_rate: {self.arrival_rate}")
        if self.num_robots <= 0:
            raise ValueError("num_robots must be positive")
        if self.num_aisles <= 0:
            raise ValueError("num_aisles must be positive")

    @property
    def bay_count(self) -> int:
        return CAPACITY_TO_BAYS[self.capacity]

    @property
    def arrival_interval(self) -> int:
        return ARRIVAL_INTERVALS[self.arrival_rate]
