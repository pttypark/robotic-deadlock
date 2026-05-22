"""Core AGV topology, route, state, and event types for RWARE extensions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


AGV_ALLOW_BACKWARD = False
DETECT_ONLY = True


class Heading(Enum):
    """Cardinal AGV heading encoded as clockwise degrees."""

    NORTH = 0
    EAST = 90
    SOUTH = 180
    WEST = 270


class AGVAction(Enum):
    """Allowed AGV actions for lane-graph movement."""

    WAIT = "wait"
    FORWARD = "forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    BACKWARD = "backward"


def turn_left(heading: Heading) -> Heading:
    """Return the heading after a 90 degree left turn."""

    return Heading((heading.value - 90) % 360)


def turn_right(heading: Heading) -> Heading:
    """Return the heading after a 90 degree right turn."""

    return Heading((heading.value + 90) % 360)


def opposite_heading(heading: Heading) -> Heading:
    """Return the heading opposite to the current heading."""

    return Heading((heading.value + 180) % 360)


@dataclass
class Node:
    """AGV lane graph node mapped to one RWARE grid cell.

    Args:
        node_id: Stable node identifier.
        position: Grid position as (row, col).
        node_type: normal_lane, waiting, conflict, merge, etc.
        allowed_turns: Allowed action strings at this node.
        is_blocked: Whether this node is temporarily unavailable.
        interaction_area_id: Area id this node primarily belongs to, if any.
        metadata: Free extension data for later algorithms.

    Returns:
        Node instance.
    """

    node_id: str
    position: tuple[int, int]
    node_type: str
    allowed_turns: list[str]
    is_blocked: bool = False
    interaction_area_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Directed movement relation between two AGV nodes.

    Args:
        edge_id: Stable edge identifier.
        from_node: Source node id.
        to_node: Target node id.
        direction: Cardinal direction required to traverse this edge.
        distance: Traversal cost or geometric distance.
        is_one_way: Whether reverse movement requires a separate edge.
        edge_type: lane, intersection, merge, bottleneck, turn, station_access.
        allowed_robot_types: Robot classes allowed to use this edge.
        metadata: Free extension data for later algorithms.

    Returns:
        Edge instance.
    """

    edge_id: str
    from_node: str
    to_node: str
    direction: str
    distance: float = 1.0
    is_one_way: bool = True
    edge_type: str = "lane"
    allowed_robot_types: list[str] = field(default_factory=lambda: ["AGV"])
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Route:
    """Predefined AGV route through a lane graph.

    Args:
        route_id: Stable route id.
        entry_direction: Direction from which the AGV enters the area.
        exit_direction: Direction where the AGV leaves the area.
        route_type: straight, left_turn, right_turn, pass, merge, etc.
        node_sequence: Ordered node ids.
        conflict_points_on_route: Conflict point node ids used by the route.

    Returns:
        Route instance.
    """

    route_id: str
    entry_direction: str
    exit_direction: str
    route_type: str
    node_sequence: list[str]
    conflict_points_on_route: list[str]


@dataclass
class AGVMission:
    """Purpose assigned to one AGV in the warehouse layout.

    Args:
        mission_id: Stable mission id.
        pickup_node: Shelf-access node the AGV should visit.
        dropoff_node: Station/goal node where the AGV returns.
        phase: to_pickup, to_dropoff, or completed.
        metadata: Free extension data for later algorithms.

    Returns:
        AGVMission instance.
    """

    mission_id: str
    pickup_node: str
    dropoff_node: str
    phase: str = "to_pickup"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionArea:
    """Paper-style interaction area with communication, waiting, and conflict zones.

    Args:
        area_id: Stable interaction area id.
        area_type: intersection, merge, bottleneck, or turn.
        communication_zone_nodes: Nodes where communication/scheduling starts.
        conflict_zone_nodes: Nodes inside the physical conflict zone.
        waiting_points: WP node ids.
        conflict_points: CP node ids.
        allowed_routes: Predefined area routes.
        priority_rule: Name reserved for future scheduling logic.
        metadata: Free extension data for later algorithms.

    Returns:
        InteractionArea instance.
    """

    area_id: str
    area_type: str
    communication_zone_nodes: set[str]
    conflict_zone_nodes: set[str]
    waiting_points: set[str]
    conflict_points: set[str]
    allowed_routes: dict[str, Route]
    priority_rule: str = "priority_then_fcfs"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AGVState:
    """Runtime AGV state on a predefined route and lane graph.

    Args:
        robot_id: Stable robot id, for example AGV_1.
        current_node: Current node id.
        previous_node: Previous node id, if any.
        next_node: Last proposed next node id, if any.
        heading: Current heading.
        priority: Integer priority from 1 to 5.
        assigned_route_id: Predefined route id, if assigned.
        current_interaction_area_id: Active interaction area id, if any.

    Returns:
        AGVState instance.
    """

    robot_id: str
    current_node: str
    previous_node: str | None
    next_node: str | None
    heading: Heading | str
    priority: int = 1
    assigned_route_id: str | None = None
    current_interaction_area_id: str | None = None
    has_entered_communication_zone: bool = False
    is_waiting_at_waiting_point: bool = False
    has_entered_conflict_zone: bool = False
    occupied_conflict_point: str | None = None
    waiting_time: int = 0
    delay_time: int = 0
    route_index: int = 0
    mission: AGVMission | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize robot id, heading, and priority after construction."""

        if isinstance(self.robot_id, int):
            self.robot_id = f"AGV_{self.robot_id}"
        if isinstance(self.heading, str):
            self.heading = Heading[self.heading]
        self.priority = min(5, max(1, int(self.priority)))

    @property
    def agv_id(self) -> int:
        """Return a numeric id for compatibility with earlier demos."""

        suffix = str(self.robot_id).split("_")[-1]
        return int(suffix) if suffix.isdigit() else 0


@dataclass(frozen=True)
class MoveProposal:
    """One AGV movement candidate before conflict-risk detection."""

    robot_id: str
    from_node: str
    to_node: str
    priority: int
    route_id: str | None
    action: AGVAction
    valid: bool
    reason: str = ""
    interaction_area_id: str | None = None

    @property
    def agv_id(self) -> int:
        """Return numeric id for compatibility with earlier tests."""

        suffix = str(self.robot_id).split("_")[-1]
        return int(suffix) if suffix.isdigit() else 0


@dataclass
class SimulationEvent:
    """Serializable event emitted by movement and interaction-area updates."""

    step: int
    event_type: str
    robots: list[str] = field(default_factory=list)
    interaction_area_id: str | None = None
    node: str | None = None
    edge: tuple[str, str] | None = None
    priorities: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> str:
        """Compatibility alias for previous event consumers."""

        return self.event_type

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable event dictionary."""

        return {
            "step": self.step,
            "event_type": self.event_type,
            "type": self.event_type,
            "robots": list(self.robots),
            "interaction_area_id": self.interaction_area_id,
            "node": self.node,
            "edge": self.edge,
            "priorities": dict(self.priorities),
            "metadata": dict(self.metadata),
        }


ConflictEvent = SimulationEvent
