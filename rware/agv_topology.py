"""Lane graph primitives for constraining AGVs to predefined guide paths."""

from __future__ import annotations

from dataclasses import dataclass, field

from rware.agv_types import (
    AGV_ALLOW_BACKWARD,
    AGVAction,
    Edge,
    Heading,
    Node,
    opposite_heading,
    turn_left,
    turn_right,
)


_DELTAS = {
    Heading.NORTH: (-1, 0),
    Heading.EAST: (0, 1),
    Heading.SOUTH: (1, 0),
    Heading.WEST: (0, -1),
}


def heading_between(start: tuple[int, int], target: tuple[int, int]) -> Heading:
    """Return the cardinal heading from start to an adjacent target.

    Args:
        start: Source position as (row, col).
        target: Adjacent target position as (row, col).

    Returns:
        Heading needed to move from start to target.
    """

    delta = (target[0] - start[0], target[1] - start[1])
    for heading, expected in _DELTAS.items():
        if delta == expected:
            return heading
    raise ValueError(f"Nodes are not adjacent cardinal cells: {start} -> {target}")


@dataclass
class LaneGraph:
    """Directed node-edge graph used by AGV movement and future algorithms."""

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    outgoing: dict[str, list[str]] = field(default_factory=dict)
    position_to_node: dict[tuple[int, int], str] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        """Add one node and index it by id and grid position."""

        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node id: {node.node_id}")
        if node.position in self.position_to_node:
            raise ValueError(f"Duplicate node position: {node.position}")
        self.nodes[node.node_id] = node
        self.position_to_node[node.position] = node.node_id
        self.outgoing.setdefault(node.node_id, [])

    def add_edge(self, edge: Edge) -> None:
        """Add one directed edge to the graph."""

        if edge.edge_id in self.edges:
            raise ValueError(f"Duplicate edge id: {edge.edge_id}")
        if edge.from_node not in self.nodes or edge.to_node not in self.nodes:
            raise ValueError(f"Unknown edge endpoint: {edge.from_node} -> {edge.to_node}")
        self.edges[edge.edge_id] = edge
        self.outgoing.setdefault(edge.from_node, []).append(edge.edge_id)

    def add_adjacent_edge(
        self,
        from_node: str,
        to_node: str,
        edge_type: str = "lane",
        is_one_way: bool = True,
    ) -> None:
        """Create an edge between adjacent grid nodes.

        Args:
            from_node: Source node id.
            to_node: Target node id.
            edge_type: Semantic edge type.
            is_one_way: Whether the relation is one-way.

        Returns:
            None.
        """

        start = self.nodes[from_node].position
        target = self.nodes[to_node].position
        direction = heading_between(start, target).name
        edge = Edge(
            edge_id=f"{from_node}->{to_node}",
            from_node=from_node,
            to_node=to_node,
            direction=direction,
            is_one_way=is_one_way,
            edge_type=edge_type,
        )
        self.add_edge(edge)

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        """Return all outgoing edges for one node id."""

        return [self.edges[edge_id] for edge_id in self.outgoing.get(node_id, [])]

    def edge_for_heading(self, node_id: str, heading: Heading) -> Edge | None:
        """Return the outgoing edge matching a heading, if present."""

        for edge in self.outgoing_edges(node_id):
            if edge.direction == heading.name:
                return edge
        return None

    def next_for_action(
        self,
        current_node: str,
        heading: Heading,
        action: AGVAction,
        allow_backward: bool = AGV_ALLOW_BACKWARD,
    ) -> tuple[str, Heading, bool, str]:
        """Resolve an AGV action into the next node and heading.

        Args:
            current_node: Current node id.
            heading: Current AGV heading.
            action: Requested AGV action.
            allow_backward: Feature flag for future reverse movement experiments.

        Returns:
            Tuple of (next_node, next_heading, valid, reason).
        """

        node = self.nodes[current_node]
        if action == AGVAction.WAIT:
            return current_node, heading, True, "wait"
        if node.is_blocked:
            return current_node, heading, False, "current_node_blocked"

        allowed = set(node.allowed_turns)
        if action.value not in allowed:
            return current_node, heading, False, "action_not_allowed_at_node"
        if action == AGVAction.FORWARD:
            next_heading = heading
        elif action == AGVAction.TURN_LEFT:
            next_heading = turn_left(heading)
        elif action == AGVAction.TURN_RIGHT:
            next_heading = turn_right(heading)
        elif action == AGVAction.BACKWARD and allow_backward:
            next_heading = opposite_heading(heading)
        else:
            return current_node, heading, False, "backward_disabled"

        edge = self.edge_for_heading(current_node, next_heading)
        if edge is None:
            return current_node, heading, False, "no_edge_for_heading"
        target = self.nodes[edge.to_node]
        if target.is_blocked:
            return current_node, heading, False, "target_node_blocked"
        return edge.to_node, next_heading, True, "ok"
