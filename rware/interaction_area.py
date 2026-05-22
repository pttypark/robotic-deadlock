"""Interaction-area registry helpers for AGV simulations."""

from __future__ import annotations

from rware.agv_types import InteractionArea


class InteractionAreaRegistry:
    """Lookup helper for communication, waiting, and conflict zones."""

    def __init__(self, areas: list[InteractionArea]) -> None:
        """Create a registry from interaction-area definitions."""

        self.areas = {area.area_id: area for area in areas}

    def area_for_node(self, node_id: str) -> InteractionArea | None:
        """Return the first interaction area containing a node.

        Args:
            node_id: Lane-graph node id.

        Returns:
            Matching InteractionArea or None.
        """

        for area in self.areas.values():
            if self.node_in_area(node_id, area.area_id):
                return area
        return None

    def node_in_area(self, node_id: str, area_id: str) -> bool:
        """Return whether a node belongs to a specific interaction area."""

        area = self.areas[area_id]
        return (
            node_id in area.communication_zone_nodes
            or node_id in area.conflict_zone_nodes
            or node_id in area.waiting_points
            or node_id in area.conflict_points
        )

    def conflict_area_for_node(self, node_id: str) -> InteractionArea | None:
        """Return the area where this node is a conflict-zone node."""

        for area in self.areas.values():
            if node_id in area.conflict_zone_nodes or node_id in area.conflict_points:
                return area
        return None

    def is_waiting_point(self, node_id: str) -> bool:
        """Return whether the node is a Waiting Point."""

        return any(node_id in area.waiting_points for area in self.areas.values())

    def is_conflict_point(self, node_id: str) -> bool:
        """Return whether the node is a Conflict Point."""

        return any(node_id in area.conflict_points for area in self.areas.values())

    def summary(self) -> list[dict]:
        """Return serializable interaction-area debug data."""

        return [
            {
                "area_id": area.area_id,
                "area_type": area.area_type,
                "waiting_points": sorted(area.waiting_points),
                "conflict_points": sorted(area.conflict_points),
                "routes": sorted(area.allowed_routes),
                "priority_rule": area.priority_rule,
            }
            for area in self.areas.values()
        ]
