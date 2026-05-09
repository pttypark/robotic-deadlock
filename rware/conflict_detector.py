"""Conflict-risk detection for AGV lane-graph movement proposals."""

from __future__ import annotations

from collections import defaultdict

from rware.agv_types import MoveProposal, SimulationEvent
from rware.interaction_area import InteractionAreaRegistry


class ConflictDetector:
    """Detect collision risks without resolving or optimizing them."""

    def __init__(self, registry: InteractionAreaRegistry) -> None:
        """Create a detector using interaction-area definitions.

        Args:
            registry: InteractionAreaRegistry for conflict point and zone lookup.

        Returns:
            None.
        """

        self.registry = registry

    def detect_conflicts(
        self,
        proposed_moves: dict[str, MoveProposal],
        step: int,
    ) -> list[SimulationEvent]:
        """Detect node, edge-swap, CP occupation, and conflict-zone entry risks.

        Args:
            proposed_moves: AGV별 from_node, to_node, priority, route_id 정보를 담은 dict.
            step: Current simulation step.

        Returns:
            List of detected conflict-risk events.
        """

        events: list[SimulationEvent] = []
        valid_moves = [proposal for proposal in proposed_moves.values() if proposal.valid]

        by_target: dict[str, list[MoveProposal]] = defaultdict(list)
        for proposal in valid_moves:
            if proposal.to_node != proposal.from_node:
                by_target[proposal.to_node].append(proposal)

        for node_id, proposals in by_target.items():
            if len(proposals) > 1:
                events.append(self._event(step, "node_collision", proposals, node=node_id))
                if self.registry.is_conflict_point(node_id):
                    area = self.registry.conflict_area_for_node(node_id)
                    events.append(
                        self._event(
                            step,
                            "conflict_point_occupation",
                            proposals,
                            node=node_id,
                            area_id=area.area_id if area else None,
                        )
                    )

        for idx, left in enumerate(valid_moves):
            for right in valid_moves[idx + 1 :]:
                if left.from_node == right.to_node and left.to_node == right.from_node:
                    events.append(
                        self._event(
                            step,
                            "edge_swap_collision",
                            [left, right],
                            edge=(left.from_node, left.to_node),
                        )
                    )

        by_conflict_area: dict[str, list[MoveProposal]] = defaultdict(list)
        for proposal in valid_moves:
            area = self.registry.conflict_area_for_node(proposal.to_node)
            if area is not None and proposal.from_node not in area.conflict_zone_nodes:
                by_conflict_area[area.area_id].append(proposal)
        for area_id, proposals in by_conflict_area.items():
            if len(proposals) > 1:
                events.append(
                    self._event(
                        step,
                        "conflict_zone_simultaneous_entry",
                        proposals,
                        area_id=area_id,
                    )
                )

        return events

    @staticmethod
    def _event(
        step: int,
        event_type: str,
        proposals: list[MoveProposal],
        node: str | None = None,
        edge: tuple[str, str] | None = None,
        area_id: str | None = None,
    ) -> SimulationEvent:
        robots = sorted(proposal.robot_id for proposal in proposals)
        priorities = {proposal.robot_id: proposal.priority for proposal in proposals}
        return SimulationEvent(
            step=step,
            event_type=event_type,
            robots=robots,
            interaction_area_id=area_id,
            node=node,
            edge=edge,
            priorities=priorities,
            metadata={
                "routes": {
                    proposal.robot_id: proposal.route_id for proposal in proposals
                }
            },
        )
