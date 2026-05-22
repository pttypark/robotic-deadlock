"""State updates and event logging for AGV interaction-area membership."""

from __future__ import annotations

from rware.agv_types import AGVState, SimulationEvent
from rware.interaction_area import InteractionAreaRegistry


class InteractionAreaManager:
    """Update AGV communication, waiting, and conflict state."""

    def __init__(self, registry: InteractionAreaRegistry) -> None:
        """Create a manager from an interaction-area registry.

        Args:
            registry: InteractionAreaRegistry for the current layout.

        Returns:
            None.
        """

        self.registry = registry

    def update_agv_state(self, agv: AGVState, step: int) -> list[SimulationEvent]:
        """Update one AGV's area flags and return newly generated events.

        Args:
            agv: AGV state to update.
            step: Current simulation step.

        Returns:
            List of SimulationEvent records for newly reached states.
        """

        events: list[SimulationEvent] = []
        area = self.registry.area_for_node(agv.current_node)
        previous_area_id = agv.current_interaction_area_id

        if area is None:
            if previous_area_id is not None:
                events.append(
                    self._event(step, "interaction_area_exited", agv, previous_area_id)
                )
            agv.current_interaction_area_id = None
            agv.has_entered_communication_zone = False
            agv.is_waiting_at_waiting_point = False
            agv.has_entered_conflict_zone = False
            agv.occupied_conflict_point = None
            return events

        agv.current_interaction_area_id = area.area_id

        if (
            agv.current_node in area.communication_zone_nodes
            and not agv.has_entered_communication_zone
        ):
            agv.has_entered_communication_zone = True
            events.append(self._event(step, "communication_zone_entered", agv, area.area_id))

        is_waiting = agv.current_node in area.waiting_points
        if is_waiting and not agv.is_waiting_at_waiting_point:
            events.append(self._event(step, "waiting_point_reached", agv, area.area_id))
        agv.is_waiting_at_waiting_point = is_waiting

        if agv.current_node in area.conflict_zone_nodes and not agv.has_entered_conflict_zone:
            agv.has_entered_conflict_zone = True
            events.append(self._event(step, "conflict_zone_entered", agv, area.area_id))

        if agv.current_node in area.conflict_points:
            if agv.occupied_conflict_point != agv.current_node:
                events.append(
                    self._event(step, "conflict_point_occupied", agv, area.area_id)
                )
            agv.occupied_conflict_point = agv.current_node
        else:
            agv.occupied_conflict_point = None

        return events

    @staticmethod
    def _event(
        step: int,
        event_type: str,
        agv: AGVState,
        area_id: str,
    ) -> SimulationEvent:
        return SimulationEvent(
            step=step,
            event_type=event_type,
            robots=[agv.robot_id],
            interaction_area_id=area_id,
            node=agv.current_node,
            priorities={agv.robot_id: agv.priority},
            metadata={"route_id": agv.assigned_route_id},
        )
