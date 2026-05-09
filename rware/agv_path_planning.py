"""A* path planning on AGV lane graphs."""

from __future__ import annotations

import heapq

from rware.agv_topology import LaneGraph


def astar_path(
    graph: LaneGraph,
    start_node: str,
    goal_node: str,
    blocked_nodes: set[str] | None = None,
) -> list[str]:
    """Plan a shortest node path using A* over the lane graph.

    Args:
        graph: LaneGraph containing nodes and directed edges.
        start_node: Start node id.
        goal_node: Goal node id.
        blocked_nodes: Optional node ids to avoid.

    Returns:
        Ordered node id path from start to goal. Empty list if no path exists.
    """

    blocked = blocked_nodes or set()
    if start_node in blocked or goal_node in blocked:
        return []
    frontier: list[tuple[float, int, str]] = []
    heapq.heappush(frontier, (0.0, 0, start_node))
    came_from: dict[str, str | None] = {start_node: None}
    cost_so_far: dict[str, float] = {start_node: 0.0}
    counter = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal_node:
            return _reconstruct_path(came_from, goal_node)

        for edge in graph.outgoing_edges(current):
            if edge.to_node in blocked:
                continue
            new_cost = cost_so_far[current] + edge.distance
            if edge.to_node not in cost_so_far or new_cost < cost_so_far[edge.to_node]:
                cost_so_far[edge.to_node] = new_cost
                counter += 1
                priority = new_cost + _manhattan(graph, edge.to_node, goal_node)
                heapq.heappush(frontier, (priority, counter, edge.to_node))
                came_from[edge.to_node] = current

    return []


def _reconstruct_path(came_from: dict[str, str | None], goal_node: str) -> list[str]:
    path = [goal_node]
    current = goal_node
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _manhattan(graph: LaneGraph, left_node: str, right_node: str) -> int:
    left = graph.nodes[left_node].position
    right = graph.nodes[right_node].position
    return abs(left[0] - right[0]) + abs(left[1] - right[1])
