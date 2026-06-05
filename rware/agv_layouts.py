"""AGV warehouse layouts with lane graphs and paper-style interaction areas."""

from __future__ import annotations

from dataclasses import dataclass, field

from rware.agv_topology import LaneGraph
from rware.agv_types import AGVAction, InteractionArea, Node, Route


GRID_SIZE = (17, 17)


@dataclass
class AGVLayout:
    """Complete AGV layout bundle used by wrapper and demo simulations.

    Args:
        name: Stable layout name.
        grid_size: Grid shape as (rows, cols).
        graph: Directed AGV lane graph.
        interaction_areas: Interaction areas in the layout.
        rware_layout: RWARE-compatible text layout.
        routes: Predefined routes available for assignment.
        initial_nodes: Default AGV start nodes.
        initial_headings: Default AGV headings by robot id.
        stations: Named station node ids.
        shelf_storage_cells: Shelf/storage cells around the lanes.

    Returns:
        AGVLayout instance.
    """

    name: str
    grid_size: tuple[int, int]
    graph: LaneGraph
    interaction_areas: list[InteractionArea]
    rware_layout: str
    routes: dict[str, Route]
    initial_nodes: dict[str, str]
    initial_headings: dict[str, str]
    stations: dict[str, str] = field(default_factory=dict)
    task_nodes: dict[str, str] = field(default_factory=dict)
    shelf_storage_cells: set[tuple[int, int]] = field(default_factory=set)

    @property
    def route_nodes(self) -> dict[str, list[str]]:
        """Return route id to node-sequence mapping for compatibility."""

        return {route_id: route.node_sequence for route_id, route in self.routes.items()}


def build_graph_rware_intersection_v1() -> AGVLayout:
    """Build the 17x17 AGV warehouse layout requested for foundation tests.

    Args:
        None.

    Returns:
        AGVLayout with a central 4-way intersection, merge, bottleneck, turn
        area, station nodes, shelf/storage cells, and RWARE layout text.
    """

    graph = LaneGraph()
    position_to_id: dict[tuple[int, int], str] = {}
    special = _special_nodes()
    routes = _build_routes()
    areas: list[InteractionArea] = []

    def ensure_node(position: tuple[int, int], default_type: str = "normal_lane") -> str:
        if position in position_to_id:
            return position_to_id[position]
        node_id, node_type, area_id = special.get(
            position,
            (f"N_{position[0]:02d}_{position[1]:02d}", default_type, None),
        )
        graph.add_node(
            Node(
                node_id=node_id,
                position=position,
                node_type=node_type,
                allowed_turns=_allowed_turns(node_type),
                interaction_area_id=area_id,
            )
        )
        position_to_id[position] = node_id
        return node_id

    traversable = _base_lane_cells()
    traversable.update(_bottleneck_cells())
    traversable.update(_merge_cells())
    traversable.update(_turn_cells())
    for route in routes.values():
        traversable.update(_positions_for_route(route))
    for position in sorted(traversable):
        ensure_node(position)

    for route in routes.values():
        _add_route_edges(graph, route)

    cross = _cross_interaction_area(routes)
    bottleneck = _bottleneck_interaction_area(routes)
    merge = _merge_interaction_area(routes)
    turn = _turn_interaction_area(routes)
    areas.extend([cross, bottleneck, merge, turn])

    rware_layout = _to_rware_layout(graph)
    return AGVLayout(
        name="graph_rware_intersection_v1",
        grid_size=GRID_SIZE,
        graph=graph,
        interaction_areas=areas,
        rware_layout=rware_layout,
        routes=routes,
        initial_nodes={
            "AGV_1": "STATION_NORTH",
            "AGV_2": "STATION_WEST",
            "AGV_3": "STATION_EAST",
            "AGV_4": "STATION_SOUTH",
        },
        initial_headings={
            "AGV_1": "SOUTH",
            "AGV_2": "EAST",
            "AGV_3": "WEST",
            "AGV_4": "NORTH",
        },
        stations={
            "NORTH": "STATION_NORTH",
            "SOUTH": "STATION_SOUTH",
            "WEST": "STATION_WEST",
            "EAST": "STATION_EAST",
        },
        shelf_storage_cells=_shelf_storage_cells(traversable),
    )


def build_four_way_intersection_layout() -> AGVLayout:
    """Compatibility alias for older scripts.

    Args:
        None.

    Returns:
        The 17x17 graph_rware_intersection_v1 layout.
    """

    return build_graph_rware_intersection_v1()


def build_fcfs_cross_shared_area_layout(
    corridor_length: int = 5,
    west_exit_extension: int = 0,
) -> AGVLayout:
    """Build the 12-AMR FCFS cross-shaped shared-area experiment layout.

    Args:
        corridor_length: Number of lane cells from each outer start to the
            shared 2x2 conflict zone. The historical layout uses 5.
        west_exit_extension: Extra outbound cells after the west exit. This
            creates an asymmetric downstream tail used by the showcase scenario.

    Returns:
        AGVLayout with four starts, four exits, one 2x2 conflict zone, and
        straight A* routes through the shared area.
    """

    if corridor_length < 5:
        raise ValueError("corridor_length must be at least 5")
    if west_exit_extension < 0:
        raise ValueError("west_exit_extension must be non-negative")

    grid_size = (
        2 * corridor_length + 3,
        2 * corridor_length + 3 + west_exit_extension,
    )
    graph = LaneGraph()
    roads = _fcfs_cross_road_cells(corridor_length, west_exit_extension)
    special = _fcfs_cross_special_nodes(corridor_length, west_exit_extension)

    for row, col in sorted(roads):
        node_id, node_type, area_id = special.get(
            (row, col),
            (f"X_{row:02d}_{col:02d}", "normal_lane", None),
        )
        graph.add_node(
            Node(
                node_id=node_id,
                position=(row, col),
                node_type=node_type,
                allowed_turns=_allowed_turns(node_type),
                interaction_area_id=area_id,
            )
        )

    for node in list(graph.nodes.values()):
        row, col = node.position
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            target_pos = (row + drow, col + dcol)
            if target_pos not in graph.position_to_node:
                continue
            target_id = graph.position_to_node[target_pos]
            if not _is_allowed_fcfs_conflict_edge(graph, node.node_id, target_id):
                continue
            edge_id = f"{node.node_id}->{target_id}"
            if edge_id not in graph.edges:
                graph.add_adjacent_edge(
                    node.node_id,
                    target_id,
                    edge_type="intersection"
                    if graph.nodes[target_id].node_type == "conflict"
                    else "lane",
                )

    routes = _fcfs_cross_routes(graph)
    area = InteractionArea(
        area_id="IA_CROSS_FCFS_001",
        area_type="intersection",
        communication_zone_nodes={"N_WAIT", "S_WAIT", "W_WAIT", "E_WAIT"},
        conflict_zone_nodes={"CP_NW", "CP_NE", "CP_SW", "CP_SE"},
        waiting_points={"N_WAIT", "S_WAIT", "W_WAIT", "E_WAIT"},
        conflict_points={"CP_NW", "CP_NE", "CP_SW", "CP_SE"},
        allowed_routes=routes,
        priority_rule="fcfs",
        metadata={
            "rule": "one_amr_in_shared_area",
            "decision_node": "waiting point immediately before conflict zone",
        },
    )

    return AGVLayout(
        name=_fcfs_cross_layout_name(corridor_length, west_exit_extension),
        grid_size=grid_size,
        graph=graph,
        interaction_areas=[area],
        rware_layout=_fcfs_cross_to_rware_layout(grid_size, roads, special),
        routes=routes,
        initial_nodes={
            "NORTH": "N_START",
            "SOUTH": "S_START",
            "WEST": "W_START",
            "EAST": "E_START",
        },
        initial_headings={
            "NORTH": "SOUTH",
            "SOUTH": "NORTH",
            "WEST": "EAST",
            "EAST": "WEST",
        },
        stations={
            "N_START": "N_START",
            "S_START": "S_START",
            "W_START": "W_START",
            "E_START": "E_START",
            "N_EXIT": "N_EXIT",
            "S_EXIT": "S_EXIT",
            "W_EXIT": "W_EXIT",
            "E_EXIT": "E_EXIT",
        },
    )


def build_fcfs_double_cross_shared_area_layout(corridor_length: int = 8) -> AGVLayout:
    """Build experiment layout v2 with two left-right symmetric FCFS crosses.

    Args:
        corridor_length: Number of lane cells from each outside edge to the
            nearest 2x2 conflict zone. The requested experiment uses 8.

    Returns:
        AGVLayout with the original single-cross geometry repeated as two
        mirrored 2x2 conflict zones connected by the shared horizontal aisle.
    """

    if corridor_length < 5:
        raise ValueError("corridor_length must be at least 5")

    grid_size = (2 * corridor_length + 3, 3 * corridor_length + 5)
    graph = LaneGraph()
    roads = _fcfs_double_cross_road_cells(corridor_length)
    special = _fcfs_double_cross_special_nodes(corridor_length)

    for row, col in sorted(roads):
        node_id, node_type, area_id = special.get(
            (row, col),
            (f"DX_{row:02d}_{col:02d}", "normal_lane", None),
        )
        graph.add_node(
            Node(
                node_id=node_id,
                position=(row, col),
                node_type=node_type,
                allowed_turns=_allowed_turns(node_type),
                interaction_area_id=area_id,
            )
        )

    for node in list(graph.nodes.values()):
        row, col = node.position
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            target_pos = (row + drow, col + dcol)
            if target_pos not in graph.position_to_node:
                continue
            target_id = graph.position_to_node[target_pos]
            if not _is_allowed_fcfs_double_cross_conflict_edge(node.node_id, target_id):
                continue
            edge_id = f"{node.node_id}->{target_id}"
            if edge_id not in graph.edges:
                graph.add_adjacent_edge(
                    node.node_id,
                    target_id,
                    edge_type="intersection"
                    if graph.nodes[target_id].node_type == "conflict"
                    else "lane",
                )

    routes = _fcfs_double_cross_routes(graph)
    areas = [
        _fcfs_double_cross_interaction_area("L", routes),
        _fcfs_double_cross_interaction_area("R", routes),
    ]

    return AGVLayout(
        name="fcfs_double_cross_shared_area_v2",
        grid_size=grid_size,
        graph=graph,
        interaction_areas=areas,
        rware_layout=_fcfs_cross_to_rware_layout(grid_size, roads, special),
        routes=routes,
        initial_nodes={
            "L_NORTH": "L_N_START",
            "L_SOUTH": "L_S_START",
            "WEST": "W_START",
            "R_NORTH": "R_N_START",
            "R_SOUTH": "R_S_START",
            "EAST": "E_START",
        },
        initial_headings={
            "L_NORTH": "SOUTH",
            "L_SOUTH": "NORTH",
            "WEST": "EAST",
            "R_NORTH": "SOUTH",
            "R_SOUTH": "NORTH",
            "EAST": "WEST",
        },
        stations={
            "L_N_START": "L_N_START",
            "L_N_EXIT": "L_N_EXIT",
            "L_S_START": "L_S_START",
            "L_S_EXIT": "L_S_EXIT",
            "W_START": "W_START",
            "W_EXIT": "W_EXIT",
            "R_N_START": "R_N_START",
            "R_N_EXIT": "R_N_EXIT",
            "R_S_START": "R_S_START",
            "R_S_EXIT": "R_S_EXIT",
            "E_START": "E_START",
            "E_EXIT": "E_EXIT",
        },
        shelf_storage_cells={
            (row, col)
            for row in range(grid_size[0])
            for col in range(grid_size[1])
            if (row, col) not in roads
        },
    )


def build_warehouse_aisle_layout_v1() -> AGVLayout:
    """Build a full warehouse-style AGV layout with A* task destinations.

    Args:
        None.

    Returns:
        AGVLayout with shelf blocks, aisle roads, bottom G stations, four
        interaction areas, and shelf-access task nodes.
    """

    grid_size = (19, 29)
    graph = LaneGraph()
    roads = _warehouse_road_cells()
    station_positions = {
        "STATION_1": (18, 2),
        "STATION_2": (18, 8),
        "STATION_3": (18, 14),
        "STATION_4": (18, 20),
    }
    task_positions = {
        "TASK_A": (2, 8),
        "TASK_B": (6, 23),
        "TASK_C": (10, 11),
        "TASK_D": (10, 5),
        "TASK_E": (14, 23),
        "TASK_F": (14, 11),
    }
    special_positions = _warehouse_special_positions()

    for row, col in sorted(roads):
        node_id = f"W_{row:02d}_{col:02d}"
        node_type = _warehouse_default_node_type(row, col)
        area_id = None
        for station_id, position in station_positions.items():
            if position == (row, col):
                node_id = station_id
                node_type = "station"
        for task_id, position in task_positions.items():
            if position == (row, col):
                node_id = task_id
                node_type = "shelf_access"
        if (row, col) in special_positions:
            node_id, node_type, area_id = special_positions[(row, col)]
        graph.add_node(
            Node(
                node_id=node_id,
                position=(row, col),
                node_type=node_type,
                allowed_turns=_allowed_turns(node_type),
                interaction_area_id=area_id,
                metadata={"is_task": node_id.startswith("TASK_")},
            )
        )

    for node in list(graph.nodes.values()):
        row, col = node.position
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            target_pos = (row + drow, col + dcol)
            if target_pos not in graph.position_to_node:
                continue
            target_id = graph.position_to_node[target_pos]
            edge_id = f"{node.node_id}->{target_id}"
            if edge_id not in graph.edges:
                graph.add_adjacent_edge(node.node_id, target_id, edge_type=_warehouse_edge_type(node, graph.nodes[target_id]))
    _enable_turns_at_junctions(graph)

    routes: dict[str, Route] = {}
    areas = [
        _warehouse_cross_area(),
        _warehouse_bottleneck_area(),
        _warehouse_merge_area(),
        _warehouse_turn_area(),
    ]
    return AGVLayout(
        name="warehouse_aisle_layout_v1",
        grid_size=grid_size,
        graph=graph,
        interaction_areas=areas,
        rware_layout=_warehouse_to_rware_layout(grid_size, roads, station_positions),
        routes=routes,
        initial_nodes={
            "AGV_1": "STATION_1",
            "AGV_2": "STATION_2",
            "AGV_3": "STATION_3",
            "AGV_4": "STATION_4",
        },
        initial_headings={
            "AGV_1": "NORTH",
            "AGV_2": "NORTH",
            "AGV_3": "NORTH",
            "AGV_4": "NORTH",
        },
        stations={key: key for key in station_positions},
        task_nodes={key: key for key in task_positions},
        shelf_storage_cells={
            (row, col)
            for row in range(grid_size[0])
            for col in range(grid_size[1])
            if (row, col) not in roads
        },
    )


def _special_nodes() -> dict[tuple[int, int], tuple[str, str, str | None]]:
    special = {
        (0, 8): ("STATION_NORTH", "station", None),
        (16, 8): ("STATION_SOUTH", "station", None),
        (8, 0): ("STATION_WEST", "station", None),
        (8, 16): ("STATION_EAST", "station", None),
        (4, 8): ("COMM_NORTH", "normal_lane", "IA_CROSS_001"),
        (12, 8): ("COMM_SOUTH", "normal_lane", "IA_CROSS_001"),
        (8, 4): ("COMM_WEST", "normal_lane", "IA_CROSS_001"),
        (8, 12): ("COMM_EAST", "normal_lane", "IA_CROSS_001"),
        (6, 8): ("WP_NORTH", "waiting", "IA_CROSS_001"),
        (10, 8): ("WP_SOUTH", "waiting", "IA_CROSS_001"),
        (8, 6): ("WP_WEST", "waiting", "IA_CROSS_001"),
        (8, 10): ("WP_EAST", "waiting", "IA_CROSS_001"),
        (7, 7): ("CP_1", "conflict", "IA_CROSS_001"),
        (7, 8): ("CP_2", "conflict", "IA_CROSS_001"),
        (7, 9): ("CP_3", "conflict", "IA_CROSS_001"),
        (8, 7): ("CP_4", "conflict", "IA_CROSS_001"),
        (8, 8): ("CP_5", "conflict", "IA_CROSS_001"),
        (8, 9): ("CP_6", "conflict", "IA_CROSS_001"),
        (9, 7): ("CP_7", "conflict", "IA_CROSS_001"),
        (9, 8): ("CP_8", "conflict", "IA_CROSS_001"),
        (9, 9): ("CP_9", "conflict", "IA_CROSS_001"),
        (4, 1): ("BN_COMM_WEST", "normal_lane", "IA_BOTTLENECK_001"),
        (4, 2): ("BN_WP_WEST", "waiting", "IA_BOTTLENECK_001"),
        (4, 3): ("BN_CP", "bottleneck", "IA_BOTTLENECK_001"),
        (4, 4): ("BN_WP_EAST", "waiting", "IA_BOTTLENECK_001"),
        (4, 5): ("BN_COMM_EAST", "normal_lane", "IA_BOTTLENECK_001"),
        (12, 2): ("MERGE_COMM_LEFT", "normal_lane", "IA_MERGE_001"),
        (12, 3): ("MERGE_WP_LEFT", "waiting", "IA_MERGE_001"),
        (14, 2): ("MERGE_COMM_RIGHT", "normal_lane", "IA_MERGE_001"),
        (14, 3): ("MERGE_WP_RIGHT", "waiting", "IA_MERGE_001"),
        (13, 4): ("MERGE_CP", "merge", "IA_MERGE_001"),
        (2, 12): ("TURN_COMM", "normal_lane", "IA_TURN_001"),
        (2, 13): ("TURN_WP", "waiting", "IA_TURN_001"),
        (2, 14): ("TURN_CP", "turn", "IA_TURN_001"),
    }
    return special


def _base_lane_cells() -> set[tuple[int, int]]:
    cells = {(row, 8) for row in range(17)}
    cells.update((8, col) for col in range(17))
    cells.update((row, col) for row in range(7, 10) for col in range(7, 10))
    return cells


def _bottleneck_cells() -> set[tuple[int, int]]:
    return {(4, col) for col in range(1, 6)}


def _merge_cells() -> set[tuple[int, int]]:
    return {
        (12, 2),
        (12, 3),
        (12, 4),
        (13, 4),
        (13, 5),
        (13, 6),
        (14, 2),
        (14, 3),
        (14, 4),
    }


def _turn_cells() -> set[tuple[int, int]]:
    return {(2, 12), (2, 13), (2, 14), (3, 14), (4, 14)}


def _build_routes() -> dict[str, Route]:
    route_specs = {
        "NORTH_TO_SOUTH": ("NORTH", "SOUTH", "straight", _north_to_south()),
        "NORTH_TO_EAST": ("NORTH", "EAST", "left_turn", _north_to_east()),
        "NORTH_TO_WEST": ("NORTH", "WEST", "right_turn", _north_to_west()),
        "SOUTH_TO_NORTH": ("SOUTH", "NORTH", "straight", _south_to_north()),
        "SOUTH_TO_EAST": ("SOUTH", "EAST", "right_turn", _south_to_east()),
        "SOUTH_TO_WEST": ("SOUTH", "WEST", "left_turn", _south_to_west()),
        "WEST_TO_EAST": ("WEST", "EAST", "straight", _west_to_east()),
        "WEST_TO_NORTH": ("WEST", "NORTH", "right_turn", _west_to_north()),
        "WEST_TO_SOUTH": ("WEST", "SOUTH", "left_turn", _west_to_south()),
        "EAST_TO_WEST": ("EAST", "WEST", "straight", _east_to_west()),
        "EAST_TO_NORTH": ("EAST", "NORTH", "left_turn", _east_to_north()),
        "EAST_TO_SOUTH": ("EAST", "SOUTH", "right_turn", _east_to_south()),
        "BOTTLENECK_WEST_TO_EAST": (
            "WEST",
            "EAST",
            "pass",
            [(4, col) for col in range(1, 6)],
        ),
        "BOTTLENECK_EAST_TO_WEST": (
            "EAST",
            "WEST",
            "pass",
            [(4, col) for col in range(5, 0, -1)],
        ),
        "LEFT_BRANCH_TO_MAIN": (
            "LEFT_BRANCH",
            "MAIN",
            "merge",
            [(12, 2), (12, 3), (12, 4), (13, 4), (13, 5), (13, 6)],
        ),
        "RIGHT_BRANCH_TO_MAIN": (
            "RIGHT_BRANCH",
            "MAIN",
            "merge",
            [(14, 2), (14, 3), (14, 4), (13, 4), (13, 5), (13, 6)],
        ),
        "TURN_LEFT_ROUTE": (
            "WEST",
            "SOUTH",
            "left_turn",
            [(2, 12), (2, 13), (2, 14), (3, 14), (4, 14)],
        ),
        "TURN_RIGHT_ROUTE": (
            "SOUTH",
            "WEST",
            "right_turn",
            [(4, 14), (3, 14), (2, 14), (2, 13), (2, 12)],
        ),
    }
    routes = {}
    for route_id, (entry, exit_, route_type, positions) in route_specs.items():
        node_sequence = [_node_id_for_position(position) for position in positions]
        routes[route_id] = Route(
            route_id=route_id,
            entry_direction=entry,
            exit_direction=exit_,
            route_type=route_type,
            node_sequence=node_sequence,
            conflict_points_on_route=[
                node_id for node_id in node_sequence if node_id.startswith("CP_") or node_id.endswith("_CP")
            ],
        )
    return routes


def _north_to_south() -> list[tuple[int, int]]:
    return [(row, 8) for row in range(0, 8)] + [(8, 8), (9, 8)] + [(row, 8) for row in range(10, 17)]


def _south_to_north() -> list[tuple[int, int]]:
    return list(reversed(_north_to_south()))


def _west_to_east() -> list[tuple[int, int]]:
    return [(8, col) for col in range(0, 8)] + [(8, 8), (8, 9)] + [(8, col) for col in range(10, 17)]


def _east_to_west() -> list[tuple[int, int]]:
    return list(reversed(_west_to_east()))


def _north_to_east() -> list[tuple[int, int]]:
    return [(row, 8) for row in range(0, 7)] + [(7, 8), (7, 9), (8, 9)] + [(8, col) for col in range(10, 17)]


def _north_to_west() -> list[tuple[int, int]]:
    return [(row, 8) for row in range(0, 7)] + [(7, 8), (7, 7), (8, 7)] + [(8, col) for col in range(6, -1, -1)]


def _south_to_east() -> list[tuple[int, int]]:
    return [(row, 8) for row in range(16, 9, -1)] + [(9, 8), (9, 9), (8, 9)] + [(8, col) for col in range(10, 17)]


def _south_to_west() -> list[tuple[int, int]]:
    return [(row, 8) for row in range(16, 9, -1)] + [(9, 8), (9, 7), (8, 7)] + [(8, col) for col in range(6, -1, -1)]


def _west_to_north() -> list[tuple[int, int]]:
    return [(8, col) for col in range(0, 7)] + [(8, 7), (7, 7), (7, 8)] + [(row, 8) for row in range(6, -1, -1)]


def _west_to_south() -> list[tuple[int, int]]:
    return [(8, col) for col in range(0, 7)] + [(8, 7), (9, 7), (9, 8)] + [(row, 8) for row in range(10, 17)]


def _east_to_north() -> list[tuple[int, int]]:
    return [(8, col) for col in range(16, 9, -1)] + [(8, 9), (7, 9), (7, 8)] + [(row, 8) for row in range(6, -1, -1)]


def _east_to_south() -> list[tuple[int, int]]:
    return [(8, col) for col in range(16, 9, -1)] + [(8, 9), (9, 9), (9, 8)] + [(row, 8) for row in range(10, 17)]


def _node_id_for_position(position: tuple[int, int]) -> str:
    return _special_nodes().get(position, (f"N_{position[0]:02d}_{position[1]:02d}", "", None))[0]


def _positions_for_route(route: Route) -> list[tuple[int, int]]:
    lookup = {node_id: position for position, (node_id, _, _) in _special_nodes().items()}
    positions = []
    for node_id in route.node_sequence:
        if node_id in lookup:
            positions.append(lookup[node_id])
        else:
            _, row, col = node_id.split("_")
            positions.append((int(row), int(col)))
    return positions


def _allowed_turns(node_type: str) -> list[str]:
    actions = [AGVAction.FORWARD.value, AGVAction.WAIT.value]
    if node_type in {"intersection", "turn", "waiting", "conflict", "shelf_access", "station"}:
        actions.extend([AGVAction.TURN_LEFT.value, AGVAction.TURN_RIGHT.value])
    return actions


def _add_route_edges(graph: LaneGraph, route: Route) -> None:
    for from_node, to_node in zip(route.node_sequence, route.node_sequence[1:]):
        if f"{from_node}->{to_node}" in graph.edges:
            continue
        edge_type = _edge_type(graph.nodes[from_node].node_type, graph.nodes[to_node].node_type)
        graph.add_adjacent_edge(from_node, to_node, edge_type=edge_type)


def _edge_type(left_type: str, right_type: str) -> str:
    for edge_type in ("bottleneck", "merge", "turn", "conflict"):
        if edge_type in {left_type, right_type}:
            return "intersection" if edge_type == "conflict" else edge_type
    if "station" in {left_type, right_type}:
        return "station_access"
    return "lane"


def _cross_interaction_area(routes: dict[str, Route]) -> InteractionArea:
    communication = {
        "N_03_08",
        "COMM_NORTH",
        "N_05_08",
        "WP_NORTH",
        "WP_SOUTH",
        "N_11_08",
        "COMM_SOUTH",
        "N_13_08",
        "N_08_03",
        "COMM_WEST",
        "N_08_05",
        "WP_WEST",
        "WP_EAST",
        "N_08_11",
        "COMM_EAST",
        "N_08_13",
    }
    conflict_points = {f"CP_{idx}" for idx in range(1, 10)}
    cross_routes = {
        route_id: route
        for route_id, route in routes.items()
        if route_id
        in {
            "NORTH_TO_SOUTH",
            "NORTH_TO_EAST",
            "NORTH_TO_WEST",
            "SOUTH_TO_NORTH",
            "SOUTH_TO_EAST",
            "SOUTH_TO_WEST",
            "WEST_TO_EAST",
            "WEST_TO_NORTH",
            "WEST_TO_SOUTH",
            "EAST_TO_WEST",
            "EAST_TO_NORTH",
            "EAST_TO_SOUTH",
        }
    }
    return InteractionArea(
        area_id="IA_CROSS_001",
        area_type="intersection",
        communication_zone_nodes=communication,
        conflict_zone_nodes=conflict_points,
        waiting_points={"WP_NORTH", "WP_SOUTH", "WP_WEST", "WP_EAST"},
        conflict_points=conflict_points,
        allowed_routes=cross_routes,
        metadata={"center": (8, 8), "conflict_zone_shape": "3x3"},
    )


def _bottleneck_interaction_area(routes: dict[str, Route]) -> InteractionArea:
    return InteractionArea(
        area_id="IA_BOTTLENECK_001",
        area_type="bottleneck",
        communication_zone_nodes={"BN_COMM_WEST", "BN_WP_WEST", "BN_WP_EAST", "BN_COMM_EAST"},
        conflict_zone_nodes={"BN_CP"},
        waiting_points={"BN_WP_WEST", "BN_WP_EAST"},
        conflict_points={"BN_CP"},
        allowed_routes={
            "BOTTLENECK_WEST_TO_EAST": routes["BOTTLENECK_WEST_TO_EAST"],
            "BOTTLENECK_EAST_TO_WEST": routes["BOTTLENECK_EAST_TO_WEST"],
        },
    )


def _merge_interaction_area(routes: dict[str, Route]) -> InteractionArea:
    return InteractionArea(
        area_id="IA_MERGE_001",
        area_type="merge",
        communication_zone_nodes={
            "MERGE_COMM_LEFT",
            "MERGE_WP_LEFT",
            "N_12_04",
            "MERGE_COMM_RIGHT",
            "MERGE_WP_RIGHT",
            "N_14_04",
        },
        conflict_zone_nodes={"MERGE_CP"},
        waiting_points={"MERGE_WP_LEFT", "MERGE_WP_RIGHT"},
        conflict_points={"MERGE_CP"},
        allowed_routes={
            "LEFT_BRANCH_TO_MAIN": routes["LEFT_BRANCH_TO_MAIN"],
            "RIGHT_BRANCH_TO_MAIN": routes["RIGHT_BRANCH_TO_MAIN"],
        },
    )


def _turn_interaction_area(routes: dict[str, Route]) -> InteractionArea:
    return InteractionArea(
        area_id="IA_TURN_001",
        area_type="turn",
        communication_zone_nodes={"TURN_COMM", "TURN_WP", "TURN_CP", "N_03_14"},
        conflict_zone_nodes={"TURN_CP"},
        waiting_points={"TURN_WP"},
        conflict_points={"TURN_CP"},
        allowed_routes={
            "TURN_LEFT_ROUTE": routes["TURN_LEFT_ROUTE"],
            "TURN_RIGHT_ROUTE": routes["TURN_RIGHT_ROUTE"],
        },
    )


def _to_rware_layout(graph: LaneGraph) -> str:
    rows = [["x" for _ in range(GRID_SIZE[1])] for _ in range(GRID_SIZE[0])]
    for node in graph.nodes.values():
        row, col = node.position
        rows[row][col] = "g" if node.node_type == "station" else "."
    return "\n".join("".join(row) for row in rows)


def _shelf_storage_cells(traversable: set[tuple[int, int]]) -> set[tuple[int, int]]:
    rows, cols = GRID_SIZE
    return {
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if (row, col) not in traversable
    }


def _warehouse_road_cells() -> set[tuple[int, int]]:
    roads: set[tuple[int, int]] = set()
    rows = {4, 8, 12, 16, 18}
    cols = {2, 5, 8, 11, 14, 17, 20, 23, 26}
    for row in rows:
        roads.update((row, col) for col in range(29))
    for col in cols:
        roads.update((row, col) for row in range(19))

    # Central interaction area and small local lane expansions.
    roads.update((row, col) for row in range(7, 10) for col in range(13, 16))
    roads.update((6, col) for col in range(18, 24))
    roads.update((row, 20) for row in range(4, 9))
    roads.update({(13, 3), (14, 4), (15, 5), (16, 6), (15, 7), (14, 8), (13, 9)})
    roads.update({(2, 22), (2, 23), (2, 24), (3, 24), (4, 24)})
    return roads


def _warehouse_default_node_type(row: int, col: int) -> str:
    horizontal_rows = {4, 8, 12, 16, 18}
    vertical_cols = {2, 5, 8, 11, 14, 17, 20, 23, 26}
    if row in horizontal_rows and col in vertical_cols:
        return "intersection"
    return "normal_lane"


def _warehouse_special_positions() -> dict[tuple[int, int], tuple[str, str, str | None]]:
    return {
        (4, 14): ("COMM_NORTH", "normal_lane", "IA_CROSS_001"),
        (12, 14): ("COMM_SOUTH", "normal_lane", "IA_CROSS_001"),
        (8, 8): ("COMM_WEST", "normal_lane", "IA_CROSS_001"),
        (8, 20): ("COMM_EAST", "normal_lane", "IA_CROSS_001"),
        (6, 14): ("WP_NORTH", "waiting", "IA_CROSS_001"),
        (10, 14): ("WP_SOUTH", "waiting", "IA_CROSS_001"),
        (8, 12): ("WP_WEST", "waiting", "IA_CROSS_001"),
        (8, 16): ("WP_EAST", "waiting", "IA_CROSS_001"),
        (7, 13): ("CP_1", "conflict", "IA_CROSS_001"),
        (7, 14): ("CP_2", "conflict", "IA_CROSS_001"),
        (7, 15): ("CP_3", "conflict", "IA_CROSS_001"),
        (8, 13): ("CP_4", "conflict", "IA_CROSS_001"),
        (8, 14): ("CP_5", "conflict", "IA_CROSS_001"),
        (8, 15): ("CP_6", "conflict", "IA_CROSS_001"),
        (9, 13): ("CP_7", "conflict", "IA_CROSS_001"),
        (9, 14): ("CP_8", "conflict", "IA_CROSS_001"),
        (9, 15): ("CP_9", "conflict", "IA_CROSS_001"),
        (6, 18): ("BN_COMM_WEST", "normal_lane", "IA_BOTTLENECK_001"),
        (6, 19): ("BN_WP_WEST", "waiting", "IA_BOTTLENECK_001"),
        (6, 20): ("BN_CP", "bottleneck", "IA_BOTTLENECK_001"),
        (6, 21): ("BN_WP_EAST", "waiting", "IA_BOTTLENECK_001"),
        (6, 22): ("BN_COMM_EAST", "normal_lane", "IA_BOTTLENECK_001"),
        (13, 3): ("MERGE_COMM_LEFT", "normal_lane", "IA_MERGE_001"),
        (14, 4): ("MERGE_WP_LEFT", "waiting", "IA_MERGE_001"),
        (13, 9): ("MERGE_COMM_RIGHT", "normal_lane", "IA_MERGE_001"),
        (14, 8): ("MERGE_WP_RIGHT", "waiting", "IA_MERGE_001"),
        (16, 6): ("MERGE_CP", "merge", "IA_MERGE_001"),
        (2, 22): ("TURN_COMM", "normal_lane", "IA_TURN_001"),
        (2, 23): ("TURN_WP", "waiting", "IA_TURN_001"),
        (2, 24): ("TURN_CP", "turn", "IA_TURN_001"),
    }


def _warehouse_edge_type(left: Node, right: Node) -> str:
    for edge_type in ("bottleneck", "merge", "turn", "conflict"):
        if edge_type in {left.node_type, right.node_type}:
            return "intersection" if edge_type == "conflict" else edge_type
    if "station" in {left.node_type, right.node_type}:
        return "station_access"
    return "lane"


def _enable_turns_at_junctions(graph: LaneGraph) -> None:
    for node in graph.nodes.values():
        directions = {edge.direction for edge in graph.outgoing_edges(node.node_id)}
        if len(directions) <= 1:
            continue
        if AGVAction.TURN_LEFT.value not in node.allowed_turns:
            node.allowed_turns.append(AGVAction.TURN_LEFT.value)
        if AGVAction.TURN_RIGHT.value not in node.allowed_turns:
            node.allowed_turns.append(AGVAction.TURN_RIGHT.value)
        if node.node_type == "normal_lane":
            node.node_type = "turn"


def _warehouse_cross_area() -> InteractionArea:
    communication = {
        "COMM_NORTH",
        "W_05_14",
        "WP_NORTH",
        "WP_SOUTH",
        "W_11_14",
        "COMM_SOUTH",
        "COMM_WEST",
        "W_08_09",
        "W_08_10",
        "W_08_11",
        "WP_WEST",
        "WP_EAST",
        "W_08_17",
        "W_08_18",
        "W_08_19",
        "COMM_EAST",
    }
    conflict_points = {f"CP_{idx}" for idx in range(1, 10)}
    return InteractionArea(
        area_id="IA_CROSS_001",
        area_type="intersection",
        communication_zone_nodes=communication,
        conflict_zone_nodes=conflict_points,
        waiting_points={"WP_NORTH", "WP_SOUTH", "WP_WEST", "WP_EAST"},
        conflict_points=conflict_points,
        allowed_routes={},
        metadata={"center": (8, 14), "conflict_zone_shape": "3x3"},
    )


def _warehouse_bottleneck_area() -> InteractionArea:
    return InteractionArea(
        area_id="IA_BOTTLENECK_001",
        area_type="bottleneck",
        communication_zone_nodes={"BN_COMM_WEST", "BN_WP_WEST", "BN_WP_EAST", "BN_COMM_EAST"},
        conflict_zone_nodes={"BN_CP"},
        waiting_points={"BN_WP_WEST", "BN_WP_EAST"},
        conflict_points={"BN_CP"},
        allowed_routes={},
    )


def _warehouse_merge_area() -> InteractionArea:
    return InteractionArea(
        area_id="IA_MERGE_001",
        area_type="merge",
        communication_zone_nodes={"MERGE_COMM_LEFT", "MERGE_WP_LEFT", "MERGE_COMM_RIGHT", "MERGE_WP_RIGHT"},
        conflict_zone_nodes={"MERGE_CP"},
        waiting_points={"MERGE_WP_LEFT", "MERGE_WP_RIGHT"},
        conflict_points={"MERGE_CP"},
        allowed_routes={},
    )


def _warehouse_turn_area() -> InteractionArea:
    return InteractionArea(
        area_id="IA_TURN_001",
        area_type="turn",
        communication_zone_nodes={"TURN_COMM", "TURN_WP", "TURN_CP", "W_03_24"},
        conflict_zone_nodes={"TURN_CP"},
        waiting_points={"TURN_WP"},
        conflict_points={"TURN_CP"},
        allowed_routes={},
    )


def _warehouse_to_rware_layout(
    grid_size: tuple[int, int],
    roads: set[tuple[int, int]],
    station_positions: dict[str, tuple[int, int]],
) -> str:
    station_set = set(station_positions.values())
    rows = [["x" for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    for row, col in roads:
        rows[row][col] = "."
    for row, col in station_set:
        rows[row][col] = "g"
    return "\n".join("".join(row) for row in rows)


def _fcfs_cross_layout_name(corridor_length: int, west_exit_extension: int) -> str:
    if corridor_length == 5 and west_exit_extension == 0:
        return "fcfs_cross_shared_area_v1"
    name = f"fcfs_cross_shared_area_corridor{corridor_length}"
    if west_exit_extension:
        name += f"_westtail{west_exit_extension}"
    return name


def _fcfs_cross_road_cells(
    corridor_length: int = 5,
    west_exit_extension: int = 0,
) -> set[tuple[int, int]]:
    rows = 2 * corridor_length + 3
    cols = rows + west_exit_extension
    center_row = corridor_length
    center_col = corridor_length + west_exit_extension
    roads = {
        (row, col)
        for row in range(rows)
        for col in (center_col, center_col + 1)
    }
    roads.update(
        (row, col)
        for row in (center_row, center_row + 1)
        for col in range(cols)
    )
    return roads


def _fcfs_cross_special_nodes(
    corridor_length: int = 5,
    west_exit_extension: int = 0,
) -> dict[tuple[int, int], tuple[str, str, str | None]]:
    area_id = "IA_CROSS_FCFS_001"
    rows = 2 * corridor_length + 3
    cols = rows + west_exit_extension
    last_row = rows - 1
    last_col = cols - 1
    center_row = corridor_length
    center_col = corridor_length + west_exit_extension
    return {
        (0, center_col): ("N_START", "station", None),
        (0, center_col + 1): ("N_EXIT", "station", None),
        (last_row, center_col + 1): ("S_START", "station", None),
        (last_row, center_col): ("S_EXIT", "station", None),
        (center_row + 1, west_exit_extension): ("W_START", "station", None),
        (center_row, 0): ("W_EXIT", "station", None),
        (center_row, last_col): ("E_START", "station", None),
        (center_row + 1, last_col): ("E_EXIT", "station", None),
        (center_row - 2, center_col): ("N_APPROACH", "normal_lane", area_id),
        (center_row - 1, center_col): ("N_WAIT", "waiting", area_id),
        (center_row + 4, center_col + 1): ("S_APPROACH", "normal_lane", area_id),
        (center_row + 2, center_col + 1): ("S_WAIT", "waiting", area_id),
        (center_row + 1, center_col - 2): ("W_APPROACH", "normal_lane", area_id),
        (center_row + 1, center_col - 1): ("W_WAIT", "waiting", area_id),
        (center_row, center_col + 4): ("E_APPROACH", "normal_lane", area_id),
        (center_row, center_col + 2): ("E_WAIT", "waiting", area_id),
        (center_row, center_col): ("CP_NW", "conflict", area_id),
        (center_row, center_col + 1): ("CP_NE", "conflict", area_id),
        (center_row + 1, center_col): ("CP_SW", "conflict", area_id),
        (center_row + 1, center_col + 1): ("CP_SE", "conflict", area_id),
    }


def _fcfs_cross_routes(graph: LaneGraph) -> dict[str, Route]:
    route_specs = {
        "NORTH_TO_SOUTH": ("NORTH", "SOUTH", "straight", "N_START", "S_EXIT"),
        "SOUTH_TO_NORTH": ("SOUTH", "NORTH", "straight", "S_START", "N_EXIT"),
        "WEST_TO_EAST": ("WEST", "EAST", "straight", "W_START", "E_EXIT"),
        "EAST_TO_WEST": ("EAST", "WEST", "straight", "E_START", "W_EXIT"),
    }
    routes = {}
    from rware.agv_path_planning import astar_path

    for route_id, (entry, exit_, route_type, start, goal) in route_specs.items():
        path = astar_path(graph, start, goal)
        routes[route_id] = Route(
            route_id=route_id,
            entry_direction=entry,
            exit_direction=exit_,
            route_type=route_type,
            node_sequence=path,
            conflict_points_on_route=[
                node_id for node_id in path if node_id.startswith("CP_")
            ],
        )
    return routes


def _is_allowed_fcfs_conflict_edge(graph: LaneGraph, from_node: str, to_node: str) -> bool:
    """Allow only right-hand circulation inside the FCFS 2x2 conflict zone."""

    conflict_cycle = {
        ("CP_NW", "CP_SW"),
        ("CP_SW", "CP_SE"),
        ("CP_SE", "CP_NE"),
        ("CP_NE", "CP_NW"),
    }
    from_is_conflict = from_node.startswith("CP_")
    to_is_conflict = to_node.startswith("CP_")
    if from_is_conflict and to_is_conflict:
        return (from_node, to_node) in conflict_cycle
    return True


def _fcfs_cross_to_rware_layout(
    grid_size: tuple[int, int],
    roads: set[tuple[int, int]],
    special: dict[tuple[int, int], tuple[str, str, str | None]],
) -> str:
    rows = [["x" for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    for row, col in roads:
        rows[row][col] = "."
    for (row, col), (_, node_type, _) in special.items():
        if node_type == "station":
            rows[row][col] = "g"
    return "\n".join("".join(row) for row in rows)


def _fcfs_double_cross_road_cells(corridor_length: int) -> set[tuple[int, int]]:
    rows = 2 * corridor_length + 3
    cols = 3 * corridor_length + 5
    center_row = corridor_length
    left_col = corridor_length
    right_col = cols - corridor_length - 2

    roads = {
        (row, col)
        for row in range(rows)
        for col in (left_col, left_col + 1, right_col, right_col + 1)
    }
    roads.update(
        (row, col)
        for row in (center_row, center_row + 1)
        for col in range(cols)
    )
    return roads


def _fcfs_double_cross_special_nodes(
    corridor_length: int,
) -> dict[tuple[int, int], tuple[str, str, str | None]]:
    rows = 2 * corridor_length + 3
    cols = 3 * corridor_length + 5
    center_row = corridor_length
    left_col = corridor_length
    right_col = cols - corridor_length - 2
    last_row = rows - 1
    last_col = cols - 1

    special: dict[tuple[int, int], tuple[str, str, str | None]] = {
        (0, left_col): ("L_N_START", "station", None),
        (0, left_col + 1): ("L_N_EXIT", "station", None),
        (last_row, left_col + 1): ("L_S_START", "station", None),
        (last_row, left_col): ("L_S_EXIT", "station", None),
        (center_row + 1, 0): ("W_START", "station", None),
        (center_row, 0): ("W_EXIT", "station", None),
        (0, right_col): ("R_N_START", "station", None),
        (0, right_col + 1): ("R_N_EXIT", "station", None),
        (last_row, right_col + 1): ("R_S_START", "station", None),
        (last_row, right_col): ("R_S_EXIT", "station", None),
        (center_row, last_col): ("E_START", "station", None),
        (center_row + 1, last_col): ("E_EXIT", "station", None),
    }
    special.update(
        _fcfs_double_cross_area_special_nodes(
            "L",
            "IA_DOUBLE_CROSS_LEFT",
            center_row,
            left_col,
        )
    )
    special.update(
        _fcfs_double_cross_area_special_nodes(
            "R",
            "IA_DOUBLE_CROSS_RIGHT",
            center_row,
            right_col,
        )
    )
    return special


def _fcfs_double_cross_area_special_nodes(
    prefix: str,
    area_id: str,
    center_row: int,
    center_col: int,
) -> dict[tuple[int, int], tuple[str, str, str | None]]:
    return {
        (center_row - 2, center_col): (f"{prefix}_N_APPROACH", "normal_lane", area_id),
        (center_row - 1, center_col): (f"{prefix}_N_WAIT", "waiting", area_id),
        (center_row + 4, center_col + 1): (f"{prefix}_S_APPROACH", "normal_lane", area_id),
        (center_row + 2, center_col + 1): (f"{prefix}_S_WAIT", "waiting", area_id),
        (center_row + 1, center_col - 2): (f"{prefix}_W_APPROACH", "normal_lane", area_id),
        (center_row + 1, center_col - 1): (f"{prefix}_W_WAIT", "waiting", area_id),
        (center_row, center_col + 4): (f"{prefix}_E_APPROACH", "normal_lane", area_id),
        (center_row, center_col + 2): (f"{prefix}_E_WAIT", "waiting", area_id),
        (center_row, center_col): (f"{prefix}_CP_NW", "conflict", area_id),
        (center_row, center_col + 1): (f"{prefix}_CP_NE", "conflict", area_id),
        (center_row + 1, center_col): (f"{prefix}_CP_SW", "conflict", area_id),
        (center_row + 1, center_col + 1): (f"{prefix}_CP_SE", "conflict", area_id),
    }


def _fcfs_double_cross_interaction_area(
    prefix: str,
    routes: dict[str, Route],
) -> InteractionArea:
    area_id = "IA_DOUBLE_CROSS_LEFT" if prefix == "L" else "IA_DOUBLE_CROSS_RIGHT"
    conflict_points = {
        f"{prefix}_CP_NW",
        f"{prefix}_CP_NE",
        f"{prefix}_CP_SW",
        f"{prefix}_CP_SE",
    }
    waiting_points = {
        f"{prefix}_N_WAIT",
        f"{prefix}_S_WAIT",
        f"{prefix}_W_WAIT",
        f"{prefix}_E_WAIT",
    }
    communication = waiting_points | {
        f"{prefix}_N_APPROACH",
        f"{prefix}_S_APPROACH",
        f"{prefix}_W_APPROACH",
        f"{prefix}_E_APPROACH",
    }
    area_routes = {
        route_id: route
        for route_id, route in routes.items()
        if any(node_id.startswith(f"{prefix}_CP_") for node_id in route.conflict_points_on_route)
    }
    return InteractionArea(
        area_id=area_id,
        area_type="intersection",
        communication_zone_nodes=communication,
        conflict_zone_nodes=conflict_points,
        waiting_points=waiting_points,
        conflict_points=conflict_points,
        allowed_routes=area_routes,
        priority_rule="fcfs",
        metadata={
            "rule": "one_amr_in_each_shared_area",
            "conflict_zone_shape": "2x2",
            "layout_role": "left_cross" if prefix == "L" else "right_cross",
        },
    )


def _fcfs_double_cross_routes(graph: LaneGraph) -> dict[str, Route]:
    route_specs = {
        "L_NORTH_TO_SOUTH": ("L_NORTH", "L_SOUTH", "straight", "L_N_START", "L_S_EXIT"),
        "L_SOUTH_TO_NORTH": ("L_SOUTH", "L_NORTH", "straight", "L_S_START", "L_N_EXIT"),
        "WEST_TO_EAST": ("WEST", "EAST", "straight", "W_START", "E_EXIT"),
        "R_NORTH_TO_SOUTH": ("R_NORTH", "R_SOUTH", "straight", "R_N_START", "R_S_EXIT"),
        "R_SOUTH_TO_NORTH": ("R_SOUTH", "R_NORTH", "straight", "R_S_START", "R_N_EXIT"),
        "EAST_TO_WEST": ("EAST", "WEST", "straight", "E_START", "W_EXIT"),
    }
    routes = {}
    from rware.agv_path_planning import astar_path

    for route_id, (entry, exit_, route_type, start, goal) in route_specs.items():
        path = astar_path(graph, start, goal)
        routes[route_id] = Route(
            route_id=route_id,
            entry_direction=entry,
            exit_direction=exit_,
            route_type=route_type,
            node_sequence=path,
            conflict_points_on_route=[
                node_id for node_id in path if "_CP_" in node_id
            ],
        )
    return routes


def _is_allowed_fcfs_double_cross_conflict_edge(from_node: str, to_node: str) -> bool:
    for prefix in ("L", "R"):
        conflict_cycle = {
            (f"{prefix}_CP_NW", f"{prefix}_CP_SW"),
            (f"{prefix}_CP_SW", f"{prefix}_CP_SE"),
            (f"{prefix}_CP_SE", f"{prefix}_CP_NE"),
            (f"{prefix}_CP_NE", f"{prefix}_CP_NW"),
        }
        from_is_conflict = from_node.startswith(f"{prefix}_CP_")
        to_is_conflict = to_node.startswith(f"{prefix}_CP_")
        if from_is_conflict and to_is_conflict:
            return (from_node, to_node) in conflict_cycle
    return True
