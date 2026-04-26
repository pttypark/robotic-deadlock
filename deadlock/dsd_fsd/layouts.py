from deadlock.dsd_fsd.model import ConflictZone, PointMap


BASELINE_LAYOUT = """
g...........g
x.x.x.x.x.x.x
x.x.x.x.x.x.x
g...........g
x.x.x.x.x.x.x
x.x.x.x.x.x.x
g...........g
"""


def build_baseline_points(n_agents=3):
    aisle_columns = [1, 3, 5, 7, 9, 11]
    transition_rows = [0, 3, 6]
    decision_points = {
        (x, y) for x in aisle_columns for y in transition_rows
    }
    waiting_points = {
        (x, y)
        for x in aisle_columns
        for y in (1, 2, 4, 5)
    }
    storage_points = sorted(waiting_points)
    buffer_points = [(0, y) for y in transition_rows] + [(12, y) for y in transition_rows]
    escape_points = set()
    conflict_zones = []

    for x in aisle_columns:
        for y in transition_rows:
            local_escape = set()
            for ex in (x - 1, x + 1):
                if 0 <= ex <= 12:
                    local_escape.add((ex, y))
                for ey in (y - 1, y + 1):
                    if 0 <= ex <= 12 and 0 <= ey <= 6:
                        local_escape.add((ex, ey))
            escape_points.update(local_escape)
            cells = {(x, y)}
            cells.update((x + dx, y) for dx in (-1, 1) if 0 <= x + dx <= 12)
            cells.update((x, y + dy) for dy in (-1, 1) if 0 <= y + dy <= 6)
            nearby_waiting = {
                point
                for point in waiting_points
                if abs(point[0] - x) + abs(point[1] - y) <= 2
            }
            conflict_zones.append(
                ConflictZone(
                    id=f"cz_{x}_{y}",
                    center=(x, y),
                    cells=cells,
                    decision_points={(x, y)},
                    waiting_points=nearby_waiting,
                    escape_points=local_escape,
                )
            )

    zones = _split_zones(storage_points, aisle_columns, n_agents)
    return PointMap(
        layout=BASELINE_LAYOUT,
        aisle_columns=aisle_columns,
        transition_rows=transition_rows,
        storage_points=storage_points,
        buffer_points=buffer_points,
        decision_points=decision_points,
        waiting_points=waiting_points,
        escape_points=escape_points,
        conflict_zones=conflict_zones,
        zones=zones,
    )


def build_paper_points(system="fsd", n_agents=4, aisles=8, bay_rows=18):
    if system == "dsd":
        return _build_paper_dsd_points(n_agents=n_agents, aisles=aisles, bay_rows=bay_rows)
    return _build_paper_fsd_points(n_agents=n_agents, aisles=aisles, bay_rows=bay_rows)


def build_shared_area_points(n_agents=4):
    width = 21
    height = 17
    center = (10, 8)
    rows = [["x" for _ in range(width)] for _ in range(height)]

    for y in range(height):
        rows[y][center[0]] = "."
    for x in range(width):
        rows[center[1]][x] = "."
    for y in range(center[1] - 2, center[1] + 3):
        for x in range(center[0] - 5, center[0] + 6):
            rows[y][x] = "."
    for x, y in (
        (center[0] - 5, center[1] - 1),
        (center[0] - 5, center[1] + 1),
        (center[0] + 5, center[1] - 1),
        (center[0] + 5, center[1] + 1),
        (center[0] - 1, center[1] - 5),
        (center[0] + 1, center[1] - 5),
        (center[0] - 1, center[1] + 5),
        (center[0] + 1, center[1] + 5),
    ):
        rows[y][x] = "."

    for x, y in ((center[0], 0), (center[0], height - 1), (0, center[1]), (width - 1, center[1])):
        rows[y][x] = "g"

    core = {
        (x, y)
        for y in range(center[1] - 2, center[1] + 3)
        for x in range(center[0] - 5, center[0] + 6)
    }
    decision_points = {
        (center[0], center[1] - 2),
        (center[0], center[1] + 2),
        (center[0] - 2, center[1]),
        (center[0] + 2, center[1]),
    }
    waiting_points = {
        (center[0], center[1] - 4),
        (center[0], center[1] + 4),
        (center[0] - 4, center[1]),
        (center[0] + 4, center[1]),
    }
    escape_points = {
        (center[0] - 1, center[1] - 1),
        (center[0] + 1, center[1] - 1),
        (center[0] - 1, center[1] + 1),
        (center[0] + 1, center[1] + 1),
    }
    conflict_zones = [
        ConflictZone(
            id="shared_core",
            center=center,
            cells=core,
            decision_points=decision_points,
            waiting_points=waiting_points,
            escape_points=escape_points,
        )
    ]
    storage_points = sorted(decision_points | waiting_points | escape_points)
    zones = _split_zones(storage_points, [center[0]], n_agents)
    return PointMap(
        layout=_rows_to_layout(rows),
        aisle_columns=[center[0]],
        transition_rows=[center[1]],
        storage_points=storage_points,
        buffer_points=[(center[0], 0), (center[0], height - 1), (0, center[1]), (width - 1, center[1])],
        decision_points=decision_points,
        waiting_points=waiting_points,
        escape_points=escape_points,
        conflict_zones=conflict_zones,
        zones=zones,
    )


def _build_paper_fsd_points(n_agents=4, aisles=8, bay_rows=18):
    aisle_columns = [2 + idx * 3 for idx in range(aisles)]
    width = aisle_columns[-1] + 3
    transition_rows = [3, 8, 13]
    buffer_row = bay_rows + 1
    height = buffer_row + 2
    rows = [["x" for _ in range(width)] for _ in range(height)]

    for x in aisle_columns:
        for y in range(height):
            rows[y][x] = "."
    for y in transition_rows:
        for x in range(width):
            rows[y][x] = "."
    for x in range(width):
        rows[buffer_row][x] = "."
    for x in aisle_columns:
        rows[buffer_row][x] = "g"

    storage_points = _storage_points(aisle_columns, transition_rows, buffer_row, width)
    buffer_points = [(x, buffer_row) for x in aisle_columns]
    decision_points = {(x, y) for x in aisle_columns for y in transition_rows}
    waiting_points = {
        (x, y + dy)
        for x in aisle_columns
        for y in transition_rows
        for dy in (-1, 1)
        if 0 <= y + dy < buffer_row
    }
    escape_points = _escape_points(aisle_columns, transition_rows, width, height)
    conflict_zones = _conflict_zones(
        aisle_columns,
        transition_rows,
        width,
        height,
        waiting_points,
        escape_points,
    )
    zones = _split_zones(storage_points, aisle_columns, n_agents)
    return PointMap(
        layout=_rows_to_layout(rows),
        aisle_columns=aisle_columns,
        transition_rows=transition_rows,
        storage_points=storage_points,
        buffer_points=buffer_points,
        decision_points=decision_points,
        waiting_points=waiting_points,
        escape_points=escape_points,
        conflict_zones=conflict_zones,
        zones=zones,
    )


def _build_paper_dsd_points(n_agents=4, aisles=8, bay_rows=18):
    aisle_columns = [2 + idx * 3 for idx in range(aisles)]
    width = aisle_columns[-1] + 3
    buffer_row = bay_rows + 1
    transition_rows = [buffer_row]
    height = buffer_row + 2
    rows = [["x" for _ in range(width)] for _ in range(height)]

    for x in aisle_columns:
        for y in range(height):
            rows[y][x] = "."
    for x in range(width):
        rows[buffer_row][x] = "."
    for x in aisle_columns:
        rows[buffer_row][x] = "g"

    storage_points = _storage_points(aisle_columns, (), buffer_row, width)
    buffer_points = [(x, buffer_row) for x in aisle_columns]
    decision_points = {(x, buffer_row) for x in aisle_columns}
    waiting_points = {
        (x, buffer_row - 1)
        for x in aisle_columns
        if buffer_row - 1 >= 0
    }
    escape_points = set(waiting_points)
    conflict_zones = _conflict_zones(
        aisle_columns,
        transition_rows,
        width,
        height,
        waiting_points,
        escape_points,
    )
    zones = _split_zones(storage_points, aisle_columns, n_agents)
    return PointMap(
        layout=_rows_to_layout(rows),
        aisle_columns=aisle_columns,
        transition_rows=transition_rows,
        storage_points=storage_points,
        buffer_points=buffer_points,
        decision_points=decision_points,
        waiting_points=waiting_points,
        escape_points=escape_points,
        conflict_zones=conflict_zones,
        zones=zones,
    )


def zone_for_target(points, target):
    for idx, zone in enumerate(points.zones):
        if target in zone:
            return idx
    return 0


def _split_zones(storage_points, aisle_columns, n_zones):
    zones = [set() for _ in range(n_zones)]
    for point in storage_points:
        aisle_idx = min(
            range(len(aisle_columns)),
            key=lambda idx: abs(aisle_columns[idx] - point[0]),
        )
        zone_idx = min(n_zones - 1, aisle_idx * n_zones // len(aisle_columns))
        zones[zone_idx].add(point)
    return zones


def buffer_for_target(points, target):
    return min(points.buffer_points, key=lambda point: abs(point[0] - target[0]) + abs(point[1] - target[1]))


def _storage_points(aisle_columns, transition_rows, buffer_row, width):
    transition_rows = set(transition_rows)
    points = []
    for x in aisle_columns:
        for y in range(buffer_row):
            if y in transition_rows:
                continue
            points.append((x, y))
    return sorted(set(points))


def _escape_points(aisle_columns, transition_rows, width, height):
    points = set()
    for x in aisle_columns:
        for y in transition_rows:
            for sx in (x - 1, x + 1):
                for sy in (y - 1, y, y + 1):
                    if 0 <= sx < width and 0 <= sy < height:
                        points.add((sx, sy))
    return points


def _conflict_zones(aisle_columns, transition_rows, width, height, waiting_points, escape_points):
    zones = []
    for x in aisle_columns:
        for y in transition_rows:
            cells = {(x, y)}
            cells.update((x + dx, y) for dx in (-1, 1) if 0 <= x + dx < width)
            cells.update((x, y + dy) for dy in (-1, 1) if 0 <= y + dy < height)
            nearby_waiting = {
                point
                for point in waiting_points
                if abs(point[0] - x) + abs(point[1] - y) <= 2
            }
            nearby_escape = {
                point
                for point in escape_points
                if abs(point[0] - x) + abs(point[1] - y) <= 2
            }
            zones.append(
                ConflictZone(
                    id=f"cz_{x}_{y}",
                    center=(x, y),
                    cells=cells,
                    decision_points={(x, y)},
                    waiting_points=nearby_waiting,
                    escape_points=nearby_escape,
                )
            )
    return zones


def _rows_to_layout(rows):
    return "\n".join("".join(row) for row in rows)
