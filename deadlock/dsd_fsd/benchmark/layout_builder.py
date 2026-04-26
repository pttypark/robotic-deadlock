from __future__ import annotations

from dataclasses import dataclass

from deadlock.dsd_fsd.benchmark.config import ExperimentConfig

Cell = tuple[int, int]


@dataclass(frozen=True)
class BenchmarkLayout:
    """Deterministic warehouse layout shared by all policies.

    Args:
        layout: RWARE-style text layout. x=shelf/storage, .=road, g=workstation.
        width: Grid width.
        height: Grid height.
        aisle_columns: Vertical access aisle x-coordinates.
        transition_rows: Horizontal transition-road y-coordinates.
        storage_points: Bay access cells where tasks are generated.
        workstations: Buffer/workstation cells.
        decision_points: Aisle/transition intersections.
        waiting_points: Cells adjacent to decision points for short waits.
        escape_points: Explicit safe cells used by FSD trigger/recovery.
        traversable: All cells AMRs can occupy.
        zones: Storage points partitioned by dedicated zone.
        zone_allowed: Traversable cells each DSD zone may enter.
    """

    layout: str
    width: int
    height: int
    aisle_columns: tuple[int, ...]
    transition_rows: tuple[int, ...]
    storage_points: tuple[Cell, ...]
    workstations: tuple[Cell, ...]
    decision_points: frozenset[Cell]
    waiting_points: frozenset[Cell]
    escape_points: frozenset[Cell]
    traversable: frozenset[Cell]
    zones: tuple[frozenset[Cell], ...]
    zone_allowed: tuple[frozenset[Cell], ...]


def build_layout(config: ExperimentConfig) -> BenchmarkLayout:
    """Build the same deterministic base warehouse for all policies.

    Args:
        config: Experiment condition controlling size and aisle count.

    Returns:
        BenchmarkLayout containing RWARE-compatible layout text and policy points.
    """

    aisle_columns = tuple(2 + idx * 3 for idx in range(config.num_aisles))
    width = aisle_columns[-1] + 3
    buffer_row = config.bay_count + 1
    height = buffer_row + 2
    transition_rows = _transition_rows(config.bay_count)

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

    decision_points = frozenset((x, y) for x in aisle_columns for y in transition_rows)
    waiting_points = _waiting_points(aisle_columns, transition_rows, buffer_row)
    escape_points = _escape_points(aisle_columns, transition_rows, width, buffer_row)

    # FSD needs explicitly available adjacent cells around decision areas. These
    # cells are functional waiting/escape pockets, not an extra mandatory road gap.
    for x, y in waiting_points | escape_points:
        rows[y][x] = "."

    workstations = tuple((x, buffer_row) for x in aisle_columns)
    storage_points = _storage_points(aisle_columns, transition_rows, buffer_row)
    traversable = frozenset(
        (x, y)
        for y, row in enumerate(rows)
        for x, char in enumerate(row)
        if char in {".", "g"}
    )
    zones = _split_zones(storage_points, aisle_columns, config.num_robots)
    zone_allowed = _build_zone_allowed(zones, aisle_columns, traversable, width, config.num_robots)

    return BenchmarkLayout(
        layout="\n".join("".join(row) for row in rows),
        width=width,
        height=height,
        aisle_columns=aisle_columns,
        transition_rows=transition_rows,
        storage_points=tuple(storage_points),
        workstations=workstations,
        decision_points=decision_points,
        waiting_points=frozenset(point for point in waiting_points if point in traversable),
        escape_points=frozenset(point for point in escape_points if point in traversable),
        traversable=traversable,
        zones=zones,
        zone_allowed=zone_allowed,
    )


def zone_for_target(layout: BenchmarkLayout, target: Cell) -> int:
    """Return the deterministic DSD zone id for a storage target.

    Args:
        layout: Benchmark layout.
        target: Storage/bay access cell.

    Returns:
        Zone index containing the target, or the nearest aisle-derived zone.
    """

    for zone_id, zone in enumerate(layout.zones):
        if target in zone:
            return zone_id
    aisle_idx = nearest_aisle_index(layout.aisle_columns, target[0])
    return min(len(layout.zones) - 1, aisle_idx * len(layout.zones) // len(layout.aisle_columns))


def buffer_for_target(layout: BenchmarkLayout, target: Cell) -> Cell:
    """Select the nearest workstation/buffer for a target.

    Args:
        layout: Benchmark layout.
        target: Storage/bay access cell.

    Returns:
        Workstation cell nearest to the target.
    """

    return min(layout.workstations, key=lambda point: (manhattan(point, target), point[0], point[1]))


def nearest_aisle_index(aisle_columns: tuple[int, ...], x: int) -> int:
    """Find the nearest aisle index for an x-coordinate."""

    return min(range(len(aisle_columns)), key=lambda idx: (abs(aisle_columns[idx] - x), idx))


def manhattan(left: Cell, right: Cell) -> int:
    """Return Manhattan distance between two grid cells."""

    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _transition_rows(bay_count: int) -> tuple[int, ...]:
    first = max(2, bay_count // 3)
    second = max(first + 3, (2 * bay_count) // 3)
    return tuple(row for row in (first, second) if 0 < row < bay_count + 1)


def _storage_points(
    aisle_columns: tuple[int, ...],
    transition_rows: tuple[int, ...],
    buffer_row: int,
) -> tuple[Cell, ...]:
    transition_set = set(transition_rows)
    return tuple(
        (x, y)
        for x in aisle_columns
        for y in range(buffer_row)
        if y not in transition_set
    )


def _waiting_points(
    aisle_columns: tuple[int, ...],
    transition_rows: tuple[int, ...],
    buffer_row: int,
) -> frozenset[Cell]:
    return frozenset(
        (x, y + dy)
        for x in aisle_columns
        for y in transition_rows
        for dy in (-1, 1)
        if 0 <= y + dy < buffer_row
    )


def _escape_points(
    aisle_columns: tuple[int, ...],
    transition_rows: tuple[int, ...],
    width: int,
    buffer_row: int,
) -> frozenset[Cell]:
    points: set[Cell] = set()
    for x in aisle_columns:
        for y in transition_rows:
            for sx in (x - 1, x + 1):
                for sy in (y - 1, y, y + 1):
                    if 0 <= sx < width and 0 <= sy < buffer_row:
                        points.add((sx, sy))
    return frozenset(points)


def _split_zones(
    storage_points: tuple[Cell, ...],
    aisle_columns: tuple[int, ...],
    num_zones: int,
) -> tuple[frozenset[Cell], ...]:
    zones = [set() for _ in range(num_zones)]
    for point in storage_points:
        aisle_idx = nearest_aisle_index(aisle_columns, point[0])
        zone_id = min(num_zones - 1, aisle_idx * num_zones // len(aisle_columns))
        zones[zone_id].add(point)
    return tuple(frozenset(zone) for zone in zones)


def _build_zone_allowed(
    zones: tuple[frozenset[Cell], ...],
    aisle_columns: tuple[int, ...],
    traversable: frozenset[Cell],
    width: int,
    num_zones: int,
) -> tuple[frozenset[Cell], ...]:
    allowed_by_zone = []
    for zone_id, zone in enumerate(zones):
        zone_aisles = [
            aisle_columns[idx]
            for idx in range(len(aisle_columns))
            if min(num_zones - 1, idx * num_zones // len(aisle_columns)) == zone_id
        ]
        if zone_aisles:
            lo = max(0, min(zone_aisles) - 1)
            hi = min(width - 1, max(zone_aisles) + 1)
        elif zone:
            xs = [point[0] for point in zone]
            lo = max(0, min(xs) - 1)
            hi = min(width - 1, max(xs) + 1)
        else:
            lo = hi = min(width - 1, max(0, zone_id))
        allowed = {cell for cell in traversable if lo <= cell[0] <= hi}
        allowed.update(zone)
        allowed_by_zone.append(frozenset(allowed))
    return tuple(allowed_by_zone)
