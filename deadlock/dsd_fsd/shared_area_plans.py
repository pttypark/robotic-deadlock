from deadlock.dsd_fsd.model import TransactionKind


def deadlock_plan(scenario, points):
    pattern = scenario.get("deadlock_pattern")
    if pattern == "shared_cross":
        return shared_cross_plan(points)
    if pattern == "shared_queue":
        return shared_queue_plan(points)
    if pattern == "corridor_pairs":
        return corridor_pair_deadlock_plan(points)
    if pattern == "hotspot_queue":
        return hotspot_queue_deadlock_plan(points)
    return None


def corridor_pair_deadlock_plan(points):
    aisles = _middle_aisles(points, count=2)
    top_row = 0
    bottom_row = points.height - 3
    starts = []
    targets = []
    for x in aisles:
        starts.extend(
            [
                (x, top_row, "DOWN"),
                (x, bottom_row, "UP"),
            ]
        )
        targets.extend(
            [
                (x, bottom_row),
                (x, top_row),
            ]
        )
    return {"starts": starts, "targets": targets}


def shared_cross_plan(points):
    center = points.conflict_zones[0].center
    left = (0, center[1])
    right = (points.width - 1, center[1])
    top = (center[0], 0)
    bottom = (center[0], points.height - 1)
    purpose_left = (center[0] - 5, center[1])
    purpose_right = (center[0] + 5, center[1])
    purpose_top = (center[0], center[1] - 5)
    purpose_bottom = (center[0], center[1] + 5)
    return {
        "starts": [
            (left[0], left[1], "RIGHT"),
            (right[0], right[1], "LEFT"),
            (top[0], top[1], "DOWN"),
            (bottom[0], bottom[1], "UP"),
        ],
        "targets": [purpose_right, purpose_left, purpose_bottom, purpose_top],
    }


def shared_queue_plan(points):
    center = points.conflict_zones[0].center
    left = (0, center[1])
    right = (points.width - 1, center[1])
    top = (center[0], 0)
    purpose_left = (center[0] - 5, center[1])
    purpose_right_1 = (center[0] + 5, center[1])
    purpose_right_2 = (center[0] + 4, center[1])
    purpose_bottom = (center[0], center[1] + 5)
    return {
        "starts": [
            (left[0], left[1], "RIGHT"),
            (left[0] + 1, left[1], "RIGHT"),
            (right[0], right[1], "LEFT"),
            (top[0], top[1], "DOWN"),
        ],
        "targets": [purpose_right_1, purpose_right_2, purpose_left, purpose_bottom],
    }


def hotspot_queue_deadlock_plan(points):
    x = _middle_aisles(points, count=1)[0]
    top_row = 0
    bottom_row = points.height - 3
    starts = [
        (x, top_row, "DOWN"),
        (x, top_row + 1, "DOWN"),
        (x, bottom_row, "UP"),
        (x, bottom_row - 1, "UP"),
    ]
    targets = [
        (x, bottom_row),
        (x, bottom_row - 1),
        (x, top_row),
        (x, top_row + 1),
    ]
    return {"starts": starts, "targets": targets}


def seed_deadlock_transactions(ledger, plan):
    for idx, target in enumerate(plan["targets"]):
        ledger.create(
            target=target,
            zone_id=idx,
            step=0,
            kind=TransactionKind.RETRIEVAL,
            buffer=target,
        )


def _middle_aisles(points, count):
    middle = len(points.aisle_columns) // 2
    start = max(0, middle - count // 2)
    aisles = points.aisle_columns[start:start + count]
    if len(aisles) < count:
        aisles = points.aisle_columns[:count]
    return aisles
