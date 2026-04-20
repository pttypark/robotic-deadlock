import heapq

from rware.warehouse import Direction

from deadlock.navigation import FORWARD, NOOP, TURN_LEFT, TURN_RIGHT


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def direction_to(current, nxt):
    dx = nxt[0] - current[0]
    dy = nxt[1] - current[1]
    if dx == 1:
        return Direction.RIGHT
    if dx == -1:
        return Direction.LEFT
    if dy == 1:
        return Direction.DOWN
    if dy == -1:
        return Direction.UP
    return None


def turn_action(current, desired):
    wrap = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    ci = wrap.index(current)
    di = wrap.index(desired)
    cw = (di - ci) % 4
    ccw = (ci - di) % 4
    return TURN_RIGHT if cw <= ccw else TURN_LEFT


def action_towards(agent, path):
    if len(path) < 2:
        return NOOP, (agent.x, agent.y)
    nxt = path[1]
    desired = direction_to((agent.x, agent.y), nxt)
    if desired is None:
        return NOOP, (agent.x, agent.y)
    if agent.dir != desired:
        return turn_action(agent.dir, desired), (agent.x, agent.y)
    return FORWARD, nxt


def find_path(env, start, goal, allowed=None, blocked=None, extra_cost=None):
    blocked = set(blocked or ())
    if start == goal:
        return [start]
    if goal in blocked:
        blocked.discard(goal)

    frontier = [(0, 0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for nxt in neighbors(env, current, allowed=allowed):
            if nxt in blocked:
                continue
            new_cost = cost_so_far[current] + 1
            if extra_cost:
                new_cost += extra_cost(nxt)
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + manhattan(nxt, goal)
                heapq.heappush(frontier, (priority, len(came_from), nxt))
                came_from[nxt] = current

    if goal not in came_from:
        return [start]

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def neighbors(env, pos, allowed=None):
    x, y = pos
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if not (0 <= nx < env.unwrapped.grid_size[1] and 0 <= ny < env.unwrapped.grid_size[0]):
            continue
        if allowed is not None and (nx, ny) not in allowed:
            continue
        if allowed is None and not env.unwrapped._is_highway(nx, ny):
            continue
        yield (nx, ny)

