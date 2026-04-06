import heapq

from rware.warehouse import Direction

from deadlock.navigation import FORWARD, NOOP, TURN_LEFT, TURN_RIGHT


class GlobalTrafficManager:
    """Human-aware centralized traffic manager for multi-robot routing."""

    def __init__(
        self,
        env,
        route_library,
        hotspot_radius=1,
        hotspot_backpressure=1,
        corridor_penalty=2,
        human_penalty=6,
    ):
        self.env = env
        self.route_library = [list(route) for route in route_library]
        self.hotspot_radius = hotspot_radius
        self.hotspot_backpressure = hotspot_backpressure
        self.corridor_penalty = corridor_penalty
        self.human_penalty = human_penalty
        self.route_progress = [0 for _ in env.unwrapped.agents]
        self.wait_steps = [0 for _ in env.unwrapped.agents]
        self.reservation_state = {}
        self.hotspots = self._extract_hotspots()
        self.corridor_rows, self.spine_cols = self._extract_corridors()
        self.adaptive_blocked_cells = set()
        self.adaptive_penalty_cells = {}

    def update_wait_steps(self, wait_steps):
        self.wait_steps = list(wait_steps)

    def set_adaptive_restrictions(self, blocked_cells=None, penalty_cells=None):
        self.adaptive_blocked_cells = set(blocked_cells or [])
        self.adaptive_penalty_cells = dict(penalty_cells or {})

    def compute_actions(self):
        agents = self.env.unwrapped.agents
        actions = [NOOP] * len(agents)
        proposals = []
        hotspot_load = {}
        occupied_now = {(agent.x, agent.y) for agent in agents}
        corridor_load = self._build_corridor_load()

        for idx, agent in enumerate(agents):
            route = self.route_library[idx % len(self.route_library)]
            target, self.route_progress[idx] = self._next_waypoint(
                agent, route, self.route_progress[idx]
            )
            if (agent.x, agent.y) == target:
                actions[idx] = NOOP
                continue

            path = self._find_path(
                start=(agent.x, agent.y),
                goal=target,
                occupied=occupied_now,
                corridor_load=corridor_load,
            )
            if len(path) < 2:
                actions[idx] = NOOP
                continue

            next_cell = path[1]
            desired = self._direction_to((agent.x, agent.y), next_cell)
            if desired is None:
                actions[idx] = NOOP
                continue
            if agent.dir != desired:
                actions[idx] = self._turn_action(agent.dir, desired)
                continue

            hotspot_load[next_cell] = hotspot_load.get(next_cell, 0) + 1
            proposals.append(
                {
                    "idx": idx,
                    "agent": agent,
                    "target": next_cell,
                    "path": path,
                }
            )

        actions = self._approve_proposals(actions, proposals, hotspot_load, occupied_now)
        self._cleanup_reservations()
        return actions

    def _approve_proposals(self, actions, proposals, hotspot_load, occupied_now):
        humans = {(human.x, human.y) for human in self.env.unwrapped.humans}
        blocked = set(self.env.unwrapped.dynamic_blocked_cells).union(
            self.env.unwrapped.static_blocked_cells
        )
        blocked.update(self.adaptive_blocked_cells)
        claimed_targets = set()
        zone_claims = set()
        approved = set()

        proposals.sort(
            key=lambda item: self._proposal_priority(item["idx"], item["agent"], hotspot_load),
            reverse=True,
        )

        for proposal in proposals:
            idx = proposal["idx"]
            target = proposal["target"]

            if target in claimed_targets or target in humans or target in blocked:
                continue
            if target in occupied_now:
                continue

            zone = self._reservation_zone(target)
            if zone is not None:
                owner = self.reservation_state.get(zone)
                if owner is not None and owner != idx:
                    continue
                if zone in zone_claims:
                    continue
                zone_path_cells = {
                    cell
                    for cell in proposal["path"][: self.hotspot_backpressure + 2]
                    if self._reservation_zone(cell) == zone
                }
                if any(cell in claimed_targets for cell in zone_path_cells):
                    continue
                self.reservation_state[zone] = idx
                zone_claims.add(zone)

            approved.add(idx)
            claimed_targets.add(target)

        for idx in approved:
            actions[idx] = FORWARD
        return actions

    def _cleanup_reservations(self):
        agents = self.env.unwrapped.agents
        for zone, owner in list(self.reservation_state.items()):
            if owner >= len(agents):
                self.reservation_state.pop(zone, None)
                continue
            agent = agents[owner]
            if self._reservation_zone((agent.x, agent.y)) != zone:
                self.reservation_state.pop(zone, None)

    def _proposal_priority(self, idx, agent, hotspot_load):
        return (
            self.wait_steps[idx] if idx < len(self.wait_steps) else 0,
            1 if (agent.x, agent.y) in self.hotspots else 0,
            hotspot_load.get((agent.x, agent.y), 0),
            -idx,
        )

    def _next_waypoint(self, agent, route, route_progress):
        target = route[route_progress % len(route)]
        if (agent.x, agent.y) == target:
            route_progress = (route_progress + 1) % len(route)
            target = route[route_progress]
        return target, route_progress

    def _find_path(self, start, goal, occupied, corridor_load):
        blocked = set(self.env.unwrapped.static_blocked_cells).union(
            self.env.unwrapped.dynamic_blocked_cells
        )
        blocked.update(self.adaptive_blocked_cells)
        human_danger = self._human_danger_cells()
        forbidden = blocked.union(pos for pos in occupied if pos != start and pos != goal)
        if goal in forbidden:
            forbidden.discard(goal)

        frontier = []
        heapq.heappush(frontier, (0, 0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for nxt in self._neighbors(current):
                if nxt in forbidden:
                    continue
                new_cost = cost_so_far[current] + 1
                new_cost += self._corridor_penalty(nxt, corridor_load)
                new_cost += self.adaptive_penalty_cells.get(nxt, 0)
                if nxt in human_danger and nxt != goal:
                    new_cost += self.human_penalty
                zone = self._reservation_zone(nxt)
                if zone is not None:
                    new_cost += 2
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self._heuristic(nxt, goal)
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

    def _extract_hotspots(self):
        hotspots = set()
        for y in range(self.env.unwrapped.grid_size[0]):
            for x in range(self.env.unwrapped.grid_size[1]):
                if not self.env.unwrapped._is_highway(x, y):
                    continue
                degree = sum(1 for _ in self._neighbors((x, y)))
                if degree >= 3:
                    hotspots.add((x, y))
        return hotspots

    def _extract_corridors(self):
        row_counts = {}
        col_counts = {}
        for y in range(self.env.unwrapped.grid_size[0]):
            count = 0
            for x in range(self.env.unwrapped.grid_size[1]):
                if self.env.unwrapped._is_highway(x, y):
                    count += 1
            if count >= 4:
                row_counts[y] = count
        for x in range(self.env.unwrapped.grid_size[1]):
            count = 0
            for y in range(self.env.unwrapped.grid_size[0]):
                if self.env.unwrapped._is_highway(x, y):
                    count += 1
            if count >= 4:
                col_counts[x] = count
        return tuple(row_counts), tuple(col_counts)

    def _build_corridor_load(self):
        load = {}
        for idx, agent in enumerate(self.env.unwrapped.agents):
            route = self.route_library[idx % len(self.route_library)]
            target = route[self.route_progress[idx] % len(route)]
            if agent.y != target[1] and agent.x in self.spine_cols:
                load[("col", agent.x)] = load.get(("col", agent.x), 0) + 1
            if agent.x != target[0] and agent.y in self.corridor_rows:
                load[("row", agent.y)] = load.get(("row", agent.y), 0) + 1
        return load

    def _corridor_penalty(self, cell, corridor_load):
        x, y = cell
        penalty = 0
        if y in self.corridor_rows:
            penalty += corridor_load.get(("row", y), 0) * self.corridor_penalty
        if x in self.spine_cols:
            penalty += corridor_load.get(("col", x), 0)
        return penalty

    def _human_danger_cells(self):
        cells = set()
        for human in self.env.unwrapped.humans:
            for y in range(self.env.unwrapped.grid_size[0]):
                for x in range(self.env.unwrapped.grid_size[1]):
                    if (
                        abs(human.x - x) + abs(human.y - y)
                        <= self.env.unwrapped.human_safety_radius
                    ):
                        cells.add((x, y))
        return cells

    def _reservation_zone(self, cell):
        best = None
        best_dist = 10**9
        for hotspot in self.hotspots:
            dist = abs(cell[0] - hotspot[0]) + abs(cell[1] - hotspot[1])
            if dist < best_dist:
                best = hotspot
                best_dist = dist
        if best is None or best_dist > self.hotspot_radius:
            return None
        return best

    def _neighbors(self, pos):
        x, y = pos
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < self.env.unwrapped.grid_size[1] and 0 <= ny < self.env.unwrapped.grid_size[0]:
                if self.env.unwrapped._is_highway(nx, ny):
                    yield (nx, ny)

    @staticmethod
    def _heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _direction_to(current, nxt):
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

    @staticmethod
    def _turn_action(current, desired):
        wrap = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        ci = wrap.index(current)
        di = wrap.index(desired)
        cw = (di - ci) % 4
        ccw = (ci - di) % 4
        return TURN_RIGHT if cw <= ccw else TURN_LEFT
