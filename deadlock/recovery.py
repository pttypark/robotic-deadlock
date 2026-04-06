class AdaptiveDeadlockRecovery:
    """Continuously reroute traffic away from recurring deadlock hotspots."""

    def __init__(
        self,
        manager,
        close_radius=0,
        penalty_radius=2,
        closure_ttl=16,
        penalty_ttl=28,
        wait_threshold=6,
        base_penalty=8,
    ):
        self.manager = manager
        self.close_radius = close_radius
        self.penalty_radius = penalty_radius
        self.closure_ttl = closure_ttl
        self.penalty_ttl = penalty_ttl
        self.wait_threshold = wait_threshold
        self.base_penalty = base_penalty
        self.blocked_until = {}
        self.penalty_until = {}
        self.heat = {}

    def refresh(self, step):
        self._prune(step)
        self.manager.set_adaptive_restrictions(
            blocked_cells=self.active_blocked_cells(step),
            penalty_cells=self.active_penalty_cells(step),
        )

    def register_deadlock(self, step, positions, wait_steps):
        hotspot = self._select_hotspot(positions, wait_steps)
        if hotspot is None:
            return None

        self.heat[hotspot] = self.heat.get(hotspot, 0) + 1
        severity = min(4, self.heat[hotspot])

        for cell in self._cells_within(hotspot, self.close_radius):
            self.blocked_until[cell] = max(
                self.blocked_until.get(cell, -1), step + self.closure_ttl + severity
            )

        for cell in self._cells_within(hotspot, self.penalty_radius):
            self.penalty_until[cell] = max(
                self.penalty_until.get(cell, (0, -1))[1],
                step + self.penalty_ttl + severity,
            )
            self.penalty_until[cell] = (
                self.base_penalty + severity * 2,
                self.penalty_until[cell],
            )

        self.refresh(step)
        return hotspot

    def active_blocked_cells(self, step):
        return {
            cell for cell, until in self.blocked_until.items() if until >= step
        }

    def active_penalty_cells(self, step):
        return {
            cell: penalty
            for cell, (penalty, until) in self.penalty_until.items()
            if until >= step
        }

    def _prune(self, step):
        self.blocked_until = {
            cell: until for cell, until in self.blocked_until.items() if until >= step
        }
        self.penalty_until = {
            cell: payload
            for cell, payload in self.penalty_until.items()
            if payload[1] >= step
        }

    def _select_hotspot(self, positions, wait_steps):
        candidates = []
        for idx, pos in enumerate(positions):
            wait = wait_steps[idx] if idx < len(wait_steps) else 0
            if wait < self.wait_threshold:
                continue
            hotspot = self._nearest_hotspot(pos)
            if hotspot is None:
                continue
            dist = abs(pos[0] - hotspot[0]) + abs(pos[1] - hotspot[1])
            candidates.append((wait, -dist, hotspot))

        if not candidates:
            for pos in positions:
                hotspot = self._nearest_hotspot(pos)
                if hotspot is not None:
                    candidates.append((0, 0, hotspot))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return candidates[0][2]

    def _nearest_hotspot(self, pos):
        best = None
        best_dist = 10**9
        for hotspot in self.manager.hotspots:
            dist = abs(pos[0] - hotspot[0]) + abs(pos[1] - hotspot[1])
            if dist < best_dist:
                best = hotspot
                best_dist = dist
        return best

    def _cells_within(self, center, radius):
        cells = set()
        for y in range(self.manager.env.unwrapped.grid_size[0]):
            for x in range(self.manager.env.unwrapped.grid_size[1]):
                if not self.manager.env.unwrapped._is_highway(x, y):
                    continue
                if abs(center[0] - x) + abs(center[1] - y) <= radius:
                    cells.add((x, y))
        return cells
