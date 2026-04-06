import heapq
from collections import defaultdict

import numpy as np
from rware.warehouse import Direction

from deadlock.navigation import FORWARD, NOOP, TURN_LEFT, TURN_RIGHT


class GraphNeuralTrafficManager:
    """GNN-style message-passing coordinator for AMR and human-aware traffic."""

    def __init__(
        self,
        env,
        route_library,
        hotspots,
        corridor_rows,
        spine_cols,
        message_passing_steps=3,
        interaction_radius=5,
        hotspot_radius=1,
        human_penalty=10.0,
        human_prediction_horizon=5,
        predicted_human_penalty=6.0,
    ):
        self.env = env
        self.route_library = [list(route) for route in route_library]
        self.hotspots = set(hotspots)
        self.corridor_rows = tuple(corridor_rows)
        self.spine_cols = tuple(spine_cols)
        self.message_passing_steps = message_passing_steps
        self.interaction_radius = interaction_radius
        self.hotspot_radius = hotspot_radius
        self.human_penalty = human_penalty
        self.human_prediction_horizon = human_prediction_horizon
        self.predicted_human_penalty = predicted_human_penalty
        self.route_progress = [0 for _ in env.unwrapped.agents]
        self.wait_steps = [0 for _ in env.unwrapped.agents]
        self.reservation_state = {}
        self.override_targets = {}
        self.held_agents = set()

    def update_wait_steps(self, wait_steps):
        self.wait_steps = list(wait_steps)

    def set_recovery_plan(self, override_targets=None, held_agents=None):
        self.override_targets = dict(override_targets or {})
        self.held_agents = set(held_agents or [])

    def clear_recovery_plan(self):
        self.override_targets = {}
        self.held_agents = set()

    def compute_actions(self):
        graph_state = self._build_graph_state()
        agent_scores = graph_state["agent_scores"]
        hotspot_penalties = graph_state["hotspot_penalties"]
        predicted_human_risk = graph_state["predicted_human_risk"]

        agents = self.env.unwrapped.agents
        actions = [NOOP] * len(agents)
        proposals = []
        occupied_now = {(agent.x, agent.y) for agent in agents}
        corridor_load = self._build_corridor_load()

        for idx, agent in enumerate(agents):
            if idx in self.held_agents and idx not in self.override_targets:
                continue
            route = self.route_library[idx % len(self.route_library)]
            if idx in self.override_targets:
                target = self.override_targets[idx]
                if (agent.x, agent.y) == target:
                    self.override_targets.pop(idx, None)
                    continue
            else:
                target, self.route_progress[idx] = self._next_waypoint(
                    agent, route, self.route_progress[idx]
                )
            if (agent.x, agent.y) == target:
                continue

            path = self._find_path(
                start=(agent.x, agent.y),
                goal=target,
                occupied=occupied_now,
                corridor_load=corridor_load,
                hotspot_penalties=hotspot_penalties,
                predicted_human_risk=predicted_human_risk,
            )
            if len(path) < 2:
                continue

            next_cell = path[1]
            desired = self._direction_to((agent.x, agent.y), next_cell)
            if desired is None:
                continue
            if agent.dir != desired:
                actions[idx] = self._turn_action(agent.dir, desired)
                continue

            proposals.append(
                {
                    "idx": idx,
                    "target": next_cell,
                    "path": path,
                    "priority": agent_scores[idx]
                    + (3.0 if idx in self.override_targets else 0.0),
                }
            )

        approved = self._approve_forward_moves(proposals, occupied_now)
        for idx in approved:
            actions[idx] = FORWARD
        self._cleanup_reservations()
        return actions

    def _build_graph_state(self):
        agents = self.env.unwrapped.agents
        humans = self.env.unwrapped.humans
        nodes = []
        features = []
        predicted_human_risk = self._predicted_human_occupancy()

        for idx, agent in enumerate(agents):
            goal = self.route_library[idx % len(self.route_library)][
                self.route_progress[idx] % len(self.route_library[idx % len(self.route_library)])
            ]
            features.append(
                np.array(
                    [
                        1.0,
                        agent.x / max(1, self.env.unwrapped.grid_size[1] - 1),
                        agent.y / max(1, self.env.unwrapped.grid_size[0] - 1),
                        min(1.0, self.wait_steps[idx] / 10.0),
                        self._nearest_human_distance((agent.x, agent.y)) / 10.0,
                        self._nearest_hotspot_distance((agent.x, agent.y)) / 10.0,
                        self._manhattan((agent.x, agent.y), goal) / 20.0,
                    ],
                    dtype=np.float32,
                )
            )
            nodes.append(("agent", idx, (agent.x, agent.y)))

        hotspot_congestion = self._hotspot_congestion()
        for hotspot in sorted(self.hotspots):
            features.append(
                np.array(
                    [
                        0.8,
                        hotspot[0] / max(1, self.env.unwrapped.grid_size[1] - 1),
                        hotspot[1] / max(1, self.env.unwrapped.grid_size[0] - 1),
                        hotspot_congestion.get(hotspot, 0) / 4.0,
                        self._nearest_human_distance(hotspot) / 10.0,
                        min(1.0, predicted_human_risk.get(hotspot, 0.0) / 6.0),
                        0.0,
                    ],
                    dtype=np.float32,
                )
            )
            nodes.append(("hotspot", len(nodes), hotspot))

        for human in humans:
            features.append(
                np.array(
                    [
                        -1.0,
                        human.x / max(1, self.env.unwrapped.grid_size[1] - 1),
                        human.y / max(1, self.env.unwrapped.grid_size[0] - 1),
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                    ],
                    dtype=np.float32,
                )
            )
            nodes.append(("human", len(nodes), (human.x, human.y)))

        X = np.stack(features)
        adjacency = self._build_adjacency(nodes)
        embeddings = self._message_passing(X, adjacency)

        agent_scores = []
        for idx in range(len(agents)):
            emb = embeddings[idx]
            score = (
                1.6 * emb[3]
                + 1.2 * emb[0]
                - 0.9 * emb[4]
                - 0.5 * emb[5]
                - 0.4 * emb[6]
            )
            agent_scores.append(float(score))

        hotspot_penalties = {}
        for node_idx, (kind, _, pos) in enumerate(nodes):
            if kind != "hotspot":
                continue
            emb = embeddings[node_idx]
            hotspot_penalties[pos] = max(
                0.0,
                5.0 * emb[3] + 4.0 * (1.0 - emb[4]) + 2.0 * emb[0],
            )

        return {
            "agent_scores": agent_scores,
            "hotspot_penalties": hotspot_penalties,
            "predicted_human_risk": predicted_human_risk,
        }

    def _build_adjacency(self, nodes):
        n = len(nodes)
        adjacency = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            adjacency[i, i] = 1.0
            kind_i, _, pos_i = nodes[i]
            for j in range(i + 1, n):
                kind_j, _, pos_j = nodes[j]
                dist = self._manhattan(pos_i, pos_j)
                connect = False
                if "agent" in (kind_i, kind_j) and dist <= self.interaction_radius:
                    connect = True
                elif kind_i == "hotspot" and kind_j == "hotspot" and dist <= 4:
                    connect = True
                elif "human" in (kind_i, kind_j) and dist <= self.interaction_radius:
                    connect = True
                if connect:
                    weight = 1.0 / max(1, dist)
                    adjacency[i, j] = weight
                    adjacency[j, i] = weight
        row_sums = adjacency.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return adjacency / row_sums

    def _message_passing(self, X, adjacency):
        H = X.copy()
        for _ in range(self.message_passing_steps):
            neigh = adjacency @ H
            H = np.tanh(0.65 * H + 0.35 * neigh)
        return H

    def _approve_forward_moves(self, proposals, occupied_now):
        proposals.sort(key=lambda item: item["priority"], reverse=True)
        humans = {(human.x, human.y) for human in self.env.unwrapped.humans}
        blocked = set(self.env.unwrapped.dynamic_blocked_cells).union(
            self.env.unwrapped.static_blocked_cells
        )
        claimed_targets = set()
        zone_claims = set()
        approved = []

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
                self.reservation_state[zone] = idx
                zone_claims.add(zone)
            approved.append(idx)
            claimed_targets.add(target)
        return approved

    def _cleanup_reservations(self):
        agents = self.env.unwrapped.agents
        for zone, owner in list(self.reservation_state.items()):
            if owner >= len(agents):
                self.reservation_state.pop(zone, None)
                continue
            agent = agents[owner]
            if self._reservation_zone((agent.x, agent.y)) != zone:
                self.reservation_state.pop(zone, None)

    def _build_corridor_load(self):
        load = defaultdict(int)
        for idx, agent in enumerate(self.env.unwrapped.agents):
            route = self.route_library[idx % len(self.route_library)]
            target = route[self.route_progress[idx] % len(route)]
            if agent.y != target[1] and agent.x in self.spine_cols:
                load[("col", agent.x)] += 1
            if agent.x != target[0] and agent.y in self.corridor_rows:
                load[("row", agent.y)] += 1
        return load

    def _find_path(
        self,
        start,
        goal,
        occupied,
        corridor_load,
        hotspot_penalties,
        predicted_human_risk,
    ):
        blocked = set(self.env.unwrapped.static_blocked_cells).union(
            self.env.unwrapped.dynamic_blocked_cells
        )
        forbidden = blocked.union(pos for pos in occupied if pos != start and pos != goal)
        if goal in forbidden:
            forbidden.discard(goal)

        frontier = []
        heapq.heappush(frontier, (0.0, 0, start))
        came_from = {start: None}
        cost_so_far = {start: 0.0}

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for nxt in self._neighbors(current):
                if nxt in forbidden:
                    continue
                new_cost = cost_so_far[current] + 1.0
                new_cost += self._corridor_penalty(nxt, corridor_load)
                new_cost += self._gnn_hotspot_penalty(nxt, hotspot_penalties)
                new_cost += (
                    predicted_human_risk.get(nxt, 0.0) * self.predicted_human_penalty
                )
                if self._in_human_danger(nxt) and nxt != goal:
                    new_cost += self.human_penalty
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self._manhattan(nxt, goal)
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

    def _corridor_penalty(self, cell, corridor_load):
        x, y = cell
        penalty = 0.0
        if y in self.corridor_rows:
            penalty += corridor_load.get(("row", y), 0) * 1.5
        if x in self.spine_cols:
            penalty += corridor_load.get(("col", x), 0) * 0.8
        return penalty

    def _gnn_hotspot_penalty(self, cell, hotspot_penalties):
        zone = self._reservation_zone(cell)
        if zone is None:
            return 0.0
        return hotspot_penalties.get(zone, 0.0)

    def _hotspot_congestion(self):
        congestion = defaultdict(int)
        for agent in self.env.unwrapped.agents:
            zone = self._reservation_zone((agent.x, agent.y))
            if zone is not None:
                congestion[zone] += 1
        return congestion

    def _in_human_danger(self, cell):
        for human in self.env.unwrapped.humans:
            if self._manhattan(cell, (human.x, human.y)) <= self.env.unwrapped.human_safety_radius:
                return True
        return False

    def _predicted_human_occupancy(self):
        risk = defaultdict(float)
        horizon = max(1, int(self.human_prediction_horizon))
        for human in self.env.unwrapped.humans:
            for step_idx, pos in enumerate(
                self._predict_human_positions(human, horizon), start=1
            ):
                weight = (horizon - step_idx + 1) / horizon
                risk[pos] += weight
                for neighbor in self._neighbors(pos):
                    risk[neighbor] += weight * 0.4
        return risk

    def _predict_human_positions(self, human, horizon):
        if not getattr(human, "route", None):
            return [(human.x, human.y)] * horizon
        route = list(human.route)
        if not route:
            return [(human.x, human.y)] * horizon

        positions = []
        route_index = human.route_index
        for _ in range(horizon):
            route_index = (route_index + 1) % len(route)
            positions.append(tuple(route[route_index]))
        return positions

    def _nearest_human_distance(self, pos):
        if not self.env.unwrapped.humans:
            return 10
        return min(self._manhattan(pos, (human.x, human.y)) for human in self.env.unwrapped.humans)

    def _nearest_hotspot_distance(self, pos):
        if not self.hotspots:
            return 10
        return min(self._manhattan(pos, hotspot) for hotspot in self.hotspots)

    def _reservation_zone(self, cell):
        best = None
        best_dist = 10**9
        for hotspot in self.hotspots:
            dist = self._manhattan(cell, hotspot)
            if dist < best_dist:
                best = hotspot
                best_dist = dist
        if best is None or best_dist > self.hotspot_radius:
            return None
        return best

    def _next_waypoint(self, agent, route, route_progress):
        target = route[route_progress % len(route)]
        if (agent.x, agent.y) == target:
            route_progress = (route_progress + 1) % len(route)
            target = route[route_progress]
        return target, route_progress

    def _neighbors(self, pos):
        x, y = pos
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < self.env.unwrapped.grid_size[1] and 0 <= ny < self.env.unwrapped.grid_size[0]:
                if self.env.unwrapped._is_highway(nx, ny):
                    yield (nx, ny)

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

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
