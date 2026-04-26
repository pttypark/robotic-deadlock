from collections import defaultdict

from deadlock.dsd_fsd.model import TransactionStatus
from deadlock.dsd_fsd.routing import action_towards, find_path, manhattan
from deadlock.navigation import FORWARD, NOOP


class BaseTransactionController:
    def __init__(self, env, points, service_steps=3, stall_threshold=8, recovery_ttl=25):
        self.env = env
        self.points = points
        self.service_steps = service_steps
        self.stall_threshold = stall_threshold
        self.recovery_ttl = recovery_ttl
        self.agent_tx = {}
        self.service_until = {}
        self.recovery_targets = {}
        self.recovery_started = {}
        self.stats = {
            "holds": 0,
            "reroutes": 0,
            "triggers": 0,
            "forced_escapes": 0,
            "blocked_moves": 0,
            "stall_recoveries": 0,
            "blocker_recoveries": 0,
            "local_escapes": 0,
            "recovery_expired": 0,
        }

    def compute_actions(self, step, ledger):
        self.attach_ledger(ledger)
        self._finish_or_start_service(step, ledger)
        self._assign_waiting(step, ledger)
        self._refresh_recovery_targets(step)
        actions, targets = self._nominal_actions()
        return actions

    def _finish_or_start_service(self, step, ledger):
        for idx, tx_id in list(self.agent_tx.items()):
            tx = ledger.transactions[tx_id]
            agent = self.env.unwrapped.agents[idx]
            pos = (agent.x, agent.y)
            if tx.status == TransactionStatus.SERVICING:
                if step >= self.service_until.get(idx, step):
                    completed = ledger.advance_after_service(tx, step)
                    if completed:
                        self.agent_tx.pop(idx, None)
                    self.service_until.pop(idx, None)
                continue
            if tx.status == TransactionStatus.ASSIGNED and pos == tx.current_target():
                ledger.start_service(tx, step)
                self.service_until[idx] = step + self.service_steps

    def _free_agents(self):
        busy = set(self.agent_tx)
        busy.update(self.recovery_targets)
        return [
            idx
            for idx, _ in enumerate(self.env.unwrapped.agents)
            if idx not in busy
        ]

    def _assign(self, ledger, tx, agent_idx, step):
        self.recovery_targets.pop(agent_idx, None)
        self.recovery_started.pop(agent_idx, None)
        ledger.assign(tx, agent_idx, step)
        self.agent_tx[agent_idx] = tx.id

    def _target_for_agent(self, idx):
        task_target = self._task_target_for_idx(idx)
        if task_target is not None:
            return task_target
        return self._parking_target_for_idx(idx)

    def _nearest_decision_point(self, idx):
        agent = self.env.unwrapped.agents[idx]
        pos = (agent.x, agent.y)
        return min(self.points.decision_points, key=lambda point: manhattan(pos, point))

    def _path_allowed(self, idx):
        return None

    def _path_allowed_for_target(self, idx, target):
        allowed = self._path_allowed(idx)
        agent = self.env.unwrapped.agents[idx]
        needs_micro_escape_path = (
            agent.carrying_shelf is None
            and (target in self.points.escape_points or not self.env.unwrapped._is_highway(*target))
        )
        if idx not in self.recovery_targets and not needs_micro_escape_path:
            return allowed

        allowed = set(allowed or self.points.highway_points)
        allowed.add(target)
        if agent.carrying_shelf is None:
            allowed.update(self.points.escape_points)
        return allowed

    def _nominal_actions(self):
        actions = [NOOP] * len(self.env.unwrapped.agents)
        targets = {}
        occupied = {
            (agent.x, agent.y)
            for agent in self.env.unwrapped.agents
        }
        for idx, agent in enumerate(self.env.unwrapped.agents):
            if idx in self.service_until:
                actions[idx] = NOOP
                self.stats["holds"] += 1
                continue
            target = self._target_for_idx(idx)
            blocked = occupied - {(agent.x, agent.y), target}
            path = find_path(
                self.env,
                (agent.x, agent.y),
                target,
                allowed=self._path_allowed_for_target(idx, target),
                blocked=blocked,
            )
            action, requested_target = action_towards(agent, path)
            if (
                action == NOOP
                and idx in self.agent_tx
                and self._agent_wait_steps()[idx] >= self.stall_threshold
            ):
                action, requested_target = self._local_escape_action(idx, occupied)
            actions[idx] = action
            targets[idx] = requested_target
        return actions, targets

    def _target_for_idx(self, idx):
        if idx in self.recovery_targets:
            return self.recovery_targets[idx]
        task_target = self._task_target_for_idx(idx)
        if task_target is not None:
            return task_target
        return self._parking_target_for_idx(idx)

    def _task_target_for_idx(self, idx):
        tx_id = self.agent_tx.get(idx)
        if tx_id is None:
            return None
        tx = self._ledger.transactions[tx_id]
        return self._decision_based_target(idx, tx.current_target())

    def _decision_based_target(self, idx, target):
        agent = self.env.unwrapped.agents[idx]
        pos = (agent.x, agent.y)
        if target in self.points.buffer_points:
            return target
        if pos[0] == target[0] or manhattan(pos, target) <= 1:
            return target
        last_decision = min(
            self.points.decision_points,
            key=lambda point: manhattan(point, target),
        )
        if pos == last_decision:
            return target
        return last_decision

    def _parking_target_for_idx(self, idx):
        agent = self.env.unwrapped.agents[idx]
        start = (agent.x, agent.y)
        allowed = self._path_allowed(idx)
        candidate_groups = (
            self.points.escape_points if agent.carrying_shelf is None else (),
            self.points.waiting_points,
            self.points.decision_points,
        )

        for candidates in candidate_groups:
            valid = [
                cell
                for cell in candidates
                if cell != start
                and self._can_enter(idx, cell)
                and (allowed is None or cell in allowed)
            ]
            if valid:
                return min(valid, key=lambda cell: manhattan(start, cell))
        return self._nearest_decision_point(idx)

    def _refresh_recovery_targets(self, step):
        occupied = {
            (agent.x, agent.y)
            for agent in self.env.unwrapped.agents
        }
        wait_steps = self._agent_wait_steps()

        for idx, target in list(self.recovery_targets.items()):
            agent = self.env.unwrapped.agents[idx]
            pos = (agent.x, agent.y)
            expired = step - self.recovery_started.get(idx, step) > self.recovery_ttl
            if pos == target or idx in self.service_until:
                self.recovery_targets.pop(idx, None)
                self.recovery_started.pop(idx, None)
            elif expired:
                self.recovery_targets.pop(idx, None)
                self.recovery_started.pop(idx, None)
                self.stats["recovery_expired"] += 1

        stalled = [
            idx
            for idx in self.agent_tx
            if idx not in self.service_until
            and idx not in self.recovery_targets
            and wait_steps[idx] >= self.stall_threshold
        ]
        if not stalled:
            return

        # Move one low-priority blocker at a time; moving everyone at once often
        # creates a second jam around the same conflict zone.
        idx = min(stalled, key=lambda candidate: self._recovery_priority(candidate, step))
        blocker = self._blocking_agent_for(idx)
        if blocker is not None and blocker not in self.recovery_targets:
            idx = blocker
            self.stats["blocker_recoveries"] += 1

        target = self._select_recovery_target(idx, occupied)
        if target is None:
            return
        self.recovery_targets[idx] = target
        self.recovery_started[idx] = step
        self.stats["stall_recoveries"] += 1

    def _recovery_priority(self, idx, step):
        return (0, idx)

    def _select_recovery_target(self, idx, occupied):
        agent = self.env.unwrapped.agents[idx]
        start = (agent.x, agent.y)
        blocked = occupied - {start}
        candidate_groups = (
            self.points.escape_points,
            self.points.waiting_points,
            self.points.decision_points,
        )
        for candidates in candidate_groups:
            valid = [
                cell
                for cell in candidates
                if cell != start
                and cell not in blocked
                and self._can_enter(idx, cell)
                and self._has_recovery_path(idx, start, cell, blocked)
            ]
            if valid:
                return min(valid, key=lambda cell: manhattan(start, cell))
        return None

    def _has_recovery_path(self, idx, start, target, blocked):
        allowed = set(self.points.highway_points)
        allowed.add(target)
        if self.env.unwrapped.agents[idx].carrying_shelf is None:
            allowed.update(self.points.escape_points)
        path = find_path(self.env, start, target, allowed=allowed, blocked=blocked)
        return len(path) > 1

    def _blocking_agent_for(self, idx):
        target = self._task_target_for_idx(idx)
        if target is None:
            return None

        agent = self.env.unwrapped.agents[idx]
        start = (agent.x, agent.y)
        path = find_path(
            self.env,
            start,
            target,
            allowed=self._path_allowed_for_target(idx, target),
            blocked=set(),
        )
        if len(path) < 2:
            return None

        occupied = {
            (other.x, other.y): other_idx
            for other_idx, other in enumerate(self.env.unwrapped.agents)
        }
        for cell in path[1:8]:
            blocker = occupied.get(cell)
            if blocker is None or blocker == idx:
                continue
            if blocker in self.service_until:
                continue
            return blocker
        return None

    def _local_escape_action(self, idx, occupied):
        agent = self.env.unwrapped.agents[idx]
        start = (agent.x, agent.y)
        for target in self._adjacent_escape_candidates(idx, occupied):
            action, requested_target = action_towards(agent, [start, target])
            self.stats["local_escapes"] += 1
            return action, requested_target
        self.stats["holds"] += 1
        return NOOP, start

    def _adjacent_escape_candidates(self, idx, occupied):
        agent = self.env.unwrapped.agents[idx]
        x, y = agent.x, agent.y
        candidates = []
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (0 <= nx < self.env.unwrapped.grid_size[1] and 0 <= ny < self.env.unwrapped.grid_size[0]):
                continue
            cell = (nx, ny)
            if cell in occupied or not self._can_enter(idx, cell):
                continue
            score = (
                0 if cell in self.points.escape_points else 1,
                0 if self.env.unwrapped._is_highway(nx, ny) else 1,
                manhattan(cell, self._target_for_idx(idx)),
            )
            candidates.append((score, cell))
        return [cell for _, cell in sorted(candidates)]

    def _can_enter(self, idx, cell):
        x, y = cell
        agent = self.env.unwrapped.agents[idx]
        if not (0 <= x < self.env.unwrapped.grid_size[1] and 0 <= y < self.env.unwrapped.grid_size[0]):
            return False
        if agent.carrying_shelf is None:
            return True
        return self.env.unwrapped._is_highway(x, y) and self.env.unwrapped.grid[1, y, x] == 0

    def _agent_wait_steps(self):
        wait_steps = getattr(self.env.unwrapped, "_agent_wait_steps", None)
        if wait_steps is None:
            return [0 for _ in self.env.unwrapped.agents]
        return list(wait_steps)

    def _assign_waiting(self, step, ledger):
        raise NotImplementedError

    def attach_ledger(self, ledger):
        self._ledger = ledger
        return self


class DSDController(BaseTransactionController):
    def __init__(self, env, points, service_steps=3, stall_threshold=8, recovery_ttl=25):
        super().__init__(
            env,
            points,
            service_steps=service_steps,
            stall_threshold=stall_threshold,
            recovery_ttl=recovery_ttl,
        )
        self.zone_allowed = self._build_zone_allowed()

    def _assign_waiting(self, step, ledger):
        self.attach_ledger(ledger)
        free = set(self._free_agents())
        for tx in sorted(ledger.waiting(), key=lambda item: item.created_step):
            agent_idx = tx.zone_id % len(self.env.unwrapped.agents)
            if agent_idx not in free:
                continue
            self._assign(ledger, tx, agent_idx, step)
            free.remove(agent_idx)

    def _path_allowed(self, idx):
        return self.zone_allowed[idx % len(self.zone_allowed)]

    def _build_zone_allowed(self):
        allowed_by_zone = []
        max_x = self.points.width - 1
        for zone in self.points.zones:
            if not zone:
                allowed_by_zone.append(set(self.points.highway_points))
                continue
            xs = [point[0] for point in zone]
            lo = max(0, min(xs) - 1)
            hi = min(max_x, max(xs) + 1)
            allowed = {
                (x, y)
                for x, y in self.points.highway_points
                if lo <= x <= hi
            }
            allowed.update(zone)
            allowed_by_zone.append(allowed)
        return allowed_by_zone


class DeadlockProneRuleController(BaseTransactionController):
    def __init__(self, env, points, service_steps=3):
        super().__init__(
            env,
            points,
            service_steps=service_steps,
            stall_threshold=10**9,
            recovery_ttl=10**9,
        )

    def _assign_waiting(self, step, ledger):
        self.attach_ledger(ledger)
        free = set(self._free_agents())
        for tx in sorted(ledger.waiting(), key=lambda item: item.created_step):
            agent_idx = tx.zone_id % len(self.env.unwrapped.agents)
            if agent_idx not in free:
                continue
            self._assign(ledger, tx, agent_idx, step)
            free.remove(agent_idx)

    def _refresh_recovery_targets(self, step):
        return

    def _parking_target_for_idx(self, idx):
        agent = self.env.unwrapped.agents[idx]
        return (agent.x, agent.y)

    def _nominal_actions(self):
        actions = [NOOP] * len(self.env.unwrapped.agents)
        targets = {}
        for idx, agent in enumerate(self.env.unwrapped.agents):
            if idx in self.service_until:
                actions[idx] = NOOP
                self.stats["holds"] += 1
                continue
            target = self._target_for_idx(idx)
            path = find_path(
                self.env,
                (agent.x, agent.y),
                target,
                allowed=self._path_allowed_for_target(idx, target),
                blocked=set(),
            )
            action, requested_target = action_towards(agent, path)
            actions[idx] = action
            targets[idx] = requested_target
        return actions, targets


class LocalGraphAdmissionController(BaseTransactionController):
    def __init__(self, env, points, service_steps=3):
        super().__init__(
            env,
            points,
            service_steps=service_steps,
            stall_threshold=10**9,
            recovery_ttl=10**9,
        )
        self.core_min_y = 1
        self.core_max_y = max(1, points.height - 4)
        self.shared_areas = {
            zone.id: set(zone.cells)
            for zone in points.conflict_zones
            if zone.id.startswith("shared_")
        }
        self.area_owner = {}
        self.stats.update(
            {
                "graph_decisions": 0,
                "admission_holds": 0,
                "max_core_occupancy": 0,
                "max_queue_length": 0,
            }
        )

    def _assign_waiting(self, step, ledger):
        self.attach_ledger(ledger)
        free = set(self._free_agents())
        for tx in sorted(ledger.waiting(), key=lambda item: item.created_step):
            agent_idx = tx.zone_id % len(self.env.unwrapped.agents)
            if agent_idx not in free:
                continue
            self._assign(ledger, tx, agent_idx, step)
            free.remove(agent_idx)

    def _refresh_recovery_targets(self, step):
        return

    def _parking_target_for_idx(self, idx):
        agent = self.env.unwrapped.agents[idx]
        return (agent.x, agent.y)

    def _nominal_actions(self):
        actions = [NOOP] * len(self.env.unwrapped.agents)
        targets = {}
        occupied = {
            (agent.x, agent.y)
            for agent in self.env.unwrapped.agents
        }
        for idx, agent in enumerate(self.env.unwrapped.agents):
            if idx in self.service_until:
                actions[idx] = NOOP
                self.stats["holds"] += 1
                continue
            target = self._target_for_idx(idx)
            path = find_path(
                self.env,
                (agent.x, agent.y),
                target,
                allowed=self._path_allowed_for_target(idx, target),
                blocked=set(),
            )
            action, requested_target = action_towards(agent, path)
            actions[idx] = action
            targets[idx] = requested_target
        self._apply_shared_area_admission(actions, targets)
        return actions, targets

    def _apply_shared_area_admission(self, actions, targets):
        candidates_by_area = defaultdict(list)
        occupants_by_area = defaultdict(set)

        for idx, agent in enumerate(self.env.unwrapped.agents):
            pos = (agent.x, agent.y)
            area = self._shared_area_for_cell(pos)
            if area is not None:
                occupants_by_area[area].add(idx)

        for idx, action in enumerate(actions):
            if action == NOOP:
                continue
            agent = self.env.unwrapped.agents[idx]
            pos = (agent.x, agent.y)
            target = targets.get(idx, pos)
            direction = self._movement_direction(idx, target)
            current_area = self._shared_area_for_cell(pos)
            target_area = self._shared_area_for_cell(target)
            approach_area = self._approach_area_for_cell(pos, direction)
            area = current_area or target_area or approach_area
            if area is None:
                continue
            candidates_by_area[area].append(
                {
                    "idx": idx,
                    "pos": pos,
                    "target": target,
                    "current_area": current_area,
                    "target_area": target_area,
                    "direction": direction,
                }
            )

        for area, candidates in candidates_by_area.items():
            occupants = occupants_by_area.get(area, set())
            active_ids = occupants | {candidate["idx"] for candidate in candidates}
            owner = self.area_owner.get(area)
            if owner is not None and owner not in active_ids:
                self.area_owner.pop(area, None)
                owner = None

            self.stats["graph_decisions"] += 1
            self.stats["max_core_occupancy"] = max(
                self.stats["max_core_occupancy"],
                len(occupants),
            )
            self.stats["max_queue_length"] = max(
                self.stats["max_queue_length"],
                len(candidates),
            )

            if owner is not None:
                selectable = [
                    candidate
                    for candidate in candidates
                    if candidate["idx"] == owner
                ]
                if not selectable and owner in occupants:
                    held = 0
                    for candidate in candidates:
                        actions[candidate["idx"]] = self._retreat_action(area, candidate) or NOOP
                        held += 1
                    self.stats["admission_holds"] += held
                    continue
            elif occupants:
                selectable = [
                    candidate
                    for candidate in candidates
                    if candidate["idx"] in occupants
                ]
                if not selectable:
                    owner = min(occupants)
                    self.area_owner[area] = owner
                    held = 0
                    for candidate in candidates:
                        actions[candidate["idx"]] = self._retreat_action(area, candidate) or NOOP
                        held += 1
                    self.stats["admission_holds"] += held
                    continue
            else:
                selectable = candidates

            if not selectable:
                for candidate in candidates:
                    actions[candidate["idx"]] = NOOP
                self.stats["admission_holds"] += len(candidates)
                continue

            scores = self._local_graph_scores(area, selectable)
            winner = max(selectable, key=lambda item: (scores[item["idx"]], -item["idx"]))
            winner_idx = winner["idx"]
            self.area_owner[area] = winner_idx
            held = 0
            for candidate in candidates:
                idx = candidate["idx"]
                if idx == winner_idx:
                    continue
                actions[idx] = self._retreat_action(area, candidate) or NOOP
                held += 1
            self.stats["admission_holds"] += held

    def _local_graph_scores(self, area, candidates):
        n = len(candidates)
        if n == 1:
            return {candidates[0]["idx"]: self._candidate_base_score(area, candidates[0])}

        features = []
        for candidate in candidates:
            others = [other for other in candidates if other is not candidate]
            wait = self._agent_wait_steps()[candidate["idx"]]
            inside = 1.0 if candidate["current_area"] == area else 0.0
            exit_free = 1.0 if self._exit_is_free(area, candidate) else 0.0
            opposite = sum(
                1
                for other in others
                if self._opposite_direction(other["direction"], candidate["direction"])
            )
            same_queue = sum(
                1
                for other in others
                if other["direction"] == candidate["direction"]
            )
            distance = manhattan(candidate["pos"], candidate["target"])
            features.append(
                [
                    min(1.0, wait / 20.0),
                    inside,
                    exit_free,
                    min(1.0, opposite / 3.0),
                    min(1.0, same_queue / 3.0),
                    1.0 / (1.0 + distance),
                ]
            )

        adjacency = [[0.0 for _ in range(n)] for _ in range(n)]
        for i, left in enumerate(candidates):
            adjacency[i][i] = 1.0
            for j, right in enumerate(candidates):
                if i == j:
                    continue
                if self._opposite_direction(left["direction"], right["direction"]):
                    adjacency[i][j] = 1.4
                elif left["direction"] == right["direction"]:
                    adjacency[i][j] = 0.8
                else:
                    adjacency[i][j] = 0.4
        for i, row in enumerate(adjacency):
            total = sum(row) or 1.0
            adjacency[i] = [value / total for value in row]

        hidden = [row[:] for row in features]
        for _ in range(2):
            next_hidden = []
            for i in range(n):
                mixed = [0.0 for _ in hidden[i]]
                for j in range(n):
                    weight = adjacency[i][j]
                    for k, value in enumerate(hidden[j]):
                        mixed[k] += weight * value
                next_hidden.append(
                    [
                        0.65 * hidden[i][k] + 0.35 * mixed[k]
                        for k in range(len(hidden[i]))
                    ]
                )
            hidden = next_hidden

        scores = {}
        for candidate, emb in zip(candidates, hidden):
            scores[candidate["idx"]] = (
                2.4 * emb[0]
                + 3.0 * emb[1]
                + 1.5 * emb[2]
                - 1.2 * emb[3]
                - 0.4 * emb[4]
                + 0.8 * emb[5]
            )
        return scores

    def _candidate_base_score(self, area, candidate):
        wait = self._agent_wait_steps()[candidate["idx"]]
        return wait + 2.0 if self._exit_is_free(area, candidate) else wait

    def _shared_area_for_cell(self, cell):
        for area_id, cells in self.shared_areas.items():
            if cell in cells:
                return area_id
        if self.shared_areas:
            return None

        x, y = cell
        if x not in self.points.aisle_columns:
            return None
        if self.core_min_y <= y <= self.core_max_y:
            return x
        return None

    def _movement_direction(self, idx, requested_target):
        agent = self.env.unwrapped.agents[idx]
        if requested_target != (agent.x, agent.y):
            return (_sign(requested_target[0] - agent.x), _sign(requested_target[1] - agent.y))
        task_target = self._task_target_for_idx(idx) or requested_target
        dx = task_target[0] - agent.x
        dy = task_target[1] - agent.y
        if abs(dx) >= abs(dy) and dx:
            return (_sign(dx), 0)
        if dy:
            return (0, _sign(dy))
        return (0, 0)

    def _approach_area_for_cell(self, cell, direction):
        if direction == (0, 0):
            return None
        for distance in (1, 2, 3):
            nxt = (
                cell[0] + direction[0] * distance,
                cell[1] + direction[1] * distance,
            )
            area = self._shared_area_for_cell(nxt)
            if area is not None:
                return area
        return None

    def _exit_is_free(self, area, candidate):
        direction = candidate["direction"]
        if area in self.shared_areas:
            cells = self.shared_areas[area]
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            x, y = candidate["target"]
            if direction[0] > 0:
                exit_cell = (max(xs) + 1, y)
            elif direction[0] < 0:
                exit_cell = (min(xs) - 1, y)
            elif direction[1] > 0:
                exit_cell = (x, max(ys) + 1)
            elif direction[1] < 0:
                exit_cell = (x, min(ys) - 1)
            else:
                exit_cell = candidate["target"]
        elif direction[1] > 0:
            exit_cell = (area, self.core_max_y + 1)
        elif direction[1] < 0:
            exit_cell = (area, self.core_min_y - 1)
        else:
            exit_cell = candidate["target"]
        for idx, agent in enumerate(self.env.unwrapped.agents):
            if idx == candidate["idx"]:
                continue
            if (agent.x, agent.y) == exit_cell:
                return False
        return True

    def _opposite_direction(self, left, right):
        return left != (0, 0) and left[0] == -right[0] and left[1] == -right[1]

    def _retreat_action(self, area, candidate):
        direction = candidate["direction"]
        if direction == (0, 0):
            return None
        agent = self.env.unwrapped.agents[candidate["idx"]]
        start = candidate["pos"]
        target = (start[0] - direction[0], start[1] - direction[1])
        if not self._retreat_cell_is_open(candidate["idx"], target):
            return None
        action, _ = action_towards(agent, [start, target])
        return action

    def _retreat_cell_is_open(self, idx, cell):
        x, y = cell
        if not (0 <= x < self.env.unwrapped.grid_size[1] and 0 <= y < self.env.unwrapped.grid_size[0]):
            return False
        if not self.env.unwrapped._is_highway(x, y):
            return False
        for other_idx, other in enumerate(self.env.unwrapped.agents):
            if other_idx == idx:
                continue
            if (other.x, other.y) == cell:
                return False
        return True


class FSDTriggerController(BaseTransactionController):
    def _assign_waiting(self, step, ledger):
        self.attach_ledger(ledger)
        free_agents = set(self._free_agents())
        waiting = ledger.waiting()
        if not waiting or not free_agents:
            return

        avg_wait = _average_wait(ledger, step)
        aisle_density = self._aisle_density(ledger)
        while waiting and free_agents:
            tx = self._select_transaction(waiting, step, avg_wait, aisle_density)
            idx = self._closest_free_agent(tx, free_agents)
            self._assign(ledger, tx, idx, step)
            free_agents.remove(idx)
            waiting.remove(tx)

    def _select_transaction(self, waiting, step, avg_wait, aisle_density):
        if len(waiting) == 1:
            return waiting[0]
        overdue = [tx for tx in waiting if tx.waiting_time(step) > avg_wait]
        if overdue:
            return min(overdue, key=lambda tx: tx.created_step)
        return min(
            waiting,
            key=lambda tx: (
                aisle_density.get(_nearest_aisle(self.points.aisle_columns, tx.target[0]), 0),
                manhattan(tx.buffer, tx.target),
                tx.created_step,
            ),
        )

    def _closest_free_agent(self, tx, free_agents):
        return min(
            free_agents,
            key=lambda idx: manhattan(
                (self.env.unwrapped.agents[idx].x, self.env.unwrapped.agents[idx].y),
                tx.current_target(),
            ),
        )

    def compute_actions(self, step, ledger):
        self.attach_ledger(ledger)
        self._finish_or_start_service(step, ledger)
        self._assign_waiting(step, ledger)
        self._refresh_recovery_targets(step)
        actions, targets = self._nominal_actions()
        return self._resolve_conflicts(actions, targets, step)

    def _assignment_score(self, tx, idx, step, avg_wait, aisle_density, agent_pos):
        wait = tx.waiting_time(step)
        waited_long = 1 if wait > avg_wait else 0
        density = aisle_density.get(tx.target[0], 0)
        distance = manhattan(agent_pos, tx.target)
        return (
            waited_long,
            wait,
            -density,
            -distance,
            -idx,
        )

    def _aisle_density(self, ledger):
        density = defaultdict(int)
        for tx in ledger.active():
            density[_nearest_aisle(self.points.aisle_columns, tx.target[0])] += 1
        return density

    def _resolve_conflicts(self, actions, targets, step):
        self._resolve_head_on_aisles(actions, step)

        groups = defaultdict(list)
        for idx, target in targets.items():
            zone = self._conflict_zone_for(target)
            if zone is not None:
                groups[zone.id].append(idx)
            groups[("target", target)].append(idx)

        for key, members in groups.items():
            if len(members) <= 1:
                continue
            winner = max(members, key=lambda idx: self._priority(idx, step))
            for idx in members:
                if idx == winner or idx in self.recovery_targets:
                    continue
                action = self._avoidance_action(idx, targets)
                actions[idx] = action
                self.stats["triggers"] += 1
        return actions

    def _resolve_head_on_aisles(self, actions, step):
        occupied = {
            (agent.x, agent.y)
            for agent in self.env.unwrapped.agents
        }
        active = [
            idx
            for idx in self.agent_tx
            if idx not in self.service_until and idx not in self.recovery_targets
        ]

        for left_pos, idx in enumerate(active):
            agent = self.env.unwrapped.agents[idx]
            pos = (agent.x, agent.y)
            target = self._task_target_for_idx(idx)
            if target is None:
                continue
            for other_idx in active[left_pos + 1:]:
                other = self.env.unwrapped.agents[other_idx]
                other_pos = (other.x, other.y)
                other_target = self._task_target_for_idx(other_idx)
                if other_target is None:
                    continue
                if not self._is_head_on_pair(pos, target, other_pos, other_target):
                    continue

                winner = max((idx, other_idx), key=lambda candidate: self._aisle_priority(candidate, step))
                loser = other_idx if winner == idx else idx
                escape = self._select_side_escape_target(loser, occupied)
                if escape is None:
                    continue
                self.recovery_targets[loser] = escape
                self.recovery_started[loser] = step
                actions[loser] = self._action_to_recovery(loser, escape, occupied)
                self.stats["triggers"] += 1
                self.stats["forced_escapes"] += 1
                self.stats["reroutes"] += 1
                occupied.add(escape)

    def _is_head_on_pair(self, pos, target, other_pos, other_target):
        if pos[0] != other_pos[0] or pos[0] not in self.points.aisle_columns:
            return False
        if target[0] != pos[0] or other_target[0] != other_pos[0]:
            return False

        dy = _sign(target[1] - pos[1])
        other_dy = _sign(other_target[1] - other_pos[1])
        if dy == 0 or other_dy == 0 or dy == other_dy:
            return False

        upper_y, lower_y = sorted((pos[1], other_pos[1]))
        upper_target = target if pos[1] == upper_y else other_target
        lower_target = other_target if pos[1] == upper_y else target
        return upper_target[1] >= lower_y or lower_target[1] <= upper_y

    def _aisle_priority(self, idx, step):
        target = self._task_target_for_idx(idx)
        leaving_aisle = 1 if target in self.points.buffer_points else 0
        return (leaving_aisle, self._priority(idx, step))

    def _select_side_escape_target(self, idx, occupied):
        agent = self.env.unwrapped.agents[idx]
        start = (agent.x, agent.y)
        candidates = []

        for dx in (-1, 1):
            side = (start[0] + dx, start[1])
            if self._is_open_highway(side, occupied):
                candidates.append((0, side))

        lateral_points = [
            point
            for point in self.points.highway_points
            if point[0] != start[0] and point not in occupied
        ]
        for point in self.points.escape_points:
            if point[0] != start[0] and point not in occupied:
                candidates.append((1, point))
        for point in lateral_points:
            if point[1] == start[1] or point[1] in self.points.transition_rows:
                candidates.append((2, point))

        reachable = []
        blocked = occupied - {start}
        for tier, target in candidates:
            path = find_path(
                self.env,
                start,
                target,
                allowed=self._path_allowed_for_target(idx, target),
                blocked=blocked,
            )
            if len(path) > 1:
                reachable.append((tier, len(path), manhattan(start, target), target))
        if not reachable:
            return None
        return min(reachable)[-1]

    def _is_open_highway(self, cell, occupied):
        x, y = cell
        if not (0 <= x < self.env.unwrapped.grid_size[1] and 0 <= y < self.env.unwrapped.grid_size[0]):
            return False
        return cell not in occupied and self.env.unwrapped._is_highway(x, y)

    def _action_to_recovery(self, idx, target, occupied):
        agent = self.env.unwrapped.agents[idx]
        start = (agent.x, agent.y)
        path = find_path(
            self.env,
            start,
            target,
            allowed=self._path_allowed_for_target(idx, target),
            blocked=occupied - {start},
        )
        action, _ = action_towards(agent, path)
        return action

    def _recovery_priority(self, idx, step):
        return self._priority(idx, step)

    def _priority(self, idx, step):
        tx_id = self.agent_tx.get(idx)
        carrying = 1 if tx_id is not None else 0
        agent = self.env.unwrapped.agents[idx]
        pos = (agent.x, agent.y)
        exit_free = 1 if self._nearest_free_decision(pos) is not None else 0
        density = sum(
            1
            for other in self.env.unwrapped.agents
            if manhattan(pos, (other.x, other.y)) <= 2
        )
        dist = min(manhattan(pos, point) for point in self.points.decision_points)
        wait = self.env.unwrapped._agent_wait_steps[idx]
        return 2 * wait + 3 * carrying + exit_free - density - 0.1 * dist - 0.01 * idx

    def _avoidance_action(self, idx, targets):
        agent = self.env.unwrapped.agents[idx]
        start = (agent.x, agent.y)
        occupied = {
            (other.x, other.y)
            for other in self.env.unwrapped.agents
            if other is not agent
        }
        blocked = occupied.union(targets.values())
        for candidates, stat_key in (
            (self.points.escape_points, "forced_escapes"),
            (self.points.waiting_points, "blocked_moves"),
            (self.points.decision_points, "blocked_moves"),
        ):
            target = self._nearest_free(start, candidates, blocked)
            if target is None:
                continue
            allowed = set(self.points.highway_points)
            allowed.add(target)
            path = find_path(self.env, start, target, allowed=allowed, blocked=occupied)
            action, _ = action_towards(agent, path)
            self.stats[stat_key] += 1
            return action
        self.stats["holds"] += 1
        return NOOP

    def _nearest_free(self, start, candidates, blocked):
        free = [point for point in candidates if point not in blocked]
        if not free:
            return None
        return min(free, key=lambda point: manhattan(start, point))

    def _nearest_free_decision(self, start):
        return self._nearest_free(start, self.points.decision_points, set())

    def _conflict_zone_for(self, cell):
        for zone in self.points.conflict_zones:
            if cell in zone.cells:
                return zone
        return None


def _average_wait(ledger, step):
    waits = [tx.waiting_time(step) for tx in ledger.transactions.values()]
    if not waits:
        return 0.0
    return sum(waits) / len(waits)


def _nearest_aisle(aisle_columns, x):
    return min(aisle_columns, key=lambda aisle_x: abs(aisle_x - x))


def _sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
