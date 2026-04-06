import gymnasium as gym
import heapq
import numpy as np
import rware
from rware.warehouse import Direction

from deadlock.core import DeadlockDetector, EpisodeRecorder, hold_render_window
from deadlock.global_manager import GlobalTrafficManager
from deadlock.navigation import FORWARD, NOOP, TURN_LEFT, TURN_RIGHT
from deadlock.recovery import AdaptiveDeadlockRecovery


LAYOUT = """
xxxxgxxxgxxxgxxxgxxxgxxxx
xxxx.xxx.xxx.xxx.xxx.xxxx
xx.....................xx
xxxx.xxx.xxx.xxx.xxx.xxxx
g.......................g
xxxx.xxx.xxx.xxx.xxx.xxxx
xx.....................xx
xxxx.xxx.xxx.xxx.xxx.xxxx
g.......................g
xxxx.xxx.xxx.xxx.xxx.xxxx
xx.....................xx
xxxx.xxx.xxx.xxx.xxx.xxxx
g.......................g
xxxx.xxx.xxx.xxx.xxx.xxxx
xx.....................xx
xxxx.xxx.xxx.xxx.xxx.xxxx
xxxxgxxxgxxxgxxxgxxxgxxxx
"""

STARTS = [
    (4, 1, "DOWN"),
    (8, 1, "DOWN"),
    (12, 1, "DOWN"),
    (16, 1, "DOWN"),
    (20, 1, "DOWN"),
    (2, 4, "RIGHT"),
    (22, 4, "LEFT"),
    (4, 15, "UP"),
    (8, 15, "UP"),
    (12, 15, "UP"),
    (16, 15, "UP"),
    (20, 15, "UP"),
]

ROUTE_LIBRARY = [
    [(4, 1), (4, 4), (20, 4), (20, 14), (4, 14), (4, 15)],
    [(8, 1), (8, 2), (20, 2), (20, 8), (8, 8), (8, 15)],
    [(12, 1), (12, 4), (0, 4), (0, 10), (12, 10), (12, 15)],
    [(16, 1), (16, 6), (24, 6), (24, 12), (16, 12), (16, 15)],
    [(20, 1), (20, 4), (4, 4), (4, 8), (20, 8), (20, 15)],
    [(0, 8), (8, 8), (8, 14), (24, 14), (24, 4), (0, 4)],
]

HUMAN_ROUTES = [
    [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4), (9, 4)],
    [(9, 8), (10, 8), (11, 8), (12, 8), (13, 8), (14, 8), (15, 8), (16, 8)],
    [(15, 12), (16, 12), (17, 12), (18, 12), (19, 12), (20, 12), (21, 12), (22, 12)],
    [(12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10)],
    [(4, 14), (5, 14), (6, 14), (7, 14), (8, 14), (9, 14), (10, 14)],
]

HOTSPOTS = {
    (4, 4),
    (8, 4),
    (12, 4),
    (16, 4),
    (20, 4),
    (4, 8),
    (8, 8),
    (12, 8),
    (16, 8),
    (20, 8),
    (4, 12),
    (8, 12),
    (12, 12),
    (16, 12),
    (20, 12),
}

CORRIDOR_ROWS = (4, 8, 12)
SPINE_COLS = (4, 8, 12, 16, 20)

CLOSURE_SCHEDULE = [
    {
        "start": 25,
        "end": 55,
        "cells": [(x, 8) for x in range(10, 15)],
    },
    {
        "start": 65,
        "end": 95,
        "cells": [(12, y) for y in range(5, 11)],
    },
    {
        "start": 105,
        "end": 135,
        "cells": [(x, 14) for x in range(15, 21)],
    },
    {
        "start": 145,
        "end": 175,
        "cells": [(8, y) for y in range(7, 13)],
    },
]


def _stage_agents(env):
    for agent, (x, y, direction) in zip(env.unwrapped.agents, STARTS):
        agent.x = x
        agent.y = y
        agent.dir = getattr(Direction, direction)
        agent.carrying_shelf = None
        agent.has_delivered = False
    env.unwrapped._recalc_grid()


def _desired_direction(agent, target):
    tx, ty = target
    if agent.x < tx:
        return Direction.RIGHT
    if agent.x > tx:
        return Direction.LEFT
    if agent.y < ty:
        return Direction.DOWN
    if agent.y > ty:
        return Direction.UP
    return agent.dir


def _turn_action(current, desired):
    wrap = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    ci = wrap.index(current)
    di = wrap.index(desired)
    cw = (di - ci) % 4
    ccw = (ci - di) % 4
    return TURN_RIGHT if cw <= ccw else TURN_LEFT


def _next_waypoint(agent, route, route_progress):
    target = route[route_progress % len(route)]
    if (agent.x, agent.y) == target:
        route_progress = (route_progress + 1) % len(route)
        target = route[route_progress]
    return target, route_progress


def _forward_target(agent):
    if agent.dir == Direction.UP:
        return (agent.x, agent.y - 1)
    if agent.dir == Direction.DOWN:
        return (agent.x, agent.y + 1)
    if agent.dir == Direction.LEFT:
        return (agent.x - 1, agent.y)
    return (agent.x + 1, agent.y)


def _in_bounds(env, pos):
    return 0 <= pos[0] < env.unwrapped.grid_size[1] and 0 <= pos[1] < env.unwrapped.grid_size[0]


def _proposal_priority(idx, agent, wait_steps, hotspot_load):
    return (
        wait_steps[idx],
        1 if (agent.x, agent.y) in HOTSPOTS else 0,
        hotspot_load.get((agent.x, agent.y), 0),
        -idx,
    )


def _neighbors(env, pos):
    x, y = pos
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if not _in_bounds(env, (nx, ny)):
            continue
        if not env.unwrapped._is_highway(nx, ny):
            continue
        yield (nx, ny)


def _heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _nearest_hotspot(cell):
    best = None
    best_dist = 10**9
    for hotspot in HOTSPOTS:
        dist = abs(cell[0] - hotspot[0]) + abs(cell[1] - hotspot[1])
        if dist < best_dist:
            best_dist = dist
            best = hotspot
    return best, best_dist


def _reservation_zone(cell):
    hotspot, dist = _nearest_hotspot(cell)
    if hotspot is None or dist > 1:
        return None
    return hotspot


def _corridor_penalty(cell, corridor_load):
    x, y = cell
    penalty = 0
    if y in CORRIDOR_ROWS:
        penalty += corridor_load.get(("row", y), 0) * 2
    if x in SPINE_COLS:
        penalty += corridor_load.get(("col", x), 0)
    return penalty


def _build_corridor_load(env, route_progress):
    load = {}
    for idx, agent in enumerate(env.unwrapped.agents):
        route = ROUTE_LIBRARY[idx % len(ROUTE_LIBRARY)]
        target = route[route_progress[idx] % len(route)]
        if agent.y != target[1]:
            load[("col", agent.x)] = load.get(("col", agent.x), 0) + 1
        if agent.x != target[0]:
            load[("row", agent.y)] = load.get(("row", agent.y), 0) + 1
    return load


def _find_path(env, start, goal, occupied, corridor_load):
    blocked = set(env.unwrapped.static_blocked_cells).union(env.unwrapped.dynamic_blocked_cells)
    humans = {(human.x, human.y) for human in env.unwrapped.humans}
    forbidden = blocked.union(humans)
    forbidden.update(pos for pos in occupied if pos != start and pos != goal)
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
        for nxt in _neighbors(env, current):
            if nxt in forbidden:
                continue
            new_cost = cost_so_far[current] + 1 + _corridor_penalty(nxt, corridor_load)
            zone = _reservation_zone(nxt)
            if zone is not None:
                new_cost += 2
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + _heuristic(nxt, goal)
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


def _traffic_managed_actions(env, route_progress, wait_steps, reservation_state):
    agents = env.unwrapped.agents
    actions = [NOOP] * len(agents)
    proposals = []
    hotspot_load = {}
    occupied_now = {(agent.x, agent.y) for agent in agents}
    corridor_load = _build_corridor_load(env, route_progress)

    for idx, agent in enumerate(agents):
        route = ROUTE_LIBRARY[idx % len(ROUTE_LIBRARY)]
        target, route_progress[idx] = _next_waypoint(agent, route, route_progress[idx])
        if (agent.x, agent.y) == target:
            actions[idx] = NOOP
            continue

        path = _find_path(
            env,
            (agent.x, agent.y),
            target,
            occupied_now,
            corridor_load,
        )
        if len(path) < 2:
            actions[idx] = NOOP
            continue
        next_cell = path[1]
        desired = _direction_to((agent.x, agent.y), next_cell)
        if desired is None:
            actions[idx] = NOOP
            continue
        if agent.dir != desired:
            actions[idx] = _turn_action(agent.dir, desired)
            continue

        hotspot_load[next_cell] = hotspot_load.get(next_cell, 0) + 1
        proposals.append(
            {
                "idx": idx,
                "agent": agent,
                "target": next_cell,
                "desired": desired,
                "path": path,
            }
        )

    humans = {(human.x, human.y) for human in env.unwrapped.humans}
    blocked = set(env.unwrapped.dynamic_blocked_cells).union(env.unwrapped.static_blocked_cells)
    claimed_targets = set()
    zone_claims = set()
    approved = set()

    proposals.sort(
        key=lambda item: _proposal_priority(
            item["idx"], item["agent"], wait_steps, hotspot_load
        ),
        reverse=True,
    )

    for proposal in proposals:
        idx = proposal["idx"]
        agent = proposal["agent"]
        target = proposal["target"]

        if target in claimed_targets:
            continue
        if target in humans or target in blocked:
            continue
        if target in occupied_now:
            continue
        zone = _reservation_zone(target)
        if zone is not None:
            owner = reservation_state.get(zone)
            if owner is not None and owner != idx:
                continue
            if zone in zone_claims:
                continue
            reservation_state[zone] = idx
            zone_claims.add(zone)

            # Backpressure: only allow one entrant near the same hotspot per step.
            zone_path_cells = {
                cell for cell in proposal["path"][:3] if _reservation_zone(cell) == zone
            }
            if any(cell in claimed_targets for cell in zone_path_cells):
                continue
        approved.add(idx)
        claimed_targets.add(target)

    for zone, owner in list(reservation_state.items()):
        if owner >= len(agents):
            reservation_state.pop(zone, None)
            continue
        agent = agents[owner]
        if _reservation_zone((agent.x, agent.y)) != zone:
            reservation_state.pop(zone, None)

    for idx in approved:
        actions[idx] = FORWARD

    return actions


def _scheduled_closures(step):
    active = []
    for closure in CLOSURE_SCHEDULE:
        if closure["start"] <= step < closure["end"]:
            active.extend(closure["cells"])
    return active


def run(
    seed=0,
    max_steps=160,
    verbose=True,
    render=False,
    hold=False,
    save_dir=None,
    save_name=None,
    fps=6,
):
    print("=" * 60)
    print("SCENARIO : FACTORY COMPLEX")
    print("layout   : multi-zone factory with 5 vertical spines and 3 main corridors")
    print("traffic  : 12 AMRs cycle across zones while 4 humans cross key aisles")
    print("events   : timed aisle closures emulate spills, maintenance, and temporary staging")
    print("=" * 60)

    recorder = EpisodeRecorder(
        save_dir=save_dir,
        save_name=save_name or "factory_complex",
        fps=fps,
    )
    env = gym.make(
        "rware:rware-tiny-12ag-v2",
        layout=LAYOUT,
        render_mode="human" if render or recorder.enabled else "rgb_array",
        human_patrol_routes=HUMAN_ROUTES,
        human_count=len(HUMAN_ROUTES),
        human_safety_radius=1,
        human_move_prob=1.0,
        human_wait_prob=0.25,
        deadlock_patience=6,
        reward_shaping={
            "wait": -0.015,
            "deadlock": -0.35,
            "human_block": -0.04,
            "zone_block": -0.03,
            "progress": 0.002,
        },
    )
    env.reset(seed=seed)
    _stage_agents(env)
    env.unwrapped._agent_wait_steps = np.zeros(env.unwrapped.n_agents, dtype=np.int32)
    env.unwrapped._position_history.clear()

    manager = GlobalTrafficManager(
        env,
        route_library=ROUTE_LIBRARY,
        hotspot_radius=1,
        hotspot_backpressure=1,
        corridor_penalty=2,
        human_penalty=8,
    )
    recovery = AdaptiveDeadlockRecovery(
        manager,
        close_radius=1,
        penalty_radius=3,
        closure_ttl=24,
        penalty_ttl=40,
        wait_threshold=5,
        base_penalty=12,
    )
    detector = DeadlockDetector(patience=6)
    stats = {
        "deadlock_events": 0,
        "recovery_events": 0,
        "manager_holds": 0,
        "max_blocked_human": 0,
        "max_blocked_agent": 0,
        "max_blocked_zone": 0,
        "max_avg_wait": 0.0,
    }

    for step in range(max_steps):
        recovery.refresh(step)
        scheduled_blocks = _scheduled_closures(step)
        adaptive_blocks = sorted(recovery.active_blocked_cells(step))
        env.unwrapped.set_dynamic_blocked_cells(scheduled_blocks + adaptive_blocks)
        actions = manager.compute_actions()
        stats["manager_holds"] += sum(action == NOOP for action in actions)
        if recorder.enabled:
            recorder.capture(env)
        elif render:
            env.render()

        _, rewards, done, truncated, info = env.step(actions)
        metrics = info.get("metrics", {})
        wait_steps = info.get("wait_steps", manager.wait_steps)
        manager.update_wait_steps(wait_steps)
        positions = [(agent.x, agent.y) for agent in env.unwrapped.agents]
        if detector.update(positions):
            stats["deadlock_events"] += 1
            hotspot = recovery.register_deadlock(step, positions, wait_steps)
            if hotspot is not None:
                stats["recovery_events"] += 1
            detector.reset()

        stats["max_blocked_human"] = max(
            stats["max_blocked_human"], metrics.get("blocked_by_human", 0)
        )
        stats["max_blocked_agent"] = max(
            stats["max_blocked_agent"], metrics.get("blocked_by_agent", 0)
        )
        stats["max_blocked_zone"] = max(
            stats["max_blocked_zone"], metrics.get("blocked_by_zone", 0)
        )
        stats["max_avg_wait"] = max(
            stats["max_avg_wait"], metrics.get("avg_wait_steps", 0.0)
        )

        if verbose and step % 10 == 0:
            print(
                f"  step {step:3d} | "
                f"deadlock={metrics.get('deadlock_active', False)} "
                f"blocked(H/A/Z)="
                f"{metrics.get('blocked_by_human', 0)}/"
                f"{metrics.get('blocked_by_agent', 0)}/"
                f"{metrics.get('blocked_by_zone', 0)} "
                f"avg_wait={metrics.get('avg_wait_steps', 0.0):.2f} "
                f"throughput={metrics.get('throughput', 0.0):.2f} "
                f"reward_sum={sum(rewards):.2f} "
                f"holds={stats['manager_holds']} "
                f"recoveries={stats['recovery_events']}"
            )

        if done or truncated:
            break

    outputs = recorder.save(stats)
    if outputs:
        stats["artifacts"] = outputs
    hold_render_window(env, enabled=(render or recorder.enabled) and hold)
    env.close()
    print(stats)
    return stats
