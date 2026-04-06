import gymnasium as gym
import numpy as np
import rware

from deadlock.core import DeadlockDetector, EpisodeRecorder, hold_render_window
from deadlock.gnn_manager import GraphNeuralTrafficManager
from deadlock.scenarios.factory_complex import (
    CORRIDOR_ROWS,
    CLOSURE_SCHEDULE,
    HOTSPOTS,
    HUMAN_ROUTES,
    LAYOUT,
    ROUTE_LIBRARY,
    SPINE_COLS,
    _stage_agents,
)


def _scheduled_closures(step):
    active = []
    for closure in CLOSURE_SCHEDULE:
        if closure["start"] <= step < closure["end"]:
            active.extend(closure["cells"])
    return active


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(env, pos):
    x, y = pos
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if 0 <= nx < env.unwrapped.grid_size[1] and 0 <= ny < env.unwrapped.grid_size[0]:
            if env.unwrapped._is_highway(nx, ny):
                yield (nx, ny)


def _nearest_hotspot(pos):
    best = None
    best_dist = 10**9
    for hotspot in HOTSPOTS:
        dist = _manhattan(pos, hotspot)
        if dist < best_dist:
            best = hotspot
            best_dist = dist
    return best, best_dist


def _find_escape_cell(env, start, occupied, blocked):
    hotspot, _ = _nearest_hotspot(start)
    if hotspot is None:
        return None

    queue = [start]
    seen = {start}
    primary_candidates = []
    fallback_candidates = []
    while queue:
        current = queue.pop(0)
        for nxt in _neighbors(env, current):
            if nxt in seen:
                continue
            seen.add(nxt)
            if nxt in blocked:
                continue
            if nxt in occupied and nxt != start:
                continue
            dist_to_hotspot = _manhattan(nxt, hotspot)
            degree = sum(1 for _ in _neighbors(env, nxt))
            if dist_to_hotspot >= 3 and degree <= 2:
                primary_candidates.append((dist_to_hotspot, -_manhattan(start, nxt), nxt))
            elif dist_to_hotspot >= 2:
                fallback_candidates.append((dist_to_hotspot, -_manhattan(start, nxt), nxt))
            queue.append(nxt)

    if primary_candidates:
        primary_candidates.sort(reverse=True)
        return primary_candidates[0][2]
    if fallback_candidates:
        fallback_candidates.sort(reverse=True)
        return fallback_candidates[0][2]
    return None


def _build_recovery_plan(env, wait_steps, min_wait=5, hold_radius=2):
    agents = env.unwrapped.agents
    occupied = {(agent.x, agent.y) for agent in agents}
    blocked = set(env.unwrapped.static_blocked_cells).union(env.unwrapped.dynamic_blocked_cells)
    for human in env.unwrapped.humans:
        blocked.add((human.x, human.y))

    stalled_by_hotspot = {}
    for idx, agent in enumerate(agents):
        wait = wait_steps[idx] if idx < len(wait_steps) else 0
        hotspot, dist = _nearest_hotspot((agent.x, agent.y))
        if hotspot is None:
            continue
        stalled_by_hotspot.setdefault(hotspot, []).append((wait, -dist, idx))

    override_targets = {}
    held_agents = set()
    reserved_escapes = set()

    hotspot_items = []
    for hotspot, entries in stalled_by_hotspot.items():
        hotspot_wait = sum(max(0, wait) for wait, _, _ in entries)
        hotspot_items.append((hotspot_wait, len(entries), hotspot, entries))
    hotspot_items.sort(reverse=True)

    for _, _, hotspot, entries in hotspot_items:
        active_entries = [entry for entry in entries if entry[0] >= min_wait]
        candidates = active_entries if active_entries else entries
        candidates = sorted(candidates, reverse=True)

        mover_idx = None
        escape_cell = None
        local_blocked = blocked.union(reserved_escapes)
        for _, _, idx in candidates:
            agent = agents[idx]
            escape_cell = _find_escape_cell(
                env,
                (agent.x, agent.y),
                occupied,
                local_blocked,
            )
            if escape_cell is not None:
                mover_idx = idx
                break

        if mover_idx is None or escape_cell is None:
            continue

        override_targets[mover_idx] = escape_cell
        reserved_escapes.add(escape_cell)

        for _, _, idx in entries:
            if idx == mover_idx:
                continue
            agent = agents[idx]
            if _manhattan((agent.x, agent.y), hotspot) <= hold_radius:
                held_agents.add(idx)

    return override_targets, held_agents


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
    print("SCENARIO : FACTORY COMPLEX GNN")
    print("layout   : multi-zone factory with graph-neural traffic coordination")
    print("traffic  : AMR-AMR plus human-AMR deadlock handled by message passing")
    print("model    : numpy-based GNN-style coordinator with predictive human occupancy")
    print("=" * 60)

    recorder = EpisodeRecorder(
        save_dir=save_dir,
        save_name=save_name or "factory_complex_gnn",
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

    manager = GraphNeuralTrafficManager(
        env,
        route_library=ROUTE_LIBRARY,
        hotspots=HOTSPOTS,
        corridor_rows=CORRIDOR_ROWS,
        spine_cols=SPINE_COLS,
        message_passing_steps=3,
        interaction_radius=5,
        hotspot_radius=1,
        human_penalty=10.0,
        human_prediction_horizon=6,
        predicted_human_penalty=7.5,
    )
    detector = DeadlockDetector(patience=6)
    recovery_plan = {
        "override_targets": {},
        "held_agents": set(),
        "until_step": -1,
        "mover_indices": set(),
    }
    stats = {
        "deadlock_events": 0,
        "forced_recoveries": 0,
        "manager_holds": 0,
        "max_blocked_human": 0,
        "max_blocked_agent": 0,
        "max_blocked_zone": 0,
        "max_avg_wait": 0.0,
    }

    for step in range(max_steps):
        env.unwrapped.set_dynamic_blocked_cells(_scheduled_closures(step))
        remaining_override_targets = {}
        for mover_idx, target in recovery_plan["override_targets"].items():
            mover = env.unwrapped.agents[mover_idx]
            if (mover.x, mover.y) != target:
                remaining_override_targets[mover_idx] = target
        recovery_plan["override_targets"] = remaining_override_targets
        recovery_plan["mover_indices"] = set(remaining_override_targets)

        if not recovery_plan["override_targets"] or step > recovery_plan["until_step"]:
            recovery_plan = {
                "override_targets": {},
                "held_agents": set(),
                "until_step": -1,
                "mover_indices": set(),
            }
            manager.clear_recovery_plan()
        else:
            manager.set_recovery_plan(
                recovery_plan["override_targets"],
                recovery_plan["held_agents"],
            )
        actions = manager.compute_actions()
        stats["manager_holds"] += sum(action == 0 for action in actions)

        if recorder.enabled:
            recorder.capture(env)
        elif render:
            env.render()

        _, rewards, done, truncated, info = env.step(actions)
        metrics = info.get("metrics", {})
        manager.update_wait_steps(info.get("wait_steps", manager.wait_steps))
        positions = [(agent.x, agent.y) for agent in env.unwrapped.agents]
        local_stall_trigger = step % 8 == 0 and max(manager.wait_steps, default=0) >= 10
        if detector.update(positions) or local_stall_trigger:
            stats["deadlock_events"] += 1
            override_targets, held_agents = _build_recovery_plan(
                env,
                manager.wait_steps,
                min_wait=5,
                hold_radius=2,
            )
            if override_targets:
                recovery_plan = {
                    "override_targets": override_targets,
                    "held_agents": held_agents,
                    "until_step": step + 16,
                    "mover_indices": set(override_targets),
                }
                manager.set_recovery_plan(override_targets, held_agents)
                stats["forced_recoveries"] += 1
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
                f"forced={stats['forced_recoveries']} "
                f"active_recovery={len(recovery_plan['override_targets'])}"
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
