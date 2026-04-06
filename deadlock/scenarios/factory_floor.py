import gymnasium as gym
import numpy as np
import rware
from rware.warehouse import Direction

from deadlock.core import (
    DeadlockDetector,
    EpisodeRecorder,
    hold_render_window,
)
from deadlock.global_manager import GlobalTrafficManager


LAYOUT = """
xxxxgxxxx.xxxxxxx.gxxx
xxxx.xxxx.xxxxxxx.xxxx
xxxx.xxxx.xxxxxxx.xxxx
xxxx.xxxx.xxxxxxx.xxxx
g....................g
xxxx.xxxx.xxx.xxx.xxxx
xxxx.xxxx.xxx.xxx.xxxx
xxxx.xxxx.xxx.xxx.xxxx
g....................g
xxxx.xxxx.xxx.xxx.xxxx
xxxx.xxxx.xxx.xxx.xxxx
xxxx.xxxx.xxx.xxx.xxxx
g....................g
xxxx.xxxx.xxxxxxx.xxxx
xxxx.xxxx.xxxxxxx.xxxx
xxxx.xxxx.xxxxxxx.xxxx
xxxxgxxxx.xxxxxxx.gxxx
"""

ROBOT_STARTS = [
    (2, 4, "RIGHT"),
    (10, 4, "RIGHT"),
    (18, 4, "LEFT"),
    (3, 8, "RIGHT"),
    (11, 8, "LEFT"),
    (19, 8, "LEFT"),
    (2, 12, "RIGHT"),
    (18, 12, "LEFT"),
]

HUMAN_ROUTES = [
    [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4), (8, 4)],
    [(12, 8), (13, 8), (14, 8), (15, 8), (16, 8), (17, 8), (18, 8), (19, 8)],
    [(10, 12), (11, 12), (12, 12), (13, 12), (14, 12), (15, 12), (16, 12)],
]

ROUTE_LIBRARY = [
    [(2, 4), (10, 4), (18, 4), (18, 12), (2, 12), (2, 4)],
    [(10, 4), (19, 8), (18, 12), (2, 12), (3, 8), (10, 4)],
    [(18, 4), (11, 8), (2, 12), (2, 4), (18, 4)],
    [(3, 8), (11, 8), (19, 8), (18, 12), (2, 12), (3, 8)],
    [(11, 8), (19, 8), (18, 4), (10, 4), (3, 8), (11, 8)],
    [(19, 8), (18, 12), (10, 12), (2, 12), (11, 8), (19, 8)],
    [(2, 12), (10, 12), (18, 12), (19, 8), (2, 4), (2, 12)],
    [(18, 12), (10, 12), (2, 12), (3, 8), (18, 4), (18, 12)],
]


def _stage_agents(env):
    agents = env.unwrapped.agents
    for agent, (x, y, direction) in zip(agents, ROBOT_STARTS):
        agent.x = x
        agent.y = y
        agent.dir = getattr(Direction, direction)
        agent.carrying_shelf = None
        agent.has_delivered = False
    env.unwrapped._recalc_grid()


def run(
    seed=0,
    max_steps=120,
    verbose=True,
    render=False,
    hold=False,
    save_dir=None,
    save_name=None,
    fps=6,
):
    print("=" * 60)
    print("SCENARIO : FACTORY FLOOR")
    print("layout   : multi-intersection factory traffic grid")
    print("trigger  : eight AMRs share three main corridors with human crossings")
    print("resolve  : human-aware yielding baseline with global congestion metrics")
    print("=" * 60)

    recorder = EpisodeRecorder(
        save_dir=save_dir,
        save_name=save_name or "factory_floor",
        fps=fps,
    )
    env = gym.make(
        "rware:rware-tiny-8ag-v2",
        layout=LAYOUT,
        render_mode="human" if render or recorder.enabled else "rgb_array",
        human_patrol_routes=HUMAN_ROUTES,
        human_count=len(HUMAN_ROUTES),
        human_safety_radius=1,
        human_move_prob=1.0,
        human_wait_prob=0.2,
        deadlock_patience=5,
        reward_shaping={
            "wait": -0.015,
            "deadlock": -0.3,
            "human_block": -0.03,
            "progress": 0.003,
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
    detector = DeadlockDetector(patience=5)
    stats = {
        "deadlock_events": 0,
        "manager_holds": 0,
        "max_blocked_human": 0,
        "max_blocked_agent": 0,
        "max_avg_wait": 0.0,
    }

    for step in range(max_steps):
        actions = manager.compute_actions()
        stats["manager_holds"] += sum(action == 0 for action in actions)

        if recorder.enabled:
            recorder.capture(env)
        elif render:
            env.render()

        _, _, done, truncated, info = env.step(actions)
        metrics = info.get("metrics", {})
        manager.update_wait_steps(info.get("wait_steps", manager.wait_steps))
        positions = [(agent.x, agent.y) for agent in env.unwrapped.agents]
        if detector.update(positions):
            stats["deadlock_events"] += 1
            detector.reset()

        stats["max_blocked_human"] = max(
            stats["max_blocked_human"], metrics.get("blocked_by_human", 0)
        )
        stats["max_blocked_agent"] = max(
            stats["max_blocked_agent"], metrics.get("blocked_by_agent", 0)
        )
        stats["max_avg_wait"] = max(
            stats["max_avg_wait"], metrics.get("avg_wait_steps", 0.0)
        )

        if verbose and step % 10 == 0:
            print(
                f"  step {step:3d} | "
                f"deadlock={metrics.get('deadlock_active', False)} "
                f"blocked(H/A)="
                f"{metrics.get('blocked_by_human', 0)}/"
                f"{metrics.get('blocked_by_agent', 0)} "
                f"avg_wait={metrics.get('avg_wait_steps', 0.0):.2f} "
                f"throughput={metrics.get('throughput', 0.0):.2f} "
                f"holds={stats['manager_holds']}"
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
