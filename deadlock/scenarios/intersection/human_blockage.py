import gymnasium as gym
import numpy as np
import rware

from deadlock.core import (
    DeadlockDetector,
    EpisodeRecorder,
    hold_render_window,
)
from deadlock.global_manager import GlobalTrafficManager
from deadlock.navigation import navigate_to


LAYOUT = """
xxxxgxxxx
xxxx.xxxx
xxxx.xxxx
xxxx.xxxx
g.......g
xxxx.xxxx
xxxx.xxxx
xxxx.xxxx
xxxxgxxxx
"""

ENTRY_POSITIONS = [
    (2, 4, "RIGHT"),
    (6, 4, "LEFT"),
    (4, 2, "DOWN"),
    (4, 6, "UP"),
]

HUMAN_ROUTE = [
    (1, 4),
    (2, 4),
    (3, 4),
    (4, 4),
    (5, 4),
    (6, 4),
    (7, 4),
    (6, 4),
    (5, 4),
    (4, 4),
    (3, 4),
    (2, 4),
]

ROUTE_LIBRARY = [
    [(2, 4), (6, 4), (4, 6), (2, 4)],
    [(6, 4), (2, 4), (4, 2), (6, 4)],
    [(4, 2), (4, 6), (6, 4), (4, 2)],
    [(4, 6), (4, 2), (2, 4), (4, 6)],
]


def run(
    seed=0,
    max_steps=80,
    verbose=True,
    render=False,
    hold=False,
    save_dir=None,
    save_name=None,
    fps=6,
):
    print("=" * 60)
    print("SCENARIO : HUMAN-INDUCED DEADLOCK")
    print("layout   : four-way intersection with a human crossing the center")
    print("trigger  : AMRs compete for the same bottleneck while a human patrols")
    print("resolve  : robots yield until the corridor clears")
    print("=" * 60)

    recorder = EpisodeRecorder(
        save_dir=save_dir,
        save_name=save_name or "human_blockage",
        fps=fps,
    )
    env = gym.make(
        "rware:rware-tiny-4ag-v2",
        layout=LAYOUT,
        render_mode="human" if render or recorder.enabled else "rgb_array",
        human_patrol_routes=[HUMAN_ROUTE],
        human_count=1,
        human_safety_radius=1,
        human_move_prob=1.0,
        human_wait_prob=0.15,
        reward_shaping={
            "wait": -0.02,
            "deadlock": -0.35,
            "human_block": -0.04,
            "progress": 0.004,
        },
    )
    env.reset(seed=seed)

    for idx, (tx, ty, td) in enumerate(ENTRY_POSITIONS):
        navigate_to(env, idx, tx, ty, td)
    env.unwrapped._agent_wait_steps = np.zeros(env.unwrapped.n_agents, dtype=np.int32)
    env.unwrapped._position_history.clear()

    manager = GlobalTrafficManager(
        env,
        route_library=ROUTE_LIBRARY,
        hotspot_radius=1,
        hotspot_backpressure=1,
        corridor_penalty=1,
        human_penalty=8,
    )
    detector = DeadlockDetector(patience=4)
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
                f"  step {step:2d} | "
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
