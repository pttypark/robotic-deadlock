import gymnasium as gym
import rware
from rware.warehouse import Direction

from deadlock.core import (
    DeadlockDetector,
    EpisodeRecorder,
    hold_render_window,
    resolve_human_blockage,
    run_loop,
)


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

    stats = run_loop(
        env,
        DeadlockDetector(patience=5),
        resolve_human_blockage,
        max_steps=max_steps,
        verbose=verbose,
        render=render,
        recorder=recorder,
    )
    outputs = recorder.save(stats)
    if outputs:
        stats["artifacts"] = outputs
    hold_render_window(env, enabled=(render or recorder.enabled) and hold)
    env.close()
    print(stats)
    return stats
