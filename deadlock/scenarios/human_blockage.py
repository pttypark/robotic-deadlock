import gymnasium as gym
import rware

from deadlock.core import (
    DeadlockDetector,
    EpisodeRecorder,
    hold_render_window,
    resolve_human_blockage,
    run_loop,
)
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

    stats = run_loop(
        env,
        DeadlockDetector(patience=4),
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
