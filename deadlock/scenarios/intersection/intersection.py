import gymnasium as gym
import rware

from deadlock.core import (
    DeadlockDetector,
    EpisodeRecorder,
    hold_render_window,
    resolve_intersection,
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
    (3, 4, "RIGHT"),
    (4, 3, "DOWN"),
    (5, 4, "LEFT"),
    (4, 5, "UP"),
]


def run(
    seed=0,
    max_steps=60,
    verbose=True,
    render=False,
    hold=False,
    save_dir=None,
    save_name=None,
    fps=6,
):
    print("=" * 60)
    print("SCENARIO : INTERSECTION DEADLOCK")
    print("layout   : 9x9 four-way intersection")
    print("trigger  : four robots enter the center simultaneously")
    print("resolve  : release one branch to break the cycle")
    print("=" * 60)

    recorder = EpisodeRecorder(
        save_dir=save_dir,
        save_name=save_name or "intersection",
        fps=fps,
    )
    env = gym.make(
        "rware:rware-tiny-4ag-v2",
        layout=LAYOUT,
        render_mode="human" if render or recorder.enabled else "rgb_array",
    )
    env.reset(seed=seed)

    for idx, (tx, ty, td) in enumerate(ENTRY_POSITIONS):
        navigate_to(env, idx, tx, ty, td)

    stats = run_loop(
        env,
        DeadlockDetector(),
        resolve_intersection,
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
