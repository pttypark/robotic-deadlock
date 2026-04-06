import gymnasium as gym
import rware

from deadlock.core import (
    DeadlockDetector,
    EpisodeRecorder,
    hold_render_window,
    resolve_single_lane,
    run_loop,
)
from deadlock.navigation import face_robot


LAYOUT = """
xxxxxxxxxxxxx
g...........g
xxxxxxxxxxxxx
"""


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
    print("SCENARIO : SINGLE-LANE DEADLOCK")
    print("layout   : g...........g")
    print("trigger  : two robots advance into a one-cell corridor")
    print("resolve  : priority-based yielding")
    print("=" * 60)

    recorder = EpisodeRecorder(
        save_dir=save_dir,
        save_name=save_name or "single_lane",
        fps=fps,
    )
    env = gym.make(
        "rware:rware-tiny-2ag-v2",
        layout=LAYOUT,
        render_mode="human" if render or recorder.enabled else "rgb_array",
    )
    env.reset(seed=seed)

    agents = env.unwrapped.agents
    left_idx = 0 if agents[0].x <= agents[1].x else 1
    right_idx = 1 - left_idx
    face_robot(env, left_idx, "RIGHT")
    face_robot(env, right_idx, "LEFT")

    stats = run_loop(
        env,
        DeadlockDetector(),
        resolve_single_lane,
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
